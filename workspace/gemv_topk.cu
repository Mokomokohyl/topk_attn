// Usage:
// nvcc -O3 -std=c++17 -arch=sm_120a -o gemv_topk ./gemv_topk.cu && ./gemv_topk --verify 4
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <utility>
#include <math.h>
#include <random>
#include "cuda_fp16.h"
#include <string.h>
#include <numeric>

// Problem constants
#define HD 128
#define HN 32
#define CLEN 2048
#define CSZ 20  // number of clusters
#define FSL (CSZ * CLEN)
#define TOPC 5
#define TOPK_PER_CLUSTER 512
#define OUT_PER_HEAD (TOPC * TOPK_PER_CLUSTER)

// for GeMV
#define TILE_SIZE 32
#define WEIGHT_BUFFER_LEN (TILE_SIZE * 2)
#define NUM_ROW_PER_WARP (TILE_SIZE / 4)
#define DEC_TILE (NUM_ROW_PER_WARP / 2)
static constexpr int CENTER_SORT_CAP = 32;
static constexpr size_t GEMV_SHARED_K_BUFFER_ELEMS = static_cast<size_t>(WEIGHT_BUFFER_LEN) * HD;
static constexpr size_t GEMV_SHARED_BYTES =
    sizeof(half) * GEMV_SHARED_K_BUFFER_ELEMS +
    sizeof(float) * CENTER_SORT_CAP +
    sizeof(int) * CENTER_SORT_CAP +
    sizeof(float) * CLEN +
    sizeof(int) * CLEN;
// wrapper of cp.async
__device__ __forceinline__ void cp_async_pred_load_128b(half* smem_ptr, const half* gmem_ptr, bool predicate) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_in_bytes = predicate ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
}

// wrapper of cp.async.commit_group
__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// wrapper of cp.async.wait_group
template <size_t n>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ---------------- CPU 参考实现与校验 ----------------
// 参考实现：与 GPU 算法的数学目标一致（精确 top-5 簇 + 每簇 top-512），按分数降序输出全局 token 索引。
static void cpu_reference_indices(const half* kv, const half* q, const half* centers,
                                 std::vector<int>& out_indices_cpu,
                                 int heads_to_check = HN, bool verbose = false) {
    out_indices_cpu.resize((size_t)HN * OUT_PER_HEAD);

    // 临时缓冲
    std::vector<float> qf(HD);
    std::vector<float> center_scores(CSZ);
    std::vector<int> center_order(CSZ);
    std::vector<std::pair<float,int>> token_scores(CLEN);

    for (int h = 0; h < heads_to_check; ++h) {
        // q -> float
        for (int d = 0; d < HD; ++d) qf[d] = __half2float(q[h * HD + d]);

        // 1) 计算每个簇的中心分数：直接使用传入的中心向量
        for (int c = 0; c < CSZ; ++c) {
            double acc = 0.0;
            size_t base = ((size_t)h * CSZ + c) * HD; // centers layout: [HN, CSZ, HD]
            for (int d = 0; d < HD; ++d) {
                acc += (double)qf[d] * (double)__half2float(centers[base + d]);
            }
            center_scores[c] = (float)acc;
            center_order[c] = c;
        }
        std::partial_sort(center_order.begin(), center_order.begin() + TOPC, center_order.end(),
            [&](int a, int b){ return center_scores[a] > center_scores[b]; });

        // 2) 对每个选中的簇，计算 token 分数并取 top-512
        for (int r = 0; r < TOPC; ++r) {
            int c = center_order[r];
            // 计算所有 token 的分数
            for (int t = 0; t < CLEN; ++t) {
                size_t seq_idx = (size_t)c * CLEN + t;
                double acc = 0.0;
                size_t base = ((size_t)seq_idx * HN + h) * HD;
                for (int d = 0; d < HD; ++d) {
                    acc += (double)qf[d] * (double)__half2float(kv[base + d]);
                }
                token_scores[t] = { (float)acc, t };
            }
            // 取 top-512（降序）
            std::partial_sort(token_scores.begin(), token_scores.begin() + TOPK_PER_CLUSTER, token_scores.end(),
                [](const auto& A, const auto& B){ return A.first > B.first; });

            // 写入全局索引
            for (int i = 0; i < TOPK_PER_CLUSTER; ++i) {
                int t = token_scores[i].second;
                int global_idx = c * CLEN + t;
                size_t out_off = (size_t)h * OUT_PER_HEAD + r * TOPK_PER_CLUSTER + i;
                out_indices_cpu[out_off] = global_idx;
            }
        }

        if (verbose && h == 0) {
            printf("[CPU] head0 top centers: ");
            for (int r = 0; r < TOPC; ++r) printf("%d(%.5f) ", center_order[r], center_scores[center_order[r]]);
            printf("\n");
        }
    }
}

// 校验：忽略簇内顺序，仅比较集合是否相等，同时报告 recall@512。
static bool verify_gpu_vs_cpu(const std::vector<int>& cpu_idx, const std::vector<int>& gpu_idx,
                              int heads_to_check, double& avg_recall, bool verbose = true) {
    bool all_ok = true;
    avg_recall = 0.0;
    std::vector<int> a(TOPK_PER_CLUSTER), b(TOPK_PER_CLUSTER);

    for (int h = 0; h < heads_to_check; ++h) {
        double head_recall_sum = 0.0;
        bool head_ok = true;
        for (int r = 0; r < TOPC; ++r) {
            size_t off = (size_t)h * OUT_PER_HEAD + r * TOPK_PER_CLUSTER;
            // 复制切片并排序（按索引值）以忽略内部顺序差异
            std::copy(cpu_idx.begin() + off, cpu_idx.begin() + off + TOPK_PER_CLUSTER, a.begin());
            std::copy(gpu_idx.begin() + off, gpu_idx.begin() + off + TOPK_PER_CLUSTER, b.begin());
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            // 计算交集大小
            std::vector<int> inter; inter.reserve(TOPK_PER_CLUSTER);
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
            double recall = (double)inter.size() / (double)TOPK_PER_CLUSTER;
            head_recall_sum += recall;
            if (inter.size() != TOPK_PER_CLUSTER) head_ok = false;
            if (verbose && h == 0 && r == 0) {
                printf("[VERIFY] head=%d cluster-rank=%d recall@%d = %.2f%%\n", h, r, TOPK_PER_CLUSTER, recall*100.0);
            }
        }
        double head_avg_recall = head_recall_sum / (double)TOPC;
        avg_recall += head_avg_recall;
        if (!head_ok) all_ok = false;
    }
    avg_recall /= (double)heads_to_check;
    return all_ok;
}

// ---------------- gemv_topk kernel (grid = HEAD_NUM * TOPC) ----------------
__global__ void gemv_topk_kernel(const __half* __restrict__ kv,
                                  const __half* __restrict__ q,
                                  const __half* __restrict__ centers,
                                  int* __restrict__ out_indices) {
    // Grid is launched with (HN * TOPC) blocks. Each block handles one head and one top-r cluster.
    const int g = blockIdx.x;
    const int h = g / TOPC;            // head id
    const int r_assigned = g % TOPC;   // which top-r (0..TOPC-1) this block is responsible for
    const int tid = threadIdx.x;       // thread in CTA
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane_id = tid & 31;
    // indexes for centers
    const uint32_t input_idx_0 = (lane_id % 16) * 8;              // head_dim
    const uint32_t weight_idx_0 = warp_id * 4 + lane_id / 16 * 2; // seq_len. tile_size = 16
    // indexes for CLEN q@k^T
    const uint32_t input_idx = (lane_id % 16) * 8;              // head_dim
    const uint32_t weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / 16 * DEC_TILE; // seq_len. tile_size = 16

    // gemv regs
    half __align__(16) reg_input[8], reg_weight[8];
    float __align__(16) qk[DEC_TILE];
    // dynamic shared memory buffers
    extern __shared__ __align__(16) unsigned char shared_storage[];
    half* k_buffer = reinterpret_cast<half*>(shared_storage);
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CENTER_SORT_CAP);
    float* cand_vals = reinterpret_cast<float*>(center_idx + CENTER_SORT_CAP);
    int* cand_idx = reinterpret_cast<int*>(cand_vals + CLEN);


    // Load q into reg
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[h * HD + input_idx_0]);

    // Phase 1: 使用传入的聚类中心，直接计算每个中心分数  score_c = dot(q, center[h,c,:])
    for (int i = 0; i < 2; i++) {
        cp_async_pred_load_128b(
            &k_buffer[(weight_idx_0 + i) * HD + input_idx_0],
            &centers[h * CSZ * HD + (weight_idx_0 + i) * HD + input_idx_0],
            (weight_idx_0 + i < CSZ)
        );
    }
    cp_async_commit_group();
    for (int i = 0; i < 2; i++) {
        cp_async_pred_load_128b(
            &k_buffer[(16 + weight_idx_0 + i) * HD + input_idx_0],
            &centers[h * CSZ * HD + (16 + weight_idx_0 + i) * HD + input_idx_0],
            (16 + weight_idx_0 + i < CSZ)
        );
    }
    cp_async_commit_group();
    cp_async_wait_group<1>();
    __syncthreads();
    for (int i = 0; i < 2; i++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(weight_idx_0 + i) * HD + input_idx_0]);
        qk[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        center_vals[weight_idx_0 + i] = qk[i];
        center_idx[weight_idx_0 + i] = weight_idx_0 + i;
    }
    cp_async_wait_group<0>();
    __syncthreads();
    for (int i = 0; i < 2; i++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(16 + weight_idx_0 + i) * HD + input_idx_0]);
        qk[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        center_vals[16 + weight_idx_0 + i] = qk[i];
        center_idx[16 + weight_idx_0 + i] = 16 + weight_idx_0 + i;
    }
    __syncthreads();
    
    // pad remaining slots to -inf
    if (tid < CENTER_SORT_CAP) {
        if (tid >= CSZ) { center_vals[tid] = -INFINITY; center_idx[tid] = -1; }
    }
    __syncthreads();

    // bitonic sort 32 elements (descending) for centers
    for (int kseq = 2; kseq <= CENTER_SORT_CAP; kseq <<= 1) {
        for (int j = kseq >> 1; j > 0; j >>= 1) {
            int i = tid;
            if (i < CENTER_SORT_CAP) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & kseq) != 0);
                    float vi = center_vals[i];
                    float vx = center_vals[ixj];
                    if (((vi > vx) == up)) {
                        center_vals[i] = vx; center_vals[ixj] = vi;
                        int ti = center_idx[i]; center_idx[i] = center_idx[ixj]; center_idx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Phase 2: this block handles its assigned top-r cluster index
    int c = center_idx[r_assigned];
    // 计算所有 token 的分数：每个线程计算若干 t 并写入共享内存
    
    // preload k
    int common_offset = (c * CLEN + weight_idx) * HN * HD + h * HD + input_idx;
    for (int i = 0; i < DEC_TILE; i++) {
        cp_async_pred_load_128b(
            &k_buffer[(weight_idx + i) * HD + input_idx],
            &kv[common_offset + i * HN * HD],
            true
        );
    }
    cp_async_commit_group();
    // main loop
    for (int id = 1; id < CLEN / TILE_SIZE; id++) {
        // fill current buffer
        for (int i = 0; i < DEC_TILE; i++) {
            cp_async_pred_load_128b(
                &k_buffer[((id % 2) * TILE_SIZE + weight_idx + i) * HD + input_idx],
                &kv[common_offset + (id * TILE_SIZE + i) * HN * HD],
                true
            );
        }
        cp_async_commit_group();

        // wait for last buffer
        cp_async_wait_group<1>();
        __syncthreads();
        // consume last buffer
        for (int i = 0; i < DEC_TILE; i++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((id - 1) % 2) * TILE_SIZE + weight_idx + i) * HD + input_idx]);
            qk[i] = 0.0f;
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (16 >> 1); mask > 0; mask >>=1) {
                qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
            }
            cand_vals[(id - 1) * TILE_SIZE + weight_idx + i] = qk[i];
            cand_idx[(id - 1) * TILE_SIZE + weight_idx + i] = (id - 1) * TILE_SIZE + weight_idx + i;
        }
    }
    // epilogue
    int id = CLEN / TILE_SIZE;
    cp_async_wait_group<0>();
    __syncthreads();
    for (int i = 0; i < DEC_TILE; i++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((id - 1) % 2) * TILE_SIZE + weight_idx + i) * HD + input_idx]);
        qk[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (16 >> 1); mask > 0; mask >>=1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        cand_vals[(id - 1) * TILE_SIZE + weight_idx + i] = qk[i];
        cand_idx[(id - 1) * TILE_SIZE + weight_idx + i] = (id - 1) * TILE_SIZE + weight_idx + i;
    }
    __syncthreads();

    // 对 CLEN=2048 个候选做全量 bitonic 排序（降序）
    for (int kseq = 2; kseq <= CLEN; kseq <<= 1) {
        for (int j = kseq >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < CLEN; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & kseq) != 0);
                    float vi = cand_vals[i];
                    float vx = cand_vals[ixj];
                    if (((vi > vx) == up)) {
                        cand_vals[i] = vx; cand_vals[ixj] = vi;
                        int ti = cand_idx[i]; cand_idx[i] = cand_idx[ixj]; cand_idx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    // 写回 top-512 的全局索引
    for (int i = tid; i < TOPK_PER_CLUSTER; i += blockDim.x) {
        int local = cand_idx[i];
        int global_idx = c * CLEN + local;
        int out_offset = h * OUT_PER_HEAD + r_assigned * TOPK_PER_CLUSTER + i;
        out_indices[out_offset] = global_idx;
    }
    __syncthreads();
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    printf("Device-side topk pipeline: HN=%d HD=%d FSL=%d CSZ=%d CLEN=%d TOPC=%d TOPK_PER_CLUSTER=%d\n",
        HN, HD, FSL, CSZ, CLEN, TOPC, TOPK_PER_CLUSTER);

    // CLI: --verify [heads]
    bool do_verify = false; int verify_heads = HN;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--verify") == 0) {
            do_verify = true;
            if (i + 1 < argc) {
                int v = atoi(argv[i+1]);
                if (v > 0 && v <= HN) { verify_heads = v; ++i; }
            }
        }
    }

    // Host init (random floats -> half)
    size_t kv_elems = (size_t)FSL * HN * HD;
    size_t q_elems = (size_t)HN * HD;
    size_t center_elems = (size_t)HN * CSZ * HD;
    half* h_kv = (half*)malloc(sizeof(half) * kv_elems);
    half* h_q = (half*)malloc(sizeof(half) * q_elems);
    half* h_centers = (half*)malloc(sizeof(half) * center_elems);
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    std::uniform_real_distribution<float> uni(-0.5f, 0.5f);

    // 生成每个 head 的 20 个随机聚类中心
    for (int h = 0; h < HN; ++h) {
        for (int c = 0; c < CSZ; ++c) {
            size_t base = ((size_t)h * CSZ + c) * HD;
            for (int d = 0; d < HD; ++d) {
                float v = uni(rng);
                h_centers[base + d] = __float2half(v);
            }
        }
    }

    // 让 kv_cache token 围绕各自簇中心分布：token = center + 正态噪声
    for (int c = 0; c < CSZ; ++c) {
        for (int t = 0; t < CLEN; ++t) {
            size_t seq_idx = (size_t)c * CLEN + t;
            for (int h = 0; h < HN; ++h) {
                size_t kv_base = ((size_t)seq_idx * HN + h) * HD;
                size_t cen_base = ((size_t)h * CSZ + c) * HD;
                for (int d = 0; d < HD; ++d) {
                    float v = __half2float(h_centers[cen_base + d]) + noise(rng);
                    h_kv[kv_base + d] = __float2half(v);
                }
            }
        }
    }

    // q 随机（可选：让 head0 的 q 更接近某个中心以便观察）
    for (int h = 0; h < HN; ++h) {
        size_t base_q = (size_t)h * HD;
        for (int d = 0; d < HD; ++d) {
            h_q[base_q + d] = __float2half(uni(rng));
        }
    }

    // Device allocations
    half *d_kv, *d_q, *d_centers;
    CUDA_CHECK(cudaMalloc(&d_kv, sizeof(half) * kv_elems));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(half) * q_elems));
    CUDA_CHECK(cudaMalloc(&d_centers, sizeof(half) * center_elems));
    CUDA_CHECK(cudaMemcpy(d_kv, h_kv, sizeof(half) * kv_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, sizeof(half) * q_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centers, h_centers, sizeof(half) * center_elems, cudaMemcpyHostToDevice));

    int* d_out_indices; // [HN, OUT_PER_HEAD]
    CUDA_CHECK(cudaMalloc(&d_out_indices, sizeof(int) * (size_t)HN * OUT_PER_HEAD));

    const size_t shmem_bytes = GEMV_SHARED_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(gemv_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_bytes)));
    printf("Using %zu bytes of dynamic shared memory per block.\n", shmem_bytes);

    // Profile like flash_decoding.cu — launch TOPC blocks per head
    dim3 grid(HN * TOPC), block(128);
    int warmup = 10, iters = 50;
    for (int i = 0; i < warmup; ++i) {
        gemv_topk_kernel<<<grid, block, shmem_bytes>>>(d_kv, d_q, d_centers, d_out_indices);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < iters; ++i) {
        gemv_topk_kernel<<<grid, block, shmem_bytes>>>(d_kv, d_q, d_centers, d_out_indices);
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float ms = 0.f; cudaEventElapsedTime(&ms, st, ed);
    printf("Fused kernel latency: %.3f us\n", (ms / iters) * 1000.0f);

    // Copy back and print sample
    std::vector<int> h_out(HN * OUT_PER_HEAD);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out_indices, sizeof(int) * (size_t)HN * OUT_PER_HEAD, cudaMemcpyDeviceToHost));
    printf("head0 kv_indices[0..31]: ");
    for (int i = 0; i < 32; ++i) printf("%d ", h_out[i]);
    printf("\n");

    // Optional: CPU 参考 + 校验
    if (do_verify) {
        printf("Running CPU reference for %d head(s)... this may take a while.\n", verify_heads);
    std::vector<int> cpu_idx; cpu_reference_indices(h_kv, h_q, h_centers, cpu_idx, verify_heads, true);
        double avg_recall = 0.0;
        bool ok = verify_gpu_vs_cpu(cpu_idx, h_out, verify_heads, avg_recall, true);
        printf("Average recall(head-avg over %d heads) = %.2f%%\n", verify_heads, avg_recall*100.0);
        if (ok) printf("✓ GPU indices match CPU reference (set equality per cluster).\n");
        else    printf("✗ GPU indices differ from CPU reference.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_out_indices));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_centers));
    CUDA_CHECK(cudaFree(d_kv));
    free(h_q);
    free(h_centers);
    free(h_kv);
    printf("Done: computed %d x %d indices on device.\n", HN, OUT_PER_HEAD);
    return 0;
}
