// Usage:
// nvcc -O3 -std=c++17 -arch=sm_120a -o gemv_topk ./gemv_topk.cu && ./gemv_topk --verify 4 && rm ./gemv_topk
// Fused kernel latency: 20.561 us
// clusterized kernel latency: 18.446 us
// nvcc -O3 -std=c++17 -arch=sm_90 -o gemv_topk ./gemv_topk.cu && ./gemv_topk --verify 4 && rm ./gemv_topk
// Fused kernel latency: 24.594 us
// clusterized kernel latency: 16.401 us
#include <cuda_runtime.h>
#include <cooperative_groups.h>
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
#include <stdint.h>
#include <cfloat>
#include <cub/block/block_radix_sort.cuh>

namespace cg = cooperative_groups;

#define CLUSTER_SIZE 5

// Problem constants
#define HD 128
#define HN 32
#define CLEN 128
#define CSZ 1024  // number of clusters
#define FSL (CSZ * CLEN)
#define TOPC 64
#define OUT_PER_HEAD (TOPC * CLEN)

// for GeMV
#define CENTERS_PER_CLUSTER ((CSZ + CLUSTER_SIZE - 1) / CLUSTER_SIZE)
#define GEMV_TILE_SIZE 32
#define GEMV_WEIGHT_BUFFER_LEN (GEMV_TILE_SIZE * 2)
#define GEMV_NUM_ROW_PER_WARP (GEMV_TILE_SIZE / 4)
#define GEMV_DEC_TILE (GEMV_NUM_ROW_PER_WARP / 2)
static constexpr size_t GEMV_SHARED_K_BUFFER_ELEMS = static_cast<size_t>(GEMV_WEIGHT_BUFFER_LEN) * HD;
static constexpr size_t GEMV_SHARED_BYTES =
    sizeof(half) * GEMV_SHARED_K_BUFFER_ELEMS +
    sizeof(float) * CSZ +
    sizeof(int) * CSZ;
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

        // 2) 对每个选中的簇，写入全局索引
        for (int r = 0; r < TOPC; ++r) {
            int c = center_order[r];
            for (int i = 0; i < CLEN; ++i) {
                int global_idx = c * CLEN + i;
                size_t out_off = (size_t)h * OUT_PER_HEAD + r * CLEN + i;
                out_indices_cpu[out_off] = global_idx;
            }
        }
    }
}

// 校验：忽略簇内顺序，仅比较集合是否相等，同时报告 recall@512。
static bool verify_gpu_vs_cpu(const std::vector<int>& cpu_idx, const std::vector<int>& gpu_idx,
                              int heads_to_check, double& avg_recall, bool verbose = true) {
    bool all_ok = true;
    avg_recall = 0.0;
    std::vector<int> a(CLEN), b(CLEN);

    for (int h = 0; h < heads_to_check; ++h) {
        double head_recall_sum = 0.0;
        bool head_ok = true;
        for (int r = 0; r < TOPC; ++r) {
            size_t off = (size_t)h * OUT_PER_HEAD + r * CLEN;
            // 复制切片并排序（按索引值）以忽略内部顺序差异
            std::copy(cpu_idx.begin() + off, cpu_idx.begin() + off + CLEN, a.begin());
            std::copy(gpu_idx.begin() + off, gpu_idx.begin() + off + CLEN, b.begin());
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            // 计算交集大小
            std::vector<int> inter; inter.reserve(CLEN);
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
            double recall = (double)inter.size() / (double)CLEN;
            head_recall_sum += recall;
            if (inter.size() != CLEN) head_ok = false;
            if (verbose && h == 0 && r == 0) {
                printf("[VERIFY] head=%d cluster-rank=%d recall@%d = %.2f%%\n", h, r, CLEN, recall*100.0);
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
__global__ void gemv_topk_kernel(const __half* __restrict__ q,
                                 const __half* __restrict__ centers,
                                 int* __restrict__ out_indices) {
    // Grid is launched with (HN * TOPC) blocks. Each block handles one head and one top-r cluster.
    const int g = blockIdx.x;
    const int h = g;
    const int tid = threadIdx.x;       // thread in CTA
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane_id = tid & 31;
    // indexes for CSZ q @ centers^T
    const uint32_t input_idx_0 = (lane_id % 16) * 8;              // head_dim
    const uint32_t weight_idx_0 = warp_id * GEMV_NUM_ROW_PER_WARP + lane_id / 16 * GEMV_DEC_TILE; // seq_len. tile_size = 16

    // gemv regs
    half __align__(16) reg_input[8], reg_weight[8];
    float __align__(16) qk[GEMV_DEC_TILE];
    // dynamic shared memory buffers
    extern __shared__ __align__(16) unsigned char shared_storage[];
    half* k_buffer = reinterpret_cast<half*>(shared_storage);
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CSZ);

    constexpr int BLOCK_THREADS = 128;
    constexpr int CENTERS_PER_THREAD = CSZ / BLOCK_THREADS;
    using CenterRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, CENTERS_PER_THREAD, int>;
    __shared__ typename CenterRadixSort::TempStorage center_sort_storage;

    if (blockDim.x != BLOCK_THREADS) {
        return;
    }

    // Load q into reg
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[h * HD + input_idx_0]);

    // Phase 1: 使用传入的聚类中心，直接计算每个中心分数  score_c = dot(q, center[h,c,:])
    
    // preload centers
    for (int i = 0; i < GEMV_DEC_TILE; i++) {
        cp_async_pred_load_128b(
            &k_buffer[(weight_idx_0 + i) * HD + input_idx_0],
            &centers[h * CSZ * HD + (weight_idx_0 + i) * HD + input_idx_0],
            (weight_idx_0 + i < CSZ)
        );
    }
    cp_async_commit_group();
    // main loop
    for (int tile_id = 1; tile_id < ((CSZ + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE); tile_id++) {
        // commit current stage cp.async load
        for (int i = 0; i < GEMV_DEC_TILE; i++) {
            cp_async_pred_load_128b(
                &k_buffer[(tile_id % 2) * GEMV_TILE_SIZE * HD + (weight_idx_0 + i) * HD + input_idx_0],
                &centers[h * CSZ * HD + (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i) * HD + input_idx_0],
                (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i < CSZ)
            );
        }
        cp_async_commit_group();
        // wait for last cp.async load
        cp_async_wait_group<1>();
        __syncthreads();
        // consume last cp.async buffer
        for (int i = 0; i < GEMV_DEC_TILE; i++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HD + input_idx_0]);
            qk[i] = 0.0f;
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
                qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
            }
            center_vals[(tile_id - 1) * GEMV_TILE_SIZE + weight_idx_0 + i] = qk[i];
            center_idx[(tile_id - 1) * GEMV_TILE_SIZE + weight_idx_0 + i] = (tile_id - 1) * GEMV_TILE_SIZE + weight_idx_0 + i;
        }
    }
    cp_async_wait_group<0>();
    __syncthreads();
    int last_tile_id = (CSZ + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE;
    for (int i = 0; i < GEMV_DEC_TILE; i++) {
    *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((last_tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HD + input_idx_0]);
        qk[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        center_vals[(last_tile_id - 1) * GEMV_TILE_SIZE + weight_idx_0 + i] = qk[i];
        center_idx[(last_tile_id - 1) * GEMV_TILE_SIZE + weight_idx_0 + i] = (last_tile_id - 1) * GEMV_TILE_SIZE + weight_idx_0 + i;
    }
    __syncthreads();
    
    // radix sort CSZ scores
    float center_scores_local[CENTERS_PER_THREAD];
    int center_index_local[CENTERS_PER_THREAD];
    #pragma unroll
    for (int item = 0; item < CENTERS_PER_THREAD; ++item) {
        int idx = tid * CENTERS_PER_THREAD + item;
        center_scores_local[item] = center_vals[idx];
        center_index_local[item] = center_idx[idx];
    }
    CenterRadixSort(center_sort_storage).SortDescending(center_scores_local, center_index_local);
    __syncthreads();

    #pragma unroll
    for (int item = 0; item < CENTERS_PER_THREAD; ++item) {
        int idx = tid * CENTERS_PER_THREAD + item;
        center_vals[idx] = center_scores_local[item];
        center_idx[idx] = center_index_local[item];
    }
    __syncthreads();

    // 写回top-CSZ centers对应的全局索引
    for (int j = 0; j < TOPC; j++) {
        int c = center_idx[j];
        for (int i = tid; i < CLEN; i += blockDim.x) {
            int global_idx = c * CLEN + i;
            int out_offset = h * OUT_PER_HEAD + j * CLEN + i;
            out_indices[out_offset] = global_idx;
        }
    }
}

// ---------------- gemv_topk kernel (grid = HEAD_NUM * TOPC) ----------------
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) gemv_topk_cluster_kernel(const __half* __restrict__ q,
                                 const __half* __restrict__ centers,
                                 int* __restrict__ out_indices) {
    // Grid is launched with (HN * TOPC) blocks. Each block handles one head and one top-r cluster.

    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const int h = grid.cluster_rank();
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane_id = tid & 31;
    // indexes for CSZ q @ centers^T
    const uint32_t input_idx_0 = (lane_id % 16) * 8;              // head_dim
    const uint32_t weight_idx_0 = warp_id * GEMV_NUM_ROW_PER_WARP + lane_id / 16 * GEMV_DEC_TILE; // seq_len. tile_size = 16

    // gemv regs
    half __align__(16) reg_input[8], reg_weight[8];
    float __align__(16) qk[GEMV_DEC_TILE];
    // dynamic shared memory buffers
    extern __shared__ __align__(16) unsigned char shared_storage[];
    half* k_buffer = reinterpret_cast<half*>(shared_storage);
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CSZ);


    // Load q into reg
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[h * HD + input_idx_0]);

    // Phase 1: 使用传入的聚类中心，直接计算每个中心分数  score_c = dot(q, center[h,c,:])
    // 每个cluster处理一部分csz                                
    const int cluster_center_offset = static_cast<int>(cluster_block_id) * CENTERS_PER_CLUSTER;
    const int remaining_centers = CSZ - cluster_center_offset;
    const int cluster_center_len = remaining_centers > 0
        ? (remaining_centers < CENTERS_PER_CLUSTER ? remaining_centers : CENTERS_PER_CLUSTER)
        : 0;
    const int cluster_tile_count = cluster_center_len > 0
        ? (cluster_center_len + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE
        : 0;

    // Initialize local slice with sentinel values
    for (int i = tid; i < CENTERS_PER_CLUSTER; i += blockDim.x) {
        int global_idx = cluster_center_offset + i;
        if (global_idx < CSZ) {
            center_vals[global_idx] = -FLT_MAX;
            center_idx[global_idx] = global_idx;
        }
    }
    __syncthreads();

    if (cluster_center_len > 0) {
        // preload centers (tile 0)
        for (int i = 0; i < GEMV_DEC_TILE; ++i) {
            cp_async_pred_load_128b(
                &k_buffer[(weight_idx_0 + i) * HD + input_idx_0],
                &centers[h * CSZ * HD + (cluster_center_offset + weight_idx_0 + i) * HD + input_idx_0],
                (weight_idx_0 + i < cluster_center_len)
            );
        }
        cp_async_commit_group();

        // pipeline remaining tiles
        for (int tile_id = 1; tile_id < cluster_tile_count; ++tile_id) {
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                cp_async_pred_load_128b(
                    &k_buffer[(tile_id % 2) * GEMV_TILE_SIZE * HD + (weight_idx_0 + i) * HD + input_idx_0],
                    &centers[h * CSZ * HD + (cluster_center_offset + tile_id * GEMV_TILE_SIZE + weight_idx_0 + i) * HD + input_idx_0],
                    (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i < cluster_center_len)
                );
            }
            cp_async_commit_group();
            cp_async_wait_group<1>();
            __syncthreads();

            const int tile_base = (tile_id - 1) * GEMV_TILE_SIZE;
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HD + input_idx_0]);
                qk[i] = 0.0f;
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
                }
                #pragma unroll
                for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
                    qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
                }
                const int local_index = tile_base + weight_idx_0 + i;
                const int global_center_idx = cluster_center_offset + local_index;
                if (local_index < cluster_center_len && global_center_idx < CSZ) {
                    center_vals[global_center_idx] = qk[i];
                    center_idx[global_center_idx] = global_center_idx;
                }
            }
        }

        cp_async_wait_group<0>();
        __syncthreads();

        const int last_tile_base = (cluster_tile_count - 1) * GEMV_TILE_SIZE;
        for (int i = 0; i < GEMV_DEC_TILE; ++i) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((cluster_tile_count - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HD + input_idx_0]);
            qk[i] = 0.0f;
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
                qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
            }
            const int local_index = last_tile_base + weight_idx_0 + i;
            const int global_center_idx = cluster_center_offset + local_index;
            if (local_index < cluster_center_len && global_center_idx < CSZ) {
                center_vals[global_center_idx] = qk[i];
                center_idx[global_center_idx] = global_center_idx;
            }
        }
    }

    __syncthreads();
    cluster.sync();

    // block 1-4 send center_vals and center_idx to block 0
    if (cluster_block_id != 0) {
        float *float_dst_shmem;
        int *int_dst_shmem;
        float_dst_shmem = cluster.map_shared_rank(center_vals, 0);
        int_dst_shmem = cluster.map_shared_rank(center_idx, 0);
        for (int i = tid; i < CENTERS_PER_CLUSTER; i += blockDim.x) {
            int global_idx = cluster_center_offset + i;
            if (global_idx < CSZ) {
                float_dst_shmem[global_idx] = center_vals[global_idx];
                int_dst_shmem[global_idx] = center_idx[global_idx];
            }
        }
    }

    cluster.sync();
    
    if (cluster_block_id == 0) {
        constexpr int BLOCK_THREADS = 128;
        constexpr int CENTERS_PER_THREAD = CSZ / BLOCK_THREADS;
        using CenterRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, CENTERS_PER_THREAD, int>;
        __shared__ typename CenterRadixSort::TempStorage center_sort_storage;
        // radix sort CSZ scores
        float center_scores_local[CENTERS_PER_THREAD];
        int center_index_local[CENTERS_PER_THREAD];
        #pragma unroll
        for (int item = 0; item < CENTERS_PER_THREAD; ++item) {
            int idx = tid * CENTERS_PER_THREAD + item;
            center_scores_local[item] = center_vals[idx];
            center_index_local[item] = center_idx[idx];
        }
        CenterRadixSort(center_sort_storage).SortDescending(center_scores_local, center_index_local);
        __syncthreads();

        #pragma unroll
        for (int item = 0; item < CENTERS_PER_THREAD; ++item) {
            int idx = tid * CENTERS_PER_THREAD + item;
            center_vals[idx] = center_scores_local[item];
            center_idx[idx] = center_index_local[item];
        }
        __syncthreads();
        // 写回top-CSZ centers对应的全局索引
        for (int j = 0; j < TOPC; j++) {
            int c = center_idx[j];
            for (int i = tid; i < CLEN; i += blockDim.x) {
                int global_idx = c * CLEN + i;
                int out_offset = h * OUT_PER_HEAD + j * CLEN + i;
                out_indices[out_offset] = global_idx;
            }
        }
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    printf("Device-side topk pipeline: HN=%d HD=%d FSL=%d CSZ=%d CLEN=%d TOPC=%d\n",
        HN, HD, FSL, CSZ, CLEN, TOPC);

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
    int* d_out_indices_2; // [HN, OUT_PER_HEAD]
    CUDA_CHECK(cudaMalloc(&d_out_indices_2, sizeof(int) * (size_t)HN * OUT_PER_HEAD));

    const size_t shmem_bytes = GEMV_SHARED_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(gemv_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_bytes)));
    printf("Using %zu bytes of dynamic shared memory per block.\n", shmem_bytes);

    // HN
    dim3 grid(HN), block(128);
    int warmup = 10, iters = 50;
    for (int i = 0; i < warmup; ++i) {
        gemv_topk_kernel<<<grid, block, shmem_bytes>>>(d_q, d_centers, d_out_indices);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < iters; ++i) {
        gemv_topk_kernel<<<grid, block, shmem_bytes>>>(d_q, d_centers, d_out_indices);
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float ms = 0.f; cudaEventElapsedTime(&ms, st, ed);
    printf("Fused kernel latency: %.3f us\n", (ms / iters) * 1000.0f);

    // HN * CLUSTER_SIZE
    dim3 grid_cluster(HN * CLUSTER_SIZE);
    for (int i = 0; i < warmup; ++i) {
        gemv_topk_cluster_kernel<<<grid_cluster, block, shmem_bytes>>>(d_q, d_centers, d_out_indices_2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t st_2, ed_2; cudaEventCreate(&st_2); cudaEventCreate(&ed_2);
    cudaEventRecord(st_2);
    for (int i = 0; i < iters; ++i) {
        gemv_topk_cluster_kernel<<<grid_cluster, block, shmem_bytes>>>(d_q, d_centers, d_out_indices_2);
    }
    cudaEventRecord(ed_2); cudaEventSynchronize(ed_2);
    float ms_2 = 0.f; cudaEventElapsedTime(&ms_2, st_2, ed_2);
    printf("clusterized kernel latency: %.3f us\n", (ms_2 / iters) * 1000.0f);

    // Copy back and print sample
    std::vector<int> h_out(HN * OUT_PER_HEAD);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out_indices, sizeof(int) * (size_t)HN * OUT_PER_HEAD, cudaMemcpyDeviceToHost));
    std::vector<int> h_out_2(HN * OUT_PER_HEAD);
    CUDA_CHECK(cudaMemcpy(h_out_2.data(), d_out_indices_2, sizeof(int) * (size_t)HN * OUT_PER_HEAD, cudaMemcpyDeviceToHost));
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

        // cluster kenrle verification
        printf("===== cluster kernel =====\n");
        avg_recall = 0.0;
        ok = verify_gpu_vs_cpu(cpu_idx, h_out_2, verify_heads, avg_recall, true);
        printf("Average recall(head-avg over %d heads) = %.2f%%\n", verify_heads, avg_recall*100.0);
        if (ok) printf("✓ GPU indices match CPU reference (set equality per cluster).\n");
        else    printf("✗ GPU indices differ from CPU reference.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_out_indices));
    CUDA_CHECK(cudaFree(d_out_indices_2));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_centers));
    CUDA_CHECK(cudaFree(d_kv));
    free(h_q);
    free(h_centers);
    free(h_kv);
    printf("Done: computed %d x %d indices on device.\n", HN, OUT_PER_HEAD);
    return 0;
}
