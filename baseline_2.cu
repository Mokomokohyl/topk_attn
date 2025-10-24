// nvcc -O3 -std=c++17 -arch=sm_120a -o baseline_2 baseline_2.cu && ./baseline_2

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>
#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>
#include <cstring>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s (%d)\n", #call, __FILE__, __LINE__, cudaGetErrorString(err), (int)err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_TRY(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA-ERR] %s failed at %s:%d -> %s (%d)\n", #call, __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e); \
    } \
} while(0)

// ---- Problem constants ----
#define HEAD_DIM 128
#define HEAD_NUM 32
#define CSZ 20                 // number of clusters
#define CLEN 2048              // tokens per cluster
#define FULL_KV_SEQ_LEN (CSZ * CLEN)   // 40960
#define TOPC 5
#define TOPK_PER_CLUSTER 512
#define OUT_PER_HEAD (TOPC * TOPK_PER_CLUSTER) // 2560

// for GeMV
#define GEMV_TILE_SIZE 32
#define GEMV_WEIGHT_BUFFER_LEN (GEMV_TILE_SIZE * 2)
#define GEMV_NUM_ROW_PER_WARP (GEMV_TILE_SIZE / 4)
#define GEMV_DEC_TILE (GEMV_NUM_ROW_PER_WARP / 2)
static constexpr int CENTER_SORT_CAP = 32;
static constexpr size_t GEMV_SHARED_K_BUFFER_ELEMS = static_cast<size_t>(GEMV_WEIGHT_BUFFER_LEN) * HEAD_DIM;
static constexpr size_t GEMV_SHARED_BYTES =
    sizeof(half) * GEMV_SHARED_K_BUFFER_ELEMS +
    sizeof(float) * CENTER_SORT_CAP +
    sizeof(int) * CENTER_SORT_CAP +
    sizeof(float) * CLEN +
    sizeof(int) * CLEN +
    sizeof(int) * TOPK_PER_CLUSTER;

// ---- Flash decoding constants ----
#define SEQ_LEN OUT_PER_HEAD
#define CLUSTER_SIZE 5
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) // 512

static_assert(CLUSTER_SIZE == TOPC, "CLUSTER_SIZE must equal TOPC in fused baseline");
static_assert(KV_DIM_PER_BLOCK == TOPK_PER_CLUSTER, "Per-block KV length must equal TOPK_PER_CLUSTER");

#define NUM_WARPS 4
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS)
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP)
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW)

#define TMA_LOAD_ONCE 64
#define TMA_LOAD_ONCE_MAX 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * HEAD_DIM)
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 2)
#define TMA_LOAD_ONCE_NUM_ATTN ((TMA_LOAD_ONCE * HEAD_DIM) / 2)
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))

#define NUM_THREAD_PER_ROW_2 (HEAD_DIM / NUM_PER_THREAD)
#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE_ATTN / NUM_WARPS)
#define DIM_BLOCK_REDUCE (2 * BLOCK_SIZE / NUM_THREAD_PER_ROW_2)
#define DEC_TILE (NUM_ROW_PER_WARP_2 / (WARP_SIZE / NUM_THREAD_PER_ROW_2))
#define NUM_ROW_PER_WARP_3 (TMA_LOAD_ONCE / NUM_WARPS)
#define NUM_THREAD_PER_ROW_3 (WARP_SIZE / NUM_ROW_PER_WARP_3)
#define NUM_PER_ROW_3 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_3)

// ---- Flash decoding helpers ----
__forceinline__ __device__ float ptx_exp2(float x) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ void cp_async_pred_load_128b(half* smem_ptr, const half* gmem_ptr, bool predicate) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_in_bytes = predicate ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" : : "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;\n" ::); }
template <size_t n>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" : : "n"(n)); }

// ---------------- Fused top-k + decode kernel ----------------
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) topk_attn_kernel(
    half* __restrict__ output,
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const half* __restrict__ centers)
{
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = grid.cluster_rank() / HEAD_NUM;
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id          = tid & 31;
    const uint32_t warp_id          = tid >> 5;
    const int r_assigned            = static_cast<int>(cluster_block_id);

    // Shared memory (aligned to 128B for cp.async)
    extern __shared__ uint8_t shmem_raw[];
    uint8_t* shmem_aligned = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(shmem_raw) + 127) & ~127);
    half* shmem_half = reinterpret_cast<half*>(shmem_aligned);

    // ---- Stage 1: each cluster CTA selects its top cluster and gathers tokens ----
    half __align__(16) reg_q[8];
    half __align__(16) reg_weight_gemv[8];
    float __align__(16) qk_tile[GEMV_DEC_TILE];

    const uint32_t input_idx_0 = (lane_id % 16) * 8;
    const uint32_t weight_idx_0 = warp_id * 4 + (lane_id / 16) * 2;
    const uint32_t input_idx = (lane_id % 16) * 8;
    const uint32_t weight_idx = warp_id * GEMV_NUM_ROW_PER_WARP + (lane_id / 16) * GEMV_DEC_TILE;

    half* k_buffer = shmem_half;
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CENTER_SORT_CAP);
    float* cand_vals = reinterpret_cast<float*>(center_idx + CENTER_SORT_CAP);
    int* cand_idx_full = reinterpret_cast<int*>(cand_vals + CLEN);
    __shared__ int kv_topk[TOPK_PER_CLUSTER];

    const half* q_head = q_ptr + (batch_id * HEAD_NUM + head_id) * HEAD_DIM;
    *(uint4*)(&reg_q[0]) = *(const uint4*)(&q_head[input_idx_0]);

    for (int i = 0; i < 2; ++i) {
        cp_async_pred_load_128b(
            &k_buffer[(weight_idx_0 + i) * HEAD_DIM + input_idx_0],
            &centers[(batch_id * HEAD_NUM + head_id) * CSZ * HEAD_DIM + (weight_idx_0 + i) * HEAD_DIM + input_idx_0],
            (weight_idx_0 + i) < CSZ);
    }
    cp_async_commit_group();
    for (int i = 0; i < 2; ++i) {
        cp_async_pred_load_128b(
            &k_buffer[(16 + weight_idx_0 + i) * HEAD_DIM + input_idx_0],
            &centers[(batch_id * HEAD_NUM + head_id) * CSZ * HEAD_DIM + (16 + weight_idx_0 + i) * HEAD_DIM + input_idx_0],
            (16 + weight_idx_0 + i) < CSZ);
    }
    cp_async_commit_group();
    cp_async_wait_group<1>();
    __syncthreads();

    for (int i = 0; i < 2; ++i) {
    *(uint4*)(&reg_weight_gemv[0]) = *(uint4*)(&k_buffer[(weight_idx_0 + i) * HEAD_DIM + input_idx_0]);
        qk_tile[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            qk_tile[i] += __half2float(reg_q[d]) * __half2float(reg_weight_gemv[d]);
        }
#pragma unroll
        for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
            qk_tile[i] += __shfl_xor_sync(0xffffffff, qk_tile[i], mask);
        }
        center_vals[weight_idx_0 + i] = qk_tile[i];
        center_idx[weight_idx_0 + i] = weight_idx_0 + i;
    }
    cp_async_wait_group<0>();
    __syncthreads();
    for (int i = 0; i < 2; ++i) {
    *(uint4*)(&reg_weight_gemv[0]) = *(uint4*)(&k_buffer[(16 + weight_idx_0 + i) * HEAD_DIM + input_idx_0]);
        qk_tile[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            qk_tile[i] += __half2float(reg_q[d]) * __half2float(reg_weight_gemv[d]);
        }
#pragma unroll
        for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
            qk_tile[i] += __shfl_xor_sync(0xffffffff, qk_tile[i], mask);
        }
        center_vals[16 + weight_idx_0 + i] = qk_tile[i];
        center_idx[16 + weight_idx_0 + i] = 16 + weight_idx_0 + i;
    }
    __syncthreads();

    if (tid < CENTER_SORT_CAP) {
        if (tid >= CSZ) {
            center_vals[tid] = -CUDART_INF_F;
            center_idx[tid] = -1;
        }
    }
    __syncthreads();

    for (int kseq = 2; kseq <= CENTER_SORT_CAP; kseq <<= 1) {
        for (int j = kseq >> 1; j > 0; j >>= 1) {
            int i = tid;
            if (i < CENTER_SORT_CAP) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & kseq) != 0);
                    float vi = center_vals[i];
                    float vx = center_vals[ixj];
                    if ((vi > vx) == up) {
                        center_vals[i] = vx; center_vals[ixj] = vi;
                        int ti = center_idx[i]; center_idx[i] = center_idx[ixj]; center_idx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    int c = (r_assigned < TOPC) ? center_idx[r_assigned] : -1;
    if (c < 0) {
        for (int i = tid; i < TOPK_PER_CLUSTER; i += blockDim.x) {
            kv_topk[i] = -1;
        }
        __syncthreads();
    } else {
        int common_offset = (c * CLEN + weight_idx) * HEAD_NUM * HEAD_DIM + head_id * HEAD_DIM + input_idx;

        for (int i = 0; i < GEMV_DEC_TILE; ++i) {
            cp_async_pred_load_128b(
                &k_buffer[(weight_idx + i) * HEAD_DIM + input_idx],
                &k_cache[common_offset + i * HEAD_NUM * HEAD_DIM],
                true);
        }
        cp_async_commit_group();

        for (int tile_id = 1; tile_id < CLEN / GEMV_TILE_SIZE; ++tile_id) {
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                cp_async_pred_load_128b(
                    &k_buffer[((tile_id % 2) * GEMV_TILE_SIZE + weight_idx + i) * HEAD_DIM + input_idx],
                    &k_cache[common_offset + (tile_id * GEMV_TILE_SIZE + i) * HEAD_NUM * HEAD_DIM],
                    true);
            }
            cp_async_commit_group();

            cp_async_wait_group<1>();
            __syncthreads();
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                *(uint4*)(&reg_weight_gemv[0]) = *(uint4*)(&k_buffer[(((tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx + i) * HEAD_DIM + input_idx]);
                qk_tile[i] = 0.0f;
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    qk_tile[i] += __half2float(reg_q[d]) * __half2float(reg_weight_gemv[d]);
                }
#pragma unroll
                for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
                    qk_tile[i] += __shfl_xor_sync(0xffffffff, qk_tile[i], mask);
                }
                int out_idx = (tile_id - 1) * GEMV_TILE_SIZE + weight_idx + i;
                cand_vals[out_idx] = qk_tile[i];
                cand_idx_full[out_idx] = out_idx;
            }
        }

        int last_tile = CLEN / GEMV_TILE_SIZE;
        cp_async_wait_group<0>();
        __syncthreads();
        for (int i = 0; i < GEMV_DEC_TILE; ++i) {
            *(uint4*)(&reg_weight_gemv[0]) = *(uint4*)(&k_buffer[(((last_tile - 1) % 2) * GEMV_TILE_SIZE + weight_idx + i) * HEAD_DIM + input_idx]);
            qk_tile[i] = 0.0f;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                qk_tile[i] += __half2float(reg_q[d]) * __half2float(reg_weight_gemv[d]);
            }
#pragma unroll
            for (int mask = (16 >> 1); mask > 0; mask >>= 1) {
                qk_tile[i] += __shfl_xor_sync(0xffffffff, qk_tile[i], mask);
            }
            int out_idx = (last_tile - 1) * GEMV_TILE_SIZE + weight_idx + i;
            cand_vals[out_idx] = qk_tile[i];
            cand_idx_full[out_idx] = out_idx;
        }
        __syncthreads();

        for (int kseq = 2; kseq <= CLEN; kseq <<= 1) {
            for (int j = kseq >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < CLEN; i += blockDim.x) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        bool up = ((i & kseq) != 0);
                        float vi = cand_vals[i];
                        float vx = cand_vals[ixj];
                        if ((vi > vx) == up) {
                            cand_vals[i] = vx; cand_vals[ixj] = vi;
                            int ti = cand_idx_full[i]; cand_idx_full[i] = cand_idx_full[ixj]; cand_idx_full[ixj] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }

        for (int i = tid; i < TOPK_PER_CLUSTER; i += blockDim.x) {
            int local_idx = cand_idx_full[i];
            int global_idx = c * CLEN + local_idx;
            kv_topk[i] = global_idx;
        }
        __syncthreads();
    }
    cluster.sync();

    // ---- Stage 2: Flash decoding across cluster ----
    const uint32_t tile_row = tid / 16;
    const uint32_t tile_col = tid % 16;

    half* weight = shmem_half;
    half* local_output = weight + 2 * TMA_LOAD_ONCE * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(local_output + HEAD_DIM);

    __shared__ float cluster_local_sum, cluster_local_max;
    // #pragma nv_diag_suppress static_var_with_dynamic_init
    // __shared__ barrier bar[4];
    // __shared__ uint64_t barrier_token;
    // uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_token));
    // if (tid == 0) {
        // init(&bar[0], block.size());
        // cde::fence_proxy_async_shared_cta();
        // init(&bar[1], block.size());
        // cde::fence_proxy_async_shared_cta();
        // init(&bar[2], block.size());
        // cde::fence_proxy_async_shared_cta();
        // init(&bar[3], block.size());
        // cde::fence_proxy_async_shared_cta();
    // }
    // block.sync();

    float local_sum = 0.0f;
    float local_max = -CUDART_INF_F;
    float pre_max = -CUDART_INF_F;
    float scale = 0.0f;
    const float softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;

    half __align__(16) reg_input[NUM_PER_THREAD];
    half __align__(16) reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem = nullptr;
    float __align__(16) qk[DEC_TILE];

    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint cluster_head_idx = (batch_id * HEAD_NUM + head_id) * HEAD_DIM;

    local_sum = 0.0f;
    for (int i = 0; i < NUM_PER_THREAD; ++i) reg_reduce[i] = 0.0f;
    *(uint4*)(&reg_input[0]) = *(const uint4*)(&q_ptr[cluster_head_idx + input_idx_2]);
    block.sync();

    for (int j = 0; j < DEC_TILE; ++j) {
        int lane_offset = weight_idx_2 + j;
        int seq_idx = (lane_offset < KV_DIM_PER_BLOCK) ? kv_topk[lane_offset] : -1;
        bool valid = seq_idx >= 0;
        size_t k_offset = static_cast<size_t>(valid ? seq_idx : 0) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2;
        cp_async_pred_load_128b(
            &weight[(weight_idx_2 + j) * HEAD_DIM + input_idx_2],
            &k_cache[k_offset],
            valid);
    }
    cp_async_commit_group();
    for (int j = 0; j < DEC_TILE; ++j) {
        int lane_offset = weight_idx_2 + j;
        int seq_idx = (lane_offset < KV_DIM_PER_BLOCK) ? kv_topk[lane_offset] : -1;
        bool valid = seq_idx >= 0;
        size_t v_offset = static_cast<size_t>(valid ? seq_idx : 0) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2;
        cp_async_pred_load_128b(
            &weight[TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2],
            &v_cache[v_offset],
            valid);
    }
    cp_async_commit_group();

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; ++id) {
        for (int j = 0; j < DEC_TILE; ++j) {
            int lane_offset = weight_idx_2 + j;
            int chunk_offset = id * TMA_LOAD_ONCE_ATTN + lane_offset;
            int seq_idx = (chunk_offset < KV_DIM_PER_BLOCK) ? kv_topk[chunk_offset] : -1;
            bool valid = seq_idx >= 0;
            size_t k_offset = static_cast<size_t>(valid ? seq_idx : 0) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2;
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2],
                &k_cache[k_offset],
                valid);
        }
        cp_async_commit_group();
    cp_async_wait_group<2>();
    block.sync();

        pre_max = local_max;
#pragma unroll
        for (int j = 0; j < DEC_TILE; ++j) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            int token_idx_prev = (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
            bool prev_valid = (token_idx_prev < KV_DIM_PER_BLOCK) && (kv_topk[token_idx_prev] >= 0);
            if (prev_valid) {
                float acc = 0.0f;
#pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; ++d) {
                    acc += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
                }
#pragma unroll
                for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                    acc += __shfl_xor_sync(0xffffffff, acc, mask);
                }
                qk[j] = acc * softmax_scale;
            } else {
                qk[j] = -CUDART_INF_F;
            }
            local_max = fmaxf(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
#pragma unroll
        for (int j = 0; j < DEC_TILE; ++j) {
            int token_idx = (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
            if (token_idx < KV_DIM_PER_BLOCK && kv_topk[token_idx] >= 0) {
                qk[j] = ptx_exp2(qk[j] - local_max);
                local_sum += qk[j];
            } else {
                qk[j] = 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; ++j) {
            reg_reduce[j] *= scale;
        }
        for (int j = 0; j < DEC_TILE; ++j) {
            int lane_offset = weight_idx_2 + j;
            int chunk_offset = id * TMA_LOAD_ONCE_ATTN + lane_offset;
            int seq_idx = (chunk_offset < KV_DIM_PER_BLOCK) ? kv_topk[chunk_offset] : -1;
            bool valid = seq_idx >= 0;
            size_t v_offset = static_cast<size_t>(valid ? seq_idx : 0) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2;
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2],
                &v_cache[v_offset],
                valid);
        }
        cp_async_commit_group();
    cp_async_wait_group<2>();
    block.sync();
        for (int j = 0; j < DEC_TILE; ++j) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
#pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; ++d) {
                reg_reduce[d] += qk[j] * __half2float(reg_weight[d]);
            }
        }
    }

    cp_async_wait_group<1>();
    block.sync();

    pre_max = local_max;
#pragma unroll
    for (int j = 0; j < DEC_TILE; ++j) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        int token_idx_prev = (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
        bool prev_valid = (token_idx_prev < KV_DIM_PER_BLOCK) && (kv_topk[token_idx_prev] >= 0);
        if (prev_valid) {
            float acc = 0.0f;
#pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; ++d) {
                acc += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
#pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                acc += __shfl_xor_sync(0xffffffff, acc, mask);
            }
            qk[j] = acc * softmax_scale;
        } else {
            qk[j] = -CUDART_INF_F;
        }
        local_max = fmaxf(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
#pragma unroll
    for (int j = 0; j < DEC_TILE; ++j) {
        int token_idx = (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
        if (token_idx < KV_DIM_PER_BLOCK && kv_topk[token_idx] >= 0) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        } else {
            qk[j] = 0.0f;
        }
    }
#pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; ++j) {
        reg_reduce[j] *= scale;
    }

    cp_async_wait_group<0>();
    block.sync();

    for (int j = 0; j < DEC_TILE; ++j) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
#pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; ++d) {
            reg_reduce[d] += qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

#pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for (int i = 0; i < NUM_PER_THREAD; ++i) reg_reduce[i] = 0.0f;
    local_sum = 0.0f; local_max = 0.0f;
#pragma unroll
    for (int j = 0; j < DIM_BLOCK_REDUCE / 2; ++j) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = fmaxf(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
#pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; ++d) {
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if (tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    for (int i = 1; i < cluster.num_blocks() - 1; ++i) {
        if (tid == 0) {
            local_max = cluster_local_max;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_max, dst_cta);
        }
        cluster.sync();
        if (tid == 0) {
            *dst_shmem = fmaxf(*dst_shmem, local_max);
        }
        cluster.sync();
    }
    scale = ptx_exp2(pre_max - cluster_local_max);
    local_sum *= scale;
#pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; ++j) {
        reg_reduce[j] *= scale;
    }
    if (tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    for (int i = 1; i < cluster.num_blocks() - 1; ++i) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
        }
        cluster.sync();
        if (tid == 0) {
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
#pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; ++j) {
        reg_reduce[j] *= __frcp_rn(cluster_local_sum);
    }
    if (tid < NUM_THREAD_PER_ROW_2) {
#pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; ++i) {
            local_output[tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    block.sync();

    atomicAdd(&output[head_id * HEAD_DIM + tid], local_output[tid]);
}

// ---------------- Main: fused baseline ----------------
int main(int argc, char** argv) {
    printf("Fused baseline: HEAD_NUM=%d HEAD_DIM=%d CSZ=%d CLEN=%d OUT_PER_HEAD=%d FULL_KV_SEQ_LEN=%d\n",
           HEAD_NUM, HEAD_DIM, CSZ, CLEN, OUT_PER_HEAD, FULL_KV_SEQ_LEN);

    size_t kv_elems = (size_t)FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM;
    size_t q_elems  = (size_t)HEAD_NUM * HEAD_DIM;
    size_t center_elems = (size_t)HEAD_NUM * CSZ * HEAD_DIM;

    std::vector<half> h_k(kv_elems), h_v(kv_elems), h_q(q_elems), h_centers(center_elems);
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    std::uniform_real_distribution<float> uni(-0.5f, 0.5f);

    for (int h = 0; h < HEAD_NUM; ++h) {
        for (int c = 0; c < CSZ; ++c) {
            size_t base = ((size_t)h * CSZ + c) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) h_centers[base + d] = __float2half(uni(rng));
        }
    }

    for (int c = 0; c < CSZ; ++c) {
        for (int t = 0; t < CLEN; ++t) {
            size_t seq_idx = (size_t)c * CLEN + t;
            for (int h = 0; h < HEAD_NUM; ++h) {
                size_t kv_base = ((size_t)seq_idx * HEAD_NUM + h) * HEAD_DIM;
                size_t cen_base = ((size_t)h * CSZ + c) * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float v = __half2float(h_centers[cen_base + d]) + noise(rng);
                    h_k[kv_base + d] = __float2half(v);
                    h_v[kv_base + d] = __float2half(v);
                }
            }
        }
    }

    for (int h = 0; h < HEAD_NUM; ++h) {
        for (int d = 0; d < HEAD_DIM; ++d) h_q[h * HEAD_DIM + d] = __float2half(uni(rng));
    }

    half *d_k=nullptr, *d_v=nullptr, *d_q=nullptr, *d_centers=nullptr, *d_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(half) * kv_elems));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(half) * kv_elems));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(half) * q_elems));
    CUDA_CHECK(cudaMalloc(&d_centers, sizeof(half) * center_elems));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));

    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(half) * kv_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(half) * kv_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(half) * q_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), sizeof(half) * center_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));

    size_t flash_shmem_bytes = (2 * TMA_LOAD_ONCE * HEAD_DIM + HEAD_DIM) * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float);
    size_t fused_shmem_bytes = std::max(GEMV_SHARED_BYTES, flash_shmem_bytes);

    int dev = 0; cudaDeviceProp prop{}; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    printf("topk_attn requested dyn shmem: %zu bytes (device optin %zu)\n", fused_shmem_bytes, (size_t)prop.sharedMemPerBlockOptin);

    CUDA_TRY(cudaFuncSetAttribute(topk_attn_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute(topk_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(fused_shmem_bytes)));

    dim3 grid(HEAD_NUM * CLUSTER_SIZE), block(BLOCK_SIZE);
    int warmup = 10, iters = 50; float ms = 0.f;
    for (int i = 0; i < warmup; ++i) {
        topk_attn_kernel<<<grid, block, fused_shmem_bytes>>>(d_out, d_q, d_k, d_v, d_centers);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < iters; ++i) {
        topk_attn_kernel<<<grid, block, fused_shmem_bytes>>>(d_out, d_q, d_k, d_v, d_centers);
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed); cudaEventElapsedTime(&ms, st, ed);
    printf("topk_attn kernel latency: %.3f us\n", (ms / iters) * 1000.0f);

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_centers));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_k));
    printf("Done.\n");
    return 0;
}
