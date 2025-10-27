/* 
Correctness check:
nvcc -O3 -std=c++17 -arch=sm_120a -DDEBUG -o baselines_dbg baselines.cu && ./baselines_dbg && rm ./baselines_dbg

Latency test:
nvcc -O3 -std=c++17 -arch=sm_120a -o baselines baselines.cu && ./baselines && rm ./baselines
*/

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
#include <cmath>
#include <cfloat>
#include <cub/block/block_radix_sort.cuh>

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

// ---- Problem constants (unified across both stages) ----
#define HEAD_DIM 128
#define HEAD_NUM 32
#define CSZ 1024                 // number of clusters
#define CLEN 128              // tokens per cluster
#define FULL_KV_SEQ_LEN (CSZ * CLEN)
#define TOPC 100
#define OUT_PER_HEAD (TOPC * CLEN)

#define CLUSTER_SIZE 5

// for GeMV
#define CENTERS_PER_CLUSTER ((CSZ + CLUSTER_SIZE - 1) / CLUSTER_SIZE)
#define GEMV_TILE_SIZE 32
#define GEMV_WEIGHT_BUFFER_LEN (GEMV_TILE_SIZE * 2)
#define GEMV_NUM_ROW_PER_WARP (GEMV_TILE_SIZE / 4)
#define GEMV_DEC_TILE (GEMV_NUM_ROW_PER_WARP / 2)
static constexpr size_t GEMV_SHARED_K_BUFFER_ELEMS = static_cast<size_t>(GEMV_WEIGHT_BUFFER_LEN) * HEAD_DIM;
static constexpr size_t GEMV_SHARED_BYTES =
    sizeof(half) * GEMV_SHARED_K_BUFFER_ELEMS +
    sizeof(float) * CSZ +
    sizeof(int) * CSZ;

// ---- Flash decoding constants ----
#define SEQ_LEN OUT_PER_HEAD
#define C_PER_BLOCK (TOPC / CLUSTER_SIZE)
#define KV_DIM_PER_BLOCK ((SEQ_LEN + CLUSTER_SIZE - 1) / CLUSTER_SIZE)
#define C_PER_BLOCK_BS (TOPC / (CLUSTER_SIZE - 1))
#define KV_DIM_PER_BLOCK_BS (SEQ_LEN / (CLUSTER_SIZE - 1))

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
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(n)); }

// ##########################################
// #  Baseline 1: topk kernel + attn kernel #
// ##########################################

// New cluster GEMV-topk kernel synced with gemv_topk.cu
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) gemv_topk_cluster_kernel(const __half* __restrict__ q,
                                 const __half* __restrict__ centers,
                                 int* __restrict__ out_indices) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const int h = grid.cluster_rank();
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane_id = tid & 31;
    const uint32_t input_idx_0 = (lane_id % 16) * 8;
    const uint32_t weight_idx_0 = warp_id * GEMV_NUM_ROW_PER_WARP + lane_id / 16 * GEMV_DEC_TILE;

    half __align__(16) reg_input[8], reg_weight[8];
    float __align__(16) qk[GEMV_DEC_TILE];
    extern __shared__ __align__(16) unsigned char shared_storage[];
    half* k_buffer = reinterpret_cast<half*>(shared_storage);
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CSZ);

    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[h * HEAD_DIM + input_idx_0]);

    const int cluster_center_offset = static_cast<int>(cluster_block_id) * CENTERS_PER_CLUSTER;
    const int remaining_centers = CSZ - cluster_center_offset;
    const int cluster_center_len = remaining_centers > 0
        ? (remaining_centers < CENTERS_PER_CLUSTER ? remaining_centers : CENTERS_PER_CLUSTER)
        : 0;
    const int cluster_tile_count = cluster_center_len > 0
        ? (cluster_center_len + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE
        : 0;

    for (int i = tid; i < CENTERS_PER_CLUSTER; i += blockDim.x) {
        int global_idx = cluster_center_offset + i;
        if (global_idx < CSZ) {
            center_vals[global_idx] = -FLT_MAX;
            center_idx[global_idx] = global_idx;
        }
    }
    __syncthreads();

    if (cluster_center_len > 0) {
        for (int i = 0; i < GEMV_DEC_TILE; ++i) {
            cp_async_pred_load_128b(
                &k_buffer[(weight_idx_0 + i) * HEAD_DIM + input_idx_0],
                &centers[h * CSZ * HEAD_DIM + (cluster_center_offset + weight_idx_0 + i) * HEAD_DIM + input_idx_0],
                (weight_idx_0 + i < cluster_center_len)
            );
        }
        cp_async_commit_group();

        for (int tile_id = 1; tile_id < cluster_tile_count; ++tile_id) {
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                cp_async_pred_load_128b(
                    &k_buffer[(tile_id % 2) * GEMV_TILE_SIZE * HEAD_DIM + (weight_idx_0 + i) * HEAD_DIM + input_idx_0],
                    &centers[h * CSZ * HEAD_DIM + (cluster_center_offset + tile_id * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx_0],
                    (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i < cluster_center_len)
                );
            }
            cp_async_commit_group();
            cp_async_wait_group<1>();
            __syncthreads();

            const int tile_base = (tile_id - 1) * GEMV_TILE_SIZE;
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx_0]);
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
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((cluster_tile_count - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx_0]);
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

    if (cluster_block_id != 0) {
        float *float_dst_shmem = cluster.map_shared_rank(center_vals, 0);
        int *int_dst_shmem = cluster.map_shared_rank(center_idx, 0);
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

// ---- Flash decoding kernel ----
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) MHAFlashDecodeKernel(
    half* output,        // [H, D]
    half* q_ptr,         // [1, H, D]
    half* k_cache,       // [N, H, D]
    half* v_cache,
    int* kv_indices      // [H, SEQ_LEN]
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; 
    const uint32_t warp_id = tid / 32;
    const uint32_t tile_row = tid / 16;
    const uint32_t tile_col = tid % 16;

    // Init shared memory
    extern __shared__ uint8_t shmem_base[];
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_output = weight + 2 * TMA_LOAD_ONCE * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(local_output + HEAD_DIM);

    __shared__ float cluster_local_sum, cluster_local_max;

    block.sync();

    // Init registers
    float local_sum = 0.0, local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    float __align__(16) qk[DEC_TILE];

    // Precompute some indices
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD; // (lane_id % 16) * 8
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint cluster_head_idx = head_id * HEAD_DIM;

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q_ptr[cluster_head_idx + input_idx_2]);
    block.sync();

    // Preload kv_cache
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[0 + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
            &k_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
            true
        );
    }
    cp_async_commit_group();
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
            &v_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
            true
        );
    }
    cp_async_commit_group();

    // mainloop
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
                &k_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
                true
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();

        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
                qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
        // For filled zeros
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            if ((KV_DIM_PER_BLOCK * cluster_block_id + (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
                qk[j] = ptx_exp2(qk[j] - local_max);
                local_sum += qk[j];
            }
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
            reg_reduce[j] = reg_reduce[j] * scale;
        }
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
                &v_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
                true
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
                reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
            }
        }
    }
    // end: mainloop

    // epilogue
    cp_async_wait_group<1>();
    block.sync();

    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if ((KV_DIM_PER_BLOCK * cluster_block_id + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        }
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }

    cp_async_wait_group<0>();
    block.sync();

    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
            reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(ptx_exp2(pre_max - local_max))), __hmul(reg_input[d], __float2half(scale)));
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    // ClusterReduce: local_max
    for (int i = 1; i < cluster.num_blocks(); i++) {
        if (tid == 0) {
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
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // ClusterReduce: local_sum
    for (int i = 1; i < cluster.num_blocks(); i++) {
        if (tid == 0) {
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
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(cluster_local_sum);
    }
    if(tid < NUM_THREAD_PER_ROW_2) {
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            local_output[tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    block.sync();

    atomicAdd(&output[head_id * HEAD_DIM + tid], local_output[tid]);
}

// ################################################
// #  Baseline 2: topk_attn directly fused kernel #
// ################################################

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) topk_attn_fused_kernel(
    half* output,        // [H, D]
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const __half* __restrict__ q,
    const __half* __restrict__ centers) {
    // flash decoding setup
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; 
    const uint32_t warp_id = tid / 32;
    const uint32_t tile_row = tid / 16;
    const uint32_t tile_col = tid % 16;

    // Init shared memory
    extern __shared__ uint8_t shmem_base[];
    // gemv-topk
    half* k_buffer = reinterpret_cast<half*>(shmem_base);
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CSZ);
    // flash-decoding
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_output = weight + 2 * TMA_LOAD_ONCE * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(local_output + HEAD_DIM);

    __shared__ float cluster_local_sum, cluster_local_max;
    // store selected kv indices in shared memory in fused kernel
    __shared__ int kv_indices[KV_DIM_PER_BLOCK];

    block.sync();

    // Init registers
    float local_sum = 0.0, local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    float __align__(16) qk[GEMV_DEC_TILE];

    // Precompute some indices
    const uint32_t input_idx = (lane_id % 16) * 8;                // head_dim
    // indices for CSZ q @ centers^T
    const uint32_t weight_idx_0 = warp_id * GEMV_NUM_ROW_PER_WARP + lane_id / 16 * GEMV_DEC_TILE; // seq_len. tile_size = TILE_SIZE
    // indices for flash-decoding
    const uint32_t weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    const uint32_t cluster_head_idx = head_id * HEAD_DIM;

    // Load q into reg_input
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[cluster_head_idx + input_idx]);

    // ---------------- STAGE 1: gemv topk ----------------
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
            center_vals[global_idx] = -CUDART_INF_F;
            center_idx[global_idx] = global_idx;
        }
    }
    __syncthreads();

    if (cluster_center_len > 0) {
        // preload centers (tile 0)
        for (int i = 0; i < GEMV_DEC_TILE; ++i) {
            cp_async_pred_load_128b(
                &k_buffer[(weight_idx_0 + i) * HEAD_DIM + input_idx],
                &centers[head_id * CSZ * HEAD_DIM + (cluster_center_offset + weight_idx_0 + i) * HEAD_DIM + input_idx],
                (weight_idx_0 + i < cluster_center_len)
            );
        }
        cp_async_commit_group();

        // pipeline remaining tiles
        for (int tile_id = 1; tile_id < cluster_tile_count; ++tile_id) {
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                cp_async_pred_load_128b(
                    &k_buffer[(tile_id % 2) * GEMV_TILE_SIZE * HEAD_DIM + (weight_idx_0 + i) * HEAD_DIM + input_idx],
                    &centers[head_id * CSZ * HEAD_DIM + (cluster_center_offset + tile_id * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx],
                    (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i < cluster_center_len)
                );
            }
            cp_async_commit_group();
            cp_async_wait_group<1>();
            __syncthreads();

            const int tile_base = (tile_id - 1) * GEMV_TILE_SIZE;
            for (int i = 0; i < GEMV_DEC_TILE; ++i) {
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx]);
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
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((cluster_tile_count - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx]);
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
        // block 0 do radix sort
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
        // top-CSZ centers对应的全局索引写到各block的kv_indices
        for (int dst_cta_id = 0; dst_cta_id < cluster.num_blocks(); dst_cta_id++) {
            int *dst_shmem;
            if (dst_cta_id == 0) {
                dst_shmem = kv_indices;
            } else {
                dst_shmem = cluster.map_shared_rank(kv_indices, dst_cta_id);
            }
            for (int j = 0; j < C_PER_BLOCK; j++) {
                int c = center_idx[dst_cta_id * C_PER_BLOCK + j];
                for (int i = tid; i < CLEN; i += blockDim.x) {
                    int global_idx = c * CLEN + i;
                    int out_offset = j * CLEN + i;
                    dst_shmem[out_offset] = global_idx;
                }
            }
        }
    }

    cluster.sync();
    // ---------------- STAGE 2: flash decoding ----------------
    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    block.sync();

    // Preload kv_cache
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[0 + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
            &k_cache[(kv_indices[weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
            true
        );
    }
    cp_async_commit_group();
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
            &v_cache[(kv_indices[weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
            true
        );
    }
    cp_async_commit_group();

    // mainloop
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
                &k_cache[(kv_indices[id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
                true
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();

        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
                qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
        // For filled zeros
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
            reg_reduce[j] = reg_reduce[j] * scale;
        }
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
                &v_cache[(kv_indices[id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
                true
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
                reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
            }
        }
    }
    // end: mainloop

    // epilogue
    cp_async_wait_group<1>();
    block.sync();

    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        qk[j] = ptx_exp2(qk[j] - local_max);
        local_sum += qk[j];
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }

    cp_async_wait_group<0>();
    block.sync();

    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
            reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(ptx_exp2(pre_max - local_max))), __hmul(reg_input[d], __float2half(scale)));
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    // ClusterReduce: local_max
    for (int i = 1; i < cluster.num_blocks(); i++) {
        if (tid == 0) {
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
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // ClusterReduce: local_sum
    for (int i = 1; i < cluster.num_blocks(); i++) {
        if (tid == 0) {
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
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(cluster_local_sum);
    }
    if(tid < NUM_THREAD_PER_ROW_2) {
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            local_output[tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    block.sync();

    atomicAdd(&output[head_id * HEAD_DIM + tid], local_output[tid]);
}

// ######################################################
// #  Baseline 3: topk_attn block specialization kernel #
// ######################################################

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) topk_attn_block_specialization_kernel(
    half* output,        // [H, D]
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const __half* __restrict__ q,
    const __half* __restrict__ centers) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    // common indices
    const uint32_t lane_id = tid % 32; 
    const uint32_t warp_id = tid / 32;
    const uint32_t tile_row = tid / 16;
    const uint32_t tile_col = tid % 16;
    const uint32_t input_idx = (lane_id % 16) * 8;                // head_dim
    const uint32_t cluster_head_idx = head_id * HEAD_DIM;

    // Init shared memory
    // Flash-decoding blocks only need KV_DIM_PER_BLOCK_BS indices
    // Gemv-topk block needs full TOPC * TOPK_PER_CLUSTER space
    __shared__ __align__(16) int kv_indices[KV_DIM_PER_BLOCK_BS];
    __shared__ __align__(16) int lock;
    volatile int* lock_ptr = &lock;
    int *dst_shmem;
    
    if (cluster_block_id != CLUSTER_SIZE - 1) {
        lock = 0;  // Flash-decoding blocks initialize lock to 0
    }

    extern __shared__ uint8_t shmem_base[];

    if (cluster_block_id == CLUSTER_SIZE - 1) {
    // begin gemv_topk block 
    // Init shared memory - gemv-topk block uses dynamic shared memory for full kv_indices array
    half* k_buffer = reinterpret_cast<half*>(shmem_base);
    float* center_vals = reinterpret_cast<float*>(k_buffer + GEMV_SHARED_K_BUFFER_ELEMS);
    int* center_idx = reinterpret_cast<int*>(center_vals + CSZ);

    // Init registers
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float __align__(16) qk[GEMV_DEC_TILE];

    // indices for CSZ q @ centers^T
    const uint32_t weight_idx_0 = warp_id * GEMV_NUM_ROW_PER_WARP + lane_id / 16 * GEMV_DEC_TILE; // seq_len. tile_size = TILE_SIZE

    // for cub::BlockRadixSort
    constexpr int BLOCK_THREADS = 128;
    constexpr int CENTERS_PER_THREAD = CSZ / BLOCK_THREADS;
    using CenterRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, CENTERS_PER_THREAD, int>;
    __shared__ typename CenterRadixSort::TempStorage center_sort_storage;
    
    // Load q into reg_input
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[cluster_head_idx + input_idx]);

    // Phase 1: 使用传入的聚类中心，直接计算每个中心分数  score_c = dot(q, center[h,c,:])
    // preload centers
    for (int i = 0; i < GEMV_DEC_TILE; i++) {
        cp_async_pred_load_128b(
            &k_buffer[(weight_idx_0 + i) * HEAD_DIM + input_idx],
            &centers[head_id * CSZ * HEAD_DIM + (weight_idx_0 + i) * HEAD_DIM + input_idx],
            (weight_idx_0 + i < CSZ)
        );
    }
    cp_async_commit_group();
    // main loop
    for (int tile_id = 1; tile_id < ((CSZ + GEMV_TILE_SIZE - 1) / GEMV_TILE_SIZE); tile_id++) {
        // commit current stage cp.async load
        for (int i = 0; i < GEMV_DEC_TILE; i++) {
            cp_async_pred_load_128b(
                &k_buffer[(tile_id % 2) * GEMV_TILE_SIZE * HEAD_DIM + (weight_idx_0 + i) * HEAD_DIM + input_idx],
                &centers[head_id * CSZ * HEAD_DIM + (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx],
                (tile_id * GEMV_TILE_SIZE + weight_idx_0 + i < CSZ)
            );
        }
        cp_async_commit_group();
        // wait for last cp.async load
        cp_async_wait_group<1>();
        __syncthreads();
        // consume last cp.async buffer
        for (int i = 0; i < GEMV_DEC_TILE; i++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx]);
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
    *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((last_tile_id - 1) % 2) * GEMV_TILE_SIZE + weight_idx_0 + i) * HEAD_DIM + input_idx]);
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

    // write to other cta's shmem
    for (int dst_cta_id = 0; dst_cta_id < cluster.num_blocks() - 1; dst_cta_id++) {
        dst_shmem = (int *)cluster.map_shared_rank(&kv_indices, dst_cta_id);
        // Each dst_cta_id should receive KV_DIM_PER_BLOCK_BS indices
        for (int j = 0; j < C_PER_BLOCK_BS; j++) {
            int c = center_idx[dst_cta_id * C_PER_BLOCK_BS + j];
            for (int i = tid; i < CLEN; i += blockDim.x) {
                int global_idx = c * CLEN + i;
                int out_offset = j * CLEN + i;
                dst_shmem[out_offset] = global_idx;
            }
        }
        block.sync();
        if (tid == 0) {
            volatile int* dst_lock = (volatile int*)cluster.map_shared_rank(&lock, dst_cta_id);
            *dst_lock = 1;
        }
        block.sync();
    }

    // end gemv_topk block
    } else {
    // begin flash-decoding blocks

    // wait for lock to release
    while (*lock_ptr == 0) {
        __nanosleep(1);
    };
    // Print or not will affect final output. NOTE: fixed after dereference lock with lock_ptr defined as below
    // volatile int* lock_ptr = &lock;

    // Debug print
#ifdef DEBUG
    if (head_id == 0 && cluster_block_id == 0 && tid == 0) {
        printf("baseline 3 head0 cluster_block_id=0 kv_indices[0..15]: ");
        for (int i = 0; i < 16; ++i) {
            printf("%d ", kv_indices[i]);
        }
        printf("\n");
        printf("baseline 3 head0 cluster_block_id=0 KV_DIM_PER_BLOCK_BS=%d\n", KV_DIM_PER_BLOCK_BS);
    }
    if (head_id == 0 && cluster_block_id == 1 && tid == 0) {
        printf("baseline 3 head0 cluster_block_id=1 kv_indices[0..15]: ");
        for (int i = 0; i < 16; ++i) {
            printf("%d ", kv_indices[i]);
        }
        printf("\n");
    }
#endif

    block.sync();
    // Init shared memory
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_output = weight + 2 * TMA_LOAD_ONCE * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(local_output + HEAD_DIM);
    __shared__ float cluster_local_sum, cluster_local_max;

    // Init registers
    float local_sum = 0.0, local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    float __align__(16) qk[DEC_TILE];

    // indices for flash-decoding
    const uint32_t weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    // CRITICAL FIX: Load q into reg_input for flash-decoding computation!
    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[cluster_head_idx + input_idx]);
    block.sync();

    // Preload kv_cache - flash-decoding blocks use their local kv_indices directly (no offset needed)
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[0 + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
            &k_cache[(kv_indices[weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
            true
        );
    }
    cp_async_commit_group();
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
            &v_cache[(kv_indices[weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
            true
        );
    }
    cp_async_commit_group();

    // mainloop
    for (int id = 1; id < KV_DIM_PER_BLOCK_BS / TMA_LOAD_ONCE_ATTN; id++) {
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
                &k_cache[(kv_indices[id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
                true
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();

        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
                qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
            reg_reduce[j] = reg_reduce[j] * scale;
        }
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx], 
                &v_cache[(kv_indices[id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx],
                true
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
                reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
            }
        }
    }
    // end: mainloop

    // epilogue
    cp_async_wait_group<1>();
    block.sync();

    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK_BS / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        qk[j] = ptx_exp2(qk[j] - local_max);
        local_sum += qk[j];
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }

    cp_async_wait_group<0>();
    block.sync();

    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK_BS / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
            reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(ptx_exp2(pre_max - local_max))), __hmul(reg_input[d], __float2half(scale)));
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    // ClusterReduce: local_max
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            int dst_cta = (cluster_block_id + i) % (cluster.num_blocks() - 1);
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
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // ClusterReduce: local_sum
    for (int i = 1; i < (cluster.num_blocks() - 1); i++) {
        if (tid == 0) {
            int dst_cta = (cluster_block_id + i) % (cluster.num_blocks() - 1);
            dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(cluster_local_sum);
    }
    if(tid < NUM_THREAD_PER_ROW_2) {
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            local_output[tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    block.sync();

    atomicAdd(&output[head_id * HEAD_DIM + tid], local_output[tid]);
    // end flash-decoding block
    }
}

// ---------------- Main: generate indices then decode ----------------
int main(int argc, char** argv) {
	printf("Topk-attn Problem Size: HEAD_NUM=%d HEAD_DIM=%d CSZ=%d CLEN=%d OUT_PER_HEAD(SEQ_LEN)=%d FULL_KV_SEQ_LEN=%d\n",
				 HEAD_NUM, HEAD_DIM, CSZ, CLEN, OUT_PER_HEAD, FULL_KV_SEQ_LEN);

	// Host buffers
	size_t kv_elems = (size_t)FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM;
	size_t q_elems  = (size_t)HEAD_NUM * HEAD_DIM;
	size_t center_elems = (size_t)HEAD_NUM * CSZ * HEAD_DIM;
	std::vector<half> h_k(kv_elems), h_v(kv_elems), h_q(q_elems), h_centers(center_elems);
    std::vector<half> h_out(HEAD_NUM * HEAD_DIM), h_out_fused(HEAD_NUM * HEAD_DIM), h_out_bs(HEAD_NUM * HEAD_DIM);
	std::mt19937 rng(12345);
	std::normal_distribution<float> noise(0.0f, 0.1f);
	std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

	// centers
	for (int h = 0; h < HEAD_NUM; ++h) for (int c = 0; c < CSZ; ++c) {
		size_t base = ((size_t)h * CSZ + c) * HEAD_DIM;
		for (int d = 0; d < HEAD_DIM; ++d) h_centers[base + d] = __float2half(uni(rng));
	}
	// kv around its center, v identical to k for simplicity
	for (int c = 0; c < CSZ; ++c) for (int t = 0; t < CLEN; ++t) {
		size_t seq_idx = (size_t)c * CLEN + t;
		for (int h = 0; h < HEAD_NUM; ++h) {
			size_t kv_base = ((size_t)seq_idx * HEAD_NUM + h) * HEAD_DIM;
			size_t cen_base = ((size_t)h * CSZ + c) * HEAD_DIM;
			for (int d = 0; d < HEAD_DIM; ++d) {
				float v = __half2float(h_centers[cen_base + d]) + noise(rng);
				h_k[kv_base + d] = __float2half(v);
				h_v[kv_base + d]  = __float2half(v);
			}
		}
	}
	// q random
	for (int h = 0; h < HEAD_NUM; ++h) for (int d = 0; d < HEAD_DIM; ++d) h_q[h*HEAD_DIM + d] = __float2half(uni(rng));

	// Device buffers
	half *d_k=nullptr, *d_v=nullptr, *d_q=nullptr, *d_centers=nullptr; int *d_kv_indices=nullptr; half *d_out=nullptr;
    // for baseline 2&3 output.
    half *d_out_2=nullptr, *d_out_3=nullptr;
	CUDA_CHECK(cudaMalloc(&d_k, sizeof(half) * kv_elems));
	CUDA_CHECK(cudaMalloc(&d_v,  sizeof(half) * kv_elems));
	CUDA_CHECK(cudaMalloc(&d_q,  sizeof(half) * q_elems));
	CUDA_CHECK(cudaMalloc(&d_centers, sizeof(half) * center_elems));
	CUDA_CHECK(cudaMalloc(&d_kv_indices, sizeof(int) * (size_t)HEAD_NUM * OUT_PER_HEAD));
	CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
	CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(half) * kv_elems, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_v,  h_v.data(),  sizeof(half) * kv_elems, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_q,  h_q.data(),  sizeof(half) * q_elems,  cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_centers, h_centers.data(), sizeof(half) * center_elems, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(d_out, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    // for baseline 2
	CUDA_CHECK(cudaMalloc(&d_out_2, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
	CUDA_CHECK(cudaMemset(d_out_2, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    // for baseline 3
	CUDA_CHECK(cudaMalloc(&d_out_3, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
	CUDA_CHECK(cudaMemset(d_out_3, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));

	// ---- Baseline 1 for reference ----
    // grid & block & dynamic shared memory config
	int dev = 0; cudaDeviceProp prop{}; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    // topk kernel
	dim3 grid_topk(HEAD_NUM * CLUSTER_SIZE), block_topk(128);
    uint32_t gemv_topk_shmem_bytes = GEMV_SHARED_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(gemv_topk_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(gemv_topk_shmem_bytes)));
    // attn kernel
	dim3 grid_dec(HEAD_NUM * CLUSTER_SIZE), block_dec(BLOCK_SIZE);
	uint32_t max_shmem_size = (2 * TMA_LOAD_ONCE * HEAD_DIM + HEAD_DIM) * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float);
	CUDA_TRY(cudaFuncSetAttribute(MHAFlashDecodeKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
	CUDA_TRY(cudaFuncSetAttribute(MHAFlashDecodeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size));
#ifndef DEBUG
    printf("---- Baseline 1 (topk kernel + attn kernel) latency test ----\n");
    int warmup_combo = 5, iters_combo = 50; float ms_combo = 0.f;
    for (int i = 0; i < warmup_combo; ++i) {
        gemv_topk_cluster_kernel<<<grid_topk, block_topk, gemv_topk_shmem_bytes>>>(d_q, d_centers, d_kv_indices);
        MHAFlashDecodeKernel<<<grid_dec, block_dec, max_shmem_size>>>(d_out, d_q, d_k, d_v, d_kv_indices);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t st_combo, ed_combo; cudaEventCreate(&st_combo); cudaEventCreate(&ed_combo);
    cudaEventRecord(st_combo);
    for (int i = 0; i < iters_combo; ++i) {
        gemv_topk_cluster_kernel<<<grid_topk, block_topk, gemv_topk_shmem_bytes>>>(d_q, d_centers, d_kv_indices);
        MHAFlashDecodeKernel<<<grid_dec, block_dec, max_shmem_size>>>(d_out, d_q, d_k, d_v, d_kv_indices);
    }
    cudaEventRecord(ed_combo); cudaEventSynchronize(ed_combo); cudaEventElapsedTime(&ms_combo, st_combo, ed_combo);
    printf("baseline 1 latency: %.3f us\n", (ms_combo / iters_combo) * 1000.0f);

	int warmup_topk = 5, iters_topk = 20; float ms_topk = 0.f;
    for (int i = 0; i < warmup_topk; ++i) gemv_topk_cluster_kernel<<<grid_topk, block_topk, gemv_topk_shmem_bytes>>>(d_q, d_centers, d_kv_indices);
	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEvent_t st1, ed1; cudaEventCreate(&st1); cudaEventCreate(&ed1);
	cudaEventRecord(st1);
    for (int i = 0; i < iters_topk; ++i) gemv_topk_cluster_kernel<<<grid_topk, block_topk, gemv_topk_shmem_bytes>>>(d_q, d_centers, d_kv_indices);
	cudaEventRecord(ed1); cudaEventSynchronize(ed1); cudaEventElapsedTime(&ms_topk, st1, ed1);
	printf("gemv_topk (indices build) latency: %.3f us\n", (ms_topk / iters_topk) * 1000.0f);

	int warmup_dec = 10, iters_dec = 50; float ms_dec = 0.f;
	for (int i = 0; i < warmup_dec; ++i) {
		MHAFlashDecodeKernel<<<grid_dec, block_dec, max_shmem_size>>>(d_out, d_q, d_k, d_v, d_kv_indices);
	}
	CUDA_CHECK(cudaDeviceSynchronize());
	cudaEvent_t st2, ed2; cudaEventCreate(&st2); cudaEventCreate(&ed2);
	cudaEventRecord(st2);
	for (int i = 0; i < iters_dec; ++i) {
		MHAFlashDecodeKernel<<<grid_dec, block_dec, max_shmem_size>>>(d_out, d_q, d_k, d_v, d_kv_indices);
	}
	cudaEventRecord(ed2); cudaEventSynchronize(ed2); cudaEventElapsedTime(&ms_dec, st2, ed2);
	printf("flash decode latency: %.3f us\n", (ms_dec / iters_dec) * 1000.0f);
#endif


    // ---- Baseline 2 ----
    dim3 grid_fused(HEAD_NUM * CLUSTER_SIZE), block_fused(BLOCK_SIZE);
    uint32_t fused_shmem_bytes = std::max<uint32_t>(
        GEMV_SHARED_BYTES,
        (2 * TMA_LOAD_ONCE * HEAD_DIM + HEAD_DIM) * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float));
    CUDA_TRY(cudaFuncSetAttribute(topk_attn_fused_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_TRY(cudaFuncSetAttribute(topk_attn_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(fused_shmem_bytes)));
#ifndef DEBUG
    printf("---- Baseline 2 (topk-attn directly fused kernel) latency test ----\n");
    int warmup_fused = 5, iters_fused = 50; float ms_fused = 0.f;
    CUDA_CHECK(cudaMemset(d_out_2, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    for (int i = 0; i < warmup_fused; ++i) {
        topk_attn_fused_kernel<<<grid_fused, block_fused, fused_shmem_bytes>>>(d_out_2, d_k, d_v, d_q, d_centers);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t st_fused, ed_fused; cudaEventCreate(&st_fused); cudaEventCreate(&ed_fused);
    CUDA_CHECK(cudaMemset(d_out_2, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    cudaEventRecord(st_fused);
    for (int i = 0; i < iters_fused; ++i) {
        topk_attn_fused_kernel<<<grid_fused, block_fused, fused_shmem_bytes>>>(d_out_2, d_k, d_v, d_q, d_centers);
    }
    cudaEventRecord(ed_fused); cudaEventSynchronize(ed_fused); cudaEventElapsedTime(&ms_fused, st_fused, ed_fused);
    printf("baseline 2 latency: %.3f us\n", (ms_fused / iters_fused) * 1000.0f);
#endif

    // ---- Baseline 3 ----
    dim3 grid_bs(HEAD_NUM * CLUSTER_SIZE), block_bs(BLOCK_SIZE);
    uint32_t bs_shmem_bytes = std::max<uint32_t>(
        GEMV_SHARED_BYTES,
        (2 * TMA_LOAD_ONCE * HEAD_DIM + HEAD_DIM) * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float));
    CUDA_TRY(cudaFuncSetAttribute(topk_attn_block_specialization_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_TRY(cudaFuncSetAttribute(topk_attn_block_specialization_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bs_shmem_bytes)));
#ifndef DEBUG
    printf("---- Baseline 3 (topk-attn block specialization kernel) latency test ----\n");
    // Commented for debugging.
    int warmup_bs = 5, iters_bs = 50; float ms_bs = 0.f;
    CUDA_CHECK(cudaMemset(d_out_3, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    topk_attn_block_specialization_kernel<<<grid_bs, block_bs, bs_shmem_bytes>>>(d_out_3, d_k, d_v, d_q, d_centers);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < warmup_bs; ++i) {
        topk_attn_block_specialization_kernel<<<grid_bs, block_bs, bs_shmem_bytes>>>(d_out_3, d_k, d_v, d_q, d_centers);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t st_bs, ed_bs; cudaEventCreate(&st_bs); cudaEventCreate(&ed_bs);
    CUDA_CHECK(cudaMemset(d_out_3, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    cudaEventRecord(st_bs);
    for (int i = 0; i < iters_bs; ++i) {
        topk_attn_block_specialization_kernel<<<grid_bs, block_bs, bs_shmem_bytes>>>(d_out_3, d_k, d_v, d_q, d_centers);
    }
    cudaEventRecord(ed_bs); cudaEventSynchronize(ed_bs); cudaEventElapsedTime(&ms_bs, st_bs, ed_bs);
    printf("baseline 3 latency: %.3f us\n", (ms_bs / iters_bs) * 1000.0f);
#endif

    // ---- Correctness check ----
#ifdef DEBUG
    // run baseline 1
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    gemv_topk_cluster_kernel<<<grid_topk, block_topk, gemv_topk_shmem_bytes>>>(d_q, d_centers, d_kv_indices);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("baseline 1 gemv_topk_cluster_kernel error: %s\n", cudaGetErrorString(err1));
    }
    MHAFlashDecodeKernel<<<grid_dec, block_dec, max_shmem_size>>>(d_out, d_q, d_k, d_v, d_kv_indices);
    CUDA_CHECK(cudaDeviceSynchronize());
    err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("baseline 1 MHAFlashDecodeKernel error: %s\n", cudaGetErrorString(err1));
    }
    
    // run baseline 2
    CUDA_CHECK(cudaMemset(d_out_2, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    topk_attn_fused_kernel<<<grid_fused, block_fused, fused_shmem_bytes>>>(d_out_2, d_k, d_v, d_q, d_centers);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("baseline 2 error: %s\n", cudaGetErrorString(err2));
    }
    
    // run baseline 3
    CUDA_CHECK(cudaMemset(d_out_3, 0, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM));
    topk_attn_block_specialization_kernel<<<grid_bs, block_bs, bs_shmem_bytes>>>(d_out_3, d_k, d_v, d_q, d_centers);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("baseline 3 error: %s\n", cudaGetErrorString(err3));
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_fused.data(), d_out_2, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_bs.data(), d_out_3, sizeof(half) * (size_t)HEAD_NUM * HEAD_DIM, cudaMemcpyDeviceToHost));
    // Inspect top-k indices produced by baseline 1
    std::vector<int> h_kv_indices(HEAD_NUM * OUT_PER_HEAD);
    CUDA_CHECK(cudaMemcpy(
        h_kv_indices.data(),
        d_kv_indices,
        sizeof(int) * static_cast<size_t>(HEAD_NUM) * OUT_PER_HEAD,
        cudaMemcpyDeviceToHost));
    printf("【baseline 1 head0 kv_indices[0..15]】: \n");
    for (int i = 0; i < 16; ++i) {
        printf("%d ", h_kv_indices[i]);
    }
    printf("\n");

    printf("【baseline 1 head0 output[0..15]】: ");
    printf("\n");
    for (int i = 0; i < 16; ++i) {
        printf("%f ", __half2float(h_out[i]));
    }
    printf("\n");
    printf("【baseline 2 head0 output[0..15]】: ");
    printf("\n");
    for (int i = 0; i < 16; ++i) {
        printf("%f ", __half2float(h_out_fused[i]));
    }
    printf("\n");
    printf("【baseline 3 head0 output[0..15]】: ");
    printf("\n");
    for (int i = 0; i < 16; ++i) {
        printf("%f ", __half2float(h_out_bs[i]));
    }
    printf("\n");

    float max_diff = 0.0f;
    int max_diff_idx = 0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        float diff = fabsf(__half2float(h_out[i]) - __half2float(h_out_fused[i]));
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    printf("max abs diff between baseline_1&2 outputs: %.6f at index %d (bl1=%.6f, bl2=%.6f)\n", 
           max_diff, max_diff_idx, __half2float(h_out[max_diff_idx]), __half2float(h_out_fused[max_diff_idx]));
    
    max_diff = 0.0f;
    max_diff_idx = 0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        float diff = fabsf(__half2float(h_out[i]) - __half2float(h_out_bs[i]));
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    printf("max abs diff between baseline_1&3 outputs: %.6f at index %d (bl1=%.6f, bl3=%.6f)\n", 
           max_diff, max_diff_idx, __half2float(h_out[max_diff_idx]), __half2float(h_out_bs[max_diff_idx]));
#endif

	// Cleanup
    cudaFree(d_out); cudaFree(d_kv_indices);
    cudaFree(d_centers); cudaFree(d_q); cudaFree(d_v); cudaFree(d_k);
    cudaFree(d_out_2);
    cudaFree(d_out_3);
	printf("Done.\n");
	return 0;
}

