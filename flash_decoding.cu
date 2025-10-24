// nvcc -O3 -arch=sm_120a -o flash_decoding flash_decoding.cu && ./flash_decoding
#include <iostream>
#include <cstdio>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <random>
#include <math_constants.h> 
#include <cuda.h>  
#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// ---- Debug helpers ----
#define CUDA_TRY(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA-ERR] %s failed at %s:%d -> %s (%d)\n", #call, __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e); \
    } \
} while(0)

#define CUDA_LAUNCH_CHECK(tag) do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[KERNEL-ERR] after %s -> %s (%d)\n", tag, cudaGetErrorString(_e), (int)_e); \
    } \
} while(0)

// setup
#define HEAD_DIM 128    
#define HEAD_NUM 32     
#define SEQ_LEN 2560
#define FULL_KV_SEQ_LEN 40960
#define CLUSTER_SIZE 4 
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 

#define NUM_WARPS 4
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 

#define TMA_LOAD_ONCE 32 // 8 16 32 64 128 256
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

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 0.1);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(0.01f);
        }   
    }   
}

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

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
    const uint32_t batch_id         = grid.cluster_rank() / HEAD_NUM;
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

    // Init barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[4];
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[2], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[3], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
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
            &k_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
            true
        );
    }
    cp_async_commit_group();
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
            &v_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
            true
        );
    }
    cp_async_commit_group();

    // mainloop
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
                &k_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
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
                &v_cache[(kv_indices[head_id * SEQ_LEN + KV_DIM_PER_BLOCK * cluster_block_id + j]) * HEAD_NUM * HEAD_DIM + cluster_head_idx + input_idx_2],
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
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
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
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // ClusterReduce: local_sum
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
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

    // // ClusterReduce
    // size = HEAD_DIM * sizeof(half);
    // src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[2 * HEAD_DIM]));
    // dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        // size, tid, HEAD_DIM, cluster_block_id,  
        // src_addr, dst_addr, bar_ptr, 
        // neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);
    atomicAdd(&output[head_id * HEAD_DIM + tid], local_output[tid]);
}

int main(int argc, char** argv) {
    printf("HEAD_DIM=%d HEAD_NUM=%d SEQ_LEN=%d FULL_KV_SEQ_LEN=%d CLUSTER_SIZE=%d\n", HEAD_DIM, HEAD_NUM, SEQ_LEN, FULL_KV_SEQ_LEN, CLUSTER_SIZE);
    uint32_t max_shmem_size = (2 * TMA_LOAD_ONCE * HEAD_DIM + HEAD_DIM) * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float);
    printf("Requested dynamic shared memory: %u bytes\n", max_shmem_size);
    int dev = 0; cudaDeviceProp prop{}; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    printf("Device %d name=%s sharedMemPerBlock=%zu sharedMemPerBlockOptin=%zu maxThreadsPerBlock=%d\n",
        dev, prop.name, (size_t)prop.sharedMemPerBlock, (size_t)prop.sharedMemPerBlockOptin, prop.maxThreadsPerBlock);
    CUDA_TRY(cudaFuncSetAttribute(MHAFlashDecodeKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_TRY(cudaFuncSetAttribute(MHAFlashDecodeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size));

    // Init input
    half *h_q, *d_q;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    int *h_kv_indices, *d_kv_indices;
    h_q = new half[1 * HEAD_NUM * HEAD_DIM];
    h_k_cache = new half[FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_v_cache = new half[FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_kv_indices = new int[HEAD_NUM * SEQ_LEN];

    fill_matrix(h_q, 1 * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_k_cache, FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_v_cache, FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM);
    // Fill h_kv_indices with random integers in [0, FULL_KV_SEQ_LEN-1]
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, FULL_KV_SEQ_LEN - 1);
        for (int i = 0; i < HEAD_NUM * SEQ_LEN; ++i) {
            h_kv_indices[i] = dist(gen);
        }
    }

    size_t sz_q = sizeof(half) * 1ULL * HEAD_NUM * HEAD_DIM;
    size_t sz_cache = sizeof(half) * 1ULL * FULL_KV_SEQ_LEN * HEAD_NUM * HEAD_DIM;
    size_t sz_idx = sizeof(int) * 1ULL * HEAD_NUM * SEQ_LEN;
    printf("Alloc sizes: q=%zu cache=%zu idx=%zu\n", sz_q, sz_cache, sz_idx);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_q), sz_q));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sz_cache));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sz_cache));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_kv_indices), sz_idx));
    CUDA_TRY(cudaMemcpy(reinterpret_cast<void*>(d_q), h_q, sz_q, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sz_cache, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sz_cache, cudaMemcpyHostToDevice));
    // Debug: first few indices
    printf("kv_indices[0..7]: ");
    for (int i = 0; i < 8 && i < SEQ_LEN; ++i) printf("%d ", h_kv_indices[i]);
    printf("\n");
    CUDA_TRY(cudaMemcpy(reinterpret_cast<void*>(d_kv_indices), h_kv_indices, sz_idx, cudaMemcpyHostToDevice));

    half* h_output, *d_output;
    h_output = new half[1 * HEAD_NUM * HEAD_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * 1 * HEAD_NUM * HEAD_DIM);
    
    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);
    printf("Launching kernel: grid=(%u) block=(%u) dyn_shmem=%u bytes\n", grid.x, block.x, max_shmem_size);

    int wmup = 50;
    int test = 100;
    for (int i = 0; i < wmup; i++) {
        MHAFlashDecodeKernel<<<grid, block, max_shmem_size>>>(
            d_output,
            d_q,
            d_k_cache,
            d_v_cache,
            d_kv_indices
        );
        CUDA_LAUNCH_CHECK("warmup");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after warmup: %s (%d)\n", cudaGetErrorString(err), (int)err);
    }
    CUDA_TRY(cudaDeviceSynchronize());

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        MHAFlashDecodeKernel<<<grid, block, max_shmem_size>>>(
            d_output,
            d_q,
            d_k_cache,
            d_v_cache,
            d_kv_indices
        );
        CUDA_LAUNCH_CHECK("timed");
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HEAD_NUM * HEAD_DIM, cudaMemcpyDeviceToHost);

    return 0;
}