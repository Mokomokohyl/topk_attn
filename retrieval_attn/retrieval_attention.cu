/*
nvcc -O3 -arch=sm_120a -o test_ra retrieval_attention.cu && ./test_ra && rm ./test_ra
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <algorithm>
#include <stdio.h>
#include <chrono>
#include <cub/block/block_radix_sort.cuh>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Configuration
// ============================================================================
constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int SEQ_LEN = 8192;
constexpr int QUERY_LEN = 1;
constexpr int CLUSTER_SIZE = 5;
constexpr int QK_BLOCKS_PER_CLUSTER = 4;
constexpr int PV_BLOCKS_PER_CLUSTER = 1;

// Configurable TOPK per QK block
template<int TOPK_PER_BLOCK>
struct KernelConfig {
    static constexpr int TOPK = TOPK_PER_BLOCK;
    static constexpr int TOTAL_TOPK = TOPK * QK_BLOCKS_PER_CLUSTER;
    static constexpr int KEYS_PER_QK_BLOCK = SEQ_LEN / QK_BLOCKS_PER_CLUSTER;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int ITEMS_PER_THREAD = KEYS_PER_QK_BLOCK / BLOCK_SIZE;
};

// ============================================================================
// Baseline Attention Kernel (No Retrieval)
// ============================================================================
__global__ void baseline_attention_kernel(
    const half* __restrict__ query,      // [1, NUM_HEADS, HEAD_DIM]
    const half* __restrict__ key_cache,  // [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    const half* __restrict__ value_cache,// [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    half* __restrict__ output            // [1, NUM_HEADS, HEAD_DIM]
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ half shared_mem[];
    half* s_scores = shared_mem;
    half* s_output = &shared_mem[SEQ_LEN];
    __shared__ half s_reduce[256];
    
    const half* q = query + head_idx * HEAD_DIM;
    const half* k_base = key_cache + head_idx * HEAD_DIM;
    const half* v_base = value_cache + head_idx * HEAD_DIM;
    
    // Compute QK scores
    for (int seq_idx = tid; seq_idx < SEQ_LEN; seq_idx += block_size) {
        const half* k = k_base + seq_idx * NUM_HEADS * HEAD_DIM;
        half score = __float2half(0.0f);
        for (int d = 0; d < HEAD_DIM; d++) {
            score = __hadd(score, __hmul(q[d], k[d]));
        }
        s_scores[seq_idx] = score;
    }
    __syncthreads();
    
    // Find max
    half max_score = __float2half(-65504.0f);
    for (int i = tid; i < SEQ_LEN; i += block_size) {
        max_score = __hmax(max_score, s_scores[i]);
    }
    
    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_score = __hmax(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    }
    
    // Write warp results to shared memory
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        s_reduce[warp_id] = max_score;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid == 0) {
        max_score = __float2half(-65504.0f);
        int num_warps = (block_size + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            max_score = __hmax(max_score, s_reduce[i]);
        }
        s_reduce[0] = max_score;
    }
    __syncthreads();
    max_score = s_reduce[0];
    
    // Compute exp and sum
    half sum = __float2half(0.0f);
    for (int i = tid; i < SEQ_LEN; i += block_size) {
        half exp_val = hexp(__hsub(s_scores[i], max_score));
        s_scores[i] = exp_val;
        sum = __hadd(sum, exp_val);
    }
    
    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        s_reduce[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid == 0) {
        sum = __float2half(0.0f);
        int num_warps = (block_size + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            sum = __hadd(sum, s_reduce[i]);
        }
        s_reduce[0] = sum;
    }
    __syncthreads();
    sum = s_reduce[0];
    
    // Normalize
    for (int i = tid; i < SEQ_LEN; i += block_size) {
        s_scores[i] = __hdiv(s_scores[i], sum);
    }
    __syncthreads();
    
    // Compute P @ V
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        half acc = __float2half(0.0f);
        for (int seq_idx = 0; seq_idx < SEQ_LEN; seq_idx++) {
            const half* v = v_base + seq_idx * NUM_HEADS * HEAD_DIM;
            acc = __hadd(acc, __hmul(s_scores[seq_idx], v[d]));
        }
        s_output[d] = acc;
    }
    __syncthreads();
    
    // Write output
    half* out = output + head_idx * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += block_size) {
        out[d] = s_output[d];
    }
}

// ============================================================================
// TopK Selection Helper
// ============================================================================
template<int K>
__device__ void select_topk_indices(
    const half* scores,
    int num_scores,
    int* topk_indices,
    half* topk_scores,
    half* merge_scores_workspace,  // Shared memory workspace
    int* merge_indices_workspace,  // Shared memory workspace
    int tid,
    int block_size
) {
    // Simple selection sort for small K
    // Each thread maintains local topk
    half local_scores[K];
    int local_indices[K];
    
    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_scores[i] = __float2half(-65504.0f);
        local_indices[i] = -1;
    }
    
    // Each thread processes its elements
    for (int i = tid; i < num_scores; i += block_size) {
        half score = scores[i];
        
        // Insert into local topk if better than minimum
        if (__hgt(score, local_scores[K-1])) {
            local_scores[K-1] = score;
            local_indices[K-1] = i;
            
            // Bubble up
            for (int j = K-1; j > 0; j--) {
                if (__hgt(local_scores[j], local_scores[j-1])) {
                    half tmp_score = local_scores[j];
                    int tmp_idx = local_indices[j];
                    local_scores[j] = local_scores[j-1];
                    local_indices[j] = local_indices[j-1];
                    local_scores[j-1] = tmp_score;
                    local_indices[j-1] = tmp_idx;
                }
            }
        }
    }
    
    // Write to shared memory for merging
    #pragma unroll
    for (int i = 0; i < K; i++) {
        merge_scores_workspace[tid * K + i] = local_scores[i];
        merge_indices_workspace[tid * K + i] = local_indices[i];
    }
    __syncthreads();
    
    // Single thread merges all local topks
    if (tid == 0) {
        int num_candidates = min(block_size, num_scores / block_size + 1) * K;
        
        for (int k = 0; k < K; k++) {
            int best_idx = -1;
            half best_score = __float2half(-65504.0f);
            
            for (int i = 0; i < num_candidates; i++) {
                if (merge_indices_workspace[i] != -1 && __hgt(merge_scores_workspace[i], best_score)) {
                    best_score = merge_scores_workspace[i];
                    best_idx = i;
                }
            }
            
            if (best_idx != -1) {
                topk_scores[k] = best_score;
                topk_indices[k] = merge_indices_workspace[best_idx];
                merge_indices_workspace[best_idx] = -1; // Mark as used
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// Retrieval Attention Kernel with DSM and Cluster
// ============================================================================
template<int TOPK_PER_BLOCK>
__cluster_dims__(1, CLUSTER_SIZE, 1)
__global__ void retrieval_attention_kernel(
    const half* __restrict__ query,      // [1, NUM_HEADS, HEAD_DIM]
    const half* __restrict__ key_cache,  // [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    const half* __restrict__ value_cache,// [SEQ_LEN, NUM_HEADS, HEAD_DIM]
    half* __restrict__ output            // [1, NUM_HEADS, HEAD_DIM]
) {
    using Config = KernelConfig<TOPK_PER_BLOCK>;
    constexpr int TOPK = Config::TOPK;
    constexpr int TOTAL_TOPK = Config::TOTAL_TOPK;
    constexpr int KEYS_PER_BLOCK = Config::KEYS_PER_QK_BLOCK;
    constexpr int BLOCK_SIZE = Config::BLOCK_SIZE;
    constexpr int ITEMS_PER_THREAD = Config::ITEMS_PER_THREAD;
    
    // Cluster and block info
    cg::cluster_group cluster = cg::this_cluster();
    int cluster_rank = cluster.block_rank(); // 0-4 within cluster
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Determine role: QK blocks (0-3) or PV block (4)
    bool is_qk_block = (cluster_rank < QK_BLOCKS_PER_CLUSTER);
    bool is_pv_block = (cluster_rank == QK_BLOCKS_PER_CLUSTER);
    
    // Pointers for this head
    const half* q = query + head_idx * HEAD_DIM;
    const half* k_base = key_cache + head_idx * HEAD_DIM;
    const half* v_base = value_cache + head_idx * HEAD_DIM;
    
    // Shared memory layout
    extern __shared__ half shared_mem[];
    
    // PV block: Initialize ready counter at the start of shared memory
    if (is_pv_block) {
        if (tid == 0) {
            // Ready counter tracks how many QK blocks have finished
            volatile int* ready_counter = (int*)shared_mem;
            *ready_counter = 0;
        }
        __syncthreads();
    }
    
    // ALL blocks sync to ensure counter is initialized before anyone maps it
    cluster.sync();
    
    // QK blocks: compute scores and find topk
    if (is_qk_block) {
        // Offset shared memory to avoid counter (first int)
        // Use float for scores to support CUB sort
        float* s_scores = (float*)((int*)shared_mem + 1);
        int* s_indices = (int*)&s_scores[KEYS_PER_BLOCK];
        
        // CUB Sort Storage
        using BlockSort = cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;
        __shared__ typename BlockSort::TempStorage sort_storage;
        
        // Compute scores for this block's key range
        int key_start = cluster_rank * KEYS_PER_BLOCK;
        
        // Thread-local storage for sorting
        float thread_keys[ITEMS_PER_THREAD];
        int thread_values[ITEMS_PER_THREAD];
        
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            int local_idx = tid * ITEMS_PER_THREAD + item;
            int global_key_idx = key_start + local_idx;
            const half* k = k_base + global_key_idx * NUM_HEADS * HEAD_DIM;
            
            float score = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                score += __half2float(q[d]) * __half2float(k[d]);
            }
            thread_keys[item] = score;
            thread_values[item] = local_idx;
        }
        
        // Sort
        BlockSort(sort_storage).SortDescending(thread_keys, thread_values);
        
        // Store back to shared memory (only need top K)
        // We need to gather the top K from the sorted thread data
        // Since it's blocked striping, the top elements are distributed.
        // Wait, BlockRadixSort output is blocked arrangement.
        // Thread 0 has items [0, ITEMS_PER_THREAD-1] which are the largest?
        // Yes, for SortDescending, rank 0 has the largest elements.
        
        __syncthreads();
        
        // Write sorted results to shared memory to be picked up
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            int rank = tid * ITEMS_PER_THREAD + item;
            if (rank < TOPK) {
                s_scores[rank] = thread_keys[item];
                s_indices[rank] = thread_values[item];
            }
        }
        __syncthreads();
        
        // Map distributed shared memory to PV block (rank 4)
        // Map to location AFTER the ready counter
        half* dsm_base = (half*)cluster.map_shared_rank(shared_mem, QK_BLOCKS_PER_CLUSTER);
        half* dsm_scores = (half*)((int*)dsm_base + 1);  // Skip counter
        int* dsm_indices = (int*)&dsm_scores[TOTAL_TOPK];
        
        // Also map the ready counter
        volatile int* dsm_ready_counter = (volatile int*)dsm_base;
        
        // Write to PV block's memory at offset for this QK block
        int offset = cluster_rank * TOPK;
        for (int i = tid; i < TOPK; i += BLOCK_SIZE) {
            dsm_scores[offset + i] = __float2half(s_scores[i]);
        }
        __threadfence_block();  // Ensure scores are written
        
        // Write indices
        for (int i = tid; i < TOPK; i += BLOCK_SIZE) {
            int local_idx = s_indices[i];
            dsm_indices[offset + i] = key_start + local_idx;
        }
        __threadfence_block();  // Ensure indices are written
        
        // Signal completion: increment ready counter
        if (tid == 0) {
            atomicAdd_system((int*)dsm_ready_counter, 1);
        }
    }
    
    // PV block: wait for all QK blocks, then compute softmax and P@V
    if (is_pv_block) {
        volatile int* ready_counter = (volatile int*)shared_mem;
        
        // Wait until all QK blocks have finished (spin wait)
        if (tid == 0) {
            while (*ready_counter < QK_BLOCKS_PER_CLUSTER) {
                // Busy wait for all QK blocks to signal completion
            }
        }
        __syncthreads();
        
        // Now safe to read data written by QK blocks
        // Data layout: [counter][scores: TOTAL_TOPK][indices: TOTAL_TOPK][probs][output][reduction]
        half* s_all_scores = (half*)((int*)shared_mem + 1);
        int* s_all_indices = (int*)&s_all_scores[TOTAL_TOPK];
        half* s_probs = (half*)&s_all_indices[TOTAL_TOPK];
        half* s_output = &s_probs[TOTAL_TOPK];
        
        // Scores and indices are already in shared memory (written by QK blocks via DSM)
        
        // Simple softmax (no numerical stability as requested)
        half sum = __float2half(0.0f);
        for (int i = tid; i < TOTAL_TOPK; i += block_size) {
            half exp_val = hexp(s_all_scores[i]);
            s_probs[i] = exp_val;
            sum = __hadd(sum, exp_val);
        }
        
        // Reduce sum across threads
        __syncthreads();
        __shared__ half s_sum_reduce[32];
        
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, offset));
        }
        
        if (lane_id == 0) {
            s_sum_reduce[warp_id] = sum;
        }
        __syncthreads();
        
        // Final reduction
        if (tid == 0) {
            sum = __float2half(0.0f);
            int num_warps = (block_size + 31) / 32;
            for (int i = 0; i < num_warps; i++) {
                sum = __hadd(sum, s_sum_reduce[i]);
            }
            s_sum_reduce[0] = sum;
        }
        __syncthreads();
        sum = s_sum_reduce[0];
        
        // Normalize
        for (int i = tid; i < TOTAL_TOPK; i += block_size) {
            s_probs[i] = __hdiv(s_probs[i], sum);
        }
        __syncthreads();
        
        // Compute P @ V
        for (int d = tid; d < HEAD_DIM; d += block_size) {
            half acc = __float2half(0.0f);
            for (int i = 0; i < TOTAL_TOPK; i++) {
                int key_idx = s_all_indices[i];
                if (key_idx >= 0 && key_idx < SEQ_LEN) {
                    const half* v = v_base + key_idx * NUM_HEADS * HEAD_DIM;
                    acc = __hadd(acc, __hmul(s_probs[i], v[d]));
                }
            }
            s_output[d] = acc;
        }
        __syncthreads();
        
        // Write output
        half* out = output + head_idx * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += block_size) {
            out[d] = s_output[d];
        }
    }
    
    // ALL blocks sync at the end to ensure completion
    cluster.sync();
}

// ============================================================================
// Host Code
// ============================================================================

template<int TOPK_PER_BLOCK>
void run_retrieval_attention(
    const half* d_query,
    const half* d_key_cache,
    const half* d_value_cache,
    half* d_output,
    int num_iterations = 100
) {
    using Config = KernelConfig<TOPK_PER_BLOCK>;
    constexpr int BLOCK_SIZE = Config::BLOCK_SIZE;
    constexpr int TOPK = Config::TOPK;
    constexpr int TOTAL_TOPK = Config::TOTAL_TOPK;
    constexpr int KEYS_PER_BLOCK = Config::KEYS_PER_QK_BLOCK;
    
    // Calculate shared memory size
    // QK blocks: counter(int) + scores(float) + topk_indices(int) + CUB storage
    // Note: s_scores and s_indices are reused for output of sort
    // We need space for KEYS_PER_BLOCK floats and ints? No, CUB sorts in-place or uses temp storage.
    // But we need to store the initial scores.
    // s_scores: KEYS_PER_BLOCK * sizeof(float)
    // s_indices: KEYS_PER_BLOCK * sizeof(int) (actually we only need TOPK indices at the end, but CUB might need full array if we sort keys and values)
    // Wait, BlockRadixSort takes input arrays.
    // We can use register-based sort if we load everything into registers.
    // My implementation uses `thread_keys` and `thread_values` registers.
    // So we don't need full s_scores/s_indices in shared memory for the sort input!
    // We only need shared memory for CUB temp storage and for the final output (TOPK elements).
    // But wait, I used `s_scores` and `s_indices` to store the TOPK results to be picked up by the DSM writer.
    // So we need TOPK * sizeof(float) + TOPK * sizeof(int).
    
    using BlockSort = cub::BlockRadixSort<float, BLOCK_SIZE, Config::ITEMS_PER_THREAD, int>;
    
    size_t qk_shared_size = sizeof(int) +  // ready counter space
                            TOPK * sizeof(float) + // s_scores (output)
                            TOPK * sizeof(int) +   // s_indices (output)
                            sizeof(typename BlockSort::TempStorage); // CUB storage

    // PV blocks: counter(int) + all_scores + all_indices + probs + output + reduction
    size_t pv_shared_size = sizeof(int) +  // ready counter
                           TOTAL_TOPK * sizeof(half) +  // all_scores
                           TOTAL_TOPK * sizeof(int) +   // all_indices
                           TOTAL_TOPK * sizeof(half) +  // probs
                           HEAD_DIM * sizeof(half) +    // output
                           32 * sizeof(half);           // reduction workspace
    size_t shared_mem_size = max(qk_shared_size, pv_shared_size);
    
    printf("Retrieval Attention Config: TOPK=%d, TOTAL_TOPK=%d, BLOCK_SIZE=%d\n", 
           TOPK, TOTAL_TOPK, BLOCK_SIZE);
    printf("Shared memory per block: %zu bytes (%.1f KB)\n", shared_mem_size, shared_mem_size / 1024.0f);
    
    // Set shared memory config
    CUDA_CHECK(cudaFuncSetAttribute(
        retrieval_attention_kernel<TOPK_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    ));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        retrieval_attention_kernel<TOPK_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100  // 100% shared memory carveout
    ));
    
    // Warmup
    dim3 grid(NUM_HEADS, CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);
    
    for (int i = 0; i < 10; i++) {
        retrieval_attention_kernel<TOPK_PER_BLOCK><<<grid, block, shared_mem_size>>>(
            d_query, d_key_cache, d_value_cache, d_output
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        retrieval_attention_kernel<TOPK_PER_BLOCK><<<grid, block, shared_mem_size>>>(
            d_query, d_key_cache, d_value_cache, d_output
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Retrieval Attention - Avg time: %.3f ms, Throughput: %.2f iter/s\n",
           milliseconds / num_iterations,
           num_iterations * 1000.0f / milliseconds);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

#ifdef TORCH_EXTENSION_NAME
// Wrapper for PyTorch extension
void launch_retrieval_attention_128(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
) {
    run_retrieval_attention<128>(query, key_cache, value_cache, output, 1);
}

void launch_retrieval_attention_256(
    const half* query,
    const half* key_cache,
    const half* value_cache,
    half* output
) {
    run_retrieval_attention<256>(query, key_cache, value_cache, output, 1);
}
#else

void initialize_data(half* query, half* key_cache, half* value_cache) {
    // Initialize with small random values
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) {
        query[i] = __float2half((float)rand() / RAND_MAX * 0.1f);
    }
    
    for (int i = 0; i < SEQ_LEN * NUM_HEADS * HEAD_DIM; i++) {
        key_cache[i] = __float2half((float)rand() / RAND_MAX * 0.1f);
        value_cache[i] = __float2half((float)rand() / RAND_MAX * 0.1f);
    }
}


void run_baseline_attention(
    const half* d_query,
    const half* d_key_cache,
    const half* d_value_cache,
    half* d_output,
    int num_iterations = 100
) {
    const int BLOCK_SIZE = 256;
    size_t shared_mem_size = (SEQ_LEN + HEAD_DIM) * sizeof(half) + 256 * sizeof(half);
    
    printf("\nBaseline Attention Config: BLOCK_SIZE=%d\n", BLOCK_SIZE);
    printf("Shared memory per block: %zu bytes (%.1f KB)\n", shared_mem_size, shared_mem_size / 1024.0f);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        baseline_attention_kernel<<<NUM_HEADS, BLOCK_SIZE, shared_mem_size>>>(
            d_query, d_key_cache, d_value_cache, d_output
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        baseline_attention_kernel<<<NUM_HEADS, BLOCK_SIZE, shared_mem_size>>>(
            d_query, d_key_cache, d_value_cache, d_output
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Baseline Attention - Avg time: %.3f ms, Throughput: %.2f iter/s\n",
           milliseconds / num_iterations,
           num_iterations * 1000.0f / milliseconds);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    // Check device capability
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Shared Memory per Block: %zu bytes (%.1f KB)\n", 
           prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0f);
    
    if (prop.major < 9) {
        printf("ERROR: This kernel requires SM 90 or above (Hopper architecture)\n");
        printf("Your device has SM %d.%d\n", prop.major, prop.minor);
        return 1;
    }
    
    // Allocate host memory
    size_t query_size = NUM_HEADS * HEAD_DIM * sizeof(half);
    size_t cache_size = SEQ_LEN * NUM_HEADS * HEAD_DIM * sizeof(half);
    
    half* h_query = (half*)malloc(query_size);
    half* h_key_cache = (half*)malloc(cache_size);
    half* h_value_cache = (half*)malloc(cache_size);
    half* h_output = (half*)malloc(query_size);
    
    // Initialize data
    initialize_data(h_query, h_key_cache, h_value_cache);
    
    // Allocate device memory
    half *d_query, *d_key_cache, *d_value_cache, *d_output;
    CUDA_CHECK(cudaMalloc(&d_query, query_size));
    CUDA_CHECK(cudaMalloc(&d_key_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&d_value_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&d_output, query_size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_query, h_query, query_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key_cache, h_key_cache, cache_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_cache, h_value_cache, cache_size, cudaMemcpyHostToDevice));
    
    printf("\n=== Running Baseline Attention ===\n");
    run_baseline_attention(d_query, d_key_cache, d_value_cache, d_output);
    
    printf("\n=== Running Retrieval Attention (TOPK=128 per block) ===\n");
    run_retrieval_attention<128>(d_query, d_key_cache, d_value_cache, d_output);
    
    printf("\n=== Running Retrieval Attention (TOPK=256 per block) ===\n");
    run_retrieval_attention<256>(d_query, d_key_cache, d_value_cache, d_output);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, query_size, cudaMemcpyDeviceToHost));
    
    // Verify output
    printf("\nSample output values (head 0, first 8 dims): ");
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", __half2float(h_output[i]));
    }
    printf("\n");
    
    // Cleanup
    free(h_query);
    free(h_key_cache);
    free(h_value_cache);
    free(h_output);
    
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key_cache));
    CUDA_CHECK(cudaFree(d_value_cache));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("\nDone!\n");
    return 0;
}
#endif
