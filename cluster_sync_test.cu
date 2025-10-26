// nvcc -std=c++17 -arch=sm_90a cluster_sync_test.cu -o cluster_sync_test && ./cluster_sync_test && rm cluster_sync_test
// Terminal output:
// block 2 value -> 20.000000
// block 0 value -> 0.000000
// block 3 value -> 30.000000
// block 1 value -> 10.000000
// block 3 cluster_sum -> 30.000000
// block 2 cluster_sum -> 40.000000
// block 0 cluster_sum -> 60.000000
// block 1 cluster_sum -> 50.000000
// block 3 cluster_max -> 20.000000
// block 2 cluster_max -> 30.000000
// block 0 cluster_max -> 30.000000
// block 1 cluster_max -> 30.000000

// 看起来传递的锁能正常工作，cluster.sync()不会报错但没有正常工作。也可能是尝试reduce的操作没有写对。


#include <cstdio>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

// DEBUG macro
#define DEBUG

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error %s at %s:%d -> %s (%d)\n", #call, __FILE__, __LINE__, cudaGetErrorString(err), (int)err); \
		exit(EXIT_FAILURE); \
	} \
} while(0)

#define CLUSTER_SIZE 5

constexpr int BLOCK_THREADS = 64;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) block_specialization_kernel() 
{
    cg::cluster_group cluster = cg::this_cluster();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid = threadIdx.x;

    __shared__ float value, cluster_sum, cluster_max;
    void *dst_shmem;

    // lock
    __shared__ int lock;
    volatile int* lock_ptr = &lock;

    if (cluster_block_id == CLUSTER_SIZE - 1) {
        // Assign value to other blocks and quit
        for (int dst_cta_id = 0; dst_cta_id < cluster.num_blocks() - 1; dst_cta_id++) {
            if (tid == 0) {
                dst_shmem = (void *)cluster.map_shared_rank(&value, dst_cta_id);
                *(float *)dst_shmem = dst_cta_id * 10;
                dst_shmem = (void *)cluster.map_shared_rank(&lock, dst_cta_id);
                *(int *)dst_shmem = 1;
            }
        }
        return;
    } else {
        while (*lock_ptr == 0) {
            __nanosleep(32);
        }

        if (tid == 0) {
            printf("block %d value -> %f\n", cluster_block_id, value);
        }

        // clutser reduce: sum
        if (tid == 0) {
            cluster_sum = 0.0f;
        }
        cluster.sync();
        for (int id = 1; id < cluster.num_blocks() - 1; id++) {
            if (tid == 0) {
                int dst_cta_id = (cluster_block_id + id) % (cluster.num_blocks() - 1);
                dst_shmem = (void *)cluster.map_shared_rank(&cluster_sum, dst_cta_id);
            }
            cluster.sync();
            if (tid == 0) {
                atomicAdd((float *)dst_shmem, value);
            }
            cluster.sync();
        }
        if (tid == 0) {
            printf("block %d cluster_sum -> %f\n", cluster_block_id, cluster_sum);
        }

        if (tid == 0) {
            cluster_max = 0.0f;
        }
        cluster.sync();
        // cluster reduce: max
        for (int id = 1; id < cluster.num_blocks() - 1; id++) {
            if (tid == 0) {
                int dst_cta_id = (cluster_block_id + id) % (cluster.num_blocks() - 1);
                dst_shmem = (void *)cluster.map_shared_rank(&cluster_max, dst_cta_id);
            }
            cluster.sync();
            if (tid == 0) {
                *(float *)dst_shmem = fmaxf(*(float *)dst_shmem, value);
            }
            cluster.sync();
        }

        if (tid == 0) {
            printf("block %d cluster_max -> %f\n", cluster_block_id, cluster_max);
        }
    }

}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) block_specialization_kernel_with_global_lock(int* global_lock) 
{
    cg::cluster_group cluster = cg::this_cluster();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid = threadIdx.x;

    __shared__ float value, cluster_sum, cluster_max;
    void *dst_shmem;

    // lock
    __shared__ int lock;
    volatile int* lock_ptr = &lock;
    volatile int* global_lock_ptr = global_lock;

    if (cluster_block_id == CLUSTER_SIZE - 1) {
        if (tid == 0) {
            printf("---- second kernel ----\n");
        }
        // Assign value to other blocks and quit
        for (int dst_cta_id = 0; dst_cta_id < cluster.num_blocks() - 1; dst_cta_id++) {
            if (tid == 0) {
                dst_shmem = (void *)cluster.map_shared_rank(&value, dst_cta_id);
                *(float *)dst_shmem = dst_cta_id * 10;
                dst_shmem = (void *)cluster.map_shared_rank(&lock, dst_cta_id);
                *(int *)dst_shmem = 1;
            }
        }
        return;
    } else {
        while (*lock_ptr == 0) {
            __nanosleep(1);
        }

        if (tid == 0) {
            printf("block %d value -> %f\n", cluster_block_id, value);
        }

        // clutser reduce: sum
        if (tid == 0) {
            cluster_sum = 0.0f;
            atomicAdd((int *)global_lock_ptr, 1);
        }
        while (*global_lock_ptr != 4) {
            __nanosleep(32);
        }
        if (tid == 0) {
            printf("block %d reached here\n", cluster_block_id);
        }
        for (int id = 1; id < cluster.num_blocks() - 1; id++) {
            if (tid == 0) {
                int dst_cta_id = (cluster_block_id + id) % (cluster.num_blocks() - 1);
                dst_shmem = (void *)cluster.map_shared_rank(&cluster_sum, dst_cta_id);
                atomicAdd((float *)dst_shmem, value);
                atomicAdd((int *)global_lock_ptr, 1);
            }
            while (*global_lock_ptr != (4 + id * 4)) {
                __nanosleep(32);
            }
            if (tid == 0) {
                printf("block %d id -> %d, global_lock -> %d \n", cluster_block_id, id, *global_lock_ptr);
            }
        }
        if (tid == 0) {
            printf("block %d cluster_sum -> %f\n", cluster_block_id, cluster_sum);
        }

        if (tid == 0) {
            cluster_max = 0.0f;
        }
        cluster.sync();
        // cluster reduce: max
        for (int id = 1; id < cluster.num_blocks() - 1; id++) {
            if (tid == 0) {
                int dst_cta_id = (cluster_block_id + id) % (cluster.num_blocks() - 1);
                dst_shmem = (void *)cluster.map_shared_rank(&cluster_max, dst_cta_id);
            }
            cluster.sync();
            if (tid == 0) {
                *(float *)dst_shmem = fmaxf(*(float *)dst_shmem, value);
            }
            cluster.sync();
        }

        if (tid == 0) {
            printf("block %d cluster_max -> %f\n", cluster_block_id, cluster_max);
        }
    }

}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    dim3 grid(CLUSTER_SIZE, 1, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    int h_global_lock = 0;
    int* d_global_lock = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_lock, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_global_lock, &h_global_lock, sizeof(int), cudaMemcpyHostToDevice));
    
    block_specialization_kernel<<<grid, block>>>();
    block_specialization_kernel_with_global_lock<<<grid, block>>>(d_global_lock);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_global_lock));
    return EXIT_SUCCESS;
}