// Usage:
// 1024线程/block
// nvcc -O3 -arch=sm_120a -o topk topk.cu && ./topk 1024
// 128线程/block
// nvcc -O3 -arch=sm_120a -o topk topk.cu && ./topk 128
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <utility>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU实现：使用部分排序
void topk_cpu(const float* data, int n, int k, float* values, int* indices) {
    std::vector<std::pair<float, int>> pairs(n);
    for (int i = 0; i < n; i++) {
        pairs[i] = std::make_pair(data[i], i);
    }
    
    // 部分排序，只排序前k个元素
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                     [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                         return a.first > b.first;
                     });
    
    for (int i = 0; i < k; i++) {
        values[i] = pairs[i].first;
        indices[i] = pairs[i].second;
    }
}

// CUDA Kernel: Bitonic Sort 
// 
// Bitonic Sort原理：
// 1. 将数组分成多个bitonic序列（一半升序一半降序交替）
// 2. 通过bitonic merge逐步合并，最终得到有序序列
// 3. 关键：direction决定最终是升序还是降序
//
// 实现细节：
// - 每个stage构建更大的bitonic序列
// - 每个step执行compare-exchange操作
// - direction: 0=降序, 1=升序
template<int TILE_SIZE>
__global__ void topk_kernel_bitonic_tile(const float* __restrict__ data,
                                         int n,
                                         int k,
                                         float* __restrict__ values,
                                         int* __restrict__ indices) {
    __shared__ float s_vals[TILE_SIZE];
    __shared__ int s_idx[TILE_SIZE];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const float* input = data + row * n;

    // 加载到共享内存；不足部分以 -INF / -1 填充
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
        if (i < n) {
            s_vals[i] = input[i];
            s_idx[i] = i;
        } else {
            s_vals[i] = -INFINITY;
            s_idx[i] = -1;
        }
    }
    __syncthreads();

    // 标准 Bitonic 网络（最终产出降序）：
    // 使用方向位 up = ((i & kseq) != 0) 以获得降序整体结果
    for (int kseq = 2; kseq <= TILE_SIZE; kseq <<= 1) {
        for (int j = kseq >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & kseq) != 0); // 与升序相反，得到降序
                    float vi = s_vals[i];
                    float vx = s_vals[ixj];
                    if (((vi > vx) == up)) {
                        s_vals[i] = vx;
                        s_vals[ixj] = vi;
                        int ti = s_idx[i];
                        s_idx[i] = s_idx[ixj];
                        s_idx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    // 写回：允许线程数小于 k 时按 stride 回写多个结果
    for (int i = tid; i < k; i += blockDim.x) {
        values[row * k + i] = s_vals[i];
        indices[row * k + i] = s_idx[i];
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void verify_results(const float* cpu_vals, const int* cpu_idx,
                   const float* gpu_vals, const int* gpu_idx,
                   int batch_size, int k) {
    bool correct = true;
    int errors = 0;
    
    for (int i = 0; i < batch_size && errors < 5; i++) {
        for (int j = 0; j < k; j++) {
            int cpu_pos = i * k + j;
            int gpu_pos = i * k + j;
            
            if (fabs(cpu_vals[cpu_pos] - gpu_vals[gpu_pos]) > 1e-5) {
                printf("Mismatch at batch %d, position %d: CPU=%.6f, GPU=%.6f\n",
                       i, j, cpu_vals[cpu_pos], gpu_vals[gpu_pos]);
                correct = false;
                errors++;
                if (errors >= 5) break;
            }
        }
    }
    
    if (correct) {
        printf("✓ Results match!\n");
    } else {
        printf("✗ Results differ (showing first %d errors)\n", errors);
    }
}

int main(int argc, char** argv) {
    // 参数设置 - 默认为您的场景: n=2560, k=512
    int batch_size = 32;
    int n = 2560;    // 每个batch的元素数量
    int k = 512;     // top-k的k值
    int use_small_block = 0;  // 是否使用小block版本(128线程)
    int threads_arg = -1;     // 可选：强制线程数（例如 128 或 1024），-1 表示自动
    
    if (argc >= 2) threads_arg = atoi(argv[1]);
    if (k > n) k = n;
    
    printf("=== Top-K Performance Comparison ===\n");
    printf("Batch size: %d\n", batch_size);
    printf("Elements per batch (n): %d\n", n);
    printf("K: %d\n", k);
    if (threads_arg > 0) {
        printf("Threads (forced): %d\n\n", threads_arg);
    } else {
        printf("Threads: auto (min(tile, 1024))\n\n");
    }
    
    // 分配主机内存
    size_t data_size = batch_size * n * sizeof(float);
    size_t result_size = batch_size * k * sizeof(float);
    size_t idx_size = batch_size * k * sizeof(int);
    
    float* h_data = (float*)malloc(data_size);
    float* h_values_cpu = (float*)malloc(result_size);
    int* h_indices_cpu = (int*)malloc(idx_size);
    float* h_values_gpu = (float*)malloc(result_size);
    int* h_indices_gpu = (int*)malloc(idx_size);
    
    // 初始化随机数据
    srand(42);
    for (int i = 0; i < batch_size * n; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 1000.0f;
    }
    
    // 打印第一个batch的前10个元素用于调试
    printf("First 10 elements of first batch:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] = %.2f\n", i, h_data[i]);
    }
    printf("\n");
    
    // CPU测试
    printf("Running CPU implementation...\n");
    double cpu_start = get_time();
    for (int i = 0; i < batch_size; i++) {
        topk_cpu(h_data + i * n, n, k, h_values_cpu + i * k, h_indices_cpu + i * k);
    }
    double cpu_time = get_time() - cpu_start;
    printf("CPU Time: %.3f ms\n", cpu_time * 1000);
    printf("CPU Top-5 values: %.2f, %.2f, %.2f, %.2f, %.2f\n", 
           h_values_cpu[0], h_values_cpu[1], h_values_cpu[2], 
           h_values_cpu[3], h_values_cpu[4]);
    printf("CPU Top-5 indices: %d, %d, %d, %d, %d\n\n", 
           h_indices_cpu[0], h_indices_cpu[1], h_indices_cpu[2],
           h_indices_cpu[3], h_indices_cpu[4]);
    
    // 分配设备内存
    float *d_data, *d_values;
    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_values, result_size));
    CUDA_CHECK(cudaMalloc(&d_indices, idx_size));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float gpu_time_ms;
    
    dim3 grid(batch_size);
    
    if (use_small_block) {
        printf("Running GPU implementation (Small Block) is disabled, falling back to tile.\n");
    }

    auto next_pow2 = [](int x) { int v = 1; while (v < x) v <<= 1; return v; };
    int tile = next_pow2(n);
    if (tile < 32) tile = 32;
    if (tile > 4096) {
        fprintf(stderr, "n=%d requires TILE_SIZE=%d (>4096). Reduce n or implement streaming.\n", n, tile);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_indices));
        free(h_data); free(h_values_cpu); free(h_indices_cpu); free(h_values_gpu); free(h_indices_gpu);
        return 2;
    }
    int threads;
    if (threads_arg > 0) {
        if (threads_arg > 1024) {
            fprintf(stderr, "Requested threads=%d exceeds 1024, clamping to 1024.\n", threads_arg);
            threads = 1024;
        } else if (threads_arg < 1) {
            fprintf(stderr, "Requested threads=%d is invalid, falling back to auto.\n", threads_arg);
            threads = tile < 1024 ? tile : 1024;
        } else {
            threads = threads_arg;
        }
    } else {
        threads = tile < 1024 ? tile : 1024;
    }
    dim3 block(threads);
    printf("Running GPU implementation (Bitonic Tile=%d, Threads=%d) ...\n", tile, threads);

    // 预热 + 计时
    switch (tile) {
        case 32:
            topk_kernel_bitonic_tile<32><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<32><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 64:
            topk_kernel_bitonic_tile<64><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<64><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 128:
            topk_kernel_bitonic_tile<128><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<128><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 256:
            topk_kernel_bitonic_tile<256><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<256><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 512:
            topk_kernel_bitonic_tile<512><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<512><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 1024:
            topk_kernel_bitonic_tile<1024><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<1024><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 2048:
            topk_kernel_bitonic_tile<2048><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<2048><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        case 4096:
            topk_kernel_bitonic_tile<4096><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            topk_kernel_bitonic_tile<4096><<<grid, block>>>(d_data, n, k, d_values, d_indices);
            break;
        default:
            fprintf(stderr, "Unhandled TILE_SIZE=%d\n", tile);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            break;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    printf("GPU Time: %.3f ms\n", gpu_time_ms);
    printf("Speedup: %.2fx\n", cpu_time * 1000 / gpu_time_ms);
    
    CUDA_CHECK(cudaMemcpy(h_values_gpu, d_values, result_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices_gpu, d_indices, idx_size, cudaMemcpyDeviceToHost));
    
    printf("GPU Top-5 values: %.2f, %.2f, %.2f, %.2f, %.2f\n", 
           h_values_gpu[0], h_values_gpu[1], h_values_gpu[2], 
           h_values_gpu[3], h_values_gpu[4]);
    printf("GPU Top-5 indices: %d, %d, %d, %d, %d\n", 
           h_indices_gpu[0], h_indices_gpu[1], h_indices_gpu[2],
           h_indices_gpu[3], h_indices_gpu[4]);
    
    // 验证结果
    printf("\nVerifying results...\n");
    verify_results(h_values_cpu, h_indices_cpu, h_values_gpu, h_indices_gpu, batch_size, k);
    
    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));
    
    free(h_data);
    free(h_values_cpu);
    free(h_indices_cpu);
    free(h_values_gpu);
    free(h_indices_gpu);
    
    printf("\n=== Performance Summary ===\n");
    printf("Scenario: n=%d, k=%d\n", n, k);
    printf("CPU: %.3f ms\n", cpu_time * 1000);
    printf("GPU: %.3f ms\n", gpu_time_ms);
    printf("Overall Speedup: %.2fx\n", cpu_time * 1000 / gpu_time_ms);
    printf("\nUsage: ./topk_cuda <batch_size> <n> <k> [threads]\n");
    
    return 0;
}
