// Benchmark split GEMV and Bitonic phases from gemv_topk.cu
// nvcc -O3 -std=c++17 -arch=sm_120a -o gemv_profile_split gemv_profile_split.cu && ./gemv_profile_split --iters 50 --warmup 10

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

// Problem constants (match gemv_topk.cu)
#define HD 128
#define HN 32
#define CLEN 2048
#define CSZ 20
#define FSL (CSZ * CLEN)
#define TOPC 5
#define TOPK_PER_CLUSTER 512
#define OUT_PER_HEAD (TOPC * TOPK_PER_CLUSTER)

#define WEIGHT_BUFFER_LEN 32
#define STAGE_OFFSET (WEIGHT_BUFFER_LEN / 2)

__device__ __forceinline__ void cp_async_pred_load_128b(half* smem_ptr, const half* gmem_ptr, bool predicate) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_in_bytes = predicate ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" : : "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <size_t n>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------------- GEMV phase kernel (without 2048-bitonic) ----------------
__global__ void gemv_phase_kernel(const __half* __restrict__ kv,
                                  const __half* __restrict__ q,
                                  const __half* __restrict__ centers,
                                  float* __restrict__ cand_vals_out,
                                  int* __restrict__ cand_idx_out,
                                  int* __restrict__ chosen_centers) {
    const int g = blockIdx.x;
    const int h = g / TOPC;
    const int r_assigned = g % TOPC;
    const int tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane_id = tid & 31;
    const uint32_t input_idx = (lane_id % 16) * 8;
    const uint32_t weight_idx = warp_id * 4 + (lane_id / 16) * 2;

    __shared__ float center_vals[32];
    __shared__ int center_idx[32];
    __shared__ float cand_vals[CLEN];
    __shared__ int cand_idx[CLEN];
    __shared__ half k_buffer[WEIGHT_BUFFER_LEN * HD];

    half __align__(16) reg_input[8], reg_weight[8];
    float __align__(16) qk[2];

    *(uint4*)(&reg_input[0]) = *(uint4*)(&q[h * HD + input_idx]);

    // Load center tiles into shared via cp.async
    for (int i = 0; i < 2; ++i) {
        cp_async_pred_load_128b(&k_buffer[(weight_idx + i) * HD + input_idx],
                                &centers[h * CSZ * HD + (weight_idx + i) * HD + input_idx],
                                (weight_idx + i < CSZ));
    }
    cp_async_commit_group();
    for (int i = 0; i < 2; ++i) {
        cp_async_pred_load_128b(&k_buffer[(STAGE_OFFSET + weight_idx + i) * HD + input_idx],
                                &centers[h * CSZ * HD + (STAGE_OFFSET + weight_idx + i) * HD + input_idx],
                                (STAGE_OFFSET + weight_idx + i < CSZ));
    }
    cp_async_commit_group();
    cp_async_wait_group<1>();
    __syncthreads();

    for (int i = 0; i < 2; ++i) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(weight_idx + i) * HD + input_idx]);
        qk[i] = 0.f;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = 16 >> 1; mask > 0; mask >>= 1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        center_vals[weight_idx + i] = qk[i];
        center_idx[weight_idx + i] = weight_idx + i;
    }
    cp_async_wait_group<0>();
    __syncthreads();

    for (int i = 0; i < 2; ++i) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(STAGE_OFFSET + weight_idx + i) * HD + input_idx]);
        qk[i] = 0.f;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = 16 >> 1; mask > 0; mask >>= 1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        center_vals[STAGE_OFFSET + weight_idx + i] = qk[i];
        center_idx[STAGE_OFFSET + weight_idx + i] = STAGE_OFFSET + weight_idx + i;
    }
    __syncthreads();

    if (tid < 32) {
        if (tid >= CSZ) {
            center_vals[tid] = -INFINITY;
            center_idx[tid] = -1;
        }
    }
    __syncthreads();

    // Bitonic sort 32 centers (same as gemv_topk)
    for (int kseq = 2; kseq <= 32; kseq <<= 1) {
        for (int j = kseq >> 1; j > 0; j >>= 1) {
            int i = tid;
            if (i < 32) {
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

    int c = center_idx[r_assigned];
    if (tid == 0) {
        chosen_centers[g] = c;
    }
    __syncthreads();

    // Preload kv
    int common_offset = (c * CLEN + weight_idx) * HN * HD + h * HD + input_idx;
    for (int i = 0; i < 2; ++i) {
        cp_async_pred_load_128b(&k_buffer[(weight_idx + i) * HD + input_idx],
                                &kv[common_offset + i * HN * HD], true);
    }
    cp_async_commit_group();

    for (int id = 1; id < CLEN / 16; ++id) {
        for (int i = 0; i < 2; ++i) {
            cp_async_pred_load_128b(&k_buffer[((id % 2) * STAGE_OFFSET + weight_idx + i) * HD + input_idx],
                                    &kv[common_offset + (id * 16 + i) * HN * HD], true);
        }
        cp_async_commit_group();

        cp_async_wait_group<1>();
        __syncthreads();
        for (int i = 0; i < 2; ++i) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((id - 1) % 2) * STAGE_OFFSET + weight_idx + i) * HD + input_idx]);
            qk[i] = 0.f;
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = 16 >> 1; mask > 0; mask >>= 1) {
                qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
            }
            cand_vals[(id - 1) * 16 + weight_idx + i] = qk[i];
            cand_idx[(id - 1) * 16 + weight_idx + i] = (id - 1) * 16 + weight_idx + i;
        }
    }

    int id = CLEN / 16;
    cp_async_wait_group<0>();
    __syncthreads();
    for (int i = 0; i < 2; ++i) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&k_buffer[(((id - 1) % 2) * STAGE_OFFSET + weight_idx + i) * HD + input_idx]);
        qk[i] = 0.f;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            qk[i] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = 16 >> 1; mask > 0; mask >>= 1) {
            qk[i] += __shfl_xor_sync(0xffffffff, qk[i], mask);
        }
        cand_vals[(id - 1) * 16 + weight_idx + i] = qk[i];
        cand_idx[(id - 1) * 16 + weight_idx + i] = (id - 1) * 16 + weight_idx + i;
    }
    __syncthreads();

    int base = g * CLEN;
    for (int t = tid; t < CLEN; t += blockDim.x) {
        cand_vals_out[base + t] = cand_vals[t];
        cand_idx_out[base + t] = cand_idx[t];
    }
}

// ---------------- Bitonic phase kernel ----------------
__global__ void bitonic_phase_kernel(const float* __restrict__ cand_vals_in,
                                     const int* __restrict__ cand_idx_in,
                                     const int* __restrict__ chosen_centers,
                                     int* __restrict__ out_indices) {
    const int g = blockIdx.x;
    const int h = g / TOPC;
    const int r_assigned = g % TOPC;
    const int tid = threadIdx.x;

    (void)cand_vals_in;
    (void)cand_idx_in;

    __shared__ float cand_vals[CLEN];
    __shared__ int cand_idx[CLEN];

    for (int i = tid; i < CLEN; i += blockDim.x) {
        unsigned int seed = (unsigned int)(g * 0x45d9f3bu) ^ (unsigned int)(i * 0x27d4eb2du);
        seed = 1664525u * seed + 1013904223u;
        float val = (float)(seed & 0xFFFF) / 65535.0f - 0.5f;
        cand_vals[i] = val;
        cand_idx[i] = i;
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
                        int ti = cand_idx[i]; cand_idx[i] = cand_idx[ixj]; cand_idx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    int c = chosen_centers[g];
    for (int i = tid; i < TOPK_PER_CLUSTER; i += blockDim.x) {
        int local = cand_idx[i];
        int global_idx = c * CLEN + local;
        int out_offset = h * OUT_PER_HEAD + r_assigned * TOPK_PER_CLUSTER + i;
        out_indices[out_offset] = global_idx;
    }
}

// ---------------- Host harness ----------------
int main(int argc, char** argv) {
    int warmup = 10;
    int iters = 50;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        }
    }

    printf("Split profiling: HN=%d HD=%d CSZ=%d CLEN=%d TOPC=%d\n", HN, HD, CSZ, CLEN, TOPC);

    size_t kv_elems = (size_t)FSL * HN * HD;
    size_t q_elems = (size_t)HN * HD;
    size_t center_elems = (size_t)HN * CSZ * HD;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uni(-0.5f, 0.5f);
    std::normal_distribution<float> noise(0.0f, 0.05f);

    half* h_kv = (half*)malloc(sizeof(half) * kv_elems);
    half* h_q = (half*)malloc(sizeof(half) * q_elems);
    half* h_centers = (half*)malloc(sizeof(half) * center_elems);

    for (int h = 0; h < HN; ++h) {
        for (int c = 0; c < CSZ; ++c) {
            size_t base = ((size_t)h * CSZ + c) * HD;
            for (int d = 0; d < HD; ++d) {
                h_centers[base + d] = __float2half(uni(rng));
            }
        }
    }

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

    for (int h = 0; h < HN; ++h) {
        size_t base_q = (size_t)h * HD;
        for (int d = 0; d < HD; ++d) {
            h_q[base_q + d] = __float2half(uni(rng));
        }
    }

    half *d_kv, *d_q, *d_centers;
    CUDA_CHECK(cudaMalloc(&d_kv, sizeof(half) * kv_elems));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(half) * q_elems));
    CUDA_CHECK(cudaMalloc(&d_centers, sizeof(half) * center_elems));
    CUDA_CHECK(cudaMemcpy(d_kv, h_kv, sizeof(half) * kv_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, sizeof(half) * q_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centers, h_centers, sizeof(half) * center_elems, cudaMemcpyHostToDevice));

    size_t cand_buffer_elems = (size_t)HN * TOPC * CLEN;
    float* d_cand_vals;
    int* d_cand_idx;
    int* d_chosen_centers;
    int* d_out_indices;
    CUDA_CHECK(cudaMalloc(&d_cand_vals, sizeof(float) * cand_buffer_elems));
    CUDA_CHECK(cudaMalloc(&d_cand_idx, sizeof(int) * cand_buffer_elems));
    CUDA_CHECK(cudaMalloc(&d_chosen_centers, sizeof(int) * HN * TOPC));
    CUDA_CHECK(cudaMalloc(&d_out_indices, sizeof(int) * (size_t)HN * OUT_PER_HEAD));

    dim3 grid(HN * TOPC);
    dim3 block(128);

    for (int i = 0; i < warmup; ++i) {
        gemv_phase_kernel<<<grid, block>>>(d_kv, d_q, d_centers, d_cand_vals, d_cand_idx, d_chosen_centers);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t st, ed;
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&ed));

    CUDA_CHECK(cudaEventRecord(st));
    for (int i = 0; i < iters; ++i) {
        gemv_phase_kernel<<<grid, block>>>(d_kv, d_q, d_centers, d_cand_vals, d_cand_idx, d_chosen_centers);
    }
    CUDA_CHECK(cudaEventRecord(ed));
    CUDA_CHECK(cudaEventSynchronize(ed));
    float gemv_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gemv_ms, st, ed));
    gemv_ms = (gemv_ms / iters) * 1000.0f;

    for (int i = 0; i < warmup; ++i) {
        bitonic_phase_kernel<<<grid, block>>>(d_cand_vals, d_cand_idx, d_chosen_centers, d_out_indices);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(st));
    for (int i = 0; i < iters; ++i) {
        bitonic_phase_kernel<<<grid, block>>>(d_cand_vals, d_cand_idx, d_chosen_centers, d_out_indices);
    }
    CUDA_CHECK(cudaEventRecord(ed));
    CUDA_CHECK(cudaEventSynchronize(ed));
    float bitonic_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&bitonic_ms, st, ed));
    bitonic_ms = (bitonic_ms / iters) * 1000.0f;

    printf("Average GEMV phase latency: %.3f us\n", gemv_ms);
    printf("Average Bitonic phase latency: %.3f us\n", bitonic_ms);

    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(ed));

    CUDA_CHECK(cudaFree(d_out_indices));
    CUDA_CHECK(cudaFree(d_chosen_centers));
    CUDA_CHECK(cudaFree(d_cand_idx));
    CUDA_CHECK(cudaFree(d_cand_vals));
    CUDA_CHECK(cudaFree(d_centers));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_kv));

    free(h_q);
    free(h_centers);
    free(h_kv);

    return 0;
}
