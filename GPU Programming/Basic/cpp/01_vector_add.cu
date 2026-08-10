/*
 * 01_vector_add.cu
 * -----------------
 * Basic CUDA C++ vector addition, with CPU-vs-GPU timing comparison.
 * Full line-by-line explanation: ../../Lec_03_VectorAdd_MatrixMul/cpp/gpu_lec03_cpp_code.md
 *
 * Compile (Colab/RunPod, T4/A-series GPU):
 *   nvcc 01_vector_add.cu -o vector_add -arch=sm_75
 * Run:
 *   ./vector_add
 *
 * Expected output (varies by GPU/array size):
 *   CPU Time: 3.09124 ms
 *   GPU Time: 0.181472 ms
 */

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// Kernel: each thread computes ONE element of c = a + b
__global__ void add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 20;              // 1,048,576 elements
    int *a, *b, *c;                // host arrays (pinned)
    int *d_a, *d_b, *d_c;          // device arrays

    // Pinned host memory transfers faster to/from the GPU than plain malloc
    cudaMallocHost(&a, N * sizeof(int));
    cudaMallocHost(&b, N * sizeof(int));
    cudaMallocHost(&c, N * sizeof(int));

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // ---- CPU baseline (sequential loop) ----
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;

    // ---- GPU version ----
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;                                   // multiple of warp size (32)
    int numBlocks = (N + blockSize - 1) / blockSize;        // ceiling division

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c);

    return 0;
}
