/*
 * 02_matrix_mul.cu
 * -----------------
 * Basic CUDA C++ matrix multiplication (5x5, human-checkable output).
 * Full line-by-line explanation: ../../Lec_03_VectorAdd_MatrixMul/cpp/gpu_lec03_cpp_code.md
 *
 * Compile (Colab/RunPod, T4/A-series GPU):
 *   nvcc 02_matrix_mul.cu -o matrix_mul -arch=sm_75
 * Run:
 *   ./matrix_mul
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 5

// Kernel: each thread computes ONE output cell C[row][col]
__global__ void matrix_mul(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // row uses the Y axis
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // col uses the X axis

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];      // flattened 2D -> 1D indexing
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int n = N;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * n * sizeof(int);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Simple, hand-verifiable test patterns
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
            b[i * n + j] = i * j;
        }

    printf("Matrix A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%d ", a[i * n + j]);
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%d ", b[i * n + j]);
        printf("\n");
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(N, N);                                // 5x5 = 25 threads/block
    dim3 gridSize((n + N - 1) / N, (n + N - 1) / N);      // ceiling division, both dims

    matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaError_t err = cudaGetLastError();                 // catches launch-time errors
    cudaDeviceSynchronize();                              // catches runtime errors too
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Product Matrix C = A x B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%d ", c[i * n + j]);
        printf("\n");
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    return 0;
}
