"""
02_matrix_mul_pycuda.py
-------------------------
Matrix multiplication using PyCUDA — a raw CUDA C++ kernel written as a string,
compiled at runtime, and driven from Python.
Full line-by-line explanation: ../../Lec_03_VectorAdd_MatrixMul/python/gpu_lec03_python_code.md

Run on Colab or RunPod (GPU runtime required):
    !pip install pycuda
    python 02_matrix_mul_pycuda.py
"""

import numpy as np
import pycuda.autoinit                       # auto-initializes the CUDA context
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

cuda_kernel_source = """
__global__ void matrix_mul_kernel(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < size && col < size) {
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}
"""

mod = SourceModule(cuda_kernel_source)                    # compiled by nvcc at runtime
matrix_mul_gpu = mod.get_function("matrix_mul_kernel")

MATRIX_SIZE = 512
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
c_gpu_result_cpu = np.empty_like(a_cpu)

a_gpu = cuda.mem_alloc(a_cpu.nbytes)
b_gpu = cuda.mem_alloc(b_cpu.nbytes)
c_gpu = cuda.mem_alloc(c_gpu_result_cpu.nbytes)

cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(b_gpu, b_cpu)

BLOCK_SIZE = (16, 16, 1)
GRID_SIZE = (
    (MATRIX_SIZE + BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0],
    (MATRIX_SIZE + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1],
)

start_gpu = time.time()
# NOTE: PyCUDA kwargs are block= THEN grid= -- opposite order from CUDA C++'s <<<grid,block>>>
matrix_mul_gpu(a_gpu, b_gpu, c_gpu, np.int32(MATRIX_SIZE), block=BLOCK_SIZE, grid=GRID_SIZE)
cuda.Context.synchronize()
gpu_time = time.time() - start_gpu

cuda.memcpy_dtoh(c_gpu_result_cpu, c_gpu)

start_cpu = time.time()
c_cpu_result = np.dot(a_cpu, b_cpu)
cpu_time = time.time() - start_cpu

print(f"GPU Time: {gpu_time:.6f} s")
print(f"CPU Time: {cpu_time:.6f} s")
print("Results match:", np.allclose(c_cpu_result, c_gpu_result_cpu, atol=1e-3))
if gpu_time > 0:
    print(f"GPU is approximately {cpu_time / gpu_time:.2f}x faster than the CPU.")
