"""
03_matrix_mul_cupy.py
-----------------------
Matrix multiplication using CuPy — plain NumPy-style code, GPU kernel fully hidden.
Full line-by-line explanation: ../../Lec_03_VectorAdd_MatrixMul/python/gpu_lec03_python_code.md

Real benchmark from this exact script (9048x9048 matrices, T4 GPU):
    GPU Time: 1.046969 s
    CPU Time: 19.567448 s
    GPU is approximately 18.69x faster than the CPU.

Run on Colab or RunPod (GPU runtime required; CuPy is usually pre-installed on Colab):
    python 03_matrix_mul_cupy.py
"""

import numpy as np
import cupy as cp
import time

MATRIX_SIZE = 9048

a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

a_gpu = cp.asarray(a_cpu)          # allocate on GPU + copy H2D, in one line
b_gpu = cp.asarray(b_cpu)

start_gpu = time.time()
c_gpu = cp.dot(a_gpu, b_gpu)       # or a_gpu @ b_gpu -- kernel details fully hidden
cp.cuda.Stream.null.synchronize()  # wait for the GPU to actually finish before timing
gpu_time = time.time() - start_gpu

result_gpu_to_cpu = cp.asnumpy(c_gpu)   # D2H copy back to plain NumPy

start_cpu = time.time()
c_cpu_result = np.dot(a_cpu, b_cpu)
cpu_time = time.time() - start_cpu

difference = np.abs(c_cpu_result - result_gpu_to_cpu)
print(f"Max absolute difference: {np.max(difference):.8f}")
print(f"Avg absolute difference: {np.mean(difference):.8f}")

if np.allclose(c_cpu_result, result_gpu_to_cpu, rtol=1e-5, atol=1e-4):
    print("Results match within floating-point tolerance.")
else:
    print("WARNING: results differ beyond expected tolerance.")

print(f"GPU Time: {gpu_time:.6f} s")
print(f"CPU Time: {cpu_time:.6f} s")
if gpu_time > 0:
    print(f"GPU is approximately {cpu_time / gpu_time:.2f}x faster than the CPU.")
