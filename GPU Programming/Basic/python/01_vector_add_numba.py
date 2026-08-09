"""
01_vector_add_numba.py
-----------------------
Vector addition using Numba's @cuda.jit — pure Python, no separate compile step.
Full line-by-line explanation: ../../00_Environment_Setup_RunPod_Colab/README.md (Part B)
                                ../../Lec_03_VectorAdd_MatrixMul/python/gpu_lec03_python_code.md

Run on Colab or RunPod (GPU runtime required):
    !pip install numba
    python 01_vector_add_numba.py
"""

from numba import cuda
import numpy as np


@cuda.jit
def vector_add_kernel(a, b, out):
    idx = cuda.grid(1)          # this thread's global 1D index
    if idx < out.size:          # boundary check
        out[idx] = a[idx] + b[idx]


def cuda_vector_addition(a_host, b_host):
    N = a_host.size

    a_device = cuda.to_device(a_host)              # H2D copy
    b_device = cuda.to_device(b_host)               # H2D copy
    out_device = cuda.device_array_like(a_host)      # empty GPU array for the result

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    vector_add_kernel[blocks_per_grid, threads_per_block](a_device, b_device, out_device)

    return out_device.copy_to_host()                # D2H copy


if __name__ == "__main__":
    N = 1_000_000
    a = 100 * np.random.rand(N)
    b = 100 * np.random.rand(N)

    result_cuda = cuda_vector_addition(a, b)
    result_cpu = a + b

    assert np.allclose(result_cuda, result_cpu)
    print("CUDA vector addition successful!")
    print("First 5 GPU results:", result_cuda[:5])
    print("First 5 CPU results:", result_cpu[:5])
