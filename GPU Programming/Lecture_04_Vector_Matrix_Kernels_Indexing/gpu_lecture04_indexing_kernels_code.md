# 💻 Lecture 04 — Vector/Matrix Kernels & Indexing: HANDS-ON CODE

### *Runnable Python + CUDA C — from block-only parallelism to a boundary-safe kernel, JIT, and reduction*

> **Nav:** [← Lecture 04 README](README.md) | [📖 THEORY](gpu_lecture04_indexing_kernels_theory.md) | [🔢 NUMERICAL](gpu_lecture04_indexing_kernels_numerical.md) | **CODE** | [🎯 PRACTICE](gpu_lecture04_indexing_kernels_practice.md)

---

## 🏗️ Setup — Google Colab (easiest)

```
Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4) -> Save
```
```python
!nvidia-smi
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```

---

## Ex 1. Block-only parallelism — `add<<<N,1>>>()`

```python
%%cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(void) {
    int N = 8;
    int h_a[8] = {0,1,2,3,4,5,6,7};
    int h_b[8] = {10,20,30,40,50,60,70,80};
    int h_c[8];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N, 1>>>(d_a, d_b, d_c);   // N blocks, 1 thread each

    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("c[%d] = %d\n", i, h_c[i]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

### 👶 What this does
This is the literal `add<<<N,1>>>()` kernel from [theory.md §1](gpu_lecture04_indexing_kernels_theory.md#1-moving-to-parallel-blocks-first) — each BLOCK (not thread) does one addition, indexed purely by `blockIdx.x`. Deliberately tiny (`N=8`) so you can verify every single output by hand.

### 👶 What you should see
```
c[0] = 10
c[1] = 21
c[2] = 32
c[3] = 43
c[4] = 54
c[5] = 65
c[6] = 76
c[7] = 87
```

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-hands-on-code)

---

## Ex 2. Combined block+thread indexing — verify the "red element" problem

```python
%%cuda
#include <stdio.h>

__global__ void whoAmI(int *out) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    out[index] = index;   // each thread just writes its own computed index
}

int main(void) {
    int M = 8;          // threads per block, matching numerical.md's worked example
    int numBlocks = 4;
    int N = M * numBlocks;
    int h_out[32];

    int *d_out;
    cudaMalloc(&d_out, N*sizeof(int));

    whoAmI<<<numBlocks, M>>>(d_out);

    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("index 21 was written by the thread that computed: %d (should be 21)\n", h_out[21]);
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);
    printf("\n");

    cudaFree(d_out);
    return 0;
}
```

### 👶 What this does
Every thread computes `index = threadIdx.x + blockIdx.x * blockDim.x` (the exact formula from [theory.md §3](gpu_lecture04_indexing_kernels_theory.md#3-the-combined-vector-add-kernel--full-main)) and simply writes ITS OWN index value into that position of the output array. If the formula is correct, `out[i]` should equal `i` for every position — a self-checking test.

### 👶 What you should see
```
index 21 was written by the thread that computed: 21 (should be 21)
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```
A perfectly ordered `0` to `31` sequence confirms the indexing formula is behaving exactly as derived in [numerical.md §1](gpu_lecture04_indexing_kernels_numerical.md#1-the-red-element-problem-rebuilt-from-zero) — this specifically re-verifies that `threadIdx.x=5, blockIdx.x=2, M=8` really does land on index 21.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-hands-on-code)

---

## Ex 3. Boundary-safe kernel — deliberately awkward size

```python
%%cuda
#include <stdio.h>

__global__ void add_safe(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

int main(void) {
    int n = 1000;              // deliberately NOT a clean multiple of 256
    int M = 256;
    int numBlocks = (n + M - 1) / M;   // ceiling division -- see numerical.md section 4
    printf("n=%d, blockSize=%d -> launching %d blocks (%d total threads, %d 'extra')\n",
           n, M, numBlocks, numBlocks*M, numBlocks*M - n);

    int *h_a = (int*)malloc(n*sizeof(int));
    int *h_b = (int*)malloc(n*sizeof(int));
    int *h_c = (int*)malloc(n*sizeof(int));
    for (int i=0;i<n;i++){ h_a[i]=i; h_b[i]=1; }

    int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));
    cudaMalloc(&d_c, n*sizeof(int));
    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(int), cudaMemcpyHostToDevice);

    add_safe<<<numBlocks, M>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("c[0]=%d  c[999]=%d  (expected 1 and 1000)\n", h_c[0], h_c[999]);

    free(h_a);free(h_b);free(h_c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    return 0;
}
```

### 👶 What this does
This reproduces "Case B" from [numerical.md §4](gpu_lecture04_indexing_kernels_numerical.md#4-ceiling-division-for-arbitrary-sizes-every-case) with real hardware: `n=1000` does NOT divide evenly by `256`, so without the ceiling-division launch AND the `if (index < n)` guard, this would either silently miss the last 232 elements or crash trying to read/write past the array end.

### 👶 What you should see
```
n=1000, blockSize=256 -> launching 4 blocks (1024 total threads, 24 'extra')
c[0]=1  c[999]=1000  (expected 1 and 1000)
```
The `24 'extra'` threads are the ones the `if (index < n)` guard correctly does nothing for — try DELETING that `if` line and re-running to see what happens (on many systems this specific example may still "work" by luck since the extra reads land in allocated-but-out-of-bounds memory, but it's undefined behavior — never rely on it).

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-hands-on-code)

---

## Ex 4. JIT in Python — the same vector add, Numba style

```python
!pip install numba -q
```
```python
from numba import cuda
import numpy as np
import time

@cuda.jit
def add_kernel(a, b, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = a[idx] + b[idx]

n = 1000
a = np.arange(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_out = cuda.device_array_like(a)

threads = 256
blocks = (n + threads - 1) // threads
add_kernel[blocks, threads](d_a, d_b, d_out)

result = d_out.copy_to_host()
print("c[0] =", result[0], " c[999] =", result[999], " (expected 1.0 and 1000.0)")
```

### 👶 What this does
The exact same "1000 elements, 256 threads/block, ceiling division, boundary check" pattern as Ex 3 — but now in Python via Numba's `@cuda.jit`, directly demonstrating [theory.md §6](gpu_lecture04_indexing_kernels_theory.md#6-jit-compilation-for-python-numba-cupy-pycuda)'s claim that Numba eliminates the need to write C/C++ CUDA code directly. Compare this file side-by-side with Ex 3 — every concept (ceiling division, boundary check, H2D/D2H copy) maps 1:1, just in different syntax.

### 👶 What you should see
```
c[0] = 1.0  c[999] = 1000.0  (expected 1.0 and 1000.0)
```

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-hands-on-code)

---

## Ex 5. Matrix multiplication as N² inner products — count them yourself

```python
import numpy as np
import time

for N in [64, 128, 256, 512]:
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    inner_products = N * N
    total_ops = N**3

    start = time.time()
    C = np.dot(A, B)
    elapsed = time.time() - start

    print(f"N={N:4d}: {inner_products:9,} inner products, "
          f"{total_ops:14,} total multiply-adds, CPU time={elapsed:.4f}s")
```

### 👶 What this does
Directly computes and prints the `N²` (inner products) and `N³` (total multiply-add operations) numbers derived in [numerical.md §5](gpu_lecture04_indexing_kernels_numerical.md#5-matrix-multiplication-counting-the-independent-work), for four increasing matrix sizes, alongside the ACTUAL time NumPy takes to compute each one on the CPU.

### 👶 What you should see (your exact timings will vary)
```
N=  64:      4,096 inner products,          262,144 total multiply-adds, CPU time=0.0003s
N= 128:     16,384 inner products,        2,097,152 total multiply-adds, CPU time=0.0021s
N= 256:     65,536 inner products,       16,777,216 total multiply-adds, CPU time=0.0156s
N= 512:    262,144 inner products,      134,217,728 total multiply-adds, CPU time=0.1183s
```
Notice the total-multiply-adds column growing MUCH faster than the inner-products column — exactly the N³ vs N² growth called out as an exam trap in [numerical.md §6](gpu_lecture04_indexing_kernels_numerical.md#6-exam-style-numerical-traps): doubling `N` from 256 to 512 (2×) grows total operations by roughly 8× (512³/256³ = 8), and you can see the CPU time grow by roughly that same ~8× ratio too.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-hands-on-code)

---

## Ex 6. Shared-memory inner-product reduction — the real (naive) kernel

```python
%%cuda
#include <stdio.h>
#define SIZE 8

__global__ void innerProduct(int *a, int *b, int *c) {
    __shared__ int product[SIZE];
    int i = threadIdx.x;

    if (i < SIZE)
        product[i] = a[i] * b[i];

    __syncthreads();

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int k = 0; k < SIZE; k++)
            sum += product[k];
        *c = sum;
    }
}

int main(void) {
    int h_a[SIZE] = {1,2,3,4,5,6,7,8};
    int h_b[SIZE] = {8,7,6,5,4,3,2,1};
    int h_c;

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, SIZE*sizeof(int));
    cudaMalloc(&d_b, SIZE*sizeof(int));
    cudaMalloc(&d_c, sizeof(int));
    cudaMemcpy(d_a, h_a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    innerProduct<<<1, SIZE>>>(d_a, d_b, d_c);

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Inner product result: %d\n", h_c);

    int expected = 0;
    for (int i=0;i<SIZE;i++) expected += h_a[i]*h_b[i];
    printf("Expected (CPU-computed): %d\n", expected);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

### 👶 What this does
This is the EXACT shared-memory reduction kernel from [theory.md §8](gpu_lecture04_indexing_kernels_theory.md#8-vectorinner-product-kernel--shared-memory-reduction), run for real with hand-checkable numbers. `a = [1..8]`, `b = [8..1]` (reversed), so the expected dot product is `1×8 + 2×7 + 3×6 + 4×5 + 5×4 + 6×3 + 7×2 + 8×1 = 8+14+18+20+20+18+14+8 = 120`.

### 👶 What you should see
```
Inner product result: 120
Expected (CPU-computed): 120
```
Try commenting out the `__syncthreads();` line and re-running several times — on some runs you may get an INCORRECT, inconsistent result (a real, visible race condition), directly demonstrating why that barrier is not optional — exactly the warning from theory.md's line-by-line breakdown of this kernel.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-hands-on-code)

---

## 🧭 What we learned

- `add<<<N,1>>>()` (block-only) really does work, indexed purely by `blockIdx.x` — a good stepping stone before combining blocks AND threads.
- The combined index formula is now something you've verified with a self-checking kernel, not just trusted from a slide.
- The ceiling-division + boundary-check pattern is something you've now BROKEN on purpose (by removing the guard) to see why it matters.
- Numba's `@cuda.jit` really does mirror the C++ pattern 1:1, just in Python syntax.
- N³ growth in matrix multiplication is now something you've measured with a real stopwatch, not just read as a formula.
- Removing `__syncthreads()` can produce a real, visible race condition — not just a theoretical warning.

---

> *GPU Programming · Lecture 04 · github.com/rpaut03l/TS-02-03*
