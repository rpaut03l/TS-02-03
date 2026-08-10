# 📖 Lecture 04 — Vector/Matrix Kernels & Indexing: THEORY

> **Nav:** [← Lecture 04 README](README.md) | **THEORY** | [🎯 PRACTICE](gpu_lecture04_indexing_kernels_practice.md)

---

## 🧠 MNEMONIC: **"MIVA-CJMS"**

> **M**oving to parallel · **I**ndexing (blocks+threads) · **V**ector add main() · **A**rbitrary sizes/boundary checks · **C**oordinating host/device · **J**IT (Numba/CuPy/PyCUDA) · **M**atrix mult · **S**hared-memory reduction

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|---|---|
| 1 | Moving to Parallel — Blocks First | [§1](#1-moving-to-parallel-blocks-first) |
| 2 | Indexing Arrays with Blocks AND Threads | [§2](#2-indexing-arrays-with-blocks-and-threads) |
| 3 | The Combined Vector-Add Kernel & Full `main()` | [§3](#3-the-combined-vector-add-kernel--full-main) |
| 4 | Handling Arbitrary Vector Sizes | [§4](#4-handling-arbitrary-vector-sizes) |
| 5 | Coordinating Host & Device | [§5](#5-coordinating-host--device) |
| 6 | JIT Compilation for Python — Numba, CuPy, PyCUDA | [§6](#6-jit-compilation-for-python-numba-cupy-pycuda) |
| 7 | Matrix Multiplication as N² Independent Inner Products | [§7](#7-matrix-multiplication-as-n²-independent-inner-products) |
| 8 | Vector/Inner-Product Kernel & Shared-Memory Reduction | [§8](#8-vectorinner-product-kernel--shared-memory-reduction) |
| 9 | Cheat Sheet & Exam Hacks | [§9](#9-cheat-sheet--exam-hacks) |

---

## 1. Moving to Parallel — Blocks First

**GPU computing is about massive parallelism.** The very first step away from a single-thread kernel is trivial:
```cpp
add<<< 1, 1 >>>();     // executes add() ONCE
add<<< N, 1 >>>();     // executes add() N TIMES, in parallel
```
Instead of executing `add()` once, we execute it **N times in parallel** — one invocation per block.

**Vector addition, block-parallel only:**
```cpp
__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
```
**Terminology:**
- Each **parallel invocation** of `add()` is referred to as a **block.**
- The **set of blocks** is referred to as a **grid.**
- Each invocation can refer to its own block index using **`blockIdx.x`.**

By using `blockIdx.x` to index into the array, each block handles a DIFFERENT array element:
```
   Block 0            Block 1            Block 2            Block 3
c[0]=a[0]+b[0];    c[1]=a[1]+b[1];    c[2]=a[2]+b[2];    c[3]=a[3]+b[3];
```
On the device, each block executes **in parallel** — this is the whole point.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 2. Indexing Arrays with Blocks AND Threads

Once you use BOTH blocks *and* threads, indexing is no longer as simple as using `blockIdx.x` alone. Consider indexing an array with one element per thread, **8 threads/block**:

```
 threadIdx.x            threadIdx.x            threadIdx.x            threadIdx.x
0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
|-------blockIdx.x=0-| |-------blockIdx.x=1-| |-------blockIdx.x=2-| |-------blockIdx.x=3-|
```

**With `M` threads/block, a unique index for each thread is:**
```cpp
int index = threadIdx.x + blockIdx.x * M;
```

**Worked example — "which thread operates on the red element (array index 21)?"**
```
Array (32 elements, M=8 threads/block):

 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 [21] 22 23 24 25 26 27 28 29 30 31
 |------- M=8 -------|                              ^
                                              threadIdx.x = 5
                                       |---------- blockIdx.x = 2 ----------|

  int index = threadIdx.x + blockIdx.x * M;
            =      5       +      2      * 8;
            =  21;
```
Element 21 is touched by `threadIdx.x = 5` in `blockIdx.x = 2`. Notice `M=8` is being used as a HARDCODED constant here — the next section replaces it with a proper built-in.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 3. The Combined Vector-Add Kernel & Full `main()`

**What changes in `main()`:** use the built-in `blockDim.x` for threads-per-block instead of a hardcoded `M`:
```cpp
int index = threadIdx.x + blockIdx.x * blockDim.x;
```

**The combined kernel — parallel across BOTH blocks and threads:**
```cpp
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}
```

**Full `main()` — Part 1 (setup and allocation):**
```cpp
#define N (2048*2048)
#define THREADS_PER_BLOCK 512

int main(void) {
    int *a, *b, *c;             // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);
```

**Full `main()` — Part 2 (copy, launch, copy back, cleanup):**
```cpp
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```
`N/THREADS_PER_BLOCK` = `(2048*2048)/512` = `4,194,304 / 512` = **8,192 blocks**, each with 512 threads — 4,194,304 threads total, exactly matching `N`.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 4. Handling Arbitrary Vector Sizes

The `main()` above only works cleanly because `N` divides evenly by `THREADS_PER_BLOCK`. **Typical problems are NOT friendly multiples of `blockDim.x`.**

**Updated kernel launch (ceiling division):**
```cpp
add<<<(N + M-1) / M, M>>>(d_a, d_b, d_c, N);
```

**And the kernel MUST avoid accessing beyond the end of the arrays:**
```cpp
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}
```
The `if (index < n)` guard is what makes this kernel safe for ANY `n`, not just clean multiples of the block size — the ceiling-division launch always creates enough threads to cover `n`, sometimes with a few "extra" threads at the tail end that this guard simply does nothing for.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 5. Coordinating Host & Device

**Kernel launches are asynchronous** — control returns to the CPU immediately. The CPU needs to explicitly synchronize before consuming the results.

| Function | Behavior |
|---|---|
| `cudaMemcpy()` | **Blocks** the CPU until the copy is complete. The copy itself only begins once all preceding CUDA calls have completed. |
| `cudaMemcpyAsync()` | **Asynchronous** — does NOT block the CPU. |
| `cudaDeviceSynchronize()` | **Blocks** the CPU until ALL preceding CUDA calls have completed. |

**The practical takeaway:** if you use `cudaMemcpy()` (the blocking version) to pull results back, you get correctness "for free" because it implicitly waits. If you use `cudaMemcpyAsync()` for performance, YOU are responsible for adding an explicit `cudaDeviceSynchronize()` (or equivalent) before touching the results on the CPU side.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 6. JIT Compilation for Python — Numba, CuPy, PyCUDA

**Just-In-Time (JIT) compilation** for Python with CUDA enables the **dynamic compilation of Python code into machine code optimized for NVIDIA GPUs, at runtime.**

| Library | What it is |
|---|---|
| **Numba** | A widely-used open-source JIT compiler that translates a subset of Python and NumPy code into fast machine code for both CPUs and GPUs. For CUDA, Numba provides the **`@cuda.jit`** decorator, letting you write GPU kernels directly in Python and have them compiled and executed on the fly — **eliminating the need to write C/C++ CUDA code directly.** |
| **CuPy** | A NumPy-compatible array library for GPU-accelerated computing. Provides a familiar NumPy-style interface for numerical operations on GPUs, often relying on CUDA under the hood. Also offers low-level CUDA support and can integrate with JIT-compiled Python functions, similar to Numba. |
| **PyCUDA** | Provides **more fine-grained control** over the CUDA API within Python. Lets developers manage CUDA contexts, memory, and kernel execution directly — **greater flexibility, but requires a deeper understanding of CUDA programming.** |

This maps directly onto the [Basic/python](../Basic/python/) code in this repo: `01_vector_add_numba.py` (Numba), `02_matrix_mul_pycuda.py` (PyCUDA), and `03_matrix_mul_cupy.py` (CuPy) — one working example of each library named on this slide.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 7. Matrix Multiplication as N² Independent Inner Products

Each element of the product matrix **C** is generated by **row-column multiplication and reduction** of matrices **A** and **B** — this operation is exactly the **inner product (dot product)** of a row-vector and a column-vector.

```
       A                    B                   C
+--------------+       +-----+             +-----+
| row (1 x N)  |   x   |col  |     =       | one  |
|              |       |(Nx1)|             |value |
+--------------+       +-----+             +-----+
   N by N               N by N               N by N
```

**The key parallelism insight:** for size N×N matrices, the matrix multiplication `C = A · B` is equivalent to **N² independent (hence parallel) inner products** — one inner product computed per output cell, and every one of those N² inner products can run on its own thread simultaneously.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 8. Vector/Inner-Product Kernel & Shared-Memory Reduction

### Serial representation (the baseline)
```
   c = sum_i( a_i * b_i )
```
```cpp
double c = 0.0;
for (int i = 0; i < SIZE; i++)
    c += a[i] * b[i];
```

### Simple parallelization strategy
```
  a:  [ ] [ ] [ ] [ ] [ ] [ ]
       |   |   |   |   |   |     <- multiplications done IN PARALLEL
  b:  [ ] [ ] [ ] [ ] [ ] [ ]
       \___\___\___\___\___/
              +
              |                  <- summation is SEQUENTIAL
              v
              c
```
The element-wise multiplications are embarrassingly parallel; the final summation, naively, is not.

### A simple (naive) kernel — parallel multiply only
```cpp
__global__ void innerProduct(int *a, int *b, int *c) {
    int product[SIZE];
    int i = threadIdx.x;
    if (i < SIZE)
        product[i] = a[i] * b[i];
}
```
Called from host code as: `innerProduct<<<grid, block>>>(...);`

### The complete version — shared memory + `__syncthreads()` + reduction
```cpp
__global__ void innerProduct(int *a, int *b, int *c) {
    __shared__ int product[SIZE];      // (1) visible to ALL threads in the block

    int i = threadIdx.x;
    if (i < SIZE)
        product[i] = a[i] * b[i];

    __syncthreads();                    // (2) wait for every thread to finish writing

    if (threadIdx.x == 0) {             // (3) ONE thread does the final summation
        int sum = 0;
        for (int k = 0; k < SIZE; k++)
            sum += product[k];
        *c = sum;
    }
}
```

**What each numbered step is doing:**
1. **`__shared__ int product[SIZE];`** — first, make `product[i]` visible to ALL threads in the block by placing it in shared memory (a plain local array would only be visible to the thread that wrote it).
2. **`__syncthreads();`** — a **barrier**: makes sure every thread has finished its multiplication (finished its "workload") before ANY thread moves on to read the full `product[]` array. Without this, thread 0 might start summing before threads 5, 6, 7... have written their values — a race condition.
3. **`if (threadIdx.x == 0) { ... }`** — finally, the summation is assigned to just ONE thread, which loops over the whole `product[]` array sequentially. The slides are explicit that this is an **"extremely inefficient reduction"** — one thread doing `SIZE` sequential additions while the other 31+ threads in its warp sit idle is a textbook example of *not* using the GPU's parallelism where it still could be used (a proper tree-based reduction, doing `log2(SIZE)` parallel steps instead of `SIZE` sequential ones, is the standard fix — a topic for a later lecture).

**Aside:** `cudaThreadSynchronize()` is used on the HOST side to synchronize host and device (an older, now-superseded name for what modern CUDA calls `cudaDeviceSynchronize()`, from [§5](#5-coordinating-host--device)).

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-theory)

---

## 9. Cheat Sheet & Exam Hacks

```
BLOCK-ONLY PARALLEL:   add<<<N,1>>>();  kernel uses c[blockIdx.x]=a[blockIdx.x]+b[blockIdx.x];

BLOCK+THREAD INDEX:    int index = threadIdx.x + blockIdx.x * blockDim.x;
                       (use blockDim.x, NOT a hardcoded M, in real code)

ARBITRARY SIZE LAUNCH: add<<<(N+M-1)/M, M>>>(...);
BOUNDARY CHECK:        if (index < n) { ... }        <- always needed for non-multiple N

SYNC FUNCTIONS:
  cudaMemcpy()             -> BLOCKS CPU, waits for prior calls, then copies
  cudaMemcpyAsync()        -> does NOT block CPU
  cudaDeviceSynchronize()  -> BLOCKS CPU until ALL prior CUDA calls finish

JIT LIBRARIES:  Numba (@cuda.jit, Python kernel) | CuPy (NumPy-style, hidden kernel)
                | PyCUDA (raw C++ kernel string, fine-grained Python control)

MATMUL:  N x N matrices -> N^2 independent inner products (1 per output cell)

REDUCTION KERNEL PATTERN:
  __shared__ array          -> make visible to whole block
  __syncthreads()            -> barrier: wait for ALL threads before reading
  if (threadIdx.x==0) { ... }-> naive: 1 thread sums everything (SLOW, exam flags this)
```

### ⚡ Exam Hacks
1. **"Why use `blockDim.x` instead of a hardcoded constant `M`?"** — `blockDim.x` is always correct regardless of how the kernel is launched; a hardcoded `M` silently breaks if the launch configuration ever changes. Always prefer the built-in.
2. **Ceiling-division launch questions** — always show `(N + M - 1) / M` explicitly; don't just state the resulting block count without the formula, since that's where marks are awarded.
3. **"What's wrong with this reduction kernel?"** questions — if you see `if (threadIdx.x == 0) { for(...) sum += ...; }` after a shared-memory write, flag it as a naive/inefficient reduction: one thread does ALL the summation work sequentially while the rest of its warp is idle.
4. **`__syncthreads()` placement** — it must come AFTER all threads write to shared memory and BEFORE any thread reads values written by OTHER threads. Placing it in the wrong spot (or omitting it) is a classic "spot the race condition" question.
5. **Numba vs CuPy vs PyCUDA — one-line differentiator** — Numba: Python kernel via `@cuda.jit`. CuPy: NumPy-style, kernel fully hidden. PyCUDA: raw CUDA C++ kernel string + fine-grained Python-side control. If a question asks "which gives the MOST manual control," the answer is PyCUDA; "least code, most abstraction" is CuPy.

---

> *GPU Programming · Lecture 04 · github.com/rpaut03l/TS-02-03*
