# 💻 Lecture 03 — Thread Hierarchy, Memory & Your First Kernel: HANDS-ON CODE

### *Runnable Python + CUDA C — see the register limit, dim3, and index math with your own eyes*

> **Nav:** [← Lecture 03 README](README.md) | [📖 THEORY](gpu_lecture03_threads_memory_theory.md) | [🔢 NUMERICAL](gpu_lecture03_threads_memory_numerical.md) | **CODE** | [🎯 PRACTICE](gpu_lecture03_threads_memory_practice.md)

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

## Ex 1. Your literal first kernel — the real Hello World

```python
%%cuda
#include <stdio.h>

__global__ void mykernel(void) {
}

int main(void) {
    mykernel<<<1,1>>>();
    printf("Hello World!\n");
    return 0;
}
```

### 👶 What this does
This is the EXACT program from [theory.md §8](gpu_lecture03_threads_memory_theory.md#8-hello-world-with-device-code), typed in and run for real. `mykernel<<<1,1>>>()` launches 1 block with 1 thread — the smallest possible GPU launch. The kernel body does nothing at all; the point of this exercise is purely to prove the launch mechanism itself works before we ask it to do anything useful.

### 👶 What you should see
```
Hello World!
```
Just that one line — because the kernel is empty, there's no GPU-side output to see; only the CPU's `printf` produces visible text.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-hands-on-code)

---

## Ex 2. Make the kernel actually DO something — print from the GPU

```python
%%cuda
#include <stdio.h>

__global__ void mykernel(void) {
    printf("Hello from the GPU! blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
}

int main(void) {
    mykernel<<<2,4>>>();
    cudaDeviceSynchronize();
    printf("Hello from the CPU!\n");
    return 0;
}
```

### 👶 What this does
Now the kernel body has `printf` too — yes, CUDA kernels CAN print, which is one of the easiest ways to "see" what each individual thread is doing while learning. We launch `<<<2,4>>>` — **2 blocks, 4 threads each** — so 8 threads total will each print their own identity.

### 👶 Important detail — `cudaDeviceSynchronize()`
Without this line, the CPU's `printf("Hello from the CPU!\n")` might print BEFORE the GPU's output even appears on screen, because kernel launches are asynchronous (the CPU doesn't wait by default — this is exactly the synchronization rule from [Lecture 04 §5](../Lecture_04_Vector_Matrix_Kernels_Indexing/gpu_lecture04_indexing_kernels_theory.md#5-coordinating-host--device)). Try commenting this line out and re-running — notice the CPU line can appear mixed in with, or even before, some of the GPU lines.

### 👶 What you should see (order of GPU lines may vary!)
```
Hello from the GPU! blockIdx.x=0 threadIdx.x=0
Hello from the GPU! blockIdx.x=0 threadIdx.x=1
Hello from the GPU! blockIdx.x=0 threadIdx.x=2
Hello from the GPU! blockIdx.x=0 threadIdx.x=3
Hello from the GPU! blockIdx.x=1 threadIdx.x=0
Hello from the GPU! blockIdx.x=1 threadIdx.x=1
Hello from the GPU! blockIdx.x=1 threadIdx.x=2
Hello from the GPU! blockIdx.x=1 threadIdx.x=3
Hello from the CPU!
```
The GPU lines might NOT come out in this exact order every run — this is a live demonstration of the "execution order is undefined" warning from [theory.md §6](gpu_lecture03_threads_memory_theory.md#6-the-dim3-type). Run the cell 3–4 times and watch the block order occasionally shuffle.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-hands-on-code)

---

## Ex 3. See `dim3` with your own eyes — print all 4 built-in variables

```python
%%cuda
#include <stdio.h>

__global__ void showIndices(void) {
    printf("blockIdx=(%d,%d,%d)  threadIdx=(%d,%d,%d)  blockDim=(%d,%d,%d)  gridDim=(%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(void) {
    dim3 blocks(2, 2, 1);
    dim3 threads(3, 3, 1);
    showIndices<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 👶 What this does
This launches a 2×2 grid of blocks, each block a 3×3 grid of threads — `2×2×3×3 = 36` total threads, each printing all four built-in `dim3` values it can see. This is the single best way to build real intuition for the table in [theory.md §6](gpu_lecture03_threads_memory_theory.md#6-the-dim3-type) — instead of memorizing the table, you WATCH the actual numbers each thread receives.

### 👶 What you should see (36 lines, order may shuffle)
```
blockIdx=(0,0,0)  threadIdx=(0,0,0)  blockDim=(3,3,1)  gridDim=(2,2,1)
blockIdx=(0,0,0)  threadIdx=(1,0,0)  blockDim=(3,3,1)  gridDim=(2,2,1)
...
blockIdx=(1,1,0)  threadIdx=(2,2,0)  blockDim=(3,3,1)  gridDim=(2,2,1)
```
Notice `blockDim` and `gridDim` are IDENTICAL on every single line — every thread sees the SAME launch shape, no matter which block/thread it is. Only `blockIdx` and `threadIdx` change per-thread. Try changing `dim3 blocks(2,2,1)` to `dim3 blocks(4,1,1)` and re-running — predict what changes BEFORE you run it, then check yourself.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-hands-on-code)

---

## Ex 4. Verify the thread-343 problem from `numerical.md` — for real

```python
%%cuda
#include <stdio.h>

__global__ void findMe(int target) {
    int blockLinear  = blockIdx.y * gridDim.x + blockIdx.x;
    int threadLinear = threadIdx.y * blockDim.x + threadIdx.x;
    int globalIndex  = blockLinear * (blockDim.x * blockDim.y) + threadLinear;

    if (globalIndex == target) {
        printf("FOUND target %d -> blockIdx=(%d,%d,%d)  threadIdx=(%d,%d,%d)\n",
               target, blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

int main(void) {
    dim3 blocks(5, 4, 1);   // gridDim = (5,4,1), matching numerical.md section 1
    dim3 threads(5, 5, 1);  // blockDim = (5,5,1)
    findMe<<<blocks, threads>>>(343);
    cudaDeviceSynchronize();
    return 0;
}
```

### 👶 What this does
This kernel computes each thread's OWN flat global index (using the same "linearize the block, linearize the thread, combine them" method from [numerical.md §1](gpu_lecture03_threads_memory_numerical.md#1-the-thread-343-problem-rebuilt-from-zero)), and only the ONE thread whose computed index matches `target=343` prints anything.

### 👶 What you should see
```
FOUND target 343 -> blockIdx=(3,2,0)  threadIdx=(2,3,0)
```
This should match EXACTLY what you calculated by hand in [numerical.md §1](gpu_lecture03_threads_memory_numerical.md#1-the-thread-343-problem-rebuilt-from-zero) — a real GPU independently confirming your own paper arithmetic. Try changing `target` to `0`, `250`, `799`, and `1799` and confirm each against the worked examples in [numerical.md §2](gpu_lecture03_threads_memory_numerical.md#2-four-more-fully-worked-index-problems) (note: those examples use a different grid shape, `(6,3,1)` blocks × `(10,10,1)` threads — adjust `dim3 blocks`/`dim3 threads` to match if you want to verify those specific ones instead).

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-hands-on-code)

---

## Ex 5. See the register limit in action (why threads/block is capped)

```python
!nvcc --version
```
```python
%%cuda
#include <stdio.h>

__global__ void heavyKernel(float *out) {
    // Deliberately declares MANY local variables to inflate register usage per thread
    float a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8,i=9,j=10;
    float k=11,l=12,m=13,n=14,o=15,p=16,q=17,r=18,s=19,t=20;
    out[threadIdx.x] = a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t;
}

int main(void) {
    float *d_out;
    cudaMalloc(&d_out, 1024 * sizeof(float));
    heavyKernel<<<1, 1024>>>(d_out);   // try launching with 1024 threads in ONE block
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Launch succeeded with 1024 threads in 1 block.\n");
    }
    cudaFree(d_out);
    return 0;
}
```
```bash
!nvcc -Xptxas -v -c heavy.cu 2>&1 | grep -i register || echo "(compile the .cu directly to see register count in nvcc -Xptxas -v output)"
```

### 👶 What this does
This kernel deliberately declares 20 local `float` variables per thread — inflating register usage on purpose, directly demonstrating the register-file constraint from [theory.md §2](gpu_lecture03_threads_memory_theory.md#2-organization-of-thread-blocks). With `1024` threads and only 20-ish registers each, this is still well within typical modern GPU limits (T4 has a large register file), so it should launch successfully — but the exercise is about SEEING the compiler-reported register count, not necessarily triggering a failure.

### 👶 What you should see
```
Launch succeeded with 1024 threads in 1 block.
```
For the register-count check, compile the same kernel as a standalone `.cu` file with `nvcc -Xptxas -v` (verbose PTX assembler output) to see a line like `Used 22 registers` — THAT number, multiplied by your thread count, is exactly the arithmetic from [Lecture 02 numerical.md §1](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_numerical.md#1-the-occupancy-calculation-every-step-slowly), now grounded in a real compiled number instead of a made-up example.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-hands-on-code)

---

## Ex 6. The full 9-step sequence, end to end, for real

```python
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void doubleValues(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] = data[idx] * 2;
}

int main(void) {
    // STEP 1: Allocate CPU Data Structure
    int n = 20;
    int *h_data = (int *)malloc(n * sizeof(int));

    // STEP 2: Initialize Data on CPU
    for (int i = 0; i < n; i++) h_data[i] = i + 1;   // 1,2,3,...,20

    printf("Before: ");
    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);
    printf("\n");

    // STEP 3: Allocate GPU Data Structure
    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));

    // STEP 4: Copy Data from CPU to GPU
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    // STEP 5: Define Execution Configuration
    int threadsPerBlock = 8;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // STEP 6: Run Kernel
    doubleValues<<<numBlocks, threadsPerBlock>>>(d_data, n);

    // STEP 7: CPU synchronizes with GPU
    cudaDeviceSynchronize();

    // STEP 8: Copy Data from GPU to CPU
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("After:  ");
    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);
    printf("\n");

    // STEP 9: De-allocate GPU and CPU memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

### 👶 What this does
Every single one of the 9 steps from [theory.md §3](gpu_lecture03_threads_memory_theory.md#3-the-9-step-sequence-for-gpu-programming) is labeled with a comment in this program, in order, doing genuinely useful (if simple) work: doubling every number in a 20-element list. Notice `n=20` does NOT divide evenly by `threadsPerBlock=8` (20/8 = 2.5), so this ALSO exercises the ceiling-division launch pattern you'll see formalized in [Lecture 04](../Lecture_04_Vector_Matrix_Kernels_Indexing/gpu_lecture04_indexing_kernels_theory.md#4-handling-arbitrary-vector-sizes) — a nice preview of what's coming next.

### 👶 What you should see
```
Before: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
After:  2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40
```
Every value is exactly doubled — a satisfying, fully self-checking confirmation that all 9 steps worked correctly together. Try changing `threadsPerBlock` to `3` (an even more awkward, non-power-of-2 number) and confirm it still works identically — the 9-step recipe doesn't care what specific numbers you plug in, as long as steps 5 and 6 handle the launch math correctly.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-hands-on-code)

---

## 🧭 What we learned

- Kernels really can `printf` — the fastest way to build intuition for thread/block indices.
- `cudaDeviceSynchronize()` isn't optional if you want GPU output to appear before your next CPU line runs.
- Execution order across blocks is genuinely undefined — you can watch it shuffle between runs.
- The thread-343 (and friends) index math from `numerical.md` isn't just paper arithmetic — a real kernel computes the identical answer.
- Register usage per thread is a real, compiler-reported number (`nvcc -Xptxas -v`), not an abstract idea.

---

> *GPU Programming · Lecture 03 · github.com/rpaut03l/TS-02-03*
