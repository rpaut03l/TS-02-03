# 📖 Basic — Code Walkthrough (Every Line, Explained)

> **Nav:** [← Basic README](README.md)

This file explains **every single line** of all 5 programs in this folder — for someone who has never read CUDA code before. Read it side by side with the actual `.cu`/`.py` files.

---

## 📚 Table of Contents

1. [cpp/01_vector_add.cu](#1-cppo1_vector_addcu)
2. [cpp/02_matrix_mul.cu](#2-cpp02_matrix_mulcu)
3. [python/01_vector_add_numba.py](#3-python01_vector_add_numbapy)
4. [python/02_matrix_mul_pycuda.py](#4-python02_matrix_mul_pycudapy)
5. [python/03_matrix_mul_cupy.py](#5-python03_matrix_mul_cupypy)

---

## 1. `cpp/01_vector_add.cu`

Imagine two long shopping lists of numbers, and you want a third list where each item is the sum of the matching items on the first two lists. That's all this program does — just at a scale of over a million items, and with a stopwatch comparing "one person doing it alone" (the CPU) against "a thousand people doing it together" (the GPU).

```cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
```
- `#include <iostream>` — brings in the tools to print text to the screen (`std::cout`).
- `#include <chrono>` — brings in a precise stopwatch/timer toolkit, used later to measure the CPU's speed.
- `#include <cuda_runtime.h>` — brings in every CUDA-specific word this file uses: `cudaMalloc`, `cudaMemcpy`, `__global__`, and so on. Without this line, the compiler wouldn't recognize any of them.

```cpp
__global__ void add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```
- `__global__` — a special label meaning "this function is a GPU kernel: the CPU can call it, but it actually runs on the GPU, across many workers at once."
- `void add(int *a, int *b, int *c, int n)` — the function's name is `add`. It takes four things: pointers to three number-lists (`a`, `b`, `c`) and `n`, the total count of numbers in each list. A "pointer" here just means "the address in memory where a list starts."
- `int i = threadIdx.x + blockIdx.x * blockDim.x;` — this is the single most important line in the whole file. Every single GPU worker that runs this function figures out its OWN unique seat number `i` this way: "which group (block) am I in" times "how many workers per group" plus "my position inside my own group." No two workers ever compute the same `i`.
- `if (i < n) {` — a safety check. Because of how the launch numbers are rounded (explained below), we sometimes create a few more workers than there are actual list items. This line makes sure a "leftover" worker doesn't try to touch a list position that doesn't exist.
- `c[i] = a[i] + b[i];` — the actual arithmetic: THIS worker takes item number `i` from list `a`, item number `i` from list `b`, adds them, and puts the answer into item number `i` of list `c`. Because a huge number of workers each do this simultaneously with a different `i`, the WHOLE list gets added all at once instead of one item at a time.

```cpp
int main() {
    int N = 1 << 20;
```
- `int main() {` — the starting point of the program, just like in any normal C/C++ program. This part runs on the CPU.
- `int N = 1 << 20;` — a shortcut way of writing "2 to the power of 20," which equals **1,048,576**. `1 << 20` means "take the number 1, and shift its bits 20 places to the left," which is a fast way to compute powers of 2. So `N` is our list length: just over a million numbers.

```cpp
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
```
- These are just NAME DECLARATIONS — we're saying "later, `a`, `b`, `c` will point to lists living in the computer's normal memory (RAM)," and "`d_a`, `d_b`, `d_c` will point to lists living in the GPU's own separate memory." The `d_` prefix is just a naming habit programmers use to remind themselves "this one lives on the Device (GPU)," not a special keyword.

```cpp
    cudaMallocHost(&a, N * sizeof(int));
    cudaMallocHost(&b, N * sizeof(int));
    cudaMallocHost(&c, N * sizeof(int));
```
- `cudaMallocHost(&a, N * sizeof(int));` — reserves enough CPU memory to hold `N` integers, and makes `a` point to the start of that reserved space. `N * sizeof(int)` calculates exactly how many bytes that is (`sizeof(int)` is almost always 4 bytes, so this is `N × 4` bytes). The special thing about `cudaMallocHost` (versus a plain `malloc`) is that this memory is "pinned" — locked in place so the operating system can't move it around, which makes copying it to the GPU faster later.
- The next two lines do the exact same thing for `b` and `c`.

```cpp
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
```
- Same idea as above, but `cudaMalloc` reserves memory **on the GPU** instead of the CPU. This is a completely separate pool of memory, physically located on the graphics card.

```cpp
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
```
- A normal CPU loop (nothing GPU-related yet) that fills our two input lists with test numbers: `a` becomes `[0, 1, 2, 3, ...]` and `b` becomes `[0, 2, 4, 6, ...]`. This just gives us predictable numbers so we can sanity-check the answer later (e.g., item 5 of `c` should end up being `5 + 10 = 15`).

```cpp
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;
```
- `cpu_start = ...now();` — starts a precise stopwatch, recording the exact current time.
- The `for` loop — this is the SAME addition, but done the OLD-FASHIONED way: one item at a time, in a simple loop, entirely on the CPU. This is our "one person doing it alone" baseline.
- `cpu_end = ...now();` — stops the stopwatch by recording the time again.
- `cpu_time = cpu_end - cpu_start;` — subtracts the two timestamps to get the elapsed duration, in milliseconds (`std::milli`).
- `std::cout << "CPU Time: " << cpu_time.count() << " ms"` — prints that duration to the screen.

```cpp
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
```
- `cudaMemcpy(destination, source, howManyBytes, direction)` — copies memory from one place to another. Here we copy list `a` (which lives on the CPU) into `d_a` (which lives on the GPU). `cudaMemcpyHostToDevice` tells CUDA the direction: FROM the host (CPU) TO the device (GPU). The GPU physically cannot reach into the CPU's memory on its own — this copy is required before the GPU can touch the data at all.
- The next line does the same thing for `b`.

```cpp
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
```
- `blockSize = 256` — we're choosing to organize our GPU workers into groups of 256. This is a common, GPU-friendly group size.
- `numBlocks = (N + blockSize - 1) / blockSize;` — figures out HOW MANY groups of 256 we need to cover all `N` items. The `+ blockSize - 1` part is a "round up" trick: without it, plain division would round DOWN and leave some items uncovered whenever `N` doesn't divide evenly by 256. With `N=1,048,576` and `blockSize=256`, this actually divides evenly (4,096 groups exactly), but the formula is written this "safe" way as a general habit.

```cpp
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
```
- These three lines set up a GPU-specific stopwatch (an "event" is CUDA's name for a timestamp marker recorded directly on the GPU's own timeline, which is more accurate than timing from the CPU side for GPU work).

```cpp
    cudaEventRecord(start);
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
```
- `cudaEventRecord(start);` — drops a timestamp flag right before we ask the GPU to start working.
- `add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);` — **this is the actual moment thousands of GPU workers spring into action.** The triple angle brackets `<<<numBlocks, blockSize>>>` tell the GPU "launch `numBlocks` groups, each with `blockSize` workers" — together, roughly 1,048,576 individual workers each run the `add` function body once, each with a different `i` (as computed inside the function, way up at the top of this walkthrough).
- `cudaEventRecord(stop);` — drops a second timestamp flag right after the launch instruction is issued.

```cpp
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
```
- `cudaMemcpy(c, d_c, ..., cudaMemcpyDeviceToHost);` — copies the FINISHED answer back from the GPU's memory (`d_c`) into the CPU's memory (`c`), so we can read/print it. Notice the direction flag flipped to `cudaMemcpyDeviceToHost` (opposite of before).
- `cudaEventSynchronize(stop);` — forces the CPU to WAIT until the GPU has genuinely reached the `stop` timestamp before continuing. This is necessary because GPU work happens in the background (asynchronously) — without this line, we might try to read the elapsed time before the GPU has actually finished, getting a meaningless number.

```cpp
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
```
- `cudaEventElapsedTime(&gpu_time, start, stop);` — calculates the time gap between our two GPU timestamps and stores it (in milliseconds) into the `gpu_time` variable.
- The `std::cout` line prints it, exactly like we did for the CPU time earlier.

```cpp
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c);
    return 0;
}
```
- `cudaFree(...)` — releases the GPU memory we reserved earlier with `cudaMalloc`, giving it back to the system. Skipping this is a "memory leak" — the GPU thinks that memory is still in use forever, even after our program stops needing it.
- `cudaFreeHost(...)` — the matching release for the pinned CPU memory we reserved with `cudaMallocHost`.
- `return 0;` — tells the operating system the program finished successfully (by C/C++ convention, `0` means "no errors").

[🔝 Back to Top](#-basic--code-walkthrough-every-line-explained)

---

## 2. `cpp/02_matrix_mul.cu`

A "matrix" is just a grid of numbers, like a spreadsheet. Multiplying two matrices means: for every cell in the answer grid, walk along one row of the first matrix and one column of the second matrix, multiplying matching pairs and adding them all up. This file does that using a small, human-checkable 5×5 example.

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define N 5
```
- `#include <stdio.h>` — brings in `printf`, the classic C way to print text (used instead of C++'s `std::cout` in this file).
- `#define N 5` — creates a "find and replace" constant: everywhere the compiler sees the word `N` later in this file, it substitutes the number `5`. So our matrices are fixed at 5×5, deliberately small enough to print and check by eye.

```cpp
__global__ void matrix_mul(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
```
- `__global__ void matrix_mul(...)` — again, a GPU kernel, this time named `matrix_mul`.
- `int row = blockIdx.y * blockDim.y + threadIdx.y;` — the SAME "which group, times group size, plus my position" formula from the vector-add example, but this time using the **`.y`** part of each value. This gives each worker its ROW number in the output grid.
- `int col = blockIdx.x * blockDim.x + threadIdx.x;` — the identical formula, but using the **`.x`** part, giving each worker its COLUMN number. Together, `(row, col)` is this worker's unique cell in the output grid — remember it as "row uses Y, column uses X."

```cpp
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}
```
- `if (row < n && col < n) {` — a safety check for BOTH dimensions this time, since we're now working on a 2D grid, not a 1D list.
- `int sum = 0;` — a running total, starting at zero, that will hold this worker's final answer.
- `for (int k = 0; k < n; k++) { sum += a[row*n+k] * b[k*n+col]; }` — this is the actual "row times column" multiplication. `k` walks from `0` to `n-1`. Each step: grab the `k`-th number along row `row` of matrix `a`, grab the `k`-th number down column `col` of matrix `b`, multiply them, and add the result into `sum`. After all `n` steps, `sum` holds the complete dot product.
- `a[row * n + k]` — matrices are stored as ONE long flat list in computer memory, not as a real 2D grid. `row * n + k` converts a "row, column" position into the correct single flat position — think of it as "skip past `row` complete rows (each `n` items long), then move `k` more steps into the current row."
- `c[row * n + col] = sum;` — writes this worker's final answer into its one cell of the output matrix, using the same flattening trick.

```cpp
int main() {
    int n = N;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * n * sizeof(int);
```
- `int n = N;` — copies our `#define`d constant (5) into a normal variable, so it can be passed as a function argument later (you can't pass a `#define` directly the same way).
- `int size = n * n * sizeof(int);` — calculates the total bytes needed for one whole matrix: `n × n` total numbers, times 4 bytes each (for an `int`).

```cpp
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
```
- `malloc(size)` — the classic (non-CUDA) way to reserve `size` bytes of regular CPU memory. `(int *)` in front just tells the compiler "treat the returned memory address as a pointer-to-integers." This is simpler than `cudaMallocHost` from the vector-add example — this program didn't bother with pinned memory since the matrices here are tiny.

```cpp
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
            b[i * n + j] = i * j;
        }
```
- A nested loop (a loop inside a loop): `i` walks through every row, `j` walks through every column within that row. For each cell, we fill matrix `a` with the value `row + col`, and matrix `b` with `row × col` — simple, predictable patterns you can check by hand, rather than random numbers.

```cpp
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
```
- Same pattern as the vector-add example: reserve matching space on the GPU, then copy the CPU-side matrices over to the GPU-side ones.

```cpp
    dim3 blockSize(N, N);
    dim3 gridSize((n + N - 1) / N, (n + N - 1) / N);
```
- `dim3 blockSize(N, N);` — creates a 2D group shape of `5 × 5 = 25` workers per group (remember `dim3` from Lecture 03 — this fills in `x=5, y=5, z=1`).
- `dim3 gridSize(...)` — the SAME ceiling-division "round up" trick from the vector-add example, but applied separately to BOTH the x and y directions, to figure out how many GROUPS we need to cover a `n × n` grid. Since `n` and `N` are both 5 here, this works out to exactly `1 × 1` group needed.

```cpp
    matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```
- The actual launch — this time using our `dim3` shapes instead of plain numbers, spawning a 2D grid of workers, one per output cell.

```cpp
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
```
- `cudaGetLastError()` — asks CUDA "did anything go wrong with that most recent launch instruction itself?" (like an invalid launch shape) and stores the answer.
- `cudaDeviceSynchronize();` — waits for the GPU to ACTUALLY finish running the kernel. This also lets us catch errors that happen DURING execution (like a thread reading outside an array), not just at launch time.
- `if (err != cudaSuccess) printf(...)` — if something went wrong, print a human-readable description of the error instead of silently producing garbage output.

```cpp
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
```
- Copies the finished answer matrix back from the GPU to the CPU, same direction-flag pattern as before.

```cpp
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
}
```
- Cleanup: `cudaFree` releases the GPU-side memory, and plain `free` releases the CPU-side memory (matching the plain `malloc` used earlier instead of `cudaMallocHost`).

[🔝 Back to Top](#-basic--code-walkthrough-every-line-explained)

---

## 3. `python/01_vector_add_numba.py`

The exact same vector-addition idea as file #1, but written entirely in Python, using a library called Numba that lets Python code run directly on the GPU.

```python
from numba import cuda
import numpy as np
```
- `from numba import cuda` — brings in Numba's GPU toolkit, including the special decorator we're about to use.
- `import numpy as np` — brings in NumPy, the standard Python library for working with lists of numbers efficiently; we'll use it to create our CPU-side test data.

```python
@cuda.jit
def vector_add_kernel(a, b, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = a[idx] + b[idx]
```
- `@cuda.jit` — this single line, placed directly above a function, is Numba's way of saying "compile the function below into a real GPU kernel." Without it, this would just be an ordinary (slow, CPU-only) Python function.
- `def vector_add_kernel(a, b, out):` — the function's name and its three inputs: two input lists and one output list.
- `idx = cuda.grid(1)` — Numba's shortcut for the "which group, times group size, plus my position" formula from the C++ example — it asks "what is MY unique worker number, in a 1-dimensional layout?" and every worker gets a different answer automatically.
- `if idx < out.size:` — the same boundary safety check as before; `.size` is NumPy's way of asking a list "how many items do you hold?"
- `out[idx] = a[idx] + b[idx]` — the actual work: this ONE worker adds ONE pair of numbers.

```python
def cuda_vector_addition(a_host, b_host):
    N = a_host.size

    a_device = cuda.to_device(a_host)
    b_device = cuda.to_device(b_host)
    out_device = cuda.device_array_like(a_host)
```
- `def cuda_vector_addition(a_host, b_host):` — a regular Python "wrapper" function that handles all the setup around the actual kernel — nothing GPU-specific yet in this line.
- `N = a_host.size` — asks the input list how many items it has.
- `a_device = cuda.to_device(a_host)` — copies list `a` from normal Python memory over to the GPU's memory, and gives us `a_device`, a handle pointing to that GPU copy. This is Numba's version of the C++ `cudaMemcpy(..., HostToDevice)`.
- `b_device = ...` — the same copy for `b`.
- `out_device = cuda.device_array_like(a_host)` — creates a brand-new, EMPTY array directly on the GPU, matching the size/type of `a_host`. This is where our answers will be written — Numba's version of `cudaMalloc`.

```python
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
```
- The exact same numbers and the exact same "round up" ceiling-division formula as the C++ file — 256 workers per group, and enough groups to cover every item in `N`. The `//` symbol in Python means "integer division" (throw away any remainder), matching C++'s default division behavior for whole numbers.

```python
    vector_add_kernel[blocks_per_grid, threads_per_block](a_device, b_device, out_device)
```
- **This is the launch.** Numba uses SQUARE BRACKETS `[blocks, threads]` instead of C++'s triple angle brackets `<<<blocks, threads>>>` — same meaning, different Python-friendly punctuation. The parentheses afterward `(a_device, b_device, out_device)` pass in our GPU-resident data, just like a normal Python function call.

```python
    return out_device.copy_to_host()
```
- Copies the finished answer back from the GPU into a normal Python/NumPy array we can print or use further — Numba's version of `cudaMemcpy(..., DeviceToHost)`.

```python
if __name__ == "__main__":
    N = 1_000_000
    a = 100 * np.random.rand(N)
    b = 100 * np.random.rand(N)
```
- `if __name__ == "__main__":` — a standard Python idiom meaning "only run the code below if this file was run directly (not imported by another file)."
- `N = 1_000_000` — one million elements this time. The underscores are purely for human readability — Python ignores them (`1_000_000` is identical to `1000000`).
- `a = 100 * np.random.rand(N)` — creates a list of `N` random decimal numbers between 0 and 1, then multiplies every one of them by 100 (so they range from 0 to 100 instead).
- `b = ...` — an independent second list of random numbers, same idea.

```python
    result_cuda = cuda_vector_addition(a, b)
    result_cpu = a + b

    assert np.allclose(result_cuda, result_cpu)
    print("CUDA vector addition successful!")
    print("First 5 GPU results:", result_cuda[:5])
    print("First 5 CPU results:", result_cpu[:5])
```
- `result_cuda = cuda_vector_addition(a, b)` — runs our whole GPU pipeline and stores the answer.
- `result_cpu = a + b` — NumPy lets you just write `+` between two lists to add them elementwise, entirely on the CPU — our "known correct" comparison answer.
- `assert np.allclose(result_cuda, result_cpu)` — checks that the two answers match closely enough (allowing for the tiny floating-point rounding differences explained in this repo's floating-point lecture); if they DON'T match, `assert` stops the program immediately with an error, which is a deliberate safety net during testing.
- The `print(...)` lines just display confirmation and a small sample of the actual numbers.

[🔝 Back to Top](#-basic--code-walkthrough-every-line-explained)

---

## 4. `python/02_matrix_mul_pycuda.py`

This one is different from the other two Python files: instead of writing the GPU logic in Python, we write it in ACTUAL CUDA C++ (as a text string!), and Python's job is only to compile that string and manage feeding data in and out.

```python
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
```
- `import numpy as np` — same as before, for creating test data.
- `import pycuda.autoinit` — importing this line has a hidden but important SIDE EFFECT: it automatically sets up ("initializes") a connection to the GPU for us, so we don't have to write that setup code ourselves.
- `import pycuda.driver as cuda` — brings in PyCUDA's toolkit for manually managing GPU memory and running kernels, and nicknames it `cuda` for shorter typing.
- `from pycuda.compiler import SourceModule` — brings in the tool that will compile our CUDA C++ code (written as a string, below) into something the GPU can run.
- `import time` — Python's basic stopwatch toolkit, used later for timing.

```python
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
```
- This whole block is a Python STRING (the triple quotes `"""..."""` mark the start and end of a multi-line piece of text) — but the text INSIDE is real, genuine CUDA C++ code, identical in spirit to the `matrix_mul` kernel from `cpp/02_matrix_mul.cu` (row/col via `.y`/`.x`, boundary check, dot-product loop, flattened indexing). Python isn't reading this as Python code at all right now — it's just holding onto it as plain text, to be handed to a real C++ compiler in the next step.

```python
mod = SourceModule(cuda_kernel_source)
matrix_mul_gpu = mod.get_function("matrix_mul_kernel")
```
- `mod = SourceModule(cuda_kernel_source)` — this is the moment that text string ACTUALLY gets compiled into real GPU machine code, using `nvcc` behind the scenes, right there while the Python program is running (this is why the very first run of a PyCUDA program is a bit slower — it's compiling C++ on the fly).
- `matrix_mul_gpu = mod.get_function("matrix_mul_kernel")` — grabs a Python-callable "handle" pointing at that now-compiled kernel, found by matching its name (`"matrix_mul_kernel"`) inside the compiled module.

```python
MATRIX_SIZE = 512
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
c_gpu_result_cpu = np.empty_like(a_cpu)
```
- `MATRIX_SIZE = 512` — this time we're using real-sized 512×512 matrices, not the tiny 5×5 example from the C++ file.
- `np.random.randn(MATRIX_SIZE, MATRIX_SIZE)` — creates a proper 2D grid (512 rows × 512 columns) of random decimal numbers.
- `.astype(np.float32)` — converts those numbers into 32-bit floating point, matching the `float` type used in our CUDA kernel string above (types must match between Python and the C++ kernel, or the GPU would misread the bytes).
- `c_gpu_result_cpu = np.empty_like(a_cpu)` — creates an empty CPU-side array with the same shape/type as `a_cpu`, ready to receive the GPU's answer later.

```python
a_gpu = cuda.mem_alloc(a_cpu.nbytes)
b_gpu = cuda.mem_alloc(b_cpu.nbytes)
c_gpu = cuda.mem_alloc(c_gpu_result_cpu.nbytes)
```
- `cuda.mem_alloc(a_cpu.nbytes)` — PyCUDA's version of `cudaMalloc`: reserves that many BYTES of memory on the GPU. `.nbytes` is a built-in NumPy property that tells you exactly how many bytes an array occupies.

```python
cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(b_gpu, b_cpu)
```
- `memcpy_htod` — PyCUDA's naming: **h**ost **to** **d**evice. Copies our CPU-side matrices into their matching GPU-side reserved spaces.

```python
BLOCK_SIZE = (16, 16, 1)
GRID_SIZE = (
    (MATRIX_SIZE + BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0],
    (MATRIX_SIZE + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1],
)
```
- `BLOCK_SIZE = (16, 16, 1)` — a group shape of 16×16 = 256 workers (the trailing `1` is the unused third dimension, same idea as CUDA C++'s `dim3`).
- `GRID_SIZE = (...)` — the SAME ceiling-division "round up" trick, computed separately for each dimension, to figure out how many 16×16 groups are needed to cover a 512×512 grid of output cells. Since `512` divides evenly by `16`, this works out to exactly `(32, 32)`.

```python
start_gpu = time.time()
matrix_mul_gpu(a_gpu, b_gpu, c_gpu, np.int32(MATRIX_SIZE), block=BLOCK_SIZE, grid=GRID_SIZE)
cuda.Context.synchronize()
gpu_time = time.time() - start_gpu
```
- `start_gpu = time.time()` — records the current time.
- `matrix_mul_gpu(...)` — the actual kernel launch. Notice the keyword arguments `block=` and `grid=` — this is PyCUDA's way of specifying launch shape, and IMPORTANTLY, `block` is listed BEFORE `grid` here, which is the OPPOSITE order from CUDA C++'s `<<<grid, block>>>` syntax — a very common point of confusion. Also notice `np.int32(MATRIX_SIZE)` — the CUDA kernel expects a specific-width C integer type, so we must explicitly convert the plain Python integer.
- `cuda.Context.synchronize()` — waits for the GPU to actually finish, exactly like `cudaDeviceSynchronize()` in C++ — necessary before we can trust our elapsed-time measurement.
- `gpu_time = time.time() - start_gpu` — subtracts the start time from the current time to get the elapsed duration.

```python
cuda.memcpy_dtoh(c_gpu_result_cpu, c_gpu)
```
- `memcpy_dtoh` — the reverse direction: **d**evice **to** **h**ost. Copies the finished answer back into our CPU-side array.

```python
start_cpu = time.time()
c_cpu_result = np.dot(a_cpu, b_cpu)
cpu_time = time.time() - start_cpu
```
- A plain CPU-only matrix multiplication using NumPy's built-in `np.dot`, purely as a "known correct" baseline to compare timing and correctness against — nothing GPU-related here.

```python
print(f"GPU Time: {gpu_time:.6f} s")
print(f"CPU Time: {cpu_time:.6f} s")
print("Results match:", np.allclose(c_cpu_result, c_gpu_result_cpu, atol=1e-3))
if gpu_time > 0:
    print(f"GPU is approximately {cpu_time / gpu_time:.2f}x faster than the CPU.")
```
- The `f"..."` strings are Python "f-strings" — the `{gpu_time:.6f}` part means "insert the value of `gpu_time` here, formatted to 6 decimal places."
- `np.allclose(..., atol=1e-3)` — checks the two results match within a small allowed error margin (`1e-3` = 0.001), never expecting bit-for-bit identical results between CPU and GPU floating-point math.
- The final `if` prints a friendly "X times faster" summary, guarding against a division-by-zero if `gpu_time` somehow came out as exactly `0`.

[🔝 Back to Top](#-basic--code-walkthrough-every-line-explained)

---

## 5. `python/03_matrix_mul_cupy.py`

This is the simplest-LOOKING file of the five — but that's the whole point. CuPy hides every single GPU detail (memory allocation, kernel writing, launch configuration) behind code that looks IDENTICAL to ordinary NumPy.

```python
import numpy as np
import cupy as cp
import time
```
- `import cupy as cp` — CuPy is designed to be a "drop-in" replacement for NumPy: almost every `np.something()` call has a matching `cp.something()` call that does the SAME thing, but runs on the GPU instead of the CPU.

```python
MATRIX_SIZE = 9048
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
```
- A much bigger matrix this time — 9048×9048 — to make the GPU's advantage dramatically visible. These lines create ordinary NumPy (CPU-side) random matrices, same as the PyCUDA example.

```python
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)
```
- `cp.asarray(a_cpu)` — this ONE line does the job that took TWO separate steps in the PyCUDA example (`cuda.mem_alloc` + `cuda.memcpy_htod`): it reserves space on the GPU AND copies the data there, all in a single function call. This is the essence of what "CuPy hides the details" means in practice.

```python
start_gpu = time.time()
c_gpu = cp.dot(a_gpu, b_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start_gpu
```
- `c_gpu = cp.dot(a_gpu, b_gpu)` — the matrix multiplication itself. There is NO kernel source code anywhere in this file, no block/grid sizing — CuPy automatically generates and launches an optimized kernel internally (using the same row/column/dot-product math as the other two files), completely invisible to us here.
- `cp.cuda.Stream.null.synchronize()` — CuPy operations run in the background (asynchronously), just like raw CUDA kernel launches — this line is CuPy's version of `cudaDeviceSynchronize()`, needed here specifically so our timing measurement is accurate (without it, we'd measure "how long it took to SCHEDULE the work," not "how long the work actually took").
- `gpu_time = time.time() - start_gpu` — the elapsed duration.

```python
result_gpu_to_cpu = cp.asnumpy(c_gpu)
```
- `cp.asnumpy(c_gpu)` — pulls the GPU-resident CuPy array back into an ordinary CPU-resident NumPy array — CuPy's one-line equivalent of `cudaMemcpy(..., DeviceToHost)`.

```python
start_cpu = time.time()
c_cpu_result = np.dot(a_cpu, b_cpu)
cpu_time = time.time() - start_cpu
```
- Same idea as the PyCUDA file: a plain NumPy CPU-only matrix multiply, timed the same way, purely as our comparison baseline.

```python
difference = np.abs(c_cpu_result - result_gpu_to_cpu)
print(f"Max absolute difference: {np.max(difference):.8f}")
print(f"Avg absolute difference: {np.mean(difference):.8f}")
```
- `np.abs(c_cpu_result - result_gpu_to_cpu)` — subtracts the two result matrices element-by-element and takes the absolute value of every difference (turning any negative differences positive), giving us a full grid of "how far off was each individual number."
- `np.max(difference)` — the single BIGGEST difference found anywhere in the whole grid.
- `np.mean(difference)` — the AVERAGE difference across the whole grid. Both of these numbers are expected to be small-but-nonzero — this is normal floating-point behavior, not a bug (explained fully in this repo's floating-point considerations lecture).

```python
if np.allclose(c_cpu_result, result_gpu_to_cpu, rtol=1e-5, atol=1e-4):
    print("Results match within floating-point tolerance.")
else:
    print("WARNING: results differ beyond expected tolerance.")
```
- `np.allclose(..., rtol=1e-5, atol=1e-4)` — checks the match using BOTH a relative tolerance (`rtol`, scales with the size of the numbers being compared) and an absolute tolerance (`atol`, a fixed small allowance) — the standard, correct way to compare floating-point results, never using the plain `==` operator for this.
- The `if`/`else` just prints a friendly message depending on the outcome.

```python
print(f"GPU Time: {gpu_time:.6f} s")
print(f"CPU Time: {cpu_time:.6f} s")
if gpu_time > 0:
    print(f"GPU is approximately {cpu_time / gpu_time:.2f}x faster than the CPU.")
```
- The exact same closing pattern as the PyCUDA file — print both times, then a friendly speedup summary. On a real T4 GPU, this consistently shows something dramatic like "18.69x faster" for a matrix this size — see [`Lec_03_VectorAdd_MatrixMul/python/gpu_lec03_python_code.md §5`](../Lec_03_VectorAdd_MatrixMul/python/gpu_lec03_python_code.md#5-real-benchmark-numbers) for the actual recorded numbers from this exact script.

[🔝 Back to Top](#-basic--code-walkthrough-every-line-explained)

---

> *GPU Programming · Basic · github.com/rpaut03l/TS-02-03*
