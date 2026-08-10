# 📖 Lecture 03 — Thread Hierarchy, Memory & Your First Kernel: THEORY

> **Nav:** [← Lecture 03 README](README.md) | **THEORY** | [🎯 PRACTICE](gpu_lecture03_threads_memory_practice.md)

---

## 🧠 MNEMONIC: **"ROSGH"**

> **R**ecap of execution flow · **O**rganization of thread blocks (register limit) · **S**equence of 9 programming steps · **G**PU H/W specs + `dim3` · **H**ello World

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|---|---|
| 1 | Program Execution — Recap | [§1](#1-program-execution-recap) |
| 2 | Organization of Thread Blocks | [§2](#2-organization-of-thread-blocks) |
| 3 | The 9-Step Sequence for GPU Programming | [§3](#3-the-9-step-sequence-for-gpu-programming) |
| 4 | Real Fermi Hardware Specs & S/W Abstraction Numbers | [§4](#4-real-fermi-hardware-specs--sw-abstraction-numbers) |
| 5 | Memory Hierarchy — Recap with Scope Rules | [§5](#5-memory-hierarchy-recap-with-scope-rules) |
| 6 | The `dim3` Type | [§6](#6-the-dim3-type) |
| 7 | Fully Worked Example — Thread 343 | [§7](#7-fully-worked-example-thread-343) |
| 8 | Hello World with Device Code | [§8](#8-hello-world-with-device-code) |
| 9 | Cheat Sheet & Exam Hacks | [§9](#9-cheat-sheet--exam-hacks) |

---

## 1. Program Execution — Recap

**The 5-step dataflow** (from [Lecture 02 §10](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_theory.md#10-workload-dataflow-start-to-finish)), now shown as a host-side timeline:

```
   CPU/Host                                          GPU/Device
      |
      | 1. Copy to GPU mem
      |------------------------------------------------->
      | 2. Launch GPU Kernel
      |------------------------------------------------->
      |                                              [ kernel runs on
      |                                                many threads ]
      | 2'. Synchronize with GPU
      |<---------------------------------------------(wait)
      | 3. Copy from GPU mem
      |<-------------------------------------------------
      v
    time
```

Notice step **2'** — synchronization is a DISTINCT step from the launch itself. Kernel launches are asynchronous (the CPU doesn't automatically wait), so the program must explicitly synchronize before it can safely read results back.

**Kernel/Block/Warp/Thread hierarchy recap** (triple-chevron launch syntax):
```
  Kernel1<<<100, 256>>>();   -> 100 blocks x 256 threads = 25,600 threads
  Kernel2<<<50, 1024>>>();   -> 50 blocks x 1024 threads = 51,200 threads
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 2. Organization of Thread Blocks

- It is convenient to organize thread blocks into **1D, 2D, or 3D arrays of threads.**
- The blocks in a grid **must be able to execute independently** — communication or cooperation BETWEEN blocks in a grid is **not possible.**
- When a kernel is launched, the number of threads per thread block AND the number of thread blocks together define the **total number of CUDA threads launched.**
- ⚠️ **The limit on threads per block exists because the number of REGISTERS that can be allocated across all threads is limited** — not an arbitrary software cap. This is the exact same register-file constraint worked through numerically in [Lecture 02 §7–8](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_theory.md#7-scheduling-decisions-worked-occupancy-example).

*(Reference: thebeardsage.com/cuda-dimensions-mapping-and-indexing/)*

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 3. The 9-Step Sequence for GPU Programming

Every CUDA program you write, without exception, follows this sequence:

```
1. Allocate CPU Data Structure
2. Initialize Data on CPU
3. Allocate GPU Data Structure
4. Copy Data from CPU to GPU
5. Define Execution Configuration     (grid size, block size)
6. Run Kernel
7. CPU synchronizes with GPU
8. Copy Data from GPU to CPU
9. De-allocate GPU and CPU memory
```

**Simple Processing Flow (the 3-step compressed version of the same idea):**
```
1. Copy input data from CPU memory to GPU memory
2. Load GPU program and execute, caching data on chip for performance
3. Copy results from GPU memory to CPU memory
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 4. Real Fermi Hardware Specs & S/W Abstraction Numbers

**GPU H/W Example — NVIDIA Fermi, real chip specs:**
```
16 Streaming Multiprocessors (SM)
512 CUDA cores (32/SM)
IEEE 754-2008 floating point (Double Precision AND Single Precision)
6 GB GDDR5 DRAM (Global Memory)
ECC Memory support
Two DMA interfaces                    (DMA = Direct Memory Access)

Reconfigurable L1 Cache / Shared Memory:  48 KB / 16 KB  (or the reverse split)
L2 Cache: 768 KB

CUDA Core internals:
  Load/Store address width: 64 bits
  Can calculate addresses of 16 threads per clock
```

**S/W Abstraction — the numbers that govern scheduling:**
```
Threads:  a kernel is executed by threads, processed by a CUDA Core.

Blocks:   512-1024 threads per block (typical range)
          Maximum 8 blocks per SM
          32 parallel threads execute at the same time in a WARP

Grids:    One grid per kernel, with multiple concurrent kernels possible
```

These are the SAME numbers behind the [Lecture 02 occupancy worked example](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_theory.md#7-scheduling-decisions-worked-occupancy-example) — now grounded in a real, named chip (Fermi) instead of abstract limits.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 5. Memory Hierarchy — Recap with Scope Rules

| Memory type | Visibility / scope |
|---|---|
| **Private memory** | Visible only to the thread |
| **Shared memory** | Visible to all threads in a block |
| **Global memory** | Visible to all threads; visible to host; accessible to multiple kernels; data stored in **row-major order** |
| **Constant memory** (read-only) | Visible to all threads in a block |

```
                                                          Registers
   Thread  <---------------------------------------->   Local Memory
   Block                                                  per Thread
+-----------+                                           Shared Memory
| Thread     |<-------------------------------------->    per Block
| Blocks x N |
+-----------+           +----------------+
   Grid 0    <---------> | Global Memory  |
+-----------+           |                |
   Grid 1    <---------> | Constant Memory|
+-----------+           +----------------+
```

**Row-major storage** matters for indexing: for a matrix stored in Global Memory, element `[row][col]` lives at flat offset `row * width + col` — the same formula used throughout [Lecture 04](../Lecture_04_Vector_Matrix_Kernels_Indexing/gpu_lecture04_indexing_kernels_theory.md) for matrix multiplication.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 6. The `dim3` Type

`dim3` is a **3D structure (vector type)** with three integers `x`, `y`, and `z`. You can initialize as many of the three coordinates as you like — the rest default to `1`:

```cpp
dim3 threads(256);            // x = 256, y = 1, z = 1
dim3 blocks(100, 100);        // x = 100, y = 100, z = 1
dim3 anotherOne(10, 54, 32);  // x = 10, y = 54, z = 32
```

**The four built-in variables every kernel can read**, all of type `dim3` (or its per-dimension `.x`/`.y`/`.z` components):

| Name | Description | .x | .y | .z |
|---|---|---|---|---|
| `threadIdx` | Thread index **within the block** (zero-based) | `threadIdx.x` | `threadIdx.y` | `threadIdx.z` |
| `blockIdx` | Block index **within the grid** (zero-based) | `blockIdx.x` | `blockIdx.y` | `blockIdx.z` |
| `blockDim` | Block dimensions, **in threads** | `blockDim.x` | `blockDim.y` | `blockDim.z` |
| `gridDim` | Grid dimensions, **in blocks** | `gridDim.x` | `gridDim.y` | `gridDim.z` |

⚠️ **Execution order is UNDEFINED.** During execution, CUDA threads/blocks are mapped to the problem and complete in an undefined order — you cannot assume block 0 finishes before block 1, or that threads within a block finish in numeric order. Never write correctness-dependent code that assumes a specific completion order.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 7. Fully Worked Example — Thread 343

**Setup:** A grid has **5 blocks in x, 4 blocks in y, 1 in z** (`gridDim = (5,4,1)`). Each block has **5 threads in x, 5 threads in y, 1 in z** (`blockDim = (5,5,1)`).

```
GRID LEVEL (blockIdx.y = index of block in y)          BLOCK LEVEL (threadIdx.y = index of thread in y)
              blockIdx.x -->                                          threadIdx.x -->
        (0,0,0) (1,0,0) (2,0,0) (3,0,0) (4,0,0)              (0,0,0) (1,0,0) (2,0,0) (3,0,0) (4,0,0)
          [0]     [5]    [10]    [15]    [20]                  [0]     [5]    [10]    [15]    [20]
        (0,1,0) (1,1,0) (2,1,0) (3,1,0) (4,1,0)              (0,1,0) (1,1,0) (2,1,0) (3,1,0) (4,1,0)
          [1]     [6]    [11]    [16]    [21]                  [1]     [6]    [11]    [16]    [21]
        (0,2,0) (1,2,0) (2,2,0) (3,2,0) (4,2,0)              (0,2,0) (1,2,0) (2,2,0) (3,2,0) (4,2,0)
          [2]     [7]    [12]    [17]    [22]                  [2]     [7]    [12]    [17]    [22]
        (0,3,0) (1,3,0) (2,3,0) (3,3,0) (4,3,0)              (0,3,0) (1,3,0) (2,3,0) (3,3,0) (4,3,0)
          [3]     [8]    [13]    [18]    [23]                  [3]     [8]    [13]    [18]    [23]
                                                               (0,4,0) (1,4,0) (2,4,0) (3,4,0) (4,4,0)
                                                                 [4]     [9]    [14]    [19]    [24]
```

**Total:** 20 blocks × 25 threads/block = 500 threads. Find the indices for **thread 343.**

**Step-by-step:**
```
STEP 1: Express thread 343 in terms of block size (25 threads/block).
        343 = 25 x 13 + 18

STEP 2: With 0-indexing, this means: the 13th block (0-indexed), position 18
        within that block, is thread 343.

STEP 3: Map the 13th block (linear index 13) to grid coordinates.
        From the GRID LEVEL layout above: linear index 13 -> block (3, 2, 0)

STEP 4: Map the in-block position to thread coordinates.
        From the BLOCK LEVEL layout above: linear index 17 -> thread (2, 3, 0)

RESULT: Thread 343 is indexed by:
        blockIdx.x  = 3        threadIdx.x = 2
        blockIdx.y  = 2        threadIdx.y = 3
        blockIdx.z  = 0        threadIdx.z = 0
```

**The general method, reusable for ANY thread number `T`, block size `B`, and grid width `W` (blocks per row):**
```
1. quotient  = T div B    -> which block (linear index)
2. remainder = T mod B    -> which thread within that block (linear index)
3. block  (x,y) = (quotient  mod W,  quotient  div W)
4. thread (x,y) = (remainder mod blockWidth,  remainder div blockWidth)
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 8. Hello World with Device Code

**Step 1 — the empty kernel:**
```cpp
__global__ void mykernel(void) {
}
```
- `__global__` is a CUDA C/C++ keyword marking a function that: **runs on the device**, and **is called from host code.**
- `nvcc` (the NVIDIA compiler driver) separates source code into host and device components: device functions (like `mykernel()`) are processed by the NVIDIA compiler; host functions (like `main()`) are processed by the standard host compiler (e.g. `gcc`).

**Step 2 — launching it:**
```cpp
mykernel<<<1,1>>>();
```
- The triple angle brackets `<<< >>>` mark a call from host code to device code — a **"kernel launch."**
- The parameters `(1,1)` mean **1 block, 1 thread** — the smallest possible launch.
- That's genuinely all that's required to execute a function on the GPU.

**Step 3 — the complete program:**
```cpp
__global__ void mykernel(void) {
}

int main(void) {
    mykernel<<<1,1>>>();
    printf("Hello World!\n");
    return 0;
}
```
`mykernel()` does nothing (empty body) — this first example is purely about proving the launch mechanism works, not about doing real GPU computation yet.

**Compiling and running:**
```
$ nvcc hello.cu
$ a.out
Hello World!
$
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-theory)

---

## 9. Cheat Sheet & Exam Hacks

```
9-STEP SEQUENCE:  Allocate CPU -> Init CPU data -> Allocate GPU -> Copy CPU->GPU
                  -> Define exec config -> Run kernel -> Sync CPU/GPU
                  -> Copy GPU->CPU -> De-allocate

REGISTER LIMIT:   threads/block is capped by register-file size, NOT an arbitrary number
BLOCK INDEPENDENCE: blocks in a grid CANNOT communicate/cooperate with each other

FERMI SPECS:      16 SM | 512 cores (32/SM) | 6GB GDDR5 | ECC | 2 DMA interfaces
                  L1/Shared: 48KB/16KB configurable | L2: 768KB

dim3:             dim3 v(x[,y[,z]]);  unspecified dims default to 1
BUILT-INS:        threadIdx (in block) | blockIdx (in grid) | blockDim (threads/block)
                  | gridDim (blocks/grid)

INDEX RECIPE:     quotient = T div blockSize  -> which block (linearly)
                  remainder = T mod blockSize -> which thread (linearly)
                  then unflatten quotient/remainder into (x,y) using grid/block WIDTH

HELLO WORLD:      __global__ void k(void){}   mykernel<<<1,1>>>();
```

### ⚡ Exam Hacks
1. **"Why is there a limit on threads per block?"** — always answer with the REGISTER FILE, not "hardware limitation" vaguely — examiners want the specific resource named.
2. **Thread-index-from-flat-number problems** (like thread 343) — ALWAYS do quotient/remainder first (which block, which in-block position), THEN unflatten each into (x,y) coordinates using the grid/block width. Doing it in the wrong order is the #1 mistake.
3. **"Can block 3 read data written by block 1 in the same kernel?"** — No, by design: blocks must be independent and cannot communicate/cooperate within a grid.
4. **`dim3` partial-initialization questions** — remember any unspecified trailing dimensions default to 1, not undefined/garbage.
5. **"What does `__global__` mean vs a normal function?"** — always state BOTH properties together: runs ON the device, AND is called FROM the host — a one-sided answer loses marks.

---

> *GPU Programming · Lecture 03 · github.com/rpaut03l/TS-02-03*
