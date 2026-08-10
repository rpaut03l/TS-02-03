# 📖 Lecture 02 — CUDA Programming Model, SIMT & Fermi Architecture: THEORY

> **Nav:** [← Lecture 02 README](README.md) | **THEORY** | [🎯 PRACTICE](gpu_lecture02_simt_fermi_practice.md)

---

## 🧠 MNEMONIC: **"PSG-WOM"**

> **P**rogramming models · **S**IMT vs SIMD · **G**PU microarchitecture/Fermi · **W**arps · **O**ccupancy scheduling · **M**emory hierarchy + dataflow

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|---|---|
| 1 | Programming Models — CUDA vs OpenCL | [§1](#1-programming-models-cuda-vs-opencl) |
| 2 | Program Execution — Software-to-Hardware Mapping | [§2](#2-program-execution-software-to-hardware-mapping) |
| 3 | SIMT vs SIMD | [§3](#3-simt-vs-simd) |
| 4 | GPU Microarchitecture | [§4](#4-gpu-microarchitecture) |
| 5 | Nvidia Fermi Architecture & the SM | [§5](#5-nvidia-fermi-architecture--the-sm) |
| 6 | Warps | [§6](#6-warps) |
| 7 | Scheduling Decisions — Worked Occupancy Example | [§7](#7-scheduling-decisions-worked-occupancy-example) |
| 8 | Latency × Throughput = Required Warps | [§8](#8-latency-x-throughput--required-warps) |
| 9 | GPU Memory Hierarchy | [§9](#9-gpu-memory-hierarchy) |
| 10 | Workload Dataflow — Start to Finish | [§10](#10-workload-dataflow-start-to-finish) |
| 11 | Kernel Launch Syntax — The Triple Chevron | [§11](#11-kernel-launch-syntax-the-triple-chevron) |
| 12 | Cheat Sheet & Exam Hacks | [§12](#12-cheat-sheet--exam-hacks) |

---

## 1. Programming Models — CUDA vs OpenCL

Multiple parallel programming models exist for GPUs: **CUDA** (Compute Unified Device Architecture, NVIDIA-only), **OpenACC**, **OpenCL** (cross-vendor), **Microsoft's DirectCompute**, plus third-party wrappers for Python, Perl, Fortran, Java, Ruby, Lua, MATLAB, IDL, and Mathematica, and compilers from PGI, RCC, HMPP, and Copperhead.

**CUDA and OpenCL use different names for the SAME underlying concepts:**

| OpenCL | CUDA |
|---|---|
| **Platform Model** | **Hardware Model** |
| CPU + OpenCL devices | CPU + NVIDIA GPUs |
| Compute Units | Streaming Multiprocessors (SMs) |
| Processing Elements | CUDA cores |
| **Execution Model** | **Programming Model** |
| NDRange | grid index |
| work-group | thread block |
| work-item | thread |
| **Memory Model** | **Memory Model** |
| global memory | global memory |
| constant memory | constant memory |
| local memory | shared memory |
| private memory | local memory |

⚠️ **The #1 trap in this table:** OpenCL's "local memory" = CUDA's "shared memory" (fast, per-block), while OpenCL's "private memory" = CUDA's "local memory" (per-thread). The word "local" means the OPPOSITE thing in each model — always double-check which framework a question is asking about.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 2. Program Execution — Software-to-Hardware Mapping

```
       HARDWARE (physical chip)                    SOFTWARE (what you write)
  +---------------------------------+     +---------------------------------------+
  |          CUDA Device            |     |                                       |
  +---------------------------------+     |                                       |
             |            |               |                                       |
  +------------------+  +------------------+
  | Streaming        |  | Streaming        |
  | Multiprocessor   |..| Multiprocessor   |
  |     (SM)         |  |     (SM)         |
  +------------------+  +------------------+
             |
             v
  +------------------+  +------------------+
  |   Thread Block    |..|   Thread Block  |
  +------------------+  +------------------+
             |
             v
  +----------+     +----------+     +----------+
  |   Warp   |  .. |   Warp   |     |   Warp   |
  +----------+     +----------+     +----------+
             |
             v
  +--------+  +--------+       +--------+
  | Thread |  | Thread |  ..   | Thread |
  +--------+  +--------+       +--------+
```

**A program consists of:**
- one or more **sequential threads** running on the **host** (CPU), and
- one or more **parallel kernels** suitable for execution on the parallel-computing GPU.

**Execution rules:**
- Only **one kernel executes at a time**, and that kernel runs on a set of lightweight parallel threads.
- Threads are grouped into **thread blocks** — a programming abstraction representing a group of threads that can execute serially OR in parallel.
- Multiple thread blocks are grouped to form a **grid**.

**Memory scope per level:**
```
       Thread  <----> Per-thread PRIVATE local memory
     Thread Block <--> Per-block SHARED memory
    Grid 0, Grid 1 <-> Per-application GLOBAL memory
```

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 3. SIMT vs SIMD

**SIMD (Single Instruction, Multiple Data):**
```
  Data:          [1][8]   [3][6]   [7][4]   [7][6]
                    \  /     \  /     \  /     \  /
  Instructions:      +         +        +        +
                     |         |        |        |
  Threads:          Th0       Th1      Th2      Th3
                     |         |        |        |
  SIMD Unit:      [busy]    [busy]    [busy]   [busy]
```
One instruction is applied to MULTIPLE pieces of data at once — a single fixed-width vector operation.

**SIMT (Single Instruction, Multiple Threads)** — the model GPUs actually use — takes SIMD one step further: instead of one instruction acting on packed data lanes, the SAME instruction is issued to many independent THREADS, each with its own registers and its own data address. This gives GPUs the flexibility of "many independent workers" while still getting SIMD-style hardware efficiency underneath (see [Lecture_04](../Lecture_04_Vector_Matrix_Kernels_Indexing/gpu_lecture04_indexing_kernels_theory.md) for JIT/Python framing of this same idea, and [warps](#6-warps) below for exactly how 32 threads execute in true lockstep).

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 4. GPU Microarchitecture

```
+------------------------------------------------------------------------------+
|                                    GPU                                       |
| +-------------------+  +-------------------+          +--------------------+ |
| | SIMT Core Cluster  | | SIMT Core Cluster  |          | SIMT Core Cluster  ||
| | +--------+--------+| | +--------+--------+|   ...    | +--------+--------+||
| | |  SIMT  |  SIMT  || | |  SIMT  |  SIMT  ||          | |  SIMT  |  SIMT  |||
| | |  Core  |  Core  || | |  Core  |  Core  ||          | |  Core  |  Core  |||
| | +--------+--------+| | +--------+--------+|          | +--------+--------+||
| +-------------------+  +-------------------+          +-------------------+  |
+---------------------------------+-------------------------------------------+
|                     Interconnection Network                                  |
+------------+------------------+--------------------------+-------------------+
| Memory     |    Memory        |          ...              |    Memory        |
| Partition  |    Partition     |                            |    Partition    |
+------------+------------------+--------------------------+-------------------+
| GDDR3/GDDR5|    GDDR3/GDDR5   |     Off-chip DRAM          |   GDDR3/GDDR5   |
+------------+------------------+--------------------------+-------------------+
```
GDDR = **G**raphics **D**ual-**D**ata **R**ate — the type of high-bandwidth off-chip memory used on discrete GPUs.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 5. Nvidia Fermi Architecture & the SM

**Chip-level layout:**
```
+----------------------------------------------------------------------+
|DRAM|H|                                                          |DRAM|
|I/F |o|      [SM] [SM] [SM] [SM] [SM] [SM] [SM] [SM]             |I/F |
|    |s|                                                          |    |
|    |t|------------------------------------------------------    |    |
|DRAM|I|                        L2 Cache                          |DRAM|
|I/F |/|------------------------------------------------------    |I/F |
|    |F|                                                          |    |
|    |G|      [SM] [SM] [SM] [SM] [SM] [SM] [SM] [SM]             |    |
|DRAM|i|                                                          |DRAM|
|I/F |g|                                                          |I/F |
|    |a|                                                          |    |
|    |T|                                                          |    |
|DRAM|h|                                                          |DRAM|
|I/F |d|                                                          |I/F |
+----------------------------------------------------------------------+
   HOST I/F = Host Interface   |   GigaThread = kernel/block dispatch engine
```

**Inside ONE Fermi SM (Streaming Multiprocessor):**
```
+-------------------------------------------------------------------------+
|                         Instruction Cache                               |
+-----------------------------------+-------------------------------------+
|    Warp Scheduler                 |      Warp Scheduler                 |
|    Dispatch Unit                  |      Dispatch Unit                  |
+-----------------------------------+-------------------------------------+
|              Register File (32,768 x 32-bit)                            |
+--------+--------+--------+--------+
| Core   | Core   | LD/ST  | SFU    |    (repeated columns of Core pairs,
| Core   | Core   | LD/ST  |        |     16 LD/ST units total, 4 SFUs)
| Core   | Core   | LD/ST  | SFU    |
| Core   | Core   | LD/ST  |        |
| Core   | Core   | LD/ST  | SFU    |
| Core   | Core   | LD/ST  |        |
| Core   | Core   | LD/ST  | SFU    |
| Core   | Core   | LD/ST  |        |
+--------+--------+--------+--------+
|                  Interconnect Network                                     |
+---------------------------------------------------------------------------+
|              64 KB Shared Memory / L1 Cache (configurable)                |
+---------------------------------------------------------------------------+
|                            Uniform Cache                                  |
+---------------------------------------------------------------------------+

  ONE CUDA Core internally:
  Dispatch Port -> Operand Collector -> [ FP Unit | INT Unit ] -> Result Queue
```

**Key facts about the Fermi SM:**
- **16 Load/Store (LD/ST) units** — calculate source and destination addresses for **16 threads per clock**.
- **4 Special Function Units (SFUs)** per SM — execute transcendental instructions: **sin, cosine, reciprocal, and square root.**
- **Dual warp schedulers** — two warps can be issued instructions in the same clock cycle.
- **32,768 × 32-bit register file** shared dynamically across all threads on that SM.
- **64 KB configurable memory** — split as Shared Memory / L1 Cache depending on kernel needs.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 6. Warps

- A **thread block is composed of warps.**
- A **warp** is a set of **32 threads** within a thread block such that all threads in the warp **execute the same instruction** at any given time. Warps are selected **serially by the SM.**
- Each warp executes in a **SIMD fashion** — this is the "T" in SIMT: the hardware bundles 32 individual threads and marches them through the same instruction stream together.

**How threads are numbered into warps:**
```
Warp 0: thread  0, thread  1, thread  2, ... thread 31
Warp 1: thread 32, thread 33, thread 34, ... thread 63
Warp 2: thread 64, thread 65, thread 66, ... thread 95
Warp 3: thread 96, thread 97, thread 98, ... thread 127
```
32 threads per warp, always — this is a fixed hardware constant across NVIDIA GPU generations.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 7. Scheduling Decisions — Worked Occupancy Example

**Given:** a 16×16 thread block, max 1024 threads/SM, max 8 blocks/SM, 32 warps/SM supported.

```
STEP 1:  16 x 16 thread block = 256 threads/block

STEP 2:  Max threads per SM = 1024
         1024 / 256 = 4  ->  4 blocks/SM (thread-slot limit)

STEP 3:  Block-slot limit is 8 blocks/SM.
         4 blocks is WITHIN that limit  ->  4 blocks CAN be scheduled on one SM.

STEP 4:  Total threads on this SM = 4 blocks x 256 threads = 1024 threads

STEP 5:  Threads split into warps of 32:
         1024 threads / 32 threads-per-warp = 32 warps

STEP 6:  Device supports 32 warps/SM.
         We are USING all 32 warps/SM  ->  100% OCCUPANCY achieved!
```

**Visual pipeline (logical -> hardware -> execution):**
```
 Logical view          Hardware view           Execution
+--------------+     +----------------+     +---------------------+
| Thread Block |     |  32 threads    |     |    CONTROL LOGIC    |
|  (wavy line  | --> |  32 threads    | --> |  [core][core]...    |
|   pattern)   |     |  32 threads    |     |  [core][core]...    |
|              |     |  32 threads    |     |  [core][core]...    |
|              |     |  32 threads    |     |     Multiprocessor  |
+--------------+     +----------------+     +---------------------+
```

**Why this matters:** occupancy is the single most-tested "does the student actually understand scheduling" question in this course — every number in the 6-step derivation above (256, 4, 8, 1024, 32, 32) has to be checked against a DIFFERENT hardware limit at each step, and the smallest limiting factor wins.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 8. Latency × Throughput = Required Warps

```
   Number of Required Warps = Instruction Latency (cycles) x Throughput (ops/cycle)
```

| GPU Model | Instruction Latency (cycles) | Throughput (ops/cycle) | Parallelism Needed (operations) |
|---|---|---|---|
| **Fermi** | 20 | 32 | 640 |
| **Kepler** | 20 | 192 | 3,840 |

**In warps:** Fermi needs `640 / 32 = 20 warps` in flight to fully hide latency; Kepler, with 6× the throughput per cycle, needs `3,840 / 32 = 120 warps` to stay saturated.

**The register trade-off (same idea as [Lecture 04's performance considerations](../Lec_04_Performance_Considerations/README.md) in more depth):**
```
Registers per SM:   Fermi = 32K   |   Kepler = 64K

  MORE THREADS, fewer registers/thread     FEWER THREADS, more registers/thread
  +--+--+--+--+--+--+                      +------+------+------+
  +--+--+--+--+--+--+                      +------+------+------+
  +--+--+--+--+--+--+                      +------+------+------+
  +--+--+--+--+--+--+                      +------+------+------+
  (many small squares = more warps          (fewer, bigger squares = fewer
   available to hide latency)                 warps, more risk of stalling)
```
```
                <----------- THROUGHPUT ----------->
        Cycle 1   v   v   v   v   v   v   v   v   v
        Cycle 2   v   v   v   v   v   v   v   v   v
LATENCY  ...      v   v   v   v   v   v   v   v   v
        Cycle 20  v   v   v   v   v   v   v   v   v
       [ every column must stay filled to maximize performance ]
```

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 9. GPU Memory Hierarchy

**Block/thread view:**
```
+----------------------------+     +----------------------------+
|          Block 0           |     |          Block 1           |
|       Shared memory        |     |       Shared memory        |
+---------------+------------+     +---------------+------------+
| Thread 0      | Thread 1   |     | Thread 0      | Thread 2   |
| Local Regs    | Local Regs |     | Local Regs    | Local Regs |
+---------------+------------+     +---------------+------------+
| Thread 2      | Thread 3   |     | Thread 1      | Thread 3   |
| Local Regs    | Local Regs |     | Local Regs    | Local Regs |
+---------------+------------+     +---------------+------------+
              |                                  |
              +----------------+-----------------+
                               v
              +-------------------------------------+
              |               Global                |
              |   Constant            Texture       |
              +-------------------------------------+
                               ^
                        +-------------+
                        |  CPU Host   |
                        +-------------+
```

**Complete picture, speed-ordered (slower on the left, faster on the right):**
```
[System DRAM]  --PCI Bus-->  [Device DRAM]  --L2 Cache-->  [On-Chip per SM]  --> [Per-Thread]
CPU controlled                Global Memory                 Shared Memory        Registers
                               Local Memory                  L1 Cache
                               Constant Memory                Texture Cache
                               Texture Memory                 Constant Cache

        <---------------------------  SLOWER                FASTER  --------------------------->
```
**Texture memory (special-purpose):** a dedicated read-only memory system for storing 2D/3D texture data (image/pattern data used to render detailed 3D scenes). It features hardware caching optimized for spatial access, and supports specialized addressing modes like clamping and interpolation.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 10. Workload Dataflow — Start to Finish

```
STEP 1: Data is copied from the HOST memory to the DEVICE memory, via the PCIe Bus.
STEP 2: The Host launches a kernel on the Device.
STEP 3: The kernel is executed by MULTIPLE THREADS concurrently.
STEP 4: The data within the device is accessed by threads through the memory hierarchy.
STEP 5: Results are moved back to device memory and transferred back to the host via PCIe.
```
```
   Host (CPU)                                   Device (GPU)
+-------------+   (1) Copy to GPU mem      +--------------------------+
|             | -------------------------> |                          |
| Host Memory |   (2) Launch kernel        |     Device Memory        |
|             | -------------------------> |  (accessed via memory    |
|             |                            |    hierarchy by many     |
|             |   (3) Results copied back  |    concurrent threads)   |
|             | <------------------------- |                          |
+-------------+                             +-------------------------+
```
This is the SAME 3-step "Simple Processing Flow" from Lecture 01/02's slides, just spelled out into its full 5-numbered form: **copy in → launch → compute in parallel → access memory hierarchy → copy out.**

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 11. Kernel Launch Syntax — The Triple Chevron

The host calls a kernel using a **triple chevron** `<<< >>>`. Inside the chevrons go **(number of blocks, number of threads per block).**

```
  Kernel1<<<100, 256>>>();   -> launches 100 blocks x 256 threads/block
                                 = 25,600 threads total

  Kernel2<<<50, 1024>>>();  -> launches 50 blocks x 1024 threads/block
                                 = 51,200 threads total
```

**Multiple kernels can run across multiple streams, overlapping in time:**
```
GPU
stream 0:  [ kernel 0 ]              [ kernel 2 ]
stream 1:            [ kernel 1 ]              [ kernel 3 ]
           ---------------------------------------------------> Time
```

**Grids of blocks of threads, with real index values (`someKernel<<<1, 1>>>()`):**
```
gridDim.x = 1     blockDim.x = 1     blockIdx.x = 0     threadIdx.x = 0
```
**A bigger example** — `dim3 blocks(2,1,1); someKernel<<<blocks, 4>>>();`:
```
gridDim.x = 2      blockDim.x = 4
blockIdx.x = 0,1
threadIdx.x = 0,1,2,3, 0,1,2,3     (resets to 0 at the start of EACH block)
```

**The full grid/block/thread picture:**
```
Host                    Device
+--------+     +-------------------------------------+
|Kernel 1|---->| Grid 1                              |
+--------+     |  Block(0,0) Block(0,1) Block(0,2)   |
               |  Block(1,0) Block(1,1) Block(1,2)   |
               +-------------------------------------+
+--------+     +-------------------------------------+
|Kernel 2|---->| Grid 2  (empty in this example)     |
+--------+     +-------------------------------------+

  Zoomed into Block(1,1):
  +-------------------------------------------------+
  | Thread(0,0) Thread(0,1) Thread(0,2) Thread(0,3) |
  | Thread(1,0) Thread(1,1) Thread(1,2) Thread(1,3) |
  +-------------------------------------------------+
```

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-theory)

---

## 12. Cheat Sheet & Exam Hacks

```
CUDA <-> OpenCL:   grid<->NDRange | thread block<->work-group | thread<->work-item
                   shared mem<->local mem | local mem<->private mem  (NAMES SWAP!)

SIMD: 1 instruction, packed data lanes (CPU-style, fixed width)
SIMT: 1 instruction, many independent THREADS (GPU-style, hardware-bundled)

FERMI SM: 32 CUDA cores | 16 LD/ST units | 4 SFUs | 2 warp schedulers
          32,768x32-bit register file | 64KB configurable Shared/L1

WARP = 32 threads, always. Warp n = threads [32n, 32n+31].

OCCUPANCY (16x16 block example):
  256 threads/block -> 1024/256=4 blocks/SM (within 8-block limit)
  4 blocks x 256 = 1024 threads -> /32 = 32 warps -> 100% occupancy

REQUIRED WARPS = LATENCY(cycles) x THROUGHPUT(ops/cycle) / 32
  Fermi: 20x32=640 ops -> 20 warps.  Kepler: 20x192=3840 ops -> 120 warps.

LAUNCH:  kernel<<<numBlocks, threadsPerBlock>>>(args);
```

### ⚡ Exam Hacks
1. **CUDA-OpenCL term-mapping questions** — the "local"/"private" swap between the two models is THE classic trap; write out the full 4-row memory table before answering.
2. **"Explain SIMT vs SIMD in one line each"** — SIMD: one instruction, packed vector data. SIMT: one instruction, many independent threads (each with own registers), hardware-bundled into warps.
3. **Fermi SM numbers are frequently tested as-is** — memorize: 32 cores, 16 LD/ST, 4 SFU, dual scheduler. Don't confuse "16 LD/ST units" with "16 cores" — they're different rows in the diagram.
4. **Occupancy problems ALWAYS have multiple limits** (thread-slot limit, block-slot limit, register limit) — compute each independently and take the MINIMUM; never assume only one limit applies.
5. **`<<<blocks, threads>>>` vs `dim3` multi-value examples** — when a question gives `dim3 blocks(2,1,1)` and asks for `blockIdx.x` values across the whole launch, remember `threadIdx.x` RESETS to 0 at the start of every new block — don't let it run continuously across blocks.

---

> *GPU Programming · Lecture 02 · github.com/rpaut03l/TS-02-03*
