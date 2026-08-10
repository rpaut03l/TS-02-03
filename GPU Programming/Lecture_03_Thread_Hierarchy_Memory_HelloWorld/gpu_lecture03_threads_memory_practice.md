# 🎯 Lecture 03 — Thread Hierarchy, Memory & Your First Kernel: PRACTICE

> **Nav:** [← Lecture 03 README](README.md) | [📖 THEORY](gpu_lecture03_threads_memory_theory.md) | **PRACTICE**

---

## Self-Test Questions

**Q1.** List all 9 steps of the GPU programming sequence, in order.

<details><summary>Answer</summary>

1. Allocate CPU Data Structure, 2. Initialize Data on CPU, 3. Allocate GPU Data Structure, 4. Copy Data from CPU to GPU, 5. Define Execution Configuration, 6. Run Kernel, 7. CPU synchronizes with GPU, 8. Copy Data from GPU to CPU, 9. De-allocate GPU and CPU memory.
</details>

**Q2.** What resource actually limits the maximum number of threads per block?

<details><summary>Answer</summary>

The number of registers that can be allocated across all threads (the SM's register file size) — not an arbitrary software cap.
</details>

**Q3.** `dim3 grid(4, 3);` — what are `gridDim.x`, `gridDim.y`, and `gridDim.z`?

<details><summary>Answer</summary>

`gridDim.x = 4`, `gridDim.y = 3`, `gridDim.z = 1` (unspecified z defaults to 1).
</details>

**Q4.** Your own worked index problem: a grid has `gridDim = (4, 3, 1)` (4 blocks wide, 3 tall) and each block has `blockDim = (4, 4, 1)` (16 threads/block). Find the `blockIdx` and `threadIdx` coordinates for thread number 30 (0-indexed, flat).

<details><summary>Answer</summary>

Block size = 16. `30 = 16 × 1 + 14` → quotient = 1 (block, linear), remainder = 14 (thread, linear).

Block linear index 1, grid width 4 → `blockIdx.x = 1 mod 4 = 1`, `blockIdx.y = 1 div 4 = 0` → `(1, 0, 0)`.

Thread linear index 14, block width 4 → `threadIdx.x = 14 mod 4 = 2`, `threadIdx.y = 14 div 4 = 3` → `(2, 3, 0)`.
</details>

**Q5.** True or False: you can rely on block 0 finishing its work before block 1 starts, as long as they're in the same kernel launch.

<details><summary>Answer</summary>

**False.** Execution order across blocks (and threads) is undefined — never write code whose correctness depends on a specific completion order.
</details>

**Q6.** What does the `__global__` keyword tell the compiler about a function, in exactly two properties?

<details><summary>Answer</summary>

(1) The function runs ON the device (GPU). (2) The function is called FROM host code. Both properties must be stated for full credit.
</details>

**Q7.** What are the real Fermi hardware numbers for: number of SMs, CUDA cores total, and global memory size, as given in this lecture?

<details><summary>Answer</summary>

16 SMs, 512 CUDA cores total (32 per SM), 6 GB GDDR5 DRAM.
</details>

---

## Mini Exercise

Write out, from memory, the minimal three-part Hello World program: the empty `__global__` kernel, its launch line, and the `main()` function that calls it and prints "Hello World!". Then compile it for real on Colab/RunPod ([setup guide](../00_Environment_Setup_RunPod_Colab/README.md)) and confirm the output matches exactly.

---

## More Self-Test Questions

**Q8.** `dim3 g(7);` — write out `g.x`, `g.y`, `g.z`.

<details><summary>Answer</summary>

`g.x = 7`, `g.y = 1`, `g.z = 1` (unspecified trailing dimensions default to 1, not 0).
</details>

**Q9.** In a grid of `gridDim=(4,5,1)` with blocks of `blockDim=(8,8,1)`, find the `blockIdx`/`threadIdx` coordinates for the flat thread number 1500.

<details><summary>Answer</summary>

Threads/block = 64. `1500 div 64 = 23` remainder `28` (since 64×23=1472, 1500-1472=28). Block linear=23, thread linear=28.

Block (x,y) = (23 mod 4, 23 div 4) = (3, 5). Thread (x,y) = (28 mod 8, 28 div 8) = (4, 3).

`blockIdx=(3,5,0)`, `threadIdx=(4,3,0)`.
</details>

**Q10.** Why can't block 2 in a grid read a value that block 5 wrote to shared memory during the same kernel launch?

<details><summary>Answer</summary>

Shared memory is scoped per BLOCK, not per grid — each block gets its own private shared-memory allocation, invisible to every other block. This is consistent with the rule that blocks must be able to execute independently and cannot communicate/cooperate.
</details>

**Q11.** What's the difference between `__global__`, `__device__`, and `__host__` (the qualifiers mentioned alongside `__global__` in theory.md §8)?

<details><summary>Answer</summary>

`__global__`: runs on the device, callable from the host (a kernel). `__device__`: runs on the device, callable ONLY from device code (a helper function used inside kernels). `__host__`: runs on the host, callable from host code — effectively the default for ordinary CPU functions like `main()`. A function can even be marked `__host__ __device__` to compile a version for both.
</details>

**Q12.** Why does `nvcc` need to split source code into host and device components rather than compiling the whole `.cu` file with one single compiler?

<details><summary>Answer</summary>

Host code (like `main()`) needs to become ordinary CPU machine instructions, compiled by a standard host compiler like `gcc`. Device code (kernels) needs to become GPU machine instructions, compiled by NVIDIA's own compiler toolchain, which understands CUDA-specific constructs like thread/block indexing and the memory hierarchy. A single generic compiler can't produce both kinds of machine code from the same source, so `nvcc` acts as a coordinating driver that routes each part to the appropriate backend compiler.
</details>

**Q13.** A grid has 12 blocks total, arranged as `gridDim=(4,3,1)`. What is `blockIdx` for the LAST block in the grid (the highest-numbered one)?

<details><summary>Answer</summary>

`blockIdx = (3, 2, 0)` — the last valid x-index is `gridDim.x - 1 = 3`, and the last valid y-index is `gridDim.y - 1 = 2`. This is the same "last tenant" sanity-check pattern from numerical.md §2.
</details>

**Q14.** True or False: if you launch `kernel<<<1,1>>>()`, the register-file limit discussed in theory.md §2 is irrelevant since there's only one thread.

<details><summary>Answer</summary>

**Mostly true for THAT single launch**, since one thread's register usage will almost never approach a full SM's register file capacity alone. But the register-file limit is what determines how many CONCURRENT thread blocks/threads an SM can run — with a `<<<1,1>>>` launch, you're using a tiny fraction of the SM's capacity, so the register limit simply isn't the bottleneck in this specific (deliberately trivial) case. The limit still exists; it's just not the binding constraint here.
</details>

---

## Challenge Exercises

**Challenge 1 — Build your own worked index problem.** Pick your own grid size (e.g. `gridDim=(8,6,1)`) and block size (e.g. `blockDim=(4,4,1)`), pick a flat thread number of your choice, and work through the full quotient/remainder derivation from numerical.md §1, showing every step. Verify your answer using the `findMe` kernel pattern from code.md Ex 4 (adjust the `dim3` values to match your chosen sizes).

**Challenge 2 — Trace the 9-step sequence for a real problem.** Pick a computational task you understand well (e.g. resizing an image, computing a running average of stock prices). Write out, in your own words, exactly what each of the 9 GPU-programming steps would involve for that specific task — what data structure gets allocated, what the execution configuration would look like, etc.

**Challenge 3 — Extend the register-limit kernel.** Take the `heavyKernel` from code.md Ex 5 and keep adding more local `float` variables (try 40, then 80, then 160) — at each step, note the `nvcc -Xptxas -v` reported register count, and use the occupancy formulas from Lecture 02 to compute how many blocks/threads an SM could support at each register count. At what point does adding more registers start meaningfully reducing occupancy?

**Challenge 4 — Race condition scavenger hunt.** Modify the `showIndices` kernel from code.md Ex 3 so that every thread in a block writes to the SAME shared-memory location without any synchronization or ordering guarantee. Run it multiple times and observe whether the final value is consistent — this is a hands-on preview of exactly the kind of race condition warned about in Lecture 04's reduction kernel.

---

## 📝 Wrap-up — Reflect

Before moving to Lecture 04, confirm you can do these without notes:
1. Recite the 9-step GPU programming sequence in order.
2. Given ANY flat thread number and grid/block shape, derive the `blockIdx`/`threadIdx` coordinates using the quotient/remainder method.
3. Explain why `__global__`, `__device__`, and `__host__` are different, and give an example use case for each.

---

## ⚡ Rapid-Fire True/False Round

1. Kernel launches wait for the CPU before starting. <details><summary>A</summary>False — kernel launches are asynchronous; the CPU moves on immediately unless explicitly synchronized.</details>
2. `dim3 d(5);` sets `d.y` and `d.z` to 0. <details><summary>A</summary>False — they default to 1, never 0.</details>
3. Blocks in the same grid can communicate via shared memory. <details><summary>A</summary>False — shared memory is scoped per-block; blocks cannot communicate/cooperate.</details>
4. The maximum threads/block limit exists purely as an arbitrary software choice. <details><summary>A</summary>False — it's fundamentally limited by the SM's register file size.</details>
5. `__device__` functions can be called directly from host code. <details><summary>A</summary>False — only `__global__` functions can be launched from the host; `__device__` functions are callable only from other device code.</details>
6. Thread and block execution order is guaranteed to match their numeric index order. <details><summary>A</summary>False — explicitly undefined; never rely on completion order.</details>
7. Global memory data is stored in row-major order. <details><summary>A</summary>True.</details>
8. `nvcc` compiles BOTH host and device code with the same single compiler pass. <details><summary>A</summary>False — it splits source into host (via gcc-like compiler) and device (NVIDIA compiler) components.</details>
9. The 9-step GPU programming sequence always ends with de-allocating memory. <details><summary>A</summary>True — step 9 is "de-allocate GPU and CPU memory."</details>
10. Private memory (per-thread) is the FASTEST memory tier in the hierarchy. <details><summary>A</summary>True (via registers) — though "local memory" that spills to DRAM is a slower fallback within that same conceptual tier.</details>

---

## 🗂️ Quick Reference Card

```
9-STEP SEQUENCE:
  1. Allocate CPU data    4. Copy CPU->GPU        7. Sync CPU/GPU
  2. Init CPU data        5. Define exec config    8. Copy GPU->CPU
  3. Allocate GPU data    6. Run kernel            9. De-allocate both

dim3:  dim3 v(x[,y[,z]]);  unspecified trailing dims default to 1 (never 0)
BUILT-INS: threadIdx (in block) | blockIdx (in grid) | blockDim (threads/block)
           | gridDim (blocks/grid)

INDEX RECIPE (any flat number T, block size B, grid width W, block width Bw):
  block_linear  = T div B         block  (x,y) = (block_linear mod W,  block_linear div W)
  room_linear   = T mod B         room   (x,y) = (room_linear  mod Bw, room_linear  div Bw)

HELLO WORLD:  __global__ void k(void){}   mykernel<<<1,1>>>();
QUALIFIERS:   __global__ (host calls, device runs) | __device__ (device-only) |
              __host__ (default CPU function)
```

---

## 🏋️ Extended Challenge — Teach It Forward

Explain the 9-step GPU programming sequence to someone who has never seen CUDA before, using an analogy of your own invention (not the building/apartment one from numerical.md). Write it out as if it were a new "Easy Story" section for this lecture, then check: does your analogy correctly capture WHY steps 7 (synchronize) and 9 (de-allocate) matter, not just what they do?

---

## 🧩 Concept-Connection Questions

**C1.** Connect the register-file limit (§2) to the 9-step programming sequence (§3). At which specific step(s) would a programmer actually need to think about register usage, and why not earlier?

<details><summary>Answer</summary>

Register usage becomes relevant at **Step 5 (Define Execution Configuration)** — this is when you choose your block/grid sizes, which is exactly when the register-per-thread cost of your kernel interacts with the SM's fixed register file to determine actual achievable occupancy. It's not relevant at Steps 1-4 (those are pure data setup, no GPU execution decisions yet), and by Step 6 (Run Kernel) the configuration is already locked in — Step 5 is the one true decision point.
</details>

**C2.** Connect the memory hierarchy scope table (§5) to the Hello World kernel (§8). The empty `mykernel(void)` function doesn't touch ANY memory tier explicitly — but does it still technically use registers?

<details><summary>Answer</summary>

Yes, technically — even an "empty" kernel still needs SOME minimal register allocation for basic execution bookkeeping (e.g., holding `threadIdx`/`blockIdx` values internally, program counter state), though for a truly empty function body this is minimal to negligible. The larger point is that the register-file constraint applies to EVERY kernel launch, even trivial ones — it's just not the binding factor when the kernel does almost nothing.
</details>

**C3.** Connect `dim3`'s "undefined execution order" warning to why the thread-343 style index problems always compute a THREAD's OWN position, rather than relying on threads finishing in a predictable sequence.

<details><summary>Answer</summary>

Because completion order is undefined, kernels can NEVER assume "thread 5 finishes before thread 6" or similar orderings for correctness. Instead, every thread must be able to independently compute exactly where it stands (its own `blockIdx`/`threadIdx`) using only its own identity — never by inferring position from observed timing or sequence. This is precisely why the index formulas in this lecture are formulas (pure functions of `blockIdx`/`threadIdx`/`blockDim`/`gridDim`), not counters that increment as threads finish.
</details>

---

## 📋 Self-Check Before Moving On

- [ ] I can recite the 9-step GPU programming sequence in order, from memory
- [ ] I understand WHY the threads/block limit exists (register file), not just THAT it exists
- [ ] I can correctly initialize a `dim3` for 1D, 2D, and 3D shapes
- [ ] I can derive `blockIdx`/`threadIdx` from any flat thread number, AND go the reverse direction
- [ ] I can explain `__global__`/`__device__`/`__host__` and give a use case for each
- [ ] I've compiled and run a real Hello World kernel myself (not just read about it)

If any box is shaky, revisit [gpu_lecture03_threads_memory_theory.md](gpu_lecture03_threads_memory_theory.md) or rerun the exercises in [gpu_lecture03_threads_memory_code.md](gpu_lecture03_threads_memory_code.md) — Lecture 04 assumes fluent index-formula derivation as a prerequisite.

---

## 📝 Sample Exam Question (Multi-Part, Full Marks Breakdown)

> **Question (10 marks):** A grid has `gridDim=(8,5,1)` and `blockDim=(10,10,1)`. (a) [2 marks] Compute total tenants in the whole grid. (b) [3 marks] Find the `blockIdx`/`threadIdx` coordinates for flat tenant number 612. (c) [2 marks] Find the flat tenant number for `blockIdx=(3,2,0)`, `threadIdx=(7,1,0)` (the reverse direction). (d) [3 marks] Explain, in your own words, why threads/block here (100) would very likely be considered a "safe" choice with respect to the register-file limit, referencing the reasoning (not just the number) from theory.md §2.

<details><summary>Full worked answer</summary>

**(a)** Total tenants = `8 × 5 × 100 = 4000`.

**(b)** `612 div 100 = 6` remainder `12` → block_linear=6, room_linear=12.
Block (x,y) = `(6 mod 8, 6 div 8) = (6, 0)`.
Room (x,y) = `(12 mod 10, 12 div 10) = (2, 1)`.
**Answer: blockIdx=(6,0,0), threadIdx=(2,1,0)**

**(c)** block_linear = `blockIdx.y × gridDim.x + blockIdx.x = 2×8+3 = 19`.
room_linear = `threadIdx.y × blockDim.x + threadIdx.x = 1×10+7 = 17`.
tenant = `block_linear × roomsPerBuilding + room_linear = 19×100+17 = 1917`.
**Answer: tenant 1917**

**(d)** 100 threads/block is well below the typical 512-1024 thread/block guidance from real Fermi-class hardware (gpu_lecture03_threads_memory_theory.md §4), and well below common register-file capacities (32K-64K registers/SM) even at a generous 30-40 registers/thread — `100 threads × 40 registers = 4000 registers`, a small fraction of a 32,768-register file. The REASONING (not just "the number is small") is that the register-file constraint only becomes a binding concern when `threads_per_block × registers_per_thread` approaches the total register file size — at 100 threads/block, you'd need an unusually high per-thread register count before this became the limiting factor over the block-slot or thread-slot limits instead.
</details>

---


## 🔁 Formula Recap (memorize before the exam)

| Concept | Formula |
|---|---|
| Amdahl's Law speedup | S = 1 / ((1-P) + P/N) |
| Amdahl's Law max speedup | S_max = 1/(1-P) |
| Thread global index (1D) | idx = threadIdx.x + blockIdx.x * blockDim.x |
| Ceiling division (block count) | blocks = (N + M - 1) / M |
| Warp number from thread number | warp = threadNum div 32 |
| Occupancy | (blocks actually scheduled x threads/block) / 32 / maxWarpsPerSM |

> *GPU Programming · Lecture 03 · github.com/rpaut03l/TS-02-03*
