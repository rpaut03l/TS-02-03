# 🎯 Lecture 04 — Vector/Matrix Kernels & Indexing: PRACTICE

> **Nav:** [← Lecture 04 README](README.md) | [📖 THEORY](gpu_lecture04_indexing_kernels_theory.md) | **PRACTICE**

---

## Self-Test Questions

**Q1.** With `M = 16` threads/block, which thread (give `blockIdx.x` and `threadIdx.x`) touches array element 50?

<details><summary>Answer</summary>

`index = threadIdx.x + blockIdx.x * M` → `50 = threadIdx.x + blockIdx.x * 16`. `50 / 16 = 3` remainder `2` → `blockIdx.x = 3`, `threadIdx.x = 2`. Check: `2 + 3*16 = 2 + 48 = 50` ✓.
</details>

**Q2.** Write the boundary-safe launch configuration and kernel guard for adding two vectors of size `n = 10,000` with `256` threads/block.

<details><summary>Answer</summary>

Launch: `add<<<(10000 + 256 - 1) / 256, 256>>>(d_a, d_b, d_c, 10000);` → `(10255)/256 = 40` blocks (integer division), giving `40 × 256 = 10,240` threads. Kernel: `if (index < n) c[index] = a[index] + b[index];` — the guard silently skips the 240 "extra" threads beyond index 9999.
</details>

**Q3.** Which CUDA function blocks the CPU until the copy is complete, and only begins copying once all preceding CUDA calls have finished?

<details><summary>Answer</summary>

`cudaMemcpy()`.
</details>

**Q4.** You call `cudaMemcpyAsync()` to copy a result back to the host, then immediately read that host array in your next line of C code. What's wrong, and how do you fix it?

<details><summary>Answer</summary>

`cudaMemcpyAsync()` does NOT block the CPU — the copy may not have finished by the time you read the array, giving stale or garbage data. Fix: insert a `cudaDeviceSynchronize()` (or equivalent) call between the async copy and reading the result.
</details>

**Q5.** In one sentence each, differentiate Numba, CuPy, and PyCUDA.

<details><summary>Answer</summary>

Numba: write GPU kernels in plain Python via `@cuda.jit`, JIT-compiled at runtime. CuPy: NumPy-compatible array library, GPU kernel entirely hidden behind familiar array operations. PyCUDA: fine-grained control — you write a raw CUDA C++ kernel string and manage context/memory/execution directly from Python.
</details>

**Q6.** Why is `C = A · B` for two N×N matrices described as "N² independent inner products"?

<details><summary>Answer</summary>

Each of the N² output cells `C[row][col]` is computed as one inner (dot) product between row `row` of A and column `col` of B — and since every output cell's computation doesn't depend on any other output cell, all N² of these inner products can be computed simultaneously, in parallel.
</details>

**Q7.** In the shared-memory inner-product kernel, why is `__syncthreads()` called BEFORE the final summation, and what would go wrong without it?

<details><summary>Answer</summary>

It's a barrier ensuring every thread has finished writing its `product[i] = a[i] * b[i]` value into shared memory before thread 0 starts reading the whole array to sum it. Without it, thread 0 could start summing while some other threads haven't written their values yet — a race condition producing a wrong (and non-deterministic) sum.
</details>

**Q8.** What specific criticism does the lecture make about the reduction step `if (threadIdx.x == 0) { for(...) sum += product[k]; }`?

<details><summary>Answer</summary>

It's called out as an "extremely inefficient reduction" — a single thread does all `SIZE` additions sequentially while every other thread in its warp (and block) sits idle, wasting the GPU's parallel capability at exactly the step where it matters.
</details>

---

## Mini Exercise — Complete the Kernel

Given this skeleton, fill in the blanks to make it a correct, boundary-safe vector-add kernel for arbitrary `n`:

```cpp
__global__ void add(int *a, int *b, int *c, int n) {
    int index = ______________________________;
    if (____________)
        c[index] = a[index] + b[index];
}
```

<details><summary>Answer</summary>

```cpp
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}
```
</details>

---

## More Self-Test Questions

**Q9.** With `M=512` threads/block, which thread (`blockIdx.x`, `threadIdx.x`) touches array element 20,000?

<details><summary>Answer</summary>

`20000 div 512 = 39` remainder `32` (512×39=19968, 20000-19968=32). `blockIdx.x=39`, `threadIdx.x=32`. Verify: `32 + 39×512 = 32+19968=20000` ✓.
</details>

**Q10.** For `N = 3,000,000` elements and `THREADS_PER_BLOCK = 512`, compute: (a) does N divide evenly by 512? (b) the ceiling-division block count, (c) total threads launched, (d) how many "extra" threads the boundary check has to skip.

<details><summary>Answer</summary>

(a) `3000000 / 512 = 5859.375` — does NOT divide evenly.
(b) `(3000000 + 512 - 1) / 512 = 3000511/512 = 5860.36...` → truncates to **5860 blocks**.
(c) `5860 × 512 = 3,000,320 threads`.
(d) `3,000,320 - 3,000,000 = 320` extra threads skipped by the `if (index < n)` guard.
</details>

**Q11.** Why does `cudaMemcpyAsync()` require the programmer to be MORE careful than `cudaMemcpy()`, even though it can be faster?

<details><summary>Answer</summary>

`cudaMemcpyAsync()` returns control to the CPU immediately without waiting for the copy to finish, so the CPU could race ahead and read/use data that hasn't actually finished transferring yet, producing stale or garbage results. `cudaMemcpy()` is safer by default because it blocks the CPU until the copy genuinely completes — the tradeoff is potential wasted CPU time waiting.
</details>

**Q12.** In one sentence, why can CuPy's `cp.dot()` call be described as "hiding" the same computation that PyCUDA's raw kernel string performs explicitly?

<details><summary>Answer</summary>

Both ultimately launch a real CUDA kernel using the same block/thread/grid execution model and the same `row*n+k`/`k*n+col` style indexing math underneath — CuPy just auto-generates and manages that kernel internally so the programmer never sees or writes it, while PyCUDA requires the programmer to write and manage that exact logic by hand.
</details>

**Q13.** A matrix multiplication of two 1024×1024 matrices is run. How many independent inner products are computed, and how many total multiply-add operations does that represent?

<details><summary>Answer</summary>

Inner products = `1024² = 1,048,576`. Total multiply-adds = `1024³ = 1,073,741,824` (over 1 billion operations).
</details>

**Q14.** In the shared-memory inner-product kernel, what specific role does the line `if (threadIdx.x == 0)` play, and why is it criticized in the lecture?

<details><summary>Answer</summary>

It restricts the final summation loop to run on ONLY thread 0 within the block — every other thread in the block (and the rest of its warp) sits completely idle during that step. It's criticized as "extremely inefficient" because it wastes almost all of the GPU's parallel capability at exactly the step (the reduction) where more clever techniques (like a tree-based reduction) could still exploit parallelism.
</details>

**Q15.** True or False: if `N` divides evenly by `THREADS_PER_BLOCK`, you can safely skip the `if (index < n)` boundary check without any risk.

<details><summary>Answer</summary>

**True for that SPECIFIC launch configuration** — if the numbers divide evenly, ceiling division produces exactly enough threads with zero "extra," so there's no out-of-bounds risk. However, it's still considered defensive best practice to include the check anyway, since it protects the kernel against future changes to `N` or block size that might reintroduce a non-clean division — omitting it creates a silent landmine for future code changes.
</details>

---

## Challenge Exercises

**Challenge 1 — Build the "arbitrary size" test yourself.** Modify code.md Ex 3 to test THREE different values of `n` (e.g. 999, 1000, 1001) with the same `M=256`, and for each, compute by hand: the block count, total threads launched, and number of "extra" threads — before running the code to check yourself.

**Challenge 2 — Time the JIT vs raw C++ gap.** Run both code.md Ex 3 (raw CUDA C++) and Ex 4 (Numba) with `n = 10,000,000` instead of 1000, and time each using each language's own timing tools (`cudaEvent` for C++, `time.time()` + explicit sync for Python). Is there a meaningful performance difference? What could explain it?

**Challenge 3 — Extend the reduction kernel to be less naive.** The kernel in code.md Ex 6 assigns ALL summation work to thread 0. Try modifying it so that TWO threads (thread 0 and thread 4) each sum half of the `product[]` array, then thread 0 adds the two partial sums together. Does dividing the reduction work across 2 threads instead of 1 actually help at `SIZE=8`? At what array size might it start to matter more?

**Challenge 4 — Matrix multiplication scaling experiment.** Using the pattern from code.md Ex 5, extend the loop to `N = 1024, 2048` (warning: 2048 may take a while on CPU-only NumPy). Plot (by hand or in a spreadsheet) `total_ops` vs `elapsed_time` — does the relationship look linear, or does it match the N³ growth predicted in numerical.md §5?

---

## 📝 Wrap-up — Reflect

Before considering this lecture complete, confirm you can do these without notes:
1. Derive the combined block+thread index formula and explain why `blockDim.x` is preferred over a hardcoded constant.
2. Write a boundary-safe kernel launch for ANY array size, including the ceiling-division formula from memory.
3. Explain, precisely, what `__syncthreads()` guarantees and why the reduction kernel would be wrong without it.
4. Differentiate Numba, CuPy, and PyCUDA in one sentence each, and say which one you'd reach for if you needed maximum control vs minimum code.

---

## ⚡ Rapid-Fire True/False Round

1. `add<<<N,1>>>()` parallelizes across threads, not blocks. <details><summary>A</summary>False — it's block-only parallelism; each block runs 1 thread, indexed by `blockIdx.x`.</details>
2. `blockDim.x` is always preferable to a hardcoded constant `M` in real kernels. <details><summary>A</summary>True — it always matches the actual launch, even if launch parameters change.</details>
3. `cudaMemcpyAsync()` blocks the CPU until the transfer finishes. <details><summary>A</summary>False — it's explicitly non-blocking; you must synchronize separately if needed.</details>
4. Numba's `@cuda.jit` still requires you to write CUDA C++ somewhere. <details><summary>A</summary>False — the whole point is eliminating the need to write C/C++ CUDA directly.</details>
5. PyCUDA gives you LESS control than CuPy. <details><summary>A</summary>False — PyCUDA gives MORE fine-grained control (raw kernel + manual memory management); CuPy hides the most detail.</details>
6. Matrix multiplication of two N×N matrices requires N² total multiply-add operations. <details><summary>A</summary>False — it's N³ total operations (N² independent inner products, each doing N multiply-adds).</details>
7. `__syncthreads()` guarantees ALL threads across the ENTIRE grid have reached that line. <details><summary>A</summary>False — it only synchronizes threads WITHIN the same block, not across the whole grid.</details>
8. The naive reduction kernel in this lecture assigns summation work to every thread equally. <details><summary>A</summary>False — it's explicitly criticized for assigning ALL summation to just thread 0.</details>
9. Ceiling division `(N+M-1)/M` gives the same result as plain `N/M` whenever N divides evenly by M. <details><summary>A</summary>True — the "+M-1" only changes the result when there's a nonzero remainder.</details>
10. `cudaDeviceSynchronize()` and `cudaThreadSynchronize()` are, historically, referring to the same host-side synchronization idea. <details><summary>A</summary>True — `cudaThreadSynchronize()` is the older, now-superseded name for what modern CUDA calls `cudaDeviceSynchronize()`.</details>

---

## 🗂️ Quick Reference Card

```
BLOCK-ONLY:      add<<<N,1>>>();  c[blockIdx.x]=a[blockIdx.x]+b[blockIdx.x];
COMBINED INDEX:  int index = threadIdx.x + blockIdx.x * blockDim.x;
CEILING LAUNCH:  add<<<(N+M-1)/M, M>>>(...);
BOUNDARY GUARD:  if (index < n) { ... }

SYNC FUNCTIONS:
  cudaMemcpy()             -> blocks CPU, waits for prior calls
  cudaMemcpyAsync()        -> does NOT block CPU
  cudaDeviceSynchronize()  -> blocks CPU until ALL prior CUDA calls finish

JIT LIBRARIES:  Numba(@cuda.jit,Python kernel) | PyCUDA(raw C++ kernel string) |
                CuPy(NumPy-style, kernel hidden)

MATMUL:  N x N matrices -> N^2 independent inner products -> N^3 total operations

REDUCTION PATTERN:
  __shared__ array           -> visible to whole block
  __syncthreads()             -> barrier: ALL threads in block, not just some
  if (threadIdx.x==0){...}    -> naive: 1 thread sums (flagged as inefficient)
```

---

## 🏋️ Extended Challenge — Design a Better Reduction

The naive reduction kernel from this lecture assigns all summation work to thread 0. Sketch (in pseudocode, no need to fully implement) a TREE-based reduction instead, where in step 1 half the threads each add a neighbor's value, in step 2 half of THOSE add another neighbor, and so on, until one thread holds the final sum. How many steps does a tree reduction need for `SIZE=8` compared to the naive version's 8 sequential additions? Generalize: for a `SIZE` that's a power of 2, how many steps does the tree version need, expressed as a function of `SIZE`?

<details><summary>Hint / partial answer</summary>

A tree reduction needs `log2(SIZE)` steps (for `SIZE=8`, that's `log2(8)=3` steps) instead of `SIZE-1` sequential additions (7 for `SIZE=8`) — each step still needs a `__syncthreads()` between it and the next, since threads read values written by other threads.
</details>

---

## 🧩 Concept-Connection Questions

**C1.** Connect the boundary-check pattern (§4) to the shared-memory reduction kernel (§8). Does the reduction kernel's `if (i < SIZE)` guard serve the SAME purpose as the boundary check in the arbitrary-vector-size kernel, or a different one?

<details><summary>Answer</summary>

Same fundamental PURPOSE (prevent out-of-bounds access) but applied in a slightly different context: in the vector-add kernel, it guards against "extra" threads created by ceiling-division rounding up past `n`. In the reduction kernel, it guards `product[i] = a[i]*b[i]` specifically for the case where more threads are launched than `SIZE` — the same defensive pattern, reused because launching slightly more threads than needed (and guarding accordingly) is a general, recurring CUDA idiom, not a one-off trick specific to vector addition.
</details>

**C2.** Connect the JIT library comparison (§6) to the matrix multiplication N³ operation count (§7). If you were prototyping a NEW matrix algorithm and wanted to iterate quickly before writing optimized C++, which of Numba/CuPy/PyCUDA would you reach for first, and why?

<details><summary>Answer</summary>

CuPy — because for a NEW algorithm still being designed, you want to focus on the MATH/algorithm correctness first (does `cp.dot()` or a sequence of CuPy operations produce the right answer?), not on manually managing memory or hand-writing a kernel. Once the algorithm is validated and you know exactly what performance-critical inner loop needs hand-tuning, THEN dropping to PyCUDA (or Numba's `@cuda.jit` for a middle ground) for that specific bottleneck becomes worthwhile — premature hand-kernel-writing on an unproven algorithm wastes effort.
</details>

**C3.** Connect `cudaDeviceSynchronize()` (§5) to the reduction kernel's `__syncthreads()` (§8). Are these the same function under a different name, or genuinely different mechanisms? What's the key scope difference?

<details><summary>Answer</summary>

Genuinely different mechanisms, differing in SCOPE. `cudaDeviceSynchronize()` is called from HOST code and blocks the CPU until ALL preceding GPU work (potentially many kernels, across the whole device) completes. `__syncthreads()` is called from DEVICE code (inside a kernel) and only synchronizes threads WITHIN the same block — it has no effect on, and no awareness of, threads in other blocks or other kernels. Confusing the two (e.g., expecting `__syncthreads()` to synchronize the whole grid) is a common and serious correctness bug.
</details>

---

## 📋 Self-Check Before Moving On

- [ ] I can write the block-only `add<<<N,1>>>()` kernel from memory
- [ ] I can derive AND verify the combined block+thread index formula on paper and in code
- [ ] I can write a boundary-safe kernel launch for any array size, unprompted
- [ ] I understand the exact scope difference between `cudaDeviceSynchronize()` and `__syncthreads()`
- [ ] I can differentiate Numba / CuPy / PyCUDA and justify a choice for a given scenario
- [ ] I can explain why matrix multiplication is O(N³) total work despite N² parallel threads
- [ ] I've personally triggered and observed a race condition by removing `__syncthreads()`

This is the last lecture in the current four-part sequence — see [Syllabus_Roadmap.md](../Syllabus_Roadmap.md) for what comes next (Debugging/Profiling, Memory deep-dive, Synchronization primitives, and beyond).

---

## 📝 Sample Exam Question (Multi-Part, Full Marks Breakdown)

> **Question (10 marks):** You are given `n = 70,000` elements to process with `M = 512` threads/block. (a) [2 marks] Write the boundary-safe kernel launch line. (b) [2 marks] Compute the exact number of blocks launched. (c) [2 marks] Compute how many "extra" (guarded-off) threads exist. (d) [2 marks] For a SEPARATE matrix-multiplication task on two 300×300 matrices, compute the number of independent inner products and the total multiply-add operations. (e) [2 marks] Explain, referencing `cudaMemcpy()` vs `cudaMemcpyAsync()`, what could go wrong if a programmer replaced `cudaMemcpy()` with `cudaMemcpyAsync()` in a boundary-safe vector-add program without adding any other changes.

<details><summary>Full worked answer</summary>

**(a)** `add<<<(n + M - 1) / M, M>>>(d_a, d_b, d_c, n);`

**(b)** `(70000 + 512 - 1) / 512 = 70511 / 512 = 137.7...` → truncates to **137 blocks**.

**(c)** Total threads launched = `137 × 512 = 70,144`. Extra threads = `70,144 - 70,000 = 144`.

**(d)** Inner products = `300² = 90,000`. Total multiply-add operations = `300³ = 27,000,000`.

**(e)** `cudaMemcpyAsync()` does not block the CPU — if the programmer swaps it in for the final device-to-host copy (`cudaMemcpy(c, d_c, ..., DeviceToHost)`) without adding an explicit synchronization call (like `cudaDeviceSynchronize()`) before the CPU reads array `c`, the CPU could read `c` BEFORE the GPU has actually finished writing the results into it — producing stale, partially-written, or garbage data, with no compiler or runtime error to flag the mistake. This is a silent correctness bug, not a crash, which makes it especially dangerous.
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

> *GPU Programming · Lecture 04 · github.com/rpaut03l/TS-02-03*
