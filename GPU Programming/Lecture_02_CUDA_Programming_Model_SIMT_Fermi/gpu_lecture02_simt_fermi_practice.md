# 🎯 Lecture 02 — CUDA Programming Model, SIMT & Fermi Architecture: PRACTICE

> **Nav:** [← Lecture 02 README](README.md) | [📖 THEORY](gpu_lecture02_simt_fermi_theory.md) | **PRACTICE**

---

## Self-Test Questions

**Q1.** In OpenCL terminology, what is the equivalent of CUDA's "thread block"? What about CUDA's "shared memory"?

<details><summary>Answer</summary>

"thread block" → OpenCL's **work-group**. "shared memory" → OpenCL's **local memory** (note: OpenCL's "private memory" is CUDA's "local memory" — the opposite pairing, a common trap).
</details>

**Q2.** How many threads are in a warp, and how are threads 64–95 grouped?

<details><summary>Answer</summary>

32 threads per warp. Threads 64–95 form **Warp 2** (Warp 0 = 0–31, Warp 1 = 32–63, Warp 2 = 64–95).
</details>

**Q3.** Full occupancy worked problem: You have a GPU where each SM supports a max of 1536 threads, max 8 blocks/SM, and 48 warps/SM. Your kernel uses a thread block size of 192 threads. Compute: (a) blocks/SM by the thread-slot limit, (b) whether the block-slot limit is a further constraint, (c) total threads actually running, (d) total warps, (e) occupancy percentage.

<details><summary>Answer</summary>

(a) 1536 / 192 = 8 blocks/SM (thread-slot limit)
(b) Block-slot limit is also 8 — exactly matches, not a further constraint.
(c) 8 blocks × 192 threads = 1536 threads
(d) 1536 / 32 = 48 warps
(e) 48/48 = 100% occupancy.
</details>

**Q4.** Name the 4 Special Function Units (SFU) operations listed for the Fermi SM.

<details><summary>Answer</summary>

sin, cosine, reciprocal, and square root.
</details>

**Q5.** Using `Number of Required Warps = Latency × Throughput`, if a hypothetical GPU has 20-cycle instruction latency and 128 ops/cycle throughput, how many warps are needed to fully hide latency?

<details><summary>Answer</summary>

Total operations needed = 20 × 128 = 2,560. Warps = 2,560 / 32 = **80 warps.**
</details>

**Q6.** `someKernel<<<3, 4>>>();` — how many total threads are launched, and what values does `threadIdx.x` take across the whole launch?

<details><summary>Answer</summary>

3 blocks × 4 threads = 12 threads total. `threadIdx.x` = 0,1,2,3 in block 0, then RESETS to 0,1,2,3 in block 1, then 0,1,2,3 again in block 2 — it never goes above 3, since it always resets per block. (`blockIdx.x` is what distinguishes 0,1,2 across the three blocks.)
</details>

**Q7.** Trace the 5-step workload dataflow from host to device and back, in order.

<details><summary>Answer</summary>

1. Data copied from host memory to device memory via PCIe.
2. Host launches the kernel on the device.
3. Kernel executed by multiple threads concurrently.
4. Data within the device accessed by threads through the memory hierarchy.
5. Results moved back to device memory and transferred to the host via PCIe.
</details>

---

## More Self-Test Questions

**Q8.** Using the "Number of Required Warps = Latency × Throughput" rule, a hypothetical GPU has instruction latency of 25 cycles and throughput of 128 ops/cycle. How many warps are needed to fully hide latency?

<details><summary>Answer</summary>

Total operations = 25 × 128 = 3,200. Warps = 3,200 / 32 = **100 warps.**
</details>

**Q9.** Full occupancy problem: an SM supports max 2048 threads, max 16 blocks/SM, and 64 warps/SM. Your kernel uses a 32×8 (256-thread) block. Work through: (a) blocks by thread limit, (b) blocks by block-slot limit, (c) actual blocks scheduled, (d) total threads, (e) total warps, (f) occupancy %.

<details><summary>Answer</summary>

(a) 2048/256 = 8 blocks (thread-slot limit)
(b) block-slot limit is 16 — not binding here since 8 ≤ 16
(c) min(8, 16) = 8 blocks
(d) 8 × 256 = 2048 threads
(e) 2048/32 = 64 warps
(f) 64/64 = 100% occupancy
</details>

**Q10.** In the OpenCL/CUDA terminology table, what does OpenCL call the CUDA concept of "constant memory"? Is this one of the terms that swaps meaning, or does it stay consistent?

<details><summary>Answer</summary>

OpenCL also calls it "constant memory" — this is one of the terms that stays CONSISTENT between the two models (unlike "local"/"private" which swap meaning). Global memory and constant memory use the same name in both frameworks; only local/shared and private/local swap.
</details>

**Q11.** Which thread numbers (give the range) belong to Warp 5?

<details><summary>Answer</summary>

Warp n covers threads `[32n, 32n+31]`. Warp 5: `32×5=160` to `32×5+31=191`. So threads 160 through 191.
</details>

**Q12.** A Fermi SM has a 32,768×32-bit register file. If a kernel uses 40 registers/thread, what's the MAXIMUM number of threads that register file alone could support (ignoring block/thread-slot limits)?

<details><summary>Answer</summary>

32,768 registers ÷ 40 registers/thread = 819.2 → **819 threads** (you can't have a fractional thread, so this rounds down to the nearest whole thread count the register file can actually support).
</details>

**Q13.** True or False: SIMD hardware can natively execute different instructions on different data lanes simultaneously, the same way SIMT threads can take different branch paths.

<details><summary>Answer</summary>

**False for both, actually — but for different reasons worth distinguishing.** Neither SIMD nor SIMT hardware executes genuinely DIFFERENT instructions simultaneously on their lanes/threads. SIMD requires the PROGRAMMER to manually compute both branch paths and blend results with a bitmask. SIMT handles branching automatically via the warp scheduler's active-mask mechanism, but it still SERIALIZES the two paths (executes path A with some threads masked off, then path B with the others masked off) — it does not achieve true simultaneous divergent execution either.
</details>

**Q14.** Why does a GPU dedicate so much less die area to cache compared to a CPU, given that memory access is objectively slower than compute?

<details><summary>Answer</summary>

A CPU has few cores doing complex, unpredictable work, so large caches meaningfully reduce average memory latency for that small number of execution contexts. A GPU has thousands of simple cores; rather than trying to cache for all of them (which would require an impractically large cache), it instead relies on having so many independent warps in flight that the SCHEDULER can always switch to a different ready warp while one is waiting on memory — trading cache size for warp-level parallelism as the latency-hiding strategy.
</details>

---

## Challenge Exercises

**Challenge 1 — Build your own occupancy table.** For an SM with max 1536 threads/SM, max 8 blocks/SM, and 48 warps/SM, compute the occupancy for block sizes 32, 96, 192, 384, and 768. Which sizes hit 100%? Which waste capacity, and on which limit (thread-slot vs block-slot)?

**Challenge 2 — Reverse-engineer the throughput.** A GPU needs exactly 96 warps to fully hide a 24-cycle instruction latency. What is its throughput in ops/cycle?

**Challenge 3 — Draw your own SIMT-core-cluster diagram.** Using the GPU Microarchitecture diagram from theory.md §4 as a template, redraw it for a hypothetical GPU with 6 SIMT Core Clusters, each containing 4 SIMT Cores (instead of the 2 shown in the original), and label the interconnection network and memory partitions accordingly.

**Challenge 4 — Trace a real kernel launch.** Given `dim3 blocks(3,2,1); someKernel<<<blocks, 6>>>();`, write out EVERY value that `blockIdx.x`, `blockIdx.y`, and `threadIdx.x` take across the entire launch (18 total threads). Cross-check against the `dim3` reset-per-block behavior explained in theory.md §11.

---

## 📝 Wrap-up — Reflect

Before moving to Lecture 03, confirm you can do these three things without notes:
1. Fill in the full CUDA↔OpenCL terminology table from memory, including which two terms "swap" meaning.
2. Walk through a 6-step occupancy calculation for any given block size and SM limits.
3. Explain SIMT in one sentence that correctly distinguishes it from SIMD.

---

## ⚡ Rapid-Fire True/False Round

1. OpenCL's "local memory" is the same concept as CUDA's "local memory." <details><summary>A</summary>False — OpenCL "local"=CUDA "shared"; OpenCL "private"=CUDA "local". They swap.</details>
2. A warp always contains exactly 32 threads on NVIDIA GPUs. <details><summary>A</summary>True — a fixed hardware constant.</details>
3. SIMD hardware automatically bundles scalar threads into vectors at runtime. <details><summary>A</summary>False — that's SIMT (GPU). SIMD requires the programmer to manage vector width explicitly.</details>
4. A Fermi SM has 4 Special Function Units. <details><summary>A</summary>True — executing sin, cosine, reciprocal, square root.</details>
5. Occupancy is determined by only ONE hardware limit at a time. <details><summary>A</summary>False — always check thread-slot, block-slot, AND register limits; take the minimum.</details>
6. `Kernel<<<50,1024>>>()` launches fewer total threads than `Kernel<<<100,256>>>()`. <details><summary>A</summary>False — 50×1024=51,200 vs 100×256=25,600. The first launches MORE threads.</details>
7. Kepler requires fewer warps than Fermi to hide the same latency, because its throughput is lower. <details><summary>A</summary>False — Kepler's throughput is HIGHER (192 vs 32 ops/cycle), so it needs MORE warps (120 vs 20) for the same latency.</details>
8. `threadIdx.x` keeps incrementing continuously across multiple blocks in the same launch. <details><summary>A</summary>False — it resets to 0 at the start of every new block.</details>
9. Global memory is visible to the host CPU as well as GPU threads. <details><summary>A</summary>True.</details>
10. A block size that isn't a multiple of 32 never causes any inefficiency. <details><summary>A</summary>False — it wastes warp capacity on padding, as shown in numerical.md §4.</details>

---

## 🗂️ Quick Reference Card

```
CUDA <-> OpenCL:  grid<->NDRange | block<->work-group | thread<->work-item
                  shared mem<->local mem | local mem<->private mem (SWAPPED!)

WARP = 32 threads always.  Warp n = threads [32n, 32n+31].

FERMI SM: 32 cores | 16 LD/ST | 4 SFU | 2 warp schedulers | 32,768x32-bit regs

OCCUPANCY RECIPE:
  1. blocks by thread-slot limit = maxThreads/blockSize
  2. blocks by block-slot limit  = maxBlocks (given)
  3. blocks by register limit    = regFile / (blockSize x regsPerThread)
  4. actual blocks = MIN of all three
  5. total threads = actual_blocks x blockSize
  6. total warps = total_threads / 32
  7. occupancy % = total_warps / maxWarpsPerSM

REQUIRED WARPS = (Latency_cycles x Throughput_ops_per_cycle) / 32

LAUNCH: kernel<<<numBlocks, threadsPerBlock>>>(args);
```

---

## 🏋️ Extended Challenge — Design Your Own GPU

Invent a hypothetical GPU architecture: pick your own values for max threads/SM, max blocks/SM, register file size, and instruction latency/throughput. Then design a kernel (choose block size and registers/thread) that achieves the HIGHEST possible occupancy on YOUR invented hardware, showing the full 7-step occupancy recipe above. Compare with a classmate's invented GPU — whose hardware allows higher theoretical occupancy, and why?

---

## 🧩 Concept-Connection Questions

**C1.** Connect SIMT to the occupancy worked example (16×16 block, 100% occupancy). If a warp within one of those 32-warp-wide blocks hits a divergent `if/else` branch, does that change the OCCUPANCY number itself, or does it change something else?

<details><summary>Answer</summary>

Occupancy (how many warps are RESIDENT on the SM at once) is unaffected by divergence — all 32 warps are still scheduled and occupying their slots. What divergence actually hurts is UTILIZATION/EFFICIENCY within those already-resident warps: some threads sit idle (masked off) during each divergent branch phase, wasting cycles even though the warp itself still counts toward occupancy. Occupancy and efficiency are related but distinct metrics — a kernel can have 100% occupancy and still run inefficiently due to divergence.
</details>

**C2.** Connect the CUDA/OpenCL terminology table to the Fermi SM diagram. Which OpenCL term would you use to refer to "one Fermi SM," and which term would you use for "one CUDA core inside it"?

<details><summary>Answer</summary>

One Fermi SM = a "Compute Unit" in OpenCL terms. One CUDA core inside it = a "Processing Element" in OpenCL terms — directly following the Hardware Model mapping from theory.md §1.
</details>

**C3.** Connect the register-pressure trade-off (§7) to the "more threads vs fewer threads" diagram. If a kernel MUST use 60 registers/thread to do its job correctly (can't be reduced further), what's the practical consequence for how many blocks an SM running that kernel can host, compared to a kernel using only 15 registers/thread?

<details><summary>Answer</summary>

With a fixed register file size (say 32,768 for Fermi), a 60-register/thread kernel can host far FEWER concurrently resident threads/blocks than a 15-register/thread kernel — roughly 4× fewer threads' worth of register space per block. This directly reduces the number of warps available for latency-hiding, which is why performance-tuning often specifically targets reducing register usage per thread when occupancy is found to be the bottleneck.
</details>

---

## 📋 Self-Check Before Moving On

- [ ] I can fill in the full CUDA↔OpenCL terminology table from memory
- [ ] I understand SIMT well enough to explain it differently from SIMD to a beginner
- [ ] I can name the Fermi SM's key components (cores, LD/ST units, SFUs, schedulers) with their counts
- [ ] I can perform a full 6-7 step occupancy calculation for any given block size and SM limits
- [ ] I understand the register-file trade-off between "many threads, few registers" and "few threads, many registers"
- [ ] I can trace the 5-step workload dataflow and the kernel launch chevron syntax

If any box feels shaky, revisit that section of [gpu_lecture02_simt_fermi_theory.md](gpu_lecture02_simt_fermi_theory.md) — Lecture 03 builds directly on the register-limit and dim3 concepts introduced here.

---

## 📝 Sample Exam Question (Multi-Part, Full Marks Breakdown)

> **Question (10 marks):** An SM supports max 1024 threads, max 8 blocks/SM, 32,768 registers, and 32 max warps/SM. A kernel is launched with 20×10 (200-thread) blocks, using 18 registers/thread. (a) [2 marks] Compute blocks allowed by thread-slot limit. (b) [2 marks] Compute blocks allowed by register limit. (c) [2 marks] Determine the actual number of blocks scheduled and state which limit binds. (d) [2 marks] Compute total warps and occupancy %. (e) [2 marks] If registers/thread increased to 30, recompute occupancy and explain the change.

<details><summary>Full worked answer</summary>

**(a)** Thread-slot: `1024/200 = 5.12` → 5 blocks (also check against the 8-block-slot limit: 5 ≤ 8, not binding here)

**(b)** Register limit: registers/block = `200 × 18 = 3600`. `32768/3600 = 9.1` → 9 blocks

**(c)** Actual blocks = `min(5, 8, 9) = 5 blocks`. The THREAD-SLOT limit binds (5 is the smallest).

**(d)** Total threads = `5 × 200 = 1000`. Total warps = `1000/32 = 31.25` → since partial warps still occupy a full warp slot (with padding), this rounds UP to 32 warps allocated per block-count... more precisely: each 200-thread block needs `ceil(200/32)=7` warps (200/32=6.25, rounds up to 7, with the 7th warp only 8 threads "real" + 24 padded). 5 blocks × 7 warps/block = 35 warps needed — but max is 32 warps/SM, so this actually EXCEEDS the warp-slot limit too! Recomputing: `min(5 blocks-by-threads, 8-by-blocks, 9-by-registers, floor(32/7)=4-by-warps) = 4 blocks`. Total threads = `4×200=800`. Total warps = `4×7=28`. Occupancy = `28/32 = 87.5%`.

*(This question deliberately reveals a 4th limiting factor — warps/SM — that a naive 3-limit calculation would miss; always check ALL relevant limits, not just the ones explicitly hinted at.)*

**(e)** At 30 registers/thread: registers/block = `200×30=6000`. Register-limited blocks = `32768/6000=5.46`→5. This doesn't change the binding limit (warps-by-7-per-block is still more restrictive at 4 blocks) — so occupancy stays at 87.5%, UNCHANGED, because the warp-slot limit (not the register limit) was already the true bottleneck both before and after the increase.
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

> *GPU Programming · Lecture 02 · github.com/rpaut03l/TS-02-03*
