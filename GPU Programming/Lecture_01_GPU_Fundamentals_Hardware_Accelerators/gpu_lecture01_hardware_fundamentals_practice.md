# 🎯 Lecture 01 — GPU Fundamentals & Hardware Accelerators: PRACTICE

> **Nav:** [← Lecture 01 README](README.md) | [📖 THEORY](gpu_lecture01_hardware_fundamentals_theory.md) | **PRACTICE**

---

## Self-Test Questions

**Q1.** Name all 5 hardware platforms covered in this lecture, and give one real-world example device for each.

<details><summary>Answer</summary>

CPU (Arm Cortex-M / Raspberry Pi), GPU (NVIDIA Jetson), TPU (Google Edge TPU), NPU (on-device phone AI chips), FPGA (Xilinx Zynq).
</details>

**Q2.** A program is 80% parallelizable. What is the maximum theoretical speedup, however many processors you use?

<details><summary>Answer</summary>

`S_max = 1/(1-P) = 1/(1-0.8) = 1/0.2 = 5x`. No matter how many cores, this program can never run faster than 5× its original time.
</details>

**Q3.** A program is 95% parallelizable and you have 20 processors. What's the actual speedup (not the theoretical max)?

<details><summary>Answer</summary>

`S = 1 / ((1-P) + P/N) = 1 / (0.05 + 0.95/20) = 1 / (0.05 + 0.0475) = 1 / 0.0975 ≈ 10.26x`

Notice this is well below the theoretical max of `1/0.05 = 20x` — you'd need far more than 20 processors to approach that ceiling.
</details>

**Q4.** In the battery thermal simulation case study, why did the GPU speedup INCREASE as grid size grew from 256×256 to 2048×2048, instead of staying constant?

<details><summary>Answer</summary>

Bigger grids mean more independent cells to update in parallel — more work for the GPU's thousands of threads to chew on simultaneously, while fixed overheads (kernel launch, memory transfer setup) become a smaller fraction of the total runtime. This is Amdahl's Law in action: effectively, `P` (the parallel fraction) stays high while the useful parallel work grows.
</details>

**Q5.** Map these three CUDA software concepts to their hardware equivalents: Thread, Thread Block, Kernel Grid.

<details><summary>Answer</summary>

Thread → Core. Thread Block → Streaming Multiprocessor (SM). Kernel Grid → Complete GPU Unit.
</details>

**Q6.** True or False: GPUs are always faster than CPUs, for any program.

<details><summary>Answer</summary>

**False.** CPUs can be over 10× faster than GPUs for sequential, latency-sensitive code. GPUs win specifically on parallel, throughput-oriented workloads.
</details>

---

## Mini Exercise

Using the CNN inference case study table in [theory.md §8](gpu_lecture01_hardware_fundamentals_theory.md#8-case-study-cnn-inference-across-hardware), compute the GPU speedup (GPU FPS ÷ CPU FPS) for the MobileNetV2 and ResNet50 rows. Which model benefits more from the GPU, and does that match the "bigger/more parallel models benefit more" pattern from the theory section?

<details><summary>Answer</summary>

MobileNetV2: 11.76 / 8.01 ≈ 1.47×. ResNet50: 12.62 / 7.26 ≈ 1.74×. ResNet50 (53 conv layers, deeper) benefits more than MobileNetV2 (35 conv layers but optimized for efficiency) — consistent with the pattern that heavier, more parallel-friendly models see a larger relative GPU speedup, though the exact ranking also depends on each architecture's specific operator mix (e.g. MobileNetV2's depthwise convolutions are inherently less parallel-friendly than ResNet50's standard convolutions).
</details>

---

## More Self-Test Questions

**Q7.** A program is 60% parallelizable. Compute the speedup with N=2, N=8, and N=32 processors, then the theoretical maximum. What do you notice about the gaps between consecutive results?

<details><summary>Answer</summary>

```
N=2:  S = 1/((1-0.6)+0.6/2)  = 1/(0.4+0.30) = 1/0.70 = 1.43x
N=8:  S = 1/((1-0.6)+0.6/8)  = 1/(0.4+0.075)= 1/0.475= 2.11x
N=32: S = 1/((1-0.6)+0.6/32) = 1/(0.4+0.01875)=1/0.41875=2.39x
S_max = 1/(1-0.6) = 1/0.4 = 2.5x
```
Gap from N=2→8 (quadrupling N): +0.68x. Gap from N=8→32 (quadrupling N again): only +0.28x. Each further quadrupling of N buys a SMALLER improvement — classic diminishing returns as you approach the 2.5x ceiling.
</details>

**Q8.** Explain, in the language of Amdahl's Law, why a GPU with 10,000 cores doesn't make a web browser start 10,000× faster.

<details><summary>Answer</summary>

Most of what a web browser does — parsing HTML, running JavaScript event handlers, waiting on network requests, updating UI state — is inherently sequential (`P` is very low for these tasks). Only specific, narrow parts (e.g. rendering many independent pixels, decoding video frames) are highly parallel. Since `S_max = 1/(1-P)`, a low `P` caps the achievable speedup at a small number regardless of how many GPU cores exist — the GPU literally cannot help with the sequential 95%+ of the workload.
</details>

**Q9.** Rank the 5 hardware platforms (CPU, GPU, TPU, NPU, FPGA) from MOST flexible/general-purpose to LEAST, and briefly justify the ranking.

<details><summary>Answer</summary>

CPU (most flexible — runs any program) → GPU (flexible within parallel-math workloads, still fully programmable) → FPGA (reconfigurable hardware logic, requires hardware-level design) → NPU/TPU (fixed-function-ish, optimized specifically for tensor/neural-net operations, least flexible but most efficient for that narrow task). NPU and TPU are close in flexibility/specialization level; TPU tends to be even more rigidly optimized for large-scale tensor ops in data centers, while NPUs are tuned for low-power on-device inference.
</details>

**Q10.** In the Comparative Evaluation table (gpu_lecture01_hardware_fundamentals_theory.md §3), which platform category has the HIGHEST development cost and LONGEST time-to-market, and why does that make sense given its performance profile?

<details><summary>Answer</summary>

ASIC-based devices. They trade flexibility for extremely low power and highest performance — but because the chip's logic is physically fixed at manufacture time (not reconfigurable like an FPGA, not software-programmable like a CPU/GPU), any design mistake requires a full new manufacturing run. This inherent rigidity is exactly why ASICs are reserved for extremely high-volume, well-understood, stable workloads (e.g. Google's TPU for their own data centers) rather than general use.
</details>

**Q11.** A grid-based simulation has its side length doubled 3 times in a row (e.g. 128→256→512→1024). By what total factor does the total cell count grow?

<details><summary>Answer</summary>

Each doubling of a 2D grid's side length quadruples (2²) the cell count. Three doublings in a row: `4 × 4 × 4 = 64×` growth in total cells. (This directly extends the pattern shown in [numerical.md §3](gpu_lecture01_hardware_fundamentals_numerical.md#3-the-battery-thermal-simulation-table-deep-dive).)
</details>

**Q12.** True or False: Amdahl's Law says adding more processors always makes a parallel program run faster, just by smaller and smaller amounts each time.

<details><summary>Answer</summary>

**Mostly true, with a nuance worth stating precisely.** Mathematically, `S = 1/((1-P)+P/N)` is a strictly increasing function of `N` (more processors never makes it WORSE in the idealized formula) — but the gains shrink toward zero as `N→∞`, approaching the ceiling `S_max=1/(1-P)`. In real systems, adding processors can eventually make things WORSE due to overheads Amdahl's Law doesn't model (coordination/communication cost, memory bandwidth contention) — the idealized law is a best-case upper bound, not a full real-world performance model.
</details>

---

## Challenge Exercises

**Challenge 1 — Build your own comparison table.** Pick any 3 real AI/ML applications you're familiar with (e.g. a recommendation system, a chatbot, a self-driving perception stack). For each, estimate roughly what fraction of the workload is parallelizable (`P`), and compute the theoretical max speedup `S_max = 1/(1-P)` for each. Which application benefits most from GPU acceleration, and why does that match your intuition about the underlying computation?

**Challenge 2 — The overhead-adjusted model.** Amdahl's Law assumes zero cost for coordinating N processors. Suppose each processor addition costs a FIXED overhead of `0.01` time units (representing synchronization cost), so the real formula becomes `S = 1 / ((1-P) + P/N + 0.01*N)`. For `P=0.9`, compute `S` at `N=10, 50, 100, 200` and find (by trial) the value of `N` that MAXIMIZES `S`. What does this tell you about "more cores is always better"?

**Challenge 3 — Recreate the case-study math.** Using only the raw CPU/GPU millisecond numbers from the battery thermal simulation table (gpu_lecture01_hardware_fundamentals_theory.md §7), reconstruct the speedup column yourself with a calculator or spreadsheet, without looking at the printed speedup numbers first. Compare your results to the table — are they identical, or do you see the same small rounding differences noted in numerical.md §3?

---

## 📝 Wrap-up — Reflect

Before moving to Lecture 02, make sure you can answer these three questions from memory, without looking back at the theory file:
1. What's the single-sentence difference between "low-latency, low-throughput" and "high-latency, high-throughput" design philosophies?
2. Given a parallel fraction P, can you write down the maximum-speedup formula without hesitating?
3. Can you map "Thread → Core", "Thread Block → SM", "Kernel Grid → Complete GPU" without needing to reread the diagram?

If any of these feel shaky, revisit [gpu_lecture01_hardware_fundamentals_theory.md](gpu_lecture01_hardware_fundamentals_theory.md) before continuing — Lecture 02 builds directly on this hardware-mapping intuition.

---

## ⚡ Rapid-Fire True/False Round

Answer each in your head first, THEN expand to check — don't peek early.

1. GPUs are always the better choice for any AI workload. <details><summary>A</summary>False — depends on parallelizability.</details>
2. Amdahl's Law's `S_max` formula requires knowing `N`. <details><summary>A</summary>False — `S_max=1/(1-P)` only needs `P`.</details>
3. FPGAs are more flexible than CPUs. <details><summary>A</summary>False — CPUs are the MOST flexible in the spectrum.</details>
4. ASICs have the lowest development cost among the 4 device categories. <details><summary>A</summary>False — highest development cost, longest time-to-market.</details>
5. A thread block maps to a Streaming Multiprocessor. <details><summary>A</summary>True.</details>
6. Doubling a 2D grid's side length doubles its total cell count. <details><summary>A</summary>False — it QUADRUPLES it (2²).</details>
7. CPUs generally have more cache per core than GPUs. <details><summary>A</summary>True — CPUs prioritize large caches; GPUs rely on many threads instead.</details>
8. In the CNN case study, every single model showed a GPU speedup greater than 1.5×. <details><summary>A</summary>False — 3Conv-CNN showed only ~1.13×.</details>
9. `P=1` (fully parallel) with any `N>1` always gives `S=N` exactly. <details><summary>A</summary>True — the special case where Amdahl's Law reduces to perfectly linear speedup.</details>
10. NPUs are typically found in large data-center racks rather than phones. <details><summary>A</summary>False — NPUs are associated with low-power, on-device inference (phones), while TPUs are more the data-center tensor accelerator.</details>

---

## 🗂️ Quick Reference Card

```
5 PLATFORMS:  CPU | GPU | TPU | NPU | FPGA  (+ASIC as extreme specialization)
SPECTRUM:     CPU -> GPU -> FPGA -> ASIC   (flexible -> specialized/efficient)

AMDAHL'S LAW:      S = 1 / ((1-P) + P/N)
MAX SPEEDUP:       S_max = 1/(1-P)          as N -> infinity
SPECIAL CASES:     P=1 -> S=N (perfect linear).  P=0 -> S=1 (no benefit ever).

CPU: low-latency, low-throughput  | few complex cores | big caches
GPU: high-latency, high-throughput| many simple cores  | tiny cache, needs many threads

MAPPING:  Thread->Core | Thread Block->SM | Kernel Grid->Complete GPU

2D GRID SCALING: doubling side length -> 4x (2^2) the total cells
```

---

## 🏋️ Extended Challenge — Bring Your Own Data

Find (or estimate) real CPU vs GPU benchmark numbers for a task you personally care about — a video encode, a physics simulation, a game you play, an ML model you've trained. Compute the actual speedup ratio, and if you can estimate `P` for that workload, compute how close the observed speedup is to the Amdahl's-Law theoretical maximum. Bring your findings to the next query session for discussion.

---

## 🧩 Concept-Connection Questions

**C1.** Connect Amdahl's Law to the hardware accelerator spectrum (CPU→GPU→FPGA→ASIC): why might a team choose an FPGA over a GPU for a workload with a LOW parallel fraction `P`, even though FPGAs are usually pitched as "more specialized/efficient"?

<details><summary>Answer</summary>

If `P` is low, GPU's massive core count barely helps (Amdahl's ceiling is low regardless), so you're not "wasting" much potential parallel speedup by NOT using a GPU. An FPGA's advantage in this case would come from custom-designing hardware logic that minimizes the fixed SERIAL overhead itself (the `1-P` portion) — a workload-specific pipeline can sometimes shrink or restructure the serial bottleneck in ways a general-purpose GPU cannot, even if the FPGA has less raw parallel throughput than the GPU.
</details>

**C2.** Connect the "Program Execution Illustration" mapping (Thread→Core, Block→SM, Grid→GPU) to the CNN inference case study. Why would a model with MORE convolutional layers (like ResNet50) naturally launch MORE thread blocks than a shallower model (like 3Conv-CNN)?

<details><summary>Answer</summary>

Each convolutional layer's computation (sliding a filter across the image) is itself parallelized across many threads/blocks. A deeper model has more layers, each contributing its own wave of thread-block launches sequentially through the network — more layers means more total kernel launches and more aggregate parallel work spread across the GPU's SMs over the course of one inference pass, which is exactly why deeper models tend to show a LARGER relative GPU speedup (more opportunity to fill the GPU's parallel capacity).
</details>

**C3.** The battery thermal simulation uses a "Shared-memory GPU" variant alongside the plain GPU version. Without having covered shared memory in depth yet (that's a later lecture), what would you PREDICT about why a shared-memory version might be even faster at LARGE grid sizes, based purely on the "closer to the compute core = faster" idea from the memory hierarchy diagrams?

<details><summary>Answer</summary>

Shared memory sits much closer to the compute cores (on-chip, per-SM) than global/device memory — so if neighboring grid cells need to read each other's temperature values repeatedly (which the heat-diffusion formula requires), caching those values in shared memory instead of re-reading them from slower global memory every time should reduce memory traffic and speed things up, especially as the grid (and thus the amount of neighbor-reading) grows.
</details>

---

## 📋 Self-Check Before Moving On

Rate your confidence (1-5) on each, honestly, before starting Lecture 02:

- [ ] I can name all 5 hardware platforms and one example device for each
- [ ] I can derive Amdahl's Law from scratch, not just recite the formula
- [ ] I can compute both `S` (given N) and `S_max` (theoretical ceiling) for any P
- [ ] I understand why GPU speedup grows with problem size (battery sim example)
- [ ] I can map Thread→Core, Block→SM, Grid→GPU without hesitation
- [ ] I know the flexibility-vs-efficiency tradeoff across CPU→GPU→FPGA→ASIC

If any box is a "2 or lower," reread that specific section of [gpu_lecture01_hardware_fundamentals_theory.md](gpu_lecture01_hardware_fundamentals_theory.md) before continuing — Lecture 02 assumes this foundation is solid.

---

## 📝 Sample Exam Question (Multi-Part, Full Marks Breakdown)

> **Question (10 marks):** A rendering pipeline is 92% parallelizable. (a) [2 marks] State the Amdahl's Law formula and identify each symbol. (b) [3 marks] Compute the speedup with N=16 processors. (c) [2 marks] Compute the theoretical maximum speedup as N→∞. (d) [3 marks] The team currently has N=16 and is considering doubling to N=32. Compute the new speedup and comment on whether doubling the hardware was "worth it" in relative terms.

<details><summary>Full worked answer</summary>

**(a)** `S = 1/((1-P)+P/N)` where `S`=speedup, `P`=parallel fraction, `N`=number of processors, `(1-P)`=serial fraction.

**(b)** `S = 1/((1-0.92)+0.92/16) = 1/(0.08+0.0575) = 1/0.1375 = 7.27x`

**(c)** `S_max = 1/(1-0.92) = 1/0.08 = 12.5x`

**(d)** At N=32: `S = 1/((1-0.92)+0.92/32) = 1/(0.08+0.02875) = 1/0.10875 = 9.20x`

Going from N=16 to N=32 (doubling hardware) improved speedup from 7.27× to 9.20× — a gain of about 1.93×, NOT the 2× you might naively expect from doubling processors. This is diminishing returns in action: at N=16 we were already at `7.27/12.5 = 58%` of the theoretical ceiling; doubling hardware moved us to `9.20/12.5 = 73.6%` of the ceiling — real, but shrinking, marginal value. Whether it's "worth it" depends on cost: a team paying for 2× the hardware to get only 1.27× more actual speedup should weigh that against the marginal cost of the additional processors.
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

> *GPU Programming · Lecture 01 · github.com/rpaut03l/TS-02-03*
