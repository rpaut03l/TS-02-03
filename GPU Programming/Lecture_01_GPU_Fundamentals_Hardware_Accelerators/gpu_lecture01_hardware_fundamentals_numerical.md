# 🔢 Lecture 01 — GPU Fundamentals & Hardware Accelerators: NUMERICAL DEEP DIVE

> **Nav:** [← Lecture 01 README](README.md) | [📖 THEORY](gpu_lecture01_hardware_fundamentals_theory.md) | **NUMERICAL** | [🎯 PRACTICE](gpu_lecture01_hardware_fundamentals_practice.md)

---

## 📚 Table of Contents

1. [Amdahl's Law — Every Step of the Arithmetic](#1-amdahls-law-every-step-of-the-arithmetic)
2. [Amdahl's Law — Five Fully Worked Problems](#2-amdahls-law-five-fully-worked-problems)
3. [The Battery Thermal Simulation Table — Deep Dive](#3-the-battery-thermal-simulation-table-deep-dive)
4. [The CNN Inference Table — Deep Dive](#4-the-cnn-inference-table-deep-dive)
5. [Exam-Style Numerical Traps](#5-exam-style-numerical-traps)
6. [Two More Fully Worked Amdahl's Law Problems](#6-two-more-fully-worked-amdahls-law-problems)
7. [Recomputing the Hardware Spec Numbers](#7-recomputing-the-hardware-spec-numbers)
8. [Extra Practice Set — Amdahl's Law Speed Drills](#8-extra-practice-set-amdahls-law-speed-drills)

---

## 1. Amdahl's Law — Every Step of the Arithmetic

Picture a class of students copying a 100-page book by hand. 90 of those pages can be split among as many students as you want — hand out pages 1–90 to 90 different students, and they all copy at the same time. But the last 10 pages are the **cover, title page, and a hand-signed certificate at the end** — only the teacher can write those, one at a time, no matter how many students you have standing around. That "only-the-teacher-can-do-it" chunk is exactly what `(1-P)` represents in Amdahl's Law, and it never goes away, no matter how many students (processors) you add.

**The formula, piece by piece:**
```
              1
  S = -----------------
       (1 - P) + P/N
```

| Symbol | What it means | In our book example |
|---|---|---|
| `P` | Fraction of the work that CAN be split among many workers | 90 pages / 100 pages = 0.9 |
| `1 - P` | Fraction that must be done by ONE worker, no matter what | 10 pages / 100 pages = 0.1 |
| `N` | Number of workers (processors) | however many students you have |
| `S` | Overall speedup vs. one worker doing everything alone | what we're solving for |

**Building the formula from scratch, one piece at a time:**
```
Time for 1 worker to do EVERYTHING = 1.0 (normalize the total time to "1 unit")

Time for the SERIAL part (can't split)   = (1 - P) x 1.0 = (1-P)
Time for the PARALLEL part with N workers = (P x 1.0) / N = P/N

TOTAL time with N workers = (1-P) + P/N

Speedup S = (time with 1 worker) / (time with N workers)
          =        1.0           / [ (1-P) + P/N ]
          =            1
              -----------------
               (1 - P) + P/N
```

**Why the maximum speedup formula drops the `N`:**
```
As N gets bigger and bigger (more and more students helping)...
   P/N gets SMALLER and SMALLER (each student's share shrinks toward zero)
   ...but (1-P) NEVER shrinks, because it was never split up in the first place.

So as N -> infinity:
   S = 1 / ( (1-P) + P/N )   ---->   S = 1 / ( (1-P) + 0 )   =   1 / (1-P)
```

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 2. Amdahl's Law — Five Fully Worked Problems

### Problem 1: P = 0.5, N = 4
```
S = 1 / ( (1 - 0.5) + 0.5/4 )
  = 1 / ( 0.5 + 0.125 )
  = 1 / 0.625
  = 1.6x
```
**What this feels like:** if only HALF your program is parallelizable, even 4 workers barely gets you past 1.6× — the other half is dragging everything down like an anchor.

### Problem 2: P = 0.9, N = 4
```
S = 1 / ( (1 - 0.9) + 0.9/4 )
  = 1 / ( 0.1 + 0.225 )
  = 1 / 0.325
  = 3.08x
```

### Problem 3: P = 0.9, N = 100 (same P as Problem 2, way more workers)
```
S = 1 / ( (1 - 0.9) + 0.9/100 )
  = 1 / ( 0.1 + 0.009 )
  = 1 / 0.109
  = 9.17x
```
**Compare Problem 2 vs 3:** going from 4 workers to 100 workers (25× more workers!) only improved speedup from 3.08× to 9.17× (about 3× better) — because that stubborn `0.1` serial chunk is now DOMINATING the denominator. This is the whole point of the law: throwing more hardware at a partially-serial problem has fast-diminishing returns.

### Problem 4: What's the absolute ceiling for P = 0.9, no matter how many workers?
```
S_max = 1 / (1 - 0.9) = 1 / 0.1 = 10x
```
Notice Problem 3 (100 workers!) only reached 9.17× — already very close to the 10× ceiling. Adding EVEN MORE workers beyond 100 would only inch you a little closer to 10×, never past it.

### Problem 5: Reverse problem — what P do you need for a 50x speedup with N=1000?
```
Start with:  S = 1 / ( (1-P) + P/N )
Plug in S=50, N=1000, solve for P:

  50 = 1 / ( (1-P) + P/1000 )
  (1-P) + P/1000 = 1/50 = 0.02
  1 - P + 0.001P = 0.02
  1 - 0.999P = 0.02
  0.999P = 0.98
  P = 0.98 / 0.999
  P ≈ 0.981
```
**Reading this:** to get even a modest 50× speedup with 1000 workers, you need your program to be **98.1% parallelizable** — leaving less than 2% serial. This is why real HPC/GPU teams obsess over shrinking that serial fraction; it's the single biggest lever in the whole equation.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 3. The Battery Thermal Simulation Table — Deep Dive

Recall the table from [theory.md §7](gpu_lecture01_hardware_fundamentals_theory.md#7-case-study-battery-thermal-simulation):

| Grid Size | CPU (ms) | GPU (ms) | GPU Speedup |
|---|---|---|---|
| 256×256 | 51.55 | 0.48 | 107.76× |
| 512×512 | 236.54 | 1.48 | 160.28× |
| 1024×1024 | 906.35 | 5.34 | 169.79× |
| 2048×2048 | 3593.75 | 17.75 | 202.44× |

**Verifying the speedup column ourselves (never trust a table blindly — recompute it):**
```
256x256:    51.55 / 0.48   = 107.395...  (table shows 107.76 -- close, rounding in original)
512x512:   236.54 / 1.48   = 159.824...  (table shows 160.28)
1024x1024: 906.35 / 5.34   = 169.720...  (table shows 169.79)
2048x2048: 3593.75 / 17.75 = 202.465...  (table shows 202.44)
```
Small differences are normal — they come from more decimal precision in the original raw timing data than what's printed in the table.

**Now look at how the CELL COUNT scales, and connect it to why speedup keeps climbing:**
```
256x256   =    65,536 cells    (this is our baseline "1x" cell count)
512x512   =   262,144 cells    = 4x  the cells of 256x256
1024x1024 = 1,048,576 cells    = 16x the cells of 256x256
2048x2048 = 4,194,304 cells    = 64x the cells of 256x256
```
Every time you double the grid's side length, you QUADRUPLE (2²) the total number of cells, because it's a 2D grid. This is why the GPU's advantage keeps growing: each doubling of the side length gives the GPU 4× more independent parallel work to fill its threads with, while the CPU still has to trudge through every single cell one at a time — the CPU's runtime roughly quadruples too (51.55 → 236.54 is about 4.6×, close to the expected 4× plus some overhead), but the GPU's runtime barely grows at all in comparison, because it just spreads the extra cells across still-mostly-idle thread capacity.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 4. The CNN Inference Table — Deep Dive

Recall from [theory.md §8](gpu_lecture01_hardware_fundamentals_theory.md#8-case-study-cnn-inference-across-hardware):

| Model | FPS: CPU | FPS: GPU | GPU Speedup |
|---|---|---|---|
| 3Conv-CNN | 13.39 | 15.07 | ? |
| MLP | 9.41 | 14.38 | ? |
| 2Dense-CNN | 10.42 | 14.41 | ? |
| MobileNetV2 | 8.01 | 11.76 | ? |
| ResNet50 | 7.26 | 12.62 | ? |

**Compute every speedup ratio, step by step:**
```
3Conv-CNN:   15.07 / 13.39 = 1.125x   (only a 12.5% improvement)
MLP:         14.38 / 9.41  = 1.528x
2Dense-CNN:  14.41 / 10.42 = 1.383x
MobileNetV2: 11.76 / 8.01  = 1.468x
ResNet50:    12.62 / 7.26  = 1.738x
```

**Rank them from smallest GPU benefit to largest:**
```
3Conv-CNN (1.13x)  <  2Dense-CNN (1.38x)  <  MobileNetV2 (1.47x)  <  MLP (1.53x)  <  ResNet50 (1.74x)
```

**Why is 3Conv-CNN's improvement so small compared to ResNet50's?** Think of it like the battery grid example again: 3Conv-CNN has only 3 convolutional layers — a small, shallow "grid" of work. ResNet50 has 53 convolutional layers plus 53 batch-normalization layers — a MUCH bigger pile of independent per-pixel, per-channel math for the GPU to chew through in parallel. A shallow model finishes so fast on EITHER chip that fixed overheads (loading the model, moving data, kernel launch setup) eat up a bigger fraction of the total time on the GPU run — echoing the exact same Amdahl's-Law idea from §1: a fixed "serial-ish" overhead cost matters more when the actual parallel workload is small.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 5. Exam-Style Numerical Traps

1. **Forgetting `S_max` needs `P` only, not `N`.** If a question gives you both `P` and `N` but asks for the "maximum possible" or "theoretical ceiling" speedup, `N` is a distractor — use `S_max = 1/(1-P)` and ignore `N` entirely.
2. **Percent vs fraction confusion.** If a question says "80% parallelizable," that's `P = 0.8`, not `P = 80`. Plugging in `80` instead of `0.8` is the single most common arithmetic error on this topic.
3. **Doubling a 2D grid ≠ doubling the work.** As shown in §3, doubling the SIDE LENGTH of a square 2D grid quadruples the total CELL COUNT (2² = 4). This trips people up when reasoning about "how much more parallel work" a bigger problem actually represents.
4. **Speedup ratios are not additive.** If model A gets 1.5× and model B gets 1.5×, that does NOT mean running them together gets 3×. Always recompute ratios from the raw times, never combine ratios directly.

---

## 6. Two More Fully Worked Amdahl's Law Problems

### Problem 6: The "diminishing helper" scenario — P=0.75, comparing N=2 vs N=200
```
N=2:
  S = 1 / ( (1-0.75) + 0.75/2 )
    = 1 / ( 0.25 + 0.375 )
    = 1 / 0.625
    = 1.6x

N=200:
  S = 1 / ( (1-0.75) + 0.75/200 )
    = 1 / ( 0.25 + 0.00375 )
    = 1 / 0.25375
    = 3.94x

S_max = 1/(1-0.75) = 1/0.25 = 4.0x
```
**What jumps out:** going from N=2 all the way to N=200 (100× more processors!) only takes you from 1.6× to 3.94× — you're stuck fighting to close the last small gap to the 4.0× ceiling, and no amount of additional processors will ever cross it.

### Problem 7: Combining two stages with different P values
Suppose a pipeline has two sequential stages: Stage A takes 40% of total time and is 95% parallelizable; Stage B takes 60% of total time and is 70% parallelizable. If both stages get N=16 processors, what's the speedup of the WHOLE pipeline?
```
STEP 1: Compute each stage's own speedup separately.
  Stage A: S_A = 1/((1-0.95)+0.95/16) = 1/(0.05+0.059375) = 1/0.109375 = 9.14x
  Stage B: S_B = 1/((1-0.70)+0.70/16) = 1/(0.30+0.04375)  = 1/0.34375  = 2.91x

STEP 2: Compute each stage's NEW time (as a fraction of original total time),
        by dividing its original time-share by its own speedup.
  New Stage A time = 0.40 / 9.14 = 0.0438
  New Stage B time = 0.60 / 2.91 = 0.2062

STEP 3: Total new time = 0.0438 + 0.2062 = 0.2500

STEP 4: Overall speedup = original total time / new total time
       = 1.0 / 0.2500
       = 4.0x
```
**Why this matters:** real programs are rarely ONE uniform serial/parallel split — they're pipelines of stages with DIFFERENT parallelizability. The overall speedup ends up dominated by whichever stage has the WORST `P` and the biggest time-share — here, Stage B (60% of the time, only 70% parallelizable) drags the combined result down to 4.0×, even though Stage A alone would have hit 9.14×.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 7. Recomputing the Hardware Spec Numbers

Revisit the "GPU H/W Example" real Fermi specs from [Lecture 03's theory](../Lecture_03_Thread_Hierarchy_Memory_HelloWorld/gpu_lecture03_threads_memory_theory.md#4-real-fermi-hardware-specs--sw-abstraction-numbers): 16 SMs, 512 CUDA cores total, 32 cores/SM.

**Sanity-check the division ourselves:**
```
512 total cores / 16 SMs = 32 cores/SM   <- matches the stated "32/SM" figure exactly
```

**If each core can (roughly) issue 1 floating-point operation per clock cycle, and the GPU runs at, say, 1.15 GHz (a realistic Fermi clock speed), what's the theoretical peak FLOPS (floating-point operations per second)?**
```
Total cores = 512
Clock speed = 1.15 x 10^9 cycles/second

Peak FLOPS (single operation per core per cycle, simplified) =
  512 cores x 1.15x10^9 cycles/sec = 588.8 x 10^9 = 588.8 GFLOPS
```
**Why "roughly" and "simplified"?** Real GPUs typically execute a fused-multiply-add (FMA) per cycle per core, which counts as TWO floating-point operations (one multiply, one add) — so real peak FLOPS figures for Fermi-class GPUs are often quoted around double this simplified estimate. This exercise is about practicing the ARITHMETIC METHOD (cores × clock = raw operation rate), not memorizing an exact marketing FLOPS number.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 8. Extra Practice Set — Amdahl's Law Speed Drills

Try these without looking at the answer first, then check:

| # | P | N | S = ? |
|---|---|---|---|
| a | 0.3 | 4 | ? |
| b | 0.99 | 10 | ? |
| c | 0.5 | 1000 | ? |
| d | 1.0 | 8 | ? |
| e | 0.0 | 8 | ? |

<details><summary>Answers</summary>

```
a) S = 1/((1-0.3)+0.3/4) = 1/(0.7+0.075) = 1/0.775 = 1.29x
b) S = 1/((1-0.99)+0.99/10) = 1/(0.01+0.099) = 1/0.109 = 9.17x
c) S = 1/((1-0.5)+0.5/1000) = 1/(0.5+0.0005) = 1/0.5005 = 1.998x (barely above 2x --
   at N=1000, we're deep into diminishing returns already since P is only 0.5)
d) S = 1/((1-1.0)+1.0/8) = 1/(0+0.125) = 1/0.125 = 8x  -- a FULLY parallel program gets
   the FULL linear speedup of N (this is the special case P=1)
e) S = 1/((1-0.0)+0.0/8) = 1/(1+0) = 1/1 = 1x -- a FULLY serial program gets ZERO
   benefit from any number of processors (P=0 special case)
```
Cases (d) and (e) are the two "boundary" special cases worth memorizing: `P=1` gives perfect linear speedup `S=N`; `P=0` gives no speedup at all, `S=1`, no matter how large `N` is.
</details>

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

## 9. One Fully Worked "Combine Everything" Problem

A team is deciding whether to port a simulation to GPU. The simulation is 85% parallelizable. On a 512×512 grid, their CPU baseline takes 300ms. Assume GPU speedup scales similarly to the battery thermal case study (roughly 160× at this grid size).

```
STEP 1 -- Amdahl's ceiling for this workload:
   S_max = 1/(1-0.85) = 1/0.15 = 6.67x

STEP 2 -- The "observed" 160x GPU speedup from the case study is for the FULLY
   parallel portion running on GPU hardware -- it does NOT account for the 15%
   serial portion that Amdahl's Law says can NEVER be sped up by the GPU at all.

STEP 3 -- Apply Amdahl's Law properly: even with an enormous per-parallel-region
   speedup, the WHOLE-PROGRAM speedup is still capped at S_max=6.67x, because
   15% of the work simply cannot benefit from ANY amount of GPU acceleration.

STEP 4 -- Expected wall-clock time on GPU (using the realistic Amdahl-capped speedup,
   not the raw 160x case-study number):
   New time = 300ms / 6.67 = 45ms  (NOT 300/160=1.875ms, which would be wrong --
   that number ignores the serial bottleneck entirely)
```
**The takeaway for the team:** the case study's headline "160× speedup" number describes the PARALLEL PORTION's kernel-level performance, not the whole application's realistic speedup — a common and costly mistake when estimating GPU-porting ROI. Always ask "what fraction of my ACTUAL end-to-end workflow is this speedup measuring?"

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-numerical-deep-dive)

---

> *GPU Programming · Lecture 01 · github.com/rpaut03l/TS-02-03*
