# 🔢 Lecture 04 — Vector/Matrix Kernels & Indexing: NUMERICAL DEEP DIVE

> **Nav:** [← Lecture 04 README](README.md) | [📖 THEORY](gpu_lecture04_indexing_kernels_theory.md) | **NUMERICAL** | [🎯 PRACTICE](gpu_lecture04_indexing_kernels_practice.md)

---

## 📚 Table of Contents

1. [The "Red Element" Problem, Rebuilt From Zero](#1-the-red-element-problem-rebuilt-from-zero)
2. [Three More Combined-Index Problems](#2-three-more-combined-index-problems)
3. [Sizing the Real `main()` — Memory and Launch Numbers](#3-sizing-the-real-main-memory-and-launch-numbers)
4. [Ceiling Division for Arbitrary Sizes — Every Case](#4-ceiling-division-for-arbitrary-sizes-every-case)
5. [Matrix Multiplication — Counting the Independent Work](#5-matrix-multiplication-counting-the-independent-work)
6. [Exam-Style Numerical Traps](#6-exam-style-numerical-traps)
7. [Three More Combined-Index Speed Drills](#7-three-more-combined-index-speed-drills)
8. [Full Worked Memory-Sizing Problem — A Different N](#8-full-worked-memory-sizing-problem-a-different-n)
9. [Extra Practice Set — Ceiling Division Speed Drills](#9-extra-practice-set-ceiling-division-speed-drills)

---

## 1. The "Red Element" Problem, Rebuilt From Zero

Picture a long train made of identical train cars, each with exactly 8 numbered seats. You're told a passenger's overall seat number on the WHOLE train (counting from the very front), and you need to say: which car, and which seat inside that car?

**The setup:** `M = 8` seats per car. Find the car and seat for **overall position 21.**

```
STEP 1 -- How many FULL cars come before this passenger?
   21 divided by 8 = 2 remainder 5
       (because 8 x 2 = 16, and 21 - 16 = 5)

   This means: 2 FULL cars (car 0 and car 1) are completely filled
   before we reach this passenger's car.
   So this passenger is in CAR NUMBER 2  ->  blockIdx.x = 2

STEP 2 -- Which seat WITHIN that car?
   The remainder from Step 1 is 5.
   So this passenger sits in SEAT NUMBER 5 (0-indexed) inside car 2.
   ->  threadIdx.x = 5

STEP 3 -- Verify with the formula from theory.md:
   index = threadIdx.x + blockIdx.x * M
         =      5       +      2      * 8
         =      5       +      16
         =      21                       <- matches! Confirmed correct.
```

**The pattern, generalized:** `blockIdx.x = index DIV M`, `threadIdx.x = index MOD M`. This is the EXACT same divide-and-remainder recipe from [Lecture 03's numerical deep dive](../Lecture_03_Thread_Hierarchy_Memory_HelloWorld/gpu_lecture03_threads_memory_numerical.md#1-the-thread-343-problem-rebuilt-from-zero) — just in 1D instead of 2D.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 2. Three More Combined-Index Problems

**Setup:** `M = 256` threads/block (a realistic real-world block size).

### Find the block and thread for array index 1000
```
1000 div 256 = 3 remainder 232
    (256 x 3 = 768, 1000 - 768 = 232)
blockIdx.x = 3,  threadIdx.x = 232
Verify: 232 + 3*256 = 232 + 768 = 1000  correct.
```

### Find the block and thread for array index 4,194,303 (the LAST index in a 2048×2048 = 4,194,304-element array, 0-indexed)
```
4194303 div 256 = 16384 remainder 255
    (256 x 16384 = 4194304... wait, that's ONE too many -- let's redo carefully)
    256 x 16383 = 4194048
    4194303 - 4194048 = 255
blockIdx.x = 16383,  threadIdx.x = 255
Verify: 255 + 16383*256 = 255 + 4194048 = 4194303   correct.
```
**Sanity check:** with `N = 2048*2048 = 4,194,304` elements and `256` threads/block, the total blocks needed = `4194304 / 256 = 16384` blocks exactly (0 through 16383) — and our last-index calculation landed on `blockIdx.x = 16383`, the very last block. That matches perfectly, exactly like the "last tenant" sanity check from Lecture 03.

### Reverse problem: given `blockIdx.x = 50`, `threadIdx.x = 100`, `M = 256`, find the array index
```
index = threadIdx.x + blockIdx.x * M
      = 100 + 50 * 256
      = 100 + 12800
      = 12900
```

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 3. Sizing the Real `main()` — Memory and Launch Numbers

Recall from [theory.md §3](gpu_lecture04_indexing_kernels_theory.md#3-the-combined-vector-add-kernel--full-main):
```cpp
#define N (2048*2048)
#define THREADS_PER_BLOCK 512
```

**Every number this program actually uses, computed step by step:**
```
STEP 1 -- How many total elements?
   N = 2048 x 2048 = 4,194,304 elements

STEP 2 -- How many bytes does ONE array take up? (assuming 4-byte ints)
   size = N x sizeof(int) = 4,194,304 x 4 bytes = 16,777,216 bytes
        = 16,777,216 / 1024 = 16,384 KB
        = 16,384 / 1024 = 16 MB

STEP 3 -- We allocate THREE such arrays on the device (d_a, d_b, d_c). Total device memory used:
   3 x 16 MB = 48 MB

STEP 4 -- How many blocks does the kernel launch use?
   blocks = N / THREADS_PER_BLOCK = 4,194,304 / 512 = 8,192 blocks

STEP 5 -- Sanity check: total threads launched should equal N exactly (clean division).
   8,192 blocks x 512 threads/block = 4,194,304 threads  = N  ✓ matches exactly,
   which is WHY this simple version of main() doesn't need a boundary check --
   N divides evenly by THREADS_PER_BLOCK with no remainder.
```

**What if someone changed `THREADS_PER_BLOCK` to `500` instead of `512`?**
```
blocks = N / THREADS_PER_BLOCK = 4,194,304 / 500 = 8,388.608

Using plain integer division in C, this TRUNCATES to 8,388 blocks (the .608 is thrown away!).
8,388 blocks x 500 threads/block = 4,194,000 threads

MISSING elements = N - 4,194,000 = 4,194,304 - 4,194,000 = 304 elements
                    that would NEVER get processed -- a silent bug!
```
This is exactly why [theory.md §4](gpu_lecture04_indexing_kernels_theory.md#4-handling-arbitrary-vector-sizes) insists on the ceiling-division launch formula and the `if (index < n)` boundary guard whenever `N` might not divide evenly — 512 was chosen specifically because `4,194,304 / 512` happens to divide with zero remainder, which is a bit of a "lucky" convenience, not something you should rely on in general code.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 4. Ceiling Division for Arbitrary Sizes — Every Case

The formula: `blocks = (N + M - 1) / M` (using C's truncating integer division).

**Why adding `M - 1` before dividing works — a step-by-step demonstration:**

### Case A: N divides evenly (N=1024, M=256)
```
(1024 + 256 - 1) / 256 = 1279 / 256 = 4.99... -> truncates to 4
Plain N/M would ALSO give 1024/256 = 4.  Same answer -- the "+M-1" trick doesn't
hurt anything when the division is already clean.
```

### Case B: N does NOT divide evenly (N=1000, M=256)
```
Plain N/M = 1000/256 = 3.90... -> truncates to 3 blocks
   3 blocks x 256 threads = 768 threads -- NOT ENOUGH! (1000-768=232 elements missed)

Ceiling version: (1000 + 256 - 1)/256 = 1255/256 = 4.90... -> truncates to 4
   4 blocks x 256 threads = 1024 threads -- covers all 1000, with 24 "extra"
   threads that the boundary check (if index < n) will correctly skip.
```

### Case C: N is JUST ONE more than a clean multiple (N=1025, M=256)
```
Plain N/M = 1025/256 = 4.003... -> truncates to 4 blocks = 1024 threads -- 1 SHORT!
Ceiling: (1025+256-1)/256 = 1280/256 = 5.0 exactly -> 5 blocks = 1280 threads -- covers it.
```
**The pattern to notice:** the ceiling trick effectively asks "if there's ANY remainder at all, round up one whole extra block" — adding `M-1` before truncating-divide is a classic, reusable programming pattern for exactly this kind of "round up, never round down" arithmetic, useful far beyond CUDA.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 5. Matrix Multiplication — Counting the Independent Work

For two N×N matrices, `C = A · B` requires **N² independent inner products** — let's build real numbers around that claim.

```
For a 5x5 matrix (N=5):
   Independent inner products = N^2 = 5^2 = 25
   Each inner product sums N=5 multiply-add pairs.
   Total multiply-add OPERATIONS = N^2 x N = N^3 = 5^3 = 125 operations.

For a 512x512 matrix (N=512):
   Independent inner products = 512^2 = 262,144
   Each does 512 multiply-adds.
   Total operations = 512^3 = 134,217,728  (over 134 MILLION operations)

For a 2048x2048 matrix (N=2048):
   Independent inner products = 2048^2 = 4,194,304
   Total operations = 2048^3 = 8,589,934,592  (over 8.5 BILLION operations)
```

**Why this matters for thread launch planning:** if you assign ONE thread per output cell (the natural mapping from [theory.md §7](gpu_lecture04_indexing_kernels_theory.md#7-matrix-multiplication-as-n²-independent-inner-products)), a 512×512 matrix multiply needs **262,144 threads** — far more than any single thread block can hold (max ~1024), which is exactly why matrix multiplication kernels always use a full 2D GRID of blocks (as shown in the real `matrix_mul.cu` code in [`Basic/cpp/`](../Basic/cpp/02_matrix_mul.cu)), never just one block.

**Quick launch-size estimate for 512×512 with 16×16 blocks:**
```
Threads per block = 16 x 16 = 256
Total threads needed = 512 x 512 = 262,144
Blocks needed = 262,144 / 256 = 1,024 blocks
   (matches the grid arrangement of 32x32 blocks, since 32x16=512 in each dimension)
```

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 6. Exam-Style Numerical Traps

1. **Integer division TRUNCATES in C/C++, it never rounds to nearest.** `1000/256` is `3`, not `4` — this is exactly why the `+M-1` ceiling trick is needed; forgetting this truncation behavior is the root cause of most "off by a few elements" bugs.
2. **Bytes vs elements vs KB/MB** — always convert carefully: `sizeof(int)` is typically 4 bytes, so element count and byte count differ by a factor of 4 (or whatever the type size is). Don't report `size` in "elements" when a question asks for memory footprint.
3. **N³ growth is brutal** — matrix multiplication's operation count grows with the CUBE of the side length, not the square. Going from a 512×512 to a 1024×1024 matrix (only 2× the side length) means `1024³/512³ = 8×` the total work, not 2× or 4×.
4. **"How many blocks" questions always want the CEILING, not the floor**, unless the question explicitly says the size divides evenly. Always check divisibility before deciding which formula to use.

---

## 7. Three More Combined-Index Speed Drills

Setup: `M = 128` threads/block.

### Find blockIdx/threadIdx for array index 5000
```
5000 div 128 = 39 remainder 8   (128x39=4992, 5000-4992=8)
blockIdx.x = 39,  threadIdx.x = 8
Verify: 8 + 39*128 = 8 + 4992 = 5000  correct.
```

### Find blockIdx/threadIdx for array index 127 (last thread of the FIRST block)
```
127 div 128 = 0 remainder 127
blockIdx.x = 0,  threadIdx.x = 127
Verify: 127 + 0*128 = 127  correct -- this is the highest threadIdx.x possible
within block 0, since valid thread indices run 0 to 127 for a 128-thread block.
```

### Find blockIdx/threadIdx for array index 128 (first thread of the SECOND block)
```
128 div 128 = 1 remainder 0
blockIdx.x = 1,  threadIdx.x = 0
Verify: 0 + 1*128 = 128  correct -- notice indices 127 and 128 are adjacent in the
array, but land in DIFFERENT blocks (0 vs 1) with threadIdx.x resetting to 0.
This block-boundary-crossing behavior is worth internalizing: consecutive array
positions don't always mean consecutive (block, thread) pairs in the same block.
```

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 8. Full Worked Memory-Sizing Problem — A Different N

Redo the [§3](#3-sizing-the-real-main-memory-and-launch-numbers) memory-sizing exercise, but for `N = 4096*4096` elements, `THREADS_PER_BLOCK = 1024`, using `double` (8-byte) elements instead of `int`:

```
STEP 1: N = 4096 x 4096 = 16,777,216 elements

STEP 2: size (bytes) = N x sizeof(double) = 16,777,216 x 8 = 134,217,728 bytes
        = 134,217,728 / 1024 = 131,072 KB
        = 131,072 / 1024 = 128 MB      (per array!)

STEP 3: total device memory for 3 arrays (a, b, c) = 3 x 128 MB = 384 MB

STEP 4: blocks = N / THREADS_PER_BLOCK = 16,777,216 / 1024 = 16,384 blocks

STEP 5: sanity check: 16,384 blocks x 1024 threads/block = 16,777,216 threads = N exactly
        (clean division again, since both N and THREADS_PER_BLOCK are powers of 2 here)
```
**Why this matters practically:** 384 MB might sound small, but this is just for a single kernel's THREE working arrays — a real application juggling many such buffers, at even bigger sizes, is exactly why GPU memory capacity (e.g. the A100's 80GB, referenced elsewhere in this repo's Enterprise_Ops track) becomes a hard practical constraint, not just a theoretical one.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

## 9. Extra Practice Set — Ceiling Division Speed Drills

Compute `blocks = (N + M - 1) / M` for each pair, using C-style truncating integer division:

| # | N | M | blocks = ? | total threads launched | "extra" threads |
|---|---|---|---|---|---|
| a | 500 | 128 | ? | ? | ? |
| b | 4096 | 512 | ? | ? | ? |
| c | 1 | 256 | ? | ? | ? |
| d | 999,999 | 1000 | ? | ? | ? |

<details><summary>Answers</summary>

```
a) (500+127)/128 = 627/128 = 4.898 -> 4 blocks
   4 x 128 = 512 threads,  extra = 512-500 = 12

b) (4096+511)/512 = 4607/512 = 8.998 -> 8 blocks
   8 x 512 = 4096 threads, extra = 4096-4096 = 0  (N divides evenly here!)

c) (1+255)/256 = 256/256 = 1.0 -> 1 block
   1 x 256 = 256 threads, extra = 256-1 = 255  (a LOT of wasted threads for
   just 1 real element -- an extreme but valid edge case worth knowing about)

d) (999999+999)/1000 = 1000998/1000 = 1000.998 -> 1000 blocks
   1000 x 1000 = 1,000,000 threads, extra = 1,000,000-999,999 = 1
```
</details>

**The lesson from row (c):** ceiling division is CORRECT even for tiny or oddly-shaped `N` — but it can be wasteful in relative terms (255 out of 256 launched threads doing nothing) when `N` is much smaller than a single block. Real code sometimes special-cases very small `N` to avoid this kind of waste, though it's rarely worth the added complexity unless `N` is a common, hot-path case.

[🔝 Back to Top](#-lecture-04--vectormatrix-kernels--indexing-numerical-deep-dive)

---

> *GPU Programming · Lecture 04 · github.com/rpaut03l/TS-02-03*
