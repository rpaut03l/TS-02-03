# 🔢 Lecture 03 — Thread Hierarchy, Memory & Your First Kernel: NUMERICAL DEEP DIVE

> **Nav:** [← Lecture 03 README](README.md) | [📖 THEORY](gpu_lecture03_threads_memory_theory.md) | **NUMERICAL** | [🎯 PRACTICE](gpu_lecture03_threads_memory_practice.md)

---

## 📚 Table of Contents

1. [The Thread-343 Problem, Rebuilt From Zero](#1-the-thread-343-problem-rebuilt-from-zero)
2. [Four More Fully Worked Index Problems](#2-four-more-fully-worked-index-problems)
3. [`dim3` Initialization — Every Combination Traced](#3-dim3-initialization-every-combination-traced)
4. [Exam-Style Numerical Traps](#4-exam-style-numerical-traps)
5. [Three More Full Building/Room Problems](#5-three-more-full-buildingroom-problems)
6. [The Register-Limit Math, Applied to This Lecture's Real Fermi Numbers](#6-the-register-limit-math-applied-to-this-lectures-real-fermi-numbers)
7. [Extra Practice Set — Index Speed Drills](#7-extra-practice-set-index-speed-drills)

---

## 1. The Thread-343 Problem, Rebuilt From Zero

Picture a big apartment complex laid out as a grid of **buildings** (blocks), and inside each building, a grid of **numbered rooms** (threads). You're told one specific tenant's overall ID number, and asked: which building, and which exact room, do they live in?

**The setup:**
```
The complex has 5 buildings across (x) and 4 buildings down (y)  = 20 buildings total
Each building has 5 rooms across (x) and 5 rooms down (y)        = 25 rooms per building
Total tenants in the whole complex = 20 buildings x 25 rooms = 500 tenants
```

**Find tenant number 343 (using 0-indexed counting, so tenant "0" is the very first one):**

```
STEP 1 -- How many FULL buildings come before this tenant?
   Every building holds 25 tenants.
   343 divided by 25 = 13 remainder 18
       (because 25 x 13 = 325, and 343 - 325 = 18)

   This means: 13 FULL buildings (0 through 12) are completely filled
   BEFORE we even reach tenant 343's building.
   So tenant 343 lives in BUILDING NUMBER 13 (the 14th building, if
   you count starting from 1 -- but we always count buildings starting at 0).

STEP 2 -- Which room number WITHIN that building?
   The remainder from Step 1 tells us: room number 18 (0-indexed)
   within building 13.

STEP 3 -- Convert "building number 13" into an (x, y) COORDINATE.
   The complex is 5 buildings wide. To go from a single "building number"
   to a (column, row) pair, use:
       column (x) = building_number MOD 5
       row    (y) = building_number DIV 5   (integer division, throw away remainder)

   column = 13 mod 5 = 3   (13 = 5x2 + 3, remainder 3)
   row    = 13 div 5 = 2   (5 goes into 13 twice, with something left over)

   So building 13 sits at coordinate (3, 2) in the grid of buildings.

STEP 4 -- Convert "room number 17" into an (x, y) coordinate INSIDE the building.
   (Using the building-level linear index 17, matching the block-level
    diagram layout in theory.md section 7.)
   Each building is 5 rooms wide, so:
       column (x) = room_number MOD 5
       row    (y) = room_number DIV 5

   column = 17 mod 5 = 2   (17 = 5x3 + 2, remainder 2)
   row    = 17 div 5 = 3   (5 goes into 17 three times, with 2 left over)

   So the room sits at coordinate (2, 3) inside its building.

FINAL ANSWER:
   blockIdx.x = 3     threadIdx.x = 2
   blockIdx.y = 2     threadIdx.y = 3
   blockIdx.z = 0     threadIdx.z = 0
```

**The reusable 4-line recipe, for ANY tenant number:**
```
1.  block_linear  = tenant_number DIV roomsPerBuilding
2.  room_linear   = tenant_number MOD roomsPerBuilding
3.  block (x,y)   = ( block_linear MOD gridWidth,  block_linear DIV gridWidth )
4.  room  (x,y)   = ( room_linear  MOD blockWidth, room_linear  DIV blockWidth )
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

## 2. Four More Fully Worked Index Problems

**Setup for all four:** a grid of `gridDim = (6, 3, 1)` (6 blocks wide, 3 tall), each block `blockDim = (10, 10, 1)` (100 threads/block). Total tenants = `6 × 3 × 100 = 1800`.

### Find tenant 0
```
0 div 100 = 0 remainder 0   -> block_linear=0, room_linear=0
block (x,y) = (0 mod 6, 0 div 6) = (0, 0)
room  (x,y) = (0 mod 10, 0 div 10) = (0, 0)
ANSWER: blockIdx=(0,0,0), threadIdx=(0,0,0)   -- the very first tenant, first room, first building.
```

### Find tenant 250
```
250 div 100 = 2 remainder 50   -> block_linear=2, room_linear=50
block (x,y) = (2 mod 6, 2 div 6) = (2, 0)
room  (x,y) = (50 mod 10, 50 div 10) = (0, 5)
ANSWER: blockIdx=(2,0,0), threadIdx=(0,5,0)
```

### Find tenant 799 (last tenant of block_linear = 7)
```
799 div 100 = 7 remainder 99   -> block_linear=7, room_linear=99
block (x,y) = (7 mod 6, 7 div 6) = (1, 1)
room  (x,y) = (99 mod 10, 99 div 10) = (9, 9)
ANSWER: blockIdx=(1,1,0), threadIdx=(9,9,0)   -- the LAST room (9,9) in its building, makes
        sense since 99 is the highest possible room_linear value (100 rooms, 0-99).
```

### Find tenant 1799 (the very LAST tenant in the whole complex)
```
1799 div 100 = 17 remainder 99   -> block_linear=17, room_linear=99
block (x,y) = (17 mod 6, 17 div 6) = (5, 2)
room  (x,y) = (99 mod 10, 99 div 10) = (9, 9)
ANSWER: blockIdx=(5,2,0), threadIdx=(9,9,0)   -- the bottom-right-most building (5,2) is
        indeed the last one in a 6-wide, 3-tall grid (0-5 across, 0-2 down), and room
        (9,9) is the last room in a 10x10 room grid. This is a great sanity check:
        the LAST overall tenant should always land on the LAST block AND the LAST room.
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

## 3. `dim3` Initialization — Every Combination Traced

Recall `dim3` fills in `1` for any dimension you don't specify. Let's trace exactly what the compiler stores for each call:

```cpp
dim3 a(256);
```
```
You gave it ONE number.  The rule: only the FIRST slot (x) gets your number;
everything else silently becomes 1.
Result:  a.x = 256,  a.y = 1,  a.z = 1
```

```cpp
dim3 b(100, 100);
```
```
You gave it TWO numbers.  The first two slots (x, y) get your numbers;
the third (z) silently becomes 1.
Result:  b.x = 100,  b.y = 100,  b.z = 1
```

```cpp
dim3 c(10, 54, 32);
```
```
You gave it all THREE numbers.  Nothing is defaulted.
Result:  c.x = 10,  c.y = 54,  c.z = 32
```

**A common follow-up question: how many THREADS does each of these represent, if used as a block shape?**
```
a: 256 x 1 x 1   = 256 threads
b: 100 x 100 x 1 = 10,000 threads
c: 10 x 54 x 32  = 17,280 threads
```
⚠️ Note: `c` (17,280 threads) would be an ILLEGAL block size on real hardware, since the maximum threads per block is capped (commonly 1024) by the register-file limit discussed in [theory.md §2](gpu_lecture03_threads_memory_theory.md#2-organization-of-thread-blocks) — this is exactly the kind of number you should sanity-check against the hardware limit whenever you see a `dim3` this large used as a BLOCK shape (as opposed to a GRID shape, where much larger numbers are normal).

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

## 4. Exam-Style Numerical Traps

1. **Building/room index order matters** — ALWAYS compute the block-level (building) coordinates and thread-level (room) coordinates SEPARATELY, using DIV/MOD against their own respective widths (grid width for blocks, block width for threads). Mixing up which width to use for which step is the most common error.
2. **0-indexing vs "the Nth one" language** — "the 18th room" in casual speech could mean index 17 (0-indexed) or 18 — always confirm which convention a question uses, and this repo consistently uses 0-indexing throughout (tenant 0 is the FIRST tenant).
3. **`dim3` defaults to 1, not 0** — unspecified dimensions become `1` (meaning "one slice/row/layer"), never `0` (which would make the total threads multiply out to zero).
4. **The LAST tenant is a great sanity check** — for a grid of size `gridDim.x × gridDim.y` blocks with `blockDim.x × blockDim.y` threads each, the very last tenant (index `total-1`) should always land on `blockIdx = (gridDim.x-1, gridDim.y-1)` and `threadIdx = (blockDim.x-1, blockDim.y-1)` — if your arithmetic doesn't land there for the last index, you made a mistake somewhere.

---

## 5. Three More Full Building/Room Problems

**Setup:** a bigger complex, `gridDim=(10, 8, 1)` (10 buildings across, 8 down), `blockDim=(12, 12, 1)` (144 rooms/building). Total tenants = `10 × 8 × 144 = 11,520`.

### Find tenant 5000
```
STEP 1: 5000 div 144 = 34 remainder 104   (144x34=4896, 5000-4896=104)
        block_linear=34, room_linear=104

STEP 2: block (x,y) = (34 mod 10, 34 div 10) = (4, 3)
STEP 3: room  (x,y) = (104 mod 12, 104 div 12) = (8, 8)

ANSWER: blockIdx=(4,3,0), threadIdx=(8,8,0)
```

### Find tenant 0 (sanity check — should be the very first)
```
0 div 144 = 0 remainder 0  -> block_linear=0, room_linear=0
block (x,y) = (0,0),  room (x,y) = (0,0)
ANSWER: blockIdx=(0,0,0), threadIdx=(0,0,0)  -- as expected, the very first position.
```

### Find tenant 11519 (the LAST tenant — sanity check)
```
11519 div 144 = 79 remainder 143   (144x79=11376, 11519-11376=143)
block_linear=79, room_linear=143

block (x,y) = (79 mod 10, 79 div 10) = (9, 7)
room  (x,y) = (143 mod 12, 143 div 12) = (11, 11)

ANSWER: blockIdx=(9,7,0), threadIdx=(11,11,0)
CHECK: gridDim=(10,8,1) means the last valid block index is (9,7) -- matches!
       blockDim=(12,12,1) means the last valid room index is (11,11) -- matches!
```

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

## 6. The Register-Limit Math, Applied to This Lecture's Real Fermi Numbers

Recall from [theory.md §4](gpu_lecture03_threads_memory_theory.md#4-real-fermi-hardware-specs--sw-abstraction-numbers): Fermi allows **512-1024 threads/block**, max **8 blocks/SM**, **32 threads/warp**.

**If a kernel is launched with the MAXIMUM allowed 1024 threads in a single block, how many blocks can even fit on one SM, purely from the thread-count angle (ignoring registers for a moment)?**
```
Max threads per SM (implied by "512-1024 threads/block" and "max 8 blocks/SM" together) --
if we assume the SM's own thread ceiling is also 1024 (a single maximal block fills it):
   1024 threads (this ONE block) / 1024 threads/SM = exactly 1 block fits.

This means: with a maximal 1024-thread block, the SM can host only 1 block at a time,
LEAVING THE OTHER 7 BLOCK-SLOTS COMPLETELY UNUSED -- because the thread-slot limit
was reached first.
```

**Compare to a more modest 128-thread block:**
```
1024 threads/SM / 128 threads/block = 8 blocks by thread-slot limit
Block-slot limit is ALSO 8.
Both limits agree exactly -> all 8 block-slots get used, 100% utilization of BOTH resources.
```
**The lesson, stated plainly:** the biggest block size isn't automatically the best choice — a block using ALL 1024 threads leaves 7 of 8 block-slots empty, while a well-chosen SMALLER block size (128 here) can use every block-slot AND every thread-slot simultaneously. This is the exact same "seats vs bays" trade-off worked out in [Lecture 02's numerical deep dive](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_numerical.md#2-occupancy-four-more-fully-worked-block-sizes), now grounded in this lecture's specific named hardware (Fermi) and its stated 512-1024 threads/block guidance.

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

## 7. Extra Practice Set — Index Speed Drills

Try these without checking the answer first. Setup: `gridDim=(6,6,1)`, `blockDim=(6,6,1)` (36 threads/block, 1296 tenants total).

| # | Tenant # | blockIdx (x,y) | threadIdx (x,y) |
|---|---|---|---|
| a | 40 | ? | ? |
| b | 200 | ? | ? |
| c | 700 | ? | ? |
| d | 1295 (last) | ? | ? |

<details><summary>Answers</summary>

```
a) 40 div 36 = 1 remainder 4   -> block_linear=1, room_linear=4
   block (1 mod 6, 1 div 6) = (1,0)   room (4 mod 6, 4 div 6) = (4,0)
   ANSWER: blockIdx=(1,0), threadIdx=(4,0)

b) 200 div 36 = 5 remainder 20  -> block_linear=5, room_linear=20
   block (5 mod 6, 5 div 6) = (5,0)   room (20 mod 6, 20 div 6) = (2,3)
   ANSWER: blockIdx=(5,0), threadIdx=(2,3)

c) 700 div 36 = 19 remainder 16  -> block_linear=19, room_linear=16
   block (19 mod 6, 19 div 6) = (1,3)   room (16 mod 6, 16 div 6) = (4,2)
   ANSWER: blockIdx=(1,3), threadIdx=(4,2)

d) 1295 div 36 = 35 remainder 35  -> block_linear=35, room_linear=35
   block (35 mod 6, 35 div 6) = (5,5)   room (35 mod 6, 35 div 6) = (5,5)
   ANSWER: blockIdx=(5,5), threadIdx=(5,5)  -- matches the "last tenant lands on the
           last block AND last room" sanity check, since gridDim and blockDim are
           both (6,6,1) here, so both maximal indices happen to be (5,5).
```
</details>

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

## 8. One Fully Worked "Reverse Direction" Problem

Given `blockIdx=(2,1,0)`, `threadIdx=(3,4,0)`, `gridDim=(5,3,1)`, `blockDim=(6,6,1)` — find the ORIGINAL flat tenant number (the reverse of every problem so far).

```
STEP 1 -- Convert blockIdx (x,y) back into a linear block number.
   block_linear = blockIdx.y * gridDim.x + blockIdx.x
                = 1 * 5 + 2
                = 7

STEP 2 -- Convert threadIdx (x,y) back into a linear room number.
   room_linear = threadIdx.y * blockDim.x + threadIdx.x
               = 4 * 6 + 3
               = 27

STEP 3 -- Combine: overall tenant number = block_linear x roomsPerBuilding + room_linear
   roomsPerBuilding = blockDim.x * blockDim.y = 6*6 = 36
   tenant = 7 * 36 + 27
          = 252 + 27
          = 279

VERIFY (forward direction, using the STANDARD recipe from section 1):
   279 div 36 = 7 remainder 27   -> block_linear=7, room_linear=27  -- matches!
   block (7 mod 5, 7 div 5) = (2,1)   -- matches blockIdx=(2,1,0) given!
   room  (27 mod 6, 27 div 6) = (3,4) -- matches threadIdx=(3,4,0) given!
```
**This is the exact mirror-image of every earlier problem** — going from (block, thread) coordinates BACK to a flat number uses the same building blocks (linearize, then combine), just applied in reverse order. Being able to go BOTH directions confidently is what separates "memorized a formula" from "understands the indexing model."

[🔝 Back to Top](#-lecture-03--thread-hierarchy-memory--your-first-kernel-numerical-deep-dive)

---

> *GPU Programming · Lecture 03 · github.com/rpaut03l/TS-02-03*
