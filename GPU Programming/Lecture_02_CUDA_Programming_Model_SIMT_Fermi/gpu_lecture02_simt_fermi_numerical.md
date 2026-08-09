# 🔢 Lecture 02 — CUDA Programming Model, SIMT & Fermi Architecture: NUMERICAL DEEP DIVE

> **Nav:** [← Lecture 02 README](README.md) | [📖 THEORY](gpu_lecture02_simt_fermi_theory.md) | **NUMERICAL** | [🎯 PRACTICE](gpu_lecture02_simt_fermi_practice.md)

---

## 📚 Table of Contents

1. [The Occupancy Calculation — Every Step, Slowly](#1-the-occupancy-calculation-every-step-slowly)
2. [Occupancy — Four More Fully Worked Block Sizes](#2-occupancy-four-more-fully-worked-block-sizes)
3. [Latency × Throughput — Building the Table From Scratch](#3-latency-x-throughput-building-the-table-from-scratch)
4. [Warp Numbering for Any Thread Count](#4-warp-numbering-for-any-thread-count)
5. [Exam-Style Numerical Traps](#5-exam-style-numerical-traps)
6. [Two More Fully Worked Occupancy Problems](#6-two-more-fully-worked-occupancy-problems)
7. [Register File Math — Combining Occupancy with Register Pressure](#7-register-file-math-combining-occupancy-with-register-pressure)
8. [Extra Practice Set — Speed Drills](#8-extra-practice-set-speed-drills)

---

## 1. The Occupancy Calculation — Every Step, Slowly

Imagine a school bus depot (an SM) that has exactly **1024 numbered seats** and exactly **8 parking bays** for whole buses (blocks). Each "bus" (thread block) you send in has to have the SAME number of seats filled, and it takes up ONE parking bay regardless of how full it is.

**The problem:** you're sending buses that each hold **256 passengers** (a 16×16 thread block = 256 threads). How many buses fit, and how full does the depot get?

```
STEP 1 -- Check the SEAT limit:
  Total seats at depot = 1024
  Seats per bus        = 256
  Buses allowed by seats = 1024 / 256 = 4 buses

STEP 2 -- Check the BAY limit:
  Total parking bays = 8
  Buses we WANT to send (from Step 1) = 4
  Is 4 <= 8?  YES -- the bay limit does not block us here.

STEP 3 -- Take the smaller (limiting) number.
  min(4 buses from seats, 8 buses allowed by bays) = 4 buses

STEP 4 -- Total passengers actually seated:
  4 buses x 256 passengers/bus = 1024 passengers
  (matches the seat limit exactly -- no seats wasted!)

STEP 5 -- Convert passengers into "groups of 32" (a warp):
  1024 passengers / 32 per group = 32 groups (warps)

STEP 6 -- Compare to the MAXIMUM groups the depot can handle at once:
  Depot can handle 32 groups (warps) max.
  We are using exactly 32 groups.

STEP 7 -- Occupancy percentage:
  (groups used / groups possible) x 100 = (32/32) x 100 = 100%
```

**The one sentence that explains WHY this matters:** every seat left empty, or every bay left unused, is a physical part of the chip sitting idle instead of doing math — occupancy is just "what fraction of the hardware's parallel capacity is actually being used right now."

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

## 2. Occupancy — Four More Fully Worked Block Sizes

Using the SAME depot rules (1024 seats, 8 bays, 32 max warp-groups):

### Block size = 64 threads
```
STEP 1: seats: 1024 / 64 = 16 buses allowed by seats
STEP 2: bays: is 16 <= 8?  NO!  Only 8 bays exist.
STEP 3: limiting number = min(16, 8) = 8 buses  (BAY limit wins this time)
STEP 4: passengers seated = 8 x 64 = 512  (HALF the seats sit empty!)
STEP 5: warps = 512 / 32 = 16 warps
STEP 6: occupancy = 16/32 = 50%
```
**Lesson:** tiny blocks waste bays. Even though you technically COULD fit 16 buses by seat-count, you only have 8 parking bays, so half your seating capacity goes unused.

### Block size = 128 threads
```
STEP 1: seats: 1024 / 128 = 8 buses allowed by seats
STEP 2: bays: is 8 <= 8?  YES, exactly matches.
STEP 3: limiting number = 8 buses
STEP 4: passengers = 8 x 128 = 1024  (full seats!)
STEP 5: warps = 1024/32 = 32 warps
STEP 6: occupancy = 32/32 = 100%
```
**Lesson:** 128 threads/block is the SMALLEST block size that hits both limits simultaneously (matches the "you need at least 128 threads/block" rule mentioned in [Lecture_04's performance considerations](../Lec_04_Performance_Considerations/gpu_lec04_theory.md#3-dynamic-partitioning-of-sm-resources-the-occupancy-cliff)).

### Block size = 512 threads
```
STEP 1: seats: 1024 / 512 = 2 buses allowed by seats
STEP 2: bays: is 2 <= 8?  YES.
STEP 3: limiting number = 2 buses
STEP 4: passengers = 2 x 512 = 1024  (full seats!)
STEP 5: warps = 1024/32 = 32 warps
STEP 6: occupancy = 32/32 = 100%
```

### Block size = 1024 threads
```
STEP 1: seats: 1024 / 1024 = 1 bus allowed by seats
STEP 2: bays: is 1 <= 8?  YES.
STEP 3: limiting number = 1 bus
STEP 4: passengers = 1 x 1024 = 1024  (full!)
STEP 5: warps = 1024/32 = 32 warps
STEP 6: occupancy = 32/32 = 100%
```
**Lesson:** a single giant block CAN reach 100% occupancy too — but it leaves NO room for the scheduler to overlap multiple independent blocks, which matters for other reasons covered in later performance lectures (register pressure, latency hiding flexibility). Occupancy alone doesn't tell the whole performance story.

**Summary table of everything computed above:**

| Block size | Buses (limit) | Total threads | Warps | Occupancy |
|---|---|---|---|---|
| 64 | 8 (bay-limited) | 512 | 16 | 50% |
| 128 | 8 (both limits) | 1024 | 32 | 100% |
| 256 | 4 (seat-limited) | 1024 | 32 | 100% |
| 512 | 2 (seat-limited) | 1024 | 32 | 100% |
| 1024 | 1 (seat-limited) | 1024 | 32 | 100% |

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

## 3. Latency × Throughput — Building the Table From Scratch

Think of an assembly line where a part takes **20 seconds to travel down the conveyor belt** before it's finished (that's the "latency"), but the belt is WIDE enough to have **32 parts moving down it side-by-side at once** (that's the "throughput" — parts finished per second, once the line is full).

**Question: how many parts need to be ON the belt at once to keep it fully busy, with zero gaps?**
```
STEP 1: If it takes 20 seconds for ONE part to finish...
STEP 2: ...and the belt can output 32 finished parts EVERY second once full...
STEP 3: ...then in those 20 seconds, you need enough parts already loaded to
        keep producing 32-per-second output continuously.
STEP 4: Total parts needed = 20 seconds x 32 parts/second = 640 parts in flight.
```
This is exactly the Fermi row of the table:
```
Fermi:  Latency=20 cycles, Throughput=32 ops/cycle
        Required parallelism = 20 x 32 = 640 operations
        In warps: 640 / 32 (threads per warp) = 20 warps
```

**Now the Kepler row, same method, wider belt:**
```
Kepler: Latency=20 cycles (SAME wait time), Throughput=192 ops/cycle (6x WIDER belt)
        Required parallelism = 20 x 192 = 3,840 operations
        In warps: 3,840 / 32 = 120 warps
```

**Why did Kepler need 6× more warps for the SAME latency?** Because the belt got 6× wider (32 → 192 ops/cycle) but the travel time down the belt stayed the same (20 cycles) — a wider belt just means you need proportionally MORE parts loaded onto it at once to keep every lane busy the whole time. The formula scales linearly with throughput: double the throughput, double the required warps, for the same latency.

**Extending the pattern — what if a hypothetical GPU had latency=30 cycles and throughput=64 ops/cycle?**
```
Required operations = 30 x 64 = 1,920
Required warps = 1,920 / 32 = 60 warps
```

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

## 4. Warp Numbering for Any Thread Count

**The rule, stated as a simple recipe:** to find which warp a given thread number `T` belongs to, divide by 32 and throw away the remainder (integer division).
```
  warp_number = T div 32
```

**Worked examples:**
```
Thread   0:  0 div 32 = 0   -> Warp 0
Thread  31: 31 div 32 = 0   -> Warp 0  (last thread still in warp 0)
Thread  32: 32 div 32 = 1   -> Warp 1  (first thread of the NEXT warp)
Thread  99: 99 div 32 = 3   -> Warp 3   (since 32x3=96, 99-96=3, still within warp 3's range 96-127)
Thread 200: 200 div 32 = 6  -> Warp 6   (32x6=192, 32x7=224, so 200 falls between 192 and 223)
```
**Double-check thread 200 the long way:**
```
Warp 6 covers threads 192 to 223 (inclusive)   [32x6=192,  32x7-1=223]
Is 200 between 192 and 223?  Yes.  Confirmed: thread 200 is in Warp 6.
```

**A block with 100 threads — how many WHOLE warps, and is there a padded/partial warp?**
```
100 / 32 = 3.125

3 FULL warps = 96 threads (warps 0, 1, 2 -- threads 0-95)
Remaining threads = 100 - 96 = 4 threads (threads 96-99)

These 4 threads form the START of Warp 3, which needs 32 threads to be "full."
The hardware PADS the remaining 28 slots in Warp 3 with inactive placeholder threads.

Total warps allocated = 4 (even though only 100 REAL threads exist)
Wasted/padded thread-slots = 4 x 32 - 100 = 128 - 100 = 28 slots doing nothing.
```
**Lesson:** any block size that isn't a clean multiple of 32 wastes some warp capacity on padding — this is exactly why block sizes like 256, 512, 128 (all multiples of 32) are preferred over odd numbers like 100 or 250.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

## 5. Exam-Style Numerical Traps

1. **Occupancy problems always have (at least) TWO independent limits** (thread-slot/seat limit AND block-slot/bay limit) — compute both separately and take the smaller one. Computing only one and stopping is the #1 lost-marks mistake.
2. **"Required warps" formula order matters** — always compute `Latency × Throughput` FIRST to get total operations, THEN divide by 32 to convert to warps. Don't divide by 32 first.
3. **Warp number = integer division by 32, always round DOWN**, never round to nearest. Thread 31 is warp 0, not warp 1, even though 31/32 = 0.97 looks "almost 1."
4. **Non-multiple-of-32 block sizes always waste warp capacity** — the wasted amount is `(warps_allocated × 32) − actual_thread_count`, not zero, even if the "missing" threads are simply inactive rather than causing an error.

---

## 6. Two More Fully Worked Occupancy Problems

### Problem: A "wasteful" block size on a modern-style SM
```
SM limits: 2048 max threads, 32 max blocks/SM, 64 max warps/SM
Block size chosen: 48 threads (an unusual, non-multiple-of-32 choice)

STEP 1 -- Seats: 2048 / 48 = 42.67 -> only 42 WHOLE blocks fit by thread count
STEP 2 -- Bays: is 42 <= 32?  NO -- the 32-block limit is now the binding constraint!
STEP 3 -- limiting number = min(42, 32) = 32 blocks
STEP 4 -- passengers = 32 x 48 = 1536 threads (512 threads' worth of capacity WASTED,
           since 2048-1536=512 seats sit empty)
STEP 5 -- warps: each 48-thread block needs ceil(48/32)=2 warps (1 full warp of 32,
           plus 1 PARTIAL warp with only 16 real threads + 16 padded/inactive slots)
           32 blocks x 2 warps/block = 64 warps allocated
STEP 6 -- occupancy (by warp count): 64/64 = 100% ... but this is MISLEADING, because
           16 of the 32 "threads" in each block's second warp are just padding, not
           real work -- a great example of why raw warp-count occupancy alone doesn't
           tell the whole efficiency story.
```

### Problem: Finding the block size that minimizes waste for a given SM
```
SM limits: 1024 max threads, 8 max blocks/SM
Which of these block sizes achieves 100% occupancy: 96, 100, 112, 128?

  96:  1024/96 = 10.67 -> 10 blocks by threads, but bay limit=8 -> 8 blocks win
       8 x 96 = 768 threads (256 threads WASTED)  -> NOT 100%

  100: 1024/100 = 10.24 -> 10 blocks by threads, bay limit=8 -> 8 blocks win
       8 x 100 = 800 threads (224 threads WASTED) -> NOT 100%
       (also: 100 isn't a multiple of 32, so warps are padded too -- doubly wasteful)

  112: 1024/112 = 9.14 -> 9 blocks by threads, bay limit=8 -> 8 blocks win
       8 x 112 = 896 threads (128 threads WASTED) -> NOT 100%

  128: 1024/128 = 8 exactly -> 8 blocks by threads, bay limit=8 -> matches exactly
       8 x 128 = 1024 threads (0 WASTED) -> 100% occupancy!  AND 128 is a clean
       multiple of 32, so no warp padding either.
```
**Lesson:** 128 threads/block is a "sweet spot" precisely because it satisfies BOTH the thread-slot and block-slot limits simultaneously on many common SM configurations, AND divides cleanly into whole warps.

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

## 7. Register File Math — Combining Occupancy with Register Pressure

Building on [Lecture 04's occupancy-cliff discussion](../Lecture_04_Performance_Considerations/gpu_lec04_theory.md#3-dynamic-partitioning-of-sm-resources-the-occupancy-cliff), let's redo a register-limited calculation using THIS lecture's Fermi numbers directly.

```
Fermi register file: 32,768 registers/SM
Block size: 256 threads
Registers used per thread: 24

STEP 1 -- Registers needed per block = 256 threads x 24 registers/thread = 6,144 registers

STEP 2 -- How many blocks fit in the register file alone?
   32,768 / 6,144 = 5.33 -> only 5 WHOLE blocks fit by register count

STEP 3 -- Compare against the thread-slot and block-slot limits (1024 threads/SM, 8 blocks/SM):
   Thread-slot limit: 1024/256 = 4 blocks
   Block-slot limit:  8 blocks
   Register limit (from Step 2): 5 blocks

STEP 4 -- The ACTUAL number of blocks scheduled = the MINIMUM of all three limits:
   min(4, 8, 5) = 4 blocks   <- the THREAD-SLOT limit is what actually binds here,
                                 not the register limit, even though we computed it!

STEP 5 -- Total threads = 4 x 256 = 1024 threads -> 1024/32 = 32 warps -> 100% occupancy
           (same result as the original 256-thread-block example, register pressure
            wasn't actually the bottleneck at 24 registers/thread)
```
**Now bump registers/thread to 40, everything else the same:**
```
Registers per block = 256 x 40 = 10,240
Register limit: 32,768/10,240 = 3.2 -> only 3 WHOLE blocks fit

Compare all three limits: thread-slot=4, block-slot=8, register=3
Actual blocks = min(4,8,3) = 3 blocks   <- NOW the register limit IS the bottleneck!

Total threads = 3 x 256 = 768 -> 768/32 = 24 warps -> 24/32 = 75% occupancy
   (occupancy DROPPED from 100% to 75% purely from increasing registers/thread from
    24 to 40 -- a smaller-scale echo of the "performance cliff" idea)
```

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

## 8. Extra Practice Set — Speed Drills

Try these without looking at the answer first:

| # | Question | Answer |
|---|---|---|
| a | Which warp does thread 500 belong to? | ? |
| b | Latency=15 cycles, Throughput=64 ops/cycle — required warps? | ? |
| c | Block size 64, SM limits 1024 threads/8 blocks — occupancy %? | ? |
| d | Registers/SM=65536, 512 threads/block, 20 regs/thread — blocks by register limit? | ? |

<details><summary>Answers</summary>

```
a) 500 div 32 = 15 remainder 20 -> Warp 15 (covers threads 480-511, and 500 is in that range)
b) 15 x 64 = 960 operations -> 960/32 = 30 warps
c) 1024/64=16 blocks by threads, but bay limit=8 -> 8 blocks win
   8x64=512 threads (half the seats empty) -> 512/32=16 warps
   -- need to know max warps/SM to get %, but relative to the 1024-thread capacity,
      only 512/1024 = 50% of thread capacity is used
d) 512 threads x 20 regs = 10,240 regs/block. 65536/10240 = 6.4 -> 6 blocks by register limit
```
</details>

[🔝 Back to Top](#-lecture-02--cuda-programming-model-simt--fermi-architecture-numerical-deep-dive)

---

> *GPU Programming · Lecture 02 · github.com/rpaut03l/TS-02-03*
