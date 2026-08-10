# 📖 Lecture 01 — GPU Fundamentals & Hardware Accelerators: THEORY

> **Nav:** [← Lecture 01 README](README.md) | **THEORY** | [🎯 PRACTICE](gpu_lecture01_hardware_fundamentals_practice.md)

---

## 🧠 MNEMONIC: **"HAG-CAM"**

> **H**ardware platforms · **A**ccelerator tradeoffs · **G**PU design philosophy · **C**ompute/memory hierarchy · **A**mdahl's Law · **M**apping (thread→core→SM→GPU)

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|---|---|
| 1 | Why This Field Exists — An AI Task Example | [§1](#1-why-this-field-exists-an-ai-task-example) |
| 2 | The 5 Hardware Platforms | [§2](#2-the-5-hardware-platforms) |
| 3 | Comparative Evaluation of Devices | [§3](#3-comparative-evaluation-of-devices) |
| 4 | GPU Design Philosophy — CPU vs GPU | [§4](#4-gpu-design-philosophy-cpu-vs-gpu) |
| 5 | CPU & GPU Compute + Memory Hierarchy | [§5](#5-cpu--gpu-compute--memory-hierarchy) |
| 6 | Amdahl's Law | [§6](#6-amdahls-law) |
| 7 | Case Study — Battery Thermal Simulation | [§7](#7-case-study-battery-thermal-simulation) |
| 8 | Case Study — CNN Inference Across Hardware | [§8](#8-case-study-cnn-inference-across-hardware) |
| 9 | The Hardware Accelerator Spectrum | [§9](#9-the-hardware-accelerator-spectrum) |
| 10 | Program Execution Illustration | [§10](#10-program-execution-illustration) |
| 11 | Cheat Sheet & Exam Hacks | [§11](#11-cheat-sheet--exam-hacks) |

---

## 1. Why This Field Exists — An AI Task Example

### 👶 Easy Story
Take a photo of a cat. To recognize it, a computer runs the SAME small math operation (a "convolution") over and over — once for every tiny patch of the image, at every layer. Millions of identical little multiply-and-add jobs. That repetition is exactly the kind of workload a GPU was built for.

### The pipeline
```
Model Creation <====> Training  ---->  Inference
                                           |
                                           v
   Input Image
        |
        v
 +------------+  +------------+       +------------+   +---------+   +---------+
 |   Conv.    |->|   Conv.    |->...->|   Conv.    |-->|  Pool.  |-->|   FC    |--> Output
 |   Layer    |  |   Layer    |       |   Layer    |   |  Layer  |   |  Layer  |    (e.g. "IIT
 +------------+  +------------+       +------------+   +---------+   +---------+     Jodhpur")
  [Feature Extraction: Convolutional Layers]           [Summarize features]
```
- **Convolutional layers** extract features (edges, textures, shapes) — this is where the repetitive math lives.
- **Pooling layers** summarize/downsample those features.
- **Fully-connected (FC) layers** turn the summarized features into a final decision.

Every one of those convolution operations across every layer is **independent of every other one at the same layer** — a textbook data-parallel workload, and exactly why this course exists.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 2. The 5 Hardware Platforms

| # | Platform | Full name | Typical role |
|---|---|---|---|
| 1 | **CPU** | Central Processing Unit | General-purpose, sequential/control-heavy work |
| 2 | **GPU** | Graphics Processing Unit | Massive data-parallel math |
| 3 | **TPU** | Tensor Processing Unit | Google's custom chip, tensor ops (matmul/conv) |
| 4 | **NPU** | Neural Processing Unit | Low-power, on-device AI inference (phones) |
| 5 | **FPGA** | Field-Programmable Gate Array | Reconfigurable hardware logic |

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 3. Comparative Evaluation of Devices

| | **CPU-based** | **GPU-based** | **FPGA-based** | **ASIC-based** |
|---|---|---|---|---|
| **Advantages** | Good versatility, lowest price, multitasking, high programmability | Medium versatility, massive parallelism, moderate programmability | Customized designs, low latency, high performance/watt | Extremely low power, highest performance |
| **Limitations** | Limited parallelism | Power hungry | Limited on-chip memory, requires design expertise | High development cost, long time-to-market, low flexibility |
| **Example devices** | Arm Cortex-M, Raspberry Pi, NanoPi, Sipeed MAIX | NVIDIA Jetson, AMD Ryzen, Arm Mali GPUs | Xilinx Zynq, Intel Arria 10, Lattice iCE40 | Google Edge TPU, Ascend 310, in-memory chips, neuromorphic chips |
| **Development tools** | Arm NN, TensorFlow Lite | TensorRT, Intel OpenVino | Intel OpenVino, Xilinx Edge AI platform | Apache TVM |

*(Adapted from Hao et al., "Enabling Design Methodologies and Future Trends for Edge AI: Specialization and Co-design", IEEE Design & Test, 2021)*

**The pattern to notice:** as you move CPU → GPU → FPGA → ASIC, you trade **flexibility** for **efficiency**. This exact trade-off reappears in [§9](#9-the-hardware-accelerator-spectrum).

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 4. GPU Design Philosophy — CPU vs GPU

### The core distinction
```
CPU:  low-latency,  low-throughput   -> high clock freq, large caches,
                                         sophisticated control, powerful ALUs
GPU:  high-latency, high-throughput  -> moderate clock freq, small caches,
                                         simple control, MANY energy-efficient ALUs
                                         -> requires MASSIVE threads to hide latency
```

### The classic diagram
```
        CPU                                    GPU
+------------+--------+--------+    +----+----+----+----+----+----+----+----+
|            |  ALU   |  ALU   |    | c  |    |    |    |    |    |    |    |
|  Control   +--------+--------+    +----+----+----+----+----+----+----+----+
|            |  ALU   |  ALU   |    | c  |    |    |    |    |    |    |    |
+------------+--------+--------+    +----+----+----+----+----+----+----+----+
|                              |    | c  |    |    |    |    |    |    |    |
|             Cache            |    +----+----+----+----+----+----+----+----+
|                              |    | c  |    |          GPU                |
+------------------------------+    +----+----+----+----+----+----+----+----+
|             DRAM             |    | c  |    |    |    |    |    |    |    |
+------------------------------+    +----+----+----+----+----+----+----+----+
                                    |            DRAM                       |
                                    +---------------------------------------+
```
- CPU: a few big, complex, general-purpose ALUs, a large control unit, deep caches.
- GPU: hundreds/thousands of small, simple ALUs — most of the chip area goes to compute lanes, not control logic or cache.

### The rule of thumb (from the slides)
```
CPUs are good for SEQUENTIAL parts where LATENCY matters
  -> CPUs can be > 10x FASTER than GPUs for sequential code

GPUs are good for PARALLEL parts where THROUGHPUT wins
  -> GPUs can be > 10x FASTER than CPUs for parallel code
```
Neither chip is universally "better" — the right question is always "is this workload sequential or parallel?"

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 5. CPU & GPU Compute + Memory Hierarchy

```
              CPU                                     GPU
 +---------+--------++---------+--------+   +--------------------------+
 |  Core   |Control ||  Core   |Control |   |  (many small cores,      |
 |  L1     Cache    ||  L1     Cache    |   |   each with its own      |
 +---------+--------++---------+--------+   |   tiny control slice)    |
 |  Core   |Control ||  Core   |Control |   +--------------------------+
 |  L1     Cache    ||  L1     Cache    |   |         L2 Cache         |
 +---------+--------++---------+--------+   +--------------------------+
 |     L2 Cache      ||     L2 Cache    |   |          DRAM            |
 +--------------------+-----------------+   +--------------------------+
 |              L3 Cache                |
 +--------------------------------------+
 |               DRAM                   |
 +--------------------------------------+
```
**The key contrast:** a CPU stacks up THREE cache levels (L1 per-core, L2 per-core, L3 shared) precisely because each of its few cores is doing complex, unpredictable work that benefits from big caches. A GPU has only ONE shared cache level (L2) above DRAM — with thousands of simple cores, the GPU instead hides memory latency by having so many threads in flight that some are always ready to compute while others wait on memory (more on this in [Lecture 02](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_theory.md)).

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 6. Amdahl's Law

### 👶 Easy Story
Say cooking dinner takes 60 minutes: 50 minutes of chopping (which 4 friends can split between them) and 10 minutes of baking (which only the oven can do — no amount of friends speeds that up). No matter how many friends you add to the chopping, that 10-minute bake time is a hard floor on how fast dinner can ever be ready.

### The core idea
> Even if part of your system is very fast (parallelized), the **non-parallel (serial) part limits the total speedup.**

### The formula
```
              1
  S = -----------------
       (1 - P) + P/N
```
Where:
- `S` = overall speedup
- `P` = fraction of the program that **can** be parallelized
- `(1 - P)` = the serial portion (cannot be sped up no matter what)
- `N` = number of processors / parallel units

### Key insight
- If even a **small portion is serial**, it caps the maximum achievable speedup.
- You **cannot get infinite speedup**, no matter how many processors you add.

### Maximum theoretical speedup
As `N → ∞` (infinite processors), the `P/N` term vanishes:
```
                1
  S_max = -----------
            1 - P
```
**Worked example:** if 90% of a program is parallelizable (`P = 0.9`), the best possible speedup — even with infinite processors — is:
```
  S_max = 1 / (1 - 0.9) = 1 / 0.1 = 10x
```
No matter how many GPU cores you throw at it, that program can never run faster than 10× the original. This is *the* number every performance-tuning conversation eventually runs into.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 7. Case Study — Battery Thermal Simulation

Heat diffusion inside a battery is governed by the PDE:
```
  dT/dt = alpha * (d2T/dx2 + d2T/dy2) + Q
```
where `T` = temperature, `alpha` = thermal diffusivity, `Q` = heat source.

Using finite differences, the update rule becomes:
```
  T[i,j]^(n+1) = T[i,j]^n + alpha*(T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1] - 4*T[i,j]) + beta*Q
```
In plain words: **New temperature = Current temperature + Heat received from neighbors + Internally generated heat.** Every grid cell's update depends only on its immediate neighbors — another textbook data-parallel pattern, perfect for one-thread-per-cell GPU execution.

**Real measured results (grid size vs. runtime):**

| Grid Size | CPU (ms) | GPU (ms) | Shared-mem GPU (ms) | GPU Speedup | Shared-mem Speedup |
|---|---|---|---|---|---|
| 256×256 | 51.55 | 0.48 | 0.53 | 107.76× | 96.44× |
| 512×512 | 236.54 | 1.48 | 1.87 | 160.28× | 126.34× |
| 1024×1024 | 906.35 | 5.34 | 6.13 | 169.79× | 147.79× |
| 2048×2048 | 3593.75 | 17.75 | 16.32 | 202.44× | 220.14× |

**What this table teaches:** speedup keeps *growing* as the grid size grows — more cells means more independent work to spread across the GPU's thousands of threads, so the GPU's advantage compounds at scale, exactly the pattern Amdahl's Law predicts when `P` stays high and `N` (effectively) grows with problem size.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 8. Case Study — CNN Inference Across Hardware

**Question:** does the hardware platform actually change real AI model performance? A set of 10 image-classification CNNs (gesture recognition, digit recognition, malaria detection, face mask detection, etc.) were benchmarked in Frames Per Second (FPS) on CPU (Intel Xeon) vs GPU (Tesla T4):

| Model | Application | Layers | FPS: CPU (Xeon) | FPS: GPU (T4) |
|---|---|---|---|---|
| 3Conv-CNN | Gesture Recognition | 3C+3M+1F+4D | 13.39 | 15.07 |
| MLP | Hand Digit Recognition | 3D | 9.41 | 14.38 |
| 2Dense-CNN | Malaria Recognition | 3C+3M+1F+2D+1DR | 10.42 | 14.41 |
| MobileNetV2 | Face Mask Detection | 35C+52BN+17DC+1AP+1F+4D+1DR | 8.01 | 11.76 |
| ResNet50 | Face Mask Detection | 53C+53BN+1M+1GA+10D | 7.26 | 12.62 |

*(Layer key: C=Conv2D, M=MaxPooling2D, F=Flatten, D=Dense, DR=Dropout, BN=BatchNormalization, DC=DepthwiseConv2D, AP=AveragePooling2D, GA=GlobalAveragePooling2D. Results measured on an AMD-Xilinx ZCU104 board.)*

**What this teaches:** the GPU wins on EVERY model in the table, but the *margin* varies a lot — a small 3-conv-layer model barely benefits (13.39 → 15.07 FPS), while heavier models with many layers (ResNet50, MobileNetV2) show a bigger relative jump. Bigger, more parallel workloads benefit more from GPU acceleration — the same Amdahl's-Law intuition from §6 and §7, now shown on real production-style models instead of a toy simulation.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 9. The Hardware Accelerator Spectrum

> **Hardware accelerator** (Wikipedia): *"computer hardware designed to perform specific functions more efficiently compared to software running on a CPU."*

There is always a **tradeoff between flexibility and efficiency** — you invest time and money for better performance and efficiency, at the cost of general-purpose flexibility:

```
More flexible                                          More specialized
More general                                            More efficient
     <---------------------------------------------------------->
   CPU              GPU              FPGA              ASIC
 (general      (parallel math,    (reconfigurable   (fixed-function,
  purpose)      moderate           hardware logic)     built for ONE
                programmability)                        job only)
                                                    -------> ACCELERATOR
```

Every step to the right buys you speed and efficiency for a NARROWER set of tasks — an ASIC is blazingly fast at exactly one job and useless at everything else; a CPU is mediocre at everything but can do anything.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 10. Program Execution Illustration

This is the map you'll use for the rest of the course — every CUDA program you write maps your SOFTWARE structure onto this HARDWARE structure:

```
  SOFTWARE CONCEPT         "Executed by"          HARDWARE COMPONENT
  ------------------                              --------------------
       Thread          ------------------------->        Core
     (1 wavy line)                                    (1 execution lane)

    Thread Block        ------------------------->  Streaming Multiprocessor (SM)
  (group of threads)                                (group of cores + scheduler)

    Kernel Grid          ------------------------->    Complete GPU Unit
  (group of blocks)                                  (all SMs on the chip)
```

**In one sentence:** a **thread** is one worker, a **thread block** is a team of workers sharing one supervisor (the SM), and the **kernel grid** is the entire workforce for one job (the whole GPU). This mapping is the backbone of everything covered in [Lecture 02](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/gpu_lecture02_simt_fermi_theory.md) onward.

[🔝 Back to Top](#-lecture-01--gpu-fundamentals--hardware-accelerators-theory)

---

## 11. Cheat Sheet & Exam Hacks

```
5 PLATFORMS:        CPU, GPU, TPU, NPU, FPGA  (+ ASIC as the extreme specialization case)
FLEXIBILITY SPECTRUM: CPU -> GPU -> FPGA -> ASIC   (more flexible -> more specialized/efficient)

CPU: low-latency, low-throughput   | few complex ALUs | big caches (L1/L2/L3)
GPU: high-latency, high-throughput | many simple ALUs | small cache, needs MASSIVE threads

RULE OF THUMB: CPU > 10x faster for SEQUENTIAL code
               GPU > 10x faster for PARALLEL code

AMDAHL'S LAW:   S = 1 / ( (1-P) + P/N )
MAX SPEEDUP:    S_max = 1 / (1-P)     as N -> infinity
WORKED EX:      P=0.9  ->  S_max = 1/0.1 = 10x   (hard ceiling, however many cores)

MAPPING:  Thread -> Core | Thread Block -> SM | Kernel Grid -> Complete GPU
```

### ⚡ Exam Hacks
1. **"Compute the max speedup given P"** — always use `S_max = 1/(1-P)`, NOT the full Amdahl formula (that one needs `N` too, and the question about "maximum possible" implies `N → ∞`).
2. **"Why doesn't adding more GPU cores keep helping forever?"** — the answer is always Amdahl's Law: the serial portion `(1-P)` becomes the bottleneck no matter how large `N` gets.
3. **"CPU vs GPU — which is faster?"** — never answer with just one word; always qualify by workload type: sequential → CPU, parallel/data-parallel → GPU.
4. **Case-study numbers (battery sim, CNN FPS)** — if asked "why does speedup increase with problem size," the answer is: bigger problems give the GPU MORE independent parallel work to fill its thousands of threads with, so the fixed serial/setup overhead becomes a smaller fraction of the total.
5. **Accelerator spectrum question** — always frame it as flexibility-vs-efficiency, and place FPGA between GPU and ASIC (reconfigurable, but still hardware-level, not general-purpose).

---

**Course evaluation scheme (tentative):** Major 60% · Assignments 20% · Quizzes 20% (2 quizzes, 10+10 marks; a valid absence proof allows a replacement assignment).

---

> *GPU Programming · Lecture 01 · github.com/rpaut03l/TS-02-03*
