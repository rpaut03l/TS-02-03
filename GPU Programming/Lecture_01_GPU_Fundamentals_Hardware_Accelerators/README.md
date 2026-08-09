# 🎓 Lecture 01 — GPU Fundamentals & Hardware Accelerators

### *Why GPUs exist, the hardware accelerator landscape, and how much speedup parallelism can actually buy you*

> **Nav:** [← GPU Programming](../README.md) | **Lecture 01** | [Lecture 02 →](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/README.md)
> **Source:** GPU Programming, Lecture-01 slides (Prof. Binod Kumar)

---

## 👶 30-second story

Before touching a single line of CUDA code, this lecture answers "why does this whole field exist?" — an AI model like an image classifier is really just millions of identical multiply-and-add operations happening over and over on different pixels. A CPU does these one at a time, brilliantly. A GPU does thousands at once, unglamorously. This lecture is the tour of *which* chip does that job (CPU/GPU/TPU/NPU/FPGA/ASIC), *why* GPUs won for AI specifically, and the one math law (Amdahl's) that tells you exactly how much you can ever hope to gain from parallelizing.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_lecture01_hardware_fundamentals_theory.md](gpu_lecture01_hardware_fundamentals_theory.md) | Hardware platform landscape, CPU vs GPU design philosophy, memory hierarchy, Amdahl's Law (derived + worked), the hardware accelerator spectrum, a real battery-thermal-simulation case study, program execution illustration (thread→core, block→SM, grid→GPU) |
| [gpu_lecture01_hardware_fundamentals_numerical.md](gpu_lecture01_hardware_fundamentals_numerical.md) | Deep step-by-step Amdahl's Law derivations (5 fully worked problems), the battery-thermal and CNN-inference tables recomputed and explained cell by cell |
| [gpu_lecture01_hardware_fundamentals_practice.md](gpu_lecture01_hardware_fundamentals_practice.md) | Self-test questions and a worked Amdahl's Law problem set |

---

## 🎯 After this lecture you should be able to…

- Name the 5 major hardware platforms for AI workloads and one advantage + one limitation of each
- Explain in one sentence why GPUs are "high-latency, high-throughput" while CPUs are "low-latency, low-throughput"
- State Amdahl's Law from memory and compute a maximum theoretical speedup given a parallel fraction `P`
- Explain the CPU→GPU relationship (thread→core, thread block→SM, kernel grid→complete GPU unit) at a glance
- Place CPU/GPU/FPGA/ASIC correctly on the flexibility-vs-efficiency spectrum

---

> *GPU Programming · Lecture 01 · github.com/rpaut03l/TS-02-03*
