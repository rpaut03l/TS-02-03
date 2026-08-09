# 🎓 Lecture 02 — CUDA Programming Model, SIMT & Fermi Architecture

### *How software (thread/block/grid) maps to real silicon (core/SM/GPU), SIMT vs SIMD, and the Fermi SM in full detail*

> **Nav:** [← Lecture 01](../Lecture_01_GPU_Fundamentals_Hardware_Accelerators/README.md) | **Lecture 02** | [Lecture 03 →](../Lecture_03_Thread_Hierarchy_Memory_HelloWorld/README.md)
> **Source:** GPU Programming, Lecture-02 slides (Prof. Binod Kumar)

---

## 👶 30-second story

Lecture 01 told you a GPU has "thousands of tiny workers." This lecture opens the hood: how does the software concept of a "kernel" actually become physical electrical activity inside real silicon? You'll meet the CUDA hardware/software mapping, the SIMT execution model (and how it differs from SIMD), the Fermi Streaming Multiprocessor's actual internal parts, warps, and a fully worked occupancy calculation — the exact math that decides how many threads truly run at once on one physical chip.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_lecture02_simt_fermi_theory.md](gpu_lecture02_simt_fermi_theory.md) | Programming models (CUDA/OpenCL comparison), CUDA hardware/software stack, SIMT vs SIMD, GPU microarchitecture, Nvidia Fermi architecture + SM internals, warps, the 16×16-block occupancy worked example, latency×throughput scheduling math, GPU memory hierarchy, the 5-step workload dataflow, kernel launch chevron syntax |
| [gpu_lecture02_simt_fermi_numerical.md](gpu_lecture02_simt_fermi_numerical.md) | Deep step-by-step occupancy calculations (5 block sizes fully traced), latency x throughput built from scratch, warp-numbering recipe for any thread count |
| [gpu_lecture02_simt_fermi_practice.md](gpu_lecture02_simt_fermi_practice.md) | Self-test questions including a full occupancy calculation exercise |

---

## 🎯 After this lecture you should be able to…

- Map every CUDA software term (grid/block/warp/thread) to its OpenCL equivalent (NDRange/work-group/—/work-item) and its physical hardware equivalent (GPU/SM/warp scheduler/CUDA core)
- Explain SIMT precisely and state how it differs from SIMD (not just "similar sounding names")
- Describe the Fermi SM's internal components: dual warp schedulers, 32 CUDA cores, 4 SFUs, 16 LD/ST units, configurable shared memory/L1
- Work through a full occupancy calculation: given a block size and per-SM limits, compute blocks/SM, total threads, and warps
- State the "Number of Required Warps = Latency × Throughput" relationship and apply it to Fermi vs Kepler numbers
- Trace the 5-step dataflow of a GPU workload from host memory to device and back

---

> *GPU Programming · Lecture 02 · github.com/rpaut03l/TS-02-03*
