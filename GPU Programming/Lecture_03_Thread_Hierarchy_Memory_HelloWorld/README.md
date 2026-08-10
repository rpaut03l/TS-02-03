# 🎓 Lecture 03 — Thread Hierarchy, Memory & Your First Kernel

### *The 9-step GPU programming recipe, dim3 indexing, a fully worked thread-343 index problem, and Hello World with real device code*

> **Nav:** [← Lecture 02](../Lecture_02_CUDA_Programming_Model_SIMT_Fermi/README.md) | **Lecture 03** | [Lecture 04 →](../Lecture_04_Vector_Matrix_Kernels_Indexing/README.md)
> **Source:** GPU Programming, Lecture-03 slides (Prof. Binod Kumar)

---

## 👶 30-second story

Lectures 01–02 built the mental model. This lecture hands you the actual **recipe** — the exact 9 steps every CUDA program follows, the `dim3` type used to shape grids and blocks, a fully worked "which thread touches this exact array element" problem, and finally your literal first working `__global__` kernel and `<<<1,1>>>` launch.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_lecture03_threads_memory_theory.md](gpu_lecture03_threads_memory_theory.md) | Program execution recap, organization of thread blocks (register-count limit explained), the 9-step GPU programming sequence, real Nvidia Fermi hardware specs, S/W abstraction numbers, `dim3` syntax + 3 initialization patterns, a fully worked grid(5,4,1)/block(5,5,1) index problem (thread 343), and Hello World with device code |
| [gpu_lecture03_threads_memory_numerical.md](gpu_lecture03_threads_memory_numerical.md) | Deep step-by-step thread-index derivations (4 more fully worked problems beyond thread 343), every `dim3` initialization pattern traced |
| [gpu_lecture03_threads_memory_code.md](gpu_lecture03_threads_memory_code.md) | Hands-on runnable exercises: a real Hello World, GPU-side `printf`, live `dim3` inspection, verifying the thread-343 problem on real hardware, and seeing the register limit via `nvcc -Xptxas -v` |
| [gpu_lecture03_threads_memory_practice.md](gpu_lecture03_threads_memory_practice.md) | Self-test questions including your own `dim3` index-tracing exercise |

---

## 🎯 After this lecture you should be able to…

- Recite the 9-step sequence for writing any CUDA program, from memory
- Explain WHY thread-block size is capped by the register file, not by an arbitrary number
- Write and initialize a `dim3` variable correctly for 1D, 2D, and 3D shapes
- Given a grid/block size and a flat thread number, derive its exact `blockIdx`/`threadIdx` coordinates by hand
- Write, compile, and explain a minimal `__global__` "Hello World" kernel

---

> *GPU Programming · Lecture 03 · github.com/rpaut03l/TS-02-03*
