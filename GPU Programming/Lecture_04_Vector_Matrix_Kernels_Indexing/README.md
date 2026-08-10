# 🎓 Lecture 04 — Vector/Matrix Kernels & Indexing

### *From `add<<<N,1>>>()` to a real combined block+thread index formula, boundary-safe kernels, host/device sync, and inner-product reduction with shared memory*

> **Nav:** [← Lecture 03](../Lecture_03_Thread_Hierarchy_Memory_HelloWorld/README.md) | **Lecture 04**
> **Source:** GPU Programming, Lecture-04 slides (Prof. Binod Kumar)

---

## 👶 30-second story

This lecture is where the course stops being purely conceptual and becomes genuinely runnable at scale. Starting from one thread doing one addition, you'll build up — block by block, then thread by thread — to a real, arbitrary-sized vector-add kernel with a full `main()`, learn the exact synchronization functions that keep host and device honest, get introduced to the Python JIT ecosystem (Numba/CuPy/PyCUDA), and finish with matrix multiplication and a shared-memory inner-product reduction kernel.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_lecture04_indexing_kernels_theory.md](gpu_lecture04_indexing_kernels_theory.md) | Moving to parallel with blocks, vector addition on the device, the full block+thread indexing derivation (worked "which thread touches the red element" example), arbitrary vector sizes with boundary checks, the complete `main()` for 2048×2048-element addition, host/device synchronization functions, JIT/Numba/CuPy/PyCUDA definitions, matrix multiplication as N² independent inner products, and the shared-memory reduction kernel |
| [gpu_lecture04_indexing_kernels_numerical.md](gpu_lecture04_indexing_kernels_numerical.md) | Deep step-by-step combined block+thread index problems, real `main()` memory/launch sizing, ceiling-division traced for 3 cases, matrix multiplication operation counts (N to N^3) |
| [gpu_lecture04_indexing_kernels_code.md](gpu_lecture04_indexing_kernels_code.md) | Hands-on runnable exercises: block-only `add<<<N,1>>>()`, a self-checking combined-index kernel, breaking (and fixing) the boundary check on purpose, Numba JIT side-by-side, measuring N^3 growth with a stopwatch, and triggering a real race condition by removing `__syncthreads()` |
| [gpu_lecture04_indexing_kernels_practice.md](gpu_lecture04_indexing_kernels_practice.md) | Self-test questions plus an index-tracing and kernel-completion exercise |

---

## 🎯 After this lecture you should be able to…

- Explain the difference between `add<<<N,1>>>()` (block-parallel only) and the combined `index = threadIdx.x + blockIdx.x * blockDim.x` formula (block- AND thread-parallel)
- Correctly compute which thread touches a given array element, given `M` (threads/block) and the element's flat index
- Write a boundary-safe kernel launch for a vector size that isn't a clean multiple of the block size
- Name the three CPU/GPU synchronization functions and state exactly what each one blocks on
- Explain, at a glance, what Numba, CuPy, and PyCUDA each are and how they differ
- Explain why matrix multiplication of two N×N matrices is N² independent inner products
- Explain what `__syncthreads()` guarantees and why the reduction example calls it before summing

---

> *GPU Programming · Lecture 04 · github.com/rpaut03l/TS-02-03*
