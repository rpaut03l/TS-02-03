# 🧩 Basic — Core CUDA Programs (C++ & Python)

### *The two canonical programs — Vector Addition and Matrix Multiplication — in every language track*

> **Nav:** [← GPU Programming](../README.md)

---

## 👶 What this folder is

Every GPU programming course teaches the same two "hello world" problems first: **adding two lists of numbers** and **multiplying two grids of numbers**. This folder collects clean, runnable, heavily-commented versions of both — in raw CUDA C++ and in three Python flavors (Numba, PyCUDA, CuPy) — kept **separate from the lecture folders** so you always have a working reference copy to run, copy, or diff against your own code.

> ⚠️ **Note on the Google Drive source:** the shared Drive folder (`GPU_Programming`) doesn't expose a browsable file listing to automated tools, so these files are built from the same verified working code already extracted and explained line-by-line in [`Lec_03_VectorAdd_MatrixMul/`](../Lec_03_VectorAdd_MatrixMul/) — confirmed to compile and run (real CPU/GPU timing output included in comments). If your Drive folder has additional or different files, upload them directly and I'll add/reconcile them here.

---

## 📁 Structure

```
Basic/
├── README.md
├── CODE_WALKTHROUGH.md        <- every single line of every file below, explained
├── cpp/
│   ├── 01_vector_add.cu       <- CUDA C++, 1D indexing, cudaEvent timing
│   └── 02_matrix_mul.cu       <- CUDA C++, 2D indexing, 5x5 human-checkable
└── python/
    ├── 01_vector_add_numba.py     <- Numba @cuda.jit, gentlest Python entry point
    ├── 02_matrix_mul_pycuda.py    <- PyCUDA, raw kernel string + Python driver
    └── 03_matrix_mul_cupy.py      <- CuPy, NumPy-style, kernel fully hidden
```

## 🧭 How to use

1. **New to CUDA?** Start with `python/01_vector_add_numba.py` — no compiler step, gentlest syntax.
2. **Want every line explained, newbie-friendly?** Read [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md) side-by-side with any file — it covers all 5 programs, line by line, no prior CUDA knowledge assumed.
3. **Want to see the real C++ underneath?** `cpp/01_vector_add.cu` → `cpp/02_matrix_mul.cu`, in that order.
4. **Want to see three levels of abstraction over the SAME matmul problem?** Run `cpp/02_matrix_mul.cu` (full control) → `python/02_matrix_mul_pycuda.py` (Python-driven, C++ kernel) → `python/03_matrix_mul_cupy.py` (kernel fully hidden) back to back.
5. Every file can run standalone on Colab (`Runtime → GPU → T4`) or RunPod — see [`00_Environment_Setup_RunPod_Colab/`](../00_Environment_Setup_RunPod_Colab/README.md) if you haven't set that up yet.
6. For the same explanations organized by CONCEPT rather than by FILE, see [`Lec_03_VectorAdd_MatrixMul/cpp/`](../Lec_03_VectorAdd_MatrixMul/cpp/gpu_lec03_cpp_code.md) and [`Lec_03_VectorAdd_MatrixMul/python/`](../Lec_03_VectorAdd_MatrixMul/python/gpu_lec03_python_code.md).

---

> *GPU Programming · Basic · github.com/rpaut03l/TS-02-03*
