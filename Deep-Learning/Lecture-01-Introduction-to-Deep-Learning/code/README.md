# DL Lecture 01 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-01--code)`

> Folder: `Deep-Learning/Lecture-01-Introduction-to-Deep-Learning/code/`
> This folder is where every lecture's runnable code lives, separate from the written explanation files, so the repo stays easy to browse.

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture01_param_counter.py` | A pure-Python calculator that reproduces every parameter-count number from the numerical README, and lets you plug in your own image sizes. |

## Why no libraries are needed yet

Lecture 1 is conceptual — it's about *why* CNNs exist, before any actual model gets built. So this first code file is deliberately dependency-free (plain Python, no NumPy/PyTorch/TensorFlow) — it's just arithmetic, verified against the lecture's own numbers. Real model-building code (with libraries) starts appearing from the Neural Networks / CNN lectures onward, and each future `code/` folder will list exactly which libraries it needs and why.

## How to run this file

You have three easy options — pick whichever matches where you already work:

1. **Locally on your MacBook (venv):**
   ```bash
   cd Deep-Learning/Lecture-01-Introduction-to-Deep-Learning/code
   python3 dl_lecture01_param_counter.py
   ```
   No `pip install` needed for this particular file — it's pure Python 3.

2. **Google Colab:** upload `dl_lecture01_param_counter.py`, open a new notebook, and either `%run dl_lecture01_param_counter.py` or just copy-paste the two functions into a cell and call `run_lecture_examples()`.

3. **Kaggle (T4 GPU environment):** works identically to Colab — GPU is not needed for this file since there's no model training yet, but it's fine to run in any Kaggle notebook.

## Expected output

Running the file prints three worked examples, matching the numerical README exactly: the 1-megapixel/1000-neuron example (~1.0 billion parameters), the 256×256×3 in-class question (6,291,488 FC vs 896 conv, ≈7,022× ratio), and a customizable 512×512×3 example (50,331,712 FC vs 4,864 conv, ≈10,348× ratio).

`[← Exercises](../exercises/dl_lecture01_exercises.md) · [🔝 Top](#dl-lecture-01--code) · [🔝 Lecture Hub](../README.md)`
