# DL Lecture 03 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-03--code)`

> Folder: `Deep-Learning/Lecture-03-Convolutional-Neural-Networks/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture03_conv_pool_batchnorm.py` | Implements convolution, max pooling, and batch normalization from scratch in NumPy, and checks every result against the numerical README's worked examples. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `conv_output_size()` | `W2 = (W1-F+2P)/S + 1` |
| `convolve2d()` | The "cookie cutter stamped everywhere" sliding-window operation |
| `max_pool2d()` | Pooling — keeps the largest value per window |
| `batch_norm()` | `x̂ = (x−μ)/√(σ²+ε)`, `y = γx̂ + β` |
| Check 6 (vertical edge kernel) | Why CNN filters act like edge detectors — a `[[1,0,-1],[1,0,-1],[1,0,-1]]` kernel is a classic vertical-edge (Sobel-like) detector; run it and watch it fire strongly exactly where the toy image transitions from bright to dark |

## How to run this file

```bash
cd Deep-Learning/Lecture-03-Convolutional-Neural-Networks/code
pip install numpy --break-system-packages
python3 dl_lecture03_conv_pool_batchnorm.py
```
Works identically on Google Colab (`!python dl_lecture03_conv_pool_batchnorm.py`) or Kaggle — no GPU required, this is small pure-CPU code meant for understanding, not production speed. Real convolution implementations (PyTorch's `nn.Conv2d`, etc.) use highly optimized, vectorized/GPU-accelerated code instead of the explicit nested loops used here — the loops here are intentional, so every step stays visible and traceable to the formulas.

## Expected output (verified — every "expected" value below was produced by actually running this script)

- Output volume checks: 28×28, 64×64 (same padding), and 55×55 (AlexNet CONV1) all match the numerical file exactly.
- AlexNet CONV1 parameter count: exactly **34,944**, matching Worked Example 4.
- Max pooling on the 4×4 example: exactly `[[6,4],[8,9]]`, matching Worked Example 6.
- Batch Normalization on `[2,4,4,8]`: exactly `[-1.294, 0.541, 0.541, 4.212]`, matching Worked Example 5.
- The toy vertical-edge convolution clearly lights up (30) exactly at the bright-to-dark boundary column and stays 0 in flat regions — a live demonstration of "why CNN filters act like edge detectors" from the theory file.

`[← Exercises](../exercises/dl_lecture03_exercises.md) · [🔝 Top](#dl-lecture-03--code) · [🔝 Lecture Hub](../README.md)`
