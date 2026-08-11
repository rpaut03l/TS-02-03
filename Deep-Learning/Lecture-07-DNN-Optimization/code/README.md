# DL Lecture 07 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-07--code)`

> Folder: `Deep-Learning/Lecture-07-DNN-Optimization/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture07_optimizers_from_scratch.py` | Implements SGD, Momentum, RMSProp, and Adam as small Python classes, verified against every numerical README worked example, then races all four on a simple toy loss function. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code class | Theory/Numerical concept |
|---|---|
| `SGD` | `θ = θ - η·g` |
| `Momentum` | `v = γv + ηg`, `θ = θ - v` |
| `RMSProp` | `E[g²]=βE[g²]+(1-β)g²`, `θ=θ-η/√(E[g²]+ε)·g` |
| `Adam` | Bias-corrected combination of both `m` and `v` |
| `race_optimizers()` | A live, watchable demonstration of convergence behaviour differences |

## A real, honest teaching moment from the race demo

Running `race_optimizers()`, you'll notice **Momentum actually OVERSHOOTS the minimum** (θ=3.0) badly around step 5 (reaching θ≈8.67!) before eventually swinging back — this is not a bug, it's a genuine, important real-world phenomenon directly explained in the theory file: momentum builds up "velocity" from consistent gradients, and if that velocity gets too large relative to how sharply the loss curves, it will sail right past the minimum before correcting, oscillating for a while before settling. This is exactly why real training often pairs momentum with a carefully tuned (and often decaying) learning rate. RMSProp and Adam, with the learning rates chosen here, converge more cautiously but don't overshoot the same way — try changing the learning rates in `race_optimizers()` yourself to see how each optimizer's behaviour changes.

## How to run this file

```bash
cd Deep-Learning/Lecture-07-DNN-Optimization/code
pip install numpy --break-system-packages
python3 dl_lecture07_optimizers_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required.

## Expected output (verified — produced by actually running this script)

- Checks 1–4 reproduce Worked Examples 1–4 EXACTLY (SGD θ=0.6; Momentum θ=0.6 then -0.06; RMSProp θ=0.6838 then 0.8799; Adam θ=0.9 then 0.8017).
- Check 5 shows a 15-step race table, all four optimizers starting at θ=-5.0, all converging toward the true minimum θ=3.0 — with Momentum's dramatic overshoot to θ≈8.67 clearly visible around step 5.

`[← Exercises](../exercises/dl_lecture07_exercises.md) · [🔝 Top](#dl-lecture-07--code) · [🔝 Lecture Hub](../README.md)`
