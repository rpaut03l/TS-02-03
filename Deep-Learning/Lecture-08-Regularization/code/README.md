# DL Lecture 08 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-08--code)`

> Folder: `Deep-Learning/Lecture-08-Regularization/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture08_regularization_from_scratch.py` | Implements closed-form L2-regularized regression, L1/L2 penalty computation, dropout (train + test), and an early-stopping epoch finder — every result checked against the numerical README. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `l2_regularized_weight_1d()` | `w = Σ(x_i·y_i)/(Σ(x_i²)+λ)` |
| `l1_penalty()` / `l2_penalty()` | `Σ|w_j|` and `0.5·Σw_j²` |
| `dropout_train()` | `output = mask ⊙ activation`, `mask ~ Bernoulli(p)` |
| `dropout_test()` | `output_test = p × activation` |
| `find_early_stopping_epoch()` | Scans validation loss for its minimum |

## How to run this file

```bash
cd Deep-Learning/Lecture-08-Regularization/code
pip install numpy --break-system-packages
python3 dl_lecture08_regularization_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required.

## Expected output (verified — produced by actually running this script)

- Check 1: w shrinks from 1.7857 (unregularized) to exactly 1.5625 (λ=2) — matches Worked Example 1.
- Check 2: L1=8.5, L2=13.125 — matches Worked Example 2 exactly. The bonus comparison confirms L2 penalizes a concentrated weight `[4,0,0]` (penalty 8.0) far more than a spread-out equivalent `[2,1,1]` (penalty 3.0), despite equal L1 totals.
- Check 3: training output `[1,0,3,4]` and test output `[0.5,1,1.5,2]` — match Worked Examples 3 and 4 exactly; the 1,024 thinned-network count for n=10 also matches Worked Example 5.
- Check 4: correctly identifies epoch 5 (loss 0.40) as the early-stopping point — matches Worked Example 6.

`[← Exercises](../exercises/dl_lecture08_exercises.md) · [🔝 Top](#dl-lecture-08--code) · [🔝 Lecture Hub](../README.md)`
