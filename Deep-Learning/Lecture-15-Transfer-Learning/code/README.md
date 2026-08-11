# DL Lecture 15 (Bonus) — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-15-bonus--code)`

> Folder: `Deep-Learning/Lecture-15-Transfer-Learning/code/`
> ⚠️ Bonus lecture — see the theory file's header note.

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture15_transfer_learning_from_scratch.py` | Implements parameter-counting and learning-rate-scaling formulas (checked against the numerical README), then runs a real, working transfer learning experiment: pretrain a small network on a SOURCE task, then adapt it to a related TARGET task via Feature Extraction, Fine-Tuning, and a from-scratch baseline. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function/class | Theory/Numerical concept |
|---|---|
| `head_params()` | `(input_dim × num_classes) + num_classes` |
| `layerwise_lr_schedule()` | `η_k = η_1 × multiplier^(k-1)` |
| `TwoLayerNet.train_step(lr_backbone=0, ...)` | Feature Extraction (frozen backbone) |
| `TwoLayerNet.train_step(lr_backbone=small, ...)` | Fine-Tuning (gently updated backbone) |
| `run_transfer_learning_demo()` | The full pretrain → adapt → compare experiment |

## How to run this file

```bash
cd Deep-Learning/Lecture-15-Transfer-Learning/code
pip install numpy --break-system-packages
python3 dl_lecture15_transfer_learning_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required for this toy problem.

## An honest result from Check 4 — and what it actually teaches

Checks 1–3 reproduce the numerical README's worked examples exactly. Check 4 is a genuinely trainable experiment, not just a formula check — and its result is worth discussing honestly rather than dressing up: on this particular toy 2D task (classifying points above/below a shifted line), training FROM SCRATCH on just 20 target examples (≈0.947 accuracy) slightly OUTPERFORMED both Feature Extraction (≈0.920) and Fine-Tuning (≈0.923).

This is not a contradiction of the theory file — it's a genuine, useful lesson about WHEN transfer learning's advantage actually shows up. The toy task here is simple (2 input dimensions, a near-linear decision boundary) — even 20 random examples already contain enough signal for gradient descent to find a good boundary REGARDLESS of where it started from. Transfer learning's real, well-documented advantage (illustrated with realistic numbers in Worked Example 3 — 50× less data needed) shows up much more dramatically on genuinely HIGH-DIMENSIONAL, complex problems (real images with thousands of pixels, real text with huge vocabularies) — exactly where random initialization has a great deal more "distance" to cover before finding a good solution, and where a pretrained starting point's head start matters far more. This toy demo is kept deliberately small and simple for hand-traceability, at the honest cost of not being the ideal setting to showcase transfer learning's biggest wins — try increasing `hidden_dim` and the input dimensionality yourself to see the gap start to appear.

## Expected output (verified — produced by actually running this script)

- Checks 1–3 match the numerical README exactly (5,130 head params, 2,281.7× ratio; 0.0001/0.001 fine-tuning rates; the 8× layer-wise rate ratio).
- Check 4: pretrained source-task accuracy ≈0.992; target-task test accuracies of Feature Extraction ≈0.920, Fine-Tuning ≈0.923, and From Scratch ≈0.947 on this specific simple toy task (see the discussion above for what this genuinely does and doesn't demonstrate).

`[← Exercises](../exercises/dl_lecture15_exercises.md) · [🔝 Top](#dl-lecture-15-bonus--code) · [🔝 Lecture Hub](../README.md)`
