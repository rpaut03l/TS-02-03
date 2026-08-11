# DL Lecture 11 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-11--code)`

> Folder: `Deep-Learning/Lecture-11-Real-World-End-to-End-Framework/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture11_pipeline_tools.py` | Six small, reusable tools implementing this lecture's pipeline math: Pearson correlation, grid search combination counting, ablation delta ranking, bagging variance reduction, an AdaBoost weight update, and stacking meta-model combination — every result checked against the numerical README. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `pearson_correlation()` | Step 2 of the pipeline — "Find Feature Correlation" |
| `count_grid_combinations()` | Step 3 — hyperparameter search space size |
| `rank_ablation_deltas()` | Step 5 — Model Ablation Study |
| `bagged_variance()` | Bagging's `σ²/n` variance reduction |
| `adaboost_weight_update()` | Boosting's sequential error-correction mechanism |
| `stacking_combine()` | Stacking's learned meta-model combination |

## How to run this file

```bash
cd Deep-Learning/Lecture-11-Real-World-End-to-End-Framework/code
pip install numpy --break-system-packages
python3 dl_lecture11_pipeline_tools.py
```
Works identically on Google Colab or Kaggle — no GPU required, these are small utility functions useful at any stage of a real project, not model-training code itself.

## Try it yourself

`rank_ablation_deltas()` and `pearson_correlation()` are genuinely reusable — try plugging in your own Titanic-dataset features (from Lecture 2's practice problem) into `pearson_correlation()` to see which features correlate most with survival, before building your model.

## Expected output (verified — produced by actually running this script)

- Check 1: Pearson r = 0.7746 — matches Worked Example 1 exactly.
- Check 2: 36 grid search combinations — matches Worked Example 2 exactly.
- Check 3: ablation ranking (attention 6.7 > dropout 3.6 > augmentation 3.1) — matches Worked Example 3 exactly.
- Check 4: bagged variance 4.0 → 0.8 (n=5) → 0.2 (n=20) — matches Worked Example 4 exactly.
- Check 5: AdaBoost alpha=0.2027, updated weights [0.1667×3, 0.25×2] — matches Worked Example 5 exactly.
- Check 6: stacked prediction 0.733 vs plain average 0.7267 — matches Worked Example 6 exactly.

`[← Exercises](../exercises/dl_lecture11_exercises.md) · [🔝 Top](#dl-lecture-11--code) · [🔝 Lecture Hub](../README.md)`
