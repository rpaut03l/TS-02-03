# DL Lecture 11 — Real-World End-to-End Framework (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

> Folder: `Deep-Learning/Lecture-11-Real-World-End-to-End-Framework/numerical/`
> Pairs with: [`theory/dl_lecture11_e2e_framework_theory.md`](../theory/dl_lecture11_e2e_framework_theory.md) · [`practice/dl_lecture11_e2e_framework_practice.md`](../practice/dl_lecture11_e2e_framework_practice.md) · [`exercises/dl_lecture11_exercises.md`](../exercises/dl_lecture11_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Feature Correlation (Pearson r), By Hand](#worked-example-1--feature-correlation-pearson-r-by-hand)
3. [Worked Example 2 — Counting Grid Search Combinations](#worked-example-2--counting-grid-search-combinations)
4. [Worked Example 3 — Ablation Study Deltas](#worked-example-3--ablation-study-deltas)
5. [Worked Example 4 — Bagging's Variance Reduction](#worked-example-4--baggings-variance-reduction)
6. [Worked Example 5 — Boosting Weight Update (AdaBoost-Style)](#worked-example-5--boosting-weight-update-adaboost-style)
7. [Worked Example 6 — Stacking, A Simple Meta-Model](#worked-example-6--stacking-a-simple-meta-model)
8. [Master Formula Cheatsheet](#master-formula-cheatsheet)
9. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
10. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| x, y | a feature and a target variable (used in the correlation example) |
| x̄, ȳ | the mean (average) of x and y |
| r | Pearson correlation coefficient, ranges from -1 to +1 |
| σ² | variance (used in the bagging example) |
| n | number of models in an ensemble, or number of samples |
| e | error rate (used in the boosting example) |
| α (alpha) | a model's "say" or weight in AdaBoost |
| w | a sample's weight (used in the boosting example) |
| w_i | the i-th base model's weight in a stacking meta-model |

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Worked Example 1 — Feature Correlation (Pearson r), By Hand

**Given:** feature x=[1,2,3,4,5] and target y=[2,4,5,4,5].

**Step 1 — Compute the means.**
```
mean_x = (1+2+3+4+5)/5 = 15/5 = 3.0
mean_y = (2+4+5+4+5)/5 = 20/5 = 4.0
```

**Step 2 — Compute the covariance term (sum of products of deviations).**
```
(1-3)(2-4) + (2-3)(4-4) + (3-3)(5-4) + (4-3)(4-4) + (5-3)(5-4)
= (-2)(-2) + (-1)(0) + (0)(1) + (1)(0) + (2)(1)
= 4 + 0 + 0 + 0 + 2
= 6.0
```

**Step 3 — Compute the standard deviation terms (square root of sum of squared deviations).**
```
std_x = sqrt[(1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2] = sqrt[4+1+0+1+4] = sqrt(10) ≈ 3.1623
std_y = sqrt[(2-4)^2+(4-4)^2+(5-4)^2+(4-4)^2+(5-4)^2] = sqrt[4+0+1+0+1] = sqrt(6) ≈ 2.4495
```

**Step 4 — Compute Pearson correlation coefficient r.**
```
r = covariance_term / (std_x x std_y) = 6.0 / (3.1623 x 2.4495) ≈ 6.0/7.746 ≈ 0.7746
```

**Result: r ≈ 0.775**, a fairly strong positive correlation between feature x and target y — this feature would likely be flagged as informative during the "Find Feature Correlation" pipeline step.

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Worked Example 2 — Counting Grid Search Combinations

**Given:** a hyperparameter grid search over 3 learning rates `[0.001, 0.01, 0.1]`, 4 batch sizes `[16,32,64,128]`, and 3 layer-count choices `[2,4,6]`.

**Step 1 — Multiply the number of choices in each dimension.**
```
Total combinations = 3 x 4 x 3 = 36
```

**Result: 36 total hyperparameter configurations** must be trained and evaluated to exhaustively cover this grid — this rapid multiplicative growth (the "combinatorial explosion" of grid search) is exactly why random search or smarter automated methods (Bayesian optimization) become necessary as the number of tunable hyperparameters grows.

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Worked Example 3 — Ablation Study Deltas

**Given:** a full model achieves 91.2% validation accuracy. Three ablated variants are tested: removing dropout → 87.6%; removing data augmentation → 88.1%; removing the attention mechanism → 84.5%.

**Step 1 — Compute the performance DROP (delta) for each ablation.**
```
Delta (no dropout)      = 91.2% - 87.6% = 3.6 percentage points
Delta (no augmentation) = 91.2% - 88.1% = 3.1 percentage points
Delta (no attention)    = 91.2% - 84.5% = 6.7 percentage points
```

**Step 2 — Rank components by importance (larger delta = more important).**
```
1st (most important): attention mechanism (6.7 pp drop)
2nd: dropout (3.6 pp drop)
3rd: data augmentation (3.1 pp drop)
```

**Result:** the attention mechanism contributes the most to this model's performance (removing it costs 6.7 percentage points), while dropout and data augmentation contribute smaller, roughly comparable amounts. This is exactly the kind of quantified, ranked insight an ablation study is designed to produce — telling you WHERE to focus further improvement effort, and confirming that no component is "dead weight."

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Worked Example 4 — Bagging's Variance Reduction

**Given:** a single model's predictions have variance σ²=4.0 (i.e., predictions vary quite a bit from one training run to another). We build a bagged ensemble.

**Step 1 — Apply the variance-of-an-average formula, ASSUMING the models' errors are independent.** For n independent models, each with variance σ², averaging their predictions gives:
```
Variance of the average = sigma^2 / n
```

**Step 2 — Compute for n=5 models.**
```
Variance (n=5) = 4.0 / 5 = 0.8
```

**Step 3 — Compute for n=20 models.**
```
Variance (n=20) = 4.0 / 20 = 0.2
```

**Result:** going from a single model (variance 4.0) to a 5-model bagged ensemble drops variance to 0.8 — a **5× reduction**; a 20-model ensemble drops it to 0.2 — a **20× reduction**. This is the precise mathematical reason bagging works: averaging independent, noisy predictions cancels out much of that noise, though in practice models are rarely perfectly independent (they're trained on overlapping/correlated data), so real-world variance reduction is somewhat less dramatic than this idealized formula suggests.

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Worked Example 5 — Boosting Weight Update (AdaBoost-Style)

**Given:** 5 training samples, all starting with equal weight 0.2 each (summing to 1.0). A weak model is trained, and gets 2 out of 5 samples wrong (error rate e=2/5=0.4).

**Step 1 — Compute the model's "say" (alpha), a standard AdaBoost formula.**
```
alpha = 0.5 x ln((1-e)/e) = 0.5 x ln(0.6/0.4) = 0.5 x ln(1.5) ≈ 0.5 x 0.4055 ≈ 0.2027
```

**Step 2 — Increase weight for misclassified samples, decrease for correctly classified ones.**
```
w_wrong (before normalizing) = 0.2 x e^(alpha) = 0.2 x e^0.2027 ≈ 0.2 x 1.2247 ≈ 0.2449
w_right (before normalizing) = 0.2 x e^(-alpha) = 0.2 x e^(-0.2027) ≈ 0.2 x 0.8165 ≈ 0.1633
```

**Step 3 — Normalize so all weights sum back to 1.0.** Total = (2 wrong x 0.2449) + (3 right x 0.1633) = 0.4899+0.4899 ≈ 0.9798.
```
Normalized w_wrong = 0.2449 / 0.9798 ≈ 0.2500
Normalized w_right = 0.1633 / 0.9798 ≈ 0.1667
```

**Result:** each misclassified sample's weight roughly rose from 0.2 to 0.25, while each correctly-classified sample's weight fell from 0.2 to about 0.167. The NEXT weak model in the boosting sequence will be trained with these updated weights, forcing it to pay MORE attention to the samples the previous model got wrong — exactly the "each new model corrects previous errors" mechanism described in the theory file.

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Worked Example 6 — Stacking, A Simple Meta-Model

**Given:** three base models (a CNN, an LSTM, and a gradient-boosted tree) each produce a prediction for the same input: `CNN=0.72`, `LSTM=0.65`, `GBT=0.81` (all probabilities that some event occurs). A simple LEARNED meta-model has been trained to combine them with weights `w_CNN=0.5, w_LSTM=0.2, w_GBT=0.3` (learned weights, summing to 1.0, NOT equal weights like a plain average).

**Step 1 — Compute the weighted combination.**
```
final_prediction = 0.5x0.72 + 0.2x0.65 + 0.3x0.81
                  = 0.36 + 0.13 + 0.243
                  = 0.733
```

**Step 2 — Compare to a naive plain average (equal weights, 1/3 each) for contrast.**
```
plain_average = (0.72+0.65+0.81)/3 = 2.18/3 ≈ 0.7267
```

**Result:** the meta-model's weighted combination (0.733) differs slightly from the plain average (0.7267) — because the meta-model LEARNED that the gradient-boosted tree's predictions (weight 0.3) and the CNN's (weight 0.5) are somewhat more reliable/informative for this task than the LSTM's (weight 0.2), rather than trusting all three equally. This is exactly what makes stacking more powerful than simple averaging: the meta-model can learn WHICH base models to trust more, potentially even per-input (in more sophisticated stacking setups), rather than blending everything with fixed, equal weight.

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Pearson correlation | `r = Σ(x_i-x̄)(y_i-ȳ) / (std_x · std_y)` |
| Grid search combinations | product of each hyperparameter's choice count |
| Ablation delta | `full_model_score - ablated_model_score` |
| Bagging variance | `σ²/n` (for n independent models) |
| AdaBoost model weight (alpha) | `0.5 · ln((1-e)/e)` |
| AdaBoost sample weight update | `w · e^(±alpha)`, then renormalize to sum to 1 |
| Stacking meta-model | `Σ w_i · prediction_i` (learned weights, not necessarily equal) |

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting to divide by the product of standard deviations when computing Pearson r — the raw covariance term alone is NOT the correlation coefficient; it must be normalized.
- **Trap:** assuming bagging's variance reduction (σ²/n) applies exactly in practice — it assumes fully INDEPENDENT models; real bagged models share overlapping training data and correlated errors, so real-world variance reduction is typically less than the idealized formula predicts.
- **Trap:** forgetting to RENORMALIZE sample weights after an AdaBoost update — the raw updated weights (before normalizing) do NOT sum to 1.0, and must be rescaled.
- **Exam hack:** for ablation study questions, always compute and explicitly RANK the deltas (not just report raw numbers) — ranking is what turns raw numbers into an actionable insight about component importance.
- **Exam hack:** for stacking vs plain averaging questions, always explicitly contrast the two (compute both), and explain that stacking's advantage is LEARNING which base models to trust more, rather than assuming equal weighting — this comparison is a favourite exam/interview question.

`[🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical)`

---

## Summary

This file turned Lecture 11's pipeline and ensembling concepts into fully worked arithmetic. A hand-computed Pearson correlation coefficient between a toy feature and target produced r≈0.775, demonstrating the exact mechanics behind the "Find Feature Correlation" pipeline step. Counting a 3×4×3 hyperparameter grid gave 36 total configurations, illustrating grid search's combinatorial growth. An ablation study example computed and ranked performance deltas for three removed components (attention: 6.7pp, dropout: 3.6pp, augmentation: 3.1pp), showing exactly how ablation studies quantify component importance. Bagging's variance formula (σ²/n) showed a single model's variance of 4.0 dropping to 0.8 with 5 models and 0.2 with 20 models — the precise mathematical reason averaging independent models reduces prediction variance. A full AdaBoost-style weight update walked through computing a model's "alpha" (≈0.203) from a 40% error rate, then updating and renormalizing sample weights so misclassified samples rise from 0.2 to 0.25 and correctly-classified samples fall to about 0.167 — exactly the "focus more on previous mistakes" mechanism underlying boosting. Finally, a stacking example compared a learned meta-model's weighted combination (0.733) against a plain unweighted average (0.7267) of the same three base-model predictions, demonstrating why a trained meta-model can outperform naive averaging by learning which base models deserve more trust. The master formula table consolidates every reusable calculation from this lecture for fast review.

`[← Theory](../theory/dl_lecture11_e2e_framework_theory.md) · [🔝 Top](#dl-lecture-11--real-world-end-to-end-framework-numerical) · [Next: Practice →](../practice/dl_lecture11_e2e_framework_practice.md)`
