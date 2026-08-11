# DL Lecture 08 — Regularization (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-08--regularization-numerical)`

> Folder: `Deep-Learning/Lecture-08-Regularization/numerical/`
> Pairs with: [`theory/dl_lecture08_regularization_theory.md`](../theory/dl_lecture08_regularization_theory.md) · [`practice/dl_lecture08_regularization_practice.md`](../practice/dl_lecture08_regularization_practice.md) · [`exercises/dl_lecture08_exercises.md`](../exercises/dl_lecture08_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Closed-Form L2 Regression, By Hand](#worked-example-1--closed-form-l2-regression-by-hand)
3. [Worked Example 2 — L1 vs L2 Penalty Values](#worked-example-2--l1-vs-l2-penalty-values)
4. [Worked Example 3 — Dropout Forward Pass, By Hand](#worked-example-3--dropout-forward-pass-by-hand)
5. [Worked Example 4 — Dropout at Test Time (Scaling)](#worked-example-4--dropout-at-test-time-scaling)
6. [Worked Example 5 — Counting Thinned Networks](#worked-example-5--counting-thinned-networks)
7. [Worked Example 6 — Finding the Early-Stopping Point](#worked-example-6--finding-the-early-stopping-point)
8. [Master Formula Cheatsheet](#master-formula-cheatsheet)
9. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
10. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| w | weight (scalar, for hand-tractability in this file) |
| λ (lambda) | regularization coefficient |
| p | dropout retention probability |
| n | number of nodes eligible for dropout |

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Worked Example 1 — Closed-Form L2 Regression, By Hand

**Given:** a single-feature (no bias) regression problem: `x = [1, 2, 3]`, `y = [2, 4, 5]`, regularization coefficient λ=2.

**Step 1 — Compute the unregularized closed-form solution.** For single-feature least squares, `w = Σ(x_i·y_i) / Σ(x_i²)`.
```
Σ(x_i . y_i) = 1x2 + 2x4 + 3x5 = 2 + 8 + 15 = 25
Σ(x_i^2) = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
w_unregularized = 25 / 14 ≈ 1.7857
```

**Step 2 — Compute the L2-regularized closed-form solution.** Adding λ to the denominator (the regularized normal equation for this simple 1D case): `w = Σ(x_i·y_i) / (Σ(x_i²) + λ)`.
```
w_regularized = 25 / (14 + 2) = 25 / 16 = 1.5625
```

**Result: w shrinks from ≈1.7857 (unregularized) to exactly 1.5625 (regularized)** — a smaller magnitude weight, exactly the "weight decay" effect described in the theory file. Notice: as λ→0, this formula recovers the unregularized solution; as λ→∞, w→0 (the weight gets crushed toward zero entirely).

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Worked Example 2 — L1 vs L2 Penalty Values

**Given:** a weight vector `w = [3, -1, 0.5, -4]`.

**Step 1 — Compute the L1 penalty (sum of absolute values).**
```
L1 = |3| + |-1| + |0.5| + |-4| = 3 + 1 + 0.5 + 4 = 8.5
```

**Step 2 — Compute the L2 penalty (half the sum of squares, per the theory file's convention).**
```
L2 = 0.5 x (3^2 + (-1)^2 + 0.5^2 + (-4)^2)
   = 0.5 x (9 + 1 + 0.25 + 16)
   = 0.5 x 26.25
   = 13.125
```

**Result: L1 penalty = 8.5, L2 penalty = 13.125.** Notice the large weight (-4) contributes disproportionately more to L2 (via squaring: 16, more than half the total L2 penalty) than to L1 (contributing just 4, less than half the total L1 penalty) — this numerically illustrates why L2 "more heavily impacts larger values" as stated in the theory file: squaring amplifies large weights' contribution much more than the linear absolute-value penalty does.

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Worked Example 3 — Dropout Forward Pass, By Hand

**Given:** a layer's pre-dropout activations `a = [1.0, 2.0, 3.0, 4.0]`, and a sampled Bernoulli dropout mask (with retention probability p=0.5) `mask = [1, 0, 1, 1]` (this particular random draw: neurons 1, 3, 4 survive; neuron 2 is dropped).

**Step 1 — Apply the mask elementwise (this IS the dropout operation).**
```
output = mask * a = [1x1.0, 0x2.0, 1x3.0, 1x4.0] = [1.0, 0.0, 3.0, 4.0]
```

**Result: output = [1.0, 0.0, 3.0, 4.0].** Neuron 2's entire contribution is completely zeroed out for this training step — its weight will NOT receive any gradient update from this particular batch (per the theory file's "only active neurons' weights are updated" rule), while neurons 1, 3, and 4 train normally this round.

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Worked Example 4 — Dropout at Test Time (Scaling)

**Given:** the SAME layer, now at test time, with the FULL (undropped) network active: `a = [1.0, 2.0, 3.0, 4.0]`, retention probability p=0.5 (the same value used during training).

**Step 1 — Scale every activation by p (no masking — every neuron is active at test time).**
```
output_test = p * a = 0.5 x [1.0, 2.0, 3.0, 4.0] = [0.5, 1.0, 1.5, 2.0]
```

**Result: output_test = [0.5, 1.0, 1.5, 2.0].** This scaling compensates for the fact that DURING training, on average only half of the neurons (p=0.5) contributed to any given forward pass, so downstream layers effectively learned to expect roughly half-strength combined signal — scaling by p at test time (when ALL neurons are active) keeps the expected magnitude of the layer's output consistent between training and testing.

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Worked Example 5 — Counting Thinned Networks

**Given:** a hidden layer with n=10 nodes eligible for dropout.

**Step 1 — Apply the formula: total thinned networks = 2ⁿ.**
```
2^10 = 1024
```

**Result: 1,024 distinct thinned network configurations are possible**, just from this one 10-node layer — each of the 10 nodes can independently be present or dropped, giving 2 choices per node, multiplied across all 10 nodes (2×2×2×2×2×2×2×2×2×2 = 1024). For a more realistic layer with n=100 nodes, this becomes `2^100` — an astronomically large number (over 10³⁰), numerically illustrating exactly why "training each thinned network separately is computationally infeasible," as stated in the theory file.

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Worked Example 6 — Finding the Early-Stopping Point

**Given:** validation loss recorded at the end of each training epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Validation loss | 0.90 | 0.70 | 0.50 | 0.42 | 0.40 | 0.41 | 0.45 | 0.50 |

**Step 1 — Scan for the minimum validation loss.** Comparing all 8 values: 0.90, 0.70, 0.50, 0.42, **0.40**, 0.41, 0.45, 0.50 — the minimum is **0.40, at epoch 5**.

**Step 2 — Confirm the trend reverses after this point.** Epochs 6, 7, 8 show INCREASING validation loss (0.41 → 0.45 → 0.50) — the classic overfitting signature described in the theory file.

**Result: the early-stopping point is epoch 5.** Training should be halted here (or the epoch-5 model checkpoint restored as the final model), even though training could technically continue further — continuing past epoch 5 only makes the model fit the training data more while actively getting WORSE on held-out validation data.

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Total regularized loss | `E(w) = E_D(w) + (λ/2)w^Tw` |
| L2 (1D) closed-form solution | `w = Σ(x_i·y_i) / (Σ(x_i²) + λ)` |
| L1 penalty | `Σ|w_j|` |
| L2 penalty | `(1/2)Σw_j²` |
| General Lq penalty | `(1/2)Σ|w_j|^q` |
| Dropout forward pass | `output = mask ⊙ activation`, `mask ~ Bernoulli(p)` |
| Dropout test-time scaling | `output_test = p × activation` |
| Number of thinned networks | `2ⁿ` |
| Early stopping | epoch minimizing validation loss |

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting that λ appears ADDED to the denominator (not multiplied) in the simple 1D closed-form solution — always double check the formula's exact structure before plugging in numbers.
- **Trap:** computing L1 and L2 penalties with the wrong exponent/operation — L1 uses absolute value (`|w_j|`, degree 1), L2 uses squares (`w_j²`, degree 2, often with a leading 1/2 factor for a cleaner derivative).
- **Trap:** forgetting the test-time scaling step for dropout, or scaling by the WRONG probability (some implementations use "inverted dropout," scaling by 1/p DURING training instead of by p at test time — always check which convention a given question/framework uses).
- **Exam hack:** for early-stopping questions, always explicitly scan and STATE the full sequence of validation losses, not just announce the answer — showing the comparison is what earns marks.
- **Exam hack:** the `2ⁿ` thinned-network count is a very testable exact formula — practice computing it for several different small n values (e.g., n=5,10,20) to build fluency with how fast it grows.

`[🔝 Top](#dl-lecture-08--regularization-numerical)`

---

## Summary

This file worked every regularization formula from the theory file into fully shown arithmetic. A simple 1D closed-form L2 regression example showed a weight shrinking from ≈1.7857 (unregularized) to exactly 1.5625 (regularized with λ=2) — concretely demonstrating "weight decay." Comparing L1 and L2 penalties on the same weight vector `[3,-1,0.5,-4]` produced an L1 penalty of 8.5 and an L2 penalty of 13.125, numerically showing how squaring in L2 disproportionately amplifies the contribution of the largest weight. A hand-computed dropout forward pass applied a Bernoulli mask `[1,0,1,1]` to activations `[1,2,3,4]`, zeroing out neuron 2 entirely; the corresponding test-time example showed the SAME activations instead scaled by p=0.5 to `[0.5,1,1.5,2]`, illustrating dropout's train/test consistency trick. Counting thinned networks for n=10 nodes gave exactly 1,024 possible configurations, with a note that realistic layer sizes (n=100) produce astronomically large counts, justifying why training each configuration separately is infeasible. Finally, a full early-stopping example scanned an 8-epoch validation loss sequence and correctly identified epoch 5 (loss=0.40) as the minimum, with epochs 6–8 clearly showing the overfitting-signature loss increase that early stopping is designed to avoid training past. The master formula table consolidates every reusable calculation from this lecture for fast review.

`[← Theory](../theory/dl_lecture08_regularization_theory.md) · [🔝 Top](#dl-lecture-08--regularization-numerical) · [Next: Practice →](../practice/dl_lecture08_regularization_practice.md)`
