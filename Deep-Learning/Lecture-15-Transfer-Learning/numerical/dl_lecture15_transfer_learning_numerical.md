# DL Lecture 15 (Bonus) — Transfer Learning (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

> Folder: `Deep-Learning/Lecture-15-Transfer-Learning/numerical/`
> Pairs with: [`theory/dl_lecture15_transfer_learning_theory.md`](../theory/dl_lecture15_transfer_learning_theory.md) · [`practice/dl_lecture15_transfer_learning_practice.md`](../practice/dl_lecture15_transfer_learning_practice.md) · [`exercises/dl_lecture15_exercises.md`](../exercises/dl_lecture15_exercises.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Trainable Parameters: Feature Extraction vs Fine-Tuning](#worked-example-1--trainable-parameters-feature-extraction-vs-fine-tuning)
3. [Worked Example 2 — Scaling the Fine-Tuning Learning Rate](#worked-example-2--scaling-the-fine-tuning-learning-rate)
4. [Worked Example 3 — Data Efficiency: Transfer Learning vs From Scratch](#worked-example-3--data-efficiency-transfer-learning-vs-from-scratch)
5. [Worked Example 4 — A Layer-Wise Learning Rate Schedule](#worked-example-4--a-layer-wise-learning-rate-schedule)
6. [Master Formula Cheatsheet](#master-formula-cheatsheet)
7. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
8. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| η (eta) | learning rate |
| θ (theta) | a model's parameters (weights) |
| g | a gradient |
| k | a scaling multiplier (e.g. "divide by 10", "triple each block") |

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Worked Example 1 — Trainable Parameters: Feature Extraction vs Fine-Tuning

**Given:** a pretrained CNN backbone with 11,700,000 parameters, producing a 512-dimensional feature vector. A new classification head is a single linear layer mapping 512 features to 10 target classes.

**Step 1 — Compute the new head's parameter count.**
```
head_params = (512 x 10) + 10 (bias)  = 5,120 + 10 = 5,130
```

**Step 2 — Feature Extraction: only the head is trainable.**
```
trainable_FE = 5,130
```

**Step 3 — Partial fine-tuning: unfreeze the backbone's last block (say, 1,200,000 params), plus the head.**
```
trainable_partial = 1,200,000 + 5,130 = 1,205,130
```

**Step 4 — Full fine-tuning: the ENTIRE backbone plus the head are trainable.**
```
trainable_full = 11,700,000 + 5,130 = 11,705,130
```

**Result:** comparing Feature Extraction (5,130 trainable params) to Full Fine-Tuning (11,705,130 trainable params) — a difference of about **2,282×**. This huge gap directly explains why Feature Extraction trains so much faster and needs so much less data: you're optimizing a MUCH smaller number of free parameters, dramatically reducing overfitting risk on a small target dataset.

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Worked Example 2 — Scaling the Fine-Tuning Learning Rate

**Given:** a base learning rate of `η=0.01`, used for training a similar architecture from scratch. Standard fine-tuning practice: scale down by 10× to 100×.

**Step 1 — Compute the "gentle fine-tuning" learning rate (100× smaller).**
```
eta_finetune = 0.01 / 100 = 0.0001
```

**Step 2 — Compute the "moderate fine-tuning" learning rate (10× smaller), sometimes used for less similar domains needing a bit more adaptation.**
```
eta_finetune_moderate = 0.01 / 10 = 0.001
```

**Result: fine-tuning learning rates of 0.0001 (very gentle) to 0.001 (moderate)**, both dramatically smaller than the from-scratch rate of 0.01. Recalling Lecture 7's gradient descent update formula `θ=θ-η·g`, a 100× smaller η means each weight update step is 100× smaller in magnitude for the SAME gradient — exactly the caution needed to avoid catastrophically overwriting valuable pretrained weights in just a few aggressive steps.

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Worked Example 3 — Data Efficiency: Transfer Learning vs From Scratch

**Given:** empirically, a particular from-scratch CNN needs roughly 100,000 labelled training images to reach a target accuracy on a vision task; a transfer-learning approach (fine-tuning an ImageNet-pretrained CNN) reaches the SAME target accuracy with only about 2,000 labelled images.

**Step 1 — Compute the data-efficiency ratio.**
```
ratio = 100,000 / 2,000 = 50
```

**Result: the transfer learning approach needs 50× LESS labelled data** to reach the same accuracy target. This is precisely why transfer learning has been transformative for applied deep learning in data-scarce domains (medical imaging, specialized industrial inspection, niche scientific applications) — collecting and labelling 2,000 images is a vastly more tractable project than collecting and labelling 100,000.

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Worked Example 4 — A Layer-Wise Learning Rate Schedule

**Given:** a 4-block pretrained network (Block 1 = earliest/most generic, Block 4 = latest/most task-specific), fine-tuned with a layer-wise learning rate schedule where each block's rate is DOUBLE the previous (earlier) block's rate, starting from Block 1's rate of `η_1=0.00005`.

**Step 1 — Compute each block's learning rate.**
```
eta_1 (earliest)  = 0.00005
eta_2 = 0.00005 x 2 = 0.0001
eta_3 = 0.0001 x 2  = 0.0002
eta_4 (latest)    = 0.0002 x 2 = 0.0004
```

**Step 2 — Compute the ratio between the latest and earliest block's learning rate.**
```
ratio = eta_4 / eta_1 = 0.0004 / 0.00005 = 8
```

**Result: the latest block's learning rate is 8× larger than the earliest block's.** This numerically implements the "progressive unfreezing" philosophy from the theory file: early, generic layers get the smallest, most cautious updates (preserving their valuable, broadly-useful features), while later, task-specific layers get proportionally larger updates, since they need to adapt the most to the new target task.

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| New head parameter count | `(input_dim × num_classes) + num_classes` (bias) |
| Feature Extraction trainable params | just the new head's parameters |
| Full fine-tuning trainable params | backbone params + head params |
| Typical fine-tuning LR scaling | `η_finetune = η_scratch / (10 to 100)` |
| Data efficiency ratio | `data_needed_from_scratch / data_needed_transfer` |
| Layer-wise LR (doubling per block) | `η_k = η_1 × 2^(k-1)` |

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting to include the bias term when counting a new classifier head's parameters — always add `+num_classes` for the bias vector.
- **Trap:** using the SAME learning rate across all layers during fine-tuning without justification — always be ready to explain WHY layer-wise or globally-reduced learning rates are standard practice (protecting pretrained knowledge, per the theory file).
- **Trap:** treating the "50× less data" or "2,282× fewer trainable parameters" figures as universal constants — these are ILLUSTRATIVE numbers from one worked example, not fixed rules; actual ratios vary hugely by architecture, dataset, and task similarity.
- **Exam hack:** always show BOTH the raw numbers AND the ratio when comparing Feature Extraction vs Fine-Tuning — the ratio (e.g., "2,282× fewer trainable parameters") is usually the more memorable, exam-worthy takeaway.
- **Exam hack:** for learning-rate-scaling questions, always connect back to Lecture 7's `θ=θ-η·g` update formula explicitly — the smaller η directly means smaller, more cautious weight movement per step, which is the actual mechanism protecting pretrained knowledge.

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical)`

---

## Summary

This file turned every transfer learning trade-off from the theory file into fully shown arithmetic. Comparing trainable parameter counts for a pretrained 11.7-million-parameter backbone showed Feature Extraction requiring only 5,130 trainable parameters (just the new head) versus Full Fine-Tuning requiring all 11,705,130 — a 2,282× difference, directly explaining Feature Extraction's speed and data-efficiency advantages. Scaling a base learning rate of 0.01 down by 10× to 100× for fine-tuning produced concrete gentle (0.0001) and moderate (0.001) fine-tuning rates, connected directly back to Lecture 7's weight update formula to explain WHY smaller rates protect pretrained knowledge. A data-efficiency example showed a hypothetical transfer-learning approach needing 50× less labelled data (2,000 vs 100,000 images) to reach the same target accuracy as training from scratch. Finally, a layer-wise learning rate schedule example computed doubling rates across four network blocks (0.00005 → 0.0001 → 0.0002 → 0.0004), giving the latest block an 8× larger learning rate than the earliest — the exact numeric implementation of the progressive-unfreezing philosophy, protecting generic early features while allowing task-specific later features to adapt more freely. The master formula table consolidates every reusable calculation from this bonus lecture for fast review.

`[← Theory](../theory/dl_lecture15_transfer_learning_theory.md) · [🔝 Top](#dl-lecture-15-bonus--transfer-learning-numerical) · [Next: Practice →](../practice/dl_lecture15_transfer_learning_practice.md)`
