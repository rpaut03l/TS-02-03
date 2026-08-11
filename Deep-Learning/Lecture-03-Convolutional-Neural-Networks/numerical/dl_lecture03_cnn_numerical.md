# DL Lecture 03 — Convolutional Neural Networks (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

> Folder: `Deep-Learning/Lecture-03-Convolutional-Neural-Networks/numerical/`
> Pairs with: [`theory/dl_lecture03_cnn_theory.md`](../theory/dl_lecture03_cnn_theory.md) · [`practice/dl_lecture03_cnn_practice.md`](../practice/dl_lecture03_cnn_practice.md) · [`exercises/dl_lecture03_exercises.md`](../exercises/dl_lecture03_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — FC vs Locally-Connected vs Convolutional (200×200×3)](#worked-example-1--fc-vs-locally-connected-vs-convolutional-200x200x3)
3. [Worked Example 2 — Output Volume Formula, From Scratch](#worked-example-2--output-volume-formula-from-scratch)
4. [Worked Example 3 — The 64×64×3 In-Class Example](#worked-example-3--the-64x64x3-in-class-example)
5. [Worked Example 4 — AlexNet CONV1, Fully Solved](#worked-example-4--alexnet-conv1-fully-solved)
6. [Worked Example 5 — Batch Normalization By Hand](#worked-example-5--batch-normalization-by-hand)
7. [Worked Example 6 — Max Pooling By Hand](#worked-example-6--max-pooling-by-hand)
8. [Master Formula Cheatsheet](#master-formula-cheatsheet)
9. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
10. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| W1, H1, D1 | input volume's width, height, depth (channels) |
| F | filter's spatial size (F×F) |
| K | number of filters (= number of output feature maps) |
| S | stride |
| P | zero-padding amount |
| W2, H2, D2 | output volume's width, height, depth |
| μ (mu), σ² | mean and variance (used in Batch Normalization) |
| γ, β | BatchNorm's learnable scale and shift |

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Worked Example 1 — FC vs Locally-Connected vs Convolutional (200×200×3)

**Given:** input image 200×200×3, and a hidden layer producing 120,000 hidden units (this matches a 200×200 spatial grid × ~3 feature maps, i.e. 200×200×3=120,000 — check: 200×200=40,000, ×3=120,000 ✓).

**Case A — Fully Connected layer.**
```
Inputs  = 200 x 200 x 3 = 120,000
Hidden units = 120,000
Params = 120,000 x 120,000 = 14,400,000,000 = 14.4 billion
```

**Case B — Locally Connected layer (local connectivity, but NO parameter sharing — every location gets its OWN independent 3×3×3 filter).**
```
Params per hidden unit = 3 x 3 x 3 = 27   (local patch, full depth)
Total hidden units = 120,000
Total params = 27 x 120,000 = 3,240,000 ≈ 3.2 million
```

**Case C — Convolutional layer (local connectivity AND parameter sharing — one 3×3×3 filter reused everywhere, per feature map).**
```
Params per filter = 27 (+1 bias = 28)
Number of feature maps, say K=3 (matching the 3 in 120,000 = 200x200x3)
Total params = 28 x 3 = 84
```

**Side-by-side:**

| Layer type | Parameters |
|---|---|
| Fully Connected | 14,400,000,000 |
| Locally Connected (no sharing) | 3,240,000 |
| Convolutional (sharing) | 84 |

**Take-away:** going from FC to local-only connectivity already saves a **4,444×** reduction; adding parameter sharing on top saves a further **~38,571×** reduction — over **171 million times fewer** parameters than the naive fully connected approach, for the exact same 120,000 hidden units.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Worked Example 2 — Output Volume Formula, From Scratch

**Given:** input volume 32×32×3, filter size F=5, K=10 filters, stride S=1, no padding (P=0).

**Step 1 — Apply the width formula.**
```
W2 = (W1 - F)/S + 1 = (32 - 5)/1 + 1 = 27/1 + 1 = 27 + 1 = 28
```

**Step 2 — Height is identical (square input, square filter).**
```
H2 = 28
```

**Step 3 — Depth equals number of filters.**
```
D2 = K = 10
```

**Result: output volume = 28×28×10.** This matches the lecture's own "Access Convolutional Challenge" example: repeatedly applying 5×5 convolutions on a 32×32 input shrinks it 32→28→24→..., illustrating exactly why padding becomes necessary if you want to preserve spatial size across many layers.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Worked Example 3 — The 64×64×3 In-Class Example

**Given (exact in-class question):** input volume 64×64×3, 20 filters of size 3×3, stride S=1, padding P=1.

**Step 1 — Apply the padded output volume formula.**
```
W2 = (W1 - F + 2P)/S + 1 = (64 - 3 + 2x1)/1 + 1 = (64 - 3 + 2)/1 + 1 = 63/1 + 1 = 63 + 1 = 64
```

**Step 2 — Height, same as width (square input/filter).**
```
H2 = 64
```

**Step 3 — Depth equals number of filters.**
```
D2 = K = 20
```

**Result: output volume = 64×64×20.** Notice the spatial size (64×64) is *exactly preserved* from input to output — this is what padding=1 with a 3×3 filter and stride 1 is specifically designed to do ("same" padding), and is one of the most common conv-layer configurations used in real architectures.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Worked Example 4 — AlexNet CONV1, Fully Solved

**Given:** input 227×227×3, CONV1 = 96 filters of size 11×11, stride S=4, no padding.

**Step 1 — Output width.**
```
W2 = (227 - 11)/4 + 1 = 216/4 + 1 = 54 + 1 = 55
```

**Step 2 — Output height (identical, square input/filter).**
```
H2 = 55
```

**Step 3 — Output depth.**
```
D2 = K = 96
```

**Result: output volume = 55×55×96.**

**Step 4 — Parameters per filter (spans full input depth, 3 channels).**
```
Weights per filter = 11 x 11 x 3 = 363
+ 1 bias = 364 parameters per filter
```

**Step 5 — Total parameters in CONV1.**
```
Total = 364 x 96 = 34,944 ≈ 35,000 (35K)
```

**Result: CONV1 alone has 34,944 parameters** — tiny compared to AlexNet's ~60 million total, because most of AlexNet's parameters actually live in its later fully-connected layers, not its convolutional layers (a very common, exam-favourite "which layer has more parameters" surprise — conv layers are parameter-cheap; FC layers are parameter-expensive).

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Worked Example 5 — Batch Normalization By Hand

**Given:** a mini-batch of 4 activation values for one feature: `x = [2, 4, 4, 8]`. Use ε (epsilon) = 0 for simplicity, and learnable parameters γ=2, β=1.

**Step 1 — Compute the mean.**
```
mean = (2 + 4 + 4 + 8) / 4 = 18 / 4 = 4.5
```

**Step 2 — Compute the variance.**
```
variance = [(2-4.5)^2 + (4-4.5)^2 + (4-4.5)^2 + (8-4.5)^2] / 4
         = [6.25 + 0.25 + 0.25 + 12.25] / 4
         = 19.0 / 4
         = 4.75
```

**Step 3 — Normalize each value: x̂ = (x − mean) / sqrt(variance).**
```
x̂1 = (2 - 4.5) / sqrt(4.75) = -2.5 / 2.179 ≈ -1.147
x̂2 = (4 - 4.5) / sqrt(4.75) = -0.5 / 2.179 ≈ -0.229
x̂3 = (4 - 4.5) / sqrt(4.75) = -0.5 / 2.179 ≈ -0.229
x̂4 = (8 - 4.5) / sqrt(4.75) =  3.5 / 2.179 ≈  1.606
```

**Step 4 — Apply the learnable scale and shift: y = γ·x̂ + β.**
```
y1 = 2 x (-1.147) + 1 = -2.294 + 1 = -1.294
y2 = 2 x (-0.229) + 1 = -0.458 + 1 =  0.542
y3 = 2 x (-0.229) + 1 = -0.458 + 1 =  0.542
y4 = 2 x (1.606) + 1  =  3.212 + 1 =  4.212
```

**Result:** the original batch `[2, 4, 4, 8]` becomes `[-1.294, 0.542, 0.542, 4.212]` after BatchNorm — notice the *shape* of the relative differences is preserved (the two middle 4's are still tied and still the two smallest-magnitude-change values), but the scale and center have been deliberately reshaped by the learned γ and β.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Worked Example 6 — Max Pooling By Hand

**Given:** a 4×4 feature map, pooling with a 2×2 window and stride 2 (the standard "non-overlapping" max pool configuration):
```
Input feature map:
[ 1  3  2  4 ]
[ 5  6  1  2 ]
[ 8  2  9  0 ]
[ 3  7  4  5 ]
```

**Step 1 — Top-left 2×2 window: [[1,3],[5,6]].** Max = **6**.

**Step 2 — Top-right 2×2 window: [[2,4],[1,2]].** Max = **4**.

**Step 3 — Bottom-left 2×2 window: [[8,2],[3,7]].** Max = **8**.

**Step 4 — Bottom-right 2×2 window: [[9,0],[4,5]].** Max = **9**.

**Result — output 2×2 feature map:**
```
[ 6  4 ]
[ 8  9 ]
```

Notice the output is exactly a quarter the size of the input (4×4=16 values down to 2×2=4 values) — pooling has thrown away 12 of the 16 original numbers, keeping only the strongest signal from each local window, exactly as described in the theory file.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Output width/height (no padding) | `W2 = (W1 - F)/S + 1` |
| Output width/height (with padding) | `W2 = (W1 - F + 2P)/S + 1` |
| Output depth | `D2 = K` |
| Params per conv filter | `F × F × Cin + 1` |
| Total conv layer params | `(F×F×Cin + 1) × K` |
| FC layer params (compare) | `inputs × hidden units` |
| BatchNorm normalize | `x̂ = (x − μ) / √(σ² + ε)` |
| BatchNorm scale/shift | `y = γ x̂ + β` |
| Max pooling | keep the single largest value per window |

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** using the no-padding formula when padding is actually given — always check whether `P` is stated before picking your formula.
- **Trap:** forgetting `+2P` (padding is added on BOTH sides of each spatial dimension, hence the factor of 2) — a very common off-by-one-ish mistake.
- **Trap:** computing variance using the wrong denominator — the lecture's BatchNorm convention divides by the batch size N (not N−1); using N−1 (sample variance) is a different convention seen elsewhere and will give a slightly different, "wrong for this exam" answer.
- **Exam hack:** always compute output volume in the order Width → Height → Depth, and never forget Depth — `D2 = K` is the single most commonly *forgotten* part of an otherwise-correct answer.
- **Exam hack:** for AlexNet-style parameter questions, always separate "parameters per filter" from "total parameters in the layer" as two explicit steps — graders reward the intermediate 364-per-filter number, not just the final 34,944.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical)`

---

## Summary

This file worked every CNN formula from the theory file down to fully shown arithmetic. Comparing Fully Connected, Locally Connected, and Convolutional layers on the same 200×200×3 image with 120,000 hidden units showed the dramatic savings chain: 14.4 billion parameters (FC) down to 3.24 million (local connectivity alone) down to just 84 (adding parameter sharing) — over 171 million times fewer parameters overall. The output volume formula `W2=(W1-F)/S+1` (or `(W1-F+2P)/S+1` with padding) was derived and applied to three separate scenarios: a plain 32×32×3 input shrinking to 28×28×10 without padding, the in-class 64×64×3 example with padding=1 exactly preserving spatial size at 64×64×20, and AlexNet's real CONV1 layer producing a 55×55×96 output volume with 34,944 (≈35K) parameters — a number worth memorizing given how frequently it reappears in exams. A fully hand-computed Batch Normalization example walked through mean, variance, normalization, and the learnable γ/β scale-and-shift on a 4-value batch, and a hand-computed 2×2 max pooling example on a 4×4 feature map showed exactly how pooling shrinks a representation to a quarter of its size while keeping only the strongest local signal. The master formula table and trap-watch list consolidate every reusable calculation from this lecture for fast pre-exam review.

`[← Theory](../theory/dl_lecture03_cnn_theory.md) · [🔝 Top](#dl-lecture-03--convolutional-neural-networks-numerical) · [Next: Practice →](../practice/dl_lecture03_cnn_practice.md)`
