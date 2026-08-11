# DL Lecture 03 — Exercise Bank (Convolutional Neural Networks)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-03--exercise-bank-convolutional-neural-networks)`

> Folder: `Deep-Learning/Lecture-03-Convolutional-Neural-Networks/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture03_cnn_theory.md) · [numerical](../numerical/dl_lecture03_cnn_numerical.md) · [practice](../practice/dl_lecture03_cnn_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q3.1.** Name the four "original" big ideas of CNNs listed in the theory file.

**Q3.2.** Write the output volume formula for width, without padding.

**Q3.3.** What does BatchNorm's γ and β do?

**Q3.4.** Name the three pooling types mentioned in the lecture.

**Q3.5.** How many parameters does AlexNet have in total (approximately)?

---

## 🟡 Medium — Applied Reasoning

**Q3.6.** For an input volume of 28×28×3, a filter size of 5, stride 1, and no padding, compute the output width and height.

**Q3.7.** Explain why a Locally Connected layer (no parameter sharing) still needs far fewer parameters than a Fully Connected layer, even though neither shares weights across positions.

**Q3.8.** A network trained with Batch Normalization is deployed to production, where it processes exactly one image at a time. Explain what BatchNorm does differently at this point compared to during training.

**Q3.9.** Explain, using the "infinitely strong prior" framing, why a convolutional network can be seen as a special, restricted case of a fully connected network.

**Q3.10.** If you increase the stride of a convolutional layer from 1 to 2 (keeping filter size and padding fixed), what happens to the output spatial size, and why?

---

## 🔴 Hard — Derivation & Multi-Step

**Q3.11.** For an input volume 128×128×3, using 16 filters of size 5×5, stride 2, padding 2: compute the full output volume (W2, H2, D2) and the total number of parameters in the layer.

**Q3.12.** A 6×6 feature map is max-pooled with a 3×3 window and stride 3 (non-overlapping). Given the feature map below, compute the full 2×2 output.
```
[ 2  1  0  4  6  1 ]
[ 3  5  2  8  2  0 ]
[ 1  0  9  1  3  4 ]
[ 7  6  2  0  1  5 ]
[ 0  4  1  3  9  2 ]
[ 8  2  5  1  0  7 ]
```

**Q3.13.** For a mini-batch of activations `x = [10, 12, 14, 16]` for one feature, with γ=1 and β=0 and ε=0, compute the full Batch Normalization output step by step.

**Q3.14.** Compare the total parameter count of an FC layer vs a Convolutional layer for an input of 100×100×3 producing 50,000 hidden units/feature-map-positions, using a 5×5 filter for the conv case with K=1 filter. Show both calculations and state the ratio.

`[🔝 Top](#dl-lecture-03--exercise-bank-convolutional-neural-networks)`

---

## Answer Key

<details>
<summary>Q3.1 – Q3.5 (Easy)</summary>

- **Q3.1:** Invariance, Local Connectivity, Parameter Sharing, Pooling.
- **Q3.2:** `W2 = (W1 - F)/S + 1`.
- **Q3.3:** They let the network learn the best scale (γ) and offset (β) for each normalized feature, effectively giving the network an "undo button" if strict zero-mean/unit-variance isn't optimal.
- **Q3.4:** Max, Average, L2.
- **Q3.5:** About 60 million parameters.
</details>

<details>
<summary>Q3.6 – Q3.10 (Medium)</summary>

- **Q3.6:** `(28-5)/1+1 = 23/1+1 = 24`. Output width = height = 24 (depth = number of filters, not given here).
- **Q3.7:** A Locally Connected layer still restricts each hidden unit to only look at a small local patch (e.g. 3×3×3=27 weights) instead of the entire image — this locality restriction alone is what saves the parameters versus FC, even without additionally sharing those 27 weights across different spatial locations. Sharing (making it Convolutional) then saves even more, since a fully connected layer connects EVERY output to EVERY input.
- **Q3.8:** During training, BatchNorm computes mean and variance from the current mini-batch of examples. At test/inference time with a single image, there's no "batch" to compute meaningful statistics from, so BatchNorm instead uses a running average of the mean and variance accumulated across many batches during training, applying those fixed stored values instead.
- **Q3.9:** A fully connected network technically has enough flexibility to represent a convolution (it could, in principle, learn weights that happen to be local and shared) — but convolution forces this structure from the start, effectively assigning zero probability to any other configuration of weights. This exactly matches the "infinitely strong prior" definition from the prior-strength table: entropy ≈ 0, zero probability on all non-conforming parameter settings.
- **Q3.10:** The output spatial size roughly halves. Since stride controls how many pixels the filter jumps each step, doubling the stride means the filter visits roughly half as many positions across the image, directly shrinking W2 and H2 (visible in the denominator S of the output volume formula).
</details>

<details>
<summary>Q3.11 – Q3.14 (Hard)</summary>

- **Q3.11:** W2 = (128 - 5 + 2×2)/2 + 1 = (128-5+4)/2+1 = 127/2+1 = 63.5+1 → using integer division, 63+1 = **64** (in practice frameworks floor this division). H2 = **64**. D2 = K = **16**. Params per filter = 5×5×3+1 = 76. Total params = 76×16 = **1,216**.
- **Q3.12:** Top-left window `[[2,1,0],[3,5,2],[1,0,9]]` → max=9. Top-right window `[[4,6,1],[8,2,0],[1,3,4]]` → max=8. Bottom-left window `[[7,6,2],[0,4,1],[8,2,5]]` → max=8. Bottom-right window `[[0,1,5],[3,9,2],[1,0,7]]` → max=9. Output: `[[9,8],[8,9]]`.
- **Q3.13:** mean = (10+12+14+16)/4 = 52/4 = 13. variance = [(10-13)²+(12-13)²+(14-13)²+(16-13)²]/4 = [9+1+1+9]/4 = 20/4 = 5. std = √5 ≈ 2.236. x̂ = [(10-13)/2.236, (12-13)/2.236, (14-13)/2.236, (16-13)/2.236] = [-1.342, -0.447, 0.447, 1.342]. With γ=1, β=0: y = x̂ (unchanged) = **[-1.342, -0.447, 0.447, 1.342]**.
- **Q3.14:** FC: inputs = 100×100×3 = 30,000. hidden units = 50,000. Params = 30,000 × 50,000 = **1,500,000,000 (1.5 billion)**. Conv: params per filter = 5×5×3 = 75 (+1 bias = 76), K=1 filter → **76 total parameters**. Ratio = 1,500,000,000 / 76 ≈ **19.7 million times** fewer parameters for the convolutional layer.
</details>

`[🔝 Top](#dl-lecture-03--exercise-bank-convolutional-neural-networks)`

---

## Summary

This exercise bank drills Lecture 3's convolution, pooling, and normalization formulas across three difficulty tiers. Easy questions cover pure recall — the four foundational CNN ideas, the output volume formula, BatchNorm's γ/β role, pooling types, and AlexNet's total parameter count. Medium questions apply the formulas to new numbers (a 28×28×3 input through a 5×5 filter), and push conceptual reasoning about why locally-connected layers save parameters even without sharing, how BatchNorm's behaviour differs between training and single-image inference, the infinitely-strong-prior framing for convolution, and the effect of increasing stride. Hard questions require full multi-step derivations: a complete padded-output-volume-plus-parameter-count problem on a 128×128×3 input, a hand-computed 6×6 max pooling exercise, a full Batch Normalization walkthrough on a new 4-value batch, and a full FC-vs-Conv parameter comparison on a 100×100×3 image showing a nearly 20-million-times parameter reduction. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture03_cnn_practice.md) · [🔝 Top](#dl-lecture-03--exercise-bank-convolutional-neural-networks) · [Code →](../code/README.md)`
