# DL Lecture 01 — Introduction to Deep Learning (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

> Folder: `Deep-Learning/Lecture-01-Introduction-to-Deep-Learning/numerical/`
> Pairs with: [`theory/dl_lecture01_introduction_theory.md`](../theory/dl_lecture01_introduction_theory.md) · [`practice/dl_lecture01_introduction_practice.md`](../practice/dl_lecture01_introduction_practice.md) · [`exercises/dl_lecture01_exercises.md`](../exercises/dl_lecture01_exercises.md)
> This file is the "show me every single number" companion to the theory file. Every formula below is worked out digit by digit — nothing is skipped.

---

## Table of Contents
1. [Why This File Exists](#why-this-file-exists)
2. [Notation Used in This File](#notation-used-in-this-file)
3. [Worked Example 1 — Fully Connected Layer Parameter Explosion](#worked-example-1--fully-connected-layer-parameter-explosion)
4. [Worked Example 2 — Naive 4D Convolution Weight Tensor](#worked-example-2--naive-4d-convolution-weight-tensor)
5. [Worked Example 3 — After Applying Translation Invariance (Weight Sharing)](#worked-example-3--after-applying-translation-invariance-weight-sharing)
6. [Worked Example 4 — After Applying Locality (The Real Convolution Kernel)](#worked-example-4--after-applying-locality-the-real-convolution-kernel)
7. [Worked Example 5 — The In-Class Question: 256×256×3 Image](#worked-example-5--the-in-class-question-2562563-image)
8. [Master Formula Cheatsheet](#master-formula-cheatsheet)
9. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
10. [Summary](#summary)

---

## Why This File Exists

The theory file explains **why** CNNs exist. This file exists to make sure you can never be caught off-guard by a number in an exam. Every single parameter-count example from Lecture 1 is redone here from absolute scratch, one arithmetic step at a time — like counting on your fingers before you're allowed to use a calculator. If you can reproduce every line of this file on a blank sheet of paper without looking, you have mastered the numerical side of Lecture 1.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| H, W | Image height and width, in pixels |
| C | Number of input channels (1 = grayscale, 3 = RGB) |
| N | Number of hidden units / neurons in a layer |
| K | Filter (kernel) size, e.g. K=3 means a 3×3 filter |
| F | Number of filters in a convolutional layer |
| Δ (delta) | The locality "radius" — how far a kernel is allowed to look, in pixels |
| [H]ᵢ,ⱼ | The output value at row i, column j |
| [X]ₖ,ₗ | The input value at row k, column l |
| Wᵢ,ⱼ,ₖ,ₗ | A single weight in the full 4D weight tensor connecting output (i,j) to input (k,l) |
| [V]ₐ,ᵦ | A single weight inside the small, shared convolution kernel, at relative offset (a,b) |

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Worked Example 1 — Fully Connected Layer Parameter Explosion

**Given:** a 1-megapixel grayscale image feeding into a fully connected hidden layer of 1000 neurons.

**Step 1 — Count the inputs.**
1 megapixel = 1,000,000 pixels = **10⁶ inputs**.

**Step 2 — Count the neurons in the next layer.**
Given directly: **1000 neurons** = 10³ neurons.

**Step 3 — Count the weights.**
In a fully connected layer, *every* input connects to *every* neuron. So:

```
Weights = (number of inputs) x (number of neurons)
        = 10^6 x 10^3
        = 10^9
```

**Step 4 — (Optional, often skipped in slides but good exam habit) Count the biases.**
Each of the 1000 neurons has its own bias term → +1000 parameters. This is negligible next to 10⁹, so the slide rounds it away, but *mention it in an exam answer* to show you know it exists.

**Result: ≈ 10⁹ (one billion) parameters** for just one hidden layer, on a modest 1-megapixel image.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Worked Example 2 — Naive 4D Convolution Weight Tensor

**Given:** a 1000 × 1000 image, and we try to write a convolution-like operation using a *full* 4D weight tensor Wᵢ,ⱼ,ₖ,ₗ (i.e., we have NOT yet applied invariance or locality — this is the "looks like convolution but isn't yet" stage from theory).

**Step 1 — Count output positions.**
Output has one value per pixel: 1000 × 1000 = **10⁶ output positions**.

**Step 2 — Count input positions each output can connect to.**
Since nothing is restricted yet, each output pixel can depend on *any* input pixel: 1000 × 1000 = **10⁶ input positions**.

**Step 3 — Multiply.**
```
Total weights = (output positions) x (input positions)
              = 10^6 x 10^6
              = 10^12
```

**Result: 10¹² (one trillion) parameters.** This is the exact number flagged in the lecture as "computationally infeasible" — and it's 1000× worse than the already-huge MLP example above, because now *every output pixel* gets its *own* private set of weights (no sharing at all).

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Worked Example 3 — After Applying Translation Invariance (Weight Sharing)

**Change:** instead of a separate weight for every (i,j)-to-(k,l) pair, we now force the weight to depend only on the **relative offset** (a,b) = (k−i, l−j) — the *same* small relationship is reused at every position (weight sharing).

**Step 1 — Range of the offset.**
For a 1000×1000 image, the offset (a,b) can still, in principle, range across almost the whole image: a ∈ (−1000, 1000) and b ∈ (−1000, 1000).

**Step 2 — Count distinct offsets.**
```
Number of distinct a values ~ 2 x 1000 = 2000
Number of distinct b values ~ 2 x 1000 = 2000
Total distinct (a,b) pairs  = 2000 x 2000 = 4,000,000 = 4 x 10^6
```

**Result: 4 × 10⁶ parameters.** Compare this to 10¹² from Example 2 — that is a **250,000× reduction**, achieved purely by reusing the same weight at every location instead of learning independent weights per location. This is weight sharing in action — but notice we *still* have not restricted how far the offset is allowed to range, which is where locality comes in next.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Worked Example 4 — After Applying Locality (The Real Convolution Kernel)

**Change:** now force the offset window to be *small*: only keep [V]ₐ,ᵦ for |a| ≤ Δ and |b| ≤ Δ, and set everything outside that window to zero. The lecture states Δ is typically smaller than 10.

**Step 1 — Count allowed values of a.**
From −Δ to +Δ inclusive → **(2Δ + 1)** values. The lecture rounds this to "≈ 2Δ" for a clean order-of-magnitude estimate — we'll show both.

**Step 2 — Count allowed values of b.**
Same as a → **(2Δ + 1)** values, rounded to "≈ 2Δ".

**Step 3 — Multiply for the total kernel size.**
```
Exact:      (2∆+1) x (2∆+1)
Rounded (slide's version): 2∆ x 2∆ = 4∆^2
```

**Step 4 — Plug in a typical Δ, say Δ = 5** (kernel roughly 10-ish pixels wide, matching "Δ smaller than 10" from the slide):
```
4∆^2 = 4 x 5^2 = 4 x 25 = 100 parameters
```

**Result: order of 4Δ² parameters** — for Δ=5, just **100 weights** for the *entire* convolution kernel, versus 10¹² for the naive version and 10⁹ for the plain MLP. This is the punchline of the whole lecture: locality + weight sharing together take you from "computationally infeasible" (10¹²) down to "a handful of numbers" (~100), by combining a 250,000× reduction (sharing) with a further roughly 40,000× reduction (locality, going from 4×10⁶ down to ~100).

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Worked Example 5 — The In-Class Question: 256×256×3 Image

This is the exact homework/in-class question posed in the lecture. Solve both parts fully.

**Setup:** input image is 256 × 256 pixels, with 3 channels (RGB), i.e. H=256, W=256, C=3.

### Part 1 — Fully connected (feedforward) layer with 32 hidden units

**Step 1 — Flatten the input to count total input numbers.**
```
Inputs = H x W x C = 256 x 256 x 3
       = 65,536 x 3
       = 196,608
```

**Step 2 — Multiply by the number of hidden units.**
```
Weights = 196,608 x 32 = 6,291,456
```

**Step 3 — Add biases (one per hidden unit).**
```
Biases = 32
```

**Step 4 — Total.**
```
Total parameters = 6,291,456 + 32 = 6,291,488
```

**Result: ≈ 6.29 million parameters**, just for this one fully connected layer.

### Part 2 — Convolutional layer, 3×3 filter, 32 such filters

**Step 1 — Count weights in a single filter.**
A filter must span the *full depth* of the input (all 3 channels), even though it is only 3×3 spatially:
```
Weights per filter = K x K x C = 3 x 3 x 3 = 27
```

**Step 2 — Add 1 bias per filter.**
```
Parameters per filter = 27 + 1 = 28
```

**Step 3 — Multiply by the number of filters (32).**
```
Total parameters = 28 x 32 = 896
```

**Result: 896 parameters.**

### Side-by-side comparison

| Layer type | Parameters | Ratio vs conv layer |
|---|---|---|
| Fully connected (32 units) | 6,291,488 | ~7,022× more |
| Convolutional (3×3, 32 filters) | 896 | baseline |

**Take-away:** for the exact same input and the exact same "32 output units/filters," the convolutional layer needs about **7,000× fewer parameters** than the fully connected layer — and this ratio only gets *more* extreme as the image gets bigger, because the FC layer's parameter count scales with the *entire* image size (H×W×C), while the conv layer's parameter count only scales with the *filter* size (K×K×C), completely independent of image size.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula | Plug-in example | Result |
|---|---|---|---|
| FC layer parameters | (inputs) × (hidden units) [+ hidden units for bias] | 10⁶ × 10³ | 10⁹ |
| Naive 4D conv (no sharing, no locality) | (H×W)² | (1000×1000)² | 10¹² |
| After weight sharing only | ≈ (2H) × (2W) | 2000 × 2000 | 4×10⁶ |
| After sharing + locality | ≈ 4Δ² | Δ=5 → 4×25 | 100 |
| Conv layer general formula | (K × K × C_in + 1) × F | 3×3×3+1=28, ×32 | 896 |
| FC layer on flattened image | (H×W×C) × N + N | 196,608×32 + 32 | 6,291,488 |

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting to multiply by **C** (input channels) when counting weights in a single conv filter. A "3×3 filter" on an RGB image is actually a **3×3×3** filter — students very often write just 9 instead of 27, losing easy marks.
- **Trap:** forgetting the **+1 bias per filter/neuron**. Rarely changes the leading digit, but examiners give a specific mark for including it.
- **Trap:** mixing up **(2Δ+1)** (exact) with **2Δ** (the slide's rounded shortcut). If a question says "exact," use (2Δ+1); if it says "approximately" or "order of," 4Δ² is fine.
- **Exam hack:** always show the **unit conversion step** explicitly (megapixel → 10⁶, etc.) — graders reward visible work, not just the final number.
- **Exam hack:** memorize the chain **10⁹ → 10¹² → 4×10⁶ → ~100** as the "story of the numbers" for Lecture 1 — if you can recite this chain with *why* each step happens (MLP → naive conv → weight sharing → +locality), you can answer almost any numerical question this lecture could ask.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical)`

---

## Summary

This file turned every parameter-count claim from Lecture 1 into fully shown arithmetic. Starting point: a plain fully connected layer on a 1-megapixel image with 1000 hidden neurons needs about 10⁹ (one billion) weights, simply because every input connects to every neuron. Trying to write a convolution-like operation the "naive" way, with a full 4D weight tensor and zero restrictions, is even worse — about 10¹² (one trillion) parameters for a 1000×1000 image, because every output pixel effectively gets its own private full-image-sized set of weights. Applying **translation invariance** (reusing the same weight at every position instead of a separate one per position — i.e., weight sharing) cuts this down to about 4×10⁶ parameters. Applying **locality** on top of that (restricting the kernel to only look at a small window of radius Δ, typically under 10) shrinks it further to roughly 4Δ² parameters — for Δ=5, just 100 numbers. The in-class worked question made this concrete with real numbers: for a 256×256×3 image, a fully connected layer with 32 hidden units needs 6,291,488 parameters, while a convolutional layer with 3×3 filters and 32 filters needs only 896 parameters — about 7,000× fewer, for equivalent output capacity. The master formula table above is your one-stop reference for every parameter-counting question this lecture can throw at you; the two traps to watch hardest are forgetting to multiply by the number of input channels inside a filter, and forgetting the small but exam-relevant +1 bias term per unit or filter.

`[← Theory](../theory/dl_lecture01_introduction_theory.md) · [🔝 Top](#dl-lecture-01--introduction-to-deep-learning-numerical) · [Next: Practice →](../practice/dl_lecture01_introduction_practice.md)`
