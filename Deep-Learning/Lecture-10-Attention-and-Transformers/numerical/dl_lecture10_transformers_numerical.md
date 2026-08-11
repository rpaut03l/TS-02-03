# DL Lecture 10 — Attention and Transformers (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

> Folder: `Deep-Learning/Lecture-10-Attention-and-Transformers/numerical/`
> Pairs with: [`theory/dl_lecture10_transformers_theory.md`](../theory/dl_lecture10_transformers_theory.md) · [`practice/dl_lecture10_transformers_practice.md`](../practice/dl_lecture10_transformers_practice.md) · [`exercises/dl_lecture10_exercises.md`](../exercises/dl_lecture10_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — ViT Patch Count](#worked-example-1--vit-patch-count)
3. [Worked Example 2 — Residual Connection, By Hand](#worked-example-2--residual-connection-by-hand)
4. [Worked Example 3 — Layer Normalization After a Residual Connection](#worked-example-3--layer-normalization-after-a-residual-connection)
5. [Worked Example 4 — Transformer Scaling Ratios](#worked-example-4--transformer-scaling-ratios)
6. [Worked Example 5 — Counting Matrix Multiplications](#worked-example-5--counting-matrix-multiplications)
7. [Worked Example 6 — Decoder Autoregressive Steps](#worked-example-6--decoder-autoregressive-steps)
8. [Master Formula Cheatsheet](#master-formula-cheatsheet)
9. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
10. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| D | Transformer embedding dimension |
| H | number of attention heads |
| N | context length (max sequence length) |
| L | number of stacked Transformer blocks |
| x | a single token's input vector |
| μ, σ² | mean and variance (used in Layer Normalization) |
| σ | also used generically for "sigma," the Greek letter for standard deviation — context makes clear which is meant |

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Worked Example 1 — ViT Patch Count

**Given:** input image 224×224×3, patch size 16×16.

**Step 1 — Compute how many patches fit along one side.**
```
patches_per_side = 224 / 16 = 14
```

**Step 2 — Compute the total number of patches (a 2D grid of patches).**
```
total_patches = 14 x 14 = 196
```

**Step 3 — Compute the flattened dimension of a single patch.**
```
patch_dim = 16 x 16 x 3 = 768
```

**Result: the image becomes a sequence of exactly 196 tokens, each starting as a 768-dimensional vector** (before the linear projection into the Transformer's embedding dimension D) — this "196 words" sequence length is exactly why the ViT paper is titled "An Image is Worth 16×16 Words": literally, a 224×224 image with 16×16 patches becomes a 196-token "sentence."

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Worked Example 2 — Residual Connection, By Hand

**Given:** input vector `x = [1.0, 2.0, 3.0]`, and a sub-layer's output (before adding the residual) `sublayer_out = [0.5, -0.2, 0.1]`.

**Step 1 — Apply the residual connection formula: output = sublayer_out + x.**
```
output = [0.5+1.0, -0.2+2.0, 0.1+3.0] = [1.5, 1.8, 3.1]
```

**Result: [1.5, 1.8, 3.1].** Notice the original input `x` is still clearly "present" in the output (each component is close to the original x value, just nudged by the sub-layer's small adjustment) — this is exactly the theory file's point: each layer only needs to learn a small RESIDUAL adjustment, not reconstruct the entire representation from scratch.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Worked Example 3 — Layer Normalization After a Residual Connection

**Given:** the residual output from Example 2: `[1.5, 1.8, 3.1]`.

**Step 1 — Compute the mean (across this ONE token's own features — this is what makes it Layer Norm, not Batch Norm).**
```
mean = (1.5 + 1.8 + 3.1) / 3 = 6.4 / 3 ≈ 2.1333
```

**Step 2 — Compute the variance.**
```
variance = [(1.5-2.1333)^2 + (1.8-2.1333)^2 + (3.1-2.1333)^2] / 3
         = [0.4011 + 0.1111 + 0.9344] / 3
         = 1.4467 / 3
         ≈ 0.4822
```

**Step 3 — Normalize (using ε≈0 for simplicity).**
```
normalized = (x - mean) / sqrt(variance)
normalized_1 = (1.5-2.1333)/sqrt(0.4822) ≈ -0.9120
normalized_2 = (1.8-2.1333)/sqrt(0.4822) ≈ -0.4800
normalized_3 = (3.1-2.1333)/sqrt(0.4822) ≈  1.3920
```

**Result: [-0.9120, -0.4800, 1.3920].** Notice this normalization happens ACROSS this single token's own 3 features (mean and variance computed from just these 3 numbers) — completely independent of any other token in the batch or sequence, which is exactly why Layer Norm behaves identically at train and test time, unlike Batch Norm.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Worked Example 4 — Transformer Scaling Ratios

**Given:** Original Transformer = 213M parameters, GPT-2 = 1.5B parameters, GPT-3 = 175B parameters.

**Step 1 — Compute the GPT-2 / Original ratio.**
```
1,500,000,000 / 213,000,000 ≈ 7.04x
```

**Step 2 — Compute the GPT-3 / GPT-2 ratio.**
```
175,000,000,000 / 1,500,000,000 ≈ 116.67x
```

**Step 3 — Compute the overall GPT-3 / Original ratio.**
```
175,000,000,000 / 213,000,000 ≈ 821.6x
```

**Result:** parameters grew about **7×** from Original to GPT-2, then a further **~117×** from GPT-2 to GPT-3 — compounding to roughly **822× total growth** from the Original Transformer to GPT-3, in just three years (2017 to 2020).

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Worked Example 5 — Counting Matrix Multiplications

**Given:** a Transformer with L=12 blocks (matching the Original Transformer's depth), where each block uses 6 matmuls (4 from self-attention, 2 from the MLP).

**Step 1 — Multiply matmuls-per-block by number of blocks.**
```
total_matmuls = 6 x 12 = 72
```

**Result: 72 total matrix multiplications** across the entire 12-block stack — a genuinely small, clean number of large, GPU-friendly operations, which is a big part of why Transformers train so efficiently at scale compared to the many small, strictly-sequential operations required by RNN/LSTM-based architectures processing the same sequence length.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Worked Example 6 — Decoder Autoregressive Steps

**Given:** generating the 4-word sentence "I am very happy" (4 tokens + an end token).

**Step 1 — Count the total generation steps needed.** Each step produces exactly ONE new token, and you need one extra step to produce the final `<end>` token.
```
Steps = (number of output words) + 1 (for <end>) = 4 + 1 = 5
```

**Step 2 — List each step's input and output explicitly, following the lecture's exact pattern.**
```
Step 1: input=<start>                    -> predicts "I"
Step 2: input=<start> I                  -> predicts "am"
Step 3: input=<start> I am               -> predicts "very"
Step 4: input=<start> I am very          -> predicts "happy"
Step 5: input=<start> I am very happy    -> predicts <end>, STOP
```

**Result: 5 total decoding steps.** Notice the input grows by exactly one token every step (this is the "autoregressive" pattern) — a general formula: for an output sequence of length **n** words, decoding requires exactly **n+1** steps (n steps to produce each real word, plus 1 more step to produce the `<end>` token that signals "stop").

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| ViT patches per side | `image_size / patch_size` |
| ViT total patches | `(image_size/patch_size)^2` |
| ViT flattened patch dim | `patch_size^2 x channels` |
| Residual connection | `output = sublayer(x) + x` |
| Layer Norm | `(x - mean_of_this_token) / sqrt(variance_of_this_token)` |
| Matmuls per Transformer block | 4 (self-attention) + 2 (MLP) = 6 |
| Total matmuls | `6 x number_of_blocks` |
| Autoregressive decoding steps | `output_length + 1` |

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** computing ViT patch dimension as just `16×16` (forgetting the channel dimension) — always multiply by the number of input channels (3 for RGB): `16×16×3=768`, not 256.
- **Trap:** computing Layer Norm's mean/variance ACROSS THE BATCH (a Batch Norm mindset) instead of across a single token's own features — Layer Norm's statistics come from just ONE token's feature vector, computed independently of every other token.
- **Trap:** forgetting the "+1" when counting autoregressive decoding steps — the `<end>` token itself requires its own generation step.
- **Exam hack:** for scaling-ratio questions, always compute BOTH the pairwise ratios (Original→GPT-2, GPT-2→GPT-3) AND the overall compounded ratio (Original→GPT-3) — questions often ask for one specific ratio, but showing all three demonstrates full understanding.
- **Exam hack:** the "6 matmuls per block" fact (4 attention + 2 MLP) is a precise, testable number worth memorizing exactly — it's frequently used to justify claims about Transformer parallelizability/efficiency.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-numerical)`

---

## Summary

This file worked every Transformer-scale calculation from the theory file into fully shown arithmetic. A 224×224×3 image with 16×16 patches produces exactly 196 patches (14×14 grid), each starting as a 768-dimensional flattened vector (16×16×3) — directly explaining the ViT paper's "16×16 Words" title, since a 224×224 image literally becomes a 196-token "sentence." A hand-computed residual connection example showed how adding a small sub-layer adjustment `[0.5,-0.2,0.1]` to the original input `[1.0,2.0,3.0]` produces an output `[1.5,1.8,3.1]` that clearly still carries the original input's identity. Applying Layer Normalization to that residual output computed its own token-specific mean (≈2.1333) and variance (≈0.4822), producing a normalized vector `[-0.9120,-0.4800,1.3920]` — explicitly independent of any other token, unlike Batch Norm. Comparing Original Transformer, GPT-2, and GPT-3 parameter counts revealed a ~7× jump then a further ~117× jump, compounding to roughly 822× total growth across just three years. Counting matrix multiplications for a 12-block Transformer gave exactly 72 total matmuls, illustrating the architecture's clean, GPU-friendly compute pattern. Finally, a full autoregressive decoding walkthrough for a 4-word sentence required exactly 5 steps (n+1, accounting for the final `<end>` token), with each step's growing input and single-token output shown explicitly. The master formula table consolidates every reusable calculation from this lecture for fast review.

`[← Theory](../theory/dl_lecture10_transformers_theory.md) · [🔝 Top](#dl-lecture-10--attention-and-transformers-numerical) · [Next: Practice →](../practice/dl_lecture10_transformers_practice.md)`
