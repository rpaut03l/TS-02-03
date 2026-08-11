# DL Lecture 03 — Convolutional Neural Networks (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice)`

> Folder: `Deep-Learning/Lecture-03-Convolutional-Neural-Networks/practice/`
> Pairs with: [`theory/dl_lecture03_cnn_theory.md`](../theory/dl_lecture03_cnn_theory.md) · [`numerical/dl_lecture03_cnn_numerical.md`](../numerical/dl_lecture03_cnn_numerical.md) · [`exercises/dl_lecture03_exercises.md`](../exercises/dl_lecture03_exercises.md)

---

## Table of Contents
1. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
2. [Explain-It-Back Prompts](#explain-it-back-prompts)
3. [Quick-Fire True / False](#quick-fire-true--false)
4. [Keras CONV Layer Practice](#keras-conv-layer-practice)
5. [Mini Interview-Style Round](#mini-interview-style-round)
6. [Summary](#summary)

---

## Concept Check — Fill in the Blank

1. A "3×3 filter" applied to an RGB image actually has shape ______.
2. The same filter reused at every location of an image is called ______.
3. Pooling is described as an "infinitely strong ______" that each unit should be invariant to small translations.
4. The normalization method that behaves identically at train and test time, and is preferred for RNNs/Transformers, is called ______.
5. AlexNet's first convolutional layer produces an output volume of shape ______.

<details>
<summary>Show answers</summary>

1. 3×3×3
2. parameter sharing
3. prior
4. Layer Normalization
5. 55×55×96
</details>

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice)`

---

## Explain-It-Back Prompts

1. Explain, in your own words, why hand-designed filters like Sobel "failed to scale," referencing the specific reasons given in the lecture.
2. Walk through the four-step "parameter sharing" story (blue filter, yellow filter, red filter) from memory.
3. Explain why BatchNorm behaves differently at train time vs test time, and why that's a common source of bugs.
4. Explain the "infinitely strong prior" framing for both convolution and pooling, using the prior-strength table from theory.
5. Explain why AlexNet's convolutional layers are relatively cheap in parameters while its fully connected layers are relatively expensive.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice)`

---

## Quick-Fire True / False

1. Local connectivity and parameter sharing are the same idea. — **False** (two separate, combinable restrictions).
2. Zero-padding is used to prevent spatial dimensions from shrinking with every convolutional layer. — **True**.
3. Max pooling keeps the average value in each window. — **False** (that's average pooling; max pooling keeps the largest value).
4. Batch Normalization's γ and β parameters are fixed, non-learnable constants. — **False** (they are learned during training).
5. Group Normalization avoids dependency on other samples in the batch, unlike standard Batch Normalization. — **True**.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice)`

---

## Keras CONV Layer Practice

The lecture notes that a Conv layer in Keras needs four hyperparameters. Match each one to its symbol from the theory/numerical files:

| Keras hyperparameter | Symbol used in this course | Your answer |
|---|---|---|
| Number of filters | ? | |
| The filter size | ? | |
| The stride | ? | |
| The zero padding | ? | |

<details>
<summary>Show answers</summary>

Number of filters → **K**. Filter size → **F**. Stride → **S**. Zero padding → **P**.
</details>

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice)`

---

## Mini Interview-Style Round

**Q1.** "We're deploying a model that processes one image at a time in production (batch size = 1). Someone suggests using standard Batch Normalization. What's your concern?"

<details>
<summary>Show answer</summary>

With a batch size of 1, computing a mean and variance "across the batch" is either meaningless or extremely noisy/unstable — BatchNorm's statistics become unreliable at very small batch sizes. At test/inference time this is usually handled by using running averages accumulated during training rather than the current (tiny) batch's statistics — but if training itself also uses very small batches, consider Group Normalization or Layer Normalization instead, since both avoid depending on other samples in the batch entirely and behave identically at train and test time.
</details>

**Q2.** "A colleague claims 'pooling always improves CNN performance, so use as much of it as possible.' How do you respond, using this lecture's material?"

<details>
<summary>Show answer</summary>

You'd push back using the theory file's prior-belief framing: pooling imposes an infinitely strong prior that each unit should be invariant to small translations. That's genuinely useful for tasks where exact position doesn't matter (e.g., "is a face present somewhere in this region"), but harmful for tasks where exact position *does* matter (e.g., precise pixel-level segmentation or exact keypoint localization) — in those cases, aggressive pooling throws away needed spatial information and increases error rather than reducing it.
</details>

**Q3.** "Explain to a junior teammate why AlexNet needed two GPUs in 2012."

<details>
<summary>Show answer</summary>

AlexNet's 96 first-layer filters and the rest of its ~60 million parameters, together with the memory needed for all its intermediate activations during training, exceeded what a single 2012-era GPU's memory could hold. The solution was to split the network — the paper describes the top 48 kernels being learned on one GPU and the bottom 48 on a second GPU — with limited cross-GPU communication at certain layers. This is a very early, hands-on example of "model parallelism," a technique still used (in far more sophisticated forms) to train today's much larger models across multiple GPUs.
</details>

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice)`

---

## Summary

This practice file reinforces Lecture 3's dense set of CNN building blocks through active recall. The fill-in-the-blank check anchors the depth-spanning filter rule, parameter sharing terminology, the "infinitely strong prior" framing, Layer Normalization's train/test consistency, and AlexNet's CONV1 output shape. Five explain-it-back prompts push you to reproduce the Sobel-filter failure story, the parameter-sharing walkthrough, BatchNorm's train/test mismatch, the prior-strength table, and AlexNet's conv-vs-FC parameter asymmetry entirely in your own words. A quick-fire true/false round targets common mix-ups (local connectivity vs parameter sharing, max vs average pooling, whether γ/β are learnable). A dedicated Keras hyperparameter-matching drill ties the course's symbolic notation (K, F, S, P) directly to real framework terminology you'll actually type in code. Finally, a three-question interview-style round rehearses reasoning about BatchNorm at small batch sizes, when pooling helps versus hurts, and why AlexNet needed multi-GPU training — the kind of applied "why," not just "what," questions a real interviewer would ask. Move to the exercises file next for a tiered, exam-format question bank with full worked answers.

`[← Numerical](../numerical/dl_lecture03_cnn_numerical.md) · [🔝 Top](#dl-lecture-03--convolutional-neural-networks-practice) · [Next: Exercises →](../exercises/dl_lecture03_exercises.md)`
