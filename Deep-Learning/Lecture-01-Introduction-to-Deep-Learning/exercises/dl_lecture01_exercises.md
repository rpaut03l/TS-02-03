# DL Lecture 01 — Exercise Bank (Introduction to Deep Learning)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-01--exercise-bank-introduction-to-deep-learning)`

> Folder: `Deep-Learning/Lecture-01-Introduction-to-Deep-Learning/exercises/`
> This is a growing, real-exam/real-interview style question bank, kept separate from the practice file so it can expand independently as more lectures are added. Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture01_introduction_theory.md) · [numerical](../numerical/dl_lecture01_introduction_numerical.md) · [practice](../practice/dl_lecture01_introduction_practice.md)

---

## Table of Contents
1. [How This Exercise Bank Grows](#how-this-exercise-bank-grows)
2. [🟢 Easy — Definitions & Recall](#-easy--definitions--recall)
3. [🟡 Medium — Applied Reasoning](#-medium--applied-reasoning)
4. [🔴 Hard — Derivation & Multi-Step](#-hard--derivation--multi-step)
5. [Answer Key](#answer-key)
6. [Summary](#summary)

---

## How This Exercise Bank Grows

Every lecture gets its own `exercises/dl_lectureNN_exercises.md` file, following this exact same tier structure (Easy/Medium/Hard), so that by the end of the trimester you have one consistent, browsable question bank across all of Deep Learning — perfect for a final pre-exam sprint. Numbering restarts per lecture (Q1.1 = Lecture 1, Question 1); cross-lecture cumulative review questions will be added to a `Deep-Learning/cumulative-review/` folder once at least 3–4 lectures exist.

`[🔝 Top](#dl-lecture-01--exercise-bank-introduction-to-deep-learning)`

---

## 🟢 Easy — Definitions & Recall

**Q1.1.** Define Deep Learning in one sentence, using the word "hierarchical."

**Q1.2.** Name the three "keys" that had to line up together to cause the recent Deep Learning boom.

**Q1.3.** What dataset did LeCun et al. (1998) use for the first practical deep learning success?

**Q1.4.** In RGB, what color does (0,0,0) represent? In CMY, what color does (0,0,0) represent?

**Q1.5.** What is the name of the property that lets a pattern-detector work no matter where in the image the pattern appears?

---

## 🟡 Medium — Applied Reasoning

**Q1.6.** A grayscale image is a 2D matrix, but it's still called "single-channel." Explain why, in 2–3 sentences.

**Q1.7.** Explain, using the "early / middle / deep layers" idea, why locality is described as looking at small patches first and integrating global information last.

**Q1.8.** A classmate says "weight sharing and locality are basically the same thing." Explain, in your own words, why they are two distinct ideas that just happen to work together.

**Q1.9.** Why does the lecture call tabular data a case where "we do not assume any structure a priori concerning how the features interact" — and why is that assumption fine for tabular data but wrong for images?

**Q1.10.** List one low-level, one mid-level, and one high-level vision task that were *not* explicitly named in the lecture slides (i.e., come up with your own examples that still fit each category correctly).

---

## 🔴 Hard — Derivation & Multi-Step

**Q1.11.** Starting from a fully connected layer's 4D weight tensor Wᵢ,ⱼ,ₖ,ₗ, derive — step by step, in words and light notation — how applying translation invariance followed by locality results in a standard convolution kernel [V]ₐ,ᵦ. (This mirrors the derivation in the theory file — attempt it from memory first.)

**Q1.12.** For a 512×512×3 input image: (a) compute the number of parameters in a fully connected layer with 64 hidden units, (b) compute the number of parameters in a convolutional layer with 5×5 filters and 64 filters, (c) compute the exact ratio between the two.

**Q1.13.** A 1D audio clip is 16,000 samples long (1 second at 16kHz), single channel. You want to detect a short sound event that is about 400 samples long, and it could occur anywhere in the clip. Using the ideas of locality and translation invariance from this lecture, explain in words what kernel size range would make sense, and why a fully-connected approach over the full 16,000 samples would be a poor design choice — include an approximate parameter count comparison.

**Q1.14.** Explain why the "4Δ²" formula in the theory/numerical files is described as an *approximation* rather than an exact count, and derive the exact formula it is approximating.

`[🔝 Top](#dl-lecture-01--exercise-bank-introduction-to-deep-learning)`

---

## Answer Key

<details>
<summary>Q1.1 – Q1.5 (Easy)</summary>

- **Q1.1:** Deep Learning is the study of many-layered neural networks that automatically learn hierarchical features from raw data, without manual feature engineering.
- **Q1.2:** Data, Compute, Algorithms.
- **Q1.3:** The MNIST handwritten digit dataset.
- **Q1.4:** RGB (0,0,0) = black. CMY (0,0,0) = white (no ink).
- **Q1.5:** (Translation) invariance.
</details>

<details>
<summary>Q1.6 – Q1.10 (Medium)</summary>

- **Q1.6:** "Channel" counts independent measurements *per location*, not spatial dimensionality. A grayscale image has exactly 1 intensity value per pixel location (even though there are many locations arranged in 2D), so it is single-channel, same as a timeseries having 1 value per timestep.
- **Q1.7:** Early layers only "see" small local windows, so they can only detect small local patterns (edges/textures). As you go deeper, each layer's effective receptive field grows because it's built from combinations of the previous layer's outputs, so it can represent progressively larger and more abstract structures — parts, then whole objects, then full scenes.
- **Q1.8:** Weight sharing addresses *how many independent copies* of a weight exist (one, reused everywhere, vs. one per position). Locality addresses *how far* a single copy of that weight is allowed to "reach" (a small window vs. the whole image). You could in principle have sharing without locality (Worked Example 3 in the numerical file — shared weights that still span nearly the whole image) — they are separate restrictions that happen to be applied together to get a standard convolution kernel.
- **Q1.9:** Tabular data (spreadsheet-like rows/columns) has columns that are just different independent features (e.g., age, income, height) with no assumed physical adjacency — column 3 is not "next to" column 4 in any meaningful sense, so treating all pairs of features as equally connectable (as MLPs do) is a reasonable, harmless assumption. Images violate this because pixel (i,j) genuinely is physically adjacent to pixel (i,j+1); ignoring that adjacency throws away real, useful structure.
- **Q1.10:** Any reasonable, correctly-categorized examples are acceptable — e.g., low-level: corner detection in a scanned document; mid-level: detecting individual traffic signs in a street photo; high-level: judging whether a video clip shows a "birthday party" versus a "sports match."
</details>

<details>
<summary>Q1.11 – Q1.14 (Hard)</summary>

- **Q1.11:** Step 1: start with Wᵢ,ⱼ,ₖ,ₗ, a fully independent weight for every (output, input) position pair — no structure assumed. Step 2 (translation invariance): re-parameterize the weight to depend only on the relative offset (a,b) = (k−i, l−j) instead of the absolute positions, so Wᵢ,ⱼ,ₖ,ₗ collapses into a single shared function Vₐ,ᵦ reused at every output location. Step 3 (locality): force Vₐ,ᵦ = 0 whenever |a| > Δ or |b| > Δ, keeping only a small local window of nonzero weights. Step 4: what remains — a small, shared, local weight window applied identically at every output position — is by definition a convolution kernel, and the resulting operation ([H]ᵢ,ⱼ = Σ over small a,b of [V]ₐ,ᵦ[X]ᵢ₊ₐ,ⱼ₊ᵦ) is a standard 2D convolution.
- **Q1.12:** (a) FC: inputs = 512×512×3 = 786,432; weights = 786,432 × 64 = 50,331,648; + 64 biases = **50,331,712**. (b) Conv: weights per filter = 5×5×3 = 75; +1 bias = 76; × 64 filters = **4,864**. (c) Ratio = 50,331,712 / 4,864 ≈ **10,349×** — the FC layer needs roughly ten thousand times more parameters for the same "64 output units" budget.
- **Q1.13:** A kernel roughly in the same ballpark as the event length (a few hundred samples, e.g. 200–400-ish, not the full 16,000) is appropriate — big enough to capture the whole short sound pattern, small enough to stay local and translation-invariant so the same detector works regardless of *when* the sound occurs. A fully connected layer over all 16,000 samples would need on the order of 16,000 × (hidden units) parameters just for one layer (e.g., with 100 hidden units, 1.6 million parameters) while also having to separately learn to detect the event at every possible time offset since it has no built-in shift invariance — a small 1D convolution kernel achieves the same detection with only a few hundred shared weights, reused automatically at every time step.
- **Q1.14:** The exact number of integer offsets satisfying |a| ≤ Δ is (2Δ + 1), not 2Δ — you must include a=0 itself as one of the allowed offsets (the "no shift" position), plus Δ positive and Δ negative offsets, giving 2Δ+1 total. So the exact 2D kernel size is (2Δ+1) × (2Δ+1), and 4Δ² is simply the leading-order approximation of that expression for reasonably large Δ, dropping the smaller +4Δ+1 correction terms — good enough for order-of-magnitude reasoning, but not the exact count.
</details>

`[🔝 Top](#dl-lecture-01--exercise-bank-introduction-to-deep-learning)`

---

## Summary

This exercise bank is the graded-question-bank counterpart to the more conversational practice file, organized into three difficulty tiers so it can double as a structured revision path: Easy questions test pure recall (definitions, named datasets, the three DL-boom pillars), Medium questions test applied reasoning (explaining channel-counting, layer-depth behaviour, and the tabular-vs-image structure assumption in your own words), and Hard questions test full derivations and multi-step numerical work, including a fresh 512×512×3 parameter-count problem and a from-scratch re-derivation of the convolution kernel from a fully connected weight tensor. All answers are collapsed behind spoiler tags so this file can be reused for timed self-testing. As more Deep Learning lectures are documented, each will get its own similarly-tiered exercises file in its own `exercises/` folder, and once several lectures exist a `cumulative-review/` folder will aggregate cross-lecture questions for full-course revision sprints.

`[← Practice](../practice/dl_lecture01_introduction_practice.md) · [🔝 Top](#dl-lecture-01--exercise-bank-introduction-to-deep-learning) · [Code →](../code/README.md)`
