# DL Lecture 02 — Exercise Bank (Neural Networks)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-02--exercise-bank-neural-networks)`

> Folder: `Deep-Learning/Lecture-02-Neural-Networks/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture02_neural_networks_theory.md) · [numerical](../numerical/dl_lecture02_neural_networks_numerical.md) · [practice](../practice/dl_lecture02_neural_networks_practice.md)

---

## Table of Contents
1. [🟢 Easy — Definitions & Recall](#-easy--definitions--recall)
2. [🟡 Medium — Applied Reasoning](#-medium--applied-reasoning)
3. [🔴 Hard — Derivation & Multi-Step](#-hard--derivation--multi-step)
4. [Answer Key](#answer-key)
5. [Summary](#summary)

---

## 🟢 Easy — Definitions & Recall

**Q2.1.** Write the formula for a linear classifier and label the shape of every symbol.

**Q2.2.** Write the formula for a 2-layer neural network and name the activation function used.

**Q2.3.** What are the five steps in the gradient descent training loop, in order?

**Q2.4.** What is the more precise/technical name for what this lecture calls a "Neural Network"?

**Q2.5.** What does ReLU output when its input is negative?

---

## 🟡 Medium — Applied Reasoning

**Q2.6.** If D = 15 (input features), H = 8 (hidden units), and C = 2 (classes), what are the shapes of W1, b1, W2, and b2?

**Q2.7.** Explain why the ring-of-blue-around-red example cannot be solved by a linear classifier, but can be solved after a polar coordinate transform.

**Q2.8.** A friend says "since W1 and W2 are both just matrices, I can multiply them together ahead of time to make the network faster." Under what condition would this actually be mathematically valid (and therefore a genuinely bad idea for network expressiveness)?

**Q2.9.** For the Titanic dataset, name two categorical columns and explain how you would prepare them before feeding them into a neural network.

**Q2.10.** Why is feature scaling especially important when using gradient descent, specifically referencing the weight update formula?

---

## 🔴 Hard — Derivation & Multi-Step

**Q2.11.** Given D=10, H=20, C=3, compute the total number of learnable parameters in the 2-layer network (including biases). Show every step.

**Q2.12.** A single neuron receives inputs `x1=3, x2=-1, x3=2` with weights `w1=0.4, w2=0.9, w3=-0.5` and bias `b=0.2`. Compute the neuron's output using (a) ReLU activation and (b) sigmoid activation. Show every arithmetic step.

**Q2.13.** Prove, using your own 2×2 matrices (different from the numerical file's example), that stacking two linear layers without a non-linear activation between them is equivalent to a single combined linear layer.

**Q2.14.** A network predicts `ŷ = 0.65` for a true label `y = 1`. Using the simplified update rule `w_new = w_old + η × error × input`, compute the new value of a weight `w_old = 0.5` connected to an input of `1.2`, using learning rate `η = 0.05`. Then explain what would happen differently if `η` were instead `0.5` (ten times larger).

`[🔝 Top](#dl-lecture-02--exercise-bank-neural-networks)`

---

## Answer Key

<details>
<summary>Q2.1 – Q2.5 (Easy)</summary>

- **Q2.1:** `f = Wx`, where `x ∈ R^D` and `W ∈ R^(C×D)`, producing `f ∈ R^C`.
- **Q2.2:** `f = W2 max(0, W1x+b1) + b2`; the activation function is ReLU (`max(0, ·)`).
- **Q2.3:** Initialise (random weights) → Present (a training pattern) → Feed forward (get output) → Compare (with target output) → Adjust (weights based on error).
- **Q2.4:** A fully-connected network, also called a Multi-Layer Perceptron (MLP).
- **Q2.5:** Exactly 0.
</details>

<details>
<summary>Q2.6 – Q2.10 (Medium)</summary>

- **Q2.6:** W1: 8×15, b1: 8×1, W2: 2×8, b2: 2×1.
- **Q2.7:** In Cartesian (x,y) coordinates, the red (center) and blue (ring) points are mixed such that any straight line either cuts through both groups or misses separating them cleanly — a ring has no "one side vs other side" straight-line split. After transforming to polar (r, θ), the center points all have small `r` and the ring points all have large `r`, so a single straight-line cut purely on the `r` axis now perfectly separates them — the transform turned an angularly-mixed problem into a radially-separated one.
- **Q2.8:** This would only be mathematically valid (i.e., give an identical result) if there is *no non-linear activation function* between W1 and W2. If that's the case, then yes, `W1 W2` really can be precomputed into one matrix — but that's exactly the theory file's point: doing so proves the two-layer network was never more expressive than a single linear layer to begin with, which is a bad thing, not a valid speed optimization, for any network that's supposed to learn non-linear patterns.
- **Q2.9:** `Sex` (male/female) and `Embarked` (port of boarding, e.g. C/Q/S) are categorical. `Sex` can be binary-encoded (0/1). `Embarked` has more than 2 categories, so it should be one-hot encoded (a separate 0/1 column per port) rather than assigned arbitrary numbers, since arbitrary numbering would falsely imply an ordering between ports that doesn't exist.
- **Q2.10:** The weight update is `w_new = w_old + η × error × input × ...` — the `input` term directly multiplies into the update size. If one feature (e.g., Fare, up to 500+) is on a much larger scale than another (e.g., a 0/1 encoded Sex column), its weight updates will be proportionally much larger and more volatile every step, making training unstable or slow to converge; scaling all inputs to a similar range keeps updates balanced across features.
</details>

<details>
<summary>Q2.11 – Q2.14 (Hard)</summary>

- **Q2.11:** W1 params = H×D = 20×10 = 200. b1 params = H = 20. W2 params = C×H = 3×20 = 60. b2 params = C = 3. Total = 200+20+60+3 = **283 parameters**.
- **Q2.12:** Weighted sum: (0.4×3) + (0.9×−1) + (−0.5×2) + 0.2 = 1.2 − 0.9 − 1.0 + 0.2 = **−0.5**. (a) ReLU: max(0, −0.5) = **0**. (b) Sigmoid: σ(−0.5) = 1/(1+e^0.5) = 1/(1+1.6487) = 1/2.6487 ≈ **0.3775**.
- **Q2.13:** Any valid worked example following the same pattern as the numerical file (pick X, W1, W2, compute `(XW1)W2` step by step, then compute `W1W2` first and apply once, and confirm both give the same number) is correct — the key deliverable is showing both computation orders arrive at an identical result.
- **Q2.14:** error = y − ŷ = 1 − 0.65 = 0.35. w_new = 0.5 + (0.05 × 0.35 × 1.2) = 0.5 + 0.021 = **0.521**. With η=0.5 instead: w_new = 0.5 + (0.5 × 0.35 × 1.2) = 0.5 + 0.21 = **0.71** — a much bigger jump. A learning rate ten times larger produces an update ten times larger; too large a learning rate risks overshooting the optimal weight value and can make training unstable or cause the loss to oscillate/diverge instead of smoothly decreasing.
</details>

`[🔝 Top](#dl-lecture-02--exercise-bank-neural-networks)`

---

## Summary

This exercise bank mirrors Lecture 1's tiered structure for Lecture 2's Neural Networks material. Easy questions cover pure recall of the linear classifier formula, the 2-layer network formula, the five-beat gradient descent loop, MLP terminology, and ReLU's behaviour on negative inputs. Medium questions push into applied reasoning: shape-tracking a concrete D=15/H=8/C=2 network, explaining the polar-coordinate separability trick in your own words, spotting the (bad) implication of pre-multiplying weight matrices without activations, preparing Titanic's categorical columns correctly, and connecting feature scaling directly to the gradient descent update formula. Hard questions require full derivations and multi-step numerical work: a complete 283-parameter count for a D=10/H=20/C=3 network, a hand-computed single-neuron forward pass under both ReLU and sigmoid activations, an independent proof of the no-activation linear-collapse result, and a full gradient descent weight-update calculation comparing a normal versus a ten-times-larger learning rate — directly illustrating why learning rate choice matters for training stability. All answers are fully worked and spoiler-tagged for timed self-testing.

`[← Practice](../practice/dl_lecture02_neural_networks_practice.md) · [🔝 Top](#dl-lecture-02--exercise-bank-neural-networks) · [Code →](../code/README.md)`
