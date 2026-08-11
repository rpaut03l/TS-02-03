# DL Lecture 02 — Neural Networks (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-02--neural-networks-practice)`

> Folder: `Deep-Learning/Lecture-02-Neural-Networks/practice/`
> Pairs with: [`theory/dl_lecture02_neural_networks_theory.md`](../theory/dl_lecture02_neural_networks_theory.md) · [`numerical/dl_lecture02_neural_networks_numerical.md`](../numerical/dl_lecture02_neural_networks_numerical.md) · [`exercises/dl_lecture02_exercises.md`](../exercises/dl_lecture02_exercises.md)

---

## Table of Contents
1. [How to Use This File](#how-to-use-this-file)
2. [Official Practice Problem — Kaggle Titanic](#official-practice-problem--kaggle-titanic)
3. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
4. [Explain-It-Back Prompts](#explain-it-back-prompts)
5. [Quick-Fire True / False](#quick-fire-true--false)
6. [Mini Interview-Style Round](#mini-interview-style-round)
7. [Summary](#summary)

---

## How to Use This File

Same rhythm as Lecture 1's practice file: try every question yourself first (out loud or on paper), then open the spoiler tag.

`[🔝 Top](#dl-lecture-02--neural-networks-practice)`

---

## Official Practice Problem — Kaggle Titanic

**Task.** Using the Titanic dataset (passenger age, class, fare, sex, family size, etc.), build a small neural network to predict survival (binary: 0 or 1).

**Getting-started checklist:**

<details>
<summary>Show step-by-step starter plan</summary>

1. **Load and inspect the data.** Look at column types — you'll find both numeric columns (age, fare) and categorical columns (sex, embarked port).
2. **Handle missing values.** The Titanic dataset famously has missing `Age` values — a common beginner move is filling them with the median age.
3. **Encode categorical columns.** Neural networks only understand numbers — convert "male"/"female" into 0/1, and one-hot encode multi-category columns like embarked port.
4. **Scale numeric features.** Features like `Fare` (0 to 500+) and `Age` (0 to 80) are on very different scales — scaling them (e.g., to 0–1) helps gradient descent converge faster and more stably.
5. **Split into train/validation sets.** So you can measure whether your model generalizes, not just memorizes.
6. **Build a small 2-layer network** — literally `f = W2 max(0, W1x+b1) + b2` from the theory file, with a sigmoid on the final output since this is binary classification.
7. **Train with gradient descent** (in practice, an optimizer like Adam — covered in the DNN Optimization lecture — handles this automatically).
8. **Evaluate accuracy** on the validation set, and compare against a simple baseline (e.g., "always predict the majority class") to make sure your network is actually learning something useful.
</details>

`[🔝 Top](#dl-lecture-02--neural-networks-practice)`

---

## Concept Check — Fill in the Blank

1. A linear classifier can be written as `f = ______`.
2. The activation function used in the 2-layer formula `f = W2 max(0, W1x+b1) + b2` is called ______.
3. Manually transforming Cartesian coordinates to ______ coordinates was the classical trick to make a ring-shaped dataset linearly separable.
4. Removing all activation functions from a multi-layer network causes it to mathematically collapse into a single ______ layer.
5. The precise technical name for what this lecture calls a "Neural Network" is a ______ network, also known as an ______.

<details>
<summary>Show answers</summary>

1. `Wx`
2. ReLU (rectified linear unit)
3. polar
4. linear
5. fully-connected; MLP (Multi-Layer Perceptron)
</details>

`[🔝 Top](#dl-lecture-02--neural-networks-practice)`

---

## Explain-It-Back Prompts

1. Explain, without writing any formulas, why a single straight line cannot separate a ring of points from a cluster at its center.
2. Walk through the five-beat gradient descent training loop from memory, in your own words.
3. Explain why `H` (hidden units) is called a "design choice" rather than something learned from data.
4. Prove, using your own small numeric example (not the one in the numerical file), that two stacked linear layers without activation collapse into one linear layer.
5. Explain the difference between a "layer" and a "neuron" to someone who has never seen either term before.

`[🔝 Top](#dl-lecture-02--neural-networks-practice)`

---

## Quick-Fire True / False

1. A single-layer perceptron and the final dense layer of a deep network both compute `f = Wx` underneath. — **True**.
2. Activation functions change during training, just like weights do. — **False** (they're fixed at architecture-design time; only weights and biases change).
3. `W1` in a 2-layer network has shape `D × H`. — **False** (it's `H × D`).
4. ReLU outputs a small negative number for negative inputs. — **False** (it outputs exactly 0).
5. "Neural Network" and "Multi-Layer Perceptron" can refer to the same thing in this lecture's context. — **True**.

`[🔝 Top](#dl-lecture-02--neural-networks-practice)`

---

## Mini Interview-Style Round

**Q1.** "Your teammate suggests just making the network 'really deep' (50 layers) to solve a hard image problem, but forgets to add activation functions between layers. What would you tell them?"

<details>
<summary>Show answer</summary>

You'd explain that without non-linear activation functions between every pair of linear layers, all 50 layers mathematically collapse into a single combined linear transform — the model would have the exact same limited expressive power as one plain linear classifier, no matter how many layers or parameters it technically has. All the extra compute and memory would be wasted. You'd insist on inserting a non-linear activation (like ReLU) after every linear layer except typically the very last one (which often uses a task-specific activation like sigmoid or softmax instead).
</details>

**Q2.** "Why do we scale/normalize numeric features (like Fare and Age in Titanic) before training a neural network?"

<details>
<summary>Show answer</summary>

Because gradient descent's weight updates are sensitive to the scale of the inputs (recall from the numerical file: `w_new = w_old + η × error × input × ...` — the input value directly scales the size of each update). If `Fare` ranges up to 500+ while `Age` ranges up to ~80, the network's weight updates for the Fare-connected weights would be dramatically larger and more unstable than for Age-connected weights, making training slower and less stable. Scaling both to a similar range (e.g., 0–1) keeps gradient updates well-behaved across all input features.
</details>

**Q3.** "How would you explain the difference between a linear classifier and a neural network to a complete beginner in one sentence?"

<details>
<summary>Show answer</summary>

A linear classifier draws exactly one straight line (or flat plane) to separate classes, while a neural network stacks many such straight-line layers together with small non-linear "twists" in between, letting it bend and combine those straight lines into shapes complex enough to separate almost any pattern.
</details>

`[🔝 Top](#dl-lecture-02--neural-networks-practice)`

---

## Summary

This practice file turns Lecture 2's ideas into active-recall drills. The official Kaggle Titanic practice problem is broken into a concrete eight-step starter checklist covering data loading, missing-value handling, categorical encoding, feature scaling, train/validation splitting, building a small `f = W2 max(0, W1x+b1) + b2` network, training with gradient descent, and evaluating against a baseline — everything needed to go from "theory" to "a working first model" on real data. A fill-in-the-blank concept check reinforces the linear classifier formula, ReLU, polar coordinates, the no-activation-collapse result, and the fully-connected/MLP terminology. Five explain-it-back prompts push you to reproduce the ring-vs-center intuition, the five-beat gradient descent loop, and the linear-collapse proof entirely in your own words. A quick-fire true/false round targets the most common shape and terminology mix-ups (especially `W1`'s `H×D` shape and ReLU's hard zero-clipping), and a mini interview-style round rehearses how to explain "why non-linearity matters" and "why feature scaling matters" the way a real interviewer or viva examiner might probe them. Move to the exercises file next for a tiered, exam-format question bank with a full answer key.

`[← Numerical](../numerical/dl_lecture02_neural_networks_numerical.md) · [🔝 Top](#dl-lecture-02--neural-networks-practice) · [Next: Exercises →](../exercises/dl_lecture02_exercises.md)`
