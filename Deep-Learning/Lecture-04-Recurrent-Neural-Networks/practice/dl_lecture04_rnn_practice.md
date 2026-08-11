# DL Lecture 04 — Recurrent Neural Networks (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

> Folder: `Deep-Learning/Lecture-04-Recurrent-Neural-Networks/practice/`
> Pairs with: [`theory/dl_lecture04_rnn_theory.md`](../theory/dl_lecture04_rnn_theory.md) · [`numerical/dl_lecture04_rnn_numerical.md`](../numerical/dl_lecture04_rnn_numerical.md) · [`exercises/dl_lecture04_exercises.md`](../exercises/dl_lecture04_exercises.md)

---

## Table of Contents
1. [Official In-Class Question](#official-in-class-question)
2. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
3. [Explain-It-Back Prompts](#explain-it-back-prompts)
4. [Quick-Fire True / False](#quick-fire-true--false)
5. [RNN Type Matching Drill](#rnn-type-matching-drill)
6. [Mini Interview-Style Round](#mini-interview-style-round)
7. [Summary](#summary)

---

## Official In-Class Question

**Q.** When an RNN reads the very first word in a sentence, there is no previous word or context, so where does its memory come from? What do you think the network starts with?

<details>
<summary>Show answer</summary>

The network starts with an empty memory, often all zeros: `h₀ = 0` (equivalently `s₋₁ = 0`). This initial all-zero hidden state is then fed into the very same recurrence formula, `s_0 = f(U x_0 + W s_{-1})`, alongside the first real input `x_0` — since `s_{-1}` is all zeros, the very first hidden state update effectively depends only on the first input, and the "memory" component only starts contributing from the second timestep onward.
</details>

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

---

## Concept Check — Fill in the Blank

1. The recurrence formula for the hidden state is `s_t = f(______)`.
2. RNNs share weights across the ______ dimension; CNNs share weights across the ______ dimension.
3. Training an RNN's weights using gradients computed by unrolling across time is called ______.
4. As sequence length grows, gradients from far-away timesteps can shrink toward zero — this is called the ______ problem.
5. Two gated architectures designed specifically to fight vanishing gradients are ______ and ______.

<details>
<summary>Show answers</summary>

1. `U x_t + W s_{t-1}`
2. temporal; spatial
3. Backpropagation Through Time (BPTT)
4. vanishing gradient
5. LSTM; GRU
</details>

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

---

## Explain-It-Back Prompts

1. Explain, using the "man who wore a wig" example, why feedforward networks are unsuitable for language modeling.
2. Walk through the RNN recurrence formula from memory, labelling every symbol.
3. Explain the vanishing gradient problem using the escalating "banana and apple" example, connecting it to the mechanism of repeated multiplication.
4. Explain the difference between vanishing and exploding gradients, and which kinds of activation functions/derivatives are associated with each.
5. Explain, in your own words, why Truncated BPTT is described as making a "Markov-like assumption."

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

---

## Quick-Fire True / False

1. RNNs assume all inputs and outputs are independent of each other. — **False** (that's the feedforward assumption RNNs are built to fix).
2. The same weight matrices are reused at every timestep in a vanilla RNN. — **True**.
3. Exploding gradients are typically associated with saturating activations like tanh and sigmoid. — **False** (that's vanishing gradients; exploding is more associated with non-saturating activations/large weights).
4. Truncated BPTT computes gradients across the entire sequence, no matter how long. — **False** (it deliberately limits how far back gradients are computed).
5. Gradient clipping preserves a gradient's direction while capping its magnitude. — **True**.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

---

## RNN Type Matching Drill

Match each RNN type to its correct application:

| Type | Application | Your match |
|---|---|---|
| One → Many | ? | |
| Many → One | ? | |
| Many → Many (aligned) | ? | |
| Many → Many (offset) | ? | |

Options: (a) Image Captioning, (b) Video classification on frame level, (c) Action prediction from a video frame sequence, (d) Video Captioning

<details>
<summary>Show answers</summary>

One → Many = (a) Image Captioning. Many → One = (c) Action prediction. Many → Many (offset, sequence in → different sequence out) = (d) Video Captioning. Many → Many (aligned, one output per input frame) = (b) Video classification on frame level.
</details>

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

---

## Mini Interview-Style Round

**Q1.** "Your model trains fine for short sentences but performs poorly on paragraph-length inputs. What RNN-specific issue would you suspect first, and what's your first cheap fix to try?"

<details>
<summary>Show answer</summary>

Suspect the vanishing gradient problem — as sequence length grows, gradient signal from early timesteps shrinks toward zero, so the network effectively can't learn long-range dependencies needed for paragraph-length text. The cheapest first fix is usually switching to a non-saturating activation like ReLU, or — more robustly — switching to a gated architecture like LSTM or GRU, which are specifically designed to preserve gradient flow across long sequences.
</details>

**Q2.** "Training loss suddenly turns into NaN partway through training an RNN. What's your first hypothesis, and how would you address it?"

<details>
<summary>Show answer</summary>

First hypothesis: exploding gradients causing numeric overflow. A standard, cheap fix is gradient clipping — capping the gradient's norm at some threshold before applying the weight update, so no single update can blow up the weights into NaN/infinity territory.
</details>

**Q3.** "Explain to a teammate why an RNN's parameter count doesn't grow even if you train on much longer sequences."

<details>
<summary>Show answer</summary>

Because the same three weight matrices (U, W, V) are reused identically at every single timestep — the recurrence formula is applied repeatedly using those same weights, rather than learning a fresh set of weights per timestep. This is the RNN's version of parameter sharing, directly analogous to a CNN filter being reused at every spatial location rather than learning a separate filter per pixel.
</details>

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice)`

---

## Summary

This practice file drills Lecture 4's RNN concepts through active recall. The official in-class question on initial hidden states is answered in full, explaining exactly why `h₀=0` and how that interacts with the first real recurrence step. A fill-in-the-blank check reinforces the recurrence formula, the temporal-vs-spatial sharing distinction, BPTT terminology, and the vanishing gradient problem and its gated-architecture fixes. Five explain-it-back prompts push you to reproduce the wig/man independence-assumption failure, the full recurrence formula with every symbol labelled, the escalating banana/apple vanishing-gradient example, the vanishing-vs-exploding distinction, and the Markov-like framing of Truncated BPTT. A quick-fire true/false round and a dedicated RNN-type-to-application matching drill (One-to-Many, Many-to-One, and both Many-to-Many variants) test both terminology and applied recognition. A three-question interview-style round rehearses real debugging scenarios — diagnosing vanishing gradients from poor long-sequence performance, diagnosing exploding gradients from NaN losses, and explaining constant parameter count under weight sharing — the kind of applied troubleshooting a real engineering interview would probe. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture04_rnn_numerical.md) · [🔝 Top](#dl-lecture-04--recurrent-neural-networks-practice) · [Next: Exercises →](../exercises/dl_lecture04_exercises.md)`
