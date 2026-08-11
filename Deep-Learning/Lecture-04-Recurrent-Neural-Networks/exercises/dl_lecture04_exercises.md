# DL Lecture 04 — Exercise Bank (Recurrent Neural Networks)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-04--exercise-bank-recurrent-neural-networks)`

> Folder: `Deep-Learning/Lecture-04-Recurrent-Neural-Networks/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture04_rnn_theory.md) · [numerical](../numerical/dl_lecture04_rnn_numerical.md) · [practice](../practice/dl_lecture04_rnn_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q4.1.** Write the RNN hidden state recurrence formula.

**Q4.2.** What value does the initial hidden state typically take?

**Q4.3.** Name the algorithm used to train RNNs by unrolling across time.

**Q4.4.** What are the two gradient problems that make training RNNs on long sequences difficult?

**Q4.5.** Name the two gated architectures mentioned as the most thorough fix for vanishing gradients.

---

## 🟡 Medium — Applied Reasoning

**Q4.6.** For D=10 (input size), H=20 (hidden size), k=10 (output size), compute the total RNN parameter count (U, W, V only, ignoring biases).

**Q4.7.** Explain why RNN parameter count does not depend on sequence length, referencing weight sharing explicitly.

**Q4.8.** A gradient's per-timestep multiplication factor is exactly 1.0. What happens to the gradient as it's backpropagated across many timesteps, and why is this actually a desirable property (this is part of the motivation for certain LSTM design choices, previewed in the next lecture)?

**Q4.9.** Classify each of the following tasks as One-to-Many, Many-to-One, or Many-to-Many: (a) sentiment classification of a full movie review, (b) generating a caption for a single photo, (c) translating an English sentence into French.

**Q4.10.** Explain why Truncated BPTT trades off long-range learning ability for training speed and stability.

---

## 🔴 Hard — Derivation & Multi-Step

**Q4.11.** Given `U = [[1,0],[0,1]]`, `W = [[0.5,0],[0,0.5]]`, `s_{-1}=[0,0]`, `x_0=[2,1]`, and activation = identity (no squashing, for simplicity), compute `s_0` step by step.

**Q4.12.** Continuing from Q4.11, with `x_1 = [1,1]`, compute `s_1` step by step.

**Q4.13.** A gradient's per-timestep multiplication factor is 0.9. After how many timesteps does the cumulative gradient factor first drop below 0.1? Show your working (trial exponents are acceptable).

**Q4.14.** A gradient vector is `g = [6, 8]` and the clipping threshold is 3. Compute the clipped gradient vector, showing every step, and verify its final magnitude equals the threshold.

`[🔝 Top](#dl-lecture-04--exercise-bank-recurrent-neural-networks)`

---

## Answer Key

<details>
<summary>Q4.1 – Q4.5 (Easy)</summary>

- **Q4.1:** `s_t = f(U x_t + W s_{t-1})`.
- **Q4.2:** All zeros (`s_{-1}=0` or `h_0=0`).
- **Q4.3:** Backpropagation Through Time (BPTT).
- **Q4.4:** Vanishing gradients and exploding gradients.
- **Q4.5:** LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit).
</details>

<details>
<summary>Q4.6 – Q4.10 (Medium)</summary>

- **Q4.6:** U: H×D=20×10=200. W: H×H=20×20=400. V: k×H=10×20=200. Total = 200+400+200 = **800 parameters**.
- **Q4.7:** The exact same U, W, V weight matrices are reused identically at every timestep, regardless of how many timesteps exist — there is no per-timestep set of independent weights, so adding more timesteps to a sequence adds more *computation* but zero additional *parameters*.
- **Q4.8:** If the per-timestep factor is exactly 1.0, then `1.0^n = 1.0` for any n — the gradient neither vanishes nor explodes no matter how many timesteps it's backpropagated through, staying perfectly stable. This is a big part of the motivation behind LSTM's design (specifically its "cell state" pathway), which is engineered to have a close-to-1 gradient-preserving path by default.
- **Q4.9:** (a) sentiment classification of a full review = Many-to-One (many words in, one sentiment label out). (b) captioning a single photo = One-to-Many (one image in, a sequence of words out). (c) translating a sentence = Many-to-Many (a sequence of words in, a sequence of words out).
- **Q4.10:** By limiting how far back gradients are computed, Truncated BPTT reduces both computation time per update and the risk of vanishing/exploding gradients from very long chains — but it also means the network's weight updates never "see" or learn from dependencies that stretch further back than the truncation window, effectively imposing a Markov-like assumption that only recent history matters for learning, even if the true task has genuinely longer-range dependencies.
</details>

<details>
<summary>Q4.11 – Q4.14 (Hard)</summary>

- **Q4.11:** U·x_0 = [1×2+0×1, 0×2+1×1] = [2,1]. W·s_{-1} = [0.5×0+0×0, 0×0+0.5×0] = [0,0]. Sum = [2,1]. Identity activation → **s_0 = [2, 1]**.
- **Q4.12:** U·x_1 = [1×1+0×1, 0×1+1×1] = [1,1]. W·s_0 = [0.5×2+0×1, 0×2+0.5×1] = [1, 0.5]. Sum = [1+1, 1+0.5] = **s_1 = [2, 1.5]**.
- **Q4.13:** 0.9^1=0.9, 0.9^5≈0.590, 0.9^10≈0.349, 0.9^15≈0.206, 0.9^20≈0.122, 0.9^22≈0.098. So the cumulative factor first drops below 0.1 at **n=22 timesteps** (0.9^21≈0.109, 0.9^22≈0.098).
- **Q4.14:** ‖g‖ = √(6²+8²) = √(36+64) = √100 = 10. Since 10 > 3, clip: g_clipped = [6,8] × (3/10) = [1.8, 2.4]. Verify: ‖[1.8,2.4]‖ = √(3.24+5.76) = √9 = **3** ✓, matches the threshold.
</details>

`[🔝 Top](#dl-lecture-04--exercise-bank-recurrent-neural-networks)`

---

## Summary

This exercise bank drills Lecture 4's RNN mechanics across three tiers. Easy questions recall the recurrence formula, initial hidden state convention, BPTT terminology, the two gradient problems, and the two named gated architectures. Medium questions apply the parameter-count formula to new dimensions (800 total parameters for D=10/H=20/k=10), explain weight sharing's role in keeping parameter count sequence-length-independent, reason about the special stability of a per-timestep factor of exactly 1.0 (foreshadowing LSTM design in the next lecture), classify three new tasks by RNN input/output pattern, and explain Truncated BPTT's speed-vs-long-range-learning trade-off. Hard questions require full multi-step derivations: two consecutive hand-computed hidden state updates using simplified identity-activation arithmetic, a trial-and-error exponent search for when a 0.9-per-step vanishing factor first drops below 0.1 (22 timesteps), and a full gradient-clipping computation reducing a magnitude-10 gradient down to exactly the threshold of 3 while preserving its direction. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture04_rnn_practice.md) · [🔝 Top](#dl-lecture-04--exercise-bank-recurrent-neural-networks) · [Code →](../code/README.md)`
