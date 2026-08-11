# DL Lecture 05 — LSTM (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-05--lstm-practice)`

> Folder: `Deep-Learning/Lecture-05-LSTM/practice/`
> Pairs with: [`theory/dl_lecture05_lstm_theory.md`](../theory/dl_lecture05_lstm_theory.md) · [`numerical/dl_lecture05_lstm_numerical.md`](../numerical/dl_lecture05_lstm_numerical.md) · [`exercises/dl_lecture05_exercises.md`](../exercises/dl_lecture05_exercises.md)

---

## Table of Contents
1. [Official In-Class Question](#official-in-class-question)
2. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
3. [Explain-It-Back Prompts](#explain-it-back-prompts)
4. [Quick-Fire True / False](#quick-fire-true--false)
5. [Gate Matching Drill](#gate-matching-drill)
6. [Mini Interview-Style Round](#mini-interview-style-round)
7. [Summary](#summary)

---

## Official In-Class Question

**Q.** Can you guess some issues with LSTM?

<details>
<summary>Show a reasoned answer</summary>

Building on the lecture's own trajectory (which moves next into Attention): (1) LSTMs are still fundamentally **sequential** — each timestep depends on the previous one finishing, so they can't be parallelized across time the way CNNs can be parallelized across space, making them slow to train on long sequences. (2) Despite being far more stable than vanilla RNNs, they can still struggle with **very** long-range dependencies — the forget gate has to consistently learn to stay near 1 across many steps for information to survive, which isn't guaranteed. (3) LSTMs have roughly **4× the parameters** of a comparably-sized vanilla RNN cell (four separate gate weight matrices), adding real memory and compute cost. These limitations directly motivate the Attention mechanism, which lets a model directly access any earlier position in a sequence without routing information through a long chain of sequential updates.
</details>

`[🔝 Top](#dl-lecture-05--lstm-practice)`

---

## Concept Check — Fill in the Blank

1. LSTM adds a dedicated ______ (in addition to the hidden state) for relaying long-term memory.
2. The ______ gate decides what to erase from the old cell state.
3. The candidate memory content uses the ______ activation function, while all three gates use ______.
4. The cell state update rule, in the lecture's own words, is "cell state = (keep useful old info) + (______)".
5. Hochreiter and Schmidhuber's key insight was to make the feedback coefficient a function of the ______, instead of a fixed weight.

<details>
<summary>Show answers</summary>

1. cell state (C)
2. forget
3. tanh; sigmoid
4. add relevant new info
5. input
</details>

`[🔝 Top](#dl-lecture-05--lstm-practice)`

---

## Explain-It-Back Prompts

1. Explain the "sticky note vs notebook" analogy for hidden state vs cell state in your own words.
2. Walk through all six LSTM equations from memory, explaining what each computes.
3. Explain why an LSTM cell has roughly 4× the parameters of a vanilla RNN cell.
4. Explain, using Worked Example 4's numbers, why LSTMs are more gradient-stable than vanilla RNNs, without claiming they "completely solve" the problem.
5. Explain the two extreme forget-gate scenarios (topic continues vs topic changes) using your own example sentence.

`[🔝 Top](#dl-lecture-05--lstm-practice)`

---

## Quick-Fire True / False

1. LSTMs completely eliminate the vanishing gradient problem. — **False** (much less likely, not impossible).
2. The forget gate output of 0 means "keep everything." — **False** (0 means forget/discard everything; 1 means keep everything).
3. The cell state update combines a multiplicative "forget" term and an additive "write new info" term. — **True**.
4. All four LSTM weight matrices (forget, input, candidate, output) operate on the concatenation of the previous hidden state and current input. — **True**.
5. LSTMs can be fully parallelized across timesteps, just like CNNs across spatial positions. — **False** (still sequential).

`[🔝 Top](#dl-lecture-05--lstm-practice)`

---

## Gate Matching Drill

Match each gate/component to its correct activation function and purpose:

| Component | Activation | Purpose |
|---|---|---|
| Forget gate | ? | ? |
| Input gate | ? | ? |
| Candidate memory | ? | ? |
| Output gate | ? | ? |

<details>
<summary>Show answers</summary>

Forget gate → sigmoid → decides what to erase from old cell state. Input gate → sigmoid → decides how much new info to write. Candidate memory → tanh → proposes the actual new content (positive/negative/neutral). Output gate → sigmoid → decides how much of the (tanh-filtered) cell state to expose as hidden state.
</details>

`[🔝 Top](#dl-lecture-05--lstm-practice)`

---

## Mini Interview-Style Round

**Q1.** "A junior engineer says 'let's just use LSTMs everywhere instead of vanilla RNNs, no downside.' How do you respond?"

<details>
<summary>Show answer</summary>

You'd note there IS a real cost: LSTMs have roughly 4× the parameters of a comparably-sized vanilla RNN cell, meaning more memory usage and more compute per timestep. For short sequences where vanishing gradients aren't a practical concern, a vanilla RNN (or even a simpler architecture) might genuinely be sufficient and cheaper. LSTMs are the right default when you specifically expect long-range dependencies to matter, not an unconditionally strictly-better replacement.
</details>

**Q2.** "Explain why LSTM's cell state update being 'additive' matters for gradient flow, in terms someone who just learned about vanishing gradients would understand."

<details>
<summary>Show answer</summary>

In a vanilla RNN, gradient flowing backward through time has to pass through the SAME multiplicative weight matrix at every single step, and repeated multiplication by the same (fractional) number shrinks toward zero fast. In an LSTM, the cell state update is `C_t = f_t*C_{t-1} + i_t*C~_t` — the `+` here means gradient has a much more direct additive path backward through the cell state, only being scaled by the forget gate `f_t` at each step rather than by a completely separate learned weight matrix. If the network learns to keep `f_t` close to 1 for information that matters, the gradient can flow back through many timesteps with comparatively little decay — this is the mechanistic reason additive updates preserve gradient better than purely multiplicative ones.
</details>

**Q3.** "Someone asks you to justify, numerically, why 'less likely to vanish' doesn't mean 'never vanishes' for LSTMs."

<details>
<summary>Show answer</summary>

Point to Worked Example 4: if the forget gate learns to stay near 1 (e.g., 0.98), decay is slow (≈36% signal remaining after 50 steps). But nothing structurally prevents the forget gate from learning to stay near 0 for some sequences/tasks — if it does, the SAME fast-decay math that hurts vanilla RNNs would apply to the LSTM's cell state pathway too. The gating mechanism gives the network the *option* to preserve gradient flow; it doesn't force it to.
</details>

`[🔝 Top](#dl-lecture-05--lstm-practice)`

---

## Summary

This practice file drills Lecture 5's LSTM gating mechanics through active recall. The official in-class "what issues does LSTM have" question is answered with a reasoned three-part response (sequential computation, imperfect very-long-range handling, and added parameter cost) that connects forward to the course's next topic, Attention. A fill-in-the-blank check reinforces cell state terminology, gate names, activation function assignments, the cell-state-update wording, and Hochreiter & Schmidhuber's core insight. Five explain-it-back prompts push you to reproduce the sticky-note/notebook analogy, all six LSTM equations, the 4× parameter ratio, the gradient-stability numeric comparison, and the two forget-gate extremes in your own words. A quick-fire true/false round targets the most common LSTM misconceptions (especially "completely solves vanishing gradients" and gate-value direction confusion), and a dedicated gate-matching drill ties each component to its correct activation function and purpose. A three-question interview-style round rehearses realistic engineering judgment calls — when NOT to default to LSTM, why additive cell-state updates help gradient flow mechanistically, and why "less likely to vanish" is not the same claim as "never vanishes." Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture05_lstm_numerical.md) · [🔝 Top](#dl-lecture-05--lstm-practice) · [Next: Exercises →](../exercises/dl_lecture05_exercises.md)`
