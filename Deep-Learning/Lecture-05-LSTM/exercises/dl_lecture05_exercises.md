# DL Lecture 05 — Exercise Bank (LSTM)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-05--exercise-bank-lstm)`

> Folder: `Deep-Learning/Lecture-05-LSTM/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture05_lstm_theory.md) · [numerical](../numerical/dl_lecture05_lstm_numerical.md) · [practice](../practice/dl_lecture05_lstm_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q5.1.** Name the three gates in an LSTM cell.

**Q5.2.** Write the cell state update formula.

**Q5.3.** Which activation function do the gates use? Which does the candidate memory use?

**Q5.4.** What insight did Hochreiter and Schmidhuber propose to fix vanishing gradients?

**Q5.5.** True or False: LSTMs completely eliminate vanishing/exploding gradients.

---

## 🟡 Medium — Applied Reasoning

**Q5.6.** For D=6, H=12, compute the total LSTM parameter count (4 gates, ignoring biases).

**Q5.7.** Explain why the output gate's formula also depends on `H_{t-1}`, and what this implies about how the previous hidden state indirectly shapes the current one.

**Q5.8.** A forget gate outputs `f_t = 0.1` for a particular cell state dimension. Explain, in words, what this means is happening to that dimension's memory.

**Q5.9.** Explain the difference between "hidden state" and "cell state" using the notebook/sticky-note analogy, then explain which one is more analogous to a vanilla RNN's hidden state.

**Q5.10.** Why does an LSTM cell's parameter count not depend on sequence length, just like a vanilla RNN?

---

## 🔴 Hard — Derivation & Multi-Step

**Q5.11.** Given H=1, D=1, `H_{t-1}=0.1`, `C_{t-1}=1.0`, `x_t=0.5`, and weights: forget gate (w_h=0.4, w_x=0.6, b=0), input gate (w_h=0.3, w_x=0.7, b=0), candidate (w_h=0.5, w_x=0.5, b=0), output gate (w_h=0.2, w_x=0.8, b=0) — compute f_t, i_t, C~_t, C_t, o_t, and H_t, showing every step.

**Q5.12.** Using the gradient-decay comparison idea from the numerical file, compute the cumulative decay factor after 30 timesteps for a forget gate consistently at (a) 0.95 and (b) 0.99. Compare the two results.

**Q5.13.** Compute the exact LSTM-to-vanilla-RNN parameter ratio for D=50, H=50 (using the standard formulas, ignoring biases), and explain why this ratio is always exactly 4, regardless of D and H.

**Q5.14.** A cell state dimension has `C_{t-1}=10`. At timestep t, `f_t=0.3` and the input gate contributes `i_t*C~_t = -2.0` (a negative candidate value, e.g. representing "negative information" per the theory file). Compute `C_t`, and explain what a negative candidate value means conceptually.

`[🔝 Top](#dl-lecture-05--exercise-bank-lstm)`

---

## Answer Key

<details>
<summary>Q5.1 – Q5.5 (Easy)</summary>

- **Q5.1:** Forget gate, Input gate, Output gate.
- **Q5.2:** `C_t = f_t * C_{t-1} + i_t * C~_t`.
- **Q5.3:** Gates use sigmoid; candidate memory uses tanh.
- **Q5.4:** Make the feedback/carry-over coefficient a function of the input, instead of always multiplying by the same fixed weight W at every timestep.
- **Q5.5:** False — they're much less likely, but can still occur.
</details>

<details>
<summary>Q5.6 – Q5.10 (Medium)</summary>

- **Q5.6:** Params per gate = H×(H+D) = 12×(12+6) = 12×18 = 216. Total = 4×216 = **864**.
- **Q5.7:** The output gate's formula `o_t = σ(W_o·[H_{t-1},x_t]+b_o)` includes `H_{t-1}` as an input, meaning the previous hidden state influences how much of the CURRENT (tanh-filtered) cell state gets exposed. This creates an indirect chain: `H_{t-1}` → affects `o_t` → affects `H_t` (since `H_t = o_t * tanh(C_t)`) — so even though `H_t`'s main content comes from the cell state, its previous value still shapes the current one through the output gate.
- **Q5.8:** A forget gate of 0.1 means about 90% of that dimension's previous memory content gets discarded — the network has decided this particular piece of information is now largely irrelevant (similar to Worked Example 3's Scenario B, "the conversation has shifted").
- **Q5.9:** The hidden state is like the sticky note — small, frequently rewritten, short-term summary exposed at every step (this is the vanilla-RNN-equivalent quantity — a vanilla RNN only has this one memory). The cell state is like the notebook — a more durable, carefully-edited long-term memory that an LSTM adds on top, which a vanilla RNN simply does not have.
- **Q5.10:** Because the same four weight matrices (W_f, W_i, W_C, W_o) are reused identically at every timestep, exactly like a vanilla RNN's U and W — adding more timesteps to a sequence adds more computation but zero additional parameters.
</details>

<details>
<summary>Q5.11 – Q5.14 (Hard)</summary>

- **Q5.11:** z_f = 0.4×0.1+0.6×0.5 = 0.04+0.30 = 0.34 → f_t=σ(0.34)≈0.5842. z_i = 0.3×0.1+0.7×0.5 = 0.03+0.35 = 0.38 → i_t=σ(0.38)≈0.5939. z_c = 0.5×0.1+0.5×0.5 = 0.05+0.25 = 0.30 → C~_t=tanh(0.30)≈0.2913. C_t = 0.5842×1.0 + 0.5939×0.2913 ≈ 0.5842+0.1730 = **0.7572**. z_o = 0.2×0.1+0.8×0.5 = 0.02+0.40 = 0.42 → o_t=σ(0.42)≈0.6034. H_t = 0.6034×tanh(0.7572) ≈ 0.6034×0.6398 ≈ **0.3861**.
- **Q5.12:** (a) 0.95^30 ≈ 0.2146. (b) 0.99^30 ≈ 0.7397. The 0.99 forget gate retains over 3.4× more signal after the same 30 timesteps than the 0.95 forget gate — even a small difference in how close the forget gate stays to 1 has a large compounding effect over many timesteps.
- **Q5.13:** LSTM params = 4×H×(H+D) = 4×50×100 = 20,000. Vanilla RNN params = H×D+H×H = 50×50+50×50 = 2,500+2,500=5,000. Ratio = 20,000/5,000 = **4.0**. This ratio is always exactly 4 because both the LSTM's per-gate term `H×(H+D)` and the vanilla RNN's `H×D+H×H` terms expand to the identical quantity `H×(H+D)` — the LSTM simply has four independent copies of that same quantity (one per gate) versus the vanilla RNN's one copy, so the ratio is always 4 regardless of D and H.
- **Q5.14:** C_t = f_t×C_{t-1} + (i_t×C~_t) = 0.3×10 + (-2.0) = 3.0 - 2.0 = **1.0**. A negative candidate value means the new information being written actively pushes that memory dimension DOWN/toward a more negative value, rather than reinforcing or neutrally leaving it — e.g., if that dimension tracked "how positive is the sentiment so far," a negative candidate would represent newly-encountered negative-sentiment content actively being incorporated.
</details>

`[🔝 Top](#dl-lecture-05--exercise-bank-lstm)`

---

## Summary

This exercise bank drills Lecture 5's LSTM gating formulas across three tiers. Easy questions recall the three gates, the cell state update formula, activation function assignments, Hochreiter & Schmidhuber's core insight, and the "does LSTM completely solve vanishing gradients" trap. Medium questions apply the parameter formula to new dimensions (864 total for D=6/H=12), explain the output gate's dependency on the previous hidden state, interpret a low forget-gate value, connect the notebook/sticky-note analogy to which memory corresponds to a vanilla RNN's hidden state, and explain sequence-length-independent parameter counts. Hard questions require full derivations: a complete six-quantity hand-computed LSTM timestep (f_t≈0.584, i_t≈0.594, C~_t≈0.291, C_t≈0.757, o_t≈0.603, H_t≈0.386), a comparative decay calculation showing a 0.99 forget gate retains over 3.4× more signal than a 0.95 forget gate after 30 timesteps, an algebraic proof that the LSTM-to-vanilla-RNN parameter ratio is always exactly 4 regardless of D and H, and a cell-state update involving a negative candidate value with conceptual interpretation. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture05_lstm_practice.md) · [🔝 Top](#dl-lecture-05--exercise-bank-lstm) · [Code →](../code/README.md)`
