# DL Lecture 05 — LSTM (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-05--lstm-numerical)`

> Folder: `Deep-Learning/Lecture-05-LSTM/numerical/`
> Pairs with: [`theory/dl_lecture05_lstm_theory.md`](../theory/dl_lecture05_lstm_theory.md) · [`practice/dl_lecture05_lstm_practice.md`](../practice/dl_lecture05_lstm_practice.md) · [`exercises/dl_lecture05_exercises.md`](../exercises/dl_lecture05_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — LSTM Parameter Count vs Vanilla RNN](#worked-example-1--lstm-parameter-count-vs-vanilla-rnn)
3. [Worked Example 2 — Full LSTM Cell, One Timestep, By Hand](#worked-example-2--full-lstm-cell-one-timestep-by-hand)
4. [Worked Example 3 — The Forget Gate in Action (Two Extremes)](#worked-example-3--the-forget-gate-in-action-two-extremes)
5. [Worked Example 4 — Gradient Path Comparison, RNN vs LSTM](#worked-example-4--gradient-path-comparison-rnn-vs-lstm)
6. [Master Formula Cheatsheet](#master-formula-cheatsheet)
7. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
8. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| D | input size |
| H | hidden state size (= cell state size) |
| σ | sigmoid activation, `σ(z)=1/(1+e^-z)` |
| f_t, i_t, o_t | forget, input, output gate outputs at time t |
| C~_t | candidate memory content at time t |
| C_t | cell state at time t |
| H_t | hidden state at time t |

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Worked Example 1 — LSTM Parameter Count vs Vanilla RNN

**Given:** D=10 (input size), H=20 (hidden/cell state size). Recall from Lecture 4's numerical file, a vanilla RNN's hidden-update weights are `U (H×D)` and `W (H×H)`.

**Step 1 — Vanilla RNN hidden-update parameters (U and W only, ignoring output V and biases for a fair comparison).**
```
U params = H x D = 20 x 10 = 200
W params = H x H = 20 x 20 = 400
Vanilla RNN total = 200 + 400 = 600
```

**Step 2 — LSTM parameters.** Each of the four components (forget gate, input gate, candidate memory, output gate) has its OWN weight matrix operating on the concatenated `[H_{t-1}, x_t]` (size H+D):
```
Params per gate/component = H x (H + D) = 20 x (20+10) = 20 x 30 = 600
Four such components (f, i, C~, o):
Total = 4 x 600 = 2,400
```

**Step 3 — Compare.**
```
LSTM / Vanilla RNN ratio = 2,400 / 600 = 4.0
```

**Result: an LSTM cell has exactly 4× the parameters of a comparably-sized vanilla RNN cell** (ignoring biases in both) — directly confirming the theory file's "roughly 4×" claim with an exact number for this configuration. Adding one bias vector (size H) per component adds `4×20=80` more parameters to the LSTM and `20` more to the vanilla RNN's hidden update, a small correction that doesn't change the overall ~4× conclusion.

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Worked Example 2 — Full LSTM Cell, One Timestep, By Hand

**Given (toy 1-dimensional example, H=1, D=1, for full hand-tractability):** previous cell state `C_{t-1}=0.5`, previous hidden state `H_{t-1}=0.2`, current input `x_t=1.0`. Illustrative weights (each gate: weight on H_{t-1}, weight on x_t, bias):
```
Forget gate:   w_fh=0.5, w_fx=0.3, b_f=0.0
Input gate:    w_ih=0.6, w_ix=0.4, b_i=0.0
Candidate:     w_ch=0.4, w_cx=0.5, b_c=0.0
Output gate:   w_oh=0.3, w_ox=0.6, b_o=0.0
```

**Step 1 — Forget gate.**
```
z_f = w_fh*H_{t-1} + w_fx*x_t + b_f = 0.5x0.2 + 0.3x1.0 + 0 = 0.10 + 0.30 = 0.40
f_t = sigmoid(0.40) = 1/(1+e^-0.40) = 1/(1+0.6703) = 1/1.6703 ≈ 0.5987
```

**Step 2 — Input gate.**
```
z_i = w_ih*H_{t-1} + w_ix*x_t + b_i = 0.6x0.2 + 0.4x1.0 + 0 = 0.12 + 0.40 = 0.52
i_t = sigmoid(0.52) = 1/(1+e^-0.52) = 1/(1+0.5945) = 1/1.5945 ≈ 0.6272
```

**Step 3 — Candidate memory.**
```
z_c = w_ch*H_{t-1} + w_cx*x_t + b_c = 0.4x0.2 + 0.5x1.0 + 0 = 0.08 + 0.50 = 0.58
C~_t = tanh(0.58) ≈ 0.5227
```

**Step 4 — Update the cell state.**
```
C_t = f_t x C_{t-1} + i_t x C~_t
    = 0.5987 x 0.5 + 0.6272 x 0.5227
    = 0.29935 + 0.327886
    ≈ 0.6272
```

**Step 5 — Output gate.**
```
z_o = w_oh*H_{t-1} + w_ox*x_t + b_o = 0.3x0.2 + 0.6x1.0 + 0 = 0.06 + 0.60 = 0.66
o_t = sigmoid(0.66) = 1/(1+e^-0.66) = 1/(1+0.5169) = 1/1.5169 ≈ 0.6593
```

**Step 6 — New hidden state.**
```
H_t = o_t x tanh(C_t) = 0.6593 x tanh(0.6272) = 0.6593 x 0.5561 ≈ 0.3667
```

**Result: C_t ≈ 0.6272, H_t ≈ 0.3667.** Notice how the cell state grew slightly (from 0.5 to 0.6272) because the forget gate (0.5987) kept a majority of the old memory while the input gate (0.6272) wrote in a meaningful chunk of new candidate content — exactly the "keep useful old + add relevant new" rule from the theory file, now fully quantified.

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Worked Example 3 — The Forget Gate in Action (Two Extremes)

**Scenario A — Forget gate ≈ 1 (topic continues).** Say `f_t = 0.98`, `C_{t-1}=4.0`, and the input gate contributes very little new information, `i_t*C~_t ≈ 0.05`.
```
C_t = 0.98 x 4.0 + 0.05 = 3.92 + 0.05 = 3.97
```
Almost all of the old memory (3.97 out of 4.0, about 99.25%) survives — the network has decided this information is still relevant.

**Scenario B — Forget gate ≈ 0 (topic completely changes).** Say `f_t = 0.02`, same `C_{t-1}=4.0`, and this time the input gate writes strongly, `i_t*C~_t ≈ 2.5`.
```
C_t = 0.02 x 4.0 + 2.5 = 0.08 + 2.5 = 2.58
```
Almost all of the old memory is wiped (only 0.08 out of the original 4.0 remains, about 2%), and the cell state is now dominated by the freshly-written information — exactly modelling "the conversation has shifted" from the theory file's forget-gate story.

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Worked Example 4 — Gradient Path Comparison, RNN vs LSTM

**Vanilla RNN (from Lecture 4's numerical file):** with a per-timestep multiplicative factor of 0.5, after 20 timesteps: `0.5^20 ≈ 0.00000095` — gradient nearly vanished.

**LSTM cell state pathway:** if the forget gate stays close to 1 (say, `f_t ≈ 0.98` consistently, because the network has learned this information remains relevant), the equivalent multiplicative decay after 20 timesteps is:
```
0.98^20 ≈ 0.6676
```

**Comparison:**

| Timesteps back | Vanilla RNN (factor 0.5) | LSTM cell state (factor 0.98) |
|---|---|---|
| 5 | 0.03125 | 0.9039 |
| 10 | 0.0009766 | 0.8171 |
| 20 | 0.00000095 | 0.6676 |
| 50 | ≈0 (effectively vanished) | 0.3642 |

**Result:** even after 50 timesteps, an LSTM whose forget gate has learned to stay near 1 for relevant information retains about **36%** of its original gradient signal, compared to a vanilla RNN's gradient having vanished to effectively **zero** by that point. This is the precise numeric backbone of "vanishing/exploding gradients can still occur in LSTMs, but are much less likely" — the LSTM doesn't make vanishing *impossible* (a forget gate that learns to stay near 0 would still cause fast decay), but gives the network the *learned option* to keep the pathway close to 1 when it matters.

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Forget gate | `f_t = σ(W_f·[H_{t-1},x_t]+b_f)` |
| Input gate | `i_t = σ(W_i·[H_{t-1},x_t]+b_i)` |
| Candidate memory | `C~_t = tanh(W_C·[H_{t-1},x_t]+b_C)` |
| Cell state update | `C_t = f_t*C_{t-1} + i_t*C~_t` |
| Output gate | `o_t = σ(W_o·[H_{t-1},x_t]+b_o)` |
| Hidden state | `H_t = o_t * tanh(C_t)` |
| LSTM params (per gate) | `H × (H+D) [+ H bias]` |
| LSTM total params | `4 × H × (H+D) [+ 4H bias]` |
| Vanilla RNN params (for comparison) | `H×D + H×H` |

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** using `H×D` for a gate's parameter count instead of `H×(H+D)` — every LSTM gate takes the CONCATENATION of the previous hidden state AND the current input, so its input dimension is `H+D`, not just `D`.
- **Trap:** forgetting there are FOUR separate weight matrices (forget, input, candidate, output) — a common mistake is computing only one gate's parameters and forgetting to multiply by 4.
- **Trap:** applying tanh where sigmoid belongs (or vice versa) in the hand-computed forward pass — always double check: forget/input/output gates → sigmoid; candidate memory and the final cell-state filtering → tanh.
- **Exam hack:** the ~4× parameter ratio between LSTM and vanilla RNN cells is a very testable, memorable number — always be ready to derive it, not just state it.
- **Exam hack:** for "explain why LSTMs are more gradient-stable" numerical questions, always contrast a near-1 forget gate's slow decay (e.g., 0.98^n) against a vanilla RNN's much faster decay (e.g., 0.5^n) using actual computed numbers, exactly as done in Worked Example 4 — bare qualitative answers lose marks versus quantified ones.

`[🔝 Top](#dl-lecture-05--lstm-numerical)`

---

## Summary

This file worked every LSTM formula from the theory file into fully shown arithmetic. Comparing parameter counts for D=10, H=20 showed a vanilla RNN's hidden-update needs 600 parameters (U+W) while an LSTM needs 2,400 (four gates, each `H×(H+D)`), an exact 4× ratio. A complete hand-computed single-timestep LSTM forward pass (H=D=1 for tractability) walked through all four gates — forget (≈0.599), input (≈0.627), candidate memory (≈0.523), and output (≈0.659) — producing an updated cell state C_t≈0.627 and hidden state H_t≈0.367, concretely demonstrating the "keep useful old info + add relevant new info" rule. Two extreme forget-gate scenarios (near-1 "topic continues" retaining ~99% of old memory, and near-0 "topic changes" wiping ~98% of old memory) showed the forget gate's real behavioural range. Finally, a direct gradient-path comparison contrasted a vanilla RNN's factor-0.5-per-step decay (vanished to near-zero by 20 timesteps) against an LSTM cell state pathway with a factor-0.98-per-step forget gate (still retaining about 36% of its signal after 50 timesteps) — the precise quantitative reason LSTMs, while not immune to vanishing/exploding gradients, are dramatically more robust to them than vanilla RNNs.

`[← Theory](../theory/dl_lecture05_lstm_theory.md) · [🔝 Top](#dl-lecture-05--lstm-numerical) · [Next: Practice →](../practice/dl_lecture05_lstm_practice.md)`
