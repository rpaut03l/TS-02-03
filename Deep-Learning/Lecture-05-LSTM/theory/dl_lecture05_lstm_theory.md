# DL Lecture 05 — LSTM (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-05--lstm-theory)`

> Folder: `Deep-Learning/Lecture-05-LSTM/theory/`
> Pairs with: [`numerical/dl_lecture05_lstm_numerical.md`](../numerical/dl_lecture05_lstm_numerical.md) · [`practice/dl_lecture05_lstm_practice.md`](../practice/dl_lecture05_lstm_practice.md) · [`exercises/dl_lecture05_exercises.md`](../exercises/dl_lecture05_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "LSTM" deck, parts 1 & 2 (combined here)

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Recap: Why Vanilla RNNs Struggle](#recap-why-vanilla-rnns-struggle)
3. [Hochreiter & Schmidhuber's Key Insight](#hochreiter--schmidhubers-key-insight)
4. [LSTM's Two Memories: Cell State and Hidden State](#lstms-two-memories-cell-state-and-hidden-state)
5. [Step-by-Step LSTM Walkthrough](#step-by-step-lstm-walkthrough)
6. [The Forget Gate](#the-forget-gate)
7. [The Input Gate](#the-input-gate)
8. [Updating the Cell State](#updating-the-cell-state)
9. [The Output Gate and Hidden State](#the-output-gate-and-hidden-state)
10. [All Four LSTM Equations, Together](#all-four-lstm-equations-together)
11. [LSTMs vs RNNs: Gradient Stability](#lstms-vs-rnns-gradient-stability)
12. [Known Issues With LSTMs](#known-issues-with-lstms)
13. [LSTMs for Sequential Modelling — Usage Patterns](#lstms-for-sequential-modelling--usage-patterns)
14. [Mnemonics](#mnemonics)
15. [Cheatsheet](#cheatsheet)
16. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
17. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine you're taking notes during a long meeting, but instead of an infinite notebook, you only have one small sticky note that you keep erasing and rewriting after every single sentence (this is a vanilla RNN's hidden state — perpetually rewritten). By the end of a two-hour meeting, almost everything from the first ten minutes is long gone, overwritten many times over. Now imagine instead you're given a proper notebook (a **dedicated memory cell**) alongside your sticky note. After every sentence, you make three small decisions: (1) *should I cross out anything in my notebook that's no longer relevant?* (the **forget gate**), (2) *should I write any new important point into my notebook?* (the **input gate**), and (3) *based on what's in my notebook right now, what should I actually say out loud as my current summary?* (the **output gate**, producing the hidden state). This three-decision, notebook-plus-sticky-note system is exactly what an **LSTM (Long Short-Term Memory)** network is.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Recap: Why Vanilla RNNs Struggle

From Lecture 4: RNNs help capture context while avoiding the sparsity, storage, and compute issues of, say, keeping a full lookup table of every possible sequence. The hidden layer is what we ultimately care about — it represents each word's "meaning" in context, updated as the sequence is read. But vanilla RNNs suffer from the **vanishing gradient problem**: because the *same* recurrent weight matrix W is multiplied in at every timestep, long-term dependencies get progressively harder to learn as the gap between relevant information grows.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Hochreiter & Schmidhuber's Key Insight

The original fix for the vanishing gradient problem, proposed by Hochreiter and Schmidhuber (who introduced LSTMs): **instead of multiplying by the same fixed weight W at every timestep, make the feedback/carry-over coefficient a *function of the input* instead of a fixed constant.** In other words, rather than a rigid "always multiply memory by this same number every step" rule (which either vanishes or explodes depending on whether that number is <1 or >1), let the network **learn, at every timestep, how much of its memory to keep versus discard**, based on the actual content it's currently seeing. This single insight — making the memory-carry coefficient *data-dependent* rather than fixed — is the seed that grows into the full gating mechanism covered below.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## LSTM's Two Memories: Cell State and Hidden State

An LSTM is a type of RNN specifically designed to better handle long-range dependencies. The key structural change: in addition to the traditional hidden state **h** (short-term, constantly-refreshed memory, same as vanilla RNNs), an LSTM adds a **dedicated memory cell, c** (cell state), specifically built to relay long-term information across many timesteps with much less interference. Think of **C** (cell state) as relaying **long-term memory**, and **H** (hidden state) as relaying **short-term memory** — both flow forward to the next timestep, and inside each timestep's "cell," a number of learned weights (via the gates) control exactly how these two memories interact and update.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Step-by-Step LSTM Walkthrough

Following the classic colah's-blog-style diagram referenced in the lecture, at every timestep an LSTM cell takes in three things — the previous cell state `C_{t-1}`, the previous hidden state `H_{t-1}`, and the current input `x_t` — and produces two things: a new cell state `C_t` and a new hidden state `H_t`. Internally, this happens through **three gates**, each a small neural network layer (with a **sigmoid** activation, squashing outputs to between 0 and 1) that decides "how much" of something to let through — a gate value of **0 means block everything, 1 means let everything through**.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## The Forget Gate

**Purpose:** decide what to throw away from the old cell state. Imagine the cell state currently holds information related to a previous topic, but the conversation has shifted — the forget gate looks at the previous hidden state `H_{t-1}` and the current input `x_t`, and outputs a number between 0 and 1 for *each* value in the cell state: **0 → forget everything at that position, 1 → keep everything at that position.** This is a sigmoid-activated layer, since sigmoid naturally squashes any input into exactly the (0,1) range needed for a "how much to keep" decision.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## The Input Gate

**Purpose:** decide what new information to write into the cell state. This has two parts: (1) a sigmoid layer decides **which values to update** (again, 0-to-1 "how much"), and (2) a tanh layer creates a vector of **candidate new memory content** — which can be positive information, negative information, or effectively zero/neutral information (since tanh outputs between -1 and 1). The model combines these to decide **how much of this new candidate information actually gets written** into the cell state.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Updating the Cell State

Now combine the forget gate and input gate's decisions into the actual memory update, captured perfectly by the lecture's own one-line summary:

```
Cell state = (keep useful old info) + (add relevant new info)
```

Concretely: multiply the old cell state by the forget gate's output (this erases whatever the forget gate decided to drop), then add the input gate's newly-written information on top. This additive (not purely multiplicative-through-many-layers) update is precisely *why* LSTMs are so much better at preserving gradient signal across long sequences — more on this in the gradient stability section below.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## The Output Gate and Hidden State

**Purpose:** decide what to actually "say out loud" as the current hidden state, based on the (now-updated) cell state. First, the cell state is squashed through **tanh** (producing the "filtered cell state," `tanh(C_t)`, forcing values back into a bounded -1 to 1 range). Then, a sigmoid **output gate** `o_t` decides how much of that filtered cell state to actually expose as the hidden state. The final hidden state is the elementwise product: `H_t = o_t × tanh(C_t)` — a "weighted version" of the filtered cell state. Note that the *previous* hidden state `H_{t-1}` also influences `o_t` (since the output gate itself looks at `H_{t-1}` and `x_t`), so the recurrent hidden layer indirectly shapes the current hidden state through this chain.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## All Four LSTM Equations, Together

Putting every gate together, using the standard notation (concatenating `[H_{t-1}, x_t]` as the combined input to each gate's weight matrix):

```
Forget gate:    f_t = sigmoid( W_f . [H_{t-1}, x_t] + b_f )
Input gate:     i_t = sigmoid( W_i . [H_{t-1}, x_t] + b_i )
Candidate mem:  C~_t = tanh( W_C . [H_{t-1}, x_t] + b_C )
Cell state:     C_t = f_t * C_{t-1} + i_t * C~_t
Output gate:    o_t = sigmoid( W_o . [H_{t-1}, x_t] + b_o )
Hidden state:   H_t = o_t * tanh(C_t)
```

(`*` here denotes elementwise multiplication, not matrix multiplication.) Four separate weight matrices (`W_f, W_i, W_C, W_o`) exist, one per gate/candidate — this is the main reason an LSTM cell has roughly **4×** the parameters of a comparably-sized vanilla RNN cell for the same hidden size (verified precisely in the numerical file).

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## LSTMs vs RNNs: Gradient Stability

Vanishing/exploding gradients **can still occur** in LSTMs, but are **much less likely** than in vanilla RNNs. Understanding why requires contrasting the two designs:

**Issue in vanilla RNNs:** long-term memory depends entirely on repeatedly multiplying by the same recurrent weight matrix W at every timestep. This W must be delicately balanced — too small and gradients vanish, too large and gradients explode — making stable learning over long sequences genuinely hard.

**Why LSTMs are more stable:** instead of one fixed multiplicative pathway, LSTMs use **gating mechanisms to explicitly control information flow**: the forget gate decides what past information to retain or discard, the input gate controls how much new information gets added, and the output gate controls how much internal information gets exposed. Because the cell state update is **additive** (`C_t = f_t*C_{t-1} + i_t*C~_t`) rather than being repeatedly passed through a saturating non-linearity and multiplied by the same weight every step, gradient signal has a much more direct, less-obstructed path backward through time — the gates *learn* when to let gradient flow through largely unimpeded (when `f_t≈1`) versus when to actively cut it off (when `f_t≈0`), rather than this behaviour being a rigid, fixed side-effect of one shared weight matrix. The lecture summarizes this as: **"the cell learns an operative time to 'turn on'"** — i.e., the network learns exactly when memory should persist strongly versus reset, rather than that behaviour being baked in uniformly for all timesteps.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Known Issues With LSTMs

The lecture poses this as an open question ("can you guess some issues with LSTM?") rather than listing exhaustive answers on the slide itself — but based on the material's own trajectory (into Attention in a later lecture), the implied issues include: LSTMs are still fundamentally **sequential** (each timestep must wait for the previous one to finish, unlike CNNs which can process spatial positions in parallel), making them slow to train on long sequences and hard to parallelize; despite being far more robust than vanilla RNNs, they can still struggle with *very* long-range dependencies in practice; and the four-gates-plus-candidate design adds significant extra parameters and computation per cell compared to a vanilla RNN cell. These exact limitations are a major part of the motivation for the Attention mechanism covered later in this course, which allows a model to directly access *any* earlier position in a sequence without having to route information through a long chain of sequential memory updates at all.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## LSTMs for Sequential Modelling — Usage Patterns

Exactly like vanilla RNNs (Lecture 4's architecture types), LSTMs support multiple input/output patterns:
- **Many-to-One (classification):** feed in an entire sequence, read out a single label at the end (e.g., sentiment classification of a whole sentence/review).
- **Many-to-Many:** feed in a sequence, read out another sequence — either aligned (one output per input timestep) or offset (e.g., sequence-to-sequence translation, where the entire input is read before output generation begins).

The internal gating mechanics described above are identical regardless of which usage pattern you apply the LSTM to — what changes is simply which timesteps you attach a "read out an output" step to.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Mnemonics

- **"Sticky note vs notebook"** → hidden state (short-term, constantly rewritten) vs cell state (long-term, carefully edited).
- **"FIO: Forget, Input, Output"** → the three gates, in the order they conceptually act within one timestep.
- **"0 = block, 1 = let through"** → what every sigmoid gate output means.
- **"Cell state = keep useful old + add relevant new"** → the one-line cell state update rule, verbatim from the lecture.
- **"The cell learns WHEN to turn on"** → why LSTMs are more gradient-stable — the behaviour is learned per-input, not fixed per-architecture.
- **4 gates → roughly 4x the parameters of a vanilla RNN cell** → a quick sanity-check number for parameter questions.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Cheatsheet

| Gate/Component | Formula | Purpose |
|---|---|---|
| Forget gate | `f_t = σ(W_f·[H_{t-1},x_t]+b_f)` | What to erase from old cell state |
| Input gate | `i_t = σ(W_i·[H_{t-1},x_t]+b_i)` | How much new info to write |
| Candidate memory | `C~_t = tanh(W_C·[H_{t-1},x_t]+b_C)` | The actual new content proposed |
| Cell state update | `C_t = f_t*C_{t-1} + i_t*C~_t` | Keep useful old + add relevant new |
| Output gate | `o_t = σ(W_o·[H_{t-1},x_t]+b_o)` | How much of cell state to expose |
| Hidden state | `H_t = o_t * tanh(C_t)` | The exposed, filtered summary |

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting that gate outputs use **sigmoid** (range 0–1, "how much") while the candidate memory uses **tanh** (range -1 to 1, "what content") — mixing these up is one of the most common LSTM exam mistakes.
- **Trap:** writing the cell state update as purely additive or purely multiplicative — it's genuinely both: `f_t * C_{t-1}` (multiplicative forgetting) **plus** `i_t * C~_t` (multiplicative-then-additive writing).
- **Trap:** claiming LSTMs completely solve vanishing/exploding gradients — the lecture explicitly states these can *still* occur in LSTMs, just much less likely than in vanilla RNNs. Never claim "completely solved" in an exam answer.
- **Exam hack:** if asked "why are LSTMs more stable than vanilla RNNs," always name the *mechanism* (gates make the memory-carry coefficient data-dependent/learned, rather than a single fixed weight applied identically every step) rather than just asserting "they have gates."
- **Exam hack:** the four-equation block (forget, input, candidate, cell update, output, hidden) is very likely to be tested as a "label this diagram" or "fill in the missing equation" question — practice writing all four/six equations from memory, in order, without looking.

`[🔝 Top](#dl-lecture-05--lstm-theory)`

---

## Summary

LSTMs (Long Short-Term Memory networks) directly address vanilla RNNs' vanishing gradient problem by replacing the single fixed recurrent weight matrix (which must be delicately balanced to avoid vanishing or exploding gradients) with a learned, data-dependent gating system, following Hochreiter and Schmidhuber's key insight of making the memory-carry coefficient a function of the input rather than a fixed constant. Structurally, an LSTM adds a dedicated cell state **C** (relaying long-term memory) alongside the traditional hidden state **H** (relaying short-term memory), and uses three sigmoid-activated gates to control information flow at every timestep: the **forget gate** decides what to erase from the old cell state, the **input gate** (paired with a tanh-activated candidate memory vector) decides what new information to write in, and the **output gate** decides how much of the (tanh-filtered) cell state to expose as the new hidden state. The cell state update follows the memorable rule "keep useful old info + add relevant new info" (`C_t = f_t*C_{t-1} + i_t*C~_t`), and this largely-additive pathway is precisely why gradients can flow backward through many timesteps far more robustly than in vanilla RNNs — though the lecture is careful to note vanishing/exploding gradients can still occur in LSTMs, just much less often. LSTMs are described as learning "when to turn on" — i.e., the gating behaviour itself is learned per-input rather than being a fixed architectural property. Despite being a major improvement, LSTMs still have real limitations (fundamentally sequential computation, added parameter/compute cost from four separate weight matrices, and still-imperfect handling of very long-range dependencies), setting up the motivation for the Attention mechanism covered in a later lecture. Like vanilla RNNs, LSTMs support flexible usage patterns including many-to-one (classification) and many-to-many (sequence-to-sequence) modelling.

`[← Lecture 04](../../Lecture-04-Recurrent-Neural-Networks/README.md) · [🔝 Top](#dl-lecture-05--lstm-theory) · [Next: Numerical →](../numerical/dl_lecture05_lstm_numerical.md)`
