# DL Lecture 06 — Attention (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-06--attention-theory)`

> Folder: `Deep-Learning/Lecture-06-Attention/theory/`
> Pairs with: [`numerical/dl_lecture06_attention_numerical.md`](../numerical/dl_lecture06_attention_numerical.md) · [`practice/dl_lecture06_attention_practice.md`](../practice/dl_lecture06_attention_practice.md) · [`exercises/dl_lecture06_exercises.md`](../exercises/dl_lecture06_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "Attention" deck

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [The Many-to-Many Problem](#the-many-to-many-problem)
3. [Encoder-Decoder RNNs](#encoder-decoder-rnns)
4. [The Fixed-Length Bottleneck Problem](#the-fixed-length-bottleneck-problem)
5. [First (Flawed) Fix: Averaging All Hidden States](#first-flawed-fix-averaging-all-hidden-states)
6. [The Real Solution: A Different Weighted Sum Per Output Step](#the-real-solution-a-different-weighted-sum-per-output-step)
7. [Computing Attention Weights, Step by Step](#computing-attention-weights-step-by-step)
8. [Walking Through Attention-Based Decoding](#walking-through-attention-based-decoding)
9. [Self-Attention: Query, Key, Value](#self-attention-query-key-value)
10. [Why Multiply Query by Key? Why Multiply by Value?](#why-multiply-query-by-key-why-multiply-by-value)
11. [Mnemonics](#mnemonics)
12. [Cheatsheet](#cheatsheet)
13. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
14. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine translating a long paragraph from English to German, but you're only allowed to read the ENTIRE English paragraph once, then close the book, and write the entire German translation purely from a single one-sentence summary you jotted down. You'd inevitably lose details — names, specific word choices, subtle nuances — because one summary sentence simply cannot hold everything a whole paragraph contains. Now imagine instead you're allowed to keep the English paragraph open on your desk, and every time you write a new German word, you're allowed to **glance back** at whichever specific English words are most relevant to that particular word you're writing right now. That glancing-back-at-exactly-what's-relevant behaviour is **Attention** — a mechanism that lets a decoder look directly at every input position, weighing each one by how relevant it is to the current output being generated, instead of being forced to work from one overloaded, fixed-size summary.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## The Many-to-Many Problem

Recall from Lecture 4/5: **Many-to-Many** tasks (like translation) take in a full input sequence and produce a full output sequence. The classic approach: first process the input and generate a hidden representation for it, then use that representation to generate an output. **The problem:** in a naive setup, each output word depends only on the *current* hidden state, and not directly on previous outputs or a rich view of the input — this leads to poor long-sequence performance, since so much has to be compressed into so little.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Encoder-Decoder RNNs

The standard architecture for sequence-to-sequence tasks:
- **Encoder:** an RNN (often an LSTM) that reads the entire input sequence and compresses it into a final hidden representation. The input sequence ends with an explicit **`<eos>`** (end of sequence) token, and the final hidden state at that point acts as a compact summary of the entire input sequence.
- **Decoder:** a second RNN that uses this final encoder hidden state as its *initial* state (and often an initial `<sos>`/start symbol) to generate a sequence of outputs, one at a time. Each predicted output word is fed back in as the input for the *next* decoding step, and generation continues until the decoder itself produces an `<eos>` token.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## The Fixed-Length Bottleneck Problem

**The core issue with plain encoder-decoder RNNs:** the *entire* input sequence gets compressed into a **single, fixed-length vector** (the final encoder hidden state). This one vector acts as the compact summary carrying ALL essential information from the whole input — and it becomes badly **"overloaded"** with information, particularly as the input sequence grows longer. In reality, *each* hidden state along the way stores only *partial* information about the sequence, and relying purely on the very last one means a lot of that partial information from earlier in the sequence can be lost by the time you reach the end.

**A deeper issue:** each output word typically depends on *specific* parts of the input, not the whole thing equally. Using only the single final encoder state ignores this direct input-output alignment entirely, leading to loss of important details needed for accurate generation — e.g., when translating "I ate an apple" to German, the output word "Ich" most directly corresponds to the input word "I," and "gegessen" most directly corresponds to "ate" — but a single fixed summary vector has no way to represent *which* input word matters most for *which* output word.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## First (Flawed) Fix: Averaging All Hidden States

A tempting first idea: instead of using only the *final* encoder hidden state, take the **average** of ALL encoder hidden states (a "global context"), and feed that same average into every stage of the decoder.

**Why this fails:** it still gives **all inputs equal importance** — no distinction between which words matter more for which output. The *same* averaged context gets reused for every single output step, even though different outputs genuinely need different inputs (as in the "Ich"↔"I", "habe/gegessen"↔"ate" example). Because the model cannot focus on the specific words that are relevant to each output, input-output alignment stays weak. **This exact failure is what directly motivates Attention.**

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## The Real Solution: A Different Weighted Sum Per Output Step

Instead of a single fixed vector OR a single fixed average, use a **different weighted sum of encoder hidden states for each output step**. For generating the 0th output word, assign one set of importance weights across all input words; for generating the 1st output word, assign a *different* set of importance weights — dynamically recomputed at every decoding step, based on what's currently relevant. This dynamically-weighted-sum idea is exactly what "Attention" refers to.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Computing Attention Weights, Step by Step

**Requirement:** the weights must be positive and sum to exactly 1.0 (i.e., form a proper probability distribution) — ideally high for the most relevant input positions for the current output, and low everywhere else.

**The two-step recipe:**
1. **Compute raw scores** — a function of the decoder's current state and each encoder hidden state; these raw scores can be positive or negative, with no constraint yet.
2. **Apply softmax** — converts the raw scores into a proper probability distribution (all positive, summing to 1.0).

Formally, for decoder state `s_{t-1}` and encoder hidden states `h_i` (for each input position i):
```
e_ti = score(s_{t-1}, h_i)              (raw, unbounded score)
alpha_ti = softmax_i(e_ti)              (attention weight for input i, at output step t)
c_t = sum_i ( alpha_ti * h_i )          (context vector: weighted sum of encoder states)
```
The context vector `c_t` is then fed into the decoder alongside its previous state and previous output word, to help produce the next output.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Walking Through Attention-Based Decoding

The lecture's own worked walkthrough, generating a German translation word by word:

**Generating word 1:**
1. Since there's no output yet, the decoder starts with **`<sos>`** (Start of Sequence).
2. Attention scores each input word: "how relevant is this input word for generating the first output word?"
3. Convert those raw scores into attention weights (via softmax) — e.g., the decoder learns to focus more on "apple" and "ate" for a particular first output word.
4. The decoder computes its new state and produces its first output word (via a softmax output layer over the vocabulary).

**Generating word 2 (timestep 1):**
1. The **previous decoder state** `s_0` is used as one of the inputs to the attention calculation.
2. Using `s_0`, attention is **recomputed from scratch** over ALL encoder hidden states `h_i` — this recomputation at every single decoding step is the key mechanical detail that makes attention dynamic.
3. Using the newly-computed weights, a **new context vector** `c_1` is created (a fresh weighted sum, different from whatever was used for word 1).
4. The decoder then uses: the previous output word (`Y_0`, e.g. "Ich"), the previous state (`s_0`), and the new context (`c_1`) together to produce the next output.

This pattern (recompute attention → build a fresh context vector → combine with previous state and previous output) repeats identically at every single decoding timestep until `<eos>` is generated.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Self-Attention: Query, Key, Value

The encoder-decoder attention described above computes attention *between* two different sequences (decoder attending to encoder). **Self-attention** is the closely related idea of a sequence attending **to itself** — every word looks at every other word *within the same sequence* to build a richer, context-aware representation. Self-attention is formally described as **mapping a query and a set of key-value pairs to an output**.

For each word, the model creates **three separate vectors**, each via its own learned weight matrix (projection):
- **Query (Q)** → "What am I looking for?"
- **Key (K)** → "What information do I offer?"
- **Value (V)** → "What information should I pass?"

These three vectors come from multiplying the same input word representation by three different learned weight matrices — i.e., `Q = X W_Q`, `K = X W_K`, `V = X W_V`, where `W_Q, W_K, W_V` are the learned projection matrices.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Why Multiply Query by Key? Why Multiply by Value?

**Why multiply Query with Key?** To compute a **similarity score** between what a word is "looking for" (its Query) and what every other word (including itself) "offers" (their Keys). Example: if "Machines" is strongly related to "Thinking" in context, the score between "Machines"'s Query and "Thinking"'s Key becomes high — and a higher score means the model should pay more attention to that relationship.

**Why multiply attention weights by Value?** After converting the Query-Key similarity scores into proper attention weights (via softmax, exactly like the encoder-decoder case above), those weights are used to compute a weighted combination of the **Value** vectors (not the Key vectors) — this combines information from the *important* words (as determined by the Query-Key scores) into the final output. The final output for each word becomes a **weighted mixture of the Values of the most relevant words** in the sequence, according to the learned attention weights.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Mnemonics

- **"One overloaded summary vector vs. glancing back at what matters"** → the core fixed-length-bottleneck-vs-attention story.
- **"Averaging treats everyone equally; attention plays favourites — correctly"** → why averaging fails and attention succeeds.
- **"Score → Softmax → Weighted Sum"** → the universal attention-weight recipe (works for both encoder-decoder attention AND self-attention).
- **"Q asks, K answers, V delivers"** → Query/Key/Value in one line.
- **"High QK score = pay more attention; weights × V = mix in that attention"** → why we multiply Query by Key, then attention weights by Value.
- **"Recompute EVERY output step"** → attention weights are never reused as-is between decoding steps — they're recalculated fresh, every single time.

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Cheatsheet

| Concept | One-liner | Formula |
|---|---|---|
| Raw attention score | Unbounded relevance measure | `e_ti = score(s_{t-1}, h_i)` |
| Attention weight | Normalized probability distribution | `alpha_ti = softmax_i(e_ti)` |
| Context vector | Weighted sum of encoder states | `c_t = Σ_i alpha_ti · h_i` |
| Self-attention output | Weighted mixture of Values | `Attention(Q,K,V) = softmax(QKᵀ) · V` (simplified) |
| Query | "What am I looking for?" | `Q = X W_Q` |
| Key | "What information do I offer?" | `K = X W_K` |
| Value | "What information should I pass?" | `V = X W_V` |
| Fixed bottleneck problem | One vector, overloaded | Solved by per-step context vectors |
| Averaging problem | Equal weight to all inputs | Solved by learned, dynamic weights |

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** confusing "averaging all hidden states" with "attention" — averaging gives EQUAL, FIXED weight to every input at every step; attention gives DIFFERENT, LEARNED, DYNAMICALLY-RECOMPUTED weight to every input at every step. These are explicitly contrasted in the lecture as the flawed fix vs the real solution.
- **Trap:** forgetting that attention weights must be recomputed at EVERY decoding step — a common mistake is treating attention as computed once and reused; the "Generating Word 2" walkthrough explicitly shows attention recomputed from scratch using the new decoder state `s_0`.
- **Trap:** mixing up Query/Key/Value roles — Query is "what THIS word wants," Key is "what OTHER words offer" (compared against Query), Value is "what actually gets passed through" once relevance is determined. The output uses attention weights times VALUE, never times Key directly.
- **Exam hack:** any question about "why does attention outperform a plain encoder-decoder RNN" should reference the fixed-length bottleneck explicitly — a single vector cannot hold arbitrarily much information without loss, especially for long sequences.
- **Exam hack:** the Score → Softmax → Weighted Sum pattern is universal across this entire lecture — both encoder-decoder attention and self-attention follow this exact three-step recipe, just with different definitions of what the "score" compares (decoder state vs encoder state, or Query vs Key).

`[🔝 Top](#dl-lecture-06--attention-theory)`

---

## Summary

This lecture motivates and explains Attention as the fix for a fundamental weakness of plain encoder-decoder RNNs: compressing an entire input sequence into one single fixed-length vector, which becomes badly "overloaded" for long sequences and ignores the fact that different output words genuinely depend on different, specific input words. A first tempting fix — averaging all encoder hidden states into one global context reused at every decoder step — fails because it gives equal importance to every input, when in reality different outputs need different inputs (illustrated by "Ich"↔"I" and "gegessen"↔"ate"). The real solution is to compute a **different weighted sum of encoder hidden states for every output step**, using attention weights that are dynamically computed as a function of the current decoder state: raw scores are computed via a scoring function comparing decoder state to each encoder hidden state, then converted into a proper probability distribution via softmax (positive, summing to 1.0), then used to build a context vector as a weighted sum of encoder hidden states. This entire process — score, softmax, weighted sum — is explicitly recomputed from scratch at every single decoding timestep, letting the decoder dynamically "look back" at whichever input words are currently most relevant. The lecture then generalizes this idea into **self-attention**, where a sequence attends to itself: each word generates a Query ("what am I looking for?"), Key ("what do I offer?"), and Value ("what should I pass?") vector via three learned projection matrices; Query is compared against every Key to compute similarity scores (higher score = more relevant), those scores are softmaxed into attention weights, and the weights are used to compute a weighted mixture of Value vectors as the final output — the exact same Score→Softmax→Weighted-Sum recipe, now applied within a single sequence rather than between an encoder and a decoder. This Query/Key/Value self-attention mechanism is the direct foundation for the Transformer architecture covered in a later lecture.

`[← Lecture 05](../../Lecture-05-LSTM/README.md) · [🔝 Top](#dl-lecture-06--attention-theory) · [Next: Numerical →](../numerical/dl_lecture06_attention_numerical.md)`
