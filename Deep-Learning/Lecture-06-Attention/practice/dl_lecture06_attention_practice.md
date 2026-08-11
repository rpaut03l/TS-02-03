# DL Lecture 06 — Attention (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-06--attention-practice)`

> Folder: `Deep-Learning/Lecture-06-Attention/practice/`
> Pairs with: [`theory/dl_lecture06_attention_theory.md`](../theory/dl_lecture06_attention_theory.md) · [`numerical/dl_lecture06_attention_numerical.md`](../numerical/dl_lecture06_attention_numerical.md) · [`exercises/dl_lecture06_exercises.md`](../exercises/dl_lecture06_exercises.md)

---

## Table of Contents
1. [Official In-Class Questions](#official-in-class-questions)
2. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
3. [Explain-It-Back Prompts](#explain-it-back-prompts)
4. [Quick-Fire True / False](#quick-fire-true--false)
5. [QKV Role Matching Drill](#qkv-role-matching-drill)
6. [Mini Interview-Style Round](#mini-interview-style-round)
7. [Summary](#summary)

---

## Official In-Class Questions

**Q1.** Guess the problem with the "process input, then generate output from just the final hidden state" setting.

<details>
<summary>Show answer</summary>

Each output word ends up depending only on the current hidden state, not directly on previous outputs or specific parts of the input — since the entire input has to be squeezed into one fixed-length final hidden state, this single vector becomes overloaded, especially for long inputs, and information from earlier in the sequence tends to get lost by the time generation happens.
</details>

**Q2.** There is still a problem with averaging all encoder hidden states and feeding that same average to every decoder stage. What is it?

<details>
<summary>Show answer</summary>

All inputs get equal importance with no distinction between them, and the exact same context gets reused for every single output step — but different outputs genuinely need different inputs (e.g. "Ich" aligns with "I," "gegessen" aligns with "ate"). Because the model can't focus on the specific words relevant to each output, alignment stays weak. This is exactly the gap that motivates real Attention: a DIFFERENT, dynamically-computed weighted sum for each output step, instead of one fixed average reused everywhere.
</details>

`[🔝 Top](#dl-lecture-06--attention-practice)`

---

## Concept Check — Fill in the Blank

1. The encoder's final hidden state, in a plain encoder-decoder RNN, acts as a compact ______ of the entire input sequence.
2. Attention weights must be ______ and sum to ______.
3. The two-step recipe for computing attention weights is: compute raw ______, then apply ______.
4. In self-attention, the three vectors created for each word are ______, ______, and ______.
5. Query is compared against Key to compute a ______ score; attention weights are then multiplied by ______ to produce the final output.

<details>
<summary>Show answers</summary>

1. summary
2. positive; 1.0
3. scores; softmax
4. Query (Q); Key (K); Value (V)
5. similarity; Value
</details>

`[🔝 Top](#dl-lecture-06--attention-practice)`

---

## Explain-It-Back Prompts

1. Explain the fixed-length bottleneck problem using the "one summary sentence for a whole paragraph" analogy.
2. Walk through the "Generating Word 2" attention-decoding steps from memory, in order.
3. Explain why averaging all encoder hidden states fails, using the "Ich"/"I" and "gegessen"/"ate" example.
4. Explain the Query/Key/Value roles using the "What am I looking for / What do I offer / What should I pass" phrasing, in your own words.
5. Explain why attention weights must be recomputed at every decoding step, rather than computed once and reused.

`[🔝 Top](#dl-lecture-06--attention-practice)`

---

## Quick-Fire True / False

1. In a plain encoder-decoder RNN, the encoder compresses the entire input into one fixed-length vector. — **True**.
2. Averaging all encoder hidden states gives every input word different, learned importance. — **False** (it gives every input EQUAL importance — that's exactly why it fails).
3. Attention weights are computed once at the start of decoding and reused for every output word. — **False** (recomputed at every decoding step).
4. In self-attention, Query is multiplied by Key to compute similarity scores. — **True**.
5. The final self-attention output is a weighted mixture of Key vectors. — **False** (it's a weighted mixture of Value vectors).

`[🔝 Top](#dl-lecture-06--attention-practice)`

---

## QKV Role Matching Drill

| Vector | Question it answers | Used for |
|---|---|---|
| Query | ? | ? |
| Key | ? | ? |
| Value | ? | ? |

<details>
<summary>Show answers</summary>

Query → "What am I looking for?" → compared against every Key to compute similarity scores. Key → "What information do I offer?" → compared against Queries to determine relevance. Value → "What information should I pass?" → combined (weighted by attention scores) to form the final output.
</details>

`[🔝 Top](#dl-lecture-06--attention-practice)`

---

## Mini Interview-Style Round

**Q1.** "A teammate proposes fixing a poorly-performing translation model by simply making the encoder's final hidden state vector much larger (more dimensions), instead of adding attention. Evaluate this idea."

<details>
<summary>Show answer</summary>

A larger fixed-length vector can hold somewhat more information, but it doesn't fix the fundamental problem: it's still ONE vector, reused identically for generating every single output word, with no mechanism to dynamically focus on different input words for different outputs. Attention solves a structurally different problem — direct input-output alignment — that raw vector size alone can't address; you'd still lose the ability to have "Ich" specifically attend to "I" versus "gegessen" specifically attending to "ate." Attention is a better fix than simply scaling up the bottleneck vector.
</details>

**Q2.** "Explain to a teammate why self-attention needs THREE separate learned projections (Q, K, V) instead of just reusing the same vector for all three roles."

<details>
<summary>Show answer</summary>

Each role does a genuinely different job: Query needs to represent "what this word is looking for" in a way that's comparable against other words' Keys; Key needs to represent "what this word offers" in a way that's easy for other words' Queries to match against; Value needs to represent "what content should actually get passed forward" once relevance is established. Using the same vector for all three would conflate these different jobs — a word's own raw representation might not be the best fit for all three purposes simultaneously, so separate learned projections let the network optimize each role independently for the task.
</details>

**Q3.** "Someone asks why the score is scaled by √d_k in self-attention. What's the intuition?"

<details>
<summary>Show answer</summary>

As the dimensionality of Q and K grows, dot products between them tend to grow larger in magnitude purely due to summing more terms, which can push softmax into regions where its gradient is very small (extremely peaked outputs), hurting training. Dividing by √d_k keeps the scores in a more moderate, well-behaved range regardless of dimensionality, helping softmax produce a more stable, learnable distribution.
</details>

`[🔝 Top](#dl-lecture-06--attention-practice)`

---

## Summary

This practice file drills Lecture 6's Attention mechanics through active recall. Both official in-class questions are answered in full — the fixed-final-hidden-state problem and the averaging-hidden-states problem — establishing the exact two-step motivation trail that leads to real Attention. A fill-in-the-blank check reinforces the encoder summary concept, the positive-sum-to-1.0 requirement, the score→softmax recipe, and Query/Key/Value terminology. Five explain-it-back prompts push you to reproduce the bottleneck analogy, the "Generating Word 2" walkthrough, the averaging-failure example, the QKV role descriptions, and the per-step recomputation requirement in your own words. A quick-fire true/false round and a QKV role-matching drill test both conceptual understanding and precise terminology. A three-question interview-style round rehearses realistic reasoning: why scaling up a bottleneck vector doesn't substitute for attention, why Q/K/V need to be three separate learned projections rather than one shared vector, and the intuition behind the √d_k scaling factor. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture06_attention_numerical.md) · [🔝 Top](#dl-lecture-06--attention-practice) · [Next: Exercises →](../exercises/dl_lecture06_exercises.md)`
