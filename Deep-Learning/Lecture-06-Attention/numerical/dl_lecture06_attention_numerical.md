# DL Lecture 06 — Attention (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-06--attention-numerical)`

> Folder: `Deep-Learning/Lecture-06-Attention/numerical/`
> Pairs with: [`theory/dl_lecture06_attention_theory.md`](../theory/dl_lecture06_attention_theory.md) · [`practice/dl_lecture06_attention_practice.md`](../practice/dl_lecture06_attention_practice.md) · [`exercises/dl_lecture06_exercises.md`](../exercises/dl_lecture06_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Raw Scores to Attention Weights (Softmax)](#worked-example-1--raw-scores-to-attention-weights-softmax)
3. [Worked Example 2 — Building the Context Vector](#worked-example-2--building-the-context-vector)
4. [Worked Example 3 — Why Averaging Fails, Quantified](#worked-example-3--why-averaging-fails-quantified)
5. [Worked Example 4 — Self-Attention, Fully Worked (Tiny Example)](#worked-example-4--self-attention-fully-worked-tiny-example)
6. [Master Formula Cheatsheet](#master-formula-cheatsheet)
7. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
8. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| h_i | encoder hidden state at input position i |
| s_{t-1} | decoder hidden state before generating output t |
| e_ti | raw (unbounded) attention score for input i, output step t |
| alpha_ti | normalized attention weight (softmax of e_ti) |
| c_t | context vector at output step t |
| Q, K, V | Query, Key, Value matrices in self-attention |
| d_k | dimensionality of the Key vectors (used for scaling) |

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Worked Example 1 — Raw Scores to Attention Weights (Softmax)

**Given:** raw attention scores for 4 input words, computed for some output step t: `e = [2.0, 0.5, -1.0, 3.0]`.

**Step 1 — Exponentiate each score.**
```
exp(2.0) = 7.389
exp(0.5) = 1.649
exp(-1.0) = 0.368
exp(3.0) = 20.086
```

**Step 2 — Sum the exponentials.**
```
sum = 7.389 + 1.649 + 0.368 + 20.086 = 29.492
```

**Step 3 — Divide each exponential by the sum.**
```
alpha_1 = 7.389 / 29.492 ≈ 0.2506
alpha_2 = 1.649 / 29.492 ≈ 0.0559
alpha_3 = 0.368 / 29.492 ≈ 0.0125
alpha_4 = 20.086 / 29.492 ≈ 0.6811
```

**Step 4 — Verify they sum to 1.0.**
```
0.2506 + 0.0559 + 0.0125 + 0.6811 = 1.0001 ≈ 1.0 (rounding)
```

**Result: attention weights ≈ [0.2506, 0.0559, 0.0125, 0.6811].** Notice input word 4 (the highest raw score, 3.0) receives by far the most attention (68%), while input word 3 (the lowest/negative score) receives almost none (1.25%) — exactly the "high for relevant, low elsewhere" behaviour required by the theory file.

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Worked Example 2 — Building the Context Vector

**Given:** the attention weights from Example 1, and 4 encoder hidden states (each a 3-dimensional toy vector for hand-tractability):
```
h_1 = [1, 0, 1]
h_2 = [0, 1, 0]
h_3 = [1, 1, 0]
h_4 = [0, 0, 1]
```

**Step 1 — Multiply each hidden state by its attention weight.**
```
alpha_1 * h_1 = 0.2506 x [1,0,1] = [0.2506, 0, 0.2506]
alpha_2 * h_2 = 0.0559 x [0,1,0] = [0, 0.0559, 0]
alpha_3 * h_3 = 0.0125 x [1,1,0] = [0.0125, 0.0125, 0]
alpha_4 * h_4 = 0.6811 x [0,0,1] = [0, 0, 0.6811]
```

**Step 2 — Sum all four weighted vectors.**
```
c_t = [0.2506+0+0.0125+0, 0+0.0559+0.0125+0, 0.2506+0+0+0.6811]
    = [0.2631, 0.0684, 0.9317]
```

**Result: c_t ≈ [0.2631, 0.0684, 0.9317].** Notice this context vector is heavily dominated by `h_4` (weight 0.6811) — the third component (0.9317, close to h_4's own third component of 1) shows this clearly. This is exactly the mechanism by which attention lets the decoder "mostly look at" the most relevant input, while still blending in small contributions from the rest.

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Worked Example 3 — Why Averaging Fails, Quantified

**Given:** the same 4 hidden states as Example 2. Compute the plain (unweighted) average, and compare to the attention-weighted context vector.

**Step 1 — Compute the plain average.**
```
average = (h_1 + h_2 + h_3 + h_4) / 4
        = ([1,0,1]+[0,1,0]+[1,1,0]+[0,0,1]) / 4
        = [2,2,2] / 4
        = [0.5, 0.5, 0.5]
```

**Step 2 — Compare to the attention context vector from Example 2: [0.2631, 0.0684, 0.9317].**

**Step 3 — Observe the key difference.** The plain average [0.5, 0.5, 0.5] is IDENTICAL no matter which output word is currently being generated — it's a fixed vector, always the same regardless of context. The attention-weighted context vector, by contrast, changes completely depending on what attention scores were computed for the CURRENT output step — if a different output word needed to focus on `h_2` instead of `h_4`, its context vector would look completely different (heavily weighted toward [0,1,0] instead). **This numerically proves the theory file's claim: averaging reuses the same context for every output, while attention builds a genuinely different, purpose-built context vector for every single output step.**

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Worked Example 4 — Self-Attention, Fully Worked (Tiny Example)

**Given:** a toy sequence of 2 words, each with a 2-dimensional embedding: `x_1 = [1, 0]`, `x_2 = [0, 1]`. Illustrative learned projection matrices (2×2, for simplicity):
```
W_Q = [[1, 0], [0, 1]]     (identity, for simplicity)
W_K = [[1, 0], [0, 1]]     (identity, for simplicity)
W_V = [[2, 0], [0, 2]]     (doubles the input)
```

**Step 1 — Compute Query, Key, Value for each word.**
```
Q1 = x1 . W_Q = [1,0],  Q2 = x2 . W_Q = [0,1]
K1 = x1 . W_K = [1,0],  K2 = x2 . W_K = [0,1]
V1 = x1 . W_V = [2,0],  V2 = x2 . W_V = [0,2]
```

**Step 2 — Compute raw similarity scores for word 1's Query against both Keys (dot product).**
```
score(Q1,K1) = [1,0] . [1,0] = 1x1 + 0x0 = 1
score(Q1,K2) = [1,0] . [0,1] = 1x0 + 0x1 = 0
```

**Step 3 — Scale by sqrt(d_k) (standard practice; here d_k=2, so sqrt(2)≈1.414 — included for completeness, though the raw scores here are small).**
```
scaled_score(Q1,K1) = 1 / 1.414 ≈ 0.7071
scaled_score(Q1,K2) = 0 / 1.414 = 0
```

**Step 4 — Softmax the scaled scores.**
```
exp(0.7071) ≈ 2.0281, exp(0) = 1.0
sum = 2.0281 + 1.0 = 3.0281
alpha(1,1) = 2.0281/3.0281 ≈ 0.6697
alpha(1,2) = 1.0/3.0281 ≈ 0.3303
```

**Step 5 — Compute word 1's self-attention output as a weighted sum of Values.**
```
output_1 = 0.6697 x V1 + 0.3303 x V2
         = 0.6697 x [2,0] + 0.3303 x [0,2]
         = [1.3394, 0] + [0, 0.6606]
         = [1.3394, 0.6606]
```

**Result: word 1's self-attention output is [1.3394, 0.6606]** — a blend dominated by its own Value (V1, since word 1's Query matched its own Key most strongly, score=1 vs 0), but still incorporating some information from word 2. This is precisely the "weighted mixture of relevant words' Values" described in the theory file, worked fully by hand.

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Raw attention score | `e_ti = score(s_{t-1}, h_i)` |
| Softmax (attention weight) | `alpha_ti = exp(e_ti) / Σ_j exp(e_tj)` |
| Context vector | `c_t = Σ_i alpha_ti · h_i` |
| Self-attention Q/K/V | `Q=XW_Q`, `K=XW_K`, `V=XW_V` |
| Scaled dot-product score | `score = (Q·Kᵀ)/√d_k` |
| Self-attention output | `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V` |

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting the softmax normalization step and treating raw scores as if they were already valid weights — raw scores can be negative and don't sum to 1; always apply softmax before using them as weights.
- **Trap:** computing the context vector as a simple average instead of a WEIGHTED sum — always multiply each hidden state by its own specific attention weight before summing, per Worked Example 2.
- **Trap:** forgetting the `√d_k` scaling factor in self-attention — while it doesn't change which input gets the most attention, it DOES change the exact numeric weights, and is frequently tested.
- **Exam hack:** always show the four-step softmax computation explicitly (exponentiate → sum → divide → verify sums to 1) — showing the verification step is an easy, often-overlooked way to catch your own arithmetic errors before submitting.
- **Exam hack:** the averaging-vs-attention comparison (Worked Example 3) is a favourite "explain the difference numerically" exam question — always compute BOTH the plain average AND the attention-weighted context vector side by side to make the contrast concrete.

`[🔝 Top](#dl-lecture-06--attention-numerical)`

---

## Summary

This file turned every Attention formula from the theory file into fully worked arithmetic. Starting from raw scores `[2.0, 0.5, -1.0, 3.0]`, the full softmax computation (exponentiate, sum, divide, verify) produced attention weights `[0.2506, 0.0559, 0.0125, 0.6811]`, correctly concentrating most weight on the highest-scoring input. Those weights were then used to build a context vector as a weighted sum of four toy encoder hidden states, producing `[0.2631, 0.0684, 0.9317]` — heavily dominated by the highest-weighted hidden state. A direct comparison against the plain unweighted average `[0.5, 0.5, 0.5]` numerically proved why averaging fails: the average is a fixed vector regardless of which output is being generated, while the attention-weighted context vector genuinely changes shape based on the current output's specific needs. Finally, a complete self-attention example computed Query, Key, and Value vectors for a 2-word toy sequence, computed and scaled similarity scores, softmaxed them into attention weights (≈0.67 self-attention, ≈0.33 to the other word), and produced a final weighted-Value output of `[1.3394, 0.6606]` — a fully hand-traceable version of the Query/Key/Value mechanism that underlies modern Transformers. The master formula table consolidates every reusable calculation from this lecture for fast review.

`[← Theory](../theory/dl_lecture06_attention_theory.md) · [🔝 Top](#dl-lecture-06--attention-numerical) · [Next: Practice →](../practice/dl_lecture06_attention_practice.md)`
