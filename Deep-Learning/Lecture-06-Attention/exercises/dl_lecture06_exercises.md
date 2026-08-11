# DL Lecture 06 — Exercise Bank (Attention)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-06--exercise-bank-attention)`

> Folder: `Deep-Learning/Lecture-06-Attention/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture06_attention_theory.md) · [numerical](../numerical/dl_lecture06_attention_numerical.md) · [practice](../practice/dl_lecture06_attention_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q6.1.** What two properties must attention weights satisfy?

**Q6.2.** Write the formula for the context vector, given attention weights and encoder hidden states.

**Q6.3.** Name the three vectors computed for each word in self-attention.

**Q6.4.** What operation converts raw attention scores into valid attention weights?

**Q6.5.** What is the fixed-length bottleneck problem?

---

## 🟡 Medium — Applied Reasoning

**Q6.6.** Given raw scores `e = [1.0, 3.0, 0.0]`, compute the attention weights via softmax.

**Q6.7.** Using the weights from Q6.6 and hidden states `h_1=[1,2]`, `h_2=[3,1]`, `h_3=[0,0]`, compute the context vector.

**Q6.8.** Explain why a plain average of encoder hidden states can never produce input-output alignment, referencing what stays constant across output steps.

**Q6.9.** In self-attention, explain what it means for a word's Query to have a high similarity score with its OWN Key, versus a high score with a DIFFERENT word's Key.

**Q6.10.** Explain why the encoder's `<eos>` token's hidden state is described as carrying special significance in a plain (non-attention) encoder-decoder RNN.

---

## 🔴 Hard — Derivation & Multi-Step

**Q6.11.** Given raw scores `e = [0.0, 0.0, 0.0, 0.0]` (all equal), compute the resulting attention weights, and explain what this reveals about the relationship between attention and plain averaging in this special case.

**Q6.12.** For a self-attention setup with `Q1=[2,0]`, `K1=[1,0]`, `K2=[0,1]`, `d_k=2`: compute the scaled dot-product scores for Q1 against both keys, then the softmax attention weights.

**Q6.13.** Continuing Q6.12, with `V1=[1,1]`, `V2=[3,3]`, compute the final self-attention output for word 1.

**Q6.14.** Explain, with a worked numeric argument, why self-attention's computational cost grows quadratically with sequence length (i.e., with n words, roughly n² pairwise score computations are needed) — compute the number of pairwise scores needed for sequences of length 10, 100, and 1000.

`[🔝 Top](#dl-lecture-06--exercise-bank-attention)`

---

## Answer Key

<details>
<summary>Q6.1 – Q6.5 (Easy)</summary>

- **Q6.1:** They must be positive, and must sum to exactly 1.0 (form a valid probability distribution).
- **Q6.2:** `c_t = Σ_i alpha_ti · h_i`.
- **Q6.3:** Query (Q), Key (K), Value (V).
- **Q6.4:** Softmax.
- **Q6.5:** The problem where an entire input sequence is compressed into a single fixed-length vector (the encoder's final hidden state), which becomes overloaded with information — especially for long inputs — losing important input-output alignment detail.
</details>

<details>
<summary>Q6.6 – Q6.10 (Medium)</summary>

- **Q6.6:** exp(1.0)=2.718, exp(3.0)=20.086, exp(0.0)=1.0. Sum=23.804. Weights ≈ [0.1142, 0.8438, 0.0420].
- **Q6.7:** c = 0.1142×[1,2] + 0.8438×[3,1] + 0.0420×[0,0] = [0.1142,0.2284] + [2.5314,0.8438] + [0,0] ≈ **[2.6456, 1.0722]**.
- **Q6.8:** A plain average sums ALL hidden states with the SAME fixed 1/n weight regardless of which output word is currently being generated — this weighting never changes across output steps, so there is no mechanism by which the context vector can shift to reflect "this particular output word needs this particular input word" — the same blended vector is fed to every decoding step no matter what.
- **Q6.9:** A high Query-Key score with its OWN Key means the word's own content is highly relevant to what it's "looking for" — it will attend strongly to itself. A high score with a DIFFERENT word's Key means that other word's information is highly relevant and should be incorporated strongly into this word's new representation — e.g., a pronoun's Query might score highly against the Key of the noun it refers to.
- **Q6.10:** In a plain encoder-decoder RNN, the `<eos>` token marks the end of the input sequence, and the hidden state AT that point is used as the sole summary handed to the decoder — since RNNs process sequentially and hidden states are continuously overwritten, this final hidden state is the last (and often most "complete," though imperfect) available summary of everything the encoder has read, making it the natural (if flawed) choice for initializing the decoder.
</details>

<details>
<summary>Q6.11 – Q6.14 (Hard)</summary>

- **Q6.11:** All four scores are equal (0.0), so `exp(0)=1` for each, sum=4, weights = [0.25, 0.25, 0.25, 0.25] — a perfectly uniform distribution. This reveals that plain averaging is actually a SPECIAL CASE of attention, occurring exactly when all raw scores are equal (i.e., the scoring function finds every input equally relevant) — attention is strictly more general/flexible, capable of both uniform averaging AND sharply focused weighting, depending on what the scoring function computes.
- **Q6.12:** score(Q1,K1) = [2,0]·[1,0] = 2. score(Q1,K2) = [2,0]·[0,1] = 0. Scaled: 2/√2≈1.4142, 0/√2=0. Softmax: exp(1.4142)≈4.1133, exp(0)=1. Sum=5.1133. Weights ≈ [0.8045, 0.1955].
- **Q6.13:** output = 0.8045×[1,1] + 0.1955×[3,3] = [0.8045,0.8045] + [0.5865,0.5865] = **[1.391, 1.391]**.
- **Q6.14:** For n words, every word computes a score against every OTHER word (including itself), giving n×n = n² pairwise scores. For n=10: 100 scores. For n=100: 10,000 scores. For n=1000: 1,000,000 scores. This quadratic growth is why plain self-attention becomes expensive for very long sequences, motivating research into more efficient attention variants for long-context applications.
</details>

`[🔝 Top](#dl-lecture-06--exercise-bank-attention)`

---

## Summary

This exercise bank drills Lecture 6's attention formulas across three tiers. Easy questions recall the attention-weight properties, the context vector formula, Query/Key/Value naming, the softmax operation, and the fixed-length bottleneck problem. Medium questions apply the full score→softmax→weighted-sum pipeline to new numbers, producing a context vector of [2.6456, 1.0722], and push conceptual reasoning about why plain averaging can never achieve input-output alignment, what high Query-Key similarity means for self- versus cross-word attention, and the special role of the `<eos>` hidden state in plain encoder-decoder RNNs. Hard questions require deeper derivations: showing that uniform raw scores make attention mathematically collapse into plain averaging (revealing averaging as a special case of attention), a full scaled dot-product self-attention computation producing weights [0.8045, 0.1955] and a final output of [1.391, 1.391], and a quantitative argument for self-attention's quadratic computational cost (100, 10,000, and 1,000,000 pairwise scores for sequences of length 10, 100, and 1000 respectively). All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture06_attention_practice.md) · [🔝 Top](#dl-lecture-06--exercise-bank-attention) · [Code →](../code/README.md)`
