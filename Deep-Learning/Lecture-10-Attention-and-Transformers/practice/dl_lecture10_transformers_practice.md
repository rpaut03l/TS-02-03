# DL Lecture 10 — Attention and Transformers (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-10--attention-and-transformers-practice)`

> Folder: `Deep-Learning/Lecture-10-Attention-and-Transformers/practice/`
> Pairs with: [`theory/dl_lecture10_transformers_theory.md`](../theory/dl_lecture10_transformers_theory.md) · [`numerical/dl_lecture10_transformers_numerical.md`](../numerical/dl_lecture10_transformers_numerical.md) · [`exercises/dl_lecture10_exercises.md`](../exercises/dl_lecture10_exercises.md)

---

## Table of Contents
1. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
2. [Explain-It-Back Prompts](#explain-it-back-prompts)
3. [Quick-Fire True / False](#quick-fire-true--false)
4. [Encoder vs Decoder Matching Drill](#encoder-vs-decoder-matching-drill)
5. [Mini Interview-Style Round](#mini-interview-style-round)
6. [Summary](#summary)

---

## Concept Check — Fill in the Blank

1. In the food court analogy, your craving is the ______, each stall's menu is the ______, and the actual dish is the ______.
2. Positional encoding must satisfy two requirements: some representation of ______, and it must be ______ for each position.
3. Residual connections were originally introduced by ______ et al., 2016.
4. Inside a Transformer block, the ONLY module where tokens interact with each other is ______.
5. The decoder differs from the encoder in exactly two ways: an extra ______ module, and ______ self-attention.

<details>
<summary>Show answers</summary>

1. Query; Key; Value
2. time/position; unique (not cyclic)
3. He
4. (multi-head) self-attention
5. cross-attention; masked
</details>

`[🔝 Top](#dl-lecture-10--attention-and-transformers-practice)`

---

## Explain-It-Back Prompts

1. Explain the food court analogy for Query, Key, and Value in your own words.
2. Explain why residual connections and Layer Normalization solve two DIFFERENT problems, even though both appear in the same part of the Transformer block diagram.
3. Walk through the full decoder generation process for the sentence "I am happy," from `<start>` to `<end>`.
4. Explain the two specific structural differences between a Transformer decoder and encoder.
5. Explain, using the ViT patch-count numbers, how an image becomes a sequence a Transformer can process.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-practice)`

---

## Quick-Fire True / False

1. Self-attention has a built-in sense of token order, so positional encoding is optional. — **False** (self-attention has no inherent order sense; positional encoding is necessary).
2. LayerNorm and the MLP inside a Transformer block let tokens exchange information with each other. — **False** (only self-attention does this; LayerNorm and MLP work per-token independently).
3. The encoder generates the final output sequence (e.g., the translated sentence). — **False** (that's the decoder's job; the encoder only builds contextual understanding).
4. GPT-3 has roughly 175 billion parameters. — **True**.
5. Vision Transformers process raw pixels directly, without any patching step. — **False** (ViT breaks the image into patches first, then flattens and linearly projects each patch).

`[🔝 Top](#dl-lecture-10--attention-and-transformers-practice)`

---

## Encoder vs Decoder Matching Drill

| Property | Encoder | Decoder |
|---|---|---|
| Role | ? | ? |
| Generates output? | ? | ? |
| Extra module vs the other | ? | ? |
| Self-attention type | ? | ? |

<details>
<summary>Show answers</summary>

Role: Encoder = Understanding; Decoder = Speaking/Generating. Generates output: Encoder = No; Decoder = Yes (autoregressively). Extra module: Decoder has an extra cross-attention module (attending to the encoder's top layer) that the encoder lacks. Self-attention type: Encoder = full/unmasked self-attention; Decoder = masked (causal) self-attention, preventing access to future tokens.
</details>

`[🔝 Top](#dl-lecture-10--attention-and-transformers-practice)`

---

## Mini Interview-Style Round

**Q1.** "A teammate suggests removing the residual connections to 'simplify the architecture and save compute.' What's your concern?"

<details>
<summary>Show answer</summary>

Residual connections exist specifically to fix vanishing gradients and degradation in deep stacks of layers — removing them would make training the many stacked Transformer blocks (12+ in the original paper, 96 in GPT-3) significantly harder, likely causing worse convergence or requiring much more careful initialization/tuning to compensate. The compute savings from removing a simple elementwise addition are negligible compared to the training stability they provide — this would be a bad trade-off.
</details>

**Q2.** "Explain why Transformers are more parallelizable than RNNs/LSTMs for processing a sequence, in terms a beginner would understand."

<details>
<summary>Show answer</summary>

An RNN/LSTM must process a sequence strictly one timestep at a time — computing the hidden state at position 5 REQUIRES first computing positions 1 through 4, since each timestep depends on the previous one (recall the recurrence formula `s_t=f(Ux_t+Ws_{t-1})` from Lecture 4). A Transformer's self-attention, by contrast, lets every token compute its relationship with every other token SIMULTANEOUSLY — there's no such sequential dependency between different token positions within a single self-attention computation, so it can be computed largely as a small number of big matrix multiplications, which GPUs are extremely good at running in parallel.
</details>

**Q3.** "Why does a decoder need masked self-attention, but an encoder doesn't?"

<details>
<summary>Show answer</summary>

The encoder's job is purely to build a rich understanding of an ALREADY COMPLETE input sequence — it has access to the entire input at once, so there's no "future" to hide; every token can freely attend to every other token, including ones that come later in the sequence. The decoder, by contrast, GENERATES output one token at a time, left to right — at the moment it's predicting the 3rd output word, the 4th, 5th, etc. output words don't exist yet (that's literally what it's trying to predict). Letting the decoder's self-attention see future tokens during training would mean it could "cheat" by looking at the answer before generating it, so masking is required to block that.
</details>

`[🔝 Top](#dl-lecture-10--attention-and-transformers-practice)`

---

## Summary

This practice file drills Lecture 10's Transformer architecture through active recall. A fill-in-the-blank check reinforces the food-court QKV analogy, positional encoding's two requirements, residual connections' origin, the only-self-attention-lets-tokens-interact rule, and the decoder's two specific structural differences. Five explain-it-back prompts push you to reproduce the food court analogy, the residual-vs-LayerNorm distinction, the full decoder generation walkthrough, the encoder-decoder structural differences, and the ViT image-to-sequence conversion in your own words. A quick-fire true/false round targets common misconceptions (self-attention's lack of built-in order, which module actually generates output, GPT-3's parameter count, and ViT's patching step), and a dedicated encoder-vs-decoder matching drill consolidates the full comparison into one table. A three-question interview-style round rehearses realistic architectural reasoning: pushing back on removing residual connections, explaining Transformer parallelizability versus RNNs in beginner-friendly terms, and justifying exactly why masked self-attention is decoder-specific. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture10_transformers_numerical.md) · [🔝 Top](#dl-lecture-10--attention-and-transformers-practice) · [Next: Exercises →](../exercises/dl_lecture10_exercises.md)`
