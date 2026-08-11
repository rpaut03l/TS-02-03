# DL Lecture 10 — Attention and Transformers (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

> Folder: `Deep-Learning/Lecture-10-Attention-and-Transformers/theory/`
> Pairs with: [`numerical/dl_lecture10_transformers_numerical.md`](../numerical/dl_lecture10_transformers_numerical.md) · [`practice/dl_lecture10_transformers_practice.md`](../practice/dl_lecture10_transformers_practice.md) · [`exercises/dl_lecture10_exercises.md`](../exercises/dl_lecture10_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "Attention and Transformers" deck

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [The Food Court Analogy for QKV](#the-food-court-analogy-for-qkv)
3. [Self-Attention, Positional Encoding, Multi-Head Attention](#self-attention-positional-encoding-multi-head-attention)
4. [Why Transformers Revolutionized NLP and AI](#why-transformers-revolutionized-nlp-and-ai)
5. [Tokenization and Embedding](#tokenization-and-embedding)
6. [Positional Encoding — Requirements](#positional-encoding--requirements)
7. [The Transformer Block, Piece by Piece](#the-transformer-block-piece-by-piece)
8. [Residual Connections](#residual-connections)
9. [Layer Normalization](#layer-normalization)
10. [The Full Transformer Block — Summary Diagram](#the-full-transformer-block--summary-diagram)
11. [Encoder vs Decoder](#encoder-vs-decoder)
12. [Decoder Generation, Step by Step](#decoder-generation-step-by-step)
13. [Two Ways Decoders Differ From Encoders](#two-ways-decoders-differ-from-encoders)
14. [Transformer Scale — Original, GPT-2, GPT-3](#transformer-scale--original-gpt-2-gpt-3)
15. [Vision Transformers (ViT)](#vision-transformers-vit)
16. [Mnemonics](#mnemonics)
17. [Cheatsheet](#cheatsheet)
18. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
19. [Summary](#summary)

---

## The Big Picture — A Story First

Lecture 6 introduced attention as "glancing back at what's relevant." This lecture builds the FULL architecture around that idea — the **Transformer** — which has become the backbone of nearly every major modern AI system (GPT, BERT, ViT, and more). If Lecture 6 taught you the single most important LEGO brick, this lecture teaches you how to snap dozens of those bricks together — with residual connections holding pieces firmly in place, layer normalization keeping every piece the right size, multiple attention "heads" looking for different kinds of relationships simultaneously, and positional encoding making sure the model knows WHERE each word sits in a sentence, since attention itself has no built-in sense of order.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## The Food Court Analogy for QKV

A richer, more intuitive version of Lecture 6's Query/Key/Value story: imagine ordering food at a large food court. Your **craving** is the **Query**. Each food stall displays a **menu/description** of what it serves — that's the **Key**, telling you what kind of information that "source" offers. You compare your craving (Query) against every stall's menu (Key) to decide which stalls are most RELEVANT to what you want. Crucially, in attention you don't pick just ONE stall — you give DIFFERENT IMPORTANCE to MULTIPLE stalls simultaneously: "this pasta stall matches my craving the most," "the pizza stall is somewhat relevant," "the dessert stall adds a little extra." The **final meal** you walk away with is a **weighted combination of dishes (Values)** from different stalls, weighted exactly by how relevant each stall was. This is precisely what the lecture calls a **"soft match"** — instead of a hard, single winner-take-all selection, the model gathers information from SEVERAL sources at once, each contributing according to its own relevance weight.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Self-Attention, Positional Encoding, Multi-Head Attention

Three closely related building blocks, introduced together:
- **Self-attention:** learns what to focus on for EACH token (a token is a word or word-piece) — every token can attend to every other token in the same sequence.
- **Positional encoding:** adds a sense of ORDER to sequences — since self-attention by itself treats a sequence more like an unordered "bag" of tokens all mutually attending to each other, positional encoding is needed to inject information about WHERE each token sits.
- **Multi-head attention:** using MORE attention "heads" (more independent sets of Q/K/V projections, running in parallel) lets the model capture more DIVERSE kinds of relationships simultaneously — one head might learn to track subject-verb relationships, another might track which adjective modifies which noun, and so on.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Why Transformers Revolutionized NLP and AI

The Transformer architecture (Vaswani et al., "Attention Is All You Need," NeurIPS 2017) is widely credited with **making AI dramatically more scalable**. Unlike RNNs/LSTMs (Lectures 4–5), which process a sequence strictly one timestep at a time (fundamentally sequential, hard to parallelize), Transformers process an ENTIRE sequence's self-attention computation largely in parallel — enabling training on vastly larger datasets with vastly larger models, which directly enabled the current era of large language models.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Tokenization and Embedding

Before any attention computation happens, raw text must be converted into numbers. **Tokenization** breaks a sentence (e.g., "A cute teddy bear is reading.") into smaller pieces — tokens — which might be whole words or word-fragments, depending on the tokenizer. Each token is then converted into a numeric vector via a learned **embedding matrix** (the lecture references Word2vec-style embeddings as a classic example) — this embedding matrix is itself LEARNED at the start of the model, converting each token into a dense vector that captures something about its meaning.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Positional Encoding — Requirements

Since self-attention has no inherent sense of sequence order (every token can "see" every other token symmetrically), we need to explicitly inject positional information. Two key REQUIREMENTS for a good positional encoding scheme: (1) it should provide **some representation of position/time** (so the model can tell "this token is earlier" from "this token is later"), and (2) it should be **unique for each position — not cyclic** (if the encoding scheme repeated itself after some length, the model couldn't distinguish position 5 from position 5+cycle-length). The original Transformer paper uses sinusoidal functions (combinations of sine and cosine at different frequencies) specifically because they satisfy both requirements while also allowing the model to easily learn relative position relationships.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## The Transformer Block, Piece by Piece

**Input:** a set of vectors x (one per token, already embedded + positionally encoded). Inside a Transformer block, **all vectors interact through (multi-headed) Self-Attention** — e.g., computing the similarity of token x₁ with some earlier token x₂, and every other token, simultaneously.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Residual Connections

Looking at what happens BETWEEN layers (not just within one layer): training very deep networks is hard because of two related problems: **vanishing gradients** (again — this time occurring between many stacked Transformer layers, not just RNN timesteps) and **degradation** (deep networks genuinely struggle to learn good representations "from scratch" as depth increases, even setting aside pure vanishing-gradient concerns).

**The fix (He et al., 2016 — originally from ResNet, adapted here): Residual Connections.** Instead of forcing each layer to learn a completely fresh transformation from nothing, add the PREVIOUS layer's output directly to the current layer's output: `output = layer(input) + input`. This gives gradients a direct, un-obstructed "shortcut" path backward through the network (similar in spirit to LSTM's additive cell-state pathway from Lecture 5), and lets each layer focus on learning just the RESIDUAL — the "extra adjustment" needed — rather than an entire representation from scratch.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Layer Normalization

**The problem:** it's difficult to train the parameters of a layer because the input arriving from the PREVIOUS layer keeps CHANGING as training progresses (the previous layer's weights are also being updated simultaneously) — this constantly-shifting input distribution makes optimization harder.

**The solution: Layer Normalization** (Ba et al., 2016) — reduce this variation by normalizing the activations WITHIN each layer (recall from Lecture 3: Layer Norm normalizes per-sample, across all of that sample's features, unlike Batch Norm which normalizes per-feature across the batch — and crucially, Layer Norm behaves identically at train and test time, which is part of why it's the standard choice for Transformers rather than Batch Norm). In the Transformer block diagram, Layer Normalization is applied to ALL vectors (all tokens), right after the residual connection.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## The Full Transformer Block — Summary Diagram

Putting every piece together, a standard Transformer block follows this sequence:

```
 Input vectors x
       |
       v
 Multi-Head Self-Attention  (all vectors interact with each other)
       |
       v
 Add residual connection (+x)  ->  Layer Norm
       |
       v
 MLP  (applied INDEPENDENTLY to each vector - no cross-token interaction here)
       |
       v
 Add residual connection  ->  Layer Norm
       |
       v
 Output vectors y
```

**Why the residual connection after the MLP exists for the same three reasons as after self-attention:** (1) preserve original information, (2) help gradients flow easily, (3) prevent deep networks from degrading.

**Key structural insight (a favourite exam point):** **Self-Attention is the ONLY place where vectors interact with each other** — LayerNorm and the MLP both work on each vector INDEPENDENTLY. This is precisely what makes Transformers so **highly scalable and parallelizable**: most of the compute is just **6 matrix multiplications** — 4 from Self-Attention (Q, K, V projections plus the output projection) and 2 from the MLP (its two linear layers).

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Encoder vs Decoder

**Encoder = Understanding. Decoder = Speaking/Generating.** The encoder's job is only to create a rich, meaningful CONTEXTUAL representation of the input — e.g., input "I love AI" gets converted into rich contextual vectors. Critically: **the encoder does NOT itself generate anything** — not a translated sentence, not a summary, not a next word, not a chatbot response. Generation is entirely the DECODER's job.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Decoder Generation, Step by Step

The lecture's own worked example, generating "I am happy" during inference/testing:

```
Step 1: Decoder input = <start> (BOS, Beginning Of Sentence token)
        Decoder predicts: "I"

Step 2: Decoder input = <start> I
        Decoder predicts: "am"

Step 3: Decoder input = <start> I am
        Decoder predicts: "happy"

Step 4: Decoder input = <start> I am happy
        Decoder predicts: <end>
        -> generation STOPS here
```

Notice the pattern: at every step, the ENTIRE sequence generated so far is fed back in as input, and the decoder predicts just the NEXT token — this is called autoregressive generation, and it continues until an end-of-sequence token is produced.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Two Ways Decoders Differ From Encoders

Many of a decoder's internal modules are identical to an encoder's, but there are exactly TWO key structural differences:

**Difference 1 — Cross-attention.** The decoder must also "look at" the encoder's representation of the input — it has an EXTRA cross-attention module, similar in mechanism to self-attention, but instead of attending to the PREVIOUS layer's own tokens, it attends to the TOP layer of the ENCODER's output. This is exactly the classic encoder-decoder attention mechanism from Lecture 6, now embedded as one specific sub-module inside the Transformer decoder.

**Difference 2 — Masked (causal) self-attention.** The decoder is a GENERATION module, producing output left-to-right, one token at a time — so its self-attention must NOT be allowed to "see" future tokens that haven't been generated yet (that would be cheating, effectively looking at the answer before generating it). This is enforced via **masking**: future positions are explicitly blocked from contributing to the current position's attention computation.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Transformer Scale — Original, GPT-2, GPT-3

A striking, exam-favourite progression showing how Transformer models have scaled up over time:

| Model | Blocks (layers) | D (embedding dim) | H (attention heads) | N (context length) | Parameters |
|---|---|---|---|---|---|
| **Original** (Vaswani et al., 2017) | 12 | 1024 | 16 | 512 | 213M |
| **GPT-2** (Radford et al., 2019) | 48 | 1600 | 25 | 1024 | 1.5B |
| **GPT-3** (Brown et al., 2020) | 96 | 12288 | 96 | 2048 | 175B |

Every single dimension scaled up dramatically across just three years: 8× more blocks, 12× wider embeddings, 6× more attention heads, 4× longer context — compounding into roughly **820× more total parameters** from the Original Transformer to GPT-3.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Vision Transformers (ViT)

**Vision Transformers** (Dosovitskiy et al., "An Image is Worth 16×16 Words," ICLR 2021) apply the SAME Transformer architecture — originally designed for text — directly to IMAGES, by cleverly reframing an image as a "sentence" of patches:

1. **Start** with an input image, e.g. 224×224×3.
2. **Break into patches**, e.g. 16×16×3 each (this is literally the "16×16 Words" from the paper's title — each patch is treated like one "word/token").
3. **Flatten and apply a linear transform** — each 16×16×3 patch (768 raw values) is flattened into a single vector, then linearly projected from 768 dimensions into whatever embedding dimension D the Transformer uses.

Once images are converted into a sequence of patch "tokens" this way, the exact same Transformer encoder architecture (self-attention, residual connections, layer norm, MLP) used for text can process images directly — a striking demonstration of how general and reusable the Transformer architecture turned out to be.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Mnemonics

- **"Food court: craving=Query, menu=Key, dish=Value"** → the richer QKV analogy.
- **"Soft match, not hard pick"** → attention blends multiple sources by relevance, never picks just one.
- **"Multiple heads see multiple relationships"** → why multi-head attention exists.
- **"Add the shortcut, then normalize"** → residual connection followed by LayerNorm, at every sub-block.
- **"Only attention talks between tokens; LayerNorm and MLP mind their own business"** → the key parallelizability insight.
- **"Encoder understands, decoder speaks"** → the one-line encoder/decoder split.
- **"Decoder: cross-attend to encoder, and never peek at the future"** → the two decoder-specific differences.
- **"16×16 Words"** → how ViT turns an image into a Transformer-readable sequence.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Cheatsheet

| Concept | One-liner |
|---|---|
| Self-attention | Every token attends to every other token in its own sequence |
| Positional encoding | Injects order information; must be unique per position, non-cyclic |
| Multi-head attention | Multiple parallel Q/K/V projections, capturing diverse relationships |
| Residual connection | `output = layer(input) + input`; fixes vanishing gradients + degradation |
| Layer normalization | Normalizes each token's own feature vector; stable train/test behaviour |
| Transformer block compute | 6 matmuls total: 4 from self-attention, 2 from MLP |
| Encoder | Builds contextual understanding; does NOT generate |
| Decoder | Generates output autoregressively, left to right |
| Cross-attention | Decoder's extra module, attends to encoder's top layer |
| Masked self-attention | Decoder can't see future tokens |
| ViT | Image → patches → flattened + linearly projected → Transformer tokens |

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** thinking LayerNorm or the MLP allow tokens to interact with each other — they explicitly do NOT; only self-attention (and cross-attention, in the decoder) lets information flow BETWEEN different token positions.
- **Trap:** confusing the encoder's job with the decoder's — the encoder builds understanding/context; it is the decoder that actually GENERATES output, one token at a time.
- **Trap:** forgetting either of the decoder's two specific differences — cross-attention (looking at the encoder) AND masked self-attention (not looking at the future) are BOTH required; naming only one loses marks.
- **Trap:** believing residual connections and Layer Normalization solve the same problem — residuals fix gradient flow/degradation ACROSS layers; LayerNorm fixes unstable, shifting input distributions WITHIN a layer's own inputs. They're complementary, not redundant.
- **Exam hack:** the Original/GPT-2/GPT-3 scaling table is extremely likely to reappear, sometimes with numbers swapped or partially blanked — memorize the ORDER OF MAGNITUDE progression (213M → 1.5B → 175B) even if you can't recall every exact D/H/N value.
- **Exam hack:** for ViT questions, always mention BOTH steps explicitly — breaking into patches, AND the flatten-plus-linear-projection step — a common partial-answer mistake is describing only the patching step.

`[🔝 Top](#dl-lecture-10--attention-and-transformers-theory)`

---

## Summary

This lecture builds the full Transformer architecture on top of Lecture 6's attention foundations. The food-court analogy deepens QKV intuition: Query = your craving, Key = each stall's menu, Value = the actual dish, and attention performs a "soft match," blending multiple relevant sources by weighted importance rather than picking just one. Self-attention lets every token attend to every other token; positional encoding injects necessary order information (required to be unique per position and non-cyclic, since attention alone has no sense of sequence order); multi-head attention runs multiple attention computations in parallel to capture diverse relationship types simultaneously. A full Transformer block wraps multi-head self-attention and a per-token MLP each with a residual connection (fixing vanishing gradients and degradation across many stacked layers, `output=layer(input)+input`) followed by Layer Normalization (stabilizing each layer's shifting input distribution, with identical train/test behaviour) — and critically, self-attention is the ONLY place tokens interact with each other, since LayerNorm and the MLP both operate independently per token, which is exactly what makes Transformers so parallelizable (just 6 matrix multiplications per block: 4 from attention, 2 from the MLP). Encoders build contextual UNDERSTANDING of input but never generate anything themselves; decoders GENERATE output autoregressively, one token at a time (feeding the growing output sequence back in as input at every step, starting from a `<start>` token and stopping at `<end>`), and differ from encoders in exactly two ways: an extra cross-attention module attending to the encoder's top-layer output, and masked self-attention preventing any token from attending to future, not-yet-generated tokens. The dramatic scaling from the Original Transformer (213M parameters) to GPT-2 (1.5B) to GPT-3 (175B) — roughly 820× growth in just three years — illustrates why this architecture's parallelizability mattered so much for enabling ever-larger models. Finally, Vision Transformers (ViT) apply this exact same architecture to images by breaking an image into fixed-size patches (e.g., 16×16×3), flattening each patch, and linearly projecting it into the Transformer's embedding dimension — turning an image into a "sentence" of patch tokens that the standard Transformer encoder can then process directly.

`[← Lecture 09](../../Lecture-09-Graph-Neural-Networks/README.md) · [🔝 Top](#dl-lecture-10--attention-and-transformers-theory) · [Next: Numerical →](../numerical/dl_lecture10_transformers_numerical.md)`
