# DL Lecture 10 — Exercise Bank (Attention and Transformers)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-10--exercise-bank-attention-and-transformers)`

> Folder: `Deep-Learning/Lecture-10-Attention-and-Transformers/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture10_transformers_theory.md) · [numerical](../numerical/dl_lecture10_transformers_numerical.md) · [practice](../practice/dl_lecture10_transformers_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q10.1.** Write the residual connection formula.

**Q10.2.** What are the two requirements for a good positional encoding scheme?

**Q10.3.** How many total matrix multiplications does a single Transformer block use, and how are they split?

**Q10.4.** Name the two structural differences between a decoder and an encoder.

**Q10.5.** What does the ViT paper's title "An Image is Worth 16×16 Words" literally refer to?

---

## 🟡 Medium — Applied Reasoning

**Q10.6.** For a 384×384×3 image with 16×16 patches, compute the number of patches and the flattened patch dimension.

**Q10.7.** Given input vector x=[2.0,4.0,6.0] and sub-layer output [1.0,-1.0,0.0], compute the residual connection output, then apply Layer Normalization to the result.

**Q10.8.** Explain why the "6 matmuls per block" fact is often cited as a reason Transformers scale so efficiently, connecting it to GPU computation.

**Q10.9.** For a Transformer with 48 blocks (matching GPT-2's depth), compute the total number of matrix multiplications.

**Q10.10.** For an autoregressively-generated 6-word output sentence, compute the total number of decoding steps required.

---

## 🔴 Hard — Derivation & Multi-Step

**Q10.11.** For a 96×96×3 image with 8×8 patches, compute the total number of patches and the flattened patch dimension. Then compute how these numbers would change if the patch size were doubled to 16×16 instead.

**Q10.12.** Compute the parameter-count ratio between GPT-3 (175B) and the Original Transformer (213M), then explain what this ratio implies about compute requirements, assuming compute scales roughly linearly with parameter count.

**Q10.13.** A Transformer decoder is generating a 10-word sentence. At the moment it is predicting word 6, explain exactly which tokens its masked self-attention IS and IS NOT allowed to attend to, and why.

**Q10.14.** Given a residual output `[0.0, 5.0, 10.0]`, compute the full Layer Normalization (mean, variance, normalized output), then explain why a Layer Norm computation for a DIFFERENT token in the same batch would use completely different mean/variance values, unlike Batch Norm.

`[🔝 Top](#dl-lecture-10--exercise-bank-attention-and-transformers)`

---

## Answer Key

<details>
<summary>Q10.1 – Q10.5 (Easy)</summary>

- **Q10.1:** `output = sublayer(x) + x`.
- **Q10.2:** Some representation of time/position, and uniqueness for each position (not cyclic).
- **Q10.3:** 6 total: 4 from self-attention, 2 from the MLP.
- **Q10.4:** An extra cross-attention module (attending to the encoder's top layer), and masked (causal) self-attention (can't see future tokens).
- **Q10.5:** A 224×224 image, broken into 16×16 patches, produces exactly 196 patches — treated as a "sentence" of 196 "words" (tokens), one per patch.
</details>

<details>
<summary>Q10.6 – Q10.10 (Medium)</summary>

- **Q10.6:** patches_per_side = 384/16 = 24. Total patches = 24×24 = **576**. Flattened patch dim = 16×16×3 = **768**.
- **Q10.7:** Residual output = [2.0+1.0, 4.0+(-1.0), 6.0+0.0] = **[3.0, 3.0, 6.0]**. Mean = (3+3+6)/3 = 4.0. Variance = [(3-4)²+(3-4)²+(6-4)²]/3 = [1+1+4]/3 = 2.0. Normalized = [(3-4)/√2, (3-4)/√2, (6-4)/√2] ≈ **[-0.7071, -0.7071, 1.4142]**.
- **Q10.8:** GPUs are extremely efficient at large matrix multiplications specifically (highly parallel hardware designed for exactly this operation). Because a Transformer block's computation reduces to just 6 large matmuls, rather than many small sequential operations (as in RNNs, where each timestep must wait for the previous one), a GPU can process an entire sequence's self-attention and MLP computations largely in parallel, making training dramatically faster for a given amount of compute.
- **Q10.9:** 6 × 48 = **288** total matrix multiplications.
- **Q10.10:** 6 words + 1 (for `<end>`) = **7 decoding steps**.
</details>

<details>
<summary>Q10.11 – Q10.14 (Hard)</summary>

- **Q10.11:** With 8×8 patches: patches_per_side = 96/8=12, total=12×12=**144** patches, flattened dim = 8×8×3=**192**. With 16×16 patches instead: patches_per_side=96/16=6, total=6×6=**36** patches, flattened dim=16×16×3=**768**. Doubling the patch size QUARTERS the number of patches (144→36, a 4× reduction, since area scales with the square of side length) while QUADRUPLING the flattened dimension per patch (192→768) — a direct trade-off between sequence length and per-token dimensionality.
- **Q10.12:** Ratio = 175,000,000,000/213,000,000 ≈ **821.6×**. If compute scales roughly linearly with parameter count, training GPT-3 would require on the order of 800+ times more compute than training the Original Transformer, for a single comparable pass over data — in practice, real training compute requirements scale even more steeply than linearly with model size when accounting for the correspondingly larger datasets and longer training runs typically used with bigger models, but the parameter-count ratio alone already indicates an enormous compute gap.
- **Q10.13:** While predicting word 6, masked self-attention IS allowed to attend to positions 1 through 5 (all previously generated tokens, including the `<start>` token) — these represent everything "already known" at this point in generation. It is NOT allowed to attend to positions 7 through 10 (future, not-yet-generated words), because those tokens don't exist yet during actual generation, and allowing the model to see them during training would let it "cheat" by looking at the answer rather than learning to predict it from only the preceding context.
- **Q10.14:** Mean = (0+5+10)/3 = 5.0. Variance = [(0-5)²+(5-5)²+(10-5)²]/3 = [25+0+25]/3 ≈ 16.667. Normalized ≈ [(0-5)/4.082, (5-5)/4.082, (10-5)/4.082] ≈ **[-1.2247, 0, 1.2247]**. A different token in the same batch has its OWN separate feature vector, entirely independent of this token's values — Layer Norm computes mean/variance from ONLY the current token's own features, never mixing in information from other tokens/samples in the batch (unlike Batch Norm, which explicitly averages across the batch dimension), so each token's normalization statistics are naturally different and computed in complete isolation from every other token.
</details>

`[🔝 Top](#dl-lecture-10--exercise-bank-attention-and-transformers)`

---

## Summary

This exercise bank drills Lecture 10's Transformer architecture formulas across three tiers. Easy questions recall the residual connection formula, positional encoding's requirements, the 6-matmul-per-block breakdown, the two encoder-decoder differences, and ViT's patch-based framing. Medium questions apply the patch-count formula to a new 384×384 image (576 patches), compute a full residual-plus-LayerNorm pipeline by hand, connect the matmul count to GPU parallelization, scale the matmul count to GPT-2's 48-block depth (288 total), and compute autoregressive decoding steps for a new sentence length. Hard questions require deeper reasoning: a patch-size trade-off analysis showing how doubling patch size quarters patch count while quadrupling per-patch dimensionality, a compute-scaling implication from the ~822× GPT-3/Original parameter ratio, a precise explanation of exactly which positions masked self-attention can and cannot see mid-generation, and a full Layer Normalization computation with an explicit explanation of why different tokens in the same batch get entirely independent normalization statistics. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture10_transformers_practice.md) · [🔝 Top](#dl-lecture-10--exercise-bank-attention-and-transformers) · [Code →](../code/README.md)`
