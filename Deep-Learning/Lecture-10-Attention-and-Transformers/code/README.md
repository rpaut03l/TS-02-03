# DL Lecture 10 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-10--code)`

> Folder: `Deep-Learning/Lecture-10-Attention-and-Transformers/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture10_transformer_block_from_scratch.py` | Implements a full Transformer block (self-attention → residual → LayerNorm → MLP → residual → LayerNorm) in NumPy, plus a ViT patch counter and a decoder autoregressive-step simulator — every piece verified against the numerical README. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `self_attention()` | Reuses Lecture 6's Query/Key/Value mechanism |
| `residual_connection()` | `output = sublayer(x) + x` |
| `layer_norm()` | Per-token normalization, independent of other tokens/batch |
| `transformer_block()` | The full 6-step pipeline from the theory file's block diagram |
| `vit_patch_info()` | ViT's patch count and flattened dimension formulas |
| `simulate_decoding()` | The `<start>` → word-by-word → `<end>` autoregressive process |

## How to run this file

```bash
cd Deep-Learning/Lecture-10-Attention-and-Transformers/code
pip install numpy --break-system-packages
python3 dl_lecture10_transformer_block_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required for this small demo (real Transformers absolutely need GPUs/TPUs at scale — this file is for understanding the mechanics, not production training).

## Expected output (verified — produced by actually running this script)

- Check 1: residual output `[1.5,1.8,3.1]` and layer-normalized `[-0.912,-0.48,1.392]` — match Worked Examples 2 & 3 exactly.
- Check 2: a full 3-token Transformer block forward pass — every output row has mean ≈0 and standard deviation ≈1, a live confirmation that Layer Normalization is working correctly at the end of the block.
- Check 3: 224×224 image with 16×16 patches → exactly 196 patches, each 768-dimensional — matches Worked Example 1 exactly.
- Check 4: a full decoder simulation for "I am happy" — 4 steps total, matching the theory file's own worked walkthrough exactly.

`[← Exercises](../exercises/dl_lecture10_exercises.md) · [🔝 Top](#dl-lecture-10--code) · [🔝 Lecture Hub](../README.md)`
