# DL Lecture 06 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-06--code)`

> Folder: `Deep-Learning/Lecture-06-Attention/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture06_attention_from_scratch.py` | Implements both classic encoder-decoder attention and self-attention (Query/Key/Value) from scratch in NumPy, verified against every numerical README example, plus a bigger 4-word self-attention demo. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `softmax()` | Converts raw scores into a valid probability distribution |
| `compute_attention_context()` | `alpha_ti = softmax(e_ti)`, `c_t = Σ alpha_ti·h_i` |
| `check_encoder_decoder_attention()` | Reproduces Worked Examples 1, 2, and 3 (including the plain-average comparison) |
| `self_attention()` | `Q=XW_Q, K=XW_K, V=XW_V`, `Attention(Q,K,V)=softmax(QKᵀ/√d_k)·V` |
| `check_self_attention()` | Reproduces Worked Example 4 exactly |
| `run_bigger_demo()` | A more realistic 4-word sentence, showing a full attention weight MATRIX (every word attending to every other word) |

## How to run this file

```bash
cd Deep-Learning/Lecture-06-Attention/code
pip install numpy --break-system-packages
python3 dl_lecture06_attention_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required.

## Expected output (verified — produced by actually running this script)

- Check 1: attention weights `[0.2506, 0.0559, 0.0125, 0.6811]` (sum exactly 1.0), context vector `[0.2630, 0.0684, 0.9316]` — matches the numerical file (tiny rounding only).
- Check 2: word 1's self-attention output `[1.3395, 0.6605]` — matches Worked Example 4 almost exactly.
- Check 3: a full 4×4 attention weight matrix for a random toy sentence, where every row sums to exactly 1.0 — this is what a real (though untrained) self-attention layer's weight matrix looks like, the same shape of object visualized in Transformer "attention map" diagrams you'll see in later coursework.

`[← Exercises](../exercises/dl_lecture06_exercises.md) · [🔝 Top](#dl-lecture-06--code) · [🔝 Lecture Hub](../README.md)`
