# DL Lecture 12 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-12--code)`

> Folder: `Deep-Learning/Lecture-12-Encoder-Decoder-and-VAE/code/`
> This is the final code folder of the Deep Learning course.

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture12_vae_from_scratch.py` | Implements compression-ratio math, closed-form Gaussian KL divergence, the reparameterization trick, and a full mini VAE (encoder → reparameterize → decoder → generation) in NumPy — every formula checked against the numerical README. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function/class | Theory/Numerical concept |
|---|---|
| `compression_ratio()` | Input size ÷ latent size |
| `vae_encoder_output_size()` | `2×N` — mean + std-dev vectors |
| `kl_divergence_gaussian()` | `0.5·Σ(σ²+μ²−1−ln(σ²))` |
| `reparameterize()` | `z = μ + σ×ε` |
| `MiniVAE.encode()` | The encoder producing μ, σ |
| `MiniVAE.forward()` | Full training-time pass: encode → reparameterize → decode → KL |
| `MiniVAE.generate_new_sample()` | The two-step "sample from prior, then decode" generation process |

## How to run this file

```bash
cd Deep-Learning/Lecture-12-Encoder-Decoder-and-VAE/code
pip install numpy --break-system-packages
python3 dl_lecture12_vae_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required for this small demo (real VAEs for images/audio/time-series absolutely benefit from GPU training — this file is for understanding the mechanics end to end).

## Expected output (verified — produced by actually running this script)

- Check 1: compression ratios 819.2 and 40.625, VAE output size 128 for N=64 — all match the numerical file exactly.
- Check 2: KL divergence ≈1.044 for the example distribution, and exactly 0.0 when the encoder's output matches the prior — matches Worked Example 3 exactly.
- Check 3: reparameterized z=2.5 — matches Worked Example 4 exactly.
- Check 4: a full, working (though untrained/random-weight) mini VAE — encoding a real input into μ/σ, sampling via reparameterization, decoding a reconstruction, computing that input's KL divergence, and then generating 3 brand-new synthetic samples purely by sampling from the prior and decoding — a live, hands-on demonstration of every mechanism described in the theory file, from end to end.

## Where to go from here

This is the final lecture's code file in the course. A natural next step: swap `MiniVAE`'s single linear encoder/decoder layers for small multi-layer networks (reusing the from-scratch 2-layer network pattern from Lecture 2's code, or moving to a real framework like PyTorch), and train it on a real small image dataset (e.g., MNIST) to see the reconstruction and generation quality improve from "random noise" (as in this demo) to genuinely recognizable digit reconstructions and generations.

`[← Exercises](../exercises/dl_lecture12_exercises.md) · [🔝 Top](#dl-lecture-12--code) · [🔝 Lecture Hub](../README.md)`
