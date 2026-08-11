# DL Lecture 12 — Encoder-Decoder and VAE (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

> Folder: `Deep-Learning/Lecture-12-Encoder-Decoder-and-VAE/numerical/`
> Pairs with: [`theory/dl_lecture12_vae_theory.md`](../theory/dl_lecture12_vae_theory.md) · [`practice/dl_lecture12_vae_practice.md`](../practice/dl_lecture12_vae_practice.md) · [`exercises/dl_lecture12_exercises.md`](../exercises/dl_lecture12_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Autoencoder Compression Ratios](#worked-example-1--autoencoder-compression-ratios)
3. [Worked Example 2 — Standard AE vs VAE Output Size](#worked-example-2--standard-ae-vs-vae-output-size)
4. [Worked Example 3 — KL Divergence for a Gaussian, By Hand](#worked-example-3--kl-divergence-for-a-gaussian-by-hand)
5. [Worked Example 4 — The Reparameterization Trick](#worked-example-4--the-reparameterization-trick)
6. [Worked Example 5 — Sampling and Decoding a New Data Point](#worked-example-5--sampling-and-decoding-a-new-data-point)
7. [Master Formula Cheatsheet](#master-formula-cheatsheet)
8. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
9. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| N | latent space dimension |
| z | a latent vector (either a fixed point, for a plain autoencoder, or a sample, for a VAE) |
| μ (mu) | the mean vector output by a VAE's encoder |
| σ (sigma) | the standard deviation vector output by a VAE's encoder |
| σ² | variance (σ multiplied by itself) |
| ε (epsilon) | a random noise sample drawn from the standard normal distribution N(0,1) |
| p_θ(z) | the prior distribution over latent variables |
| KL | Kullback-Leibler divergence, a measure of how different two probability distributions are |

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Worked Example 1 — Autoencoder Compression Ratios

**Given:** the lecture's own two examples — (a) 16,384 pixels compressed to 20 latent numbers, (b) a 1,300-point sonobuoy signal compressed to 32 latent values.

**Step 1 — Compute compression ratio for example (a).**
```
ratio_a = 16,384 / 20 = 819.2
```

**Step 2 — Compute compression ratio for example (b).**
```
ratio_b = 1,300 / 32 = 40.625
```

**Result:** the image autoencoder achieves an 819:1 compression ratio, while the sonobuoy autoencoder achieves a more modest 40.6:1 ratio. Both are dramatic compressions, but the difference illustrates that the "right" amount of compression genuinely depends on the application — images (with lots of redundant, spatially-correlated pixel information, per Lecture 1's data-type/invariance discussion) can often be compressed far more aggressively than a 1D acoustic signal while still reconstructing well.

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Worked Example 2 — Standard AE vs VAE Output Size

**Given:** latent dimension N=64.

**Step 1 — Compute a standard autoencoder's latent output size.**
```
Standard AE output size = N = 64
```

**Step 2 — Compute a VAE's latent output size (mean vector + standard deviation vector).**
```
VAE output size = N (mean) + N (std dev) = 64 + 64 = 128
```

**Step 3 — Compute the ratio.**
```
ratio = 128 / 64 = 2.0
```

**Result: a VAE's encoder outputs EXACTLY twice as many numbers as a plain autoencoder's encoder**, for the same latent dimension N — a direct, memorable numeric consequence of the theory file's "distribution, not a point" idea: you need both a mean AND a spread (standard deviation) to define a distribution, versus just one number per dimension for a fixed point.

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Worked Example 3 — KL Divergence for a Gaussian, By Hand

**Given:** for a VAE, when both the prior and posterior are Gaussian, there's a convenient CLOSED-FORM formula for KL divergence between the encoder's learned distribution `N(μ, σ²)` and the standard normal prior `N(0, 1)`:
```
KL = 0.5 * sum_over_dimensions( sigma_i^2 + mu_i^2 - 1 - ln(sigma_i^2) )
```

Take a 3-dimensional latent space with `μ = [0.5, -0.2, 1.0]` and `σ = [1.2, 0.8, 0.5]`.

**Step 1 — Compute each dimension's contribution.**
```
Dim 1: 1.2^2 + 0.5^2 - 1 - ln(1.2^2) = 1.44 + 0.25 - 1 - ln(1.44) = 1.44+0.25-1-0.3646 ≈ 0.3254
Dim 2: 0.8^2 + (-0.2)^2 - 1 - ln(0.8^2) = 0.64+0.04-1-ln(0.64) = 0.64+0.04-1-(-0.4463) ≈ 0.1263
Dim 3: 0.5^2 + 1.0^2 - 1 - ln(0.5^2) = 0.25+1.0-1-ln(0.25) = 0.25+1.0-1-(-1.3863) ≈ 1.6363
```

**Step 2 — Sum the three dimensions and multiply by 0.5.**
```
sum = 0.3254 + 0.1263 + 1.6363 ≈ 2.088
KL = 0.5 x 2.088 ≈ 1.044
```

**Result: KL ≈ 1.044.** For comparison, if the encoder had instead output EXACTLY the prior itself (μ=[0,0,0], σ=[1,1,1] — i.e., the encoder "gives up" and just outputs the prior, ignoring the actual input x), the KL divergence would be EXACTLY **0** (verify: `1²+0²-1-ln(1²) = 1+0-1-0 = 0` for every dimension) — this is the "zero penalty" baseline that the KL term in the ELBO loss pulls the encoder toward, balanced against the competing pressure to encode enough INPUT-SPECIFIC information for good reconstruction.

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Worked Example 4 — The Reparameterization Trick

**Given:** the encoder outputs `μ=2.0` and `σ=0.5` for one latent dimension. A standard-normal random sample `ε=1.0` is drawn (ε ~ N(0,1)).

**Step 1 — Apply the reparameterization formula.**
```
z = mu + sigma x epsilon = 2.0 + 0.5 x 1.0 = 2.0 + 0.5 = 2.5
```

**Result: z = 2.5.** This "reparameterization trick" is the standard practical technique for training VAEs: instead of directly sampling z from `N(μ,σ²)` (an operation that can't be backpropagated through, since randomness has no gradient), you sample a plain, FIXED-distribution random number `ε ~ N(0,1)` externally, then compute z as a simple, fully-differentiable FORMULA involving μ, σ, and ε — letting gradients flow cleanly through μ and σ during training, while ε just acts as external, non-trainable noise.

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Worked Example 5 — Sampling and Decoding a New Data Point

**Given:** a trained VAE's prior is the standard normal `N(0,1)` in each of 3 latent dimensions. We draw a random sample `z = [0.3, -1.1, 0.7]` from this prior (simulating step 1 of generation).

**Step 1 — This IS "sample z from the prior" (step 1 of the two-step generation process from the theory file).** No computation needed beyond the random draw itself — just confirm z has the right shape (3 values, matching the latent dimension).

**Step 2 — Feed z through the trained decoder to get `p_θ(x|z)`, producing a new data sample x.** (This step requires the actual trained decoder network — illustrated conceptually here, and implemented in code.) If, for example, a toy linear decoder is `x = W·z + b` with `W` a 3×2 matrix and `b` a 2-vector, applying it to z produces a brand-new, 2-dimensional synthetic data point that the VAE has never seen during training — yet, because z was drawn from the SAME prior distribution the encoder was trained to approximate, the resulting x should resemble the general "style" of the real training data.

**Result:** this two-step process — sample from the prior, then decode — is exactly how EVERY new data point a trained VAE generates comes to exist; there is no other generation mechanism.

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Compression ratio | `input_size / latent_size` |
| VAE output size (vs standard AE) | `2 x N` (mean + std dev) vs `N` |
| KL divergence (Gaussian vs standard normal) | `0.5 x Σ(σᵢ² + μᵢ² − 1 − ln(σᵢ²))` |
| Reparameterization trick | `z = μ + σ x ε`, where `ε ~ N(0,1)` |
| VAE generation | 1. Sample `z ~ p_θ(z)`  2. Decode `x ~ p_θ(x\|z)` |

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting the `-1` and `-ln(σ²)` terms in the KL divergence formula — all four terms (`σ²`, `μ²`, `-1`, `-ln(σ²)`) are required; dropping any one gives a wrong answer.
- **Trap:** computing `ln(σ²)` as `2ln(σ)` incorrectly, or mixing up σ and σ² throughout the formula — always be explicit about which one you're given, and square/unsquare consistently.
- **Trap:** believing z is sampled directly from `N(μ,σ²)` during training — the REPARAMETERIZATION TRICK is used specifically because direct sampling isn't differentiable; always mention `ε ~ N(0,1)` and the `z=μ+σε` formula when explaining VAE training.
- **Exam hack:** the "KL=0 exactly when the encoder outputs the prior itself" fact (Worked Example 3) is a favourite conceptual check — verify you can explain WHY this makes sense (the two distributions being compared become literally identical) as well as compute it.
- **Exam hack:** compression-ratio questions (Worked Example 1) are simple, fast, easy marks — always show the division explicitly rather than just stating a rounded ratio.

`[🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical)`

---

## Summary

This file worked every VAE-related calculation from the theory file into fully shown arithmetic. Computing compression ratios for the lecture's own two examples gave 819.2:1 for the 16,384-pixel-to-20-number image autoencoder and 40.625:1 for the 1,300-point-to-32-value sonobuoy autoencoder, illustrating how compression aggressiveness varies by data type. Comparing standard autoencoder and VAE output sizes for latent dimension N=64 showed the VAE needs exactly 128 output values (a mean vector plus a standard deviation vector) versus a plain autoencoder's 64 — precisely double, directly reflecting the "distribution, not a point" conceptual shift. A full hand-computed KL divergence between a 3-dimensional learned Gaussian (μ=[0.5,-0.2,1.0], σ=[1.2,0.8,0.5]) and the standard normal prior gave KL≈1.044, with a companion check confirming KL=0 exactly when the encoder's output matches the prior itself. The reparameterization trick was demonstrated with a single concrete example (μ=2.0, σ=0.5, ε=1.0 → z=2.5), showing exactly how VAEs make sampling differentiable for training. Finally, a full "sample then decode" generation walkthrough demonstrated the exact two-step mechanism by which every new VAE-generated sample comes into existence. The master formula table consolidates every reusable calculation from this final lecture for fast review.

`[← Theory](../theory/dl_lecture12_vae_theory.md) · [🔝 Top](#dl-lecture-12--encoder-decoder-and-vae-numerical) · [Next: Practice →](../practice/dl_lecture12_vae_practice.md)`
