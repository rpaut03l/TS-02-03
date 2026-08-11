# DL Lecture 12 — Exercise Bank (Encoder-Decoder and VAE)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-12--exercise-bank-encoder-decoder-and-vae)`

> Folder: `Deep-Learning/Lecture-12-Encoder-Decoder-and-VAE/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture12_vae_theory.md) · [numerical](../numerical/dl_lecture12_vae_numerical.md) · [practice](../practice/dl_lecture12_vae_practice.md)
> This is the final exercise bank of the Deep Learning course.

---

## 🟢 Easy — Definitions & Recall

**Q12.1.** What defines an Autoencoder as a special case of an Encoder-Decoder model?

**Q12.2.** Name the explicit and implicit density model examples given in the lecture.

**Q12.3.** What two vectors does a VAE's encoder output, instead of one?

**Q12.4.** Write the reparameterization trick formula.

**Q12.5.** What was Denoising Autoencoders' primary design motivation?

---

## 🟡 Medium — Applied Reasoning

**Q12.6.** For a 4096-pixel input compressed to 8 latent numbers, compute the compression ratio.

**Q12.7.** For latent dimension N=128, compute a VAE's total encoder output size.

**Q12.8.** For μ=[1.0, 0.0], σ=[1.0, 1.0], compute the KL divergence against the standard normal prior.

**Q12.9.** Explain why the true posterior p_θ(z|x) is generally described as "intractable," and why this motivates learning an approximate posterior instead.

**Q12.10.** Explain why an Autoencoder's latent space "acts as a bottleneck" while a general Encoder-Decoder's latent space does not have to.

---

## 🔴 Hard — Derivation & Multi-Step

**Q12.11.** For μ=-1.0, σ=2.0, ε=0.5, compute z using the reparameterization trick, and explain what would happen to z if ε were instead sampled as -0.5.

**Q12.12.** A VAE is trained on 28×28 grayscale images (784 pixels) with a latent dimension of 16. Compute (a) the plain-autoencoder-equivalent compression ratio, and (b) the VAE's total encoder output size, and (c) the ratio of encoder output size to original input size.

**Q12.13.** Explain, step by step, the full two-stage process for generating a brand-new synthetic image from a trained VAE, and explain what would go wrong if you tried to do this with a plain (non-variational) autoencoder instead.

**Q12.14.** For a 2-dimensional latent space, μ=[0.2, -0.3], σ=[0.9, 1.1], compute the KL divergence against the standard normal prior, and compare the result to a case where σ=[1.0,1.0] (with the same μ) — explain in words what this comparison reveals about how σ moving away from 1 affects the KL penalty, independent of μ.

`[🔝 Top](#dl-lecture-12--exercise-bank-encoder-decoder-and-vae)`

---

## Answer Key

<details>
<summary>Q12.1 – Q12.5 (Easy)</summary>

- **Q12.1:** Input = Target Output — the model is trained to reconstruct its own input.
- **Q12.2:** VAE = explicit density model example. GAN = implicit density model example.
- **Q12.3:** A mean vector (μ) and a standard deviation vector (σ).
- **Q12.4:** `z = μ + σ × ε`, where `ε ~ N(0,1)`.
- **Q12.5:** To fight overfitting in autoencoders, particularly when there are more parameters than data points — by forcing reconstruction from corrupted inputs, the model must learn genuine relationships among features rather than memorizing exact values.
</details>

<details>
<summary>Q12.6 – Q12.10 (Medium)</summary>

- **Q12.6:** 4096/8 = **512:1**.
- **Q12.7:** 128 × 2 = **256** total output values (128 mean + 128 std dev).
- **Q12.8:** Dim1: 1²+1²−1−ln(1²) = 1+1−1−0 = 1. Dim2: 1²+0²−1−ln(1²) = 1+0−1−0 = 0. Sum = 1. KL = 0.5×1 = **0.5**.
- **Q12.9:** The true posterior requires computing p(x) (the overall probability of the data) as a normalizing constant, via an integral over ALL possible values of z — for complex, high-dimensional, non-linear models (like a deep neural network decoder), this integral has no closed-form solution and is computationally infeasible to evaluate directly. Since the true posterior can't be computed exactly, VAEs instead train a separate encoder network to LEARN an approximate posterior `q_φ(z|x)` that gets as close as possible to the true one, measured via KL divergence — turning an intractable exact-computation problem into a tractable optimization problem.
- **Q12.10:** In an Autoencoder, the training objective IS reconstruction of the input from the latent code — the latent space MUST be smaller than the input, or else the model could trivially "cheat" by just passing the input through unchanged (a bottleneck forces genuine compression/learning to happen). In a general Encoder-Decoder, the target output is something DIFFERENT from the input (e.g., a translation, or predicted ground motion from source parameters) — there's no risk of a trivial "just copy the input" shortcut, so the latent space is free to be as large or small as whatever helps the decoder's actual (different) task, including larger than the input if that's genuinely useful.
</details>

<details>
<summary>Q12.11 – Q12.14 (Hard)</summary>

- **Q12.11:** z = -1.0 + 2.0×0.5 = -1.0+1.0 = **0.0**. If ε were instead -0.5: z = -1.0 + 2.0×(-0.5) = -1.0-1.0 = **-2.0**. Different random draws of ε produce different sampled z values for the SAME μ and σ — this is exactly the point: μ and σ define a whole DISTRIBUTION of plausible z values, and each training/generation pass draws a fresh random ε to sample a specific point from that distribution.
- **Q12.12:** (a) 784/16 = **49:1**. (b) 16×2 = **32** total encoder output values. (c) 32/784 ≈ **0.0408**, i.e. the VAE's encoder output is only about 4.1% the size of the original input — still a substantial compression, even after doubling for the mean+std-dev split.
- **Q12.13:** Step 1: sample a latent vector z from the PRIOR distribution p_θ(z) (typically standard normal, N(0,1), for each latent dimension) — this requires no input image at all, just a random draw. Step 2: feed this sampled z through the TRAINED DECODER network to obtain p_θ(x|z), producing a brand-new synthetic image. If you tried this with a plain (non-variational) autoencoder: there is no defined prior distribution to sample FROM in the first place (a plain AE has no `p_θ(z)` — only whatever specific latent points happened to result from encoding real training images), and even if you picked some arbitrary point in latent space, there's no guarantee the AE's decoder would produce anything realistic from it, since the AE's latent space was never regularized/organized to make nearby or randomly-sampled points decode sensibly — a genuine structural difference that specifically enables VAEs (and not plain AEs) to generate new data reliably.
- **Q12.14:** For σ=[0.9,1.1], μ=[0.2,-0.3]: Dim1: 0.9²+0.2²−1−ln(0.9²) = 0.81+0.04−1−ln(0.81) = 0.81+0.04-1-(-0.2107) ≈ 0.0607. Dim2: 1.1²+(-0.3)²−1−ln(1.1²) = 1.21+0.09−1−ln(1.21) = 1.21+0.09-1-0.1906 ≈ 0.1094. Sum ≈ 0.1701. KL ≈ 0.5×0.1701 ≈ **0.0851**. For σ=[1.0,1.0] (same μ): Dim1: 1+0.04-1-0=0.04. Dim2: 1+0.09-1-0=0.09. Sum=0.13. KL=0.5×0.13=**0.065**. The σ=[0.9,1.1] case has a SLIGHTLY HIGHER KL (0.0851 vs 0.065) than the σ=[1,1] case with the same μ — showing that moving σ away from 1 in EITHER direction (smaller like 0.9, or larger like 1.1) adds extra KL penalty on top of whatever penalty μ alone contributes; the KL term penalizes any std-dev deviating from 1, not just std-devs that are too small or too large in one specific direction.
</details>

`[🔝 Top](#dl-lecture-12--exercise-bank-encoder-decoder-and-vae)`

---

## Summary

This final exercise bank of the course drills Encoder-Decoder and VAE concepts across three tiers. Easy questions recall the autoencoder's input-equals-target-output definition, explicit/implicit density model examples, the VAE encoder's dual mean/std-dev output, the reparameterization formula, and denoising autoencoders' overfitting-fighting motivation. Medium questions apply compression-ratio and encoder-output-size formulas to new numbers (512:1 compression; 256-value VAE output for N=128), compute a full KL divergence (0.5), and explain conceptually why the true posterior is intractable and why autoencoder latent spaces specifically must be bottlenecks while general encoder-decoder latent spaces need not be. Hard questions require deeper derivations: a reparameterization-trick computation showing how different ε draws (0.5 vs -0.5) produce very different sampled z values (0.0 vs -2.0) from the identical μ,σ, a full three-part VAE-on-MNIST-style compression analysis (49:1 ratio, 32-value encoder output, ~4.1% of original size), a complete worded explanation of why plain autoencoders can't reliably generate new data the way VAEs can, and a comparative KL divergence calculation showing that σ deviating from 1 in EITHER direction (0.9 or 1.1) adds extra KL penalty beyond whatever μ alone contributes. All answers are fully worked and spoiler-tagged — this closes out the exercise-bank sequence for the entire twelve-lecture Deep Learning course.

`[← Practice](../practice/dl_lecture12_vae_practice.md) · [🔝 Top](#dl-lecture-12--exercise-bank-encoder-decoder-and-vae) · [Code →](../code/README.md)`
