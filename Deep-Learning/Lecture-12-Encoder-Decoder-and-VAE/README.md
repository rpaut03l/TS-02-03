# Lecture 12 — Encoder-Decoder and Variational Autoencoders

`[← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-12--encoder-decoder-and-variational-autoencoders)`

**Instructor:** Dr. Anushka Joshi, IIT Jodhpur | **Date:** May 2026 | **Source slides:** "Encoder Decoder and Variational Autoencoder" deck
**This is the final lecture of the Deep Learning course.**

Covers general Encoder-Decoder models, Autoencoders as a special case (with a bottleneck latent space), Denoising Autoencoders, and Variational Autoencoders (VAE) — the prior/likelihood/posterior distributions, the ELBO/KL-divergence loss, the reparameterization trick, and generation via sampling.

## Files in this lecture

| File | Focus |
|---|---|
| 📘 [`theory/dl_lecture12_vae_theory.md`](theory/dl_lecture12_vae_theory.md) | Encoder-decoder vs autoencoder, denoising AE, VAE distributions and ELBO |
| 🔢 [`numerical/dl_lecture12_vae_numerical.md`](numerical/dl_lecture12_vae_numerical.md) | Compression ratios, KL divergence by hand, reparameterization, generation |
| ✍️ [`practice/dl_lecture12_vae_practice.md`](practice/dl_lecture12_vae_practice.md) | Fill-in-blank, VAE distribution matching, interview Qs |
| 🧪 [`exercises/dl_lecture12_exercises.md`](exercises/dl_lecture12_exercises.md) | Tiered Easy/Medium/Hard question bank with answer key |
| 💻 [`code/`](code/README.md) | A full mini VAE (encode, reparameterize, decode, generate) built from scratch in NumPy |

## Suggested reading order

Theory → Numerical → Practice → Exercises → Code.

## Course complete

This is the last lecture in the Deep Learning course documentation. See the [Deep Learning course hub](../README.md) for the full 12-lecture index, or revisit [Lecture 01](../Lecture-01-Introduction-to-Deep-Learning/README.md) to start again from the beginning.

A **bonus 13th lecture on Generative Adversarial Networks** — not part of the original slide deck, but built to complete the generative-modeling picture alongside this lecture's VAE content — is available: [Lecture 13 →](../Lecture-13-Generative-Adversarial-Networks/README.md)

`[← Lecture 11](../Lecture-11-Real-World-End-to-End-Framework/README.md) · [← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-12--encoder-decoder-and-variational-autoencoders) · [Next: Lecture 13 (Bonus) →](../Lecture-13-Generative-Adversarial-Networks/README.md)`
