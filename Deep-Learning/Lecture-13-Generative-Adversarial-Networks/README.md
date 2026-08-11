# Lecture 13 (Bonus) — Generative Adversarial Networks

`[← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-13-bonus--generative-adversarial-networks)`

> ⚠️ **Bonus lecture — not part of the original 541-page slide deck.** The source material only mentions GANs twice, as a one-line comparison point in Lecture 12. This lecture was built in full, in the same house style as the rest of the course, to cover the other major deep generative model family alongside VAE. Treat it as supplementary enrichment, not something guaranteed to appear on an exam based on the official slides.

Covers the Generator/Discriminator adversarial framework, the minimax objective, the alternating training loop, the vanishing gradient problem and its non-saturating-loss fix, mode collapse, the theoretical global optimum, DCGAN's practical design guidelines, Conditional GANs, evaluation via Inception Score/FID, and a full GAN-vs-VAE comparison.

## Files in this lecture

| File | Focus |
|---|---|
| 📘 [`theory/dl_lecture13_gan_theory.md`](theory/dl_lecture13_gan_theory.md) | Generator/Discriminator, minimax, mode collapse, DCGAN, GAN vs VAE |
| 🔢 [`numerical/dl_lecture13_gan_numerical.md`](numerical/dl_lecture13_gan_numerical.md) | Loss calculations, gradient magnitude comparison, global optimum, toy FID |
| ✍️ [`practice/dl_lecture13_gan_practice.md`](practice/dl_lecture13_gan_practice.md) | Fill-in-blank, GAN vs VAE matching, interview Qs |
| 🧪 [`exercises/dl_lecture13_exercises.md`](exercises/dl_lecture13_exercises.md) | Tiered Easy/Medium/Hard question bank with answer key |
| 💻 [`code/`](code/README.md) | Loss/gradient formulas plus a genuinely trainable tiny 1D GAN in NumPy |

## Suggested reading order

Theory → Numerical → Practice → Exercises → Code.

`[← Lecture 12](../Lecture-12-Encoder-Decoder-and-VAE/README.md) · [← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-13-bonus--generative-adversarial-networks) · [Next: Lecture 14 (Bonus) →](../Lecture-14-Restricted-Boltzmann-Machines/README.md)`
