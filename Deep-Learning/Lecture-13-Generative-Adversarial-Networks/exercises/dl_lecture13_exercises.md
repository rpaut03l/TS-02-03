# DL Lecture 13 (Bonus) — Exercise Bank (Generative Adversarial Networks)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-13-bonus--exercise-bank-generative-adversarial-networks)`

> Folder: `Deep-Learning/Lecture-13-Generative-Adversarial-Networks/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture13_gan_theory.md) · [numerical](../numerical/dl_lecture13_gan_numerical.md) · [practice](../practice/dl_lecture13_gan_practice.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## 🟢 Easy — Definitions & Recall

**Q13.1.** Write the full GAN minimax objective function.

**Q13.2.** What does the Discriminator output, and what range is it in?

**Q13.3.** Name the two metrics used to evaluate GANs, and state whether higher or lower is better for each.

**Q13.4.** What activation functions does DCGAN recommend for the Generator's output layer and the Discriminator's hidden layers?

**Q13.5.** What does a Conditional GAN add, compared to a plain GAN?

---

## 🟡 Medium — Applied Reasoning

**Q13.6.** For D(x)=0.8 and D(G(z))=0.3, compute the Discriminator's loss.

**Q13.7.** For D(G(z))=0.05, compute both the minimax and non-saturating loss gradient magnitudes, and their ratio.

**Q13.8.** Explain why DCGAN replaces pooling layers with strided convolutions, connecting this back to Lecture 3's pooling discussion.

**Q13.9.** Explain why a GAN's Discriminator loss does NOT go to zero at the theoretical global optimum, using the D(x)=0.5 condition.

**Q13.10.** Explain, in your own words, why the Generator's training signal is described as "a moving target," unlike a standard classifier's fixed labels.

---

## 🔴 Hard — Derivation & Multi-Step

**Q13.11.** For a 1D toy FID calculation with real data μ=1.0, σ=0.8 and generated data μ=1.3, σ=1.0, compute the FID.

**Q13.12.** Compute the exact Discriminator loss value at the global optimum (D(x)=D(G(z))=0.5), showing every step, and express it in terms of ln(2).

**Q13.13.** Explain, with a full worked numeric argument, why training a GAN using ONLY the original minimax Generator loss (never switching to non-saturating) could plausibly cause training to stall completely in early epochs. Use the gradient formulas to justify your answer at D(G(z))=0.001.

**Q13.14.** A Conditional GAN's Discriminator receives `D(x, y)` instead of `D(x)`. Explain what would go wrong (in terms of controllability) if only the GENERATOR received the condition y, but the Discriminator did not — i.e., the Discriminator only ever saw `D(x)`, ignoring y entirely.

`[🔝 Top](#dl-lecture-13-bonus--exercise-bank-generative-adversarial-networks)`

---

## Answer Key

<details>
<summary>Q13.1 – Q13.5 (Easy)</summary>

- **Q13.1:** `min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1-D(G(z)))]`.
- **Q13.2:** The Discriminator outputs a single probability estimate that its input is real, in the range [0, 1].
- **Q13.3:** Inception Score (higher is better) and Fréchet Inception Distance / FID (lower is better).
- **Q13.4:** Generator output layer: Tanh. Discriminator hidden layers: LeakyReLU.
- **Q13.5:** An additional condition/label input y, fed into both the Generator (`G(z,y)`) and Discriminator (`D(x,y)`), enabling controllable generation.
</details>

<details>
<summary>Q13.6 – Q13.10 (Medium)</summary>

- **Q13.6:** L_D = -[log(0.8)+log(1-0.3)] = -[log(0.8)+log(0.7)] = -[-0.2231+(-0.3567)] = -(-0.5798) = **0.5798**.
- **Q13.7:** Minimax grad = 1/(1-0.05) = 1/0.95 ≈ **1.0526**. Non-saturating grad = 1/0.05 = **20.0**. Ratio = 20.0/1.0526 ≈ **19×** larger for the non-saturating loss.
- **Q13.8:** Lecture 3 explained pooling as a fixed (non-learned) operation for shrinking spatial resolution while keeping the strongest signal. DCGAN instead uses STRIDED convolutions (in the Discriminator, for downsampling) and transposed/fractional-strided convolutions (in the Generator, for upsampling), because these ARE learnable — the network can learn the best way to resize its feature maps for the specific task of distinguishing/generating images, rather than being stuck with a fixed max/average rule.
- **Q13.9:** D(x)=0.5 means the Discriminator is maximally UNCERTAIN — it literally cannot do better than a coin flip, because real and fake data have become statistically indistinguishable. `log(0.5)` is a specific, nonzero negative number (≈-0.693), not zero — so the resulting loss (`-2×log(0.5)≈1.386`) reflects this irreducible uncertainty rather than confident correct classification (which WOULD drive the loss toward zero).
- **Q13.10:** A standard classifier has a FIXED, unchanging ground-truth label for each training example throughout training. The Generator's "target" — what counts as fooling the Discriminator — depends entirely on the CURRENT Discriminator's weights, which are themselves being updated every step; so what counted as "successfully fooling" the Discriminator last step may no longer work this step, and vice versa, making the optimization landscape the Generator navigates a constantly shifting one, unlike the fixed landscape a normal supervised classifier optimizes against.
</details>

<details>
<summary>Q13.11 – Q13.14 (Hard)</summary>

- **Q13.11:** (1.0-1.3)² + 0.8² + 1.0² - 2×√(0.8²×1.0²) = 0.09 + 0.64+1.0 - 2×√(0.64) = 0.09+1.64-2×0.8 = 0.09+1.64-1.6 = **0.13**.
- **Q13.12:** L_D = -[log(0.5)+log(1-0.5)] = -[log(0.5)+log(0.5)] = -2×log(0.5). Since log(0.5) = log(1/2) = -log(2) = **-ln(2)**, this gives L_D = -2×(-ln(2)) = **2ln(2) ≈ 1.3863**.
- **Q13.13:** At D(G(z))=0.001 (an extremely bad, very-early-training Generator), minimax gradient magnitude = 1/(1-0.001) = 1/0.999 ≈ **1.001** — essentially no stronger than at D(G(z))=0.5 (grad=2.0) or even weaker in relative terms; the gradient barely responds to how catastrophically bad the Generator currently is. Meanwhile non-saturating gradient = 1/0.001 = **1000** — a massive, appropriately large corrective signal. If training relied ONLY on the minimax loss, the Generator would receive almost the SAME tiny gradient regardless of whether it's moderately bad (D(G(z))=0.1, grad≈1.11) or catastrophically bad (D(G(z))=0.001, grad≈1.001) — providing essentially no useful "how urgently must I improve" signal exactly when that signal matters most, plausibly causing training to stall with the Generator unable to escape its initial poor state.
- **Q13.14:** If the Discriminator only ever sees `D(x)` (ignoring y), it has no way to check whether a generated sample `G(z,y)` actually MATCHES its intended condition y — it can only judge overall realism, not label-correctness. The Generator would then have no gradient pressure to make its output correspond to the requested condition y at all; it could satisfy the Discriminator by producing realistic-looking samples of ANY class, completely ignoring y, defeating the entire purpose of conditioning. Both networks must see y for conditioning to actually work — the Discriminator needs it to check "is x a REAL example specifically of class y," not just "is x realistic in general."
</details>

`[🔝 Top](#dl-lecture-13-bonus--exercise-bank-generative-adversarial-networks)`

---

## Summary

This bonus exercise bank drills GAN training mechanics across three tiers. Easy questions recall the minimax objective formula, the Discriminator's output range, GAN evaluation metrics, DCGAN's recommended activations, and what Conditional GANs add. Medium questions apply the Discriminator loss formula to new values (0.5798), compute and compare gradient magnitudes at D(G(z))=0.05 (a 19× ratio favoring the non-saturating loss), connect DCGAN's strided-convolution design choice back to Lecture 3's pooling discussion, explain why the global optimum's loss is nonzero, and articulate why the Generator's training target is a "moving target." Hard questions require deeper derivations: a full toy FID calculation (0.13), an exact symbolic derivation of the global optimum's Discriminator loss as 2ln(2), a detailed numeric argument for why relying solely on the minimax loss could stall training at D(G(z))=0.001 (minimax gradient ≈1.001 vs non-saturating ≈1000), and a conceptual explanation of why BOTH networks — not just the Generator — must receive the conditioning label y for a Conditional GAN to actually work. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture13_gan_practice.md) · [🔝 Top](#dl-lecture-13-bonus--exercise-bank-generative-adversarial-networks) · [Code →](../code/README.md)`
