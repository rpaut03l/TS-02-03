# DL Lecture 13 (Bonus) — Generative Adversarial Networks (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice)`

> Folder: `Deep-Learning/Lecture-13-Generative-Adversarial-Networks/practice/`
> Pairs with: [`theory/dl_lecture13_gan_theory.md`](../theory/dl_lecture13_gan_theory.md) · [`numerical/dl_lecture13_gan_numerical.md`](../numerical/dl_lecture13_gan_numerical.md) · [`exercises/dl_lecture13_exercises.md`](../exercises/dl_lecture13_exercises.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## Table of Contents
1. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
2. [Explain-It-Back Prompts](#explain-it-back-prompts)
3. [Quick-Fire True / False](#quick-fire-true--false)
4. [GAN vs VAE Matching Drill](#gan-vs-vae-matching-drill)
5. [Mini Interview-Style Round](#mini-interview-style-round)
6. [Summary](#summary)

---

## Concept Check — Fill in the Blank

1. The two networks in a GAN are the ______ and the ______.
2. In the minimax objective, the Discriminator plays ______, and the Generator plays ______.
3. The practical fix for the Generator's vanishing gradient problem is called the ______ loss.
4. ______ is GAN training's most notorious failure mode, where the Generator produces low-diversity output.
5. At the theoretical global optimum, D(x) = ______ for every input.

<details>
<summary>Show answers</summary>

1. Generator; Discriminator
2. max; min
3. non-saturating
4. Mode collapse
5. 0.5
</details>

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice)`

---

## Explain-It-Back Prompts

1. Explain the forger-versus-detective analogy for GANs in your own words.
2. Walk through the alternating training loop (Discriminator step, then Generator step) from memory.
3. Explain why the Generator's gradient vanishes early in training, and how the non-saturating loss fixes it, using Worked Example 3's numbers.
4. Explain mode collapse, and why it's a direct consequence of the adversarial training dynamic rather than a random bug.
5. Explain the GAN vs VAE trade-off (sharpness vs stability) in your own words.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice)`

---

## Quick-Fire True / False

1. The Generator directly sees real training data during its forward pass. — **False** (it only receives noise z; it learns indirectly via the Discriminator's gradient signal).
2. Both networks are updated simultaneously from the same forward pass, without freezing either one. — **False** (training alternates, freezing one network while updating the other).
3. GANs can compute an explicit likelihood value for a given data point. — **False** (GANs are implicit density models; they cannot).
4. DCGAN uses LeakyReLU in the Discriminator and ReLU/Tanh in the Generator. — **True**.
5. A lower FID score indicates a better-performing GAN. — **True**.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice)`

---

## GAN vs VAE Matching Drill

| Property | GAN | VAE |
|---|---|---|
| Density model type | ? | ? |
| Typical output sharpness | ? | ? |
| Typical training stability | ? | ? |

<details>
<summary>Show answers</summary>

Density model type: GAN=implicit, VAE=explicit. Typical output sharpness: GAN=sharper, VAE=smoother/blurrier. Typical training stability: GAN=less stable (mode collapse, vanishing gradients), VAE=more stable (single reconstruction+KL objective).
</details>

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice)`

---

## Mini Interview-Style Round

**Q1.** "Your GAN's generated images all look nearly identical after a few thousand training steps. What's your first hypothesis, and what would you check?"

<details>
<summary>Show answer</summary>

First hypothesis: mode collapse — the Generator has found one (or a very small number of) outputs that reliably fool the current Discriminator and has stopped exploring the rest of the data distribution. You'd check by feeding many DIFFERENT random noise vectors z into the Generator and visually/statistically comparing the outputs — if wildly different z inputs produce nearly identical outputs, that confirms mode collapse. Common fixes include using mini-batch discrimination, unrolled GANs, or switching to a more stable training objective like Wasserstein GAN.
</details>

**Q2.** "A teammate implements a GAN but updates both the Generator and Discriminator from the exact same forward pass, without freezing either network. Why is this a bug?"

<details>
<summary>Show answer</summary>

Updating both networks simultaneously means each network's gradient computation is based on a target (the other network's CURRENT weights) that's ALSO about to change in the same step — this is even less stable than the already-tricky standard alternating approach, since neither network gets a stable, fixed target to optimize against for even one clean step. The standard discipline is: freeze the Generator, update only the Discriminator using the current Generator's outputs; then freeze the (now-updated) Discriminator, and update only the Generator using ITS feedback.
</details>

**Q3.** "Why can't you simply compute 'the probability of this specific image occurring' with a trained GAN, the way you could with a VAE?"

<details>
<summary>Show answer</summary>

A GAN is an implicit density model — it was never trained to model p(x) explicitly at all; it only ever learned a mapping from noise z to realistic-looking samples, with no probabilistic interpretation attached to that mapping. A VAE, by contrast, is explicitly built around probability distributions (the prior, likelihood, and posterior from Lecture 12) and its ELBO training objective directly relates to a (lower bound on) the data's log-likelihood — giving it at least an approximate way to answer "how likely is this data point," a question a GAN structurally cannot answer.
</details>

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice)`

---

## Summary

This practice file drills the bonus GAN lecture's adversarial training concepts through active recall. A fill-in-the-blank check reinforces the Generator/Discriminator naming, the minimax max/min roles, the non-saturating loss, mode collapse terminology, and the global optimum's D(x)=0.5 condition. Five explain-it-back prompts push you to reproduce the forger-detective analogy, the alternating training loop, the vanishing-gradient-and-fix numeric argument, why mode collapse is a structural consequence rather than a random bug, and the GAN-vs-VAE sharpness/stability trade-off. A quick-fire true/false round and a GAN-vs-VAE matching drill test both conceptual accuracy and precise comparative terminology. A three-question interview-style round rehearses realistic diagnostic reasoning: spotting mode collapse from repetitive outputs, explaining why simultaneous (non-alternating) updates are a training bug, and explaining why GANs structurally cannot compute explicit likelihoods the way VAEs can. Move to the exercises file next for this bonus lecture's tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture13_gan_numerical.md) · [🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-practice) · [Next: Exercises →](../exercises/dl_lecture13_exercises.md)`
