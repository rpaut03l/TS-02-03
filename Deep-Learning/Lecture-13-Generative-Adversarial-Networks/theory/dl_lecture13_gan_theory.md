# DL Lecture 13 (Bonus) — Generative Adversarial Networks (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

> Folder: `Deep-Learning/Lecture-13-Generative-Adversarial-Networks/theory/`
> Pairs with: [`numerical/dl_lecture13_gan_numerical.md`](../numerical/dl_lecture13_gan_numerical.md) · [`practice/dl_lecture13_gan_practice.md`](../practice/dl_lecture13_gan_practice.md) · [`exercises/dl_lecture13_exercises.md`](../exercises/dl_lecture13_exercises.md)
> **⚠️ Bonus lecture — not part of the original 541-page slide deck.** The source material only mentions GANs twice (as VAE's "implicit density model" counterpart in Lecture 12). This lecture was built from scratch, in the same house style as the rest of the course, to fill that gap, since GAN is one of the two foundational deep generative model families (alongside VAE) and comes up constantly in interviews and further coursework. Treat it as supplementary, not something your instructor necessarily covered or will test on directly — but genuinely worth knowing.

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Recap: Where GANs Fit Among Generative Models](#recap-where-gans-fit-among-generative-models)
3. [The Two Networks: Generator and Discriminator](#the-two-networks-generator-and-discriminator)
4. [The Adversarial Game, Step by Step](#the-adversarial-game-step-by-step)
5. [The Minimax Objective Function](#the-minimax-objective-function)
6. [Why "Minimax"? Reading the Formula Like a Story](#why-minimax-reading-the-formula-like-a-story)
7. [Training a GAN — The Alternating Loop](#training-a-gan--the-alternating-loop)
8. [The Vanishing Gradient Problem in GANs](#the-vanishing-gradient-problem-in-gans)
9. [The Non-Saturating Generator Loss — A Practical Fix](#the-non-saturating-generator-loss--a-practical-fix)
10. [Mode Collapse](#mode-collapse)
11. [How to Diagnose and Fix Common GAN Problems — A Step-by-Step Troubleshooting Guide](#how-to-diagnose-and-fix-common-gan-problems--a-step-by-step-troubleshooting-guide)
12. [The Global Optimum — What "Winning" Looks Like](#the-global-optimum--what-winning-looks-like)
13. [DCGAN — Making GANs Work in Practice](#dcgan--making-gans-work-in-practice)
14. [Conditional GANs (cGAN)](#conditional-gans-cgan)
15. [Evaluating a GAN — Inception Score and FID](#evaluating-a-gan--inception-score-and-fid)
16. [GAN vs VAE — The Full Comparison](#gan-vs-vae--the-full-comparison)
17. [Mnemonics](#mnemonics)
18. [Cheatsheet](#cheatsheet)
19. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
20. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine an art forger and an art detective locked in an ongoing duel. The forger's job: paint fake masterpieces convincing enough to fool the detective. The detective's job: examine any painting — real or fake — and correctly say "genuine" or "forgery." At the start, the forger is terrible, and the detective catches every fake instantly. But every time the detective catches a fake, the forger studies exactly WHY it got caught and improves their technique. Every time the forger produces something that fools the detective, the detective sharpens their eye and gets better at spotting the next generation of fakes. Round after round, BOTH get better — and eventually, the forger's paintings become so convincing that even an expert detective can't reliably tell them apart from the real thing. This forger-versus-detective duel, played out as two competing neural networks trained together, IS a **Generative Adversarial Network (GAN)** — introduced by Ian Goodfellow and colleagues in 2014. The forger is called the **Generator**; the detective is called the **Discriminator**.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Recap: Where GANs Fit Among Generative Models

Recall from Lecture 12's density-model split: **explicit density models** (like VAE) directly estimate the data's probability distribution p(x) and can compute actual likelihood values. **Implicit density models** (like GAN) do NOT explicitly define p(x) at all — they only learn to produce realistic SAMPLES from the data distribution, with no explicit likelihood computation anywhere in the picture. A GAN never tells you "this image has an 8% probability of occurring" — it only ever produces new images and gets FEEDBACK on how convincing they are. This is the single biggest philosophical difference between GANs and VAEs, and it shapes everything else about how GANs are trained and evaluated.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## The Two Networks: Generator and Discriminator

**The Generator (G):** takes as input a random noise vector `z`, sampled from a simple, fixed prior distribution (typically `z ~ N(0,1)`, exactly like a VAE's prior from Lecture 12) — and transforms it into a synthetic data sample `G(z)` (e.g., a fake image). The Generator NEVER sees real training data directly during the forward pass that produces its output — its only source of information about what "realistic" means comes indirectly, through the gradient signal it receives from the Discriminator's feedback.

**The Discriminator (D):** a binary classifier. It takes EITHER a real data sample `x` (from the actual training set) OR a fake sample `G(z)` (from the Generator), and outputs a single number between 0 and 1 — its estimate of the probability that the input is REAL. `D(x)` close to 1 means "I'm confident this is real"; `D(G(z))` close to 0 means "I'm confident this is fake."

**The key structural insight:** these two networks have COMPLETELY OPPOSING objectives. The Discriminator wants to correctly separate real from fake (maximize its own classification accuracy). The Generator wants to fool the Discriminator (make `D(G(z))` as close to 1 as possible, i.e., make its fakes indistinguishable from real data). This opposition is precisely why the framework is called "adversarial."

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## The Adversarial Game, Step by Step

```
 1. Sample noise z ~ N(0,1)
 2. Generator produces a fake sample: x_fake = G(z)
 3. Discriminator scores a REAL sample:  D(x_real)  -> wants this near 1
 4. Discriminator scores the FAKE sample: D(x_fake) -> wants this near 0
 5. Update Discriminator to get BETTER at telling real from fake
 6. Update Generator to get BETTER at fooling the (now-updated) Discriminator
 7. Repeat thousands of times
```

Notice this is fundamentally different from every other training setup in this course: there is no single fixed "correct answer" or "target label" for the Generator's output the way there is for, say, an image classifier. The Generator's target keeps MOVING, because it's defined entirely relative to whatever the Discriminator currently believes — a genuinely unusual, co-evolving training dynamic.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## The Minimax Objective Function

The full GAN training objective, from the original Goodfellow et al. (2014) paper, is a single formula both networks are simultaneously working against:

```
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

Reading each piece:
- **E_{x~p_data}[log D(x)]** — the AVERAGE, over real training data x, of `log D(x)`. The Discriminator wants D(x) close to 1 for real data, which makes `log D(x)` close to 0 (its best possible value, since log(1)=0 and log of anything less than 1 is negative).
- **E_{z~p_z}[log(1-D(G(z)))]** — the AVERAGE, over noise samples z, of `log(1-D(G(z)))`. The Discriminator wants D(G(z)) close to 0 for fake data, which makes `1-D(G(z))` close to 1, making `log(1-D(G(z)))` close to 0 (again its best value).
- **max_D** — the Discriminator is trying to MAXIMIZE this whole expression (get both terms as close to 0 — i.e., as large/least-negative — as possible, by correctly classifying both real and fake).
- **min_G** — the Generator is trying to MINIMIZE this same expression — specifically, it can only affect the second term (`D(G(z))`), and it wants to make `D(G(z))` close to 1 (fool the discriminator), which makes `log(1-D(G(z)))` a very large negative number, dragging the whole expression DOWN.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Why "Minimax"? Reading the Formula Like a Story

"Minimax" describes exactly the forger-detective duel: the Discriminator plays MAX (trying to make V as large as possible, i.e., classify perfectly), while the Generator plays MIN (trying to make V as small as possible, i.e., fool the Discriminator completely) — and BOTH players are optimizing the SAME formula, just in opposite directions. This is a classic two-player zero-sum game, borrowed directly from game theory (the same mathematical framework used to analyze competitive games like chess). Unlike every other loss function in this course (which a single network tries to minimize), the GAN objective is fundamentally a TUG-OF-WAR between two networks with opposing goals over the same scoreboard.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Training a GAN — The Alternating Loop

In practice, you can't update both networks by "solving" the minimax game exactly in one step — instead, training ALTERNATES between the two networks, usually within each mini-batch:

```
FOR each training step:
    --- Step A: Update the Discriminator (usually k=1 step, sometimes more) ---
    1. Sample a batch of real data x, and a batch of noise z
    2. Compute D(x) and D(G(z))
    3. Compute the Discriminator's loss (wants D(x)->1, D(G(z))->0)
    4. Backpropagate and update ONLY the Discriminator's weights (Generator frozen)

    --- Step B: Update the Generator (usually 1 step) ---
    5. Sample a NEW batch of noise z
    6. Compute D(G(z)) using the JUST-UPDATED Discriminator
    7. Compute the Generator's loss (wants D(G(z))->1)
    8. Backpropagate and update ONLY the Generator's weights (Discriminator frozen)
```

Notice the careful "freeze one, update the other" discipline — at every single step, exactly ONE network's weights change, while the other's stay fixed, providing a stable target to optimize against for that one step. Get this backward (updating both simultaneously from the same forward pass without careful freezing) and training becomes even more unstable than it already tends to be.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## The Vanishing Gradient Problem in GANs

Here's a genuinely important, practical failure mode: EARLY in training, the Generator is terrible — its fakes are obviously fake, so the Discriminator confidently outputs `D(G(z)) ≈ 0` for nearly all of them. Look back at the Generator's loss term, `log(1-D(G(z)))`: when `D(G(z))` is very close to 0, this term is close to `log(1) = 0`, and — crucially — its GRADIENT with respect to the Generator's parameters is also very close to zero (the function `log(1-x)` is nearly flat near x=0). This means, right when the Generator most desperately needs a strong learning signal (because it's terrible and everything it produces gets rejected), the gradient it receives is almost nonexistent — a frustrating, self-defeating dynamic, similar in spirit to Lecture 4's vanishing gradient problem, but arising from a completely different mechanism (the shape of the loss surface near the Discriminator's confident-rejection region, not repeated multiplication across timesteps).

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## The Non-Saturating Generator Loss — A Practical Fix

Goodfellow's own paper proposed a simple, widely-adopted practical fix: instead of having the Generator MINIMIZE `log(1-D(G(z)))` (which has the vanishing-gradient problem above), have it instead MAXIMIZE `log(D(G(z)))` directly — an equivalent GOAL (both push the Generator toward fooling the Discriminator), but with much STRONGER gradients exactly when the Generator is doing poorly (when `D(G(z))` is near 0, `log(D(G(z)))` has a steep, informative gradient, unlike the nearly-flat `log(1-D(G(z)))`). This reformulated loss is called the **non-saturating loss**, and it's the version almost universally used in practice, rather than the original paper's mathematically "cleaner" minimax formulation.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Mode Collapse

**Mode collapse** is GAN training's most notorious failure mode: instead of learning to produce the full DIVERSITY of the real data distribution, the Generator discovers that producing just ONE (or a small handful of) particularly convincing fake sample(s) reliably fools the current Discriminator — so it collapses to producing nearly IDENTICAL outputs regardless of the input noise z, completely ignoring the diversity that different z values are supposed to encode. Picture the forger discovering ONE forged painting style that always fools the detective, and just churning out endless near-identical copies of that one style, rather than continuing to develop a genuinely diverse portfolio. This is a direct, damaging consequence of the adversarial game's dynamics — the Generator is only ever rewarded for fooling the CURRENT Discriminator, with no explicit pressure to also cover the full diversity of the real data distribution, unlike VAE's reconstruction objective (Lecture 12), which inherently pushes toward reconstructing the SPECIFIC input it was given, encouraging diversity by construction.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## How to Diagnose and Fix Common GAN Problems — A Step-by-Step Troubleshooting Guide

GAN training has a well-earned reputation for being finicky. This section is a practical, symptom-by-symptom troubleshooting guide — the kind of checklist a practitioner actually works through when a GAN isn't training well.

### Step 1 — Identify which failure mode you're seeing

Before fixing anything, correctly diagnose the symptom:

```
SYMPTOM                                    LIKELY CAUSE
------------------------------------------------------------------------
Generator loss stays high, barely moves    Vanishing gradient (Generator
                                            using the original minimax loss)

All generated samples look nearly          Mode collapse (Generator found
identical, regardless of input noise z     one "safe" output and stopped
                                            exploring)

Discriminator loss crashes toward 0        Discriminator has become too
almost immediately, Generator loss         strong, too fast - it's winning
explodes upward                            so completely the Generator gets
                                            no useful gradient signal at all

Losses oscillate wildly, never settle      Learning rates too high, or
                                            imbalance between G and D
                                            training speed

Generated samples are visually poor/       Architecture mismatch, or not
blurry even after long training            enough training, or unsuitable
                                            hyperparameters for the data
```

### Step 2 — Fix vanishing Generator gradients

**Symptom:** the Generator's loss stops decreasing early in training, even though its outputs are clearly still poor quality.
**Fix:** switch from the original minimax loss (`log(1-D(G(z)))`, minimized) to the **non-saturating loss** (`-log(D(G(z)))`, minimized) — covered in detail above. This is almost always the FIRST thing to check, since it's a one-line loss-function change with a large, well-documented effect (recall Worked Example 3 in the numerical file: nearly a 100× gradient-magnitude difference at the point the Generator most needs help).

### Step 3 — Fix a Discriminator that's "winning too easily"

**Symptom:** Discriminator loss drops to near-zero very quickly and stays there; Generator loss stays high or grows.
**Fixes, roughly in order of how commonly they're tried:**
1. **Slow the Discriminator down** — train it less often than the Generator (e.g., 1 Discriminator step per 2–5 Generator steps), giving the Generator more chances to catch up.
2. **Reduce the Discriminator's learning rate** relative to the Generator's — an imbalanced "race" where D learns much faster than G is a common root cause.
3. **Add noise to the Discriminator's inputs** (both real and fake) — this makes the classification task slightly harder, preventing D from becoming overconfident too early, and is a simple, often-effective regularization trick.
4. **Use label smoothing** on the Discriminator's real-data targets — instead of training D to output exactly 1.0 for real data, train it toward a softer target like 0.9, discouraging overconfidence.

### Step 4 — Fix mode collapse

**Symptom:** wildly different noise vectors z produce nearly identical (or a very small handful of distinct) generated outputs.
**Fixes, roughly in order of complexity:**
1. **Mini-batch discrimination** — let the Discriminator look at an entire BATCH of generated samples at once (not just one at a time), specifically so it can notice "these all look suspiciously similar to each other" and penalize that directly — giving the Generator a gradient signal that explicitly rewards diversity.
2. **Unrolled GANs** — let the Generator's update "look ahead" a few steps into how the Discriminator would respond, rather than reacting only to the Discriminator's CURRENT state, reducing the Generator's incentive to exploit a narrow, temporary weakness.
3. **Switch to a more stable objective entirely** — architectures like **Wasserstein GAN (WGAN)**, which replaces the original minimax objective with a distance metric (the Earth Mover's/Wasserstein distance) between real and fake distributions, are specifically designed to provide smoother, more informative gradients throughout training and are widely reported to suffer from mode collapse far less than the original formulation.

### Step 5 — Fix oscillating, unstable losses

**Symptom:** both G and D losses swing up and down repeatedly, never settling into a stable pattern.
**Fixes:**
1. **Lower BOTH learning rates** — oscillation is frequently a sign that updates are simply too large relative to the loss surface's curvature (directly recalling Lecture 7's optimizer material).
2. **Use Adam with tuned β₁** — the original DCGAN paper specifically recommends `β₁=0.5` (lower than Adam's typical default of 0.9) for GAN training, since a high `β₁` (heavy momentum) can amplify the natural oscillation of adversarial training rather than damping it.
3. **Apply gradient penalty / spectral normalization** — modern techniques that directly constrain how much the Discriminator's output can change in response to small input changes, smoothing out the training dynamics substantially.

### Step 6 — General hygiene checklist (apply proactively, not just when something breaks)

- Normalize input images to **[-1, 1]** and use **Tanh** on the Generator's output layer to match (per DCGAN's guidelines above).
- Use **Batch Normalization** in both networks (except the Generator's output layer and the Discriminator's input layer).
- Avoid max-pooling; use **strided convolutions** instead, letting the network learn its own resizing.
- Track BOTH losses over time, and periodically inspect actual generated SAMPLES visually — loss values alone can be misleading in adversarial training (a "good-looking" loss number doesn't always mean good-looking output), so visual inspection remains an essential, non-optional diagnostic step throughout training.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## The Global Optimum — What "Winning" Looks Like

Goodfellow's original paper proves that, under idealized conditions (infinite model capacity, perfect optimization), the minimax game has a unique global optimum: the Generator's distribution `p_g` exactly MATCHES the real data distribution `p_data`, and at that point, the Discriminator can do NO BETTER than random guessing — `D(x) = 0.5` for every input, real or fake, because the fakes have become truly statistically indistinguishable from real data. This is the theoretical "finish line" of GAN training: not a Discriminator that's gotten really good at its job, but a Discriminator that's been rendered COMPLETELY USELESS, because the Generator's output is genuinely as realistic as real data. In practice, this exact equilibrium is rarely reached perfectly (optimization is not perfect, model capacity is finite, and the alternating training dynamic can oscillate rather than converge cleanly) — but it's the target the whole framework is built around.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## DCGAN — Making GANs Work in Practice

**DCGAN** (Deep Convolutional GAN; Radford, Metz & Chintala, 2015) was a landmark paper that turned GANs from a promising-but-finicky idea into something reliably trainable for images, by combining GANs with CNN architecture (Lecture 3) plus a specific set of empirically-validated design guidelines:
- Replace any pooling layers with **strided convolutions** (Discriminator) and **fractional-strided/transposed convolutions** (Generator) — letting the network learn its own spatial up/downsampling instead of using a fixed, non-learned pooling operation.
- Use **Batch Normalization** (Lecture 3) in both Generator and Discriminator — stabilizing training significantly, though notably NOT on the Generator's output layer or the Discriminator's input layer.
- Remove fully-connected hidden layers in favor of a fully convolutional architecture.
- Use **ReLU** activation in the Generator for all layers except the output (which uses **Tanh**, matching pixel values typically normalized to [-1,1]).
- Use **LeakyReLU** activation in the Discriminator for all layers (helps gradient flow, avoiding the "dead ReLU" issue from Lecture 2 in a network that's already prone to unstable gradients).

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Conditional GANs (cGAN)

A **plain GAN** generates samples from the overall data distribution with no control over WHICH kind of sample you get — you can't ask specifically for "a fake cat photo" versus "a fake dog photo," only "a fake photo, whatever the Generator feels like producing." A **Conditional GAN** (Mirza & Osindero, 2014) fixes this by feeding an additional CONDITION/label `y` (e.g., a class label, a text description, or another image) into BOTH the Generator and Discriminator: the Generator now produces `G(z, y)` (a fake sample conditioned on y), and the Discriminator evaluates `D(x, y)` (is x a real sample belonging to condition y?). This simple addition unlocks controllable generation — e.g., "generate a fake image specifically of class y" — and is the direct ancestor of modern text-to-image and image-to-image generation systems (recall Lecture 12's Palette diffusion-model examples for colorization/inpainting/outfilling, which follow a conceptually similar conditioning idea, though via a different underlying generative mechanism).

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Evaluating a GAN — Inception Score and FID

Since GANs are implicit density models (no explicit likelihood to evaluate, per the density-model recap above), evaluating "how good" a GAN is requires different tools than a simple loss-value comparison:

- **Inception Score (IS):** feeds generated images through a pretrained image classifier (originally, the "Inception" network) and measures two things at once: are individual generated images CONFIDENTLY classified as a specific class (sharp, meaningful images, not blurry noise), and is there good DIVERSITY of predicted classes across the whole generated batch (not mode collapse)? Higher IS is better.
- **Fréchet Inception Distance (FID):** compares the STATISTICAL DISTRIBUTION of real images to generated images, in the pretrained classifier's feature space (rather than pixel space) — computing something like a distance between two Gaussian distributions fit to real vs. generated feature activations. Lower FID is better (0 would mean the two distributions are statistically identical). FID is widely considered more reliable than Inception Score, in particular because it directly compares against real data rather than evaluating generated samples in isolation.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## GAN vs VAE — The Full Comparison

Directly extending Lecture 12's closing comparison:

| | GAN | VAE |
|---|---|---|
| Density model type | Implicit (no explicit p(x)) | Explicit (models p(x) via θ) |
| Training objective | Adversarial minimax game (2 networks) | Single reconstruction + KL-regularization objective (ELBO) |
| Training stability | Often UNSTABLE (mode collapse, vanishing gradients, oscillation) | Generally SMOOTHER and more stable |
| Output sharpness | Typically SHARPER, more detailed | Typically SMOOTHER, sometimes blurrier |
| Latent space structure | No explicit regularization toward a nice prior | Explicitly regularized via KL divergence toward a known prior |
| Likelihood computation | Not possible (implicit) | Possible, at least approximately (via the ELBO) |
| Can directly encode a specific input to latent space? | No native encoder (though variants like BiGAN exist) | Yes — that's literally the encoder's job |

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Mnemonics

- **"Forger vs detective, both get better"** → the core GAN adversarial dynamic.
- **"Discriminator plays MAX, Generator plays MIN, same scoreboard"** → the minimax objective.
- **"D(x)→1 for real, D(G(z))→0 for fake — Generator wants to flip that"** → what each network wants.
- **"Freeze one, train the other, every single step"** → the alternating training discipline.
- **"Non-saturating: maximize log(D(G(z))), not minimize log(1-D(G(z)))"** → the practical vanishing-gradient fix.
- **"One convincing fake, forever"** → mode collapse in one image.
- **"D(x)=0.5 everywhere = Generator has truly won"** → the theoretical global optimum.
- **"GAN: sharp but unstable. VAE: smooth but blurrier."** → the one-line GAN vs VAE trade-off.
- **"D winning too fast? Slow it down. G stuck? Switch to non-saturating. Outputs identical? Add mini-batch diversity."** → the three most common fixes, in one line each.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Cheatsheet

| Concept | One-liner | Formula/Fact |
|---|---|---|
| Generator | Produces fake samples from noise | `G(z)`, `z~N(0,1)` |
| Discriminator | Classifies real vs fake | `D(x) ∈ [0,1]`, probability "real" |
| Minimax objective | The full adversarial game | `min_G max_D E[log D(x)] + E[log(1-D(G(z)))]` |
| Non-saturating G loss | Practical vanishing-gradient fix | `max log(D(G(z)))` instead of `min log(1-D(G(z)))` |
| Mode collapse | G produces low-diversity output | Ignores z diversity, repeats one convincing fake |
| Global optimum | Theoretical "finish line" | `p_g = p_data`, `D(x)=0.5` everywhere |
| DCGAN | CNN + GAN design guidelines | Strided conv, BatchNorm, ReLU/Tanh (G), LeakyReLU (D) |
| Conditional GAN | Controllable generation | `G(z,y)`, `D(x,y)` |
| Inception Score | Quality + diversity metric | Higher = better |
| FID | Distributional distance metric | Lower = better |
| D too strong fix | Slow D down, add noise, label smoothing | rebalances the adversarial race |
| Mode collapse fix | Mini-batch discrimination, unrolled GANs, WGAN | rewards/enforces diversity |
| Oscillation fix | Lower LR, Adam β₁=0.5, gradient penalty | smooths the training dynamics |

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** describing the Generator as trying to "minimize" the WHOLE minimax objective — it can only actually influence the SECOND term (`log(1-D(G(z)))`), since the first term doesn't depend on G at all.
- **Trap:** forgetting the alternating "freeze one network" training discipline — updating both networks' weights from the same shared forward pass without careful freezing is a common implementation bug, not a valid alternative training strategy.
- **Trap:** confusing vanishing gradients here with Lecture 4's RNN vanishing gradients — same NAME, completely different MECHANISM (loss-surface flatness near confident-rejection, vs. repeated multiplication across timesteps).
- **Trap:** claiming GANs have NO way to be conditioned/controlled — plain GANs don't, but Conditional GANs explicitly add this capability via an extra input y to both networks.
- **Exam hack:** always state BOTH halves of the GAN vs VAE trade-off (GAN: sharper but less stable; VAE: smoother/blurrier but more stable) — a one-sided answer misses half the intended comparison.
- **Exam hack:** the global optimum condition `D(x)=0.5` for all x is a favourite "what does perfect GAN convergence look like" exam question — always explain WHY (fakes are statistically indistinguishable from real data, so the best any classifier can do is a coin flip), not just state the number.
- **Exam hack:** if asked "your GAN isn't training well, what do you check first," always follow the diagnostic ORDER from the troubleshooting guide — first correctly IDENTIFY the symptom (which loss is doing what), THEN apply the matching fix — jumping straight to a fix without diagnosing the actual symptom is a common, marks-losing shortcut.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory)`

---

## Summary

This bonus lecture introduces Generative Adversarial Networks, the implicit-density-model counterpart to Lecture 12's VAE, via the forger-versus-detective analogy: a Generator (G) transforms random noise z into fake samples G(z), while a Discriminator (D) tries to correctly separate real data from G's fakes, with both networks trained simultaneously in direct opposition via the minimax objective `min_G max_D E[log D(x)] + E[log(1-D(G(z)))]` — a genuine two-player zero-sum game, unlike every other single-network loss function in this course. Training alternates between updating D (with G frozen) and updating G (with D frozen), and a critical practical problem — vanishing gradients for G early in training, when D confidently rejects G's obviously-fake outputs — is fixed via the widely-used non-saturating loss, having G maximize `log(D(G(z)))` instead of minimizing `log(1-D(G(z)))`. GAN training is notoriously prone to mode collapse (G discovering one convincing fake and repeating it, ignoring the diversity noise z should encode), in contrast to the theoretical global optimum where G's distribution exactly matches the real data distribution and D can do no better than random guessing (D(x)=0.5 everywhere). A dedicated step-by-step troubleshooting guide walks through diagnosing and fixing the most common real-world GAN training failures: vanishing Generator gradients (fixed by the non-saturating loss), a Discriminator that wins too easily and too fast (fixed by slowing D's training pace, adding input noise, or label smoothing), mode collapse (fixed by mini-batch discrimination, unrolled GANs, or switching to more stable objectives like Wasserstein GAN), and oscillating/unstable losses (fixed by lowering learning rates, tuning Adam's β₁ down to 0.5 as DCGAN recommends, or applying gradient penalty/spectral normalization) — plus a proactive hygiene checklist (input normalization, Tanh output, Batch Normalization placement, strided convolutions instead of pooling, and always inspecting actual generated samples visually rather than trusting loss values alone). DCGAN established the practical CNN-based design recipe (strided convolutions, batch normalization, ReLU/Tanh in the Generator, LeakyReLU in the Discriminator) that made GANs reliably trainable for images, while Conditional GANs add an extra label/condition input to both networks for controllable generation. Because GANs have no explicit likelihood to evaluate, quality is assessed instead via Inception Score (image quality + class diversity) and Fréchet Inception Distance (distributional distance from real data in a pretrained classifier's feature space, generally the more trusted metric). The lecture closes with a full GAN-vs-VAE comparison extending Lecture 12's conclusion: GANs typically produce sharper outputs but train less stably (mode collapse, vanishing gradients, oscillation), while VAEs train more smoothly via a single well-behaved reconstruction-plus-KL objective but tend to produce smoother, sometimes blurrier outputs — two genuinely complementary approaches to the same underlying generative modeling problem.

`[← Lecture 12](../../Lecture-12-Encoder-Decoder-and-VAE/README.md) · [🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-theory) · [Next: Numerical →](../numerical/dl_lecture13_gan_numerical.md)`
