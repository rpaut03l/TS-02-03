# DL Lecture 14 (Bonus) — Restricted Boltzmann Machines (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice)`

> Folder: `Deep-Learning/Lecture-14-Restricted-Boltzmann-Machines/practice/`
> Pairs with: [`theory/dl_lecture14_rbm_theory.md`](../theory/dl_lecture14_rbm_theory.md) · [`numerical/dl_lecture14_rbm_numerical.md`](../numerical/dl_lecture14_rbm_numerical.md) · [`exercises/dl_lecture14_exercises.md`](../exercises/dl_lecture14_exercises.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## Table of Contents
1. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
2. [Explain-It-Back Prompts](#explain-it-back-prompts)
3. [Quick-Fire True / False](#quick-fire-true--false)
4. [RBM vs Autoencoder vs VAE Matching Drill](#rbm-vs-autoencoder-vs-vae-matching-drill)
5. [Mini Interview-Style Round](#mini-interview-style-round)
6. [Summary](#summary)

---

## Concept Check — Fill in the Blank

1. In an energy-based model, ______ energy corresponds to ______ probability.
2. "Restricted" in RBM refers to forbidding connections ______ a layer.
3. The two hidden/visible activation formulas both use the ______ activation function.
4. RBM training uses an approximation called ______, since computing the true partition function Z is intractable.
5. A Deep Belief Network stacks multiple RBMs, trained ______, one layer at a time.

<details>
<summary>Show answers</summary>

1. lower; higher
2. within (intra-layer)
3. sigmoid
4. Contrastive Divergence
5. greedily
</details>

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice)`

---

## Explain-It-Back Prompts

1. Explain the party-mood analogy for energy-based models in your own words.
2. Explain why the "restricted" bipartite structure makes RBMs tractable when full Boltzmann Machines are not.
3. Walk through the positive-phase/negative-phase distinction in Contrastive Divergence from memory.
4. Explain why DBN pretraining mattered historically, and why it's less commonly used today.
5. Explain the difference between RBM, Autoencoder, and VAE in one sentence each.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice)`

---

## Quick-Fire True / False

1. In a Restricted Boltzmann Machine, hidden units can be directly connected to other hidden units. — **False**.
2. Lower energy corresponds to higher probability under the Boltzmann distribution. — **True**.
3. Gibbs sampling in an RBM alternates between sampling h given v, and v given h. — **True**.
4. Contrastive Divergence computes the exact partition function Z. — **False** (it's specifically designed to avoid ever computing Z).
5. Modern deep networks typically require RBM pretraining before training with backpropagation. — **False** (better initialization/ReLU/BatchNorm/ResNet made this largely unnecessary).

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice)`

---

## RBM vs Autoencoder vs VAE Matching Drill

| Property | RBM | Autoencoder | VAE |
|---|---|---|---|
| Probabilistic? | ? | ? | ? |
| Training method | ? | ? | ? |

<details>
<summary>Show answers</summary>

Probabilistic: RBM=Yes (Boltzmann distribution), Autoencoder=No (fixed point), VAE=Yes (learned distribution). Training method: RBM=Contrastive Divergence, Autoencoder=direct backpropagation on reconstruction loss, VAE=backpropagation on the ELBO.
</details>

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice)`

---

## Mini Interview-Style Round

**Q1.** "A teammate suggests using RBM pretraining for a new deep classifier project in 2026. What would you tell them?"

<details>
<summary>Show answer</summary>

You'd point out that RBM/DBN pretraining was a historically important workaround for training deep networks in the mid-2000s, before better solutions existed — modern techniques (proper weight initialization, ReLU activations, Batch Normalization, residual connections) solve the same underlying "how do we train deep networks" problem far more simply and effectively, without needing an unsupervised pretraining phase at all. For a new project today, you'd recommend a standard modern architecture trained end-to-end with backpropagation, reserving RBM-style approaches only for cases specifically requiring energy-based modeling.
</details>

**Q2.** "Explain why Contrastive Divergence is called an 'approximation' rather than an exact training method."

<details>
<summary>Show answer</summary>

The mathematically correct training objective requires computing (or at least accurately estimating) statistics from the model's TRUE equilibrium distribution, which technically requires running Gibbs sampling to full convergence — a very long chain. Contrastive Divergence instead runs just k steps (often only k=1), which is a much cruder, biased estimate of the true equilibrium statistics — hence "approximation." In practice, this crude approximation still works well enough to train useful RBMs, which was itself a significant practical discovery.
</details>

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice)`

---

## Summary

This practice file drills the bonus RBM lecture's energy-based modeling concepts through active recall. A fill-in-the-blank check reinforces the energy-probability relationship, the bipartite restriction, the shared sigmoid activation formula, Contrastive Divergence terminology, and DBN's greedy layer-wise training. Five explain-it-back prompts push you to reproduce the party-mood analogy, the tractability argument for the bipartite structure, the CD positive/negative phase distinction, DBN's historical significance, and the RBM/Autoencoder/VAE comparison. A quick-fire true/false round and a three-way matching drill test conceptual accuracy across all three generative model families covered in Lectures 12–14. A two-question interview-style round rehearses realistic judgment: advising against RBM pretraining for a modern project, and explaining precisely why Contrastive Divergence is an approximation rather than exact training. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture14_rbm_numerical.md) · [🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-practice) · [Next: Exercises →](../exercises/dl_lecture14_exercises.md)`
