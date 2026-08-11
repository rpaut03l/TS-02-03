# DL Lecture 14 (Bonus) — Restricted Boltzmann Machines (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

> Folder: `Deep-Learning/Lecture-14-Restricted-Boltzmann-Machines/theory/`
> Pairs with: [`numerical/dl_lecture14_rbm_numerical.md`](../numerical/dl_lecture14_rbm_numerical.md) · [`practice/dl_lecture14_rbm_practice.md`](../practice/dl_lecture14_rbm_practice.md) · [`exercises/dl_lecture14_exercises.md`](../exercises/dl_lecture14_exercises.md)
> **⚠️ Bonus lecture — not part of the original 541-page slide deck.** RBMs and Deep Belief Networks aren't mentioned anywhere in the source material. This lecture was built from scratch, in the same house style as the rest of the course, because RBMs are the historical bridge between Hinton's 2006 breakthrough (cited in Lecture 12's autoencoder history) and the modern deep learning era — genuinely useful context, even though your instructor's slides moved straight to autoencoders instead. Treat it as supplementary enrichment.

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Energy-Based Models — A New Way of Thinking About Probability](#energy-based-models--a-new-way-of-thinking-about-probability)
3. [Boltzmann Machines — The Full (Unrestricted) Idea](#boltzmann-machines--the-full-unrestricted-idea)
4. [Why "Restricted"? The Bipartite Structure](#why-restricted-the-bipartite-structure)
5. [The RBM Energy Function](#the-rbm-energy-function)
6. [From Energy to Probability](#from-energy-to-probability)
7. [The Conditional Independence Trick](#the-conditional-independence-trick)
8. [Computing Hidden and Visible Activations](#computing-hidden-and-visible-activations)
9. [Gibbs Sampling — Letting the Network Dream](#gibbs-sampling--letting-the-network-dream)
10. [Training an RBM — Contrastive Divergence](#training-an-rbm--contrastive-divergence)
11. [The CD-k Update Rule](#the-cd-k-update-rule)
12. [Deep Belief Networks — Stacking RBMs](#deep-belief-networks--stacking-rbms)
13. [RBMs in the History of Deep Learning](#rbms-in-the-history-of-deep-learning)
14. [Why RBMs Faded — And Why They Still Matter](#why-rbms-faded--and-why-they-still-matter)
15. [RBM vs Autoencoder vs VAE](#rbm-vs-autoencoder-vs-vae)
16. [Mnemonics](#mnemonics)
17. [Cheatsheet](#cheatsheet)
18. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
19. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine a room full of people at a party, where each person's mood (happy or grumpy) is influenced by the moods of the people standing near them — happy people tend to cluster near other happy people, grumpy people near other grumpy people, and certain pairs of people just naturally clash or complement each other regardless of the general mood in the room. If you wanted to describe the "overall harmony" of this party mathematically, you might assign a single number — call it **energy** — that's LOW when everyone's moods fit together nicely, and HIGH when there's a lot of clashing and tension. A **Restricted Boltzmann Machine (RBM)** does exactly this, but for a neural network: it has two layers of simple on/off units — a **visible layer** (the data you observe, like party-goers' visible moods) and a **hidden layer** (unobserved, latent "factors" that help explain the pattern, like an invisible layer of "party themes" pulling people's moods together) — and it defines a single ENERGY number for every possible visible+hidden configuration, then converts that energy into a probability: **low energy = high probability, high energy = low probability.** Training an RBM means adjusting the connections between visible and hidden units until LOW ENERGY configurations line up with what real data actually looks like.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Energy-Based Models — A New Way of Thinking About Probability

Every model so far in this course (Lectures 1–13) has directly computed an output — a class score, a hidden state, a reconstruction, a fake image. An **energy-based model** works differently: instead of directly computing "the answer," it defines an **energy function** `E(x)` over configurations x, where LOWER energy means "more plausible/likely," and HIGHER energy means "less plausible/unlikely." Probability is then derived FROM energy via the **Boltzmann distribution** (borrowed directly from statistical physics, hence the name):
```
P(x) = exp(-E(x)) / Z
```
where **Z** (the "partition function") is a normalizing constant — the sum of `exp(-E(x))` over EVERY possible configuration x, ensuring all probabilities sum to 1. This is a fundamentally different philosophy from every feedforward network in this course: rather than learning a direct input→output FUNCTION, an energy-based model learns a SCORING LANDSCAPE, where training pulls the energy of real data down and pushes the energy of everything else up.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Boltzmann Machines — The Full (Unrestricted) Idea

A **Boltzmann Machine** (Hinton & Sejnowski, 1985) is a network of binary (0/1) units, ALL of which can be connected to ALL others — visible units connected to visible units, hidden units connected to hidden units, and visible units connected to hidden units, with a learned weight on every single connection. This full generality is exactly what makes plain Boltzmann Machines painfully SLOW to train: computing anything useful requires accounting for every unit's influence on every other unit simultaneously, and the training procedure (which requires estimating expectations over the full joint distribution) becomes computationally intractable for anything beyond tiny toy networks.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Why "Restricted"? The Bipartite Structure

The **Restricted** Boltzmann Machine (Smolensky, 1986; popularized and made practically trainable by Hinton in the 2000s) fixes this intractability with one simple structural rule: **NO connections are allowed WITHIN a layer** — visible units cannot connect to other visible units, and hidden units cannot connect to other hidden units. Only visible-to-hidden connections are allowed. This produces a **bipartite graph** structure (recall the graph terminology from Lecture 9 — a bipartite graph is one where nodes split into two groups, with edges only ever crossing BETWEEN groups, never within one):
```
 VISIBLE LAYER          HIDDEN LAYER
   v1 --*--*--*------->  h1
   v2 --*--*--*------->  h2      *  = every visible unit connects
   v3 --*--*--*------->  h3         to every hidden unit (fully
                                     bipartite-connected)

  NO v-to-v connections.  NO h-to-h connections.
```
This restriction is what makes RBMs practically trainable: because there are no within-layer connections, all the HIDDEN units become **conditionally independent of each other, GIVEN the visible units** (and vice versa) — a property exploited heavily in the sections below.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## The RBM Energy Function

For a visible vector `v` (length n_v) and hidden vector `h` (length n_h), both binary (each entry is 0 or 1), the RBM's energy function is:
```
E(v, h) = - a^T v - b^T h - v^T W h
```
Breaking down each term:
- **a** = a bias vector for the visible units (length n_v) — a's own baseline preference for each visible unit being "on."
- **b** = a bias vector for the hidden units (length n_h) — b's own baseline preference for each hidden unit being "on."
- **W** = the weight matrix connecting visible and hidden units (shape n_v × n_h) — `W_ij` captures how strongly visible unit i and hidden unit j "agree" or "disagree."
- **v^T W h** = the interaction term — literally `Σ_i Σ_j v_i · W_ij · h_j`, summing up every active visible-hidden pair's connection strength.

**Why the minus signs?** Because LOWER energy should mean HIGHER probability (recall `P(x)=exp(-E(x))/Z`), and we WANT the model to prefer configurations where biases and weighted interactions are large and POSITIVE (i.e., visible/hidden units that are "on" together, in a way the connection weight rewards) — the minus signs flip large positive bias/interaction sums into large NEGATIVE energy, which then translates into large POSITIVE probability once exponentiated.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## From Energy to Probability

Plugging the RBM's energy function into the general Boltzmann distribution formula:
```
P(v, h) = exp(-E(v,h)) / Z
```
where Z sums `exp(-E(v,h))` over EVERY possible combination of v and h — for even modest-sized layers, this is an astronomically large sum (recall Lecture 8's `2ⁿ` dropout-configuration count — a similar combinatorial explosion appears here), which is exactly why Z is never computed directly in practice; instead, RBMs are trained using a clever workaround (Contrastive Divergence, covered below) that never requires computing Z at all.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## The Conditional Independence Trick

Because of the bipartite (no within-layer connections) structure, a beautiful simplification falls out: **given the visible units, all hidden units are conditionally independent of each other** (and vice versa). This means you can compute each hidden unit's activation probability SEPARATELY and INDEPENDENTLY, rather than needing to jointly consider the whole hidden layer at once — turning an otherwise intractable joint computation into n_h simple, independent, one-at-a-time calculations. This single structural property (a direct consequence of the "restricted"/no-intra-layer-connections rule) is THE reason RBMs are tractable to train at all, where full Boltzmann Machines are not.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Computing Hidden and Visible Activations

Thanks to conditional independence, each unit's probability of being "on" (=1) has a clean, closed-form **sigmoid** formula (recall the sigmoid activation from Lecture 2):
```
P(h_j = 1 | v) = sigmoid( b_j + Σ_i v_i * W_ij )
P(v_i = 1 | h) = sigmoid( a_i + Σ_j W_ij * h_j )
```
Reading the first formula: hidden unit j's probability of activating depends on its own bias `b_j`, PLUS a weighted sum of every visible unit's value, weighted by that visible unit's connection strength to hidden unit j — exactly the same "weighted sum, then squash through an activation" pattern from a single neuron in Lecture 2, just now producing a PROBABILITY (via sigmoid) rather than a plain activation value. The second formula runs the exact same computation in REVERSE — reconstructing visible units from hidden units — which is why RBMs are naturally suited to reconstruction-style tasks, echoing Lecture 12's autoencoder story.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Gibbs Sampling — Letting the Network Dream

To actually DRAW samples from an RBM (or estimate the statistics needed for training), we use **Gibbs sampling** — an iterative back-and-forth sampling procedure exploiting the conditional independence property:
```
1. Start with some visible vector v (e.g., a real training example)
2. Sample h ~ P(h|v)   (sample every hidden unit, using the sigmoid formula above)
3. Sample v ~ P(v|h)   (sample every visible unit, using the reverse sigmoid formula)
4. Repeat steps 2-3 many times
```
Each full round (v→h→v) is one **Gibbs step**. After enough Gibbs steps, the chain of samples converges toward the model's true underlying distribution — informally, the network is "dreaming up" what it believes plausible data looks like, purely from its learned weights, with no real data involved after the very first step.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Training an RBM — Contrastive Divergence

The theoretically "correct" way to train an RBM (maximizing the likelihood of the training data) requires computing the intractable partition function Z — Hinton's key practical innovation, **Contrastive Divergence (CD)**, sidesteps this entirely with a clever approximation: instead of running Gibbs sampling to FULL convergence (which would take a very long chain), run it for just **k steps** (commonly k=1, called **CD-1**), starting from a REAL training example, and compare:
- **The "positive phase"**: statistics computed from the REAL data and its hidden activations (v→h, using the actual training example).
- **The "negative phase"**: statistics computed after k Gibbs steps of "dreaming" starting from that same real example (v→h→v→h→..., k times).

The core training INTUITION: push the energy of the real data DOWN (make it more probable), and push the energy of the model's own "dreamed" reconstruction UP (make it less probable) — training the model to prefer real data patterns over whatever it currently tends to hallucinate.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## The CD-k Update Rule

The actual weight update formula, using CD-k:
```
delta_W_ij = eta * ( <v_i h_j>_data - <v_i h_j>_reconstruction )
```
Where `<v_i h_j>_data` is computed using the REAL visible data and its sampled hidden activations (the positive phase), and `<v_i h_j>_reconstruction` is computed after k Gibbs steps (the negative phase, the model's own "dream"). **η (eta)** is the learning rate — the exact same concept from Lecture 7's optimizers, just applied to this very different training procedure. Bias updates follow the same positive-minus-negative pattern:
```
delta_a_i = eta * ( v_i_data - v_i_reconstruction )
delta_b_j = eta * ( h_j_data - h_j_reconstruction )
```
Notice the elegant symmetry: if the model's "dream" (reconstruction) already looks exactly like the real data, the positive and negative terms are equal, the update becomes zero, and training has converged — the model's internal beliefs now match reality.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Deep Belief Networks — Stacking RBMs

A **Deep Belief Network (DBN)** stacks multiple RBMs on top of each other, trained **one layer at a time**, greedily:
```
1. Train RBM #1 on the raw input data (visible = raw data, hidden = layer-1 features)
2. FREEZE RBM #1. Use its hidden activations as the "visible" data for RBM #2
3. Train RBM #2 on THOSE activations (visible = layer-1 features, hidden = layer-2 features)
4. Repeat: stack as many RBMs as desired, each one trained on the previous one's output
5. (Optional) Fine-tune the WHOLE stack together using backpropagation, treating the
   stacked RBMs as an initialization for a normal deep feedforward network
```
This **greedy layer-wise pretraining** procedure was historically significant: in the mid-2000s, training deep networks directly with backpropagation from random initialization was notoriously unstable (severe vanishing-gradient problems, similar in spirit to Lecture 4's RNN vanishing gradients, but for depth rather than sequence length). DBN pretraining gave each layer a good, unsupervised starting point BEFORE any supervised fine-tuning even began, which was often the difference between a deep network training successfully at all versus failing completely.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## RBMs in the History of Deep Learning

Recall from Lecture 12: Hinton & Salakhutdinov's landmark 2006 paper, "Reducing the Dimensionality of Data with Neural Networks," is usually credited (alongside a companion 2006 DBN paper by Hinton, Osindero, and Teh) as one of the key sparks that reignited serious interest in deep learning after the field's long "AI winter." That companion paper used exactly the RBM/DBN greedy pretraining procedure described above to successfully train networks far deeper than was previously considered practical — directly enabling the autoencoder-based dimensionality reduction results covered in Lecture 12, and setting the stage for the entire modern deep learning era that followed (including everything covered in Lectures 1–13 of this course).

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Why RBMs Faded — And Why They Still Matter

By the early 2010s, RBM/DBN-style greedy pretraining was largely REPLACED by simpler, more direct methods: better weight initialization schemes (e.g., Xavier/He initialization), ReLU activations (Lecture 2) which don't saturate the way sigmoid/tanh do, Batch Normalization (Lecture 3), and residual connections (Lecture 10) — all of which solved the SAME underlying "how do we train very deep networks" problem, but far more simply and effectively, without needing RBM pretraining as a crutch. Modern networks are trained end-to-end with plain backpropagation and gradient descent from the very start.

**Why they still matter:** RBMs remain the clearest, most historically important example of an **energy-based generative model** — a family of ideas that has genuinely resurged in modern research (score-based generative models and diffusion models, referenced in Lecture 12's Palette application, share deep conceptual roots with energy-based thinking), and understanding RBMs gives real intuition for how probability, energy, and sampling connect in generative modeling more broadly — useful background for reading modern generative AI research, even though RBMs themselves are rarely used directly in production systems today.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## RBM vs Autoencoder vs VAE

Extending the generative-model comparisons from Lectures 12 and 13:

| | RBM | Autoencoder | VAE |
|---|---|---|---|
| Core idea | Energy function over binary units | Deterministic compress-then-reconstruct | Probabilistic compress-then-reconstruct |
| Probabilistic? | Yes (Boltzmann distribution) | No (a single fixed latent point) | Yes (learns a latent distribution) |
| Training | Contrastive Divergence (approximate) | Direct backpropagation (reconstruction loss) | Backpropagation on the ELBO (reconstruction + KL) |
| Sampling new data | Gibbs sampling (iterative) | No native generation mechanism | Sample from prior, then decode (one pass) |
| Historical role | 2006-era breakthrough enabling deep training | Modern standard for compression/representation learning | Modern standard for probabilistic generation |

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Mnemonics

- **"Low energy = high probability, like a ball settling into a valley"** → the core energy-based model intuition.
- **"Restricted = no gossip within your own layer"** → the bipartite, no-intra-layer-connection rule.
- **"v talks to h, h talks to v, but v never talks to v"** → the bipartite structure in one line.
- **"Positive phase = reality, negative phase = the model's dream"** → Contrastive Divergence's two phases.
- **"Stack RBMs like Lego, one layer at a time, greedily"** → Deep Belief Network pretraining.
- **"RBM pretraining did ReLU/BatchNorm/ResNet's job, the hard way, first"** → why RBMs faded but still matter historically.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Cheatsheet

| Concept | One-liner | Formula |
|---|---|---|
| Energy-based model | Lower energy = higher probability | `P(x)=exp(-E(x))/Z` |
| RBM energy function | Bias + bias + interaction, all negated | `E(v,h)=-a^Tv-b^Th-v^TWh` |
| Bipartite structure | No within-layer connections | v-v: none, h-h: none, v-h: full |
| Hidden activation | Sigmoid of weighted visible sum | `P(h_j=1\|v)=σ(b_j+Σv_iW_ij)` |
| Visible reconstruction | Sigmoid of weighted hidden sum | `P(v_i=1\|h)=σ(a_i+ΣW_ijh_j)` |
| Gibbs sampling | Iterative v→h→v→h... | converges to model distribution |
| Contrastive Divergence | Approximate training via short Gibbs chains | CD-k, commonly k=1 |
| CD weight update | Positive minus negative phase | `ΔW_ij=η(⟨v_ih_j⟩_data-⟨v_ih_j⟩_recon)` |
| DBN | Stack of RBMs, trained greedily layer-by-layer | pretraining before fine-tuning |

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting the minus signs in the energy function, or forgetting WHY they're there — always connect back to `P(x)=exp(-E(x))/Z`: lower energy must mean higher probability.
- **Trap:** claiming RBM units can connect within their own layer — the entire point of "restricted" is that they CANNOT; only cross-layer (visible-hidden) connections exist.
- **Trap:** trying to compute Z directly — it's intentionally intractable for any real-sized RBM; that's exactly why Contrastive Divergence exists, as a workaround that never needs Z at all.
- **Trap:** confusing the positive and negative phases in CD — positive phase = statistics FROM real data; negative phase = statistics from the model's OWN reconstruction/dream after k Gibbs steps.
- **Exam hack:** if asked why RBMs are tractable but full Boltzmann Machines aren't, always cite the SPECIFIC mechanism — conditional independence of all hidden units given the visible units (and vice versa), a direct consequence of the bipartite/no-intra-layer-connection structure.
- **Exam hack:** DBN's greedy layer-wise pretraining is a favourite "describe the historical significance" essay question — always mention it was a practical workaround for training deep networks BEFORE better initialization/ReLU/BatchNorm/ResNet existed, not a technique still commonly used today.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory)`

---

## Summary

This bonus lecture introduces Restricted Boltzmann Machines as historically foundational energy-based generative models, connecting directly to Lecture 12's mention of Hinton's 2006 breakthrough. An energy-based model defines a scalar energy `E(x)` over configurations, converted to probability via the Boltzmann distribution `P(x)=exp(-E(x))/Z`, where lower energy means higher probability. A full Boltzmann Machine allows connections between every pair of units, making it computationally intractable; the RESTRICTED version fixes this by forbidding within-layer connections entirely (visible-visible and hidden-hidden connections are banned, leaving only a bipartite visible-hidden structure), with energy function `E(v,h)=-a^Tv-b^Th-v^TWh`. This bipartite restriction produces a crucial simplification — all hidden units become conditionally independent given the visible units (and vice versa) — enabling clean, closed-form sigmoid activation formulas (`P(h_j=1|v)=σ(b_j+Σv_iW_ij)` and its reverse), and enabling Gibbs sampling (an iterative v→h→v→h chain) to draw samples or "dream" from the model. Since computing the true training objective requires the intractable partition function Z, Hinton's Contrastive Divergence (CD-k) approximates training instead, comparing a "positive phase" (statistics from real data) against a "negative phase" (statistics from a short k-step Gibbs chain, the model's own reconstruction), pushing real data's energy down and the model's own hallucinated reconstructions' energy up. Stacking multiple RBMs into a Deep Belief Network, trained greedily one layer at a time before optional end-to-end fine-tuning, was historically pivotal — it was one of the key techniques that made training genuinely deep networks practical in the mid-2000s, directly enabling the autoencoder-based results in Lecture 12 and helping spark the modern deep learning era. RBM/DBN pretraining has since been largely superseded by simpler, more direct solutions to the same underlying problem (better initialization, ReLU, Batch Normalization, residual connections), but understanding RBMs provides genuine, transferable intuition for energy-based and probabilistic generative modeling more broadly, connecting conceptually to the modern resurgence of energy-based and score-based generative models.

`[← Lecture 13](../../Lecture-13-Generative-Adversarial-Networks/README.md) · [🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-theory) · [Next: Numerical →](../numerical/dl_lecture14_rbm_numerical.md)`
