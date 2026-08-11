# DL Lecture 14 (Bonus) — Exercise Bank (Restricted Boltzmann Machines)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-14-bonus--exercise-bank-restricted-boltzmann-machines)`

> Folder: `Deep-Learning/Lecture-14-Restricted-Boltzmann-Machines/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture14_rbm_theory.md) · [numerical](../numerical/dl_lecture14_rbm_numerical.md) · [practice](../practice/dl_lecture14_rbm_practice.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## 🟢 Easy — Definitions & Recall

**Q14.1.** Write the RBM energy function.

**Q14.2.** What does "restricted" mean structurally, in an RBM?

**Q14.3.** Write the hidden unit activation probability formula.

**Q14.4.** What training algorithm is used to train RBMs, and why is it needed?

**Q14.5.** What does a Deep Belief Network do with multiple RBMs?

---

## 🟡 Medium — Applied Reasoning

**Q14.6.** For v=[0,1,1], h=[1,0], a=[0.1,0.2,-0.1], b=[0.3,-0.2], W=[[0.2,0.1],[-0.3,0.4],[0.5,-0.2]], compute the energy E(v,h).

**Q14.7.** Using the same v, b, W from Q14.6, compute P(h1=1|v) and P(h2=1|v).

**Q14.8.** Explain why computing the RBM's partition function Z directly is intractable for any real-sized network.

**Q14.9.** Explain the conditional independence property in an RBM, and why it doesn't hold in a full (unrestricted) Boltzmann Machine.

**Q14.10.** Explain why RBM/DBN pretraining became less necessary after the introduction of ReLU, Batch Normalization, and residual connections.

---

## 🔴 Hard — Derivation & Multi-Step

**Q14.11.** For v=[1,1,0], a=[0.0,0.3,-0.2], b=[0.1,0.1], W=[[0.4,-0.1],[0.2,0.3],[-0.5,0.2]]: compute the full energy E(v,h) for h=[0,1].

**Q14.12.** Using the same setup as Q14.11, compute P(h1=1|v) and P(h2=1|v), and state which hidden unit is more likely to activate given this visible vector.

**Q14.13.** A DBN is built by stacking 3 RBMs: RBM1 maps 100 visible units to 50 hidden units, RBM2 maps 50 to 20, RBM3 maps 20 to 10. If each RBM's weight matrix (ignoring biases) uses one parameter per visible-hidden connection, compute the total number of parameters across all three RBMs.

**Q14.14.** Explain, step by step, what would happen numerically to a hidden unit's activation probability if its bias `b_j` were an extremely large negative number (e.g., -100), even if every incoming weighted visible contribution were strongly positive. Connect this to sigmoid's saturation behavior from Lecture 2.

`[🔝 Top](#dl-lecture-14-bonus--exercise-bank-restricted-boltzmann-machines)`

---

## Answer Key

<details>
<summary>Q14.1 – Q14.5 (Easy)</summary>

- **Q14.1:** `E(v,h) = -a^Tv - b^Th - v^TWh`.
- **Q14.2:** No connections are allowed within a layer — no visible-visible connections, no hidden-hidden connections; only visible-hidden connections exist (a bipartite structure).
- **Q14.3:** `P(h_j=1|v) = σ(b_j + Σ_i v_i·W_ij)`.
- **Q14.4:** Contrastive Divergence (CD-k). It's needed because the mathematically exact training objective requires computing the intractable partition function Z, which CD sidesteps entirely using short Gibbs sampling chains instead.
- **Q14.5:** It stacks multiple RBMs on top of each other, training them greedily one layer at a time (each RBM's hidden output becomes the next RBM's visible input), historically used to pretrain deep networks before backpropagation-based fine-tuning.
</details>

<details>
<summary>Q14.6 – Q14.10 (Medium)</summary>

- **Q14.6:** a.v = 0.1×0+0.2×1+(-0.1)×1 = 0+0.2-0.1 = 0.1. b.h = 0.3×1+(-0.2)×0 = 0.3. v^TWh = v2×(W21h1+W22h2)+v3×(W31h1+W32h2) = 1×(-0.3×1+0.4×0)+1×(0.5×1+(-0.2)×0) = -0.3+0.5 = 0.2. E = -0.1-0.3-0.2 = **-0.6**.
- **Q14.7:** z_h1 = 0.3+(0×0.2+1×(-0.3)+1×0.5) = 0.3+0.2 = 0.5. z_h2 = -0.2+(0×0.1+1×0.4+1×(-0.2)) = -0.2+0.2 = 0.0. P(h1=1|v)=σ(0.5)≈**0.6225**. P(h2=1|v)=σ(0.0)=**0.5**. Hidden unit 1 is more likely to activate.
- **Q14.8:** Z sums `exp(-E(v,h))` over EVERY possible combination of visible and hidden unit values — for n_v visible units and n_h hidden units (each binary), there are `2^(n_v+n_h)` total configurations, a number that grows exponentially and becomes computationally infeasible to enumerate for any network beyond a tiny toy size (recall the `2ⁿ` combinatorial explosion pattern from Lecture 8's dropout discussion).
- **Q14.9:** In an RBM, since there are NO within-layer connections, each hidden unit's activation only depends on the VISIBLE units (not on other hidden units) — so once you know v, all hidden units can be computed independently and simultaneously. In a full Boltzmann Machine, hidden units CAN be connected to each other, so one hidden unit's value depends on OTHER hidden units' values too, breaking the independence and requiring much more expensive joint computation.
- **Q14.10:** These techniques directly solve the SAME underlying problem RBM pretraining was working around — training instability/vanishing gradients in deep networks — but more simply and effectively: proper initialization avoids starting in a bad region of weight space, ReLU avoids the vanishing-gradient-prone saturation of sigmoid/tanh, Batch Normalization keeps layer inputs well-scaled throughout training, and residual connections give gradients a direct path through many layers. With these tools, deep networks can be trained end-to-end from random initialization without needing an unsupervised RBM pretraining phase first.
</details>

<details>
<summary>Q14.11 – Q14.14 (Hard)</summary>

- **Q14.11:** a.v = 0.0×1+0.3×1+(-0.2)×0 = 0.3. b.h = 0.1×0+0.1×1 = 0.1. v^TWh = v1×(W11h1+W12h2)+v2×(W21h1+W22h2)+v3×(...) = 1×(0.4×0+(-0.1)×1)+1×(0.2×0+0.3×1)+0×(...) = -0.1+0.3+0 = 0.2. E = -0.3-0.1-0.2 = **-0.6**.
- **Q14.12:** z_h1 = 0.1+(1×0.4+1×0.2+0×(-0.5)) = 0.1+0.6 = 0.7. z_h2 = 0.1+(1×(-0.1)+1×0.3+0×0.2) = 0.1+0.2 = 0.3. P(h1=1|v)=σ(0.7)≈**0.6682**. P(h2=1|v)=σ(0.3)≈**0.5744**. Hidden unit 1 is more likely to activate.
- **Q14.13:** RBM1: 100×50 = 5,000. RBM2: 50×20 = 1,000. RBM3: 20×10 = 200. Total = 5,000+1,000+200 = **6,200 parameters** across all three RBMs (excluding biases).
- **Q14.14:** As `b_j → -100`, the pre-activation `z_h_j = b_j + (weighted visible sum)` stays strongly negative even if the weighted visible sum is a reasonably large positive number (e.g., +10, giving z=-90) — sigmoid saturates hard toward 0 for very negative inputs (`σ(-90)` is astronomically close to 0), so `P(h_j=1|v)` would be pinned near 0 essentially regardless of what the visible units actually show. This directly mirrors Lecture 2's saturation discussion: an extreme bias can effectively "kill" a unit's ability to respond to its inputs, exactly the way a very negative pre-activation saturates sigmoid/tanh and starves the unit of any meaningful gradient signal too.
</details>

`[🔝 Top](#dl-lecture-14-bonus--exercise-bank-restricted-boltzmann-machines)`

---

## Summary

This bonus exercise bank drills RBM energy, activation, and training mechanics across three tiers. Easy questions recall the energy function, the bipartite restriction, the hidden activation formula, Contrastive Divergence's purpose, and DBN's greedy stacking. Medium questions apply the energy and activation formulas to new numbers (E=-0.6, hidden probabilities ≈0.6225/0.5), and explain why computing Z is intractable, why conditional independence holds in RBMs but not full Boltzmann Machines, and why modern techniques made RBM pretraining largely unnecessary. Hard questions require full derivations: a complete energy and dual hidden-activation computation on a fresh configuration, a three-RBM DBN parameter count (6,200 total), and a detailed sigmoid-saturation argument for what happens when a hidden unit's bias is extremely negative, directly connecting back to Lecture 2's activation function discussion. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture14_rbm_practice.md) · [🔝 Top](#dl-lecture-14-bonus--exercise-bank-restricted-boltzmann-machines) · [Code →](../code/README.md)`
