# DL Lecture 08 — Exercise Bank (Regularization)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-08--exercise-bank-regularization)`

> Folder: `Deep-Learning/Lecture-08-Regularization/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture08_regularization_theory.md) · [numerical](../numerical/dl_lecture08_regularization_numerical.md) · [practice](../practice/dl_lecture08_regularization_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q8.1.** Write the total regularized loss formula.

**Q8.2.** What value of q in the general Lq regularizer gives Lasso? What value gives L2?

**Q8.3.** What is the formula for the number of possible thinned dropout networks, given n nodes?

**Q8.4.** What happens to a neuron's output when its dropout mask value is 0?

**Q8.5.** Which data split should be used only once, at the very end of a project?

---

## 🟡 Medium — Applied Reasoning

**Q8.6.** For x=[2,4], y=[3,7], λ=3, compute the unregularized and L2-regularized closed-form weight (1D case).

**Q8.7.** For weight vector w=[2,-3,1,0.5], compute the L1 and L2 penalty values.

**Q8.8.** For n=8 dropout-eligible nodes, compute the number of possible thinned networks.

**Q8.9.** Explain why models are described in the lecture as often being deliberately "overparametrized" in practice, and why this doesn't automatically doom generalization.

**Q8.10.** Explain the difference between what data augmentation does and what dropout does, even though both are regularization techniques.

---

## 🔴 Hard — Derivation & Multi-Step

**Q8.11.** Given the following validation losses by epoch: `[1.2, 0.9, 0.6, 0.55, 0.58, 0.65, 0.70]`, identify the early-stopping epoch and justify your answer using the full sequence.

**Q8.12.** A dropout layer has activations `a=[5.0, 1.0, 2.0, 8.0, 3.0]` and mask `[0,1,1,0,1]` (p=0.5). Compute the training-time output, then separately compute the test-time output (full network, scaled by p).

**Q8.13.** Explain, with a worked numeric argument, why L2's penalty disproportionately punishes one large weight versus several small weights of the same total magnitude. Compute the L2 penalty for w_a=[4,0,0] and w_b=[2,1,1] (both sum to the same total absolute magnitude of 4), and compare.

**Q8.14.** For a network with 3 dropout-eligible layers of sizes 5, 10, and 20 nodes respectively (dropout applied independently within each layer), compute the total number of distinct thinned-network configurations across the whole network (i.e., the product across all three layers).

`[🔝 Top](#dl-lecture-08--exercise-bank-regularization)`

---

## Answer Key

<details>
<summary>Q8.1 – Q8.5 (Easy)</summary>

- **Q8.1:** `E(w) = E_D(w) + (λ/2)w^Tw`.
- **Q8.2:** q=1 gives Lasso (L1); q=2 gives L2.
- **Q8.3:** `2ⁿ`.
- **Q8.4:** The neuron's output is completely zeroed out (removed) for that forward pass, and its weights are not updated by that batch's backward pass.
- **Q8.5:** The test set.
</details>

<details>
<summary>Q8.6 – Q8.10 (Medium)</summary>

- **Q8.6:** Σ(xy)=2×3+4×7=6+28=34. Σ(x²)=4+16=20. w_unreg=34/20=**1.7**. w_reg=34/(20+3)=34/23≈**1.4783**.
- **Q8.7:** L1 = |2|+|-3|+|1|+|0.5| = 2+3+1+0.5 = **6.5**. L2 = 0.5×(4+9+1+0.25) = 0.5×14.25 = **7.125**.
- **Q8.8:** 2⁸ = **256**.
- **Q8.9:** Overparametrized means the model has far more parameters than strictly needed to fit the training data perfectly. Research (Neyshabur et al., Belkin et al.) shows that good architecture choices combined with SGD's own implicit regularization properties allow such models to still generalize well — the raw parameter COUNT alone doesn't determine generalization; how the model is trained (optimizer, batch size, normalization) matters just as much, so overparametrization isn't automatically harmful when paired with these other factors (and often explicit regularization on top).
- **Q8.10:** Data augmentation operates on the INPUT data, generating new synthetic training examples via transformations (rotation, flipping, noise, etc.) — it changes what the network SEES. Dropout operates on the NETWORK'S INTERNAL STRUCTURE during training, randomly deactivating neurons — it changes how the network COMPUTES, not what data it's given. Both reduce overfitting, but through entirely different mechanisms (richer/more diverse input data vs. more robust internal representations).
</details>

<details>
<summary>Q8.11 – Q8.14 (Hard)</summary>

- **Q8.11:** Sequence: 1.2, 0.9, 0.6, 0.55, 0.58, 0.65, 0.70. The minimum value is 0.55, at epoch 4 (1-indexed). Epochs 5, 6, 7 show strictly increasing loss (0.58→0.65→0.70), confirming the overfitting signature — the early-stopping epoch is **epoch 4**.
- **Q8.12:** Training output = mask×a = [0×5.0, 1×1.0, 1×2.0, 0×8.0, 1×3.0] = **[0, 1.0, 2.0, 0, 3.0]**. Test output = p×a = 0.5×[5.0,1.0,2.0,8.0,3.0] = **[2.5, 0.5, 1.0, 4.0, 1.5]**.
- **Q8.13:** L2(w_a) = 0.5×(4²+0²+0²) = 0.5×16 = **8.0**. L2(w_b) = 0.5×(2²+1²+1²) = 0.5×(4+1+1) = 0.5×6 = **3.0**. Even though both vectors sum to the same total absolute magnitude (|4|+|0|+|0|=4 and |2|+|1|+|1|=4), the L2 penalty is nearly 2.7× larger for the concentrated single large weight (8.0) than for the spread-out smaller weights (3.0) — this numerically demonstrates why L2 regularization actively discourages concentrating magnitude into a few large weights, preferring instead to spread weight magnitude out more evenly.
- **Q8.14:** Layer 1 (5 nodes): 2⁵=32. Layer 2 (10 nodes): 2¹⁰=1024. Layer 3 (20 nodes): 2²⁰=1,048,576. Total combinations = 32×1024×1,048,576 = **34,359,738,368** (over 34 billion) distinct thinned-network configurations across the whole network — vividly illustrating why training each one separately is utterly infeasible, and why the dropout trick (one shared set of weights) is essential.
</details>

`[🔝 Top](#dl-lecture-08--exercise-bank-regularization)`

---

## Summary

This exercise bank drills Lecture 8's regularization formulas across three tiers. Easy questions recall the total regularized loss formula, the Lq-to-L1/L2 mapping, the 2ⁿ thinned-network formula, dropout's masking behaviour, and the test set's single-use rule. Medium questions apply the closed-form L2 formula to new numbers (w: 1.7→1.4783), compute L1/L2 penalties for a new weight vector (6.5 vs 7.125), count thinned networks for n=8 (256), and reason about why overparametrization isn't automatically harmful and how data augmentation differs mechanistically from dropout. Hard questions require full derivations: identifying an early-stopping epoch from a 7-point validation loss sequence (epoch 4), a complete dropout training-vs-test output comparison, a numeric proof that L2 penalizes concentrated large weights far more than spread-out smaller weights of equal total magnitude (8.0 vs 3.0, despite equal L1 totals), and a multi-layer thinned-network count reaching over 34 billion combinations across a modestly-sized 3-layer network. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture08_regularization_practice.md) · [🔝 Top](#dl-lecture-08--exercise-bank-regularization) · [Code →](../code/README.md)`
