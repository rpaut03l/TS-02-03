# DL Lecture 07 — Exercise Bank (DNN Optimization)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-07--exercise-bank-dnn-optimization)`

> Folder: `Deep-Learning/Lecture-07-DNN-Optimization/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture07_optimization_theory.md) · [numerical](../numerical/dl_lecture07_optimization_numerical.md) · [practice](../practice/dl_lecture07_optimization_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q7.1.** Write the plain SGD weight update formula.

**Q7.2.** What does the momentum coefficient γ control?

**Q7.3.** What does RMSProp adapt, and based on what quantity?

**Q7.4.** Name the two things Adam tracks (its "moments").

**Q7.5.** What is the difference between a local minimum and a saddle point?

---

## 🟡 Medium — Applied Reasoning

**Q7.6.** For θ=2.0, η=0.5, g=1.5, compute the plain SGD update.

**Q7.7.** For θ=2.0, v=0.2 (previous velocity), γ=0.8, η=0.1, g=2.0, compute the new velocity and updated θ using momentum.

**Q7.8.** Explain why Batch GD's "well-understood convergence conditions" advantage doesn't make it the practical default choice.

**Q7.9.** Explain what happens to RMSProp's effective step size for a parameter whose gradients have historically been very large, versus one whose gradients have been consistently small.

**Q7.10.** Why is bias correction more impactful at t=1 than at, say, t=1000, in Adam?

---

## 🔴 Hard — Derivation & Multi-Step

**Q7.11.** For θ=0.5, E[g²]₀=0.5, β=0.9, η=0.2, g=2.0: compute the updated E[g²], the update magnitude, and the new θ using RMSProp.

**Q7.12.** For θ=0.0, m₀=0, v₀=0, β1=0.9, β2=0.999, η=0.05, g=6.0 (first step, t=1): compute m₁, v₁, the bias-corrected m̂₁ and v̂₁, and the new θ using Adam.

**Q7.13.** A step-decay schedule starts at η₀=0.2 and halves every 15 epochs. Compute η at epoch 0, 15, 35, and 55.

**Q7.14.** Explain, with a short worked numeric argument, why in Worked Example 4 of the numerical file, Adam's bias-corrected v̂₁ (16.0) is dramatically larger than the raw v₁ (0.016) at t=1, but this gap shrinks quickly at later timesteps — compute v̂ vs raw v at t=1, t=5, and t=20 for β2=0.999 assuming v stays roughly constant at 0.016 (i.e., only compute how the correction factor `1/(1-β2^t)` itself changes).

`[🔝 Top](#dl-lecture-07--exercise-bank-dnn-optimization)`

---

## Answer Key

<details>
<summary>Q7.1 – Q7.5 (Easy)</summary>

- **Q7.1:** `θ = θ - η·g`.
- **Q7.2:** How much of the PREVIOUS update/velocity carries forward into the current update — larger γ means past gradients influence the current step more strongly.
- **Q7.3:** RMSProp adapts each parameter's individual effective learning rate, based on an exponentially decaying moving average of that parameter's own PAST SQUARED GRADIENTS.
- **Q7.4:** A first-moment estimate (exponentially decaying average of past gradients, like momentum) and a second-moment estimate (exponentially decaying average of past squared gradients, like RMSProp).
- **Q7.5:** A local minimum is a point where the function value is lower than all nearby points (though possibly not the global lowest point). A saddle point also has zero gradient, but is NOT a minimum at all — it curves upward in some directions and downward in others, like a horse saddle.
</details>

<details>
<summary>Q7.6 – Q7.10 (Medium)</summary>

- **Q7.6:** θ_new = 2.0 - 0.5×1.5 = 2.0-0.75 = **1.25**.
- **Q7.7:** v_new = 0.8×0.2 + 0.1×2.0 = 0.16+0.20 = **0.36**. θ_new = 2.0 - 0.36 = **1.64**.
- **Q7.8:** Batch GD requires processing the ENTIRE training set before making even a single parameter update — for large datasets this means very few updates per unit of time/compute, making training painfully slow in practice, even though its convergence behaviour is theoretically cleaner and easier to analyze. Practical training speed usually outweighs theoretical convergence-analysis convenience, which is why Mini-Batch SGD (not Batch GD) is the standard default.
- **Q7.9:** For a parameter with historically LARGE gradients, `E[g²]` grows large, so dividing by `√(E[g²]+ε)` shrinks the effective step size (preventing overshooting/instability). For a parameter with historically SMALL gradients, `E[g²]` stays small, so the effective step size stays relatively larger (helping it still make meaningful progress despite a "flat-feeling" direction).
- **Q7.10:** At t=1, both `m` and `v` are computed from just ONE gradient observation combined with an initial value of exactly 0 (heavily biased toward 0, since `(1-β1)` and `(1-β2)` are small fractions of the true signal). The correction factors `1/(1-β1^t)` and `1/(1-β2^t)` are largest exactly when t is small (since β^t is close to 1 for small t, making `1-β^t` close to 0, so its reciprocal is huge) — as t grows, `β^t → 0`, so the correction factor approaches 1 and becomes nearly irrelevant, meaning bias correction matters most in early training and fades away naturally as training progresses.
</details>

<details>
<summary>Q7.11 – Q7.14 (Hard)</summary>

- **Q7.11:** E[g²]_new = 0.9×0.5 + 0.1×2.0² = 0.45+0.4 = **0.85**. update = 0.2/√(0.85+ε)×2.0 ≈ 0.2/0.9220×2.0 ≈ **0.4339**. θ_new = 0.5 - 0.4339 ≈ **0.0661**.
- **Q7.12:** m₁ = 0.9×0+0.1×6.0 = **0.6**. v₁ = 0.999×0+0.001×36 = **0.036**. m̂₁ = 0.6/(1-0.9) = **6.0**. v̂₁ = 0.036/(1-0.999) = **36.0**. update = 0.05×6.0/(√36+ε) = 0.3/6.0 = **0.05**. θ_new = 0.0 - 0.05 = **-0.05**.
- **Q7.13:** epoch 0: floor(0/15)=0 → η=0.2×0.5⁰=**0.2**. epoch 15: floor(15/15)=1 → η=0.2×0.5¹=**0.1**. epoch 35: floor(35/15)=2 → η=0.2×0.5²=**0.05**. epoch 55: floor(55/15)=3 → η=0.2×0.5³=**0.025**.
- **Q7.14:** Correction factor 1/(1-0.999^t): at t=1, 0.999^1=0.999, 1-0.999=0.001, factor=1000 (so v̂=0.016×1000/16... using the file's actual v=0.016, v̂=0.016×1000=16.0, matching Worked Example 4). At t=5, 0.999^5≈0.99501, 1-0.99501≈0.00499, factor≈200.4 → v̂≈0.016×200.4≈3.206. At t=20, 0.999^20≈0.9802, 1-0.9802≈0.0198, factor≈50.5 → v̂≈0.016×50.5≈0.808. The correction factor shrinks dramatically from 1000 (t=1) to about 200 (t=5) to about 50 (t=20) — confirming bias correction's effect fades rapidly as training progresses, exactly as explained in Q7.10.
</details>

`[🔝 Top](#dl-lecture-07--exercise-bank-dnn-optimization)`

---

## Summary

This exercise bank drills Lecture 7's optimizer formulas across three tiers. Easy questions recall the SGD update rule, momentum's γ coefficient, RMSProp's adaptive mechanism, Adam's two tracked moments, and the local-minimum-vs-saddle-point distinction. Medium questions apply the SGD and Momentum formulas to new numbers, and reason about why Batch GD's theoretical cleanliness doesn't make it the practical default, how RMSProp's step size responds to gradient history, and why Adam's bias correction matters most early in training. Hard questions require full derivations: a complete RMSProp step (θ: 0.5→0.0661), a complete bias-corrected Adam step from scratch (θ: 0.0→-0.05), a four-point step-decay schedule calculation (0.2→0.1→0.05→0.025), and a quantitative demonstration of how Adam's bias-correction factor shrinks from 1000× at t=1 down to about 50× by t=20 — directly explaining why the correction's impact fades as training progresses. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture07_optimization_practice.md) · [🔝 Top](#dl-lecture-07--exercise-bank-dnn-optimization) · [Code →](../code/README.md)`
