# DL Lecture 07 — DNN Optimization (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

> Folder: `Deep-Learning/Lecture-07-DNN-Optimization/numerical/`
> Pairs with: [`theory/dl_lecture07_optimization_theory.md`](../theory/dl_lecture07_optimization_theory.md) · [`practice/dl_lecture07_optimization_practice.md`](../practice/dl_lecture07_optimization_practice.md) · [`exercises/dl_lecture07_exercises.md`](../exercises/dl_lecture07_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Plain SGD Update](#worked-example-1--plain-sgd-update)
3. [Worked Example 2 — Momentum, Two Steps](#worked-example-2--momentum-two-steps)
4. [Worked Example 3 — RMSProp, Two Steps](#worked-example-3--rmsprop-two-steps)
5. [Worked Example 4 — Adam, Two Steps (With Bias Correction)](#worked-example-4--adam-two-steps-with-bias-correction)
6. [Worked Example 5 — Learning Rate Decay Schedule](#worked-example-5--learning-rate-decay-schedule)
7. [Master Formula Cheatsheet](#master-formula-cheatsheet)
8. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
9. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| θ (theta) | a parameter (weight) being optimized |
| η (eta) | learning rate |
| g | gradient at the current step |
| γ (gamma) | momentum coefficient |
| v | velocity (momentum) or moving average (RMSProp/Adam context-dependent) |
| β, β1, β2 | decay rates for moving averages (RMSProp uses β; Adam uses β1, β2) |
| m | first-moment estimate (mean of gradients, Adam) |
| ε (epsilon) | tiny constant preventing division by zero |

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Worked Example 1 — Plain SGD Update

**Given:** θ=1.0, learning rate η=0.1, gradient g=4.0.

**Step 1 — Apply the SGD update formula.**
```
theta_new = theta - eta * g = 1.0 - 0.1 x 4.0 = 1.0 - 0.4 = 0.6
```

**Result: θ moves from 1.0 to 0.6.** Notice the direction: since the gradient is positive (4.0, pointing toward increasing loss), the parameter moves DOWN (decreases) — exactly the "move opposite the gradient" rule from the theory file.

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Worked Example 2 — Momentum, Two Steps

**Given:** θ=1.0, v=0 (initial velocity), γ=0.9, η=0.1, gradients over two steps: g₁=4.0, g₂=3.0.

**Step 1 (t=1) — Compute new velocity.**
```
v_1 = gamma*v_0 + eta*g_1 = 0.9x0 + 0.1x4.0 = 0 + 0.4 = 0.4
```

**Step 2 (t=1) — Update θ.**
```
theta_1 = theta_0 - v_1 = 1.0 - 0.4 = 0.6
```

**Step 3 (t=2) — Compute new velocity (carrying over previous velocity).**
```
v_2 = gamma*v_1 + eta*g_2 = 0.9x0.4 + 0.1x3.0 = 0.36 + 0.30 = 0.66
```

**Step 4 (t=2) — Update θ.**
```
theta_2 = theta_1 - v_2 = 0.6 - 0.66 = -0.06
```

**Result: θ moves 1.0 → 0.6 → -0.06.** Notice the SECOND step's move (0.66) is LARGER than the first step's move (0.4), even though the second gradient (3.0) was smaller than the first (4.0) — this is momentum's "carrying speed forward" effect in action: 90% of the previous velocity (0.4×0.9=0.36) got added on top of the new gradient's own contribution (0.1×3.0=0.30).

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Worked Example 3 — RMSProp, Two Steps

**Given:** θ=1.0, E[g²]₀=0 (initial squared-gradient average), decay β=0.9, η=0.1, ε=1e-8 (negligibly small, included only for completeness), gradients: g₁=4.0, g₂=-3.0.

**Step 1 (t=1) — Update the decaying average of squared gradients.**
```
E[g^2]_1 = beta*E[g^2]_0 + (1-beta)*g_1^2 = 0.9x0 + 0.1x16 = 1.6
```

**Step 2 (t=1) — Compute the update.**
```
update_1 = eta / sqrt(E[g^2]_1 + eps) * g_1 = 0.1/sqrt(1.6) x 4.0 = 0.0791 x 4.0 ≈ 0.3162
theta_1 = theta_0 - update_1 = 1.0 - 0.3162 = 0.6838
```

**Step 3 (t=2) — Update the decaying average again.**
```
E[g^2]_2 = 0.9x1.6 + 0.1x(-3.0)^2 = 1.44 + 0.9 = 2.34
```

**Step 4 (t=2) — Compute the update.**
```
update_2 = eta / sqrt(E[g^2]_2 + eps) * g_2 = 0.1/sqrt(2.34) x (-3.0) = 0.0654 x (-3.0) ≈ -0.1961
theta_2 = theta_1 - update_2 = 0.6838 - (-0.1961) = 0.8799
```

**Result: θ moves 1.0 → 0.6838 → 0.8799.** Notice at step 2, the gradient flips sign (negative), so the update ALSO flips sign, moving θ back UP — RMSProp still correctly follows the sign of the current gradient; what it adaptively controls is the STEP SIZE (scaled by the accumulated squared-gradient history), not the direction.

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Worked Example 4 — Adam, Two Steps (With Bias Correction)

**Given:** θ=1.0, m₀=0, v₀=0, β1=0.9, β2=0.999, η=0.1, ε=1e-8, gradients: g₁=4.0, g₂=3.0.

**Step 1 (t=1) — Update first-moment (mean) and second-moment (squared) estimates.**
```
m_1 = beta1*m_0 + (1-beta1)*g_1 = 0.9x0 + 0.1x4.0 = 0.4
v_1 = beta2*v_0 + (1-beta2)*g_1^2 = 0.999x0 + 0.001x16 = 0.016
```

**Step 2 (t=1) — Apply BIAS CORRECTION (crucial for early steps, since m and v start at 0 and are biased toward 0 initially).**
```
m_hat_1 = m_1 / (1 - beta1^1) = 0.4 / (1-0.9) = 0.4/0.1 = 4.0
v_hat_1 = v_1 / (1 - beta2^1) = 0.016 / (1-0.999) = 0.016/0.001 = 16.0
```

**Step 3 (t=1) — Compute the update and new θ.**
```
update_1 = eta * m_hat_1 / (sqrt(v_hat_1) + eps) = 0.1 x 4.0 / (sqrt(16)+eps) = 0.4/4.0 = 0.1
theta_1 = theta_0 - update_1 = 1.0 - 0.1 = 0.9
```

**Step 4 (t=2) — Repeat for the second gradient.**
```
m_2 = 0.9x0.4 + 0.1x3.0 = 0.36+0.30 = 0.66
v_2 = 0.999x0.016 + 0.001x9 = 0.015984+0.009 = 0.024984
m_hat_2 = 0.66/(1-0.9^2) = 0.66/0.19 ≈ 3.4737
v_hat_2 = 0.024984/(1-0.999^2) ≈ 0.024984/0.001999 ≈ 12.4982
update_2 = 0.1 x 3.4737/(sqrt(12.4982)+eps) ≈ 0.1x3.4737/3.5352 ≈ 0.0983
theta_2 = 0.9 - 0.0983 = 0.8017
```

**Result: θ moves 1.0 → 0.9 → 0.8017.** Notice how much smaller and smoother these Adam steps are (0.1, then 0.098) compared to plain SGD's single 0.4 step or momentum's escalating 0.4-then-0.66 steps — this is Adam's combined bias-corrected momentum-plus-adaptive-scaling behaviour producing steadier, more controlled updates.

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Worked Example 5 — Learning Rate Decay Schedule

**Given:** an initial learning rate η₀=0.1, and a step-decay schedule that HALVES the learning rate every 10 epochs.

**Step 1 — Epochs 0–9: η = 0.1** (unchanged).
**Step 2 — Epochs 10–19: η = 0.1/2 = 0.05.**
**Step 3 — Epochs 20–29: η = 0.05/2 = 0.025.**
**Step 4 — Epochs 30–39: η = 0.025/2 = 0.0125.**

**General formula for step decay:**
```
eta(epoch) = eta_0 x (decay_factor)^floor(epoch / decay_every)
           = 0.1 x (0.5)^floor(epoch / 10)
```
Check at epoch=25: `floor(25/10)=2`, so `eta = 0.1 x 0.5^2 = 0.1x0.25 = 0.025` ✓ matches Step 3.

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Master Formula Cheatsheet

| Optimizer | Update formula |
|---|---|
| Plain SGD | `θ = θ - η·g` |
| Momentum | `v = γv + ηg`;  `θ = θ - v` |
| RMSProp | `E[g²] = βE[g²] + (1-β)g²`;  `θ = θ - η/√(E[g²]+ε) · g` |
| Adam (raw moments) | `m = β1·m + (1-β1)g`;  `v = β2·v + (1-β2)g²` |
| Adam (bias-corrected) | `m̂ = m/(1-β1^t)`;  `v̂ = v/(1-β2^t)` |
| Adam (final update) | `θ = θ - η·m̂/(√v̂+ε)` |
| Step decay | `η(epoch) = η₀ · decay^⌊epoch/step_size⌋` |

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting bias correction in Adam, especially at early timesteps — without it, `m̂` and `v̂` are severely underestimated near t=1 (both m and v start at exactly 0), causing incorrect, overly-small early updates. Always compute `m_hat` and `v_hat` explicitly, never skip straight from `m,v` to the final update.
- **Trap:** in momentum, forgetting that `v` carries over between steps — a common mistake is resetting v to 0 every step, which would make momentum behave identically to plain SGD (defeating its purpose).
- **Trap:** applying the SAME learning rate scaling to every parameter in RMSProp/Adam — the entire point is that `E[g²]` (or `v`) is tracked PER-PARAMETER independently, so different weights can end up with very different effective step sizes.
- **Exam hack:** always show bias-corrected `m_hat`/`v_hat` as SEPARATE explicit steps in Adam calculations — graders specifically check for this intermediate step, not just the final theta.
- **Exam hack:** for "explain why momentum's second step was bigger despite a smaller gradient" style questions, always explicitly compute and cite the carried-over `γ×v_previous` term — this is the crux of the explanation examiners want.

`[🔝 Top](#dl-lecture-07--dnn-optimization-numerical)`

---

## Summary

This file worked every optimizer formula from the theory file into fully shown arithmetic, all starting from θ=1.0 with comparable gradients, to make direct comparison easy. Plain SGD with η=0.1 and g=4.0 moved θ from 1.0 to 0.6 in one clean step. Momentum, with γ=0.9 across two steps (gradients 4.0 then 3.0), moved θ to 0.6 then -0.06 — the second step (0.66) was actually LARGER than the first (0.4) despite a smaller gradient, because 90% of the previous velocity carried forward on top of the new gradient's contribution. RMSProp, tracking a decaying average of squared gradients, moved θ to 0.6838 then 0.8799 across a sign-flipping gradient sequence (4.0, then -3.0), correctly reversing direction while adaptively scaling step size based on gradient history. Adam, combining bias-corrected first and second moment estimates, produced smaller, steadier updates (1.0 → 0.9 → 0.8017) than either plain SGD or Momentum alone — explicitly demonstrating why bias correction matters most at early timesteps, when the raw moment estimates are still heavily biased toward zero. Finally, a step-decay learning rate schedule example showed how η=0.1 nearly halves every 10 epochs, following the formula `η(epoch)=η₀·decay^⌊epoch/step_size⌋`, verified against a direct epoch-25 calculation. The master formula table consolidates every optimizer's update rule for fast side-by-side comparison and review.

`[← Theory](../theory/dl_lecture07_optimization_theory.md) · [🔝 Top](#dl-lecture-07--dnn-optimization-numerical) · [Next: Practice →](../practice/dl_lecture07_optimization_practice.md)`
