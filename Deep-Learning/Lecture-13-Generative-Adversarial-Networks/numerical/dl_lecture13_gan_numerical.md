# DL Lecture 13 (Bonus) — Generative Adversarial Networks (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

> Folder: `Deep-Learning/Lecture-13-Generative-Adversarial-Networks/numerical/`
> Pairs with: [`theory/dl_lecture13_gan_theory.md`](../theory/dl_lecture13_gan_theory.md) · [`practice/dl_lecture13_gan_practice.md`](../practice/dl_lecture13_gan_practice.md) · [`exercises/dl_lecture13_exercises.md`](../exercises/dl_lecture13_exercises.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Discriminator Loss, By Hand](#worked-example-1--discriminator-loss-by-hand)
3. [Worked Example 2 — Generator Loss, Two Ways](#worked-example-2--generator-loss-two-ways)
4. [Worked Example 3 — The Vanishing Gradient Problem, Quantified](#worked-example-3--the-vanishing-gradient-problem-quantified)
5. [Worked Example 4 — The Global Optimum, Verified](#worked-example-4--the-global-optimum-verified)
6. [Worked Example 5 — FID, A Toy Calculation](#worked-example-5--fid-a-toy-calculation)
7. [Master Formula Cheatsheet](#master-formula-cheatsheet)
8. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
9. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| D(x) | Discriminator's estimated probability that x is real |
| D(G(z)) | Discriminator's estimated probability that a fake sample is real |
| L_D | Discriminator's loss |
| L_G | Generator's loss |

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Worked Example 1 — Discriminator Loss, By Hand

**Given:** for one batch, the Discriminator outputs `D(x)=0.9` for a real sample (fairly confident it's real — good), and `D(G(z))=0.2` for a fake sample (fairly confident it's fake — good).

**Step 1 — Write the Discriminator's loss (negative log-likelihood form, the practical implementation of maximizing the minimax V).**
```
L_D = -[ log(D(x)) + log(1 - D(G(z))) ]
```

**Step 2 — Plug in the numbers.**
```
log(0.9) ≈ -0.1054
log(1 - 0.2) = log(0.8) ≈ -0.2231
L_D = -[ -0.1054 + (-0.2231) ] = -(-0.3285) = 0.3285
```

**Result: L_D ≈ 0.3285.** This is a relatively LOW loss (good for the Discriminator) — it correctly leaned toward "real" for the real sample and "fake" for the fake sample. If the Discriminator instead got BOTH wrong (say D(x)=0.1, D(G(z))=0.9), the loss would be dramatically higher, reflecting worse performance.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Worked Example 2 — Generator Loss, Two Ways

**Given:** the same `D(G(z))=0.2` from Example 1 (the Discriminator is fairly confident this particular fake is fake — bad news for the Generator).

**Step 1 — Compute the ORIGINAL minimax Generator loss (to be minimized): `log(1 - D(G(z)))`.**
```
L_G_minimax = log(1 - 0.2) = log(0.8) ≈ -0.2231
```

**Step 2 — Compute the NON-SATURATING Generator loss (to be minimized): `-log(D(G(z)))`.**
```
L_G_nonsaturating = -log(0.2) ≈ -(-1.6094) = 1.6094
```

**Result:** both losses correctly signal "the Generator is doing poorly here" (D(G(z))=0.2 is far from the Generator's goal of 1.0) — but notice they're on very different SCALES (-0.2231 vs 1.6094). What matters more than the raw values is comparing their GRADIENTS, worked out next.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Worked Example 3 — The Vanishing Gradient Problem, Quantified

**Given:** the derivative of `log(1-x)` with respect to x is `-1/(1-x)`; the derivative of `-log(x)` with respect to x is `-1/x`. Compute both gradients at four different values of `D(G(z))`, representing the Generator at different stages of training (0.01 = very early/bad, 0.9 = very good/nearly winning).

**Step 1 — Compute the minimax loss gradient magnitude at each point.**
```
D(G(z))=0.01: |grad| = 1/(1-0.01) = 1/0.99 ≈ 1.0101
D(G(z))=0.10: |grad| = 1/(1-0.10) = 1/0.90 ≈ 1.1111
D(G(z))=0.50: |grad| = 1/(1-0.50) = 1/0.50 = 2.0000
D(G(z))=0.90: |grad| = 1/(1-0.90) = 1/0.10 = 10.000
```

**Step 2 — Compute the non-saturating loss gradient magnitude at each point.**
```
D(G(z))=0.01: |grad| = 1/0.01 = 100.00
D(G(z))=0.10: |grad| = 1/0.10 = 10.000
D(G(z))=0.50: |grad| = 1/0.50 = 2.0000
D(G(z))=0.90: |grad| = 1/0.90 ≈ 1.1111
```

**Result — the critical comparison, at D(G(z))=0.01 (the Generator is doing VERY poorly, exactly when it most needs a strong learning signal):**

| D(G(z)) | Minimax gradient | Non-saturating gradient |
|---|---|---|
| 0.01 (very bad) | 1.01 | **100.00** |
| 0.90 (very good) | 10.00 | 1.11 |

**At the point where the Generator needs help most (D(G(z))=0.01), the non-saturating loss's gradient is nearly 100× LARGER than the minimax loss's gradient (100.0 vs 1.01)** — this is the precise numeric proof of the vanishing gradient problem, and exactly why the non-saturating loss is used in practice. Notice the pattern flips at the OTHER end (D(G(z))=0.9, Generator nearly winning): here the ORIGINAL minimax loss actually has the larger gradient — but this region matters far less in practice, since a Generator that's already winning doesn't need as strong a push.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Worked Example 4 — The Global Optimum, Verified

**Given:** at the theoretical global optimum, `D(x) = 0.5` for every input (real or fake).

**Step 1 — Compute the Discriminator's loss AT this optimum, for one real and one fake sample.**
```
L_D = -[ log(0.5) + log(1 - 0.5) ] = -[ log(0.5) + log(0.5) ] = -2 x log(0.5)
    = -2 x (-0.6931) = 1.3863
```

**Step 2 — Compare to a "confident and correct" Discriminator, e.g. D(x)=0.99, D(G(z))=0.01 (from Worked Example 1's style, but even more confident).**
```
L_D = -[ log(0.99) + log(1-0.01) ] = -[ log(0.99)+log(0.99) ] = -2 x log(0.99)
    = -2 x (-0.01005) = 0.0201
```

**Result:** the global-optimum Discriminator's loss (1.3863) is MUCH HIGHER than a confidently-correct Discriminator's loss (0.0201) — this might look "worse" numerically, but it's actually the theoretical BEST the Discriminator can ever achieve once the Generator has truly won, because `D=0.5` for everything is a mathematical necessity when real and fake are genuinely statistically indistinguishable, not a sign of the Discriminator performing poorly by choice. `log(0.5)` for both terms is precisely the value `-ln(2) ≈ -0.6931`, and `-2×(-0.6931)=1.3863=2ln(2)` is the exact, provable value of the minimax game at its true global optimum — a specific, memorable number.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Worked Example 5 — FID, A Toy Calculation

**Given:** a simplified 1-dimensional version of FID (real FID uses high-dimensional feature vectors and full covariance matrices; this toy version uses single numbers for hand-tractability). Real data: mean μ₁=0.0, std σ₁=1.0. Generated data: mean μ₂=0.5, std σ₂=1.2.

**Step 1 — Apply the (1D-simplified) FID formula.**
```
FID = (mu1 - mu2)^2 + sigma1^2 + sigma2^2 - 2 x sqrt(sigma1^2 x sigma2^2)
```

**Step 2 — Plug in the numbers.**
```
(0.0-0.5)^2 = 0.25
sigma1^2 + sigma2^2 = 1.0^2 + 1.2^2 = 1.0 + 1.44 = 2.44
sqrt(sigma1^2 x sigma2^2) = sqrt(1.0 x 1.44) = sqrt(1.44) = 1.2
2 x 1.2 = 2.4

FID = 0.25 + 2.44 - 2.4 = 0.29
```

**Step 3 — Compare to the "identical distributions" case (μ₂=0.0, σ₂=1.0, i.e., generated perfectly matches real).**
```
FID = (0-0)^2 + 1+1 - 2xsqrt(1x1) = 0+2-2 = 0
```

**Result: FID=0.29 for the mismatched distributions, FID=0 exactly when the two distributions are identical** — confirming "lower FID is better, 0 means statistically identical" from the theory file, with a concrete worked number showing what a real gap between real and generated data distributions looks like numerically.

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Discriminator loss | `L_D = -[log(D(x)) + log(1-D(G(z)))]` |
| Generator loss (minimax) | `L_G = log(1-D(G(z)))` (minimize) |
| Generator loss (non-saturating) | `L_G = -log(D(G(z)))` (minimize) |
| Minimax loss gradient magnitude | `1/(1-D(G(z)))` |
| Non-saturating loss gradient magnitude | `1/D(G(z))` |
| Global optimum Discriminator loss | `2 ln(2) ≈ 1.3863` |
| FID (1D simplified) | `(μ1-μ2)² + σ1² + σ2² - 2√(σ1²σ2²)` |

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting the negative sign when converting the minimax formulation into a practical "loss to minimize" — the Discriminator's loss is the NEGATIVE of what it's trying to maximize.
- **Trap:** computing gradient magnitude without taking the absolute value / correct sign convention — always double check whether a question wants signed derivative or magnitude.
- **Trap:** assuming the global optimum means the Discriminator's loss goes to ZERO — it actually converges to `2ln(2)≈1.3863`, a specific nonzero value, precisely because D(x)=0.5 is NOT a confident, low-loss prediction; it reflects genuine, irreducible uncertainty.
- **Exam hack:** the vanishing gradient comparison table (Worked Example 3) is the single most important numeric result in this bonus lecture — be ready to reproduce the ~100× gradient ratio at D(G(z))=0.01 from scratch, including the two derivative formulas.
- **Exam hack:** for FID questions, always state clearly that lower is better and 0 means identical distributions — and that the real formula operates on high-dimensional feature covariance matrices, not single numbers (this toy version is for building intuition only).

`[🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical)`

---

## Summary

This file worked every core GAN loss calculation from the theory file into fully shown arithmetic. A concrete Discriminator loss computation (D(x)=0.9, D(G(z))=0.2) gave L_D≈0.3285, a low/good value reflecting mostly-correct classification. Comparing the original minimax Generator loss against the non-saturating alternative at the same D(G(z))=0.2 showed very different raw values (-0.2231 vs 1.6094), but the real story emerged in Worked Example 3's full gradient-magnitude comparison: at D(G(z))=0.01 (Generator performing very poorly, exactly when strong learning signal matters most), the non-saturating loss's gradient (100.0) is nearly 100× larger than the minimax loss's gradient (1.01) — the precise numeric proof of why GAN training in practice almost universally uses the non-saturating loss. The theoretical global optimum was verified numerically: at D(x)=0.5 everywhere, the Discriminator's loss settles at exactly 2ln(2)≈1.3863, notably HIGHER than a confidently-correct Discriminator's loss, reflecting genuine irreducible uncertainty rather than poor performance. Finally, a simplified 1D FID calculation compared a real distribution (μ=0,σ=1) against a mismatched generated distribution (μ=0.5,σ=1.2), producing FID≈0.29, with a companion calculation confirming FID=0 exactly for identical distributions. The master formula table consolidates every reusable GAN calculation from this bonus lecture for fast review.

`[← Theory](../theory/dl_lecture13_gan_theory.md) · [🔝 Top](#dl-lecture-13-bonus--generative-adversarial-networks-numerical) · [Next: Practice →](../practice/dl_lecture13_gan_practice.md)`
