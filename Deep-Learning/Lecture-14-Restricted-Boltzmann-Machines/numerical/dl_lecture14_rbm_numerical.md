# DL Lecture 14 (Bonus) — Restricted Boltzmann Machines (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

> Folder: `Deep-Learning/Lecture-14-Restricted-Boltzmann-Machines/numerical/`
> Pairs with: [`theory/dl_lecture14_rbm_theory.md`](../theory/dl_lecture14_rbm_theory.md) · [`practice/dl_lecture14_rbm_practice.md`](../practice/dl_lecture14_rbm_practice.md) · [`exercises/dl_lecture14_exercises.md`](../exercises/dl_lecture14_exercises.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Computing the Energy of a Configuration](#worked-example-1--computing-the-energy-of-a-configuration)
3. [Worked Example 2 — Hidden Unit Activation Probabilities](#worked-example-2--hidden-unit-activation-probabilities)
4. [Worked Example 3 — Visible Reconstruction Probabilities](#worked-example-3--visible-reconstruction-probabilities)
5. [Worked Example 4 — One Full CD-1 Weight Update](#worked-example-4--one-full-cd-1-weight-update)
6. [Master Formula Cheatsheet](#master-formula-cheatsheet)
7. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
8. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| v | visible unit vector |
| h | hidden unit vector |
| a | visible bias vector |
| b | hidden bias vector |
| W | visible-hidden weight matrix |
| η (eta) | learning rate |

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Worked Example 1 — Computing the Energy of a Configuration

**Given:** `v=[1,0,1]`, `h=[1,1]`, `a=[0.2,-0.1,0.3]`, `b=[0.1,-0.2]`, `W=[[0.5,-0.3],[0.2,0.4],[-0.1,0.6]]`.

**Step 1 — Compute the visible bias term, `a^T v`.**
```
a.v = 0.2x1 + (-0.1)x0 + 0.3x1 = 0.2 + 0 + 0.3 = 0.5
```

**Step 2 — Compute the hidden bias term, `b^T h`.**
```
b.h = 0.1x1 + (-0.2)x1 = 0.1 - 0.2 = -0.1
```

**Step 3 — Compute the interaction term, `v^T W h`.** First compute `Wh`, then dot with v (or equivalently, sum every `v_i·W_ij·h_j` term):
```
v^T W h = v1(W11h1+W12h2) + v2(W21h1+W22h2) + v3(W31h1+W32h2)
        = 1x(0.5x1+(-0.3)x1) + 0x(0.2x1+0.4x1) + 1x((-0.1)x1+0.6x1)
        = 1x(0.2) + 0x(0.6) + 1x(0.5)
        = 0.2 + 0 + 0.5 = 0.7
```

**Step 4 — Combine into the full energy formula.**
```
E(v,h) = -(a.v) - (b.h) - (v^T W h) = -0.5 - (-0.1) - 0.7 = -0.5+0.1-0.7 = -1.1
```

**Result: E(v,h) = -1.1.** A NEGATIVE energy here means this particular (v,h) configuration is being pulled toward HIGHER probability by the model — remember, lower/more-negative energy always corresponds to higher probability via `P∝exp(-E)`.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Worked Example 2 — Hidden Unit Activation Probabilities

**Given:** the same v, b, W as Example 1. Compute `P(h_j=1|v)` for both hidden units.

**Step 1 — Compute the pre-activation for h1: `b_1 + Σ_i v_i·W_i1`.**
```
z_h1 = 0.1 + (1x0.5 + 0x0.2 + 1x(-0.1)) = 0.1 + (0.5+0-0.1) = 0.1+0.4 = 0.5
```

**Step 2 — Compute the pre-activation for h2: `b_2 + Σ_i v_i·W_i2`.**
```
z_h2 = -0.2 + (1x(-0.3) + 0x0.4 + 1x0.6) = -0.2 + (-0.3+0+0.6) = -0.2+0.3 = 0.1
```

**Step 3 — Apply sigmoid to each.**
```
P(h1=1|v) = sigmoid(0.5) = 1/(1+e^-0.5) ≈ 0.6225
P(h2=1|v) = sigmoid(0.1) = 1/(1+e^-0.1) ≈ 0.5250
```

**Result: P(h1=1|v)≈0.6225, P(h2=1|v)≈0.5250.** Given this particular visible vector, hidden unit 1 is somewhat more likely to activate (62.25%) than hidden unit 2 (52.50%) — these two probabilities were computed COMPLETELY INDEPENDENTLY of each other, exactly the conditional-independence property the theory file describes as the key benefit of the bipartite "restricted" structure.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Worked Example 3 — Visible Reconstruction Probabilities

**Given:** the same a, W from Example 1, and `h=[1,1]` (both hidden units "on").

**Step 1 — Compute the pre-activation for each visible unit: `a_i + Σ_j W_ij·h_j`.**
```
z_v1 = 0.2 + (0.5x1 + (-0.3)x1) = 0.2 + 0.2 = 0.4
z_v2 = -0.1 + (0.2x1 + 0.4x1)   = -0.1 + 0.6 = 0.5
z_v3 = 0.3 + ((-0.1)x1 + 0.6x1) = 0.3 + 0.5 = 0.8
```

**Step 2 — Apply sigmoid to each.**
```
P(v1=1|h) = sigmoid(0.4) ≈ 0.5987
P(v2=1|h) = sigmoid(0.5) ≈ 0.6225
P(v3=1|h) = sigmoid(0.8) ≈ 0.6900
```

**Result: [0.5987, 0.6225, 0.6900].** This is the RBM "reconstructing" a visible vector purely from the hidden units — exactly the reverse direction of Example 2's computation, using the SAME weight matrix W (just transposed in role, not literally transposed in math — `W_ij` connects v_i to h_j either direction) and the visible bias a instead of the hidden bias b.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Worked Example 4 — One Full CD-1 Weight Update

**Given:** real training example `v_data=[1,0,1]`, same a, b, W as before, learning rate η=0.1. Using MEAN-FIELD CD-1 (using probabilities directly instead of stochastic 0/1 samples, for a cleaner hand-computable example — a common simplification in practice).

**Step 1 — Positive phase: compute h_data (hidden probabilities given real v_data).** (Same computation as Example 2.)
```
h_data ≈ [0.6225, 0.5250]
```

**Step 2 — Positive phase statistics: outer product `v_data ⊗ h_data`.**
```
pos = v_data^T . h_data =
[[1x0.6225, 1x0.5250],
 [0x0.6225, 0x0.5250],
 [1x0.6225, 1x0.5250]]
=
[[0.6225, 0.5250],
 [0,      0     ],
 [0.6225, 0.5250]]
```

**Step 3 — Negative phase: reconstruct v from h_data (one Gibbs step), then recompute h from that reconstruction.**
```
v_recon ≈ [0.5875, 0.5584, 0.6348]    (sigmoid(a + W.h_data), same formula as Example 3)
h_recon ≈ [0.6087, 0.5567]             (sigmoid(b + v_recon.W), same formula as Example 2, using v_recon)
```

**Step 4 — Negative phase statistics: outer product `v_recon ⊗ h_recon`.**
```
neg ≈
[[0.5875x0.6087, 0.5875x0.5567],
 [0.5584x0.6087, 0.5584x0.5567],
 [0.6348x0.6087, 0.6348x0.5567]]
=
[[0.3576, 0.3271],
 [0.3399, 0.3109],
 [0.3864, 0.3534]]
```

**Step 5 — Compute the weight update: `ΔW = η × (pos - neg)`.**
```
pos - neg ≈
[[0.2649, 0.1979],
 [-0.3399, -0.3109],
 [0.2361, 0.1716]]

dW = 0.1 x (pos - neg) ≈
[[0.0265, 0.0198],
 [-0.0340, -0.0311],
 [0.0236, 0.0172]]
```

**Result:** weights connecting `v1` and `v3` (which were ACTUALLY ON in the real data, v_data=[1,0,1]) to both hidden units get pushed UP (positive updates), while weights connecting `v2` (which was OFF in the real data, but the model's reconstruction incorrectly assigned it a fairly high ~0.56 probability of being on) get pushed DOWN (negative updates) — the model is being corrected to more strongly favour the ACTUAL pattern (v1 and v3 on, v2 off) over its current, slightly-wrong reconstruction tendency.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Energy function | `E(v,h) = -a^Tv - b^Th - v^TWh` |
| Boltzmann distribution | `P(v,h) = exp(-E(v,h))/Z` |
| Hidden activation | `P(h_j=1\|v) = σ(b_j + Σ_i v_i·W_ij)` |
| Visible reconstruction | `P(v_i=1\|h) = σ(a_i + Σ_j W_ij·h_j)` |
| CD-k weight update | `ΔW_ij = η(⟨v_ih_j⟩_data - ⟨v_ih_j⟩_recon)` |
| CD-k bias updates | `Δa_i=η(v_i_data-v_i_recon)`, `Δb_j=η(h_j_data-h_j_recon)` |

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** computing the interaction term `v^TWh` incorrectly — always compute it as a sum over ALL (i,j) pairs, `Σ_i Σ_j v_i·W_ij·h_j`, not just a simple dot product of v and h directly.
- **Trap:** forgetting that hidden and visible activation formulas use DIFFERENT bias vectors — hidden uses b, visible uses a; mixing them up is a very common mistake.
- **Trap:** in CD-k, using the WRONG sign — the update is `data MINUS reconstruction`, not the other way around; the model should move TOWARD the real data's statistics.
- **Exam hack:** always show the pre-activation (`z`) as an explicit intermediate step before applying sigmoid — graders reward the visible intermediate calculation, not just the final probability.
- **Exam hack:** the CD-1 update's sign pattern (positive for units active in real data, negative for units the model reconstructs but shouldn't) is exactly analogous to Lecture 7's gradient descent direction — both push the model TOWARD lower error/energy on real data.

`[🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical)`

---

## Summary

This file worked every RBM formula from the theory file into fully shown arithmetic. Computing the energy of a specific configuration (v=[1,0,1], h=[1,1]) with given biases and weights gave E(v,h)=-1.1, breaking the calculation into its three component terms (visible bias, hidden bias, interaction). Hidden activation probabilities were computed independently for both hidden units (≈0.6225 and ≈0.5250), demonstrating the conditional-independence property that makes RBMs tractable, and the reverse visible-reconstruction computation produced [0.5987, 0.6225, 0.6900] using the same weight matrix in the opposite direction. A complete mean-field CD-1 weight update walked through both the positive phase (statistics from real data) and negative phase (statistics from a one-step Gibbs reconstruction), producing a full 3×2 weight update matrix that correctly pushes weights toward the real data's actual on/off pattern and away from the model's current (slightly incorrect) reconstruction tendency. The master formula table consolidates every reusable RBM calculation from this bonus lecture for fast review.

`[← Theory](../theory/dl_lecture14_rbm_theory.md) · [🔝 Top](#dl-lecture-14-bonus--restricted-boltzmann-machines-numerical) · [Next: Practice →](../practice/dl_lecture14_rbm_practice.md)`
