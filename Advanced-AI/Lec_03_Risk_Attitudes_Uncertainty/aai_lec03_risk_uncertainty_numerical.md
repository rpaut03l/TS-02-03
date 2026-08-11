# 🧮 Advanced AI — Lec 03: Risk Attitudes & Rational Choice Under Uncertainty — NUMERICAL

### *Every worked example, every step, every number verified*

> **Nav:** [⬅️ Prev: Lec 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md) | [← Lec 03 README](README.md) | [📖 THEORY](aai_lec03_risk_uncertainty_theory.md) | **NUMERICAL** | [🎯 PRACTICE](aai_lec03_risk_uncertainty_practice.md)

---

## 📚 Table of Contents

| # | Worked Example | Jump |
|---|---|---|
| 1 | Risk Attitudes — Three Utility Functions, Same Gamble | [§1](#1-risk-attitudes--three-utility-functions-same-gamble) |
| 2 | The St. Petersburg Paradox — Full Derivation | [§2](#2-the-st-petersburg-paradox--full-derivation) |
| 3 | Resolving the Paradox — Log Utility vs Squared Utility | [§3](#3-resolving-the-paradox--log-utility-vs-squared-utility) |
| 4 | Rational Choice Under Uncertainty — Worked Comparison | [§4](#4-rational-choice-under-uncertainty--worked-comparison) |
| 5 | Formula Cheat Sheet | [§5](#5-formula-cheat-sheet) |

---

## 1. Risk Attitudes — Three Utility Functions, Same Gamble

### The Setup

A single lottery: `X ∈ {0, 4}`, each with probability `1/2`. Its plain average payout never changes, no matter which `u` we test:
```
E[X|Fₐ] = (0)(0.5) + (4)(0.5) = 0 + 2 = 2
```

We'll test three different utility functions against this same gamble, to see how each one classifies its own risk attitude.

### Case 1 — `u(x) = x` (testing for Risk Neutral)

**Step 1 — compute `u(E[X|Fₐ])` (utility of the average):**
```
u(E[X|Fₐ]) = u(2) = 2
```

**Step 2 — compute `E[u(X)|Fₐ]` (average of the utilities):**
```
E[u(X)|Fₐ] = u(0)·(0.5) + u(4)·(0.5) = 0·(0.5) + 4·(0.5) = 0 + 2 = 2
```

**Step 3 — compare:**
```
E[u(X)|Fₐ] = 2   =   u(E[X|Fₐ]) = 2
```
✅ **Equal → confirms Risk Neutral.**

### Case 2 — `u(x) = √x` (testing for Risk Averse)

**Step 1:**
```
u(E[X|Fₐ]) = u(2) = √2 ≈ 1.414
```

**Step 2:**
```
E[u(X)|Fₐ] = √0·(0.5) + √4·(0.5) = 0·(0.5) + 2·(0.5) = 0 + 1 = 1
```

**Step 3 — compare:**
```
E[u(X)|Fₐ] = 1   ≤   u(E[X|Fₐ]) ≈ 1.414
```
✅ **`E[u(X)] < u(E[X])` → confirms Risk Averse.** The gamble feels worse (utility 1) than being simply handed the guaranteed average of 2 (utility ≈1.414).

### Case 3 — `u(x) = x²` (testing for Risk Loving)

**Step 1:**
```
u(E[X|Fₐ]) = u(2) = 2² = 4
```

**Step 2:**
```
E[u(X)|Fₐ] = 0²·(0.5) + 4²·(0.5) = 0·(0.5) + 16·(0.5) = 0 + 8 = 8
```

**Step 3 — compare:**
```
E[u(X)|Fₐ] = 8   ≥   u(E[X|Fₐ]) = 4
```
✅ **`E[u(X)] > u(E[X])` → confirms Risk Loving.** The gamble feels better (utility 8) than being handed the guaranteed average (utility 4).

### Summary Table

| `u(x)` | `E[u(X)]` | `u(E[X])` | Comparison | Attitude |
|---|---|---|---|---|
| `x` | 2 | 2 | `=` | Risk Neutral |
| `√x` | 1 | 1.414 | `≤` | Risk Averse |
| `x²` | 8 | 4 | `≥` | Risk Loving |

The **shape** of `u` alone determines the risk attitude — nothing about the gamble itself changed across all three cases.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--numerical)

---

## 2. The St. Petersburg Paradox — Full Derivation

### The Setup

Fair coin, tossed until the first Tail. First Tail on toss `k` → payoff `2^(k-1)` dollars, probability `(1/2)^k`.

### Step 1 — Build the table for the first several tosses

| Toss `k` | Payoff `2^(k-1)` | Probability `(1/2)^k` | Product |
|---|---|---|---|
| 1 | $1 | 1/2 | 1/2 |
| 2 | $2 | 1/4 | 1/2 |
| 3 | $4 | 1/8 | 1/2 |
| 4 | $8 | 1/16 | 1/2 |
| 5 | $16 | 1/32 | 1/2 |

### Step 2 — Prove algebraically that every term equals exactly `1/2`

```
2^(k-1) × (1/2)^k
= 2^(k-1) / 2^k                        [rewrite (1/2)^k as 1/2^k]
= 2^((k-1) - k)                        [subtract exponents, same base]
= 2^(-1)
= 1/2
```
This holds for **every** value of `k` — the doubling payoff and halving probability cancel perfectly, every single time.

### Step 3 — Sum infinitely many copies of `1/2`

```
E[X] = Σ_{k=1}^∞ (1/2) = 1/2 + 1/2 + 1/2 + ...
```
A constant positive number, added infinitely many times, has no finite total:
```
after n terms: n/2  →  as n → ∞, this → ∞
```
```
E[X] = ∞
```

### The Paradox

Under plain expected value (`u(x)=x`, risk-neutral), a rational person should be willing to pay **any finite amount** to play this game once. Real people pay only a modest amount — a stark mismatch between the model and observed behavior.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--numerical)

---

## 3. Resolving the Paradox — Log Utility vs Squared Utility

### Case A — `u(x) = ln(x)` (concave, risk-averse — the historical fix)

**Step 1 — write the general term:**
```
u(payoff) × (probability) = ln(2^{k-1}) × (1/2)^k
```

**Step 2 — apply the log power rule `ln(a^b) = b·ln(a)`:**
```
ln(2^{k-1}) = (k-1)·ln(2)
```
So the term becomes: `(k-1)·ln(2) × (1/2)^k`

**Step 3 — factor the constant `ln(2)` out of the whole sum:**
```
E[u(X)] = ln(2) × Σ_{k=1}^∞ (k-1)·(1/2)^k
```
Call the remaining sum `S = Σ_{k=1}^∞ (k-1)·(1/2)^k`.

**Step 4 — split `(k-1)` into `k − 1`, breaking `S` into two known geometric series:**
```
S = Σ k·(1/2)^k  −  Σ (1/2)^k  = S₁ − S₂
```

**Step 5 — evaluate `S₂` (standard geometric series, `Σ_{k=1}^∞ r^k = r/(1-r)`, with `r=1/2`):**
```
S₂ = (1/2)/(1 - 1/2) = (1/2)/(1/2) = 1
```

**Step 6 — evaluate `S₁` (weighted geometric series, `Σ_{k=1}^∞ k·r^k = r/(1-r)²`, with `r=1/2`):**
```
S₁ = (1/2)/(1-1/2)² = (1/2)/(1/4) = 2
```

**Step 7 — combine:**
```
S = S₁ − S₂ = 2 − 1 = 1
```

**Step 8 — plug back in:**
```
E[u(X)] = ln(2) × 1 = ln(2) ≈ 0.693     ← FINITE
```

**Step 9 — convert to a certainty equivalent (the dollar-value fair price):**
```
u(CE) = E[u(X)]
ln(CE) = ln(2)
CE = 2
```
✅ **A log-utility individual would pay at most $2 to play.**

### Case B — `u(x) = x²` (convex, risk-loving — testing whether ANY nonlinear u works)

**Step 1 — write the general term:**
```
(2^{k-1})² × (1/2)^k
```

**Step 2 — simplify `(2^{k-1})²` using `(a^b)^c = a^{bc}`:**
```
(2^{k-1})² = 2^{2(k-1)} = 2^{2k-2}
```

**Step 3 — combine with `(1/2)^k = 1/2^k`:**
```
2^{2k-2}/2^k = 2^{(2k-2)-k} = 2^{k-2}
```

**Step 4 — inspect the terms as `k` grows:**

| `k` | Term `2^{k-2}` |
|---|---|
| 1 | 0.5 |
| 2 | 1 |
| 3 | 2 |
| 4 | 4 |
| 5 | 8 |

The terms **double every step** — they never shrink at all.

**Step 5 — conclude:**
```
E[u(X)] = Σ_{k=1}^∞ 2^{k-2} = 0.5 + 1 + 2 + 4 + 8 + ... = ∞
```
❌ **Still infinite — actually diverges even "faster" in spirit than the original risk-neutral case, because a convex function rewards the enormous rare payoffs even more heavily than raw dollars would.**

### Full Comparison Table

| `u(x)` | Shape | `E[u(X)]` | Certainty Equivalent | Resolves paradox? |
|---|---|---|---|---|
| `x` | Linear | ∞ | undefined (∞) | ❌ No — this IS the paradox |
| `x²` | Convex | ∞ | undefined (∞) | ❌ No — makes it worse |
| `ln(x)` | Concave | `ln(2) ≈ 0.693` | **$2** | ✅ Yes |

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--numerical)

---

## 4. Rational Choice Under Uncertainty — Worked Comparison

### The Setup

Recall the movie-theatre example from Lec 02. Now make "Go to PVR" risky: there's a 70% chance you get your first choice (Aliens, payoff 6) and a 30% chance the show is sold out and you're stuck watching The Matrix instead (payoff 1). Compare against the safe, guaranteed option "Go to Inox, watch Casablanca" (payoff 4, no randomness at all — a lottery placing probability 1 on a single outcome).

### Step 1 — compute expected utility of the risky option (PVR)

```
E[u(X)|F_PVR] = u(6)·(0.7) + u(1)·(0.3)
              = 6(0.7) + 1(0.3)
              = 4.2 + 0.3
              = 4.5
```

### Step 2 — compute expected utility of the safe option (Inox)

Since Inox is guaranteed (probability 1 on Casablanca, payoff 4):
```
E[u(X)|F_Inox] = u(4)·(1.0) = 4.0
```

### Step 3 — apply the rational choice rule

```
E[u(X)|F_PVR] = 4.5   ≥   E[u(X)|F_Inox] = 4.0
```
✅ **Choose PVR**, despite the sold-out risk — its expected payoff still edges out the guaranteed alternative.

### A twist — what if the sold-out probability were higher?

Suppose instead PVR only has a 50% chance of showing Aliens:
```
E[u(X)|F_PVR] = 6(0.5) + 1(0.5) = 3.0 + 0.5 = 3.5
```
```
E[u(X)|F_PVR] = 3.5   <   E[u(X)|F_Inox] = 4.0
```
🔄 **The choice flips — now Inox is optimal.** This is the same "the exact numbers matter, not just the structure" lesson from Lec 02's Numerical §3, now applied to a full rational-choice-under-uncertainty comparison instead of a bare lottery comparison.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--numerical)

---

## 5. Formula Cheat Sheet

```
╔════════════════════════════════════════════════════════════════════╗
║  FORMULAS — COPY THESE DOWN BEFORE THE EXAM                        ║
╠════════════════════════════════════════════════════════════════════╣
║  Risk Neutral:  E[u(X)|Fₐ] = u(E[X|Fₐ])                            ║
║  Risk Averse:   E[u(X)|Fₐ] ≤ u(E[X|Fₐ])                            ║
║  Risk Loving:   E[u(X)|Fₐ] ≥ u(E[X|Fₐ])                            ║
║                                                                    ║
║  St. Petersburg general term: 2^(k-1) · (1/2)^k = 1/2  (always!)   ║
║  Geometric series:      Σ_{k=1}^∞ r^k   = r/(1-r)                  ║
║  Weighted geometric:    Σ_{k=1}^∞ k·r^k = r/(1-r)²                 ║
║                                                                    ║
║  Certainty equivalent:  u(CE) = E[u(X)]  →  solve for CE           ║
║                                                                    ║
║  Rational choice under uncertainty:                                ║
║    choose a* with E[u(X)|F_{a*}] ≥ E[u(X)|Fₐ]  for all a ∈ A       ║
╚════════════════════════════════════════════════════════════════════╝
```

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--numerical)

---

## 📝 Summary

Every worked example in this file circles back to the same core comparison — `E[u(X)]` versus `u(E[X])` — and shows how just changing the shape of `u` flips which side wins. Testing the same 50/50 gamble against three different utility functions (§1) confirmed the pattern exactly as the theory predicted: linear gives equality, a square-root curve gives risk aversion, and squaring gives risk-loving behavior, with nothing about the gamble itself ever changing. The St. Petersburg derivation (§2) is the file's centerpiece — proving, term by term, that every single element of the sum equals exactly 1/2 was the key insight that made the infinite divergence undeniable rather than just asserted. Section 3 then earned its keep by testing *two* different fixes side by side: `ln(x)` tames the paradox down to a tidy $2 certainty equivalent, while `x²` was shown to make things actively worse, proving that "any nonlinear function" is not a valid shortcut — the concavity has to be strong enough to beat the specific growth rate of the payoffs involved. The rational-choice-under-uncertainty example (§4) closed the loop by showing the exact same "recompute and watch it flip" behavior from Lecture 02, now applied to a full decision between a risky and a safe action rather than a bare lottery comparison.

---

> **Next:** [🎯 PRACTICE →](aai_lec03_risk_uncertainty_practice.md) · [← back to THEORY](aai_lec03_risk_uncertainty_theory.md)
>
> *Advanced AI · Lec 03 · github.com/rpaut03l/TS-02-03*
