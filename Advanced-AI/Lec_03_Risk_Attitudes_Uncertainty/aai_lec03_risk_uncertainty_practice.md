# 🎯 Advanced AI — Lec 03: Risk Attitudes & Rational Choice Under Uncertainty — PRACTICE

### *Self-test problems, a Q&A bank, and a mini project — try before you peek*

> **Nav:** [⬅️ Prev: Lec 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md) | [← Lec 03 README](README.md) | [📖 THEORY](aai_lec03_risk_uncertainty_theory.md) | [🧮 NUMERICAL](aai_lec03_risk_uncertainty_numerical.md) | **PRACTICE**

---

## 📚 Table of Contents

| # | Section | Jump |
|---|---|---|
| 1 | Risk Attitude Concept Checks | [§1](#1-risk-attitude-concept-checks) |
| 2 | Risk Attitude Calculation Drills | [§2](#2-risk-attitude-calculation-drills) |
| 3 | St. Petersburg Paradox Drills | [§3](#3-st-petersburg-paradox-drills) |
| 4 | Rational Choice Under Uncertainty Practice | [§4](#4-rational-choice-under-uncertainty-practice) |
| 5 | Mini Project — Test Your Own Utility Function | [§5](#5-mini-project--test-your-own-utility-function) |
| 6 | Exam-Style Q&A Bank | [§6](#6-exam-style-qa-bank) |

> 💡 Every problem below is inside a `<details>` block. **Try to solve it on paper first**, then click "Show Answer" to check yourself.

---

## 1. Risk Attitude Concept Checks

**Q1.1.** A friend says: "I'd rather have a guaranteed ₹500 than a coin flip between ₹0 and ₹1000, even though they're worth the same on average." What risk attitude does this describe?

<details>
<summary>Show Answer</summary>

**Risk Averse.** They prefer the sure amount over a gamble with the identical expected value — the defining behavior of `E[u(X)] ≤ u(E[X])`.
</details>

---

**Q1.2.** True or False: a risk-loving individual's utility function must be concave.

<details>
<summary>Show Answer</summary>

**False.** Risk-loving corresponds to a **convex** utility function (curves upward), not concave. Concave corresponds to risk-*averse*.
</details>

---

**Q1.3.** What geometric fact (from Jensen's Inequality) explains why a concave utility function produces risk aversion?

<details>
<summary>Show Answer</summary>

For a concave curve, the straight chord connecting any two points on the curve lies **below** the curve everywhere in between. Since `E[u(X)]` is the height of that chord's midpoint, and `u(E[X])` is the curve's actual height at that same x-value, the chord being below the curve directly gives `E[u(X)] ≤ u(E[X])` — exactly the risk-averse inequality.
</details>

---

## 2. Risk Attitude Calculation Drills

**Q2.1.** `X ∈ {0, 9}`, each with probability 1/2. Using `u(x) = √x`, compute both `E[u(X)]` and `u(E[X])`, and state the risk attitude this confirms.

<details>
<summary>Show Answer</summary>

```
E[X] = (0)(0.5) + (9)(0.5) = 4.5
u(E[X]) = √4.5 ≈ 2.121

E[u(X)] = √0(0.5) + √9(0.5) = 0(0.5) + 3(0.5) = 1.5
```
`E[u(X)] = 1.5 ≤ u(E[X]) ≈ 2.121` → **Risk Averse.**
</details>

---

**Q2.2.** Same lottery as Q2.1 (`X ∈ {0,9}`, 50/50), but now with `u(x) = x³`. Compute both sides and state the attitude.

<details>
<summary>Show Answer</summary>

```
u(E[X]) = u(4.5) = 4.5³ = 91.125

E[u(X)] = 0³(0.5) + 9³(0.5) = 0(0.5) + 729(0.5) = 364.5
```
`E[u(X)] = 364.5 ≥ u(E[X]) = 91.125` → **Risk Loving.**
</details>

---

## 3. St. Petersburg Paradox Drills

**Q3.1.** In the St. Petersburg game, what is the probability that the first Tail lands exactly on toss 5? What is the payoff if it does?

<details>
<summary>Show Answer</summary>

```
Probability = (1/2)^5 = 1/32
Payoff = 2^(5-1) = 2^4 = $16
```
</details>

---

**Q3.2.** Verify by direct multiplication that the "payoff × probability" term for toss 5 equals 1/2, matching every other toss.

<details>
<summary>Show Answer</summary>

```
16 × (1/32) = 16/32 = 1/2   ✓
```
Confirms the general pattern: every single term in the expected-value sum equals exactly 1/2, regardless of `k`.
</details>

---

**Q3.3.** If a casino offered a *capped* version of the St. Petersburg game — the payoff maxes out at $1,024 no matter how many tosses it takes to get the first Tail — would the expected value still be infinite? (Conceptual, no full derivation needed — just reason about it.)

<details>
<summary>Show Answer</summary>

**No.** Capping the payoff means only a *finite* number of terms (up to the toss number where `2^(k-1)` first hits the cap) contribute their full uncapped value; every toss beyond that point contributes a shrinking `(capped value) × (shrinking probability)`, and — critically — there are no longer infinitely many "constant 1/2" terms, since the payoff stops doubling once it's capped. The sum becomes a finite number. This is actually how real casinos and insurers handle this problem in practice: capping maximum payouts avoids the paradox entirely.
</details>

---

## 4. Rational Choice Under Uncertainty Practice

**Q4.1.** Action A gives a 60% chance of payoff 10 and a 40% chance of payoff 2. Action B gives a guaranteed payoff of 7. Using `u(x) = x`, which action does a rational individual choose?

<details>
<summary>Show Answer</summary>

```
E[u(X)|A] = 10(0.6) + 2(0.4) = 6 + 0.8 = 6.8
E[u(X)|B] = 7(1.0) = 7
```
`6.8 < 7` → **Choose B** (the guaranteed option).
</details>

---

**Q4.2.** Same two actions as Q4.1, but now the individual is risk-loving with `u(x) = x²`. Recompute and see if the choice changes.

<details>
<summary>Show Answer</summary>

```
E[u(X)|A] = 10²(0.6) + 2²(0.4) = 100(0.6) + 4(0.4) = 60 + 1.6 = 61.6
E[u(X)|B] = 7²(1.0) = 49
```
`61.6 > 49` → **Choice flips to A!** This shows how risk attitude — not just the raw numbers — can change the rational decision, even when the underlying lottery and guaranteed alternative haven't changed at all.
</details>

---

## 5. Mini Project — Test Your Own Utility Function

No spoiler answer here — genuinely yours to build.

**Task:**
1. Invent your own simple lottery `X` with at least 2 outcomes and assign probabilities that sum to 1.
2. Pick three different utility functions to test against it: one linear (like `u(x)=x`), one concave (like `u(x)=√x` or `u(x)=ln(x)`), and one convex (like `u(x)=x²`).
3. For each of the three, compute both `E[u(X)]` and `u(E[X])`, and classify the risk attitude each one implies.
4. Now design your own version of the St. Petersburg game — pick a different growth rate for the payoff (e.g., payoff triples instead of doubles each round) and a different coin-bias if you like. Determine: does `E[X]` still diverge? Does `u(x)=ln(x)` still tame it to a finite number? (Hint: compare the payoff's growth rate against `(1/2)^k`'s shrink rate — if the payoff grows *faster* than 2x per step, does `ln` still catch up in time?)

---

## 6. Exam-Style Q&A Bank

<details>
<summary><b>Q. Write the three risk-attitude inequalities.</b></summary>

Risk Neutral: `E[u(X)|Fₐ] = u(E[X|Fₐ])`. Risk Averse: `E[u(X)|Fₐ] ≤ u(E[X|Fₐ])`. Risk Loving: `E[u(X)|Fₐ] ≥ u(E[X|Fₐ])`.
</details>

<details>
<summary><b>Q. What shape of u corresponds to each risk attitude?</b></summary>

Linear → risk neutral. Concave → risk averse. Convex → risk loving.
</details>

<details>
<summary><b>Q. State the St. Petersburg game's rules in one sentence.</b></summary>

A fair coin is tossed until the first Tail; if it lands on toss `k`, the payoff is `2^(k-1)` dollars.
</details>

<details>
<summary><b>Q. Why does E[X] diverge in the St. Petersburg game?</b></summary>

Every term in the expected-value sum, `2^(k-1)·(1/2)^k`, simplifies to exactly `1/2` regardless of `k`, because the doubling payoff exactly cancels the halving probability. Summing infinitely many copies of a positive constant diverges to infinity.
</details>

<details>
<summary><b>Q. Which utility function resolves the paradox, and what is the resulting certainty equivalent?</b></summary>

`u(x) = ln(x)`. It gives `E[u(X)] = ln(2)`, and solving `ln(CE) = ln(2)` gives a certainty equivalent of `CE = $2`.
</details>

<details>
<summary><b>Q. Does every nonlinear utility function fix the St. Petersburg paradox? Explain.</b></summary>

No. Convex functions like `u(x) = x²` make it worse — they reward large payoffs even more than raw dollars, so `E[u(X)]` still diverges to infinity. Only a sufficiently concave function (like `ln(x)`) tames the exponential payoff growth.
</details>

<details>
<summary><b>Q. What is the one new item added to the Rational Choice Assumption for decisions under uncertainty?</b></summary>

Knowing which probability distribution (lottery) `Fₐ` Nature will use as a consequence of each specific action `a` — not merely that randomness exists, but the exact `Fₐ` for every `a ∈ A`.
</details>

<details>
<summary><b>Q. Write the rational choice rule under uncertainty, and compare it to the certainty version from Lecture 1.</b></summary>

Under uncertainty: choose `a*` with `E[u(X)|F_{a*}] ≥ E[u(X)|Fₐ]` for all `a ∈ A`. Under certainty (Lecture 1): choose `a*` with `v(a*) = u(x*(a*)) ≥ v(a)` for all `a ∈ A`. The structure — "pick the biggest number across all options" — is identical; only the definition of "the number" changes from a certain payoff to an expected one.
</details>

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--practice)

---

## 📝 Summary

The risk-attitude drills should have left you able to instantly recognize the pattern: same lottery, three different utility function shapes, three different verdicts, with the direction of the inequality (`≤` for averse, `≥` for loving) being the one detail worth triple-checking on any exam. The St. Petersburg drills pushed further into the mechanics than the main derivation did — computing an individual term by hand, verifying it matches the general formula, and then reasoning conceptually about what capping the payoff would do (namely, breaking the "every term equals 1/2" trick that caused the divergence in the first place). The rational-choice-under-uncertainty drills reinforced the sharpest lesson of this whole lecture: risk attitude isn't just theoretical decoration, it can genuinely change which real-world action counts as "rational" — the exact same lottery-versus-guaranteed-payoff comparison flipped its answer purely because the utility function changed from linear to convex. The mini project, extending the St. Petersburg game with a different growth rate, exists specifically to test whether you understand *why* `ln(x)` worked — if you change the payoff's growth rate, does the logarithm still catch up in time to keep the sum finite? Chasing that question down is the best possible proof that the concept, not just the specific numbers, has actually landed.

---

> **← Back:** [🧮 NUMERICAL](aai_lec03_risk_uncertainty_numerical.md) · [📖 THEORY](aai_lec03_risk_uncertainty_theory.md) · [🏠 Lec 03 README](README.md) · [⬅️ Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)
>
> *Advanced AI · Lec 03 · github.com/rpaut03l/TS-02-03*
