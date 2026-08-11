# 📖 Advanced AI — Lec 03: Risk Attitudes & Rational Choice Under Uncertainty — THEORY

### *Risk Neutral/Averse/Loving · Jensen's Inequality · The St. Petersburg Paradox · Rational Choice Under Uncertainty*

> **Nav:** [⬅️ Prev: Lec 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md) | [← Lec 03 README](README.md) | **THEORY** | [🧮 NUMERICAL](aai_lec03_risk_uncertainty_numerical.md) | [🎯 PRACTICE](aai_lec03_risk_uncertainty_practice.md)

---

## 🧠 MNEMONIC: **"RNA-Loves-Jensen → StP → RCU"**

> **R**isk **N**eutral, **A**verse, **Loving** → **J**ensen's Inequality (why they happen) → **St**.**P**etersburg paradox (why it matters) → **R**ational **C**hoice under **U**ncertainty (the formal finish line)

This lecture continues directly from Lec 02's Expected Utility (§5). Everything here is about answering one question: *given two options with the exact same average payoff, why would a rational person ever prefer one over the other?*

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Risk Attitudes — Neutral, Averse, Loving | [§1](#1-risk-attitudes--neutral-averse-loving) |
| 2 | Why It Happens — Jensen's Inequality & the Shape of `u` | [§2](#2-why-it-happens--jensens-inequality--the-shape-of-u) |
| 3 | The St. Petersburg Paradox | [§3](#3-the-st-petersburg-paradox) |
| 4 | Resolving the Paradox — Concave Utility to the Rescue | [§4](#4-resolving-the-paradox--concave-utility-to-the-rescue) |
| 5 | Rational Choice Under Uncertainty | [§5](#5-rational-choice-under-uncertainty) |
| 6 | Cheat Sheet & Exam Hacks | [§6](#6-cheat-sheet--exam-hacks) |

---

## 1. Risk Attitudes — Neutral, Averse, Loving

### 👶 Easy Story

Picture two offers with the exact same "fairness" on average: a coin flip between getting nothing and getting a big pile of candy, versus a guaranteed medium-sized pile that equals the *average* of the flip. Some kids would happily take the flip, hoping for the big pile. Other kids would rather just take the guaranteed medium pile and not risk ending up with nothing at all. Neither kid is wrong — they just feel differently about *uncertainty itself*, above and beyond the raw average outcome. That feeling is called a **risk attitude**, and this section gives it exact mathematical shape.

### The Setup

Recall from Lec 02 §5: `E[u(X)|Fₐ]` is the **expected utility** of a lottery — average the *utility* of each outcome, weighted by likelihood. This section introduces a second, different quantity: `u(E[X|Fₐ])` — first average the raw *outcomes* themselves (no utility applied yet), then apply `u` to that single average number.

These two quantities answer subtly different questions:
- `E[u(X)|Fₐ]` = "How good does the *gamble itself* feel, on average?"
- `u(E[X|Fₐ])` = "How good would it feel to just be *handed* the gamble's average payout for certain?"

### The Three Formal Definitions

```
┌──────────────────────────────────────────────────────────────────┐
│  RISK NEUTRAL:   E[u(X)|Fₐ]  =  u(E[X|Fₐ])                        │
│      "Averaging first or applying u first gives the SAME number." │
│                                                                    │
│  RISK AVERSE:    E[u(X)|Fₐ]  ≤  u(E[X|Fₐ])                        │
│      "The gamble feels WORSE than its own guaranteed average."    │
│                                                                    │
│  RISK LOVING:    E[u(X)|Fₐ]  ≥  u(E[X|Fₐ])                        │
│      "The gamble feels BETTER than its own guaranteed average."   │
└──────────────────────────────────────────────────────────────────┘
```

> 🍼 **Kid version:** Offer a coin flip between ₹0 and ₹4 (average = ₹2) versus a guaranteed ₹2. A **risk-neutral** person doesn't care which one they get — they're exactly indifferent. A **risk-averse** person picks the guaranteed ₹2 every time, because losing feels worse than winning feels good, even though both options pay the same on average. A **risk-loving** person actually prefers the coin flip — the thrill/upside of possibly getting more outweighs the risk of getting less.

### Why This Matters
Most real people and organizations are risk-averse for large stakes (insurance exists because of this — people gladly pay slightly more than the "fair" expected cost of an accident, just to avoid the uncertainty). This single distinction — comparing `E[u(X)]` against `u(E[X])` — is the entire mathematical foundation of insurance, diversification, and portfolio theory in economics and finance.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--theory)

---

## 2. Why It Happens — Jensen's Inequality & the Shape of `u`

### 👶 Easy Story

Draw any curve on paper connecting two dots. Now draw a straight line (a "chord") connecting those same two dots. If your curve bends downward like a frown (or a hill), the straight line sits *above* the curve everywhere in between. If your curve bends upward like a smile (a valley), the straight line sits *below* the curve. This simple geometric fact — called **Jensen's Inequality** — is the entire reason risk attitudes exist.

### The Geometry

```
   u(x)                                     u(x)
    |        ___----●  u(x̄)                  |                    ●  u(x̄)
    |    ___/                                 |                 __/
    |  _/         ← CONCAVE curve             |             ___/
    | /  ●  ← midpoint of the CHORD            |          __/
    |/   ↑ this is E[u(X)]                     |      ___/
    ●    (sits BELOW the curve)                |  ___/
    |    ↑ u(E[X]) sits ON the curve            | ●  ← CONVEX curve
    +------------------------- x               +------------------------- x
    RISK AVERSE: chord dips below the curve    RISK LOVING: chord sits above the curve
```

| Shape of `u` | Technical name | Risk attitude |
|---|---|---|
| Bends downward (like a hill/frown) | **Concave** | Risk **Averse** |
| Perfectly straight | **Linear** | Risk **Neutral** |
| Bends upward (like a valley/smile) | **Convex** | Risk **Loving** |

### Why Concave Means Risk-Averse (Plain Intuition)

A concave `u` means **each additional dollar helps you less than the previous dollar did** — this is called **diminishing marginal utility**. Going from ₹0 to ₹4 feels like a much bigger jump in happiness than going from ₹96 to ₹100, even though both are "+₹4." Because of this shrinking benefit, the pain of the low outcome in a gamble outweighs the joy of the high outcome by more than a straight-line calculation would suggest — so the average *utility* falls short of the utility of the *average* payout.

> 🍼 **Kid version:** Your very first scoop of ice cream is amazing. Your fifth scoop barely adds anything, and might even make your tummy hurt. That's diminishing returns — and it's exactly why a guaranteed medium amount of something often feels safer and better than gambling for a chance at a lot more.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--theory)

---

## 3. The St. Petersburg Paradox

### 👶 Easy Story

Imagine a magic game: flip a coin again and again until you finally get Tails. Whenever that first Tails shows up, your prize *doubles* for every extra flip it took to get there. Flip 1 Tails? You get $1. Flip 2? $2. Flip 3? $4. Flip 10? $512! Since the prize keeps doubling forever the longer you wait, ask yourself: **how much would you pay, right now, for a single ticket to play this game once?**

### The Formal Setup

A fair coin is tossed repeatedly until the first Tail appears. If the first Tail lands on toss `k`, the payoff is `2^(k-1)` dollars.

**Probability the first Tail lands exactly on toss `k`:** you need `k−1` Heads in a row, then a Tail:
```
P(first Tail on toss k) = (1/2)^(k-1) × (1/2) = (1/2)^k
```

```
                    TOSS 1
                 ┌───┴───┐
              Tail       Head → toss again
             ($1, p=1/2)      ┌───┴───┐
                            Tail       Head → toss again
                          ($2, p=1/4)      ┌───┴───┐
                                        Tail       Head → ...
                                      ($4, p=1/8)
```

### The Naive (Risk-Neutral) Expected Value

Using the discrete expected-utility formula from Lec 02 §5 with `u(x) = x` (the plain, risk-neutral case):
```
E[X] = Σ_{k=1}^∞  2^(k-1) · (1/2)^k
```
Simplify the general term: `2^(k-1) × (1/2)^k = 2^(k-1)/2^k = 2^(-1) = 1/2` — this is **exactly ½, no matter what `k` is**, because the doubling payoff and the halving probability cancel perfectly at every step.

```
E[X] = 1/2 + 1/2 + 1/2 + ...  (forever)  =  ∞
```

### The Paradox, Stated Plainly

A risk-neutral individual, who only cares about maximizing expected value, should logically be willing to pay **any finite amount of money**, no matter how large, for a single play of this game. But no sane person would actually pay more than a modest amount to play — creating a stark contradiction between the mathematical model and real human behavior. This is the "paradox": the theory (risk-neutral expected value) and observed reality (people pay very little) fly apart completely.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--theory)

---

## 4. Resolving the Paradox — Concave Utility to the Rescue

### 👶 Easy Story

The problem wasn't the coin, or the doubling rule — it was assuming that "twice the money" always means "twice the happiness." Real happiness doesn't work that way (remember the ice-cream-scoops idea from §2). If you use a utility function that grows *slower* than the raw dollar amount — a **concave**, risk-averse one — the runaway payoff gets tamed into a small, sensible number.

### The Fix: `u(x) = ln(x)`

Historically, Daniel Bernoulli proposed exactly this fix in the 1700s. Recompute expected utility using `u(x) = ln(x)` instead of `u(x) = x`:
```
E[u(X)] = Σ_{k=1}^∞  ln(2^{k-1}) · (1/2)^k
        = ln(2) × Σ_{k=1}^∞  (k-1)/2^k
        = ln(2) × 1
        = ln(2) ≈ 0.693      ← FINITE!
```
(The full step-by-step evaluation of that infinite sum is in the [Numerical file, §2](aai_lec03_risk_uncertainty_numerical.md#2-the-st-petersburg-paradox--full-derivation).)

### Converting Back to a Dollar Amount — the Certainty Equivalent

`E[u(X)] = ln(2)` is a utility number, not a dollar figure. The **certainty equivalent (CE)** is the guaranteed dollar amount that would give the exact same utility:
```
u(CE) = E[u(X)]
ln(CE) = ln(2)
CE = 2
```
✅ **A rational, log-utility, risk-averse individual would pay at most $2 to play this game** — small, sensible, matching real human behavior.

### A Warning — Not Every Fix Works

It's tempting to think "just apply *some* utility function and the paradox goes away." Not true. Testing `u(x) = x²` (a **convex**, risk-*loving* function) actually makes things worse:
```
E[u(X)] = Σ_{k=1}^∞ (2^{k-1})² · (1/2)^k = Σ_{k=1}^∞ 2^{k-2} = 0.5 + 1 + 2 + 4 + ...  =  ∞
```
Still infinite — because a convex function rewards big rare payoffs *even more* than raw dollars do, which only accelerates the same runaway problem.

```
╔══════════════════════════════════════════════════════════════════╗
║  u(x) = x   (linear, risk-neutral)   →  E[u(X)] = ∞   (paradox)   ║
║  u(x) = x²  (convex, risk-loving)    →  E[u(X)] = ∞   (worse!)    ║
║  u(x) = ln(x) (concave, risk-averse) →  E[u(X)] = ln(2) → CE = $2 ║
╚══════════════════════════════════════════════════════════════════╝
```

**Lesson:** resolving the St. Petersburg paradox specifically requires a **concave** utility function strong enough to counteract exponential payoff growth — not just "any" nonlinear function.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--theory)

---

## 5. Rational Choice Under Uncertainty

### 👶 Easy Story

This closes the loop all the way back to Lec 01. Back then, a rational individual just picked the action with the biggest guaranteed payoff. Now that every action can trigger a lottery instead of a certain outcome, the rule barely changes — you just swap "guaranteed payoff" for "*average* payoff," and everything else survives untouched.

### The Rational Choice Assumption, Updated for Uncertainty

Compared to Lec 01's four "must-knows," there's exactly **one new requirement**:

```
┌──────────────────────────────────────────────────────────────┐
│  RATIONAL CHOICE ASSUMPTION (under uncertainty)                │
├──────────────────────────────────────────────────────────────┤
│  1. Knows all actions A                    (same as Lec 01)    │
│  2. Knows all outcomes X                   (same as Lec 01)    │
│  3. 🆕 Knows which lottery Fₐ Nature uses,                     │
│        for EVERY possible action a                             │
│  4. Knows their own payoff function u over X (same as Lec 01)  │
└──────────────────────────────────────────────────────────────┘
```
The individual must know exactly *which* lottery each action triggers — not merely that some randomness is involved, but the full `Fₐ` (or `p|ₐ`) for every single `a ∈ A`.

### The Rational Individual — Choice Rule Under Uncertainty

An individual is rational if they satisfy the assumption above, and:
```
Choose a* ∈ A   if and only if   E[u(X)|F_{a*}] ≥ E[u(X)|Fₐ]   for all a ∈ A
```

### Side-by-Side with the Certainty Version (Lec 01)

```
┌─────────────────────────────────────────────────────────────────┐
│  NO UNCERTAINTY (Lec 01)     │  WITH UNCERTAINTY (this lecture)   │
├─────────────────────────────────────────────────────────────────┤
│  v(a) = u(x*(a))             │  v(a) = E[u(X)|Fₐ]                 │
│  "the ONE payoff you get"    │  "the AVERAGE payoff you'd get"    │
│                               │                                    │
│  Pick a* with                 │  Pick a* with                      │
│  v(a*) ≥ v(a)  ∀ a ∈ A       │  E[u(X)|F_{a*}] ≥ E[u(X)|Fₐ]  ∀ a  │
└─────────────────────────────────────────────────────────────────┘
```

Literally the same "pick the biggest number, compared across every option" rule from Lec 01 — the only thing that changed is *how you compute the number you're maximizing.*

> 🍼 **Kid version:** Before, you knew exactly what candy you'd get from each jar. Now, some jars are actually grab-bags — you don't know exactly which candy you'll pull out, only the odds. The rule is still "pick the jar you like best" — you just now have to think about it as "pick the grab-bag with the best *average* candy," instead of "pick the jar with the best guaranteed candy."

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--theory)

---

## 6. Cheat Sheet & Exam Hacks

```
╔══════════════════════════════════════════════════════════════════╗
║  ADVANCED AI — LEC 03 ONE-LINERS                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Risk Neutral:  E[u(X)|Fₐ] = u(E[X|Fₐ])   → u is LINEAR             ║
║  Risk Averse:   E[u(X)|Fₐ] ≤ u(E[X|Fₐ])   → u is CONCAVE             ║
║  Risk Loving:   E[u(X)|Fₐ] ≥ u(E[X|Fₐ])   → u is CONVEX              ║
║                                                                     ║
║  Jensen's Inequality: chord vs curve — concave→chord below curve,   ║
║                        convex→chord above curve                     ║
║                                                                     ║
║  St. Petersburg game: payoff 2^(k-1) at toss k, prob (1/2)^k        ║
║  E[X] with u(x)=x    → ∞          (the paradox)                     ║
║  E[u(X)] with u=ln(x) → ln(2)     → certainty equivalent = $2        ║
║  E[u(X)] with u=x²    → ∞          (convex makes it WORSE)           ║
║                                                                     ║
║  Rational choice under uncertainty:                                 ║
║    Pick a* with E[u(X)|F_{a*}] ≥ E[u(X)|Fₐ]  for all a ∈ A          ║
║    (new "must-know": Fₐ for every action a)                         ║
╚══════════════════════════════════════════════════════════════════╝
```

### ⚡ Exam Red Flags

1. **"Write the three risk-attitude inequalities."** → Risk-neutral uses `=`, risk-averse uses `≤` (E[u(X)] on the left, smaller), risk-loving uses `≥`. The direction of the inequality is the #1 thing students mix up — remember: risk-averse dislikes the gamble, so the gamble's *expected utility* is the smaller (≤) side.
2. **"What geometric fact explains risk aversion?"** → Jensen's Inequality: for a concave function, the chord between two points lies below the curve, so averaging outcomes-then-applying-u beats applying-u-then-averaging.
3. **"State the St. Petersburg paradox and its core arithmetic trick."** → Payoff doubles (`2^(k-1)`) exactly as fast as probability halves (`(1/2)^k`), so every term in the expected-value sum is a constant `1/2` — summing infinitely many of those diverges.
4. **"Which utility function resolves the paradox, and why does x² fail?"** → `u(x)=ln(x)` works because it's concave enough to tame the exponential growth; `u(x)=x²` fails (makes it worse) because it's convex and *amplifies* large payoffs rather than discounting them.
5. **"What's the ONE new item added to the Rational Choice Assumption for uncertainty?"** → Knowing which lottery/distribution `Fₐ` Nature uses for every action `a` — not just that randomness exists, but its exact shape for each action.
6. **"Write the rational choice rule under uncertainty."** → `E[u(X)|F_{a*}] ≥ E[u(X)|Fₐ]` for all `a ∈ A`. Same structure as Lec 01's `v(a*) ≥ v(a)`, just with expected utility replacing certain payoff.

[↑ Back to Top](#-advanced-ai--lec-03-risk-attitudes--rational-choice-under-uncertainty--theory)

---

## 📝 Summary

This lecture answers a question Lecture 02 quietly left open: if two options have the exact same expected payoff, does a rational person truly not care which one they get? The answer is a firm "not necessarily," and the whole lecture explains why. Risk attitudes — neutral, averse, loving — turn out to be entirely a matter of the *shape* of someone's utility function, and Jensen's Inequality supplies the clean geometric reason: a concave curve always keeps the straight-line "average of two points" below the curve itself, which is exactly why risk-averse people rate a gamble lower than its own guaranteed average. The St. Petersburg Paradox then delivered a genuine shock: a simple doubling coin-flip game has an infinite "fair price" under plain expected value, yet no real person would pay more than a few dollars to play — a flat contradiction between the math and observed behavior. Fixing it required a concave utility function strong enough to tame the exponential payoff growth (`ln(x)` works, giving a sensible $2 price), while a convex one (`x²`) only makes the paradox worse. The lecture closes by folding all of this back into Lecture 01's original choice rule, upgraded so the individual now must also know the exact lottery each action triggers, and picks whichever action has the highest *expected* utility rather than the highest certain one.

---

> **Next:** [🧮 NUMERICAL →](aai_lec03_risk_uncertainty_numerical.md) · [🎯 PRACTICE →](aai_lec03_risk_uncertainty_practice.md)
>
> *Advanced AI · Lec 03 · github.com/rpaut03l/TS-02-03*
