# 📖 Advanced AI — Lec 02: Decision Trees, Risk & Lotteries — THEORY

### *Decision Trees · Nature · Simple & Compound Lotteries · Expected Utility*

> **Nav:** [⬅️ Prev: Lec 01](../Lec_01_Individual_Decision_Problem/README.md) | [← Lec 02 README](README.md) | **THEORY** | [🧮 NUMERICAL](aai_lec02_decision_trees_lotteries_numerical.md) | [🎯 PRACTICE](aai_lec02_decision_trees_lotteries_practice.md)

---

## 🧠 MNEMONIC: **"DT → RNG → CL → EU"**

> **D**ecision **T**rees → **R**isk / **N**ature / **G**ambles (simple lotteries) → **C**ompound **L**otteries → **E**xpected **U**tility

This continues directly from [Lecture 01](../Lec_01_Individual_Decision_Problem/README.md)'s **"AXP → CTR → UV"** chant. Lec 01 built a rational individual who always knows *exactly* what an action gets them. This lecture removes that certainty — now **Nature** (an imaginary extra "player") rolls dice to decide the outcome, and you'll learn how a rational individual still picks the best action anyway.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | Decision Trees | [§1](#1-decision-trees) |
| 2 | Risk, Nature & Random Outcomes — Simple Lotteries | [§2](#2-risk-nature--random-outcomes--simple-lotteries) |
| 3 | Continuous Outcomes — Cumulative Distribution Functions | [§3](#3-continuous-outcomes--cumulative-distribution-functions) |
| 4 | Compound Lotteries | [§4](#4-compound-lotteries) |
| 5 | Evaluating Lotteries — Expected Utility | [§5](#5-evaluating-lotteries--expected-utility) |
| 6 | Cheat Sheet & Exam Hacks | [§6](#6-cheat-sheet--exam-hacks) |

---

## 1. Decision Trees

### 👶 Easy Story

A decision tree is just a **map of "if I go this way, then that way, I end up here"** — drawn out so you can literally see every path and its final payoff.

### Anatomy of a Decision Tree

```
┌────────────────────────────────────────────────────────────┐
│  DECISION TREE PARTS                                       │
├────────────────────────────────────────────────────────────┤
│  Node    →  a point where the individual RESIDES / decides │
│  Branch  →  an ACTION taken by the individual (an edge)    │
│  Leaf    →  the PAYOFF of the full sequence of actions     │
└────────────────────────────────────────────────────────────┘
```

Textual (ASCII) skeleton:

```
                     ROOT (individual decides)
                    /                        \
              action a1                    action a2
                /                              \
          [maybe more nodes/branches...]   [maybe more...]
              |                                 |
           LEAF: payoff                     LEAF: payoff
```

You read a decision tree **top to bottom**: start at the root, follow the branch (action) you'd take, and land on a leaf, which tells you the payoff of that entire chain of choices.

This is exactly [Lecture 01](../Lec_01_Individual_Decision_Problem/README.md)'s `(A, X, ≿)` framework, just drawn out visually for **multi-step** choices — each node is a fresh mini decision problem, and the whole tree is one big sequence of them.

### 🍼 Worked Mini-Example (concept only — full numbers in the Numerical file)

Two movie theatres near your home: **Inox** (closer) and **PVR** (farther). Each shows 3 different films. Your decision tree:

```
                          YOU (root)
                 ┌──────────────┴──────────────┐
            Go to Inox                    Go to PVR
           ┌──────┼──────┐             ┌──────┼──────┐
      Casablanca  GWTW  Dr.S       Matrix  BladeRunner Aliens
        LEAF      LEAF  LEAF        LEAF      LEAF     LEAF
```
- **Nodes**: "YOU" (the root, where you decide theatre), and the two theatre-nodes (where you decide film).
- **Branches**: "Go to Inox" / "Go to PVR" (first decision), then each specific film title (second decision).
- **Leaves**: 6 total — one payoff number per fully-specified plan ("Go to PVR, watch Aliens", etc.)

The full worked-out numbers (with alphabetic preferences, payoffs 1–6, and a walking-cost twist that changes the optimal choice) are in **[🧮 NUMERICAL](aai_lec02_decision_trees_lotteries_numerical.md#1-decision-tree--the-movie-theatre-problem)**.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--theory)

---

## 2. Risk, Nature & Random Outcomes — Simple Lotteries

### 👶 Easy Story

So far (in this lecture's §1, and all of Lecture 01), every action led to exactly ONE guaranteed outcome. Real life isn't like that — you buy a lottery ticket and Mother Nature (an imaginary extra "player" called **Nature**) rolls a die to decide what you actually get. This section is about giving that randomness precise notation.

### Setup

Actions still lead to outcomes, but now **Nature** decides *which* outcome, according to a probability distribution that depends on the action you took.

### Case A — Discrete/Finite Outcomes

Let `X = {x₁, x₂, …, xₙ}` (a finite list of possible outcomes).

For every action `a ∈ A`, Nature chooses a **probability distribution**:
```
p|ₐ = ( p(x₁|a), p(x₂|a), …, p(xₙ|a) )
```
This is called a **simple lottery** over `X`, conditioned on action `a`.

The set of ALL such valid probability distributions over `X` is written `Δ(X)` (read: "the simplex over X" — just a fancy name for "all the valid probability distributions you could put on X").

**Validity rules** (what makes `p|ₐ` a genuine probability distribution):
```
For every a ∈ A, if p|ₐ ∈ Δ(X), then:

  Rule 1 (non-negativity, each ≤ 1):
     for every i ∈ {1, 2, …, n},   0 ≤ p(xᵢ|a) ≤ 1

  Rule 2 (probabilities sum to 1):
     Σᵢ₌₁ⁿ p(xᵢ|a) = 1
```

> 🍼 **Kid version:** A lottery is just a spinner. Each slice of the spinner is one possible outcome. Rule 1 says no slice can be negative-sized or bigger than the whole spinner. Rule 2 says all the slices, added up, must exactly cover the whole spinner (100%) — no gaps, no overlaps.

```
   SIMPLE LOTTERY SPINNER  (example: X = {10, 0})

        ┌─────────────┐
        │   p=0.75    │  → outcome 10
        │   ┌───┐     │
        │   │0.25│    │  → outcome 0
        └───┴───┴─────┘
   0.75 + 0.25 = 1.00  ✓ valid lottery
```

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--theory)

---

## 3. Continuous Outcomes — Cumulative Distribution Functions

### 👶 Easy Story

Sometimes outcomes aren't a neat little list (like {10, 0}) — they're ANY number in a range, like "your profit could be anywhere between ₹0 and ₹1,00,000." You can't list infinitely many numbers, so instead of a spinner-with-slices, you use a **smooth ramp** that tells you "chance the outcome is at most this much."

### Setup

Let `X = [x, x̄] ⊆ ℝ` — outcomes now form a continuous interval (`x` = lower bound, `x̄` = upper bound).

For every action `a ∈ A`, Nature chooses a **cumulative distribution function (CDF)**:
```
Fₐ : ℝ → [0, 1],      where     Fₐ(x) = Pr(X ≤ x | a)
```
`Fₐ(x)` answers: *"given I took action a, what's the probability the outcome ends up ≤ x?"*

The set of all valid CDFs over `X` is again written `Δ(X)`.

**Validity rules** — for every `a ∈ A`, if `Fₐ ∈ Δ(X)`, then:

```
Rule 1 (starts at 0):        lim_{x → -∞} Fₐ(x) = 0
Rule 2 (ends at 1):           lim_{x → +∞} Fₐ(x) = 1
Rule 3 (non-decreasing):      for any x, y ∈ X with x ≤ y,   Fₐ(x) ≤ Fₐ(y)
Rule 4 (right-continuous):    for any x ∈ X,   Fₐ(x) = lim_{h→0} Fₐ(x + h)
```

| Rule | Plain English |
|---|---|
| Starts at 0 | Nothing can be less than the smallest possible outcome, so probability of "≤ way below range" is 0 |
| Ends at 1 | Everything is eventually covered, so probability of "≤ way above range" is 1 |
| Non-decreasing | As you allow bigger and bigger outcomes, the "probability of at most this much" can only go up (or stay flat), never down |
| Right-continuous | No sudden unexplained jumps when approaching from the right — a technical smoothness condition |

> 🍼 **Kid version:** Imagine filling a bathtub with water very slowly from empty (0%) to completely full (100%) — it only ever goes UP or stays flat, never drains backward on its own. That's a CDF.

```
   Fₐ(x)
    1 |                        ______________
      |                   ____/
      |              ____/
      |         ____/
    0 |____/
      └────────────────────────────────────── x
          x (lower bound)         x̄ (upper bound)
```

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--theory)

---

## 4. Compound Lotteries

### 👶 Easy Story

A **compound lottery** is "a lottery whose prizes are themselves other lotteries" — like a raffle where the prize isn't cash, it's ANOTHER raffle ticket for a second drawing.

### Definition

> Given a certain action, **Nature may choose a simple lottery of simple lotteries over outcomes.**

That is: instead of the branches of a probability node leading straight to final outcomes, they can lead to ANOTHER probability node, which only THEN leads to final outcomes (possibly nested several levels deep).

```
                     Player
                  ┌────┴────┐
              action g    action s
                  │            │
                  N (Nature)   N (Nature)
              ┌───┴───┐        │
          p=0.625  p=0.375     │  (single-level lottery)
             │         │       │
             N         N     0.5 / 0.5
          ┌──┴──┐   ┌──┴──┐    │    │
        0.9   0.1  0.5   0.5  10    0
         │     │    │     │
        10     0   10     0
```

Here, action `g` doesn't lead straight to a payoff — it leads to a **first Nature move** (0.625 vs 0.375), and only then a **second Nature move** decides the actual payoff (10 or 0). Action `s`, by contrast, is a plain **simple lottery**: one Nature move straight to a payoff.

> 🍼 **Kid version:** Simple lottery = one spin of one wheel. Compound lottery = spin a wheel, and whichever wedge you land on is ITSELF another wheel you now have to spin.

### Reducing a Compound Lottery to a Simple One

Any compound lottery can be **collapsed down** into an equivalent simple lottery by multiplying probabilities along each path and summing paths that lead to the same final outcome — this uses the basic rules of probability (chain rule + law of total probability). A fully worked reduction (multiplying out the 0.625/0.375 and 0.9/0.1/0.5/0.5 branches above) is in **[🧮 NUMERICAL — §4](aai_lec02_decision_trees_lotteries_numerical.md#4-compound-lottery-reduction)**.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--theory)

---

## 5. Evaluating Lotteries — Expected Utility

### 👶 Easy Story

Now that outcomes are random, "which action is better" can't just mean "which single outcome is better" — it means **which action gives the better AVERAGE payoff, weighted by how likely each outcome is.** That average is called **expected utility**.

### Setup

Let `u : X → ℝ` be the payoff (utility) function over outcomes — the same `u` from [Lecture 01 §4](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#4-the-utility-representation-theorem).

### Case A — Discrete Outcomes

`X = {x₁, x₂, …, xₙ}`. For every `a ∈ A`, Nature chooses a lottery `p|ₐ` over `X`. The **expected payoff** from `p|ₐ` is:

```
        n
E[u(X)| p|ₐ] = Σ  u(xᵢ) · p(xᵢ|a)
       i=1
```

Plain English: *for every possible outcome, multiply "how good it is" (`u(xᵢ)`) by "how likely it is" (`p(xᵢ|a)`), then add all those products up.*

### Case B — Continuous Outcomes

`X = [x, x̄] ⊆ ℝ`. For every `a ∈ A`, Nature chooses a lottery `Fₐ` over `X`. The expected payoff is:

```
             ∞
E[u(X)| Fₐ] = ∫  u(x) dFₐ(x)
            -∞
```

This is the continuous cousin of the same idea: instead of summing over a finite list of outcomes, you integrate over the whole continuous range, weighted by the CDF.

```
┌────────────────────────────────────────────────────────────┐
│  EXPECTED UTILITY — the ONE idea behind both formulas      │
├────────────────────────────────────────────────────────────┤
│  discrete:    Σ  (how good) × (how likely)                 │
│  continuous:  ∫  (how good) × (how likely, as a density)   │
└────────────────────────────────────────────────────────────┘
```

> 🍼 **Kid version:** If a scratch card has a 75% chance of ₹10 and a 25% chance of ₹0, your "fair expectation" isn't ₹10 or ₹0 — it's `0.75×10 + 0.25×0 = ₹7.5`. Expected utility is exactly this idea, just with "how good" (`u`) instead of raw cash, so it also works when outcomes aren't money.

### Comparing Lotteries — the Decision Rule

Between two lotteries `g` (risky/gamble) and `s` (safe/sure-ish), a rational individual prefers `g` over `s` if and only if:
```
E[u(X) | g]  ≥  E[u(X) | s]
```
Exactly the same "pick the bigger number" rule from [Lecture 01 §5](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#5-rational-choice-assumption--the-value-function) — just applied to *expected* payoffs instead of certain ones.

Full worked numeric comparisons (including a famous case where **changing just ONE outcome value flips which lottery wins**) are in **[🧮 NUMERICAL — §2 & §3](aai_lec02_decision_trees_lotteries_numerical.md#2-expected-utility-worked-examples)**.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--theory)

---

## 6. Cheat Sheet & Exam Hacks

```
╔════════════════════════════════════════════════════════════════════╗
║  ADVANCED AI — LEC 02 ONE-LINERS                                   ║
╠════════════════════════════════════════════════════════════════════╣
║  Decision tree: Node=decide | Branch=action | Leaf=payoff          ║
║                                                                    ║
║  Simple lottery (discrete): p|ₐ=(p(x₁|a),...,p(xₙ|a)) ∈ Δ(X)       ║
║    rules: 0 ≤ p(xᵢ|a) ≤ 1   AND   Σ p(xᵢ|a) = 1                    ║
║                                                                    ║
║  Simple lottery (continuous): Fₐ(x) = Pr(X ≤ x|a) ∈ Δ(X)           ║
║    rules: Fₐ(-∞)=0, Fₐ(+∞)=1, non-decreasing, right-continuous     ║
║                                                                    ║
║  Compound lottery = a lottery whose PRIZES are lotteries           ║
║  (collapses to a simple lottery via chain rule + total probability)║
║                                                                    ║
║  EXPECTED UTILITY:                                                 ║
║    discrete:    E[u(X)|p|ₐ] = Σᵢ u(xᵢ)·p(xᵢ|a)                     ║
║    continuous:  E[u(X)|Fₐ]  = ∫ u(x) dFₐ(x)                        ║
║  Prefer g over s  ⟺  E[u|g] ≥ E[u|s]                              ║
╚════════════════════════════════════════════════════════════════════╝
```

### ⚡ Exam Red Flags

1. **"What are the parts of a decision tree?"** → Nodes (decisions), Branches/Edges (actions), Leaves (payoffs).
2. **"Give the 2 validity rules for a discrete probability distribution `p|ₐ`."** → Each `p(xᵢ|a) ∈ [0,1]`, and they sum to 1.
3. **"Give the 4 validity rules for a CDF `Fₐ`."** → Limit at −∞ is 0, limit at +∞ is 1, non-decreasing, right-continuous.
4. **"What is a compound lottery, and how do you evaluate it?"** → A lottery whose outcomes are themselves lotteries; reduce it to a simple lottery by multiplying branch probabilities along each path and summing paths landing on the same final outcome, then apply the expected-utility formula.
5. **"Write the expected-utility formula for a discrete lottery."** → `E[u(X)|p|ₐ] = Σᵢ u(xᵢ)·p(xᵢ|a)`. This is THE most commonly tested formula in this lecture — memorize it cold.
6. **The #1 exam trap** — students forget that expected utility depends on the **actual utility numbers**, not just probabilities. Two lotteries with identical probabilities can flip their ranking if even ONE outcome's payoff value changes (see the Numerical file §3 for a worked example of exactly this).
7. **A close second exam trap** — forgetting that a walking-cost / effort-cost style penalty in a decision tree must be subtracted from the LEAF payoff, not from the branch probability. Costs affect `v(a)`, not `Δ(X)`.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--theory)

---

> **Next:** [🧮 NUMERICAL →](aai_lec02_decision_trees_lotteries_numerical.md) · [🎯 PRACTICE →](aai_lec02_decision_trees_lotteries_practice.md)
>
> *Advanced AI · Lec 02 · github.com/rpaut03l/TS-02-03*
