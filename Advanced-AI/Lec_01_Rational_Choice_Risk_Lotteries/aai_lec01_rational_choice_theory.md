# 📖 Advanced AI — Lec 01: Rational Choice, Preferences & Risk — THEORY

### *Actions · Outcomes · Preferences · Utility · Decision Trees · Lotteries · Expected Utility*

> **Nav:** [← Lec 01 README](README.md) | **THEORY** | [🧮 NUMERICAL](aai_lec01_rational_choice_numerical.md) | [🎯 PRACTICE](aai_lec01_rational_choice_practice.md)

---

## 🧠 MNEMONIC: **"AXP → CTR → UV → DT → RNG → EU"**

> **A**ctions, **X** (Outcomes), **P**references → **C**ompleteness, **T**ransitivity, **R**ationality → **U**tility, **V**alue function → **D**ecision **T**rees → **R**isk / **N**ature / **G**ambles (lotteries) → **E**xpected **U**tility

Say it like a little chant: *"Ax-P leads to C-T-R, which gives U-V, then D-T, then R-N-G, then E-U."* Every section below is one letter of this chant, in order.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | The Individual Decision Problem (A, X, ≿) | [§1](#1-the-individual-decision-problem) |
| 2 | Preference Relations — Weak, Strict, Indifference | [§2](#2-preference-relations) |
| 3 | Rational Preferences — Completeness & Transitivity | [§3](#3-rational-preferences) |
| 4 | The Utility Representation Theorem | [§4](#4-the-utility-representation-theorem) |
| 5 | Rational Choice Assumption & the Value Function | [§5](#5-rational-choice-assumption--the-value-function) |
| 6 | Decision Trees | [§6](#6-decision-trees) |
| 7 | Risk, Nature & Random Outcomes — Simple Lotteries | [§7](#7-risk-nature--random-outcomes--simple-lotteries) |
| 8 | Continuous Outcomes — Cumulative Distribution Functions | [§8](#8-continuous-outcomes--cumulative-distribution-functions) |
| 9 | Compound Lotteries | [§9](#9-compound-lotteries) |
| 10 | Evaluating Lotteries — Expected Utility | [§10](#10-evaluating-lotteries--expected-utility) |
| 11 | Cheat Sheet & Exam Hacks | [§11](#11-cheat-sheet--exam-hacks) |

---

## 1. The Individual Decision Problem

### 👶 Easy Story

Picture a kid standing in front of an ice-cream van. The kid has:
- Some **flavours to pick from** (that's what she can *do*)
- Some **scoops that actually land in the cone** depending on what she picks (that's what *happens*)
- A **favourite order of flavours in her head** (chocolate > vanilla > mango, always)

Advanced AI's very first building block just gives these three things fancy names.

### Formal Definition

Every individual decision problem has exactly three ingredients:

```
┌───────────────────────────────────────────────────────────┐
│  INDIVIDUAL DECISION PROBLEM                              │
├───────────────────────────────────────────────────────────┤
│  A  =  Actions      "what I can CHOOSE"                   │
│  X  =  Outcomes     "what actually HAPPENS"               │
│  ≿  =  Preferences  "how I RANK what happens"  (also: R)  │
└───────────────────────────────────────────────────────────┘
```

| Symbol | Name | Meaning | Ice-cream analogy |
|---|---|---|---|
| `A` | **Actions** | The set of things the individual is allowed to choose | {chocolate, vanilla, mango} |
| `X` | **Outcomes** | The set of consequences that can result from actions | {chocolate scoop, vanilla scoop, mango scoop} |
| `≿` (or `R`) | **Preferences** | A ranking (ordering) over outcomes | chocolate ≿ vanilla ≿ mango |

> 🍼 **Kid version:** `A` is the menu, `X` is what actually lands in your cone, `≿` is your own private "I like this more than that" list in your head.

Note carefully: actions and outcomes are **not automatically the same thing**. Picking "chocolate" (an action) *usually* gets you a chocolate scoop (an outcome) — but if the shop is out of chocolate, the action and the outcome can come apart. This gap is exactly why the theory keeps `A` and `X` as two separate sets instead of merging them.

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 2. Preference Relations

### 👶 Easy Story

You don't just have one flavour list — you actually have THREE different feelings hiding inside "I like this":
1. "I like this **at least as much as** that" (could be a tie)
2. "I like this **strictly more** than that" (no tie, clear winner)
3. "I like these **equally**" (a genuine tie)

### Formal Definitions

Let `X` be the set of outcomes. A preference relation `≿` is a subset of `X × X` (written `≿ ⊆ X²`).

**1. Weak preference `≿`**
```
x ≿ y     means     (x, y) ∈ ≿
```
Read: *"x is weakly preferred to y."* — x is at least as good as y (x could equal y in value, or beat it).

**2. Strict preference `≻`**
```
≻ ⊂ X²

x ≻ y   ⟺   x ≿ y   AND   ¬(y ≿ x)
```
Read: *"x beats y, and y does NOT beat x."* — a genuine, no-tie win for x.

**3. Indifference `∼`**
```
∼ ⊆ X²

x ∼ y   ⟺   x ≿ y   AND   y ≿ x
```
Read: *"x is at least as good as y, AND y is at least as good as x"* → they must be **exactly tied**.

```
┌─────────────────────────────────────────────────────────┐
│   x ≿ y   "x is at least as good as y"    (weak)         │
│   x ≻ y   "x is strictly better than y"   (strict win)   │
│   x ∼ y   "x and y are tied"              (indifference) │
└─────────────────────────────────────────────────────────┘
```

> 🍼 **Kid version:** `≿` is "I wouldn't say no to x over y." `≻` is "I'd fight for x over y." `∼` is "flip a coin, I really don't care which one."

### 🧠 Mnemonic
**"≿ is the parent, ≻ and ∼ are its two children."** Weak preference `≿` alone can be split into exactly a strict-win part (`≻`) and a tie part (`∼`) — nothing else is possible.

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 3. Rational Preferences

### 👶 Easy Story

Imagine a friend whose taste makes NO sense: she says pizza ≥ burger, burger ≥ noodles, but then noodles ≥ pizza! That's a **loop** — you could keep trading her food forever and she'd always agree to the next trade, losing money/food every time (this is literally called a "money pump" in economics). **Rational preferences forbid this loop.**

### The Two Properties

**Completeness** — *"You must have an opinion on every pair."*
```
For any x, y ∈ X:   either  x ≿ y   or   y ≿ x   (or both, if tied)
```
No outcome pair is left un-ranked. You are never allowed to shrug and say "I have no idea which I prefer."

**Transitivity** — *"No contradictions allowed."*
```
For any x, y, z ∈ X:   if  x ≿ y   and   y ≿ z   then   x ≿ z
```
If x beats-or-ties y, and y beats-or-ties z, then x must beat-or-tie z. No loops.

### Rational Preference Relation

```
≿ is RATIONAL   ⟺   ≿ is COMPLETE   AND   ≿ is TRANSITIVE
```

| Property | What breaks if it fails | Real-world symptom |
|---|---|---|
| Completeness | Some pair (x, y) has NO ranking at all | "I genuinely can't say if I prefer A or B" — decision paralysis |
| Transitivity | A ranking loop exists: x ≻ y ≻ z ≻ x | You can be "money-pumped" — someone extracts endless value by cycling your trades |

> 🍼 **Kid version:** Completeness = "you must pick a favourite (or tie) between ANY two toys, no skipping." Transitivity = "your favourite list can't go in circles."

### 🧠 Mnemonic: **"C-T = No Skips, No Loops"**
- **C**ompleteness → **No Skips** (every pair gets ranked)
- **T**ransitivity → **No Loops** (the ranking never contradicts itself)

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 4. The Utility Representation Theorem

### 👶 Easy Story

Rankings ("I like chocolate more than vanilla") are annoying to compute with. Numbers are easy — you can add, compare, and optimise over numbers with plain arithmetic. This section is about a magic promise: **any rational preference over a finite set of outcomes CAN be turned into plain numbers**, without changing what you actually prefer.

### The Payoff (Utility) Function

```
u : X → ℝ
```
`u` is just a function that assigns a real number to every outcome.

### Representation

`u` **represents** `≿` if, for every `x, y ∈ X`:
```
u(x) ≥ u(y)   ⟺   x ≿ y
```
In words: bigger number = weakly more preferred. The numbers and the rankings must always agree, both directions.

### The Existence Result

> **Theorem.** If `≿` is rational (complete + transitive) and `X` is **finite**, then there exists a `u : X → ℝ` that represents `≿`.

This is huge: it means you never *have* to reason with abstract "≿" symbols — as long as preferences are rational and the outcome set is finite, you can always replace them with ordinary numbers.

### Proof Sketch (2 steps)

**Step 1 — Find the best and worst outcomes.**
Since `≿` is rational and `X` is finite, there exist `x̄, x ∈ X` (best and worst) such that:
```
x̄ ≿ y   for all y ∈ X        ("x̄ beats or ties everyone")
z  ≿ x   for all z ∈ X        ("everyone beats or ties x, the worst")
```
> 🍼 Think of lining up every kid's favourite toy — with a finite pile of toys, there is always a most-liked one and a least-liked one (possibly tied with others).

**Step 2 — Build indifference classes, number them monotonically.**
Group all outcomes that are tied with each other into **indifference classes**. Since `≿` is complete and transitive, these classes are cleanly ordered from worst to best with NO overlaps and NO gaps in the ranking. Assign an increasing number to each class as you go from worst to best (e.g., worst class → 1, next class → 2, …, best class → k). Any two outcomes in the same class get the same number (they're tied); any outcome in a higher class gets a strictly bigger number.

```
Worst class            Middle class(es)             Best class
   u = 1        <         u = 2, 3, ...        <        u = k
```

This constructed `u` satisfies `u(x) ≥ u(y) ⟺ x ≿ y` by design — done.

> 🍼 **Kid version:** Sort every toy from "meh" to "amazing," put toys that feel exactly equal into the same pile, then just number the piles 1, 2, 3, ... from meh to amazing. That numbering IS your utility function.

### ⚠️ What this theorem does NOT say
- It does **not** say the numbers are unique — any relabeling that keeps the *order* the same also works (e.g., 1,2,3 or 10,20,30 or 1,5,100 all represent the same preference, as long as order is preserved).
- It does **not** work automatically for infinite `X` without extra continuity assumptions (out of scope here — just remember: finite + rational ⟹ representable).

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 5. Rational Choice Assumption & the Value Function

### 👶 Easy Story

Knowing your OWN taste (utility over outcomes) is only half the story. You also need to know **which action gets you which outcome** — otherwise you might love chocolate ice-cream but keep pointing at the wrong tub!

### Rational Choice Assumption (4 things the individual must know)

For the individual to be able to act rationally, they must know:

1. **All possible actions** available to them: the full set `A`.
2. **All possible outcomes**: the full set `X`.
3. **Which outcome follows from which action** — a function:
   ```
   x* : A → X
   ```
   (given action `a`, `x*(a)` tells you exactly which outcome results)
4. **Their own rational preference (payoff function) over `X`** — i.e., they know `u`.

```
┌─────────────────────────────────────────────────────────┐
│  RATIONAL CHOICE ASSUMPTION — the 4 "must-knows"        │
├─────────────────────────────────────────────────────────┤
│  1. Know A          (the menu)                          │
│  2. Know X           (what can happen)                  │
│  3. Know x*: A → X   (which action → which outcome)     │
│  4. Know u over X    (own ranking/payoff of outcomes)   │
└─────────────────────────────────────────────────────────┘
```

### Payoff From Actions

Combine `u` (payoff over outcomes) with `x*` (which action → which outcome) to get a payoff **directly over actions**:

```
v : A → ℝ,      where     v(a) = u(x*(a))
```

`v(a)` says: *"if I take action `a`, this is the number I end up getting."* It's just `u` viewed through the lens of "what does this action actually get me."

### Rational Individual — the Choice Rule

An individual is **rational** if:
1. They satisfy the rational choice assumption (the 4 must-knows above), AND
2. They choose action `a* ∈ A` **if and only if**:
   ```
   v(a*) ≥ v(a)     for all a ∈ A
   ```

> 🍼 **Kid version:** Line up every menu item, work out the "yumminess number" you'd get for picking each one, then just point at whichever number is the biggest. That's it — that's the entire definition of a "rational" choice in this framework.

### 🧠 Mnemonic: **"Know it, Map it, Rank it, Max it"**
- **Know it** → know `A` and `X`
- **Map it** → know `x*: A → X`
- **Rank it** → know `u` over `X`
- **Max it** → pick `a*` that maximizes `v(a) = u(x*(a))`

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 6. Decision Trees

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

### 🍼 Worked Mini-Example (concept only — full numbers in the Numerical file)

Two movie theatres near your home: **Inox** (closer) and **PVR** (farther). Each shows 3 different films. Your decision tree:

```
                          YOU (root)
                  ┌──────────────┴────────────┐
            Go to Inox                    Go to PVR
           ┌──────┼──────┐             ┌──────┼──────┐
      Casablanca  GWTW  Dr.S       Matrix  BladeRunner Aliens
        LEAF      LEAF  LEAF        LEAF      LEAF     LEAF
```
- **Nodes**: "YOU" (the root, where you decide theatre), and the two theatre-nodes (where you decide film).
- **Branches**: "Go to Inox" / "Go to PVR" (first decision), then each specific film title (second decision).
- **Leaves**: 6 total — one payoff number per fully-specified plan ("Go to PVR, watch Aliens", etc.)

The full worked-out numbers (with alphabetic preferences, payoffs 1–6, and a walking-cost twist) are in **[🧮 NUMERICAL](aai_lec01_rational_choice_numerical.md)**.

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 7. Risk, Nature & Random Outcomes — Simple Lotteries

### 👶 Easy Story

So far, every action led to exactly ONE guaranteed outcome. Real life isn't like that — you buy a lottery ticket and Mother Nature (an imaginary extra "player" called **Nature**) rolls a die to decide what you actually get. This section is about giving that randomness precise notation.

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

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 8. Continuous Outcomes — Cumulative Distribution Functions

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

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 9. Compound Lotteries

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

Any compound lottery can be **collapsed down** into an equivalent simple lottery by multiplying probabilities along each path and summing paths that lead to the same final outcome — this uses the basic rules of probability (chain rule + law of total probability). A fully worked reduction (multiplying out the 0.625/0.375 and 0.9/0.1/0.5/0.5 branches above) is in **[🧮 NUMERICAL — §4](aai_lec01_rational_choice_numerical.md#4-compound-lottery-reduction)**.

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 10. Evaluating Lotteries — Expected Utility

### 👶 Easy Story

Now that outcomes are random, "which action is better" can't just mean "which single outcome is better" — it means **which action gives the better AVERAGE payoff, weighted by how likely each outcome is.** That average is called **expected utility**.

### Setup

Let `u : X → ℝ` be the payoff (utility) function over outcomes, exactly as before (§4).

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
Exactly the same "pick the bigger number" rule from §5 — just applied to *expected* payoffs instead of certain ones.

Full worked numeric comparisons (including a famous case where **changing just ONE outcome value flips which lottery wins**) are in **[🧮 NUMERICAL — §2 & §3](aai_lec01_rational_choice_numerical.md#2-expected-utility-worked-examples)**.

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

## 11. Cheat Sheet & Exam Hacks

```
╔═════════════════════════════════════════════════════════════════════╗
║  ADVANCED AI — LEC 01 ONE-LINERS                                    ║
╠═════════════════════════════════════════════════════════════════════╣
║  Decision problem = (A, X, ≿)                                       ║
║    A = actions, X = outcomes, ≿ = preferences over X                ║
║                                                                     ║
║  ≿ weak | ≻ strict = ≿ AND NOT(reverse ≿) | ∼ tie = ≿ both ways     ║
║                                                                     ║
║  RATIONAL ≿  =  COMPLETE (no skips) + TRANSITIVE (no loops)         ║
║                                                                     ║
║  u: X → ℝ represents ≿   ⟺   u(x) ≥ u(y) ⟺ x ≿ y                  ║
║  THEOREM: ≿ rational + X finite  ⟹  representing u EXISTS          ║
║                                                                     ║
║  Rational Choice Assumption: know A, know X, know x*:A→X, know u    ║
║  v(a) = u(x*(a))     — payoff of an ACTION, not just an outcome     ║
║  Rational choice: pick a* with v(a*) ≥ v(a) for all a ∈ A           ║
║                                                                     ║
║  Decision tree: Node=decide | Branch=action | Leaf=payoff           ║
║                                                                     ║
║  Simple lottery (discrete): p|ₐ=(p(x₁|a),...,p(xₙ|a)) ∈ Δ(X)        ║
║    rules: 0 ≤ p(xᵢ|a) ≤ 1   AND   Σ p(xᵢ|a) = 1                     ║
║                                                                     ║
║  Simple lottery (continuous): Fₐ(x) = Pr(X ≤ x|a) ∈ Δ(X)            ║
║    rules: Fₐ(-∞)=0, Fₐ(+∞)=1, non-decreasing, right-continuous      ║
║                                                                     ║
║  Compound lottery = a lottery whose PRIZES are lotteries            ║
║  (collapses to a simple lottery via chain rule + total probability) ║
║                                                                     ║
║  EXPECTED UTILITY:                                                  ║
║    discrete:    E[u(X)|p|ₐ] = Σᵢ u(xᵢ)·p(xᵢ|a)                      ║
║    continuous:  E[u(X)|Fₐ]  = ∫ u(x) dFₐ(x)                         ║
║  Prefer g over s  ⟺  E[u|g] ≥ E[u|s]                               ║
╚═════════════════════════════════════════════════════════════════════╝
```

### ⚡ Exam Red Flags

1. **"What are the 3 components of an individual decision problem?"** → Actions (A), Outcomes (X), Preferences (≿).
2. **"Define strict preference and indifference in terms of ≿."** → `x ≻ y ⟺ x ≿ y ∧ ¬(y ≿ x)`; `x ∼ y ⟺ x ≿ y ∧ y ≿ x`. Don't swap the negation — it belongs only in the strict-preference definition.
3. **"What two properties make a preference relation rational?"** → Completeness + Transitivity. (Memory hook: "No Skips, No Loops.")
4. **"State the utility representation theorem."** → If `≿` is rational and `X` is finite, there exists `u: X → ℝ` representing `≿`, i.e. `u(x) ≥ u(y) ⟺ x ≿ y`. Don't forget the **finiteness** condition — it's a common trick to drop it in a MCQ.
5. **"What must a rational individual know (rational choice assumption)?"** → All actions `A`, all outcomes `X`, the map `x*: A → X`, and their own `u` over `X`. All four, not just some.
6. **"Distinguish u and v."** → `u` is over **outcomes**, `v` is over **actions**: `v(a) = u(x*(a))`.
7. **"What are the parts of a decision tree?"** → Nodes (decisions), Branches/Edges (actions), Leaves (payoffs).
8. **"Give the 2 validity rules for a discrete probability distribution `p|ₐ`."** → Each `p(xᵢ|a) ∈ [0,1]`, and they sum to 1.
9. **"Give the 4 validity rules for a CDF `Fₐ`."** → Limit at −∞ is 0, limit at +∞ is 1, non-decreasing, right-continuous.
10. **"What is a compound lottery, and how do you evaluate it?"** → A lottery whose outcomes are themselves lotteries; reduce it to a simple lottery by multiplying branch probabilities along each path and summing paths landing on the same final outcome, then apply the expected-utility formula.
11. **"Write the expected-utility formula for a discrete lottery."** → `E[u(X)|p|ₐ] = Σᵢ u(xᵢ)·p(xᵢ|a)`. This is THE most commonly tested formula in this unit — memorize it cold.
12. **The #1 exam trap** — students forget that expected utility depends on the **actual utility numbers**, not just probabilities. Two lotteries with identical probabilities can flip their ranking if even ONE outcome's payoff value changes (see the Numerical file, §3, for a worked example of exactly this).

[↑ Back to Top](#-advanced-ai--lec-01-rational-choice-preferences--risk--theory)

---

> **Next:** [🧮 NUMERICAL →](aai_lec01_rational_choice_numerical.md) · [🎯 PRACTICE →](aai_lec01_rational_choice_practice.md)
>
> *Advanced AI · Lec 01 · github.com/rpaut03l/TS-02-03*
