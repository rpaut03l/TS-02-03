# 📖 Advanced AI — Lec 01: Individual Rational Decision Paradigm — THEORY

### *Actions · Outcomes · Preferences · Rationality · Utility Representation · Rational Choice*

> **Nav:** [← Lec 01 README](README.md) | **THEORY** | [🧮 NUMERICAL](aai_lec01_decision_problem_numerical.md) | [🎯 PRACTICE](aai_lec01_decision_problem_practice.md) | [Next: Lec 02 ➡️](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)

---

## 🧠 MNEMONIC: **"AXP → CTR → UV"**

> **A**ctions, **X** (Outcomes), **P**references → **C**ompleteness, **T**ransitivity, **R**ationality → **U**tility, **V**alue function

Say it like a little chant: *"Ax-P leads to C-T-R, which gives U-V."* Every section below is one letter of this chant, in order. Decision Trees and Lotteries — the part where randomness enters — belong to **[Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)**.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | The Individual Decision Problem (A, X, ≿) | [§1](#1-the-individual-decision-problem) |
| 2 | Preference Relations — Weak, Strict, Indifference | [§2](#2-preference-relations) |
| 3 | Rational Preferences — Completeness & Transitivity | [§3](#3-rational-preferences) |
| 4 | The Utility Representation Theorem | [§4](#4-the-utility-representation-theorem) |
| 5 | Rational Choice Assumption & the Value Function | [§5](#5-rational-choice-assumption--the-value-function) |
| 6 | Cheat Sheet & Exam Hacks | [§6](#6-cheat-sheet--exam-hacks) |

---

## 1. The Individual Decision Problem

### 👶 Easy Story

Picture a kid standing in front of an ice-cream van. The kid has:
- Some **flavours to pick from** (that's what she can *do*)
- Some **scoops that actually land in the cone** depending on what she picks (that's what *happens*)
- A **favourite order of flavours in her head** (chocolate > vanilla > mango, always)

This lecture's very first building block just gives these three things fancy names.

### Formal Definition

Every individual decision problem has exactly three ingredients:

```
┌───────────────────────────────────────────────────────────┐
│  INDIVIDUAL DECISION PROBLEM                               │
├───────────────────────────────────────────────────────────┤
│  A  =  Actions      "what I can CHOOSE"                    │
│  X  =  Outcomes     "what actually HAPPENS"                │
│  ≿  =  Preferences  "how I RANK what happens"  (also: R)   │
└───────────────────────────────────────────────────────────┘
```

| Symbol | Name | Meaning | Ice-cream analogy |
|---|---|---|---|
| `A` | **Actions** | The set of things the individual is allowed to choose | {chocolate, vanilla, mango} |
| `X` | **Outcomes** | The set of consequences that can result from actions | {chocolate scoop, vanilla scoop, mango scoop} |
| `≿` (or `R`) | **Preferences** | A ranking (ordering) over outcomes | chocolate ≿ vanilla ≿ mango |

> 🍼 **Kid version:** `A` is the menu, `X` is what actually lands in your cone, `≿` is your own private "I like this more than that" list in your head.

Note carefully: actions and outcomes are **not automatically the same thing**. Picking "chocolate" (an action) *usually* gets you a chocolate scoop (an outcome) — but if the shop is out of chocolate, the action and the outcome can come apart. This gap is exactly why the theory keeps `A` and `X` as two separate sets instead of merging them.

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--theory)

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

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--theory)

---

## 3. Rational Preferences

### 👶 Easy Story

Imagine a friend whose taste makes NO sense: she says pizza ≥ burger, burger ≥ noodles, but then noodles ≥ pizza! That's a **loop** — you could keep trading her food forever and she'd always agree to the next trade, losing food every time (this is literally called a "money pump" in economics). **Rational preferences forbid this loop.**

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

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--theory)

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

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--theory)

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
│  RATIONAL CHOICE ASSUMPTION — the 4 "must-knows"         │
├─────────────────────────────────────────────────────────┤
│  1. Know A          (the menu)                           │
│  2. Know X           (what can happen)                   │
│  3. Know x*: A → X   (which action → which outcome)      │
│  4. Know u over X    (own ranking/payoff of outcomes)     │
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

### 🔭 Where this goes next

Everything so far assumes each action leads to exactly ONE guaranteed outcome — no luck involved. **[Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)** breaks that assumption: it introduces **decision trees** (multi-step versions of this same choice problem) and **Nature** — an imaginary extra "player" who rolls dice to decide the outcome of a risky action. The `u`, `v`, and "pick the biggest number" logic you just learned doesn't go away — it gets upgraded to **expected utility**.

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--theory)

---

## 6. Cheat Sheet & Exam Hacks

```
╔══════════════════════════════════════════════════════════════════╗
║  ADVANCED AI — LEC 01 ONE-LINERS                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Decision problem = (A, X, ≿)                                     ║
║    A = actions, X = outcomes, ≿ = preferences over X               ║
║                                                                    ║
║  ≿ weak | ≻ strict = ≿ AND NOT(reverse ≿) | ∼ tie = ≿ both ways    ║
║                                                                    ║
║  RATIONAL ≿  =  COMPLETE (no skips) + TRANSITIVE (no loops)        ║
║                                                                    ║
║  u: X → ℝ represents ≿   ⟺   u(x) ≥ u(y) ⟺ x ≿ y                  ║
║  THEOREM: ≿ rational + X finite  ⟹  representing u EXISTS          ║
║                                                                    ║
║  Rational Choice Assumption: know A, know X, know x*:A→X, know u   ║
║  v(a) = u(x*(a))     — payoff of an ACTION, not just an outcome    ║
║  Rational choice: pick a* with v(a*) ≥ v(a) for all a ∈ A          ║
╚══════════════════════════════════════════════════════════════════╝
```

### ⚡ Exam Red Flags

1. **"What are the 3 components of an individual decision problem?"** → Actions (A), Outcomes (X), Preferences (≿).
2. **"Define strict preference and indifference in terms of ≿."** → `x ≻ y ⟺ x ≿ y ∧ ¬(y ≿ x)`; `x ∼ y ⟺ x ≿ y ∧ y ≿ x`. Don't swap the negation — it belongs only in the strict-preference definition.
3. **"What two properties make a preference relation rational?"** → Completeness + Transitivity. (Memory hook: "No Skips, No Loops.")
4. **"State the utility representation theorem."** → If `≿` is rational and `X` is finite, there exists `u: X → ℝ` representing `≿`, i.e. `u(x) ≥ u(y) ⟺ x ≿ y`. Don't forget the **finiteness** condition — it's a common trick to drop it in a MCQ.
5. **"What must a rational individual know (rational choice assumption)?"** → All actions `A`, all outcomes `X`, the map `x*: A → X`, and their own `u` over `X`. All four, not just some.
6. **"Distinguish u and v."** → `u` is over **outcomes**, `v` is over **actions**: `v(a) = u(x*(a))`.

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--theory)

---

> **Next:** [🧮 NUMERICAL →](aai_lec01_decision_problem_numerical.md) · [🎯 PRACTICE →](aai_lec01_decision_problem_practice.md) · [Lecture 02 (Decision Trees & Lotteries) →](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)
>
> *Advanced AI · Lec 01 · github.com/rpaut03l/TS-02-03*
