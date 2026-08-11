# 🧮 Advanced AI — Lec 01: Individual Rational Decision Paradigm — NUMERICAL

### *Every definition, turned into real numbers, step by step*

> **Nav:** [← Lec 01 README](README.md) | [📖 THEORY](aai_lec01_decision_problem_theory.md) | **NUMERICAL** | [🎯 PRACTICE](aai_lec01_decision_problem_practice.md)

---

## 📚 Table of Contents

| # | Worked Example | Jump |
|---|---|---|
| 1 | Checking Rationality — Completeness & Transitivity | [§1](#1-checking-rationality--completeness--transitivity) |
| 2 | Utility Representation — Full Worked Proof | [§2](#2-utility-representation--full-worked-proof) |
| 3 | Rational Choice in Action — u vs v | [§3](#3-rational-choice-in-action--u-vs-v) |
| 4 | Formula Cheat Sheet | [§4](#4-formula-cheat-sheet) |

---

## 1. Checking Rationality — Completeness & Transitivity

### The Setup

`X = {Tea, Coffee, Juice}`. Someone states these preference facts:
```
Tea ≿ Coffee
Coffee ≿ Juice
Tea ≿ Juice
Juice ≿ Tea      ← so Tea ∼ Juice as well
```

### Step 1 — Check Completeness

List every pair and confirm at least one direction is stated:

| Pair | Stated? |
|---|---|
| (Tea, Coffee) | ✅ Tea ≿ Coffee |
| (Coffee, Juice) | ✅ Coffee ≿ Juice |
| (Tea, Juice) | ✅ Tea ≿ Juice (and Juice ≿ Tea) |

All 3 pairs covered → **Complete ✓**

### Step 2 — Check Transitivity

Take the chain `Tea ≿ Coffee` and `Coffee ≿ Juice`. Transitivity demands:
```
Tea ≿ Coffee  AND  Coffee ≿ Juice   ⟹   Tea ≿ Juice
```
We were given `Tea ≿ Juice` directly — ✅ **consistent, no contradiction.**

### Step 3 — Conclusion

```
Complete ✓  +  Transitive ✓   ⟹   ≿ is RATIONAL
```

### A broken example, for contrast

Now suppose instead: `A ≿ B`, `B ≿ C`, `C ≻ A`.
```
A ≿ B  and  B ≿ C   ⟹  (by transitivity) we NEED  A ≿ C
But we were told  C ≻ A,  which means  ¬(A ≿ C)
```
**Contradiction → NOT transitive → NOT rational.** This is the "money pump" loop from the Theory file §3: a trader could keep offering you A→B→C→A trades forever, and if you truly hold this preference, you'd agree to every single one, going in circles.

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--numerical)

---

## 2. Utility Representation — Full Worked Proof

Let `X = {Noodles, Burger, Pizza}` with the rational preference:
```
Pizza ≻ Burger ≻ Noodles     (pizza is strictly best, noodles strictly worst)
```

**Step 1 — find best (`x̄`) and worst (`x`) outcomes.**
```
x̄ = Pizza     (beats or ties everyone)
x = Noodles    (everyone beats or ties it)
```

**Step 2 — build indifference classes.**
Since there are no ties stated, each outcome is its own class:
```
Class 1 (worst):  {Noodles}
Class 2 (middle): {Burger}
Class 3 (best):   {Pizza}
```

**Step 3 — assign increasing numbers, worst to best.**
```
u(Noodles) = 1
u(Burger)  = 2
u(Pizza)   = 3
```

**Step 4 — verify the representation condition `u(x) ≥ u(y) ⟺ x ≿ y`.**

| Pair | u comparison | ≿ comparison | Match? |
|---|---|---|---|
| Pizza vs Burger | u(3) ≥ u(2) ✓ | Pizza ≻ Burger ✓ | ✅ |
| Burger vs Noodles | u(2) ≥ u(1) ✓ | Burger ≻ Noodles ✓ | ✅ |
| Pizza vs Noodles | u(3) ≥ u(1) ✓ | Pizza ≻ Noodles ✓ | ✅ |

✅ Every pair matches — `u` correctly represents `≿`.

### Now with a tie, to test indifference classes properly

`X = {Noodles, Burger, Pasta, Pizza}` with `Pizza ≻ Pasta ∼ Burger ≻ Noodles` (Pasta and Burger are tied).

**Step 1 — best/worst:** `x̄ = Pizza`, `x = Noodles`.

**Step 2 — indifference classes:**
```
Class 1 (worst):  {Noodles}
Class 2 (middle): {Pasta, Burger}     ← tied, SAME class
Class 3 (best):   {Pizza}
```

**Step 3 — numbering (tied items get the SAME number):**
```
u(Noodles) = 1
u(Pasta)   = 2
u(Burger)  = 2      ← same as Pasta, because Pasta ∼ Burger
u(Pizza)   = 3
```

**Step 4 — verify:** `u(Pasta) = u(Burger) = 2`, and indeed `Pasta ≿ Burger` AND `Burger ≿ Pasta` both hold (that's the definition of `∼`) — so `u(Pasta) ≥ u(Burger)` and `u(Burger) ≥ u(Pasta)` both check out. ✅ Consistent.

> ⚠️ **Common mistake:** giving tied outcomes *different* numbers (like Pasta=2, Burger=3). That would silently claim `Burger ≻ Pasta`, which contradicts the stated `Pasta ∼ Burger`. Tied outcomes MUST share the same utility number.

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--numerical)

---

## 3. Rational Choice in Action — u vs v

### The Setup

`A = {order_pizza, order_burger, order_salad}`. The outcome-map `x*: A → X` (what you actually get if the kitchen has stock) is:
```
x*(order_pizza)  = Pizza
x*(order_burger) = Burger        ← kitchen is OUT of burger buns today!
x*(order_salad)  = Salad
```
But wait — since burger buns are out, the ACTUAL outcome-map today is:
```
x*(order_pizza)  = Pizza
x*(order_burger) = Salad   (kitchen substitutes salad when out of buns)
x*(order_salad)  = Salad
```
Utility over outcomes: `u(Pizza) = 3, u(Salad) = 1` (Burger's outcome no longer independently matters — it's mapped to Salad today).

### Step 1 — compute `v(a) = u(x*(a))` for every action

```
v(order_pizza)  = u(x*(order_pizza))  = u(Pizza) = 3
v(order_burger) = u(x*(order_burger)) = u(Salad) = 1     ← NOT u(Burger)!
v(order_salad)  = u(x*(order_salad))  = u(Salad) = 1
```

### Step 2 — apply the rational choice rule

```
max(v(order_pizza), v(order_burger), v(order_salad)) = max(3, 1, 1) = 3
```
✅ **Rational choice: `order_pizza`** (`a* = order_pizza`), since `v(order_pizza) = 3 ≥ v(a)` for every other action `a`.

> 🍼 **Kid version:** This example is exactly why `A` and `X` are kept separate, and why `v` (over actions) is different from `u` (over outcomes). If you only tracked "how much I like Burger," you'd have wrongly assumed ordering burger gets you burger-happiness — but today it actually gets you salad-happiness, because of a real-world substitution. `v(a) = u(x*(a))` correctly captures what you ACTUALLY get, not what the action's name suggests.

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--numerical)

---

## 4. Formula Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════╗
║  FORMULAS — COPY THESE DOWN BEFORE THE EXAM                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Weak:      x ≿ y                                                  ║
║  Strict:    x ≻ y  ⟺  x ≿ y ∧ ¬(y ≿ x)                            ║
║  Indiff:    x ∼ y  ⟺  x ≿ y ∧ y ≿ x                                ║
║                                                                    ║
║  Completeness:  ∀ x,y ∈ X:  x ≿ y  or  y ≿ x                       ║
║  Transitivity:  ∀ x,y,z ∈ X:  x ≿ y ∧ y ≿ z  ⟹  x ≿ z              ║
║                                                                    ║
║  Representation:      u(x) ≥ u(y)  ⟺  x ≿ y                       ║
║  Tied outcomes MUST get the SAME u-value                          ║
║                                                                    ║
║  Value of an action:  v(a) = u(x*(a))                              ║
║  Rational choice:     choose a* with v(a*) ≥ v(a)  ∀ a ∈ A         ║
╚══════════════════════════════════════════════════════════════════╝
```

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--numerical)

---

## 📝 Summary

Every worked example in this file exists to turn Lecture 01's abstract symbols into something you can actually compute by hand. Checking rationality (§1) is really just two habits: scan every pair of outcomes to confirm none are left unranked (completeness), and chase any chain of "beats-or-ties" comparisons to make sure it never loops back on itself (transitivity) — and when it does loop, that's precisely the money-pump trap the theory is designed to catch. Building a representing utility function (§2) turned out to be almost mechanical once you know the trick: find the best and worst outcomes, group anything tied together into the same "class," and count upward from worst to best — with the one sharp warning that tied outcomes must always receive the exact same number, never two different ones. The u-vs-v example (§3) showed why Lecture 01 insists on keeping actions and outcomes as separate objects — an action's *name* can be misleading about what it actually delivers once real-world substitutions happen, and only `v(a) = u(x*(a))` tracks the truth. Keep this file bookmarked as your formula reference; nearly every calculation in the Chapter 1 exercises folder — especially Exercise 1.5's car-preference thresholds and Exercise 1.1's payoff assignments — leans directly on the exact same two techniques practiced here.

---

> **Next:** [🎯 PRACTICE →](aai_lec01_decision_problem_practice.md) · [← back to THEORY](aai_lec01_decision_problem_theory.md) · [Lecture 02 →](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)
>
> *Advanced AI · Lec 01 · github.com/rpaut03l/TS-02-03*
