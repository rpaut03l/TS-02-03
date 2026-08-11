# 📗 Exercise 1.5 — Buying a Car

> **Nav:** [← Exercise 1.4](aai_ch1_ex04_alcohol_consumption.md) | [← Chapter 1 Exercises](README.md) | **1.5** | [1.6 →](aai_ch1_ex06_fruit_trees.md)

---

## 📋 What's Being Asked

Budget $12,000, five cars on the lot (Corolla 2002 $9,350, Camry 2001 $10,500, LeSabre 2001 $8,825, Civic 2000 $9,215, Impreza 2000 $9,690). Your base preference (any given year): Camry ≻ Impreza ≻ Corolla ≻ Civic ≻ LeSabre — but you're willing to pay up to $999 to move up one preference tier, and up to $500 to move to a year-older version of the same model. Parts (a)–(d) want the alternative set, the derived preference relation, a decision tree with payoffs, and a demonstration that the payoff numbers aren't unique.

---

## Part (a) — The Set of Possible Alternatives

Every listed car costs less than the $12,000 budget, so all five are individually affordable:
```
A = { Corolla, Camry, LeSabre, Civic, Impreza }
```
(The "buy nothing" option could also be added in principle, but the exercise frames the decision as choosing among the available lot inventory.)

## Part (b) — Deriving the Actual Preference Relation

The base tier order (Camry ≻ Impreza ≻ Corolla ≻ Civic ≻ LeSabre) only holds automatically **if no price gap between adjacent tiers exceeds $999** — otherwise the cheaper, lower-tier car wins instead. Check every adjacent pair using the rule "prefer the higher tier unless the lower tier is cheaper by more than $999":

**Step 1 — Camry ($10,500) vs Impreza ($9,690):**
```
Threshold = 10,500 − 999 = $9,501
Impreza's price ($9,690) > $9,501 → prefer CAMRY
```

**Step 2 — Impreza ($9,690) vs Corolla ($9,350):**
```
Threshold = 9,690 − 999 = $8,691
Corolla's price ($9,350) > $8,691 → prefer IMPREZA
```

**Step 3 — Corolla ($9,350) vs Civic ($9,215):**
```
Threshold = 9,350 − 999 = $8,351
Civic's price ($9,215) > $8,351 → prefer COROLLA
```

**Step 4 — Civic ($9,215) vs LeSabre ($8,825):**
```
Threshold = 9,215 − 999 = $8,216
LeSabre's price ($8,825) > $8,216 → prefer CIVIC
```

**Step 5 — since every adjacent check preserved the base order, and preferences are transitive (Lecture 01 §3), the full chain holds:**
```
Camry ≻ Impreza ≻ Corolla ≻ Civic ≻ LeSabre
```
(The $500 year-threshold rule doesn't apply here, since no two cars in this list share the same model at different years — it's given only as part of the general preference structure.)

## Part (c) — Decision Tree, Payoffs, and the Choice

```
                                YOU (root)
        ┌──────────┬──────────┬──────────┬──────────┬
      Camry      Impreza    Corolla     Civic     LeSabre
     (leaf)      (leaf)     (leaf)      (leaf)      (leaf)
      u=5          u=4        u=3         u=2          u=1
```
A flat, single-level tree — assign payoffs matching the derived ranking (5=best down to 1=worst):
```
max(5, 4, 3, 2, 1) = 5  →  Choose: Camry
```
✅ Since $10,500 ≤ $12,000, the Camry is affordable — **buy the Camry.**

## Part (d) — A Different Payoff Set, Same Problem

**Yes.** Per the Utility Representation Theorem (Lecture 01 §4), any set of numbers preserving the *same relative order* represents the identical preference relation — the actual numeric spacing carries no meaning by itself. For example:
```
u(Camry) = 100, u(Impreza) = 80, u(Corolla) = 60, u(Civic) = 40, u(LeSabre) = 20
```
or even unevenly spaced numbers like `u = (50, 49, 10, 9, 1)` — as long as `Camry > Impreza > Corolla > Civic > LeSabre` in the numbers, the decision tree still points to the exact same optimal choice (Camry), because rational choice only cares about *which number is largest*, never about the specific gaps between them.

---

## 🧠 Mnemonic & Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════════╗
║  "CHECK EVERY ADJACENT GAP AGAINST ITS THRESHOLD"                    ║
║  For tier-jump thresholds: prefer higher tier UNLESS                 ║
║  lower tier is cheaper by MORE than the threshold amount.            ║
║  Transitivity then extends any surviving adjacent chain end-to-end.  ║
║                                                                      ║
║  Utility numbers are NEVER unique — only ORDER matters.              ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** this exercise is the practical version of Lecture 01 §4's "⚠️ what this theorem does NOT say" warning — a representing utility function is never the *only* valid one, and exams love testing whether you understand that relabeling numbers (while preserving order) changes nothing about the rational choice.

---

## 📝 Summary

This exercise is really about *earning* a preference relation rather than simply being handed one. The base tier ordering (Camry, Impreza, Corolla, Civic, LeSabre) only survives contact with real prices because every single adjacent price gap happened to fall inside the $999 threshold — had even one gap exceeded it, the actual rational preference would have deviated from the stated "ideal" ordering, and the exercise deliberately makes you verify every single link in the chain rather than assume it holds. Part (d)'s twist — showing that a completely different set of numbers (100/80/60/40/20, or even unevenly spaced ones) represents the exact same preference and leads to the exact same optimal choice — is the most direct, hands-on proof you'll get in this entire chapter that utility numbers are only ever meaningful in their *relative order*, never their absolute values or spacing. That single idea, first introduced as a warning in [Lecture 01 §4](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#4-the-utility-representation-theorem), is one of the most commonly misunderstood points in early decision theory, and this exercise is designed specifically to make the abstract warning concrete and impossible to forget.

---

> **Next:** [Exercise 1.6 — Fruit Trees →](aai_ch1_ex06_fruit_trees.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
