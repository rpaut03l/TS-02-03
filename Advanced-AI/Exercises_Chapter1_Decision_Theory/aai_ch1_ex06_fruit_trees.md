# 📗 Exercise 1.6 — Fruit Trees

> **Nav:** [← Exercise 1.5](aai_ch1_ex05_buying_a_car.md) | [← Chapter 1 Exercises](README.md) | **1.6** | [1.7 →](aai_ch1_ex07_city_parks.md)

---

## 📋 What's Being Asked

Room for up to 2 fruit trees; choices are apple, orange, or pear. Maintenance costs: apple $100, orange $70, pear $120. Food-bill reductions: apple $130, pear $145, orange $90. Parts (a)–(c) want the action set, the payoff of each action, and the decision-tree solution under the original rule. Part (d) changes the rule so a *second tree of the same kind* only gives half its usual reduction, and asks whether the optimal choice changes.

---

## Prerequisite — The Payoff of Each Tree Type Alone

| Tree | Maintenance cost | Food-bill reduction | Net savings |
|---|---|---|---|
| Apple (A) | $100 | $130 | **+$30** |
| Orange (O) | $70 | $90 | **+$20** |
| Pear (P) | $120 | $145 | **+$25** |

Define payoff `v(a)` = total reduction − total cost = total savings. A rational player maximizes `v(a)`.

## Part (a) — Set of Actions and Outcomes

Since repeats are allowed and order doesn't matter, planting 0, 1, or 2 trees gives:
```
A = { ∅, Apple, Orange, Pear, AA, OO, PP, AO, AP, OP }  →  10 total actions
```
`X = A` here — there's no randomness (no "Nature" involved), so each action deterministically produces itself as the outcome: `x*(a) = a`.

## Part (b) — Payoff of Each Action

| Action | Cost | Reduction | `v(a)` |
|---|---|---|---|
| ∅ | 0 | 0 | 0 |
| A | 100 | 130 | 30 |
| O | 70 | 90 | 20 |
| P | 120 | 145 | 25 |
| AA | 200 | 260 | **60** |
| OO | 140 | 180 | 40 |
| PP | 240 | 290 | 50 |
| AO | 170 | 220 | 50 |
| AP | 220 | 275 | 55 |
| OP | 190 | 235 | 45 |

## Part (c) — Decision Tree & Rational Choice

```
                                YOU (root, decide slot 1)
              ┌──────────┬──────────┬──────────┬
            Stop        Apple      Orange      Pear
           (leaf)      (decide     (decide    (decide
             v=0        slot 2)     slot 2)    slot 2)
```
Each "decide slot 2" node branches again into Stop/+Apple/+Orange/+Pear, giving the 10 leaves from part (b).

```
max(0, 30, 20, 25, 60, 40, 50, 50, 55, 45) = 60
```
✅ **Rational choice: two Apple trees (AA), `v(AA) = 60`** — beating every other option, including the mixed pairs.

## Part (d) — Second Tree of the Same Kind Gives Half Reduction

**Step 1 — recompute only the three same-kind pairs** (mixed pairs are untouched, since neither tree in a mixed pair is a "second tree of the same kind"):
```
AA: 130 + 65 = 195 reduction, 200 cost → v(AA) = 195 − 200 = −5   (was 60!)
OO: 90 + 45 = 135 reduction, 140 cost → v(OO) = 135 − 140 = −5   (was 40)
PP: 145 + 72.5 = 217.5 reduction, 240 cost → v(PP) = 217.5 − 240 = −22.5   (was 50)
```

**Step 2 — updated full table:**

| Action | Original `v(a)` | New `v(a)` |
|---|---|---|
| ∅, A, O, P | 0, 30, 20, 25 | unchanged |
| **AA** | 60 | **−5** |
| **OO** | 40 | **−5** |
| **PP** | 50 | **−22.5** |
| AO, AP, OP | 50, 55, 45 | unchanged |

**Step 3 — find the new maximum:**
```
max(0, 30, 20, 25, −5, 40, 50, 50, −5, 45, 55, −22.5) = 55
```
✅ **Yes — the rational choice flips to Apple + Pear (AP), `v(AP) = 55`.** Planting two apples went from the single best option to a net loss, because the new rule specifically punishes *repeating* a kind; mixed pairs like AP were never repeats, so they're completely unaffected and simply become the new best answer once AA collapses.

---

## 🧠 Mnemonic & Cheat Sheet

```
╔════════════════════════════════════════════════════════════════════╗
║  "SAME-KIND RULE CHANGES ONLY HIT SAME-KIND PAIRS"                 ║
║  When a rule specifically targets "the second X of the same        ║
║  kind," only the AA/OO/PP-style entries move — mixed pairs         ║
║  (AO/AP/OP) are mathematically untouched. Recompute ONLY what      ║
║  the rule actually changes, then re-scan the whole table for max.  ║
╚════════════════════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** this is the direct Chapter 1 counterpart to Lecture 02's "Odds don't flip you — Outcomes do" lesson — here it's "a targeted rule change doesn't flip you, it flips only the specific options it targets," and the ripple effect (AA collapsing promotes AP to the top) is exactly the kind of second-order reasoning exams test.

---

## 📝 Summary

This exercise's real lesson lives entirely in the gap between parts (c) and (d): a single rule change — "the second tree of the same kind only gets half the usual food-bill reduction" — didn't nudge the answer slightly, it completely reversed the previously unambiguous best choice, flipping two apple trees from the clear winner (payoff 60) into an active net loss (payoff −5). The reason this rule change only affected same-kind pairs (AA, OO, PP) and left every mixed pair (AO, AP, OP) completely untouched is worth internalizing as a general skill: whenever a problem changes a rule that targets a specific pattern, resist the temptation to recompute everything from scratch — instead, isolate exactly which entries the rule change actually touches, recompute only those, and then rescan the full table for a new maximum. That habit alone would have caught the ripple effect here, where AA's collapse from best-in-class quietly promoted AP into the new optimal choice by default, not because AP itself changed at all, but because its strongest competitor evaporated. This exact "isolate what changed, then rescan" discipline is the single most useful transferable skill from this whole exercise.

---

> **Next:** [Exercise 1.7 — City Parks →](aai_ch1_ex07_city_parks.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
