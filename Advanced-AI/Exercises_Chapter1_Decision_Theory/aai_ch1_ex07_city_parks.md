# 📗 Exercise 1.7 — City Parks

> **Nav:** [← Exercise 1.6](aai_ch1_ex06_fruit_trees.md) | [← Chapter 1 Exercises](README.md) | **1.7**

---

## 📋 What's Being Asked

The city's yearly budget is $20,000,000, and city code caps park spending at 5% of that budget. The mayor's constituents have a benefit function `v(c) = √(400c) − c/80` for spending `c` dollars on parks. Part (a) wants the action set. Part (b) wants the optimal spending amount. Part (c) changes the benefit function to `v(c) = √(1600c) − c/80` (reflecting increased public enthusiasm) and asks for the new action set and optimal spending.

---

## Prerequisite — Rewriting `v(c)` for Easier Differentiation

`v(c) = √(400c) − c/80` can be simplified using `√(400c) = √400 × √c = 20√c`:
```
v(c) = 20√c − c/80
```

## Part (a) — The Action Set

**Step 1 — compute the spending cap:**
```
5% of $20,000,000 = 0.05 × 20,000,000 = $1,000,000
```
**Step 2 — since spending can be any dollar amount up to that cap:**
```
A = [0, 1,000,000]   (a continuous range, not a discrete list)
```

## Part (b) — Optimal Spending

**Step 1 — take the first derivative** (recall `d/dc[√c] = 1/(2√c)`):
```
v'(c) = 20 × 1/(2√c) − 1/80 = 10/√c − 1/80
```

**Step 2 — set to zero and solve for `c*`:**
```
10/√c = 1/80
√c = 10 × 80 = 800
c* = 800² = 640,000
```

**Step 3 — confirm it's a maximum via the second derivative:**
```
v''(c) = −5c^(−3/2)
```
This is negative for every `c > 0` → **concave** → `c* = 640,000` is confirmed a maximum.

**Step 4 — check the unconstrained optimum against the cap:**
```
c* = $640,000  ≤  cap = $1,000,000  ✓  (fits comfortably within the action set)
```

**Step 5 — compute the resulting benefit** (note `√640,000 = 800` exactly, since `800² = 640,000`):
```
v(640,000) = 20(800) − 640,000/80 = 16,000 − 8,000 = 8,000
```

✅ **Optimal spending: $640,000**, delivering a net public benefit of **$8,000** (in money-equivalent value units).

## Part (c) — Public Enthusiasm Shifts

**Step 1 — the action set itself is UNCHANGED** — the 5%-of-budget cap is a city-code rule, independent of public opinion:
```
A = [0, 1,000,000]   (same as before)
```

**Step 2 — rewrite the new benefit function:**
```
v(c) = √(1600c) − c/80 = 40√c − c/80
```

**Step 3 — take the derivative and solve for the unconstrained optimum:**
```
v'(c) = 20/√c − 1/80
20/√c = 1/80
√c = 1,600
c* = 1,600² = 2,560,000
```

**Step 4 — check against the cap:**
```
c* = $2,560,000   >   cap = $1,000,000   ❌  (exceeds the allowed range!)
```

**Step 5 — since the unconstrained peak lies outside the legal spending range, verify the function is still rising right up to the cap** (confirming the true constrained optimum sits exactly at the boundary):
```
v'(1,000,000) = 20/√1,000,000 − 1/80 = 20/1000 − 1/80 = 0.02 − 0.0125 = 0.0075 > 0
```
Since the slope is still positive at the cap, benefit is still increasing right up to the maximum legal spending level — the mayor should spend as much as the law allows.

✅ **New optimal spending: the full $1,000,000 cap** — a **corner solution**, not the calculus-derived interior peak, because public enthusiasm has grown so much that the "ideal" unconstrained spending now exceeds what city code permits.

```
v(c)
    |                                    ___________ (would keep
    |                              _____/              rising past
    |                        _____/                     the cap!)
    |                  _____/
    |            _____/
    |      _____/
    |  ___/
    +──┴──────────────────────────────┴──────────── c
    0                              $1,000,000 (cap)
                                   ↑ mayor stops HERE, not at the
                                     true unconstrained peak
```

---

## 🧠 Mnemonic & Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════╗
║  "ALWAYS CHECK THE CAP AFTER SOLVING v'(c)=0"                    ║
║  1. Simplify v(c), take v'(c), set to 0, solve for c*.           ║
║  2. Check v''(c)<0 to confirm a maximum, not a minimum.          ║
║  3. Compare c* against any hard boundary/cap.                    ║
║  4. If c* > cap: check v'(cap) — if still positive, the TRUE     ║
║     optimum is the CAP itself (a corner solution), not c*.       ║
╚══════════════════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** this is the single most commonly missed step in continuous-optimization exam problems — always sanity-check your calculus answer against any stated constraint. An unconstrained maximum that falls outside the legal/physical range is not the real answer; the real answer becomes whichever boundary the constraint pushes you against, provided the function is still improving in that direction.

---

## 📝 Summary

This exercise closes the chapter with the single most commonly forgotten step in continuous-optimization problems: always check your calculus answer against any hard constraint before trusting it. Part (b)'s solution worked exactly as expected — differentiate, set to zero, solve, confirm concavity — and the resulting optimum happily fit inside the spending cap with room to spare. Part (c) then quietly changed the public's enthusiasm for parks, and the same four-step method produced an unconstrained optimum that blew straight past the legal 5%-of-budget ceiling. The critical extra step — checking whether the payoff function was still *increasing* right at the boundary — is what correctly identified the true answer as a **corner solution**: spend the maximum the law allows, not the number calculus alone would suggest. This is a genuinely common real-world pattern well beyond city budgets — any time a mathematically "ideal" amount exceeds a hard limit (a budget cap, a physical capacity, a legal restriction) while the payoff is still climbing at that limit, the rational choice is always to push right up against the boundary, and no further calculus is needed once that's confirmed.

---

## 🎓 Chapter 1 Complete

You've now worked through every action-set pattern this chapter uses: short discrete lists (1.1, 1.2, 1.6), budget-constrained combinatorial grids (1.3, 1.5), and continuous ranges requiring calculus (1.4, 1.7) — including the two sharpest traps in the whole chapter: a rule change that only ripples through part of a table (1.6), and an unconstrained optimum that quietly exceeds a hard limit (1.7). Every one of these exercises used nothing beyond the `(A, X, ≿)` and `v(a)` machinery from [Lecture 01](../Lec_01_Individual_Decision_Problem/README.md) and the decision-tree convention from [Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md) — proof that a small, well-understood toolkit goes a very long way once you know exactly which pattern a new problem is asking you to apply.

---

> **← Back to start:** [Chapter 1 Exercises Hub](README.md) | [Exercise 1.1](aai_ch1_ex01_your_decision.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
