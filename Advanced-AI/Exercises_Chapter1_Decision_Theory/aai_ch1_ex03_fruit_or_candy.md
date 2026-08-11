# 📗 Exercise 1.3 — Fruit or Candy

> **Nav:** [← Exercise 1.2](aai_ch1_ex02_going_to_movies.md) | [← Chapter 1 Exercises](README.md) | **1.3** | [1.4 →](aai_ch1_ex04_alcohol_consumption.md)

---

## 📋 What's Being Asked

Bananas cost $0.50, candy costs $0.25, budget is $1.25. Each additional banana (or candy) you eat is worth **half** the payoff of the previous one — a **diminishing-returns** payoff structure. Part (a) wants the full action set given the budget. Part (b) wants the decision tree. Part (c) asks whether you should spend the entire $1.25, with a rational-choice justification. Part (d) raises candy's price to $0.30 and asks how the action set and your answer to (c) change.

---

## Part (a) — The Action Set Given the Budget

Every action is a pair `(b, c)` = (number of bananas, number of candies) satisfying `0.50b + 0.25c ≤ 1.25`, with `b, c ≥ 0` integers.

**Step 1 — find the max bananas alone:** `1.25/0.50 = 2.5` → `b ≤ 2`.

**Step 2 — for each value of `b`, find the max `c` that still fits:**
```
b=0: 0.25c ≤ 1.25 → c ≤ 5   → c = 0,1,2,3,4,5   (6 options)
b=1: 0.25c ≤ 0.75 → c ≤ 3   → c = 0,1,2,3        (4 options)
b=2: 0.25c ≤ 0.25 → c ≤ 1   → c = 0,1            (2 options)
```

**Step 3 — total action set:** `A = {(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1)}` — **12 total actions.**

## Part (b) — The Decision Tree

```
                              YOU (root: how many bananas?)
              ┌──────────────────┬──────────────────┐
            b=0                 b=1                b=2
       (how many candy?)   (how many candy?)   (how many candy?)
      ┌─┬─┬─┬─┬─┐          ┌─┬─┬─┬─┐            ┌─┬─┐
    c=0..5 (6 leaves)   c=0..3 (4 leaves)      c=0..1 (2 leaves)
```
A two-level tree: first choose bananas, then choose the maximum affordable candy for that many bananas — 12 leaves total, matching part (a)'s action set exactly.

## Part (c) — Should You Spend It All?

Since you **value money itself** (leftover cash retains its own $1-for-$1 value), the true payoff of any action includes leftover cash: `U(b,c) = B(b) + C(c) + (1.25 − cost(b,c))`, where `B(b)` and `C(c)` are the cumulative eating-payoffs.

**Step 1 — derive a closed-form for cumulative payoff** (geometric series, each term half the last, first term `a₁`, ratio `r=0.5`):
```
B(b) = 1.20 + 0.60 + 0.30 + ... (b terms) = 2.40 × (1 − 0.5^b)
C(c) = 0.40 + 0.20 + 0.10 + ... (c terms) = 0.80 × (1 − 0.5^c)
```
Check `B(2) = 2.40×(1−0.25) = 1.80` (matches `1.20+0.60`) ✓; `C(1)=0.80×0.5=0.40` ✓.

**Step 2 — since `1.25` is constant, maximizing `U` reduces to maximizing "net value per category":**
```
netB(b) = B(b) − 0.50b        netC(c) = C(c) − 0.25c
```

| `b` | `B(b)` | `netB(b)` | `c` | `C(c)` | `netC(c)` |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1.20 | **0.70** | 1 | 0.40 | **0.15** |
| 2 | 1.80 | **0.80** ✅ | 2 | 0.60 | 0.10 |
| — | — | — | 3 | 0.70 | −0.05 |

**Step 3 — best `b=2` (netB=0.80), best `c=1` (netC=0.15).** Check: does `(2,1)` fit the budget? Cost = `0.50(2)+0.25(1) = 1.25` — **exactly the full budget**, so both individual optima are simultaneously achievable.

**Step 4 — verify by computing `U` for all 12 actions** (full table cross-checked numerically): the maximum is `U(2,1) = 1.80 + 0.40 + 0 = $2.20`, beating every other combination, including every combination that leaves money unspent.

✅ **Yes — spend all $1.25**, buying 2 bananas and 1 candy, for a maximum total value of **$2.20**.

## Part (d) — Candy Price Rises to $0.30

**Step 1 — recompute the action set:**
```
b=0: 0.30c ≤ 1.25 → c ≤ 4   (5 options)
b=1: 0.30c ≤ 0.75 → c ≤ 2   (3 options)
b=2: 0.30c ≤ 0.25 → c ≤ 0   (1 option)
```
**New total: 9 actions** (down from 12).

**Step 2 — recompute `netC(c) = C(c) − 0.30c`:**
```
c=1: 0.40 − 0.30 = 0.10 (new best for c)
c=2: 0.60 − 0.60 = 0.00
```

**Step 3 — since `b=2` now forces `c=0` (budget only leaves $0.25, can't afford $0.30 candy), compute `U` for every remaining action** — the maximum is now **$2.05, tied between two different plans**: `(b=2, c=0)` costing only $1.00, and `(b=1, c=1)` costing $0.80.

✅ **The answer to part (c) flips!** Neither optimal plan spends the full $1.25 anymore — $0.25 or $0.45 is rationally left unspent. The candy price hike made the marginal candy purchases no longer worth their cost, so the rational choice is to *not* spend everything.

---

## 🧠 Mnemonic & Cheat Sheet

```
╔══════════════════════════════════════════════════════════════╗
║  "SPEND UNTIL NET GOES NEGATIVE"                             ║
║  With diminishing returns, keep buying an item only while    ║
║  its NEXT unit's payoff exceeds its price. The moment the    ║
║  next unit's marginal payoff < price, STOP — even with       ║
║  money left over. Leftover cash still counts as value!       ║
╚══════════════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** whenever a problem says "you value money," always add back leftover budget as its own payoff term — forgetting this is the #1 way to wrongly conclude you should always spend everything.

---

## 📝 Summary

This exercise packs three separate lessons into one budget-constrained scenario. First, once quantities can range from 0 up to a budget-imposed maximum, the action set stops being a short hand-written list and becomes a genuine combinatorial grid — twelve pairs in part (a), shrinking to nine once a single price changed in part (d), and correctly enumerating that grid is half the battle before any optimization even starts. Second, diminishing returns (each extra banana or candy worth exactly half the previous one) meant the smart strategy wasn't "buy as much as the budget allows" but "keep buying only while the next unit's payoff still beats its price," which is precisely why the net-value-per-category shortcut (`netB`, `netC`) worked so cleanly. Third — and this is the twist worth remembering — because money itself carries value, leftover unspent cash always has to be added back into the total payoff, and the moment a price rose in part (d), the previously-obvious "spend it all" answer flipped into "leave some money on the table." That flip is the exercise's real punchline: rational spending isn't about exhausting a budget, it's about stopping exactly where marginal value crosses marginal cost, wherever that happens to land.

---

> **Next:** [Exercise 1.4 — Alcohol Consumption →](aai_ch1_ex04_alcohol_consumption.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
