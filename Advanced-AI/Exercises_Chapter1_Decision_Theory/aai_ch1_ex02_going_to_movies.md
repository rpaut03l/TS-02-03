# 📗 Exercise 1.2 — Going to the Movies

> **Nav:** [← Exercise 1.1](aai_ch1_ex01_your_decision.md) | [← Chapter 1 Exercises](README.md) | **1.2** | [1.3 →](aai_ch1_ex03_fruit_or_candy.md)

---

## 📋 What's Being Asked

Two theaters: **Cineclass** (1 mile away, showing *Casablanca*, *Gone with the Wind*, *Dr. Strangelove*) and **Cineblast** (3 miles away, showing *The Matrix*, *Blade Runner*, *Aliens*). Part (a) wants just the bare decision tree, no numbers. Part (b) wants alphabetic-preference payoffs (1–6) applied, and the resulting choice. Part (c) adds a walking cost of 1 payoff unit per mile and asks whether the optimal choice changes.

---

## Part (a) — The Bare Decision Tree

```
                                YOU  (root node)
                     ┌───────────────┴───────────────┐
              Go to Cineclass                 Go to Cineblast
                (action)                         (action)
          ┌─────────┼─────────┐            ┌──────────┼──────────┐
     Casablanca    GWTW     Dr.S       The Matrix  BladeRunner  Aliens
       (leaf)     (leaf)   (leaf)        (leaf)      (leaf)     (leaf)
```
**Nodes:** "YOU" (root, choosing theater) and the two theater-nodes (choosing film). **Branches:** "Go to Cineclass"/"Go to Cineblast" first, then the specific film title. **Leaves:** 6 total fully-specified plans, no payoffs yet.

## Part (b) — Alphabetic Payoffs 1 Through 6

**Step 1 — sort all six movies alphabetically:**
```
Aliens, Blade Runner, Casablanca, Dr. Strangelove, Gone with the Wind, The Matrix
```

**Step 2 — you like Aliens most, The Matrix least, so assign 6 down to 1 in that order:**

| Rank (best→worst) | Movie | Payoff |
|---|---|---|
| 1st | Aliens | **6** |
| 2nd | Blade Runner | **5** |
| 3rd | Casablanca | **4** |
| 4th | Dr. Strangelove | **3** |
| 5th | Gone with the Wind | **2** |
| 6th (worst) | The Matrix | **1** |

**Step 3 — attach to the tree and apply the rational choice rule:**
```
max(4, 2, 3, 1, 5, 6) = 6  →  "Go to Cineblast, watch Aliens"
```
✅ **Choice: Cineblast → Aliens (payoff 6).** No distance cost yet, so it's simply your literal favorite film.

## Part (c) — Add Walking Cost: 1 Payoff Unit per Mile

**Step 1 — compute distance cost per theater:**
```
Cineclass: 1 mile → cost = 1 unit
Cineblast: 3 miles → cost = 3 units
```

**Step 2 — subtract cost from every leaf at that theater:**

| Theater | Movie | Base u | − Cost | Updated v |
|---|---|---|---|---|
| Cineclass (−1) | Casablanca | 4 | −1 | **3** |
| Cineclass (−1) | Gone with the Wind | 2 | −1 | **1** |
| Cineclass (−1) | Dr. Strangelove | 3 | −1 | **2** |
| Cineblast (−3) | The Matrix | 1 | −3 | **−2** |
| Cineblast (−3) | Blade Runner | 5 | −3 | **2** |
| Cineblast (−3) | Aliens | 6 | −3 | **3** |

**Step 3 — find the new best plan:**
```
max(3, 1, 2, −2, 2, 3) = 3  →  TWO leaves tie:
   "Cineclass, Casablanca"   AND   "Cineblast, Aliens"
```

✅ **Yes, the choice changes** — from a unique winner (Cineblast/Aliens) to a **tie** between two plans. The extra 2 units of walking cost (3−1) exactly cancels the 2-point gap between Aliens (6) and Casablanca (4), leaving a rational individual indifferent (`∼`) between them.

---

## 🧠 Mnemonic & Cheat Sheet

```
╔══════════════════════════════════════════════════════════╗
║  "ALPHABET IN, DISTANCE OUT"                             ║
║  Assign payoffs from your preference order FIRST,        ║
║  then subtract cost from the LEAF, never the branch.     ║
║  Watch for TIES when cost differences match payoff gaps. ║
╚══════════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** this is the textbook example of Lecture 02 §6's exam trap — "a close second exam trap ... costs affect `v(a)`, not any probability." Costs are always subtracted directly from the leaf value, and always check whether the cost gap between two branches exactly matches (or exceeds) the payoff gap, since that's exactly when ties or flips occur.

---

## 📝 Summary

This exercise is a direct, hands-on rerun of the decision-tree machinery built in [Lecture 02 §1](../Lec_02_Decision_Trees_Risk_Lotteries/aai_lec02_decision_trees_lotteries_theory.md#1-decision-trees), and its real teaching moment lives entirely in part (c). Before any walking cost was introduced, the answer felt almost too obvious — of course you'd go watch your favorite movie. The moment a cost got attached to distance, though, the two theaters stopped being "close one, far one" and became a genuine trade-off between preference and convenience, and the walking cost happened to land at exactly the right size to create a dead-even tie between them. That's not a coincidence built into the numbers by accident — it's the whole point of the exercise: costs subtract directly from a leaf's payoff, never from anything else, and whenever a cost gap happens to match a payoff gap exactly, indifference is the mathematically correct outcome, not a sign that something went wrong in your arithmetic. Keep this exact pattern in mind any time a "which option is better" question quietly hides a cost or effort difference between the choices — always resolve it at the leaf level, and always double-check whether the gap fully cancels, partially narrows, or leaves the original winner untouched.

---

> **Next:** [Exercise 1.3 — Fruit or Candy →](aai_ch1_ex03_fruit_or_candy.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
