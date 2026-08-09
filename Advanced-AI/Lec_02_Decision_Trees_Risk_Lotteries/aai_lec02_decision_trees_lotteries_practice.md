# 🎯 Advanced AI — Lec 02: Decision Trees, Risk & Lotteries — PRACTICE

### *Self-test problems, a Q&A bank, and a mini project — try before you peek*

> **Nav:** [⬅️ Prev: Lec 01](../Lec_01_Individual_Decision_Problem/README.md) | [← Lec 02 README](README.md) | [📖 THEORY](aai_lec02_decision_trees_lotteries_theory.md) | [🧮 NUMERICAL](aai_lec02_decision_trees_lotteries_numerical.md) | **PRACTICE**

---

## 📚 Table of Contents

| # | Section | Jump |
|---|---|---|
| 1 | Decision Tree Practice | [§1](#1-decision-tree-practice) |
| 2 | Simple Lottery Concept Checks | [§2](#2-simple-lottery-concept-checks) |
| 3 | Expected Utility Practice | [§3](#3-expected-utility-practice) |
| 4 | Compound Lottery Practice | [§4](#4-compound-lottery-practice) |
| 5 | Mini Project — Build Your Own Lottery | [§5](#5-mini-project--build-your-own-lottery) |
| 6 | Exam-Style Q&A Bank | [§6](#6-exam-style-qa-bank) |

> 💡 Every problem below is inside a `<details>` block. **Try to solve it on paper first**, then click "Show Answer" to check yourself.

---

## 1. Decision Tree Practice

**Q1.1.** You're deciding between two restaurants: **Cafe A** (2 km away) serving {Pasta, Salad} and **Cafe B** (5 km away) serving {Burger, Pizza}. Draw the decision tree (nodes, branches, leaves) with NO payoffs yet.

<details>
<summary>Show Answer</summary>

```
                          YOU (root)
              ┌───────────────┴───────────────┐
          Go to Cafe A                    Go to Cafe B
          ┌──────┴──────┐               ┌──────┴──────┐
        Pasta         Salad           Burger         Pizza
        (leaf)        (leaf)          (leaf)         (leaf)
```
Root = you deciding; first branches = which cafe (action); second branches = which dish (action); leaves = 4 total fully-specified plans, payoffs not yet attached.
</details>

---

**Q1.2.** Suppose your preference ranking (best→worst) is: `Pizza ≻ Pasta ≻ Burger ≻ Salad`. Assign payoffs 4 (best) down to 1 (worst), attach them to the tree from Q1.1, and state your optimal choice **before** any distance cost.

<details>
<summary>Show Answer</summary>

```
Pizza = 4, Pasta = 3, Burger = 2, Salad = 1

                          YOU
              ┌───────────────┴───────────────┐
          Go to Cafe A                    Go to Cafe B
          ┌──────┴──────┐               ┌──────┴──────┐
        Pasta         Salad           Burger         Pizza
        v=3           v=1             v=2            v=4
```
`max(3, 1, 2, 4) = 4` → **Choose: Cafe B, Pizza.**
</details>

---

**Q1.3.** Now apply a walking cost of 1 payoff unit per km (Cafe A costs 2, Cafe B costs 5). Recompute all four leaves and state the new optimal choice. Does it change?

<details>
<summary>Show Answer</summary>

```
Cafe A (−2): Pasta = 3−2 = 1,   Salad = 1−2 = −1
Cafe B (−5): Burger = 2−5 = −3, Pizza = 4−5 = −1
```
`max(1, −1, −3, −1) = 1` → **Choose: Cafe A, Pasta.**

**Yes, the choice changes!** Pizza was the favourite dish, but the 5 km walk (5-point cost) is too steep — it drags Pizza's value from 4 down to −1, below Pasta's (1). This mirrors the Inox/PVR tie in the Numerical file §1, but here the cost is big enough to fully flip the winner rather than just create a tie.
</details>

---

## 2. Simple Lottery Concept Checks

**Q2.1.** Is `p = (0.3, 0.3, 0.3)` over `X = {x₁, x₂, x₃}` a valid simple lottery? Show which rule it violates, if any.

<details>
<summary>Show Answer</summary>

**No, invalid.** Rule 1 (each in `[0,1]`) is fine, but Rule 2 fails: `0.3 + 0.3 + 0.3 = 0.9 ≠ 1`. The probabilities must sum to exactly 1.
</details>

---

**Q2.2.** Is `F(x)` a valid CDF if it satisfies `F(−∞)=0`, `F(+∞)=1`, and is non-decreasing, but has an unexplained downward jump at `x=5`? Which rule fails?

<details>
<summary>Show Answer</summary>

**No, invalid** — a downward jump directly violates the **non-decreasing** rule (Rule 3), since `F` must never decrease as `x` increases. (This example also happens to break right-continuity if the jump is at the point itself, but the core violation here is non-decreasingness.)
</details>

---

## 3. Expected Utility Practice

**Q3.1.** Compute `E[u(X)|p]` for a lottery `p`: 0.4 chance of payoff 20, 0.6 chance of payoff 5. Assume `u(x) = x`.

<details>
<summary>Show Answer</summary>

```
E[u|p] = (20 × 0.4) + (5 × 0.6) = 8 + 3 = 11
```
</details>

---

**Q3.2.** Lottery `g`: 0.3 → 100, 0.7 → −10. Lottery `s`: guaranteed 20 (i.e., 1.0 → 20). Which does a rational, `u(x)=x` individual prefer?

<details>
<summary>Show Answer</summary>

```
E[u|g] = (100 × 0.3) + (−10 × 0.7) = 30 − 7 = 23
E[u|s] = (20 × 1.0) = 20
```
`23 > 20` → **Prefer g.** Even though `g` has a scary 70% chance of a loss, the 30% shot at 100 more than compensates on average.
</details>

---

**Q3.3.** Take `g` from Q3.2 and change the bad outcome from −10 to −40 (probabilities unchanged: 0.3 / 0.7). Does the preference between `g` and `s` flip? Show the arithmetic.

<details>
<summary>Show Answer</summary>

```
E[u|g] = (100 × 0.3) + (−40 × 0.7) = 30 − 28 = 2
E[u|s] = 20   (unchanged)
```
`2 < 20` → **Yes, it flips! Now prefer s.** This is the exact same lesson as the Numerical file §3 ("Odds don't flip you — Outcomes do"): the probabilities never moved, but making the downside big enough is all it takes to reverse the ranking.
</details>

---

## 4. Compound Lottery Practice

**Q4.1.** A compound lottery: first, Nature picks branch X (prob 0.4) or branch Y (prob 0.6). Under branch X, Nature then picks payoff 50 (prob 0.8) or payoff 0 (prob 0.2). Under branch Y, Nature picks payoff 50 (prob 0.25) or payoff 0 (prob 0.75). Reduce this to a simple lottery over {50, 0}.

<details>
<summary>Show Answer</summary>

```
Path X→50: 0.4 × 0.8 = 0.32
Path X→0:  0.4 × 0.2 = 0.08
Path Y→50: 0.6 × 0.25 = 0.15
Path Y→0:  0.6 × 0.75 = 0.45

P(50) = 0.32 + 0.15 = 0.47
P(0)  = 0.08 + 0.45 = 0.53

Check: 0.47 + 0.53 = 1.00 ✓

Reduced simple lottery:  0.47 → 50   |   0.53 → 0
```
</details>

---

**Q4.2.** Using the reduced lottery from Q4.1 and `u(x) = x`, what is its expected utility? Would a rational individual prefer this lottery to a guaranteed payoff of 25?

<details>
<summary>Show Answer</summary>

```
E[u] = (50 × 0.47) + (0 × 0.53) = 23.5
```
`23.5 < 25` → **The guaranteed 25 is preferred** — it's slightly better than the compound lottery's expected value.
</details>

---

## 5. Mini Project — Build Your Own Lottery

No spoiler answer here — this one's genuinely yours to build. It's the kind of open-ended task that shows up as a "design + justify" exam question or assignment.

**Task:**
1. Invent your own decision problem `(A, X, ≿)` with at least 3 outcomes. State whether `X` is discrete or continuous.
2. Build a `u: X → ℝ` that represents your `≿`, following the worked proof method in [Lecture 01 — Numerical §2](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_numerical.md#2-utility-representation--full-worked-proof) (find best/worst, form indifference classes, assign increasing numbers).
3. Design ONE simple lottery `p` over your outcomes (make sure the probabilities sum to 1!) and compute `E[u(X)|p]`.
4. Design a SECOND lottery over the same outcomes, and use expected utility to say which one a rational individual with your `u` would prefer.
5. **Bonus:** turn one branch of one lottery into a nested (compound) lottery, then reduce it back to a simple lottery, exactly like §4 of the Numerical file.

> 🍼 Tip: this is much easier with a fun, concrete story (like the ice-cream, movie, or scratch-card examples above) than with abstract symbols — build the story first, then attach the math.

---

## 6. Exam-Style Q&A Bank

Rapid-fire recall questions — good for a last-minute review pass. Try to answer in one line before checking.

<details>
<summary><b>Q. What are the three parts of a decision tree?</b></summary>

Nodes (where the individual resides/decides), Branches/Edges (actions taken), Leaves (payoffs of the full action sequence).
</details>

<details>
<summary><b>Q. State the two validity conditions for a discrete simple lottery `p|ₐ`.</b></summary>

Each `p(xᵢ|a) ∈ [0, 1]`, and `Σᵢ p(xᵢ|a) = 1`.
</details>

<details>
<summary><b>Q. State the four validity conditions for a continuous lottery's CDF `Fₐ`.</b></summary>

`Fₐ(x)→0` as `x→−∞`; `Fₐ(x)→1` as `x→+∞`; non-decreasing; right-continuous.
</details>

<details>
<summary><b>Q. What is a compound lottery, and how do you evaluate one?</b></summary>

A lottery whose prizes are themselves lotteries. Evaluate by reducing it to a simple lottery first: multiply probabilities along each path (chain rule), then sum the probabilities of paths landing on the same final outcome (law of total probability) — then apply the ordinary expected-utility formula.
</details>

<details>
<summary><b>Q. Write the discrete expected-utility formula.</b></summary>

`E[u(X)|p|ₐ] = Σᵢ₌₁ⁿ u(xᵢ) · p(xᵢ|a)`
</details>

<details>
<summary><b>Q. Write the continuous expected-utility formula.</b></summary>

`E[u(X)|Fₐ] = ∫₋∞^∞ u(x) dFₐ(x)`
</details>

<details>
<summary><b>Q. Can two lotteries with identical probabilities but different payoff values have different rankings? Give the intuition.</b></summary>

Yes. Expected utility depends on BOTH the probabilities AND the actual payoff values — changing just one outcome's payoff (holding probabilities fixed) can flip which lottery has the higher expected utility. See Numerical §3 for a fully worked example.
</details>

<details>
<summary><b>Q. In a decision tree with an added cost (e.g. distance/effort), where does the cost get applied?</b></summary>

Subtracted from the leaf payoff (i.e., it affects `v(a)`), not applied to any probability. Costs and payoffs live in the same units as `u`/`v`; probabilities are untouched.
</details>

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--practice)

---

> **← Back:** [🧮 NUMERICAL](aai_lec02_decision_trees_lotteries_numerical.md) · [📖 THEORY](aai_lec02_decision_trees_lotteries_theory.md) · [🏠 Lec 02 README](README.md) · [⬅️ Lecture 01](../Lec_01_Individual_Decision_Problem/README.md)
>
> *Advanced AI · Lec 02 · github.com/rpaut03l/TS-02-03*
