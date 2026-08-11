# 🎓 Advanced AI — Chapter 1 Exercises: Individual Rational Decision Paradigm

### *Seven fully worked problems — decision trees, budget constraints, calculus optimization, and preference orderings*

> **Nav:** [← Advanced AI](../README.md) | **Chapter 1 Exercises**

---

## 👶 30-second story

Every exercise in this folder is the same core skill from [Lecture 01](../Lec_01_Individual_Decision_Problem/README.md) and [Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md), applied to a real-feeling scenario: figure out the action set `A`, figure out the outcomes `X`, attach payoffs `u`, and pick the action that maximizes `v(a)`. Some exercises use a short discrete action list (movies, fruit trees), some use a budget-constrained combinatorial list (fruit or candy, buying a car), and some use a fully continuous action set that needs calculus — take a derivative, set it to zero, check the second derivative — to find the optimum (alcohol consumption, city parks). All seven are variations on one theme: **list your options, know what each one is worth to you, pick the biggest number.**

---

## 📁 Files in this folder

| # | Exercise | File | Core skill |
|---|---|---|---|
| 1.1 | Your Decision (template exercise) | [aai_ch1_ex01_your_decision.md](aai_ch1_ex01_your_decision.md) | Formalizing any everyday decision as `(A, X)` and a decision tree |
| 1.2 | Going to the Movies | [aai_ch1_ex02_going_to_movies.md](aai_ch1_ex02_going_to_movies.md) | Decision tree, alphabetic preferences, distance-cost twist |
| 1.3 | Fruit or Candy | [aai_ch1_ex03_fruit_or_candy.md](aai_ch1_ex03_fruit_or_candy.md) | Budget-constrained combinatorial action sets, diminishing returns |
| 1.4 | Alcohol Consumption | [aai_ch1_ex04_alcohol_consumption.md](aai_ch1_ex04_alcohol_consumption.md) | Continuous optimization via calculus (parabola, first/second derivative) |
| 1.5 | Buying a Car | [aai_ch1_ex05_buying_a_car.md](aai_ch1_ex05_buying_a_car.md) | Threshold-based preference construction, utility non-uniqueness |
| 1.6 | Fruit Trees | [aai_ch1_ex06_fruit_trees.md](aai_ch1_ex06_fruit_trees.md) | Combinatorial action sets, payoff-rule changes flipping the optimum |
| 1.7 | City Parks | [aai_ch1_ex07_city_parks.md](aai_ch1_ex07_city_parks.md) | Continuous optimization with a binding budget cap (corner solution) |

---

## 🧭 How to use this folder

1. Each file is self-contained: problem restated in your own words, the full `(A, X, ≿)` / `v(a)` setup, every calculation shown step by step, a decision tree or diagram where relevant, and a short cheat-sheet/mnemonic block at the end.
2. Work the problem yourself first with pen and paper — these exercises reward practicing the mechanics, not just reading the answer.
3. Every exercise cross-references the exact Lecture 01/02/03 section that supplies the tool being used (utility representation, expected value, risk attitudes, etc.) — follow those links if a step feels unfamiliar.

---

## 🧠 The Three Solution Patterns Used Across This Chapter

```
┌──────────────────────────────────────────────────────────────────┐
│  PATTERN 1 — SHORT DISCRETE LIST (1.1, 1.2, 1.6)                 │
│  List every option by hand, compute v(a) for each, pick max.     │
│                                                                  │
│  PATTERN 2 — BUDGET-CONSTRAINED COMBINATIONS (1.3, 1.5)          │
│  List every combination that fits the budget, compute total      │
│  value (or use a "net value per item" shortcut), pick max.       │
│                                                                  │
│  PATTERN 3 — CONTINUOUS ACTION SET (1.4, 1.7)                    │
│  a is any real number in a range. Take v'(a), set to 0, solve    │
│  for a*, confirm v''(a) < 0 (a maximum), check a* fits any cap.  │
└──────────────────────────────────────────────────────────────────┘
```

Recognizing which pattern a problem uses is 80% of solving it correctly — the rest is careful arithmetic.

---

## 🔗 Related reading & cross-references

- **[Lecture 01 — Individual Rational Decision Paradigm](../Lec_01_Individual_Decision_Problem/README.md)** — the `(A, X, ≿)`, `u`, and `v(a) = u(x*(a))` machinery every exercise here uses directly.
- **[Lecture 02 — Decision Trees, Risk & Lotteries](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)** — the decision-tree drawing convention (nodes/branches/leaves) used in 1.1, 1.2, 1.5, and 1.6.
- **[Lecture 03 — Risk Attitudes & Rational Choice Under Uncertainty](../Lec_03_Risk_Attitudes_Uncertainty/README.md)** — not directly needed for these Chapter 1 exercises (none involve randomness), but the natural next step once you're comfortable with certainty-based decisions.

---

## 🗺️ Exercise ↔ Lecture Concept Map

| Exercise | Primary Lecture Tool Used | Specific Section |
|---|---|---|
| 1.1 Your Decision | `(A, X)` formalization | [Lec 01 §1](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#1-the-individual-decision-problem) |
| 1.2 Going to the Movies | Decision tree + leaf-level cost | [Lec 02 §1](../Lec_02_Decision_Trees_Risk_Lotteries/aai_lec02_decision_trees_lotteries_theory.md#1-decision-trees) |
| 1.3 Fruit or Candy | Combinatorial action sets, `v(a)` maximization | [Lec 01 §5](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#5-rational-choice-assumption--the-value-function) |
| 1.4 Alcohol Consumption | Continuous optimization (calculus) | [Lec 01 §5](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#5-rational-choice-assumption--the-value-function) |
| 1.5 Buying a Car | Preference construction, utility non-uniqueness | [Lec 01 §3–§4](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#4-the-utility-representation-theorem) |
| 1.6 Fruit Trees | Combinatorial decision tree, rule-change ripple effects | [Lec 02 §1](../Lec_02_Decision_Trees_Risk_Lotteries/aai_lec02_decision_trees_lotteries_theory.md#1-decision-trees) |
| 1.7 City Parks | Continuous optimization with a binding cap (corner solution) | [Lec 01 §5](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#5-rational-choice-assumption--the-value-function) |

---

## 📝 Summary

This folder is where every abstract tool from Lecture 01 and Lecture 02 gets tested against something that at least *feels* like a real decision — buying a used car, planting trees, deciding how much to drink or spend. Three solution patterns cover every problem here: short discrete lists you can enumerate by hand, budget-constrained combinatorial grids that need systematic enumeration plus a "net value" shortcut, and continuous action sets that require actual calculus to optimize correctly. The two exercises that pack the sharpest lessons are 1.6 and 1.7 — one shows how a narrowly-targeted rule change can completely flip an optimal decision without touching most of the option table, and the other shows why every calculus-derived optimum must be checked against real-world constraints before it can be trusted. Taken together, these seven exercises prove that the entire Lecture 01–02 toolkit — list your actions, know your outcomes, assign honest payoffs, and pick the biggest number (subject to whatever constraints actually apply) — scales cleanly from toy examples all the way up to genuinely tricky, multi-step real-world decisions.

---

> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
