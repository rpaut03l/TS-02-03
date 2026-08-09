# 🎓 Advanced AI — Lec 02: Decision Trees, Risk & Lotteries

### *Decision Trees · Nature · Simple & Compound Lotteries · Expected Utility*

> **Nav:** [← Advanced AI](../README.md) | [⬅️ Prev: Lec 01](../Lec_01_Individual_Decision_Problem/README.md) | **Lec 02**

---

## 👶 30-second story

[Lecture 01](../Lec_01_Individual_Decision_Problem/README.md) built a rational individual who always knows *exactly* what an action gets them — order chocolate, get chocolate, no surprises. Real decisions rarely work that way: you order a scratch card, and an invisible extra "player" called **Nature** rolls the dice to decide what you actually walk away with.

This lecture does three things:
1. **Decision Trees** — draws out multi-step choices (like "which theatre, then which movie") as a literal tree you can read top to bottom.
2. **Lotteries** — gives Nature's dice-rolling a precise mathematical shape: a **simple lottery** (a spinner with slices) for a finite list of outcomes, or a **CDF** for a continuous range; and a **compound lottery** for "a lottery whose prize is another lottery."
3. **Expected Utility** — upgrades Lecture 01's "just pick the biggest number" rule into "pick the biggest *average* number, weighted by how likely each outcome is" — the single formula this whole lecture builds toward.

This is the last stop before real *game theory* — once you can rank one person's choices under risk, the next step (coming in later lectures) is putting a **second** rational person in the room, each one reasoning about what the other will do.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [aai_lec02_decision_trees_lotteries_theory.md](aai_lec02_decision_trees_lotteries_theory.md) | The full concepts — decision trees, risk & Nature, simple lotteries (discrete + continuous), compound lotteries, and expected utility |
| [aai_lec02_decision_trees_lotteries_numerical.md](aai_lec02_decision_trees_lotteries_numerical.md) | **Every number worked out step by step** — the full movie-theatre decision tree (with a walking-cost twist), expected-utility comparisons, a preference-flip demonstration, and a compound-lottery reduction |
| [aai_lec02_decision_trees_lotteries_practice.md](aai_lec02_decision_trees_lotteries_practice.md) | **Self-test problems with spoiler-tag answers** — decision-tree practice, lottery concept checks, expected-utility drills, a compound-lottery drill, an open-ended mini project, and a rapid-fire exam Q&A bank |

---

## 🎯 After this lecture you should be able to…

- Draw and solve a **decision tree** (nodes, branches, leaves), including how added costs (e.g. distance/effort) can change the optimal choice
- Define a **simple lottery**, both for discrete outcomes (`p|ₐ ∈ Δ(X)`) and continuous outcomes (`Fₐ ∈ Δ(X)`), and state their validity rules
- Define a **compound lottery** and reduce it to an equivalent simple lottery (chain rule + law of total probability)
- Write and apply the **expected utility** formula, discrete and continuous
- Compare two lotteries and correctly apply the rule "prefer g over s ⟺ E[u|g] ≥ E[u|s]"
- Explain — and demonstrate numerically — why changing an outcome's *payoff* (not just its probability) can flip which lottery is preferred

---

## 🧭 How to use this folder

1. **Read `theory.md` first**, top to bottom. Every section builds directly on Lecture 01's `u`, `v`, and rational-choice rule.
2. **Go through `numerical.md`.** Every formula gets a fully worked, step-by-step numeric example — redo the arithmetic yourself.
3. **Attempt `practice.md` before peeking at the answers.** Spoiler tags are there on purpose.
4. Use the **Cheat Sheet & Exam Hacks** section at the bottom of `theory.md` (and the Formula Cheat Sheet at the bottom of `numerical.md`) for a last-minute review pass.

---

## 🔗 Related reading & cross-references

### 🧠 From this same repo (TS-02-03)
- **[Lecture 01 — Individual Rational Decision Paradigm](../Lec_01_Individual_Decision_Problem/README.md)** — the direct prerequisite: this lecture's `u`, `v`, and decision-tree leaves all come straight from Lec 01's definitions. If any notation here feels unfamiliar, that's where it's introduced.
- **[Advanced AI subject README](../README.md)** — full topic roadmap for this track (strategic games, Nash equilibrium, etc. still to come).

### 📘 From the companion Trimester-1 repo (TS-01)
- **[TS-01 / AI — Topic 08: Adversarial Search (Minimax)](https://github.com/rpaut03l/TS-01/tree/main/AI/08_Adversarial_Search_Minimax)** — directly relevant: minimax game trees are literally a **two-player extension** of the single-player decision trees in this lecture's §1. Where this lecture has one decision-maker picking the biggest payoff at each node, minimax has two decision-makers alternately picking biggest/smallest. Read this next to see the natural bridge toward actual *game* theory.
- **[TS-01 / AI — Topic 10: Expectimax Search](https://github.com/rpaut03l/TS-01/tree/main/AI/10_Expectimax_Search)** — directly relevant: expectimax adds "chance nodes" to a game tree that average over outcomes by probability — exactly this lecture's expected-utility idea (§5), just embedded inside a multi-step search tree instead of a one-shot lottery.
- **[TS-01 / AI — Topic 19: MDP & Policy](https://github.com/rpaut03l/TS-01/tree/main/AI/19_RL_MDP_Policy)** — background reading: Markov Decision Processes generalize this lecture's one-shot lottery framework into a *sequential*, multi-period decision problem under uncertainty.

### 📖 Course textbook reference
This lecture's material (decision trees, risk, lotteries, expected utility) follows the standard "decision theory under uncertainty" section that typically follows the deterministic-choice chapter in game-theory textbooks (e.g., Tadelis, *Game Theory: An Introduction*, Ch. 1, later sections). Treat this folder as an ELI5-annotated, fully-worked companion — use the textbook for formal proofs (e.g. the full Expected Utility Theorem / von Neumann–Morgenstern axioms) and extra exercises, this folder for intuition and step-by-step numeric practice.

---

> *Advanced AI · Lec 02 · github.com/rpaut03l/TS-02-03*
