# 🎓 Advanced AI — Lec 01: Rational Choice, Preferences & Risk

### *Individual Decision Problem · Preferences · Utility · Decision Trees · Lotteries · Expected Utility*

> **Nav:** [← Advanced AI](../README.md) | **Lec 01**

---

## 👶 30-second story

Imagine a kid in front of a menu with different options (**Actions**), some idea of what she'll actually get if she orders each thing (**Outcomes**), and her own private list of "I like this more than that" (**Preferences**). If her list never contradicts itself, we can replace it with plain numbers (**Utility**) and just tell her: "always pick the biggest number." That's the whole first half of this lecture.

The second half adds one twist: sometimes ordering something doesn't guarantee what you get — **Nature** rolls dice and decides for you (a **lottery**). Now "biggest number" becomes "biggest *average* number, weighted by how likely each result is" (**Expected Utility**). That's the whole second half.

This is the mathematical foundation the rest of Advanced AI (game theory, strategic interaction, equilibrium concepts) is built on top of — before you can reason about *two or more* rational players, you first need a rock-solid definition of what "rational" means for just *one*.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [aai_lec01_rational_choice_theory.md](aai_lec01_rational_choice_theory.md) | The full concepts — decision problem, preference relations, rationality, utility representation theorem, rational choice, decision trees, lotteries (discrete + continuous), compound lotteries, expected utility |
| [aai_lec01_rational_choice_numerical.md](aai_lec01_rational_choice_numerical.md) | **Every number worked out step by step** — the full movie-theatre decision tree (with a walking-cost twist), expected-utility comparisons, a preference-flip demonstration, a compound-lottery reduction, and a mini worked utility-representation proof |
| [aai_lec01_rational_choice_practice.md](aai_lec01_rational_choice_practice.md) | **Self-test problems with spoiler-tag answers** — concept checks, rationality drills, decision-tree practice, expected-utility drills, a compound-lottery drill, an open-ended mini project, and a rapid-fire exam Q&A bank |

---

## 🎯 After this lecture you should be able to…

- State the 3 components of an individual decision problem: `A` (actions), `X` (outcomes), `≿` (preferences)
- Write the formal definitions of weak preference `≿`, strict preference `≻`, and indifference `∼`
- Check whether a given preference relation is **rational** (complete + transitive)
- State and apply the **Utility Representation Theorem**, including its finiteness condition
- Distinguish `u` (utility over outcomes) from `v` (payoff over actions), and write `v(a) = u(x*(a))`
- List the 4 things a **Rational Choice Assumption** requires an individual to know
- Draw and solve a **decision tree** (nodes, branches, leaves) including how added costs can change the optimal choice
- Define a **simple lottery**, both for discrete outcomes (`p|ₐ ∈ Δ(X)`) and continuous outcomes (`Fₐ ∈ Δ(X)`), and state their validity rules
- Define a **compound lottery** and reduce it to an equivalent simple lottery
- Write and apply the **expected utility** formula, discrete and continuous, and explain why changing an outcome's payoff (not just its probability) can flip a preference ranking

---

## 🧭 How to use this folder

1. **Read `theory.md` first**, top to bottom. Every section has a quick story before the formal definitions — get the intuition before the symbols.
2. **Go through `numerical.md`.** Every formula in the theory file gets a fully worked, step-by-step numeric example here — don't just read the arithmetic, redo it yourself on paper.
3. **Attempt `practice.md` before peeking at the answers.** Each problem is inside a spoiler tag — genuinely try it first, that's where the learning happens.
4. Use the **Cheat Sheet & Exam Hacks** section at the bottom of `theory.md` (and the Formula Cheat Sheet at the bottom of `numerical.md`) for a last-minute review pass.

---

## 🔗 Related reading & cross-references

### 🧠 From this same repo (TS-02-03)
- This lecture is **Lec 01** of the Advanced AI track. See the [Advanced AI subject README](../README.md) for the full topic roadmap and how later lectures (games, equilibrium) build on these definitions.

### 📘 From the companion Trimester-1 repo (TS-01)
- **[TS-01 / AI — full syllabus](https://github.com/rpaut03l/TS-01/tree/main/AI)** — the Trimester-1 companion track covering classical AI: search, CSPs, adversarial search, logic, planning, probabilistic reasoning, and reinforcement learning, all in the same ELI5 + worked-example + cheat-sheet style used here.
- **[TS-01 / AI — Topic 08: Adversarial Search (Minimax)](https://github.com/rpaut03l/TS-01/tree/main/AI/08_Adversarial_Search_Minimax)** — directly relevant: minimax game trees are literally a **two-player extension** of the single-player decision trees in this lecture's §6. Where this lecture has one decision-maker picking the biggest payoff, minimax has two decision-makers alternately picking biggest/smallest. Read this after finishing this folder to see the natural next step toward actual *game* theory.
- **[TS-01 / AI — Topic 10: Expectimax Search](https://github.com/rpaut03l/TS-01/tree/main/AI/10_Expectimax_Search)** — directly relevant: expectimax adds "chance nodes" to a game tree that average over outcomes by probability, which is *exactly* the expected-utility idea from this lecture's §10, just embedded inside a multi-step search tree instead of a one-shot lottery.
- **[TS-01 / AI — Topic 19: MDP & Policy](https://github.com/rpaut03l/TS-01/tree/main/AI/19_RL_MDP_Policy)** — background reading: Markov Decision Processes generalize this lecture's single-shot rational-choice framework into a *sequential*, multi-period decision problem under uncertainty — useful context once Advanced AI moves past one-shot games.

### 📖 Course textbook reference
This lecture's material (individual decision problems, rational preferences, utility representation, expected utility under risk) follows the standard **decision-theory foundations chapter** that opens most game-theory textbooks (e.g., Tadelis, *Game Theory: An Introduction*, Ch. 1) before those texts move on to strategic (multi-player) games. If you're using that textbook alongside the slides, treat this folder as your own from-scratch, ELI5-annotated version of that chapter — work the examples here, then cross-check tricky definitions against your copy of the book directly.

---

> *Advanced AI · Lec 01 · github.com/rpaut03l/TS-02-03*
