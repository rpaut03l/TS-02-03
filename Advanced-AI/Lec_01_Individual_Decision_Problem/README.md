# 🎓 Advanced AI — Lec 01: Individual Rational Decision Paradigm

### *Actions · Outcomes · Preferences · Rationality · Utility Representation · Rational Choice*

> **Nav:** [← Advanced AI](../README.md) | **Lec 01** | [Lec 02 ➡️](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)

---

## 👶 30-second story

Imagine a kid in front of a menu with different options (**Actions**), some idea of what she'll actually get if she orders each thing (**Outcomes**), and her own private list of "I like this more than that" (**Preferences**). If her list never contradicts itself (no un-ranked pairs, no ranking loops — that's **rationality**), we can replace it with plain numbers (**Utility**) and just tell her: "always pick the biggest number" (**Rational Choice**).

This is the bedrock of the whole Advanced AI course. Before you can reason about *two or more* players outsmarting each other (actual game theory), you need an airtight definition of what "rational" means for just **one** decision-maker with **no** randomness involved yet. That's exactly this lecture. Randomness — dice-rolling "Nature," decision trees, and lotteries — is saved for [Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md).

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [aai_lec01_decision_problem_theory.md](aai_lec01_decision_problem_theory.md) | The full concepts — decision problem, preference relations, rationality (completeness + transitivity), the utility representation theorem (with proof sketch), rational choice assumption, and the value function `v(a) = u(x*(a))` |
| [aai_lec01_decision_problem_numerical.md](aai_lec01_decision_problem_numerical.md) | **Every definition worked out with real numbers** — checking rationality on a concrete example, a full utility-representation proof (including the tied-outcomes trap), and a worked u-vs-v rational-choice example |
| [aai_lec01_decision_problem_practice.md](aai_lec01_decision_problem_practice.md) | **Self-test problems with spoiler-tag answers** — concept checks, rationality drills, utility-representation drills, an open-ended mini task, and a rapid-fire exam Q&A bank |

---

## 🎯 After this lecture you should be able to…

- State the 3 components of an individual decision problem: `A` (actions), `X` (outcomes), `≿` (preferences)
- Write the formal definitions of weak preference `≿`, strict preference `≻`, and indifference `∼`
- Check whether a given preference relation is **rational** (complete + transitive), and explain what breaks (money-pump loops, decision paralysis) when it isn't
- State and apply the **Utility Representation Theorem**, including its finiteness condition, and construct a representing `u` by hand (including correctly handling tied outcomes)
- List the 4 things a **Rational Choice Assumption** requires an individual to know
- Distinguish `u` (utility over outcomes) from `v` (payoff over actions), and write `v(a) = u(x*(a))`
- Apply the rational choice rule: pick `a*` such that `v(a*) ≥ v(a)` for all `a ∈ A`

---

## 🧭 How to use this folder

1. **Read `theory.md` first**, top to bottom. Every section has a quick story before the formal definitions — get the intuition before the symbols.
2. **Go through `numerical.md`.** Every definition and theorem gets a fully worked, step-by-step numeric example here — don't just read it, redo it yourself on paper.
3. **Attempt `practice.md` before peeking at the answers.** Each problem is inside a spoiler tag — genuinely try it first.
4. Use the **Cheat Sheet & Exam Hacks** section at the bottom of `theory.md` for a last-minute review pass.
5. Once comfortable, move on to **[Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)**, where randomness enters the picture.

---

## 🔗 Related reading & cross-references

- **[Advanced AI subject README](../README.md)** — full topic roadmap for this track.
- **[Lecture 02 — Decision Trees, Risk & Lotteries](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)** — the direct continuation: this lecture's `u`, `v`, and "biggest number wins" logic gets upgraded to expected utility once outcomes become uncertain.
- **[TS-01 / AI — full syllabus](https://github.com/rpaut03l/TS-01/tree/main/AI)** — the Trimester-1 companion track (search, CSPs, adversarial search, logic, planning, RL), same documentation style as this folder.
- **[TS-01 / AI — Topic 19: MDP & Policy](https://github.com/rpaut03l/TS-01/tree/main/AI/19_RL_MDP_Policy)** — background reading: Markov Decision Processes generalize this lecture's single-shot rational-choice framework into a *sequential* decision problem — useful once this course moves past one-shot decisions.

### 📖 Course textbook reference
This lecture's material (individual decision problems, rational preferences, utility representation) follows the standard opening section of most game-theory textbooks (e.g., Tadelis, *Game Theory: An Introduction*, Ch. 1) before those texts move on to risk and then strategic (multi-player) games. Treat this folder as an ELI5-annotated, fully-worked companion — use the textbook for formal proofs and extra exercises, this folder for intuition and step-by-step numeric practice.

### 🎯 Where these concepts appear in the Chapter 1 Exercises
- **[Exercise 1.1](../Exercises_Chapter1_Decision_Theory/aai_ch1_ex01_your_decision.md)** — directly practices §1's `(A, X)` formalization
- **[Exercise 1.5](../Exercises_Chapter1_Decision_Theory/aai_ch1_ex05_buying_a_car.md)** — builds a preference relation by hand (§2–§3) and demonstrates §4's utility non-uniqueness
- Every exercise in the folder uses §5's `v(a) = u(x*(a))` maximization rule as its final decision step

---

## 📝 Summary

Lecture 01 is the foundation the entire Advanced AI course is poured on top of. It takes the everyday, fuzzy idea of "making a good decision" and turns it into a precise, three-part object — actions, outcomes, and preferences — and then asks exactly what has to be true about your preferences for a mathematician to call you "rational": complete (no un-ranked pairs) and transitive (no contradictory loops). The lecture's biggest single result, the Utility Representation Theorem, is a genuine gift — it means you almost never have to work with clunky preference symbols directly, since any rational preference over a finite set can be replaced by ordinary numbers that behave exactly the same way. Everything closes with the Rational Choice Assumption, a checklist of four things a decision-maker must know, and the payoff-maximizing choice rule that follows from it. This lecture assumes a world with zero uncertainty — every action leads to exactly one guaranteed outcome — which is precisely the assumption Lecture 02 removes next, and precisely the toolkit the Chapter 1 Exercises folder puts to work across seven different real-feeling scenarios.

---

> *Advanced AI · Lec 01 · github.com/rpaut03l/TS-02-03*
