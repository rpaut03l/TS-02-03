# 🎓 Advanced AI — Lec 04: Decision Over Time & Value of Information

### *Sequential Decisions · Backward Induction · Discounting · Value of Information · Intertemporal Consumption*

> **Nav:** [← Advanced AI](../README.md) | [⬅️ Prev: Lec 03](../Lec_03_Risk_Attitudes_Uncertainty/README.md) | **Lec 04**

---

## 👶 30-second story

Every decision so far in this course happened in one shot — you picked an action, maybe Nature rolled some dice, and you found out what you got. This lecture breaks that mold in the most realistic way possible: **real decisions often unfold in stages, with new information arriving in between.** A company can gamble on R&D and only decide how aggressively to market the result *after* learning whether the research actually worked. That single shift — letting a later decision depend on something learned after an earlier one — is genuinely new machinery, and it's solved with a technique called **backward induction**: work out your best move at the very last decision point first, treat that as a known number, and fold the tree backward one step at a time until you reach the beginning.

Three more ideas ride along with this. **Discounting** captures the plain fact that money (or any reward) received later is worth a little less than the same reward received today, formalized with a single number, `δ`. The **Value of Information** puts an exact price tag on something intuitively obvious — that knowing more before you decide is never bad and often very good — and proves it's always a non-negative number. And a full **intertemporal consumption model** shows all of this working together with real calculus: split a fixed resource across two time periods, and the resulting optimality condition (the famous Euler equation) elegantly explains why patient people split evenly while impatient people front-load consumption toward the present.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [aai_lec04_decision_over_time_theory.md](aai_lec04_decision_over_time_theory.md) | The full concepts — sequential decisions, backward induction (with a full Mermaid decision tree), discounting and threshold analysis, the Value of Information, and the intertemporal consumption model |
| [aai_lec04_decision_over_time_numerical.md](aai_lec04_decision_over_time_numerical.md) | **Every number worked out step by step** — the complete R&D backward-induction solve, the discounting threshold derivation, the full MBA Value-of-Information calculation (decomposed state by state), and the consumption-optimization Euler equation solved with real numbers |
| [aai_lec04_decision_over_time_practice.md](aai_lec04_decision_over_time_practice.md) | **Self-test problems with spoiler-tag answers**, using entirely fresh numbers — backward induction drills, discounting threshold drills, Value of Information drills, consumption drills, an open-ended mini project, and a rapid-fire exam Q&A bank |

---

## 🎯 After this lecture you should be able to…

- Solve any multi-stage decision tree using **backward induction**: solve the last decision nodes first, collapse each into a value, and fold the tree back to the root
- Explain why the output of backward induction is a full **contingent strategy**, not a single fixed action
- Set up and solve for a **discount-factor threshold** where two time-sensitive options become equally attractive
- Define the **Value of Information** formula and explain, using the state-by-state decomposition method, exactly where a positive VOI comes from
- State and interpret the **Euler equation** for two-period consumption, `u'(x₁) = δ·u'(K−x₁)`, including the special case `δ=1`

---

## 🧭 How to use this folder

1. **Read `theory.md` first**, top to bottom — every section includes a Mermaid decision-tree diagram alongside the story and formal notation.
2. **Work through `numerical.md`.** Every calculation is broken into small, independently-checkable steps — redo each one on paper.
3. **Attempt `practice.md` before peeking at the answers.** All problems use fresh numbers you haven't already seen solved.
4. Use the **Cheat Sheet & Exam Hacks** section at the bottom of `theory.md` for a last-minute review pass.

---

## 🔗 Related reading & cross-references

### 🧠 From this same repo (TS-02-03)
- **[Lecture 01 — Individual Rational Decision Paradigm](../Lec_01_Individual_Decision_Problem/README.md)** — the `(A, X, ≿)`, `u`, and `v(a)` machinery this lecture's every decision node still relies on.
- **[Lecture 02 — Decision Trees, Risk & Lotteries](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)** — the node/branch/leaf drawing convention this lecture's Mermaid trees follow, and the expected-utility formula every backward-induction step applies.
- **[Lecture 03 — Risk Attitudes & Rational Choice Under Uncertainty](../Lec_03_Risk_Attitudes_Uncertainty/README.md)** — the concave-utility idea (`u''<0`) reused directly in this lecture's consumption model.
- **[Advanced AI subject README](../README.md)** — full topic roadmap for this track.

### 📘 From the companion Trimester-1 repo (TS-01)
- **[TS-01 / AI — full syllabus](https://github.com/rpaut03l/TS-01/tree/main/AI)** — the Trimester-1 companion track, same documentation style.
- **[TS-01 / AI — Topic 19: MDP & Policy](https://github.com/rpaut03l/TS-01/tree/main/AI/19_RL_MDP_Policy)** — directly relevant: Markov Decision Processes are the natural generalization of this lecture's two-stage backward induction into arbitrarily many sequential stages, and MDP "policies" are exactly this lecture's "contingent strategies," formalized further.
- **[TS-01 / AI — Topic 08: Adversarial Search (Minimax)](https://github.com/rpaut03l/TS-01/tree/main/AI/08_Adversarial_Search_Minimax)** — Minimax is also solved via backward induction (there called "backing up values" through the game tree) — the same core algorithmic idea, applied to a two-player adversarial setting instead of a single player facing Nature.

### 📖 Course textbook reference
This lecture's material (sequential decisions, backward induction, discounting, value of information, and intertemporal consumption) follows the standard "dynamic decision-making" section that typically closes out the single-agent decision-theory portion of game-theory textbooks (e.g., Tadelis, *Game Theory: An Introduction*, Ch. 1–2, later sections) before the course moves into genuine multi-player strategic games. Treat this folder as an ELI5-annotated, fully-worked companion — use the textbook for formal proofs and additional exercises, and this folder for intuition and step-by-step numeric practice.

---

## 📝 Summary

Lecture 04 is where the single-agent decision-theory portion of this course reaches its natural peak of realism: decisions that unfold over multiple stages, rewards that lose value the longer you wait for them, and the genuine option to learn something useful before committing. Backward induction supplied the one new algorithmic tool needed to handle staged decisions correctly — solve the future first, then use that knowledge to solve the present — and it produces not just a number but a complete, adaptive plan of action. Discounting gave time itself a price, letting two options separated by delay be compared on equal footing through a single threshold value of δ. The Value of Information turned "knowing more helps" from an intuition into an exact, always-non-negative formula, and its state-by-state decomposition revealed precisely which situations make information valuable and which don't. The intertemporal consumption model tied everything together with real calculus, producing the Euler equation as a clean, general rule connecting patience (δ) to how resources should be split across time. Every one of these four techniques — sequencing, discounting, information, and intertemporal optimization — becomes essential once the course moves into genuine multi-player strategic games, where every player is simultaneously trying to sequence their own moves, discount their own future payoffs, and infer information from what other players do.

---

> *Advanced AI · Lec 04 · github.com/rpaut03l/TS-02-03*
