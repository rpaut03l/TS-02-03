# Advanced AI

### *Game theory & strategic decision-making — notes, worked numbers, and practice*

> 🔗 **Repo:** [github.com/rpaut03l/TS-02-03](https://github.com/rpaut03l/TS-02-03) · Advanced AI Track
>
> **Style:** Every topic explained (easy story + picture), then the formal definitions, then fully worked numbers, then self-test practice. Same trio pattern used across this repo's MLOps and GPU Programming tracks.

---

## 🧠 What even is Advanced AI (this track)?

### 👶 Easy Story

Trimester-1's `AI` track (in the companion [TS-01](https://github.com/rpaut03l/TS-01/tree/main/AI) repo) was about **one agent** figuring out the smartest move against a fixed, known environment — search a maze, satisfy some constraints, plan a route. Topic 08 there even had a taste of "an opponent" with Minimax.

**Advanced AI zooms all the way into that opponent.** Before you can reason about two or more agents outsmarting each other (real game theory — strategic games, Nash equilibrium, and beyond), you first need an airtight definition of what a single "rational" decision-maker even means: what they can do, what can happen, how they rank outcomes, and how they handle pure luck (Nature rolling dice on their behalf). That foundation is exactly what Lec 01 in this folder builds, brick by brick.

```
TS-01 / AI     = one agent, known environment           (solo puzzle-solving)
Advanced AI    = one rational agent, precisely defined   (this folder's Lec 01)
                 → then MULTIPLE rational agents,
                   each reasoning about the others        (later lectures: games!)
```

---

## 📁 Contents of this folder

| # | Lecture | Folder |
|---|---|---|
| 1 | **Rational Choice, Preferences & Risk** — Individual Decision Problem, Preference Relations, Rationality, Utility Representation, Decision Trees, Simple & Compound Lotteries, Expected Utility | [Lec_01_Rational_Choice_Risk_Lotteries/](Lec_01_Rational_Choice_Risk_Lotteries/) |

Each lecture folder has the same **trio** of files:

| File | Purpose |
|---|---|
| `*_theory.md` | **Concepts explained like a kid** — easy stories, analogies, boxed ASCII diagrams, exact formal notation, then the technical depth |
| `*_numerical.md` | **Every formula worked out with real numbers**, step by step — the numeric backbone for exam-style problems |
| `*_practice.md` | **Self-test problems with spoiler-tag answers**, plus a rapid-fire exam Q&A bank and an open-ended mini project |

---

## 🧭 How to use this folder

1. **Read `theory.md` first.** Get the story and the exact notation before touching numbers.
2. **Work through `numerical.md`.** Redo every worked example yourself on paper — don't just read the arithmetic.
3. **Attempt `practice.md` before checking answers.** Every problem is hidden behind a spoiler tag on purpose.
4. **Use the Cheat Sheet & Exam Hacks** sections (end of `theory.md` and `numerical.md`) for a final review pass.

---

## 📚 Topic roadmap

Planned topics in this track (more folders will be added as the course progresses):

- ✅ **Lec 01** — Rational Choice, Preferences & Risk (individual decision theory foundations)
- 🔭 Strategic-form games & Nash Equilibrium
- 🔭 Dominant strategies & iterated elimination
- 🔭 Mixed strategies
- 🔭 Extensive-form games & backward induction
- 🔭 Repeated games
- 🔭 Bayesian games & incomplete information

---

## 🔗 External Resources & Cross-References

### 📘 Companion Trimester-1 repo (TS-01)
- **[TS-01 / AI — full syllabus](https://github.com/rpaut03l/TS-01/tree/main/AI)** — the prerequisite track: search, CSPs, adversarial search (Minimax, Alpha-Beta, Expectimax), logic, planning, probabilistic reasoning, reinforcement learning. Same documentation style as this folder.
- **[TS-01 / AI — Topic 08: Adversarial Search (Minimax)](https://github.com/rpaut03l/TS-01/tree/main/AI/08_Adversarial_Search_Minimax)** — the natural bridge into game theory: a MAX player and a MIN player alternately optimizing a shared game tree is a first, informal taste of the strategic reasoning this Advanced AI track formalizes.
- **[TS-01 / AI — Topic 10: Expectimax Search](https://github.com/rpaut03l/TS-01/tree/main/AI/10_Expectimax_Search)** — chance nodes that average outcomes by probability are the search-tree cousin of this track's expected-utility lotteries.
- **[TS-01 / AI — Topic 19: MDP & Policy](https://github.com/rpaut03l/TS-01/tree/main/AI/19_RL_MDP_Policy)** — extends single-shot rational choice (this track's Lec 01) into sequential decision-making under uncertainty.
- **[TS-01 / ML — algorithm fundamentals](https://github.com/rpaut03l/TS-01/tree/main/ML)** — if a later Advanced AI topic leans on a specific ML technique (e.g. for learning in games), that track has the theory + worked numericals + practice code for it.

### 📖 Textbook
This track's foundational material follows the standard opening chapter of most game-theory textbooks (individual decision problems → rational preferences → utility representation → risk and expected utility), the same structure used in texts such as **Tadelis, *Game Theory: An Introduction*, Chapter 1**. Treat this folder as an ELI5-annotated, fully-worked companion to that material — use the textbook for the formal proofs and additional exercises, and this folder for intuition-first explanations and step-by-step numeric practice.

---

> *github.com/rpaut03l/TS-02-03*
