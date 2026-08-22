# 🎓 Advanced AI — Lec 03: Risk Attitudes & Rational Choice Under Uncertainty

### *Risk Neutral/Averse/Loving · Jensen's Inequality · The St. Petersburg Paradox · Rational Choice Under Uncertainty*

> **Nav:** [← Advanced AI](../README.md) | [⬅️ Prev: Lec 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md) | **Lec 03** | [Lec 04 ➡️](../Lec_04_Decision_Over_Time_Value_of_Information/README.md)

---

## 👶 30-second story

[Lecture 02](../Lec_02_Decision_Trees_Risk_Lotteries/README.md) taught you how to compute the expected utility of a gamble. This lecture asks a sharper question: given two options with the **exact same** average payoff — a risky gamble and a guaranteed sure thing worth the same on average — would a rational person actually be indifferent between them?

The answer is: not necessarily, and *why not* is this whole lecture. Some people specifically prefer the safety of the guaranteed amount (**risk-averse**), some genuinely don't care either way (**risk-neutral**), and some actually prefer the thrill of the gamble (**risk-loving**). The shape of a person's utility function — bending downward, straight, or bending upward — completely determines which camp they fall into, and a neat piece of geometry called Jensen's Inequality explains exactly why.

The lecture then throws a genuinely famous curveball at you: the **St. Petersburg Paradox**, a simple coin-flip game whose "fair price" comes out to *infinity* under plain risk-neutral math — and yet no real person would pay more than a few dollars to play it. Solving that mismatch is exactly what motivated economists to invent expected utility (and risk aversion) in the first place, centuries ago. Finally, the lecture closes the loop back to Lecture 01: the rational-choice rule you learned there gets one small, clean upgrade to work correctly once uncertainty is involved.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [aai_lec03_risk_uncertainty_theory.md](aai_lec03_risk_uncertainty_theory.md) | The full concepts — risk attitudes and their formulas, Jensen's Inequality intuition, the St. Petersburg Paradox setup and its resolution, and the formal rational-choice rule under uncertainty |
| [aai_lec03_risk_uncertainty_numerical.md](aai_lec03_risk_uncertainty_numerical.md) | **Every number worked out step by step** — three utility functions tested against the same gamble, the full St. Petersburg derivation (including why `E[X]=∞`), the `ln(x)` fix and why `x²` fails, and a worked rational-choice-under-uncertainty comparison |
| [aai_lec03_risk_uncertainty_practice.md](aai_lec03_risk_uncertainty_practice.md) | **Self-test problems with spoiler-tag answers** — risk-attitude drills, St. Petersburg drills, rational-choice-under-uncertainty practice, an open-ended mini project, and a rapid-fire exam Q&A bank |

---

## 🎯 After this lecture you should be able to…

- State and correctly apply the three risk-attitude inequalities (Neutral `=`, Averse `≤`, Loving `≥`) comparing `E[u(X)]` to `u(E[X])`
- Explain, using Jensen's Inequality, why a concave utility function implies risk aversion and a convex one implies risk-loving behavior
- Set up and evaluate the St. Petersburg Paradox: derive why the naive expected value is infinite, and why that contradicts observed human behavior
- Resolve the paradox using a concave utility function (`u(x)=ln(x)`), including computing a finite certainty equivalent
- Explain why not every nonlinear utility function resolves the paradox — specifically why convex functions like `x²` make it worse
- State the updated Rational Choice Assumption and choice rule for decisions under uncertainty, and explain exactly how it differs from the certainty version in Lecture 01

---

## 🧭 How to use this folder

1. **Read `theory.md` first**, top to bottom — each section builds directly on Lec 02's expected utility formula.
2. **Go through `numerical.md`.** Every claim in the theory file gets a fully worked, step-by-step numeric example — redo the arithmetic yourself, especially the geometric-series steps in the St. Petersburg derivation.
3. **Attempt `practice.md` before peeking at the answers.**
4. Use the **Cheat Sheet & Exam Hacks** section at the bottom of `theory.md` for a last-minute review pass.

---

## 🔗 Related reading & cross-references

### 🧠 From this same repo (TS-02-03)
- **[Lecture 01 — Individual Rational Decision Paradigm](../Lec_01_Individual_Decision_Problem/README.md)** — the rational-choice rule this lecture's §5 directly extends.
- **[Lecture 02 — Decision Trees, Risk & Lotteries](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)** — the expected-utility formula (`E[u(X)|Fₐ]`) this entire lecture is built on top of.
- **[Advanced AI subject README](../README.md)** — full topic roadmap for this track.

### 📘 From the companion Trimester-1 repo (TS-01)
- **[TS-01 / ML — algorithm fundamentals](https://github.com/rpaut03l/TS-01/tree/main/ML)** — background reading: concave/convex function reasoning (Jensen's Inequality) also underlies convexity assumptions used throughout optimization-based ML algorithms.

### 📖 Course textbook reference
This lecture's material (risk attitudes, Jensen's Inequality, the St. Petersburg Paradox, and rational choice under uncertainty) follows the standard "risk and expected utility" section that typically closes out the introductory decision-theory chapter in game-theory textbooks (e.g., Tadelis, *Game Theory: An Introduction*, Ch. 1–2). Treat this folder as an ELI5-annotated, fully-worked companion — use the textbook for the formal Expected Utility Theorem / von Neumann–Morgenstern axioms and additional exercises, and this folder for intuition and step-by-step numeric practice.

### 🎯 Where these concepts appear in the Chapter 1 Exercises
None of the Chapter 1 exercises (1.1–1.7) involve randomness or lotteries — they're all certainty-based decisions, so they draw on Lecture 01 and Lecture 02's tree-drawing convention rather than this lecture's risk-attitude machinery. This lecture's tools (risk attitudes, Jensen's Inequality, the St. Petersburg Paradox) become directly relevant once the course moves into later chapters involving genuine uncertainty and strategic games under risk.

### ➡️ Where this lecture leads next
**[Lecture 04 — Decision Over Time & Value of Information](../Lec_04_Decision_Over_Time_Value_of_Information/README.md)** reuses this lecture's concave-utility idea (`u''(·) < 0`) directly in its intertemporal consumption model — the same "diminishing returns" reasoning that explained risk aversion here also explains why consumption should be smoothed evenly across time.

---

## 📝 Summary

Lecture 03 exists to answer a question that's easy to overlook once you've mastered expected utility: does averaging outcomes tell the whole story, or does uncertainty itself carry its own cost or thrill? Risk attitudes gave that question a precise answer — comparing `E[u(X)]` to `u(E[X])` sorts every person into risk-neutral, risk-averse, or risk-loving, and Jensen's Inequality explains the sorting with pure geometry: concave curves keep chords below them, convex curves keep chords above. The St. Petersburg Paradox then supplied the most memorable proof of why this distinction matters in the real world — a simple, seemingly innocent coin-flipping game breaks the plain expected-value rule completely, producing an infinite "fair price" that no sane person would ever actually pay, and only a sufficiently concave utility function can rescue the math back to something sensible. The lecture's final section ties everything back to where the whole course started: the Lecture 01 rational choice rule survives completely intact under uncertainty, with exactly one small but crucial upgrade — maximize expected utility instead of certain utility, and know the full lottery each action triggers, not just its existence.

---

> *Advanced AI · Lec 03 · github.com/rpaut03l/TS-02-03*
