# 🎯 Advanced AI — Lec 01: Individual Rational Decision Paradigm — PRACTICE

### *Self-test problems, a Q&A bank, and a mini task — try before you peek*

> **Nav:** [← Lec 01 README](README.md) | [📖 THEORY](aai_lec01_decision_problem_theory.md) | [🧮 NUMERICAL](aai_lec01_decision_problem_numerical.md) | **PRACTICE**

---

## 📚 Table of Contents

| # | Section | Jump |
|---|---|---|
| 1 | Concept Checks (quick, no calculation) | [§1](#1-concept-checks) |
| 2 | Preference & Rationality Drills | [§2](#2-preference--rationality-drills) |
| 3 | Utility Representation Drills | [§3](#3-utility-representation-drills) |
| 4 | Mini Task — Build Your Own Rational Preference | [§4](#4-mini-task--build-your-own-rational-preference) |
| 5 | Exam-Style Q&A Bank | [§5](#5-exam-style-qa-bank) |

> 💡 Every problem below is inside a `<details>` block. **Try to solve it on paper first**, then click "Show Answer" to check yourself.

---

## 1. Concept Checks

**Q1.1.** In the individual decision problem `(A, X, ≿)`, which symbol represents "what you can choose" and which represents "what actually happens"?

<details>
<summary>Show Answer</summary>

`A` = Actions = what you can choose. `X` = Outcomes = what actually happens (the consequence). They are deliberately kept separate because an action doesn't always guarantee its intended outcome.
</details>

---

**Q1.2.** True or False: `x ∼ y` means the individual has no ranking at all between x and y.

<details>
<summary>Show Answer</summary>

**False.** `x ∼ y` (indifference) means `x ≿ y` AND `y ≿ x` — i.e. they DO have a ranking, and that ranking is a **tie**. "No ranking at all" would be a violation of completeness, which rational preferences are never allowed to have.
</details>

---

**Q1.3.** Which theorem guarantees that a rational preference relation can always be written as numbers, and what is the ONE condition on `X` required for it to hold?

<details>
<summary>Show Answer</summary>

The **Utility Representation Theorem**. It requires `X` to be **finite** (in addition to `≿` being rational, i.e. complete and transitive).
</details>

---

## 2. Preference & Rationality Drills

**Q2.1.** A friend says: *"I like tea at least as much as coffee, and coffee at least as much as tea."* What relation between tea and coffee does this establish, and why?

<details>
<summary>Show Answer</summary>

`tea ∼ coffee` (indifference). By definition, `x ∼ y ⟺ x ≿ y ∧ y ≿ x` — exactly what was stated, in both directions.
</details>

---

**Q2.2.** Consider a preference relation with: `A ≿ B`, `B ≿ C`, and `C ≻ A`. Is this preference relation rational? Show your reasoning.

<details>
<summary>Show Answer</summary>

**No — it violates transitivity.** Transitivity requires: if `A ≿ B` and `B ≿ C`, then `A ≿ C` must hold. But we're told `C ≻ A`, which means `C ≿ A` AND `¬(A ≿ C)` — directly contradicting the required `A ≿ C`. This is a **preference loop** (A→B→C→A), exactly the "money pump" scenario from the Theory file §3, and worked again in the Numerical file §1.
</details>

---

**Q2.3.** `X = {gold, silver, bronze}`. Someone says: *"I haven't decided whether I prefer gold or silver — I just don't know."* Which rationality property does this threaten, and why is it a problem?

<details>
<summary>Show Answer</summary>

It threatens **completeness**, which requires that for ANY two outcomes, at least one of `x ≿ y` or `y ≿ x` must hold. Refusing to rank gold vs silver at all (not even a tie) leaves that pair completely unranked — a direct violation. (Note: "I'm indifferent between them" would be FINE — that's still a ranking, just a tied one. "I genuinely can't compare them" is the actual violation.)
</details>

---

## 3. Utility Representation Drills

**Q3.1.** `X = {Bronze, Silver, Gold}` with `Gold ≻ Silver ≻ Bronze`. Build a representing utility function `u`, following the worst→best numbering method.

<details>
<summary>Show Answer</summary>

```
u(Bronze) = 1
u(Silver) = 2
u(Gold)   = 3
```
Check: `u(Gold)=3 ≥ u(Silver)=2 ⟺ Gold ≻ Silver` ✓, and so on for every pair.
</details>

---

**Q3.2.** Now suppose `Gold ≻ Silver ∼ Bronze` (Silver and Bronze are tied, both below Gold). What goes wrong if you assign `u(Bronze)=1, u(Silver)=2, u(Gold)=3`?

<details>
<summary>Show Answer</summary>

This is **wrong**. Since `Silver ∼ Bronze`, they must get the **same** utility number. The correct assignment is:
```
u(Bronze) = 1
u(Silver) = 1     ← same as Bronze, they're tied
u(Gold)   = 2
```
Giving Silver a strictly higher number than Bronze (as in the flawed version) would incorrectly imply `Silver ≻ Bronze`, contradicting the stated tie. See Numerical file §2 for the fully worked version of this exact trap.
</details>

---

**Q3.3.** `A = {buy_A, buy_B}`, and due to a stock-out, `x*(buy_A) = ItemX` but `x*(buy_B) = ItemX` too (both actions currently lead to the same outcome!). If `u(ItemX) = 5`, what are `v(buy_A)` and `v(buy_B)`? Is the individual indifferent between the two actions?

<details>
<summary>Show Answer</summary>

```
v(buy_A) = u(x*(buy_A)) = u(ItemX) = 5
v(buy_B) = u(x*(buy_B)) = u(ItemX) = 5
```
`v(buy_A) = v(buy_B) = 5` → **Yes, indifferent.** Both actions are optimal choices `a*` since neither has a strictly higher value than the other. This shows `v` cares only about the resulting outcome, not the action's label.
</details>

---

## 4. Mini Task — Build Your Own Rational Preference

No spoiler answer here — genuinely yours to build.

**Task:**
1. Pick your own finite set `X` of at least 4 outcomes (something concrete and fun — snacks, songs, hobbies, whatever).
2. Write out a preference ranking `≿` over `X`, including at least one tie (`∼`) somewhere.
3. Explicitly verify **completeness** — check every single pair of outcomes has a stated ranking.
4. Explicitly verify **transitivity** — pick at least 2 different three-outcome chains (`x ≿ y ≿ z`) and confirm `x ≿ z` holds in each.
5. Build a representing `u: X → ℝ`, following the worst→best indifference-class method from [Numerical §2](aai_lec01_decision_problem_numerical.md#2-utility-representation--full-worked-proof). Make sure tied outcomes share the same number.
6. Invent 3 actions `A = {a₁, a₂, a₃}` and an outcome-map `x*: A → X` (it's fine, even encouraged, if two actions map to the same outcome). Compute `v(a) = u(x*(a))` for each, and state which action(s) a rational individual would choose.

---

## 5. Exam-Style Q&A Bank

<details>
<summary><b>Q. State the three components of an individual decision problem.</b></summary>

Actions (`A`), Outcomes (`X`), Preferences (`≿`).
</details>

<details>
<summary><b>Q. Write the formal definitions of ≻ and ∼ in terms of ≿.</b></summary>

`x ≻ y ⟺ x ≿ y ∧ ¬(y ≿ x)`. `x ∼ y ⟺ x ≿ y ∧ y ≿ x`.
</details>

<details>
<summary><b>Q. What are the two conditions for a rational preference relation?</b></summary>

Completeness and Transitivity.
</details>

<details>
<summary><b>Q. What does it mean for `u` to "represent" `≿`?</b></summary>

For all `x, y ∈ X`: `u(x) ≥ u(y) ⟺ x ≿ y`.
</details>

<details>
<summary><b>Q. Under what condition does a representing utility function `u` provably exist?</b></summary>

`≿` is rational (complete + transitive) AND `X` is finite.
</details>

<details>
<summary><b>Q. Describe the 2-step construction used to prove the utility representation theorem.</b></summary>

Step 1: find the best (`x̄`) and worst (`x`) outcomes (guaranteed to exist since `X` is finite and `≿` rational). Step 2: group outcomes into indifference classes (tied outcomes together), then assign strictly increasing numbers to the classes from worst to best.
</details>

<details>
<summary><b>Q. What four things must a rational individual know (Rational Choice Assumption)?</b></summary>

All actions `A`, all outcomes `X`, the outcome-map `x*: A → X`, and their own utility `u` over `X`.
</details>

<details>
<summary><b>Q. Define `v(a)` and explain how it differs from `u(x)`.</b></summary>

`v(a) = u(x*(a))` — the payoff of taking action `a`, obtained by applying `u` to the outcome that action actually produces. `u` is over outcomes; `v` is over actions. They can diverge if an action's real-world outcome differs from its "obvious" label (see Numerical §3).
</details>

<details>
<summary><b>Q. What is the rational choice rule, stated formally?</b></summary>

Choose `a* ∈ A` such that `v(a*) ≥ v(a)` for all `a ∈ A`.
</details>

[↑ Back to Top](#-advanced-ai--lec-01-individual-rational-decision-paradigm--practice)

---

> **← Back:** [🧮 NUMERICAL](aai_lec01_decision_problem_numerical.md) · [📖 THEORY](aai_lec01_decision_problem_theory.md) · [🏠 Lec 01 README](README.md)
>
> **Onward:** [Lecture 02 — Decision Trees, Risk & Lotteries →](../Lec_02_Decision_Trees_Risk_Lotteries/README.md)
>
> *Advanced AI · Lec 01 · github.com/rpaut03l/TS-02-03*
