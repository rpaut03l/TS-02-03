# 📗 Exercise 1.1 — Your Decision

> **Nav:** [← Chapter 1 Exercises](README.md) | **1.1** | [1.2 →](aai_ch1_ex02_going_to_movies.md)

---

## 📋 What's Being Asked

This exercise is deliberately personal — think of a decision you make regularly (what to eat, how to commute, when to study) and formalize it using the exact tools from [Lecture 01](../Lec_01_Individual_Decision_Problem/README.md): list the **actions** `A`, list the **outcomes** `X`, and — this is the specific instruction — do **NOT** yet write down a preference relation `≿`. Only after that, assign numeric **payoffs** to the outcomes and draw the **decision tree**. The point of withholding `≿` until the end is to force you to separate "what could happen" from "how much I like it" — exactly the Lecture 01 distinction between `X` (outcomes) and `u` (payoff/preference over outcomes).

Since this is inherently personal, below is a fully worked **template example** — "what to have for lunch" — showing exactly the method you should apply to your own regularly-faced decision.

---

## 🧠 Step-by-Step Method

### Step 1 — Name the decision precisely

Vague: "what should I eat?" Precise: "What do I order for lunch today, given the three usual options near my desk?" A decision problem needs a crisply bounded action set — open-ended decisions ("do something about lunch") can't be formalized until you narrow them.

### Step 2 — List the Actions `A`

```
A = { Sandwich shop, Salad bar, Pack lunch from home }
```
Three discrete, mutually exclusive choices — exactly the kind of short action list Lecture 01 §1 describes.

### Step 3 — List the Outcomes `X`

For this decision, each action deterministically produces one outcome (no randomness, no Nature involved yet — that's Lecture 02's territory):
```
X = { Turkey sandwich + chips, Garden salad, Home-packed meal }
```
Notice `X` and `A` are almost identical in this example — that's common for simple, low-uncertainty everyday decisions, but the exercise still wants them listed as two separate sets to reinforce the Lecture 01 habit of never merging "what I choose" with "what I get."

### Step 4 — Explicitly withhold `≿` for now

At this stage, we know the three things that *could* happen, but we haven't said anything yet about which is preferred. This matters because it's easy to accidentally build your action/outcome list already biased toward your favorite — the exercise wants you to resist that until payoffs are assigned deliberately in the next step.

### Step 5 — Assign payoffs `u`

Now attach numbers reflecting personal preference — for this template, say the sandwich shop is the most satisfying, salad is healthiest-but-less-filling, and home-packed is convenient but boring:
```
u(Turkey sandwich + chips) = 8
u(Garden salad)            = 5
u(Home-packed meal)        = 6
```

### Step 6 — Draw the decision tree

```
                         YOU (root)
              ┌──────────┼──────────┐
        Sandwich shop  Salad bar   Pack lunch
           (leaf)       (leaf)      (leaf)
            u=8           u=5         u=6
```
A flat, single-level tree — one decision, three branches, three leaves. No sequential structure needed since this is a one-shot choice.

### Step 7 — Apply the rational choice rule

```
max(8, 5, 6) = 8  →  Choose: Sandwich shop
```

---

## 🎯 Applying This to Your Own Decision

Swap in your own regularly-faced decision using the same seven steps: name it precisely, list 2–5 actions, list the matching outcomes, resist assigning preference too early, then assign honest payoff numbers, draw the flat tree, and pick the max. Common good candidates: choice of commute route, choice of workout type, choice of study subject for the evening.

---

## 🧠 Mnemonic & Cheat Sheet

```
╔════════════════════════════════════════════════════╗
║  "LIST, LIST, LATER" — the three-step discipline     ║
║  1. LIST actions A                                    ║
║  2. LIST outcomes X  (separately, even if similar!)  ║
║  3. Assign preference/payoff LATER, never upfront     ║
╚════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** this exercise is really testing whether you understand that `A`, `X`, and `≿` are three genuinely separate objects in the decision-problem definition (Lecture 01 §1), not three names for the same thing.

---

## 📝 Summary

This exercise is really a discipline check disguised as a simple task: can you name a decision's actions and outcomes as two genuinely separate lists, without letting your personal preference sneak in and bias the list before you've explicitly assigned any payoffs? The lunch template walked through all seven steps in order — naming the decision precisely, listing actions, listing outcomes, deliberately pausing before assigning any preference, only then attaching honest payoff numbers, drawing a flat single-level tree, and finally applying the same "pick the biggest number" rule from [Lecture 01 §5](../Lec_01_Individual_Decision_Problem/aai_lec01_decision_problem_theory.md#5-rational-choice-assumption--the-value-function). The reason this matters beyond a warm-up exercise: real-world decisions are rarely handed to you already formalized, and the skill of correctly separating "what I can do" from "what I get" from "how much I like it" is the single most transferable habit from this entire course — it's the same three-part split every other exercise in this folder, and every lecture concept from here forward, quietly assumes you can already do without thinking twice about it.

---

> **Next:** [Exercise 1.2 — Going to the Movies →](aai_ch1_ex02_going_to_movies.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
