# DL Lecture 07 — DNN Optimization (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-07--dnn-optimization-practice)`

> Folder: `Deep-Learning/Lecture-07-DNN-Optimization/practice/`
> Pairs with: [`theory/dl_lecture07_optimization_theory.md`](../theory/dl_lecture07_optimization_theory.md) · [`numerical/dl_lecture07_optimization_numerical.md`](../numerical/dl_lecture07_optimization_numerical.md) · [`exercises/dl_lecture07_exercises.md`](../exercises/dl_lecture07_exercises.md)

---

## Table of Contents
1. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
2. [Explain-It-Back Prompts](#explain-it-back-prompts)
3. [Quick-Fire True / False](#quick-fire-true--false)
4. [Optimizer Matching Drill](#optimizer-matching-drill)
5. [Mini Interview-Style Round](#mini-interview-style-round)
6. [Summary](#summary)

---

## Concept Check — Fill in the Blank

1. The gradient points toward the direction of maximum ______, so gradient descent moves in the ______ direction.
2. A ______ function is bowl-shaped, meaning any local minimum is also the global minimum.
3. ______ points are more common than local minima in high-dimensional spaces, and have zero gradient without being a true minimum.
4. Adam is described in the lecture as "sort of like ______ with ______."
5. The standard, most widely used gradient descent variant in practice is ______.

<details>
<summary>Show answers</summary>

1. increase; opposite
2. convex
3. Saddle
4. RMSProp; momentum
5. Mini-Batch SGD
</details>

`[🔝 Top](#dl-lecture-07--dnn-optimization-practice)`

---

## Explain-It-Back Prompts

1. Explain the foggy-mountain-hiking analogy for gradient descent in your own words.
2. Walk through why backpropagation separates the effect of weights (on pre-activations) from the effect of activation functions (on post-activations).
3. Explain, using Worked Example 2's actual numbers, why momentum's second update step was larger than its first, despite a smaller gradient.
4. Explain the difference between what Momentum controls (direction/velocity) and what RMSProp controls (per-parameter step size), and how Adam combines both.
5. Explain why learning rate decay makes intuitive sense, using the "big steps early, small steps late" framing.

`[🔝 Top](#dl-lecture-07--dnn-optimization-practice)`

---

## Quick-Fire True / False

1. Neural network loss landscapes are generally convex. — **False**.
2. Batch GD updates parameters after every single training example. — **False** (that's Stochastic GD; Batch GD updates after the WHOLE dataset).
3. RMSProp uses a single global learning rate shared by all parameters. — **False** (it adapts the learning rate per-parameter).
4. Adam requires bias correction because its moment estimates start at zero and are biased early in training. — **True**.
5. A saddle point has a non-zero gradient. — **False** (by definition, saddle points have zero gradient).

`[🔝 Top](#dl-lecture-07--dnn-optimization-practice)`

---

## Optimizer Matching Drill

| Optimizer | Key mechanism | Your match |
|---|---|---|
| Plain SGD | ? | |
| Momentum | ? | |
| RMSProp | ? | |
| Adam | ? | |

Options: (a) Combines momentum AND per-parameter adaptive scaling with bias correction, (b) Uses only the current gradient, no memory of past steps, (c) Adapts each parameter's step size using a decaying average of past squared gradients, (d) Carries forward a fraction of the previous update as "velocity"

<details>
<summary>Show answers</summary>

Plain SGD → (b). Momentum → (d). RMSProp → (c). Adam → (a).
</details>

`[🔝 Top](#dl-lecture-07--dnn-optimization-practice)`

---

## Mini Interview-Style Round

**Q1.** "Training loss oscillates wildly and never settles down, even after many epochs. What optimizer-related hypothesis would you check first?"

<details>
<summary>Show answer</summary>

Suspect the learning rate is too large relative to the loss surface's curvature — large steps can repeatedly overshoot the minimum, causing oscillation instead of convergence. First things to try: reduce the base learning rate, and/or introduce learning rate decay so steps shrink over time as training progresses. If using plain SGD, also consider switching to Momentum or Adam, since momentum specifically damps oscillation along high-curvature directions.
</details>

**Q2.** "A teammate asks why we don't just always use Batch GD, since its convergence properties are so well understood. What's your answer?"

<details>
<summary>Show answer</summary>

Batch GD requires computing gradients over the ENTIRE training set before making even one update — for large modern datasets, this is often computationally prohibitive and painfully slow, since you make very few updates per unit of compute/time spent. Mini-batch SGD gets most of the stability benefits of averaging over multiple examples while updating far more frequently, making it dramatically more practical, which is exactly why it's the standard choice despite Batch GD's cleaner theoretical convergence guarantees.
</details>

**Q3.** "Explain, in terms a beginner would understand, why Adam tends to be a safe 'default' optimizer choice."

<details>
<summary>Show answer</summary>

Adam automatically does two helpful things at once without much manual tuning: it smooths out noisy updates by remembering a running average of recent gradient directions (like momentum, reducing zig-zagging), and it automatically gives each individual parameter its own appropriately-sized step, shrinking steps for parameters with historically large/volatile gradients and allowing bigger steps for parameters with small/stable gradients (like RMSProp). Because it handles both of these adjustments automatically, it tends to work reasonably well across a wide range of problems without requiring as much careful, problem-specific learning-rate tuning as plain SGD.
</details>

`[🔝 Top](#dl-lecture-07--dnn-optimization-practice)`

---

## Summary

This practice file drills Lecture 7's optimization concepts through active recall. A fill-in-the-blank check reinforces the gradient-direction rule, convexity's definition, saddle points, Adam's "RMSProp + momentum" description, and mini-batch SGD's status as the practical standard. Five explain-it-back prompts push you to reproduce the hiking analogy, the weights-vs-activations backprop separation, the concrete momentum-acceleration numbers from the numerical file, the Momentum-vs-RMSProp-vs-Adam distinction, and the learning-rate-decay intuition in your own words. A quick-fire true/false round targets common mix-ups (Batch vs Stochastic update frequency, RMSProp's per-parameter adaptivity, saddle point gradient values), and a dedicated optimizer-matching drill ties each of the four optimizers to its core mechanism. A three-question interview-style round rehearses realistic debugging judgment: diagnosing oscillating training loss, justifying mini-batch SGD's practical dominance over theoretically-cleaner Batch GD, and explaining why Adam is a reasonable default choice for beginners. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture07_optimization_numerical.md) · [🔝 Top](#dl-lecture-07--dnn-optimization-practice) · [Next: Exercises →](../exercises/dl_lecture07_exercises.md)`
