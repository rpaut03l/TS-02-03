# DL Lecture 07 — DNN Optimization (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

> Folder: `Deep-Learning/Lecture-07-DNN-Optimization/theory/`
> Pairs with: [`numerical/dl_lecture07_optimization_numerical.md`](../numerical/dl_lecture07_optimization_numerical.md) · [`practice/dl_lecture07_optimization_practice.md`](../practice/dl_lecture07_optimization_practice.md) · [`exercises/dl_lecture07_exercises.md`](../exercises/dl_lecture07_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "DNN Optimization" deck

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [How to Train Your Network](#how-to-train-your-network)
3. [Gradients and Backpropagation](#gradients-and-backpropagation)
4. [The Loss Function](#the-loss-function)
5. [Convexity — Why It Matters](#convexity--why-it-matters)
6. [Following the Slope — The Gradient](#following-the-slope--the-gradient)
7. [Backpropagation via the Chain Rule](#backpropagation-via-the-chain-rule)
8. [Batch, Stochastic, and Mini-Batch Gradient Descent](#batch-stochastic-and-mini-batch-gradient-descent)
9. [The Problem: Local Minima and Saddle Points](#the-problem-local-minima-and-saddle-points)
10. [Momentum](#momentum)
11. [RMSProp](#rmsprop)
12. [Adam](#adam)
13. [Learning Rate Decay](#learning-rate-decay)
14. [Mnemonics](#mnemonics)
15. [Cheatsheet](#cheatsheet)
16. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
17. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine you're hiking down a foggy mountain at night, trying to reach the lowest valley, but you can only feel the ground right under your feet — you can't see the whole landscape. Every step, you feel which direction slopes downward most steeply, and you take a step that way. That's **gradient descent**: the "loss" is your altitude, the "weights" are your GPS coordinates, and the "gradient" is the slope you feel under your feet. Now imagine different hiking strategies: checking the slope after every single step you take (**Stochastic GD**), checking the average slope across your entire hiking group before anyone moves (**Batch GD**), or checking the average slope across a small cluster of nearby hikers (**Mini-Batch GD**). And imagine that instead of only trusting *this instant's* slope reading, you also remember *which direction you were already moving* and let that momentum carry you through small bumps and flat patches (**Momentum**, **RMSProp**, **Adam**). This lecture is entirely about smarter, faster ways to walk downhill in the fog.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## How to Train Your Network

Training means: **define a loss function, then find the parameters that minimize that loss on the training data.** This lecture starts with the simplest possible version: **stochastic gradient descent with a batch size of one** — processing training examples one at a time.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Gradients and Backpropagation

Multilayer neural networks are powerful — but how do they actually *learn*? The answer is **Backpropagation**: the core algorithm used to train neural networks, which computes **gradients** — numbers that tell us how much each individual weight affects the total error. Backpropagation works by efficiently applying the **Chain Rule** from calculus, propagating the error signal backward from the output layer through every earlier layer. The overall optimization algorithm is gradient descent; backpropagation is specifically the *method* used to efficiently compute the gradients that gradient descent then uses to update weights.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## The Loss Function

For a single training example `{x, y}` and the ANN's prediction `f̂(x, θ) = z` (where `θ` is the full parameter vector, containing all the network's weight coefficients), we need a loss function measuring how wrong the prediction is. For regression tasks, the lecture uses **squared error**: `L = (y - z)²` (the classic choice, penalizing larger errors disproportionately more). **Training means finding the value of θ that minimizes L(θ).** The good news: `L(θ)` is differentiable (we can compute its gradient). The bad news: in general, there is **no closed-form analytical solution** — you can't just solve an equation to jump directly to the answer; you have to search for it iteratively, step by step, which is exactly what gradient descent does.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Convexity — Why It Matters

**Convex function:** a function whose shape curves upward like a bowl — visually, it only has ONE "lowest point," so any **local** minimum you find is automatically also the **global** minimum, making optimization dramatically easier (you can never get "stuck" in a worse dip, because there isn't one).

**Convex set:** a set X is convex if, for any two points x and y inside X, the entire straight line connecting them also stays entirely inside X — no "caving in" or holes.

**Key consequence (a formal proposition worth memorizing):** *if X is a convex set and f is a convex function, then any local minimum of f in X is also a global minimum.* Unfortunately, in deep learning, X (the space of all possible weight vectors θ) is typically extremely high-dimensional, and the loss landscape `f(θ)` is generally **not** convex — so this convenient guarantee usually does NOT hold for neural networks, which is exactly why local minima and saddle points (discussed below) are real, practical concerns during training.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Following the Slope — The Gradient

In one dimension, the **derivative** of a function tells you its instantaneous slope at a point. In multiple dimensions (like a neural network's many weights), the **gradient** is the vector of **partial derivatives** along each individual dimension — one slope value per parameter. Crucially: **the gradient points in the direction of MAXIMUM increase** of the function. Since we want to *decrease* the loss, gradient descent always moves in the **opposite** direction of the gradient (hence the characteristic minus sign in every weight-update formula in this lecture).

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Backpropagation via the Chain Rule

**Chain rule recap:** if `f(x)` and `x(t)` are both functions, then the derivative of the composition `f(x(t))` with respect to `t` is `df/dt = (df/dx) × (dx/dt)` — you multiply the "local" derivatives along the chain.

A neural network's output `z` is literally a **composition of functions** — layer after layer of linear transforms and activations feeding into each other — so the loss `L(z)` is also a composition. Backpropagation exploits this directly: it applies the chain rule repeatedly, working **backward** from the loss, computing how much the loss changes with respect to each layer's output, then each layer's pre-activation, then finally each individual weight. A key structural insight from the lecture: backprop deliberately **separates** the effect of weights from the effect of activations at each layer, because **weights affect the pre-activation value** (often called `r_j` in the slides) while the **activation function affects the post-activation value** (`u_j`) — treating these as two separate chain-rule links keeps the derivation clean and modular, layer by layer.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Batch, Stochastic, and Mini-Batch Gradient Descent

Three variants, differing only in **how many training examples' gradients get averaged before each weight update**:

| Variant | Update frequency | Pros | Cons |
|---|---|---|---|
| **Batch GD** | After the ENTIRE training set | Convergence conditions well-studied and well understood; easier to analyze how weights change | Slow — one update per full pass over potentially huge data |
| **Stochastic GD (SGD)** | After EACH individual training example | Fast; noise can help escape poor local minima; adapts to changing/streaming data (online learning) | Noisy, unstable updates |
| **Mini-Batch SGD** | After a small randomly-selected GROUP of examples | Best practical balance — the standard choice in practice | Requires choosing a batch size hyperparameter |

**Practical detail:** Mini-batch SGD is the standard choice in practice; a typical small batch size might be around **20 samples**, though the ideal batch size can be increased depending on dataset size and available compute.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## The Problem: Local Minima and Saddle Points

What if the loss landscape has a **local minimum** (a dip that isn't the true lowest point) or a **saddle point** (a point where the gradient is exactly zero, but it's not a minimum at all — it curves up in some directions and down in others, like a horse saddle)? Plain gradient descent can get **stuck** at either — a zero gradient means the weight update formula computes essentially no movement, halting progress even though a better solution exists elsewhere. A key, somewhat surprising fact: **saddle points are far more common than true local minima in high-dimensional spaces** (like the millions of parameters in a real neural network) — this makes escaping flat/zero-gradient regions a genuinely important practical problem, motivating the more advanced optimizers below.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Momentum

**The problem with plain SGD:** the weight update depends ONLY on the *current* gradient — it has no memory of which direction it was already moving, so it can zig-zag inefficiently, especially on loss surfaces that are steep in some directions and flat in others.

**The fix:** add a **momentum** term that carries over a fraction of the *previous* update, so the optimizer builds up "velocity" in consistently-good directions, much like a real ball rolling downhill picks up speed and smooths over small bumps. The weight update now has two parts: (1) a **gradient term** — the current direction that reduces loss right now, and (2) a **momentum term** — a fraction of the previous step, carried forward.

**Advantages:** momentum can significantly speed up training when the loss surface is highly non-spherical (steep in some directions, shallow in others) — it **damps** step sizes along directions of high curvature (preventing overshooting/oscillation) while effectively giving a **larger effective learning rate** along directions of low curvature (accelerating through flat regions). The momentum coefficient, commonly called **γ (gamma)**, controls how much of the past matters: larger γ means past gradients influence the current step more. A common practical schedule: start with γ around **0.5** until training stabilizes, then increase to **0.9 or higher**.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## RMSProp

**RMSProp** (Tieleman & Hinton, 2012) belongs to the family of **adaptive optimizers** — instead of one single global learning rate for all parameters (as in plain Gradient Descent), adaptive optimizers automatically adjust the effective learning rate **separately for each individual parameter**, based on that parameter's own history of past gradients. Specifically, RMSProp adaptively scales each parameter's gradient using an **exponentially decaying moving average of past squared gradients** — parameters that have historically had large gradients get their effective step size shrunk down (to avoid overshooting), while parameters with historically small gradients get relatively larger effective steps (to make meaningful progress despite a flat-feeling direction).

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Adam

**Adam** (Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015) is described in the lecture as **"sort of like RMSProp with momentum"** — it combines BOTH ideas at once: it tracks an exponentially decaying average of past gradients (like Momentum, giving it a sense of consistent direction) AND an exponentially decaying average of past *squared* gradients (like RMSProp, giving it per-parameter adaptive step sizes). This combination is why Adam is one of the most widely used default optimizers in modern deep learning practice — it inherits the smoothing/acceleration benefit of momentum together with the per-parameter adaptivity of RMSProp.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Learning Rate Decay

All of the optimizers above (SGD, SGD+Momentum, RMSProp, Adam) treat the **learning rate as a hyperparameter** — and in reality, no single "best" learning rate exists universally; the right choice genuinely depends on the specific problem. A very common, effective practical technique: **Learning Rate Decay** — instead of keeping the learning rate constant throughout training, deliberately **reduce it at predefined points** (e.g., at specific epochs, or following a schedule like cosine annealing with warm restarts). The intuition: early in training, large steps help you make fast progress across the broad loss landscape; later in training, as you approach a good solution, smaller steps let you fine-tune precisely without overshooting or bouncing around the minimum. Several influential papers cited in the lecture (SGDR/"Warm Restarts," GPT's pretraining recipe, SlowFast Networks, Sparse Transformers) all rely on carefully designed learning rate schedules as a key part of their training recipe.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Mnemonics

- **"Hiking downhill in the fog, feeling the slope under your feet"** → gradient descent in one image.
- **"Convex = bowl-shaped = only one bottom"** → why convexity makes optimization easy (and why it usually doesn't hold for neural nets).
- **"Gradient points UP, so we walk the OPPOSITE way"** → the minus sign in every update formula.
- **"Batch = whole group moves together, Stochastic = one at a time, Mini-batch = small huddle"** → the three GD variants in one line.
- **"Momentum = remember which way you were already rolling"** → the ball-rolling-downhill picture.
- **"RMSProp = per-parameter volume knob based on past squared gradients"** → adaptive learning rate intuition.
- **"Adam = RMSProp + Momentum, best of both"** → the one-line Adam definition, straight from the lecture.
- **"Big steps early, small steps late"** → learning rate decay in one line.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Cheatsheet

| Concept | One-liner |
|---|---|
| Gradient | Vector of partial derivatives; points toward steepest INCREASE |
| Backpropagation | Efficient chain-rule computation of gradients, layer by layer, backward |
| Convex function | Bowl-shaped; local minimum = global minimum |
| Saddle point | Zero gradient, but not a true minimum; common in high dimensions |
| Batch GD | Update after the WHOLE dataset |
| Stochastic GD | Update after EACH example |
| Mini-Batch SGD | Update after a small random GROUP; standard in practice |
| Momentum | Adds a fraction (γ) of the previous update to the current one |
| RMSProp | Per-parameter adaptive learning rate via decaying average of squared gradients |
| Adam | Momentum + RMSProp combined |
| Learning rate decay | Reduce learning rate at predefined points during training |

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** claiming neural network loss landscapes are convex — they are generally NOT, which is exactly why local minima/saddle points are a real concern; convexity is presented as a useful reference concept, not a property of typical deep learning loss surfaces.
- **Trap:** confusing saddle points with local minima — a saddle point has zero gradient but is NOT a minimum (it curves up in some directions, down in others); both can stall plain gradient descent, but they are different phenomena, and saddle points are explicitly noted as MORE common in high dimensions.
- **Trap:** mixing up which GD variant updates most/least frequently — Batch GD updates LEAST often (once per full dataset pass), SGD updates MOST often (every single example), Mini-batch is in between.
- **Trap:** describing Adam as "just momentum" or "just RMSProp" — it is explicitly BOTH combined, tracking both a first-moment (mean) and second-moment (squared) estimate of the gradient.
- **Exam hack:** if asked to explain why the gradient descent update has a minus sign, always explicitly state "the gradient points toward the direction of maximum INCREASE, so we move in the opposite direction to decrease the loss" — this exact phrasing/reasoning is a favourite thing examiners look for.
- **Exam hack:** the γ (gamma) momentum scheduling detail (start ~0.5, increase to ~0.9+) is a specific, testable number — memorize it alongside the general momentum concept.

`[🔝 Top](#dl-lecture-07--dnn-optimization-theory)`

---

## Summary

This lecture covers how neural networks are actually trained: by defining a differentiable loss function (e.g., squared error for regression) over parameters θ, and using gradient descent to iteratively search for a minimum, since no closed-form analytical solution generally exists. Gradients — vectors of partial derivatives that point toward the direction of maximum increase — are computed efficiently via **Backpropagation**, which applies the chain rule repeatedly, working backward through the network's layered function composition, separating the effect of weights (on pre-activations) from the effect of activation functions (on post-activations). Convexity (bowl-shaped functions where any local minimum is automatically global) makes optimization easy when it holds, but neural network loss landscapes are generally NOT convex, making local minima and — even more commonly, in high dimensions — **saddle points** (zero gradient, not a true minimum) genuine practical obstacles that plain gradient descent can get stuck on. Three basic gradient descent variants differ in update frequency: Batch GD (whole dataset per update, well-understood convergence), Stochastic GD (one example per update, fast and noisy, useful for online learning), and Mini-Batch SGD (small random groups per update, the standard practical choice, often around 20 samples). To handle local minima, saddle points, and inefficient zig-zagging, **Momentum** adds a fraction (γ, typically ramped from ~0.5 to ~0.9+) of the previous update into the current one, damping oscillation in high-curvature directions while accelerating through low-curvature ones. **RMSProp** takes a different, adaptive approach — scaling each parameter's individual learning rate based on an exponentially decaying moving average of its own past squared gradients. **Adam**, described as "RMSProp with momentum," combines both ideas and is one of the most widely used default optimizers in practice. Finally, **Learning Rate Decay** — deliberately shrinking the learning rate at predefined points during training rather than keeping it fixed — lets training take large, fast steps early on and small, precise fine-tuning steps later, a technique underlying several influential training recipes cited in the lecture.

`[← Lecture 06](../../Lecture-06-Attention/README.md) · [🔝 Top](#dl-lecture-07--dnn-optimization-theory) · [Next: Numerical →](../numerical/dl_lecture07_optimization_numerical.md)`
