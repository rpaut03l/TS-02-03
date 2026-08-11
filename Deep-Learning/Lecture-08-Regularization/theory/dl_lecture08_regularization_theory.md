# DL Lecture 08 — Regularization (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-08--regularization-theory)`

> Folder: `Deep-Learning/Lecture-08-Regularization/theory/`
> Pairs with: [`numerical/dl_lecture08_regularization_numerical.md`](../numerical/dl_lecture08_regularization_numerical.md) · [`practice/dl_lecture08_regularization_practice.md`](../practice/dl_lecture08_regularization_practice.md) · [`exercises/dl_lecture08_exercises.md`](../exercises/dl_lecture08_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "Regularization" deck

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Underfitting and Overfitting](#underfitting-and-overfitting)
3. [Parameter Norm Penalties — Data Loss + Regularization](#parameter-norm-penalties--data-loss--regularization)
4. [Regularized Least Squares (L2 / Weight Decay)](#regularized-least-squares-l2--weight-decay)
5. [Why L2 Regularization Is Easy to Optimize](#why-l2-regularization-is-easy-to-optimize)
6. [The General Lq Regularizer — L1 vs L2](#the-general-lq-regularizer--l1-vs-l2)
7. [L1 vs L2 Intuition — Sparsity vs Shrinkage](#l1-vs-l2-intuition--sparsity-vs-shrinkage)
8. [Is Regularizing Parameters Enough?](#is-regularizing-parameters-enough)
9. [Ensemble Learning](#ensemble-learning)
10. [Dropout — The Mechanics](#dropout--the-mechanics)
11. [The Dropout Trick — Training Without 2ⁿ Separate Networks](#the-dropout-trick--training-without-2n-separate-networks)
12. [Dropout at Test Time](#dropout-at-test-time)
13. [Early Stopping](#early-stopping)
14. [Data Augmentation](#data-augmentation)
15. [Mnemonics](#mnemonics)
16. [Cheatsheet](#cheatsheet)
17. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
18. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine a student preparing for an exam by memorizing the EXACT wording of every single practice question and its answer, word for word — including the practice set's typos and quirky phrasing. On the practice set itself, this student scores 100%. But on the real exam, with slightly reworded questions, they fail badly, because they never learned the underlying *concepts* — just memorized noise. This is **overfitting**: a model that fits the training data (including its noise and quirks) too perfectly, at the cost of generalizing to new data. **Regularization** is the whole toolbox of techniques that gently discourage this kind of over-memorization — nudging the model toward simpler, more robust patterns that are more likely to generalize, even if that means giving up a little bit of training-set accuracy.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Underfitting and Overfitting

**Underfitting:** the model is too simple to capture the real pattern in the data — high error on BOTH training and new data. **Overfitting:** the model is too complex/flexible relative to the amount of data — very low training error, but high error on new data, because it has effectively memorized noise rather than learning the true signal. **The classic fix:** control model capacity — prefer simpler models, or explicitly regularize a more complex model — to avoid fitting the noise. A key modern reality check: models in practice are increasingly chosen to be **overparametrized** — deliberately having far more parameters than strictly needed to perfectly fit the training data — which makes regularization even more essential, not less.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Parameter Norm Penalties — Data Loss + Regularization

The total loss used for training is split into two pieces:
```
Total Loss = Data Loss + Regularization
```
- **Data loss:** measures whether the model's predictions match the training data (this is the "normal" loss from Lecture 7, e.g. squared error).
- **Regularization:** an extra term that PREVENTS the model from fitting the training data *too* well — deliberately pushing back against a perfect fit, so the model doesn't chase every last bit of noise in the training set.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Regularized Least Squares (L2 / Weight Decay)

**Sum-of-squares error (data loss) for regression:**
```
E_D(w) = (1/2) * Σ_n (y_n - w^T x_n)^2
```

**Simplest regularizer (L2 penalty):** the sum-of-squares of the weight vector's own elements:
```
E_W(w) = (1/2) * w^T w
```

**Total regularized loss:**
```
E(w) = E_D(w) + (lambda/2) * w^T w
```
where **λ (lambda)** is the **regularization coefficient** — a hyperparameter controlling how strongly the penalty is enforced (λ=0 recovers plain unregularized least squares; larger λ pushes weights more aggressively toward zero).

**Why it's called "weight decay":** in sequential (iterative, step-by-step) learning algorithms, this penalty encourages weight values to *decay* toward zero at every update step, rather than growing unboundedly large.

**Geometric picture (a favourite exam diagram):** plot weight space with axes `w1` and `w2`. The **blue contours** are the data loss's error contours — concentric ellipses, like height-lines on a bowl, with the very center being the unregularized minimum (best fit to training data alone, with no size restriction on weights). The **orange region** represents the allowed/feasible region once the L2 penalty is added — "don't use weights that are too large, even if they'd reduce training loss further." The blue-center (unregularized) solution may have very large weights, high sensitivity to noise, and overfitting — regularization deliberately pulls the solution away from that risky corner, toward smaller, more robust weights, even at the cost of a slightly higher training loss.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Why L2 Regularization Is Easy to Optimize

Two big practical advantages: (1) **the error function remains a quadratic function of w** — both the original squared-error term and the L2 penalty `w^T w` are quadratic in w, so their sum is still a nice, smooth, convex bowl shape (a parabola in higher dimensions). (2) **its exact minimizer can be found in closed form** — instead of needing iterative optimization (like gradient descent from Lecture 7), you can set the gradient with respect to w to exactly zero and directly SOLVE for the optimal w mathematically, in one step, no iteration required. This closed-form solvability is a special, convenient property of L2 regularization specifically (thanks to the whole loss staying quadratic) — it does NOT hold for every kind of regularizer.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## The General Lq Regularizer — L1 vs L2

A more general family of regularizers penalizes the sum of the **q-th power** of the absolute weight values:
```
E_W(w) = (1/2) * Σ_j |w_j|^q
```
- **q=2** → this is exactly the quadratic (L2) regularizer described above.
- **q=1** → this is the **Lasso** regularizer (Tibshirani, 1996) — a different, extremely influential and widely-used choice.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## L1 vs L2 Intuition — Sparsity vs Shrinkage

**L1 regularization** generally pushes many weight values ALL THE WAY to exactly **0** — producing **sparse** weight vectors (many weights are exactly zero, effectively performing automatic feature selection). Visualized as a histogram of weight values, L1 produces a very tall spike exactly at 0.

**L2 regularization** generally shrinks larger weight values more heavily but does NOT force many weights to exactly zero — it produces weights clustered NEAR zero, but rarely exactly at zero. Visualized as a histogram, L2 produces weights gathered near (but not exactly at) zero.

**Geometric reason (the classic "diamond vs circle" picture):** for a 2-weight setting `(w1, w2)`, the L2 penalty's contours form smooth CIRCLES around the origin, while the L1 penalty's contours form a DIAMOND (with sharp corners sitting exactly on the axes, i.e., where one weight is exactly 0). Because the L1 diamond has corners exactly on the axes, the optimal solution (where the data-loss contours first touch the regularizer's boundary) is much more likely to land exactly ON a corner — i.e., with one or more weights at exactly zero. The L2 circle has no corners, so the optimal touching point almost never lands exactly on an axis, giving small-but-nonzero weights instead.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Is Regularizing Parameters Enough?

An important, more modern nuance: **regularizing by controlling only the number/size of parameters is NOT the best or only option.** Research (Neyshabur et al. 2015; Belkin et al. 2019) shows that good architecture choices PLUS SGD itself already generalize surprisingly well in the overparametrized regime — meaning **implicit regularization** properties (which optimizer is used, batch size, normalization layers, etc. — not just explicit penalty terms) also meaningfully affect generalization, and should be considered alongside explicit parameter penalties, not instead of them.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Ensemble Learning

**Broad definition (important, easy to forget):** regularization means ANYTHING that reduces overfitting and improves generalization — not just explicit penalty terms. **Model averaging** (a "bagging" ensemble — training multiple models and averaging their predictions) is one such technique that genuinely helps generalization. **The problem:** training several large neural networks just to build an ensemble is often **prohibitively expensive** computationally.

**Two main ensemble strategies:**
1. **Different architectures:** train multiple neural networks with genuinely different designs, then combine (e.g., average) their predictions.
2. **Different data subsets:** train the SAME architecture multiple times on different subsets of the training data, then combine predictions.

**The deployment problem:** even if ensembles improve accuracy, deploying MANY full models together at inference time can be too computationally expensive for fast, real-world, latency-sensitive systems. This exact problem — wanting ensemble-like benefits, without ensemble-like deployment cost — is precisely what motivates **Dropout**.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Dropout — The Mechanics

**Dropout** refers to literally "dropping out" (temporarily removing) units from the network during training. It directly addresses BOTH of ensemble learning's limitations: it allows training something *like* many different networks WITHOUT significant extra computational overhead, acting as an efficient, approximate way of combining exponentially many different neural networks using just ONE network's worth of compute.

**The mechanics, step by step:**
1. **Temporarily remove a node** and ALL its incoming and outgoing connections, resulting in a smaller, "thinned" network for this particular training step.
2. **Each node is retained with a fixed probability p** — typically **p=0.5** for hidden nodes, and **p=0.8** for input nodes (input nodes are dropped less aggressively, since losing raw input information is more costly than losing an intermediate hidden feature).

**Formal mechanics:** for each neuron, a binary mask value is sampled from a **Bernoulli distribution** with parameter p (i.e., the mask is 1 with probability p, and 0 with probability 1-p). The neuron's output after applying dropout becomes:
```
output = mask * activation
```
— i.e., the neuron's normal output is either passed through unchanged (mask=1) or completely zeroed out (mask=0), independently, for every neuron, every training step.

**How many distinct "thinned" networks are possible?** Given a total of **n** nodes (where dropout can be applied), the total number of distinct thinned networks that can be formed is **2ⁿ** — since each of the n nodes can independently be either present or dropped, giving 2 choices per node, multiplied across all n nodes.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## The Dropout Trick — Training Without 2ⁿ Separate Networks

Since the number of possible thinned networks (2ⁿ) is astronomically large, training each one separately is completely infeasible. **The trick:** (1) keep ONE common, shared set of weights for the entire "full" network; (2) for EACH training batch, randomly drop a different random subset of neurons, creating a DIFFERENT thinned network each time — but all thinned networks share and update the SAME underlying weight parameters.

**Walking through two mini-batches:**
- **1st mini-batch:** all weights initialized; dropout randomly deactivates some neurons, forming a thinned network for this batch; loss is computed and backpropagated; **only the weights connected to currently-ACTIVE neurons are updated** — weights belonging to dropped neurons simply remain unchanged for this particular batch.
- **2nd mini-batch:** dropout is re-applied with a NEW random pattern, forming a DIFFERENT thinned network; loss is again computed and backpropagated to whichever weights are active this time.

**The key consequence — parameter sharing across thinned networks:** if a particular weight happens to be active in BOTH mini-batches, it receives multiple updates (once per batch it was active in); if it's active in only one of the two batches, it receives just one update that round. Over the course of many training batches, essentially every weight gets updated many times, across many DIFFERENT thinned-network contexts — this is Dropout's clever way of approximately training "many different networks" while only ever maintaining and updating ONE shared set of weights, directly solving ensemble learning's prohibitive-training-cost problem.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Dropout at Test Time

**The natural next question:** after training with dropout, which weights do you actually use at test/inference time, since different weights were "active" at different points during training? It's **impossible to literally aggregate the outputs of all 2ⁿ thinned networks** individually (far too many, far too expensive). **The practical solution:** at test time, use the FULL network (no dropping at all), but **scale each neuron's output by its retention probability p** (the same probability used during training) — this scaling approximates averaging over the exponentially many thinned networks in a single, cheap forward pass, without ever having to explicitly enumerate or run them individually.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Early Stopping

**Setup:** split your dataset into three parts — training set, validation set, and test set. Critically: use the **test set only ONCE, at the very end**, for a final, unbiased estimate of generalization performance; use **validation accuracy** throughout training for tuning decisions (always recommended, and reusable many times without biasing your final test-set estimate).

**The technique:** as training progresses, training error typically keeps decreasing, but VALIDATION error will decrease for a while, then eventually start increasing again — this is the visual signature of the model beginning to overfit. **Early stopping** means: stop training at the point where validation error reaches its MINIMUM (typically marked with a vertical dashed line on a training-curve plot), rather than continuing to train further and letting the model overfit past that point. Early stopping is one of the simplest and most widely used forms of regularization in practice — it costs essentially nothing extra beyond monitoring validation performance during training you were already doing anyway.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Data Augmentation

**Data Augmentation** increases the EFFECTIVE size and diversity of your training dataset by generating new training samples through random transformations applied to existing data — helping reduce overfitting and improve generalization, without needing to collect any genuinely new data.

**Common augmentation techniques (especially for images):** translation (shifting), rotation and scaling, flipping (horizontal/vertical), cropping and zooming, distortion or deformation, brightness/contrast/color adjustments, and adding noise or blur. The underlying principle: if a transformed version of an image (e.g., slightly rotated, or mirrored) should still be classified the same way, then training on both versions teaches the model to be robust/invariant to that specific kind of transformation — directly related to the invariance ideas introduced back in Lecture 1.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Mnemonics

- **"Memorizing practice questions vs learning the concept"** → overfitting vs true generalization, in one image.
- **"Data loss says fit; regularization says don't fit TOO well"** → the two-part total loss.
- **"L2 = weight decay, shrinks toward (but not to) zero"** → the closed-form-friendly, circular-contour regularizer.
- **"L1 = Lasso, diamond corners, snaps weights to exactly 0"** → the sparsity-inducing regularizer.
- **"2ⁿ thinned networks, ONE shared set of weights"** → dropout's core trick.
- **"Scale by p at test time instead of averaging 2ⁿ networks"** → dropout's test-time approximation.
- **"Stop where the validation curve turns back up"** → early stopping in one line.

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Cheatsheet

| Technique | One-liner | Key formula/fact |
|---|---|---|
| L2 / weight decay | Shrinks weights smoothly toward zero | `E(w)=E_D(w)+(λ/2)w^Tw` |
| L1 / Lasso | Pushes many weights exactly to zero | `q=1` in the general `Σ|w_j|^q` family |
| General Lq | Unifies L1 and L2 | `q=2`→L2, `q=1`→L1 |
| Ensemble learning | Train multiple models, average predictions | Expensive to train AND deploy |
| Dropout | Randomly zero neurons during training | `2ⁿ` possible thinned networks; retained w.p. p (0.5 hidden, 0.8 input) |
| Dropout at test | Use full network, scale by p | Approximates averaging all `2ⁿ` networks |
| Early stopping | Stop at minimum validation error | Uses train/validation/test split |
| Data augmentation | Synthetic new training samples via transforms | Translation, rotation, flip, crop, noise, etc. |

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** claiming L1 and L2 behave the same way — L1's diamond-shaped contours (sharp corners on the axes) cause sparsity (exact zeros); L2's circular contours almost never touch the axes exactly, causing shrinkage without sparsity. This geometric distinction is a favourite "explain why" exam question.
- **Trap:** using the test set for tuning decisions during early stopping — the theory file explicitly states the test set should be used ONLY ONCE, at the very end; use the VALIDATION set for all tuning, including early-stopping decisions.
- **Trap:** thinking dropout literally trains 2ⁿ separate networks — it trains ONE shared set of weights, with a different random subset "active" each batch; the 2ⁿ figure describes how many DIFFERENT thinned-network configurations are theoretically possible, not how many are actually separately trained.
- **Trap:** forgetting the test-time scaling step for dropout — using the full (undropped) network at test time WITHOUT scaling by p would produce outputs with a systematically different average magnitude than what the network saw during training.
- **Exam hack:** if asked to justify why L2 regularization is easy to optimize, always name BOTH reasons explicitly — stays quadratic (bowl-shaped) AND has a closed-form solution — a one-reason answer typically loses partial marks.
- **Exam hack:** the p=0.5 (hidden) / p=0.8 (input) retention probabilities are exact, testable numbers from the lecture — memorize them precisely, not just "dropout uses some probability."

`[🔝 Top](#dl-lecture-08--regularization-theory)`

---

## Summary

This lecture surveys the full toolbox of regularization techniques used to fight overfitting — where a model fits training data (including its noise) too well and fails to generalize to new data. The total training loss splits into data loss (fit the training data) plus a regularization term (don't fit it TOO well); the simplest and most classic regularizer is the **L2 penalty** (`(λ/2)w^Tw`, also called weight decay because it pushes weights to decay toward zero), which is especially convenient because it keeps the total loss quadratic (bowl-shaped) and therefore solvable in exact closed form, without needing iterative optimization. Generalizing to an `Lq` family (`Σ|w_j|^q`), `q=2` recovers L2 while `q=1` gives **Lasso (L1)** regularization — geometrically, L1's diamond-shaped penalty contours (with corners exactly on the weight axes) tend to push many weights to EXACTLY zero (sparsity), while L2's smooth circular contours shrink weights toward but not to zero. Modern research nuances this picture further: regularizing parameter count/size alone isn't the whole story — implicit regularization from the choice of optimizer, batch size, and normalization also meaningfully affects generalization in the overparametrized regime typical of modern deep learning. **Ensemble learning** (training multiple models and combining their predictions) genuinely improves generalization but is often prohibitively expensive to both train and deploy — a limitation directly addressed by **Dropout**, which randomly deactivates neurons (retained with probability p=0.5 for hidden nodes, p=0.8 for input nodes) during training, creating a different "thinned" network each batch while sharing ONE common set of weights across all `2ⁿ` possible configurations; at test time, the full (undropped) network is used with each neuron's output scaled by p, cheaply approximating an average over all those thinned networks. **Early stopping** — halting training at the point where validation error (never test error, which is reserved for one final unbiased evaluation) reaches its minimum — is one of the simplest, cheapest, and most widely used regularization techniques. Finally, **Data Augmentation** synthetically expands training data diversity through transformations like rotation, flipping, cropping, and noise injection, directly building the invariance properties introduced back in Lecture 1 into the training process itself.

`[← Lecture 07](../../Lecture-07-DNN-Optimization/README.md) · [🔝 Top](#dl-lecture-08--regularization-theory) · [Next: Numerical →](../numerical/dl_lecture08_regularization_numerical.md)`
