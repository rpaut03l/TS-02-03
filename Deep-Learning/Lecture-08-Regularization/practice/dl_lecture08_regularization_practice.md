# DL Lecture 08 — Regularization (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-08--regularization-practice)`

> Folder: `Deep-Learning/Lecture-08-Regularization/practice/`
> Pairs with: [`theory/dl_lecture08_regularization_theory.md`](../theory/dl_lecture08_regularization_theory.md) · [`numerical/dl_lecture08_regularization_numerical.md`](../numerical/dl_lecture08_regularization_numerical.md) · [`exercises/dl_lecture08_exercises.md`](../exercises/dl_lecture08_exercises.md)

---

## Table of Contents
1. [Official In-Class Questions](#official-in-class-questions)
2. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
3. [Explain-It-Back Prompts](#explain-it-back-prompts)
4. [Quick-Fire True / False](#quick-fire-true--false)
5. [Regularization Technique Matching Drill](#regularization-technique-matching-drill)
6. [Mini Interview-Style Round](#mini-interview-style-round)
7. [Summary](#summary)

---

## Official In-Class Questions

**Q1.** Is regularizing parameters enough?

<details>
<summary>Show answer</summary>

No — research (Neyshabur et al. 2015; Belkin et al. 2019) shows that good architecture choices combined with SGD itself already generalize surprisingly well in the overparametrized regime. This means IMPLICIT regularization properties — which optimizer is used, batch size, normalization layers, and so on — also meaningfully affect generalization, and should be considered alongside explicit parameter penalties (L1/L2), not as a complete substitute for them, but not ignored either.
</details>

**Q2.** Given a total of n nodes, what are the total number of thinned networks that can be formed?

<details>
<summary>Show answer</summary>

2ⁿ — since each of the n nodes can independently be either present or dropped (2 choices per node), multiplied across all n nodes.
</details>

**Q3.** After dropout training, if a weight is updated multiple times across different thinned networks, which weight is used during testing?

<details>
<summary>Show answer</summary>

It's impossible to literally aggregate the outputs of 2ⁿ thinned networks individually. Instead, the FULL (undropped) network is used at test time, with each neuron's output scaled by its retention probability p — this cheaply approximates averaging over all the thinned networks in one forward pass.
</details>

`[🔝 Top](#dl-lecture-08--regularization-practice)`

---

## Concept Check — Fill in the Blank

1. Total training loss = ______ + ______.
2. The L2 penalty is also known as ______, because it encourages weights to decay toward ______.
3. q=1 in the general Lq regularizer corresponds to the ______ regularizer.
4. Dropout nodes are typically retained with probability ______ for hidden nodes and ______ for input nodes.
5. During early stopping, the ______ set is used for tuning decisions, while the ______ set is used only once, at the very end.

<details>
<summary>Show answers</summary>

1. Data loss; Regularization
2. weight decay; zero
3. Lasso (L1)
4. 0.5; 0.8
5. validation; test
</details>

`[🔝 Top](#dl-lecture-08--regularization-practice)`

---

## Explain-It-Back Prompts

1. Explain overfitting using the "memorizing practice questions" analogy.
2. Explain why L2 regularization keeps the loss function quadratic and solvable in closed form.
3. Explain, using the diamond-vs-circle contour picture, why L1 produces sparse weights but L2 doesn't.
4. Walk through the dropout trick across two mini-batches, explaining exactly which weights get updated each time.
5. Explain why the test set must be used only once, connecting it to the purpose of the validation set.

`[🔝 Top](#dl-lecture-08--regularization-practice)`

---

## Quick-Fire True / False

1. Overparametrized models are always a bad idea and should be avoided. — **False** (models in practice are often deliberately overparametrized; good architecture + SGD can generalize well even so, especially combined with regularization).
2. L1 regularization tends to produce sparse weight vectors. — **True**.
3. Dropout literally trains 2ⁿ separate neural networks during training. — **False** (it trains ONE shared set of weights, with different thinned configurations each batch).
4. Early stopping requires no held-out data beyond the training set. — **False** (it requires a validation set).
5. Data augmentation creates entirely new, independently collected training examples. — **False** (it creates new samples via transformations of EXISTING data).

`[🔝 Top](#dl-lecture-08--regularization-practice)`

---

## Regularization Technique Matching Drill

| Technique | Key mechanism | Your match |
|---|---|---|
| L2 / weight decay | ? | |
| Dropout | ? | |
| Early stopping | ? | |
| Data augmentation | ? | |

Options: (a) Randomly zero out neurons during training, (b) Stop training when validation error is minimized, (c) Penalize the sum of squared weights, (d) Generate synthetic training samples via transformations

<details>
<summary>Show answers</summary>

L2/weight decay → (c). Dropout → (a). Early stopping → (b). Data augmentation → (d).
</details>

`[🔝 Top](#dl-lecture-08--regularization-practice)`

---

## Mini Interview-Style Round

**Q1.** "A model achieves 99% training accuracy but only 65% validation accuracy. Suggest three different regularization techniques from this lecture that could help, and briefly justify each."

<details>
<summary>Show answer</summary>

(1) Dropout — randomly deactivating neurons during training discourages the network from relying too heavily on any single neuron/co-adapted feature combination, forcing more robust, redundant representations. (2) Early stopping — if training has continued well past the point where validation loss was minimized, simply stopping earlier (or restoring an earlier checkpoint) directly addresses the gap. (3) Data augmentation — if the training set is small or not diverse enough, synthetically expanding it via transformations can reduce the model's ability to simply memorize specific training examples. L2 regularization (weight decay) is also a reasonable fourth option, directly penalizing large weights that often indicate overfitting.
</details>

**Q2.** "Explain to a teammate why using p=0.8 for input-layer dropout (rather than the usual p=0.5) makes sense."

<details>
<summary>Show answer</summary>

Input nodes carry raw, information-dense signal directly from the data — dropping them too aggressively (as with p=0.5) risks discarding genuinely important raw information the network never gets a chance to learn from at all for that example. Hidden nodes, by contrast, tend to have more redundant, overlapping representations (many hidden units may encode similar or correlated features), so dropping half of them (p=0.5) is more tolerable and still forces useful, non-redundant learning without starving the network of essential input signal.
</details>

**Q3.** "Someone claims early stopping and L2 regularization are 'basically the same thing.' Do you agree?"

<details>
<summary>Show answer</summary>

They're different techniques that happen to share a similar GOAL (preventing overfitting) but work through completely different mechanisms. Early stopping controls overfitting by limiting HOW LONG training runs, based on monitoring validation performance. L2 regularization controls overfitting by directly modifying the LOSS FUNCTION itself, penalizing large weight magnitudes throughout training regardless of how many epochs are run. In fact, they're often used TOGETHER, not as substitutes for each other — some research has drawn theoretical connections between the two under certain conditions, but they're implemented and reasoned about very differently in practice.
</details>

`[🔝 Top](#dl-lecture-08--regularization-practice)`

---

## Summary

This practice file drills Lecture 8's regularization toolbox through active recall. All three official in-class questions are answered in full — whether parameter regularization alone is enough (no — implicit regularization matters too), the 2ⁿ thinned-network count, and the test-time scaling solution to dropout's aggregation problem. A fill-in-the-blank check reinforces the data-loss-plus-regularization split, weight decay terminology, the Lasso/L1 connection, dropout's exact retention probabilities (0.5 hidden, 0.8 input), and the validation-vs-test-set distinction. Five explain-it-back prompts push you to reproduce the overfitting analogy, why L2 stays closed-form-solvable, the L1-vs-L2 geometric sparsity argument, the two-mini-batch dropout walkthrough, and the test-set-used-once principle. A quick-fire true/false round and a technique-matching drill test both conceptual accuracy and precise terminology across L2, dropout, early stopping, and data augmentation. A three-question interview-style round rehearses realistic scenarios: diagnosing a large train-validation gap with three concrete regularization suggestions, justifying dropout's different input-vs-hidden retention probabilities, and clarifying that early stopping and L2 regularization are complementary, not equivalent, techniques. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture08_regularization_numerical.md) · [🔝 Top](#dl-lecture-08--regularization-practice) · [Next: Exercises →](../exercises/dl_lecture08_exercises.md)`
