# Lecture 02 — Neural Networks

`[← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-02--neural-networks)`

**Instructor:** Dr. Anushka Joshi, IIT Jodhpur | **Date:** April 2026 | **Source slides:** "Neural Networks" deck (23 slides)

This lecture builds neural networks up from the simplest linear classifier (`f = Wx`) to a 2-layer network (`f = W2 max(0, W1x+b1) + b2`), walks through a single neuron's math and the 5-beat gradient descent training loop, and proves — both conceptually and mathematically — why non-linear activation functions are non-negotiable. It closes with real hands-on practice on the Kaggle Titanic dataset.

## Files in this lecture

| File | Focus |
|---|---|
| 📘 [`theory/dl_lecture02_neural_networks_theory.md`](theory/dl_lecture02_neural_networks_theory.md) | Linear classifiers, 2-layer NN, gradient descent, why non-linearity matters |
| 🔢 [`numerical/dl_lecture02_neural_networks_numerical.md`](numerical/dl_lecture02_neural_networks_numerical.md) | Shape-tracking, hand-computed forward passes, weight update arithmetic, linear-collapse proof |
| ✍️ [`practice/dl_lecture02_neural_networks_practice.md`](practice/dl_lecture02_neural_networks_practice.md) | Titanic starter checklist, fill-in-the-blank, explain-it-back, interview Qs |
| 🧪 [`exercises/dl_lecture02_exercises.md`](exercises/dl_lecture02_exercises.md) | Tiered (Easy/Medium/Hard) exam-style question bank with answer key |
| 💻 [`code/`](code/README.md) | NumPy-only 2-layer network solving XOR, trained with hand-written gradient descent |

## Suggested reading order

Theory → Numerical → Practice → Exercises → Code. Each file's bottom nav links straight into the next one.

`[← Lecture 01](../Lecture-01-Introduction-to-Deep-Learning/README.md) · [← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-02--neural-networks) · [Next: Lecture 03 →](../Lecture-03-Convolutional-Neural-Networks/README.md)`
