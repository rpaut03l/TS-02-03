# DL Lecture 02 — Neural Networks (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-02--neural-networks-numerical)`

> Folder: `Deep-Learning/Lecture-02-Neural-Networks/numerical/`
> Pairs with: [`theory/dl_lecture02_neural_networks_theory.md`](../theory/dl_lecture02_neural_networks_theory.md) · [`practice/dl_lecture02_neural_networks_practice.md`](../practice/dl_lecture02_neural_networks_practice.md) · [`exercises/dl_lecture02_exercises.md`](../exercises/dl_lecture02_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Shape-Tracking a 2-Layer Network](#worked-example-1--shape-tracking-a-2-layer-network)
3. [Worked Example 2 — A Single Neuron, By Hand](#worked-example-2--a-single-neuron-by-hand)
4. [Worked Example 3 — Full Forward Pass (3 Inputs → 3 Hidden → 1 Output)](#worked-example-3--full-forward-pass-3-inputs--3-hidden--1-output)
5. [Worked Example 4 — Computing the Error and One Gradient Descent Nudge](#worked-example-4--computing-the-error-and-one-gradient-descent-nudge)
6. [Worked Example 5 — Proving Linear Layers Collapse Without Activation](#worked-example-5--proving-linear-layers-collapse-without-activation)
7. [Master Formula Cheatsheet](#master-formula-cheatsheet)
8. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
9. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| D | number of input features |
| H | number of hidden units |
| C | number of output classes |
| W1, b1 | first layer's weights and bias |
| W2, b2 | second layer's weights and bias |
| ReLU / max(0,·) | activation function: outputs the input if positive, else 0 |
| η (eta) | learning rate — how big a step gradient descent takes |
| ŷ (y-hat) | the network's predicted output |
| y | the true target output |

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Worked Example 1 — Shape-Tracking a 2-Layer Network

**Given:** D = 10 input features, H = 20 hidden units (a design choice), C = 3 output classes.

**Step 1 — Shape of W1.** W1 must map a D-length vector to an H-length vector, so:
```
W1 shape = H x D = 20 x 10
```

**Step 2 — Shape of b1.** One bias per hidden unit:
```
b1 shape = H x 1 = 20 x 1
```

**Step 3 — Compute W1x shape.**
```
(H x D) . (D x 1) = (20 x 10) . (10 x 1) = (20 x 1)
```
The inner dimensions (10 and 10) match and cancel out, leaving a 20×1 result — exactly H values, one per hidden unit.

**Step 4 — Shape of W2.** W2 must map an H-length vector to a C-length vector:
```
W2 shape = C x H = 3 x 20
```

**Step 5 — Compute W2 * (hidden output) shape.**
```
(C x H) . (H x 1) = (3 x 20) . (20 x 1) = (3 x 1)
```
Result: exactly C=3 class scores, as required.

**Step 6 — Count total learnable parameters.**
```
W1 params = H x D = 20 x 10 = 200
b1 params = H     = 20
W2 params = C x H = 3 x 20 = 60
b2 params = C     = 3
-----------------------------------
Total     = 200 + 20 + 60 + 3 = 283 parameters
```

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Worked Example 2 — A Single Neuron, By Hand

**Given** (straight from the lecture's example): three inputs `x1 = -0.06`, `x2 = -2.5`, `x3 = 1.4`, arriving on wires with weights we will choose as `w1 = 0.5`, `w2 = -0.3`, `w3 = 0.8`, and bias `b = 0.1` (illustrative values, since the lecture only shows the inputs).

**Step 1 — Multiply each input by its weight.**
```
w1*x1 = 0.5  x (-0.06) = -0.03
w2*x2 = -0.3 x (-2.5)  = 0.75
w3*x3 = 0.8  x (1.4)   = 1.12
```

**Step 2 — Sum the products.**
```
sum = -0.03 + 0.75 + 1.12 = 1.84
```

**Step 3 — Add the bias.**
```
z = sum + b = 1.84 + 0.1 = 1.94
```

**Step 4 — Apply the activation function.** Using ReLU, `max(0, z)`:
```
output = max(0, 1.94) = 1.94   (positive, so ReLU passes it through unchanged)
```

**Result: this single neuron outputs 1.94.** If instead `z` had come out negative (say −0.5), ReLU would have clipped it to exactly `0`.

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Worked Example 3 — Full Forward Pass (3 Inputs → 3 Hidden → 1 Output)

This reproduces the lecture's own worked diagram: inputs `6.4, 2.8, 1.7`, true target class `y = 1`, network output `ŷ = 0.9`. Let's show one plausible full forward pass that could produce that 0.9, using illustrative weights (the slide does not print its exact hidden weights, so we choose clean numbers and verify the arithmetic end to end).

**Setup:** 3 inputs → 3 hidden neurons (fully connected) → 1 output neuron (fully connected).

**Step 1 — Hidden neuron 1.** Say its weights are `[0.2, 0.1, 0.3]` and bias `0.0`:
```
z_h1 = (0.2)(6.4) + (0.1)(2.8) + (0.3)(1.7) + 0.0
     = 1.28 + 0.28 + 0.51
     = 2.07
h1 = max(0, 2.07) = 2.07
```

**Step 2 — Hidden neuron 2.** Say its weights are `[0.05, -0.2, 0.4]` and bias `0.0`:
```
z_h2 = (0.05)(6.4) + (-0.2)(2.8) + (0.4)(1.7) + 0.0
     = 0.32 - 0.56 + 0.68
     = 0.44
h2 = max(0, 0.44) = 0.44
```

**Step 3 — Hidden neuron 3.** Say its weights are `[-0.1, 0.15, 0.05]` and bias `0.0`:
```
z_h3 = (-0.1)(6.4) + (0.15)(2.8) + (0.05)(1.7) + 0.0
     = -0.64 + 0.42 + 0.085
     = -0.135
h3 = max(0, -0.135) = 0   (ReLU clips negative values to 0)
```

**Step 4 — Output neuron.** Say output weights are `[0.3, 0.9, 0.2]` and bias `0.1`, combining h1, h2, h3:
```
z_out = (0.3)(2.07) + (0.9)(0.44) + (0.2)(0) + 0.1
      = 0.621 + 0.396 + 0 + 0.1
      = 1.117
```

**Step 5 — Final squashing (e.g., sigmoid, common for a single output "probability" neuron).** Using sigmoid, `σ(z) = 1 / (1 + e^-z)`:
```
sigma(1.117) = 1 / (1 + e^-1.117) = 1 / (1 + 0.3273) = 1 / 1.3273 ≈ 0.753
```

Our illustrative weights gave `≈0.753`, not exactly the slide's `0.9` — that's expected, since real weights are *learned*, not hand-picked. The point of this worked example is the **mechanical process** (multiply → sum → bias → activate, layer by layer) — running it correctly, digit by digit, is exactly what you'll be asked to do in a numerical exam question with *given* weights.

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Worked Example 4 — Computing the Error and One Gradient Descent Nudge

**Given:** true target `y = 1`, network output `ŷ = 0.9` (using the slide's own numbers now).

**Step 1 — Compute the error (simplest form: absolute/linear error).**
```
error = y - y_hat = 1 - 0.9 = 0.1
```

**Step 2 — Conceptual weight update rule (simplified gradient descent).**
```
w_new = w_old + eta x error x (input that fed that weight) x (activation derivative)
```

**Step 3 — Plug in illustrative numbers.** Suppose one particular weight `w = 0.3` connected an input of `2.07` (our `h1` from Example 3) to the output neuron, and we use a small learning rate `η = 0.01`, and (for simplicity) assume the local derivative term ≈ 1:
```
w_new = 0.3 + (0.01 x 0.1 x 2.07 x 1)
      = 0.3 + 0.00207
      = 0.30207
```

**Result:** the weight moved *very slightly* from `0.3` to `0.30207` — a tiny nudge in the direction that would reduce the error next time this same input pattern is seen. This is "gradient descent" in its simplest possible hand-computed form: notice the nudge is proportional to (a) how big the error was, (b) how big the learning rate is, and (c) how big the input was — a common exam question asks you to reason about what happens if any one of these three changes (e.g., "what happens if the learning rate is too large?").

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Worked Example 5 — Proving Linear Layers Collapse Without Activation

**Claim from theory:** stacking linear layers without non-linear activation functions collapses into one single linear layer. Let's verify this with small, concrete matrices.

**Step 1 — Pick tiny example matrices.** Let `X` be a 1×2 input row vector, and:
```
X  = [1, 2]
W1 = [[2, 0],
      [0, 3]]          (2x2)
W2 = [[1],
      [1]]              (2x1)
```

**Step 2 — Compute layer 1 output (no activation): X W1.**
```
X W1 = [1x2 + 2x0, 1x0 + 2x3] = [2, 6]
```

**Step 3 — Compute layer 2 output: (X W1) W2.**
```
[2, 6] . [[1],[1]] = (2x1 + 6x1) = 8
```

**Step 4 — Now compute W_combined = W1 W2 directly, and apply it to X in one step.**
```
W1 W2 = [[2,0],[0,3]] . [[1],[1]] = [[2x1+0x1],[0x1+3x1]] = [[2],[3]]

X . W_combined = [1, 2] . [[2],[3]] = (1x2 + 2x3) = 2 + 6 = 8
```

**Result: both routes give exactly 8.** Whether you apply `W1` then `W2` separately, or pre-multiply them into one combined matrix `W_combined = W1 W2` and apply it once, the answer is identical — **because there was no non-linear activation function breaking up the two multiplications.** This is the fully numeric proof behind the theory file's claim that depth without non-linearity buys you nothing beyond a single linear layer.

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Linear classifier | `f = Wx`, `x∈R^D`, `W∈R^(C×D)` |
| 2-layer NN | `f = W2 max(0, W1x+b1) + b2` |
| Shape check, layer 1 | `(H×D)(D×1) = (H×1)` |
| Shape check, layer 2 | `(C×H)(H×1) = (C×1)` |
| Total parameter count | `H×D + H + C×H + C` |
| Single neuron | `output = activation( Σ wi·xi + b )` |
| ReLU | `max(0, z)` |
| Sigmoid | `σ(z) = 1 / (1 + e^-z)` |
| Error (simple) | `error = y − ŷ` |
| Weight update (simplified) | `w_new = w_old + η × error × input × activation_derivative` |
| No-activation collapse | `X W1 W2 = X (W1 W2) = X W_combined` |

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** applying ReLU and forgetting that it clips *negative* values to exactly `0`, not to some small negative number — Worked Example 3's `h3` is a deliberate reminder of this.
- **Trap:** mixing up which matrix comes first when checking shapes — always write the operation as `(output shape) = (weight shape) . (input shape)` and cancel the matching inner dimensions, exactly as shown in Worked Example 1.
- **Trap:** forgetting the bias terms when counting total parameters — Worked Example 1 shows they're small in count but still required for full marks.
- **Exam hack:** if a question gives you exact weights and inputs and asks for the forward pass output, follow the *exact* four-step rhythm every time: multiply → sum → add bias → activate — never skip a step, even mentally, since partial marks are usually awarded per step shown.
- **Exam hack:** the "collapse without activation" proof (Worked Example 5) is a favourite short-answer/derivation question — practice reproducing it with your own small 2×2 matrices until you can do it without looking.

`[🔝 Top](#dl-lecture-02--neural-networks-numerical)`

---

## Summary

This file turned Lecture 2's every claim into fully shown, digit-by-digit arithmetic. Shape-tracking a 2-layer network with D=10, H=20, C=3 walks through exactly how `(H×D)(D×1)=(H×1)` and `(C×H)(H×1)=(C×1)` chain together, and totals 283 learnable parameters once biases are included. A single neuron's computation — multiply each input by its weight, sum the products, add a bias, then squash through ReLU — was computed by hand on the lecture's own example inputs (−0.06, −2.5, 1.4). A full illustrative forward pass through a 3-input → 3-hidden → 1-output network reproduced the lecture's own worked scenario (target 1, predicted 0.9), showing every hidden neuron's linear combination, ReLU activation (including one neuron correctly clipping to exactly 0), and the final sigmoid-squashed output. The error `y − ŷ = 1 − 0.9 = 0.1` was then used to demonstrate one simplified gradient descent weight update by hand, making the abstract "nudge the weights to reduce error" idea from the theory file completely concrete with real numbers. Finally, a from-scratch numeric proof with tiny 2×2 matrices verified that stacking linear layers without a non-linear activation function in between is mathematically identical to using one single combined linear layer — the exact numeric backbone behind why activation functions are non-negotiable in deep networks. The master formula table consolidates every reusable formula from this lecture into one place for fast pre-exam review.

`[← Theory](../theory/dl_lecture02_neural_networks_theory.md) · [🔝 Top](#dl-lecture-02--neural-networks-numerical) · [Next: Practice →](../practice/dl_lecture02_neural_networks_practice.md)`
