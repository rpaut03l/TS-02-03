# DL Lecture 04 — Recurrent Neural Networks (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

> Folder: `Deep-Learning/Lecture-04-Recurrent-Neural-Networks/numerical/`
> Pairs with: [`theory/dl_lecture04_rnn_theory.md`](../theory/dl_lecture04_rnn_theory.md) · [`practice/dl_lecture04_rnn_practice.md`](../practice/dl_lecture04_rnn_practice.md) · [`exercises/dl_lecture04_exercises.md`](../exercises/dl_lecture04_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — RNN Parameter Count](#worked-example-1--rnn-parameter-count)
3. [Worked Example 2 — Full Hidden State Update, By Hand](#worked-example-2--full-hidden-state-update-by-hand)
4. [Worked Example 3 — Unrolling 3 Timesteps](#worked-example-3--unrolling-3-timesteps)
5. [Worked Example 4 — Vanishing Gradient, Made Concrete](#worked-example-4--vanishing-gradient-made-concrete)
6. [Worked Example 5 — Exploding Gradient, Made Concrete](#worked-example-5--exploding-gradient-made-concrete)
7. [Worked Example 6 — Gradient Clipping By Hand](#worked-example-6--gradient-clipping-by-hand)
8. [Master Formula Cheatsheet](#master-formula-cheatsheet)
9. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
10. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| D | input size (e.g. vocabulary size for one-hot word vectors) |
| H | hidden state size (design choice) |
| k | output size (e.g. vocabulary size for next-word prediction) |
| U | weight matrix, input → hidden |
| W | weight matrix, hidden → hidden (the recurrent weight) |
| V | weight matrix, hidden → output |
| tanh | hyperbolic tangent activation, outputs in range (-1, 1) |

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Worked Example 1 — RNN Parameter Count

**Given:** D = 8 (vocabulary size / input dimension), H = 16 (hidden state size), k = 8 (output vocabulary size, same as input here).

**Step 1 — Shape and size of U (input → hidden).**
```
U shape = H x D = 16 x 8
U params = 16 x 8 = 128
```

**Step 2 — Shape and size of W (hidden → hidden, the recurrent weight).**
```
W shape = H x H = 16 x 16
W params = 16 x 16 = 256
```

**Step 3 — Shape and size of V (hidden → output).**
```
V shape = k x H = 8 x 16
V params = 8 x 16 = 128
```

**Step 4 — Total parameters (ignoring biases for simplicity, as in the slide's formula).**
```
Total = 128 + 256 + 128 = 512 parameters
```

**Key insight:** this parameter count does **not** grow with sequence length — whether your sentence has 5 words or 500 words, the SAME 512 parameters (U, W, V) are reused at every timestep. This is the direct numerical proof of "parameter sharing across time" from the theory file — compare this to a hypothetical (bad) design that learned separate weights per timestep, which would need 512 × (sequence length) parameters instead.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Worked Example 2 — Full Hidden State Update, By Hand

**Given:** a tiny RNN with input size D=2, hidden size H=2. Illustrative weights:
```
U = [[0.5, -0.2],
     [0.1,  0.3]]         (2x2)

W = [[0.9, 0.0],
     [0.0, 0.8]]         (2x2)

Initial hidden state: s_{-1} = [0, 0]  (starts empty, per theory)
First input: x_0 = [1, 0]
Activation: tanh
```

**Step 1 — Compute U x_0.**
```
U x_0 = [0.5x1 + (-0.2)x0,  0.1x1 + 0.3x0] = [0.5, 0.1]
```

**Step 2 — Compute W s_{-1}.**
```
W s_{-1} = [0.9x0 + 0.0x0, 0.0x0 + 0.8x0] = [0, 0]     (initial state is all zero, so this term vanishes)
```

**Step 3 — Sum and apply tanh.**
```
pre-activation = [0.5+0, 0.1+0] = [0.5, 0.1]
s_0 = tanh([0.5, 0.1]) = [tanh(0.5), tanh(0.1)] = [0.4621, 0.0997]
```

**Result: s_0 ≈ [0.4621, 0.0997]** — this is the network's "memory" after reading just the first input.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Worked Example 3 — Unrolling 3 Timesteps

**Continuing from Example 2**, feed a second input `x_1 = [0, 1]` using the same U, W.

**Step 1 — Compute U x_1.**
```
U x_1 = [0.5x0 + (-0.2)x1, 0.1x0 + 0.3x1] = [-0.2, 0.3]
```

**Step 2 — Compute W s_0 (using s_0 from Example 2).**
```
W s_0 = [0.9x0.4621 + 0.0x0.0997, 0.0x0.4621 + 0.8x0.0997]
      = [0.4159, 0.0798]
```

**Step 3 — Sum and apply tanh.**
```
pre-activation = [-0.2+0.4159, 0.3+0.0798] = [0.2159, 0.3798]
s_1 = tanh([0.2159, 0.3798]) = [0.2127, 0.3629]
```

**Result: s_1 ≈ [0.2127, 0.3629].** Notice this new hidden state is influenced by BOTH the new input `x_1` AND the memory `s_0` carried over from the previous step (via the `W s_0` term) — exactly the recurrence formula in action. If you continued to a third timestep `x_2`, you'd repeat this exact process again using `s_1` in place of `s_0` — this repeat-the-same-two-steps pattern, over and over, is literally what "unrolling" means.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Worked Example 4 — Vanishing Gradient, Made Concrete

**Given:** suppose at each timestep, backpropagation multiplies the running gradient by a derivative term of exactly 0.5 (a plausible saturating-tanh-region value).

**Step 1 — After 1 timestep back:** `gradient x 0.5^1 = 0.5`
**Step 2 — After 5 timesteps back:** `gradient x 0.5^5 = 0.03125`
**Step 3 — After 10 timesteps back:** `gradient x 0.5^10 = 0.0009765625`
**Step 4 — After 20 timesteps back:** `gradient x 0.5^20 ≈ 0.00000095`

**Result:** starting from a gradient contribution of 1.0, by the time you've backpropagated just 20 timesteps, the contribution has shrunk to about **0.00000095** — less than one-millionth of its original size. This numerically demonstrates exactly why information from 20+ words back (like the lecture's "banana" example) barely influences training at all: its gradient signal has vanished almost to nothing by the time it reaches that distant timestep.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Worked Example 5 — Exploding Gradient, Made Concrete

**Given:** suppose instead each timestep's derivative term is 1.5 (plausible if weights are larger than 1 and activations don't saturate).

**Step 1 — After 5 timesteps:** `1.5^5 = 7.59375`
**Step 2 — After 10 timesteps:** `1.5^10 ≈ 57.665`
**Step 3 — After 20 timesteps:** `1.5^20 ≈ 3325.26`
**Step 4 — After 50 timesteps:** `1.5^50 ≈ 637,621,500` (over 637 million!)

**Result:** the same repeated-multiplication mechanism, but now with a factor greater than 1, causes the gradient to explode to an enormous, numerically unstable size well before reaching 50 timesteps — concretely illustrating the "numeric overflow" danger described in the theory file.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Worked Example 6 — Gradient Clipping By Hand

**Given:** a computed gradient vector `g = [3, 4]` (its magnitude/L2-norm is `sqrt(3^2+4^2) = sqrt(25) = 5`), and a clipping threshold of `2`.

**Step 1 — Compute the gradient's norm.**
```
||g|| = sqrt(3^2 + 4^2) = sqrt(9+16) = sqrt(25) = 5
```

**Step 2 — Compare to threshold.** Since `5 > 2` (threshold), clipping is triggered.

**Step 3 — Rescale the gradient to have exactly the threshold's magnitude, preserving direction.**
```
g_clipped = g x (threshold / ||g||) = [3, 4] x (2/5) = [1.2, 1.6]
```

**Step 4 — Verify the new magnitude.**
```
||g_clipped|| = sqrt(1.2^2 + 1.6^2) = sqrt(1.44 + 2.56) = sqrt(4.0) = 2.0  ✓ matches threshold
```

**Result:** the clipped gradient `[1.2, 1.6]` points in exactly the same direction as the original `[3, 4]` (same ratio between components), but has been rescaled down to magnitude exactly 2 — preventing the exploding-gradient numeric overflow described in the theory file, while still moving the weights in the correct direction.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Hidden state update | `s_t = f(U x_t + W s_{t-1})` |
| Output | `o_t = softmax(V s_t)` |
| RNN parameter count | `(H×D) + (H×H) + (k×H)` |
| Vanishing gradient (n steps back, factor r<1) | `gradient × r^n → 0` as n grows |
| Exploding gradient (n steps back, factor r>1) | `gradient × r^n → ∞` as n grows |
| Gradient clipping | `g_clipped = g × (threshold / ‖g‖)`, applied only if `‖g‖ > threshold` |

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting the `W s_{t-1}` term entirely when computing a hidden state update by hand — always compute BOTH the `U x_t` and `W s_{t-1}` pieces separately, then sum, then activate.
- **Trap:** applying tanh/sigmoid before summing the two pieces, instead of after — the activation applies to the SUM `(U x_t + W s_{t-1})`, not to each piece individually.
- **Trap:** in gradient clipping, rescaling to the threshold value directly instead of `threshold/‖g‖` times the vector — always preserve direction, only rescale magnitude.
- **Exam hack:** for vanishing/exploding gradient "explain why" questions, always connect your answer back to repeated multiplication of per-timestep derivative factors — examiners specifically look for the "many numbers <1 multiplied together shrink; many numbers >1 multiplied together grow" mechanism, not just "long sequences are hard."
- **Exam hack:** RNN parameter-count questions are a favourite "notice it doesn't scale with sequence length" trick question — always explicitly state this fact for full marks, not just the number itself.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical)`

---

## Summary

This file turned Lecture 4's formulas into fully worked arithmetic. A parameter count for a small RNN (D=8, H=16, k=8) totalled 512 parameters across U, W, and V — and crucially, this count stays fixed regardless of sequence length, the numeric proof of cross-time parameter sharing. A full hand-computed hidden state update walked through `s_t = tanh(U x_t + W s_{t-1})` digit by digit for two consecutive timesteps, showing exactly how each new hidden state blends the current input with memory carried over from before. Two "made concrete" demonstrations showed vanishing gradients shrinking a factor-0.5-per-step signal down to under one-millionth of its original size after just 20 timesteps, and exploding gradients growing a factor-1.5-per-step signal past 637 million after 50 timesteps — turning the theory file's qualitative warnings into hard numbers. A final hand-computed gradient clipping example showed exactly how a gradient vector `[3,4]` (magnitude 5) gets rescaled down to `[1.2,1.6]` (magnitude exactly 2) when a clipping threshold of 2 is applied, preserving direction while capping magnitude. The master formula table consolidates every reusable calculation from this lecture for fast review.

`[← Theory](../theory/dl_lecture04_rnn_theory.md) · [🔝 Top](#dl-lecture-04--recurrent-neural-networks-numerical) · [Next: Practice →](../practice/dl_lecture04_rnn_practice.md)`
