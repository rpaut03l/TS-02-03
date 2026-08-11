# DL Lecture 04 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-04--code)`

> Folder: `Deep-Learning/Lecture-04-Recurrent-Neural-Networks/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture04_vanilla_rnn_from_scratch.py` | Implements `s_t = tanh(Ux_t + Ws_{t-1})` and `o_t = softmax(Vs_t)` in NumPy, runs a forward pass over a toy sentence, and numerically demonstrates vanishing gradients, exploding gradients, and gradient clipping. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## Note on this network being untrained

This file only implements the **forward pass** (not full BPTT training) — the weights are randomly initialized and never updated, so the "predicted_next_word" output is not meaningful (it's just an untrained network's random guess). The point of this file is to make the *mechanics* of the recurrence formula and the gradient problems fully visible and traceable to the theory/numerical READMEs; a full trainable version (with backpropagation through time implemented by hand, similar to Lecture 2's from-scratch training loop) is a great follow-up exercise once you're comfortable with these mechanics — and is exactly what real frameworks like PyTorch's `nn.RNN` automate for you in later, more advanced coursework.

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `rnn_step()` | `s_t = tanh(U x_t + W s_{t-1})` |
| `rnn_output()` | `o_t = softmax(V s_t)` |
| `demonstrate_vanishing_and_exploding_gradients()` | Worked Examples 4 & 5 — gradient shrinking/growing across timesteps |
| `clip_gradient()` | Worked Example 6 — gradient clipping formula |

## How to run this file

```bash
cd Deep-Learning/Lecture-04-Recurrent-Neural-Networks/code
pip install numpy --break-system-packages
python3 dl_lecture04_vanilla_rnn_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required.

## Expected output (verified — produced by actually running this script)

- The vanishing gradient demo shrinks from 0.5 (1 step back) down to about 0.00000095 (20 steps back) — matches Worked Example 4 exactly.
- The exploding gradient demo grows from 7.59 (5 steps back) up to about 637,621,500 (50 steps back) — matches Worked Example 5 exactly.
- The gradient clipping demo takes `[3,4]` (norm 5) down to `[1.2,1.6]` (norm exactly 2) — matches Worked Example 6 exactly.

`[← Exercises](../exercises/dl_lecture04_exercises.md) · [🔝 Top](#dl-lecture-04--code) · [🔝 Lecture Hub](../README.md)`
