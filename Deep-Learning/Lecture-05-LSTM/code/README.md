# DL Lecture 05 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-05--code)`

> Folder: `Deep-Learning/Lecture-05-LSTM/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture05_lstm_cell_from_scratch.py` | Implements a full LSTM cell (all 6 equations) as a small Python class using only NumPy, verified against the numerical README's hand-computed example, then run across a random toy sequence and a gradient-decay comparison. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code element | Theory/Numerical concept |
|---|---|
| `LSTMCell.__init__` | Four separate weight matrices (`W_f, W_i, W_c, W_o`) — the source of the "4× a vanilla RNN" parameter count |
| `LSTMCell.step()` | All six LSTM equations, in order: forget gate → input gate → candidate memory → cell state update → output gate → hidden state |
| `check_worked_example_2()` | Reproduces Worked Example 2 with the SAME hand-picked weights, so you can verify code and hand-arithmetic agree exactly |
| `check_parameter_count()` | Verifies the exact 4.0× LSTM-vs-RNN parameter ratio from Worked Example 1 |
| `check_gradient_decay()` | Reproduces Worked Example 4 — a 0.98-per-step forget gate retains far more gradient signal than a 0.5-per-step vanilla RNN factor |

## How to run this file

```bash
cd Deep-Learning/Lecture-05-LSTM/code
pip install numpy --break-system-packages
python3 dl_lecture05_lstm_cell_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required.

## Expected output (verified — produced by actually running this script)

- Check 1 reproduces the numerical file's hand-computed values almost exactly (f_t≈0.5987, i_t≈0.6271, C~_t≈0.5227, C_t≈0.6271, o_t≈0.6593, H_t≈0.3666 — tiny rounding differences only).
- Check 2 confirms the LSTM/vanilla-RNN weight ratio is exactly **4.00×**.
- Check 3 shows a real (randomly-initialized, untrained) LSTM cell processing a 5-step sequence, with forget/input gate values hovering near 0.5 (expected for random, untrained weights — a trained network would learn to push these toward 0 or 1 depending on the task).
- Check 4 reproduces the exact numbers from Worked Example 4: vanilla RNN decay collapses to ~0 by 20 steps, while the LSTM's 0.98 pathway still retains 0.3642 (36%) after 50 steps.

`[← Exercises](../exercises/dl_lecture05_exercises.md) · [🔝 Top](#dl-lecture-05--code) · [🔝 Lecture Hub](../README.md)`
