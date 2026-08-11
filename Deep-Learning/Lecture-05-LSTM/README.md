# Lecture 05 — LSTM

`[← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-05--lstm)`

**Instructor:** Dr. Anushka Joshi, IIT Jodhpur | **Date:** May 2026 | **Source slides:** "LSTM" deck, parts 1 & 2 (combined)

Covers why vanilla RNNs struggle with long-range dependencies, Hochreiter & Schmidhuber's gating insight, the cell state vs hidden state distinction, all three gates (forget/input/output) plus candidate memory, gradient stability vs vanilla RNNs, known LSTM limitations, and usage patterns.

## Files in this lecture

| File | Focus |
|---|---|
| 📘 [`theory/dl_lecture05_lstm_theory.md`](theory/dl_lecture05_lstm_theory.md) | Gates, cell state, all 6 equations, gradient stability, known issues |
| 🔢 [`numerical/dl_lecture05_lstm_numerical.md`](numerical/dl_lecture05_lstm_numerical.md) | Parameter counts, full hand-computed timestep, forget-gate extremes, gradient decay comparison |
| ✍️ [`practice/dl_lecture05_lstm_practice.md`](practice/dl_lecture05_lstm_practice.md) | Official in-class Q, fill-in-blank, gate-matching drill, interview Qs |
| 🧪 [`exercises/dl_lecture05_exercises.md`](exercises/dl_lecture05_exercises.md) | Tiered Easy/Medium/Hard question bank with answer key |
| 💻 [`code/`](code/README.md) | A full LSTM cell built from scratch in NumPy — verified against every numerical example |

## Suggested reading order

Theory → Numerical → Practice → Exercises → Code.

`[← Lecture 04](../Lecture-04-Recurrent-Neural-Networks/README.md) · [← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-05--lstm) · [Next: Lecture 06 →](../Lecture-06-Attention/README.md)`
