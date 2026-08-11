# DL Lecture 02 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-02--code)`

> Folder: `Deep-Learning/Lecture-02-Neural-Networks/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture02_two_layer_nn_from_scratch.py` | Builds `f = W2 max(0, W1x+b1) + b2` using only NumPy, trains it with hand-written gradient descent on the classic XOR problem, and prints the loss shrinking step by step. |

## Library required

Just **NumPy** — nothing else. This is deliberate: the goal of this file is to make every single line of math from the theory/numerical READMEs visible in code, with zero "magic" hidden inside a deep learning framework. From later lectures (CNN onward), we introduce PyTorch, since hand-writing every gradient by hand stops being practical once architectures get bigger.

```bash
pip install numpy --break-system-packages
```

## Why XOR?

XOR (exclusive-or) is the smallest possible dataset that a **linear classifier cannot solve** — it's the numeric, hands-on version of the "ring around a center" story from the theory file. Points `(0,0)` and `(1,1)` belong to class 0; points `(0,1)` and `(1,0)` belong to class 1. Try drawing a single straight line on paper that separates these two groups — it's impossible. This tiny 4-row dataset is the standard "hello world" proof that a hidden layer + non-linear activation genuinely buys you something a linear model cannot achieve.

## What each part of the code maps back to

| Code section | Theory/Numerical concept |
|---|---|
| `forward()` function | `f = W2 max(0, W1x+b1) + b2` |
| `relu()` / `sigmoid()` | The activation functions discussed in theory |
| The `for step in range(...)` loop | The 5-beat gradient descent rhythm (Init → Present → Forward → Compare → Adjust) |
| `d_loss_d_...` variables | The calculus behind "Adjust weights based on error" — the actual gradients |
| `W1 -= learning_rate * d_loss_d_W1` | The weight update formula from the numerical file |

## How to run this file

1. **Locally (venv):**
   ```bash
   cd Deep-Learning/Lecture-02-Neural-Networks/code
   pip install numpy --break-system-packages
   python3 dl_lecture02_two_layer_nn_from_scratch.py
   ```
2. **Google Colab:** upload the `.py` file, then in a cell: `!python dl_lecture02_two_layer_nn_from_scratch.py` (NumPy is pre-installed on Colab).
3. **Kaggle:** works the same as Colab — no GPU needed, this is tiny CPU-only code.

## Expected output (verified — this exact output was produced by running the script)

```
Step    0 | Loss = 0.313926
Step 1000 | Loss = 0.000747
Step 2000 | Loss = 0.000321
Step 3000 | Loss = 0.000201
Step 4000 | Loss = 0.000145

Final predictions after training:
  input=[0. 0.]  true=0  predicted=0.014
  input=[0. 1.]  true=1  predicted=0.993
  input=[1. 0.]  true=1  predicted=0.993
  input=[1. 1.]  true=0  predicted=0.014
```

Note the code comment about the random seed: try changing `np.random.seed(1)` to a different number and re-running — sometimes you'll see the loss get "stuck" around 0.16–0.17 and never improve. That's a real, important phenomenon called a **local minimum**, and it's a preview of topics covered in depth in the DNN Optimization lecture later in this course.

`[← Exercises](../exercises/dl_lecture02_exercises.md) · [🔝 Top](#dl-lecture-02--code) · [🔝 Lecture Hub](../README.md)`
