# DL Lecture 14 (Bonus) — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-14-bonus--code)`

> Folder: `Deep-Learning/Lecture-14-Restricted-Boltzmann-Machines/code/`
> ⚠️ Bonus lecture — see the theory file's header note.

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture14_rbm_from_scratch.py` | Implements the RBM energy function, hidden/visible activations, Gibbs sampling, and CD-k training — verified against the numerical README — then trains a genuinely working RBM on a toy 2-pattern binary dataset. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function/class | Theory/Numerical concept |
|---|---|
| `rbm_energy()` | `E(v,h)=-a^Tv-b^Th-v^TWh` |
| `hidden_prob()` / `visible_prob()` | The two sigmoid activation formulas |
| `gibbs_step()` | One v→h→v Gibbs sampling round |
| `RBM.train_step()` | A full CD-k update: positive phase, k-step negative phase, weight/bias updates |
| `RBM.reconstruct()` | Feed v in, sample h, reconstruct v — the autoencoder-like "dream" |

## How to run this file

```bash
cd Deep-Learning/Lecture-14-Restricted-Boltzmann-Machines/code
pip install numpy --break-system-packages
python3 dl_lecture14_rbm_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required for this toy problem.

## Expected output (verified — produced by actually running this script)

- Check 1 reproduces every numerical README worked example exactly: E(v,h)=-1.1, hidden probabilities ≈[0.6225,0.5250], visible reconstruction probabilities ≈[0.5987,0.6225,0.6900].
- Check 2 shows a REAL, working RBM: starting from ~50% reconstruction error (random weights, no information yet), training via CD-1 for 3,000 steps on two alternating toy binary patterns drives reconstruction error down to under 2% — a genuine, live demonstration of Contrastive Divergence successfully teaching the RBM to "remember" both patterns purely through its learned weights, exactly the "party mood energy landscape" story from the theory file made concrete.

`[← Exercises](../exercises/dl_lecture14_exercises.md) · [🔝 Top](#dl-lecture-14-bonus--code) · [🔝 Lecture Hub](../README.md)`
