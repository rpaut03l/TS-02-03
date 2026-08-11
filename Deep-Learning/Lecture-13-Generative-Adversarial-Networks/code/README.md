# DL Lecture 13 (Bonus) — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-13-bonus--code)`

> Folder: `Deep-Learning/Lecture-13-Generative-Adversarial-Networks/code/`
> ⚠️ Bonus lecture — see the theory file's header note.

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture13_gan_from_scratch.py` | Implements the Discriminator/Generator losses (both variants), their gradient magnitudes, the global optimum check, a toy FID calculation, and — going further than the other lectures' code files — a genuinely TRAINABLE tiny 1D GAN that learns via real adversarial training, not just formula verification. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function/class | Theory/Numerical concept |
|---|---|
| `discriminator_loss()` | `L_D = -[log(D(x))+log(1-D(G(z)))]` |
| `generator_loss_minimax()` / `generator_loss_nonsaturating()` | The two Generator loss formulations |
| `minimax_grad_magnitude()` / `nonsaturating_grad_magnitude()` | The vanishing-gradient-vs-fix comparison |
| `toy_fid_1d()` | The simplified 1D FID formula |
| `TinyGAN1D` | A REAL, trainable linear Generator + Discriminator pair, with hand-derived gradients for both networks and the alternating training loop |

## How to run this file

```bash
cd Deep-Learning/Lecture-13-Generative-Adversarial-Networks/code
pip install numpy --break-system-packages
python3 dl_lecture13_gan_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required for this 1D toy problem (real image GANs absolutely need GPUs).

## An honest result from Check 5 — a real (partial) mode collapse, live

Running `run_tiny_gan_training_demo()`, the Generator's MEAN does converge toward the real distribution's target (4.0) — but its STANDARD DEVIATION collapses down toward roughly 0.25–0.4, well short of the real data's target std of 1.5. **This is not a bug — it's a genuine, live example of the mode collapse phenomenon described in the theory file**, happening naturally with this simple linear Generator: once the Generator finds outputs clustered tightly around the right MEAN, that's often enough to fool a similarly simple linear Discriminator most of the time, so there's little pressure pushing it to also match the full SPREAD of the real distribution. This is exactly the kind of real, observable instability that motivated DCGAN's more careful architecture guidelines and techniques like Wasserstein GAN — a small, honest, hands-on taste of why GAN training has a reputation for being finicky, even in a problem this simple.

## Expected output (verified — produced by actually running this script)

- Checks 1–4 reproduce every numerical README worked example exactly (L_D≈0.3285; L_G minimax≈-0.2231, non-saturating≈1.6094; the gradient ratio table showing ~99× at D(G(z))=0.01; global optimum loss=1.3863; toy FID=0.29 and 0.0).
- Check 5 shows the tiny GAN's Discriminator loss oscillating (expected — the classic adversarial "tug of war" signature) while the Generator's mean drifts toward the real target of 4.0, with the std-collapse behaviour discussed above.

`[← Exercises](../exercises/dl_lecture13_exercises.md) · [🔝 Top](#dl-lecture-13-bonus--code) · [🔝 Lecture Hub](../README.md)`
