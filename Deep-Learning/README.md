# Deep Learning — TS-02 Course Hub

`[🔝 Top](#deep-learning--ts-02-course-hub)`

> Repo: `rpaut03l/TS-02-03` → `Deep-Learning/`
> Course: Deep Learning (Core, 3-0-0[3]) | Instructor: Dr. Anushka Joshi, IIT Jodhpur
> Textbook reference: Goodfellow, Bengio & Courville — *Deep Learning*
> Source material: 13-lecture merged slide deck (541 pages), split below into 12 clean topic folders.

This hub is the single entry point into every Deep Learning lecture folder. Each lecture folder always has the same five-piece skeleton: **theory → numerical → practice → exercises → code**, so once you know your way around Lecture 01, every later lecture feels identical to navigate.

---

## Repo Structure (this course, inside TS-02-03)

```
TS-02-03/
└── Deep-Learning/
    ├── README.md                              <- you are here (course hub)
    │
    ├── Lecture-01-Introduction-to-Deep-Learning/
    │   ├── README.md                          <- lecture hub
    │   ├── theory/dl_lecture01_introduction_theory.md
    │   ├── numerical/dl_lecture01_introduction_numerical.md
    │   ├── practice/dl_lecture01_introduction_practice.md
    │   ├── exercises/dl_lecture01_exercises.md
    │   └── code/
    │       ├── README.md
    │       └── dl_lecture01_param_counter.py
    │
    ├── Lecture-02-Neural-Networks/             <- same 5-piece skeleton (pending)
    ├── Lecture-03-Convolutional-Neural-Networks/
    ├── Lecture-04-Recurrent-Neural-Networks/
    ├── Lecture-05-LSTM/
    ├── Lecture-06-Attention/
    ├── Lecture-07-DNN-Optimization/
    ├── Lecture-08-Regularization/
    ├── Lecture-09-Graph-Neural-Networks/
    ├── Lecture-10-Attention-and-Transformers/
    ├── Lecture-11-Real-World-End-to-End-Framework/
    ├── Lecture-12-Encoder-Decoder-and-VAE/
    │
    ├── Lecture-13-Generative-Adversarial-Networks/    <- BONUS, not in original deck
    ├── Lecture-14-Restricted-Boltzmann-Machines/       <- BONUS, not in original deck
    ├── Lecture-15-Transfer-Learning/                   <- BONUS, not in original deck
    │
    └── cumulative-review/                      <- added once 3-4 lectures exist
        └── dl_cumulative_exercises.md
```

**Naming rule applied everywhere:** `dl_lectureNN_<topic>_<theory|numerical|practice>.md`, all lowercase, hyphen/underscore separated, no spaces — matches your existing TS-01/TS-02 house style so GitHub anchor links never break.

---

## Lecture Status

| # | Lecture Folder | Topic (from slides) | Status |
|---|---|---|---|
| 01 | `Lecture-01-Introduction-to-Deep-Learning` | History, applications, data types, MLP limits, invariance, locality | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 02 | `Lecture-02-Neural-Networks` | Linear classifiers, 2-layer NN, gradient descent, non-linearity, Titanic practice | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 03 | `Lecture-03-Convolutional-Neural-Networks` | Convolution vs correlation, edges, filter types, separability, loss, regularizers, CNN architectures | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 04 | `Lecture-04-Recurrent-Neural-Networks` | RNN fundamentals, sequence modelling, BPTT, vanishing/exploding gradients | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 05 | `Lecture-05-LSTM` | Gates, memory cells, vanishing gradient fix | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 06 | `Lecture-06-Attention` | Attention mechanism fundamentals | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 07 | `Lecture-07-DNN-Optimization` | Optimizers, learning rate schedules, AutoML | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 08 | `Lecture-08-Regularization` | Dropout, weight decay, batch norm, early stopping | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 09 | `Lecture-09-Graph-Neural-Networks` | GNN fundamentals, message passing | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 10 | `Lecture-10-Attention-and-Transformers` | Full Transformer architecture, self-attention, encoder/decoder blocks | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 11 | `Lecture-11-Real-World-End-to-End-Framework` | Applied end-to-end DL project framework | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 12 | `Lecture-12-Encoder-Decoder-and-VAE` | Encoder-decoder, Variational Autoencoders | ✅ **Done** — full theory/numerical/practice/exercises/code |
| 13 (Bonus) | `Lecture-13-Generative-Adversarial-Networks` | GANs — Generator/Discriminator, minimax, DCGAN, GAN vs VAE | ✅ **Done** — full theory/numerical/practice/exercises/code — **not in the original slide deck**, built to complete the generative-modeling picture (see the lecture's own README for details) |
| 14 (Bonus) | `Lecture-14-Restricted-Boltzmann-Machines` | RBM — energy-based models, Contrastive Divergence, DBN | ✅ **Done** — full theory/numerical/practice/exercises/code — **not in the original slide deck**, added to cover the historical bridge to Lecture 12's Hinton 2006 reference |
| 15 (Bonus) | `Lecture-15-Transfer-Learning` | Feature extraction vs fine-tuning, GPT/BERT, domain shift | ✅ **Done** — full theory/numerical/practice/exercises/code — **not in the original slide deck**, added since GPT/BERT (cited in your own slides) are built entirely on this technique |

**All 12 lectures from the original slide deck are complete, plus three bonus lectures (GANs, RBMs, Transfer Learning) covering major topics the source material only mentioned in passing or not at all.** Every theory/numerical/practice/exercises/code file across the full course has been written, and every code file has been executed to verify its output matches the corresponding numerical worked examples exactly.

---

## Cross-links to your other repos

- Trimester 1 coursework: https://github.com/rpaut03l/TS-01
- MLOps production pipeline example (for when DL-Ops/MLOps concepts connect to real infra): https://github.com/rpaut03l/rptl_gn_mlops/tree/mlops-pipeline
- ML workflows blog post: https://www.rohitpatel.in/2025/11/machine-learning-workflows-ml-models.html

---

## How to Upload / Sync This Folder to GitHub

You already have `rpaut03l/TS-02-03` as the live repo. To add this `Deep-Learning/` folder:

```bash
# 1. Clone if you don't already have a local copy
git clone https://github.com/rpaut03l/TS-02-03.git
cd TS-02-03

# 2. Create a feature branch (never commit Deep Learning docs straight to main)
git checkout -b feature/deep-learning-lecture-01

# 3. Copy the new Deep-Learning/ folder into your local repo root
#    (drag-drop, or `cp -r` from wherever you downloaded/unzipped this deliverable)

# 4. Stage everything (make sure a .gitignore is in place first, per your usual workflow)
git add .

# 5. Commit with a clear message
git commit -m "docs(deep-learning): add Lecture 01 - Introduction to Deep Learning (theory/numerical/practice/exercises/code)"

# 6. Push the branch and open a PR
git push -u origin feature/deep-learning-lecture-01
# then open a Pull Request on GitHub, review, and merge into main
```

For every future lecture, repeat steps 2–6 with a new branch name like `feature/deep-learning-lecture-02`, so each lecture is reviewable as its own PR — matching your existing feature-branch → PR → merge workflow, never force-pushing, and never touching files outside the new lecture folder in any single PR.

---

## Where Code + Libraries Will Live (as lectures progress)

Lecture 1's `code/` folder is dependency-free by design (it's a conceptual lecture). From Lecture 02 onward, expect this pattern in each lecture's `code/` folder:

```
Lecture-NN-Topic-Name/code/
├── README.md              <- explains exactly which libraries are needed and why,
│                              plus how to run on Colab / Kaggle / local venv
├── requirements.txt        <- pinned library versions for that lecture's code only
└── dl_lectureNN_*.py / .ipynb
```

This keeps each lecture's dependencies isolated and easy to `pip install -r requirements.txt` per-lecture, rather than one giant shared environment file that gets harder to maintain as more lectures are added.

`[🔝 Top](#deep-learning--ts-02-course-hub)`
