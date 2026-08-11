# DL Lecture 11 — Exercise Bank (Real-World End-to-End Framework)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-11--exercise-bank-real-world-end-to-end-framework)`

> Folder: `Deep-Learning/Lecture-11-Real-World-End-to-End-Framework/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture11_e2e_framework_theory.md) · [numerical](../numerical/dl_lecture11_e2e_framework_numerical.md) · [practice](../practice/dl_lecture11_e2e_framework_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q11.1.** List all six steps of the end-to-end pipeline, in order.

**Q11.2.** What does an ablation study measure?

**Q11.3.** Write the formula for bagging's variance reduction, for n independent models.

**Q11.4.** What real-world problem did the instructor's own cited research paper address?

**Q11.5.** What distinguishes stacking from simple prediction averaging?

---

## 🟡 Medium — Applied Reasoning

**Q11.6.** For feature x=[2,4,6,8] and target y=[1,3,3,5], compute the Pearson correlation coefficient by hand.

**Q11.7.** A grid search covers 6 hyperparameters, each with 2 possible values. Compute the total number of combinations, and compare it to a search over 4 hyperparameters each with 5 values.

**Q11.8.** A full model scores 88.4% accuracy. Removing component A drops it to 85.0%; removing component B drops it to 86.9%. Compute both deltas and rank the components by importance.

**Q11.9.** For a single model with prediction variance 9.0, compute the bagged variance using 3 models and using 9 models, and state the reduction factor in each case.

**Q11.10.** Explain why cascading is particularly useful in latency-sensitive real-time systems, referencing the "fast filter, then slow accurate model" structure.

---

## 🔴 Hard — Derivation & Multi-Step

**Q11.11.** An AdaBoost weak model has an error rate of e=0.25 on 4 samples (1 wrong, 3 right), all starting with equal weight 0.25. Compute alpha, the unnormalized updated weights, and the final normalized weights.

**Q11.12.** A stacking meta-model combines three base models with learned weights w1=0.4, w2=0.35, w3=0.25, given predictions p1=0.6, p2=0.9, p3=0.3. Compute the final stacked prediction, and compare it to the plain average of the three predictions.

**Q11.13.** Design a full ablation study plan (in words) for a hypothetical image classification model that uses data augmentation, dropout, batch normalization, and a residual connection. Explain what each individual ablation run would tell you, and in what order you'd recommend running them.

**Q11.14.** A cascading system has a fast Stage-1 filter that correctly routes 80% of "easy" cases directly to a final answer, sending the remaining 20% ("hard" cases) to a slower Stage-2 model. If Stage-1 takes 2ms per case and Stage-2 takes 50ms per case, compute the AVERAGE processing time per case across 1000 cases, and compare it to always using Stage-2 alone.

`[🔝 Top](#dl-lecture-11--exercise-bank-real-world-end-to-end-framework)`

---

## Answer Key

<details>
<summary>Q11.1 – Q11.5 (Easy)</summary>

- **Q11.1:** Input Dataset → Find Feature Correlation → Find Optimized Hyperparameters → Architecture Flowchart → Model Ablation Study → Final Model Selection.
- **Q11.2:** How much each individual model component contributes to overall performance, by removing/disabling it and measuring the resulting performance drop.
- **Q11.3:** `variance = σ²/n`.
- **Q11.4:** Early detection of earthquake magnitude (for earthquake early-warning purposes).
- **Q11.5:** Stacking trains a learned meta-model to combine base models' predictions (potentially with unequal, learned weights, or more complex combination rules), rather than simply averaging them with fixed, equal weight.
</details>

<details>
<summary>Q11.6 – Q11.10 (Medium)</summary>

- **Q11.6:** mean_x=5.0, mean_y=3.0. Covariance term = (2-5)(1-3)+(4-5)(3-3)+(6-5)(3-3)+(8-5)(5-3) = 6+0+0+6 = 12. std_x=√[(−3)²+(−1)²+1²+3²]=√20≈4.4721. std_y=√[(−2)²+0²+0²+2²]=√8≈2.8284. r = 12/(4.4721×2.8284) ≈ **0.9487** — a very strong positive correlation.
- **Q11.7:** 6 hyperparameters × 2 values each = 2⁶ = **64** combinations. 4 hyperparameters × 5 values each = 5⁴ = **625** combinations — even with FEWER hyperparameters, more VALUES per hyperparameter produced nearly 10× more combinations, illustrating that both the number of hyperparameters AND the number of values per hyperparameter drive combinatorial growth.
- **Q11.8:** Delta A = 88.4-85.0 = **3.4 percentage points**. Delta B = 88.4-86.9 = **1.5 percentage points**. Ranking: Component A is more important (larger performance drop when removed) than Component B.
- **Q11.9:** n=3: 9.0/3 = **3.0** (3× reduction). n=9: 9.0/9 = **1.0** (9× reduction). The variance reduction factor always equals n itself, under the idealized independence assumption.
- **Q11.10:** Most real-world inputs are "easy" (a fast, cheap model can handle them confidently and correctly), while only a minority are genuinely "hard" and need a slower, more accurate model's extra compute. By filtering with a fast Stage-1 model first, the system avoids paying the slow model's full cost for every single case — dramatically reducing AVERAGE latency across a real traffic distribution, while still reserving the accurate (but slow) model specifically for the harder cases that actually need it.
</details>

<details>
<summary>Q11.11 – Q11.14 (Hard)</summary>

- **Q11.11:** alpha = 0.5×ln(0.75/0.25) = 0.5×ln(3) ≈ 0.5×1.0986 ≈ **0.5493**. w_wrong (unnormalized) = 0.25×e^0.5493 ≈ 0.25×1.7321 ≈ **0.4330**. w_right (unnormalized) = 0.25×e^-0.5493 ≈ 0.25×0.5774 ≈ **0.1443**. Total = 1×0.4330 + 3×0.1443 ≈ 0.4330+0.4330 = 0.8660. Normalized w_wrong = 0.4330/0.8660 = **0.50**. Normalized w_right = 0.1443/0.8660 ≈ **0.1667** (each of the 3 correct samples).
- **Q11.12:** Stacked = 0.4×0.6 + 0.35×0.9 + 0.25×0.3 = 0.24+0.315+0.075 = **0.63**. Plain average = (0.6+0.9+0.3)/3 = 1.8/3 = **0.60**. The meta-model's weighted result (0.63) leans higher than the plain average (0.60), reflecting that it learned to trust model 2's higher prediction (0.9, weight 0.35) somewhat more than an equal-weighting scheme would.
- **Q11.13:** A reasonable plan: (1) Baseline — train the FULL model (augmentation + dropout + batch norm + residual connection) and record its accuracy. (2) Remove data augmentation only, retrain, record the drop. (3) Restore augmentation, remove dropout only, retrain, record the drop. (4) Restore dropout, remove batch normalization only, retrain, record the drop. (5) Restore batch norm, remove the residual connection only, retrain, record the drop. Running ablations ONE COMPONENT AT A TIME (rather than removing several at once) isolates each component's individual contribution cleanly; a sensible order is to ablate components you suspect matter LEAST first (to quickly confirm/refute assumptions) or, alternatively, in the order components were added during development, to mirror the model's actual build history.
- **Q11.14:** Average time = 0.8×2ms + 0.2×50ms = 1.6+10.0 = **11.6ms per case**, across 1000 cases → 11,600ms total (11.6 seconds). Always using Stage-2 alone: 1000×50ms = **50,000ms (50 seconds)**. The cascading system is roughly **4.3× faster on average** (11.6s vs 50s) while still routing every genuinely hard case through the accurate Stage-2 model.
</details>

`[🔝 Top](#dl-lecture-11--exercise-bank-real-world-end-to-end-framework)`

---

## Summary

This exercise bank drills Lecture 11's applied pipeline and ensembling mathematics across three tiers. Easy questions recall the full six-step pipeline, ablation study purpose, the bagging variance formula, the earthquake case study, and stacking's distinguishing feature. Medium questions apply the Pearson correlation formula to new data (r≈0.9487), compute and compare two different grid-search combination counts (64 vs 625), rank two ablated components by performance delta, apply the bagging variance formula at two different ensemble sizes, and reason about cascading's latency benefit. Hard questions require full derivations: a complete AdaBoost weight-update cycle with normalization, a stacking-vs-plain-averaging numeric comparison, a full worded ablation study design plan for a 4-component model, and a quantitative cascading-system latency analysis showing a 4.3× speedup over always using the slow, accurate model. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture11_e2e_framework_practice.md) · [🔝 Top](#dl-lecture-11--exercise-bank-real-world-end-to-end-framework) · [Code →](../code/README.md)`
