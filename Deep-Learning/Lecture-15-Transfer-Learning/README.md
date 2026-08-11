# Lecture 15 (Bonus) — Transfer Learning

`[← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-15-bonus--transfer-learning)`

> ⚠️ **Bonus lecture — not part of the original 541-page slide deck.** Transfer learning isn't covered as a dedicated topic in the source material — it only appears via two paper-title citations (GPT and BERT, both literally about "pre-training"). Built to complete the picture, since it's one of the most practically important techniques in modern deep learning, directly underlying the models your own slides cite.

Covers the feature-hierarchy argument for why transfer learning works, Feature Extraction vs Fine-Tuning, the data-size × domain-similarity decision grid, learning rate scaling and layer freezing, ImageNet pretraining, GPT/BERT's pretrain-then-fine-tune paradigm, domain shift, and catastrophic forgetting.

## Files in this lecture

| File | Focus |
|---|---|
| 📘 [`theory/dl_lecture15_transfer_learning_theory.md`](theory/dl_lecture15_transfer_learning_theory.md) | Feature extraction vs fine-tuning, decision grid, GPT/BERT, domain shift |
| 🔢 [`numerical/dl_lecture15_transfer_learning_numerical.md`](numerical/dl_lecture15_transfer_learning_numerical.md) | Parameter counts, LR scaling, data efficiency, layer-wise schedules |
| ✍️ [`practice/dl_lecture15_transfer_learning_practice.md`](practice/dl_lecture15_transfer_learning_practice.md) | Fill-in-blank, strategy decision drill, interview Qs |
| 🧪 [`exercises/dl_lecture15_exercises.md`](exercises/dl_lecture15_exercises.md) | Tiered Easy/Medium/Hard question bank with answer key |
| 💻 [`code/`](code/README.md) | A real pretrain → Feature-Extract/Fine-Tune experiment in NumPy, with an honest discussion of its results |

## Suggested reading order

Theory → Numerical → Practice → Exercises → Code.

`[← Lecture 14](../Lecture-14-Restricted-Boltzmann-Machines/README.md) · [← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-15-bonus--transfer-learning)`
