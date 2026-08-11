# DL Lecture 15 (Bonus) — Exercise Bank (Transfer Learning)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-15-bonus--exercise-bank-transfer-learning)`

> Folder: `Deep-Learning/Lecture-15-Transfer-Learning/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture15_transfer_learning_theory.md) · [numerical](../numerical/dl_lecture15_transfer_learning_numerical.md) · [practice](../practice/dl_lecture15_transfer_learning_practice.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## 🟢 Easy — Definitions & Recall

**Q15.1.** Define transfer learning in one sentence.

**Q15.2.** Name the two main transfer learning strategies.

**Q15.3.** Why does fine-tuning typically use a smaller learning rate than training from scratch?

**Q15.4.** Name two real-world NLP models cited in this course that follow the pretrain-then-fine-tune paradigm.

**Q15.5.** What is catastrophic forgetting?

---

## 🟡 Medium — Applied Reasoning

**Q15.6.** For a pretrained backbone with 5,000,000 parameters and a new head mapping 256 features to 20 classes, compute the trainable parameter count under Feature Extraction and under Full Fine-Tuning, and their ratio.

**Q15.7.** For a base learning rate of 0.02, compute the fine-tuning learning rate using a 50× reduction.

**Q15.8.** A from-scratch model needs 80,000 images; a transfer learning approach needs 1,600 images for the same accuracy. Compute the data efficiency ratio.

**Q15.9.** Explain why "small target dataset + similar domain" favors Feature Extraction, while "large target dataset + different domain" favors Fine-Tuning (or training from scratch).

**Q15.10.** Explain why early CNN layers transfer well across very different tasks (e.g., natural photos to X-rays), while late layers often don't.

---

## 🔴 Hard — Derivation & Multi-Step

**Q15.11.** A 5-block network uses a layer-wise learning rate schedule where each block's rate TRIPLES compared to the previous block, starting at `η_1=0.00002`. Compute all 5 blocks' learning rates and the ratio between the last and first.

**Q15.12.** Explain, step by step, what would likely happen if you fine-tuned a pretrained model on a target dataset that has a COMPLETELY different label space and visual domain (e.g., pretrained on natural photos, target = abstract satellite radar signatures) using full fine-tuning with a large learning rate for many epochs. Connect your answer to both domain shift and catastrophic forgetting.

**Q15.13.** A team has two options: (a) Feature Extraction, needing 30 minutes of training and reaching 82% accuracy; (b) Full Fine-Tuning, needing 6 hours of training and reaching 89% accuracy. Compute the accuracy gained per hour of training time for each option, and discuss when each choice might be preferred despite the other's advantage.

**Q15.14.** Design a full progressive-unfreezing schedule (in words, with approximate epoch numbers) for a 4-block pretrained network being adapted to a moderately different target domain with a medium-sized dataset, explaining what happens at each stage and why.

`[🔝 Top](#dl-lecture-15-bonus--exercise-bank-transfer-learning)`

---

## Answer Key

<details>
<summary>Q15.1 – Q15.5 (Easy)</summary>

- **Q15.1:** Transfer learning is the practice of reusing a model trained on a data-rich source task as the starting point for a related, typically data-scarce target task.
- **Q15.2:** Feature Extraction and Fine-Tuning.
- **Q15.3:** To avoid catastrophic forgetting — large learning rate updates risk quickly overwriting the valuable knowledge already encoded in the pretrained weights, before the (typically small) target dataset can gently guide the model toward the new task instead.
- **Q15.4:** GPT and BERT.
- **Q15.5:** The risk that fine-tuning too aggressively (too large a learning rate, too much unfreezing, too many epochs) causes the network to overwrite/erase the valuable general knowledge learned during pretraining.
</details>

<details>
<summary>Q15.6 – Q15.10 (Medium)</summary>

- **Q15.6:** Head params = 256×20+20 = 5,140. FE trainable = **5,140**. Full fine-tune trainable = 5,000,000+5,140 = **5,005,140**. Ratio ≈ **973.8×**.
- **Q15.7:** 0.02/50 = **0.0004**.
- **Q15.8:** 80,000/1,600 = **50×** less data needed for transfer learning.
- **Q15.9:** With a small target dataset, training many parameters (as full fine-tuning would) risks severe overfitting — Feature Extraction's small number of trainable parameters (just the head) is much safer. When the domain is similar, the pretrained features are already well-suited, so freezing them (Feature Extraction) works well. With a LARGE target dataset, there's enough data to safely train more parameters without overfitting, and if the domain differs meaningfully, the network genuinely needs to adapt its internal features (not just its final decision layer) to perform well — both favor Fine-Tuning, or in the most extreme domain-difference cases, even training from scratch.
- **Q15.10:** Early CNN layers learn extremely generic, low-level visual patterns (edges, corners, simple textures, color gradients) that are useful for describing almost ANY kind of image, regardless of its specific content or domain — an edge is an edge whether it's in a photo of a cat or an X-ray. Late layers, by contrast, learn increasingly SPECIFIC combinations tuned to the exact categories/patterns of the original source task — these specific combinations often don't correspond to anything meaningful in a very different target domain, so they transfer poorly and typically need to be re-learned or heavily adjusted.
</details>

<details>
<summary>Q15.11 – Q15.14 (Hard)</summary>

- **Q15.11:** η1=0.00002, η2=0.00002×3=0.00006, η3=0.00006×3=0.00018, η4=0.00018×3=0.00054, η5=0.00054×3=0.00162. Ratio = η5/η1 = 0.00162/0.00002 = **81×**.
- **Q15.12:** This scenario combines LARGE domain shift (natural photos vs abstract radar signatures) with aggressive fine-tuning settings (large LR, full unfreezing, many epochs) — both factors that maximize risk. The large domain shift means the pretrained features are a poor starting match for the target domain to begin with (limited positive transfer, possible negative transfer). The aggressive fine-tuning settings then compound this by rapidly overwriting whatever useful generic knowledge (e.g., basic edge/texture detectors) DID transfer, via catastrophic forgetting, likely BEFORE the model has had a chance to learn genuinely useful target-domain-specific features from the — probably still comparatively limited — target data. The likely outcome: performance that's no better, and possibly worse, than simply training a smaller network from scratch on the target data alone, since the "shortcuts" transfer learning was supposed to provide have been squandered.
- **Q15.13:** Feature Extraction: 82% / 0.5 hours = **164% accuracy-points per hour**. Full Fine-Tuning: 89% / 6 hours ≈ **14.83% accuracy-points per hour**. Feature Extraction is vastly more "efficient" per unit of training time. Full Fine-Tuning would still be preferred when the absolute accuracy gap (89% vs 82%, a 7-point difference) genuinely matters for the application (e.g., a safety-critical system) and the extra 5.5 hours of one-time training cost is a worthwhile investment; Feature Extraction would be preferred for rapid prototyping, tight compute budgets, or when 82% is already sufficient for the use case.
- **Q15.14:** A reasonable schedule: Epochs 1-5: freeze the ENTIRE backbone (Blocks 1-4), train only the new head — lets the head adapt to the backbone's existing features first, without risking damage to the backbone. Epochs 6-10: unfreeze Block 4 (latest, most task-specific) only, using a moderate learning rate — allow the most task-specific features to start adapting to the target domain. Epochs 11-15: additionally unfreeze Block 3, using a smaller learning rate for it than Block 4 (since Block 3 is more generic than Block 4 but less generic than Blocks 1-2) — broaden adaptation cautiously. Epochs 16+ (optional, only if the medium-sized dataset and validation performance justify it): cautiously unfreeze Block 2 with an even smaller learning rate, while keeping Block 1 (earliest, most generic) frozen throughout, since it needs the least adaptation and has the most to lose from careless updates. This schedule directly implements progressive unfreezing: task-specific layers adapt earliest and most, generic layers adapt latest (if at all) and least.
</details>

`[🔝 Top](#dl-lecture-15-bonus--exercise-bank-transfer-learning)`

---

## Summary

This bonus exercise bank drills transfer learning strategy and mechanics across three tiers. Easy questions recall the core definition, the two main strategies, why fine-tuning uses smaller learning rates, real cited NLP examples (GPT, BERT), and catastrophic forgetting. Medium questions apply parameter-counting and learning-rate-scaling formulas to new numbers (973.8× parameter ratio; a 0.0004 fine-tuning rate; a 50× data efficiency ratio), and explain the reasoning behind the decision grid and why early layers transfer better than late layers. Hard questions require deeper reasoning: a 5-block tripling learning-rate schedule (81× final-to-first ratio), a detailed worked explanation of what goes wrong when large domain shift meets aggressive fine-tuning, an accuracy-per-hour efficiency comparison between Feature Extraction and Full Fine-Tuning with genuine trade-off discussion, and a complete, staged progressive-unfreezing schedule designed from scratch for a realistic 4-block network scenario. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture15_transfer_learning_practice.md) · [🔝 Top](#dl-lecture-15-bonus--exercise-bank-transfer-learning) · [Code →](../code/README.md)`
