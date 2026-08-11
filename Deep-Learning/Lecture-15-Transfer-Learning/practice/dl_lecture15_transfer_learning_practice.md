# DL Lecture 15 (Bonus) — Transfer Learning (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice)`

> Folder: `Deep-Learning/Lecture-15-Transfer-Learning/practice/`
> Pairs with: [`theory/dl_lecture15_transfer_learning_theory.md`](../theory/dl_lecture15_transfer_learning_theory.md) · [`numerical/dl_lecture15_transfer_learning_numerical.md`](../numerical/dl_lecture15_transfer_learning_numerical.md) · [`exercises/dl_lecture15_exercises.md`](../exercises/dl_lecture15_exercises.md)
> ⚠️ Bonus lecture — see the theory file's header note.

---

## Table of Contents
1. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
2. [Explain-It-Back Prompts](#explain-it-back-prompts)
3. [Quick-Fire True / False](#quick-fire-true--false)
4. [Strategy Decision Drill](#strategy-decision-drill)
5. [Mini Interview-Style Round](#mini-interview-style-round)
6. [Summary](#summary)

---

## Concept Check — Fill in the Blank

1. Transfer learning reuses a model trained on a ______ task as a starting point for a ______ task.
2. In Feature Extraction, the pretrained network is ______, and only the new ______ is trained.
3. Fine-tuning typically uses a ______ learning rate than training from scratch.
4. GPT and BERT are both real-world examples of the ______-then-______ paradigm.
5. ______ describes source and target data distributions differing meaningfully, which can cause ______ transfer.

<details>
<summary>Show answers</summary>

1. source; target
2. frozen; head
3. smaller
4. pretrain; fine-tune
5. Domain shift; negative
</details>

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice)`

---

## Explain-It-Back Prompts

1. Explain the violinist-learning-viola analogy for transfer learning in your own words.
2. Explain the feature hierarchy argument for why transfer learning works, connecting back to Lecture 3's CNN discussion.
3. Walk through the decision grid (data size × domain similarity) from memory.
4. Explain why fine-tuning uses a smaller learning rate than training from scratch, connecting back to Lecture 7's update formula.
5. Explain catastrophic forgetting and how progressive unfreezing helps prevent it.

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice)`

---

## Quick-Fire True / False

1. Feature extraction trains the entire pretrained network, including all its original layers. — **False** (it freezes the pretrained network and trains only a new head).
2. Fine-tuning should generally use a LARGER learning rate than training from scratch. — **False** (a much SMALLER one, to protect pretrained knowledge).
3. GPT's name literally references "Generative Pre-Training." — **True**.
4. Transfer learning always improves performance compared to training from scratch, regardless of domain shift. — **False** (negative transfer is a real risk under large domain shift).
5. Progressive unfreezing typically unfreezes early (generic) layers first. — **False** (it unfreezes LATE, task-specific layers first, keeping early layers frozen longest).

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice)`

---

## Strategy Decision Drill

For each scenario, decide: Feature Extraction, Fine-Tuning, or Train From Scratch?

1. 200 labelled X-ray images, target task very different from ImageNet's natural photos.
2. 50,000 labelled natural photos, target task similar to ImageNet.
3. 5 million labelled images, a completely novel sensor modality with no related pretrained model available.

<details>
<summary>Show answers</summary>

1. Likely Fine-Tuning (small data pushes toward transfer learning, but domain difference means feature extraction alone may underperform — fine-tuning later layers, cautiously, is a reasonable middle ground) — or Feature Extraction from early layers only, per the theory file's "tricky" quadrant.
2. Fine-Tuning (or even mostly Feature Extraction) — large-ish data plus domain similarity make either strategy viable, with fine-tuning often slightly ahead given the data volume supports it.
3. Train From Scratch — abundant data plus no related pretrained source available removes transfer learning's main advantages entirely.
</details>

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice)`

---

## Mini Interview-Style Round

**Q1.** "Your team fine-tunes a pretrained model on a small target dataset using the SAME learning rate used for training from scratch, and validation accuracy gets WORSE than the pretrained model's original zero-shot performance. What likely happened?"

<details>
<summary>Show answer</summary>

Likely catastrophic forgetting — a learning rate that's appropriate for training from scratch is typically far too large for fine-tuning, causing large, destructive weight updates that overwrite the valuable pretrained knowledge faster than the small target dataset can meaningfully guide it toward the new task. The fix: reduce the learning rate by 10-100x, and/or freeze more layers (favor Feature Extraction or partial fine-tuning over full fine-tuning) given the small dataset size.
</details>

**Q2.** "Explain to a teammate why GPT and BERT don't need to be retrained from scratch for every new NLP application."

<details>
<summary>Show answer</summary>

Both were pretrained ONCE, at great computational expense, on a huge, generic text corpus, learning broad, general-purpose language understanding. For any new downstream task (sentiment analysis, question answering, etc.), you only need to fine-tune (or, for very large models, simply prompt) that SAME pretrained foundation on a comparatively small amount of task-specific data — exactly the transfer learning paradigm, letting many different downstream applications all benefit from one expensive pretraining run.
</details>

`[🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice)`

---

## Summary

This practice file drills the bonus transfer learning lecture's concepts through active recall. A fill-in-the-blank check reinforces the source/target task terminology, Feature Extraction's frozen-network-plus-new-head structure, fine-tuning's smaller learning rate, the GPT/BERT pretrain-then-fine-tune paradigm, and domain shift/negative transfer terminology. Five explain-it-back prompts push you to reproduce the violinist analogy, the feature hierarchy argument, the decision grid, the learning-rate caution (tied to Lecture 7), and catastrophic forgetting with its progressive-unfreezing mitigation. A quick-fire true/false round and a three-scenario strategy decision drill test both conceptual accuracy and applied judgment across genuinely different data-size/domain-similarity combinations. A two-question interview-style round rehearses realistic debugging (diagnosing catastrophic forgetting from a too-large fine-tuning learning rate) and explaining the GPT/BERT paradigm to a colleague. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture15_transfer_learning_numerical.md) · [🔝 Top](#dl-lecture-15-bonus--transfer-learning-practice) · [Next: Exercises →](../exercises/dl_lecture15_exercises.md)`
