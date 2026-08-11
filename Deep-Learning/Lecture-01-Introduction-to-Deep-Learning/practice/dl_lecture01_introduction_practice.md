# DL Lecture 01 — Introduction to Deep Learning (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

> Folder: `Deep-Learning/Lecture-01-Introduction-to-Deep-Learning/practice/`
> Pairs with: [`theory/dl_lecture01_introduction_theory.md`](../theory/dl_lecture01_introduction_theory.md) · [`numerical/dl_lecture01_introduction_numerical.md`](../numerical/dl_lecture01_introduction_numerical.md) · [`exercises/dl_lecture01_exercises.md`](../exercises/dl_lecture01_exercises.md)
> Purpose: talk-through, conceptual "explain it back" practice. For a pure graded question bank, see the exercises file.

---

## Table of Contents
1. [How to Use This File](#how-to-use-this-file)
2. [Official In-Class Homework Questions](#official-in-class-homework-questions)
3. [Concept Check — Fill in the Blank Story](#concept-check--fill-in-the-blank-story)
4. [Explain-It-Back Prompts](#explain-it-back-prompts)
5. [Quick-Fire True / False](#quick-fire-true--false)
6. [Diagram Labelling Practice](#diagram-labelling-practice)
7. [Summary](#summary)

---

## How to Use This File

Answers are hidden inside `<details>` spoiler tags — try to answer out loud or on paper *before* clicking them open. This mirrors how the lecture itself works: it poses a question, waits, then reveals the answer on the next slide.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Official In-Class Homework Questions

These are the exact two homework prompts given at the end of the L1 slide deck.

**Q1.** Audio data is often represented as a one-dimensional sequence. When might you want to impose locality and translation invariance for audio?

<details>
<summary>Show answer</summary>

You'd want locality + translation invariance whenever a short sound *pattern* (like a phoneme, a drumbeat, or a specific word) can occur at **any point in time** in the audio clip, and should be recognized the same way no matter *when* it happens. Example: detecting the phoneme "sh" in speech — it sounds the same whether it occurs at the 2nd second or the 20th second of a recording. Locality says: you only need a small time-window around that sound to detect it (you don't need the whole clip). Translation invariance (here it's really *time*-shift invariance) says: the same small detector should fire correctly no matter *where in time* the pattern occurs. This is exactly why 1D convolutions are used heavily for audio and speech processing.
</details>

**Q2.** Do you think convolutional layers might also be applicable for text data?

<details>
<summary>Show answer</summary>

Yes — text can be thought of as a 1D sequence of tokens/word-embeddings, and many useful patterns (like a 3-word phrase, or a common suffix pattern) can occur at different positions in a sentence while still meaning the same thing. A 1D convolution can slide a small filter across the sequence of word vectors to detect such local patterns regardless of position — this is the basis of 1D-CNN text classifiers. The caveat: unlike images or short audio clips, text also has strong *long-range* dependencies (a pronoun 30 words later referring back to a noun) that a small local convolution window struggles to capture well — which is part of why RNNs, LSTMs, and eventually Attention/Transformers (covered in later lectures) became so important for text.
</details>

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Concept Check — Fill in the Blank Story

Fill in each blank, then reveal.

1. A toddler learning to recognize a dog is a real-world analogy for how deep networks learn ______ features automatically instead of a human hand-engineering them.
2. RGB is called an ______ color model because you start from black and add light; CMY is called a ______ color model because you start from white and absorb light.
3. The property that lets a "Waldo detector" work no matter where Waldo is standing is called ______.
4. The property that says a local pattern only needs nearby pixels to be detected is called ______.
5. LeCun's 1998 model was trained on the ______ handwritten digit dataset.

<details>
<summary>Show answers</summary>

1. hierarchical
2. additive; subtractive
3. (translation) invariance
4. locality
5. MNIST
</details>

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Explain-It-Back Prompts

Say your answer out loud in your own words (this is the single best way to expose gaps before an exam does it for you):

1. Explain, without using the word "convolution," why a fully connected layer is a bad choice for a large image.
2. Explain the four-step journey from "MLP weight tensor" to "convolution kernel" using the Waldo story as your guide.
3. Why does the lecture insist Deep Learning is not "just neural networks," but specifically *many-layered* neural networks?
4. Give one real application each for low-level, mid-level, and high-level vision tasks, in your own words (not copied from the slide list).
5. Explain why "1 megapixel image, 1000 hidden neurons, 10⁹ parameters" is considered a *small* example compared to what real-world deep vision models deal with.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Quick-Fire True / False

1. In RGB, white is represented as (0,0,0). — **False** (white is (1,1,1); (0,0,0) is black in RGB, the opposite of CMY).
2. Weight sharing means the *same* filter weights are reused at every spatial location. — **True**.
3. A grayscale image and a timeseries cannot both be called "single-channel" because one is 2D and the other is 1D. — **False** (channel count is about measurements per location/timestep, not dimensionality).
4. AlexNet is considered a landmark moment because it proved learned features can beat hand-engineered ones at scale. — **True**.
5. Locality means a network should look at the *entire* image to compute every output value. — **False** (locality means the opposite: only a small nearby window).

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Diagram Labelling Practice

Below is a blank version of the "three keys" diagram from the theory file. Try to fill in the three missing labels (Data / Compute / Algorithms) yourself before checking the theory file's version.

```
   KEY 1                KEY 2                 KEY 3
+---------+          +----------+          +-----------+
|   ???   |    +     |   ???    |    +     |   ???     |   =  DL BOOM
+---------+          +----------+          +-----------+
```

<details>
<summary>Show labelled version</summary>

KEY 1 = DATA (huge labelled datasets), KEY 2 = COMPUTE (GPUs, parallel hardware), KEY 3 = ALGORITHMS (better training tricks).
</details>

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Mini Interview-Style Round

These mimic how an interviewer or a viva examiner might phrase Lecture 1 questions — slightly reworded from the slide language on purpose, so you learn to recognize the *idea* rather than memorizing exact slide wording.

**Q1.** "Your manager asks why you can't just flatten an image and feed it into a normal dense neural network for a production image classifier. What do you tell them?"

<details>
<summary>Show answer</summary>

You'd explain the parameter explosion problem concretely: flattening throws away 2D spatial structure, and the resulting fully connected layer needs parameters proportional to (image size × hidden units) — which for anything beyond tiny thumbnails quickly reaches hundreds of millions or billions of weights, demanding huge datasets and huge compute just to avoid overfitting, while still not exploiting the fact that nearby pixels are related. You'd recommend a convolutional architecture instead, which uses far fewer parameters via weight sharing and locality while typically performing *better* on images, not just cheaper.
</details>

**Q2.** "What's the difference between something being invariant to translation and something being invariant to scale or rotation?"

<details>
<summary>Show answer</summary>

Translation invariance (what Lecture 1 focuses on) means the model should give the same recognition result when the object simply *moves* to a different (x,y) position in the frame, with everything else unchanged. Scale invariance means the model should still recognize the object even if it appears bigger or smaller (closer to or farther from the camera). Rotation invariance means recognition should hold even if the object is tilted or rotated. Plain convolution with weight sharing gives you translation invariance more or less "for free," but scale and rotation invariance are *not* automatic — they usually require extra tricks like data augmentation, pooling, or specialized architectures, which is why the lecture explicitly separates "geometric invariance: translation, rotation, scale" as three related-but-different properties.
</details>

**Q3.** "Someone claims deep learning became popular purely because 'GPUs got faster.' How would you correct that claim using this lecture's material?"

<details>
<summary>Show answer</summary>

Correct them by naming all three pillars, not just one: yes, GPU compute mattered enormously, but it was only useful *together with* the explosion of large labelled datasets (like ImageNet) and improvements in training algorithms (better initialization, activation functions, optimizers, regularization). Faster GPUs alone, applied to 1990s-era algorithms and 1990s-era dataset sizes, would not have produced the same jump — all three had to arrive together.
</details>

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice)`

---

## Summary

This practice file turns Lecture 1's ideas into active-recall drills instead of passive re-reading. It starts with the two official homework prompts (locality/invariance for audio, and whether convolution applies to text), fully answered with reasoning, not just a one-word response. It then adds a fill-in-the-blank concept check covering the toddler analogy, RGB/CMY naming, invariance, locality, and the MNIST dataset; five explain-it-back prompts designed to be spoken out loud so gaps in understanding surface before an exam does; a quick-fire true/false round targeting the most common misconceptions (especially the RGB-vs-CMY white/black flip); and a blank-diagram labelling drill for the "three keys of the DL boom" story. Working through all five sections without peeking at the spoilers is a strong signal that Lecture 1's concepts are genuinely internalized rather than just recognized. Move to the exercises file next for a pure, exam-format question bank with difficulty tiers.

`[← Numerical](../numerical/dl_lecture01_introduction_numerical.md) · [🔝 Top](#dl-lecture-01--introduction-to-deep-learning-practice) · [Next: Exercises →](../exercises/dl_lecture01_exercises.md)`
