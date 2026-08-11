# DL Lecture 01 — Introduction to Deep Learning (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

> Folder: `Deep-Learning/Lecture-01-Introduction-to-Deep-Learning/theory/`
> Pairs with: [`numerical/dl_lecture01_introduction_numerical.md`](../numerical/dl_lecture01_introduction_numerical.md) · [`practice/dl_lecture01_introduction_practice.md`](../practice/dl_lecture01_introduction_practice.md) · [`exercises/dl_lecture01_exercises.md`](../exercises/dl_lecture01_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: L1 slide deck "Introduction to Deep Learning and CNN"

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [What Is Deep Learning?](#what-is-deep-learning)
3. [Why Did Deep Learning Suddenly Boom?](#why-did-deep-learning-suddenly-boom)
4. [Where Is Deep Learning Used?](#where-is-deep-learning-used)
5. [Three Levels of Vision Tasks](#three-levels-of-vision-tasks)
6. [Data Types Deep Learning Eats](#data-types-deep-learning-eats)
7. [The Problem With Plain Neural Networks on Images](#the-problem-with-plain-neural-networks-on-images)
8. [Invariance — Why Waldo Is Still Waldo](#invariance--why-waldo-is-still-waldo)
9. [Locality — Zoom In Before You Zoom Out](#locality--zoom-in-before-you-zoom-out)
10. [From Fully Connected Layers Toward Convolution](#from-fully-connected-layers-toward-convolution)
11. [Mnemonics](#mnemonics)
12. [Cheatsheet](#cheatsheet)
13. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
14. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine you hand a toddler a photo of a dog and ask, "what is this?" The toddler doesn't know grammar rules or biology textbooks — they just point and say "doggie!" How did they learn that? Nobody sat them down and gave them a rulebook like "four legs + fur + tail + barks = dog." Instead, they *saw hundreds of dogs* — big ones, small ones, sitting, running, black, white — and slowly their brain built its own internal rulebook, layer by layer. First they learned to notice edges and shapes. Then they learned fur textures and ear shapes. Then, eventually, "dog" became an idea they just *recognize* instantly.

Deep Learning tries to teach a computer the exact same way. Instead of a programmer writing thousands of "if-this-then-that" rules to detect a dog in a photo, we show the computer thousands of labelled dog photos, and we let a many-layered mathematical structure (a "neural network") figure out its own rules — first simple ones (edges), then medium ones (fur patterns, ears), then complex ones (this whole shape = dog). That is deep learning in one sentence: **letting layered models discover their own features from raw data, instead of a human hand-engineering those features.**

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## What Is Deep Learning?

Two equivalent ways to define it, both from the lecture:

1. **Interdisciplinary definition:** Deep Learning is a field that lets computers get *high-level semantic understanding* out of raw data. It borrows from Computer Science, Electrical Engineering, Mathematics, Psychology, Neuroscience, and surprisingly even Earthquake Engineering (yes — DL is used to detect quakes early too).
2. **Technical definition:** Deep Learning studies **neural network models with many stacked layers** that *automatically* learn a hierarchy of features straight from data — no manual feature engineering required.

Deep Learning sits *inside* Machine Learning, which sits *inside* Artificial Intelligence. Think of it as Russian nesting dolls:

```
+-------------------------------------------------+
|  Artificial Intelligence (AI)                    |
|   +-------------------------------------------+  |
|   |  Machine Learning (ML)                     |  |
|   |   +---------------------------------+      |  |
|   |   |  Deep Learning (DL)              |      |  |
|   |   |  "many-layer neural networks"    |      |  |
|   |   +---------------------------------+      |  |
|   +-------------------------------------------+  |
+-------------------------------------------------+
```

The word "deep" does not mean "profound" — it literally means **many layers stacked on top of each other**. A network with 2 layers is "shallow." A network with 50, 150, or 1000 layers is "deep."

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Why Did Deep Learning Suddenly Boom?

Neural networks are not a new idea — the math behind them is decades old. So why did Deep Learning explode into headlines only in the last ~15 years? Three ingredients finally lined up at the same time, like three keys that all had to turn together to open a locked door:

```
   KEY 1                KEY 2                 KEY 3
+---------+          +----------+          +-----------+
|  DATA   |    +     | COMPUTE  |    +     | ALGORITHMS|   =  DL BOOM
| (huge   |          | (GPUs,   |          | (better   |
| labelled|          | parallel |          | training  |
| datasets|          | hardware)|          | tricks)   |
+---------+          +----------+          +-----------+
```

- **Data:** The internet gave us millions of labelled images, text pages, and videos (e.g., ImageNet).
- **Compute:** GPUs (built for video games) turned out to be perfect for the matrix-multiplication-heavy math neural networks need.
- **Algorithms:** Smarter training tricks (better initialization, activation functions, regularization) made deep networks actually trainable without exploding or vanishing.

**First big practical proof:** LeCun et al. (1998) trained a CNN on the **MNIST** handwritten digit dataset and got genuinely useful error rates — this was one of the first times deep learning left the lab and worked on a real, useful task (reading handwritten postal codes and cheques).

**The tipping-point moment** most historians point to is **AlexNet** — a deep CNN that massively beat the "previous best" hand-engineered computer vision systems on image classification. That single result is often called the "big bang" of the modern deep learning era, because it proved that letting a network *learn* its own features beats a human *hand-crafting* features.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Where Is Deep Learning Used?

The lecture lists a wide spread of real applications — group them mentally into "seeing," "protecting," "moving," and "understanding" buckets:

| Bucket | Examples from the lecture |
|---|---|
| **Seeing** | Biometrics & face recognition, medical imaging, satellite image understanding |
| **Protecting / Predicting disaster** | Earthquake early-warning systems, weather nowcasting |
| **Moving** | Autonomous navigation — self-driving cars, Mars Rover, robotics |
| **Watching / Defending** | Target tracking & military applications, surveillance |
| **Talking / Interacting** | Human–computer interaction |

Notice the pattern: wherever there is a *huge pile of raw signal* (pixels, sound waves, sensor readings) and a human expert would normally have to squint at it and make a judgment call, deep learning can often be trained to make that same judgment call, faster and at scale.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Three Levels of Vision Tasks

The lecture buckets computer-vision problems into three levels of "how much understanding is needed":

```
   HIGH-LEVEL   -->  "What is happening in this whole scene?"
                     (scene understanding, activity recognition)

   MID-LEVEL    -->  "What objects/parts are here and where?"
                     (object detection, segmentation)

   LOW-LEVEL    -->  "What are the raw pixel-level patterns?"
                     (edges, corners, textures, gradients)
```

Think of it like reading a book: **low-level** = recognizing individual letters, **mid-level** = recognizing words and sentences, **high-level** = understanding what the whole paragraph *means*. A deep network stacks layers so that early layers do low-level work and later layers build up to high-level understanding — automatically.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Data Types Deep Learning Eats

Deep learning does not only eat images — it eats anything that can be turned into numbers. The lecture splits this into **single-channel** and **multi-channel** data.

### Single-channel data
- **Timeseries:** one feature value that changes over time (e.g., temperature every hour). Think of it as a single wiggly line on a graph.
- **Grayscale images:** a 2D grid (matrix) of intensity values. Convention: 0 = black, 255 = white, with everything in between a shade of grey. Each pixel is really "how many photons landed here," rounded to the nearest integer level.

### Multi-channel data
- **RGB (additive color, used by screens/cameras):** you *add* light. Black = (0,0,0) — no light at all. White = (1,1,1) — full Red + Green + Blue light blasted together. Primary colors: Red, Green, Blue.
- **CMY (subtractive color, used by printers/ink):** you *absorb* light instead of adding it. White = (0,0,0) — no ink, so all light bounces back = white paper. Black = (1,1,1) — full ink on all three channels, absorbing everything = black. Primary colors: Cyan, Magenta, Yellow.

**Baby-story way to remember the difference:** a screen is like a dark room where you're switching on colored torches (start dark, add light → additive). Paper is like a white wall where you're throwing paint at it (start white, add paint that blocks light → subtractive).

A neat trick question from the slides: *why are a grayscale image (a 2D matrix) and a univariate timeseries (a 1D sequence) both called "single-channel"* even though one is 2D and the other is 1D? Because "channel" refers to *how many independent measurements exist per location/timestep* — not the number of spatial dimensions. A grayscale image has 1 number per pixel location (however many locations there are); a timeseries has 1 number per timestep. Both = 1 channel.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## The Problem With Plain Neural Networks on Images

Before CNNs, people tried to use plain fully-connected networks (Multi-Layer Perceptrons, "MLPs") on images. MLPs are great for **tabular data** — rows and columns like a spreadsheet — because in tabular data we don't assume the columns have any particular spatial relationship to each other (column 3 is not "next to" column 4 in any physical sense).

But images are *not* like that — pixel (5,5) really is physically next to pixel (5,6). MLPs completely throw this structure away, and paying for that ignorance is very expensive:

- A modest **1-megapixel image** → about **10⁶ input numbers**.
- Connect that fully to a hidden layer of just **1000 neurons** → about **10⁹ parameters** (weights) to learn.

That's a *billion* numbers to learn from a single hidden layer, for a fairly small image. This creates three real problems, easy to remember with the acronym **M-D-S**:
- **M**emory: storing a billion weights is expensive.
- **D**ata-hunger: you need a *massive* labelled dataset to reliably estimate a billion parameters without overfitting.
- **S**tructure-blindness: the model completely ignores that nearby pixels are related — it treats pixel (1,1) and pixel (500,500) as equally "unrelated" to their neighbours, which is obviously wrong for real images.

Full formal derivation of these numbers is in the **numerical** companion file — this theory file focuses on *why* it's a problem, not the arithmetic.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Invariance — Why Waldo Is Still Waldo

Picture the "Where's Waldo?" puzzle books — Waldo in his red-and-white stripy shirt is hiding somewhere in a huge, busy picture. Here's the key insight: **what Waldo looks like does not depend on where Waldo is standing in the picture.** Waldo's face, hat, and stripes look identical whether he's hiding top-left or bottom-right.

This property is called **invariance** — specifically **translation invariance**: if you *shift* (translate) an object to a different position in the image, the network should still recognize it as the *same* object, using the *same* learned pattern-detector, without needing to re-learn it separately for every possible position.

CNNs are built to systematically exploit this. Instead of learning a completely separate "Waldo detector" for every possible (x,y) position in the image (wasteful and needs huge data), a CNN learns *one* small Waldo-shaped pattern detector (a filter) and **slides it across the whole image**, checking "does this pattern match here? ...and here? ...and here?" This is called **weight sharing** — the *same* small set of weights is reused at every spatial location, instead of having separate independent weights for each location like an MLP does.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Locality — Zoom In Before You Zoom Out

Locality is the second big idea, and it works hand-in-hand with invariance. The claim: **to recognize a small local pattern (like an edge, or Waldo's hat), you only need to look at a small neighbourhood of nearby pixels — not the entire image at once.**

The lecture describes this as a layer-by-layer zoom-out:

```
 EARLY LAYERS        MIDDLE LAYERS           DEEPER LAYERS
 (small patches)     (combine nearby         (integrate global
 edges, textures      patterns)               information)
      |                     |                        |
      v                     v                        v
  " / \ - | "  --->   " ear-shape,        --->   " full dog,
                          eye-shape "                whole scene "
```

Mathematically, this means: for an output pixel at location (i,j), we only let it depend on nearby input pixels at offsets (i+u, j+v) where u and v are *small* — not on pixels that are thousands of positions away. Formally, outside some small range (|a| > Δ or |b| > Δ), we simply force the weight to zero: **[V]ₐ,ᵦ = 0**. This is exactly what a **convolution kernel** does — it is a small local weight-window (commonly called V) that only "sees" a small neighbourhood.

**Baby-story recap:** to recognize a friend's face in a crowd, your eyes don't process the entire stadium pixel-by-pixel at once — you first notice small local cues (a nose shape here, an eyebrow there) and your brain assembles those local cues upward into "oh, that's my friend!" Locality + invariance together are *why* that works efficiently.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## From Fully Connected Layers Toward Convolution

Putting invariance + locality together is literally how you derive a convolutional layer starting from a fully-connected layer, in four conceptual steps:

1. **Start:** MLP layer — every output pixel [H]ᵢ,ⱼ depends on *every* input pixel [X]ₖ,ₗ, through a full 4D weight tensor Wᵢ,ⱼ,ₖ,ₗ. Extremely expensive, and weights are *different* for every (i,j) — no sharing at all.
2. **Apply translation invariance:** force the weight to depend only on the *relative offset* (a,b) = (k−i, l−j), not on the absolute position (i,j) itself. Now the *same* small filter is reused everywhere (weight sharing) — but it can still technically look anywhere in the image (offsets can range across the whole image).
3. **Apply locality:** restrict those offsets (a,b) to a small window, |a| ≤ Δ and |b| ≤ Δ, and zero out everything outside that window.
4. **Result:** you now have a small, shared, local weight window — this *is* a convolution kernel. A convolutional layer.

This progression (MLP → weight-sharing → locality-restricted weight-sharing → convolution) is one of the single most commonly asked "explain why CNNs exist" conceptual questions in interviews and exams — know it cold, in your own words, with the Waldo/locality story attached.

**Feature points**, a closely related idea, are used downstream for image alignment (mosaics/panoramas), 3D reconstruction, motion tracking, object recognition, database image retrieval, and robot navigation — anywhere you need to reliably say "this same distinctive point exists in two different images/frames."

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Mnemonics

- **"DL = Data + Compute + Algorithms"** → remember the boom with **D-C-A**, like "Deep Convolutional AI" turning three keys at once.
- **"MDS kills MLPs on images"** → **M**emory, **D**ata-hunger, **S**tructure-blindness.
- **"Waldo doesn't change, only his address changes"** → translation invariance in one line.
- **"Zoom in before you zoom out"** → locality: small patches first, whole scene last.
- **RGB = torches in a dark room (ADD light), CMY = paint on white paper (BLOCK light)**.
- **"Same filter, many places"** → weight sharing, the engine behind both invariance and the huge parameter savings of CNNs.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Cheatsheet

| Concept | One-line definition | Real-world hook |
|---|---|---|
| Deep Learning | Many-layer neural nets that auto-learn hierarchical features | Toddler learning "dog" from many photos |
| Invariance | Same pattern recognized regardless of position | Waldo looks the same everywhere in the picture |
| Locality | Nearby pixels matter most for local patterns | Recognizing a face from a nose + eyebrow first |
| Weight sharing | Same small filter reused at every location | One rubber stamp, pressed everywhere on the page |
| RGB | Additive color, used by screens | Torches in a dark room |
| CMY | Subtractive color, used by printers | Paint on white paper |
| Low-level task | Raw pixel patterns (edges, corners) | Recognizing individual letters |
| Mid-level task | Objects/parts and their location | Recognizing words |
| High-level task | Whole-scene understanding | Understanding a full sentence's meaning |
| Convolution kernel (V) | Small shared local weight window | The "rubber stamp" itself |

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** "Deep learning = neural networks." Not quite — *all* deep learning uses neural networks, but a *shallow* 1–2 layer neural network is **not** considered deep learning. "Deep" specifically refers to *depth* (many stacked layers).
- **Trap:** Confusing invariance with locality. Invariance is about **position not mattering** for recognition. Locality is about **only nearby pixels mattering** for computing a value. They are different ideas that both motivate convolution, but answering "define locality" with an invariance-flavoured answer is a very common mark-loss in exams.
- **Trap:** RGB vs CMY color math. Remember: in RGB, **white = all channels ON (1,1,1)**; in CMY, **white = all channels OFF (0,0,0)**. Students very commonly flip this under exam pressure — anchor it with the "torches vs paint" mnemonic above.
- **Exam hack:** If asked "why do CNNs need fewer parameters than MLPs for images," always answer with **both** ingredients — weight sharing (from invariance) **and** restricted receptive field (from locality) — examiners specifically look for both, not just one.
- **Exam hack:** If asked to justify DL's recent success, always cite all **three** pillars (data, compute, algorithms) — a one-pillar answer usually loses partial marks.

`[🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory)`

---

## Summary

This lecture is the "why do we even need CNNs" origin story. Deep Learning is defined as neural networks with many stacked layers that automatically discover hierarchical features from raw data, instead of humans hand-engineering those features — the same way a toddler learns to recognize a dog from repeated exposure rather than a rulebook. The field boomed only recently because three ingredients finally lined up together: huge labelled datasets, GPU compute power, and better training algorithms, with LeCun's 1998 MNIST results and later AlexNet's landmark win serving as the two headline proof points. Deep Learning today powers applications from biometric face recognition to earthquake early-warning to self-driving cars, and vision tasks are commonly bucketed into low-level (raw pixel patterns), mid-level (objects and parts), and high-level (whole-scene understanding) categories. Data fed into deep networks can be single-channel (grayscale images, timeseries) or multi-channel (RGB screens use additive color where white = all channels on; CMY printers use subtractive color where white = no ink at all). The core motivation for convolutional networks specifically comes from two failures of plain fully-connected MLPs on image data: the sheer parameter explosion (a 1-megapixel image into a 1000-neuron layer needs about a billion weights) and the complete disregard for spatial structure. Two properties fix this: **invariance** (a pattern like Waldo's face should be recognized the same way no matter where it appears — solved by weight sharing, i.e., reusing the same small filter everywhere) and **locality** (only nearby pixels should matter for computing a local feature — solved by restricting the filter to a small window, exactly what a convolution kernel V is). Applying both properties step by step to a fully-connected layer's weight tensor is literally the mathematical derivation of a convolutional layer, and is one of the most important "explain from first principles" answers to have ready for exams. Carry forward the M-D-S trap (Memory, Data-hunger, Structure-blindness) and the "Waldo doesn't change, only his address changes" line — they will keep showing up as the conceptual backbone for every CNN lecture that follows this one.

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-01--introduction-to-deep-learning-theory) · [Next: Numerical →](../numerical/dl_lecture01_introduction_numerical.md)`
