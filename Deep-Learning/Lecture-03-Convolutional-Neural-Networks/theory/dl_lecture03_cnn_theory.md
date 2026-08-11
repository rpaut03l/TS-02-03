# DL Lecture 03 — Convolutional Neural Networks (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

> Folder: `Deep-Learning/Lecture-03-Convolutional-Neural-Networks/theory/`
> Pairs with: [`numerical/dl_lecture03_cnn_numerical.md`](../numerical/dl_lecture03_cnn_numerical.md) · [`practice/dl_lecture03_cnn_practice.md`](../practice/dl_lecture03_cnn_practice.md) · [`exercises/dl_lecture03_exercises.md`](../exercises/dl_lecture03_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "CNN" deck, parts 1 & 2 (combined here)

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Recap: Hand-Designed Filters and Why They Failed](#recap-hand-designed-filters-and-why-they-failed)
3. [The Four Original Big Ideas of CNNs](#the-four-original-big-ideas-of-cnns)
4. [Local Connectivity, In Full Detail](#local-connectivity-in-full-detail)
5. [Parameter Sharing, Step by Step](#parameter-sharing-step-by-step)
6. [The Convolutional Layer, Formally](#the-convolutional-layer-formally)
7. [Why CNN Filters Act Like Edge Detectors](#why-cnn-filters-act-like-edge-detectors)
8. [Pooling — Keeping What Matters, Dropping the Rest](#pooling--keeping-what-matters-dropping-the-rest)
9. [Convolution + Pooling as a "Prior Belief"](#convolution--pooling-as-a-prior-belief)
10. [Normalization Layers — Keeping Numbers Well-Behaved](#normalization-layers--keeping-numbers-well-behaved)
11. [The Four Flavours of Normalization, Compared](#the-four-flavours-of-normalization-compared)
12. [Fully Connected Layer — The Classifier at the End](#fully-connected-layer--the-classifier-at-the-end)
13. [Case Study: AlexNet](#case-study-alexnet)
14. [Mnemonics](#mnemonics)
15. [Cheatsheet](#cheatsheet)
16. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
17. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine you're baking cookies with a set of cookie cutters. Each cutter has one fixed shape — a star, a heart, a circle — and you stamp it over and over across a big sheet of rolled dough. You don't design a brand-new cutter for every single spot on the sheet; you reuse the *same* cutter everywhere, and you only press it down over a *small* patch of dough at a time, not the whole sheet at once. That's a convolution, in one image: a small "cutter" (called a **filter** or **kernel**) is pressed against small local patches of an image, reused identically everywhere, stamping out a map of "how much does this patch look like my cutter's shape?" This lecture is about turning that cookie-cutter idea into a full, trainable neural network layer — and everything else (pooling, normalization, AlexNet) builds directly on top of it.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Recap: Hand-Designed Filters and Why They Failed

Before learned filters, computer vision relied on **hand-designed filters** like the **Sobel filter** — a small fixed grid of numbers that, when slid across an image, approximates the image's **partial derivatives**: the vertical Sobel operator measures how pixel values change horizontally (∂/∂x), and the horizontal Sobel operator measures how they change vertically (∂/∂y). Combining both gives you the gradient's magnitude (how strong an edge is) and phase (which direction it points).

**The problem:** the gradient from a Sobel filter is always perpendicular (90°) to lines of constant intensity — so raw Sobel output gives you a "blurry cloud" around every edge, not a single clean line. The fix, **non-maximum suppression**, checks whether each pixel is a local maximum *along the gradient direction* and keeps only that peak, thinning the edge down to one clean line. But even with this fix, Sobel-style hand-designed filters (Sobel, moving averages, etc.) share a deeper, fatal flaw: they only work for the specific task they were designed for, **cannot adapt to data**, and failed to scale because they were inflexible, limited in complexity, and required immense human expertise to design well. This is precisely the motivation for moving to **CNN filters** — filters that are *learned* from data instead of hand-designed, automatically adapting to whatever patterns the data actually contains.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## The Four Original Big Ideas of CNNs

The lecture organizes CNN design into two eras:

**Originally (the foundational four):**
1. **Invariance** — already covered in Lecture 1 (Waldo doesn't change, only his position does).
2. **Local Connectivity** — each neuron only looks at a small patch, not the whole image.
3. **Parameter Sharing** — the same filter weights are reused at every location.
4. **Pooling** — shrink the representation while keeping the most important signal.

**More recently added:**
5. **Normalization** — keep numbers well-scaled as they flow through many layers.
6. **Last-layer customization** — task-specific output layers.
7. **Loss functions** — task-specific ways of measuring "how wrong" a prediction is.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Local Connectivity, In Full Detail

**Core rule:** each hidden unit connects only to a small subregion (patch) of the input image — never the whole image at once. But there's a subtlety students often miss: **the patch spans the small spatial window (e.g. 3×3) but the FULL depth of channels.** A "3×3 filter" on a grayscale (1-channel) image really is 3×3×1 = 9 weights. The exact same "3×3 filter" on a color RGB (3-channel) image becomes **3×3×3 = 27 weights**, because at every location the filter simultaneously takes a 3×3 patch from the Red channel, a 3×3 patch from the Green channel, and a 3×3 patch from the Blue channel, and combines all of them together into one output number. (Recall this "same patch = same as a Sobel window" analogy from the slides — it's literally the same sliding-window idea, just now with learnable numbers instead of fixed Sobel numbers.)

**Why this matters:** a fully connected hidden layer looking at a 200×200×3 image would need an unmanageable number of parameters — computing linear activations for every hidden unit would be prohibitively expensive (see the exact numbers in the Cheatsheet/Numerical file). Local connectivity alone (without even sharing weights yet) already cuts this down enormously, simply because each hidden unit now only "sees" a tiny 3×3×3 patch instead of the entire 200×200×3 image.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Parameter Sharing, Step by Step

Parameter sharing is the second, separate restriction (recall from Lecture 1: sharing and locality are different ideas that combine). The rule: **the same filter (identical weights) is used at every location of the image**, to produce one full "feature map." The lecture's own step-by-step walkthrough:

```
STEP 1: You have one filter (say 3x3x3 for RGB) -- picture it colored BLUE
STEP 2: Slide that SAME blue filter over every position of the image
        -> this produces one full output "feature map"
STEP 3: Take a DIFFERENT filter (picture it YELLOW), slide IT over
        every position too -> a second feature map
STEP 4: Take yet another filter (RED), slide it too -> a third feature map
```

Each differently-colored filter detects a *different* pattern (one might detect vertical edges, another horizontal edges, another a color blob), and each produces its own full feature map. Stack all these feature maps together (depth-wise) and you get the convolutional layer's full output volume. **Local connectivity says "look at a small patch." Parameter sharing says "reuse the same patch-detector everywhere." Together, they are what makes a convolution so much cheaper than either a fully-connected layer or even a locally-connected-but-not-shared layer** (the exact parameter counts — 14.4 billion vs 3.2 million vs a mere few thousand — are worked out fully in the numerical file).

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## The Convolutional Layer, Formally

Putting local connectivity + parameter sharing together in one operation gives you **convolution**: sliding a small window (filter) over an image and checking, at every position, "how well does this patch match my learned pattern?"

Given an input volume of size **W1 × H1 × D1** (width, height, depth/channels), and a convolutional layer with:
- **F** = receptive field size (the filter's spatial size, e.g. F=3 for a 3×3 filter)
- **K** = number of feature maps (i.e., number of distinct filters)
- **S** = stride (how many pixels the filter jumps each step)
- **P** = zero-padding (how many pixels of zero-border are added around the input)

The output volume size is:
```
W2 = (W1 - F) / S + 1
H2 = (H1 - F) / S + 1
D2 = K
```
(With zero-padding P added, the more general formula is `W2 = (W1 - F + 2P)/S + 1`.) **Stride** controls how much the output resolution shrinks — a stride of 2 roughly halves the output resolution compared to stride 1, since the filter jumps 2 pixels at a time instead of 1. **Zero-padding** is the standard fix to stop your spatial dimensions shrinking every single layer (the lecture calls this the "Access Convolutional Challenge" — repeatedly applying, say, 5×5 convolutions on a 32×32 input without padding shrinks it 32→28→24→..., losing spatial information fast; padding lets you preserve the input size where needed).

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Why CNN Filters Act Like Edge Detectors

There's an elegant mathematical reason learned CNN filters so often end up looking like edge/gradient detectors (similar to Sobel), even though nobody told them to: the **differentiation property of convolution**. Convolving an image with the *derivative* of a filter is mathematically equivalent to first convolving with the plain filter, then differentiating the result — meaning a single learned convolution can effectively fold a "smooth + differentiate" pipeline into one operation, saving a computational step. This is a big part of why, when you visualize the first-layer filters a trained CNN actually learns, so many of them look strikingly like edge detectors and Gabor-like filters — the network rediscovers something close to classical hand-designed filters, purely from data, because that turns out to be a genuinely useful building block for vision tasks.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Pooling — Keeping What Matters, Dropping the Rest

**What pooling does:** convolution *finds* features; pooling *keeps the most important ones and shrinks the spatial size*, making the representation smaller and more manageable for later layers. It's useful whenever you care more about **whether** a feature is present than **exactly where** it is — the slide's example: for face detection, knowing the *approximate* location of the left and right eyes is enough; you don't need pixel-perfect eye coordinates. If a small "eye detector" filter fires strongly somewhere in a small local window, pooling just keeps that strong signal and discards the exact sub-pixel position, making the network robust to small shifts in exactly where the eye appears.

**Common pooling types:** Max pooling (keep the single largest value in each window — by far the most common), Average pooling (keep the average of each window), and L2 pooling (keep the L2 norm of each window). **Max pooling** is the default workhorse because it strongly preserves the "was this feature detected at all, anywhere in this window?" signal.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Convolution + Pooling as a "Prior Belief"

A more advanced, exam-favourite framing: think of **prior probability distributions** as encoding what we believe *before* seeing any data.

| Prior strength | What it means for a model | Entropy | Example |
|---|---|---|---|
| Weak prior | model is flexible, learns freely from data | high entropy | Gaussian with high variance |
| Strong prior | model is restricted, forced to follow rules | low entropy | Gaussian with low variance |
| Infinitely strong prior | model is hard-constrained | entropy ≈ 0 | zero probability on some parameters entirely |

**Key insight:** a convolutional network can be thought of as a fully-connected network with an **infinitely strong prior** baked into its weights — a prior that says "far-apart pixels shouldn't interact, and the same small pattern-detector should be reused everywhere." This is *exactly* the locality + weight-sharing story from Lecture 1, now reframed in the language of Bayesian priors. **Pooling adds a second, related infinitely-strong prior**: that each unit should be invariant to *small translations*. This explains precisely when CNNs (with pooling) start to struggle: if **exact position matters** for your task, pooling throws away information you actually needed, hurting accuracy; if **long-range dependencies matter** (something far away in the image affects the correct answer), plain convolution's local view struggles to capture that relationship — this exact limitation is a big part of the motivation for Attention mechanisms covered in a later lecture.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Normalization Layers — Keeping Numbers Well-Behaved

**The problem normalization solves:** consider a single layer `y = Wx`. Training gets hard when (a) inputs `x` are not centered around zero (forcing large bias terms to compensate) or (b) different input elements have wildly different scales (forcing the weights in `W` to vary enormously to compensate). **The idea:** force inputs to be "nicely scaled" at *every* layer, not just at the very start.

Three historical stepping stones toward modern normalization:
- **Local Contrast Normalization (LCN):** adjusts each pixel using its local neighbourhood so features stand out better; improves invariance and sparsity.
- **Local Response Normalization (LRN):** normalizes *across channels* at each spatial position, so strong features dominate and weak ones get suppressed — described as a kind of "lateral inhibition," borrowed from real neuroscience.
- **Batch Normalization (BatchNorm):** the modern default. Usually inserted after a Fully Connected or Convolutional layer, and *before* the non-linearity. Given a batch of activations at some layer, BatchNorm forces each feature dimension to be zero-mean, unit-variance:

```
x_hat = (x - mean) / sqrt(variance + epsilon)
y = gamma * x_hat + beta
```

Here `mean` and `variance` are computed **per feature, across all samples in the current mini-batch** (think of your data as a table: rows = samples (index i), columns = features (index j); for a fixed feature column j, you average over all rows i). The learnable parameters **γ (gamma)** and **β (beta)** let the network *undo* the forced zero-mean/unit-variance constraint if that's not actually optimal for a given feature — solving the "what if zero-mean unit-variance is too strict a constraint?" problem the raw formula would otherwise create.

**Why BatchNorm helps (memorize this list — very exam-favourite):** makes deep networks much easier to train, improves gradient flow, allows higher learning rates and faster convergence, makes networks more robust to weight initialization, acts as a form of regularization "for free" during training, and has zero overhead at test-time since it can be mathematically fused into the preceding conv/FC layer. **The one big gotcha:** BatchNorm behaves *differently* during training (uses the current mini-batch's statistics) versus testing (uses a running average of statistics accumulated during training, since at test time you may not even have a "batch" — e.g., predicting on a single image) — this training/testing behaviour mismatch is a very common source of real-world bugs.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## The Four Flavours of Normalization, Compared

| Type | Normalizes over | Same train/test behaviour? | Best suited for |
|---|---|---|---|
| **Batch Norm** | Across all samples, per channel | No (needs running averages at test time) | CNNs with large, stable batch sizes |
| **Layer Norm** | Across all features, per single sample (row-wise) | Yes | RNNs / Transformers (no dependency on other samples in the batch) |
| **Instance Norm** | Across spatial dimensions, per sample AND per channel | Yes | Style transfer / texture synthesis |
| **Group Norm** | Across a small group of channels, per sample | Yes | Small-batch-size training where BatchNorm's per-batch statistics become unreliable |

The unifying visual from the slides: picture your activations as a cube (batch × channel × spatial). Batch Norm slices "for each channel, use all samples." Layer Norm slices "for each sample, use all its features." Group Norm and Instance Norm slice somewhere in between, avoiding any mixing across the batch dimension entirely — which is exactly why they behave identically at train and test time, unlike Batch Norm.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Fully Connected Layer — The Classifier at the End

After several rounds of convolution + pooling + normalization extract and refine features, CNN architectures typically end with one or more **fully connected (FC) layers** — literally the Multi-Layer Perceptron from Lecture 2. Its role: act as the final **classifier**, taking the rich, high-level features the convolutional layers extracted (discriminative parts, higher semantic entities) and mapping them to final class scores, usually with an activation like softmax at the very last step.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Case Study: AlexNet

**AlexNet** won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, and is widely credited as the "big bang" moment referenced back in Lecture 1. Key facts: trained on 1.2 million images across 1000 classes, using SGD with regularization, with about **60 million parameters** total, using an optimized GPU implementation (cuda-convnet) — remarkable, since at the time, splitting the network's 96 first-layer filters across *two* GPUs (48 kernels learned on each) was necessary just to fit the model in memory.

**First layer worked out (this is a favourite exam numerical question, fully redone in the numerical file):** input images are 227×227×3. The first conv layer (CONV1) uses 96 filters of size 11×11, applied at stride 4, with no padding. Using the output-volume formula from earlier: `(227-11)/4 + 1 = 55`, giving an output volume of **55×55×96**. The parameter count per filter is `(kernel height × kernel width × input channels) + 1 bias = (11×11×3)+1 = 364`; across all 96 filters, `364 × 96 ≈ 35,000` (35K) parameters for this one layer alone.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Mnemonics

- **"Cookie cutter, same shape, small patch, many stamps"** → local connectivity + parameter sharing in one image.
- **"3×3 isn't really 3×3 — it's 3×3×(all channels)"** → the depth-spanning filter rule.
- **"Conv finds, pool keeps"** → the one-line convolution vs pooling distinction.
- **"BatchNorm: same team (batch), LayerNorm: same player (sample)"** → who gets averaged over.
- **"γ and β give BatchNorm an undo button"** → why the learnable scale/shift parameters exist.
- **W2=(W1-F)/S+1** → say it as "Width minus Filter, over Stride, plus one" — drill this until automatic.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Cheatsheet

| Concept | One-liner | Key formula/number |
|---|---|---|
| Local connectivity | Each unit sees a small patch, full depth | 3×3 on RGB = 3×3×3 = 27 weights |
| Parameter sharing | Same filter reused at every location | 1 filter → 1 full feature map |
| Output volume | Shrinks based on filter/stride/padding | `W2=(W1-F)/S+1`, `D2=K` |
| Params in conv layer | Independent of image size | `(F×F×Cin + 1) × K` |
| Pooling | Shrinks size, keeps strongest signal | Max / Average / L2 |
| CNN as prior | Infinitely strong prior over weights | locality + sharing = hard constraint |
| BatchNorm | Zero-mean unit-variance per feature, per batch | `x̂=(x−μ)/√(σ²+ε)`, `y=γx̂+β` |
| LayerNorm | Same, but per-sample (row-wise) | train=test behaviour |
| AlexNet CONV1 | 227×227×3 in, 96 filters 11×11 stride 4 | output 55×55×96, ≈35K params |
| FC comparison (200×200×3, 120,000 hidden units) | — | FC: 14.4B params · Local (no sharing): 3.2M · Conv (sharing): thousands |

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting to include input depth when sizing a filter — always write filters as `F×F×Cin`, never just `F×F`.
- **Trap:** confusing "local connectivity" (a spatial restriction) with "parameter sharing" (a weight-reuse restriction) — they combine to make convolution, but are graded as two distinct concepts.
- **Trap:** assuming BatchNorm behaves identically at train and test time — it does not; that mismatch is explicitly called out as a common bug source, and a favourite trick exam question.
- **Trap:** mixing up which normalization method is "safe" for small batch sizes — Group Norm and Layer Norm avoid the batch-dependency problem that hurts Batch Norm when batch size is very small.
- **Exam hack:** for any "compute the output volume" question, always write out all three components explicitly: `W2, H2, D2` — don't just answer with `W2` and forget the depth is simply `K` (number of filters), a common easy-mark loss.
- **Exam hack:** the AlexNet CONV1 numbers (55×55×96 output, ≈35K params) are extremely likely to reappear verbatim or with modified numbers (different stride/filter size) — practice the two-formula combo (output volume, then parameter count) until it's automatic.

`[🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory)`

---

## Summary

This lecture builds the full convolutional layer from first principles and then layers pooling, normalization, and a real case study on top. It starts by explaining why hand-designed filters like Sobel (which compute image gradients and need non-maximum suppression to produce clean edges) ultimately failed to scale — they're inflexible and can't adapt to data — motivating a move to *learned* CNN filters. Two originally-foundational ideas, local connectivity (each unit sees only a small patch, but spanning the full input depth, so a 3×3 filter on RGB is really 3×3×3=27 weights) and parameter sharing (the exact same filter is reused at every spatial location to produce one feature map, with different filters producing different feature maps), combine into the single convolution operation, whose output volume is computed via `W2=(W1-F)/S+1, H2=(H1-F)/S+1, D2=K`, with stride controlling resolution shrinkage and zero-padding controlling whether spatial size is preserved. A neat mathematical property (the differentiation property of convolution) explains why learned CNN filters so often end up resembling classical edge detectors. Pooling (max/average/L2) then shrinks the representation while preserving "was this feature present" signal, useful whenever exact position doesn't matter — and both convolution and pooling can be reframed, in Bayesian terms, as an "infinitely strong prior" baked into a fully-connected network's weights, which also explains exactly when CNNs struggle (when exact position matters, or when long-range dependencies matter beyond convolution's local view). Normalization layers (Local Contrast → Local Response → the modern default, Batch Normalization, which forces zero-mean/unit-variance per feature per batch via `x̂=(x−μ)/√(σ²+ε)`, `y=γx̂+β`) make deep networks dramatically easier to train, though BatchNorm's crucial train/test behaviour mismatch is a common bug source; Layer, Instance, and Group Normalization each slice the same idea differently to avoid depending on other samples in the batch. Finally, the AlexNet case study (2012 ILSVRC winner, ~60 million parameters, trained across two GPUs) walks a complete worked example — a 227×227×3 input through a 96-filter, 11×11, stride-4 first layer producing a 55×55×96 output volume with about 35,000 parameters — tying every formula in this lecture to one concrete, historically important real network.

`[← Lecture 02](../../Lecture-02-Neural-Networks/README.md) · [🔝 Top](#dl-lecture-03--convolutional-neural-networks-theory) · [Next: Numerical →](../numerical/dl_lecture03_cnn_numerical.md)`
