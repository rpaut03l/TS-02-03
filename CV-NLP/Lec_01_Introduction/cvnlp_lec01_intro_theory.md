# 📖 CV-NLP — Lec 01: Introduction — THEORY

### *What is CV · History of CV · Image Processing Foundations · Video Processing Foundations*

> **Nav:** [← Lec 01 README](README.md) | **THEORY** | [🧮 NUMERICAL](cvnlp_lec01_intro_numerical.md) | [🎯 PRACTICE](cvnlp_lec01_intro_practice.md) | [Track Home ➡️](../README.md)

---

## 🧠 MASTER MNEMONIC: "BIOM -> CV -> HIST -> IMG -> VID"

> Bio, Image-processing, Optics, Math feed CV -> CV's HISTory runs from Marr to CLIP -> then IMaGe math -> then VIDeo math.

Say it as a chant: *"Bio feeds CV, history builds CV, then image, then video."* Every section below is one link in this chain, in the exact order the lecture presents it. Read the chant out loud once before you start — by the time you reach Section 9, the chant alone should let you rebuild the entire lecture's skeleton from memory.

---

## 📚 Table of Contents

| # | Topic | Jump |
|---|-------|------|
| 1 | What is Computer Vision | [Section 1](#1-what-is-computer-vision) |
| 2 | History of CV — Biological & Ancient Roots | [Section 2](#2-history-of-cv--biological--ancient-roots) |
| 3 | Marr's Computational Vision Model | [Section 3](#3-marrs-computational-vision-model) |
| 4 | Classical Feature Detectors — SIFT & HOG | [Section 4](#4-classical-feature-detectors--sift--hog) |
| 5 | Structure from Motion & Segmentation | [Section 5](#5-structure-from-motion--segmentation) |
| 6 | The ImageNet Era — AlexNet to ViT | [Section 6](#6-the-imagenet-era--alexnet-to-vit) |
| 7 | CLIP & DALL-E — Vision Meets Language | [Section 7](#7-clip--dall-e--vision-meets-language) |
| 8 | Image Processing Foundations | [Section 8](#8-image-processing-foundations) |
| 9 | Video Processing Foundations | [Section 9](#9-video-processing-foundations) |
| 10 | Cheat Sheet & Exam Hacks | [Section 10](#10-cheat-sheet--exam-hacks) |
| 11 | Full-Lecture Summary | [Section 11](#11-full-lecture-summary) |

---

## 1. What is Computer Vision

### The Starting Picture

Think of a toddler standing at the window on a rainy morning. She doesn't just see "gray blur outside" — within seconds her brain sorts that blur into a car, a puddle, a dog running past, and a decision ("that dog looks fun, I want to go outside"). Nobody handed her a rulebook. Her eyes captured light, her brain organized that light into shapes, matched those shapes to things she'd seen before, and built a plan of action — all before she even said a word about it.

That entire chain — light in, understanding out — is exactly the job description of Computer Vision (CV). The only difference is that instead of a brain built from neurons, we're building one out of numbers, matrices, and code, and instead of twenty-some years of childhood experience, our "child" (the model) has to learn everything from a pile of example photographs.

### Why CV Refuses to Stay Inside Computer Science

The lecture's opening diagram places **Computer Vision** as a blue circle at the center of a wheel, with spokes reaching out to Biology, Psychology, Physics, Computer Science, Mathematics, and Engineering. This is not decoration — it's a warning to every new CV student: if you only study algorithms, you will hit a wall, because CV problems are fundamentally borrowed problems.

```
+-----------------------------------------------------------------+
|                 WHY COMPUTER VISION IS A TEAM SPORT              |
+-----------------------------------------------------------------+
|  BIOLOGY      -> how a real retina/optic nerve/visual cortex     |
|                   actually turns light into signals               |
|  PHYSICS      -> how light bends through a lens, how a camera     |
|                   sensor captures photons as numbers               |
|  PSYCHOLOGY   -> how humans perceive faces, depth, motion,        |
|                   illusions -- the "correct" answer a CV system    |
|                   should aim to match                              |
|  MATHEMATICS  -> linear algebra (images ARE matrices), calculus    |
|                   (gradients), probability (uncertainty)           |
|  COMPUTER SCI -> the actual algorithms, data structures, and       |
|                   systems that run all of the above at scale       |
|  ENGINEERING  -> robotics needs vision to physically act in the    |
|                   world (grab the cup, avoid the wall)             |
|  SPEECH / NLP -> once you've SEEN something, you often need to     |
|                   describe it or answer questions about it too      |
+-----------------------------------------------------------------+
```

Picture it like a school science-fair project where nobody is allowed to work alone: the biology kid explains how the eye actually behaves, the physics kid explains how a camera lens bends light onto a sensor, the math kid supplies the counting and pattern rules, and the computer-science kid is the one who finally types the code that stitches it all together. Remove any one teammate and the "seeing machine" simply won't work as well.

### The Nested-Circles Family Tree: AI, ML, DL, CNN

A second diagram in the deck shows four ovals stacked inside one another, largest to smallest:

```
+-----------------------------------------------------------------+
|  Artificial Intelligence (AI)                                    |
|  +---------------------------------------------------------+    |
|  |  Machine Learning (ML)                                    |   |
|  |  +---------------------------------------------------+    |   |
|  |  |  Deep Learning (DL)                                 |   |   |
|  |  |  +-----------------------------------------------+  |   |   |
|  |  |  |  Convolutional Neural Network (CNN)             | |   |   |
|  |  |  +-----------------------------------------------+  |   |   |
|  |  +---------------------------------------------------+    |   |
|  +---------------------------------------------------------+    |
+-----------------------------------------------------------------+
```

Think of it exactly like those Russian nesting dolls (matryoshka). The biggest, outermost doll is Artificial Intelligence — anything at all where a machine behaves in a way that looks "smart" (even old-school chess programs with hand-written rules count as AI, no learning required). Crack that doll open and you find Machine Learning — a smaller doll containing only the AI systems that *learn patterns from data* instead of following hand-written rules. Crack that one open and you find Deep Learning — an even smaller doll containing only the ML systems built from many-layered neural networks. Crack that one open, finally, and you find CNN — the smallest doll, containing only the Deep Learning systems built specifically to slide small filters across grids of pixels.

Every doll fits inside the one before it, but never the other way — you cannot crack open the tiny CNN doll and expect to find the giant AI doll hiding inside. A common exam trap is claiming "all Deep Learning is CNN" — false, because Transformers, plain RNNs, and Vision Transformers (covered later in this same lecture, Section 6) are Deep Learning models that are **not** CNNs.

### The Menu of CV Tasks

Right beside the nested-circles diagram, the deck lists the concrete jobs Computer Vision is hired to do:

```
+-----------------------------------------------------------------+
|  COMPUTER VISION -- THE JOB MENU                                  |
+-----------------------------------------------------------------+
|  - Object Detection          (find AND locate objects)            |
|  - Object Classification     (name the single main object)        |
|  - Scene Understanding       (describe the whole environment)      |
|  - Semantic Scene Segmentation (color every pixel by class)         |
|  - 3D Reconstruction         (rebuild the world in three dimensions)|
|  - Object Tracking           (follow an object across video frames) |
|  - Human Pose Estimation     (find joint positions -- elbow, knee)  |
|  - Activity Recognition      (identify what action is happening)     |
|  - Visual Question Answering  (answer natural-language questions      |
|    (VQA)                       about an image)                        |
|  - ... (and the list keeps growing every year)                        |
+-----------------------------------------------------------------+
```

### Why Vision Is Called AI's "Entrance Hall"

The lecture's exact phrase is: *"Vision is the most important source of information for the human brain and is the 'entrance hall' of AI."* Three side-by-side examples back this claim up:

1. A single RGB bedroom photo is turned, by a pipeline, into estimated depth maps, an object layout, a 3D room reconstruction, and even the camera's pitch/yaw/roll orientation — all extracted from one flat picture.
2. A photo of someone pouring water in a kitchen is parsed simultaneously for the action ("pouring"), the objects ("kettle," "cat," "TV"), and the scene ("kitchen").
3. A "joint parsing and cognitive reasoning" diagram takes that same kitchen photo and builds a graph connecting hidden objects, imagined actions, and relationships — essentially the machine asking itself *"why is the kettle near the stove, and what is likely to happen next?"*

Picture walking into a friend's house for the first time. Before you can have a conversation, plan where to sit, or figure out if dinner is ready, you first have to *see* the house — the sofa, the smell of something on the stove, the dog by the door. Every later, "smarter" decision is downstream of that first walk through the entrance hall. That is exactly why the lecture insists vision comes first, before language, before reasoning, before planning.

### Summary

Computer Vision is the field that teaches machines to turn raw pixels into understanding, and it deliberately borrows from biology, physics, psychology, math, and engineering rather than being a pure computer-science topic. It nests neatly inside the AI -> ML -> DL -> CNN family tree (each is a strict subset of the one before it, never the reverse), and its job list spans everything from simple classification to full scene understanding and visual question answering. Because so much of intelligent behavior depends on first correctly perceiving the world, the lecture calls vision the "entrance hall" of AI — the very first room any intelligent system has to pass through.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 2. History of CV — Biological & Ancient Roots

### Nature Solved This First

Long before anyone wrote a line of code, evolution had already solved "seeing" — more than once, independently, in wildly different animals. The lecture opens its history section with four close-up photos: an octopus's rubbery, suction-cupped eye; a fly's compound eye made of thousands of tiny lenses; a chameleon's independently swiveling eye; and a human baby's wide, curious eyes.

```
+-----------------------------------------------------------------+
|  FOUR DIFFERENT "CAMERAS" NATURE INVENTED                        |
+-----------------------------------------------------------------+
|  Octopus   -> soft-bodied camera eye, no blind spot               |
|  Fly       -> compound eye, thousands of tiny lenses, wide field   |
|  Chameleon -> two eyes that swivel INDEPENDENTLY of each other     |
|  Human baby -> forward-facing pair, built for depth + faces        |
+-----------------------------------------------------------------+
```

Picture a room full of different inventors, none of whom ever met each other, all separately arriving at "hey, a hole that lets light in and something behind it to catch that light is a really good idea." That is what biologists call convergent evolution, and for a CV student it's the single strongest hint in the whole lecture that vision isn't some rare accident — it is such a valuable skill that life reinvented it again and again, in bodies that share almost nothing else in common.

### Ancient Humans Build Their Own Eye: The Camera Obscura

```
+-----------------------------------------------------------------+
|  TIMELINE: HUMANS BUILD AN ARTIFICIAL "EYE"                      |
+-----------------------------------------------------------------+
|  16th century AD -> Leonardo da Vinci sketches pinhole optics,     |
|                      showing light rays crossing through a          |
|                      single small opening                            |
|  1545            -> Gemma Frisius publishes a camera obscura         |
|                      diagram, using a small hole in a dark room to    |
|                      safely observe a solar eclipse                   |
|  18th century     -> "Camera Obscura" is formalized in encyclopedias, |
|                      described as a box or tent with a tiny hole      |
|                      that projects an upside-down image of the        |
|                      outside world onto the opposite wall             |
+-----------------------------------------------------------------+
```

Here is the trick, explained the way you'd explain it to someone who has never seen a camera: take a completely dark shoebox, and poke one tiny pinhole in one side. Light from outside — say, a tree in the sunshine — travels in straight lines. Only the rays that happen to line up with that tiny hole make it inside the box, and because light travels in straight lines, the ray from the *top* of the tree ends up hitting the *bottom* of the opposite wall, and the ray from the *bottom* of the tree ends up hitting the *top*. The result: a real, projected, but upside-down picture of the tree appears on the inside wall of your shoebox — with absolutely no lenses, no electronics, nothing but a hole and darkness. That is the camera obscura, and it is the direct ancestor of every digital camera used to capture the training photos that modern CV models learn from.

### Neuroscience Proves the Brain Has Specialist Modules

The lecture leans on two classic brain-imaging studies to make a second, equally important point: even inside one single human brain, vision is not one giant do-everything blob — it is split into specialist sub-regions.

- **Kanwisher et al., J. Neuro. 1997** — an fMRI study titled effectively "Faces > Houses": when volunteers were shown photographs of faces, a very specific patch of brain tissue lit up far more strongly than when they were shown houses. This patch is known today as the Fusiform Face Area (FFA) — the brain's dedicated "face-recognition department."
- **Epstein & Kanwisher, Nature 1998** — a follow-up study repeating the same trick with faces, objects, houses, and scenes, shown both normally and scrambled into meaningless jigsaw pieces. A *different* brain region reliably lit up specifically for houses/scenes (later named the Parahippocampal Place Area) — proving this wasn't a fluke, but a second, separate specialist department.

Picture a giant office building where, instead of one employee doing every job, there's a "Faces Department" on one floor and a "Places Department" on another floor, and each department only lights up its lights when its specific kind of paperwork (face photo vs. house photo) arrives on its desk. That is a strikingly literal description of what these two studies found inside real human brains — and it hints strongly to CV engineers that maybe artificial vision systems should also be built from specialized sub-modules rather than one giant generic network trying to do everything at once.

### Mnemonic

**"Bio-Cam-Neuro"** -> Biological vision (animals) came first in evolutionary time, then ancient humans built artificial eyes with the Camera obscura, then modern Neuroscience proved the human brain runs specialized, departmentalized vision circuits (faces vs. places) rather than one single generic vision blob.

### Summary

This section establishes that Computer Vision did not start with computers at all. Vision was solved by biology millions of years before any silicon chip existed, in multiple independent animal designs (octopus, insect, reptile, mammal). Humans then reverse-engineered a crude external "eye" — the camera obscura — centuries before electricity. And modern neuroscience (Kanwisher's face-vs-house fMRI studies) revealed that even a single human brain doesn't process all visual categories the same way; it runs dedicated specialist circuits for different kinds of visual content. Together, these three threads (biology, optics history, neuroscience) form the intellectual foundation the rest of CV's engineering history builds on top of.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 3. Marr's Computational Vision Model

### The Core Idea, Told Simply

Hand someone a single, slightly blurry photograph of an orange basketball resting on a wooden court and ask them, "What is this, exactly, and how is it positioned in three-dimensional space?" Nobody — human or machine — can leap from "a grid of colored dots" straight to "it is a sphere resting flat on a floor" in one single mental jump. There has to be a sequence of smaller, well-defined steps in between. In the 1970s, a scientist named David Marr was the first person to write down, formally and rigorously, exactly what those in-between steps are and why each one is necessary. His book, simply titled *Vision*, opens with a line the lecture quotes directly:

> "3D Reconstruction — Not talent, but **computation**."

That sentence is the whole philosophy in one breath: seeing is not some mystical gift some people have and others don't — it is a sequence of well-defined computations that anyone (or anything, including a machine) can, in principle, replicate step by step.

### The Four Stages — The Single Most Important Diagram in This Lecture

```
+-------------+     +----------------+     +---------------+     +------------------+
| Input Image |  -> | Primal Sketch  |  -> | 2-and-half-D  |  -> | 3-D Model        |
|             |     |                |     | Sketch        |     | Representation   |
|-------------|     |----------------|     |---------------|     |------------------|
| Perceived   |     | Zero crossings,|     | Local surface |     | 3D models        |
| intensities |     | blobs, edges,  |     | orientation   |     | hierarchically   |
| (raw light  |     | bars, ends,    |     | and depth     |     | organized in     |
| values,     |     | virtual lines, |     | discontinuit- |     | terms of surface |
| nothing     |     | groups, curves,|     | ies -- BUT    |     | and volumetric   |
| understood  |     | boundaries     |     | only relative |     | primitives --    |
| yet)        |     |                |     | to the VIEWER |     | valid from ANY   |
|             |     |                |     |               |     | viewpoint        |
+-------------+     +----------------+     +---------------+     +------------------+
        "Stages of Visual Representation" -- David Marr, 1970s
```

| Stage | What it captures | Basketball example, told as a story |
|---|---|---|
| 1. Input Image | Raw pixel intensities, nothing "understood" yet | Just the photograph itself, exactly as the camera sensor recorded it |
| 2. Primal Sketch | 2D structure: edges, blobs, boundaries | A child with a crayon tracing only the outline and the seam-lines of the ball, ignoring the color inside |
| 3. 2-1/2-D Sketch | Depth and surface orientation, but ONLY from the current viewpoint | "The part of the ball facing me curves toward me" — with zero information about what the hidden far side looks like |
| 4. 3-D Model | Full volumetric shape, independent of any one viewpoint | "This is a sphere" — true and complete from every possible angle, including the side nobody photographed |

### Why "2-and-1/2-D" and Not Just "3D"? (The Favourite Exam Trap)

This is the single most frequently misunderstood idea in the whole lecture, so slow down here. The 2-1/2-D sketch is **viewer-centered**: it only encodes what is visible right now, from exactly where the camera or eye is standing. The full 3-D model is **object-centered**: it represents the object's complete shape as a standalone fact about the object itself, true from every angle, whether or not anyone is currently looking at it from that angle.

```
+-------------------------------------------------------------+
|   2-1/2-D SKETCH                    3-D MODEL                 |
|   ---------------                   ----------                |
|   Viewer-centered                   Object-centered            |
|   "What I can see right now,        "What the object actually  |
|    from where I'm standing"          IS, valid from any angle" |
|                                                                 |
|   Has SOME depth info, but only     Has FULL volumetric shape, |
|   relative to my current viewpoint  independent of viewpoint    |
+-------------------------------------------------------------+
```

Picture standing in front of a beach ball. You can absolutely tell that the near side curves toward you — your eyes give you that much "depth" information for free. But you have zero direct evidence about what the far side of the ball looks like right now; you're only guessing, based on prior life experience, that "balls are round all the way around." That guess-based, viewpoint-independent, complete mental model ("it's a full sphere, front and back") is the 3-D Model stage. The raw, only-what-I-can-currently-see information ("the near side curves toward me") is the 2-1/2-D Sketch stage. They sound similar, but one is a snapshot from your current spot, and the other is a complete fact about the object that doesn't care where you're standing.

### Why Does Vision Even Need Multiple Stages? (Answering the Instructor's Own Q1)

A single 2D photograph is **ambiguous and incomplete** on its own. The exact same physical object can produce wildly different-looking photographs depending on lighting, distance, camera angle, background clutter, or partial occlusion (something blocking part of the view). Because of this ambiguity, the visual system cannot safely leap directly from "raw pixels" to "it's a chair" — it has to build up confidence in stages:

```
+-------------------------------------------------------------+
|  WHAT EACH LAYER OF INFORMATION ACTUALLY TELLS YOU             |
+-------------------------------------------------------------+
|  Pixels  ->  tell us WHAT IS VISIBLE (raw brightness values)   |
|  Edges   ->  tell us WHERE structures BEGIN and END             |
|  Depth   ->  tells us HOW those structures are ARRANGED,         |
|              relative to the viewer                              |
|  3D model -> tells us WHAT the object actually IS                |
+-------------------------------------------------------------+
```

Think of assembling a jigsaw puzzle in a dim room. First you find the pieces with a straight edge (that's the Primal Sketch — just finding the boundaries). Then you notice which pieces seem to sit slightly "in front" of others based on how the shadows fall (that's the 2-1/2-D Sketch — depth relative to how the light is currently falling on the table, i.e., your current viewpoint). Only once you've assembled everything into the complete picture do you finally recognize "oh, this is a puzzle of a lighthouse" (that's the 3-D Model — the complete, viewpoint-independent understanding). Skipping straight from "loose puzzle pieces" to "it's a lighthouse" without the middle steps simply isn't possible, for a human doing a jigsaw or for a vision algorithm processing a photograph.

### Mnemonic

**"P - 2-1/2 - 3: Pixels, Pieces, Position, Presence"**
- **P**ixels (Input Image) -> raw intensities, nothing understood yet
- **P**ieces (Primal Sketch) -> edges, blobs, boundaries, the jigsaw-piece outlines
- **2-1/2** (viewer-relative depth) -> surface orientation, but only from my current seat in the room
- **3-D Presence** -> the full object, understood from every angle, not just mine

### Summary

David Marr's 1970s Computational Vision model breaks the "impossible-seeming" leap from raw pixels to full understanding into four clean, well-defined stages: the Input Image (raw intensities), the Primal Sketch (edges, blobs, boundaries), the 2-1/2-D Sketch (depth and surface orientation, but strictly relative to the current viewer's position), and finally the 3-D Model (a complete, viewpoint-independent volumetric understanding of the object). The most commonly tested distinction is that the 2-1/2-D stage is viewer-centered (only what's currently visible) while the 3-D stage is object-centered (true from every angle) — this single distinction shows up again and again in exam questions and vivas, so it deserves to be memorized word for word, not just understood loosely.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 4. Classical Feature Detectors — SIFT & HOG

### Setting the Scene

Long before neural networks could "learn" what an interesting corner or a human silhouette looked like from millions of examples, engineers had to sit down and hand-design mathematical recipes that reliably found useful structure in any photograph — no matter how it was rotated, zoomed in or out, or lit. Two of these hand-crafted recipes, SIFT and HOG, are important enough that the lecture spends multiple slides and multiple instructor Q&A rounds on each of them.

### SIFT — Scale-Invariant Feature Transform

**What it is, in the lecture's own words:** SIFT is a classic feature detection algorithm designed to detect and describe local features that are **robust to scale, rotation, and some viewpoint changes**.

```
+-------------------------------------------------------------+
|  SIFT -- WHAT MAKES A KEYPOINT COUNT AS "GOOD"?                |
+-------------------------------------------------------------+
|  [check] Distinctive          -- stands out from its neighbors  |
|  [check] Stable under scale   -- still found when photo is       |
|                                   zoomed in or out                |
|  [check] Stable under rotation -- still found when photo is       |
|                                   rotated/tilted                   |
|  [check] Reliable under lighting/viewpoint changes                |
|  -> Corners and blobs with unique, non-repeating texture           |
|     patterns work best -- easy to localize AND easy to re-match     |
+-------------------------------------------------------------+
```

**Why SIFT survives being rotated or zoomed:** SIFT constructs a **scale-space representation** — the image is deliberately blurred and shrunk repeatedly, building a whole stack ("pyramid") of increasingly zoomed-out, increasingly blurry versions of the same photo, and keypoints are detected as points that remain "extreme" (a local maximum or minimum) across this whole stack. On top of that, SIFT **assigns its own personal compass-orientation to every single keypoint**, based on the direction of the strongest local gradient. Because the final descriptor for that keypoint is always computed *relative to* its own assigned scale and orientation, rotating or resizing the entire input photo doesn't confuse the algorithm — the same physical keypoints simply get re-detected, just tagged with a different scale level or compass reading, and the actual descriptor numbers stay effectively the same.

Picture giving every single distinctive corner in a photograph its own tiny personal compass and its own personal measuring tape. No matter how you spin the whole photograph around, or how far you zoom the camera in or out, that corner's compass needle and measuring tape automatically re-adjust to point the same relative way and measure the same relative distance. So when SIFT looks at a rotated, zoomed photo later, it can still say with confidence, "yes, this is the exact same corner I marked before" — because it's not comparing raw pixel positions, it's comparing *self-corrected* readings that don't care about the overall spin or zoom of the photo.

### HOG — Histogram of Oriented Gradients

**What it computes, per pixel:**
- **Gradient magnitude** — how strong the local change in brightness is at that pixel.
- **Gradient orientation** — the compass direction in which that brightness change is pointing.

```
+-------------------------------------------------------------+
|  HOG PIPELINE (the lecture's own labeled images, a through g)  |
+-------------------------------------------------------------+
|  (a) Average human silhouette   -- a fuzzy heatmap-style        |
|      template built from many training photos                  |
|  (b) Sample HOG cells/blocks    -- coarse gradient structure     |
|      visible even at low, pixel-level resolution                 |
|  (c) Original input image       -- a real photo of a person       |
|      standing, at plain pixel level                               |
|  (d) Real-world test photo      -- same person, photographed       |
|      normally                                                      |
|  (e) HOG descriptor overlay     -- every cell now carries its       |
|      own little gradient-orientation histogram                      |
|  (f) Highlighted match regions  -- areas where the extracted          |
|      HOG features line up with the human-shaped template               |
|  (g) Positive detections        -- the final "yes, a person is          |
|      here" bounding boxes                                                 |
+-------------------------------------------------------------+
```

HOG deliberately looks at an image in small, fixed-size blocks, and for each block asks two simple questions: where are the edges, and which way do they point? A standing human silhouette has a remarkably *consistent* edge pattern even when the person's clothing, skin tone, or lighting changes completely from photo to photo:

```
+-------------------------------------------------------------+
|  WHY A HUMAN SILHOUETTE'S EDGE PATTERN STAYS CONSISTENT         |
+-------------------------------------------------------------+
|  Curved edges     -> around the head (skull/hair outline)        |
|  Diagonal edges   -> around the shoulders                         |
|  Vertical edges   -> running down along the torso and legs          |
|  Separated edges  -> corresponding to the gap between the two legs   |
+-------------------------------------------------------------+
```

Because of this, HOG deliberately throws away exact pixel colors and intensities, and keeps only the *directions* in which brightness changes — which is precisely what makes it robust to different clothing colors, different lighting conditions, and small appearance changes, so long as the overall silhouette shape and edge arrangement stays roughly the same. Imagine tracing only the outline of a person with a pencil, ignoring every detail of their clothes or skin tone entirely — that pencil outline alone, and the direction each pencil stroke points, is essentially what HOG is built to capture and compare.

### Where HOG Breaks Down (Answering the Instructor's Own Q5)

HOG struggles specifically with objects that are:

1. **Highly deformable** — an animal twisting into many different, non-rigid poses breaks the "consistent edge pattern" assumption HOG relies on.
2. **Lacking strong edges**, or having smooth, texture-less surfaces — there's simply nothing for a gradient calculation to grab onto.
3. **Occluded or sitting in cluttered backgrounds** — because HOG expects a consistent grid of edge orientations matching its learned template, missing edges (something blocking part of the person) or extra, distracting edges (a busy background) throw the match off.

Picture trying to recognize your friend's silhouette from across a crowded, noisy street, except your friend is also wearing a huge fluffy costume that changes shape every time they move. A rigid, edge-pattern-based recognizer like HOG would genuinely struggle here — not because it's a bad algorithm, but because the very thing it depends on (a stable, predictable outline) has been taken away.

### Mnemonic

**"SIFT Spins & Scales, HOG Grids & Points"**
- **SIFT** -> key**points** with built-in orientation/scale invariance — survives spinning and zooming.
- **HOG** -> a **grid** of gradient histograms — best suited to roughly rigid silhouettes, like a standing human.

### Summary

SIFT and HOG are two of the most influential hand-crafted feature detectors of the pre-deep-learning era. SIFT finds sparse, highly distinctive keypoints and gives each one its own scale and orientation, making it excellent at re-identifying the exact same physical point across very different photos of the same scene (great for matching and 3D reconstruction). HOG instead densely scans an entire region in a grid, summarizing local edge direction and strength everywhere, which makes it excellent at recognizing an overall silhouette shape like a standing human, but weak against deformable objects, smooth textureless surfaces, and occlusion/clutter. Knowing exactly which one to reach for — and exactly why the other one would fail — is one of the most commonly tested ideas in this lecture.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 5. Structure from Motion & Segmentation

### Two Eyes Are Better Than One

Close one eye and try to judge exactly how far away your coffee mug is on the desk — it's noticeably harder than doing the same thing with both eyes open. Open both eyes again, and it becomes almost effortless, because your brain automatically compares the two slightly different views from your two eyes and triangulates distance from that difference. Structure from Motion (SfM) borrows this exact trick, except instead of two fixed eyes, it uses many photographs of the same scene taken from different positions — or a single moving camera capturing many frames over time.

### Structure from Motion (SfM) — The Core Idea

**Core idea, stated plainly:** SfM finds the **same physical points or features across multiple photographs**, and uses **triangulation** to compute where each of those points must sit in 3D space. By knowing (or estimating) each photo's camera position and viewing angle, the algorithm draws imaginary straight lines from each camera through the matched point in each photo, and calculates exactly where those lines must intersect in three-dimensional space.

```
+-------------------------------------------------------------+
|  STRUCTURE FROM MOTION -- STEP BY STEP RECIPE                    |
+-------------------------------------------------------------+
|  1. Collect many overlapping photos of the same scene            |
|  2. Detect distinctive features in every photo (e.g. with SIFT)   |
|  3. Match the SAME feature across as many photos as possible       |
|  4. Estimate each photo's camera position AND orientation           |
|  5. Triangulate every matched feature -> get its 3D coordinate       |
|  6. Result: a sparse (or, with more work, dense) 3D point cloud       |
|     reconstructing the original scene                                 |
+-------------------------------------------------------------+
```

Imagine photographing a marble statue from ten completely different angles as you walk slowly around it. Suppose you notice the exact same tiny crack in the marble appears in five of your ten photos. Because you (roughly) know where you were standing and which way your camera was pointing for each of those five shots, simple geometry — draw a line from each camera position through where that crack appears in that camera's photo — tells you precisely where that crack sits in real 3D space, at the point where all five lines cross. Now repeat this same trick for thousands of tiny distinctive points across the entire statue, and you have reconstructed the whole statue's 3D shape, purely from ordinary flat 2D photographs and some careful geometry — no laser scanner required. This directly answers the instructor's own in-class Q6 about how algorithms recover 3D structure "just from photos."

### Semantic Segmentation vs. Instance Segmentation

This pairing is one of the single most commonly confused ideas in all of Computer Vision, and the lecture deliberately drills it with a direct side-by-side visual comparison: a photo of a dining table surrounded by several chairs.

```
+-------------------------------------------------------------+
|             SEMANTIC vs INSTANCE SEGMENTATION                   |
+-------------------------------------------------------------+
|  Input photo: a dining table with SEVERAL chairs around it        |
|                                                                    |
|  Semantic Segmentation output:                                    |
|    ALL chairs get painted the exact SAME color.                    |
|    -> The system knows WHAT is a chair, but NOT which               |
|       individual chair is which.                                     |
|                                                                       |
|  Instance Segmentation output:                                       |
|    Each INDIVIDUAL chair gets its OWN unique color.                   |
|    -> The system knows WHAT is a chair, AND separates each chair       |
|       as its own distinct, countable object.                           |
+-------------------------------------------------------------+
```

| Question | Semantic Segmentation | Instance Segmentation |
|---|---|---|
| "Which pixels belong to the class *chair*?" | Answered directly | Also answered |
| "Which pixels belong to THIS specific chair (chair #3)?" | Cannot answer — everything is merged into one blob | Answered directly |
| Can you count how many chairs are in the photo? | No | Yes |
| Which is harder to build? | Relatively easier | Harder — must also separate touching/overlapping instances of the SAME class |

Think about coloring a picture-book page where the rule is simply "color every chair blue." Once you've finished, you genuinely cannot tell from the coloring alone whether there were three chairs squeezed together or one giant blue blob — that is semantic segmentation's fundamental limitation. Now imagine instead you were told "give every single chair its own separate crayon color, even if two chairs are touching each other." Now, just by counting distinct colors, you can say with confidence exactly how many chairs are in the picture — that is instance segmentation, and it's a strictly harder, strictly more informative task.

### Mnemonic

**"SfM Triangulates, Semantic Groups, Instance Splits"** — Structure from Motion triangulates matched points into 3D; Semantic segmentation groups same-class pixels together under one color; Instance segmentation splits those groups apart again, one color per individual object.

### Summary

Structure from Motion recovers full 3D scene geometry from nothing but a collection of ordinary overlapping 2D photographs, by matching the same distinctive feature points across images and triangulating their 3D positions from known (or estimated) camera viewpoints — the exact same principle your own two eyes use for everyday depth perception. Semantic segmentation and instance segmentation are two closely related but importantly different pixel-labeling tasks: semantic segmentation paints all pixels of a given class the same color (answering "what"), while instance segmentation goes further and separates every individual object of that class into its own uniquely colored region (answering "which one," and enabling counting). Mixing these two up is one of the most common mistakes students make on exams and in interviews for this topic.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 6. The ImageNet Era — AlexNet to ViT

### The Collision That Changed Everything

For decades, CV researchers had no choice but to hand-design features like SIFT and HOG — clever, genuinely useful, but fundamentally limited by human creativity and intuition. Then, around 2012, three separate ingredients finally collided at the same moment: a massive, carefully labeled dataset (ImageNet), an old but data-hungry algorithm family (deep Convolutional Neural Networks), and, for the first time, fast enough hardware (GPUs) to actually train those networks at scale. This section tells the "before versus after" story of that collision.

### The Datasets That Made It Possible

```
+-------------------------------------------------------------+
|  DATASET TIMELINE -- EACH ONE BIGGER THAN THE LAST               |
+-------------------------------------------------------------+
|  Caltech-101 (Fei-Fei et al., 2004)                                |
|      -> 101 object categories, an early, modest benchmark           |
|  PASCAL VOC 2009 (Everingham et al., 2006-2012)                     |
|      -> the Visual Object Classes Challenge, with careful             |
|         annotated bounding boxes around objects                       |
|  ImageNet (Deng et al., CVPR 2009)                                    |
|      -> 22,000 categories, 15,000,000 images -- an enormous jump       |
|         in both scale and label detail                                  |
|  ILSVRC (Russakovsky et al., IJCV 2015)                                |
|      -> the annual Image Classification Challenge, a curated             |
|         SUBSET of ImageNet: 1,000 object classes, 1,431,167 images        |
+-------------------------------------------------------------+
```

### The Architecture Timeline (2010 -> 2015)

| Year | Model | Key idea | Reference |
|---|---|---|---|
| 2010 | NEC-UIUC | A dense descriptor grid (HOG, LBP) feeding into local coordinate coding, super-vector pooling (SPM), and a final linear SVM classifier — the OLD, hand-crafted pipeline | Lin, CVPR 2011 |
| 2012 | SuperVision (better known as AlexNet) | The first deep CNN to convincingly dominate ILSVRC — stacked convolution and pooling layers, trained on GPUs for the first time at this scale | Krizhevsky, NIPS 2012 |
| 2014 | GoogLeNet | "Inception modules": parallel branches inside one layer, running 1x1, 3x3, 5x5 convolutions and pooling side by side, then concatenating results | Szegedy, arXiv 2014 |
| 2014 | VGG | Extremely deep, but deliberately uniform — nothing but stacked small 3x3 convolutions, layer after layer | Simonyan, arXiv 2014 |
| 2015 | MSRA (the ResNet family) | Residual/"skip" connections that let signal bypass layers directly, enabling networks to go dramatically deeper without collapsing during training | He, ICCV 2015 |

A second diagram makes the "why 2012, and not 1998" question concrete by comparing raw scale: **LeCun et al., 1998** trained on roughly 10^6 transistors' worth of compute and roughly 10^7 pixels of training data (digit recognition on the NIST dataset). **Krizhevsky et al., 2012** trained on roughly 10^9 transistors' worth of compute (thanks to GPUs) and roughly 10^14 pixels of training data (the full ImageNet corpus). This isn't a story about a brand-new idea suddenly appearing — CNNs had existed since the 1990s. It's a story about **compute and data finally growing large enough to let an old, good idea actually work at scale**.

Picture a talented young chef who has had a genuinely great recipe sitting in a drawer since 1998, but simply didn't have a big enough kitchen, enough ingredients, or a large enough oven to cook it properly for a hundred guests. In 2012, someone finally handed that same chef a massive industrial kitchen (GPUs) and a huge delivery truck full of ingredients (ImageNet's millions of labeled photos) — and the exact same old recipe, cooked at the right scale for the first time, suddenly produced spectacular results.

### The ImageNet Error-Rate Story, Told as a Timeline

```
Top-5 error rate on the ImageNet Classification Task (illustrative shape,
matching the lecture's own chart):

  2010: ~28%    2011: ~26%    2012: ~16%  <- the AlexNet "shock" year
  2013: ~12%    2014: ~7%     2015: ~3.6%   2016: ~3.0%   2017: ~2.3%
                                    (Human-level error is roughly 5%,
                                     shown as a dashed reference line)
```

Before 2012, the very best models in the world were still wrong roughly one time out of every four attempts (26-28% error). In a single year, AlexNet very nearly cut that error rate in half. By around 2015, the leading models had already dropped their error rate below the typical human error rate on this specific, narrow benchmark task — an astonishing milestone that made international news at the time.

### The Even Longer Curve (2011 -> 2021): From Hand-Crafted Features All the Way to Vision Transformers

A separate, longer-running leaderboard chart in the lecture shows the state-of-the-art climbing continuously across an entire decade: **SIFT+FVs (2011, still hand-crafted!) -> AlexNet / ZFNet (2012-2013) -> SPPNet (2014) -> Inception V2/V3 (2015) -> ResNeXt-101 (2016-2017) -> PNASNet-5 (2018) -> FixResNeXt-101 (2019) -> ViT-H/14 (2021)**, with top-1 accuracy climbing from roughly 50% all the way past 88%. This single line graph is, in effect, the entire "hand-crafted features -> deep CNNs -> Vision Transformers" story of modern Computer Vision, compressed into one continuous curve.

### Beyond Classification: Detection & Segmentation Datasets

- **COCO** (Common Objects in Context, cocodataset.org) — today's standard benchmark for object detection, packed with cluttered, real-world scenes showing bounding boxes and labels for people, elephants, cars, parking meters, sinks, cups, clocks, and dozens of other everyday object categories, all mixed together in busy street/room/beach photos.
- **LVIS** (Large Vocabulary Instance Segmentation, lvisdataset.org) — extends instance segmentation to a much larger, "long-tail" vocabulary of object categories (going well beyond COCO's standard 80 common classes), with example scenes showing dozens of small, individually and uniquely colored instance masks for objects as varied as books, bottles, donuts, and swans, all in the same crowded photo.

### Mnemonic

**"C-P-I-I -> A-V-G-M -> ViT"**
**C**altech-101, **P**ASCAL, **I**mageNet, **I**LSVRC (the datasets, growing bigger each time) -> **A**lexNet, **V**GG, **G**oogLeNet, **M**SRA/ResNet (the architectures, each solving a different scaling problem) -> **ViT** (the Vision Transformer takeover, visible on the leaderboard by 2021).

### Summary

The 2012-2017 window is the single most important stretch of years in modern Computer Vision history. It was triggered not by a brand-new algorithmic idea, but by three old ideas — deep CNNs, huge labeled datasets, and GPU-scale compute — finally arriving together at the same moment, on the back of the ImageNet/ILSVRC benchmark. AlexNet's 2012 win nearly halved the previous year's error rate overnight, and the years that followed (VGG, GoogLeNet, ResNet, and eventually Vision Transformers by 2021) kept pushing accuracy up past typical human performance. Alongside pure classification, dedicated datasets like COCO and LVIS pushed the field into harder, richer tasks: locating multiple objects at once (detection) and separately labeling every individual object instance (instance segmentation), setting the stage for the CLIP and DALL-E models covered next.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 7. CLIP & DALL-E — Vision Meets Language

### The Missing Piece

Every single model discussed so far in this lecture — AlexNet, VGG, GoogLeNet, ResNet — only ever knew a small, fixed list of category names decided in advance ("this photo is one of exactly these 1,000 ImageNet classes, and nothing else"). CLIP and DALL-E are the models that finally broke that restriction, by connecting images with completely free-form natural language instead of a rigid, pre-decided class list. This is the literal bridge between the "CV" half and the "NLP" half of this whole course's name, and it's worth pausing on, because everything after this point in the course builds on this bridge.

### CLIP — Connecting Text and Images

**What it is, in OpenAI's own words (quoted in the lecture, Jan 5, 2021):** "We're introducing a neural network called CLIP which efficiently learns visual concepts from natural language supervision. CLIP can be applied to any visual classification benchmark by simply providing the names of the visual categories to be recognized" — a **zero-shot** capability directly comparable to what GPT-2 and GPT-3 had already shown for pure text, except now applied to images.

```
+-------------------------------------------------------------+
|  CLIP -- THE THREE-STEP PIPELINE                                 |
+-------------------------------------------------------------+
|  STEP 1: Contrastive pre-training                                 |
|    Text Encoder(caption)  -> T1, T2, ..., TN                        |
|    Image Encoder(photo)   -> I1, I2, ..., IN                         |
|    Train so MATCHING (image, caption) pairs get a HIGH similarity     |
|    score (Ii . Ti), and every MIS-matched pair gets a LOW score        |
|    (Ii . Tj, where i is not equal to j) -- this trains against a        |
|    big N-by-N grid of similarity scores, all at once                     |
|                                                                            |
|  STEP 2: Build a "zero-shot" classifier straight from label text            |
|    Take plain class names: "plane", "car", "dog", "bird", ...                |
|    Wrap each one in a simple template: "a photo of a {object}."               |
|    Run every wrapped label through the Text Encoder -> T1 ... TN               |
|                                                                                  |
|  STEP 3: Zero-shot prediction on a brand-new photo                              |
|    New test image -> Image Encoder -> I1                                         |
|    Compute I1.T1, I1.T2, ..., I1.TN  (compare against every class)                |
|    Pick whichever class's Tj gives the HIGHEST similarity score                    |
|    -> "a photo of a dog." is predicted, with NO extra training needed at all!       |
+-------------------------------------------------------------+
```

Picture teaching two very different students — one who only ever looks at pictures, and one who only ever reads sentences — to secretly agree on a shared "feeling" for every concept, such that when the picture-student sees a dog photo and the sentence-student reads "a photo of a dog," both of their private feelings end up landing in almost exactly the same spot on some enormous invisible map. Once both students have been trained to always agree like this, you can hand the picture-student a brand-new photo they've genuinely never seen before, whisper a list of candidate captions to the sentence-student, and simply ask, "which of these sentence-feelings is closest to what you're currently feeling about this new photo?" Whichever caption wins that comparison is CLIP's answer — and notice that nobody had to retrain either student specifically for this new photo or this new list of captions.

### CLIP: Image-Text Match Example, From the Lecture's Own Data

The lecture shows a cosine-similarity heatmap comparing 7 captions (for example, "a facial photo of a tabby cat," "a red motorcycle standing in a garage," "a page of text about segmentation") against 7 photos. In every single row and column, the **diagonal cell** — the cell where the caption and photo are the genuinely correct pairing — consistently shows the highest similarity score compared to every mismatched combination in that same row or column (roughly 0.30, 0.32, 0.29, 0.29, 0.32, 0.30, and 0.18 along the diagonal). This is direct, numerical confirmation that CLIP is correctly linking each caption to its true matching photo, ahead of every wrong pairing.

### DALL-E — Creating Brand-New Images from Text

Where CLIP works in the direction of **text -> find the best-matching image out of a pool that already exists**, DALL-E runs in the exact opposite direction: **text -> generate a completely brand-new image from nothing but the description.** The lecture shows several genuinely playful example prompts, each producing multiple distinct AI-generated candidate images: *"an illustration of a baby daikon radish in a tutu walking a dog,"* *"an armchair in the shape of an avocado,"* and *"a stained glass window with an image of a blue strawberry."* None of these described objects exist as real photographs anywhere in the world — DALL-E has to genuinely invent a plausible-looking image purely from the text description.

### Mnemonic

**"CLIP Connects, DALL-E Draws"**
- **CLIP** -> given some text plus a *pool of existing images*, finds the single best **match** (this is a discriminative, comparison-based task).
- **DALL-E** -> given only text, with no pool of images at all, **creates** a brand-new image from scratch (this is a generative, creation-based task).

### Summary

CLIP and DALL-E represent the moment Computer Vision stopped being locked into fixed, pre-decided category lists and started genuinely speaking the same language as text. CLIP is trained with contrastive learning to pull matching (image, caption) pairs together and push mismatched pairs apart in a shared representation space, which lets it perform zero-shot classification on entirely new categories just by describing them in plain English — no retraining required. DALL-E flips the direction entirely, generating brand-new, never-before-seen images purely from a text description. Together, these two models are the literal bridge connecting this course's "CV" half to its "NLP" half, and they set up everything that later lectures on vision-language models, captioning, and visual question answering will build on.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 8. Image Processing Foundations

### The Plumbing Behind Every CV Model

Before any fancy CNN, CLIP, or DALL-E can even begin to run, an image first has to exist as plain, storable, transmittable *data* — as numbers a computer can hold in memory, squeeze down for storage, and mathematically adjust for clarity. This section is deliberately the "plumbing" of the whole lecture: color models, compression, and enhancement math. It's less glamorous than Marr's model or CLIP, but every single pixel that ever reaches a neural network passed through exactly this plumbing first.

### Color Models

```
+-------------------------------------------------------------+
|  RGB -- Red, Green, Blue                                        |
|         Used in: digital screens (monitors, cameras, TVs)         |
|         ADDS light together -- combining all three at full           |
|         strength produces WHITE                                       |
|                                                                          |
|  HSL -- Hue, Saturation, Lightness                                       |
|         Used in: image editing, color filtering                           |
|           Hue        = the TYPE of color (red, green, etc.)                 |
|           Saturation = how PURE / intense that color is                      |
|           Lightness  = how LIGHT or DARK the color appears overall             |
|                                                                                   |
|  CMYK -- Cyan, Magenta, Yellow, Black                                             |
|          Used in: printing                                                          |
|          ADDS ink together -- combining more ink ABSORBS more light,                  |
|          moving toward BLACK                                                            |
+-------------------------------------------------------------+
```

**The single most important exam point here:** RGB and CMYK are literal *opposites* in mechanism. RGB is **additive** — a screen starts pitch black, and *adding* red, green, and blue light together gets you all the way up to white; more light equals brighter. CMYK is **subtractive** — a sheet of paper starts white, and *adding* cyan, magenta, and yellow ink absorbs more and more of the light bouncing off the page, dragging the result toward black; more ink equals darker. Screens shine light directly at your eyes (so they add light to get brighter), while printed pages only reflect whatever light isn't absorbed by the ink (so they subtract light to get darker) — the physical setup is genuinely opposite, not just a different naming convention.

### JPEG Compression — The Full Five-Step Pipeline

```
Raw Image Data
      |
      v
+--------------+   +---------------+   +-------------+   +----------------+   +-----------+
| 1. Color     |-> | 2. Downsample |-> | 3. DCT       |-> | 4. Quantization |-> | 5.Encoding|
|    Transform |   |               |   | (Discrete    |   |                 |   |           |
|--------------|   |---------------|   |  Cosine      |   |-----------------|   |-----------|
| RGB -> YCbCr |   | Reduce the    |   |  Transform)  |   | Reduce the      |   | Huffman   |
| (separates   |   | RESOLUTION of |   |--------------|   | PRECISION of    |   | coding -> |
| brightness   |   | just the      |   | Transforms   |   | frequencies     |   | final     |
| from color)  |   | COLOR         |   | the image    |   | (high, barely-  |   | compressed|
|              |   | channels --   |   | into         |   | visible         |   | bitstream |
|              |   | we're more    |   | frequency    |   | frequencies are |   |           |
|              |   | sensitive to  |   | space, using |   | often thrown    |   |           |
|              |   | brightness    |   | 8x8-pixel    |   | away entirely)  |   |           |
|              |   | than color)   |   | block DCT    |   |                 |   |           |
|              |   |               |   | basis        |   |                 |   |           |
|              |   |               |   | functions    |   |                 |   |           |
+--------------+   +---------------+   +-------------+   +----------------+   +-----------+
                                                                                     |
                                                                                     v
                                                                    JPEG-Compressed Image Data
```

**Decompression simply reverses every single step**, in the opposite order: Huffman decode, then de-quantize, then inverse-DCT, then upsample the color channels back to full resolution, then convert the color space back from YCbCr to RGB, finally producing the reconstructed raw image.

**Why convert to YCbCr before downsampling (Step 1) instead of just working directly in RGB?** Human eyes are simply much more sensitive to differences in **brightness** (technically called luminance, the Y channel) than to differences in **color** (technically called chrominance, the Cb and Cr channels). By deliberately splitting brightness apart from color right at the start, JPEG can then afford to throw away far more color detail in the very next step (downsampling) while keeping brightness sharp and crisp — and because our eyes barely notice the missing color detail anyway, this single design decision alone saves an enormous amount of file size with almost no perceived drop in quality.

Picture a printed newspaper photo under a magnifying glass: you'd notice immediately if the brightness pattern (the actual shapes, light and dark) got blurry, but you'd barely notice if the exact shade of a small colored patch shifted slightly. JPEG exploits exactly this asymmetry in human perception, on purpose, at industrial scale.

### Video Compression — I-Frames, P-Frames, and B-Frames

```
+-------------------------------------------------------------+
|  FRAME TYPES IN VIDEO COMPRESSION                                |
+-------------------------------------------------------------+
|  I-Frame (Intra, "keyframe")                                       |
|    -> Completely self-contained -- doesn't rely on ANY other        |
|       frame to be decoded                                            |
|    -> Looks like a plain static photo (essentially a full,             |
|       JPEG-like compressed frame, all on its own)                       |
|                                                                            |
|  P-Frame (Predicted)                                                        |
|    -> Rendered using the PREVIOUS frame, storing only the                    |
|       DIFFERENCES from it                                                     |
|    -> Exploits the fact that "almost always, the current picture                |
|       can be rendered using the previous frame plus small tweaks"                |
|                                                                                     |
|  B-Frame (Bi-predictive)                                                              |
|    -> Refers to BOTH the past frame AND a future frame at the same                     |
|       time, for even better compression                                                 |
|    -> Trade-off: needs BOTH past and future frames already decoded                        |
|       before it itself can be decoded -- this adds DELAY and needs                         |
|       MORE memory -- not ideal for real-time apps like video calls                          |
+-------------------------------------------------------------+

  Example GOP (Group of Pictures) ordering:  B I B P B P B P B I
```

Codecs mentioned in the lecture that all use this same I/P/B frame skeleton include H.261, H.262, H.263, H.264, H.265, AV1, and WMV — every one of them shares the same basic encoder/decoder shape: **FDCT -> Quantizer -> Entropy Encoder -> Compressed Bitstream** on the way out, mirrored by **IDCT <- De-quantizer <- Entropy Decoder** on the way back in, both sides sharing the same Quantization Tables and Huffman Tables so the decoder knows exactly how to reverse what the encoder did.

Think of an I-frame as a complete photograph pinned to a corkboard. A P-frame is like a small sticky note attached next to it that just says "everything's the same as the last photo, except THIS one corner moved slightly." A B-frame is a sticky note that has to look both backward AND forward in time before it can be written at all — it's more efficient overall (it can describe changes very compactly by referencing two anchor points instead of one), but whoever is reading the sticky notes in order has to wait until BOTH neighboring photographs are already available before that particular note even makes sense — which is exactly why B-frames add delay and aren't ideal for something that needs to feel instantaneous, like a live video call.

### Brightness, Contrast, and Histograms

**Brightness** of a grayscale image is defined as the **average intensity** across every single pixel:

```
                  1    h   w
        B(I)  =  ---  E   E  I(u, v)
                  wh  v=1 u=1
```

where `w` is the image's width in pixels, `h` is its height, and `I(u,v)` is the raw intensity value at position (u,v). Step 1 is to sum up every single pixel's intensity value across the whole image. Step 2 is to divide that giant sum by the total number of pixels (`w` times `h`), which turns the raw sum into a genuine average.

**Contrast** measures how much brightness *varies* relative to the overall average brightness:

```
                    Change in Luminance
   Contrast  =    ------------------------
                     Average Luminance
```

The bigger the variation relative to the average, the higher the contrast. (The lecture is explicit that many different precise formulas for "contrast" exist in the broader literature — this ratio form is the intuitive version taught here, and it's worth stating clearly which definition you're using if a question ever asks you to define contrast from scratch.)

**Histogram** — a simple chart, `h(i)`, that counts exactly how many pixels in the image have each possible intensity value `i`, ranging from 0 (pure black) up to 255 (pure white) for a standard 8-bit image. Critically, a histogram carries **only statistical information** — it tells you the *count* of pixels at each brightness level, but gives you **zero information about WHERE, spatially, those pixels are located** in the actual image.

```
h(i)
 10 |                    #
  9 |                    #           #
  8 |                    #     #  #  #           #
  7 |                    #     #  #  #        #  #
  5 |        #           #     #  #  #     #  #  #
  3 |        #           #     #  #  #     #  #  #
  1 |        #           #     #  #  #     #  #  #  #
  0 +--------+-----------+-----+--+--+-----+--+--+--+--> i (intensity)
             1           2     3  4  5     6  7  8  9  ...
```

Reading this chart, "10 pixels have the intensity value i = 2" is found simply by reading straight up from position i=2 on the horizontal axis to the top of that bar.

**What histogram shape tells you about contrast:**
- A narrow, tall central peak means **low contrast** — nearly all the pixels are crammed into a small, similar-brightness range.
- A wide, spread-out histogram means **good, healthy contrast**.
- Peaks stacked right at the two extreme ends (near 0 and near 255) mean **high contrast, but with likely detail loss** — this is called clipping, where lots of pixels are pure black or pure white with almost nothing in the smooth middle range.

**Dynamic range** is the number of genuinely distinct brightness levels an image actually uses.
- High dynamic range means many different shades and smooth gradations between them.
- Low dynamic range means only a handful of distinct shades, giving the image a flat, posterized look.

### Image Enhancement Techniques

**1. Histogram Equalization** — deliberately spreads out an image's pixel intensities so that the image uses the *entire* available dynamic range (0-255, for a standard 8-bit image), pushing the histogram shape as close as possible to a flat, uniform distribution. The whole goal is automatic, algorithmic contrast enhancement, with no manual tweaking required.

**2. Log Transform** — for compressing an image with a genuinely huge dynamic range down into something that can actually be displayed or printed normally:

```
s = c . log(1 + |r|)
```

- `s` is the output pixel value.
- `r` is the input pixel value.
- `c` is a scaling constant, chosen so the final result fits neatly inside the intended output range.
- The log function **compresses large values proportionally much more than it compresses small values** — meaning a handful of extremely bright outlier pixels get squashed down much harder than the ordinary mid-range pixels do.
- **Best suited for:** images with an unusually high dynamic range — astronomical images (a few blindingly bright stars next to a mostly-dark sky), medical scans, or satellite imagery — where a small number of extreme pixel values would otherwise completely wash out all the useful mid-range detail if displayed without any correction.

**3. Gray-Level Slicing** — a technique for deliberately highlighting one specific band of intensity values, `[A, B]`, while suppressing everything else:
- Pick a range `[A, B]` of pixel intensities you personally care about.
- Pixels that fall **inside** this chosen range get enhanced or brightened.
- All other pixels are pushed down to a flat **constant** value (often pure black, i.e., zero).
- This is genuinely useful for isolating one specific tonal feature — for example, "show me only the brightest road markings in this photo, and black out absolutely everything else."

**4. Spatial Filtering** — applying a small filter, or "kernel," to a small local neighborhood of the image at a time, in order to enhance or suppress a specific local feature such as noise or edges.

```
Low-Pass Filter (smoothing / general noise removal, e.g. spike or white noise):
        +--------------+
   1/9 *|  1   1   1   |     -> averages an entire 3x3 neighborhood together,
        |  1   1   1   |        which removes high-frequency noise and makes
        |  1   1   1   |        the whole image look smoother
        +--------------+

Median Filter (specifically for salt-and-pepper / impulse noise):
   Replace each pixel with the MEDIAN of its neighboring pixel values
   (this is NOT a simple fixed-weight multiply -- it's a nonlinear,
    "sort everything and pick the middle one" operation).
   -> Unlike a plain average, the median is genuinely NOT dragged
      toward extreme outlier values, so it preserves sharp edges
      much better than averaging does.

Sharpening Filter (highlight edges and fine local detail):
        +-----------------+
   1/9 *| -1   -1   -1    |     -> deliberately boosts the CENTER pixel's
        | -1    8   -1    |        value relative to its neighbors,
        | -1   -1   -1    |        which emphasizes local contrast and edges
        +-----------------+
```

**Spatial Domain versus Frequency Domain enhancement:**
- **Spatial Domain** enhancement means working directly on the raw pixels themselves, using small kernels like the ones shown above.
- **Frequency Domain** enhancement means first transforming the entire image (for example, via a DCT or FFT) into its frequency components, deliberately modifying those frequency components directly, and only then transforming the whole thing back into a normal, viewable image.

### Mnemonic

**"C-J-F-B-C-H-D-L-G-S"**
**C**olor models -> **J**PEG pipeline -> **F**rame types (I/P/B) -> **B**rightness -> **C**ontrast -> **H**istogram -> **D**ynamic range -> **L**og transform -> **G**ray-level slicing -> **S**patial filtering. Say it as a single chant: *"Color, JPEG, Frames, Bright-Contrast-Histo-Dynamic, Log-Gray-Spatial."*

### Summary

Image Processing Foundations cover everything that has to happen to a picture before it can be meaningfully analyzed or fed into a model. Color models (RGB, HSL, CMYK) describe fundamentally different — and in RGB versus CMYK's case, literally opposite — ways of representing color, additive for glowing screens and subtractive for reflective, ink-based print. JPEG's five-step pipeline (color transform, downsampling, DCT, quantization, entropy encoding) squeezes an image down by deliberately exploiting the fact that human eyes care far more about brightness than about fine color detail. Video compression extends this same logic across time using I-frames (self-contained), P-frames (predicted from the past), and B-frames (predicted from both past and future, at the cost of added delay). Finally, a handful of simple but powerful formulas — brightness, contrast, histograms, the log transform, gray-level slicing, and spatial-filter kernels — give us precise mathematical tools to measure and actively improve an image's visual quality before any higher-level CV model ever sees it.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 9. Video Processing Foundations

### From Frozen Moments to Motion

A single image is one frozen instant in time. A video is thousands of these frozen instants stitched together in sequence — and the moment you have *multiple* consecutive frames available, an entirely new kind of question becomes possible for the very first time: not just "what is this," but "what is actually MOVING, and in which direction, and how fast?" This final section of the lecture covers exactly that shift.

### Optical Flow

**Definition:** Optical flow is the **pattern of apparent motion** of objects, surfaces, and edges within a visual scene, produced by the relative motion between an observer (the camera) and the scene itself. In practice, it tells us how things in a video are moving by directly comparing one frame against the very next one.

```
   Frame at T=t              Frame at T=t+1             Optical Flow Output
  +--------------+          +--------------+           +--------------------+
  |  [car] at     |    ->    |  [car] slightly|   ->     |  colored blob:       |
  |  position A   |          |  further        |         |  the COLOR encodes    |
  |               |          |  forward         |         |  motion DIRECTION,      |
  |               |          |                  |         |  the BRIGHTNESS encodes  |
  |               |          |                  |         |  motion SPEED             |
  +--------------+          +--------------+           +--------------------+
```

Every single pixel in an optical flow field carries its own **flow vector**, made up of:
- A **direction** — which way that particular pixel appears to be moving.
- A **magnitude** — how much it moved, i.e., its apparent speed.

If `(u, v)` represents the horizontal and vertical motion of a given pixel between two frames, then the **flow magnitude** at that pixel is:

```
Magnitude = sqrt(u^2 + v^2)
```

This is literally the Pythagorean theorem, applied directly to a 2D motion vector — treat `u` and `v` as the two shorter legs of a right-angled triangle, and the magnitude is simply the hypotenuse: the actual straight-line length of that pixel's displacement, regardless of exactly which direction it happened to move in.

### HOG and HOF in Video

- **HOG**, when applied to video, captures the shape and structure of objects (especially humans) frame by frame, by analyzing local edges and their orientations — this is the exact same HOG technique from Section 4, simply run separately on each individual frame.
- **HOF (Histogram of Optical Flow)** is HOG's motion-focused cousin:
  - HOF is especially well suited to recognizing **actions** — walking, running, waving — purely from video.
  - It only looks at **motion direction**, and deliberately ignores static appearance entirely.
  - It performs best when objects move in reasonably consistent, repeated patterns.
  - HOF is constructed the same way HOG's per-cell orientation histogram is built, except instead of counting static edge directions, it counts **how many flow arrows point in each direction** — effectively, a histogram of *movement* direction rather than a histogram of *appearance* direction.

**A favourite exam trap, directly answering the instructor's own in-class Q9:** if two people walk in genuinely *opposite* directions, their HOF features will come out **different** — because the specific direction bins that get filled in the histogram completely change, even though both people are performing the exact same underlying *action* (walking). This is a real, documented limitation of HOF: it is sensitive to absolute walking direction, not merely to the general "type" of motion being performed.

### Motion Boundary Histogram (MBH)

MBH computes the gradient **of the optical flow field itself**, rather than the gradient of raw pixel brightness — split into separate MBH-x and MBH-y components — and this specifically highlights *motion boundaries*: places where the direction or speed of motion changes sharply, such as at the visible edge of a moving arm or leg against an otherwise mostly-static background.

**Worked intuition, straight from the lecture:** imagine training a system on 100 different videos, each one showing a different person walking. Every single video produces its own MBH-x and MBH-y descriptor. Averaging all 100 of these descriptors together produces a typical "walking" MBH pattern — something like *"usually, there's strong motion concentrated at the legs, and a smaller amount of motion at the arms."* When a brand-new test video arrives later, the system simply computes that video's own MBH descriptor and checks: *"does this new pattern resemble the average walking pattern I've already learned?"* — this simple template-comparison idea is the entire essence of classical, pre-deep-learning action recognition.

### The Classical Action-Recognition Pipeline: Bag-of-Space-Time-Features Plus SVM

```
+-------------------------------------------------------------+
|  TRADITIONAL ACTION CLASSIFICATION PIPELINE                       |
+-------------------------------------------------------------+
|  1. Identify space-time interest points                             |
|     (marked as red dots on the video, at exactly the spots where     |
|      motion or appearance changes -- e.g. joints/limbs in motion)     |
|                          |                                              |
|                          v                                              |
|  2. Extract a "collection of space-time patches" surrounding             |
|     each interest point                                                    |
|                          |                                                  |
|                          v                                                  |
|  3. Compute descriptors for every patch: HOG (appearance), HOF               |
|     (motion), and/or MBH (motion boundary) -- this converts each               |
|     raw patch into a compact numeric feature vector                             |
|                          |                                                       |
|                          v                                                       |
|  4. Cluster similar feature vectors together into a "dictionary"                   |
|     of visual words (exactly like clustering similar word meanings                  |
|     together in NLP -- hence the deliberate name "Bag of [visual] Words")            |
|                          |                                                            |
|                          v                                                            |
|  5. Build a single Histogram of Visual Words for the entire video                       |
|     (simply counting how many patches fell into each visual-word                          |
|      cluster, across the whole clip)                                                        |
|                          |                                                                    |
|                          v                                                                    |
|  6. Feed that final histogram into an SVM Classifier -> get the                                 |
|     predicted action label ("walking", "running", etc.)                                           |
+-------------------------------------------------------------+
```

Imagine chopping an entire video into hundreds of tiny moving "stickers" (the space-time patches), briefly describing each individual sticker with just a handful of numbers (its HOG, HOF, or MBH descriptor), then sorting visually similar-looking stickers into a set of clearly labeled piles (the "visual words," much like sorting a giant bucket of Lego bricks by their exact shape), then simply counting how many stickers from this particular video landed in each labeled pile (the final histogram), and finally handing that entire count sheet over to a judge (the SVM classifier) whose only job is to decide, based purely on those counts, "this looks like a walking video" or "this looks like a running video."

### Beyond the Classical Pipeline — A Quick Tour of Modern Video Understanding

The lecture closes its Video Processing section with a fast, deliberately brief tour of much harder, more modern video problems, without diving deep into how they're actually solved (those solutions arrive in later lectures of the course):

- **Slow-motion interpolation** — algorithmically generating brand-new intermediate frames that sit between two genuinely real, captured frames (for example, "Super SloMo," referenced at jianghz.me/projects/superslomo).
- **Person re-identification** — given just one portrait photo (a simple headshot), find every single appearance of that exact same person throughout an entire video, even as they change their pose, their clothing, or the lighting around them.
- **Efficient video object detection** — instead of wastefully treating every single frame as a completely fresh, independent detection problem, modern approaches (such as "scale-time lattice" methods) deliberately share object detections across nearby frames over time, exploiting motion and scene consistency to improve both speed and accuracy simultaneously.
- **Movie and story understanding** — going meaningfully beyond simple "detect this one action" or "detect this one object" tasks, toward automatically building a full **Event Flow** (essentially a text synopsis of an entire movie) and a **Character Interaction Map**, both extracted automatically, straight from raw video, using vision models — recognizing not just isolated actions and objects, but genuine contextual storylines, sequences of events, and relationships between characters.

### Mnemonic

**"Flow -> HOG/HOF/MBH -> Bag -> SVM"**
Optical **Flow** gives you raw motion vectors -> describe local patches using **HOG/HOF/MBH** -> cluster those descriptors into a **Bag** of visual words -> classify the resulting histogram with an **SVM**. This single arrow chain is the entire classical video-action-recognition story, compressed into five words.

### Summary

Video Processing Foundations extend everything learned about static images into the added dimension of time. Optical flow captures per-pixel motion as a direction-and-magnitude vector, with magnitude computed via the simple Pythagorean formula `sqrt(u^2 + v^2)`. HOG (appearance) and HOF (motion direction) extend classical feature detection into video, though HOF is provably direction-sensitive rather than purely action-sensitive (opposite-direction walking produces different HOF descriptors despite being the same action). MBH sharpens this further by focusing specifically on motion *boundaries* — where movement direction or speed changes sharply. All of these descriptors feed into the classical Bag-of-Space-Time-Features plus SVM pipeline, the dominant pre-deep-learning approach to action recognition, built from interest-point detection, patch description, visual-word clustering, histogram construction, and final SVM classification. The lecture closes by briefly previewing where video understanding is headed next: slow-motion frame interpolation, person re-identification across an entire video, efficient frame-to-frame object detection, and full automated movie/story understanding.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 10. Cheat Sheet & Exam Hacks

```
+=================================================================+
|  CV-NLP -- LEC 01 ONE-LINERS                                    |
+=================================================================+
|  CV is interdisciplinary: Biology + Physics + Psychology +       |
|  Math + CS + Engineering                                          |
|  AI contains ML contains DL contains CNN                           |
|  (every CNN is DL, but NOT every DL model is a CNN --                |
|   Transformers/RNNs are DL too)                                        |
|                                                                          |
|  MARR'S MODEL:  Input Image -> Primal Sketch -> 2-1/2-D Sketch ->        |
|                  3-D Model                                                |
|  2-1/2-D = viewer-centered (only the visible surface)                      |
|  3-D     = object-centered (the full shape, from any angle)                  |
|                                                                                 |
|  SIFT = scale + rotation invariant keypoints (via scale-space +                 |
|         assigned orientation)                                                    |
|  HOG  = per-pixel gradient magnitude + orientation, in a grid of                   |
|         histograms                                                                  |
|  HOG fails on: deformable objects, smooth/edge-less surfaces, clutter                |
|                                                                                          |
|  SfM = match features across many photos + triangulate -> 3D point cloud                 |
|  Semantic Seg = SAME color per CLASS  |  Instance Seg = UNIQUE color                        |
|                 per OBJECT (lets you count instances)                                         |
|                                                                                                   |
|  ImageNet: 22,000 classes / 15M images  |  ILSVRC: 1,000 classes / 1.43M                          |
|  2012 = the AlexNet moment (error dropped roughly 28% -> 16%)                                       |
|  Human-level error on this benchmark is roughly 5%                                                     |
|  Architecture order: AlexNet (2012) -> GoogLeNet/VGG (2014) -> ResNet (2015)                             |
|                                                                                                              |
|  CLIP = matches text with image via contrastive learning (discriminative)                                     |
|  DALL-E = generates a genuinely NEW image straight from text (generative)                                       |
|                                                                                                                     |
|  RGB = additive (light)  |  CMYK = subtractive (ink)  |  they are OPPOSITES                                          |
|  JPEG pipeline: Color Transform -> Downsample -> DCT -> Quantize -> Encode (Huffman)                                    |
|  I-frame = self-contained  |  P-frame = uses only the past  |  B-frame =                                                  |
|  uses BOTH past and future (adds delay/memory cost -- bad for live calls)                                                   |
|                                                                                                                                  |
|  B(I) = (1/wh) times [sum of ALL pixel values]                                                                                     |
|  Contrast = (Change in Luminance) / (Average Luminance)                                                                              |
|  Log transform: s = c times log(1+|r|)  -> compresses HIGH dynamic range                                                               |
|  Gray-level slicing: keep [A,B] bright, push everything else -> constant                                                                 |
|  Low-pass kernel = (1/9) times all-ones 3x3  |  Sharpen kernel's center                                                                     |
|  weight = 8, all 8 neighbors = -1  |  Median filter -> best against                                                                            |
|  salt-and-pepper noise specifically                                                                                                              |
|                                                                                                                                                       |
|  Optical flow magnitude = sqrt(u^2 + v^2)                                                                                                              |
|  HOF = motion-direction histogram (fails to distinguish two OPPOSITE-                                                                                    |
|  direction versions of the SAME action)                                                                                                                    |
|  MBH = gradient OF the optical flow field itself -> captures motion                                                                                          |
|  BOUNDARIES specifically                                                                                                                                        |
|  Classical action recognition pipeline: interest points -> patches ->                                                                                             |
|  HOG/HOF/MBH descriptors -> Bag-of-Visual-Words histogram -> SVM classifier                                                                                          |
+=================================================================+
```

### Exam Red Flags

1. **"Why is Marr's model called 2-1/2-D and not 3D?"** -> Because it is viewer-centered (it only encodes the currently visible surface, relative to the current viewpoint), not a full object-centered volumetric model. Avoid saying "because it's not fully accurate" — the real reason is specifically about *reference frame* (viewer-relative versus object-relative), not general accuracy.
2. **"What exactly makes SIFT rotation/scale invariant?"** -> Two distinct mechanisms must both be named for full credit: the scale-space representation (handles scale) AND the assigned keypoint orientation (handles rotation). Naming only one mechanism usually costs partial marks.
3. **"Differentiate semantic versus instance segmentation."** -> Semantic segmentation assigns the same color per class and cannot count individual instances. Instance segmentation assigns a unique color per individual object and therefore CAN count instances. A very common wrong answer is claiming they're "basically the same thing" — they are not; instance segmentation is a strict, harder refinement of semantic segmentation.
4. **"Why does JPEG convert RGB to YCbCr before doing anything else?"** -> Human eyes are far more sensitive to brightness (luminance) than to color (chrominance) differences, so this separation allows JPEG to downsample the color channels aggressively while still preserving perceived image quality. Don't just say "it's a different color space" — always explain the underlying *perceptual* reasoning.
5. **"Why not just use only B-frames in video compression, if they compress best?"** -> B-frames require BOTH the past frame AND a future frame to already be decoded before they themselves can be decoded, which adds noticeable delay and requires more memory — a bad trade-off for real-time applications such as video calls. A frequent wrong answer claims "B-frames are lower quality," which is false; the real issue is latency and memory, not visual quality.
6. **"State the brightness and contrast formulas exactly."** -> `B(I) = (1/wh) times the sum of ALL pixel values` and `Contrast = (Change in Luminance) / (Average Luminance)`. Never forget the `1/wh` normalization step in brightness — a very common mistake is to report just the raw sum without dividing by the total pixel count.
7. **"When exactly is the log transform useful?"** -> Specifically for high-dynamic-range images (astronomical, medical, or satellite imagery) — because the log function compresses large values proportionally more than small ones, which prevents a handful of extremely bright outlier pixels from washing out all the useful mid-tone detail.
8. **"Distinguish CLIP from DALL-E clearly."** -> CLIP is discriminative — it matches given text against an already-existing pool of images, using contrastive learning. DALL-E is generative — it creates a completely new image purely from a text description, with no existing pool of candidate images involved at all. They come from the same research family and era, but solve genuinely opposite-direction problems — never conflate them.
9. **"Why does HOF fail to tell apart two people walking in opposite directions?"** -> Because HOF encodes *absolute* motion direction in its histogram bins, not merely "the general type of motion" — walking left and walking right activate entirely different direction bins, producing two visibly different HOF descriptors even though the underlying *action* (walking) performed by both people is identical.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

## 11. Full-Lecture Summary

Lecture 1 lays the entire foundation for everything the CV-NLP course will build on. It opens by establishing that Computer Vision is not a self-contained computer-science topic but a genuinely interdisciplinary field, borrowing directly from biology, physics, psychology, and mathematics, and organizes itself neatly inside the AI-contains-ML-contains-DL-contains-CNN family tree. It then walks through CV's history at high speed: biological vision evolved independently across many species long before any camera existed; humans built their first artificial "eye" with the 16th-century camera obscura; and 20th-century neuroscience revealed that even the human brain runs specialized, departmentalized vision circuits rather than one generic vision module. David Marr's 1970s Computational Vision model then supplies the field's core theoretical skeleton — Input Image, Primal Sketch, 2-1/2-D Sketch, and 3-D Model — with the viewer-centered-versus-object-centered distinction between the last two stages being one of the most heavily tested ideas in the whole lecture.

From there, the lecture covers two influential hand-crafted feature detectors, SIFT (scale- and rotation-invariant keypoints, ideal for matching the same point across very different photos) and HOG (dense gradient-orientation grids, ideal for recognizing roughly rigid silhouettes like a standing human, but weak against deformable objects, smooth surfaces, and occlusion). Structure from Motion is introduced as the geometric technique for recovering 3D scene structure purely from overlapping 2D photographs, via feature matching and triangulation — the same basic trick your own two eyes use for everyday depth perception. Semantic and instance segmentation are carefully distinguished, with instance segmentation's ability to count individual objects being the key differentiator.

The lecture then narrates the deep-learning revolution in CV: how the 2012 collision of the ImageNet dataset, deep CNNs, and GPU-scale compute produced AlexNet's landmark win, nearly halving the previous year's classification error rate, followed by a rapid cascade of improved architectures (GoogLeNet, VGG, ResNet) that eventually surpassed human-level accuracy on the ILSVRC benchmark, and continued climbing all the way to Vision Transformers by 2021. CLIP and DALL-E close out the historical arc by finally connecting vision with free-form natural language — CLIP as a discriminative text-to-image matcher using contrastive learning, and DALL-E as a generative text-to-image creator — directly bridging this course's Computer Vision half with its Natural Language Processing half.

Finally, the lecture grounds all of this history in concrete, usable mathematics: color models (RGB's additive light-mixing versus CMYK's subtractive ink-mixing), the five-step JPEG compression pipeline (color transform, downsampling, DCT, quantization, entropy encoding), video compression via I/P/B frames, and a toolbox of image-enhancement formulas — brightness, contrast, histograms, the log transform, gray-level slicing, and spatial-filter kernels (low-pass, median, sharpening). The lecture then extends this same mathematical toolbox into the time dimension with video processing: optical flow (per-pixel motion vectors and their Pythagorean magnitude), HOG/HOF/MBH descriptors for motion, and the classical Bag-of-Space-Time-Features plus SVM pipeline for action recognition — closing with a brief preview of modern video understanding tasks like slow-motion interpolation, person re-identification, efficient video object detection, and full automated movie/story understanding, all of which set the stage for the deep-learning-based architectures the rest of this course will cover.

[Back to Top](#-cv-nlp--lec-01-introduction--theory)

---

> **Next:** [🧮 NUMERICAL →](cvnlp_lec01_intro_numerical.md) · [🎯 PRACTICE →](cvnlp_lec01_intro_practice.md) · [Lecture 01 README →](README.md)
>
> *CV-NLP · Lec 01 · github.com/rpaut03l/TS-02-03*
>
> *Some figures/explanations in the source lecture are themselves adapted from publicly available academic resources and research papers (Marr 1982; Lowe 1999; Dalal & Triggs 2005; Deng et al. 2009; Russakovsky et al. 2015; Kanwisher et al. 1997; Epstein & Kanwisher 1998; OpenAI CLIP/DALL-E 2021). All original-author credit is preserved here; this write-up is original explanatory text, not a slide reproduction.*
