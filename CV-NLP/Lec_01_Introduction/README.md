# CV-NLP — Lecture 01: Introduction

### *What is Computer Vision · History of CV · Image Processing Foundations · Video Processing Foundations*

> 🔗 **Repo:** [github.com/rpaut03l/TS-02-03](https://github.com/rpaut03l/TS-02-03) · [CV-NLP Track](../README.md)
>
> **Files in this folder:**
> [📖 THEORY](cvnlp_lec01_intro_theory.md) · [🧮 NUMERICAL](cvnlp_lec01_intro_numerical.md) · [🎯 PRACTICE](cvnlp_lec01_intro_practice.md)

---

## 🧸 What's In This Lecture, Told As One Continuous Story

Picture a completely blank-slate mind that has never seen a photograph before, and this lecture is the process of teaching that mind, piece by piece, how photographs actually work — starting from "why does seeing even matter" all the way to "here is the exact math for sharpening a blurry picture." Nothing is assumed; every idea builds directly on the idea right before it.

```mermaid
flowchart TD
    START(("🎬 Start: a blank slate<br/>that has never seen<br/>a photograph")) --> P1

    subgraph P1["PILLAR 1 — Why does vision even matter?"]
        direction TB
        p1a["CV borrows from Biology,<br/>Physics, Psychology, Math"] --> p1b["AI ⊃ ML ⊃ DL ⊃ CNN<br/>nested family tree"]
        p1b --> p1c["The job menu: detection,<br/>segmentation, pose, VQA..."]
    end

    P1 --> P2

    subgraph P2["PILLAR 2 — How did we get here?"]
        direction TB
        p2a["Biology solved vision first<br/>(octopus, fly, human)"] --> p2b["Camera Obscura (1545)<br/>humans build an artificial eye"]
        p2b --> p2c["Marr's Model:<br/>Image → Edges → 2½D → 3D"]
        p2c --> p2d["SIFT &amp; HOG<br/>hand-crafted features"]
        p2d --> p2e["2012: AlexNet + ImageNet<br/>+ GPUs collide"]
        p2e --> p2f["2021: CLIP &amp; DALL-E<br/>connect vision + language"]
    end

    P2 --> P3

    subgraph P3["PILLAR 3 — What IS an image, mathematically?"]
        direction TB
        p3a["Color models<br/>RGB / HSL / CMYK"] --> p3b["JPEG compression<br/>5-step pipeline"]
        p3b --> p3c["Brightness, Contrast,<br/>Histograms"]
        p3c --> p3d["Log transform, Gray-slicing,<br/>Spatial filters"]
    end

    P3 --> P4

    subgraph P4["PILLAR 4 — What IS a video, mathematically?"]
        direction TB
        p4a["I / P / B frames"] --> p4b["Optical Flow<br/>(motion vectors)"]
        p4b --> p4c["HOG/HOF/MBH<br/>motion descriptors"]
        p4c --> p4d["Bag-of-Visual-Words<br/>+ SVM action recognition"]
    end

    P4 --> DONE(("🏁 A mind that now understands<br/>images AND video as data"))

    style START fill:#fef3c7,stroke:#d97706
    style DONE fill:#dcfce7,stroke:#16a34a
    style P1 fill:#dbeafe,stroke:#2563eb
    style P2 fill:#fce7f3,stroke:#db2777
    style P3 fill:#fef9c3,stroke:#ca8a04
    style P4 fill:#e0e7ff,stroke:#4f46e5
```

That single diagram IS Lecture 1. Everything below just zooms into each of the four boxes one at a time.

---

## 📚 Contents of This Lecture Folder

| File | What it covers | Length |
|---|---|---|
| [`cvnlp_lec01_intro_theory.md`](cvnlp_lec01_intro_theory.md) | Full concept walkthrough — 9 topic sections + cheat sheet + full-lecture summary, story-first analogies throughout, formal notation, aligned ASCII diagrams, mnemonics | ~13,200 words |
| [`cvnlp_lec01_intro_numerical.md`](cvnlp_lec01_intro_numerical.md) | Every formula (brightness, contrast, log transform, optical-flow magnitude, spatial filter kernels, histogram counts, JPEG block math, ImageNet error-rate math) worked out digit-by-digit, each section closing with its own short summary | ~7,300 words |
| [`cvnlp_lec01_intro_practice.md`](cvnlp_lec01_intro_practice.md) | The instructor's own in-class Q&A (Q1–Q9, spoiler-tagged, each with an added analogy), 4 extra drill sections, 5 real-world applied scenarios, a hands-on mini project, and a 40-question rapid-fire exam bank | ~6,900 words |

---

## 🔬 Zoom-In #1: Marr's Computational Vision Pipeline

The single most important diagram of the entire lecture — everything else builds context around this one idea. A photograph cannot jump straight from "raw pixels" to "I understand this object" in one leap; it has to pass through fixed intermediate stages.

```mermaid
flowchart LR
    A["📷 Input Image<br/><i>raw pixel intensities,<br/>nothing understood yet</i>"] -->|"find edges/blobs"| B["✏️ Primal Sketch<br/><i>zero-crossings, blobs,<br/>edges, boundaries</i>"]
    B -->|"add depth, but only<br/>from THIS viewpoint"| C["🌗 2½-D Sketch<br/><i>viewer-centered depth &amp;<br/>surface orientation</i>"]
    C -->|"generalize beyond<br/>this one viewpoint"| D["🧊 3-D Model<br/><i>object-centered,<br/>true from ANY angle</i>"]

    style A fill:#f1f5f9,stroke:#334155
    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#fef9c3,stroke:#ca8a04
    style D fill:#dcfce7,stroke:#16a34a
```

Notice the labels on the arrows — each arrow is doing real work, not just connecting boxes. The jump from the 2½-D Sketch to the full 3-D Model is the single most commonly misunderstood step in exams: the 2½-D stage only knows what's visible from where the camera is currently standing, while the 3-D stage is a complete, angle-independent fact about the object. Full explanation, with the basketball example and the "why is it called 2½ and not 3" deep dive, lives in [THEORY, Section 3](cvnlp_lec01_intro_theory.md#3-marrs-computational-vision-model).

---

## 🔬 Zoom-In #2: The JPEG Compression Workflow

Every single image that ever gets stored, emailed, or fed into a model passes through some version of this pipeline first.

```mermaid
flowchart LR
    A["Raw Image<br/>(RGB)"] --> B["1. Color Transform<br/>RGB → YCbCr"]
    B --> C["2. Downsampling<br/>shrink COLOR channels<br/>(eyes care less about color)"]
    C --> D["3. DCT<br/>8×8 blocks →<br/>frequency space"]
    D --> E["4. Quantization<br/>drop barely-visible<br/>high frequencies"]
    E --> F["5. Encoding<br/>Huffman coding"]
    F --> G["📦 Compressed<br/>JPEG bitstream"]

    style A fill:#f1f5f9,stroke:#334155
    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#fef9c3,stroke:#ca8a04
    style D fill:#fce7f3,stroke:#db2777
    style E fill:#e0e7ff,stroke:#4f46e5
    style F fill:#fed7aa,stroke:#ea580c
    style G fill:#dcfce7,stroke:#16a34a
```

The trick worth remembering: Step 1 deliberately separates brightness from color specifically so Step 2 can afford to throw away far more color detail than brightness detail — because human eyes simply don't notice small color shifts the way they notice small brightness shifts. Full walkthrough, with the exact reasoning and the video-frame (I/P/B) extension of this same idea, lives in [THEORY, Section 8](cvnlp_lec01_intro_theory.md#8-image-processing-foundations).

---

## 🔬 Zoom-In #3: The Classical Video Action-Recognition Pipeline

Before deep learning, this six-step recipe was the standard way to teach a computer to recognize an action (like "walking" or "running") from raw video.

```mermaid
flowchart TD
    A["🎥 Raw Video"] --> B["1. Find space-time<br/>interest points<br/><i>(where motion happens)</i>"]
    B --> C["2. Extract space-time<br/>patches around them"]
    C --> D["3. Describe each patch<br/>HOG (shape) / HOF (motion) /<br/>MBH (motion boundary)"]
    D --> E["4. Cluster into a<br/>dictionary of<br/>'visual words'"]
    E --> F["5. Build one Histogram<br/>of Visual Words<br/>for the whole video"]
    F --> G["6. SVM Classifier"]
    G --> H["🏃 Predicted Action:<br/>'walking' / 'running'"]

    style A fill:#f1f5f9,stroke:#334155
    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#fef9c3,stroke:#ca8a04
    style D fill:#fce7f3,stroke:#db2777
    style E fill:#e0e7ff,stroke:#4f46e5
    style F fill:#fed7aa,stroke:#ea580c
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

This is essentially "Bag-of-Words" borrowed straight from NLP and applied to chopped-up pieces of video — cluster similar-looking motion patches into named piles, count how many patches from this video landed in each pile, and hand that count sheet to a classifier. Full walkthrough lives in [THEORY, Section 9](cvnlp_lec01_intro_theory.md#9-video-processing-foundations).

---

## 🧭 How to Use This Folder

```mermaid
flowchart LR
    A["Read theory.md<br/>top to bottom, once"] --> B["Redo every worked<br/>example in numerical.md<br/>on paper"]
    B --> C["Attempt practice.md<br/>cold — no peeking"]
    C --> D["Loop back to theory.md's<br/>Cheat Sheet before any quiz"]

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#fef9c3,stroke:#ca8a04
    style C fill:#fce7f3,stroke:#db2777
    style D fill:#dcfce7,stroke:#16a34a
```

1. **Read `theory.md` first**, without stopping to memorize — just get the full story straight, using the zoom-in diagrams above as your map of where you are at any given moment.
2. **Work through `numerical.md` with pen and paper.** Every single number is reproducible by hand; if your answer doesn't match, that mismatch is precisely where your gap is.
3. **Attempt `practice.md` before checking any answers.** Every problem is deliberately hidden behind a spoiler tag so you can't accidentally cheat yourself out of real practice.
4. **Use the Cheat Sheet & Exam Hacks** sections (end of `theory.md` and `numerical.md`) for a final review pass the night before anything graded.

---

## 🔭 Where This Lecture Leads Next

Lecture 1 is 100% foundation — there's no heavy hands-on numerical work here like "compute this CNN's output size," because that specific style of assignment starts once the course reaches actual CNN architectures. This folder's `numerical.md` is deliberately built around the formulas *this particular* lecture introduces (brightness, contrast, log transform, optical flow magnitude, spatial-filter convolution, JPEG block counting, error-rate math), so every worked example is genuinely grounded in what was actually taught, rather than being generic filler.

```mermaid
flowchart LR
    L1["Lec 01<br/>Foundations"] -.->|"builds toward"| CNN["Future:<br/>CNN Architectures"]
    CNN -.->|"builds toward"| DET["Future:<br/>Object Detection"]
    DET -.->|"builds toward"| VL["Future:<br/>Vision-Language Models"]

    style L1 fill:#dcfce7,stroke:#16a34a
    style CNN fill:#f8fafc,stroke:#94a3b8,stroke-dasharray: 5 5
    style DET fill:#f8fafc,stroke:#94a3b8,stroke-dasharray: 5 5
    style VL fill:#f8fafc,stroke:#94a3b8,stroke-dasharray: 5 5
```

---

## 🧠 Quick-Glance Mnemonic Bank for This Lecture

| Mnemonic | Unpacks to |
|---|---|
| BIOM -> CV -> HIST -> IMG -> VID | The master chant for this entire lecture's 4-pillar structure |
| Bio-Cam-Neuro | Biological vision -> Camera obscura -> Neuroscience specialization |
| P-2½-3: Pixels, Pieces, Position, Presence | Marr's four stages, in order |
| SIFT Spins & Scales, HOG Grids & Points | SIFT vs. HOG, in six words |
| SfM Triangulates, Semantic Groups, Instance Splits | 3D reconstruction + the two segmentation types |
| C-P-I-I -> A-V-G-M -> ViT | Dataset growth -> architecture evolution -> Transformer takeover |
| CLIP Connects, DALL-E Draws | Discriminative matching vs. generative creation |
| C-J-F-B-C-H-D-L-G-S | The full Image Processing Foundations section, in order |
| Flow -> HOG/HOF/MBH -> Bag -> SVM | The full classical video action-recognition pipeline |

---

> **Next:** [📖 THEORY →](cvnlp_lec01_intro_theory.md)
>
> *CV-NLP · Lec 01 · github.com/rpaut03l/TS-02-03*
