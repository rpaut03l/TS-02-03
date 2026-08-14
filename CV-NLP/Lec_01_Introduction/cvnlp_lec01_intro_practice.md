# 🎯 CV-NLP — Lec 01: Introduction — PRACTICE

### *The instructor's own in-class Q&A, self-test drills, a mini project, and an exam-style Q&A bank*

> **Nav:** [← Lec 01 README](README.md) | [📖 THEORY](cvnlp_lec01_intro_theory.md) | [🧮 NUMERICAL](cvnlp_lec01_intro_numerical.md) | **PRACTICE**

> 💡 Every problem below is inside a `<details>` block. **Try to solve it on paper first**, then click "Show Answer" to check yourself. Treat this file the way you'd treat a real exam room — no peeking until you've genuinely committed to an answer.

---

## 🧠 MASTER MNEMONIC: "Q-C-F-I-R-M-E"

> In-class **Q**&A first, then **C**oncept checks, **F**eature-detector drills, **I**mage/video-processing drills, **R**eal-world applied questions, a **M**ini project, and finally an **E**xam-style rapid-fire bank.

---

## 📚 Table of Contents

| # | Section | Jump |
|---|---|---|
| 1 | In-Class Q&A — As Asked by the Instructor (Q1–Q9) | [Section 1](#1-in-class-qa--as-asked-by-the-instructor) |
| 2 | Concept Checks — Marr's Model & CV Basics | [Section 2](#2-concept-checks--marrs-model--cv-basics) |
| 3 | Feature Detector Drills (SIFT / HOG) | [Section 3](#3-feature-detector-drills-sift--hog) |
| 4 | Image & Video Processing Drills | [Section 4](#4-image--video-processing-drills) |
| 5 | Real-World / Applied Practice Questions | [Section 5](#5-real-world--applied-practice-questions) |
| 6 | Mini Project — Build a Mini Marr Pipeline | [Section 6](#6-mini-project--build-a-mini-marr-pipeline) |
| 7 | Exam-Style Rapid-Fire Q&A Bank | [Section 7](#7-exam-style-rapid-fire-qa-bank) |
| 8 | Full-File Summary | [Section 8](#8-full-file-summary) |

---

## 1. In-Class Q&A — As Asked by the Instructor

> These are the exact discussion questions the instructor posed live during Lecture 1, reconstructed here with their complete answers and extra explanatory context, spoiler-tagged so you can genuinely self-test before checking your work.

**Q1.** Why do you think the human visual system needs multiple stages like edge detection and depth estimation, rather than directly recognizing a 3D object from an image?

<details>
<summary>Show Answer</summary>

A 2D image is **ambiguous and incomplete** — it does not directly contain complete information about a three-dimensional object. The same physical object may look very different under changes in lighting, distance, viewpoint, background, or occlusion. So the visual system first detects simple features (edges, colours, textures, orientations), then combines these to estimate shape, depth, motion, and relative position, and only *then* matches this structured information against previously learned object representations for recognition.

Picture trying to identify a friend purely from a single, oddly-lit, partly-shadowed photograph taken from an unusual angle — your brain doesn't magically "just know" it's your friend the instant light hits your retina. It first pulls out edges and shapes, works out roughly how those shapes sit in space, and only then compares that structured understanding against everyone you already know. Skip any one of those in-between steps, and recognition simply breaks down. In short: **pixels tell us what is visible, edges tell us where structures begin and end, depth tells us how those structures are arranged, and the 3D model tells us what the object is** — each stage adds information the raw pixels alone could never provide on their own.
</details>

---

**Q2.** In Marr's model, why is it called a "2½-D sketch" instead of 3D?

<details>
<summary>Show Answer</summary>

Because it includes depth information, but **only relative to the viewer's own viewpoint** — not a full volumetric model. It is called 2½-D rather than fully 3D because the representation is mainly constructed from the observer's current viewpoint. For example, we can estimate that the visible surface of a basketball is curved, but we cannot directly see (or represent, at this stage) its back side.

Think of standing directly in front of a beach ball at a pool party. You can absolutely tell, just by looking, that the near side curves gently toward you — that's genuine depth information, and it's free, immediate, and correct. But you have exactly zero direct visual evidence right now about what the far side of that ball looks like; you're only assuming, from a lifetime of prior experience with round objects, that it curves the same way all the way around. That "current-viewpoint-only" depth estimate is the 2½-D Sketch. A full 3-D model, by contrast, is object-centered and represents the object's complete shape independent of viewpoint — true and complete from every angle, including the one nobody happens to be looking at right now.
</details>

---

**Q3.** What makes a feature point "good" for matching across images?

<details>
<summary>Show Answer</summary>

A good feature point is one that is **distinctive**, **stable under scale and rotation**, and **reliable across lighting/viewpoint changes**. Typically, corners or blobs with unique texture patterns work best because they are easy to localize and match — a flat, featureless patch of blue sky is a genuinely bad feature point (nothing distinctive to lock onto, and every nearby patch of sky looks identical to it), while a sharp, uniquely-shaped corner of a building is excellent.

Picture trying to find a specific friend again in a huge, packed crowd photo. If your friend is wearing a plain gray shirt identical to fifty other people around them, you'll never reliably spot them a second time — there's nothing distinctive to search for. But if your friend is wearing a bright, uniquely patterned scarf, you can spot them instantly and confidently, even in a completely different photo taken moments later from a slightly different angle. Good feature points are exactly that "unique scarf" — something distinctive enough to be found again with confidence.
</details>

---

**Q4.** If I rotate or scale the image of a building, will the same keypoints be found using SIFT? Why?

<details>
<summary>Show Answer</summary>

**Yes** — SIFT is designed to be invariant to scale and rotation. It detects keypoints using a **scale-space representation** and assigns an **orientation** to each keypoint, making the descriptor rotation- and scale-invariant. So the same physical corners/blobs on the building get re-detected regardless of how the photo was rotated or zoomed.

Imagine every distinctive corner of the building carrying its own tiny personal compass and its own personal measuring tape, both of which automatically re-calibrate themselves no matter how you spin or zoom the overall photograph. When SIFT looks at a rotated, zoomed photo of the same building later, it isn't comparing raw pixel coordinates (which genuinely would change under rotation/zoom) — it's comparing these self-correcting compass-and-ruler readings, which stay consistent, so it can still confidently say "yes, this is the same corner I marked before."
</details>

---

**Q5.** What kind of object shapes do you think HOG would fail to detect effectively?

<details>
<summary>Show Answer</summary>

HOG struggles with objects that are: (1) **highly deformable** (e.g., animals in varied poses), (2) **lacking strong edges** or having smooth surfaces, and (3) **occluded or in cluttered backgrounds** — because HOG relies on consistent edge orientation patterns over a fixed grid, and all three of these situations break that consistency.

Picture trying to recognize a friend's silhouette across a busy, noisy street, except your friend is also wearing a huge, floppy, ever-changing costume that reshapes itself with every step they take. A detector built entirely around "does this match a stable, predictable outline template" genuinely struggles here — not because it's a poorly designed algorithm, but because the very thing it fundamentally depends on (a consistent, predictable edge pattern) has been taken away by the deformable costume, the smooth featureless surface, or the cluttered background hiding half the silhouette.
</details>

---

**Q6.** How do you think the algorithm figures out the 3D structure just from photos? What clues can it use?

<details>
<summary>Show Answer</summary>

It finds the **same points/features in multiple photos** and uses **triangulation** to compute their 3D position. By knowing the camera positions and viewing angles for each photo, it calculates where the point must exist in 3D space. This is the Structure from Motion (SfM) technique.

Picture photographing a marble statue from ten different spots as you slowly circle around it, and noticing the exact same tiny chip in the marble shows up in five of your ten photographs. Because you roughly know where your camera was standing and which direction it was pointed for each of those five shots, simple geometry — imagining a straight line drawn from each camera position through where that chip appears in that camera's photo — tells you exactly where the chip sits in real 3D space, at the single point where all five imaginary lines cross. Repeat this trick for thousands of tiny distinctive points, and the whole statue's 3D shape emerges, built entirely from flat photographs and careful geometry, with no laser scanner required.
</details>

---

**Q7.** Looking at semantic segmentation vs. instance segmentation outputs — can you explain the main difference?

<details>
<summary>Show Answer</summary>

**Semantic segmentation** labels all objects of the same class with the same color — it knows *what* is a chair, but not *which* chair. **Instance segmentation** goes one step further — it not only identifies what is a chair, but also separates each individual chair as a separate object (each with its own unique color label). So instance segmentation is strictly more informative — it lets you *count* distinct objects, semantic segmentation does not.

Think of coloring a picture-book page where the only instruction is "color every chair blue." Once finished, you genuinely cannot tell, just by looking at the coloring, whether there were three chairs squeezed tightly together or one giant blue blob — that's semantic segmentation's fundamental ceiling. Now imagine instead being told "give every single chair its own separate crayon color, even where two chairs are touching." Simply by counting distinct colors afterward, you can now say with confidence exactly how many chairs are in the picture — that upgrade is instance segmentation.
</details>

---

**Q8.** Why don't we use only B-frames for better compression in video?

<details>
<summary>Show Answer</summary>

Because **B-frames need both past AND future frames to be decoded**, so they increase delay and require more memory — not ideal for real-time applications like video calls. A decoder receiving only B-frames would have to wait for future frames to even exist before it could reconstruct the current one, which is unacceptable for live/real-time streaming.

Think of an I-frame as a complete photograph pinned up on a corkboard, fully understandable on its own. A B-frame is like a sticky note that only makes sense once you've already read BOTH the note pinned before it AND the note that will be pinned after it — meaning you're stuck waiting for a note that hasn't even been written yet before you can understand the one in front of you right now. That's a perfectly fine trade-off for a pre-recorded movie file sitting on a hard drive (all the "future" notes already exist), but it's a genuinely bad trade-off for a live video call, where the "future" frame literally hasn't happened yet.
</details>

---

**Q9.** If two people walk in opposite directions, will their HOF features look the same or different?

<details>
<summary>Show Answer</summary>

**Different** — because the direction bins in the Histogram of Optical Flow will change. HOF encodes the *direction* of motion (not just its presence), so a person walking right fills different histogram bins than a person walking left, even though both are performing the same underlying *action* (walking).

Picture two weathervanes spinning in a strong wind, one pointing steadily east and the other pointing steadily west. If you're only counting "how many weathervanes are spinning" (that's roughly what optical flow magnitude alone captures), both look identical. But if you're recording exactly which direction each one points (that's what HOF's histogram bins capture), the two readings are obviously, visibly different — even though "wind is blowing and something is spinning" is true for both. (See the Numerical file's Section 7, Example 3, for the exact magnitude-versus-direction arithmetic behind this.)
</details>

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 2. Concept Checks — Marr's Model & CV Basics

**Q2.1.** List, in order, the four stages of Marr's Computational Vision model.

<details>
<summary>Show Answer</summary>

1. Input Image (perceived intensities) -> 2. Primal Sketch (zero-crossings, blobs, edges, bars, ends, virtual lines, groups, curves, boundaries) -> 3. 2½-D Sketch (local surface orientation and depth discontinuities, viewer-centered) -> 4. 3-D Model Representation (hierarchically organized volumetric/surface primitives, object-centered).

Picture climbing a four-rung ladder where each rung only makes sense once you've stepped on the rung below it: raw light hitting your eye (rung 1) has to become edges and outlines (rung 2) before it can become "this curves toward me from where I'm standing" (rung 3), before it can finally become "this is a complete, viewpoint-independent sphere" (rung 4). Skipping a rung isn't just harder — in Marr's framework, it's genuinely not possible.
</details>

---

**Q2.2.** True or False: According to the AI ⊃ ML ⊃ DL ⊃ CNN nesting, every Deep Learning model is a CNN.

<details>
<summary>Show Answer</summary>

**False.** The nesting means every CNN is DL, and every DL technique is ML, and every ML technique is AI — but the reverse doesn't hold at any level. There are DL architectures that are NOT CNNs (e.g., plain RNNs, Transformers/ViT) — CNN is just one specific sub-family inside DL.

Think back to the nesting-dolls picture from the theory file: cracking open the smallest CNN doll always reveals it was sitting inside the DL doll, but cracking open the DL doll does NOT always reveal a CNN doll inside — sometimes you find a completely different-shaped doll in there instead (a Transformer-shaped one, or an RNN-shaped one). The "contains" relationship only ever runs in one direction.
</details>

---

**Q2.3.** Why does the lecture call vision "the entrance hall of AI"?

<details>
<summary>Show Answer</summary>

Because vision is described as **the most important source of information for the human brain**, and by extension, for most AI systems that interact with the physical/visual world. A huge fraction of downstream reasoning tasks (scene understanding, robotics, cognitive reasoning about "why is this object here") depend on first correctly perceiving the visual scene.

Picture walking into a friend's house for the very first time. Before you can decide where to sit, whether dinner is nearly ready, or whether the dog is friendly, you first have to physically walk through the entrance hall and simply *see* the place. Every smarter decision that follows is built on top of that very first look around — which is exactly why the lecture insists vision has to come first, structurally, before deeper AI reasoning can even begin.
</details>

---

**Q2.4.** A classmate claims the Primal Sketch stage already tells you what the object IS (e.g., "it's a chair"). Correct this misunderstanding.

<details>
<summary>Show Answer</summary>

**Incorrect.** The Primal Sketch stage only extracts low-level 2D structure — zero-crossings, blobs, edges, bars, ends, virtual lines, groups, curves, and boundaries. It has no notion of object identity whatsoever. Recognizing "this is a chair" requires matching the fully assembled 3-D Model stage against previously learned object representations — a much later step than the Primal Sketch.

Picture a child with a crayon carefully tracing only the outlines in a coloring book page, without yet knowing what any of the shapes actually represent. That tracing (the outline alone) is the Primal Sketch. Only much later, after building up a full understanding of the shape's arrangement in space and comparing it against things the child has seen before, does "oh, this is a chair!" recognition actually happen.
</details>

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 3. Feature Detector Drills (SIFT / HOG)

**Q3.1.** A colleague says: "HOG and SIFT are basically the same thing — both find features in an image." Correct or challenge this statement.

<details>
<summary>Show Answer</summary>

**Challenge it.** They solve related but distinct problems. **SIFT** detects sparse, distinctive **keypoints** (corners/blobs) and describes them with scale/rotation-invariant descriptors — good for matching the *same* point across very different views of a scene (e.g., for Structure from Motion, panorama stitching). **HOG** computes a **dense grid** of gradient-orientation histograms over an entire image region (e.g., a whole pedestrian bounding box) — good for describing an object's overall *silhouette shape* for detection/classification, not for precise point-to-point matching across viewpoints.

Think of SIFT as a detective hunting for a handful of highly specific fingerprints scattered around a crime scene, while HOG is more like an artist sketching the overall outline and shading of an entire figure in one sweep. Both are looking at the same photograph, but they're solving genuinely different jobs — fingerprint matching versus overall silhouette description — and swapping one in for the other usually produces disappointing results.
</details>

---

**Q3.2.** Why is a smooth, featureless wall a bad place for SIFT to find keypoints?

<details>
<summary>Show Answer</summary>

SIFT needs **distinctive, locally unique** structure (corners, blobs, sharp texture) to detect and reliably re-match a keypoint. A smooth wall has no local intensity variation — every small patch looks like every other patch, so there's nothing "distinctive" to lock onto, and even if a keypoint were detected there, it couldn't be reliably re-matched (any nearby patch would look equally similar).

Picture trying to find "your" specific square of a completely blank white wall in a photograph, and then trying to find that exact same square again in a second photograph. Every square of blank wall looks identical to every other square — there is genuinely no way to tell them apart, no matter how carefully you look. That total absence of distinctiveness is exactly the situation SIFT is helpless against.
</details>

---

**Q3.3.** A HOG-based pedestrian detector is deployed at a crowded train station where people are frequently partially hidden behind pillars and each other. Predict its likely failure mode, and explain why using the theory.

<details>
<summary>Show Answer</summary>

It will likely **miss occluded pedestrians** or produce weaker detection confidence for them. HOG relies on a **consistent grid of edge orientations matching a learned human-silhouette template** — occlusion removes or distorts parts of that grid (e.g., legs hidden behind a pillar), breaking the expected edge pattern the classifier was trained to recognize. This matches the theory's explicit failure case: "occluded or in cluttered backgrounds" is listed as a known HOG weakness.

Picture trying to recognize a familiar shape from a jigsaw puzzle when a third of the pieces are missing entirely and replaced with a completely different, unrelated picture underneath. The overall outline you were expecting to match against simply isn't fully there anymore — and HOG, which depends on the *whole* expected grid pattern lining up, suffers exactly this kind of confusion whenever a pillar or a second person blocks part of the target.
</details>

---

**Q3.4.** Would SIFT or HOG be the better choice for stitching together a panorama photo from several overlapping, slightly-rotated shots of the same landscape? Justify your choice.

<details>
<summary>Show Answer</summary>

**SIFT.** Panorama stitching fundamentally requires finding the SAME physical points across multiple overlapping photos, which will naturally differ slightly in rotation, scale, and viewpoint due to handheld camera movement between shots. That is exactly the scenario SIFT's scale-and-rotation-invariant keypoints are purpose-built for. HOG, by contrast, is designed to describe an object's overall silhouette shape (like a standing pedestrian) for detection/classification — it has no mechanism for finding and matching one specific physical point precisely across multiple photos, so it would be a poor fit here.

Imagine trying to line up two overlapping strips of a torn photograph. You'd naturally look for a few small, uniquely identifiable details near the torn edge — a distinctive rooftop corner, a specific tree branch — and match those exact same details in both strips to line everything up correctly. That's precisely SIFT's job. HOG, by comparison, would only tell you "there's roughly a landscape-shaped thing here," which doesn't help you line up the torn edges precisely at all.
</details>

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 4. Image & Video Processing Drills

**Q4.1.** An astronomer has a 16-bit image with pixel values from 0 to 65,535, and most stars are much dimmer than a few very bright ones, making faint stars invisible on a normal display. Which enhancement technique from this lecture should they use, and why?

<details>
<summary>Show Answer</summary>

The **log transform** (`s = c . log(1+|r|)`). It compresses large values more than small ones, so the handful of extremely bright pixels (stars) get compressed down proportionally more, while the dim, previously-invisible faint stars get relatively boosted into a visible range. The theory file explicitly lists astronomical images as a canonical use case for this technique.

Picture a very tall, unevenly spaced staircase where the first few steps are close together, but the last few steps near the top are absurdly far apart. The log transform effectively "squashes" that top section of enormous, widely-spaced steps down into something climbable, while leaving the already-close-together lower steps mostly untouched — exactly matching what needs to happen to make faint, dim stars visible next to blindingly bright ones.
</details>

---

**Q4.2.** A photographer wants to see ONLY the sky (a narrow band of blue intensities) highlighted in a black-and-white version of a photo, with everything else pushed to black. Which technique fits, and how would they set it up?

<details>
<summary>Show Answer</summary>

**Gray-level slicing.** They would identify the intensity range `[A, B]` corresponding to the sky's brightness values in the grayscale image, set pixels inside `[A, B]` to a bright/enhanced value (e.g., 255), and set everything outside that range to a constant (e.g., 0/black) — isolating just the sky tonal band.

Picture a strict bouncer standing at a club door with one very specific instruction: only let in guests wearing a badge numbered between A and B; everyone else gets turned away and sent home. Gray-level slicing runs exactly this same "guest list" check on every single pixel's brightness value, letting through (and lighting up) only the narrow band that matches the sky, and firmly rejecting (blacking out) everything else.
</details>

---

**Q4.3.** A video call app needs the lowest possible latency. Should its encoder rely heavily on B-frames? Justify using the I/P/B-frame theory.

<details>
<summary>Show Answer</summary>

**No.** B-frames require both past AND future frames to already be decoded before they themselves can be decoded — this adds delay (you must wait for "future" frames to arrive) and increases memory requirements (multiple frames must be buffered simultaneously). For a low-latency real-time application like a video call, the encoder should lean more on I-frames and P-frames, minimizing or avoiding B-frames.

Think again of the sticky-note analogy: a B-frame sticky note only makes sense once you've already read the note that comes AFTER it in time — but in a live video call, that "future" note genuinely hasn't been written yet, because the other person hasn't spoken or moved yet. Waiting around for a note that doesn't exist yet is exactly the kind of delay a real-time app cannot afford.
</details>

---

**Q4.4.** A photo has salt-and-pepper noise (random pure-black and pure-white speckles). Should you apply a low-pass (averaging) filter or a median filter, and why?

<details>
<summary>Show Answer</summary>

**Median filter.** Salt-and-pepper noise consists of extreme outlier values (pure 0 or pure 255). An averaging (low-pass) filter would be dragged toward these outliers (as shown numerically in the Numerical file Section 6b, where one outlier pixel changed the average from 50 to ~68.3). A median filter is robust to outliers — it picks the middle-ranked value, which stays unaffected by a small number of extreme values, and importantly preserves edges better than averaging.

Picture asking nine friends to guess a stranger's age, where eight of them guess sensibly (all somewhere between 30 and 40) but one joker shouts "one thousand!" as a prank. If you took the plain average of all nine guesses, that single ridiculous joke answer would drag your final estimate wildly off course. But if you instead lined up all nine guesses from lowest to highest and simply picked the middle one, the joke answer would land safely at the very end of the line and have zero effect on your final answer — exactly how a median filter shrugs off a single wild outlier pixel.
</details>

---

**Q4.5.** A security camera captures a scene with a mostly-static background and one person walking through the frame. Which classical descriptor from this lecture would most efficiently highlight just the moving person, and why?

<details>
<summary>Show Answer</summary>

**Optical flow.** Optical flow specifically captures apparent motion between consecutive frames — every pixel gets a motion vector (u, v), and pixels belonging to the static background will have a flow magnitude close to zero (`sqrt(u^2 + v^2) ~= 0`), while pixels belonging to the moving person will have a clearly nonzero magnitude. This naturally separates "moving foreground" from "static background" without needing any prior knowledge of what a person looks like.

Picture comparing two nearly identical photographs of a room taken a second apart, and simply highlighting only the pixels that changed between the two shots. Everything that stayed perfectly still (the furniture, the walls) would show essentially zero change, while the walking person would stand out clearly as the only region with real movement — that's optical flow doing exactly its job.
</details>

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 5. Real-World / Applied Practice Questions

> These go beyond the slides — applied scenarios to test whether you can *transfer* the concepts, not just recall them. Treat these as the closest simulation of a viva or open-ended exam question.

**Q5.1.** You're building a mobile app that must recognize the SAME painting in a museum from photos taken by different visitors, at different angles, zoom levels, and lighting conditions. Which classical feature detector from this lecture would you reach for first, and what specific property makes it suited to this task?

<details>
<summary>Show Answer</summary>

**SIFT.** Different visitors' photos will vary in scale (zoom/distance), rotation (camera angle/tilt), and somewhat in viewpoint — exactly the three conditions SIFT is explicitly designed to be robust against (scale-invariance via scale-space representation, rotation-invariance via assigned keypoint orientation). HOG would be a poor fit here since it's built for describing a roughly-fixed silhouette shape (like a standing pedestrian), not for matching arbitrary rigid 2D artwork across wildly different photo angles.

Imagine trying to recognize the exact same painting from a dozen visitors' phone photos, each snapped from a slightly different spot in the room, at a slightly different zoom, under slightly different gallery lighting. What you genuinely need is a way to say "these ten distinctive brushstroke corners in photo A are the exact same physical spots as those ten corners in photo B," regardless of how zoomed-in or tilted each photo happens to be — which is precisely SIFT's specialty.
</details>

---

**Q5.2.** A city wants to automatically count how many individual delivery trucks are parked (possibly overlapping/touching) in a satellite image. Should they use semantic or instance segmentation, and why does the distinction matter operationally here?

<details>
<summary>Show Answer</summary>

**Instance segmentation.** The city needs an actual *count* of individual trucks. Semantic segmentation would only tell them "this blob of pixels is truck-class," merging touching/overlapping trucks into one undifferentiated region — useless for counting. Instance segmentation assigns a unique label per truck, even where trucks are adjacent, enabling an accurate count.

Picture a satellite photo of a parking lot with a dozen delivery trucks parked bumper-to-bumper in a tight row. Semantic segmentation would paint the whole row one single color — "truck" — leaving you with no way to tell if that's four trucks or fourteen. Instance segmentation instead gives every individual truck its own distinct color, so counting distinct colors directly gives you the answer the city actually needs.
</details>

---

**Q5.3.** A satellite sends back a raw 12-bit sensor image (values 0–4095) that needs to be JPEG-compressed for transmission back to Earth over a low-bandwidth link. Walk through, in order, which of this lecture's techniques would plausibly be applied and why.

<details>
<summary>Show Answer</summary>

A plausible pipeline: (1) **Log transform** first, to compress the 12-bit high-dynamic-range data down toward a more standard 8-bit displayable/storable range without losing faint detail (astronomical/satellite imagery is the textbook use case). (2) Convert to **YCbCr** color transform (if the sensor is color) to separate luminance from chrominance. (3) **Downsample** the chroma channels (humans/downstream analysts are less sensitive to color precision than brightness precision), saving significant bandwidth (4x fewer chroma samples under 4:2:0, as computed in Numerical Section 8). (4) Apply the **DCT -> Quantization -> Huffman Encoding** steps of the standard JPEG pipeline to compress the remaining data for transmission.

Picture a satellite engineer working under an extremely tight radio-bandwidth budget, essentially trying to fit an enormous, richly-detailed photograph through a very narrow "pipe" back to Earth. Every technique in this pipeline exists specifically to squeeze that photograph down as small as possible while throwing away only the details human eyes and scientists genuinely won't miss — brightness outliers get log-compressed, color gets downsampled, and the remaining data gets transformed and entropy-coded down to the smallest possible size.
</details>

---

**Q5.4.** A sports analytics company wants to automatically classify whether a player is "sprinting" or "jogging" purely from raw video, using only classical (pre-deep-learning) techniques from this lecture. Sketch the pipeline.

<details>
<summary>Show Answer</summary>

The classical **Bag-of-Space-Time-Features + SVM** pipeline: (1) detect space-time interest points where motion/appearance changes (e.g., limb joints in motion) — likely using optical flow to locate motion-rich regions first; (2) extract space-time patches around these points; (3) compute HOG (appearance), HOF (motion direction), and/or MBH (motion boundary) descriptors per patch — HOF/MBH will be especially useful here since sprinting vs. jogging differ mainly in motion *speed/magnitude and pattern*, not appearance; (4) cluster descriptors into a visual-word dictionary; (5) build a bag-of-visual-words histogram for each video clip; (6) train an SVM classifier on labeled sprinting/jogging clips to predict the class for new clips.

Picture a coach who has spent a season watching hundreds of sprint clips and hundreds of jog clips, building up an intuitive sense of "sprinting legs move like THIS, jogging legs move like THAT" — a mental average pattern for each action. The MBH-averaging worked example from the theory file's Section 9 (averaging 100 walking videos to get a typical MBH pattern) is exactly this coach's intuition, made mathematically precise: build an average motion-pattern "fingerprint" for each action class, then compare any new clip's fingerprint against both stored averages and pick whichever one it resembles more closely.
</details>

---

**Q5.5.** A drone company wants to detect whether a rooftop solar panel installation has changed (new panels added, panels removed) by comparing two aerial photos of the same rooftop taken six months apart, from slightly different drone flight paths. Which two techniques from this lecture would you combine, and in what order?

<details>
<summary>Show Answer</summary>

First, use **SIFT** to find and match distinctive keypoints between the two photos (rooftop edges, chimney corners, existing panel frame corners) despite the slightly different drone viewpoints and altitudes — this lets you geometrically align (warp) the two photos onto a shared coordinate frame, the same underlying matching principle used in Structure from Motion. Second, once both photos are aligned to the same coordinate frame, compare pixel regions directly (potentially using **gray-level slicing** to isolate the specific brightness/reflectivity range typical of solar panel glass, separating panel regions from ordinary roof material) — regions that differ meaningfully between the two aligned, sliced images indicate genuine panel changes, rather than an artifact of the drone simply having flown a slightly different path.

Picture laying two "before" and "after" transparent photo overlays exactly on top of each other by lining up a few unmistakable landmarks — the chimney corner, a roof edge — so that any true differences you spot afterward are genuine changes, not just the result of the two photos being taken from slightly different spots. SIFT does the "lining up the landmarks" step; gray-level slicing then helps you cleanly isolate exactly the shiny-panel material you actually care about comparing.
</details>

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 6. Mini Project — Build a Mini Marr Pipeline

> **Open-ended, no single "correct" answer — this is meant to be attempted hands-on (e.g., in Python with OpenCV/NumPy), not just discussed.**

**The Task:** Take any single photograph (a simple object on a plain background works best, e.g., a ball or a mug) and manually reconstruct all four stages of Marr's Computational Vision model as separate output images:

```
+------------------------------------------------------------+
| Stage 1: Input Image                                       |
+------------------------------------------------------------+
| -> load your photo, convert it to                          |
| grayscale                                                  |
+------------------------------------------------------------+
                             |
                             v
+------------------------------------------------------------+
| Stage 2: Primal Sketch                                     |
+------------------------------------------------------------+
| -> run an edge detector (Canny or                          |
| Sobel) to approximate "zero crossings,                     |
| blobs, edges"                                              |
+------------------------------------------------------------+
                             |
                             v
+------------------------------------------------------------+
| Stage 3: 2-1/2-D Sketch                                    |
+------------------------------------------------------------+
| -> use a simple shading-based depth cue                    |
| (brighter = closer), OR a stereo pair                      |
| if you have two photos from slightly                       |
| different angles -- produce a ROUGH                        |
| depth/orientation map (this stage is                       |
| intentionally the hardest and most                         |
| open-ended; a coarse approximation is                      |
| perfectly fine)                                            |
+------------------------------------------------------------+
                             |
                             v
+------------------------------------------------------------+
| Stage 4: 3-D Model Representation                          |
+------------------------------------------------------------+
| -> write a short paragraph describing                      |
| what a full, object-centered 3D model                      |
| would need to capture that your                            |
| 2-1/2-D sketch cannot (e.g. "the back                      |
| and underside of the mug, which no                         |
| single photo shows")                                       |
+------------------------------------------------------------+
```

**Stretch goals (optional, but genuinely worthwhile for building intuition):**
- Also run a SIFT keypoint detector (`cv2.SIFT_create()` in OpenCV) on your image and overlay the detected keypoints — compare where SIFT's keypoints land versus where your Primal Sketch's edges are. They should broadly correlate, since SIFT keypoints tend to sit at corners/blobs, which are exactly the kind of features the Primal Sketch is meant to capture.
- Compute the brightness `B(I)` and contrast of your grayscale image using the exact formulas from the Numerical file Sections 1–2, and verify your hand computation against `numpy.mean()`.
- Apply the log transform (Numerical Section 4) to your image and visually compare before/after — does it visibly help if your photo has strong shadows or overexposed regions?
- Apply both a low-pass (average) filter and a median filter to a deliberately noised version of your photo (add synthetic salt-and-pepper noise using `numpy.random`), and visually confirm which filter better preserves edges, matching the numeric evidence from Numerical Section 6b.
- If you have access to any short video clip, compute optical flow between two consecutive frames (`cv2.calcOpticalFlowFarneback`) and visualize the resulting motion field as a color-coded image, matching the description in Theory Section 9.

**Deliverable suggestion:** a single notebook or folder with 4 output images (one per Marr stage) plus a short written reflection (150–200 words) on where your pipeline's approximations diverge from the "real" theoretical stage (e.g., "my Primal Sketch used Canny edges, but Marr's original formulation also includes 'bars, ends, virtual lines, groups,' which a simple edge detector doesn't explicitly separate out"). Treat this reflection paragraph as valuable exam preparation in its own right — the act of noticing exactly where a simplified implementation diverges from the full theoretical definition is precisely the kind of nuanced understanding exam questions tend to probe for.

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 7. Exam-Style Rapid-Fire Q&A Bank

> One-line-answer format — good for a final pre-exam speed-run. Cover the answer column and go down the list, checking yourself honestly on each one before revealing.

| # | Question | Answer |
|---|---|---|
| 1 | What are Marr's 4 stages, in order? | Input Image -> Primal Sketch -> 2½-D Sketch -> 3-D Model |
| 2 | Is the 2½-D Sketch viewer-centered or object-centered? | Viewer-centered |
| 3 | Is the 3-D Model viewer-centered or object-centered? | Object-centered |
| 4 | What two properties make SIFT scale/rotation invariant? | Scale-space representation + assigned keypoint orientation |
| 5 | What does HOG compute per pixel? | Gradient magnitude and gradient orientation |
| 6 | Name 3 failure cases of HOG. | Deformable objects, smooth/edge-less surfaces, occlusion/clutter |
| 7 | What does Structure from Motion use to compute 3D position? | Matched features across multiple photos + triangulation |
| 8 | Semantic segmentation output: one color per ___? | Class |
| 9 | Instance segmentation output: one color per ___? | Object instance (can count objects) |
| 10 | Which ImageNet-era model triggered the "2012 AlexNet moment"? | SuperVision / AlexNet (Krizhevsky et al., NIPS 2012) |
| 11 | ILSVRC image classification: how many classes / images? | 1,000 classes / 1,431,167 images |
| 12 | Full ImageNet dataset: how many categories / images? | 22,000 categories / 15,000,000 images |
| 13 | Is CLIP discriminative or generative? | Discriminative (matches text with existing images) |
| 14 | Is DALL-E discriminative or generative? | Generative (creates new images from text) |
| 15 | RGB is additive or subtractive? | Additive (adds light -> white) |
| 16 | CMYK is additive or subtractive? | Subtractive (adds ink -> absorbs light -> black) |
| 17 | List the 5 JPEG pipeline steps in order. | Color Transform -> Downsampling -> DCT -> Quantization -> Encoding |
| 18 | Why RGB->YCbCr before downsampling? | Eyes are more sensitive to brightness than color, so color can be downsampled more aggressively |
| 19 | Which frame type is self-contained (no dependency)? | I-frame (Intra/keyframe) |
| 20 | Which frame type depends only on the past? | P-frame (Predicted) |
| 21 | Which frame type depends on both past and future? | B-frame (Bi-predictive) |
| 22 | Brightness formula? | B(I) = (1/wh) times sum of I(u,v) |
| 23 | Contrast formula (as given)? | Change in Luminance / Average Luminance |
| 24 | What does a histogram show? | Distribution of pixel intensities (statistical only, no location info) |
| 25 | Narrow central histogram peak = ? | Low contrast |
| 26 | Log transform formula? | s = c times log(1+|r|) |
| 27 | Log transform best suited for? | High-dynamic-range images (astronomical, medical, satellite) |
| 28 | Gray-level slicing: what happens outside [A,B]? | Set to a constant (often 0/dark) |
| 29 | Low-pass filter kernel weights? | 1/9 times all-ones 3x3 |
| 30 | Which filter best removes salt-and-pepper noise? | Median filter |
| 31 | Sharpening filter center weight (in the 1/9 kernel)? | 8 (surrounded by -1s) |
| 32 | Optical flow magnitude formula? | sqrt(u^2 + v^2) |
| 33 | What does HOF capture that HOG does not? | Motion direction (not static appearance) |
| 34 | What does MBH compute the gradient of? | The optical flow field itself (motion boundaries) |
| 35 | Classical action recognition pipeline, in order? | Interest points -> patches -> HOG/HOF/MBH descriptors -> Bag-of-Visual-Words histogram -> SVM |
| 36 | Why is CV called interdisciplinary? | It draws on biology, physics, psychology, math, CS, and engineering together |
| 37 | Does AI contain ML, or does ML contain AI? | AI contains ML (ML is a subset of AI, not the reverse) |
| 38 | Who wrote the original Computational Vision book, and when? | David Marr, "Vision," 1970s (published 1982) |
| 39 | What did Kanwisher's 1997 fMRI study find? | A specific brain region (FFA) responds much more strongly to faces than to houses |
| 40 | What's the key trade-off of relying heavily on B-frames? | Better compression, but added decoding delay and higher memory use |

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

## 8. Full-File Summary

This practice file turns every idea from the Theory and Numerical files into something you have to actually retrieve from memory, rather than just recognize while reading. Section 1 reproduces the instructor's own nine live in-class discussion questions in full, each with a deeper worked-through explanation and a fresh analogy on top of the original answer, covering the full sweep of the lecture: why vision needs multiple processing stages, why Marr's depth stage is "2½-D" rather than fully 3D, what makes a keypoint genuinely good for matching, why SIFT survives rotation and scaling, where HOG breaks down, how Structure from Motion recovers 3D shape from flat photos, how semantic and instance segmentation differ, why B-frames are a poor fit for real-time video, and why two people walking in opposite directions produce different HOF signatures despite performing the same action.

Sections 2 through 4 drill the theory file's core concepts directly — Marr's four-stage pipeline, the AI/ML/DL/CNN nesting, SIFT versus HOG trade-offs, and the full suite of image/video enhancement techniques (log transform, gray-level slicing, median versus low-pass filtering, optical flow) — each with a fresh explanatory analogy layered on top of the original slide content. Section 5 pushes further into genuinely applied, real-world scenarios (museum painting recognition, satellite truck counting, satellite image transmission, sports action classification, drone rooftop comparison) specifically designed to test whether the underlying concepts can be transferred to situations the original lecture slides never explicitly covered — this is deliberately the closest simulation of what a real viva or open-ended exam question would demand. Section 6 offers a genuinely hands-on mini project, walking through building an actual, working four-stage Marr pipeline in code, with several optional stretch goals connecting directly back to the exact formulas worked out in the Numerical file. Finally, Section 7's forty-question rapid-fire bank is built for a fast, honest final self-check in the last day or two before any quiz or exam — if every single one of these forty answers comes to mind instantly and correctly, Lecture 1 is genuinely exam-ready.

[Back to Top](#-cv-nlp--lec-01-introduction--practice)

---

> **You've completed Lec 01's full trio.** Loop back to [📖 THEORY](cvnlp_lec01_intro_theory.md)'s Cheat Sheet the night before any quiz, and redo [🧮 NUMERICAL](cvnlp_lec01_intro_numerical.md) cold (no notes) as your final self-check.
>
> *CV-NLP · Lec 01 · github.com/rpaut03l/TS-02-03*
