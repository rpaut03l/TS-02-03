# 🧮 CV-NLP — Lec 01: Introduction — NUMERICAL

### *Every formula from Lec 01, worked out digit by digit, with the reasoning behind every step*

> **Nav:** [← Lec 01 README](README.md) | [📖 THEORY](cvnlp_lec01_intro_theory.md) | **NUMERICAL** | [🎯 PRACTICE](cvnlp_lec01_intro_practice.md) | [Track Home ➡️](../README.md)

---

## 🧠 MASTER MNEMONIC: "B-C-H-L-G-S-M-J-I"

> **B**rightness, **C**ontrast, **H**istogram, **L**og transform, **G**ray slicing, **S**patial kernels, **M**otion magnitude, **J**PEG block math, **I**mageNet error math — worked in exactly that order below.

Before diving in, picture a kitchen scale. Every formula in this file is really just a kitchen scale in disguise: you put some numbers in one side (pixel values, motion vectors, error rates), you follow a fixed recipe (add, divide, square, take a square root, take a log), and a single trustworthy number comes out the other side. None of the arithmetic below requires anything beyond addition, multiplication, division, square roots, and logarithms — the goal of this file is to make sure you can perform every single one of these "kitchen scale" recipes completely from memory, with pen and paper, no calculator required for the easy ones.

---

## 📚 Table of Contents

| # | Worked Example | Jump |
|---|---|---|
| 1 | Brightness B(I) on a Tiny 4x4 Image | [Section 1](#1-brightness-bi-on-a-tiny-4x4-image) |
| 2 | Contrast Ratio Calculation | [Section 2](#2-contrast-ratio-calculation) |
| 3 | Histogram Construction by Hand | [Section 3](#3-histogram-construction-by-hand) |
| 4 | Log Transform Pixel by Pixel | [Section 4](#4-log-transform-pixel-by-pixel) |
| 5 | Gray-Level Slicing on a Row of Pixels | [Section 5](#5-gray-level-slicing-on-a-row-of-pixels) |
| 6 | Spatial Filtering — Low-Pass, Median, Sharpen | [Section 6](#6-spatial-filtering--low-pass-median-sharpen) |
| 7 | Optical Flow Magnitude | [Section 7](#7-optical-flow-magnitude) |
| 8 | JPEG 8x8 Block & Downsampling Arithmetic | [Section 8](#8-jpeg-8x8-block--downsampling-arithmetic) |
| 9 | ImageNet Top-1 Error-Rate Reduction Math | [Section 9](#9-imagenet-top-1-error-rate-reduction-math) |
| 10 | Cheat Sheet & Exam Hacks | [Section 10](#10-cheat-sheet--exam-hacks) |
| 11 | Full-File Summary | [Section 11](#11-full-file-summary) |

---

## 1. Brightness B(I) on a Tiny 4x4 Image

### The Idea, Told Simply

Imagine a tiny 4x4 grayscale photograph — nothing but sixteen little gray squares laid out in a grid. To answer the question "how bright is this whole photo, on average," you don't need anything clever: you literally add up every single square's brightness number, and then divide that total by however many squares there were. That's the entire brightness formula, dressed up in mathematical notation.

### The Formula

```
                  1    h   w
        B(I)  =  ---  E   E   I(u, v)
                  wh  v=1 u=1
```

Read this exactly as: sum every pixel intensity `I(u,v)` across the whole image (the double sigma symbol just means "loop over every row, and inside that, loop over every column"), and then multiply the result by `1` over `wh` (which is just another way of writing "divide by the total pixel count").

### Worked Example

Take this 4x4 grayscale image, with values from 0 (black) to 255 (white):

```
+-----+-----+-----+-----+
| 20  | 40  | 60  | 80  |   <- Row 1
+-----+-----+-----+-----+
| 100 | 120 | 140 | 160 |   <- Row 2
+-----+-----+-----+-----+
| 180 | 200 | 220 | 240 |   <- Row 3
+-----+-----+-----+-----+
| 10  | 30  | 50  | 70  |   <- Row 4
+-----+-----+-----+-----+
```

Here `w = 4` (four columns) and `h = 4` (four rows), so the total pixel count `wh = 16`.

**Step 1 — Sum every pixel intensity, one row at a time (this is exactly how you'd do it by hand, and it keeps the arithmetic manageable).**

```
Row 1 sum = 20 + 40 + 60 + 80      = 200
Row 2 sum = 100 + 120 + 140 + 160  = 520
Row 3 sum = 180 + 200 + 220 + 240  = 840
Row 4 sum = 10 + 30 + 50 + 70      = 160
```

Now add the four row sums together, two at a time, so nothing gets lost:

```
200 + 520 = 720
720 + 840 = 1560
1560 + 160 = 1720
```

So the grand total, `Sum of I(u,v)` across the whole image, is `1720`.

**Step 2 — Divide that grand total by the total pixel count, `wh = 16`.**

```
B(I) = 1720 / 16
B(I) = 107.5
```

**Answer: B(I) = 107.5.**

### What This Number Actually Means

Picture the full brightness scale as a see-saw with pure black (0) sitting on the far left end and pure white (255) sitting on the far right end. The exact middle of that see-saw would be `255 / 2 = 127.5`. Our computed brightness of `107.5` sits a little to the LEFT of that exact middle point — meaning this particular 4x4 image leans very slightly darker than a perfectly balanced mid-gray image, even though it contains plenty of bright pixels (up to 240) mixed in with dark ones (as low as 10).

```
  0            107.5  127.5            255
  |--------------|------|--------------|
 BLACK        our B(I)  mid-gray      WHITE
```

### Summary

Brightness is nothing more than a plain average — add up every pixel's intensity value, and divide by how many pixels there are. The two most common mistakes are forgetting the division step entirely (reporting the raw sum, 1720, as if it were the brightness) and dividing by only the width or only the height instead of the full pixel count (`w times h`). Always show both the sum and the division explicitly in an exam answer, exactly as done above.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 2. Contrast Ratio Calculation

### The Idea, Told Simply

If brightness tells you the "average mood" of a photo, contrast tells you how much that mood *swings* around the average — a photo where every pixel is nearly identical has almost no contrast (a flat, boring mood), while a photo with both very dark shadows and very bright highlights has high contrast (a dramatic, swingy mood).

### The Formula

```
                    Change in Luminance
   Contrast  =    -------------------------
                     Average Luminance
```

### Worked Example

Using the exact same 4x4 image from Section 1, we already know the **Average Luminance** is `B(I) = 107.5`.

**Step 1 — Find the "Change in Luminance."** One simple, commonly used way to measure this is `maximum pixel value minus minimum pixel value` across the whole image — in other words, the full spread from the single darkest pixel to the single brightest pixel.

```
Scanning the whole 4x4 grid for the largest value:
  Max = 240   (found in Row 3, last column)

Scanning the whole 4x4 grid for the smallest value:
  Min = 10    (found in Row 4, first column)

Change in Luminance = Max - Min = 240 - 10 = 230
```

**Step 2 — Divide that spread by the Average Luminance.**

```
Contrast = 230 / 107.5
Contrast = 2.1395...
Contrast ~= 2.14
```

**Answer: Contrast is approximately 2.14.**

### What This Number Actually Means

Since the total spread (230) is more than double the average brightness (107.5), this tells us the image's pixel values are widely scattered relative to their own average, rather than tightly bunched close to it — a genuine hallmark of a fairly high-contrast image.

```
+------------------------------------------------------------+
| QUICK CONTRAST-VALUE INTUITION                             |
+------------------------------------------------------------+
| Contrast well below 1  -> pixel values cluster TIGHTLY     |
|   around                                                   |
| the average (LOW contrast, flat look)                      |
| Contrast around 1      -> spread is roughly equal to the   |
| average (moderate contrast)                                |
| Contrast well above 1  -> pixel values are widely          |
|   scattered                                                |
| relative to the average (HIGH contrast)                    |
+------------------------------------------------------------+```

### A Note on Definitions

The lecture is explicit that many different precise mathematical formulas for "contrast" exist across the wider imaging literature (RMS contrast, Michelson contrast, Weber contrast, and others each have slightly different formulas and use-cases). The ratio form shown above is the intuitive version taught in this specific lecture. If an exam question simply asks you to "define contrast" without specifying which formula, it's safest to explicitly state which definition you're using before computing anything — graders generally accept any of the standard definitions as long as the underlying concept (spread of brightness values, relative to the average) is correctly explained and consistently applied.

### Summary

Contrast measures the spread of brightness values relative to the image's own average brightness — computed here as the maximum-minus-minimum spread divided by the average luminance. A contrast value well above 1 signals a dramatic, high-contrast image; a value well below 1 signals a flat, low-contrast image. Always double check whether a given exam question wants the simple ratio form shown here, or a different, explicitly named formula (RMS, Michelson, Weber) — and if unspecified, state your chosen definition clearly before computing.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 3. Histogram Construction by Hand

### The Idea, Told Simply

Building a histogram by hand is exactly like sorting a big pile of mixed candy into separate bowls by color, and then simply counting how many pieces of candy ended up in each bowl. Nothing more complicated than counting is actually involved — the only "hard" part is being careful and not accidentally missing or double-counting a pixel.

### Worked Example

Take a small 4x4 image, deliberately using only a few distinct intensity values so the counting stays easy to follow:

```
+----+----+----+----+
| 2  | 2  | 5  | 7  |   <- Row 1
+----+----+----+----+
| 2  | 5  | 5  | 9  |   <- Row 2
+----+----+----+----+
| 7  | 9  | 9  | 9  |   <- Row 3
+----+----+----+----+
| 2  | 5  | 7  | 9  |   <- Row 4
+----+----+----+----+
```

**Step 1 — List every distinct intensity value that appears anywhere in the grid.** Scanning carefully through all sixteen cells: the only values that ever appear are 2, 5, 7, and 9.

**Step 2 — Tally exactly how many times each value appears, one value at a time, so you never lose track.**

```
Value 2: at (R1,C1), (R1,C2), (R2,C1),
         (R4,C1) -> count = 4

Value 5: at (R1,C3), (R2,C2), (R2,C3),
         (R4,C2) -> count = 4

Value 7: at (R1,C4), (R3,C1), (R4,C3)
         -> count = 3

Value 9: at (R2,C4), (R3,C2), (R3,C3),
         (R3,C4), (R4,C4) -> count = 5
```

**Step 3 — Sanity-check by adding all four counts; the total must exactly equal the total pixel count (16), or a counting mistake was made somewhere.**

```
4 + 4 + 3 + 5 = 16
CHECK: matches wh = 4 x 4 = 16,
so the count is correct
```

**Step 4 — Write out the finished histogram table, `h(i)`.**

```
i     :   2     5     7     9
h(i)  :   4     4     3     5
```

```
h(i)
 5 |                          #
 4 |    #           #
 3 |    #           #           #
 2 |    #           #           #     #
 1 |    #           #           #     #
 0 +----+-----------+-----------+-----+----> i (intensity)
        2           5           7     9
```

**Interpretation:** This histogram is deliberately spread across only 4 separate, narrow spikes rather than a smooth continuous spread — a clear sign of a low-detail, limited dynamic range image, since only 4 distinct gray levels are being used out of a possible 256. This is precisely the kind of image histogram equalization (covered in the theory file's Image Processing section) would want to spread out across the full available range.

### Summary

Constructing a histogram by hand is simply careful counting: identify every distinct pixel value present, tally how many times each one occurs, and always verify your counts sum to the total pixel count before trusting your result. Histograms deliberately throw away all spatial position information — they only tell you *how many* pixels share each brightness level, never *where* in the image those pixels are located, which is exactly why two visually very different images can sometimes still share an identical histogram.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 4. Log Transform Pixel by Pixel

### The Idea, Told Simply

Imagine a very tall, unevenly spaced staircase where the first few steps are close together but the steps near the top are absurdly far apart — a satellite photo's pixel values often behave exactly like this staircase, with a handful of blindingly bright pixels sitting enormously far above the bulk of ordinary, modest-brightness pixels. The log transform's entire job is to squash that unevenly spaced staircase down into a normal, evenly climbable one, so a regular 8-bit screen (which can only display values from 0 to 255) can actually show the whole picture without the bright outliers wrecking everything else.

### The Formula

```
s = c . log(1 + |r|)
```

### Worked Example

Suppose we have a high-dynamic-range astronomical sensor image with pixel values ranging enormously wide — say `r = 0, 15, 255, 4095` (imagine a 12-bit sensor capable of going all the way up to 4095). We want to compress this down into a normal, displayable 8-bit range (0 to 255).

**Step 1 — Choose the scaling constant `c`.** The standard, simple trick: pick `c` so that the *maximum* input value maps as close as possible to the *maximum* output value, 255. Using the natural logarithm (ln):

```
c = 255 / log(1 + r_max)
c = 255 / log(1 + 4095)
c = 255 / log(4096)
```

Computing `log(4096)` using the natural log gives `ln(4096) ~= 8.317`.

```
c = 255 / 8.317
c ~= 30.66
```

**Step 2 — Apply the transform to each pixel value, one at a time.**

For `r = 0`:
```
s = c . log(1 + 0)
s = c . log(1)
s = c . 0          (log of 1 is always exactly 0, in any base)
s = 0
```

For `r = 15`:
```
s = 30.66 . log(1 + 15)
s = 30.66 . log(16)
log(16) = ln(16) ~= 2.773
s = 30.66 x 2.773
s ~= 85.04
```

For `r = 255`:
```
s = 30.66 . log(1 + 255)
s = 30.66 . log(256)
log(256) = ln(256) ~= 5.545
s = 30.66 x 5.545
s ~= 170.03
```

For `r = 4095`:
```
s = 30.66 . log(1 + 4095)
s = 30.66 . log(4096)
log(4096) ~= 8.317
s = 30.66 x 8.317
s ~= 255.03
(rounds to 255, our chosen max --
exactly as designed in Step 1)
```

**Step 3 — Line up the input/output pairs and compare the compression effect directly.**

```
+---------+------+------+-------+-------+
| Input r |   0  |  15  |  255  |  4095 |
+---------+------+------+-------+-------+
| Output s|   0  | 85.0 | 170.0 | 255.0 |
+---------+------+------+-------+-------+
```

**The key observation, which is the entire point of this technique:** the input range from 0 up to 255 (a jump of 255) produced an output jump of only about 170 (from 0 up to 170). But the much, MUCH bigger input range from 255 up to 4095 (a jump of a whopping 3840!) only produced a further output jump of about 85 (from 170 up to 255). **The log function compressed that enormous 3840-wide jump into a smaller output range than it used for the far smaller 255-wide jump** — exactly matching the theory statement that "the log function compresses large values more than smaller ones."

```
Input axis (linear, un-squashed):
  0 ----- 255 ------------------- 4095
  |--255--|---------3840---------|

Output axis (after log compression):
  0 ---- 170 -- 255
  |-170--|--85--|
```

Notice how the huge 3840-wide gap on the input side got squeezed down into a much smaller 85-wide gap on the output side, while the modest 255-wide gap on the input side kept a comparatively generous 170-wide gap on the output side. That asymmetric squeezing is precisely why the log transform is the tool of choice whenever a handful of extreme, blindingly bright pixels would otherwise wash out all the useful, ordinary mid-tone detail.

### Summary

The log transform, `s = c . log(1+|r|)`, compresses a very wide, unevenly distributed range of pixel intensities down into a smaller, evenly displayable range, by squashing large input values proportionally far more than small input values. The scaling constant `c` is typically chosen so the largest expected input value maps close to the largest allowed output value (255, for an 8-bit image). This makes the log transform the natural choice for astronomical, medical, or satellite images, where a small number of extreme-brightness pixels would otherwise dominate and hide all the useful detail sitting in the ordinary mid-range.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 5. Gray-Level Slicing on a Row of Pixels

### The Idea, Told Simply

Gray-level slicing works exactly like a bouncer standing at a club entrance with a strict, fixed guest list: only pixel values that fall inside the chosen range `[A, B]` are allowed to shine through at full brightness, and every other pixel value gets turned away and pushed down to a flat, boring constant (usually black).

### Worked Example

Take one row of ten pixels:

```
r: 10  50  90  130  150  170  190  210  230  250
```

We want to highlight the range `[A, B] = [120, 200]` — pushing pixels inside this range up to full brightness (`255`), and pushing everything else down to a flat `0`, exactly matching the technique description in the theory file.

**Step 1 — Check every single pixel against the range `[120, 200]`, one at a time, carefully.**

```
r = 10  -> 10 < 120     -> OUTSIDE -> s = 0
r = 50  -> 50 < 120     -> OUTSIDE -> s = 0
r = 90  -> 90 < 120     -> OUTSIDE -> s = 0
r = 130 -> 120<=130<=200 -> INSIDE -> s = 255
r = 150 -> 120<=150<=200 -> INSIDE -> s = 255
r = 170 -> 120<=170<=200 -> INSIDE -> s = 255
r = 190 -> 120<=190<=200 -> INSIDE -> s = 255
r = 210 -> 210 > 200    -> OUTSIDE -> s = 0
r = 230 -> 230 > 200    -> OUTSIDE -> s = 0
r = 250 -> 250 > 200    -> OUTSIDE -> s = 0
```

**Step 2 — Write out the finished, sliced output row.**

```
s:  0   0   0  255  255  255  255   0   0   0
```

```
Input  (r): 10  50  90 130 150 170 190 210 230 250
                       |------INSIDE------|
Output (s):  0   0   0 255 255 255 255  0   0   0
                       |--LIT UP WHITE--|--|
```

**Interpretation:** exactly the four pixels that fell inside `[120, 200]` "light up" and become pure white (255), while every single pixel outside that range goes fully dark (0) — this is gray-level slicing doing precisely its job, isolating and highlighting one specific tonal band while suppressing everything else.

### Summary

Gray-level slicing is a simple range check applied to every pixel: if a pixel's value falls inside the chosen `[A, B]` window, boost it to full brightness (or otherwise enhance it); if it falls outside that window, flatten it to a constant value. This is a useful way to isolate a single feature of interest — such as one specific tonal band representing a road marking, a tumor, or a particular terrain type — while suppressing every other unrelated brightness level in the image.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 6. Spatial Filtering — Low-Pass, Median, Sharpen

### The Idea, Told Simply

Spatial filtering means sliding a small window (usually 3x3 pixels) across an image, and at every position, replacing the center pixel with some new value computed from that little neighborhood. Different filters use different recipes for computing that new value — averaging (low-pass), sorting and picking the middle (median), or deliberately exaggerating the difference between the center and its surroundings (sharpen).

### Setup — A Shared 3x3 Neighborhood

Take this 3x3 patch of an image (the pixel being filtered is the exact **center**, with value 50):

```
+----+----+----+
| 10 | 20 | 30 |
+----+----+----+
| 40 | 50 | 60 |
+----+----+----+
| 70 | 80 | 90 |
+----+----+----+
```

### 6a. Low-Pass Filter (1/9 Averaging Kernel)

```
Kernel:
     +----+----+----+
     |  1 |  1 |  1 |
 1/9 +----+----+----+
     |  1 |  1 |  1 |
     +----+----+----+
     |  1 |  1 |  1 |
     +----+----+----+
```

**Step 1 — Since every single kernel weight equals 1/9, this filter is really just a plain average. Sum all nine patch values together.**

```
10 + 20 = 30
30 + 30 = 60
60 + 40 = 100
100 + 50 = 150
150 + 60 = 210
210 + 70 = 280
280 + 80 = 360
360 + 90 = 450
```

So the running total across all nine pixels is `450`.

**Step 2 — Divide that total by 9 (the number of pixels in the neighborhood).**

```
Output = 450 / 9 = 50
```

**Result: the new center value is 50.** In this perfectly symmetric example, the plain average happens to land exactly on the original center value — but in a real, noisy photograph, this averaging step is exactly what smooths out random bright or dark speckle noise, because a single stray outlier pixel gets diluted across all nine positions rather than standing out on its own.

### 6b. Median Filter

**Step 1 — Sort all nine patch values into ascending order.**

```
10,  20,  30,  40,  50,  60,  70,  80,  90
```

**Step 2 — Pick out the exact middle value (the 5th value out of 9, since 9 is odd).**

```
Position:  1    2    3    4    5    6    7    8    9
Value:    10   20   30   40   50   60   70   80   90
                              ^
                        this is the MIDDLE (5th) value
```

**Result: the median is 50.** Now imagine a piece of salt-and-pepper noise corrupted one pixel — replace the bottom-right corner value `90` with an extreme outlier, `255`:

```
+----+----+-----+
| 10 | 20 |  30 |
+----+----+-----+
| 40 | 50 |  60 |
+----+----+-----+
| 70 | 80 | 255 |
+----+----+-----+
```

Sort these nine values again: `10, 20, 30, 40, 50, 60, 70, 80, 255`. The middle (5th) value is **still exactly 50** — the extreme outlier landed at the very end of the sorted list and had zero effect on the middle position.

**Now compare this against what the low-pass average filter would have produced on this same noisy patch:**

```
Sum = 10+20+30+40+50+60+70+80+255
Sum = 615
Average = 615 / 9
Average ~= 68.3
(visibly dragged upward by that outlier!)
```

```
+------------------------------------------------------------+
| SAME NOISY PATCH, TWO DIFFERENT FILTERS                    |
+------------------------------------------------------------+
| Low-pass (average) output:  ~68.3   <- dragged toward the  |
| outlier                                                    |
| Median filter output:        50     <- completely          |
|   UNAFFECTED                                               |
| by the outlier                                             |
+------------------------------------------------------------+```

This numeric comparison is exactly the theory statement made concrete: *"unlike average, median doesn't get affected by extreme values, so it preserves edges better."* The median filter output stayed rock-solid at exactly 50 despite the huge outlier, while the plain average filter jumped noticeably up to about 68.3.

### 6c. Sharpening Filter (1/9 Kernel with Center Weight 8)

```
Kernel:
     +----+----+----+
     | -1 | -1 | -1 |
 1/9 +----+----+----+
     | -1 |  8 | -1 |
     +----+----+----+
     | -1 | -1 | -1 |
     +----+----+----+
```

Using the original (non-noisy) 3x3 patch:

```
+----+----+----+
| 10 | 20 | 30 |
+----+----+----+
| 40 | 50 | 60 |
+----+----+----+
| 70 | 80 | 90 |
+----+----+----+
```

**Step 1 — Multiply every single pixel by its matching kernel weight, position by position, very carefully.**

```
Top-left:      10 x (-1) = -10
Top-middle:    20 x (-1) = -20
Top-right:     30 x (-1) = -30
Middle-left:   40 x (-1) = -40
Center:        50 x  8   = 400
Middle-right:  60 x (-1) = -60
Bottom-left:   70 x (-1) = -70
Bottom-mid:    80 x (-1) = -80
Bottom-right:  90 x (-1) = -90
```

**Step 2 — Sum all nine of these weighted values together.**

```
Sum = -10-20-30-40+400-60-70-80-90

Group the negative terms first
(everything except the center):
  -10-20-30-40-60-70-80-90 = -400

Now add the center term back in:
  -400 + 400 = 0
```

**Step 3 — Divide by 9.**

```
Output = 0 / 9 = 0
```

**Result: the sharpened output is exactly 0** in this perfectly uniform-gradient patch. This makes complete sense once you understand what the sharpening kernel is actually measuring: it compares the center pixel against the *average* of its eight surroundings, and reports how different the center is from that average. Since this particular patch is a smooth, evenly-increasing gradient with absolutely no sudden edge or anomaly, the center pixel (50) really is exactly the average of its surroundings, so the sharpening filter correctly reports "nothing unusual here" with an output of 0.

**Now imagine instead that the center pixel had been a sudden spike — say, the center was 200 instead of 50, with the same eight surrounding values.**

```
Center weighted term:  200 x 8 = 1600
Surrounding terms (unchanged): still sum to -400

Sum = -400 + 1600 = 1200
Output = 1200 / 9 ~= 133.3
```

**This large positive output (about 133.3, versus 0 before) is exactly how a sharpening filter flags a genuine local edge or anomaly** — a big departure from zero signals "something unusual is happening right here," which is precisely the local-contrast-boosting behavior sharpening filters are designed to highlight.

### Summary

Spatial filtering slides a small 3x3 window across an image and replaces each center pixel using a fixed recipe applied to its neighborhood. The low-pass filter is a plain average (sum of nine values divided by 9), which smooths noise but can be dragged toward extreme outlier values. The median filter sorts the nine values and picks the middle one, making it immune to a single outlier — this is precisely why it's the right tool against salt-and-pepper noise. The sharpening filter compares the center pixel against its surroundings (via the 8-in-the-center, minus-1-around-it kernel), producing near-zero output on smooth regions and a strong positive or negative output wherever a genuine local edge or anomaly exists.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 7. Optical Flow Magnitude

### The Idea, Told Simply

Optical flow magnitude is nothing more than the Pythagorean theorem from basic geometry, borrowed and applied directly to motion. If a pixel moves partly sideways and partly up-or-down between two frames, its overall "as the crow flies" speed is simply the hypotenuse of the right triangle formed by those two separate movements.

### The Formula

```
Magnitude = sqrt(u^2 + v^2)
```

### Worked Example 1 — A Clean, Simple Case

A pixel moves `u = 3` pixels horizontally and `v = 4` pixels vertically between two consecutive frames.

**Step 1 — Square each component separately.**
```
u^2 = 3^2 = 9
v^2 = 4^2 = 16
```

**Step 2 — Add the two squared components together.**
```
u^2 + v^2 = 9 + 16 = 25
```

**Step 3 — Take the square root of that sum.**
```
Magnitude = sqrt(25) = 5
```

**Answer: Magnitude = 5 pixels per frame.** Notice this is a classic 3-4-5 right triangle, deliberately chosen so the arithmetic is easy to verify by hand — exactly the same 3-4-5 triangle used in countless carpentry and construction "square-corner" tricks.

```
             v = 4
        +----------------+
        |              / |
        |             /  |
        |    hyp =   /   |
        |    5       /   |
        |           /    |
        |          /     |
        |         /      |
        +----------------+
          u = 3
```

### Worked Example 2 — A Case That Doesn't Resolve to a Whole Number

A pixel moves `u = 5`, `v = 7` between two frames — deliberately chosen to demonstrate the general procedure when the answer isn't a clean whole number.

**Step 1 — Square each component.**
```
u^2 = 5^2 = 25
v^2 = 7^2 = 49
```

**Step 2 — Add them together.**
```
u^2 + v^2 = 25 + 49 = 74
```

**Step 3 — Take the square root.** Since `8^2 = 64` and `9^2 = 81`, we know `sqrt(74)` must sit somewhere between 8 and 9. Testing more precisely:
```
8.6^2 = 73.96     (very close to 74!)
```
```
Magnitude ~= 8.60 pixels per frame
```

### Worked Example 3 — Comparing Two People Walking in Opposite Directions

This example directly connects the optical flow magnitude formula back to the theory file's HOF discussion. Person A walks to the right: `(u, v) = (5, 0)`. Person B walks to the left: `(u, v) = (-5, 0)`.

**Magnitude for Person A:**
```
Magnitude_A = sqrt(5^2 + 0^2) = sqrt(25 + 0) = sqrt(25) = 5
```

**Magnitude for Person B:**
```
Magnitude_B = sqrt((-5)^2 + 0^2) = sqrt(25 + 0) = sqrt(25) = 5
```

```
+------------------------------------------------------------+
| MAGNITUDE ALONE CANNOT TELL THESE TWO MOTIONS APART        |
+------------------------------------------------------------+
| Person A: (u,v) = (5, 0)   -> Magnitude = 5                |
| Person B: (u,v) = (-5, 0)  -> Magnitude = 5   <-           |
|   IDENTICAL!                                               |
| Squaring a negative number always makes it positive, so    |
|   the                                                      |
| sign (direction) information is completely LOST once you   |
|   compute                                                  |
| the magnitude alone.                                       |
+------------------------------------------------------------+```

**The key numeric insight:** the magnitude (speed) is exactly identical for both people (5, since squaring a negative number removes its sign entirely) — but the *direction* they're walking is genuinely opposite (0 degrees versus 180 degrees). This directly explains why the theory file states that HOF descriptors differ for opposite-direction walking: raw magnitude alone genuinely cannot distinguish the two motions, but the direction-binned histogram (HOF) absolutely can, because it separately tracks *which way* each flow vector points, not merely *how fast* it moves.

### Summary

Optical flow magnitude is a direct application of the Pythagorean theorem to a 2D motion vector `(u, v)`: square both components, add them together, and take the square root. This magnitude tells you how fast a pixel is moving, but nothing about which direction it's moving in — as demonstrated numerically above, two motions in exactly opposite directions can produce an identical magnitude, which is precisely why direction-sensitive descriptors like HOF are needed on top of raw optical flow magnitude for tasks like distinguishing "walking left" from "walking right."

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 8. JPEG 8x8 Block & Downsampling Arithmetic

### The Idea, Told Simply

JPEG never processes a whole image in one giant single step — it chops the image up into a grid of small, manageable 8x8-pixel tiles, and runs its DCT/quantization/encoding machinery separately on each tile. Figuring out exactly how many tiles a given image needs is simple division, done carefully in both directions (width and height).

### Worked Example — How Many DCT Blocks Does an Image Need?

Suppose we have a 640x480 image (a familiar, classic resolution), and JPEG processes it using 8x8-pixel blocks.

**Step 1 — Work out how many blocks fit across the width.**
```
640 / 8 = 80 blocks, laid out across the image horizontally
```

**Step 2 — Work out how many blocks fit down the height.**
```
480 / 8 = 60 blocks, laid out down the image vertically
```

**Step 3 — Multiply these two numbers together to get the total block count.**
```
Total blocks = 80 x 60 = 4800 blocks
```

**Answer: 4800 separate 8x8 DCT transforms** must be computed to JPEG-compress this single image, per color channel. (And, as you'll see in the next worked example, after chroma downsampling the two color channels typically need far fewer blocks than the luminance channel, since human eyes are much less sensitive to fine color detail than to fine brightness detail.)

```
+------------------------------------------------------------+
| 640 x 480 IMAGE, CHOPPED INTO AN 80 x 60 GRID OF 8x8       |
| BLOCKS                                                     |
+------------------------------------------------------------+
| [] [] [] [] [] [] [] [] ... (80 blocks across this row)    |
| [] [] [] [] [] [] [] [] ...                                |
| ... (60 rows of blocks, stacked vertically)                |
| Total = 80 x 60 = 4800 individual 8x8 tiles                |
+------------------------------------------------------------+
```

### Worked Example — Chroma Downsampling Savings (the 4:2:0 Scheme)

A very common JPEG and video downsampling scheme, called "4:2:0," deliberately keeps the luminance (Y) channel at full resolution, but samples the two chroma channels (Cb and Cr) at **half resolution in both directions** — half the width, and half the height.

**Step 1 — Work out the original chroma pixel count, assuming (for comparison) it started at the same full resolution as luminance.**
```
640 x 480 = 307,200 samples
(per chroma channel, if kept at
full resolution)
```

**Step 2 — Work out the new chroma pixel count after 4:2:0 downsampling (half width AND half height).**
```
New width  = 640 / 2 = 320
New height = 480 / 2 = 240
New chroma sample count =
  320 x 240 = 76,800 samples
  (per chroma channel)
```

**Step 3 — Compute exactly how much smaller the downsampled chroma data is, as a clean ratio.**
```
Reduction factor = 307,200 / 76,800 = 4
```

**Answer: exactly 4 times fewer chroma samples per channel.** Since a color image has two separate chroma channels (Cb and Cr), the *combined* chroma data drops from `2 x 307,200 = 614,400` samples all the way down to `2 x 76,800 = 153,600` samples.

```
+------------------------------------------------------------+
| CHROMA DATA, BEFORE AND AFTER 4:2:0 DOWNSAMPLING           |
+------------------------------------------------------------+
| Before (full res, both channels):  614,400 samples         |
| After  (4:2:0, both channels):     153,600 samples         |
| Savings:                           614,400 - 153,600 =     |
|   460,800                                                  |
| samples saved, PURELY from                                 |
| downsampling, before DCT,                                  |
| quantization, or entropy coding                            |
| have even begun                                            |
+------------------------------------------------------------+```

### Summary

Counting JPEG's 8x8 processing blocks is simple division: divide the image width by 8, divide the image height by 8, and multiply those two results together. The 4:2:0 chroma downsampling scheme halves both the width and the height of the two color channels, which multiplies together into a clean 4-times reduction in chroma sample count — a substantial bandwidth and storage saving that happens before any of the DCT, quantization, or entropy-coding machinery even begins its work, and it relies entirely on the same "eyes are less sensitive to color than brightness" principle covered in the theory file.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 9. ImageNet Top-1 Error-Rate Reduction Math

### The Idea, Told Simply

When someone says "AlexNet was a huge leap forward," that claim can and should be backed up with actual numbers — and it turns out there are two genuinely different, both mathematically valid, ways to measure "how big a leap" something was: the plain difference (absolute drop) and the difference relative to where you started (relative improvement). Confusing these two is one of the most common mistakes people make when discussing performance improvements in any field, not just Computer Vision.

### Worked Example — Quantifying the "AlexNet Moment"

Using the theory file's illustrative error-rate numbers: the 2011 error rate was approximately 0.26 (26%), and the 2012 error rate (AlexNet) was approximately 0.16 (16%).

**Step 1 — Compute the absolute drop in error, simply by subtracting.**
```
Absolute drop = 0.26 - 0.16 = 0.10
```
This `0.10` represents 10 **percentage points** of improvement — a plain, simple subtraction.

**Step 2 — Compute the relative (percentage) improvement, which measures the drop relative to where we started.**
```
Relative improvement = (Absolute drop / Old error) x 100
Relative improvement = (0.10 / 0.26) x 100
```

Let's carry out that division carefully:
```
0.10 / 0.26 = 0.3846...
```

```
Relative improvement = 0.3846... x 100
Relative improvement ~= 38.46%
```

**Answer: AlexNet cut the error rate by roughly 38.5% relative to the previous year's performance.** This "roughly 38.5% relative improvement" is the precise, numeric backing behind the theory file's description of this as "the huge jump."

```
+------------------------------------------------------------+
| ABSOLUTE DROP vs RELATIVE IMPROVEMENT -- NOT THE SAME      |
| NUMBER!                                                    |
+------------------------------------------------------------+
| Absolute drop:        0.26 - 0.16 = 0.10  (10 percentage   |
|   points)                                                  |
| Relative improvement: (0.10 / 0.26) x 100 ~= 38.46%        |
| These measure genuinely different things -- always state   |
|   which                                                    |
| one you are reporting, especially in an exam answer!       |
+------------------------------------------------------------+```

### Worked Example — Total Error Reduction from 2010 to 2017

Using the theory file's illustrative values: the 2010 error rate was approximately 0.28, and the 2017 error rate was approximately 0.023.

**Step 1 — Compute the absolute drop.**
```
Absolute drop = 0.28 - 0.023 = 0.257
```
This equals `25.7` percentage points of improvement over these 7 years.

**Step 2 — Compute the relative improvement.**
```
Relative improvement = (0.257 / 0.28) x 100
0.257 / 0.28 = 0.9178...
Relative improvement = 0.9178... x 100
Relative improvement ~= 91.78%
```

**Answer: roughly a 92% relative reduction in error** over the 2010-to-2017 window.

**Step 3 — A third, equally useful way to express this same improvement: as a "times smaller" ratio.**
```
Ratio = Old error / New error
Ratio = 0.28 / 0.023
Ratio ~= 12.17
```

**Interpretation:** the 2017 model was, on a relative basis, wrong roughly **12 times less often** than the 2010 model on this exact same benchmark task, even though the images and the general difficulty of the classification problem stayed the same across both years. This particular "roughly 12 times fewer mistakes" framing is an especially strong, memorable "wow-fact" worth keeping in your back pocket for any viva or interview question about why 2012-2017 is widely called the deep learning revolution in Computer Vision.

```
+------------------------------------------------------------+
| THREE WAYS TO DESCRIBE THE SAME 2010 -> 2017 IMPROVEMENT   |
+------------------------------------------------------------+
| Absolute drop:          25.7 percentage points             |
| Relative improvement:   roughly 92%                        |
| "Times fewer mistakes": roughly 12x fewer errors, on a     |
|   relative                                                 |
| basis                                                      |
+------------------------------------------------------------+```

### Summary

Comparing error rates across years requires deliberately choosing (and clearly stating) which framing you're using: the absolute drop in percentage points (a plain subtraction), the relative percentage improvement (the drop divided by the original error, times 100), or the "times fewer mistakes" ratio (old error divided by new error). All three framings are numerically valid and each tells a genuinely useful story, but they produce very different-sounding numbers from the exact same underlying data — always specify explicitly which one you are reporting, both in exam answers and in any real-world performance comparison.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 10. Cheat Sheet & Exam Hacks

```
+------------------------------------------------------------+
| BRIGHTNESS & CONTRAST                                      |
+------------------------------------------------------------+
| Brightness: B(I) = (1/wh) x [sum of ALL                    |
| pixel values]                                              |
| -> ALWAYS divide by TOTAL pixel count                      |
| (w times h), never just w or just h                        |
|                                                            |
| Contrast: (max - min) / average                            |
| (one common, working definition)                           |
+------------------------------------------------------------+

+------------------------------------------------------------+
| LOG TRANSFORM                                              |
+------------------------------------------------------------+
| s = c . log(1+|r|)                                         |
| -> find c using: c = 255 / log(1+r_max)                    |
| -> ALWAYS use (1+r) inside the log, never                  |
| log(r) alone (avoids log(0), undefined                     |
| whenever r = 0)                                            |
+------------------------------------------------------------+

+------------------------------------------------------------+
| GRAY-SLICING & SPATIAL FILTERS                             |
+------------------------------------------------------------+
| Gray-level slicing: inside [A,B] -> boost                  |
| to 255; outside -> push to 0                               |
|                                                            |
| Low-pass kernel: divide SUM of the 3x3                     |
| patch by 9 (plain average)                                 |
| Median filter: SORT all 9 values, take                     |
| the 5th (exact middle) one                                 |
| Sharpen kernel: (8 times center) MINUS                     |
| (sum of 8 neighbors), then divide by 9                     |
+------------------------------------------------------------+

+------------------------------------------------------------+
| OPTICAL FLOW & JPEG MATH                                   |
+------------------------------------------------------------+
| Optical flow magnitude: sqrt(u^2 + v^2)                    |
| -- pure Pythagoras, always POSITIVE                        |
| -> direction needs atan2(v,u) separately;                  |
| magnitude alone can NEVER tell two                         |
| opposite-direction motions apart                           |
|                                                            |
| JPEG blocks needed = (width/8) x (height/8)                |
| 4:2:0 downsampling -> chroma sample count                  |
| drops by EXACTLY 4 times                                   |
+------------------------------------------------------------+

+------------------------------------------------------------+
| COMPARING IMPROVEMENTS                                     |
+------------------------------------------------------------+
| Relative % improvement =                                   |
| ((old - new) / old) x 100                                  |
| -> this is NOT the same number as the                      |
| plain absolute percentage-point drop!                      |
+------------------------------------------------------------+```

### Exam Red Flags

1. **Computing B(I) but forgetting to divide by wh.** A raw, un-divided pixel sum is NOT brightness — brightness is always specifically an *average*. Always write out the explicit division step in your answer, even if it feels obvious.
2. **Confusing absolute versus relative error reduction.** "Error dropped from 26% to 16%" is a 10-point **absolute** drop, but roughly a 38.5% **relative** improvement — these are two different numbers computed two different ways, and a question may specifically ask for one or the other, or both.
3. **Using log(r) instead of log(1+r) in the log transform.** Using bare `log(r)` breaks completely at `r = 0`, since the logarithm of zero is mathematically undefined. Always use the exact form `log(1+|r|)` given in the formula, with no shortcuts.
4. **Averaging instead of taking the median in a median-filter question.** These two operations give genuinely DIFFERENT numeric answers whenever an outlier value is present in the neighborhood — that difference is the entire reason median filtering exists in the first place. Don't accidentally compute a mean when a question specifically asks for a median.
5. **Forgetting the final square root step in optical flow magnitude.** `u^2 + v^2` by itself is only the SQUARED magnitude, not the magnitude itself — always finish the calculation with the square root, `sqrt(...)`, as the very last step.
6. **Assuming optical flow magnitude alone can distinguish opposite-direction motion.** It genuinely cannot, as shown explicitly in Section 7's Example 3 — magnitude is completely direction-agnostic (squaring removes any negative sign); only a direction-binned histogram, like HOF, can actually tell two opposite motions apart.
7. **Miscounting JPEG blocks by forgetting to divide BOTH width and height by 8.** Always compute the full formula, `(width/8) times (height/8)` — dividing only one of the two dimensions by 8 and forgetting the other is a very common careless mistake under exam time pressure.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

## 11. Full-File Summary

This numerical file walks through every genuine formula introduced in Lecture 1, worked out completely by hand, one careful arithmetic step at a time. Brightness (B(I)) is shown to be nothing more than a plain average of all pixel intensities, always requiring an explicit division by the total pixel count. Contrast extends this by measuring how far pixel values spread out relative to that same average brightness. Building a histogram by hand is demonstrated as careful, verifiable counting — tallying how many pixels share each intensity value, and cross-checking that the tallies sum correctly to the total pixel count. The log transform is worked through with real numbers across a wide input range (0 to 4095), concretely demonstrating how it squashes large values far more aggressively than small ones — exactly the property that makes it useful for high-dynamic-range imagery. Gray-level slicing is shown as a straightforward per-pixel range check, either boosting or flattening each value depending on whether it falls inside a chosen window.

Spatial filtering is covered in the greatest depth, contrasting a plain low-pass average against a median filter on the exact same noisy patch — numerically proving that the median filter (output stays at 50) resists outlier noise far better than the plain average filter (output drifts to about 68.3) — before also working through the sharpening kernel's behavior on both a smooth patch (output near 0) and an edge-containing patch (output strongly positive). Optical flow magnitude is shown to be a direct Pythagorean-theorem calculation, and a dedicated worked example proves numerically that magnitude alone cannot distinguish two exactly opposite-direction motions, which is exactly why direction-sensitive descriptors like HOF matter. JPEG block counting and 4:2:0 chroma downsampling arithmetic quantify precisely how much data JPEG saves before any compression algorithm even runs. Finally, the ImageNet error-rate section carefully distinguishes absolute percentage-point drops from relative percentage improvements from "times-fewer-mistakes" ratios — three mathematically valid but genuinely different ways of describing the same underlying historical improvement in Computer Vision accuracy between 2010 and 2017.

[Back to Top](#-cv-nlp--lec-01-introduction--numerical)

---

> **Next:** [🎯 PRACTICE →](cvnlp_lec01_intro_practice.md) · [📖 THEORY →](cvnlp_lec01_intro_theory.md) · [Lecture 01 README →](README.md)
>
> *CV-NLP · Lec 01 · github.com/rpaut03l/TS-02-03*
