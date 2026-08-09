# 🧮 Advanced AI — Lec 02: Decision Trees, Risk & Lotteries — NUMERICAL

### *Every worked example, every number, every step shown*

> **Nav:** [⬅️ Prev: Lec 01](../Lec_01_Individual_Decision_Problem/README.md) | [← Lec 02 README](README.md) | [📖 THEORY](aai_lec02_decision_trees_lotteries_theory.md) | **NUMERICAL** | [🎯 PRACTICE](aai_lec02_decision_trees_lotteries_practice.md)

---

> ⚠️ **Note on the movie-theatre example (§1):** the original slide says *"the cost of walking each mile is equal to one unit of payoff"* while the distances are given in **km**. For a clean, self-contained worked example below we treat the distance numbers directly as the payoff-cost units (i.e. we use "1 unit of payoff per km" instead of per mile). The **method** — subtract distance-cost from the base payoff, then re-compare — is exactly what you'd do with real miles; just swap in the correct mile figure if your version of the question insists on miles.

## 📚 Table of Contents

| # | Worked Example | Jump |
|---|---|---|
| 1 | Decision Tree — The Movie Theatre Problem (full, 3 parts) | [§1](#1-decision-tree--the-movie-theatre-problem) |
| 2 | Expected Utility — Worked Examples (g vs s) | [§2](#2-expected-utility-worked-examples) |
| 3 | The Preference Flip — Why the Exact Numbers Matter | [§3](#3-the-preference-flip--why-the-exact-numbers-matter) |
| 4 | Compound Lottery Reduction | [§4](#4-compound-lottery-reduction) |
| 5 | Formula Cheat Sheet | [§5](#5-formula-cheat-sheet) |

---

## 1. Decision Tree — The Movie Theatre Problem

### The Setup

> Two movie theatres near your home: **Inox** (1 km away) and **PVR** (3 km away). Inox shows *Casablanca*, *Gone with the Wind*, *Dr. Strangelove*. PVR shows *The Matrix*, *Blade Runner*, *Aliens*. Your preferences are **alphabetical** — you like *Aliens* the most, *The Matrix* the least.

### Part (1) — Draw the tree, no payoffs yet

```
                                YOU  (root node)
                     ┌───────────────┴───────────────┐
                Go to Inox                       Go to PVR
                (action)                          (action)
          ┌─────────┼─────────┐            ┌──────────┼──────────┐
     Casablanca    GWTW     Dr.S       The Matrix  BladeRunner  Aliens
       (leaf)     (leaf)   (leaf)        (leaf)      (leaf)     (leaf)
```
- **Root node** = the individual, about to choose a theatre.
- **First branches** = the action "go to Inox" or "go to PVR."
- **Second-level nodes** = individual again, now choosing a film at that theatre.
- **Leaves** = 6 total final outcomes (one per fully-specified plan). No numbers yet — this is a pure map of possibilities.

### Part (2) — Alphabetic payoffs 1 through 6

**Step 1: sort all six movies alphabetically.**
```
Aliens, Blade Runner, Casablanca, Dr. Strangelove, Gone with the Wind, The Matrix
```

**Step 2: you like Aliens MOST and The Matrix LEAST — so assign payoff 6 down to 1, in that alphabetic order (best = 6):**

| Rank (best→worst) | Movie | Payoff |
|---|---|---|
| 1st (best) | Aliens | **6** |
| 2nd | Blade Runner | **5** |
| 3rd | Casablanca | **4** |
| 4th | Dr. Strangelove | **3** |
| 5th | Gone with the Wind | **2** |
| 6th (worst) | The Matrix | **1** |

**Step 3: attach these payoffs to the leaves.**

```
                                YOU
                     ┌───────────────┴───────────────┐
                Go to Inox                       Go to PVR
          ┌─────────┼─────────┐            ┌──────────┼──────────┐
     Casablanca    GWTW     Dr.S        The Matrix  BladeRunner  Aliens
        u=4         u=2      u=3           u=1         u=5        u=6
```

**Step 4: which option would you choose?**
```
v(a) for each full plan = the leaf payoff directly (no cost yet)
Best leaf = max(4, 2, 3, 1, 5, 6) = 6  →  "Go to PVR, watch Aliens"
```
✅ **Choice: PVR → Aliens (payoff 6).** No surprise — it's your literal favourite movie and, so far, distance costs nothing.

### Part (3) — Now add walking cost: 1 payoff unit per km

**Step 1: compute the distance-cost per theatre.**
```
Inox: 1 km  →  cost = 1 unit
PVR:  3 km  →  cost = 3 units
```

**Step 2: subtract the cost from every leaf at that theatre.**

```
v(action) = u(movie) − distance_cost(theatre)
```

| Theatre | Movie | Base u | − Cost | Updated v |
|---|---|---|---|---|
| Inox (−1) | Casablanca | 4 | −1 | **3** |
| Inox (−1) | Gone with the Wind | 2 | −1 | **1** |
| Inox (−1) | Dr. Strangelove | 3 | −1 | **2** |
| PVR (−3) | The Matrix | 1 | −3 | **−2** |
| PVR (−3) | Blade Runner | 5 | −3 | **2** |
| PVR (−3) | Aliens | 6 | −3 | **3** |

**Step 3: updated tree.**

```
                                YOU
                     ┌───────────────┴───────────────┐
                Go to Inox (−1)                 Go to PVR (−3)
          ┌─────────┼─────────┐            ┌──────────┼──────────┐
     Casablanca    GWTW     Dr.S        The Matrix  BladeRunner  Aliens
        v=3         v=1      v=2           v=−2        v=2        v=3
```

**Step 4: find the new best plan.**
```
max(3, 1, 2, −2, 2, 3) = 3  →  TWO leaves tie at 3:
   "Go to Inox, watch Casablanca"      AND      "Go to PVR, watch Aliens"
```

✅ **Would your choice change?** **Yes — it's no longer a clean single winner.** Before the walking cost, "PVR → Aliens" was the *unique* best plan. After the walking cost, it's now **tied** with "Inox → Casablanca." The extra 2 units of walking cost (3 − 1) exactly cancels out the 2-point gap between Aliens (6) and Casablanca (4). A rational individual is indifferent (`∼`) between these two plans now — either is an optimal `a*`.

> 🍼 **Kid version:** Aliens is your favourite movie, but it's a longer walk. Casablanca is not your favourite, but it's much closer. Once you count the walk as "tiredness cost," the two plans end up feeling exactly equally good.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--numerical)

---

## 2. Expected Utility Worked Examples

Throughout this section, assume `u(x) = x` (the payoff/utility of an outcome is just its face value — this is the simplest possible utility function, sometimes called "risk-neutral"). Two actions are compared: **g** (a gamble) and **s** (a safer bet).

### Example A — The base comparison

```
g:  0.75 → payoff 10   |   0.25 → payoff 0
s:  0.50 → payoff 10   |   0.50 → payoff 0
```

**Step 1 — apply the discrete expected utility formula:**
```
E[u(X)|g] = Σ u(xᵢ)·p(xᵢ|g) = (10 × 0.75) + (0 × 0.25)
          = 7.5 + 0
          = 7.5
```
```
E[u(X)|s] = (10 × 0.50) + (0 × 0.50)
          = 5.0 + 0
          = 5.0
```

**Step 2 — compare:**
```
E[u|g] = 7.5   >   E[u|s] = 5.0
```
✅ **Prefer g.** Even though `s` guarantees a 50% shot at the same top prize, `g`'s much higher chance (75% vs 50%) of hitting the ₹10 outcome wins out.

### Example B — Slightly different payoffs

```
g:  0.75 → payoff 9    |   0.25 → payoff −1
s:  0.50 → payoff 10   |   0.50 → payoff 0
```

**Step 1:**
```
E[u|g] = (9 × 0.75) + (−1 × 0.25)
       = 6.75 + (−0.25)
       = 6.5
```
```
E[u|s] = (10 × 0.50) + (0 × 0.50)
       = 5.0 + 0
       = 5.0
```

**Step 2 — compare:**
```
E[u|g] = 6.5   >   E[u|s] = 5.0
```
✅ **Still prefer g.** Even after lowering the top prize (9 instead of 10) and adding a small downside (−1), the higher-probability gamble still wins.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--numerical)

---

## 3. The Preference Flip — Why the Exact Numbers Matter

This is the single most important numeric lesson in this unit: **changing just ONE payoff number can completely flip which lottery is preferred**, even though the *probabilities* haven't moved at all.

### Example C — Same probabilities as Example B, one number changed

```
g:  0.75 → payoff 9    |   0.25 → payoff −8      ← only this number changed (was −1)
s:  0.50 → payoff 10   |   0.50 → payoff 0        ← unchanged
```

**Step 1 — recompute `E[u|g]`:**
```
E[u|g] = (9 × 0.75) + (−8 × 0.25)
       = 6.75 + (−2.0)
       = 4.75
```

**Step 2 — `E[u|s]` is unchanged from before:**
```
E[u|s] = 5.0
```

**Step 3 — compare:**
```
E[u|g] = 4.75   <   E[u|s] = 5.0
```
🔄 **The preference FLIPS! Now prefer s, not g.**

### Side-by-side: watch the flip happen

| Version | g's bad outcome | E[u\|g] | E[u\|s] | Winner |
|---|---|---|---|---|
| Example B | −1 | 6.5 | 5.0 | **g** (gamble wins) |
| Example C | −8 | 4.75 | 5.0 | **s** (safe wins) |

> 🍼 **Kid version:** A scratch card that's "75% chance of a small prize, 25% chance of losing a tiny bit" is still worth playing. The SAME scratch card where the 25%-chance downside is suddenly a HUGE loss is no longer worth playing — even though your odds of winning didn't change at all. **How much you can lose matters just as much as how likely you are to lose it.**

### 🧠 Mnemonic: **"Odds don't flip you — Outcomes do."**
If two lotteries have the exact same probabilities but you change what's actually AT STAKE in even one branch, the expected-utility comparison can flip. Always plug in the actual numbers — never eyeball probabilities alone.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--numerical)

---

## 4. Compound Lottery Reduction

### The Setup

```
Action g leads to a COMPOUND lottery:

                     g
                     │
                     N  (first Nature move)
              ┌──────┴──────┐
           p=0.625        p=0.375
              │               │
              N               N   (second Nature move)
          ┌───┴───┐       ┌───┴───┐
        0.9      0.1     0.5     0.5
         │        │       │       │
        10        0      10       0

Action s leads to a SIMPLE lottery:

                     s
                     │
                     N
              ┌──────┴──────┐
            0.5             0.5
             │               │
            10               0
```

### Step 1 — Identify every path through the compound tree for action `g`

| Path | Probability (multiply along the path) | Final payoff |
|---|---|---|
| 0.625 → 0.9 | 0.625 × 0.9 = **0.5625** | 10 |
| 0.625 → 0.1 | 0.625 × 0.1 = **0.0625** | 0 |
| 0.375 → 0.5 | 0.375 × 0.5 = **0.1875** | 10 |
| 0.375 → 0.5 | 0.375 × 0.5 = **0.1875** | 0 |

*(This uses the basic chain rule of probability: the chance of following a whole path is the product of the chances at each step along it.)*

### Step 2 — Group paths by their final payoff (law of total probability)

```
P(final payoff = 10) = 0.5625 + 0.1875 = 0.75
P(final payoff = 0)  = 0.0625 + 0.1875 = 0.25
```

### Step 3 — Check: this is exactly the reduced simple lottery

```
Reduced g:   0.75 → 10   |   0.25 → 0
```

Compare with **Example A** in §2 — this is the *exact same simple lottery* used there! This confirms an important principle: **a compound lottery is always reducible to an equivalent simple lottery**, and once reduced, you evaluate it with the ordinary expected-utility formula exactly as before.

### Step 4 — Sanity-check the probabilities sum to 1

```
0.75 + 0.25 = 1.00   ✓  valid simple lottery
```

### Step 5 — Now compute expected utility exactly as in §2

```
E[u|g] = (10 × 0.75) + (0 × 0.25) = 7.5    ← identical to Example A
E[u|s] = (10 × 0.50) + (0 × 0.50) = 5.0
```
✅ **Prefer g** — same answer, whether you evaluate the compound tree directly (by reducing it first) or the pre-reduced simple lottery. This is exactly why the theory lets you always "flatten" a compound lottery before evaluating it.

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--numerical)

---

## 5. Formula Cheat Sheet

```
╔══════════════════════════════════════════════════════════════════╗
║  FORMULAS — COPY THESE DOWN BEFORE THE EXAM                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Discrete lottery validity:                                       ║
║      0 ≤ p(xᵢ|a) ≤ 1     and     Σᵢ p(xᵢ|a) = 1                   ║
║                                                                    ║
║  Continuous lottery (CDF) validity:                                ║
║      Fₐ(−∞)=0,  Fₐ(+∞)=1,  non-decreasing,  right-continuous      ║
║                                                                    ║
║  Expected utility (discrete):                                     ║
║      E[u(X)|p|ₐ] = Σᵢ₌₁ⁿ u(xᵢ) · p(xᵢ|a)                          ║
║                                                                    ║
║  Expected utility (continuous):                                   ║
║      E[u(X)|Fₐ] = ∫₋∞^∞ u(x) dFₐ(x)                               ║
║                                                                    ║
║  Compound lottery path probability = PRODUCT along the path        ║
║  Same-outcome paths → ADD their probabilities together             ║
║                                                                    ║
║  Prefer g over s  ⟺  E[u|g] ≥ E[u|s]                               ║
╚══════════════════════════════════════════════════════════════════╝
```

[↑ Back to Top](#-advanced-ai--lec-02-decision-trees-risk--lotteries--numerical)

---

> **Next:** [🎯 PRACTICE →](aai_lec02_decision_trees_lotteries_practice.md) · [← back to THEORY](aai_lec02_decision_trees_lotteries_theory.md)
>
> *Advanced AI · Lec 02 · github.com/rpaut03l/TS-02-03*
