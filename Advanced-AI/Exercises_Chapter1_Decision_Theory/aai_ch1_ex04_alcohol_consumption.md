# 📗 Exercise 1.4 — Alcohol Consumption

> **Nav:** [← Exercise 1.3](aai_ch1_ex03_fruit_or_candy.md) | [← Chapter 1 Exercises](README.md) | **1.4** | [1.5 →](aai_ch1_ex05_buying_a_car.md)

---

## 📋 What's Being Asked

Payoff function `v(a) = θa − 4a²`, where `a` is the amount consumed and `θ` is a person-specific parameter (larger people have larger `θ`). Given `θ` ranges from 0.2 (smallest person) to 6 (largest person), find: (a) an amount nobody should ever drink; (b) the optimal `a` for `θ=1` and `θ=4`; (c) a proof that smaller people should drink less than larger people; (d) whether anyone should ever drink more than 1 liter.

---

## Prerequisite — The General Optimization Method

This is a **Pattern 3** (continuous action set) problem — `a` can be any non-negative real number, so you can't just list options and compare. Instead: take the derivative `v'(a)`, set it to zero, solve for the critical point, and confirm it's a maximum using the second derivative `v''(a) < 0`.

## Step 1 — Take the First Derivative

```
v(a) = θa − 4a²
v'(a) = θ − 8a
```

## Step 2 — Set to Zero and Solve for `a*`

```
θ − 8a = 0
a* = θ/8
```

## Step 3 — Confirm It's a Maximum

```
v''(a) = −8
```
`−8 < 0` for all `a` → the curve is **concave** (a downward-opening parabola) → `a* = θ/8` is confirmed a **maximum**.

## Part (a) — An Amount No Person Should Drink

Since `a*(θ) = θ/8` is **strictly increasing** in `θ`, the largest optimal drinking amount across the whole population occurs at the largest `θ`:
```
a*(θ_max) = a*(6) = 6/8 = 0.75
```
**Every single person's optimal amount is at most 0.75.** Therefore, **any amount greater than 0.75** (e.g., 1 liter, 2 liters) is an amount that literally no rational person in this population should ever drink, regardless of their size.

## Part (b) — Optimal Amount for Specific `θ` Values

```
θ = 1:  a* = 1/8 = 0.125
θ = 4:  a* = 4/8 = 0.5
```

## Part (c) — Proving Larger People Should Drink More

**Step 1 — restate the claim algebraically:** we must show that if `θ₁ < θ₂` (person 1 is smaller than person 2), then `a*(θ₁) < a*(θ₂)`.

**Step 2 — use the formula directly:**
```
a*(θ) = θ/8
```
This is a **linear, strictly increasing function of `θ`** — the coefficient `1/8` is positive. For any `θ₁ < θ₂`:
```
a*(θ₁) = θ₁/8  <  θ₂/8 = a*(θ₂)
```
This inequality holds simply because dividing both sides of `θ₁ < θ₂` by the positive constant `8` preserves the direction of the inequality.

**Step 3 — connect back to the problem's given fact** ("larger people have higher θs than smaller people"): since `a*` is a strictly increasing function of `θ`, and larger people have strictly higher `θ`, it follows immediately that larger people have strictly higher optimal drinking amounts `a*`. ✅ **Proven in general** — not just for the two example values in part (b), but for the entire population.

## Part (d) — Should Anyone Drink More Than 1 Liter?

From part (a), the single largest optimal amount in the entire population is `a*(6) = 0.75` liters — strictly less than 1 liter. **No — nobody should ever drink more than one 1-liter bottle**, since even the largest person's own optimum falls well under that threshold.

```
v(a)
    |            ●  ← peak at a* = θ/8 (varies by person)
    |          ╱   ╲
    |        ╱       ╲
    |      ╱           ╲
  0 |____╱_______________╲____ a
        0     a*=θ/8    2(θ/8)
        Largest possible a* (θ=6): only 0.75 — well under 1 liter
```

---

## 🧠 Mnemonic & Cheat Sheet

```
╔══════════════════════════════════════════════════════════╗
║  "DERIVE, ZERO, CHECK CONCAVE, BOUND BY EXTREMES"             ║
║  1. v'(a) = θ − 8a           (derive)                        ║
║  2. θ−8a=0 → a*=θ/8          (zero)                          ║
║  3. v''(a)=−8<0 → confirmed max (check concave)               ║
║  4. Plug in θ_min, θ_max for population-wide bounds (extremes)║
╚══════════════════════════════════════════════════════════╝
```

**Exam-relevant takeaway:** whenever a payoff is quadratic (`v(a) = pa² + qa + r` or, equivalently written, `θa − 4a²`), the optimum is always at `a* = −q/(2p)` — here that shortcut gives `a* = −θ/(2×(−4)) = θ/8`, matching the derivative method exactly. Both routes always agree for quadratics.

---

## 📝 Summary

This exercise is the cleanest possible introduction to Pattern 3 from this folder's overview — a continuous action set that genuinely requires calculus, not comparison of a short list, to solve. The whole method reduces to a four-step ritual that's worth memorizing exactly as written: differentiate the payoff, set the derivative to zero, solve for the critical point, then confirm it's a maximum (not a minimum or saddle) by checking that the second derivative is negative. What makes this particular problem elegant is that the optimal amount, `a* = θ/8`, turned out to be a simple linear function of the person-specific parameter, which made proving the general "bigger people drink more" claim almost effortless — a strictly increasing function of `θ` combined with the given fact that bigger people have bigger `θ` was all that was needed, no case-by-case checking required. The population-bound questions (parts a and d) showed a technique that reappears constantly in optimization problems: once you know the *shape* of how the optimal action depends on a parameter, you only ever need to check the parameter's extreme values to bound the optimal action across an entire population, rather than testing every person individually.

---

> **Next:** [Exercise 1.5 — Buying a Car →](aai_ch1_ex05_buying_a_car.md)
>
> *Advanced AI · Chapter 1 Exercises · github.com/rpaut03l/TS-02-03*
