# DL Lecture 04 — Recurrent Neural Networks (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

> Folder: `Deep-Learning/Lecture-04-Recurrent-Neural-Networks/theory/`
> Pairs with: [`numerical/dl_lecture04_rnn_numerical.md`](../numerical/dl_lecture04_rnn_numerical.md) · [`practice/dl_lecture04_rnn_practice.md`](../practice/dl_lecture04_rnn_practice.md) · [`exercises/dl_lecture04_exercises.md`](../exercises/dl_lecture04_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "Recurrent Neural Networks" deck, parts 1 & 2 (combined here)

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Why Feedforward Networks Fail on Sequences](#why-feedforward-networks-fail-on-sequences)
3. [CNN vs RNN — Similarities and Differences](#cnn-vs-rnn--similarities-and-differences)
4. [The Core Idea: An Internal State That Updates](#the-core-idea-an-internal-state-that-updates)
5. [The RNN Recurrence Formula](#the-rnn-recurrence-formula)
6. [Shape Breakdown of an RNN](#shape-breakdown-of-an-rnn)
7. [Types of RNN Architectures](#types-of-rnn-architectures)
8. [Learning Through the Computational Graph](#learning-through-the-computational-graph)
9. [Training with Backpropagation Through Time (BPTT)](#training-with-backpropagation-through-time-bptt)
10. [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
11. [The Exploding Gradient Problem](#the-exploding-gradient-problem)
12. [Four Ways to Fight Vanishing/Exploding Gradients](#four-ways-to-fight-vanishingexploding-gradients)
13. [Mnemonics](#mnemonics)
14. [Cheatsheet](#cheatsheet)
15. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
16. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine reading a mystery novel one word at a time, but with total amnesia — the instant you finish a sentence, you forget everything you just read, and start the next sentence completely fresh. You'd never figure out who the murderer is, because solving the mystery depends on *remembering* clues from many pages ago. Now imagine instead you keep a small notebook, and after every sentence you jot down a short summary of everything important so far, then use that notebook summary (plus the new sentence) to update your understanding. That small, constantly-updated notebook is exactly what a **Recurrent Neural Network's hidden state** is. An RNN reads a sequence one piece at a time (a word, an audio sample, a stock price), and after each piece, updates a small internal summary (the hidden state) that carries forward everything relevant it has learned so far — instead of treating each piece of input as if it appeared out of nowhere, with no history.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Why Feedforward Networks Fail on Sequences

Traditional feedforward networks (the MLPs from Lecture 2, and even CNNs from Lecture 3) assume all inputs and outputs are **independent of each other**. This assumption breaks badly for language and speech. Consider: *"The man who wore a wig on his head went inside."* Who went inside — the man, or the wig? Answering correctly requires remembering the entire sequence of words that came before "went inside," not just the last word or two in isolation. A **Recurrent Neural Network (RNN)** is a class of neural network where connections between units form a **directed cycle**, letting it process arbitrary-length sequences of inputs while carrying information forward from earlier steps to later ones.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## CNN vs RNN — Similarities and Differences

**Similarity:** both CNNs and RNNs constrain the model space by **sharing parameters** between neurons/timesteps — parameter sharing is the key to the success of both architecture families (recall Lecture 3's parameter-sharing story, now reused across *time* instead of *space*).

**Differences:**

| | CNN | RNN |
|---|---|---|
| Modeling type | Spatial | Sequential |
| Sharing dimension | Spatial dimension (usually images) | Temporal dimension (usually text/speech) |
| Computation steps | Fixed number; output depends only on current input | Variable number; hidden layers and output depend on previous hidden states too |
| Simple intuition | "Pattern detector" (like edges in images) | "Memory-based learner" (understands sequence flow) |
| Context awareness | Only sees local receptive fields | Learns "what happened before" — critical for prediction |

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## The Core Idea: An Internal State That Updates

**Recurrent** literally means "repeating the same operation again and again." The key properties: (1) RNNs **have memory** — a hidden state that persists and updates across timesteps; (2) RNNs **share weights** across all timesteps, so there are far fewer learned parameters than if you tried to learn a separate network for every possible sequence position; (3) the underlying **idea/assumption** is that temporal information matters, and using the *same rule* at every timestep lets the model generalize patterns across the whole sequence without learning a separate rule for each position.

**Where does memory come from for the very first word, when there's no previous context?** The network starts with an **empty memory, often all zeros: h₀ = 0** (or equivalently s₋₁ = 0, depending on notation) — this initial hidden state is then updated by the recurrence formula as real inputs arrive.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## The RNN Recurrence Formula

This is the single most important formula in the lecture, given in two equivalent notations across the slides. Using the WildML-style notation:

```
s_t = f( U x_t + W s_{t-1} )
o_t = softmax( V s_t )
```

Where:
- **x_t** = input at timestep t (e.g., a one-hot vector representing a word in a sentence)
- **s_t** = hidden state at timestep t (the network's "memory") — calculated from the *current* input **and** the *previous* hidden state
- **U** = weight matrix applied to the current input
- **W** = weight matrix applied to the previous hidden state (this is the "recurrent" weight — the SAME W is reused at every single timestep)
- **f** = a non-linear activation function (commonly tanh or ReLU)
- **s₋₁** = the initial hidden state, typically initialized to all zeros
- **o_t** = output at timestep t — e.g., for next-word prediction, a probability vector across the entire vocabulary, computed via softmax
- **V** = weight matrix mapping the hidden state to the output space

The equivalent Fei-Fei Li-style notation used elsewhere in the slides writes the same idea as `h_t = f_W(h_{t-1}, x_t)` ("new state = some function of old state and the input vector at this timestep") and `y_t = f_{Why}(h_t)` ("output = another function, with its own parameters, of the new state"). **Both notations describe exactly the same mechanism** — don't let the symbol-renaming (s vs h, U/W/V vs Wxh/Whh/Why) confuse you; always check which convention a given question is using.

**The single most important detail to memorize:** the *same* function `f_W` (with the *same* weights U, W, V) is reused at every timestep — this is parameter sharing across time, directly analogous to a CNN filter being reused across space.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Shape Breakdown of an RNN

- **x_t** — input at timestep t, e.g. a one-hot vector the size of your vocabulary.
- **s_t** (or h_t) — the hidden state, "memory" of the network at timestep t, whose size (call it H) is a design choice — a bigger hidden state can, in principle, remember more, at the cost of more parameters.
- **o_t** — output at timestep t. If the output size is **k** (e.g., the vocabulary size, for next-word prediction), then `o_t` is a k-length vector of probabilities, produced via softmax so all k values sum to 1.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Types of RNN Architectures

RNNs are flexible in how many inputs and outputs they handle, unlike a plain feedforward "Vanilla Neural Network" (one fixed input → one fixed output). Common patterns from the lecture:

```
ONE -> ONE       :  Vanilla Neural Network (no recurrence needed)
ONE -> MANY      :  Image Captioning (one image -> a sequence of words)
MANY -> ONE      :  Action Prediction (a sequence of video frames -> one action class)
MANY -> MANY     :  Video Captioning (a sequence of video frames -> a caption, another sequence)
MANY -> MANY     :  Video classification on a frame level (input sequence, output a label PER frame)
```

Each pattern reuses the exact same recurrence formula — what changes is simply *when* you feed in a new input, and *when* you read out an output, at each timestep.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Learning Through the Computational Graph

A **computational graph** is a diagram showing every operation and how data flows through them — useful for reasoning about both the forward pass (computing outputs) and the backward pass (computing gradients for training). The basic vanilla RNN is sometimes called an **"Elman RNN,"** named after Prof. Jeffrey Elman, who introduced this architecture. Unrolling the computational graph across timesteps (drawing a separate box for each timestep, all sharing the same weights) makes it visually obvious that an RNN is really just a very deep feedforward network in disguise — except the "layers" are timesteps, and every layer uses *identical* weights, unlike a normal deep network where every layer typically has its own independent weights.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Training with Backpropagation Through Time (BPTT)

A **gradient** tells us how much the output (or loss) changes when we slightly change a parameter — this is the same gradient descent idea from Lecture 2, now applied across an unrolled sequence. Since an RNN is a computational graph, we can use **backpropagation** to compute gradients of the loss with respect to every parameter — when applied to an RNN's unrolled-across-time structure, this specific procedure is called **Backpropagation Through Time (BPTT)**: run the network **forward through the entire sequence** to compute the total loss, then run **backward through the entire sequence** to compute gradients, summing up the error/gradient contributions from *every* timestep for one training example. Once gradients are computed, weights are updated using SGD or a variant (like Adam, covered in the DNN Optimization lecture).

The tricky part mathematically: since the SAME weight matrix W is reused at every timestep, computing ∂Loss/∂W requires summing contributions from *every* timestep, and each of those contributions itself requires a chain-rule expansion reaching back through *all* previous timesteps — this repeated chain-rule multiplication across many timesteps is exactly the mechanism that causes the vanishing/exploding gradient problems discussed next.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## The Vanishing Gradient Problem

**The problem:** as sequence length grows, the impact of far-away inputs on the current prediction diminishes, because the gradient literally **vanishes** (shrinks toward zero) as it's backpropagated through many timesteps [Bengio et al., 1994].

**Why this matters (the lecture's running example, made progressively harder):** *"I have a banana and an apple. My friend ate the banana. I was hungry and wanted a fruit. I really wished I had a banana as well, but we were all out. So I ate the ____."* The correct answer ("apple") depends on information from *far* back in the sequence. If gradients vanish, the network effectively "forgets" that far-back context and cannot learn to use it — badly hurting performance on any task with long-range dependencies.

**Why it happens, mechanically:** the gradient of common non-linear activations (tanh, sigmoid) is always a number strictly between 0 and 1 — these functions **saturate** (flatten out, gradient near zero) at extreme input values. Backpropagating through many timesteps means **multiplying many such sub-1 numbers together**, and repeatedly multiplying numbers less than 1 makes the product shrink toward zero, exponentially fast, the longer the sequence. This isn't unique to RNNs — it happens in *any* long chain of such activations, including very deep feedforward networks, but RNNs are especially prone to it because "depth" there literally equals sequence length, which can easily be 20+ words or more.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## The Exploding Gradient Problem

The mirror-image problem: if gradients are **not** fractional (for example, when using ReLU, whose derivative is exactly 1 for positive inputs rather than a fraction), repeated multiplication across many timesteps can instead cause the gradient to grow **explosively large**, quickly leading to **numeric overflow errors** and unstable, wildly-swinging training.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Four Ways to Fight Vanishing/Exploding Gradients

1. **Change the activation function.** Swap tanh/sigmoid for **ReLU**, whose derivative is exactly 1 for any positive pre-activation value — multiplying many 1's together doesn't vanish. Trade-off: ReLU's derivative is exactly **0** for negative pre-activation values, which can *completely erase* the gradient in that regime (a different failure mode, sometimes called "dead ReLUs").
2. **Don't backpropagate all the way to the beginning — Truncated BPTT (TBPTT).** Instead of computing gradients across the entire sequence, truncate the backpropagation process after some fixed number of steps. This essentially imposes a Markov-like assumption ("only the recent past really matters for this gradient update"), trading some long-range learning ability for stability and speed.
3. **Gradient clipping.** Use ReLU to fight vanishing gradients, but explicitly guard against exploding gradients: if a computed gradient's magnitude exceeds some threshold, clip/truncate it back down to that threshold before applying the update. Together, ReLU (fighting vanishing) + clipping (fighting exploding) form a simple, commonly used combo.
4. **Change the RNN's internals more thoroughly — gated architectures.** Use a fundamentally redesigned recurrent unit, such as an **LSTM (Long Short-Term Memory)** or a **GRU (Gated Recurrent Unit)**, both of which use internal "gates" specifically engineered to let gradients flow across many timesteps without vanishing. This is precisely the topic of the very next lecture in this course.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Mnemonics

- **"A notebook you keep updating, not amnesia every sentence"** → the core RNN hidden-state idea.
- **"Same W, every timestep"** → parameter sharing across time, the RNN analogue of Lecture 3's parameter sharing across space.
- **"h₀ = 0, memory starts empty"** → how RNNs handle the very first input.
- **"Multiply fractions long enough, they vanish; multiply large numbers long enough, they explode"** → the two gradient failure modes in one line.
- **"ReLU + Clip = fight both"** → the simple two-part fix combo before you need a full gated architecture.
- **BPTT = "unroll, then roll gradients back"** → forward through the whole sequence, then backward through the whole sequence.

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Cheatsheet

| Concept | One-liner | Formula |
|---|---|---|
| Hidden state update | New memory = function of old memory + current input | `s_t = f(U x_t + W s_{t-1})` |
| Output | Prediction from current hidden state | `o_t = softmax(V s_t)` |
| Initial state | Memory starts empty | `s_{-1} = 0` (or `h_0 = 0`) |
| BPTT | Forward whole sequence, then backward whole sequence | sums gradients across all timesteps |
| Vanishing gradient | Product of many <1 numbers → 0 | caused by saturating tanh/sigmoid derivatives |
| Exploding gradient | Product of many large numbers → overflow | common with non-saturating activations like ReLU |
| Fix 1 | Change activation | tanh/sigmoid → ReLU |
| Fix 2 | Truncated BPTT | Markov-like assumption, limited lookback |
| Fix 3 | Gradient clipping | cap gradient magnitude at a threshold |
| Fix 4 | Gated architecture | LSTM / GRU (next lecture) |

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** writing the recurrence formula without the previous hidden state term — always include BOTH the current input AND the previous state: `s_t = f(U x_t + W s_{t-1})`, never just `f(U x_t)`.
- **Trap:** thinking vanishing gradients only affect RNNs — the theory file explicitly states this affects *any* long chain of saturating activations, including deep feedforward networks; RNNs are just especially exposed to it because sequence length directly equals "depth."
- **Trap:** confusing vanishing and exploding gradients' causes — vanishing comes from repeatedly multiplying numbers **less than 1** (saturating activations); exploding comes from repeatedly multiplying numbers that are **not bounded below 1** (e.g., ReLU's derivative of exactly 1, or large weight values).
- **Exam hack:** if asked "why can't a plain feedforward network handle language modeling well," always anchor your answer in the independence assumption — feedforward nets assume inputs/outputs are independent, which is false for sequences with long-range dependencies like the wig/man example.
- **Exam hack:** the four fixes for vanishing/exploding gradients are a favourite "list all of X" exam question — memorize them in this exact order (change activation → truncated BPTT → gradient clipping → gated architectures) since it mirrors "cheapest/simplest fix first, most thorough fix last."

`[🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory)`

---

## Summary

This lecture introduces Recurrent Neural Networks as the fix for feedforward networks' fatal blind spot on sequential data: the assumption that inputs and outputs are independent, which breaks down badly for language and speech where meaning depends on the entire preceding sequence. RNNs share the parameter-sharing philosophy with CNNs (the key to both architectures' success) but share weights across the *temporal* dimension instead of the *spatial* dimension, using a variable number of computation steps where hidden states and outputs depend on previous hidden states. The core mechanism is the recurrence formula `s_t = f(U x_t + W s_{t-1})`, producing an output `o_t = softmax(V s_t)`, with memory starting at zero (`s_{-1}=0`) and the *same* weights U, W, V reused identically at every single timestep. RNNs support flexible input/output patterns (one-to-many for image captioning, many-to-one for action prediction, many-to-many for video captioning), all using the same underlying recurrence. Training uses Backpropagation Through Time (BPTT) — run forward through the whole sequence, then backward through the whole sequence, summing gradients across every timestep — but this repeated chain-rule multiplication across many timesteps creates two serious problems: vanishing gradients (many sub-1 saturating-activation derivatives multiplied together shrink toward zero, causing the network to effectively forget long-range context, illustrated by the lecture's escalating banana/apple example) and exploding gradients (many large numbers multiplied together cause numeric overflow). Four progressively more thorough fixes exist: switching to ReLU-style activations, truncating backpropagation (TBPTT), clipping oversized gradients, and — the most thorough fix, previewing the very next lecture — redesigning the recurrent unit's internals entirely using gated architectures like LSTM or GRU.

`[← Lecture 03](../../Lecture-03-Convolutional-Neural-Networks/README.md) · [🔝 Top](#dl-lecture-04--recurrent-neural-networks-theory) · [Next: Numerical →](../numerical/dl_lecture04_rnn_numerical.md)`
