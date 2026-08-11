"""
DL Lecture 04 - A Vanilla RNN Built From Scratch (NumPy only)
================================================================
What this file does (in plain words):
  Builds the exact recurrence formula from the theory file -
  s_t = tanh(U x_t + W s_{t-1}), o_t = softmax(V s_t) - and runs it forward
  across a short toy sequence, printing the hidden state at every timestep.
  It also includes a small demo that VISUALIZES the vanishing gradient
  problem, by tracking how a gradient signal shrinks as it is backpropagated
  further and further back through time.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np

np.random.seed(0)


# ---------------------------------------------------------------------------
# PART 1: A vanilla RNN cell - one single timestep of the recurrence formula
# ---------------------------------------------------------------------------
def rnn_step(x_t, s_prev, U, W, b):
    """
    Implements: s_t = tanh(U x_t + W s_prev + b)
    x_t    : input vector at this timestep, shape (D,)
    s_prev : previous hidden state, shape (H,)
    U      : input-to-hidden weights, shape (H, D)
    W      : hidden-to-hidden weights, shape (H, H)
    b      : bias, shape (H,)
    Returns the NEW hidden state, shape (H,)
    """
    pre_activation = U @ x_t + W @ s_prev + b
    s_t = np.tanh(pre_activation)
    return s_t


def rnn_output(s_t, V, c):
    """Implements: o_t = softmax(V s_t + c)"""
    logits = V @ s_t + c
    exp_logits = np.exp(logits - np.max(logits))   # subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)


# ---------------------------------------------------------------------------
# PART 2: Run a full forward pass across a toy sequence
# ---------------------------------------------------------------------------
def run_forward_pass():
    D, H, k = 4, 5, 4   # input size, hidden size, output size (small toy vocabulary)

    U = np.random.randn(H, D) * 0.5
    W = np.random.randn(H, H) * 0.5
    V = np.random.randn(k, H) * 0.5
    b = np.zeros(H)
    c = np.zeros(k)

    # A toy "sentence" of 5 one-hot vectors over a vocabulary of size 4
    vocabulary = ["cat", "sat", "on", "mat"]
    sentence_indices = [0, 1, 2, 3, 3]   # "cat sat on mat mat" (toy, repeats "mat")
    sequence = []
    for idx in sentence_indices:
        one_hot = np.zeros(D)
        one_hot[idx] = 1.0
        sequence.append(one_hot)

    print("=" * 65)
    print("Forward pass through a toy RNN")
    print("=" * 65)
    s_t = np.zeros(H)   # h_0 = 0, memory starts empty (per theory file)
    for t, x_t in enumerate(sequence):
        s_t = rnn_step(x_t, s_t, U, W, b)
        o_t = rnn_output(s_t, V, c)
        predicted_word = vocabulary[np.argmax(o_t)]
        print(f"t={t} | input={vocabulary[sentence_indices[t]]:5s} | "
              f"hidden_state={np.round(s_t, 3)} | predicted_next_word={predicted_word}")


# ---------------------------------------------------------------------------
# PART 3: Visualize the vanishing / exploding gradient problem numerically
#          (verified against the numerical README's Worked Examples 4 and 5)
# ---------------------------------------------------------------------------
def demonstrate_vanishing_and_exploding_gradients():
    print()
    print("=" * 65)
    print("Vanishing gradient demo (per-step factor = 0.5)")
    print("=" * 65)
    gradient = 1.0
    factor = 0.5
    for steps_back in [1, 5, 10, 20]:
        value = gradient * (factor ** steps_back)
        print(f"  After {steps_back:2d} timesteps back: gradient contribution = {value:.10f}")

    print()
    print("=" * 65)
    print("Exploding gradient demo (per-step factor = 1.5)")
    print("=" * 65)
    factor = 1.5
    for steps_back in [5, 10, 20, 50]:
        value = gradient * (factor ** steps_back)
        print(f"  After {steps_back:2d} timesteps back: gradient contribution = {value:,.2f}")


# ---------------------------------------------------------------------------
# PART 4: Gradient clipping (verified against Worked Example 6)
# ---------------------------------------------------------------------------
def clip_gradient(g, threshold):
    """Rescales g to have norm == threshold, ONLY if its norm exceeds threshold."""
    norm = np.linalg.norm(g)
    if norm > threshold:
        g = g * (threshold / norm)
    return g


def demonstrate_gradient_clipping():
    print()
    print("=" * 65)
    print("Gradient clipping demo")
    print("=" * 65)
    g = np.array([3.0, 4.0])
    threshold = 2.0
    g_clipped = clip_gradient(g, threshold)
    print(f"Original gradient: {g}, norm = {np.linalg.norm(g):.3f}")
    print(f"Clipped gradient:  {g_clipped}, norm = {np.linalg.norm(g_clipped):.3f} (should equal threshold={threshold})")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_forward_pass()
    demonstrate_vanishing_and_exploding_gradients()
    demonstrate_gradient_clipping()
