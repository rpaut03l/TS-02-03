"""
DL Lecture 06 - Encoder-Decoder Attention and Self-Attention From Scratch
============================================================================
What this file does (in plain words):
  Implements the Score -> Softmax -> Weighted Sum recipe from the theory
  file, TWICE: once as classic encoder-decoder attention (a decoder
  "looking back" at encoder hidden states), and once as self-attention
  (the Query/Key/Value mechanism). Every number is checked against the
  numerical README's worked examples.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


def softmax(scores):
    """Converts raw scores into a probability distribution (positive, sums to 1)."""
    exp_scores = np.exp(scores - np.max(scores))  # subtract max for numerical stability
    return exp_scores / np.sum(exp_scores)


# ---------------------------------------------------------------------------
# PART 1: Classic encoder-decoder attention
# ---------------------------------------------------------------------------
def compute_attention_context(raw_scores, encoder_hidden_states):
    """
    Implements: alpha = softmax(e),  c = sum(alpha_i * h_i)
    raw_scores            : array of shape (n_inputs,) - the e_ti values
    encoder_hidden_states : array of shape (n_inputs, hidden_dim) - the h_i values
    Returns (attention_weights, context_vector)
    """
    attention_weights = softmax(raw_scores)
    context_vector = attention_weights @ encoder_hidden_states   # weighted sum
    return attention_weights, context_vector


def check_encoder_decoder_attention():
    print("=" * 65)
    print("Check 1: Encoder-decoder attention (Worked Examples 1 & 2)")
    print("=" * 65)
    raw_scores = np.array([2.0, 0.5, -1.0, 3.0])
    hidden_states = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=float)

    weights, context = compute_attention_context(raw_scores, hidden_states)
    print(f"Attention weights: {np.round(weights, 4)}  (expected ~[0.2506,0.0559,0.0125,0.6811])")
    print(f"Sum of weights: {weights.sum():.6f}  (should be 1.0)")
    print(f"Context vector: {np.round(context, 4)}  (expected ~[0.2631,0.0684,0.9317])")

    plain_average = hidden_states.mean(axis=0)
    print(f"\nPlain average (for comparison): {plain_average}  (expected [0.5,0.5,0.5])")
    print("Notice: the plain average is FIXED, while the attention context vector")
    print("depends entirely on the raw_scores - change the scores, the context changes too.")


# ---------------------------------------------------------------------------
# PART 2: Self-attention (Query, Key, Value)
# ---------------------------------------------------------------------------
def self_attention(X, W_q, W_k, W_v, scale=True):
    """
    Implements: Q=XW_q, K=XW_k, V=XW_v
                Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) . V
    X : input embeddings, shape (n_words, embed_dim)
    Returns (Q, K, V, attention_weights, output)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = Q @ K.T                     # every word's Query dotted with every word's Key
    if scale:
        d_k = K.shape[-1]
        scores = scores / np.sqrt(d_k)   # scaling, as discussed in the theory file

    # softmax each ROW (each word's own distribution over all other words)
    attention_weights = np.array([softmax(row) for row in scores])
    output = attention_weights @ V
    return Q, K, V, attention_weights, output


def check_self_attention():
    print()
    print("=" * 65)
    print("Check 2: Self-attention (Worked Example 4)")
    print("=" * 65)
    X = np.array([
        [1.0, 0.0],   # word 1
        [0.0, 1.0],   # word 2
    ])
    W_q = np.array([[1.0, 0.0], [0.0, 1.0]])   # identity
    W_k = np.array([[1.0, 0.0], [0.0, 1.0]])   # identity
    W_v = np.array([[2.0, 0.0], [0.0, 2.0]])   # doubles the input

    Q, K, V, weights, output = self_attention(X, W_q, W_k, W_v)
    print(f"Q:\n{Q}")
    print(f"K:\n{K}")
    print(f"V:\n{V}")
    print(f"Attention weights (row = word, col = attends-to):\n{np.round(weights, 4)}")
    print(f"Word 1's output: {np.round(output[0], 4)}  (expected ~[1.3394, 0.6606])")


# ---------------------------------------------------------------------------
# PART 3: A slightly bigger, more realistic self-attention demo
# ---------------------------------------------------------------------------
def run_bigger_demo():
    print()
    print("=" * 65)
    print("Check 3: Self-attention on a 4-word toy sentence (random weights)")
    print("=" * 65)
    rng = np.random.default_rng(7)
    words = ["The", "cat", "sat", "down"]
    embed_dim = 4
    X = rng.normal(0, 1, (len(words), embed_dim))

    W_q = rng.normal(0, 0.5, (embed_dim, embed_dim))
    W_k = rng.normal(0, 0.5, (embed_dim, embed_dim))
    W_v = rng.normal(0, 0.5, (embed_dim, embed_dim))

    _, _, _, weights, _ = self_attention(X, W_q, W_k, W_v)
    print("Attention weight matrix (rows sum to 1.0, each row = one word's attention distribution):")
    print("        " + "  ".join(f"{w:>6s}" for w in words))
    for i, word in enumerate(words):
        row_str = "  ".join(f"{weights[i,j]:.4f}" for j in range(len(words)))
        print(f"{word:>6s}  {row_str}")
    print(f"\nRow sums (should all be 1.0): {np.round(weights.sum(axis=1), 6)}")


if __name__ == "__main__":
    check_encoder_decoder_attention()
    check_self_attention()
    run_bigger_demo()
