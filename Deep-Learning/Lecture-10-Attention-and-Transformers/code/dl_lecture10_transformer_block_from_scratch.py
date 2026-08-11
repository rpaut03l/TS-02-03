"""
DL Lecture 10 - A Mini Transformer Block Built From Scratch (NumPy only)
===========================================================================
What this file does (in plain words):
  Implements the full Transformer block pipeline from the theory file -
  self-attention (reusing Lecture 6's mechanism) -> residual connection ->
  layer norm -> per-token MLP -> residual connection -> layer norm - and
  checks the residual/LayerNorm pieces against the numerical README.
  Also includes a standalone ViT patch-counter and a decoder step simulator.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# PART 1: Self-attention (same mechanism as Lecture 6, reused here)
# ---------------------------------------------------------------------------
def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V


# ---------------------------------------------------------------------------
# PART 2: Residual connection + Layer Normalization
# ---------------------------------------------------------------------------
def residual_connection(x, sublayer_output):
    """Implements: output = sublayer(x) + x"""
    return x + sublayer_output


def layer_norm(x, eps=1e-8):
    """
    Normalizes EACH ROW (each token) independently, across its own features.
    x: shape (n_tokens, feature_dim)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def check_residual_and_layernorm():
    print("=" * 65)
    print("Check 1: Residual connection + Layer Norm (Worked Examples 2 & 3)")
    print("=" * 65)
    x = np.array([1.0, 2.0, 3.0])
    sublayer_out = np.array([0.5, -0.2, 0.1])
    residual_out = residual_connection(x, sublayer_out)
    print(f"Residual output: {residual_out}  (expected [1.5, 1.8, 3.1])")

    normed = layer_norm(residual_out.reshape(1, -1), eps=0)[0]
    print(f"Layer-normalized: {np.round(normed, 4)}  (expected ~[-0.912, -0.48, 1.392])")


# ---------------------------------------------------------------------------
# PART 3: A full mini Transformer block
# ---------------------------------------------------------------------------
def simple_mlp(x, W1, b1, W2, b2):
    """A tiny 2-layer MLP, applied to EACH token independently (per theory file)."""
    hidden = np.maximum(0, x @ W1 + b1)   # ReLU
    return hidden @ W2 + b2


def transformer_block(X, params):
    """
    Full pipeline: Self-Attention -> residual -> LayerNorm -> MLP -> residual -> LayerNorm
    X: shape (n_tokens, dim)
    """
    # --- Self-Attention sub-layer ---
    attn_out = self_attention(X, params["W_q"], params["W_k"], params["W_v"])
    X = residual_connection(X, attn_out)
    X = layer_norm(X)

    # --- MLP sub-layer (applied independently per token - no cross-token mixing) ---
    mlp_out = simple_mlp(X, params["W1"], params["b1"], params["W2"], params["b2"])
    X = residual_connection(X, mlp_out)
    X = layer_norm(X)

    return X


def run_transformer_block_demo():
    print()
    print("=" * 65)
    print("Check 2: A full mini Transformer block, 3 tokens, dim=4")
    print("=" * 65)
    rng = np.random.default_rng(5)
    n_tokens, dim = 3, 4
    X = rng.normal(0, 1, (n_tokens, dim))

    params = {
        "W_q": rng.normal(0, 0.5, (dim, dim)),
        "W_k": rng.normal(0, 0.5, (dim, dim)),
        "W_v": rng.normal(0, 0.5, (dim, dim)),
        "W1": rng.normal(0, 0.5, (dim, dim * 2)),
        "b1": np.zeros(dim * 2),
        "W2": rng.normal(0, 0.5, (dim * 2, dim)),
        "b2": np.zeros(dim),
    }

    print("Input tokens:\n", np.round(X, 3))
    output = transformer_block(X, params)
    print("Output tokens (after full Transformer block):\n", np.round(output, 3))
    print("\nEach output row's mean should be ~0 and std ~1 (Layer Norm's effect):")
    print("Row means:", np.round(output.mean(axis=1), 6))
    print("Row stds: ", np.round(output.std(axis=1), 6))


# ---------------------------------------------------------------------------
# PART 4: ViT patch counter
# ---------------------------------------------------------------------------
def vit_patch_info(image_size, patch_size, channels=3):
    patches_per_side = image_size // patch_size
    total_patches = patches_per_side ** 2
    flattened_dim = patch_size * patch_size * channels
    return total_patches, flattened_dim


def check_vit_patches():
    print()
    print("=" * 65)
    print("Check 3: ViT patch counter (Worked Example 1)")
    print("=" * 65)
    total, dim = vit_patch_info(224, 16)
    print(f"224x224 image, 16x16 patches -> {total} patches, each {dim}-dim  (expected 196, 768)")


# ---------------------------------------------------------------------------
# PART 5: Decoder autoregressive step simulator
# ---------------------------------------------------------------------------
def simulate_decoding(target_words):
    print()
    print("=" * 65)
    print(f"Check 4: Decoder autoregressive simulation for {target_words}")
    print("=" * 65)
    generated = ["<start>"]
    for step, next_word in enumerate(target_words + ["<end>"], start=1):
        print(f"Step {step}: input={' '.join(generated)}  ->  predicts: '{next_word}'")
        generated.append(next_word)
    print(f"\nTotal steps: {len(target_words) + 1}  (expected {len(target_words)+1})")


if __name__ == "__main__":
    check_residual_and_layernorm()
    run_transformer_block_demo()
    check_vit_patches()
    simulate_decoding(["I", "am", "happy"])
