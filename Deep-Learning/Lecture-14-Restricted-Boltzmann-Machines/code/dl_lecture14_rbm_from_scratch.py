"""
DL Lecture 14 (Bonus) - A Restricted Boltzmann Machine Built From Scratch
============================================================================
What this file does (in plain words):
  Implements the RBM energy function, hidden/visible activation formulas,
  Gibbs sampling, and Contrastive Divergence (CD-k) training - checked
  against the numerical README - then trains a tiny real RBM on a toy
  binary pattern dataset and shows it learning to reconstruct that pattern.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# PART 1: Energy function and activation formulas
# ---------------------------------------------------------------------------
def rbm_energy(v, h, a, b, W):
    """Implements: E(v,h) = -a.v - b.h - v^T W h"""
    return -(a @ v) - (b @ h) - (v @ W @ h)


def hidden_prob(v, b, W):
    """Implements: P(h_j=1|v) = sigmoid(b_j + sum_i v_i*W_ij)"""
    return sigmoid(b + v @ W)


def visible_prob(h, a, W):
    """Implements: P(v_i=1|h) = sigmoid(a_i + sum_j W_ij*h_j)"""
    return sigmoid(a + W @ h)


def check_energy_and_activations():
    print("=" * 65)
    print("Check 1: Energy + activations (Worked Examples 1, 2, 3)")
    print("=" * 65)
    v = np.array([1, 0, 1], dtype=float)
    h = np.array([1, 1], dtype=float)
    a = np.array([0.2, -0.1, 0.3])
    b = np.array([0.1, -0.2])
    W = np.array([[0.5, -0.3], [0.2, 0.4], [-0.1, 0.6]])

    E = rbm_energy(v, h, a, b, W)
    print(f"E(v,h) = {E:.4f}  (expected -1.1)")

    ph = hidden_prob(v, b, W)
    print(f"P(h=1|v) = {np.round(ph, 4)}  (expected ~[0.6225, 0.5250])")

    pv = visible_prob(h, a, W)
    print(f"P(v=1|h) = {np.round(pv, 4)}  (expected ~[0.5987, 0.6225, 0.6900])")


# ---------------------------------------------------------------------------
# PART 2: Gibbs sampling
# ---------------------------------------------------------------------------
def gibbs_step(v, a, b, W, rng):
    """One full v->h->v Gibbs step. Returns (h_sample, v_sample, h_prob, v_prob)."""
    h_prob = hidden_prob(v, b, W)
    h_sample = (rng.random(len(h_prob)) < h_prob).astype(float)
    v_prob = visible_prob(h_sample, a, W)
    v_sample = (rng.random(len(v_prob)) < v_prob).astype(float)
    return h_sample, v_sample, h_prob, v_prob


# ---------------------------------------------------------------------------
# PART 3: Contrastive Divergence training (CD-k)
# ---------------------------------------------------------------------------
class RBM:
    def __init__(self, n_visible, n_hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden)
        self.W = rng.normal(0, 0.1, (n_visible, n_hidden))
        self.rng = rng

    def train_step(self, v_data, k=1, lr=0.1):
        """One CD-k update for a single training example v_data."""
        # Positive phase
        h_data_prob = hidden_prob(v_data, self.b, self.W)
        h_data = (self.rng.random(len(h_data_prob)) < h_data_prob).astype(float)

        # Negative phase: run k Gibbs steps starting from h_data
        v_k = v_data.copy()
        h_k = h_data.copy()
        for _ in range(k):
            v_prob = visible_prob(h_k, self.a, self.W)
            v_k = (self.rng.random(len(v_prob)) < v_prob).astype(float)
            h_prob = hidden_prob(v_k, self.b, self.W)
            h_k = (self.rng.random(len(h_prob)) < h_prob).astype(float)

        # CD-k weight/bias updates
        pos = np.outer(v_data, h_data_prob)
        neg = np.outer(v_k, hidden_prob(v_k, self.b, self.W))
        self.W += lr * (pos - neg)
        self.a += lr * (v_data - v_k)
        self.b += lr * (h_data_prob - hidden_prob(v_k, self.b, self.W))

    def reconstruct(self, v):
        h_prob = hidden_prob(v, self.b, self.W)
        h = (self.rng.random(len(h_prob)) < h_prob).astype(float)
        v_prob = visible_prob(h, self.a, self.W)
        return v_prob


def run_rbm_training_demo():
    print()
    print("=" * 65)
    print("Check 2: Training a tiny RBM to reconstruct a toy binary pattern")
    print("=" * 65)
    # Toy dataset: two repeated binary patterns (like two "moods" from the theory story)
    pattern_A = np.array([1, 1, 0, 0, 1], dtype=float)
    pattern_B = np.array([0, 0, 1, 1, 0], dtype=float)
    dataset = [pattern_A, pattern_B]

    rbm = RBM(n_visible=5, n_hidden=3, seed=1)

    n_epochs = 3000
    for epoch in range(n_epochs):
        v = dataset[epoch % len(dataset)]
        rbm.train_step(v, k=1, lr=0.05)

        if epoch % 750 == 0:
            recon_A = rbm.reconstruct(pattern_A)
            recon_B = rbm.reconstruct(pattern_B)
            err_A = np.mean(np.abs(recon_A - pattern_A))
            err_B = np.mean(np.abs(recon_B - pattern_B))
            print(f"Epoch {epoch:4d} | recon(A)={np.round(recon_A,2)} err={err_A:.3f} | "
                  f"recon(B)={np.round(recon_B,2)} err={err_B:.3f}")

    print(f"\nPattern A = {pattern_A}")
    print(f"Pattern B = {pattern_B}")
    print("If training worked, reconstruction probabilities should be close to")
    print("the original patterns' 0/1 values (high where pattern=1, low where pattern=0).")


if __name__ == "__main__":
    check_energy_and_activations()
    run_rbm_training_demo()
