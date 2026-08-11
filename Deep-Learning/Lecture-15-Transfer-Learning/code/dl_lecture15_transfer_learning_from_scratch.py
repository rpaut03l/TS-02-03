"""
DL Lecture 15 (Bonus) - Transfer Learning Mechanics From Scratch (NumPy only)
================================================================================
What this file does (in plain words):
  Implements the parameter-counting and learning-rate-scaling formulas from
  the numerical README, then builds a genuinely trainable toy demo: a small
  "pretrained" 2-layer network is first trained on a SOURCE task, then reused
  on a DIFFERENT (but related) TARGET task using both Feature Extraction and
  Fine-Tuning, so you can directly compare how each strategy performs.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


# ---------------------------------------------------------------------------
# PART 1: Parameter counting and learning-rate scaling formulas
# ---------------------------------------------------------------------------
def head_params(input_dim, num_classes):
    """Implements: (input_dim x num_classes) + num_classes (bias)"""
    return input_dim * num_classes + num_classes


def check_parameter_counting():
    print("=" * 65)
    print("Check 1: Parameter counting (Worked Example 1)")
    print("=" * 65)
    backbone_params = 11_700_000
    head = head_params(512, 10)
    print(f"Head params: {head}  (expected 5130)")
    print(f"Feature Extraction trainable: {head}")
    full = backbone_params + head
    print(f"Full fine-tuning trainable: {full:,}  (expected 11,705,130)")
    print(f"Ratio: {full/head:,.1f}x  (expected ~2281.7x)")


def check_lr_scaling():
    print()
    print("=" * 65)
    print("Check 2: Learning rate scaling (Worked Example 2)")
    print("=" * 65)
    base_lr = 0.01
    print(f"Gentle fine-tune LR (/100): {base_lr/100}  (expected 0.0001)")
    print(f"Moderate fine-tune LR (/10): {base_lr/10}  (expected 0.001)")


def layerwise_lr_schedule(lr_first, num_blocks, multiplier):
    return [lr_first * (multiplier ** i) for i in range(num_blocks)]


def check_layerwise_schedule():
    print()
    print("=" * 65)
    print("Check 3: Layer-wise LR schedule (Worked Example 4)")
    print("=" * 65)
    rates = layerwise_lr_schedule(0.00005, 4, 2)
    print(f"Block rates: {[round(r,5) for r in rates]}  (expected [5e-5, 1e-4, 2e-4, 4e-4])")
    print(f"Ratio (last/first): {rates[-1]/rates[0]}  (expected 8.0)")


# ---------------------------------------------------------------------------
# PART 2: A genuinely trainable transfer learning demo
#   SOURCE task: classify points as "above" or "below" the line y=x
#   TARGET task: classify points as "above" or "below" the line y=x+1
#   (related but shifted - a small, clean stand-in for domain shift)
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


class TwoLayerNet:
    """A small 2-layer network: input -> hidden (frozen-able) -> output head."""

    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.5, (hidden_dim, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        out = sigmoid(z2)
        return z1, h1, out

    def train_step(self, X, y, lr_backbone, lr_head):
        """One gradient descent step. lr_backbone=0 freezes the backbone (Feature Extraction)."""
        n = X.shape[0]
        z1, h1, out = self.forward(X)
        y = y.reshape(-1, 1)

        d_out = (out - y) / n                      # MSE-style gradient (simplified)
        d_W2 = h1.T @ d_out
        d_b2 = d_out.sum(axis=0)

        d_h1 = d_out @ self.W2.T
        d_z1 = d_h1 * (z1 > 0)                       # ReLU derivative
        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # Head always updates with lr_head; backbone updates with lr_backbone (0 = frozen)
        self.W2 -= lr_head * d_W2
        self.b2 -= lr_head * d_b2
        self.W1 -= lr_backbone * d_W1
        self.b1 -= lr_backbone * d_b1

    def accuracy(self, X, y):
        _, _, out = self.forward(X)
        preds = (out.flatten() > 0.5).astype(float)
        return np.mean(preds == y)


def make_dataset(n, offset, rng):
    X = rng.normal(0, 2, (n, 2))
    y = (X[:, 1] > X[:, 0] + offset).astype(float)   # label: above the shifted line
    return X, y


def run_transfer_learning_demo():
    print()
    print("=" * 65)
    print("Check 4: Real transfer learning demo (source task -> target task)")
    print("=" * 65)
    rng = np.random.default_rng(3)

    # --- Step 1: "Pretrain" on the SOURCE task (line y = x) ---
    X_source, y_source = make_dataset(500, offset=0.0, rng=rng)
    net = TwoLayerNet(input_dim=2, hidden_dim=8, seed=5)
    for _ in range(400):
        net.train_step(X_source, y_source, lr_backbone=0.05, lr_head=0.05)
    print(f"Pretrained (source task) accuracy: {net.accuracy(X_source, y_source):.3f}")

    # --- Step 2: TARGET task (line y = x+1, a shifted/related task = mild domain shift) ---
    # Deliberately SMALL target dataset (20 examples) - this is exactly the data-scarce
    # regime where transfer learning's real advantage shows up (see Worked Example 3).
    X_target, y_target = make_dataset(20, offset=1.0, rng=rng)
    X_target_test, y_target_test = make_dataset(300, offset=1.0, rng=rng)

    # --- Option A: Feature Extraction (freeze backbone, only train the head) ---
    net_fe = TwoLayerNet(input_dim=2, hidden_dim=8, seed=5)
    net_fe.W1, net_fe.b1 = net.W1.copy(), net.b1.copy()   # copy pretrained backbone
    net_fe.W2, net_fe.b2 = net.W2.copy(), net.b2.copy()
    for _ in range(200):
        net_fe.train_step(X_target, y_target, lr_backbone=0.0, lr_head=0.1)   # backbone frozen
    acc_fe = net_fe.accuracy(X_target_test, y_target_test)

    # --- Option B: Fine-Tuning (small LR on the whole network) ---
    net_ft = TwoLayerNet(input_dim=2, hidden_dim=8, seed=5)
    net_ft.W1, net_ft.b1 = net.W1.copy(), net.b1.copy()
    net_ft.W2, net_ft.b2 = net.W2.copy(), net.b2.copy()
    for _ in range(200):
        net_ft.train_step(X_target, y_target, lr_backbone=0.01, lr_head=0.05)  # gentle full fine-tune

    acc_ft = net_ft.accuracy(X_target_test, y_target_test)

    # --- Baseline: train from scratch on the (small) target dataset only ---
    net_scratch = TwoLayerNet(input_dim=2, hidden_dim=8, seed=99)
    for _ in range(200):
        net_scratch.train_step(X_target, y_target, lr_backbone=0.05, lr_head=0.05)
    acc_scratch = net_scratch.accuracy(X_target_test, y_target_test)

    print(f"\nTarget task test accuracy (only 20 training examples - a data-scarce regime,")
    print(f"tested on 300 held-out examples):")
    print(f"  Feature Extraction (backbone frozen):  {acc_fe:.3f}")
    print(f"  Fine-Tuning (backbone gently updated):  {acc_ft:.3f}")
    print(f"  From scratch (random init, same data):  {acc_scratch:.3f}")
    print("\nNote: this toy 2D task is simple enough that even 20 random examples are")
    print("often sufficient regardless of starting point - see this file's README for")
    print("an honest discussion of what this result does (and doesn't) demonstrate.")


if __name__ == "__main__":
    check_parameter_counting()
    check_lr_scaling()
    check_layerwise_schedule()
    run_transfer_learning_demo()
