"""
DL Lecture 08 - L2 Regression, L1/L2 Penalties, Dropout, and Early Stopping
==============================================================================
What this file does (in plain words):
  Implements four separate regularization techniques from the theory file as
  small functions, each checked against the numerical README's worked
  examples: (1) closed-form L2-regularized regression, (2) L1 vs L2 penalty
  computation, (3) a dropout forward pass (train and test time), and
  (4) an early-stopping epoch finder.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


# ---------------------------------------------------------------------------
# PART 1: Closed-form L2-regularized regression (1D, for hand-traceability)
# ---------------------------------------------------------------------------
def l2_regularized_weight_1d(x, y, lam):
    """
    Implements: w = sum(x_i*y_i) / (sum(x_i^2) + lambda)
    Setting lam=0 recovers the plain (unregularized) closed-form solution.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    numerator = np.sum(x * y)
    denominator = np.sum(x ** 2) + lam
    return numerator / denominator


def check_l2_regression():
    print("=" * 65)
    print("Check 1: Closed-form L2 regression (Worked Example 1)")
    print("=" * 65)
    x = [1, 2, 3]
    y = [2, 4, 5]
    w_unreg = l2_regularized_weight_1d(x, y, lam=0)
    w_reg = l2_regularized_weight_1d(x, y, lam=2)
    print(f"w (unregularized, lambda=0): {w_unreg:.4f}  (expected ~1.7857)")
    print(f"w (regularized, lambda=2):   {w_reg:.4f}  (expected exactly 1.5625)")


# ---------------------------------------------------------------------------
# PART 2: L1 vs L2 penalty computation
# ---------------------------------------------------------------------------
def l1_penalty(w):
    """Implements: sum(|w_j|)"""
    return np.sum(np.abs(w))


def l2_penalty(w):
    """Implements: 0.5 * sum(w_j^2)"""
    return 0.5 * np.sum(np.square(w))


def check_l1_vs_l2():
    print()
    print("=" * 65)
    print("Check 2: L1 vs L2 penalty (Worked Example 2)")
    print("=" * 65)
    w = np.array([3, -1, 0.5, -4])
    print(f"L1 penalty: {l1_penalty(w)}  (expected 8.5)")
    print(f"L2 penalty: {l2_penalty(w)}  (expected 13.125)")

    print()
    print("Bonus: concentrated vs spread-out weights, same total L1 magnitude")
    w_concentrated = np.array([4, 0, 0])
    w_spread = np.array([2, 1, 1])
    print(f"L1 total for both: {l1_penalty(w_concentrated)} and {l1_penalty(w_spread)} (equal)")
    print(f"L2 penalty, concentrated {w_concentrated}: {l2_penalty(w_concentrated)}")
    print(f"L2 penalty, spread out {w_spread}: {l2_penalty(w_spread)}")
    print("-> L2 punishes the CONCENTRATED large weight far more heavily.")


# ---------------------------------------------------------------------------
# PART 3: Dropout - training pass (masked) and test pass (scaled)
# ---------------------------------------------------------------------------
def dropout_train(activations, p, rng=None, mask=None):
    """
    Training-time dropout: randomly zero out each activation with
    probability (1-p), i.e. RETAIN with probability p.
    If `mask` is provided explicitly, uses it instead of sampling (for
    reproducing a specific worked example).
    """
    activations = np.array(activations, dtype=float)
    if mask is None:
        if rng is None:
            rng = np.random.default_rng()
        mask = (rng.random(len(activations)) < p).astype(float)
    else:
        mask = np.array(mask, dtype=float)
    return mask * activations, mask


def dropout_test(activations, p):
    """Test-time dropout: use the FULL network, scaled by p."""
    activations = np.array(activations, dtype=float)
    return p * activations


def check_dropout():
    print()
    print("=" * 65)
    print("Check 3: Dropout train vs test (Worked Examples 3 & 4)")
    print("=" * 65)
    activations = [1.0, 2.0, 3.0, 4.0]
    train_output, mask = dropout_train(activations, p=0.5, mask=[1, 0, 1, 1])
    test_output = dropout_test(activations, p=0.5)
    print(f"Mask used: {mask}")
    print(f"Training-time output: {train_output}  (expected [1.0, 0.0, 3.0, 4.0])")
    print(f"Test-time output:     {test_output}  (expected [0.5, 1.0, 1.5, 2.0])")

    print()
    print("Bonus: number of thinned networks for n=10 nodes:", 2 ** 10, " (expected 1024)")


# ---------------------------------------------------------------------------
# PART 4: Early stopping - find the epoch with minimum validation loss
# ---------------------------------------------------------------------------
def find_early_stopping_epoch(validation_losses):
    """Returns (best_epoch_1_indexed, best_loss)."""
    losses = np.array(validation_losses)
    best_index = int(np.argmin(losses))   # 0-indexed
    return best_index + 1, losses[best_index]


def check_early_stopping():
    print()
    print("=" * 65)
    print("Check 4: Early stopping epoch finder (Worked Example 6)")
    print("=" * 65)
    val_losses = [0.90, 0.70, 0.50, 0.42, 0.40, 0.41, 0.45, 0.50]
    epoch, loss = find_early_stopping_epoch(val_losses)
    print(f"Validation losses: {val_losses}")
    print(f"Best epoch: {epoch}  (expected 5)")
    print(f"Best validation loss: {loss}  (expected 0.40)")


if __name__ == "__main__":
    check_l2_regression()
    check_l1_vs_l2()
    check_dropout()
    check_early_stopping()
