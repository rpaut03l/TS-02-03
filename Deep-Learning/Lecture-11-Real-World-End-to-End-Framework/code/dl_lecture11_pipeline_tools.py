"""
DL Lecture 11 - Pipeline Tools: Correlation, Ablation, Bagging, Boosting, Stacking
=====================================================================================
What this file does (in plain words):
  Implements the numeric tools behind Lecture 11's six-step pipeline and its
  ensembling methods - Pearson correlation, grid search combination counting,
  ablation delta ranking, bagging variance reduction, an AdaBoost weight
  update, and a stacking meta-model combination - all checked against the
  numerical README.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np
import itertools


# ---------------------------------------------------------------------------
# PART 1: Feature correlation (Pearson r)
# ---------------------------------------------------------------------------
def pearson_correlation(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mean_x, mean_y = x.mean(), y.mean()
    covariance_term = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))
    return covariance_term / (std_x * std_y)


def check_correlation():
    print("=" * 65)
    print("Check 1: Feature correlation (Worked Example 1)")
    print("=" * 65)
    r = pearson_correlation([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
    print(f"Pearson r: {r:.4f}  (expected ~0.7746)")


# ---------------------------------------------------------------------------
# PART 2: Grid search combination counter
# ---------------------------------------------------------------------------
def count_grid_combinations(*hyperparameter_lists):
    """Counts (and can enumerate) all combinations of a hyperparameter grid."""
    total = 1
    for values in hyperparameter_lists:
        total *= len(values)
    return total


def check_grid_search():
    print()
    print("=" * 65)
    print("Check 2: Grid search combination count (Worked Example 2)")
    print("=" * 65)
    lr = [0.001, 0.01, 0.1]
    batch = [16, 32, 64, 128]
    layers = [2, 4, 6]
    total = count_grid_combinations(lr, batch, layers)
    print(f"3 learning rates x 4 batch sizes x 3 layer counts = {total} combinations  (expected 36)")


# ---------------------------------------------------------------------------
# PART 3: Ablation study delta ranking
# ---------------------------------------------------------------------------
def rank_ablation_deltas(full_score, ablated_scores: dict):
    """
    full_score: the full model's score
    ablated_scores: {component_name: score_without_that_component}
    Returns a list of (component, delta) sorted by delta descending (most important first).
    """
    deltas = {name: full_score - score for name, score in ablated_scores.items()}
    ranked = sorted(deltas.items(), key=lambda item: item[1], reverse=True)
    return ranked


def check_ablation():
    print()
    print("=" * 65)
    print("Check 3: Ablation study ranking (Worked Example 3)")
    print("=" * 65)
    ranked = rank_ablation_deltas(
        full_score=91.2,
        ablated_scores={"dropout": 87.6, "augmentation": 88.1, "attention": 84.5},
    )
    for name, delta in ranked:
        print(f"  {name}: delta = {delta:.1f} percentage points")
    print("(expected order: attention 6.7, dropout 3.6, augmentation 3.1)")


# ---------------------------------------------------------------------------
# PART 4: Bagging variance reduction
# ---------------------------------------------------------------------------
def bagged_variance(single_model_variance, n_models):
    """Implements: variance / n (assumes independent models)"""
    return single_model_variance / n_models


def check_bagging():
    print()
    print("=" * 65)
    print("Check 4: Bagging variance reduction (Worked Example 4)")
    print("=" * 65)
    for n in [1, 5, 20]:
        v = bagged_variance(4.0, n)
        print(f"  n={n:2d} models -> variance = {v}")


# ---------------------------------------------------------------------------
# PART 5: AdaBoost-style weight update
# ---------------------------------------------------------------------------
def adaboost_weight_update(sample_weights, is_correct):
    """
    sample_weights: array of current weights (should sum to 1)
    is_correct: boolean array, True = correctly classified, False = misclassified
    Returns (alpha, new_normalized_weights)
    """
    sample_weights = np.array(sample_weights, dtype=float)
    is_correct = np.array(is_correct, dtype=bool)
    error_rate = np.sum(sample_weights[~is_correct])   # sum of weights of WRONG samples
    alpha = 0.5 * np.log((1 - error_rate) / error_rate)

    updated = np.where(
        is_correct,
        sample_weights * np.exp(-alpha),
        sample_weights * np.exp(alpha),
    )
    normalized = updated / updated.sum()
    return alpha, normalized


def check_boosting():
    print()
    print("=" * 65)
    print("Check 5: AdaBoost weight update (Worked Example 5)")
    print("=" * 65)
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    is_correct = [True, True, True, False, False]   # 2 wrong out of 5, matches e=0.4
    alpha, new_weights = adaboost_weight_update(weights, is_correct)
    print(f"alpha: {alpha:.4f}  (expected ~0.2027)")
    print(f"new weights: {np.round(new_weights, 4)}  (expected [~0.1667]*3 + [~0.25]*2)")


# ---------------------------------------------------------------------------
# PART 6: Stacking meta-model
# ---------------------------------------------------------------------------
def stacking_combine(predictions, weights):
    predictions = np.array(predictions, dtype=float)
    weights = np.array(weights, dtype=float)
    return np.sum(predictions * weights)


def check_stacking():
    print()
    print("=" * 65)
    print("Check 6: Stacking meta-model combination (Worked Example 6)")
    print("=" * 65)
    predictions = [0.72, 0.65, 0.81]   # CNN, LSTM, GBT
    weights = [0.5, 0.2, 0.3]
    stacked = stacking_combine(predictions, weights)
    plain_avg = np.mean(predictions)
    print(f"Stacked (learned weights): {stacked:.4f}  (expected 0.733)")
    print(f"Plain average (equal weights): {plain_avg:.4f}  (expected ~0.7267)")


if __name__ == "__main__":
    check_correlation()
    check_grid_search()
    check_ablation()
    check_bagging()
    check_boosting()
    check_stacking()
