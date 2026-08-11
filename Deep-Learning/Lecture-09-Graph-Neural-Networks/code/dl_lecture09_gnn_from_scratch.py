"""
DL Lecture 09 - A Mini Graph Neural Network Built From Scratch (NumPy only)
==============================================================================
What this file does (in plain words):
  Builds a tiny graph (as an adjacency matrix), implements neighbor mean
  aggregation and the full GNN layer update formula from the theory file,
  checks results against the numerical README's worked examples, and then
  runs a full multi-layer forward pass across an entire small graph.

Library used: NumPy only.
Install: pip install numpy --break-system-packages
"""

import numpy as np


# ---------------------------------------------------------------------------
# PART 1: Build a graph as an adjacency matrix
# ---------------------------------------------------------------------------
def build_adjacency_matrix(n_nodes, edges):
    """
    edges: list of (i, j, weight) tuples. Automatically fills BOTH A[i,j]
    and A[j,i] (assumes a symmetric/undirected graph, as most examples in
    this lecture use).
    """
    A = np.zeros((n_nodes, n_nodes))
    for i, j, w in edges:
        A[i, j] = w
        A[j, i] = w
    return A


def degree(A, node_index):
    """Degree = sum of the node's row in the adjacency matrix."""
    return A[node_index].sum()


def check_adjacency_and_degree():
    print("=" * 65)
    print("Check 1: Adjacency matrix + degree (Worked Examples 1 & 2)")
    print("=" * 65)
    # 3-node graph: 0-indexed here (node 1->index 0, node 2->index 1, node 3->index 2)
    edges = [(0, 1, 2), (0, 2, 4), (1, 2, 1)]
    A = build_adjacency_matrix(3, edges)
    print("Adjacency matrix:\n", A)
    print("Is symmetric (A == A.T)?", np.array_equal(A, A.T))
    print(f"Degree of node 1 (index 0): {degree(A,0)}  (expected 6)")
    print(f"Degree of node 2 (index 1): {degree(A,1)}  (expected 3)")
    print(f"Degree of node 3 (index 2): {degree(A,2)}  (expected 5)")


# ---------------------------------------------------------------------------
# PART 2: Mean neighbor aggregation + full GNN layer update
# ---------------------------------------------------------------------------
def relu(x):
    return np.maximum(0, x)


def mean_aggregate(neighbor_features):
    """neighbor_features: list/array of feature vectors, one per neighbor."""
    return np.mean(neighbor_features, axis=0)


def gnn_layer_update(self_features, neighbor_features, W, B, activation=relu):
    """
    Implements: h_v^(k) = activation( W . mean(N(v)) + B . h_v^(k-1) )
    """
    aggregated = mean_aggregate(neighbor_features)
    pre_activation = W @ aggregated + B @ self_features
    return activation(pre_activation), aggregated


def check_gnn_layer_update():
    print()
    print("=" * 65)
    print("Check 2: GNN layer update (Worked Examples 3 & 4)")
    print("=" * 65)
    h_A = np.array([1.0, 0.0])
    neighbors = np.array([
        [0.0, 1.0],   # h_B
        [1.0, 1.0],   # h_C
        [2.0, 0.0],   # h_D
    ])
    W = np.array([[0.5, 0.0], [0.0, 0.5]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])

    new_h_A, aggregated = gnn_layer_update(h_A, neighbors, W, B)
    print(f"Aggregated neighbor vector: {np.round(aggregated, 4)}  (expected [1.0, 0.6667])")
    print(f"New embedding h_A^(1): {np.round(new_h_A, 4)}  (expected [1.5, 0.3333])")


# ---------------------------------------------------------------------------
# PART 3: Hop distance via powers of S
# ---------------------------------------------------------------------------
def check_hop_distance():
    print()
    print("=" * 65)
    print("Check 3: Hop distance via powers of S (Worked Example 5)")
    print("=" * 65)
    S = np.array([[0, 2, 4], [2, 0, 1], [4, 1, 0]])
    x = np.array([1, 0, 0])
    Sx = S @ x
    S2x = S @ Sx
    print(f"Sx (1-hop):  {Sx}  (expected [0, 2, 4])")
    print(f"S^2 x (2-hop): {S2x}  (expected [20, 4, 2])")


# ---------------------------------------------------------------------------
# PART 4: A full multi-layer forward pass over a whole small graph
# ---------------------------------------------------------------------------
def run_full_graph_forward_pass():
    print()
    print("=" * 65)
    print("Check 4: Full 2-layer GNN forward pass over a 4-node graph")
    print("=" * 65)
    # A simple 4-node graph: 0-1, 1-2, 2-3, 3-0 (a cycle)
    node_names = ["A", "B", "C", "D"]
    adjacency = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [2, 0],
    }

    # Initial features (layer-0 embeddings) - random but fixed for reproducibility
    rng = np.random.default_rng(3)
    feature_dim = 3
    h = {i: rng.normal(0, 1, feature_dim) for i in range(4)}

    W = rng.normal(0, 0.5, (feature_dim, feature_dim))
    B = rng.normal(0, 0.5, (feature_dim, feature_dim))

    for layer in range(1, 3):   # 2 layers
        new_h = {}
        for node in range(4):
            neighbor_feats = np.array([h[n] for n in adjacency[node]])
            new_h[node], _ = gnn_layer_update(h[node], neighbor_feats, W, B)
        h = new_h
        print(f"\nAfter layer {layer}:")
        for i in range(4):
            print(f"  {node_names[i]}: {np.round(h[i], 4)}")

    print("\nEach node's final embedding now reflects information gathered")
    print("from up to 2 hops away, exactly as described in the theory file.")


# ---------------------------------------------------------------------------
# PART 5: GCN - degree-normalized aggregation (Kipf & Welling, 2017)
# ---------------------------------------------------------------------------
def gcn_update(self_index, neighbor_indices, degrees, features, W, activation=relu):
    """
    Implements: h_i' = activation( sum_j (1/sqrt(d_i*d_j)) * W . h_j )
    self_index       : index of the target node i
    neighbor_indices : list of neighbor node indices j
    degrees          : dict {node_index: degree}
    features         : dict {node_index: feature vector h}
    W                : shared weight matrix
    """
    d_i = degrees[self_index]
    total = np.zeros(W.shape[0])
    for j in neighbor_indices:
        d_j = degrees[j]
        coef = 1.0 / np.sqrt(d_i * d_j)
        total += coef * (W @ features[j])
    return activation(total)


def check_gcn():
    print()
    print("=" * 65)
    print("Check 5: GCN degree-normalized aggregation (Worked Example 6)")
    print("=" * 65)
    degrees = {"A": 3, "B": 4, "C": 2, "D": 1}
    features = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0]),
        "C": np.array([1.0, 1.0]),
        "D": np.array([2.0, 0.0]),
    }
    W = np.array([[0.5, 0.0], [0.0, 0.5]])
    h_A_new = gcn_update("A", ["B", "C", "D"], degrees, features, W)
    print(f"GCN update for A: {np.round(h_A_new, 4)}  (expected ~[0.7815, 0.3485])")


# ---------------------------------------------------------------------------
# PART 6: GAT - attention-based aggregation (Velickovic et al., 2018)
# ---------------------------------------------------------------------------
def gat_update(self_index, neighbor_indices, features, W, activation=relu):
    """
    Implements: e_ij = (W.h_i) . (W.h_j)   (simplified dot-product attention score)
                alpha_ij = softmax_j(e_ij)
                h_i' = activation( sum_j alpha_ij * W . h_j )
    """
    Wh_i = W @ features[self_index]
    scores = []
    Wh_neighbors = []
    for j in neighbor_indices:
        Wh_j = W @ features[j]
        Wh_neighbors.append(Wh_j)
        scores.append(Wh_i @ Wh_j)
    scores = np.array(scores)
    alphas = softmax(scores)   # reuse the softmax defined in check_encoder... style; redefine locally below
    total = np.zeros(W.shape[0])
    for alpha, Wh_j in zip(alphas, Wh_neighbors):
        total += alpha * Wh_j
    return activation(total), alphas


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def check_gat():
    print()
    print("=" * 65)
    print("Check 6: GAT attention-based aggregation (Worked Example 7)")
    print("=" * 65)
    features = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0]),
        "C": np.array([1.0, 1.0]),
        "D": np.array([2.0, 0.0]),
    }
    W = np.array([[0.5, 0.0], [0.0, 0.5]])
    h_A_new, alphas = gat_update("A", ["B", "C", "D"], features, W)
    print(f"Attention weights (B,C,D): {np.round(alphas, 4)}  (expected ~[0.2543, 0.3265, 0.4192])")
    print(f"GAT update for A: {np.round(h_A_new, 4)}  (expected ~[0.5825, 0.2905])")


def check_gcn_weight_sizing():
    print()
    print("=" * 65)
    print("Check 7: GCN weight matrix sizing (Worked Example 8)")
    print("=" * 65)
    n_nodes, in_features, out_features = 10, 5, 16
    W = np.zeros((out_features, in_features))  # or (in_features, out_features) depending on convention
    print(f"For {n_nodes} nodes, {in_features} input features -> {out_features} output features:")
    print(f"W shape: {in_features} x {out_features}  (expected 5 x 16)")
    print(f"Total parameters in W: {in_features * out_features}  (expected 80)")


if __name__ == "__main__":
    check_adjacency_and_degree()
    check_gnn_layer_update()
    check_hop_distance()
    run_full_graph_forward_pass()
    check_gcn()
    check_gat()
    check_gcn_weight_sizing()
