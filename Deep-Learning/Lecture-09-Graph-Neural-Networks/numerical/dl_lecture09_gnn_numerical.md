# DL Lecture 09 — Graph Neural Networks (Numerical)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

> Folder: `Deep-Learning/Lecture-09-Graph-Neural-Networks/numerical/`
> Pairs with: [`theory/dl_lecture09_gnn_theory.md`](../theory/dl_lecture09_gnn_theory.md) · [`practice/dl_lecture09_gnn_practice.md`](../practice/dl_lecture09_gnn_practice.md) · [`exercises/dl_lecture09_exercises.md`](../exercises/dl_lecture09_exercises.md)

---

## Table of Contents
1. [Notation Used in This File](#notation-used-in-this-file)
2. [Worked Example 1 — Building an Adjacency Matrix](#worked-example-1--building-an-adjacency-matrix)
3. [Worked Example 2 — Computing Degree](#worked-example-2--computing-degree)
4. [Worked Example 3 — Neighborhood Aggregation (Mean), By Hand](#worked-example-3--neighborhood-aggregation-mean-by-hand)
5. [Worked Example 4 — Full GNN Layer Update, By Hand](#worked-example-4--full-gnn-layer-update-by-hand)
6. [Worked Example 5 — Hop Distance via Powers of S](#worked-example-5--hop-distance-via-powers-of-s)
7. [Worked Example 6 — GCN's Degree-Normalized Aggregation](#worked-example-6--gcns-degree-normalized-aggregation)
8. [Worked Example 7 — GAT's Attention-Based Aggregation](#worked-example-7--gats-attention-based-aggregation)
9. [Worked Example 8 — The GCN Weight Matrix Sizing Question](#worked-example-8--the-gcn-weight-matrix-sizing-question)
10. [Master Formula Cheatsheet](#master-formula-cheatsheet)
11. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
12. [Summary](#summary)

---

## Notation Used in This File

| Symbol | Meaning |
|---|---|
| A (or S) | adjacency matrix / graph shift operator |
| N(v) | neighborhood of node v |
| h_v^(k) | embedding of node v after layer k |
| W_k, B_k | learned weight matrices at layer k (neighbor path, self path) |
| σ | non-linear activation (ReLU used in this file) |

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 1 — Building an Adjacency Matrix

**Given:** a 3-node symmetric weighted graph with edges: (1,2) weight 2, (1,3) weight 4, (2,3) weight 1.

**Step 1 — Set up a 3×3 matrix, all zeros initially.**
```
      node1  node2  node3
node1   0      0      0
node2   0      0      0
node3   0      0      0
```

**Step 2 — Fill in each edge's weight, in BOTH directions (since the graph is symmetric).**
```
A_12 = A_21 = 2
A_13 = A_31 = 4
A_23 = A_32 = 1
```

**Step 3 — Final adjacency matrix.**
```
A = [ 0  2  4 ]
    [ 2  0  1 ]
    [ 4  1  0 ]
```

**Result:** notice A is symmetric (A = Aᵀ), confirming the theory file's claim that symmetric graphs always produce symmetric adjacency matrices — you can literally check this by verifying the matrix looks the same when reflected across its main diagonal.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 2 — Computing Degree

**Given:** the same graph as Example 1. Compute the degree of node 1.

**Step 1 — Identify all edges incident to node 1.** From the adjacency matrix's row 1: `A_12=2`, `A_13=4`.

**Step 2 — Sum the weights.**
```
d_1 = A_12 + A_13 = 2 + 4 = 6
```

**Result: node 1's degree is 6.** (For comparison: node 2's degree = A_21+A_23 = 2+1 = 3; node 3's degree = A_31+A_32 = 4+1 = 5 — notice degree is simply the sum of a node's ROW in the adjacency matrix.)

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 3 — Neighborhood Aggregation (Mean), By Hand

**Given:** target node A, with neighbors B, C, D. Each node has a 2-dimensional feature vector:
```
h_A = [1.0, 0.0]
h_B = [0.0, 1.0]
h_C = [1.0, 1.0]
h_D = [2.0, 0.0]
```

**Step 1 — Sum the neighbor feature vectors (B, C, D — NOT including A itself).**
```
sum = h_B + h_C + h_D = [0.0+1.0+2.0, 1.0+1.0+0.0] = [3.0, 2.0]
```

**Step 2 — Divide by the number of neighbors (mean aggregation).**
```
mean(N(A)) = [3.0, 2.0] / 3 = [1.0, 0.6667]
```

**Result: the aggregated neighbor vector is [1.0, 0.6667].** This is a single, FIXED-SIZE summary of A's (potentially very differently-sized) neighborhood — exactly solving the "different nodes have different numbers of neighbors, but we need a fixed-size output" problem described in the theory file.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 4 — Full GNN Layer Update, By Hand

**Given:** continuing from Example 3, apply one full GNN layer update using the formula `h_v^(1) = σ(W·mean(N(v)) + B·h_v^(0))`, with illustrative weight matrices:
```
W = [[0.5, 0.0], [0.0, 0.5]]     (halves the aggregated neighbor vector)
B = [[1.0, 0.0], [0.0, 1.0]]     (identity - passes A's own features through unchanged)
```

**Step 1 — Apply W to the aggregated neighbor vector from Example 3.**
```
W . mean(N(A)) = [0.5x1.0 + 0.0x0.6667, 0.0x1.0 + 0.5x0.6667] = [0.5, 0.3333]
```

**Step 2 — Apply B to A's own previous embedding (h_A^(0) = [1.0, 0.0]).**
```
B . h_A^(0) = [1.0x1.0 + 0.0x0.0, 0.0x1.0 + 1.0x0.0] = [1.0, 0.0]
```

**Step 3 — Sum the two pathways.**
```
pre-activation = [0.5+1.0, 0.3333+0.0] = [1.5, 0.3333]
```

**Step 4 — Apply the non-linear activation (ReLU).**
```
h_A^(1) = max(0, [1.5, 0.3333]) = [1.5, 0.3333]     (both values already positive, so ReLU passes them through unchanged)
```

**Result: h_A^(1) = [1.5, 0.3333].** Node A's brand-new, layer-1 embedding now blends its OWN original features (via the B pathway) with a SUMMARY of its neighbors' features (via the W pathway) — exactly the "aggregate neighbors + combine with self" recipe from the theory file, fully quantified.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 5 — Hop Distance via Powers of S

**Given:** the adjacency matrix from Example 1, `S = [[0,2,4],[2,0,1],[4,1,0]]`, and a signal `x = [1, 0, 0]` (all the "signal" starts concentrated at node 1).

**Step 1 — Compute Sx (1-hop: information from node 1's direct neighbors).**
```
Sx = S . x = [0x1+2x0+4x0, 2x1+0x0+1x0, 4x1+1x0+0x0] = [0, 2, 4]
```
Notice: node 1 itself now has value 0 (it's not its own neighbor), while nodes 2 and 3 (node 1's direct neighbors) picked up values proportional to their connection strength to node 1 (2 and 4 respectively) — exactly "collecting information from 1-hop neighbors."

**Step 2 — Compute S²x = S(Sx) (2-hop: information from neighbors of neighbors).**
```
S(Sx) = S . [0,2,4] = [0x0+2x2+4x4, 2x0+0x2+1x4, 4x0+1x2+0x4] = [20, 4, 2]
```

**Result:** after 2 hops, the signal has spread back to include node 1 again (value 20 — it received signal back from both its neighbors, weighted by their own connection strengths) as well as further-diffused values at nodes 2 and 3. This numerically demonstrates the "Sx = 1-hop, S²x = 2-hop, S³x = even farther" progression from the theory file — each additional multiplication by S spreads the signal one hop further across the graph.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 6 — GCN's Degree-Normalized Aggregation

**Given:** target node A (degree d_A=3), connected to neighbors B (degree d_B=4), C (degree d_C=2), D (degree d_D=1). Feature vectors: `h_A=[1.0,0.0]`, `h_B=[0.0,1.0]`, `h_C=[1.0,1.0]`, `h_D=[2.0,0.0]`. Weight matrix `W=[[0.5,0],[0,0.5]]`.

**Step 1 — Compute each neighbor's normalization coefficient, `1/sqrt(d_A · d_neighbor)`.**
```
coef_B = 1/sqrt(3x4) = 1/sqrt(12) ≈ 0.2887
coef_C = 1/sqrt(3x2) = 1/sqrt(6)  ≈ 0.4082
coef_D = 1/sqrt(3x1) = 1/sqrt(3)  ≈ 0.5774
```
Notice: **D (lowest degree) gets the LARGEST coefficient**, and **B (highest degree) gets the SMALLEST coefficient** — GCN's normalization systematically down-weights contributions from high-degree "hub" neighbors (who are shared with many other nodes) and up-weights contributions from low-degree, more "exclusive" neighbors.

**Step 2 — Transform each neighbor's features by W, then scale by its coefficient.**
```
coef_B x (W.h_B) = 0.2887 x [0, 0.5]   = [0,      0.1443]
coef_C x (W.h_C) = 0.4082 x [0.5, 0.5] = [0.2041, 0.2041]
coef_D x (W.h_D) = 0.5774 x [1.0, 0]   = [0.5774, 0]
```

**Step 3 — Sum all contributions and apply ReLU.**
```
sum = [0+0.2041+0.5774, 0.1443+0.2041+0] = [0.7815, 0.3485]
h_A_new = ReLU([0.7815, 0.3485]) = [0.7815, 0.3485]
```

**Result: h_A_new ≈ [0.7815, 0.3485].** This is the exact GCN update — different from plain mean aggregation (Worked Example 3), because each neighbor's contribution is weighted by the degree-based coefficient rather than a flat `1/|N(v)|`.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 7 — GAT's Attention-Based Aggregation

**Given:** the same node A and neighbors B, C, D from Worked Example 6, with the same W. GAT computes raw attention scores as a simple dot product `e_ij = (W·h_i)·(W·h_j)` (a simplified stand-in for GAT's actual learned attention function `a`).

**Step 1 — Compute `W·h_i` for all four nodes.**
```
W.h_A = [0.5, 0.0]
W.h_B = [0.0, 0.5]
W.h_C = [0.5, 0.5]
W.h_D = [1.0, 0.0]
```

**Step 2 — Compute raw scores (dot products of A's transformed features against each neighbor's).**
```
e_AB = [0.5,0.0].[0.0,0.5] = 0.5x0.0 + 0.0x0.5 = 0.0
e_AC = [0.5,0.0].[0.5,0.5] = 0.5x0.5 + 0.0x0.5 = 0.25
e_AD = [0.5,0.0].[1.0,0.0] = 0.5x1.0 + 0.0x0.0 = 0.5
```

**Step 3 — Apply softmax to get attention weights.**
```
exp(0.0)=1.0, exp(0.25)≈1.2840, exp(0.5)≈1.6487
sum ≈ 1.0+1.2840+1.6487 = 3.9327
alpha_AB = 1.0/3.9327 ≈ 0.2543
alpha_AC = 1.2840/3.9327 ≈ 0.3265
alpha_AD = 1.6487/3.9327 ≈ 0.4192
```

**Step 4 — Compute the weighted sum of transformed neighbor features, then ReLU.**
```
h_A_new = alpha_AB x W.h_B + alpha_AC x W.h_C + alpha_AD x W.h_D
        = 0.2543x[0,0.5] + 0.3265x[0.5,0.5] + 0.4192x[1.0,0.0]
        = [0, 0.1272] + [0.1633,0.1633] + [0.4192,0]
        = [0.5825, 0.2905]
h_A_new = ReLU([0.5825,0.2905]) = [0.5825, 0.2905]
```

**Result: h_A_new ≈ [0.5825, 0.2905].** Compare to GCN's `[0.7815, 0.3485]` (Worked Example 6) — GAT arrived at a DIFFERENT weighting purely from the actual feature CONTENT of A, B, C, D (via the dot-product attention scores), completely independent of the nodes' degrees, which is exactly the "learned, adaptive importance" GAT is designed to provide.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Worked Example 8 — The GCN Weight Matrix Sizing Question

**Given (the lecture's own direct question):** a graph has 10 nodes, each with 5 input features. You want a GCN layer producing 16 new features per node, using one shared weight matrix W.

**Step 1 — Identify W's required input dimension.** W must accept each node's 5-dimensional feature vector: input dimension = **5**.

**Step 2 — Identify W's required output dimension.** W must produce the desired 16-dimensional new feature vector: output dimension = **16**.

**Step 3 — State W's shape.**
```
W in R^(5 x 16)
```

**Result: W ∈ R^(5×16)** — and critically, this SAME 5×16 matrix is applied identically at every one of the 10 nodes (parameter sharing, exactly like a CNN filter reused at every spatial position). Total learnable parameters in W: `5×16 = 80` (plus a bias vector of size 16, if included, for 96 total) — completely independent of how many nodes (10, 10,000, or 10 million) the graph actually has, exactly the same "parameter count doesn't scale with problem size" property seen in RNNs (Lecture 4) and LSTMs (Lecture 5).

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Master Formula Cheatsheet

| Scenario | Formula |
|---|---|
| Adjacency matrix entry | `A_ij = w_ij` for `(i,j)∈E` |
| Degree | `d_i = Σ_j A_ij` (sum of row i) |
| Mean neighbor aggregation | `mean(N(v)) = (1/|N(v)|) Σ_{u∈N(v)} h_u` |
| GNN layer update | `h_v^(k) = σ(W_k·mean(N(v)) + B_k·h_v^(k-1))` |
| k-hop signal spread | `S^k x` |
| GCN update | `h_i' = σ(Σ_j (1/√(d_i·d_j))·W·h_j)` |
| GAT attention score | `e_ij = a(W·h_i, W·h_j)` |
| GAT attention weight | `α_ij = softmax_j(e_ij) = exp(e_ij)/Σ_k exp(e_ik)` |
| GAT update | `h_i' = σ(Σ_j α_ij·W·h_j)` |
| GCN weight matrix shape | `W ∈ R^(input_dim × output_dim)` |

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Exam Hacks / Trap Watch

- **Trap:** including the node itself when computing the neighbor mean (unless the specific formulation explicitly defines self-loops) — the standard formula aggregates over N(v), the neighbors, SEPARATELY from the self-pathway (B·h_v).
- **Trap:** forgetting to divide by the neighbor count when doing MEAN aggregation — summing without dividing is a different (also valid, but different) aggregation scheme called SUM aggregation; always check which one a question specifies.
- **Trap:** applying the activation function separately to the W-pathway and B-pathway BEFORE summing, instead of summing first, then activating once — always sum the two pre-activation pathways first, exactly as shown in Worked Example 4.
- **Exam hack:** always compute degree as literally "sum the row (or column, for symmetric graphs they're identical) of the adjacency matrix" — this is faster and less error-prone than trying to recall the formal definition from memory under time pressure.
- **Exam hack:** the `S, S², S³` hop-distance progression is a favourite "compute Sx and S²x by hand for a small graph" question — practice matrix-vector multiplication carefully, since a single arithmetic slip early on cascades into every later hop.
- **Trap:** forgetting the square root in GCN's normalization coefficient — it's `1/√(d_i·d_j)`, not `1/(d_i·d_j)`; dropping the square root gives a completely different (much smaller) number.
- **Exam hack:** for GAT questions, always compute the raw scores FIRST, then explicitly show the softmax normalization step (exponentiate, sum, divide) — exactly like the attention-weight computation in Lecture 6, just now over a node's neighbors instead of a sequence.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-numerical)`

---

## Summary

This file worked every GNN formula from the theory file into fully shown arithmetic. Building a 3-node adjacency matrix from a list of weighted edges confirmed its symmetric structure (A=Aᵀ) for an undirected graph, and computing node degrees as row-sums gave d₁=6, d₂=3, d₃=5. A hand-computed mean-aggregation example combined three neighbors' 2D feature vectors into a single fixed-size summary `[1.0, 0.6667]`, directly solving the varying-neighbor-count problem described in the theory file. A full GNN layer update then combined this aggregated neighbor summary (via a learned weight matrix W) with the target node's own previous embedding (via a separate learned weight matrix B), summed both pathways, and applied ReLU to produce a new embedding `h_A^(1) = [1.5, 0.3333]` — the complete "aggregate neighbors + combine with self" recipe, fully quantified. A hop-distance example used matrix-vector multiplication with the graph's shift matrix S to show how a signal concentrated at node 1 spreads to its 1-hop neighbors (`Sx=[0,2,4]`) and then further to 2-hop reach (`S²x=[20,4,2]`), numerically demonstrating exactly how powers of S control how far information travels across a graph. Two additional worked examples then contrasted GCN and GAT directly on the SAME 4-node setup: GCN's degree-normalized aggregation produced `h_A_new≈[0.7815,0.3485]`, systematically down-weighting high-degree neighbor B and up-weighting low-degree neighbor D based purely on structure; GAT's attention-based aggregation produced a different result, `≈[0.5825,0.2905]`, weighting neighbors based on actual feature content via a Score→Softmax→Weighted-Sum computation instead of degree alone — directly illustrating GAT's "learned, adaptive importance" advantage over GCN's fixed coefficients. Finally, the lecture's own GCN sizing question was solved explicitly: for 10 nodes with 5 input features producing 16 output features, the shared weight matrix must be `W∈R^(5×16)` (80 parameters), independent of how many nodes the graph actually has. The master formula table consolidates every reusable calculation from this lecture for fast review.

`[← Theory](../theory/dl_lecture09_gnn_theory.md) · [🔝 Top](#dl-lecture-09--graph-neural-networks-numerical) · [Next: Practice →](../practice/dl_lecture09_gnn_practice.md)`
