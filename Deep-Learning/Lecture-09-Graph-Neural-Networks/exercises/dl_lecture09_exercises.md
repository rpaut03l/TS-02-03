# DL Lecture 09 — Exercise Bank (Graph Neural Networks)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-09--exercise-bank-graph-neural-networks)`

> Folder: `Deep-Learning/Lecture-09-Graph-Neural-Networks/exercises/`
> Difficulty tiers: 🟢 Easy · 🟡 Medium · 🔴 Hard.
> Related: [theory](../theory/dl_lecture09_gnn_theory.md) · [numerical](../numerical/dl_lecture09_gnn_numerical.md) · [practice](../practice/dl_lecture09_gnn_practice.md)

---

## 🟢 Easy — Definitions & Recall

**Q9.1.** Write the formal triplet definition of a graph.

**Q9.2.** What condition makes a graph's adjacency matrix symmetric?

**Q9.3.** Write the general GNN layer update formula.

**Q9.4.** What does layer-k node embedding depend on, in terms of hop distance?

**Q9.5.** Name the two systems built to scale GNNs to very large, dynamic graphs.

**Q9.6.** Write the GCN node update formula.

**Q9.7.** Write the GAT attention weight (softmax) formula, and name what MPNN's main contribution was (beyond "inventing aggregation").

---

## 🟡 Medium — Applied Reasoning

**Q9.8.** For a 4-node cycle graph (each node connects to exactly 2 neighbors, all edge weights =1), compute the degree of every node.

**Q9.9.** For the same cycle graph's adjacency matrix S=[[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]] and signal x=[1,0,0,0], compute Sx and S²x.

**Q9.10.** Explain why a graph filter (just powers of S applied to a signal) cannot solve genuinely non-linear tasks, connecting this back to Lecture 2's linear classifier limitations.

**Q9.11.** Two neighbor nodes B=[2,0] and C=[0,2] feed into target node A, whose own previous embedding is [4,4]. Using mean aggregation with W=identity and B_matrix=0.5×identity, compute the pre-activation value (before applying any non-linearity).

**Q9.12.** Explain why an unweighted graph can still be either directed or symmetric — i.e., why "unweighted" and "directed/symmetric" are independent properties.

**Q9.13.** For a node i with degree 4, connected to a neighbor j with degree 9, compute GCN's normalization coefficient `1/√(d_i·d_j)`.

**Q9.14.** Explain, in your own words, why GAT is described as sitting at a "golden middle" between GCN (cheap but rigid) and MPNN (flexible but expensive).

---

## 🔴 Hard — Derivation & Multi-Step

**Q9.15.** For a 5-node star graph (node 1 is the center, connected to nodes 2,3,4,5 each with weight 1, no other edges), write out the full 5×5 adjacency matrix, then compute the degree of node 1 and the degree of node 2.

**Q9.16.** Using the star graph from Q9.15, compute S·x where x=[0,1,1,1,1] (signal concentrated at the four outer nodes). Explain what this result represents in words.

**Q9.17.** A node has 3 neighbors with 3-dimensional feature vectors h_B=[1,2,3], h_C=[4,0,2], h_D=[1,1,1]. Compute the mean-aggregated neighbor vector, then apply a weight matrix W=2×identity to it.

**Q9.18.** Explain, with a worked numeric argument, why choosing SUM aggregation instead of MEAN aggregation makes a GNN's output sensitive to a node's DEGREE (number of neighbors) even when neighbor feature values are otherwise identical. Compute both the sum and mean aggregation for a node with 2 neighbors both having feature [1,1], and separately for a node with 5 neighbors all having feature [1,1], and compare.

**Q9.19.** For a target node A with degree d_A=2, connected to neighbor E (degree d_E=6) and neighbor F (degree d_F=1), with `h_E=[1,0]`, `h_F=[0,1]`, `W=identity`: compute the full GCN update for A (both coefficients, both contributions, the sum, and ReLU).

**Q9.20.** A GNN engineer has a graph where 10,000 nodes each have exactly 20 input features, and they want a GCN layer producing 64 output features per node. Compute the shape and total parameter count of the required shared weight matrix W (including a bias vector of size 64), and explain why this count does not depend on the 10,000 node figure at all.

`[🔝 Top](#dl-lecture-09--exercise-bank-graph-neural-networks)`

---

## Answer Key

<details>
<summary>Q9.1 – Q9.7 (Easy)</summary>

- **Q9.1:** G = (V, E, W) — vertices, edges, weights.
- **Q9.2:** The edge set and the weights are both symmetric: `(i,j)∈E ⟺ (j,i)∈E` and `w_ij=w_ji` for all edges.
- **Q9.3:** `h_v^(k) = σ(W_k·mean(N(v)) + B_k·h_v^(k-1))`.
- **Q9.4:** It depends on information from nodes up to k hops away.
- **Q9.5:** GraphSAGE and PinSAGE.
- **Q9.6:** `h_i' = σ(Σ_j (1/√(d_i·d_j))·W·h_j)`.
- **Q9.7:** `α_ij = softmax_j(e_ij) = exp(e_ij)/Σ_k exp(e_ik)`. MPNN's main contribution was a generalized theoretical framework with explicit edge-aware message functions, unifying many existing GNN variants — not inventing aggregation itself.
</details>

<details>
<summary>Q9.8 – Q9.14 (Medium)</summary>

- **Q9.8:** Every node in the cycle has exactly 2 neighbors, each edge weight 1, so every node's degree = 1+1 = **2** (all four nodes: degree 2).
- **Q9.9:** Sx = S·[1,0,0,0] = [0,1,0,1] (node 1's two direct neighbors, nodes 2 and 4, each pick up value 1). S²x = S·[0,1,0,1] = [2,0,2,0] (signal returns to nodes 1 and 3, each picking up contributions from both their neighbors).
- **Q9.10:** A graph filter is built purely from LINEAR combinations (powers of S applied to x, summed with learned coefficients) — exactly like a plain linear classifier `f=Wx` from Lecture 2, it can only represent linear relationships between input and output. Just as stacking multiple linear NN layers without activation functions collapses into one linear layer, stacking multiple graph filters without non-linearities would still only ever represent a linear function of the input signal — genuinely non-linear tasks (like complex signal analysis) require adding a pointwise non-linear activation, turning the graph filter into a graph perceptron.
- **Q9.11:** mean(B,C) = ([2,0]+[0,2])/2 = [1,1]. W·mean = identity×[1,1] = [1,1]. B_matrix·h_self = 0.5×[4,4] = [2,2]. Pre-activation = [1,1]+[2,2] = **[3,3]**.
- **Q9.12:** "Unweighted" describes whether edge WEIGHTS carry information (all weights=1 if unweighted) — it says nothing about edge DIRECTION. "Directed" vs "symmetric" describes whether influence flows one-way or both-ways. These are two completely independent design choices — you can have an unweighted directed graph (e.g., "who follows whom" on a social platform where following isn't necessarily mutual, but existence-of-follow is all that matters) OR an unweighted symmetric graph (e.g., a simple "who is friends with whom" graph with no weight/strength info).
- **Q9.13:** 1/√(4×9) = 1/√36 = 1/6 ≈ **0.1667**.
- **Q9.14:** GCN is cheap (just a fixed degree-based formula, no learned coefficients beyond W) but rigid (can't adapt neighbor importance to actual feature content). MPNN is fully flexible (learned message functions conditioned on edge features) but expensive (must store and process messages at the edge level, hard to scale). GAT sits between them: it adds LEARNED, adaptive importance (like MPNN's flexibility) via attention, but without needing full edge-level message storage (staying closer to GCN's computational cost) — hence "golden middle."
</details>

<details>
<summary>Q9.15 – Q9.20 (Hard)</summary>

- **Q9.15:** Adjacency matrix (rows/cols = nodes 1-5): row1=[0,1,1,1,1], row2=[1,0,0,0,0], row3=[1,0,0,0,0], row4=[1,0,0,0,0], row5=[1,0,0,0,0]. Node 1's degree = sum of row 1 = 1+1+1+1 = **4**. Node 2's degree = sum of row 2 = **1** (only connected to the center).
- **Q9.16:** Sx = S·[0,1,1,1,1]. Row 1: 0×0+1×1+1×1+1×1+1×1 = **4**. Rows 2-5: each row only has a 1 in column 1, and x[1]=0, so each = **0**. Result: [4,0,0,0,0]. This represents: the center node (node 1) receives the SUM of all its 4 outer neighbors' signal values (since each outer node had value 1, node 1 collects 1+1+1+1=4), while each outer node, having only the center as its neighbor, gets 0 (since the center's own signal value was 0 in x).
- **Q9.17:** mean = ([1,2,3]+[4,0,2]+[1,1,1])/3 = [6,3,6]/3 = [2,1,2]. W·mean = 2×[2,1,2] = **[4,2,4]**.
- **Q9.18:** SUM for 2 neighbors of [1,1]: [1,1]+[1,1] = [2,2]. MEAN for 2 neighbors: [2,2]/2 = [1,1]. SUM for 5 neighbors of [1,1]: 5×[1,1] = [5,5]. MEAN for 5 neighbors: [5,5]/5 = [1,1]. Under SUM aggregation, the 5-neighbor node's aggregated value ([5,5]) is 2.5× larger than the 2-neighbor node's ([2,2]), purely because it has more neighbors, EVEN THOUGH every individual neighbor has the identical feature value in both cases. Under MEAN aggregation, both nodes produce the IDENTICAL aggregated value ([1,1]), correctly reflecting that their neighbors are qualitatively the same, regardless of how many there are. This demonstrates why MEAN aggregation is often preferred when you want a node's degree not to artificially dominate its learned representation.
- **Q9.19:** coef_E = 1/√(2×6) = 1/√12 ≈ 0.2887. coef_F = 1/√(2×1) = 1/√2 ≈ 0.7071. Contribution from E = 0.2887×[1,0] = [0.2887, 0]. Contribution from F = 0.7071×[0,1] = [0, 0.7071]. Sum = [0.2887, 0.7071]. ReLU([0.2887,0.7071]) = **[0.2887, 0.7071]** (both already positive).
- **Q9.20:** W shape = **20×64** = 1,280 weight parameters, + 64 bias parameters = **1,344 total parameters**. This count depends only on the INPUT feature size (20) and OUTPUT feature size (64) — because W is SHARED and applied identically at every node, adding more nodes to the graph (10,000, or 10 million) never changes W's size, only the amount of computation (forward passes) required, exactly mirroring the "parameter count independent of sequence length" property of RNNs and LSTMs.
</details>

`[🔝 Top](#dl-lecture-09--exercise-bank-graph-neural-networks)`

---

## Summary

This exercise bank drills Lecture 9's graph theory and GNN mechanics across three tiers, including GCN, MPNN, and GAT. Easy questions recall the formal graph triplet, the symmetric-adjacency-matrix condition, the general GNN layer update formula, hop-distance dependence, GraphSAGE/PinSAGE naming, and now also the GCN update formula, the GAT softmax attention formula, and MPNN's real contribution. Medium questions apply degree and hop-distance calculations to a new 4-node cycle graph, connect graph filters' linearity limitation back to Lecture 2's linear classifier story, compute a full mean-aggregation-plus-self pre-activation value, clarify that "unweighted" and "directed/symmetric" are independent graph properties, compute a GCN normalization coefficient directly, and explain GAT's "golden middle" positioning between GCN and MPNN. Hard questions require full derivations on a 5-node star graph (building its full adjacency matrix, computing asymmetric degrees, and a full one-hop signal propagation), a 3-dimensional mean-aggregation-plus-weight-matrix computation, a quantitative demonstration of why SUM aggregation makes a node's output artificially sensitive to its degree, a complete GCN update computation with two differently-weighted neighbors, and a full weight-matrix sizing and parameter-counting exercise for a 10,000-node graph, confirming GCN's parameter count is entirely independent of graph size. All answers are fully worked and spoiler-tagged.

`[← Practice](../practice/dl_lecture09_gnn_practice.md) · [🔝 Top](#dl-lecture-09--exercise-bank-graph-neural-networks) · [Code →](../code/README.md)`
