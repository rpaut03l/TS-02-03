# DL Lecture 09 — Graph Neural Networks (Practice)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

> Folder: `Deep-Learning/Lecture-09-Graph-Neural-Networks/practice/`
> Pairs with: [`theory/dl_lecture09_gnn_theory.md`](../theory/dl_lecture09_gnn_theory.md) · [`numerical/dl_lecture09_gnn_numerical.md`](../numerical/dl_lecture09_gnn_numerical.md) · [`exercises/dl_lecture09_exercises.md`](../exercises/dl_lecture09_exercises.md)

---

## Table of Contents
1. [Official In-Class Question](#official-in-class-question)
2. [Concept Check — Fill in the Blank](#concept-check--fill-in-the-blank)
3. [Explain-It-Back Prompts](#explain-it-back-prompts)
4. [Quick-Fire True / False](#quick-fire-true--false)
5. [Graph Type Matching Drill](#graph-type-matching-drill)
6. [Mini Interview-Style Round](#mini-interview-style-round)
7. [Summary](#summary)

---

## Official In-Class Question

**Q.** Given a total of n nodes, and considering a graph where node 1 connects to nodes 2 and 3 with weights w₁₂ and w₁₃, what is node 1's neighborhood and degree?

<details>
<summary>Show answer</summary>

Node 1's neighborhood is `n(1) = {2, 3}` — the set of nodes connected to node 1. Node 1's degree is `d(1) = w₁₂ + w₁₃` — the sum of the weights of all edges incident to node 1.
</details>

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## Concept Check — Fill in the Blank

1. A graph is formally defined as the triplet G = (______, ______, ______).
2. CNNs work well on images because images have a fixed ______, consistent local ______, and ______ pixels.
3. A graph filter alone can only learn ______ maps; adding a pointwise non-linearity produces a graph ______.
4. The GNN layer update combines a transformed ______ of the neighbors with a transformed version of the node's own ______ representation.
5. ______ (Hamilton et al., 2017) solves the problem of GNNs needing the entire graph in memory, by sampling local neighborhoods.

<details>
<summary>Show answers</summary>

1. V (vertices); E (edges); W (weights)
2. grid structure; neighborhoods; ordered
3. linear; perceptron
4. aggregate/mean; previous/self
5. GraphSAGE
</details>

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## Explain-It-Back Prompts

1. Explain the "trust a person by their friends" analogy for what a GNN does.
2. Explain why plain CNNs can't be directly applied to graph-structured data, referencing all three properties images have that graphs lack.
3. Walk through the full GNN layer update formula from memory, explaining each term.
4. Explain why GNN depth (number of layers) should differ between molecule property prediction and citation network analysis.
5. Explain why GraphSAGE/PinSAGE were needed for graphs like Pinterest's, citing the specific practical problems with traditional GNNs.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## GCN / MPNN / GAT Matching Drill

| Architecture | Key mechanism | Your match |
|---|---|---|
| GCN | ? | |
| MPNN | ? | |
| GAT | ? | |

Options: (a) Learns neighbor importance automatically via attention, (b) Degree-normalized aggregation, the most commonly cited GNN, (c) Generalized, edge-aware message-passing framework unifying many GNN variants

<details>
<summary>Show answers</summary>

GCN → (b). MPNN → (c). GAT → (a).
</details>

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## Quick-Fire True / False

1. In a directed graph, edge (i,j) is always the same as edge (j,i). — **False** (that's specifically the symmetric/undirected case).
2. A graph filter (just powers of S) already has full non-linear expressive power. — **False** (it's linear; non-linearity requires an added activation function, forming a graph perceptron).
3. The graph shift operator S is a trainable parameter, learned during GNN training. — **False** (S is fixed prior structural information; W and B are the trainable parameters).
4. Layer-k node embeddings incorporate information from nodes up to k hops away. — **True**.
5. PinSAGE was applied to a graph with roughly 3 billion nodes. — **True**.
6. GCN's normalization coefficient depends on the LEARNED attention between two nodes' features. — **False** (that's GAT; GCN's coefficient is a FIXED function of node degrees, `1/√(d_i·d_j)`).
7. MPNN's main contribution was inventing the idea of aggregation itself. — **False** (aggregation already existed; MPNN's contribution was the generalized, edge-aware theoretical framework unifying many variants).

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## Graph Type Matching Drill

| Graph type | Defining property | Your match |
|---|---|---|
| Directed | ? | |
| Symmetric | ? | |
| Unweighted | ? | |
| Weighted Symmetric | ? | |

Options: (a) all weights equal 1, (b) both symmetric AND weighted, the most common in practice, (c) edge (i,j) differs from (j,i), (d) edge set and weights are both symmetric

<details>
<summary>Show answers</summary>

Directed → (c). Symmetric → (d). Unweighted → (a). Weighted Symmetric → (b).
</details>

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## Mini Interview-Style Round

**Q1.** "You're designing a GNN for predicting a molecule's toxicity. A teammate suggests using 20 GNN layers 'to be safe.' What's your concern?"

<details>
<summary>Show answer</summary>

For molecule property prediction, nearby atoms typically matter most — the theory file explicitly gives this as the canonical example of a task suited to a SHALLOW GNN. Using 20 layers would aggregate information from atoms 20 hops away, which is likely irrelevant (and could even be counterproductive, diluting locally-important signal with distant noise) for most molecular properties, while also being unnecessarily expensive to train and run. A much shallower GNN (a handful of layers) is more appropriate here.
</details>

**Q2.** "Explain why a social media platform can't just use a 'traditional' GNN that assumes the whole graph is loaded in memory."

<details>
<summary>Show answer</summary>

Social networks are extremely large, with new users, posts, and connections appearing continuously, and the graph structure changing dynamically essentially every second. Processing the whole graph at once every time a recommendation is needed would be computationally expensive, memory-intensive, and far too slow for real-time use. This is exactly why GraphSAGE (sampling and aggregating from local neighborhoods rather than requiring the full graph) and its extreme-scale successor PinSAGE were developed.
</details>

**Q3.** "A colleague proposes using plain GCN for a molecular property prediction task where certain specific atom-to-atom bonds are known to matter far more than others, regardless of the atoms' connectivity degree. Would you recommend GCN, or something else?"

<details>
<summary>Show answer</summary>

Plain GCN's neighbor coefficients are FIXED functions of node degree (`1/√(d_i·d_j)`) — they can't adapt to reflect "this specific bond matters more than that one" if both bonds involve atoms with similar degree. GAT would be a better fit here, since it LEARNS neighbor importance via attention, directly from the atoms' actual feature content, rather than from degree alone — exactly the scenario GAT was designed to improve on GCN for. If the bonds themselves carry rich, distinct features (single vs double vs triple bond, etc.) that need to directly influence the message, a full MPNN-style edge-aware message-passing approach might be worth the extra computational cost too.
</details>

**Q4.** "A colleague says 'the adjacency matrix contains all the same information as a list of edges, so it doesn't matter which representation we use.' Do you agree?"

<details>
<summary>Show answer</summary>

They contain equivalent information, but the FORMS matter practically: an adjacency matrix makes matrix operations (like computing Sx for neighbor aggregation, or S² for 2-hop reach) mathematically clean and directly usable inside a GNN's layer update formulas. An edge list is often more memory-efficient to STORE for large, sparse graphs (most real-world graphs have far fewer edges than the n² entries a dense adjacency matrix would require), which is part of why frameworks like GraphSAGE work with sampled neighbor lists rather than dense adjacency matrices for very large graphs.
</details>

`[🔝 Top](#dl-lecture-09--graph-neural-networks-practice)`

---

## Summary

This practice file drills Lecture 9's graph fundamentals and GNN mechanics through active recall. The official in-class neighborhood/degree question is answered in full, anchoring the exact notation used throughout the lecture. A fill-in-the-blank check reinforces the formal graph triplet, why CNNs fail on graphs, the graph-filter-to-graph-perceptron non-linearity story, the aggregate-plus-self update structure, and GraphSAGE's role. Five explain-it-back prompts push you to reproduce the "trust by friends" analogy, the three CNN-friendly image properties graphs lack, the full layer update formula, the shallow-vs-deep GNN depth argument, and the GraphSAGE/PinSAGE motivation in your own words. A quick-fire true/false round and a graph-type matching drill test both conceptual accuracy (directed vs symmetric, S being fixed not trainable) and precise terminology. A dedicated GCN/MPNN/GAT matching drill and two additional true/false items target the three named architectures specifically. A four-question interview-style round rehearses realistic design judgment: pushing back on an unnecessarily deep GNN for molecule prediction, explaining why traditional whole-graph-in-memory GNNs fail for social networks, recommending GAT over GCN when bond-specific importance matters more than node degree, and clarifying that adjacency matrices and edge lists are equivalent in information but different in practical usability. Move to the exercises file next for a tiered, exam-format question bank.

`[← Numerical](../numerical/dl_lecture09_gnn_numerical.md) · [🔝 Top](#dl-lecture-09--graph-neural-networks-practice) · [Next: Exercises →](../exercises/dl_lecture09_exercises.md)`
