# DL Lecture 09 — Graph Neural Networks (Theory)

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

> Folder: `Deep-Learning/Lecture-09-Graph-Neural-Networks/theory/`
> Pairs with: [`numerical/dl_lecture09_gnn_numerical.md`](../numerical/dl_lecture09_gnn_numerical.md) · [`practice/dl_lecture09_gnn_practice.md`](../practice/dl_lecture09_gnn_practice.md) · [`exercises/dl_lecture09_exercises.md`](../exercises/dl_lecture09_exercises.md)
> Instructor: Dr. Anushka Joshi, IIT Jodhpur | Source: "Graph Neural Networks" deck, parts 1 & 2 (combined here)

---

## Table of Contents
1. [The Big Picture — A Story First](#the-big-picture--a-story-first)
2. [Why Deep Learning on Graphs?](#why-deep-learning-on-graphs)
3. [Why Graphs Are Hard for Normal NNs and CNNs](#why-graphs-are-hard-for-normal-nns-and-cnns)
4. [Nodes, Edges, and Weights — The Formal Graph Definition](#nodes-edges-and-weights--the-formal-graph-definition)
5. [Types of Graphs](#types-of-graphs)
6. [The Adjacency Matrix (Graph Shift Operator)](#the-adjacency-matrix-graph-shift-operator)
7. [Neighborhoods and Degrees](#neighborhoods-and-degrees)
8. [From CNN to GNN — Time and Space as Graphs](#from-cnn-to-gnn--time-and-space-as-graphs)
9. [The Graph Perceptron and GNN Layers](#the-graph-perceptron-and-gnn-layers)
10. [Aggregating Neighbors — The Core GNN Idea](#aggregating-neighbors--the-core-gnn-idea)
11. [The First GNN Equation](#the-first-gnn-equation)
12. [GCN — The Most Common GNN Formula](#gcn--the-most-common-gnn-formula)
13. [The General Message Passing Framework (MPNN)](#the-general-message-passing-framework-mpnn)
14. [Graph Attention Networks (GAT)](#graph-attention-networks-gat)
15. [Graph Embedding — The Bigger Picture](#graph-embedding--the-bigger-picture)
16. [How Many Layers? Depth vs Locality](#how-many-layers-depth-vs-locality)
17. [Training a GNN](#training-a-gnn)
18. [Scaling to Huge Graphs — GraphSAGE](#scaling-to-huge-graphs--graphsage)
19. [Real-World Success Stories](#real-world-success-stories)
20. [Mnemonics](#mnemonics)
21. [Cheatsheet](#cheatsheet)
22. [Exam Hacks / Trap Watch](#exam-hacks--trap-watch)
23. [Summary](#summary)

---

## The Big Picture — A Story First

Imagine trying to figure out how trustworthy a person is, purely by looking at them in isolation, with a photograph — no context about who they know, who vouches for them, or what community they belong to. Now imagine instead you get to see their entire social circle: who they're friends with, who those friends are friends with, and so on. Suddenly you have vastly more signal — "this person is trusted because three of their close friends, who are themselves well-regarded, all vouch for them." A **Graph Neural Network** formalizes exactly this intuition mathematically: instead of treating each data point (a person, an atom, a road intersection, a word) in isolation, a GNN builds a representation of each node by repeatedly **gathering information from its neighbors**, and its neighbors' neighbors, layer by layer — letting local relationships shape a rich, context-aware understanding of every single node in a network.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Why Deep Learning on Graphs?

Graphs are a flexible way to represent signals, images, and other structured data by modeling **relationships between connected elements**. Several real motivating examples from the lecture:

- **Authorship attribution:** given an anonymous article, predict who wrote it. Why a graph? Words become NODES, and relationships (co-occurring words, writing patterns) become EDGES — the resulting graph captures an author's distinctive writing style (this is exactly the "Word Adjacency Network" approach used to distinguish Marlowe's writing from Shakespeare's).
- **Recommendation systems:** platforms like Netflix/Amazon/YouTube. Users AND products/movies both become nodes; interactions (purchase, watch, rating) become edges — the task becomes predicting the rating a user would give an unseen product.
- **Multiagent physical systems:** e.g., wireless communication networks managing interference when allocating bandwidth/power. Graphs here represent REAL interacting systems, not just abstract data — the objective is global, but each node only has access to LOCAL information, a genuinely graph-structured constraint.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Why Graphs Are Hard for Normal NNs and CNNs

CNNs work extremely well on images specifically because images have: (1) a **fixed grid structure**, (2) **consistent local neighborhoods** (every interior pixel has exactly 8 neighbors), and (3) **ordered pixels** (a clear, consistent left-right/top-bottom arrangement). A graph has NONE of these convenient properties: **irregular structure**, a **varying number of neighbors** per node, and **no fixed ordering** of nodes at all. Directly applying a normal neural network or plain CNN to graph data is therefore fundamentally challenging — you can't just slide a fixed-size filter across a graph the way you slide one across an image, because "neighborhood shape" is different at every single node.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Nodes, Edges, and Weights — The Formal Graph Definition

A graph is formally a triplet **G = (V, E, W)**:
- **V (vertices/nodes):** a set of n labels, typically V = {1, ..., n}.
- **E (edges):** ordered pairs of labels (i, j). We interpret `(i,j) ∈ E` as **"i can be influenced by j."**
- **W (weights):** numbers `w_ij ∈ R` associated with each edge (i,j), representing the **strength of the influence of j on i.**

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Types of Graphs

| Type | Defining property | Example use case |
|---|---|---|
| **Directed** | `(i,j)` is different from `(j,i)`; weights can differ (`w₃₅ ≠ w₅₃`) | Citation networks — paper A citing paper B doesn't mean B cites A |
| **Symmetric (undirected)** | `(i,j)∈E ⟺ (j,i)∈E`, AND `w_ij=w_ji` for all edges | Social network friendships — if A is B's friend, B is A's friend |
| **Unweighted** | All weights are exactly 1 (`w_ij=1` for all edges); can be directed OR symmetric | Molecule property prediction — a chemical bond either exists (1) or doesn't (0) |
| **Weighted Symmetric** | Both symmetric AND weighted — the MOST COMMON type in practice | Traffic networks — roads connect intersections, weighted by distance or traffic correlation |

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## The Adjacency Matrix (Graph Shift Operator)

Graphs have a natural matrix representation, also called a **Graph Shift Operator**. The **adjacency matrix A** of graph G=(V,E,W) is a (typically sparse) matrix with nonzero entries:
```
A_ij = w_ij   for all (i,j) in E
```
If the graph is symmetric, the adjacency matrix is symmetric too: **A = Aᵀ**. For an UNWEIGHTED graph, this simplifies to `A_ij = 1` for every edge in E (and 0 everywhere else).

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Neighborhoods and Degrees

- **Neighborhood of node i:** the set of nodes that influence i (i.e., all j such that `(i,j) ∈ E`).
- **Degree d_i of node i:** the SUM of the weights of all edges incident to (directly connected to) node i.

**Worked example from the slides:** for node 1 with neighbors {2, 3}, the neighborhood is `n(1) = {2, 3}`, and the degree is `d(1) = w₁₂ + w₁₃` — literally add up the weights of every edge touching node 1.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## From CNN to GNN — Time and Space as Graphs

Convolution already works on 1D signals (time series) and 2D signals (images) — can we generalize convolution to graphs too? The key trick: **first reinterpret CNN-friendly data AS graphs.**
- **Time as a graph:** each timestep is a node; edges connect neighboring time samples — this is literally a "directed line graph."
- **Space (images) as a graph:** each pixel is a node; edges connect neighboring pixels — this is literally a "grid graph."

Once you see time-series and images as special, very regular cases of graphs, you can write ordinary convolution using the graph's shift/adjacency matrix **S**: `Sx` collects information from 1-hop neighbors, `S²x` collects information from 2-hop neighbors, `S³x` collects information from even farther neighbors — and because S has a FIXED size, this generalizes cleanly to ANY arbitrary graph (a social network, a sensor network, anything), not just neat lines or grids. The polynomial built from powers of S becomes exactly a **graph convolutional filter.**

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## The Graph Perceptron and GNN Layers

A **graph filter** (built purely from powers of S, as above) has limited expressive power because it can only learn LINEAR maps — but most real deep learning tasks (like analyzing earthquake signals, which depend on many interacting factors) are fundamentally nonlinear. The fix, exactly parallel to Lecture 2's "why we need non-linearity" story: introduce a **graph perceptron** — apply a graph filter, THEN a pointwise non-linear activation function (like ReLU), just like a normal CNN's "filter bank + pointwise nonlinearity" pattern, but now generalized onto arbitrary graph structure.

**A full GNN is built by stacking (composing) several graph perceptrons — layering them, exactly like stacking layers in an MLP or CNN.** For an L-layer GNN, the trainable parameters form a **filter tensor H = [h₁, h₂, ..., h_L]** — one learned filter per layer. The graph shift **S** is NOT trainable — it's fixed, prior structural information about the graph itself (who's connected to whom). Feeding the input signal `x=x₀` through Layer 1, then Layer 1's output through Layer 2, and so on, the final layer's output is the GNN's overall output, written **Φ(x; S, H)** — a function of the input x, the (fixed) graph structure S, and the (learned) filter tensor H.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Aggregating Neighbors — The Core GNN Idea

The modern, widely-used framing (from Stanford's CS224W): **a node's neighborhood defines a computation graph.** The key idea: **generate node embeddings based on local network neighborhoods** — neighbors provide extra, useful information beyond a node's own raw features.

**Concrete walkthrough:** suppose the target node is A, with neighbors B, C, and D. A GNN says: "A should also look at its neighbors" — gather (aggregate) information FROM B, C, and D, and COMBINE it (e.g., using a simple **mean** of the neighbors' features) into A's own updated representation.

**A refinement — transform before combining:** instead of directly sending raw features, each neighbor FIRST transforms its own feature through a small neural network before sending it onward. Why? Raw features may not be directly useful as-is — a learned transformation lets each neighbor "translate" its information into a more useful form before it gets combined into the target node's new representation.

**Every node builds its OWN local computation graph**, based on its own specific neighborhood — this is a crucial structural point: unlike a CNN, where every spatial position uses an identical filter shape, in a GNN every NODE potentially has a differently-shaped computation graph, since every node can have a different number of neighbors.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## The First GNN Equation

Formally, for node v:

**Initial (layer-0) embedding — just the node's own raw input features:**
```
h_v^(0) = x_v
```

**After each subsequent layer k (this is the core recursive update, seen throughout the lecture in slightly different notations):**
```
h_v^(k) = sigma( W_k * MEAN({ h_u^(k-1) : u in N(v) }) + B_k * h_v^(k-1) )
```

Where:
- **N(v)** = the neighborhood of node v (the set of nodes connected to v)
- **W_k** = a learned weight matrix, for THIS layer, that transforms the AGGREGATED NEIGHBOR information
- **B_k** = a separate learned weight matrix, for THIS layer, that transforms the node's OWN previous representation (its "self" pathway)
- **σ (sigma)** = a non-linear activation function (e.g., ReLU)

**Final embedding** = the output of the LAST GNN layer, `h_v^(L)` — this becomes the node's learned feature vector, ready to be used for a downstream prediction task. The overall pipeline: **Node features → GNN layers → Node embeddings → Prediction layer → Output.**

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## GCN — The Most Common GNN Formula

The generic aggregation update above is a useful mental model, but the single most commonly cited, most widely used concrete GNN architecture is the **GCN (Graph Convolutional Network)**, introduced by **Kipf & Welling, ICLR 2017**. Its node update rule:
```
h_i^(l+1) = sigma( Σ_{j in N(i)} (1 / sqrt(d_i * d_j)) * W^(l) * h_j^(l) )
```
The key detail that distinguishes GCN from a plain mean-aggregation update: instead of dividing by just ONE node's neighbor count (`|N(v)|`, as in plain mean aggregation), GCN divides each neighbor's contribution by **the square root of the PRODUCT of both nodes' degrees**, `sqrt(d_i × d_j)` — a symmetric normalization that accounts for BOTH the receiving node i's degree AND the sending neighbor j's degree. This keeps the aggregated signal well-scaled even when node degrees vary wildly across a graph (a hub node with thousands of neighbors vs. a leaf node with just one). GCN is described as the "most common graph neural network... simple and powerful... most commonly cited work" — genuinely the default starting point for most real GNN projects.

**A concrete sizing question the lecture poses directly:** suppose a graph has 10 nodes, each with 5 input features, and you want to produce 16 new features per node via one GCN layer, using a shared weight matrix W (playing the same role a convolution filter plays in a CNN). What shape must W have? **Answer: `W ∈ R^(5×16)`** — it must accept 5-dimensional inputs (matching each node's feature size) and produce 16-dimensional outputs (the desired new feature size), and — crucially — this SAME W is shared and reused identically across all 10 nodes, exactly like a CNN filter is reused across all spatial positions (recall Lecture 3's parameter-sharing story, now applied to graph nodes instead of image pixels).

**Does GCN preserve the number of nodes?** Yes — after a GCN layer, you still have the same 10 nodes; each one simply now carries **its own information PLUS aggregated neighbor information**, re-encoded into the new (in this example, 16-dimensional) feature space. GCN transforms node FEATURES, not the graph's STRUCTURE.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## The General Message Passing Framework (MPNN)

**The problem with plain GCN:** it only INDIRECTLY supports edge features — the degree-based normalization `1/sqrt(d_i·d_j)` doesn't leave room for edges to carry their own rich information (e.g., "this chemical bond is a double bond," or "this road has heavy traffic"). **The solution:** design update mechanisms directly at the EDGE level — nodes exchange information by explicitly sending **messages** across graph edges, and these messages can be CONDITIONED on edge features, not just node features. Each node then **aggregates all messages sent to it**, using a **permutation-invariant function** (like sum or mean — a function whose result doesn't depend on the ORDER the messages arrive in, since a node's neighbors have no inherent ordering).

**MPNN (Message Passing Neural Network; Gilmer et al., ICML 2017)** formalizes this generic message-passing idea into a full theoretical framework. Its own major contribution, in the lecture's own words, was NOT "inventing aggregation" (that idea already existed) — it was providing: (1) a **generalized theoretical framework**, (2) **explicit edge-aware message functions**, and (3) a **unification of many existing GNN variants** under one consistent mathematical description. A generic message-passing step looks like:
```
message_ij = M( h_i, h_j, e_ij )              (compute a message from j to i, using edge features e_ij)
h_i_new    = U( h_i, aggregate_j(message_ij) ) (update node i using all incoming messages)
```
where `M` (the message function) and `U` (the update function) are both learned neural networks.

**A crucial correctness requirement MPNN emphasizes:** the whole framework must be **invariant to graph isomorphism** — if two graphs represent the exact same underlying structure (just with nodes numbered/labelled differently), the network MUST produce the same output regardless of that arbitrary numbering. The lecture's own example: in a molecule graph (atoms as nodes, chemical bonds as edges), the numbering/order in which atoms happen to be listed should never affect the model's prediction — a genuinely important correctness property for any real chemistry application.

**MPNN's limitations (the lecture poses this as a direct question, then answers it):** MPNNs require storing and processing messages at the EDGE level, leading to high memory and computational cost, making them difficult to scale to large graphs — practical mainly for smaller graph datasets. The lecture even notes MPNNs "can be viewed as MLP-like operations on graph structures" — powerful and flexible, but at real computational expense.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Graph Attention Networks (GAT)

**Towards a "golden middle"** between GCN's cheap-but-rigid fixed normalization and MPNN's flexible-but-expensive full edge-level message passing: a simplified graph convolution framework that still learns per-neighbor IMPORTANCE, without paying MPNN's full edge-message cost:
```
h_i' = sigma( Σ_j alpha_ij * W * h_j )
```
where **W** is a shared learnable weight matrix, **α_ij** is an importance coefficient between nodes i and j, and **σ** is an activation function. **The drawback of plain GCN, reframed through this lens:** GCN essentially DEFINES `α_ij` explicitly and rigidly (as the fixed `1/sqrt(d_i·d_j)` degree-normalization term) — this FIXED coefficient can fail to capture truly ADAPTIVE node importance (e.g., maybe one particular neighbor really is far more informative than another, even if their degrees are similar), leading to representational limitations on complex graphs.

**The GAT fix (Veličković et al., ICLR 2018):** instead of manually defining `α_ij`, **learn it automatically using attention** — directly reusing the Query/Key-style attention machinery from Lecture 6 and Lecture 10, now applied to graph neighbors instead of sequence tokens. Node features are first transformed using a shared weight matrix W (exactly like GCN's W), and then an **attention function `a`** computes a raw score measuring how important neighbor j is to node i:
```
e_ij = a( W*h_i, W*h_j )
```
**Attention normalization:** these raw scores are converted into proper attention weights via **softmax** (the exact same Score→Softmax→Weighted-Sum recipe from Lecture 6):
```
alpha_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k in N(i)} exp(e_ik)
```
This ensures all of node i's neighbor importance values sum to exactly 1 — a proper probability distribution over i's neighbors, exactly like attention weights over input words in Lecture 6.

**Multi-head attention in GAT:** exactly like Transformer multi-head attention (Lecture 10), GAT commonly uses **K independent attention heads** (the lecture's example uses K=3) computing SEPARATE attention distributions over the same neighborhood in parallel — different heads can learn to focus on different kinds of relationships. The K heads' outputs are then either **concatenated** (producing a wider final representation) or **averaged** (keeping the same width) to produce the final updated embedding.

**Why GAT matters:** it directly solves GCN's "fixed coefficient" limitation by making neighbor importance LEARNED and ADAPTIVE rather than a rigid function of degree alone — often improving performance on graphs where some neighbors are genuinely much more informative than others, at a computational cost that's still far cheaper than MPNN's full edge-message-passing machinery.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Graph Embedding — The Bigger Picture

Zooming out: everything above — GCN, MPNN, GAT, and the general aggregation framework — is ultimately in service of **graph embedding**: converting high-dimensional graph-structured information (nodes, edges, features) into a compact, LOW-dimensional vector space, where standard downstream machine learning tools (classifiers, clustering algorithms, nearest-neighbor search) can be applied directly. Traditional graph analytics (path analysis, connectivity analysis) can suffer from high computational cost and excessive memory requirements on real, complex industrial graphs — learned graph embeddings (exactly what every GNN layer in this lecture produces) are the modern, scalable answer to that same underlying goal: represent a complex network in a form that's both COMPACT and USEFUL for downstream tasks.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## How Many Layers? Depth vs Locality

**Why this matters:** Layer-k embeddings gather information from nodes that are exactly **k hops away**. So choosing the number of GNN layers directly controls how "far" information can travel to influence any given node's final representation.

- **When local relationships matter most → use a SHALLOW GNN.** Example: molecules — nearby atoms matter most for predicting chemical properties; information from atoms very far away in the molecular graph is usually not very relevant.
- **When larger/farther context matters → use a DEEPER GNN.** Example: citation networks — a paper's topic may genuinely depend on papers that are cited several hops away (papers-that-cite-papers-that-cite-the-original-paper), so more layers (more hops of aggregation) can help capture that longer-range structure.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Training a GNN

A node embedding `z_v` is a function of the whole input graph (its structure AND features). To train the GNN's parameters, we need a **loss function defined on the embeddings** — exactly like any other supervised learning setup: compute the loss between the model's prediction (built from `z_v`) and the true label `y` for that node/graph/task, then use gradient descent (from Lecture 7) and backpropagation to update the trainable parameters — specifically, **W** (the weight matrix for neighborhood aggregation) and **B** (the weight matrix for transforming a node's own self-representation), at every layer.

**Concrete illustration:** a user node A in a social network might START with only its own raw profile features. AFTER passing through several GNN layers of neighbor aggregation, A's final embedding also incorporates its friends' information, community-level information, and multi-hop context — a vastly richer representation than the raw profile features alone, entirely learned through the graph structure.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Scaling to Huge Graphs — GraphSAGE

**The problem:** traditional GNN formulations usually assume the ENTIRE graph is available in memory at once. This becomes genuinely difficult for real systems like social-network recommenders, because: social networks are extremely large, new users/posts/connections appear continuously, and the graph structure changes dynamically, essentially every second. Processing the whole graph at once, every time, is computationally expensive, memory-intensive, and far too slow for real-time recommendations.

**The solution: GraphSAGE** (Hamilton et al., NeurIPS 2017) — instead of requiring the full graph, GraphSAGE samples and aggregates from a node's LOCAL neighborhood, enabling GNNs to scale to enormous, dynamically-changing graphs. **PinSAGE** (Ying et al., KDD 2018) took this idea to an extreme scale, applying it to Pinterest's graph — roughly **3 billion nodes and 18 billion edges** — a scale utterly impossible with naive "load the whole graph into memory" approaches.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Real-World Success Stories

The lecture highlights several genuinely impactful GNN applications:
- **Drug discovery:** described as "arguably the most popularized success story of graph neural networks to date" (Stokes et al., Cell 2020) — using GNNs to identify promising new antibiotic candidates by learning from molecular graph structure.
- **Traffic and travel-time prediction:** transportation maps (like Google Maps) are naturally graphs — intersections are nodes, roads are edges (with features like road length, current/historical speeds). Routes get partitioned into "supersegments" (larger combined road sections), and a GNN performs GRAPH REGRESSION on the supersegment graph to estimate time of arrival (ETA).
- **Recommender systems at scale:** social networks and platforms like Pinterest use GraphSAGE/PinSAGE-style GNNs for link prediction and recommendation, precisely because of the massive-scale, dynamically-changing graph challenges described above.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Mnemonics

- **"Trust a person by their friends, not just their photo"** → the core GNN intuition.
- **"Images have grids, graphs have chaos"** → why plain CNNs don't directly work on graphs.
- **"V, E, W — who exists, who influences whom, how strongly"** → the formal graph triplet.
- **"Directed = one-way street, Symmetric = two-way street"** → directed vs undirected graphs.
- **"S¹ = neighbors, S² = neighbors of neighbors, S³ = even farther"** → powers of the shift operator = hop distance.
- **"Aggregate, then combine with self"** → the core `W·mean(neighbors) + B·self` update rule.
- **"Shallow for local (molecules), deep for global (citations)"** → choosing GNN depth.
- **"Can't fit the whole graph? Sample it — GraphSAGE"** → the scaling solution.
- **"GCN: divide by sqrt(d_i × d_j), most common GNN"** → the default, cheap, degree-normalized formula.
- **"MPNN: message, then aggregate, permutation-invariant, edge-aware"** → the general, expensive, edge-level framework.
- **"GAT: don't fix the coefficient, learn it — attention over neighbors"** → GCN's fixed-weight limitation solved.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Cheatsheet

| Concept | One-liner | Formula/Fact |
|---|---|---|
| Graph | Vertices, Edges, Weights | `G=(V,E,W)` |
| Adjacency matrix | Matrix form of a graph | `A_ij = w_ij` for `(i,j)∈E` |
| Degree | Total edge weight touching a node | `d_i = Σ w_ij` for incident edges |
| Directed graph | Asymmetric influence | `(i,j) ≠ (j,i)` |
| Symmetric graph | Mutual influence | `(i,j)∈E ⟺ (j,i)∈E`, `w_ij=w_ji` |
| Graph filter | Linear-only combination via powers of S | `Sx, S²x, S³x, ...` |
| Graph perceptron | Filter + non-linearity | Adds expressive power beyond linear |
| GNN layer update | Aggregate neighbors + self | `h_v^(k)=σ(W_k·mean(N(v))+B_k·h_v^(k-1))` |
| GCN (Kipf & Welling) | Degree-normalized aggregation; most common GNN | `h_i'=σ(Σ_j (1/√(d_i·d_j))·W·h_j)` |
| MPNN (Gilmer et al.) | General, edge-aware message passing | `msg=M(h_i,h_j,e_ij)`, `h_i'=U(h_i,agg(msg))` |
| GAT (Veličković et al.) | Learned, attention-based neighbor importance | `α_ij=softmax_j(a(Wh_i,Wh_j))`, `h_i'=σ(Σα_ij·W·h_j)` |
| Layer count = hop distance | k layers reach k-hop neighbors | shallow=local, deep=global |
| GraphSAGE | Scales GNNs via neighbor sampling | handles billion-node graphs |

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Exam Hacks / Trap Watch

- **Trap:** forgetting that a plain graph filter (just powers of S) is LINEAR — the non-linearity specifically comes from adding a pointwise activation function, forming a "graph perceptron," exactly parallel to Lecture 2's linear-classifier-to-neural-network story.
- **Trap:** confusing the graph shift operator S (fixed, NOT trainable — it's prior structural information) with the filter tensor H or weight matrices W/B (which ARE trainable/learned).
- **Trap:** assuming every node's computation graph looks the same — unlike CNN filters (identical shape everywhere), each node's aggregation neighborhood can have a completely different number of neighbors, so each node effectively builds a differently-SHAPED (though identically-PARAMETERIZED) computation graph.
- **Exam hack:** always connect "number of GNN layers" to "hop distance reached" explicitly — this is THE most frequently tested conceptual link in this lecture (shallow=local tasks like molecules, deep=long-range tasks like citation networks).
- **Exam hack:** if asked why GraphSAGE/PinSAGE were needed, always cite the SPECIFIC reasons given (graph too large for memory, continuously changing/dynamic, real-time latency requirements) rather than a vague "graphs can be big" answer.
- **Trap:** confusing GCN's FIXED `1/√(d_i·d_j)` normalization with GAT's LEARNED `α_ij` attention weight — GCN's coefficient depends only on degree (structural, not learned); GAT's coefficient is learned from the actual node features via an attention function, which is precisely GAT's advantage.
- **Trap:** describing MPNN as "just another aggregation scheme" — its real contribution was the generalized theoretical framework, explicit edge-aware messages, and unifying many GNN variants, not inventing aggregation itself.
- **Exam hack:** the GCN sizing question ("10 nodes, 5 features, want 16 new features, what shape is W?") is a favourite, easy-marks numerical question — always answer with the shape `(input_dim × output_dim)`, i.e. `5×16` here, and note W is SHARED across all nodes.

`[🔝 Top](#dl-lecture-09--graph-neural-networks-theory)`

---

## Summary

This lecture introduces Graph Neural Networks as the natural extension of CNN-style convolution to irregularly-structured data. Graphs represent relationships between connected elements (formally a triplet G=(V,E,W): vertices, edges as ordered "influence" pairs, and weights as influence strength), and are essential whenever data doesn't fit a clean grid — authorship attribution (words as nodes), recommendation systems (users/products as nodes), and multiagent physical systems all motivate graph-structured deep learning. Plain CNNs fail on graphs because graphs have irregular structure, varying neighbor counts, and no fixed node ordering — unlike images' convenient fixed grid, consistent neighborhoods, and ordered pixels. Graphs are represented via an adjacency (shift) matrix A, and the key trick for generalizing convolution is reinterpreting time series and images themselves as special, regular graphs (line graphs and grid graphs respectively), so that powers of the shift matrix S (`Sx, S²x, S³x, ...`) capture increasingly distant (multi-hop) neighborhood information on ANY graph, not just regular ones. Since plain graph filters are linear, a "graph perceptron" adds a pointwise non-linearity, and stacking multiple graph perceptrons into layers (with a trainable filter tensor H per layer, while the graph structure S itself stays fixed) forms a full GNN, `Φ(x;S,H)`. The modern, dominant framing describes this as neighborhood aggregation: each node generates its embedding by aggregating (e.g., averaging) transformed features from its neighbors, combined with its own previous representation, following the recursive update `h_v^(k)=σ(W_k·mean(N(v))+B_k·h_v^(k-1))`, starting from `h_v^(0)=x_v`. Three concrete, named architectures build on this general recipe: **GCN** (Kipf & Welling, 2017), the most commonly cited GNN, uses a degree-normalized aggregation (`1/√(d_i·d_j)`) and is the default starting point for most projects; **MPNN** (Gilmer et al., 2017) generalizes this into a full edge-aware message-passing framework (learned message function M, learned update function U, permutation-invariant aggregation, and a strict graph-isomorphism-invariance requirement) that unifies many GNN variants but is expensive and hard to scale; and **GAT** (Veličković et al., 2018) fixes GCN's rigid, degree-only coefficient by LEARNING neighbor importance via attention (`α_ij=softmax_j(a(Wh_i,Wh_j))`), reusing the exact Score→Softmax→Weighted-Sum recipe from Lecture 6, including multi-head attention analogous to Lecture 10's Transformers. All of this serves the broader goal of graph embedding — converting complex, high-dimensional graph structure into compact, useful low-dimensional vectors. Since layer-k embeddings incorporate information from nodes exactly k hops away, choosing GNN depth directly trades off locality (shallow, for tasks like molecule property prediction) against long-range context (deep, for tasks like citation-network analysis). Training uses a loss function defined on the learned node embeddings, optimized via standard backpropagation and gradient descent. Because traditional GNNs assume the whole graph fits in memory — a major problem for huge, constantly-changing graphs like social networks — **GraphSAGE** (and its extreme-scale successor **PinSAGE**, applied to a 3-billion-node, 18-billion-edge Pinterest graph) solves this by sampling and aggregating from local neighborhoods rather than requiring the full graph. Real-world GNN success stories cited include antibiotic drug discovery (Stokes et al., Cell 2020) and Google-Maps-style estimated-time-of-arrival prediction via graph regression on road "supersegment" graphs.

`[← Lecture 08](../../Lecture-08-Regularization/README.md) · [🔝 Top](#dl-lecture-09--graph-neural-networks-theory) · [Next: Numerical →](../numerical/dl_lecture09_gnn_numerical.md)`
