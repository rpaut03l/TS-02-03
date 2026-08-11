# Lecture 09 — Graph Neural Networks

`[← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-09--graph-neural-networks)`

**Instructor:** Dr. Anushka Joshi, IIT Jodhpur | **Date:** May 2026 | **Source slides:** "Graph Neural Networks" deck, parts 1 & 2 (combined)

Covers why deep learning on graphs is needed, formal graph definitions, adjacency matrices, the graph-perceptron/GNN-layer derivation, neighborhood aggregation, **GCN, MPNN, and GAT** (the three concrete named architectures from the source deck), GNN depth vs locality, training GNNs, and scaling to huge graphs via GraphSAGE/PinSAGE.

## Files in this lecture

| File | Focus |
|---|---|
| 📘 [`theory/dl_lecture09_gnn_theory.md`](theory/dl_lecture09_gnn_theory.md) | Graph types, adjacency matrix, GNN layers, aggregation, GCN, MPNN, GAT, GraphSAGE |
| 🔢 [`numerical/dl_lecture09_gnn_numerical.md`](numerical/dl_lecture09_gnn_numerical.md) | Adjacency matrix + degree, mean aggregation, full layer update, hop distance, GCN, GAT |
| ✍️ [`practice/dl_lecture09_gnn_practice.md`](practice/dl_lecture09_gnn_practice.md) | Official in-class Q, graph-type matching, interview Qs |
| 🧪 [`exercises/dl_lecture09_exercises.md`](exercises/dl_lecture09_exercises.md) | Tiered Easy/Medium/Hard question bank with answer key |
| 💻 [`code/`](code/README.md) | A mini GNN (adjacency matrix, aggregation, layer update) built from scratch in NumPy |

## Suggested reading order

Theory → Numerical → Practice → Exercises → Code.

`[← Lecture 08](../Lecture-08-Regularization/README.md) · [← Deep Learning Hub](../README.md) · [🔝 Top](#lecture-09--graph-neural-networks) · [Next: Lecture 10 →](../Lecture-10-Attention-and-Transformers/README.md)`
