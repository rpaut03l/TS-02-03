# DL Lecture 09 — Code

`[← TOC](../../README.md) · [🔝 Top](#dl-lecture-09--code)`

> Folder: `Deep-Learning/Lecture-09-Graph-Neural-Networks/code/`

---

## What's in this folder

| File | What it does |
|---|---|
| `dl_lecture09_gnn_from_scratch.py` | Builds an adjacency matrix, implements mean-neighbor aggregation and the full GNN layer update, verifies against every numerical README example, then runs a full 2-layer forward pass across a whole 4-node graph. |

## Library required

Just **NumPy**.
```bash
pip install numpy --break-system-packages
```

## What each part of the code maps back to

| Code function | Theory/Numerical concept |
|---|---|
| `build_adjacency_matrix()` | `G=(V,E,W)` → matrix form `A_ij=w_ij` |
| `degree()` | `d_i = Σ_j A_ij` |
| `mean_aggregate()` | `mean(N(v)) = (1/|N(v)|)Σh_u` |
| `gnn_layer_update()` | `h_v^(k)=σ(W·mean(N(v))+B·h_v^(k-1))` |
| `check_hop_distance()` | `Sx`, `S²x` — hop-by-hop signal propagation |
| `run_full_graph_forward_pass()` | A complete, whole-graph, multi-layer GNN forward pass |
| `gcn_update()` | GCN's degree-normalized update: `h_i'=σ(Σ_j (1/√(d_i·d_j))·W·h_j)` |
| `gat_update()` | GAT's attention-based update: raw scores → softmax → weighted sum |
| `check_gcn_weight_sizing()` | The lecture's own "10 nodes, 5→16 features, what shape is W?" question |

## How to run this file

```bash
cd Deep-Learning/Lecture-09-Graph-Neural-Networks/code
pip install numpy --break-system-packages
python3 dl_lecture09_gnn_from_scratch.py
```
Works identically on Google Colab or Kaggle — no GPU required for this small-scale demo (real GNNs on large graphs, like PinSAGE, absolutely need GPUs and specialized libraries like PyTorch Geometric or DGL — this file is for understanding the mechanics, not production-scale training).

## Expected output (verified — produced by actually running this script)

- Check 1: adjacency matrix confirmed symmetric, degrees 6/3/5 — matches Worked Examples 1 & 2 exactly.
- Check 2: aggregated neighbor vector `[1.0, 0.6667]` and new embedding `[1.5, 0.3333]` — matches Worked Examples 3 & 4 exactly.
- Check 3: `Sx=[0,2,4]` and `S²x=[20,4,2]` — matches Worked Example 5 exactly.
- Check 4: a full 2-layer forward pass over a 4-node cycle graph, showing every node's embedding evolving as it incorporates first 1-hop, then 2-hop neighborhood information — a live demonstration of the "layer depth = hop distance" principle from the theory file.
- Check 5: GCN update `[0.7815, 0.3485]` — matches Worked Example 6 exactly.
- Check 6: GAT attention weights `[0.2543, 0.3265, 0.4192]` and update `[0.5825, 0.2905]` — matches Worked Example 7 (tiny rounding only).
- Check 7: GCN weight matrix sizing — `5×16` shape, 80 parameters — matches Worked Example 8 exactly.

`[← Exercises](../exercises/dl_lecture09_exercises.md) · [🔝 Top](#dl-lecture-09--code) · [🔝 Lecture Hub](../README.md)`
