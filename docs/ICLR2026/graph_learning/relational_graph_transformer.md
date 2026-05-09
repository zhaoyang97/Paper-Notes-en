---
title: >-
  [Paper Note] Relational Graph Transformer
description: >-
  [ICLR 2026][Graph Learning][Graph Transformer] This paper proposes RelGT, the first graph Transformer specifically designed for relational databases. Through multi-element tokenization (a 5-tuple of feature/type/hop distance/time/local structure encodings) and a local–global hybrid attention mechanism, RelGT consistently outperforms GNN baselines across all 21 tasks in the RelBench benchmark, with improvements of up to 18%.
tags:
  - ICLR 2026
  - Graph Learning
  - Graph Transformer
  - Relational Deep Learning
  - Multi-Element Tokenization
  - Heterogeneous Temporal Graph
  - Positional Encoding
date: 2026-05-08
content_hash: 27257e9a3c04f0ee
---

# Relational Graph Transformer

**Conference**: ICLR 2026
**arXiv**: [2505.10960](https://arxiv.org/abs/2505.10960)
**Code**: [GitHub](https://github.com/snap-stanford/relgt)
**Area**: Graph Learning
**Keywords**: Graph Transformer, Relational Deep Learning, Multi-Element Tokenization, Heterogeneous Temporal Graph, Positional Encoding

## TL;DR

This paper proposes RelGT, the first graph Transformer specifically designed for relational databases. Through multi-element tokenization (a 5-tuple of feature/type/hop distance/time/local structure encodings) and a local–global hybrid attention mechanism, RelGT consistently outperforms GNN baselines across all 21 tasks in the RelBench benchmark, with improvements of up to 18%.

## Background & Motivation

Enterprise data (financial transactions, e-commerce records, healthcare, etc.) is predominantly stored in relational databases. Relational Deep Learning (RDL) converts multi-table data into heterogeneous temporal graphs (Relational Entity Graphs, REGs) and applies GNNs to learn representations. However, GNNs exhibit inherent limitations:

**Insufficient structural expressiveness**: Message passing fails to capture complex structural patterns, e.g., two transactions that are both 2-hop neighbors are only indirectly connected via a shared customer.

**Limited long-range dependency modeling**: In a 2-layer GNN, product nodes can never directly interact (requiring 4 hops: transaction → customer → transaction → product).

**Existing graph Transformers are ill-suited for REGs**:
   - Traditional positional encodings (Laplacian PE, node2vec) do not generalize to large-scale heterogeneous graphs.
   - They lack the ability to model temporal dynamics and schema constraints.
   - Existing tokenization schemes discard critical structural information.

## Method

### Overall Architecture

RelGT comprises two core components:

1. **Multi-Element Tokenization (§3.1)**: Each node in the REG is decomposed into 5 element encodings that are then concatenated.
2. **Hybrid Transformer Network (§3.2)**: Local attention + global centroid attention.

Pipeline: seed node → temporally-aware sampling of $K$ neighbors → 5-element tokenization → local Transformer → global centroid attention → prediction head.

### Key Designs

#### Multi-Element Tokenization (5-Tuple Representation)

Each sampled node $v_j$ is represented as a 5-tuple $(x_{v_j}, \phi(v_j), p(v_i, v_j), \tau(v_j) - \tau(v_i), \text{GNN-PE}_{v_j})$:

1. **Node features $x_{v_j}$**: A multi-modal encoder processes numerical/categorical/text/image column attributes → $h_{\text{feat}} \in \mathbb{R}^d$
2. **Node type $\phi(v_j)$**: Table-level one-hot → projected via a learnable matrix → $h_{\text{type}} \in \mathbb{R}^d$
3. **Relative hop distance $p(v_i, v_j)$**: Shortest-path hop count from seed to neighbor → one-hot encoding → $h_{\text{hop}} \in \mathbb{R}^d$
4. **Relative time $\tau(v_j) - \tau(v_i)$**: Timestamp difference → linear transformation → $h_{\text{time}} \in \mathbb{R}^d$
5. **Subgraph GNN PE**: A lightweight GNN operates on the sampled subgraph with **random initial features** → $h_{\text{pe}} \in \mathbb{R}^d$

Final combination:
$$h_{\text{token}}(v_j) = O \cdot [h_{\text{feat}} \| h_{\text{type}} \| h_{\text{hop}} \| h_{\text{time}} \| h_{\text{pe}}]$$

where $O \in \mathbb{R}^{5d \times d}$ is a learnable mixing matrix.

**Elegant design of Subgraph GNN PE**: Random node feature initialization breaks symmetry to enhance expressiveness (Sato et al., 2021), while $Z_{\text{random}}$ is re-sampled at each training step (randomization strategy) to approximately preserve permutation equivariance.

**Core advantage**: No expensive global PE precomputation over the full graph is required; all encodings are local and lightweight.

#### Transformer Network: Local + Global

**Local module**: Full pairwise self-attention over $K$ sampled tokens for the seed node ($L$ Transformer layers), covering a broader receptive field than GNN message passing. Pooling is performed via a learnable linear combination.

$$h_{\text{local}}(v_i) = \text{Pool}(\text{FFN}(\text{Attention}(v_i, \{v_j\}_{j=1}^K))_L)$$

**Global module**: The seed node attends to $B$ learnable centroid tokens (centroids are dynamically updated during training via EMA K-Means), capturing database-level patterns that span beyond the local subgraph.

$$h_{\text{global}}(v_i) = \text{Attention}(v_i, \{c_b\}_{b=1}^B)$$

**Final representation**:
$$h_{\text{output}}(v_i) = \text{FFN}([h_{\text{local}}(v_i) \| h_{\text{global}}(v_i)])$$

### Loss & Training

- **Task-specific loss**: Selected according to the downstream task (MAE for regression, AUC for classification, etc.)
- **End-to-end training**: Replaces the GNN component within the RDL pipeline (Robinson et al., 2024)
- **Model scale**: 10–20M parameters, learning rate 1e-4
- **Sampling parameters**: $K=300$ local neighbors, $B=4096$ global centroids
- **Depth**: $L \in \{1, 4, 8\}$ searched when training nodes < 1M; fixed at $L=4$ otherwise
- **Batch size**: 256 for < 1M nodes; 1024 for > 1M nodes
- **Dropout**: $\{0.3, 0.4, 0.5\}$
- Temporally-aware sampling enforces $\tau(v_j) \leq \tau(v_i)$ to prevent data leakage

## Key Experimental Results

### Main Results

**Benchmark**: RelBench (7 datasets, 21 tasks) spanning e-commerce, clinical, social, and sports domains; training set sizes range from 1.3K to 5.4M.

**Regression tasks (MAE↓)**:

| Dataset | Task | RDL (GNN) | HGT | HGT+PE | **RelGT** | Gain |
|---------|------|-----------|-----|--------|-----------|------|
| rel-avito | ad-ctr | 0.041 | 0.046 | 0.048 | **0.035** | **15.85%** |
| rel-trial | site-success | 0.400 | 0.443 | 0.440 | **0.326** | **18.43%** |
| rel-amazon | item-ltv | 50.05 | 55.87 | 55.85 | **48.92** | 2.26% |
| rel-hm | item-sales | 0.056 | 0.064 | 0.064 | **0.054** | 4.29% |

**Classification tasks (AUC↑)**:

| Dataset | Task | RDL (GNN) | HGT | HGT+PE | **RelGT** | Gain |
|---------|------|-----------|-----|--------|-----------|------|
| rel-f1 | driver-top3 | 0.755 | 0.708 | 0.763 | **0.835** | **10.56%** |
| rel-avito | user-clicks | 0.659 | 0.638 | 0.646 | **0.683** | 3.64% |
| rel-stack | user-engagement | 0.902 | 0.885 | 0.882 | **0.905** | 0.35% |

**Overall statistics** (±1% threshold): 10 tasks with clear improvement / 9 on par / 2 with marginal degradation.

### Ablation Study

| Removed Component | ad-ctr | user-clicks | site-success | Trend |
|-------------------|--------|-------------|--------------|-------|
| w/o global module | -6.00% | +7.85% | -19.08% | Task-dependent |
| w/o GNN PE | -1.14% | **-15.15%** | — | **Consistently drops** |
| w/o node type | -7.14% | +5.01% | — | Mixed |
| w/o hop encoding | -3.43% | +5.77% | — | Mixed |
| w/o relative time | -9.14% | +8.37% | — | Mixed |

### Key Findings

1. **Subgraph GNN PE is the only component critical across all tasks**: Its removal consistently degrades performance, as it is the sole explicit encoding of local structure (parent–child relationships, cycles, etc.).
2. **The global module exhibits strong task dependency**: Removing it causes a 19% drop on site-success (which requires global context), yet yields a 7.9% improvement on user-clicks (where local information suffices).
3. **HGT+PE is inferior to RelGT**: Even augmenting HGT with Laplacian PE does not match RelGT, demonstrating that multi-element decomposition outperforms a single PE scheme.
4. **No expensive precomputation required**: All encodings are computed on sampled subgraphs, saving orders of magnitude in computation compared to full-graph Laplacian decomposition.

## Highlights & Insights

- **Multi-element tokenization paradigm**: Extends the NLP Transformer's "token + position" concept to a 5-element representation, decoupling the encoding of different information dimensions — superior to compressing all information into a single PE.
- **Random-feature GNN PE**: Elegantly leverages random initialization to enhance expressiveness, combined with per-step re-sampling to maintain equivariance — a theoretically and practically sound design.
- **Engineering-friendly**: Directly replaces the GNN component in the RDL pipeline while keeping all other infrastructure unchanged.
- **Global centroid mechanism**: EMA K-Means dynamically updates centroids during training without requiring additional preprocessing steps.

## Limitations & Future Work

1. Recommendation tasks are not covered (9 of the 30 RelBench tasks are excluded), as recommendation requires specialized treatment such as pair-wise learning.
2. Temporal encoding relies on a simple linear transformation; more advanced schemes (e.g., periodic functions, learnable temporal kernels) could be incorporated.
3. The fixed sampling size of $K=300$ may be suboptimal for extremely large or small local structures.
4. Global centroids introduce noise for certain tasks; an adaptive on/off mechanism could be considered.
5. Exhaustive hyperparameter search was not conducted, suggesting that reported results may have further room for improvement.

## Related Work & Insights

- **vs. GraphGPS**: GPS targets homogeneous static graphs and cannot handle heterogeneous/temporal settings; RelGT is specifically designed for REGs.
- **vs. HGT**: HGT handles heterogeneity but lacks effective PE and temporal modeling; RelGT's 5-element representation provides comprehensive coverage.
- **vs. RelGNN / ContextGNN**: These methods enhance GNNs but lack the flexibility of the Transformer's full pairwise attention.
- Insight: The multi-element tokenization paradigm is generalizable to other multi-dimensional heterogeneous graph scenarios (e.g., knowledge graphs, molecular networks, code dependency graphs).

## Rating

- Novelty: ★★★★☆ — Multi-element tokenization and random GNN PE are significant contributions.
- Technical Depth: ★★★★☆ — Careful design with theoretical justification for each component.
- Experimental Thoroughness: ★★★★★ — 21 tasks, multiple baselines, and comprehensive ablations.
- Writing Quality: ★★★★☆ — Clear structure with excellent figures.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](../../AAAI2026/graph_learning/ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[NeurIPS 2025\] Wavy Transformer](../../NeurIPS2025/graph_learning/wavy_transformer.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)

<!-- RELATED:END -->
