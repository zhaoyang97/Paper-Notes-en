---
title: >-
  [Paper Note] Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs
description: >-
  [AAAI 2026 (Oral)][Graph Learning][Fractal Nodes] This paper proposes Fractal Nodes (FN) to enhance long-range message passing in MPNNs. Subgraph-level aggregation nodes are generated via METIS graph partitioning…
tags:
  - "AAAI 2026 (Oral)"
  - "Graph Learning"
  - "Fractal Nodes"
  - "MPNN"
  - "Long-Range Dependencies"
  - "Over-Squashing"
  - "Graph Partitioning"
  - "MLP-Mixer"
  - "METIS"
date: 2026-05-08
content_hash: 2484dd4abbe87feb
---

# Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs

**Conference**: AAAI 2026 (Oral)
**arXiv**: [2511.13010](https://arxiv.org/abs/2511.13010)
**Code**: [https://github.com/jeongwhanchoi/MPNN-FN](https://github.com/jeongwhanchoi/MPNN-FN)
**Area**: Graph Neural Networks / Message Passing
**Keywords**: Fractal Nodes, MPNN, Long-Range Dependencies, Over-Squashing, Graph Partitioning, MLP-Mixer, METIS

## TL;DR

This paper proposes Fractal Nodes (FN) to enhance long-range message passing in MPNNs. Subgraph-level aggregation nodes are generated via METIS graph partitioning, combined with low-pass and high-pass filters (LPF+HPF) and a learnable frequency parameter $\omega$. MLP-Mixer is adopted for cross-subgraph communication. The approach achieves $O(L(|V|+|E|))$ linear complexity while matching or surpassing Graph Transformer performance, earning an AAAI Oral.

## Background & Motivation

GNNs face a fundamental tension between local and global information:

**Over-squashing in MPNNs**: As information propagates through multiple hops, fixed-width node vectors create a bottleneck—signals between distant nodes are compressed exponentially. Higher effective resistance between nodes further impedes information flow.

**Cost of Graph Transformers**: Global self-attention resolves long-range dependencies but incurs $O(|V|^2)$ quadratic complexity, limiting scalability and neglecting local graph topology.

**Limitations of virtual nodes**: A single global virtual node is a common augmentation strategy, but sharing one aggregation point across all nodes still creates an information bottleneck.

**Core Problem**: *Can long-range dependencies be resolved within the MPNN framework at linear complexity?* This paper answers affirmatively via Fractal Nodes, inspired by fractal structures observed in real-world networks—graph partitioning naturally induces fractal structure, where subgraphs mirror the connectivity patterns of the full graph.

## Method

### Overall Architecture

Fractal node layers are added on top of the original graph: (1) METIS partitions the graph into $C$ subgraphs; (2) each subgraph produces one fractal node connected to all nodes within it (single-hop shortcut connections); (3) at each layer, standard MPNN message passing is performed, followed by fractal node updates and information propagation back to original nodes; (4) an optional MLP-Mixer is applied at the final layer for cross-subgraph communication.

### Key Designs

1. **Fractal Node Construction**: Rather than simple mean pooling (which retains only the DC component / lowest-frequency signal), a dual-channel LPF+HPF design is employed:
   $$f_c^{(\ell+1)} = \text{LPF}(\{h_{v,c}\}) + \omega_c^{(\ell)} \cdot \text{HPF}(\{h_{v,c}\})$$
   LPF computes the subgraph mean (global information); HPF = node features $-$ LPF (local detail); $\omega$ is a learnable parameter controlling high-frequency contribution. Classification tasks tend to favor high frequencies, while regression tasks favor low frequencies.

2. **Resistance Reduction Theorem (Theorem 4.1)**: For any nodes $u, v$ in the original graph $\mathcal{G}$, the effective resistance in the augmented graph $\mathcal{G}_f$ satisfies $R_f(u,v) \leq R(u,v)$. Fractal nodes directly reduce effective resistance via single-hop shortcut connections, theoretically guaranteeing mitigation of over-squashing.

3. **MLP-Mixer for Cross-Subgraph Communication** (FN_M variant): At the final layer, MLP-Mixer (token-mixing + channel-mixing) is applied over all fractal nodes, enabling long-range inter-subgraph interaction without constructing a coarsened graph, at a cost of only $O(Cd^2)$.

4. **Two Variants**: FN (per-layer fractal node updates only) and FN_M (FN with additional MLP-Mixer cross-subgraph communication).

### Loss & Training

The same task loss as the base MPNN is used (cross-entropy for classification, MAE/MSE for regression). Fractal nodes serve as differentiable auxiliary structures and are trained end-to-end.

## Key Experimental Results

### Main Results: Performance Comparison on 6 Benchmark Datasets

| Method | Peptides-func AP↑ | Peptides-struct MAE↓ | MNIST Acc↑ | CIFAR10 Acc↑ |
|--------|-------------------|---------------------|------------|-------------|
| GCN | 0.6328 | 0.2758 | 0.9269 | 0.5423 |
| GINE | 0.6405 | 0.2780 | 0.9705 | 0.6131 |
| GraphGPS | 0.6534 | 0.2509 | 0.9805 | 0.7230 |
| GRIT | 0.6988 | 0.2460 | 0.9811 | 0.7647 |
| Graph-ViT/MLP-Mixer | 0.6970 | 0.2449 | 0.9846 | 0.7158 |
| **GINE+FN_M** | **0.7018** | — | **0.9858** | **0.7459** |
| **GatedGCN+FN_M** | **0.7040** | — | **0.9862** | **0.7528** |

### Ablation Study: Expressivity on Synthetic Datasets

| Method | CSL Acc | SR25 Acc | EXP Acc |
|--------|---------|---------|---------|
| GCN | 10.00 | 6.67 | 52.17 |
| GCN+FN_M | 39.67 | **100.0** | 86.40 |
| GINE | 10.00 | 6.67 | 51.35 |
| GINE+FN_M | 47.33 | **100.0** | 95.58 |
| GatedGCN | 10.00 | 6.67 | 51.25 |
| GatedGCN+FN_M | 49.67 | **100.0** | 96.50 |

### Key Findings

- **MPNN+FN surpasses Graph Transformers**: GINE+FN_M achieves 0.7018 AP on Peptides-func, outperforming GraphGPS (0.6534, +7.4%), while reducing complexity from $O(L|V|^2)$ to $O(L(|V|+|E|))$.
- **100% accuracy on SR25**: All MPNN baselines achieve only 6.67% on SR25; adding FN_M raises this to 100%, demonstrating a substantial gain in expressivity (beyond 1-WL).
- **Over-squashing mitigation**: In the TreeNeighboursMatch experiment, standard MPNNs fail at tree depth > 4, while MPNN+FN generalizes to depth = 7.
- **HPF is critical**: Removing HPF leads to significant performance degradation; high-frequency signals are essential for classification tasks.
- **METIS > Louvain > Random**: METIS-generated partitions preserve community structure, yielding more representative fractal nodes.
- **Optimal partition count $C=32$**: Too few partitions lead to coarse granularity; too many increase communication overhead.

## Highlights & Insights

- **"Are Graph Transformers Necessary?"**: The question itself carries strong community impact. The answer proposed here is: not necessarily—MPNN augmented with Fractal Nodes suffices, achieving linear complexity with comparable or superior performance.
- **Fractal geometry meets GNNs**: Graph partitioning naturally induces fractal structure, where subgraphs mirror the connectivity patterns of the full graph; betweenness centrality distributions corroborate this observation.
- **Theoretical limitation of mean pooling**: Theorem 3.1 proves that mean pooling retains only the lowest-frequency (DC) component, discarding all high-frequency information—providing theoretical justification for the LPF+HPF design.
- **MLP-Mixer succeeds in the graph domain**: Attention mechanisms are not the only means of capturing global information.

## Limitations & Future Work

- METIS partitioning is a preprocessing step with limited adaptability to dynamic graphs or structurally evolving settings.
- The partition count $C$ is a hyperparameter; different graphs may require different optimal values—adaptive partitioning strategies remain to be explored.
- Only a single layer of fractal nodes is employed; hierarchical fractal structures (fractals of fractals) may yield further improvements.
- The additional $O(Cd^2)$ cost introduced by MLP-Mixer may become non-negligible on very large graphs.

## Related Work & Insights

| Method Category | Representatives | Complexity | Long-Range Capacity | Local Preservation |
|----------------|----------------|------------|--------------------|--------------------|
| Standard MPNN | GCN, GINE | $O(L|E|)$ | Weak | Strong |
| Virtual Node | VN (Gilmer 2017) | $O(L(|V|+|E|))$ | Moderate | Strong |
| Graph Rewiring | DIGL, FoSR | $O(L|E'|)$ | Moderate | Topology altered |
| Graph Transformer | GraphGPS, Graphormer | $O(L|V|^2)$ | Strong | Weak |
| **Fractal Nodes (Ours)** | **MPNN-FN/FN_M** | **$O(L(|V|+|E|))$** | **Strong** | **Strong** |

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Fractal Nodes constitute a genuinely novel concept with both theoretical grounding and intuitive appeal.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark validation, synthetic expressivity experiments, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation and methodology are clearly presented with rigorous theoretical treatment.
- Value: ⭐⭐⭐⭐⭐ Likely to reshape the GNN community's perception of the necessity of Graph Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What Expressivity Theory Misses: Message Passing Complexity for GNNs](../../NeurIPS2025/graph_learning/what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](../../ACL2026/graph_learning/from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)

</div>

<!-- RELATED:END -->
