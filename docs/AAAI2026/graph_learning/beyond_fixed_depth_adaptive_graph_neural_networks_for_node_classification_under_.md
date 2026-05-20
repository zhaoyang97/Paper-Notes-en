---
title: >-
  [Paper Note] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily
description: >-
  [AAAI2026][Graph Learning][GNN] This paper proposes AD-GNN, which theoretically analyzes node-level homophily/heterophily characteristics and adaptively assigns different aggregation depths to individual nodes…
tags:
  - "AAAI2026"
  - "Graph Learning"
  - "GNN"
  - "adaptive depth"
  - "heterophily"
  - "node classification"
  - "homophily"
date: 2026-05-08
content_hash: 7343abd2ce537719
---

# Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily

**Conference**: AAAI2026
**arXiv**: [2511.06608](https://arxiv.org/abs/2511.06608)  
**Code**: To be confirmed  
**Area**: Graph Learning
**Keywords**: GNN, adaptive depth, heterophily, node classification, homophily

## TL;DR
This paper proposes AD-GNN, which theoretically analyzes node-level homophily/heterophily characteristics and adaptively assigns different aggregation depths to individual nodes, enabling unified handling of node classification on both homophilic and heterophilic graphs within a single framework.

## Background & Motivation
Traditional GNNs are built upon the homophily assumption—connected nodes tend to share the same label. However, in real-world scenarios such as social networks and web graphs, heterophilic graphs are prevalent, where connected nodes often belong to different classes, leading to significant performance degradation.

Existing approaches suffer from two critical limitations:

1. **Fixed aggregation depth**: The vast majority of GNNs apply a uniform number of layers to all nodes, ignoring the fact that different nodes require different information propagation depths depending on their local homophily levels and neighborhood structures.
2. **Lack of a unified framework**: Existing methods are typically designed for either homophilic or heterophilic settings and cannot naturally adapt to both paradigms within a single architecture.

The authors' core observation is that even within the same graph, the optimal information propagation strategy varies across nodes. This motivates them to theoretically establish the connection between node-level structural-label characteristics and message-passing dynamics.

## Core Problem
How to adaptively determine GNN aggregation depth for each node based on its local neighborhood structure (degree, proportion of same-label/different-label neighbors), enabling a single model to efficiently handle both homophilic and heterophilic graphs?

## Method

### Theoretical Foundation: Signal Preservation Factor

For each node $v$, its profile is defined as $(d_v^+, d_v^-, d_v)$, denoting the number of same-label neighbors, different-label neighbors, and total degree, respectively. Under the Contextual Stochastic Block Model (CSBM) assumption, the Signal Preservation Factor is defined as:

$$\alpha_v = \frac{1 + d_v^+ - d_v^-}{d_v + 1}$$

**Theorem 1 (Label Aggregation Effect)**: After one layer of aggregation, the classification quality of node $v$ is:

$$Q_v = \frac{\alpha_v^2 (d_v + 1) \Delta^2}{\sigma_{\text{intra}}^2}$$

where $\Delta^2$ is the inter-class signal variance and $\sigma_{\text{intra}}^2$ is the intra-class noise variance. Three key corollaries follow:

- **Strong homophily** ($d_v^+ \gg d_v^-$): $\alpha_v \approx 1$; aggregation is always beneficial and classification quality grows linearly with degree.
- **Strong heterophily** ($d_v^- \gg d_v^+$): low-degree nodes experience signal cancellation ($Q_v \approx 0$), while high-degree nodes can recover performance by learning inverse relationships.
- **Mixed case** ($d_v^+ \approx d_v^-$): $\alpha_v \approx \frac{1}{d_v+1}$; signal cancellation worsens with increasing degree.

**Theorem 2 (Multi-layer Aggregation Effect)**: After $n$ layers, classification quality is:

$$Q_v^n = \frac{\alpha_v^{2n} (d_v+1)^n \Delta^2}{\sigma_{\text{intra}}^2}$$

If $|\alpha_v| < 1$, signal degradation worsens exponentially with depth.

### AD-GNN Architecture

Based on the theoretical analysis, the paper introduces the Depth Benefit Metric:

$$\varepsilon_v^n = \frac{Q_v^n}{Q_v^0} = (\alpha_v^2 \cdot (d_v + 1))^n$$

$\varepsilon_v^n > 1$ indicates that deeper aggregation is beneficial; $< 1$ indicates it is harmful.

**Stopping Depth Assignment**: Given a maximum allowed depth $t_{\max}$, a learnable monotonically increasing threshold function $\tau_\theta(t) = \lambda + (1-\lambda) \cdot \theta(t)$ determines the stopping layer $T(v)$ for each node. As depth increases, nodes with lower depth benefit are progressively filtered out and excluded from further aggregation.

**Message Passing**: For $t \leq T(v)$, standard aggregation-update operations are performed; for $t > T(v)$, node representations remain unchanged. This mechanism can be used as a plug-and-play enhancement for mainstream GNN backbones such as GCN, GAT, and GraphSAGE.

### Depth Benefit Metric Computation

Since labels are incomplete under semi-supervised training, a learnable function $f_\delta(\mathbf{h}_u, \mathbf{h}_v) \in [0,1]$ is used to estimate the probability that adjacent nodes share the same label, thereby approximating $\hat{\alpha}_v$ and $\hat{\varepsilon}_v$. The training objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{reg}}$$

where the regularization term $\mathcal{L}_{\text{reg}}$ leverages labeled edges in the training set to supervise $f_\delta$ toward learning meaningful similarity estimates.

### Fast Variant: AD-GNN_fast

This variant replaces the learnable similarity function with a static degree-based approximation. Exploiting degree assortativity (high-degree nodes tend to connect with high-degree nodes), the same-label probability is estimated as $p_{uv} = \frac{d_u \times d_v}{\max_{(i,j) \in E}(d_i \times d_j)}$. This eliminates the computational overhead of learning $f_\delta$ and the regularization term, reducing complexity from $\mathcal{O}(|E| \times d + t_{\max} \times (|E|+|V|))$ to $\mathcal{O}(t_{\max} \times (|E|+|V|))$.

## Key Experimental Results

Evaluations are conducted on 11 datasets (5 homophilic + 6 heterophilic) using 6 GNN backbones:

| Backbone | Baseline (Cornell) | AD-Version | Baseline (Texas) | AD-Version |
|----------|-------------------|------------|-----------------|------------|
| GCN | 55.14 | **88.51** | 60.00 | **92.30** |
| GAT | 53.64 | **86.17** | 61.21 | **90.49** |
| GraphSAGE | 75.95 | **89.57** | 82.43 | **92.95** |
| MixHop | 73.51 | **90.21** | 77.84 | **94.43** |

Key findings:

- Improvements are most pronounced on heterophilic graphs (e.g., GCN on Cornell improves from 55.14% → 88.51%, a gain of 33 percentage points).
- Consistent gains are also observed on homophilic graphs (e.g., Photo improves from 89.30% → 94.10%).
- The fast variant achieves performance close to or occasionally exceeding the full version.
- Scalability experiment (ogbn-arxiv): AD-GCN_fast has the same parameter count as GCN, incurs only ~5% additional runtime overhead, and improves accuracy from 68.39% to 70.42% at depth=8.
- AD-GNN effectively alleviates over-smoothing and maintains stable performance in deep networks.

## Highlights & Insights
1. **Theory-driven architecture design**: A complete node-level theoretical framework is built from the signal preservation factor, deriving distinct behaviors for strong homophily, strong heterophily, and mixed scenarios, with strong alignment between theoretical predictions and experimental results.
2. **Plug-and-play design**: AD-GNN integrates seamlessly into any existing GNN backbone (GCN/GAT/GraphSAGE/MixHop/GATv2/DirGNN) without modifying the backbone architecture.
3. **Unified handling of homophily and heterophily**: A single framework handles both paradigms without requiring prior knowledge of graph type.
4. **Practical fast variant**: AD-GNN_fast achieves near-full-version performance at minimal additional computational cost.
5. **Side benefit: alleviating over-smoothing**: The adaptive depth mechanism naturally prevents redundant aggregation layers from over-smoothing node signals.

## Limitations & Future Work
1. **Strong theoretical assumptions**: The framework relies on CSBM structure, Gaussian features, binary classification, and layer-independence assumptions, which may not hold in real graphs.
2. **Degree assortativity assumption in the fast variant**: AD-GNN_fast depends on degree assortativity (high-degree nodes connect to high-degree nodes), which may fail on graphs where degree is uncorrelated with labels.
3. **Hyperparameter $\lambda$**: Requires separate tuning for homophilic and heterophilic graphs ($\lambda=0$ is optimal for homophilic graphs; $\lambda > 0$ is needed for low-degree heterophilic graphs); future work could consider data-driven learning of this parameter.
4. **Stopping depth granularity**: The current approach uses a global threshold function for stopping decisions, without fully exploiting dynamic changes in node-local topology.
5. **Evaluated on node classification only**: Performance on other tasks such as graph classification and link prediction remains unexplored.

## Related Work & Insights
- **Heterophily-aware GNNs** (MixHop, DirGNN, etc.): AD-GNN is orthogonal and complementary, capable of providing further gains on top of these methods.
- **Depth-adaptive GNNs** (wu2024depth uses reinforcement learning for depth search; ADMP-GNN uses centrality-based heuristics): AD-GNN is the first to analyze the impact of neighborhood label composition on propagation depth from a heterophily-theoretic perspective.
- **Graph rewiring methods** (modifying graph structure to increase homophily): AD-GNN does not alter the graph structure but instead adaptively adjusts the number of aggregation layers per node.

The "signal preservation factor" is a concise and powerful theoretical tool that can be generalized to other scenarios requiring analysis of aggregation effectiveness. The adaptive depth idea can be extended to adaptive layer selection in Transformers (early exit), particularly in graph Transformer settings. The signal cancellation problem in nodes with mixed homophily/heterophily suggests that node-level propagation strategies, rather than global strategies, may be necessary in heterogeneous graph learning.

## Rating
- Novelty: ⭐⭐⭐⭐ (Theory-driven adaptive depth is a novel approach; the formalization of the signal preservation factor is clean and rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (11 datasets × 6 backbones, with full ablation, scalability, and over-smoothing analyses)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear, structure is complete, and corollaries align well with experiments)
- Value: ⭐⭐⭐⭐ (Plug-and-play, theoretically grounded, and of high practical value to the graph learning community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[NeurIPS 2025\] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](../../NeurIPS2025/graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)

</div>

<!-- RELATED:END -->
