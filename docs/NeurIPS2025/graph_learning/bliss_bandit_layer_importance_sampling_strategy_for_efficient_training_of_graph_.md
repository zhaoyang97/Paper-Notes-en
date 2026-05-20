---
title: >-
  [Paper Note] BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks
description: >-
  [NeurIPS 2025][Graph Learning][Graph Sampling] This paper proposes BLISS, which formulates layer-wise neighbor sampling in GNNs as a multi-armed bandit problem. Using the EXP3 algorithm…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Graph Sampling"
  - "Layer-wise Importance Sampling"
  - "Multi-Armed Bandit"
  - "GNN Training"
  - "Scalability"
date: 2026-05-08
content_hash: fea9f21d4722e5ed
---

# BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2512.22388](https://arxiv.org/abs/2512.22388)  
**Code**: [https://github.com/linhthi/BLISS-GNN](https://github.com/linhthi/BLISS-GNN)  
**Area**: Graph Neural Networks / Efficient Training
**Keywords**: Graph Sampling, Layer-wise Importance Sampling, Multi-Armed Bandit, GNN Training, Scalability

## TL;DR
This paper proposes BLISS, which formulates layer-wise neighbor sampling in GNNs as a multi-armed bandit problem. Using the EXP3 algorithm, it dynamically adjusts per-edge sampling probabilities with the variance contribution of neighbors to node representations as the reward signal, achieving accuracy on par with or exceeding full-batch training on GCN and GAT.

## Background & Motivation

**Background**: Training GNNs on large graphs suffers from the neighborhood explosion problem — recursive aggregation at each layer causes the receptive field to grow exponentially. Common solutions involve neighbor sampling, categorized into three types: node-wise (GraphSAGE), layer-wise (FastGCN/LADIES), and subgraph-based (ClusterGCN/GraphSAINT).

**Limitations of Prior Work**:
- Node-wise sampling introduces redundancy (the same node is repeatedly sampled for different targets)
- Existing layer-wise samplers (LADIES, LABOR) rely on static sampling distributions that cannot adapt to shifting node importance during training
- BS-GNN applies bandits to node-wise sampling but does not extend to the layer-wise setting

**Key Challenge**: A dynamic balance must be struck between sampling efficiency (fewer samples reduce computation) and representation quality (more informative samples reduce variance), yet this balance shifts continuously throughout training.

**Goal**: To dynamically learn per-neighbor importance within the layer-wise sampling framework and adjust sampling probabilities accordingly.

**Key Insight**: Each edge is treated as an "arm"; the variance contribution of a node's representation serves as the reward; the EXP3 bandit algorithm adapts the sampling strategy online.

**Core Idea**: Layer-wise sampling + EXP3 bandit = adaptive neighbor importance sampling.

## Method

### Overall Architecture
Sampling proceeds top-down layer by layer: starting from the last layer $L$, per-layer sampling probabilities $p_j$ are computed and $k$ nodes are selected. Forward propagation uses Monte Carlo estimation for aggregation; after backpropagation, EXP3 updates the sampling weights.

### Key Designs

1. **Layer-wise Sampling Probability Computation**:

    - Function: Determines sampling by jointly considering neighbor importance across all target nodes.
    - Mechanism: $p_j = \sqrt{\sum_i (q_{ij} / \sum_{k \in \mathcal{N}_i} q_{ik})^2}$; the sampling probability is the L2 norm of the normalized weights that node $j$ contributes to all target nodes $i$.
    - Design Motivation: Layer-wise sampling reduces redundancy over node-wise sampling — a node need only be sampled once to serve all targets that depend on it.

2. **EXP3 Reward and Weight Update**:

    - Function: Updates the sampling strategy based on the actual contribution of each neighbor.
    - Mechanism: The reward $r_{ij} = \frac{\alpha_{ij}^2}{k \cdot q_j^2} \|h_j\|_2^2$ reflects the variance contribution of neighbor $j$ to node $i$'s representation. Weights are updated as $w_{ij}^{(t+1)} = w_{ij}^{(t)} \exp(\delta \hat{r}_{ij}^{(t)} / |\mathcal{N}_i|)$, with an exploration term controlled by $\eta$ in the probability update.
    - Design Motivation: High-reward neighbors carry more information and should be sampled more frequently; EXP3 guarantees an asymptotically optimal regret bound.

3. **GAT Adaptation (PLADIES)**:

    - Function: Extends BLISS to attention-based GNNs.
    - Mechanism: The true attention weights are approximated as $\alpha'_{ij} = \sum_{j \in S_i} q_{ij} \cdot \tilde{\alpha}_{ij} / \sum_{j \in S_i} \tilde{\alpha}_{ij}$, with at least one neighbor retained per node via a skip connection.
    - Design Motivation: GAT attention coefficients depend on neighbor information; at least one neighbor must be present for computation.

### Loss & Training
- Standard node classification loss (cross-entropy)
- Adam optimizer, lr = 0.002
- Bandit hyperparameters: $\eta = 0.4$, $\delta = \eta / 10^6$

## Key Experimental Results

### Main Results
F1 scores of BLISS vs. PLADIES across 6 datasets (GAT / GraphSAGE):

| Dataset | BLISS-GAT | PLADIES-GAT | BLISS-SAGE | PLADIES-SAGE |
|--------|-----------|-------------|------------|-------------|
| Citeseer | 0.706 | 0.683 | 0.580 | 0.601 |
| Cora | 0.813 | 0.809 | 0.795 | 0.772 |
| Pubmed | 0.731 | 0.718 | 0.597 | 0.557 |
| Flickr | 0.511 | 0.507 | 0.503 | 0.505 |

### Ablation Study: BLISS vs. Static Sampling

| Property | BLISS | PLADIES |
|------|-------|---------|
| Sampling Strategy | Dynamic (EXP3) | Static |
| GAT Adaptation | Yes (feedback attention) | Yes (skip connection) |
| Citeseer GAT Test F1 | **0.706** | 0.683 |
| Pubmed SAGE Test F1 | **0.597** | 0.557 |

### Key Findings
- **BLISS shows a larger advantage on GAT**: +2.3% on Citeseer GAT and +1.3% on Pubmed GAT — dynamic sampling benefits attention mechanisms more.
- **Mixed results on GraphSAGE**: PLADIES outperforms BLISS on Citeseer (0.601 vs. 0.580), indicating that dynamic sampling does not always help with simpler aggregation schemes.
- **Diminishing gap on larger datasets**: The difference on Flickr is <0.5%, suggesting that sampling strategy matters less when data is abundant.
- **Exploration–exploitation balance is critical**: A large $\eta$ leads to excessive exploration, while a small $\eta$ causes premature convergence.

## Highlights & Insights
- **Layer-wise bandit sampling** reduces redundancy compared to node-wise bandits (BS-GNN) with a clean conceptual formulation.
- The **reward definition** directly corresponds to variance reduction, providing theoretical grounding.
- The framework integrates well with various GNN architectures and exhibits good generality.

## Limitations & Future Work
- Comparison is limited to PLADIES; subgraph methods such as GraphSAINT and ClusterGCN are not evaluated.
- Instability on GraphSAGE suggests the reward signal may be noisy under simple aggregation.
- Bandit hyperparameters ($\eta$, $\delta$) require manual tuning.
- Efficiency on very large graphs (millions of nodes and beyond) remains untested.

## Related Work & Insights
- **vs. BS-GNN (Liu, 2020)**: BS-GNN applies node-wise bandit sampling; BLISS extends this to the layer-wise setting to reduce redundancy.
- **vs. LADIES/LABOR**: These methods use static layer-wise sampling; BLISS dynamically adapts to evolving importance.
- **vs. GraphSAINT**: Subgraph-based sampling operates under a different paradigm and cannot be directly compared.

## Rating
- Novelty: ⭐⭐⭐ Layer-wise bandits represent a natural extension of BS-GNN; conceptual innovation is moderate.
- Experimental Thoroughness: ⭐⭐⭐ Covers 6 datasets and 2 architectures, but comparison is limited to a single baseline.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and notation is rigorous.
- Value: ⭐⭐⭐ Offers meaningful reference for large-scale GNN training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)

</div>

<!-- RELATED:END -->
