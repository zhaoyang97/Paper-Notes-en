---
title: >-
  [Paper Note] Cooperative Sheaf Neural Networks
description: >-
  [ICLR 2026][Graph Learning][Sheaf Neural Networks] This paper proposes in/out-degree sheaf Laplacians defined on directed graphs for cellular sheaves…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Sheaf Neural Networks"
  - "cooperative behavior"
  - "directed graphs"
  - "oversquashing"
  - "heterophilic graphs"
date: 2026-05-08
content_hash: ea9867b0e62f4398
---

# Cooperative Sheaf Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2507.00647](https://arxiv.org/abs/2507.00647)
**Code**: N/A
**Area**: Graph Learning / Graph Neural Networks
**Keywords**: Sheaf Neural Networks, cooperative behavior, directed graphs, oversquashing, heterophilic graphs

## TL;DR

This paper proposes in/out-degree sheaf Laplacians defined on directed graphs for cellular sheaves, and constructs a Cooperative Sheaf Neural Network (CSNN) that enables nodes to independently select information propagation/reception strategies, thereby simultaneously mitigating oversquashing and handling heterophilic tasks.

## Background & Motivation

**Background**: Sheaf Neural Networks (SNNs) generalize the diffusion mechanism of GNNs by defining cellular sheaves over graphs, and have been shown to handle heterophilic tasks while alleviating oversmoothing.

**Limitations of Prior Work**: Classical SNNs are defined on undirected graphs, preventing nodes from independently choosing to "only propagate" or "only receive" information. If a node $i$ wishes to block all incoming neighbor information, it must set all associated restriction maps to zero ($\mathcal{F}_{i \unlhd e}=0$), which simultaneously cuts off $i$'s ability to propagate information outward.

**Key Challenge**: The sheaf Laplacian structure in SNNs conflates PROPAGATE with LISTEN, making it impossible to fully decouple the four cooperative behaviors (STANDARD / LISTEN / PROPAGATE / ISOLATE).

**Goal**: To enable nodes in SNNs to independently decide whether to propagate and/or receive information, realizing genuine cooperative behavior and better mitigating oversquashing.

**Key Insight**: Splitting each undirected edge into a pair of directed edges, and defining cellular sheaves along with their in/out-degree sheaf Laplacians on the resulting directed graph.

**Core Idea**: By separating source maps $\mathbf{S}_i$ and target maps $\mathbf{T}_i$ via the sheaf Laplacian on directed graphs, each node can independently control the flow of information into and out of itself.

## Method

### Overall Architecture

CSNN converts the input undirected graph into a directed graph (each undirected edge is split into a pair of directed edges), learns a pair of conformal maps $\mathbf{S}_i$ (source map) and $\mathbf{T}_i$ (target map) for each node $i$, performs normalized diffusion by combining the out-degree and transposed in-degree sheaf Laplacians, and then completes node representation learning through NSD-style iterative updates.

### Key Designs

1. **Directed Graph Cellular Sheaf and In/Out-Degree Laplacians**:

    - **Function**: Defines a sheaf structure on directed graphs, distinguishing restriction maps for nodes acting as sources versus targets.
    - **Mechanism**: The out-degree sheaf Laplacian is $L_{\mathcal{F}}^{\text{out}}(\mathbf{X})_i = \sum_{j \in N(i)} (\mathbf{S}_i^\top \mathbf{S}_i \mathbf{x}_i - \mathbf{T}_i^\top \mathbf{S}_j \mathbf{x}_j)$; the in-degree counterpart is analogous but uses $\mathbf{T}$ to control the receiving end. Asymmetric diffusion is achieved by composing $(\Delta_\mathcal{F}^{\text{in}})^\top \Delta_\mathcal{F}^{\text{out}}$.
    - **Design Motivation**: In the undirected sheaf Laplacian, setting $\mathcal{F}_{i \unlhd e}=0$ simultaneously severs both incoming and outgoing connections (Proposition 3.1). With the directed decomposition, $\mathbf{S}_i=0$ (no propagation) and $\mathbf{T}_i=0$ (no listening) can be set independently.

2. **Flat Vector Bundle Parameterization**:

    - **Function**: Replaces per-edge restriction maps with only two conformal maps per node.
    - **Mechanism**: For all neighbors $j$, the maps $\mathcal{F}_{i \unlhd ij} = \mathbf{S}_i$ and $\mathcal{F}_{i \unlhd ji} = \mathbf{T}_i$ are shared. Orthogonal matrices are constructed via Householder reflections and then multiplied by a learned positive scalar.
    - **Design Motivation**: A general cellular sheaf requires $2m$ restriction maps (where $m$ is the number of edges), whereas the flat vector bundle requires only $2n$ (where $n$ is the number of nodes), substantially reducing computational cost.

3. **Extended Receptive Field and Selective Attention**:

    - **Function**: Theoretically establishes that CSNN can access $2t$-hop neighbors per layer and selectively ignore intermediate nodes along a path.
    - **Mechanism**: By appropriately configuring the $\mathbf{S}$ and $\mathbf{T}$ maps, $\partial \mathbf{x}_i^{(t)} / \partial \mathbf{x}_j^{(0)}$ exhibits high sensitivity to target node $j$ at distance $t$ while approaching zero for intermediate nodes.
    - **Design Motivation**: Traditional GNNs with $t$ layers can only access $t$-hop neighbors, and information is exponentially compressed along paths, leading to oversquashing. The selective attention mechanism of CSNN effectively alleviates this issue.

### Loss & Training

Diffusion iterations follow the NSD style, with restriction maps learned end-to-end via neural networks. Householder reflections ensure orthogonality, and multiplying by a learned positive scalar yields conformal maps.

## Key Experimental Results

### Main Results

| Dataset | Metric | CSNN | Best Baseline | Gain |
|--------|------|------|----------|------|
| roman-empire | Acc | 92.63 | BuNN 91.75 | +0.88 |
| minesweeper | AUROC | 99.07 | BuNN 98.99 | +0.08 |
| tolokers | AUROC | 85.45 | CO-GNN 84.84 | +0.61 |
| questions | AUROC | 79.31 | BuNN 78.75 | +0.56 |
| Wisconsin | Acc | 90.00 | O(d)-NSD 89.41 | +0.59 |

### Ablation Study

| Configuration | NeighborsMatch Accuracy | Note |
|------|----------------------|------|
| CSNN (r=2~8) | 100% at all depths | Perfectly resolves oversquashing |
| BuNN (r≥7) | 71%→42% | Degrades starting at r=7 |
| NSD (r≥4) | 5% | Severe oversquashing |
| GCN/GIN (r≥4) | Fails | Cannot handle long-range dependencies |

### Key Findings

- CSNN maintains 100% accuracy on NeighborsMatch across all tree depths, significantly outperforming all sheaf-based and non-sheaf baselines.
- CSNN achieves state-of-the-art results on 9 out of 11 node classification datasets, with particularly strong performance on highly heterophilic datasets.
- On the peptides-func graph classification task, CSNN achieves 73.38 AP, surpassing BuNN (72.76), GPS, SAN, and other methods.

## Highlights & Insights

- The paper rigorously proves from an algebraic topology perspective that SNNs cannot realize cooperative behaviors (Proposition 3.1), and then elegantly resolves this limitation using directed sheaves.
- The flat vector bundle design reduces the parameter count from $O(m)$ to $O(n)$, ensuring computational efficiency alongside theoretical advantages.
- The paper theoretically demonstrates that CSNN achieves a receptive field of $2t$-hop per layer rather than the conventional $t$-hop, offering a new perspective for mitigating oversquashing.

## Limitations & Future Work

- Cooperative behavior selection is determined implicitly through continuous parameters rather than through explicit modeling of discrete actions.
- CSNN does not achieve state-of-the-art results on certain datasets such as amazon-ratings, suggesting that the simplification introduced by the flat vector bundle may sacrifice flexibility.
- Experiments are conducted only on medium-scale graphs; scalability to large graphs (>100K nodes) remains to be evaluated.

## Related Work & Insights

- **vs. CO-GNN**: CO-GNN employs a discrete Gumbel-Softmax action network to select cooperative modes, whereas CSNN achieves this naturally through continuous parameters, avoiding training instability and hyperparameter sensitivity.
- **vs. NSD**: NSD is based on undirected sheaves; CSNN extends expressive power through directed sheaves and substantially outperforms NSD on NeighborsMatch.
- **vs. BuNN**: BuNN is also sheaf-based, but exhibits clear degradation in oversquashing tests at r≥7, whereas CSNN consistently maintains 100% accuracy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The directed sheaf Laplacian is an entirely novel mathematical construction with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage via synthetic benchmarks, 11 node classification datasets, and 2 graph classification datasets.
- **Writing Quality**: ⭐⭐⭐⭐ The definition–proposition–proof structure is clear and mathematically rigorous.
- **Value**: ⭐⭐⭐⭐ Provides new theoretical and practical directions for sheaf-based GNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[CVPR 2026\] Hyperbolic Busemann Neural Networks](../../CVPR2026/graph_learning/hyperbolic_busemann_neural_networks.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)

</div>

<!-- RELATED:END -->
