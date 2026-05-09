---
title: >-
  [Paper Note] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks
description: >-
  [NeurIPS 2025][Graph Learning][Graph Prompting] This paper proposes GraphTOP, the first graph topology-oriented prompting framework, which formulates topology-oriented prompting as an edge rewiring problem and relaxes it into a continuous space via Gumbel-Softmax. GraphTOP outperforms six baseline methods across five datasets and four pre-training strategies.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Graph Prompting
  - Topology
  - Edge Rewiring
  - Pre-training
  - Gumbel-Softmax
date: 2026-05-08
content_hash: a8ab4640ff10ed8a
---

# GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2510.22451](https://arxiv.org/abs/2510.22451)
**Code**: [https://github.com/xbfu/GraphTOP](https://github.com/xbfu/GraphTOP)
**Area**: Graph Learning / Graph Prompting
**Keywords**: Graph Prompting, Topology, Edge Rewiring, Pre-training, Gumbel-Softmax

## TL;DR
This paper proposes GraphTOP, the first graph topology-oriented prompting framework, which formulates topology-oriented prompting as an edge rewiring problem and relaxes it into a continuous space via Gumbel-Softmax. GraphTOP outperforms six baseline methods across five datasets and four pre-training strategies.

## Background & Motivation

**State of the Field**: The "pre-train then adapt" paradigm is widely adopted in graph learning. Graph prompting, which adapts frozen pre-trained GNNs by modifying the input graph data, is an efficient adaptation strategy.

**Limitations of Prior Work**: Existing graph prompting methods are almost exclusively feature-oriented — they operate only on node features or latent representations (e.g., GPF, All-in-one, GraphPrompt), entirely overlooking graph topology, which is an intrinsic characteristic of graph-structured data. However, graph representations are determined not only by features but also by topological structure.

**Root Cause**: How can prompting be achieved by modifying graph topology? Edge selection is a discrete optimization problem that is not directly amenable to gradient-based optimization.

**Paper Goals**: To design a topology-oriented graph prompting framework that adapts pre-trained GNN models to downstream node classification tasks through topological modification.

**Starting Point**: Topology prompting is formulated as an edge rewiring problem, relaxed into a continuous probability space via Bernoulli reparameterization and Gumbel-Softmax.

**Core Idea**: The framework learns the existence probability of each edge as a topology prompt, renders it differentiably trainable through Gumbel-Softmax, and employs entropy regularization to ensure tight relaxation and graph sparsity.

## Method

### Overall Architecture
Given a pre-trained GNN $f_{\theta^*}$ and an input graph $\mathcal{G} = (\mathbf{A}, \mathbf{X})$, GraphTOP learns a prompted graph $\tilde{\mathcal{G}} = (\mathbf{S}, \mathbf{X})$, where $\mathbf{S}$ is a learned adjacency matrix. By substituting $\mathbf{S}$ for the original $\mathbf{A}$, the frozen GNN can generate more task-appropriate representations for downstream tasks.

### Key Designs

1. **Edge Rewiring Formulation**:

    - Function: Models the topology prompt as a binary decision $s_{ij} \in \{0,1\}$ for each node pair regarding edge existence.
    - Mechanism: Each $s_{ij}$ is treated as a Bernoulli random variable with parameter $p_{ij}$, made differentiably sampleable via Gumbel-Softmax reparameterization. Theorem 1 proves that $\Pr(G_1 - G_2 + \log(p/(1-p)) \geq 0) = p$.
    - Design Motivation: Relaxes the discrete edge selection problem into continuous probability optimization, enabling gradient-based training.

2. **Shared Trainable Projector**:

    - Function: Computes edge probabilities via a 2-layer MLP applied to pre-trained GNN node representations: $p_{ij} = \sigma(W_2(\text{ReLU}(W_1(h_i + h_j))))$.
    - Mechanism: Rather than learning each $p_{ij}$ independently (which would cause parameter explosion), a shared projector infers edge probabilities from node representations.
    - Design Motivation: Substantially reduces parameter count while leveraging semantic information encoded in pre-trained representations to guide topological modification.

3. **Subgraph-Constrained Rewiring**:

    - Function: Restricts edge rewiring to the $\rho$-hop local subgraph of each target node (default $\rho=2$).
    - Mechanism: Only edges between the target node and other nodes within the subgraph are rewired; all remaining edges are preserved. Complexity is reduced from $O(D^{2\rho})$ to $O(D^{\rho})$.
    - Design Motivation: GNN representations depend primarily on local subgraphs; global rewiring is unnecessary and computationally expensive.

### Loss & Training
- Total loss: $\mathcal{L}_P + \lambda_1 \mathcal{L}_E + \lambda_2 \mathcal{L}_S$
- $\mathcal{L}_P$: Cross-entropy loss for node classification.
- $\mathcal{L}_E$: Entropy regularization (encourages probabilities toward 0 or 1 to ensure tight relaxation).
- $\mathcal{L}_S$: Sparsity regularization (controls edge density in the prompted graph, with $\gamma=0.5$).
- Temperature annealing: $\tau$ decays linearly, transitioning from probabilistic approximation to deterministic output.

## Key Experimental Results

### Main Results — 5-shot Node Classification

| Pre-training | Method | Cora | PubMed | Amazon | Minesweeper | Flickr |
|---|---|---|---|---|---|---|
| GraphCL | Linear Probe | 55.69 | 67.30 | 23.19 | 67.59 | 29.31 |
| GraphCL | ProNoG (SOTA) | 60.01 | 68.17 | 23.26 | 65.48 | 26.17 |
| GraphCL | **GraphTOP** | **65.12** | **69.42** | **27.15** | **70.23** | **31.84** |

### Ablation Study

| Configuration | Key Findings |
|---|---|
| w/o entropy regularization $\mathcal{L}_E$ | Topological instability at inference; performance degrades. |
| w/o sparsity regularization $\mathcal{L}_S$ | Graph becomes overly dense; performance degrades. |
| Feature prompt vs. topology prompt | Topology prompting and feature prompting are complementary. |

### Key Findings
- GraphTOP consistently outperforms all baselines across four pre-training strategies (GraphCL, SimGRACE, LP-GPPT, LP-GraphPrompt).
- Theorem 2 theoretically proves that edge rewiring increases inter-class representation distance: $\text{Dist}' = \frac{p+q}{|p-q|} \text{Dist} > \text{Dist}$.
- The additional computational overhead of topology prompting is minimal: $O(D^2 d_l^2)$ compared to $O(KD^2 d_l^2)$ for the GNN itself.

## Highlights & Insights
- **First exploration of topology-oriented graph prompting**: All prior work operates exclusively on features; this paper fills the gap in the topological dimension.
- **Elegant combination of Gumbel-Softmax and entropy regularization**: Gumbel-Softmax enables differentiable edge selection, while entropy regularization ensures that edge probabilities converge toward deterministic values, yielding stable topology at inference.
- **Theoretical guarantees**: Based on the CSBM model, the paper formally proves that topology prompting increases inter-class distance, providing rigorous theoretical support.

## Limitations & Future Work
- **Node classification only**: The current framework focuses on node-level tasks; topology prompt design for graph-level and edge-level tasks remains unexplored.
- **5-shot setting**: Strong performance under extremely low annotation budgets is demonstrated, but effectiveness in fully supervised settings has not been validated.
- **Information loss from subgraph constraints**: The $\rho=2$ restriction may miss important long-range connections.
- **Potential improvements**: Joint optimization of topology and feature prompts; exploration of adaptive $\rho$.

## Related Work & Insights
- **vs. GPF/All-in-one**: These methods modify only node features, whereas GraphTOP modifies topological structure; the two approaches are orthogonal and complementary.
- **vs. Graph Structure Learning (GSL)**: GSL also modifies graph structure but requires end-to-end GNN training; GraphTOP keeps the GNN frozen, making it more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First extension of graph prompting to the topological dimension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets × four pre-training strategies, though limited to node classification.
- Writing Quality: ⭐⭐⭐⭐ Theoretical analysis is clear; notation is dense but well-organized.
- Value: ⭐⭐⭐⭐ Opens a new direction for graph prompting with practical application potential.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](graph_neural_networks_for_interferometer_simulations.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)

<!-- RELATED:END -->
