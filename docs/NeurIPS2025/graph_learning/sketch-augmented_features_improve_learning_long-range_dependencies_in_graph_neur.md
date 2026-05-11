---
title: >-
  [Paper Note] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks
description: >-
  [NeurIPS 2025][Graph Learning][Graph Neural Networks] This paper proposes Sketched Random Features (SRF), which injects random kernel-space projections of node features into every layer of a standard message-passing GNN…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Oversquashing"
  - "Oversmoothing"
  - "Random Features"
  - "Johnson-Lindenstrauss Transform"
date: 2026-05-08
content_hash: 21b17d1ecd0c09ed
---

# Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2511.03824](https://arxiv.org/abs/2511.03824)
**Code**: [GitHub](https://github.com/ryienh/sketched-random-features)
**Area**: Graph Learning
**Keywords**: Graph Neural Networks, Oversquashing, Oversmoothing, Random Features, Johnson-Lindenstrauss Transform

## TL;DR

This paper proposes Sketched Random Features (SRF), which injects random kernel-space projections of node features into every layer of a standard message-passing GNN, simultaneously alleviating oversquashing, oversmoothing, and limited expressiveness, with rigorous theoretical guarantees and low computational overhead.

## Background & Motivation

Standard message-passing GNNs (MPGNNs) suffer from three well-recognized fundamental limitations:

**Oversquashing**: Signals from distant nodes are compressed into fixed-dimensional vectors over multi-hop propagation, causing severe information loss. The receptive field grows exponentially while the hidden dimension $|\mathbf{h}|$ remains fixed, making long-range dependency capture difficult.

**Oversmoothing**: As network depth increases, node representations converge exponentially to nearly identical values, measured by the Dirichlet energy: $\mathcal{D}(H^{(\ell)}) = \frac{1}{N}\sum_{i}\sum_{j \in \mathcal{N}(i)}\|\mathbf{h}_i^{(\ell)} - \mathbf{h}_j^{(\ell)}\|_2^2$, which decays rapidly as $\ell$ grows.

**Limited Expressiveness**: The expressiveness of MPGNNs is bounded by the 1-WL test, preventing them from distinguishing many pairs of non-isomorphic graphs.

Existing solutions either introduce $O(N^2)$ Graph Transformers, rely on pure random node features (slow convergence), or use topology-based encodings that are difficult to make simultaneously unique, distance-sensitive, and equivariant.

**Key Observation**: Existing methods primarily exploit topological information for positional encoding, overlooking the typically rich node features in real-world graphs. SRF directly leverages node features to provide a global, feature-distance-sensitive signal.

## Method

### Overall Architecture

SRF consists of two preprocessing steps: (1) a kernel embedding operator $\mathcal{E}$ maps node features into a random kernel feature space; (2) a sketching operator $\mathcal{S}$ applies random projections across nodes to mix global information. The resulting sketch features are concatenated to the node hidden states at every GNN layer.

### Key Designs

1. **Kernel Embedding Operator $\mathcal{E}$**: Applies kernel approximation independently to each node feature.

   For the raw feature matrix $X \in \mathbb{R}^{N \times F}$, random Fourier features are used for kernel approximation:
   $\varphi_{\text{RBF}}(\mathbf{x}) = \sqrt{\frac{2}{D}} [\cos(\boldsymbol{\omega}_1^\top \mathbf{x} + b_1), \ldots, \cos(\boldsymbol{\omega}_D^\top \mathbf{x} + b_D)]^\top$

   satisfying $\mathbb{E}[\varphi(\mathbf{x}_i)^\top \varphi(\mathbf{x}_j)] = \kappa(\mathbf{x}_i, \mathbf{x}_j)$ (unbiased kernel estimation). Three kernel choices are supported: linear, RBF, and Laplacian.

   **Design Motivation**: The kernel mapping preserves distance relationships in the feature space, rendering subsequent projections meaningful.

2. **Additive Gaussian Sketching Operator $\mathcal{S}_{AG}$**: Mixes global information via random projections across nodes.

   $\mathcal{S}_{AG}(\Phi) = \left(I + \frac{1}{\sqrt{N}} G\right) \Phi$

   where $G \in \mathbb{R}^{N \times N}$ with $G_{ij} \sim \mathcal{N}(0,1)$ i.i.d. Multi-dimensional estimation is achieved by concatenating $k$ independent sketches via the $k$-th order projection $\mathcal{S}_{AG}^{(k)}$.

   The final SRF is defined as: $Z = \mathcal{S}^{(k)}(\mathcal{E}(X)) \in \mathbb{R}^{N \times (kD)}$

   **Design Motivation**: Unlike conventional JLT used for dimensionality reduction, random projections here achieve cross-node information mixing: each $\mathbf{z}_i$ contains a linear combination of all node features, bypassing the topological bottleneck of the graph.

3. **Injection into MPGNN**: SRF is concatenated to the hidden state at every layer.

   $\tilde{\mathbf{h}}_i^{(\ell)} = [\mathbf{h}_i^{(\ell)} | \mathbf{z}_i]$
   $\mathbf{h}_i^{(\ell+1)} = f\left(\tilde{\mathbf{h}}_i^{(\ell)}, \{\tilde{\mathbf{h}}_j^{(\ell)} : j \in \mathcal{N}(i)\}\right)$

   SRF is computed only once and reused at every layer, incurring negligible training cost.

### Theoretical Properties

SRF satisfies five key properties, mathematically proving its mitigation of the three major GNN limitations:

| Property | Content | Limitation Addressed |
|----------|---------|---------------------|
| Unbiased kernel estimation | $\mathbb{E}[\mathbf{z}_i^\top \mathbf{z}_j] = \kappa(\mathbf{x}_i, \mathbf{x}_j)$ | Oversmoothing |
| Kernel distance sensitivity | $(1-c)\|\varphi(\mathbf{x}_i) - \varphi(\mathbf{x}_j)\| \leq \|\mathbf{z}_i - \mathbf{z}_j\| \leq (1+c)\|\ldots\|$ | Oversmoothing |
| Cross-node information | Each $\mathbf{z}_i$ contains a linear combination of all node embeddings | Oversquashing |
| Almost surely unique | $P(\mathbf{z}_i \neq \mathbf{z}_j) = 1, \forall i \neq j$ | Expressiveness |
| Expected permutation equivariance | $\mathbb{E}[f(P_\pi X)] = \mathbb{E}[P_\pi f(X)]$ | Equivariance |

### Complexity Analysis

With structured random matrices (SRM), the overall complexity is $O(NFD + kN^2 \log N)$ with $O(N)$ storage. SRF is precomputed only once and does not scale with training epochs.

## Key Experimental Results

### Synthetic Tasks: Expressiveness and Long-Range Dependencies

| Dataset | Baseline GNN | Ablation ($\mathcal{S}_{id}$) | SRF ($\mathcal{S}_{AG}^{(1)}$) | SRF ($\mathcal{S}_{AG}^{(8)}$) |
|---------|-------------|-------------------------------|-------------------------------|-------------------------------|
| CSL (Acc↑) | 0.100 | 0.100 | 1.000 | 1.000 |
| EXP (Acc↑) | 0.518 | 0.520 | 1.000 | 1.000 |

Non-isomorphic graphs that baselines fail to distinguish are perfectly separated after SRF augmentation.

### Real-World Dataset Performance

| Dataset | GINE (Baseline) | R-PEARL | SRF-$\mathcal{E}_{\text{RBF}}$ |
|---------|----------------|---------|-------------------------------|
| REDDIT-B (Acc↑) | 91.8 | 93.0 | **94.13** |
| REDDIT-M (Acc↑) | 56.9 | 59.4 | **60.53** |
| DrugOOD-Assay (AUC↑) | 71.68 | 72.24 | **72.63** |
| DrugOOD-Size (AUC↑) | 66.04 | 65.89 | **67.23** |

### Ablation Study

| Configuration | REDDIT-B | REDDIT-M | Note |
|--------------|----------|----------|------|
| No augmentation | 91.8 | 56.9 | Baseline GIN |
| $(\mathcal{E}_\mathcal{L}, \mathcal{S}_{id})$ | 92.56 | 58.32 | Kernel features only, no sketching |
| $(\mathcal{E}_\mathcal{L}, \mathcal{S}_{AG}^{(8)})$ | **94.00** | **60.33** | Full SRF |

The sketching projection accounts for over 60% of the total gain, far exceeding the contribution of kernel features alone.

### Key Findings

1. On the Tree-NeighborsMatch oversquashing benchmark, SRF maintains perfect accuracy within radius 4 and retains >40% accuracy at radius 8.
2. Dirichlet energy experiments show that SRF effectively preserves node distinctiveness in deep layers; increasing the number of projections $k$ further maintains energy.
3. SRF is complementary to PEARL positional encodings; their combination approaches Graph Transformer performance on Peptides-struct (MAE: 0.243 vs. 0.245).
4. SRF runs approximately 3× faster than PEARL and requires roughly two orders of magnitude less memory.

## Highlights & Insights

1. **One Solution for Three Problems**: A single mechanism simultaneously addresses oversquashing, oversmoothing, and limited expressiveness, with both theoretical proofs and experimental validation.
2. **Feature-Centric**: The augmentation signal is built from node features rather than graph topology, making it orthogonal and complementary to topology-based encodings.
3. **Plug-and-Play**: Architecture-agnostic and directly applicable to any MPGNN, including GIN, GINE, and GAT.
4. **Low Overhead**: Single precomputation significantly reduces training time and memory compared to learned encodings such as PEARL.

## Limitations & Future Work

- On graphs without node features or with degenerate features, SRF degenerates to pure random features.
- When $D \ll N$, the kernel approximation introduces bias; the ratio of feature dimensionality to node count affects performance.
- Only node-level sketching is considered; edge features and subgraph-level sketching augmentation remain unexplored.
- Equivariance is achieved in expectation rather than strictly, introducing variance from a single random realization.

## Related Work & Insights

- **Positional Encodings**: SignNet, PEARL, SPE
- **GNN Improvements**: Graph Transformers (GPS, SAN), graph rewiring methods
- **Randomized Methods**: Random Node Features, Johnson-Lindenstrauss Transform
- **Kernel Methods**: Random Fourier Features, kernel approximation

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Creatively repurposes the JL transform for GNN augmentation rather than dimensionality reduction, offering a distinctly original perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of synthetic and real-world tasks, ablations, complexity analysis, and architectural generalization.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations with clear correspondence between theoretical properties and GNN limitations.
- Value: ⭐⭐⭐⭐⭐ — A simple, efficient, plug-and-play solution with both theoretical and practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)
- [\[AAAI 2026\] Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](../../AAAI2026/graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)

</div>

<!-- RELATED:END -->
