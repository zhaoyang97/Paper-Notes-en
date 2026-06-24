---
title: >-
  [Paper Note] Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification
description: >-
  [ICML2025][Graph Learning][Over-squashing] Proposes GOKU (Densification-Sparsification Rewiring paradigm), which treats the input graph as a spectral sparsifier of an unknown dense latent graph and solves the inverse sparsification problem. It effectively mitigates the over-squashing problem in GNNs by enhancing graph connectivity while explicitly preserving the Laplacian spectrum.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Over-squashing"
  - "Graph Rewiring"
  - "Spectral Sparsification"
  - "Effective Resistance"
  - "Graph Classification"
  - "Node Classification"
date: 2026-05-08
content_hash: 404356b7b9013485
---

# Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification

**Conference**: ICML2025  
**arXiv**: [2506.16110](https://arxiv.org/abs/2506.16110)  
**Code**: [GitHub](https://github.com/Jinx-byebye/GOKU)  
**Area**: Graph Learning / Graph Neural Networks  
**Keywords**: Over-squashing, Graph Rewiring, Spectral Sparsification, Effective Resistance, Graph Classification, Node Classification

## TL;DR
Proposes GOKU (Densification-Sparsification Rewiring paradigm), which treats the input graph as a spectral sparsifier of an unknown dense latent graph and solves the inverse sparsification problem. It effectively mitigates the over-squashing problem in GNNs by enhancing graph connectivity while explicitly preserving the Laplacian spectrum.

## Background & Motivation

**Over-squashing Problem**: GNNs adopt communication via message passing. When information between distant nodes needs to pass through structural bottlenecks of the graph, the messages are compressed and distorted during aggregation, preventing GNNs from capturing long-range dependencies. This problem becomes more severe in deeper networks as the receptive field grows while the node representation vectors remain fixed in size.

**Two Major Limitations of Existing Graph Rewiring Methods**:

**Non-preservation of spectral properties**: Delaunay rewiring constructs graphs solely based on node features, completely ignoring the original topology; Graph Transformers rely on fully connected graphs; Expander rewiring even introduces completely new node sets. The graph spectrum (the set of Laplacian eigenvalues) encodes crucial topological information such as community structure and node centrality, which are essential for clustering and semi-supervised learning.

**Increased danger of over-smoothing and computational overhead due to edge accumulation**: Denser output graphs cause excessive mixing of neighborhood information, leading node features to become indistinguishable.

**Core Motivation**: Can connectivity be enhanced while maintaining sparsity and explicitly preserving the original graph spectrum? The authors experimentally verify the correlation between spectral properties and classification tasks—the spectral similarity of graphs/nodes within the same class is significantly higher than those across different classes (Table 1), proving the necessity of preserving the graph spectrum.

## Method

### DSR Paradigm: Densification-Sparsification Rewiring

The overall workflow is $G \xrightarrow{\text{稠密化}} G_l \xrightarrow{\text{稀疏化}} G_o$, where $G$ is the input graph, $G_l$ is the latent dense graph, and $G_o$ is the final output graph.

**Key Idea**: Treat the input graph $G$ as the result of spectral sparsification on an unknown dense graph $G_l$. The structural bottlenecks in $G$ stem from the removal of edges critical for connectivity during the sparsification process. By recovering these "missing edges" via inverse sparsification, and subsequently performing forward sparsification to prune unimportant edges, the final output $G_o$ simultaneously satisfies:

- Enhanced connectivity (reduced effective resistance)
- Maintained sparsity (number of edges does not exceed the original graph)
- Explicit preservation of graph spectrum: $G_o \overset{(1\pm\epsilon)^2}{\approx} G$

### Densification Stage: Inverse Process of Unimportant Spectral Sparsification (USS)

**USS Sampling Probability**: For an edge $e=(u,v)$, the sampling probability is:

$$p_e \propto \frac{\deg(u) + \deg(v) + 1}{|f_u - f_v|}$$

where $f_u, f_v$ are the components of the Fiedler vector (the eigenvector corresponding to the second smallest eigenvalue of the Laplacian). Intuition: a large $|f_u - f_v|$ indicates weak connectivity between $u$ and $v$, and small degrees represent weak connections to the rest of the graph—such edges are critical to connectivity and are thus prioritized for removal in USS.

**Inverse Sparsification MLE**: Model the problem as maximum likelihood estimation to find the latent graph $G_l$ most likely to generate $G$:

$$G_l = \underset{G_d}{\arg\max}\; \mathbb{P}(\phi(G_d) = G)$$

Through independence assumptions and simplifications, the likelihood is factorized into a product of edge-level probabilities. To restrict the search space: assume uniform edge weights in $G_l$; construct a candidate edge set (around $16|E'|$ candidate edges) from nodes with the largest Fiedler values and smallest degrees, then sample $|E'|$ missing edges based on the objective function values.

**Theoretical Guarantee** (Theorem 4.1): When the number of samples $q \geq \frac{\kappa^2}{2\epsilon^2}\log 8$, USS produces a $(1\pm\epsilon)$-spectral sparsifier with a probability of at least 3/4.

### Sparsification Stage: Importance Spectral Sparsification (ISS)

Apply Effective Resistance (ER)-based ISS to the latent graph $G_l$. For each edge $(u,v)$, the sampling probability is:

$$p_e \propto (1 + S_e) \cdot R_e$$

where $S_e = (1+\cos(x_u, x_v))/2$ is the normalized cosine similarity, and $R_e$ is the effective resistance. This design prioritizes keeping edges with high ER (critical for connectivity) and similar node features (homophily enhancement), while filtering out low-similarity noise edges.

Sampling continues until $\beta|E|$ distinct edges are obtained ($0.5 < \beta \leq 1$), ensuring that the output graph is not denser than the original graph.

**Theoretical Guarantee** (Theorem 4.2): ISS produces a $(1\pm\epsilon)$-spectral sparsifier with sampling complexity $O(n\log n / \epsilon^2)$.

### Complexity

The overall time complexity is $O(|E'|m) + \tilde{O}(m/\delta^2)$, where the ER approximation is completed in sub-linear time utilizing the method of Koutis et al. (2014). Experiments verify that the running time scales nearly linearly with the graph size.

## Key Experimental Results

Evaluated on 10 datasets (6 node-classification + 4 graph-classification), comparing against 6 SOTA rewiring methods including SDRF, FoSR, BORF, GTR, DR, and LASER.

### Classification Accuracy under the GCN Backbone (Table 2, Partial)

| Method | Texas | Cornell | Wisconsin | Mutag | Average Rank |
|------|-------|---------|-----------|-------|----------|
| NONE | 44.2 | 41.5 | 44.6 | 68.8 | 6.60 |
| BORF | 49.4 | 50.8 | 50.3 | 75.8 | 3.40 |
| DR | 67.8 | 57.8 | 62.8 | 80.1 | 4.55 |
| **GOKU** | **72.4** | **69.4** | **68.8** | **81.0** | **1.90** |

### Classification Accuracy under the GIN Backbone (Table 3, Partial)

| Method | Texas | Cornell | Wisconsin | IMDB | Average Rank |
|------|-------|---------|-----------|------|----------|
| NONE | 46.4 | 40.2 | 42.1 | 67.7 | 6.20 |
| LASER | 46.5 | 44.5 | 46.1 | 68.6 | 3.40 |
| **GOKU** | **60.2** | **49.5** | **57.6** | **71.3** | **1.80** |

GOKU exhibits particularly prominent advantages on heterophilous graphs (Texas/Cornell/Wisconsin), validating the effectiveness of spectrum-preserving rewiring in heterophily scenarios.

### Ablation Study (Table 4)

| Variant | Mutag | Chameleon | Cora |
|------|-------|-----------|------|
| GOKU-D (Densification only) | 78.0 | 62.7 | 86.4 |
| GOKU-S (Sparsification only) | 76.8 | 62.9 | 86.4 |
| GOKU (Full) | **81.0** | **63.2** | **86.8** |

Densification and sparsification are both indispensable: densification alone may lead to over-smoothing due to over-connectivity; sparsification alone cannot significantly improve connectivity.

### Connectivity Improvement

On SBM synthetic graphs, the effective resistance distribution after GOKU rewiring shifts significantly to the left (lower ER = better connectivity), while homophily is consistently improved (e.g., L-Low: 0.385→0.433).

## Highlights & Insights

1. **Novel Perspective of Inverse Sparsification**: Treats the input graph as a spectral sparsifier of an unknown dense graph, enhancing connectivity by "recovering missing edges"—a conceptually elegant approach supported by solid theory.
2. **Simultaneous Realization of Three Operations**: Enhancing connectivity + maintaining sparsity + explicitly preserving the graph spectrum. It is the first rewiring method to achieve all three conditions simultaneously.
3. **Complete Theoretical Foundation**: Both USS and ISS possess probabilistic guarantees of spectral sparsification (Theorem 4.1, 4.2), avoiding purely heuristic formulations.
4. **Substantial Lead on Heterophilous Graphs**: Texas increases from 44.2 to 72.4, Cornell from 41.5 to 69.4, with performance gains significantly exceeding existing methods.
5. **Node-Feature-Aware Sparsification**: Incorporates cosine similarity in ISS to reconcile both topology and feature information.

## Limitations & Future Work

1. **Weighted Graph Assumption**: Spectral sparsification relies on weighted graphs to guarantee spectrum preservation. Preserving the spectrum of unweighted graphs across different edge counts is inherently difficult, and the introduced edge weights may bring additional complexity to GNN training.
2. **Hyperparameters $\alpha, \beta$ Require Tuning**: Although the search space is limited ($\alpha \in \{5..30\}, \beta \in [0.5,1]$), the optimal values may vary across different datasets.
3. **Heuristic Candidate Edge Construction in the Densification Stage**: Selecting candidate nodes based on Fiedler values and degree rankings is intuitively motivated but lacks theoretical guarantees of optimality.
4. **Scalability on Large-Scale Graphs**: A graph with 2000 nodes takes approximately 65-72 seconds, which may still pose a bottleneck when dealing with larger-scale graphs (e.g., millions of nodes).

## Related Work & Insights

- **FoSR** (Karhadkar et al., 2023): Improves the first-order approximation of the spectral gap by adding edges, but does not preserve the spectrum.
- **BORF** (Nguyen et al., 2023): Curvature-based rewiring that simultaneously adds and deletes edges.
- **DR** (Attali et al., 2024): Delaunay rewiring based entirely on node features, discarding the original topology.
- **LASER** (Barbero et al., 2024): Local-aware rewiring restricted to the $k$-hop neighborhood, without explicitly preserving the spectrum.
- **Spielman & Srivastava (2008)**: A seminal work on ER-based spectral sparsification, upon which the ISS stage of GOKU is built.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The perspective of inverse sparsification is a pioneering attempt in the field of graph rewiring, with an elegantly designed DSR paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 10 datasets × 3 backbone models × ablations × spectral visualizations, but lacks experiments on massive graphs.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and rigorous theory, though requires patience due to dense notation.
- Value: ⭐⭐⭐⭐⭐ — Provides significant improvements on heterophilous graphs, delivering a theoretically sound and practically effective solution to the GNN over-squashing problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[ICLR 2026\] gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity](../../ICLR2026/graph_learning/glstm_mitigating_over-squashing_by_increasing_storage_capacity.md)
- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[ICML 2025\] GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers](grokformer_graph_fourier_kolmogorov-arnold_transformers.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

</div>

<!-- RELATED:END -->
