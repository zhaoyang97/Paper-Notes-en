---
title: >-
  [Paper Note] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering
description: >-
  [AAAI 2026][Graph Learning][Hierarchical Clustering] This paper proposes HypCSE, which relaxes discrete Structural Entropy (SE) into a Continuous Structural Entropy (CSE) defined in hyperbolic space. Combined with graph structure learning and contrastive learning, HypCSE enables end-to-end differentiable hierarchical clustering and consistently outperforms both discrete and continuous hierarchical clustering methods across 7 datasets.
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Hierarchical Clustering"
  - "Structural Entropy"
  - "Hyperbolic Space"
  - "Graph Structure Learning"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 6401329f074f0611
---

# Hyperbolic Continuous Structural Entropy for Hierarchical Clustering

**Conference**: AAAI 2026
**arXiv**: [2512.00524](https://arxiv.org/abs/2512.00524)  
**Code**: [GitHub](https://github.com/SELGroup/HypCSE)  
**Area**: Graph Learning / Hierarchical Clustering
**Keywords**: Hierarchical Clustering, Structural Entropy, Hyperbolic Space, Graph Structure Learning, Contrastive Learning

## TL;DR

This paper proposes HypCSE, which relaxes discrete Structural Entropy (SE) into a Continuous Structural Entropy (CSE) defined in hyperbolic space. Combined with graph structure learning and contrastive learning, HypCSE enables end-to-end differentiable hierarchical clustering and consistently outperforms both discrete and continuous hierarchical clustering methods across 7 datasets.

## Background & Motivation

Hierarchical clustering organizes data points into a dendrogram and is a classical unsupervised learning task. Existing methods face two major challenges:

1. **Most methods lack a global objective function**: Agglomerative methods iteratively merge the closest cluster pairs, while divisive methods iteratively split clusters — both are greedy strategies prone to yielding suboptimal trees.
2. **Graph-based continuous methods define objectives on complete or statically predefined graphs**: Complete graphs are contaminated by noisy edges, while predefined graphs cannot be updated during training — neither fully exploits the underlying data features.

Structural Entropy (SE), rooted in information theory, measures tree quality by quantifying the minimum encoding length of a random walk on the graph, making it an excellent global objective. However, SE is defined over discrete trees and is thus non-differentiable.

## Core Problem

How can discrete structural entropy be relaxed into a differentiable objective and optimized end-to-end over an adaptively learned graph structure to achieve high-quality hierarchical clustering?

## Method

### Overall Architecture

HypCSE consists of two modules: (1) a hyperbolic hierarchical clustering module that constructs a graph, encodes it in hyperbolic space, minimizes CSE, and decodes the result into a partition tree; and (2) a Graph Structure Learning (GSL) module that dynamically updates the graph structure via a contrastive learning-guided graph learner.

### Key Designs

1. **CSE Objective**: SE is reformulated in terms of the Lowest Common Ancestor (LCA):
$$\mathcal{H}^\mathcal{T}(G) = \frac{2}{\mathcal{V}(G)} \sum_{(v_i,v_j)\in E} \mathbf{W}_{ij} \log_2 \mathcal{V}_{\mathcal{T}_i \vee \mathcal{T}_j} - \frac{1}{\mathcal{V}(G)} \sum_{v_i \in V} \mathcal{V}_{\mathcal{T}_i} \log_2 \mathcal{V}_{\mathcal{T}_i}$$
By exploiting the correspondence between geodesics in hyperbolic space and shortest paths in trees, the discrete LCA is approximated by the hyperbolic LCA in the Poincaré model, and the non-differentiable indicator function is relaxed via scaled softmax.

2. **Hyperbolic Graph Encoding**: A 3-layer Lorentz convolution (LConv) encodes the graph into the Lorentz model $\mathbb{L}^{\kappa,d}$, which is then projected onto the Poincaré model for CSE computation.

3. **Graph Structure Learning**: A dual-view design is adopted with an anchor graph $G_a$ and a learner graph $G_l$. The anchor graph provides stable supervision, while the learner graph is updated from data via a GSL encoder. The anchor graph is updated via bootstrapping: $E_a \leftarrow S_\tau(E_a) + S_{1-\tau}(E_l)$.

4. **Hyperbolic Contrastive Learning**: Distances between two views of the same node are minimized in hyperbolic space, while distances between different nodes are maximized, guiding the graph learner to produce more discriminative features.

5. **Overall Objective**: $\mathcal{L}_{HypCSE} = \mathcal{L}_{cse} + \eta_1 \mathcal{L}_{con} + \eta_2 \mathcal{L}_{cen}$, where $\mathcal{L}_{cen}$ ensures that leaf embeddings are distributed around the origin.

## Key Experimental Results

### Hierarchical Clustering Quality (Dendrogram Purity %)

| Method | Zoo | Iris | Wine | Br.Cancer | OptDigits | Spambase | PenDigits |
|--------|-----|------|------|-----------|-----------|----------|-----------|
| SingleLinkage | 97.6 | 81.2 | 67.9 | 85.1 | 73.3 | 58.9 | 70.0 |
| HCSE | 97.3 | 89.7 | 71.1 | 94.2 | 81.5 | 55.2 | 76.9 |
| HypHC | 96.8 | 76.0 | 88.7 | 96.5 | 33.5 | 75.4 | 11.7 |
| FPH | 89.9 | 85.3 | 92.8 | 92.6 | 81.0 | 54.8 | 69.6 |
| **HypCSE** | **97.9** | **95.1** | **93.9** | **96.8** | **86.4** | **75.5** | **81.4** |

HypCSE ranks first on the DP metric across all 7 datasets and achieves the best SE metric on 4 datasets.

### Flexibility Evaluation (Similarity Classification, ACC %)

| Method | Zoo | Iris | Glass | Seg. |
|--------|-----|------|-------|------|
| HypHC-ETE | 87.9 | 85.6 | 54.4 | 67.7 |
| **HypCSE** | **88.3** | **91.8** | **54.7** | **78.8** |

## Highlights & Insights

- **Elegant fusion of information-theoretic objectives and hyperbolic geometry**: CSE bridges discrete trees and continuous embeddings via hyperbolic LCA.
- **GSL addresses the static graph bottleneck**: Contrastive learning-guided graph structure updates effectively remove noisy edges.
- **End-to-end differentiable and scalable**: Tree decoding complexity can be reduced to $O(n \log n)$.
- **Solid theoretical foundation**: The lower-bound relationship between SE and graph conductance is proven (Lemma 3), ensuring the optimization direction is well-grounded.

## Limitations & Future Work

- HypCSE exhibits sensitivity to hyperparameters ($\tau$, $\eta_1$) on the Spambase dataset, suggesting GSL instability when hierarchical structure is ambiguous.
- The softmax relaxation in the CSE objective introduces approximation error, and the optimal choice of temperature parameter $t_1$ is dataset-dependent.
- Validation is limited to small-scale UCI datasets; experiments on large-scale graph data are absent.

## Related Work & Insights

HypHC also performs hierarchical clustering in hyperbolic space but optimizes the Dasgupta cost on a complete graph, making it sensitive to noise and causing severe degradation on large datasets (only 11.7% on PenDigits). UFit/FPH relax the ultrametric or Dasgupta cost but rely on static graphs. HCSE optimizes discrete SE but cannot learn end-to-end. HypCSE unifies the global SE objective, the advantages of hyperbolic geometry, and adaptive graph structure learning.

The relaxation approach underlying CSE can be generalized to the continuous relaxation of other combinatorial optimization objectives (e.g., normalized cut). The combination of hyperbolic embeddings and structural entropy is applicable to community detection in social networks and hierarchical structure discovery in knowledge graphs. The GSL module is general-purpose and can be integrated into any downstream task requiring graph-structured input.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to relax structural entropy into a hyperbolic continuous space; highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablations and detailed sensitivity analysis, but large-scale experiments are missing.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and figures are clear.
- Value: ⭐⭐⭐⭐ Introduces a new information-theoretic + geometric paradigm for hierarchical clustering.
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](../../ICLR2026/graph_learning/cords_-_continuous_representations_of_discrete_structures.md)
- [\[ICLR 2026\] Bridging ML and Algorithms: Comparison of Hyperbolic Embeddings](../../ICLR2026/graph_learning/bridging_ml_and_algorithms_comparison_of_hyperbolic_embeddings.md)
- [\[ICLR 2026\] $\ell_1$ Latent Distance Based Continuous-Time Graph Representation](../../ICLR2026/graph_learning/ell_1_latent_distance_based_continuous-time_graph_representation.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](../../ICML2025/graph_learning/hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

</div>

<!-- RELATED:END -->
