---
title: >-
  [Paper Note] Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering
description: >-
  [AAAI 2026][categorical data clustering] This paper proposes DISC, a method that learns cluster-customized category relationships (rather than a globally uniform distance) for each cluster. Through joint optimization of…
tags:
  - "AAAI 2026"
  - "categorical data clustering"
  - "distance metric learning"
  - "subspace clustering"
  - "category relationship"
  - "minimum spanning tree"
date: 2026-05-08
content_hash: 0e5b6ae1cb630f94
---

# Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering

**Conference**: AAAI 2026
**arXiv**: [2511.09049](https://arxiv.org/abs/2511.09049)  
**Code**: [GitHub - ZHAO-Mingjie/SCOF](https://github.com/ZHAO-Mingjie/SCOF)  
**Area**: LLM Evaluation
**Keywords**: categorical data clustering, distance metric learning, subspace clustering, category relationship, minimum spanning tree

## TL;DR

This paper proposes DISC, a method that learns cluster-customized category relationships (rather than a globally uniform distance) for each cluster. Through joint optimization of relationship trees and cluster assignments, DISC achieves an average rank of 1.25 across 12 datasets, substantially outperforming the previous best method (average rank 5.21).

## Background & Motivation

**Categorical data lacks inherent distance**: Numerical data admits natural Euclidean distance, but categorical attributes (e.g., "lawyer" vs. "driver" under "occupation") lack well-defined distance relationships, which constitutes the fundamental challenge of categorical data clustering.

**Existing distance metrics are task-agnostic**: Methods based on Hamming distance, information entropy, and statistical features cannot adaptively adjust to different clustering tasks, limiting their expressiveness.

**Existing learning methods learn only a global uniform relationship**: Distance learning approaches (similarity learning, kernel space mapping, graph learning) jointly optimize distance and clustering, yet apply a unified category relationship across all clusters, ignoring subspace specificity.

**Subspace clustering adjusts weights but not category relationships**: Existing subspace methods learn only attribute importance weights, while the relationships among attribute values remain fixed and cannot capture intra-attribute variation across clusters.

**The same attribute should have different relationships in different clusters**: Using a pneumonia dataset as an example, "programmer" and "construction worker" are close in the COVID-19 cluster (high transmissibility) but far apart in the pneumoconiosis cluster (large difference in dust exposure), demonstrating the necessity of cluster-customized relationships.

**Experimental evidence supports customized relationships**: The authors compare random customization, uniform, and weighted strategies on k-modes, finding that the Customized strategy consistently outperforms both Uniform and Weighted strategies, and that Uniform also outperforms simple attribute weighting alone, confirming that cluster-customized category relationships are key to accurate clustering.

## Method

### Overall Architecture (DISC)

The core idea of DISC (DIStance learning from Cluster) is to learn a customized category relationship distance for each cluster. The overall pipeline is:

1. Initialize cluster partition → 2. Construct a fully connected graph for each attribute within each cluster → 3. Infer a minimum spanning tree as the relationship tree → 4. Define subspace distance based on the relationship tree → 5. Jointly optimize distance and cluster partition → Iterate until convergence

The objective is to minimize intra-cluster dissimilarity:

$$z(\mathbf{H}, M, \mathcal{T}) = \sum_{j=1}^{k} \sum_{i=1}^{n} h_{i,j} \cdot \sum_{r=1}^{l} \mathbf{D}_{j,r}(u,s)$$

### Key Design 1: Fully Connected Graph Modeling via Conditional Probability Distributions

For each attribute $\mathbf{a}_r$ within cluster $C_j$, a fully connected graph $G_{j,r}$ is constructed with nodes corresponding to all possible attribute values. Edge weights are defined as the discrepancy between the conditional probability distributions of two values within the cluster:

$$\mathbf{W}_{j,r}(u,s) = |p(v_r^u | C_j) - p(v_r^s | C_j)|$$

The conditional probability distribution reflects the distributional pattern of values within a cluster; values with similar distributions have smaller edge weights, indicating greater similarity.

### Key Design 2: Minimum Spanning Tree for Relationship Tree Inference

A fully connected graph introduces ambiguity due to multiple paths between nodes. To address this, a minimum spanning tree (MST) $\mathcal{T}_{j,r}$ is extracted from the fully connected graph as the relationship tree, ensuring a unique path between any two values and thus a determinate, unique distance definition.

A key theorem establishes that the inferred relationship tree defines a deterministic, Euclidean-compatible distance metric. This means the learned categorical distances can be naturally combined with Euclidean distances for numerical attributes, supporting mixed-type dataset clustering.

### Key Design 3: Generalized Attribute Weighting Mechanism

The relationship tree implicitly captures attribute importance: if an attribute exhibits large distributional discrepancies across values within a cluster (large conditional probability differences), its distance values are correspondingly large, contributing more to the overall distance. This is equivalent to decomposing traditional attribute weighting into finer-grained value-level distance learning, yielding greater expressiveness.

### Loss & Training

A three-step alternating optimization scheme is employed:

1. **Fix $M$ and $\mathcal{T}$, update $\mathbf{H}$**: Assign each sample to the nearest cluster center.
2. **Fix $\mathcal{T}$ and $\mathbf{H}$, update $M$**: Compute the mode of each attribute within each cluster as the cluster center, following k-modes.
3. **After Steps 1–2 converge, fix $\mathbf{H}$ and $M$, re-infer $\mathcal{T}$**: Recompute conditional probability distributions based on the current partition and reconstruct the relationship trees.

The three steps iterate cyclically until the partition matrix $\mathbf{H}$ no longer changes. Convergence is theoretically guaranteed (Theorem 3), and the time complexity is $O(nlk\mathcal{I}E)$, linear in the number of samples, attributes, and clusters.

## Key Experimental Results

### Table 1: Clustering Performance Comparison (ACC, 12 UCI Datasets, 11 Methods)

| Dataset | KMD | CoForest | HARR | SigDT | **DISC** |
|---------|------|----------|------|-------|----------|
| CA | 0.380 | 0.402 | 0.368 | 0.534 | **0.583** |
| DT | 0.597 | 0.676 | 0.711 | 0.844 | **0.859** |
| AV | 0.625 | 0.667 | 0.644 | 0.638 | **0.726** |
| OB | 0.340 | 0.383 | 0.373 | 0.381 | **0.425** |
| BM | 0.640 | 0.609 | 0.617 | 0.249 | **0.729** |
| ZO | 0.697 | 0.724 | 0.726 | 0.723 | **0.805** |
| **Avg. Rank** | 8.38 | 5.42 | 5.21 | 6.83 | **1.25** |

DISC achieves an average ACC rank of 1.25, far surpassing the runner-up HARR at 5.21; average ARI rank is 1.42 and CMP rank is 1.58.

### Table 2: Extension to Mixed-Type Datasets (ACC)

| Dataset | KPT | WOCIL | HARR | DISC (cat. only) | **DISC (mixed)** |
|---------|------|-------|------|------------------|------------------|
| CC | 0.398 | 0.386 | 0.430 | 0.443 | **0.449** |
| AP | 0.516 | 0.572 | 0.576 | 0.641 | **0.656** |
| DT | 0.531 | 0.690 | 0.615 | 0.859 | **0.862** |
| AV | 0.539 | 0.673 | 0.681 | 0.726 | **0.776** |
| BM | 0.587 | 0.572 | 0.522 | 0.729 | **0.742** |

Results validate the compatibility of the learned categorical distances with Euclidean distances, with further improvements on mixed-type datasets.

## Highlights & Insights

- **Novel problem perspective**: This is the first work to explicitly propose learning cluster-customized category relationships rather than a uniform distance, substantially enhancing the fitting capacity of clustering algorithms.
- **Elegant mathematical modeling**: The hierarchical pipeline of conditional probability distribution → fully connected graph → minimum spanning tree transforms ambiguous category relationships into deterministic, measurable distances.
- **Rigorous theoretical guarantees**: The determinism, Euclidean compatibility, and convergence of the relationship tree distance are formally proven; the method is not heuristic.
- **Strong empirical performance**: Average ACC rank of 1.25 vs. 5.21 for the runner-up, with significance tests passed and consistent superiority across all 12 datasets.
- **Natural support for mixed-type data**: Euclidean compatibility enables seamless extension to datasets with both numerical and categorical attributes.

## Limitations & Future Work

- **Experiments limited to small-scale UCI datasets**: The largest dataset, BM, contains only 45K samples; validation on large-scale (millions of samples) datasets is absent.
- **Relies on the true number of clusters $k^*$**: All experiments use the ground-truth number of clusters; strategies for automatically determining $k$ when it is unknown are not discussed.
- **Tied to the k-modes framework**: The method is built on the hard-partition framework of k-modes; soft clustering or more flexible clustering paradigms are not explored.
- **Increased MST computation cost for large attribute domains**: Although overall complexity is linear, constructing fully connected graphs and MSTs for attributes with many distinct values $o_r$ may become a bottleneck.
- **No comparison with deep clustering methods**: The paper does not compare against deep learning-based categorical data clustering methods.

## Related Work & Insights

- **Categorical distance metrics**: Hamming distance (Hamming 1950), information entropy methods (Lin 1998; Zhang & Cheung 2022b), statistical methods (Ahmad & Dey 2007) — all task-agnostic and non-adaptive to clustering.
- **Distance learning**: Probability distribution similarity (Cheung & Jia 2013), kernel space mapping (Zhu et al. 2022), graph learning (Zhang & Cheung 2022a, 2023; Zhao et al. 2024/CoForest) — learn a globally uniform relationship.
- **Subspace clustering**: Attribute weighting methods (Bai et al. 2011; Cao et al. 2013; Jia & Cheung 2017/WOCIL) — adjust weights only, not category relationships.
- **SOTA baselines**: HARR (Zhang et al. 2025b), SigDT (Hu et al. 2025c), MCDC (Cai et al. 2024).

## Rating

- Novelty: ⭐⭐⭐⭐ — Cluster-customized category relationships represent a clear and valuable new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 12 datasets, 11 baselines, ablation studies, mixed-type data extension, and convergence/efficiency analysis are all provided; however, large-scale and deep learning comparisons are missing.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, theoretical derivations are complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Represents a clear advancement in categorical data clustering, combining theoretical rigor with practical effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)

</div>

<!-- RELATED:END -->
