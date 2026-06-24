---
title: >-
  [Paper Note] SpEx: A Spectral Approach to Explainable Clustering
description: >-
  [NeurIPS 2025][Interpretability][Explainable clustering] This paper proposes SpEx, a general spectral graph partitioning-based framework for explainable clustering that can "round" any reference clustering (without requiring centroids) into an explainable clustering via coordinate-cut decision trees, or perform reference-free clustering directly on a kNN graph.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Explainable clustering"
  - "spectral graph partitioning"
  - "coordinate cuts"
  - "decision trees"
  - "Cheeger inequality"
date: 2026-05-08
content_hash: 8a3e1e7dd1ee008c
---

# SpEx: A Spectral Approach to Explainable Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2511.00885](https://arxiv.org/abs/2511.00885)  
**Code**: [GitHub](https://github.com/talargv/SpEx)  
**Area**: Explainable Clustering, Spectral Methods, Graph Partitioning
**Keywords**: Explainable clustering, spectral graph partitioning, coordinate cuts, decision trees, Cheeger inequality

## TL;DR
This paper proposes SpEx, a general spectral graph partitioning-based framework for explainable clustering that can "round" any reference clustering (without requiring centroids) into an explainable clustering via coordinate-cut decision trees, or perform reference-free clustering directly on a kNN graph.

## Background & Motivation
- Explainable clustering (Moshkovitz 2020): clustering described by binary decision trees where each internal node corresponds to a single-coordinate threshold cut.
- Limitations of prior work:
    - **IMM/EMN**: require the reference clustering to have centroids (k-means); cannot handle kernel k-means or spectral clustering.
    - **Kernel IMM**: supports only specific kernel functions, with high complexity and poor scalability.
    - **CART**: not designed for clustering; maximizes Gini impurity reduction, potentially selecting cuts incompatible with the clustering structure.
- No truly general explainable clustering method exists to date — one that is unconstrained by reference clustering type and robust on real-world data.

## Method

### Overall Architecture
Leverages a generalization of the Cheeger inequality to relate coordinate cuts to the conductance of graph cuts.

### Key Designs

#### Core Theorem (Theorem 2.2)
For a point set $X \subset \mathbb{R}^d$ and graph $G(X, E, w)$, there exists a coordinate cut $S_{j,\tau}$ such that:
$$\Psi_G(S_{j,\tau}(X)) \leq \sqrt{\frac{\mathbb{E}_{adj}\|x-y\|_2^2}{\mathbb{E}_{all}\|x-y\|_2^2}}$$
- Numerator: expected squared distance between adjacent pairs in the graph.
- Denominator: expected squared distance between all pairs.
- When the graph captures meaningful geometric structure (neighbors are closer than random pairs), the conductance upper bound is tighter.

#### SpEx-Clique (Reference Clustering Mode)
- Given **any** reference clustering, constructs a **clique graph**: edges connect points within the same cluster.
- Iteratively selects coordinate cuts with minimum conductance to build a decision tree.
- Requires neither centroids nor specific kernel functions.

#### SpEx-kNN (Reference-Free Mode)
- Constructs a kNN graph ($k=20$) directly on the dataset.
- Produces explainable clustering without any reference clustering.

#### Iterative Tree Construction (Algorithm 1)
- A priority queue maintains all leaf nodes.
- At each step, the leaf node with the greatest conductance reduction after cutting is selected for splitting.
- Objective: minimize the multi-way normalized cut objective $\bar{\Psi}_G(T) = \sum_{v \in \mathcal{L}(T)}\psi_G(X_v)$.
- The number of leaf nodes is freely chosen and need not match the number of reference clusters.

### Unified Theoretical Framework: Non-Uniform Sparse Cut
Via Trevisan's non-uniform sparse cut framework, prior methods are unified as follows:
- **IMM** → $G$ = star graph, $H$ = diameter centroid-pair single-edge graph.
- **EMN** → $G$ = star graph, $H$ = complete graph over centroids.
- **CART** → $H$ = independent set graph, $G$ = $H$-degree-weighted complete graph.

**Graph-theoretic explanation of CART's failure**: The independent set graph incentivizes separating points from different clusters but **does not penalize separating points within the same cluster** — causing greedy selection of suboptimal cuts.

### Computational Efficiency
- Per-level time: $O(dn(\log n + S))$
- SpEx-Clique: $S = O(1)$ (cut scores updated by maintaining two cluster histograms)
- SpEx-kNN: $S = O(\kappa)$, where $\kappa$ is the number of neighbors
- Total time: $O(kdn\log n)$ (SpEx-Clique)

## Key Experimental Results

### Main Results (8 Real-World Datasets, ARI/AMI Evaluation)

| Dataset | n | d | k | SpEx-Clique | SpEx-kNN | EMN | CART (k-means) | CART (spectral) |
|--------|---|---|---|-------------|----------|-----|----------------|-----------------|
| CIFAR-10 | 50K | 512 | 10 | **Best/Best** | Mid | Mid | Poor | Poor |
| MNIST | 60K | 512 | 10 | **Best** | Mid | Mid | Poor | Mid |
| 20News | 11K | 768 | 20 | **Best** | Mid | Mid | Poor | Poor |
| Beans | 13K | 16 | 7 | Mid | **Best** | Mid | Mid | Mid |
| Ecoli | 336 | 7 | 8 | Mid | **Best** | Mid | Mid | Mid |
| Iris | 150 | 4 | 3 | Mid | **Best** | Mid | Mid | Good |

### Method Stability Analysis

| Property | SpEx-Clique | SpEx-kNN | EMN | CART | Kernel IMM |
|----------|------------|---------|-----|------|-----------|
| Requires centroids | ✗ | ✗ | ✓ | ✗ | ✗ |
| Requires reference clustering | ✓ | ✗ | ✓ | ✓ | ✓ |
| Scalability to large data | ✓ | ✓ | ✓ | ✓ | ✗ |
| Consistently strong performance | ✓ | Good on low-dim | Mid | Good on small data | Uncertain |

### Key Findings
- SpEx-Clique is the most consistently high-performing algorithm, outperforming every baseline in most settings.
- SpEx-kNN performs best on low-dimensional datasets (≤16 dimensions).
- CART performs adequately on small datasets but degrades on large ones — confirming that toy-example failure modes manifest in real data.
- Kernel IMM cannot run on large datasets due to poor scalability.
- The number of leaf nodes in SpEx can be set flexibly — increasing leaves reduces clustering quality at the cost of interpretability.

## Highlights & Insights
1. **Truly general**: the first explainable clustering method compatible with arbitrary reference clustering types.
2. **Unified theoretical perspective**: the non-uniform sparse cut framework provides a unified understanding of IMM, EMN, and CART.
3. **Root cause of CART's failure**: the independent set graph does not penalize intra-cluster separation, leading to greedy selection of poor cuts.
4. **Reference-free mode**: SpEx-kNN bypasses reference clustering entirely to produce explainable clusters directly from data.
5. **Efficient implementation**: SpEx-Clique requires only $O(1)$ time to update cut scores.

## Limitations & Future Work
- Only axis-aligned (single-coordinate) cuts are considered — oblique cuts, though potentially more powerful, are unexplored.
- SpEx-kNN underperforms SpEx-Clique on high-dimensional data.
- Theoretical bounds on the explainability cost have not yet been analyzed.
- The value of $k$ in the kNN graph must be specified by the user.
- Extension to online or streaming settings remains an open direction.

## Related Work & Insights
- The explainable clustering formulation and IMM algorithm of Moshkovitz et al. (2020) have inspired a substantial body of theoretical work.
- EMN (2022) improves upon IMM via ratio-based optimization.
- Kernel IMM (2024) extends the framework to kernel k-means but with limited scalability.
- This paper is the first to introduce spectral graph partitioning theory into explainable clustering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (elegant combination of spectral methods and a unified framework)
- Technical Depth: ⭐⭐⭐⭐⭐ (theoretical contributions, algorithm design, and unified analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 datasets, multiple baselines)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear logic, well-balanced between intuition and formal proofs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)
- [\[ACL 2026\] A Structured Clustering Approach for Inducing Media Narratives](../../ACL2026/interpretability/a_structured_clustering_approach_for_inducing_media_narratives.md)
- [\[ICLR 2026\] Explainable K-means Neural Networks for Multi-view Clustering](../../ICLR2026/interpretability/explainable_k_-means_neural_networks_for_multi-view_clustering.md)
- [\[NeurIPS 2025\] Additive Models Explained: A Computational Complexity Approach](additive_models_explained_a_computational_complexity_approach.md)
- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)

</div>

<!-- RELATED:END -->
