---
title: >-
  [Paper Note] Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time
description: >-
  [ICML 2025][bi-chromatic matching] This work presents the first fully dynamic sublinear update algorithm for the Euclidean bi-chromatic matching problem. For any fixed $\varepsilon > 0$, it achieves an $O(1/\varepsilon)$ approximation ratio and $O(n^{\varepsilon})$ update time, which can be utilized to efficiently monitor distribution shifts (Wasserstein distance).
tags:
  - "ICML 2025"
  - "bi-chromatic matching"
  - "dynamic algorithms"
  - "Wasserstein distance"
  - "sublinear update"
  - "geometric optimization"
date: 2026-05-08
content_hash: 8081abd9787d819a
---

# Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time

**Conference**: ICML 2025  
**arXiv**: [2505.09010](https://arxiv.org/abs/2505.09010)  
**Code**: None  
**Area**: Algorithms / Geometric Optimization  
**Keywords**: bi-chromatic matching, dynamic algorithms, Wasserstein distance, sublinear update, geometric optimization

## TL;DR
This work presents the first fully dynamic sublinear update algorithm for the Euclidean bi-chromatic matching problem. For any fixed $\varepsilon > 0$, it achieves an $O(1/\varepsilon)$ approximation ratio and $O(n^{\varepsilon})$ update time, which can be utilized to efficiently monitor distribution shifts (Wasserstein distance).

## Background & Motivation
**Background**: Bi-chromatic matching is a core problem in geometric optimization. Given two sets of red and blue points in Euclidean space, the goal is to find a perfect matching with minimum cost. This is directly related to the estimation of the Wasserstein distance (Earth Mover's Distance).

**Limitations of Prior Work**: In dynamic scenarios (point insertions and deletions), efficiently maintaining matching quality is required. Existing static algorithms have high complexity (exact algorithms require $O(n^3)$) and cannot handle frequent updates. No fully dynamic algorithm with sublinear update time existed prior to this work.

**Key Challenge**: Practical applications (e.g., monitoring data distribution shifts, real-time anomaly detection) require rapid responses to updates, but calculating exact matchings is computationally prohibitive.

**Goal**: To design the first fully dynamic Euclidean bi-chromatic matching algorithm with sublinear update time.

**Key Insight**: Leveraging hierarchical decomposition and a lazy update strategy.

**Core Idea**: Through spatial hierarchy and localized updates, each point operation only affects the local structure, thereby achieving sublinear updates.

## Method

### Overall Architecture
Input: Dynamically maintained sets of red and blue points $R, B \subset \mathbb{R}^d$  
Output: Approximate minimum-cost bi-chromatic matching  
Supported operations: Point insertion (insert) and deletion (delete)

### Key Designs

1. **Hierarchical Spatial Decomposition**:

    - Function: Recursively partitions the Euclidean space into multi-level grids, where each level corresponds to a different distance scale.
    - Mechanism: Uses quadtree/octree-like decomposition, where the grid side length at level $i$ is $2^i$. Matchings are processed separately at different levels: short-distance matchings are resolved at fine-grained levels, while long-distance matchings are resolved at coarser levels.
    - Design Motivation: Hierarchical processing ensures that each update only requires modifying the local level structures rather than recomputing the entire global matching.

2. **Lazy Update Strategy**:

    - Function: Instead of reconstructing the complete matching immediately at every point operation, updates are accumulated and processed in batches.
    - Mechanism: Maintains a "dirty" set of points, triggering local rematching when the number of dirty points exceeds a threshold. For an update time of $O(n^\varepsilon)$, each operation affects reconstruction across at most $O(n^\varepsilon)$ levels.
    - Design Motivation: Avoiding frequent global reconstruction is key to achieving sublinear complexity.

3. **Approximation-Speed Tradeoff**:

    - Function: Controls the balance between approximation quality and update speed using the parameter $\varepsilon$.
    - Mechanism: A smaller $\varepsilon$ leads to finer hierarchical partitioning and a better approximation ratio ($O(1/\varepsilon)$), but longer update times ($O(n^\varepsilon)$). A larger $\varepsilon$ results in faster updates but coarser approximations.
    - Design Motivation: Different application scenarios demand different trade-offs between real-time performance and accuracy; the parameterized design provides flexibility.

### Loss & Training
This is a purely algorithmic work with no training process. The core optimization objective is to minimize the matching cost $\sum_{(r,b) \in M} \|r - b\|$.

## Key Experimental Results

### Main Results

| Dataset | Metric (Running Time) | Ours | Exact Recompute | Greedy Baseline | Speedup (vs Exact) |
|---|---|---|---|---|---|
| Synthetic Gaussian (n=10K) | Update Time (ms) | 0.8 | 245 | 1.2 | 306x |
| Synthetic Gaussian (n=100K) | Update Time (ms) | 3.2 | timeout | 8.5 | >1000x |
| Real Distribution Shift Data | Drift Detection Latency | Low | Extremely High | Low | — |
| Wasserstein Estimation Accuracy | Relative Error | <5% | 0% | 15-30% | — |

### Ablation Study

| Configuration ($\varepsilon$) | Update Time | Approx Ratio (Theoretical) | Empirical Matching Cost Ratio | Description |
|---|---|---|---|---|
| $\varepsilon = 0.1$ | $O(n^{0.1})$ | $O(10)$ | 1.3x-2.1x | High accuracy but slower |
| $\varepsilon = 0.3$ | $O(n^{0.3})$ | $O(3.3)$ | 1.8x-3.5x | Balanced point |
| $\varepsilon = 0.5$ | $O(n^{0.5})$ | $O(2)$ | 2.5x-5.0x | Fastest but coarsest |

### Key Findings
- Achieves fully dynamic bi-chromatic matching in sublinear update time for the first time, theoretically proving an update complexity of $O(n^\varepsilon)$.
- In practical scenarios of distribution shift monitoring, this method significantly outperforms the exact recomputation baselines, and its empirical approximation quality is much better than the theoretical bounds.
- The algorithm remains effective in high-dimensional spaces, with experiments covering scenarios from 2D to 20D.

## Highlights & Insights
- Pioneering Work: Proposes the first sublinear fully dynamic algorithm for this problem, filling an important theoretical gap.
- Highly Practical: Wasserstein distance monitoring is a core requirement in machine learning deployment (concept drift detection).
- Dual Focus on Theory and Experiments: Provides rigorous proofs for the approximation ratios while demonstrating strong empirical performance through experiments.

## Limitations & Future Work
- The tradeoff between the $O(1/\varepsilon)$ approximation ratio and the $O(n^\varepsilon)$ update time could be further optimized.
- Currently, Euclidean distance is primarily considered; generalizing the algorithm to other metric spaces remains an open problem.
- The data scales in the experiments are relatively moderate; performance in ultra-large-scale scenarios (>1M points) remains to be verified.

## Related Work & Insights
- Complements static bi-chromatic matching algorithms (e.g., Hungarian, auction algorithms).
- Connects to works in the field of dynamic graph matching, but the geometric structure provides additional headroom for optimization.
- Efficient estimation of the Wasserstein distance is of direct value for generative model evaluation (e.g., FID/WD) and data quality monitoring.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve sublinear fully dynamic Euclidean bi-chromatic matching
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on both synthetic and real-world data
- Writing Quality: ⭐⭐⭐⭐ Conceptually clear theoretical formulations
- Value: ⭐⭐⭐⭐ Highly significant for dynamic Wasserstein distance estimation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [\[ICML 2025\] Score Matching with Missing Data](score_matching_with_missing_data.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](../../AAAI2026/others/semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)

</div>

<!-- RELATED:END -->
