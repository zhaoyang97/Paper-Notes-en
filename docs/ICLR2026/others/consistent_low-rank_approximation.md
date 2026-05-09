---
title: >-
  [Paper Note] Consistent Low-Rank Approximation
description: >-
  [ICLR 2026][low-rank approximation] This paper formalizes and systematically studies the *consistent low-rank approximation* problem—maintaining a near-optimal rank-$k$ approximation of a matrix whose rows arrive in a stream while minimizing the total variation (recourse) of the solution. It proves that $O(k/\varepsilon \cdot \log(nd))$ recourse is achievable under additive error, $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$ recourse is achievable under multiplicative $(1+\varepsilon)$ error, and establishes a lower bound of $\Omega(k/\varepsilon \cdot \log(n/k))$.
tags:
  - ICLR 2026
  - low-rank approximation
  - streaming algorithm
  - consistency
  - recourse
  - online algorithm
date: 2026-05-08
content_hash: 78b7a60f73b4a840
---

# Consistent Low-Rank Approximation

**Conference**: ICLR 2026
**arXiv**: [2603.02148](https://arxiv.org/abs/2603.02148)
**Code**: To be confirmed
**Area**: Other
**Keywords**: low-rank approximation, streaming algorithm, consistency, recourse, online algorithm

## TL;DR
This paper formalizes and systematically studies the *consistent low-rank approximation* problem—maintaining a near-optimal rank-$k$ approximation of a matrix whose rows arrive in a stream while minimizing the total variation (recourse) of the solution. It proves that $O(k/\varepsilon \cdot \log(nd))$ recourse is achievable under additive error, $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$ recourse is achievable under multiplicative $(1+\varepsilon)$ error, and establishes a lower bound of $\Omega(k/\varepsilon \cdot \log(n/k))$.

## Background & Motivation

**Background**: Low-rank approximation is a core tool in machine learning (recommender systems, dimensionality reduction, feature engineering). In streaming settings where data arrives row by row, online maintenance of low-rank approximations is required. Frequent Directions and online ridge leverage score sampling are the dominant approaches.

**Limitations of Prior Work**: (a) Online algorithms focus solely on approximation quality, ignoring solution stability—frequent changes to factor matrices incur high retraining costs for downstream models; (b) Frequent Directions can suffer $O(n)$ recourse (catastrophic) when the $k$-th and $(k+1)$-th singular vectors alternate; (c) consistency/recourse has been studied in clustering and caching problems but has not been systematically addressed for low-rank approximation.

**Key Challenge**: The optimal subspace of an online low-rank approximation may change completely at every step ($\Omega(nk)$ recourse), yet downstream applications require a stable feature space. The goal is to achieve an optimal trade-off between approximation quality and solution stability.

**Goal**: Formalize the consistent low-rank approximation problem and establish matching upper and lower bounds.

**Key Insight**: Recourse is measured by the Frobenius distance between successive subspace projection matrices. Online ridge leverage score sampling is used to reduce the effective stream length, and refined singular value analysis minimizes per-step factor changes.

**Core Idea**: Compress the stream length via ridge leverage sampling, then handle each update by classifying singular values (directly replacing small singular values; stabilizing large ones), achieving sub-quadratic recourse.

## Method

### Overall Architecture
A matrix $\mathbf{A} \in \mathbb{R}^{n \times d}$ arrives row by row. The goal is to output factors $\mathbf{V}^{(t)} \in \mathbb{R}^{k \times d}$ at each time $t$ such that $\|\mathbf{A}^{(t)} - \mathbf{A}^{(t)} (\mathbf{V}^{(t)})^\top \mathbf{V}^{(t)}\|_F^2 \leq (1+\varepsilon) \cdot \text{OPT}_t$, while minimizing total recourse $\sum_t \|\mathbf{P}_{\mathbf{V}^{(t)}} - \mathbf{P}_{\mathbf{V}^{(t-1)}}\|_F^2$.

### Key Designs

1. **Additive Error Algorithm (Theorem 1.1)**:

    - Mechanism: Track the Frobenius norm. Recompute the SVD whenever $\|\mathbf{A}^{(t)}\|_F^2$ grows by a factor of $(1+\varepsilon)$.
    - Recourse: $O(1/\varepsilon \cdot \log(ndM))$ recomputations, each incurring recourse $k$, yielding total $O(k/\varepsilon \cdot \log(ndM))$.
    - Correctness: Between two recomputations, new rows contribute at most $\varepsilon \cdot \|\mathbf{A}^{(t)}\|_F^2$.

2. **Multiplicative Error Algorithm (Theorem 1.3, main contribution)**:

    - Step 1: Online ridge leverage score sampling compresses the stream length from $n$ to $k/\varepsilon \cdot \text{polylog}$.
    - Step 2: Apply refined updates to the compressed stream.
    - **Key Analysis**: Case analysis on the bottom $\sqrt{k}$ of the top-$k$ singular values:
        - If $\sum_{i=k-\sqrt{k}}^k \sigma_i^2$ is small (weak tail): new rows can replace tail singular vectors; recompute every $\sqrt{k}$ steps → $O(k)$ recourse per $\sqrt{k}$ steps.
        - If the tail is strong: the optimal subspace does not change drastically → directly recompute the top SVD, incurring recourse $\leq \sqrt{k}$.
    - Total recourse: $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$—the $\sqrt{k}$ threshold optimally balances the two cases.
    - **Special Handling for Anti-Hadamard Matrices**: The optimal low-rank cost for integer matrices can be exponentially small, requiring refined analysis in this regime.

3. **Lower Bound (Theorem 1.4)**:

    - Construction: Divide into $\Theta(1/\varepsilon \cdot \log(n/k))$ phases, each alternating the optimal subspace between two sets of orthogonal bases.
    - Result: $\Omega(k/\varepsilon \cdot \log(n/k))$ recourse is necessary.

### Recourse Metric
Recourse is measured by the Frobenius distance between subspace projection matrices rather than the difference between basis vectors—this is invariant to basis rotations and constitutes a natural, robust metric.

## Key Experimental Results

### Main Results (Empirical Evaluation)

| Algorithm | Approximation Quality | Observed Recourse | Notes |
|---|---|---|---|
| Frequent Directions | Good | $O(n)$ (catastrophic) | Singular value alternation causes frequent changes |
| Naïve SVD recomputation | Optimal | $O(nk)$ | Recomputed at every step |
| Ridge Sampling + recomputation | $(1+\varepsilon)$ | $O(k^2/\varepsilon^2)$ | Baseline |
| **Proposed Algorithm** | $(1+\varepsilon)$ | $O(k^{3/2}/\varepsilon^2)$ | Sub-quadratic, optimal |

### Key Findings
- **Catastrophic recourse of Frequent Directions**: $O(n)$ recourse renders it entirely unsuitable for stability-sensitive applications.
- **Worst-case theory vs. practice**: Recourse on real data is significantly lower than the theoretical upper bound.
- **The $\sqrt{k}$ threshold is critical**: Switching between the "weak tail" and "strong tail" cases is the central algorithmic design insight.

## Highlights & Insights
- **Pioneering problem formulation**: This is the first work to extend the consistency/recourse framework from clustering to low-rank approximation, filling an important gap in online algorithm theory.
- **Elegance of the $\sqrt{k}$ balance point**: Replacing $r$ factors incurs recourse $k \cdot r$ vs. $k^2/r$; the optimal choice $r = \sqrt{k}$ yields a clean trade-off analysis.
- **Refined treatment of Anti-Hadamard matrices**: Handling the edge case where the optimal cost for integer matrices is exponentially small demonstrates significant theoretical depth.
- **Clear practical motivation**: Frequent retraining in feature engineering is a genuine pain point—consistent low-rank approximation directly addresses this engineering challenge.

## Limitations & Future Work
- **Primarily theoretical**: Experiments are relatively simple and lack validation on large-scale real-world datasets (e.g., Netflix Prize data).
- **Gap between upper and lower bounds**: Upper bound $O(k^{3/2}/\varepsilon^2)$ vs. lower bound $\Omega(k/\varepsilon)$—it remains open whether the $\sqrt{k}$ factor is tight.
- **Row-arrival only**: More complex streaming models such as column arrival, sliding windows, and insertion-deletion streams are not covered.
- **Future directions**: (a) Close the gap between upper and lower bounds; (b) extend to dynamic streams (insertions and deletions); (c) leverage distributional assumptions to improve practical performance.

## Related Work & Insights
- **vs. Consistent Clustering (Lattanzi & Vassilvitskii)**: Consistent clustering exploits geometric properties to select robust centers. The analogy for low-rank approximation is not direct—subspace structure requires different techniques.
- **vs. Frequent Directions**: FD is an excellent streaming algorithm but suffers catastrophic recourse. The proposed algorithm is designed specifically for consistency.
- **vs. Online PCA**: Online PCA optimizes regret, not recourse. The two objectives are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering problem formulation with non-trivial algorithm design and analysis
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; experimental validation is present but limited
- Writing Quality: ⭐⭐⭐⭐⭐ Clear technical overview, well-motivated intuitions, and deep anti-Hadamard analysis
- Value: ⭐⭐⭐⭐ Opens a new direction for consistent online algorithms with practical implications for feature engineering

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](../../AAAI2026/others/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)

<!-- RELATED:END -->
