---
title: >-
  [Paper Note] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm
description: >-
  [NeurIPS 2025][Interpretability][Explainable clustering] This paper presents the first explainable k-medians clustering algorithm for general $\ell_p$ norms…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Explainable clustering"
  - "k-medians"
  - "decision trees"
  - "dynamic algorithms"
  - "approximation ratio"
date: 2026-05-08
content_hash: 69dcc9e3549d8412
---

# Dynamic Algorithm for Explainable k-medians Clustering under lp Norm

**Conference**: NeurIPS 2025
**arXiv**: [2512.01150](https://arxiv.org/abs/2512.01150)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Explainable clustering, k-medians, decision trees, dynamic algorithms, approximation ratio

## TL;DR
This paper presents the first explainable k-medians clustering algorithm for general $\ell_p$ norms, achieving an approximation ratio of $\tilde{O}(p(\log k)^{1+1/p-1/p^2})$ (improving the best known bound for $p=2$), along with the first dynamic variant: maintaining an explainable clustering under center insertions/deletions with $O(d \log^3 k)$ amortized update time and $O(\log k)$ amortized reassignments.

## Background & Motivation

**Background**: Explainable clustering was introduced by Dasgupta et al. (2020), where clustering partitions are represented by threshold decision trees—each internal node applies a threshold comparison on a single feature, making cluster assignments transparent and traceable. The central question is the price of explainability: how well can the cost of a decision-tree clustering approximate the optimal unconstrained cost?

**Limitations of Prior Work**: The competitive ratio under the $\ell_1$ norm has been tightly characterized as $1 + H_{k-1}$; under $\ell_2$, the best known bound is $\tilde{O}(\log^{3/2} k)$, leaving room for improvement; no algorithm existed for general $\ell_p$ ($p > 2$); and all prior algorithms are static.

**Key Challenge**: General $\ell_p$ norms require more carefully designed random cut distributions—uniform random cuts yield a competitive ratio of $\Theta(k^{1-1/p})$. In the dynamic setting, the hierarchical structure of decision trees makes local updates more challenging.

**Goal**: (a) Design explainable k-medians algorithms for all finite $p \geq 1$ and improve the bound for $p=2$; (b) provide the first dynamic version.

**Key Insight**: Non-uniform random cuts with CDF $x^p/R_t^p$, combined with Poisson process timestamps to enable dynamic updates.

**Core Idea**: Through $\ell_p$-adapted non-uniform random cuts and Poisson-timestamp-driven dynamic maintenance, the paper achieves near-optimal explainable clustering for all $\ell_p$ norms along with efficient dynamic updates for the first time.

## Method

### Overall Architecture
The algorithm takes $k$ reference centers $C$ as input and outputs a $k$-leaf threshold decision tree $\mathcal{T}$. The core procedure Partition_Leaf recursively splits each node containing multiple centers into child nodes. The algorithm is data-agnostic—it depends only on the reference center set.

### Key Designs

1. **Non-Uniform Random Cuts (Partition_Leaf)**:

    - **Function**: Recursively partitions the space containing center set $C_u$ until each leaf contains at most half the centers.
    - **Mechanism**: An anchor point $m^u$ (coordinate-wise median) is selected; the algorithm iteratively samples a random coordinate $i$, sign $\sigma$, and parameter $Z_t \sim \text{Uniform}[0, R_t^p]$, sets $\theta_t = Z_t^{1/p}$, and defines the cut threshold $\vartheta_t = m^u_i + \sigma \cdot \theta_t$.
    - **Key Property**: The CDF of the threshold is $x^p / R_t^p$—cut density is low near the anchor and high far from it, ensuring separation probability scales with the $p$-th power of the $\ell_p$ distance.
    - **Why Not Uniform Cuts**: Uniform cuts treat all distances equally, causing the competitive ratio to degrade to $\Theta(k^{1-1/p})$.

2. **Three-Category Cut Analysis Framework**:

    - **Function**: Precisely analyzes the expected cost of incorrectly cutting a data point $x$.
    - **Core Classification**:
        - **Safe cuts**: The retreat distance is small ($A_k M_t \leq R_t / (6^p \log^2 k)$); total contribution is $O(1) \cdot \|x-c\|_p$.
        - **Light cuts**: The radius is small ($R_t \leq 6 \log^\alpha k \cdot \max\{\|x-m^t\|, \|c-m^t\|\}$); contribution is $O(p \log^{1+\alpha} k) \cdot \|x-c\|_p$.
        - **Heavy and unsafe cuts**: Bounded via Hölder's inequality; contribution is $O((\log k)^{2-1/p-\alpha(p-1)}) \cdot \|x-c\|_p$.
    - Setting $\alpha = 1/p - 1/p^2$ balances the two terms, yielding the recurrence $A_k \leq O(p (\log k)^{1+1/p-1/p^2} \log\log k)$.

3. **Dynamic Algorithm (Poisson Timestamps)**:

    - **Function**: Efficiently updates the decision tree upon insertions or deletions in the center set.
    - **Mechanism**: Each random cut is assigned an arrival time $\rho_t$ drawn from a Poisson process; timestamps remain invariant under center changes.
        - *Insertion*: Locate the earliest cut that separates the new center from the anchor, and insert it at the corresponding position in the tree.
        - *Deletion*: Locate the leaf and remove the corresponding cut.
        - *Rebuild*: Triggered when the number of updates exceeds one quarter of the initial center count.
    - **Complexity**: $O(d \log^3 k)$ amortized update time; $O(\log k)$ amortized reassignments.

### Theoretical Guarantees

**Theorem 3.1 (Static)**: $\mathbf{E}[\text{cost}_p(X, \mathcal{T})] \leq O(p \cdot (\log k)^{1+1/p-1/p^2} \log\log k) \cdot \text{cost}_p(X, C)$

**Theorem 5.1 (Dynamic)**: Same approximation ratio, with $O(d \log^3 k)$ amortized update time and $O(\log k)$ amortized reassignments.

## Key Experimental Results

### Theoretical Approximation Ratio Comparison

| Norm | Prev. Best Upper Bound | Ours | Lower Bound | Gap |
|------|----------------------|------|-------------|-----|
| $\ell_1$ | $1 + H_{k-1}$ (tight) | $O(\log k \log\log k)$ | $\Omega(\log k)$ | $O(\log\log k)$ |
| $\ell_2$ | $\tilde{O}(\log^{3/2} k)$ | $\tilde{O}(\log^{5/4} k)$ | $\Omega(\log k)$ | $\tilde{O}(\log^{1/4} k)$ |
| General $\ell_p$ | None | $\tilde{O}(p (\log k)^{1+1/p-1/p^2})$ | $\Omega(\log k)$ | Polylogarithmic |

### Dynamic Algorithm Complexity

| Operation | Update Time (Amortized) | Reassignments (Amortized) | Approximation Guarantee |
|-----------|------------------------|--------------------------|------------------------|
| Center insertion | $O(d \log^3 k)$ | $O(\log k)$ | Same as static |
| Center deletion | $O(d \log^3 k)$ | $O(\log k)$ | Same as static |

### Key Findings
- For $\ell_2$, the exponent improves from $3/2$ to $5/4$, achieved through the more refined three-category cut analysis.
- The logarithmic exponent $1 + 1/p - 1/p^2$ always lies in $[1, 1.25]$, attaining its maximum at $p=2$.
- A lower bound theorem establishes that no $p$-oblivious algorithm can be simultaneously optimal for all $p$ (an $\Omega(d^{1/4})$ lower bound exists).
- The dynamic algorithm can be composed with the dynamic k-medians framework of Bhattacharya et al. (2025) to achieve fully dynamic explainable clustering.

## Highlights & Insights
- **Elegance of Non-Uniform Cuts**: The CDF $x^p/R_t^p$ is a natural match to the geometry of $\ell_p$, making separation probability proportional to the $p$-th power of the distance—a key enabler of the logarithmic competitive ratio.
- **Poisson Timestamps as an Algorithmic Tool**: The exponential clock technique, originally used purely for analysis, is elevated here to a design primitive; timestamps provide a globally unique identifier for each cut.
- **A General Bridge from Static to Dynamic**: The paradigm of first assigning timestamps and then maintaining them may generalize to the dynamization of other recursive randomized algorithms.

## Limitations & Future Work
- For $p=1$, the competitive ratio is $O(\log k \log\log k)$, falling short of the tight bound by an $O(\log\log k)$ factor.
- For $p=2$, a gap of $\tilde{O}(\log^{1/4} k)$ remains.
- The dynamic algorithm requires a full subtree rebuild at reconstruction thresholds, potentially causing occasional high-latency updates.
- As a purely theoretical work, empirical evaluation of practical clustering performance is entirely absent.

## Related Work & Insights
- **vs. Dasgupta et al. (2020)**: The foundational work achieves only an $O(k)$ upper bound; the present work attains logarithmic bounds for all $\ell_p$.
- **vs. Gupta et al. (2023)**: The tight $\ell_1$ bound of $1 + H_{k-1}$; this paper falls short by only $O(\log\log k)$ while covering all $\ell_p$.
- **vs. Makarychev & Shan (2021)**: The $\tilde{O}(\log^{3/2} k)$ bound for $\ell_2$ is improved to $\tilde{O}(\log^{5/4} k)$ and generalized.
- **vs. Bhattacharya et al. (2025)**: Dynamic unconstrained k-medians; the dynamic explainable clustering developed here can be directly layered on top.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First explainable clustering algorithm for general $\ell_p$ norms plus the first dynamic version; major theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐ — Purely theoretical; no empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical derivations; the three-category cut analysis framework is clearly presented.
- **Value**: ⭐⭐⭐⭐ — Significant contribution to the theoretical foundations of explainable AI; dynamization is practically relevant for streaming data scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](spex_a_spectral_approach_to_explainable_clustering.md)
- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](empowering_decision_trees_via_shape_function_branching.md)

</div>

<!-- RELATED:END -->
