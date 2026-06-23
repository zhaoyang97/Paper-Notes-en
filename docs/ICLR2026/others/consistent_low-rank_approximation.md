---
title: >-
  [Paper Note] Consistent Low-Rank Approximation
description: >-
  [ICLR 2026][Others][low-rank approximation] The paper proposes and systematically studies the "Consistent Low-Rank Approximation" problem—maintaining a near-optimal rank-$k$ approximation of a matrix whose rows arrive in a stream while minimizing the total change in the solution (recourse). It proves that $O(k/\varepsilon \cdot \log(nd))$ recourse is feasible un
tags:
  - ICLR 2026
  - Others
  - low-rank approximation
  - streaming algorithm
  - consistency
  - recourse
  - online algorithm
date: 2026-05-08
content_hash: d65f46623ab4af41
---
# Consistent Low-Rank Approximation

**Conference**: ICLR2026  
**arXiv**: [2603.02148](https://arxiv.org/abs/2603.02148)  
**Code**: To be confirmed  
**Area**: Others  
**Keywords**: low-rank approximation, streaming algorithm, consistency, recourse, online algorithm

## TL;DR
The paper proposes and systematically studies the "Consistent Low-Rank Approximation" problem—maintaining a near-optimal rank-$k$ approximation of a matrix whose rows arrive in a stream while minimizing the total change in the solution (recourse). It proves that $O(k/\varepsilon \cdot \log(nd))$ recourse is feasible under additive error, $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$ recourse is feasible under $(1+\varepsilon)$ multiplicative error, and provides a lower bound of $\Omega(k/\varepsilon \cdot \log(n/k))$.

## Background & Motivation

**Background**: Low-rank approximation is a core tool in machine learning (recommendation systems, dimensionality reduction, feature engineering). In streaming scenarios where data arrives row by row, it is necessary to maintain low-rank approximations online. Frequent Directions and online ridge leverage score sampling are the prevailing methods.

**Limitations of Prior Work**: (a) Online algorithms focus solely on approximation quality, neglecting solution stability—frequent changes to factor matrices lead to high retraining costs for downstream models; (b) Frequent Directions can incur a recourse of $O(n)$ when the $k$-th and $(k+1)$-th singular vectors alternate (disastrous); (c) Consistency (recourse) has been studied in clustering and caching problems but has not yet been systematized for low-rank approximation.

**Key Challenge**: The optimal subspace for online low-rank approximation can change completely at every step ($\Omega(nk)$ recourse), yet downstream applications require stable feature spaces. The goal is to achieve an optimal trade-off between approximation quality and solution stability.

**Goal**: To formalize the consistent low-rank approximation problem and provide upper and lower bounds.

**Key Insight**: Measure recourse using the Frobenius distance between subspace projection matrices, combine online ridge leverage score sampling to reduce effective stream length, and minimize step-wise factor changes through fine-grained singular value analysis.

**Core Idea**: Compress the stream length via ridge leverage sampling and process updates by categorizing singular values (replacing small singular values directly while maintaining stable large ones) to achieve sub-quadratic recourse.

## Method

### Overall Architecture
Matrix $\mathbf{A} \in \mathbb{R}^{n \times d}$ arrives row by row. At each time $t$, the algorithm outputs a rank-$k$ factor $\mathbf{V}^{(t)} \in \mathbb{R}^{k \times d}$. The objective is to ensure both approximation quality $\|\mathbf{A}^{(t)} - \mathbf{A}^{(t)} (\mathbf{V}^{(t)})^\top \mathbf{V}^{(t)}\|_F^2 \leq (1+\varepsilon) \cdot \text{OPT}_t$ and minimize the total recourse over the stream $\sum_t \|\mathbf{P}_{\mathbf{V}^{(t)}} - \mathbf{P}_{\mathbf{V}^{(t-1)}}\|_F^2$. To this end, the paper presents an additive error algorithm, a multiplicative error algorithm, and a matching lower bound.

### Key Designs

**1. Measuring Recourse: Frobenius Distance of Projection Matrices**

The first step in addressing consistency is selecting the correct metric for "change." If one directly compares the difference between adjacent factor matrices $\mathbf{V}^{(t)}$ and $\mathbf{V}^{(t-1)}$, a meaningless basis rotation (changing the orthogonal basis for the same subspace) would be miscalculated as a massive change. This work instead uses the Frobenius distance of subspace projection matrices $\|\mathbf{P}_{\mathbf{V}^{(t)}} - \mathbf{P}_{\mathbf{V}^{(t-1)}}\|_F^2$ to measure recourse—it only considers whether the spanned subspace actually changes and is invariant to basis rotations. This makes recourse a natural and robust metric, allowing subsequent analysis to focus on the subspace level.

**2. Additive Error Algorithm: Triggering Recomputation via Norm Growth**

Under additive error (Theorem 1.1), the algorithm is minimalist: it monitors the Frobenius norm of the current matrix. Recomputation of the SVD occurs only when $\|\mathbf{A}^{(t)}\|_F^2$ increases by a factor of $(1+\varepsilon)$ compared to the last recomputation; otherwise, the old factor is retained. This design works because the total energy of new rows between two recomputations is capped at $\leq \varepsilon \cdot \|\mathbf{A}^{(t)}\|_F^2$, naturally controlling the additive error. Since the norm doubles at most $O(1/\varepsilon \cdot \log(ndM))$ times (where $M$ is the element upper bound), and the recourse per recomputation is at most $k$, the total recourse becomes $O(k/\varepsilon \cdot \log(ndM))$—which nearly matches the lower bound.

**3. Multiplicative Error Algorithm: Compressing Stream Length and Case-wise Tail Singular Value Updates**

This is the core contribution: tightening the error guarantee to $(1+\varepsilon)$ multiplicative (Theorem 1.3). While direct recomputation at every step costs $k^2/\varepsilon^2 \cdot \text{polylog}$, this work reduces it to sub-quadratic through two steps. First, online ridge leverage score sampling compresses the effective stream length from $n$ to $k/\varepsilon \cdot \text{polylog}$, ensuring expensive updates only happen for a few "important" rows. Second, fine-grained updates are performed on the compressed stream, specifically treating the weakest $\sqrt{k}$ of the top-$k$ singular values with case work: when the energy of this tail is small ($\sum_{i=k-\sqrt{k}}^{k} \sigma_i^2$ is very small), these directions are insignificant, and the tail singular vectors can be directly replaced by new rows, accumulating enough changes over $\sqrt{k}$ steps before a unified recomputation (averaging $O(k)$ recourse per $\sqrt{k}$ steps); when tail energy is large, the optimal subspace is inherently stable and won't be violently disturbed by new rows, so the top SVD is recomputed with a single recourse not exceeding $\sqrt{k}$. Together, these yield a total recourse of $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$ for integer-bounded matrices. The $\sqrt{k}$ threshold optimizes the trade-off between "replacing $r$ factors costing $k \cdot r$" and "recomputing every $k^2/r$ steps." Fine-grained analysis is also applied to anti-Hadamard integer matrices where the optimal low-rank cost can be exponentially small. For data streams with a polynomial online condition number, the bound can be further reduced to $k/\varepsilon^2 \cdot \text{polylog}$.

**4. Lower Bound: Subspace Alternation Construction for $\Omega(k/\varepsilon \cdot \log(n/k))$**

To demonstrate that the recourse is near its limit, the paper constructs a hard instance (Theorem 1.4): the stream is divided into $\Theta(1/\varepsilon \cdot \log(n/k))$ phases, each forcing the optimal subspace to switch between two sets of orthogonal bases. Even if the entire stream is known in advance, any algorithm maintaining a multiplicative $(1+\varepsilon)$ approximation must switch subspaces, resulting in a total recourse of at least $\Omega(k/\varepsilon \cdot \log(n/k))$. This is only a $\sqrt{k}$ factor away from the additive upper bound, delineating the room for improvement in multiplicative bounds.

## Key Experimental Results

### Main Results (Empirical Evaluation)

| Algorithm | Approx Quality | Measured Recourse | Description |
|-----------|----------------|-------------------|-------------|
| Frequent Directions | Good | $O(n)$ (Catastrophic) | Singular values alternate frequently |
| Naive SVD Recompute | Optimal | $O(nk)$ | Recompute every step |
| Ridge Sampling + Recompute | $(1+\varepsilon)$ | $O(k^2/\varepsilon^2)$ | Baseline |
| **Ours** | $(1+\varepsilon)$ | $O(k^{3/2}/\varepsilon^2)$ | Sub-quadratic, optimal |

### Key Findings
- **Frequent Directions suffers catastrophic recourse**: Its $O(n)$ level makes it entirely unsuitable for applications requiring stability.
- **Theoretical Worst-case vs. Reality**: The recourse on actual data is significantly lower than the theoretical upper bound.
- **The $\sqrt{k}$ threshold is critical**: Switching case work between "weak tail" and "strong tail" is the core algorithmic design.

## Highlights & Insights
- **Pioneering Problem Definition**: First to extend consistency (recourse) from clustering to low-rank approximation—filling a significant gap in online algorithm theory.
- **Elegant $\sqrt{k}$ Balance Point**: The trade-off between replacing $r$ factors (step-wise recourse $k \cdot r$) and periodic recomputation ($k^2/r$) is neatly optimized at $r = \sqrt{k}$.
- **Rigorous Handling of Anti-Hadamard Matrices**: Addressing boundary cases where optimal costs for integer matrices are exponentially small demonstrates theoretical depth.
- **Clear Practical Motivation**: Solving the engineering pain point of frequent model retraining in feature engineering.

## Limitations & Future Work
- **Purely Theoretical Contribution**: The experiments are relatively simple and lack validation on large-scale real-world datasets (e.g., Netflix Prize).
- **Gap between Bounds**: With an upper bound of $O(k^{3/2}/\varepsilon^2)$ vs. a lower bound of $\Omega(k/\varepsilon)$, the tightness of the $\sqrt{k}$ factor remains an open question.
- **Row-only Arrival**: More complex streaming models like column arrival, sliding windows, and insertion-deletion are not covered.
- **Future Directions**: (a) Closing the gap between upper and lower bounds; (b) Expanding to dynamic streams (insertion + deletion); (c) Improving practical performance by incorporating distributional assumptions.

## Related Work & Insights
- **vs. Consistent Clustering (Lattanzi & Vassilvitskii)**: Consistent clustering uses geometric properties to select robust centers; low-rank approximation requires different subspace techniques.
- **vs. Frequent Directions**: FD is an excellent streaming algorithm but has disastrous recourse. The proposed algorithm is specifically designed for consistency.
- **vs. Online PCA**: Online PCA focuses on regret rather than recourse. These two objectives are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering problem definition + non-trivial algorithmic design and analysis.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical, with basic but sufficient experimental validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear technical overview, intuitive explanations, and deep anti-Hadamard analysis.
- Value: ⭐⭐⭐⭐ Opens a new direction for consistent online algorithms with practical inspiration for feature engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification](random_anchors_with_low-rank_decorrelated_learning_a_minimalist_pipeline_for_cla.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[CVPR 2026\] Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation](../../CVPR2026/others/basis-oriented_low-rank_transfer_for_few-shot_and_test-time_adaptation.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](../../ACL2025/others/adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[ICLR 2026\] Permutation-Consistent Variational Encoding for Incomplete Multi-View Multi-Label Classification](permutation-consistent_variational_encoding_for_incomplete_multi-view_multi-labe.md)

</div>

<!-- RELATED:END -->
