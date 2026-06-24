---
title: >-
  [Paper Note] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures
description: >-
  [ICML2025][Randomized Dimensionality Reduction] It is proved that for a broad class of Euclidean maximization problems (such as maximum matching, Max-TSP, maximum spanning tree, and subgraph diversity), reducing the dimension to $O(\lambda)$ (where $\lambda$ is the doubling dimension of the dataset) using a data-independent Gaussian JL transform preserves the value of all candidate solutions. This dependency is also shown to be tight.
tags:
  - "ICML2025"
  - "Randomized Dimensionality Reduction"
  - "Johnson-Lindenstrauss"
  - "Doubling Dimension"
  - "Maximum Matching"
  - "Max-TSP"
  - "Diversity Measures"
date: 2026-05-08
content_hash: 93302268ff728688
---

# Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures

**Conference**: ICML2025  
**arXiv**: [2506.00165](https://arxiv.org/abs/2506.00165)  
**Code**: None  
**Area**: Dimensionality Reduction / Diversity Maximization / Combinatorial Optimization  
**Keywords**: Randomized Dimensionality Reduction, Johnson-Lindenstrauss, Doubling Dimension, Maximum Matching, Max-TSP, Diversity Measures

## TL;DR

It is proved that for a broad class of Euclidean maximization problems (such as maximum matching, Max-TSP, maximum spanning tree, and subgraph diversity), reducing the dimension to $O(\lambda)$ (where $\lambda$ is the doubling dimension of the dataset) using a data-independent Gaussian JL transform preserves the value of all candidate solutions. This dependency is also shown to be tight.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Randomized dimensionality reduction (such as the JL transform) is a core technique for accelerating large-scale Euclidean optimization. The classic JL lemma projects $n$ points into $O(\log n)$ dimensions to preserve pairwise distances, but this target dimension grows with the data scale $|X|$.

For **minimization** problems (such as facility location and k-center), prior works have bound the target dimension to the doubling dimension of the dataset $\lambda_X$, achieving the dual advantages of data-independent mapping and data-dependent analysis. However, for **maximization** problems (such as maximum matching, Max-TSP, and diversity maximization), such results were previously almost completely non-existent.

The core problem of this paper: **for which maximization problems can we design data-independent dimensionality reduction where the target dimension depends only on the doubling dimension $\lambda$ rather than $\log n$?**

## Method

### Definition of Doubling Dimension

The doubling dimension $\lambda_X$ is the smallest $\lambda$ such that any ball of radius $r$ containing data points can be covered by $2^\lambda$ balls of radius $r/2$. In the worst case, $\lambda \leq \log n$, but real-world datasets are usually far smaller.

### Core Dimensionality Reduction Scheme

A standard **Gaussian JL mapping** is used: $G \in \mathbb{R}^{t \times d}$, with each element sampled independently from $\mathcal{N}(0, 1/t)$, projecting data from $\mathbb{R}^d$ to $\mathbb{R}^t$.

**Core Theorem (Theorem 2.1)**: For a point set $P \subset \mathbb{R}^d$ with a doubling dimension of $\lambda$, setting the target dimension:

$$t = O\!\left(\varepsilon^{-2}\,\lambda\,\log \frac{1}{\varepsilon}\right)$$

guarantees that with probability $\geq 2/3$, any $(1+\varepsilon)$-approximate maximum matching/Max-TSP solution in the projected space is also a $(1+O(\varepsilon))$-approximate solution in the original space.

### Key Lemmas for Proof

**Lemma 2.4 (Matching Ball Containment)**: In maximum matching, the endpoints of all "short edges" (length $\leq r/4$) are contained in the same ball of radius $r/2$. Utilizing this property, a sequence of balls $B_0, B_1, \ldots$ is recursively constructed, with the radius halved at each level.

**Proof Flow**:
1. For each ball $B_i$, construct an $\varepsilon r_i$-net whose size is controlled by the doubling dimension as $(2/\varepsilon)^{\lambda}$.
2. Use the JL lemma to guarantee that the distance distortion between net points is $\leq 1+\varepsilon\alpha$.
3. Use the Indyk-Naor Lemma (Lemma 1.2) to control the projected distance from points in the ball to the net points.
4. Aggregate the errors of all levels into $O(\varepsilon \cdot \mathrm{opt}(P))$ via Markov's inequality.

### Threshold Phenomenon

The paper discovers a compelling **sharp threshold phenomenon**:
- Target dimension $\geq O(\lambda)$: Guarantees $(1+\varepsilon)$-approximation.
- Target dimension slightly below this threshold: Distortion immediately jumps to $\sqrt{2}$.
- Continuing reduction to $O(1)$ dimensions: Distortion always remains at $\sqrt{2}$ without further deterioration.

**Theorem 4.1** Rybicki proves that without the doubling assumption, projecting to $O(\varepsilon^{-2})$ dimensions yields a $(\sqrt{2}+\varepsilon)$-approximation. This proof establishes an elegant connection between maximum matching and 1-median: $\mathrm{opt}_{\text{1-median}}(P) \leq \sqrt{2} \cdot \mathrm{opt}_{\text{max-match}}(P)$, utilizing the existence of Tverberg graphs.

### Extensions to Other Problems

| Problem | Target Dimension (Upper Bound) | Lower Bound |
|------|----------------|------|
| Maximum Weight Matching | $O(\varepsilon^{-2}\lambda\log\frac{1}{\varepsilon})$ | $\Omega(\lambda)$ |
| Max-TSP | $O(\varepsilon^{-2}\lambda\log\frac{1}{\varepsilon})$ | $\Omega(\lambda)$ |
| Max $k$-Supermatching | $O(\varepsilon^{-2}k^2\lambda\log\frac{k}{\varepsilon})$ | — |
| Maximum Spanning Tree | $O(\varepsilon^{-2}\lambda\log\frac{1}{\varepsilon})$ | $\Omega(\lambda)$ |
| Max $k$-Coverage | $O(\varepsilon^{-2}\lambda\log\frac{1}{\varepsilon})$ | $\Omega(\lambda)$ |
| Subgraph Diversity | $O(\varepsilon^{-2}(\lambda\log\frac{1}{\varepsilon}+\log k))$ | $\Omega(\lambda)$ |

## Key Experimental Results

The method is evaluated on MNIST, CIFAR embeddings, and synthetic datasets, with "low intrinsic dimension" and "high intrinsic dimension" versions prepared for each dataset. Experiments cover three representative problems: maximum matching, remote-clique, and max $k$-coverage.

**Key Observations**:
- Low intrinsic dimension data projected to extremely low dimensions (such as 10-20 dimensions) maintains near-zero relative error.
- High intrinsic dimension data shows significantly larger errors under the same projection dimensions.
- Experiments clearly validate that the doubling dimension is the decisive parameter controlling the representation capability after dimensionality reduction.
- MNIST digit "2" (known to have a low doubling dimension) performs far better under dimensionality reduction compared to its noisy counterpart.
- The low-dimensional rotated version of CIFAR and the original high-dimensional embedding behave completely differently under the same ambient dimension.

## Highlights & Insights

1. **Unified Framework**: Offers a unified treatment of dimensionality reduction for a broad class of Euclidean maximization and diversity problems for the first time, with all upper bounds sharing the same proof template (ball sequences + net points + error aggregation).
2. **Preservation Strength**: The dimensionality reduction guarantee holds for **all candidate solutions** rather than just preserving the optimal value. This implies that any algorithm can be executed in the low-dimensional space for post-processing.
3. **Tight Bounds**: The dependence on $\lambda$ is shown to be necessary across all problems ($\Omega(\lambda)$ lower bounds).
4. **Threshold Phenomenon**: The sharp jump from $(1+\varepsilon) \to \sqrt{2}$ is a novel discovery, revealing the intrinsic behavior of dimensionality reduction on maximization problems.
5. **Connection between 1-median and Maximum Matching** (Lemma 4.2) cleverly utilizes Tverberg graphs and holds independent theoretical value.

## Limitations & Future Work

- Fits only Gaussian JL mapping ($i.i.d.$ Gaussian matrices), without discussing more efficient variants like Sparse JL.
- The experimental scale is relatively small ($n=1000$) and lacks verification in large-scale real-world application scenarios.
- The diversity maximization experiments use greedy approximation rather than exact solutions.
- The target dimension for supermatching contains a factor of $k^2$, and it remains unclear whether this can be improved.
- The threshold phenomenon is verified only on maximum matching and Max-TSP; whether similar phenomena exist for other problems remains unknown.
- Calculating the doubling dimension itself is NP-hard, requiring estimation in practice.

## Related Work & Insights

- Classic work on the **JL Transform** (Johnson & Lindenstrauss 1984; Indyk & Motwani 1998).
- Pioneers in **doubling-dimension dimensionality reduction** (Indyk & Naor 2007; Narayanan et al. 2021) handled nearest neighbors and single-linkage clustering.
- **Dimensionality reduction for minimization problems** (Huang et al. 2024; Jiang et al. 2024) handled facility location and k-center.
- **Diversity maximization** (Indyk et al. 2014; Cevallos et al. 2018, 2019) under low-dimensional settings with PTAS.
- The existence theorem of **Tverberg Graphs** (Pirahmad et al. 2024) provides a key tool for $(\sqrt{2})$-approximation.

## Rating

- Novelty: ⭐⭐⭐⭐ — First unified treatment of doubling-dimension dimensionality reduction for maximization problems; the threshold phenomenon is a brand new discovery.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments clearly validate the theory, but the scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ — Elegant proof structure, clear summary table, and a good coordination between theory and experiments.
- Value: ⭐⭐⭐⭐ — Fills an important gap in the field of dimensionality reduction for maximization, offering instructional significance for algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](../../ACL2025/others/rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)
- [\[ACL 2025\] Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale](../../ACL2025/others/verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)
- [\[NeurIPS 2025\] Emergency Response Measures for Catastrophic AI Risk](../../NeurIPS2025/others/emergency_response_measures_for_catastrophic_ai_risk.md)

</div>

<!-- RELATED:END -->
