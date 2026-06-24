---
title: >-
  [Paper Note] Accelerating Spectral Clustering under Fairness Constraints
description: >-
  [ICML 2025][AI Safety][Fair Clustering] The fair spectral clustering (Fair SC) problem is formulated into a Difference-of-Convex (DC) optimization framework. By employing a variable augmentation strategy and an ADMM-type algorithm, expensive eigendecomposition computations are avoided, achieving significant acceleration on large-scale problems.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Fair Clustering"
  - "Spectral Clustering"
  - "DC Optimization"
  - "ADMM"
  - "Fairness Constraints"
date: 2026-05-08
content_hash: 2afd1f85fea9bbe3
---

# Accelerating Spectral Clustering under Fairness Constraints

**Conference**: ICML 2025  
**arXiv**: [2506.08143](https://arxiv.org/abs/2506.08143)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Fair Clustering, Spectral Clustering, DC Optimization, ADMM, Fairness Constraints

## TL;DR
The fair spectral clustering (Fair SC) problem is formulated into a Difference-of-Convex (DC) optimization framework. By employing a variable augmentation strategy and an ADMM-type algorithm, expensive eigendecomposition computations are avoided, achieving significant acceleration on large-scale problems.

## Background & Motivation

**Background**: The widespread application of algorithmic decision systems in critical areas such as healthcare and education has raised fairness concerns. Fair clustering requires the proportion of each demographic group within each cluster to be consistent with that of the overall population.

**Limitations of Prior Work**: Kleindessner et al. (2019) introduced fairness constraints into spectral clustering but relied on the computationally expensive eigendecomposition of the fairness-constrained graph Laplacian matrix, limiting its applicability to large-scale data. Wang et al. (2023) made improvements but still required eigendecomposition.

**Key Challenge**: The computational complexity of eigendecomposition is $O(n^3)$, which is unscalable as the data volume scales.

**Goal**: How to solve spectral clustering efficiently while maintaining fairness constraints?

**Key Insight**: Spectral clustering is essentially a trace maximization problem under orthogonal constraints, which can be naturally embedded into a DC framework—fairness constraints are integrated into ADMM via variable augmentation.

**Core Idea**: Replace eigendecomposition with DC-ADMM, where each subproblem can be solved using fast gradient methods.

## Method

### Overall Architecture
1. Formulate the Fair SC problem into a DC form: $\min f(X) - g(X)$, where both $f$ and $g$ are convex functions.
2. Introduce auxiliary variables to decouple the fairness constraints from the orthogonal constraints.
3. Alternately optimize each subproblem using ADMM.
4. Use gradient descent to solve approximately in the DC steps.

### Key Designs

1. **DC Problem Transformation**:

    - Function: Transform the trace maximization and fairness constraints of Fair SC into difference-of-convex optimization.
    - Mechanism: $\max \text{tr}(X^T L X)$ s.t. $X^T X = I$, fairness constraints $\rightarrow$ DC form by encoding the orthogonal constraints as penalty terms.
    - Design Motivation: Mature optimization tools exist for the DC framework, avoiding eigendecomposition.

2. **Variable Augmentation + ADMM**:

    - Function: Introduce auxiliary variables to partition different constraints into different subproblems.
    - Mechanism: Isolate $X$ (clustering indicator) and $Y$ (fairness projection), and update them alternately using ADMM.
    - Design Motivation: Each subproblem can be solved efficiently—eliminating the need for full-matrix eigendecomposition.

### Loss & Training
- DC Iterations: Outer DCA (DC Algorithm) + Inner ADMM.
- Convergence to a DC critical point.

## Key Experimental Results

### Main Results
Comparison of computation time under different data scales:

| Dataset | n | Fair SC (Original) | Wang et al. | Ours | Speedup |
|--------|---|---------------|-------------|------|--------|
| Synthetic | 1K | 2.1s | 0.8s | 0.3s | 7× |
| Synthetic | 10K | 210s | 85s | 12s | 17× |
| Bank | 45K | >1h | 1800s | 180s | ~10× |

### Ablation Study

| Configuration | Time | Clustering Quality | Description |
|------|------|---------|------|
| No Fairness Constraints (Standard SC) | Fastest | Highest NMI but Unfair | Baseline |
| Fair SC (Eigendecomposition) | Slowest | Fair + Good Quality | Prior Method |
| Ours (DC-ADMM) | 10×+ Faster | Fair + Comparable Quality | Significant Efficiency Gain |

### Key Findings
- As the data volume grows, the speedup ratio becomes increasingly larger—the scalability of our method is far superior to eigendecomposition-based methods.
- Clustering quality (NMI, balance) is on par with prior methods.
- The acceleration effect is more pronounced when the number of clusters $k$ increases.

## Highlights & Insights
- The DC framework combined with ADMM provides a general and efficient methodology for fairness-constrained optimization.
- Avoiding eigendecomposition makes it feasible to scale the method to tens of thousands of samples.
- The variable augmentation strategy naturally integrates fairness constraints into the optimization framework.

## Limitations & Future Work
- DC optimization only guarantees convergence to a critical point, not the global optimum.
- Precomputing the graph Laplacian matrix is still challenging for ultra-large-scale data.
- The definition of fairness only considers group fairness, without addressing individual fairness.

## Related Work & Insights
- **vs Kleindessner et al.**: Original Fair SC method, requiring full eigendecomposition.
- **vs Wang et al.**: Achieves partial acceleration but still relies on eigendecomposition, which is completely avoided by ours.

## Rating
- Novelty: ⭐⭐⭐⭐ Employing DC+ADMM for fair spectral clustering is a novel technical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough validation of scalability on both synthetic and real-world data.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematics and standardized structure.
- Value: ⭐⭐⭐⭐ Renders fair clustering applicable to real-scale data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](../../ICML2026/ai_safety/optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2025\] Relative Error Fair Clustering in the Weak-Strong Oracle Model](relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)
- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](../../NeurIPS2025/ai_safety/unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[NeurIPS 2025\] Fairness under Competition](../../NeurIPS2025/ai_safety/fairness_under_competition.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)

</div>

<!-- RELATED:END -->
