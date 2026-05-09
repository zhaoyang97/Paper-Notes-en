---
title: >-
  [Paper Note] Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers
description: >-
  [NeurIPS 2025][Coreset] This paper is the first to eliminate the dependency of the robust geometric median coreset size on the number of outliers $m$: under the condition $n \geq 4m$, it achieves an optimal coreset size of $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$ for $d=1$, and $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ in high dimensions. The core technical contribution is a novel **non-componentwise error analysis**.
tags:
  - NeurIPS 2025
  - Coreset
  - Robust Geometric Median
  - Outliers
  - Clustering
  - Non-Componentwise Error Analysis
date: 2026-05-08
content_hash: bdef4ed94f1cb2da
---

# Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers

**Conference**: NeurIPS 2025
**arXiv**: [2510.24621](https://arxiv.org/abs/2510.24621)
**Area**: Others
**Keywords**: Coreset, Robust Geometric Median, Outliers, Clustering, Non-Componentwise Error Analysis

## TL;DR
This paper is the first to eliminate the dependency of the robust geometric median coreset size on the number of outliers $m$: under the condition $n \geq 4m$, it achieves an optimal coreset size of $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$ for $d=1$, and $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ in high dimensions. The core technical contribution is a novel **non-componentwise error analysis**.

## Background & Motivation

**State of the Field**: The geometric median (Fermat-Weber problem) occupies a central role in computational geometry. Its robust variant allows removing $m$ outliers to resist noise or adversarial perturbations, with the objective function $\text{cost}^{(m)}(P, c) = \min_{|L|=m} \sum_{p \in P \setminus L} \|p - c\|$.

**Limitations of Prior Work**: All known coreset constructions include an $O(m)$ term, since existing methods retain all outlier points in the coreset. When $m = \Theta(n)$ (e.g., the PageBlocks dataset with $m \approx 0.1n$), the coreset loses its compression utility.

**Core Problem**: Can the $O(m)$ term in coreset size be eliminated? Prior results seemed to suggest otherwise, given an existing $\Omega(m)$ lower bound—however, this bound relies on the extreme case $m = n-1$.

**Starting Point**: This paper proves that $n - m = \Omega(n)$ is both necessary and sufficient for eliminating the $O(m)$ term, and proposes a new algorithm along with a novel non-componentwise error analysis under this condition.

## Method

### Lower Bound: $n - m = \Omega(n)$ is Necessary
**Theorem 1.1**: There exists a one-dimensional dataset such that any $\varepsilon$-coreset must have size $\Omega(m/(n-m))$. When $n - m = o(n)$, this bound is $\omega(1)$, so eliminating $O(m)$ requires $n - m = \Omega(n)$.

### One-Dimensional Case: Optimal Coreset (Theorem 1.2)
**Core Subset $P_M$**: Let $P_M = \{p_{m+1}, \ldots, p_{n-m}\}$ be the middle $n - 2m$ points; when $n \geq 4m$, $|P_M| \geq 2m$. All points in $P_M$ are inliers for any center $c$, so $\text{cost}^{(m)}(P, c) \geq \text{cost}(P_M, c)$.

**Three-Stage Algorithm**:
1. **Stage 1**: Apply the optimal coreset construction for vanilla 1D geometric median [37] to $P_M$, producing $\tilde{O}(\varepsilon^{-1/2})$ buckets.
2. **Stage 2**: Partition the outer $2m$ points $P_L \cup P_R$ into exponentially increasing intervals (inner and outer blocks) based on distance to $c_L$ and $c_R$.
3. **Stage 3**: Additional partitioning to ensure $\text{cost}^{(m)}(P, p_{n-m}) = \text{cost}^{(m)}(S, p_{n-m})$.

**Non-Componentwise Error Analysis (Core Technique)**: Traditional methods analyze the error of each bucket individually and sum them (componentwise analysis), requiring per-bucket outlier alignment, which forces all outer points to be retained. This paper instead analyzes the global error **without** requiring per-bucket error control:

$$|\text{cost}^{(m)}(P, c) - \text{cost}^{(m)}(S, c)| \leq |\text{cost}^{(m)}(P, p_{n-m}) - \text{cost}^{(m)}(S, p_{n-m})| + \int_{p_{n-m}}^{c} |f'_P(x) - f'_S(x)| dx$$

**Key Geometric Observation**: $f'_P(c)$ (the derivative of the cost function) equals the number of inliers to the left of $c$ minus those to the right. Thus $|f'_P(c) - f'_S(c)| \leq \sum_i |m_i - m_i'| + 2|B_c|$, where $m_i, m_i'$ are the outlier counts of $P$ and $S$ in each bucket, and $B_c$ is the bucket containing $c$.

As long as each bucket has size $O(\varepsilon n)$, we have $\sum_i |m_i - m_i'| \leq O(\varepsilon n)$, bounding the global error—**individual buckets may have arbitrarily large errors, but these errors cancel across buckets**.

**Coreset Size Analysis**: $P_M$ contributes $\tilde{O}(\varepsilon^{-1/2})$ buckets; the outer points $P_L \cup P_R$ contribute $O(\frac{m}{n}\varepsilon^{-1})$ buckets; the total is $\tilde{O}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$.

**Lower Bound Matching**: A worst-case construction distributes $m$ outliers across $\frac{m}{n}\varepsilon^{-1}$ exponentially increasing intervals; if the coreset omits any interval, the error reaches $2\varepsilon \cdot \text{cost}^{(m)}(P, c)$.

### High-Dimensional Case (Theorem 1.3)
Rather than retaining all $m$ outer points, the method **uniformly samples** $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ points from the outer $m$ points. Leveraging the VC dimension $O(d)$ of the ball range space, the sampled set $S_O$ is, with high probability, an $\varepsilon$-approximation over the ball range space, guaranteeing that the number of outlier-misaligned points is at most $O(\varepsilon m)$.

### Extension to Robust $(k,z)$-Clustering (Theorem 1.5)
Under Assumption 1.4 (each inlier cluster has size $\geq 4m$, with no extreme distant points), the coreset size is $\tilde{O}(k^2\varepsilon^{-2z}\min\{\varepsilon^{-2}, d\})$, eliminating the $O(m)$ term.

## Key Experimental Results

### Size–Error Tradeoff on Six Real Datasets (Robust Geometric Median)

| Dataset | Method | Coreset Size | Empirical Error ↓ |
|---------|--------|-------------|-------------------|
| Census1990 | Ours | 1000 | **0.012** |
| Census1990 | Baseline [39,40] | 2300 | 0.013 |
| PageBlocks ($m \approx 0.1n$) | Ours | 500 | **Below baseline** |
| PageBlocks ($m \approx 0.1n$) | Baseline [39,40] | 500 | Higher |

### Running Time Comparison
At the same empirical error level, the proposed algorithm runs significantly faster than baselines (Table 8) due to the smaller coreset size.

### Robustness of the Data Assumption
Condition 2 of Assumption 1.4 ($\max_p \text{dist}(p, C^*)^z \leq 4k \cdot \text{avg}$) is satisfied on all six real datasets (Table 2), and the algorithm performs well in practice even when the assumption is violated.

## Highlights & Insights
- The **non-componentwise error analysis** is a first in the coreset literature: it permits arbitrarily large errors within individual buckets (as long as they cancel globally), breaking the $\Omega(m)$ barrier of all prior methods.
- The **necessity and sufficiency of $n \geq 4m$** elegantly characterizes the problem structure—whether $O(m)$ can be eliminated depends entirely on the fraction of inliers.
- The **tight optimal coreset size** $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$ in the one-dimensional case clearly reveals the separation between the robust and vanilla settings: deviation occurs when $m > \sqrt{\varepsilon} n$.
- The range space argument is applied to outlier points for the first time, extending the $\varepsilon$-approximation toolkit from inliers to outer points.
- The algorithms are simple and efficient: the one-dimensional algorithm runs in $O(n)$ time and the high-dimensional algorithm in $O(nd)$ time, both linear in complexity.

## Related Work & Insights

| Method | Coreset Size ($d > 1$, $k=1$) | Requires $n \geq 4m$? |
|--------|-------------------------------|----------------------|
| [39] Huang et al. | $O(m) + \tilde{O}(\varepsilon^{-3}\min\{\varepsilon^{-2}, d\})$ | No |
| [40] Huang et al. | $O(m) + \tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ | No |
| [42] Jiang et al. | $O(m\varepsilon^{-1}) + \text{Vanilla size}$ | No |
| **Ours (Thm 1.3)** | $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ | Yes |

## Limitations & Future Work
- A large gap remains between the high-dimensional bound $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ and the one-dimensional optimal $\tilde{O}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$; the optimal high-dimensional coreset size remains open.
- Condition 1 of Assumption 1.4 ($\min_i |P_i^*| \geq 4m$) may fail when cluster sizes are highly imbalanced.
- The analysis is restricted to Euclidean space and the $\ell_2$ distance; extensions to more general metric spaces are mentioned but not analyzed in detail.
- Experiments cover only robust geometric median and $k$-median; $k$-means ($z=2$) is not evaluated experimentally.
- The non-componentwise analysis, while powerful, is fully exploited only in the one-dimensional case—the high-dimensional version still falls back to a range space argument.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Non-componentwise error analysis is an entirely new technical tool with independent research value.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Matching upper and lower bounds, necessary and sufficient conditions, and multi-level generalizations.
- Experimental Thoroughness: ⭐⭐⭐ — Validated on six datasets, but limited to $z=1$ and basic comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear technical overview, effective figures, and complete appendix.
- Overall: ⭐⭐⭐⭐⭐ — Outstanding theoretical contribution that resolves an important open problem in the coreset literature.

## Related Work & Insights
- **vs. Huang et al. [39,40]**: Componentwise error analysis via hierarchical sampling inherently requires per-bucket outlier alignment, making $O(m)$ unavoidable. The proposed non-componentwise analysis fundamentally breaks this bottleneck.
- **vs. Jiang et al. [42]**: A reduction from robust to vanilla yields $O(m\varepsilon^{-1}) + \text{Vanilla size}$, still containing an $m$ term. This paper eliminates it entirely.
- **vs. Feldman & Langberg [27]**: Early constructions require exponentially large size $(k+m)^{O(k+m)}$, rendering them practically infeasible.
- The non-componentwise error analysis may generalize to other compression problems in which a subset of the data can be discarded.
- The ball range space argument applied to outlier points provides a new technical tool for robust clustering coresets.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](overfitting_in_adaptive_robust_optimization.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)

<!-- RELATED:END -->
