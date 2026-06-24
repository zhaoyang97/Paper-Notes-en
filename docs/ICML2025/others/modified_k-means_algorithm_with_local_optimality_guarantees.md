---
title: >-
  [Paper Note] Modified K-means Algorithm with Local Optimality Guarantees
description: >-
  [ICML2025][K-means] This work identifies a long-standing misconception that the classic K-means algorithm always converges to a local optimum. It proposes the LO-K-means modification, which guarantees convergence to a continuous or discrete local optimum without increasing the per-step computational complexity.
tags:
  - "ICML2025"
  - "K-means"
  - "Local Optimality"
  - "Bregman Divergence"
  - "Clustering Algorithm"
  - "Combinatorial Optimization"
date: 2026-05-08
content_hash: d3f794d46e358130
---

# Modified K-means Algorithm with Local Optimality Guarantees

**Conference**: ICML2025  
**arXiv**: [2506.06990](https://arxiv.org/abs/2506.06990)  
**Code**: [GitHub](https://github.com/lmingyi/LO-K-means)  
**Area**: Others/Clustering  
**Keywords**: K-means, Local Optimality, Bregman Divergence, Clustering Algorithm, Combinatorial Optimization

## TL;DR

This work identifies a long-standing misconception that the classic K-means algorithm always converges to a local optimum. It proposes the LO-K-means modification, which guarantees convergence to a continuous or discrete local optimum without increasing the per-step computational complexity.

## Background & Motivation

K-means (Lloyd's algorithm) is one of the most widely used clustering algorithms. For a long time, the academic community has generally believed that while K-means cannot achieve global optimality, it at least converges to a **local optimum**. This view is held by scikit-learn documentation and multiple classic papers (Grunau & Rozhoň 2022, Balcan et al. 2018). This "consensus" traces back to the proof by Selim & Ismail (1984); however, that work only proved that K-means converges in a finite number of steps, without rigorously proving local optimality.

The authors found a crucial error in Lemma 7 of Selim & Ismail: the local optimality condition used in that lemma only holds when the objective function $F$ is **convex**, whereas $F(P)$ is actually a **concave function** with respect to $P$. This implies that the solution to which K-means converges may **not** be a local optimum.

**Core Problem**: Can the K-means algorithm be modified to guarantee local optimality while maintaining the same computational complexity?

## Method

### Problem Formulation

Consider the weighted K-means problem, given $N$ points $X=\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, weights $W=\{w_i\}$, and $K$ clusters, using the Bregman divergence $D_\phi$ as the metric:

$$\min_{P,C} f(P,C) = \sum_{k=1}^{K}\sum_{n=1}^{N} p_{k,n} w_n D(x_n, c_k)$$

where $P \in S_1$ (discrete assignment matrix), and $c_k$ is the cluster center. The Bregman divergence is defined as:

$$D_\phi(x,y) = \phi(x) - \phi(y) - \langle x-y, \nabla\phi(y) \rangle$$

When $\phi(x)=\|x\|_2^2$, it degenerates to the squared Euclidean distance.

### Two Definitions of Local Optimality

- **C-local (Continuous Local Optimality)**: Optimal in the neighborhood of the continuous relaxation domain $S_2$ of $P$.
- **D-local (Discrete Local Optimality)**: Optimal among all adjacent discrete points $T(P)$ of $P$ ($|T(P)|=N(K-1)$). D-local is stronger than C-local.

### Counterexample Construction

The authors provide a 1D counterexample with 5 points and 2 clusters: $x_1=-4, x_2=-2, x_3=0, x_4=1.5, x_5=2.5$, with initial centers $c_1=0, c_2=2.5$. K-means converges to $c_1^*=-2, c_2^*=2$. However, by continuously moving the assignment of $x_3$ from cluster 1 to cluster 2 ($\alpha$ from 0 to 1), the derivative of the clustering loss is:

$$\frac{d}{d\alpha}f = \frac{-20\alpha(\alpha+12)}{(\alpha-3)^2(\alpha+2)^2} < 0$$

which is strictly negative on $(0,1]$, showing that this converged solution is **neither CJ-local, C-local, nor D-local**.

### LO-K-means Algorithm

**Mechanism**: After K-means converges, verify and correct the local optimality as an additional step.

**Function 1 (C-LO)**: Guarantees C-local convergence. After K-means converges, check whether there exist multiple cluster centers equidistant from some point ($|A(C_P^*)|>1$). If so, reassign this point to the cluster with the larger index (symmetry breaking), which is guaranteed to strictly reduce the loss by Theorem 4.3.

**Function 2 (D-LO)**: Guarantees D-local convergence. Traverse all points $n$ and their non-current clusters $k_2$, and calculate the change in loss from reassignment using Lemma 4.1:

$$\Delta_1(g,a,b) = w_g\big(D(x_g, c_b^*) - D(x_g, c_a^*)\big) - \big((s_a - w_g)D(c_a^{new}, c_a^*) + (s_b + w_g)D(c_b^{new}, c_b^*)\big)$$

If $\Delta_1 < 0$, execute the reassignment. The **Min-D-LO** variant selects the $(n, k_2)$ pair that minimizes $\Delta_1$.

### Complexity

- Per-step time complexity: $O(NK\Gamma_\phi(d))$, which is **exactly the same** as vanilla K-means.
- Space complexity: $O((N+K)d)$, which is **exactly the same** as vanilla K-means.

## Key Experimental Results

### Synthetic Data (Figure 1-2)

Across 1,000 runs with varying $N, K, d$, C-LO improves K-means results in a large number of settings. The improvement ratio increases as $K$ increases and dimensionality $d$ decreases, indicating that K-means failing to converge to a C-local optimum is fairly common.

### Main Results on Real Datasets (Table 1, Euclidean Distance)

| Dataset | K | Algorithm | Average Loss | Minimum Loss | Time (s) |
|--------|---|------|----------|----------|---------|
| Iris (N=150,d=4) | 10 | K-means | 31.55 | 26.78 | <0.001 |
| | | D-LO | **30.53** | **26.18** | <0.001 |
| | | Min-D-LO | 30.55 | **26.18** | <0.001 |
| Wine Quality (N=6497,d=11) | 10 | K-means | 1,378,087 | 1,367,222 | 0.01 |
| | | D-LO | **1,377,711** | **1,367,222** | 0.02 |
| News20 (N=2000,d=1089) | 10 | K-means | 919,058 | 734,571 | 0.71 |
| | | D-LO | **637,004** | **625,281** | 19.52 |
| | | Min-D-LO | 642,980 | 637,400 | 6.65 |

### Ablation Study & Key Findings

| Comparison Dimension | Finding |
|----------|------|
| C-LO Improvement Frequency | The larger $K$ and lower $d$, the higher the probability that K-means does not converge to a C-local optimum. |
| D-LO vs K-means | Achieves lower or equivalent average/minimum loss across all datasets. |
| Min-D-LO vs D-LO | Close accuracy, but significantly faster (6.65s vs 19.52s on the News20 dataset). |
| K-means++ Initialization | Yields better results when combined with D-LO++, serving as a complement. |
| Other Bregman Divergences | Equally effective under KL divergence and Itakura-Saito divergence. |
| Comparison with Peng & Xia (2005) | Min-D-LO is faster on high-dimensional data while also guaranteeing D-local. |

## Highlights & Insights

1. **Dispelling a Classic Misconception**: First to rigorously prove that K-means does not necessarily converge to a local optimum, overturning a 40-year consensus.
2. **Minimalist Modification, Plug-and-Play**: Merely adds a verification/correction step after K-means converges, allowing direct integration into existing codebases.
3. **Zero Additional Complexity**: Per-step time and space complexities remain unchanged, with rigorous theoretical guarantees.
4. **Broad Applicability**: Applicable to all Bregman divergences (squared Euclidean, KL, Itakura-Saito, Mahalanobis, etc.) and is not limited to Euclidean distance.
5. **Complementary to K-means++**: Can be stacked to further reduce clustering loss.

## Limitations & Future Work

1. D-LO may require a large number of extra iterations on high-dimensional, large-scale data (News20 requires 870 iterations vs 35 for K-means), although Min-D-LO mitigates this issue.
2. Only considers single-point reassignment (adjacent vertices), without considering stronger definitions of local optimality involving simultaneous multi-point swaps.
3. The empty cluster handling strategy (splitting from the largest cluster) is somewhat heuristic, which may affect the quality of final solutions.
4. Lacks comparison with more complex clustering methods (such as spectral clustering and deep clustering).
5. Lacks a theoretical upper bound on the optimality gap to global optimality.

## Related Work & Insights

- **Selim & Ismail (1984)**: First to attempt to prove K-means local optimality (this work identifies its flaw).
- **Peng & Xia (2005)**: Proposed a D-local optimal algorithm, but the proposed method is easier to integrate and faster in high dimensions.
- **Arthur & Vassilvitskii (2006)**: K-means++ initialization, complementary to LO-K-means.
- **Banerjee et al. (2005)**: K-means with Bregman divergence, claimed local optimality but the proof was incomplete.
- **Insight**: This modification strategy can be extended to other robust variants of K-means (e.g., K-means—removing outliers).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Dispels a 40-year classic misconception, highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetics + real data + multiple divergences, yet lacks comparison with methods like deep clustering.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematically rigorous, clear counterexample, and well-structured.
- Value: ⭐⭐⭐⭐ — Makes significant theoretical contributions to fundamental clustering algorithms with strong practicality, though its impact is constrained to traditional clustering scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](../../AAAI2026/others/approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[ICML 2025\] OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable?](ood-chameleon_is_algorithm_selection_for_ood_generalization_learnable.md)
- [\[ACL 2025\] Theoretical Guarantees for Minimum Bayes Risk Decoding](../../ACL2025/others/theoretical_guarantees_for_minimum_bayes_risk_decoding.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](../../NeurIPS2025/others/sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](../../NeurIPS2025/others/fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)

</div>

<!-- RELATED:END -->
