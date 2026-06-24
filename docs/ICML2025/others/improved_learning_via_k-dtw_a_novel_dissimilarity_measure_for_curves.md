---
title: >-
  [Paper Note] Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves
description: >-
  [ICML2025][Dynamic Time Warping] Proposes $k$-DTW—a novel dissimilarity measure for polygonal curves that focuses only on the **sum of the $k$ largest distances** in a traversal, combining the robustness of DTW with the metric properties of the Fréchet distance, while proving for the first time a **dimension-free** learning bound for curve clustering.
tags:
  - "ICML2025"
  - "Dynamic Time Warping"
  - "Curve Similarity"
  - "Fréchet Distance"
  - "Learning Theory"
  - "Rademacher Complexity"
date: 2026-05-08
content_hash: 28409ad2d5d2873c
---

# Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves

**Conference**: ICML2025  
**arXiv**: [2505.23431](https://arxiv.org/abs/2505.23431)  
**Code**: [akrivosija/kDTW](https://github.com/akrivosija/kDTW)  
**Area**: Others/Distance Metrics  
**Keywords**: Dynamic Time Warping, Curve Similarity, Fréchet Distance, Learning Theory, Rademacher Complexity

## TL;DR

Proposes $k$-DTW—a novel dissimilarity measure for polygonal curves that focuses only on the **sum of the $k$ largest distances** in a traversal, combining the robustness of DTW with the metric properties of the Fréchet distance, while proving for the first time a **dimension-free** learning bound for curve clustering.

## Background & Motivation

Data such as handwriting recognition, time series, and sensor trajectories are essentially **polygonal curves** in Euclidean space. Comparing two curves requires a suitable dissimilarity measure, and mainstream measures have their respective drawbacks:

- **Fréchet distance**: Satisfies metric axioms (triangle inequality) but is **extremely sensitive to outlier vertices**—a single outlier can dominate the overall distance.
- **DTW distance**: Robust to outliers (summation instead of taking the maximum), but **violates the triangle inequality** with a violation factor up to $\Theta(m)$ (where $m$ is the curve complexity), severely affecting clustering and classification.

The computational complexity for both is near-quadratic ($O(m'^{} m''^{})$), and they cannot be computed in strongly sub-quadratic time under the SETH assumption. Core problem: **Can we design a measure that simultaneously possesses the metric properties of Fréchet and the robustness of DTW?**

## Method

### Core Definition: $k$-DTW

Given two curves $\sigma=(v_1,\dots,v_{m'})$ and $\tau=(w_1,\dots,w_{m''})$, $k$-DTW is defined as:

$$d_{k\text{-DTW}}(\sigma,\tau) = \min_{T\in\mathcal{T}} \sum_{l=1}^{k} s_l^{(T)}$$

where $\mathcal{T}$ is the set of all traversals, and $s_l^{(T)}$ is the **$l$-th largest matching distance** in traversal $T$. Key insights:

- When $k=1$, it degenerates to the **discrete Fréchet distance** (taking the maximum matching distance).
- When $k \geq 2m-1$, it degenerates to **standard DTW** (summing all matching distances).
- Inspired by the **Ky-Fan norm** of a matrix (the sum of the $k$ largest singular values).

### Enhanced Triangle Inequality

$k$-DTW satisfies a relaxed triangle inequality, where the relaxation factor is only $k$ instead of $m$:

$$d_{k\text{-DTW}}(\sigma,\tau) \leq k \cdot \bigl(d_{k\text{-DTW}}(\sigma,\upsilon) + d_{k\text{-DTW}}(\upsilon,\tau)\bigr)$$

This bound is **tight**. When $k \ll m$, this is significantly better than the $m^{1/q}$ factor of DTW.

### Robustness Analysis

The paper introduces the concept of the **breakdown point** of curve distances: for the curve median $t_m(\pi)$ under $k$-DTW, the breakdown point is $\beta(t_m, \pi) = \lfloor(k+1)/2\rfloor$. This implies that at least $\lfloor(k+1)/2\rfloor$ vertices need to be corrupted to make the median diverge—the larger $k$ is, the more robust it is.

### Exact Algorithm (Algorithm 1)

It is impossible to directly compute $k$-DTW using standard DTW dynamic programming (since it is proved that the optimal traversal can be completely different). The algorithm employs a **parametric search**:

1. Construct the distance matrix $D[i,j] = \|v_i - w_j\|$
2. Sort all unique distance values as $E[1] > \cdots > E[z] > 0$
3. For each guess value $E[l]$ (candidate for the $k$-th largest distance):
    - Construct the modified matrix $D'[i,j] = \max\{D[i,j] - E[l], 0\}$
    - Run standard DTW on $D'$, and add $k \cdot E[l]$
4. Take the minimum cost over all iterations

**Time Complexity**: $O(m' m'' z)$, where $z$ is the number of distinct distances (at worst $O(m'm'')$). Two heuristic speedups can reduce the standard DTW calls by 85%–97.5%.

### $(1+\varepsilon)$-Approximation Algorithm

By utilizing the Fréchet distance as an initial $k$-approximation estimate ($d_{dF} \leq d_{k\text{-DTW}} \leq k \cdot d_{dF}$), the number of distinct distances is reduced to a logarithmic scale via rounding:

$$O\!\left(m' m'' \cdot \frac{\log(k/\varepsilon)}{\varepsilon}\right)$$

### Learning Theory: Dimension-Free Generalization Bound

Core theoretical contribution: proving the first **dimension-free** learning bound for curve clustering. Utilizing chaining techniques and dimension reduction via terminal embeddings:

| Metric | Rademacher/Gaussian Complexity Upper Bound |
|------|-------------------------------|
| DTW | $O\!\bigl(\sqrt{m^3 \cdot \min\{d\log m,\, m^2\log^4(mn)\}/n}\bigr)$ |
| $k$-DTW | $O\!\bigl(\sqrt{m k^2 \cdot \min\{d\log k,\, k^2\log^4(mn)\}/n}\bigr)$ |

The lower bound for DTW is $\Omega(\sqrt{m^2/n})$, thus when $k \ll m$, the complexity of $k$-DTW is $\tilde{O}(\sqrt{m/n})$, which is $\tilde{\Omega}(\sqrt{m})$ times lower than that of DTW—this is a **strict complexity separation**.

## Key Experimental Results

### Synthetic Data: Hierarchical Clustering

Constructing 60 curves (20 for each of the 3 classes), using hierarchical agglomerative clustering (HAC):

| Method | Single Linkage | Complete Linkage |
|------|---------------|-----------------|
| DTW | Cannot distinguish classes $A_l$ and $C$ | Similar difficulty |
| Fréchet | Intra-class distance of $A_l$ is too large | Similar difficulty |
| $k$-DTW | **Clearly identifies three classes**, small intra-class distance | **Clearly identifies three classes** |

### Real Data: OULAD $l$-NN Classification

Dataset: 275 student clickstream curves ($m=294$), binary classification (Pass/Fail), $l=17$-NN, 100 repetitions of 6-fold cross-validation:

| Distance Measure | AUC | Accuracy | $F_1$ |
|---------|-----|----------|-------|
| Fréchet | 0.737 | 0.837 | 0.910 |
| 64-DTW | 0.788 | 0.856 | 0.918 |
| **72-DTW** | **0.796** | **0.855** | **0.918** |
| 76-DTW | 0.798 | 0.855 | 0.917 |
| DTW | 0.784 | 0.855 | 0.917 |

- $k$-DTW improves AUC by up to **8.2%** compared to Fréchet, and by **1.8%** compared to DTW.
- The optimal $k$ is roughly 20%–25% of the curve complexity.
- In practice, testing only a few values of $k \in \{\ln m, \sqrt{m}, m/10, m/4\}$ is sufficient.

### Comparison with Other Baselines

Compared with weak discrete Fréchet, ERP (edit distance with real penalty), and two partial DTW variants, $k$-DTW performs the best or near-optimal in most scenarios, whereas the worst results are typically produced by DTW or Fréchet.

## Highlights & Insights

- **Theoretical Elegance**: $k$-DTW naturally interpolates between Fréchet ($k=1$) and DTW (large $k$) via a single parameter $k$, with the design intuition inspired by the Ky-Fan norm.
- **First Dimension-Free Generalization Bound**: Previous bounds based on the VC-dimension inevitably dependent on the dimension $d$.
- **Strict Complexity Separation**: Proves that the Rademacher complexity of $k$-DTW is **strictly lower** than that of DTW by a factor of $\tilde{\Omega}(\sqrt{m})$.
- **Clever Algorithm**: Converts $k$-DTW into multiple calls of standard DTW through parametric search, inspired by combinatorial optimization.
- **High Practicality**: Heuristic speedups reduce computation by 85%–97.5%; the choice of $k$ does not require fine-tuning.

## Limitations & Future Work

- **Computational Cost**: The exact algorithm is at worst $O(m^4)$, and the approximation algorithm is also slower than standard DTW by a factor of $O(\log(k/\varepsilon)/\varepsilon)$.
- **Selection of $k$**: Although a few candidate values are empirically sufficient, there is a lack of theoretical guidance for adaptively selecting $k$.
- **Limited Experimental Scale**: The real dataset contains only 275 curves, which has not been validated on large-scale time series benchmarks.
- **Discrete Case Only**: Has not been extended to $k$-DTW variants for continuous curves (such as CDTW).
- **Improving the top-$k$ optimization framework** is an important open problem—its resolution would directly lead to faster $k$-DTW algorithms.

## Related Work & Insights

- **Fréchet Distance** (Alt & Godau 1995) and **DTW** (Vintsyuk 1968; Berndt & Clifford 1994): Two classical curve distances.
- **top-$k$ Optimization** (Bertsimas & Sim 2003): The inspiration source of Algorithm 1.
- **Ky-Fan Norm** (Fan 1951): The matrix theory background of the $k$-DTW definition.
- **Curve VC Dimension** (Driemel et al. 2021; Conradi et al. 2024): Previous Fréchet/DTW learning bounds all rely on the VC dimension.
- **Terminal Embeddings** (Narayanan & Nelson 2019): Dimension reduction technique used to prove the dimension-free bounds.
- Curve clustering NP-hardness (Driemel et al. 2016; Buchin et al. 2019/2020).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Proposes a new metric in the field of classical curve distance, bringing theoretical breakthroughs.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments validated the theory, but the dataset scale and diversity are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical parts with clear intuitive explanations.
- Value: ⭐⭐⭐⭐ — Possesses important theoretical and practical significance for the field of time series/curve analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](../../NeurIPS2025/others/scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](../../ACL2025/others/a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](../../ICLR2026/others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)

</div>

<!-- RELATED:END -->
