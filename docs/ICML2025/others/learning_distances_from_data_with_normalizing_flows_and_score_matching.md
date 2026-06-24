---
title: >-
  [Paper Note] Learning Distances from Data with Normalizing Flows and Score Matching
description: >-
  [ICML 2025][Fermat Distance] This paper proposes to learn density functions and score functions using normalizing flows and score matching to efficiently compute density-based Fermat distances, addressing the issues of slow convergence and rough paths in high-dimensional spaces associated with traditional graph-based methods.
tags:
  - "ICML 2025"
  - "Fermat Distance"
  - "Normalizing Flows"
  - "Score Matching"
  - "Geodesic"
  - "Density-Based Distance"
date: 2026-05-08
content_hash: a88c4fd3577faf70
---

# Learning Distances from Data with Normalizing Flows and Score Matching

**Conference**: ICML 2025  
**arXiv**: [2407.09297](https://arxiv.org/abs/2407.09297)  
**Code**: [GitHub](https://github.com/vislearn/Fermat-Distance)  
**Area**: Metric Learning / Riemannian Geometry / Density Estimation  
**Keywords**: Fermat Distance, Normalizing Flows, Score Matching, Geodesic, Density-Based Distance  

## TL;DR

This paper proposes to learn density functions and score functions using normalizing flows and score matching to efficiently compute density-based Fermat distances, addressing the issues of slow convergence and rough paths in high-dimensional spaces associated with traditional graph-based methods.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Metric learning is a fundamental task in machine learning. Traditional methods map data to Euclidean space and use Euclidean distance, which is limited by the geometric constraints of Euclidean space. A more flexible approach is to define a Riemannian metric in the data space and solve for the geodesic distance.

**Fermat distance** is a class of density-based distances (DBD). Its core idea is similar to Fermat's principle of least time in optics: defining a conformal metric tensor $g_x(u,v) = \langle u,v\rangle / p(x)^{2\beta}$, such that low-density regions are "stretched" and high-density regions are "compressed". Geodesics naturally travel along high-density regions of the data manifold.

Existing estimators of Fermat distance based on nearest-neighbor graphs face two key issues:

**Extremely slow convergence**: Due to poor accuracy in density estimation, the practical performance is very poor despite theoretical convergence guarantees.

**Rough paths in high dimensions**: Graph-based methods produce paths that are not smooth enough due to the curse of dimensionality in high-dimensional spaces.

## Method

### 1. Dimension-Adaptive Fermat Distance

In a standard Gaussian distribution, if $\beta=1$, high-dimensional data points are far from the origin (with a mean distance of $\sqrt{D}$), leading to extremely curved geodesic behaviors. The authors propose setting $\beta = 1/D$ (inversely proportional to the dimension) to make geodesic behavior across different dimensions consistent and numerically stable.

### 2. Normalizing Flows Density-Weighted Graph

Traditional methods approximate density using powers of Euclidean distances between local neighbors, which yields poor accuracy. This paper uses normalizing flows to learn precise densities, and then approximates the path integral by weighting the edges in a $k$-NN graph with the learned densities:

$$\text{dist}(x_1, x_2) \approx \frac{\|x_1 - x_2\|}{S} \sum_{i=1}^{S} \frac{1}{p_\theta(y_{i-1/2})^\beta}$$

For numerical stability, all computations are performed in the log space.

### 3. Score Matching Relaxation

The reparameterized geodesic equation (with constant Euclidean speed) is:

$$\ddot{\varphi} - \beta(s(\varphi)\cdot\dot{\varphi})\dot{\varphi} + \beta s(\varphi)\|\dot{\varphi}\|^2 = 0$$

where $s(x) = \nabla_x \log p(x)$ is the score function. The relaxation algorithm iteratively updates the intermediate points of the path to satisfy the geodesic equation. Experiments show that directly learning the score with score matching performs better than deriving it from the flow, revealing a trade-off between learning the log-density versus learning its gradient.

### 4. Ground Truth Computation

The authors developed a numerically stable relaxation method (Algorithm 1) to reparameterize paths with constant Euclidean speed. This method can compute exact geodesic distances and geodesics in known density distributions (such as Gaussian mixtures) to serve as an evaluation baseline.

## Key Experimental Results

### Main Results: Convergence Analysis on 2D Datasets

| Method | Convergence Performance |
|------|---------|
| Power-Weighted Shortest Path (PWSPD) | Extremely slow convergence, LPR much higher than 0 |
| Flow Density-Weighted Graph | Convergence rate almost identical to the ground truth density |
| Ground Truth Density-Weighted Graph | Fastest (Reference line) |

Testing 1,000 random paths across five 2D distributions demonstrates that the accuracy of density estimation is indeed the bottleneck of traditional methods.

### High-Dimensional Scaling (Up to 25 Dimensions)

| Method | High-Dimensional Performance |
|------|---------|
| All graph-based methods | Degrade sharply as dimensionality increases |
| Score Relaxation | Maintain robust performance across all dimensions |

### Ablation Study

- **Flow score vs Score matching**: Directly computing derivatives from the flow introduces excessive noise, whereas score matching is more precise.
- **Dimensional scaling**: $\beta = 1/D$ significantly improves numerical stability.

## Highlights & Insights

- For the first time, a ground truth comparative evaluation is provided in the field of Fermat distance.
- Reveals the root cause of the massive gap between theoretical guarantees and practical performance in traditional methods: insufficient accuracy in density estimation.
- Dimension-adaptive scaling of $\beta=1/D$ enables the method to generalize to high dimensions.
- The score relaxation method effectively overcomes the curse of dimensionality.

## Limitations & Future Work

- High-dimensional experiments were only conducted on standard Gaussian distributions, without validation on real-world high-dimensional datasets.
- Score matching and normalizing flows require training themselves, increasing the initial computational overhead.
- For highly complex manifold structures, the expressiveness of the score model might be insufficient.

## Rating

⭐⭐⭐⭐ — Technically sound and elegant method. For the first time, an exact baseline for Fermat distance is established, and practical improvements are proposed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Score Matching with Missing Data](score_matching_with_missing_data.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time](fully_dynamic_euclidean_bi-chromatic_matching_in_sublinear_update_time.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](curvature_enhanced_data_augmentation_for_regression.md)

</div>

<!-- RELATED:END -->
