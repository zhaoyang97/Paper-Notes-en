---
title: >-
  [Paper Note] LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection
description: >-
  [ICML2025][differentiable sorting] The authors propose LapSum, a unified framework for four major differentiable sorting tasks (differentiable ranking, sorting, top-k selection, and permutation matrices) based on a closed-form invertible formula of the sum of cumulative density functions of the Laplace distribution. It operates with a time complexity of only $O(n\log n)$ and $O(n)$ space complexity, significantly outperforming existing methods in large-scale scenarios.
tags:
  - "ICML2025"
  - "differentiable sorting"
  - "differentiable top-k"
  - "soft ranking"
  - "Laplace distribution"
  - "permutation learning"
date: 2026-05-08
content_hash: 43e587ee6d3be200
---

# LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection

**Conference**: ICML2025  
**arXiv**: [2503.06242](https://arxiv.org/abs/2503.06242)  
**Code**: [github.com/gmum/LapSum](https://github.com/gmum/LapSum)  
**Area**: Differentiable Sorting / Differentiable Optimization  
**Keywords**: differentiable sorting, differentiable top-k, soft ranking, Laplace distribution, permutation learning

## TL;DR

The authors propose LapSum, a unified framework for four major differentiable sorting tasks (differentiable ranking, sorting, top-k selection, and permutation matrices) based on a closed-form invertible formula of the sum of cumulative density functions of the Laplace distribution. It operates with a time complexity of only $O(n\log n)$ and $O(n)$ space complexity, significantly outperforming existing methods in large-scale scenarios.

## Background & Motivation

Operations such as sorting, ranking, and top-k selection are widely utilized in tasks like recommendation systems, multi-label classification, and sparse network extraction. However, these operations are fundamentally **piecewise constant functions**, meaning they are non-differentiable and cannot be directly optimized using gradient descent. Existing solutions suffer from the following pain points:

- **Optimal Transport Methods** (Cuturi et al., 2019; Xie et al., 2020): Mathematically elegant but computationally expensive; difficult to apply in large-scale scenarios.
- **Permutation Learning Methods** (NeuralSort, SoftSort, SinkhornSort): Require Sinkhorn iterations or complex matrix operations, lacking closed-form solutions.
- **Smooth Approximation Methods** (Berrada et al., 2018): Trade-off exists between accuracy and speed.
- Most methods have a time complexity of $O(n^2)$ or higher, leading to memory explosion when $n$ and $k$ are large.
- Some methods do not output probability vectors and do not support GPU parallelization.

**Goal**: Design a **theoretically unified, closed-form, and highly parallelizable** framework to solve all differentiable sorting problems under a single umbrella.

## Method

### Core Idea: F-Sum Framework

By choosing an arbitrary even-symmetric positive density function $f$ and its cumulative distribution function $F$, the scale-parameterized F-Sum with parameter $\alpha$ is defined as:

$$F\text{-Sum}_\alpha(r, x) = \sum_{i=0}^{n-1} F_\alpha(x - r_i), \quad F_\alpha(x) = F\left(\frac{x}{\alpha}\right)$$

**Key property**: $F_\alpha$ is strictly monotonic $\rightarrow$ $F\text{-Sum}_\alpha$ is strictly monotonic with respect to $x$ $\rightarrow$ there exists a unique inverse function $F\text{-Sum}^{-1}_\alpha(r, k)$.

Based on this framework, the four major differentiable sorting operations are unified as follows:

| Operation | Definition |
|------|------|
| **Soft Ranking** | $F\text{-Rank}_\alpha(r_j) = F\text{-Sum}_\alpha(r, r_j) - \frac{1}{2}$ |
| **Soft Sorting** | $(F\text{-Sort}_\alpha(r))_l = F\text{-Sum}^{-1}_\alpha(\frac{1}{2} + l)$ |
| **Soft Top-k** | $p_i = F_\alpha(b - r_i)$，where $b = F\text{-Sum}^{-1}_\alpha(r, k)$ |
| **Soft Permutation** | $[F_\alpha(b_{i+1} - r_j) - F_\alpha(b_i - r_j)]_{i,j}$ |

As $\alpha \to 0^+$, all soft operations converge to their corresponding hard operations (sorting, ranking, top-k selection, and permutation matrix).

### Why Laplace Distribution?

Choosing $F$ as the standard Laplace cumulative distribution function (CDF):

$$\text{Lap}(x) = \begin{cases} \frac{1}{2}e^x & x \leq 0 \\ 1 - \frac{1}{2}e^{-x} & x > 0 \end{cases}$$

The piecewise exponential structure of the Laplace CDF allows $\text{Lap-Sum}$ to be expressed on each interval $[r_j, r_{j+1}]$ as:

$$\text{Lap-Sum}(x) = \frac{1}{2}a_j e^{(x-r_{j+1})/\alpha} - \frac{1}{2}b_{j+1}e^{(r_j - x)/\alpha} + c_{j+1}$$

where $a_j, b_j, c_j$ can be computed sequentially in $O(n)$ time (supporting prefix scan parallelization). The inverse function also has a **closed-form solution**, eliminating the need for iterative solvers.

### Gradient Computation

The derivative matrix of Top-k is $D = \frac{\partial P}{\partial r} = s\,q^T - \text{diag}(s)$, where:
- $s_i = \frac{1}{\alpha}\min(p_i, 1-p_i)$
- $q = \text{softmax}(-|b-r_i|/\alpha)$

Direct computation is $O(n^2)$, but leveraging the matrix-vector multiplication $Dv = \langle q,v\rangle s - s \odot v$ allows gradient propagation in just $O(n)$. The overall time complexity (including preprocessing sorting) is $O(n\log n)$.

## Key Experimental Results

### Top-k Classification (CIFAR-100, ResNet18)

| Method | ACC@1 | ACC@5 |
|------|-------|-------|
| SinkhornSort | 61.89 | 86.94 |
| DiffSortNets | 62.00 | 86.73 |
| **Lap-Top-k** | **64.53** | **88.51** |

### ImageNet-21K-P Fine-tuning (ResNeXt-101)

| Method | ACC@1 | ACC@5 |
|------|-------|-------|
| DiffSortNets | 40.22 | 70.88 |
| **Lap-Top-k** | **40.48** | **71.05** |

### kNN Image Classification

| Method | MNIST | CIFAR-10 |
|------|-------|----------|
| kNN+NeuralSort | 99.5 | 90.7 |
| kNN+SOFT Top-k | 99.4 | **92.6** |
| **kNN+Lap-Top-k** | **99.4** | 92.2 |

### Large-MNIST Permutation Learning Accuracy (%)

| Method | n=3 | n=5 | n=7 |
|------|-----|-----|-----|
| DNeural | 93.0 | 83.7 | 73.8 |
| SNeural | 92.7 | 83.5 | 74.1 |
| **LapSum** | **94.2** | **85.3** | 74.1 |

### Time/Memory Complexity

Under the setting of $k=n/2$, both the memory footprint and runtime of LapSum are superior to or on par with all baseline methods, especially exhibiting a significant advantage in high-dimensional (large $n$) settings. Critical difference statistical tests confirm that LapSum ranks first overall.

## Highlights & Insights

1. **Strong Theoretical Unification**: A single F-Sum framework elegantly and concisely unifies the derivation of four major operations: ranking, sorting, top-k, and permutation.
2. **Closed-form Solutions and Gradients**: Free of Sinkhorn iterations and LP solvers, implemented in just 26 lines of pseudocode.
3. **$O(n\log n)$ Time & $O(n)$ Space**: Matches the complexity of the sorting operation itself, achieving theoretical near-optimality.
4. **Parallelization Support**: The recursive sequences $a_j, b_j$ can be parallelized using prefix scans, with both CPU and CUDA implementations provided.
5. **Trainable $\alpha$**: The scale parameter can act as a learnable parameter to automatically adapt between hard and soft behavior.
6. **Guaranteed Probability Vectors**: The property $\sum_i p_i = k$ strictly holds, outperforming some alternative methods that fail to guarantee probability normalization.

## Limitations & Future Work

1. **Degradation in Permutation Learning for Large $n$**: For $n=9, 15$, LapSum is outperformed by NeuralSort, likely because the rapid tail-decay of the Laplace distribution makes the soft permutation matrix less distinguishable in higher dimensions.
2. **Sensitivity to $\alpha$**: Experiments demonstrate that the value of $\alpha$ heavily influences outcomes, requiring meticulous hyperparameter tuning or learning.
3. **Evaluation Limited to Classification**: The performance has not yet been validated in practical downstream application scenarios such as recommendation systems, NLP ranking, or information retrieval.
4. **Under-discussed Relationship with Sampling Methods**: The integration of LapSum in scenarios requiring discrete sampling (e.g., Gumbel-Softmax) remains unexplored.
5. **Numerical Stability**: Although the paper claims numerical stability, its practical behavior under extreme values of $\alpha$ or exceptionally large $n$ requires further verification.

## Related Work & Insights

- **NeuralSort** (Grover et al., 2019): Differentiable permutation based on the Gumbel distribution. LapSum is more efficient by utilizing the Laplace distribution.
- **DiffSortNets** (Petersen et al., 2022): Differentiable sorting utilizing sorting networks, but suffers from higher complexity.
- **Fast Differentiable Sorting** (Blondel et al., 2020): Based on regularized optimal transport; does not output probabilities.
- **Optimal Transport Top-k** (Xie et al., 2020): Differentiable top-k based on the OT framework, which suffers from computational bottlenecks at scale.
- *Insight*: Selecting an appropriate probability distribution (such as the piecewise exponential structure of Laplace) can bring massive computational payloads down. This methodology can be extended to other scenarios demanding differentiable discrete operations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of inverting the Laplace CDF is highly novel, and the unified framework is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers top-k, permutations, and kNN, but the target application scenarios remain somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations and comprehensive experimental comparisons.
- Value: ⭐⭐⭐⭐⭐ — Provides out-of-the-box CPU/CUDA implementations, promising to become a standard tool in differentiable sorting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](../../ICCV2025/others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)
- [\[ICML 2025\] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)
- [\[ICML 2025\] K²IE: Kernel Method-based Kernel Intensity Estimators for Inhomogeneous Poisson Processes](k2ie_kernel_method-based_kernel_intensity_estimators_for_inhomogeneous_poisson_p.md)
- [\[ICML 2025\] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)

</div>

<!-- RELATED:END -->
