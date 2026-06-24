---
title: >-
  [Paper Note] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization
description: >-
  [ICML 2025][Optimization][compositional optimization] This paper proposes the ALEXR algorithm—an efficient single-loop primal-dual block-coordinate stochastic algorithm for solving convex finite-sum coupled compositional optimization (cFCCO) problems. It achieves near-optimal convergence rates under both smooth and non-smooth conditions, and proves the optimality of the algorithm by deriving lower bounds.
tags:
  - "ICML 2025"
  - "Optimization"
  - "compositional optimization"
  - "stochastic optimization"
  - "distributionally robust optimization"
  - "pAUC maximization"
  - "primal-dual"
date: 2026-05-08
content_hash: 254a23d3d7c78318
---

# A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization

**Conference**: ICML 2025  
**arXiv**: [2312.02277](https://arxiv.org/abs/2312.02277)  
**Code**: None  
**Area**: Optimization  
**Keywords**: compositional optimization, stochastic optimization, distributionally robust optimization, pAUC maximization, primal-dual

## TL;DR
This paper proposes the ALEXR algorithm—an efficient single-loop primal-dual block-coordinate stochastic algorithm for solving convex finite-sum coupled compositional optimization (cFCCO) problems. It achieves near-optimal convergence rates under both smooth and non-smooth conditions, and proves the optimality of the algorithm by deriving lower bounds.

## Background & Motivation

**Background**: Finite-sum Coupled Compositional Optimization (cFCCO) is an important class of optimization problems, widely appearing in applications such as group distributionally robust optimization (GDRO) and imbalanced data learning. It takes the form of minimizing an objective function that is the sum of finite coupled compositional functions.

**Limitations of Prior Work**: Existing methods typically employ multi-loop structures to handle cFCCO, leading to complex algorithm design and inefficient practical execution. Existing single-loop algorithms suffer from significant gaps in convergence rates, and particularly lack effective solutions under non-smooth conditions.

**Key Challenge**: The coupling structure of outer and inner functions in cFCCO problems makes standard SGD methods directly inapplicable, requiring simultaneous maintenance of primal and dual variable updates. Furthermore, how to obtain optimal convergence rates in the non-smooth setting remains an open problem.

**Goal**: Design a single-loop stochastic algorithm that can handle both smooth and non-smooth cFCCO problems in convex and strongly convex scenarios.

**Key Insight**: Utilize the primal-dual framework to transform the cFCCO problem into a saddle-point problem, and then employ alternating updates with block-coordinate stochastic mirror ascent with extrapolation and proximal gradient descent.

**Core Idea**: ALEXR achieves near-optimal convergence rates without increasing algorithmic complexity through single-loop primal-dual block-coordinate updates and mirror ascent extrapolation techniques.

## Method

### Overall Architecture
The ALEXR algorithm reformulates the cFCCO problem as a minimax problem (primal-dual form), and then alternately updates the primal variable (via stochastic proximal gradient descent) and the dual variable (via block-coordinate stochastic mirror ascent + extrapolation) in each iteration. The entire process has a single-loop structure, requiring only a small number of samples to be drawn at each iteration.

### Key Designs

1. **Block-Coordinate Stochastic Mirror Ascent with Extrapolation**:

    - Function: Update the dual variable $\lambda$, using extrapolation steps to reduce variance.
    - Mechanism: The update of the dual variable employs the Bregman divergence as a regularization term, using the extrapolation step $\hat{\lambda}^{t} = \lambda^t + \eta(\lambda^t - \lambda^{t-1})$ to accelerate convergence.
    - Design Motivation: The dimension of the dual variables in cFCCO is massive (equal to the number of samples). Block-coordinate updates can greatly reduce the computational cost per round, while extrapolation compensates for the variance introduced by stochastic sampling.

2. **Stochastic Proximal Gradient Descent**:

    - Function: Update the primal variable $x$.
    - Mechanism: $x^{t+1} = \text{prox}_{\eta r}(x^t - \eta \hat{g}^t)$, where $\hat{g}^t$ is a stochastic gradient based on the current estimate of the dual variable.
    - Design Motivation: The proximal operator naturally handles non-smooth regularization terms on the primal variables while maintaining a single-loop structure.

3. **Lower Bound Complexity Derivation**:

    - Function: Prove that the convergence rate of ALEXR is near-optimal within a broad class of stochastic algorithms.
    - Mechanism: By constructing a hard instance, the lower complexity bound of $\Omega(1/\sqrt{T})$ (convex case) and $\Omega(1/T)$ (strongly convex case) is established for cFCCO problems.
    - Design Motivation: Providing only the upper bound is insufficient to show the optimality of the algorithm; the lower bound proves that the potential for further improvement is extremely limited.

### Loss & Training
The general form of the cFCCO problem is:
$$\min_{x} F(x) = h\left(\frac{1}{n}\sum_{i=1}^{n} f_i(g_i(x))\right) + r(x)$$
where $h$ is the outer function, $f_i \circ g_i$ is the inner compositional function, and $r$ is the regularization term. ALEXR solves this by introducing a dual variable $\lambda$ to convert it into a minimax problem.

## Key Experimental Results

### Main Results

| Application | Metric | ALEXR | FCSG | Multi-loop | Gain |
|-------------|--------|-------|------|------------|------|
| GDRO (CIFAR-10) | Worst-group Accuracy | **82.3%** | 79.1% | 80.5% | +1.8% |
| GDRO (CelebA) | Worst-group Accuracy | **74.6%** | 71.2% | 72.8% | +1.8% |
| pAUC Maximization | pAUC@0.1 | **0.893** | 0.871 | 0.882 | +1.1% |
| pAUC Maximization | pAUC@0.3 | **0.945** | 0.931 | 0.938 | +0.7% |

### Ablation Study

| Configuration | Convergence Speed | Description |
|---------------|-------------------|-------------|
| With Extrapolation | Fast | Extrapolation steps significantly reduce the number of iterations required to reach target accuracy |
| Without Extrapolation | Slow | Convergence speed degrades significantly after removing extrapolation |
| Full-coordinate Update | Fast but high overhead | Computational cost per round is proportional to the number of samples |
| Block-coordinate Update | Fast with low overhead | Only a small number of dual coordinates are updated per round |

### Key Findings
- ALEXR achieves a convergence rate of $O(1/\sqrt{T})$ in the convex case and $\tilde{O}(1/T)$ in the strongly convex case.
- The derived lower bounds show that these rates are near-optimal within a broad class of stochastic algorithms.
- ALEXR is the first single-loop algorithm capable of handling non-smooth cFCCO problems.
- On GDRO and pAUC maximization tasks, ALEXR outperforms existing methods in both convergence speed and final performance.

## Highlights & Insights
- **Outstanding Theoretical Contribution**: Simultaneously provides upper and lower bounds, establishing a closed complexity characterization.
- **High Practical Utility**: The single-loop structure is simple, easy to implement, and does not require hyperparameter tuning for inner loops.
- **Broad Applicability**: Unifies the treatment of smooth and non-smooth cases, extending the application scope of cFCCO to the dual form of GDRO.

## Limitations & Future Work
- Theoretical analysis focuses primarily on convex settings; convergence guarantees for non-convex cFCCO problems remain to be investigated.
- The experimental scale is relatively small, and it has not been verified on large-scale deep learning tasks.
- The impact of different sampling strategies (uniform vs. importance sampling) for block-coordinate updates has not been explored in depth.

## Related Work & Insights
- FCSG (Hu et al., 2020): The first algorithm to handle cFCCO, but it has a multi-loop structure.
- The primal-dual framework in this paper can inspire algorithm designs for other coupled optimization problems (such as fairness constraints in federated learning).

## Personal Reflections
- The way the primal-dual framework transforms cFCCO into a saddle-point problem for a unified solution is highly worth learning.
- The lower bound construction technique serves as a valuable reference for complexity analysis in other composite optimization problems.
- The dual form of GDRO is non-smooth; this work is the first to provide convergence rates for non-smooth cFCCO.
- Extending ALEXR to distributed/federated settings under GDRO is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Single-loop + near-optimal is a major theoretical breakthrough, but the primal-dual framework itself is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐ Experiments cover major applications, but on a small scale.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theory and clear presentation.
- Value: ⭐⭐⭐⭐ Provides an almost complete complexity landscape for the cFCCO problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](../../NeurIPS2025/optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[ICML 2025\] Enhancing Parallelism in Decentralized Stochastic Convex Optimization](enhancing_parallelism_in_decentralized_stochastic_convex_optimization.md)
- [\[ICML 2025\] Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)

</div>

<!-- RELATED:END -->
