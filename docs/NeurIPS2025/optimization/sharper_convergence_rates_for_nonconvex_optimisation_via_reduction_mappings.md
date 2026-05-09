---
title: >-
  [Paper Note] Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings
description: >-
  [NeurIPS 2025 (**Spotlight**)][Optimization][nonconvex optimization] This paper proposes the Reduction Mapping framework, which exploits the manifold structure of the optimal solution set (arising from over-parameterization or symmetry) to reparameterize the optimization problem, and proves that this improves curvature properties and theoretically accelerates the convergence of gradient-based methods.
tags:
  - "NeurIPS 2025 (**Spotlight**)"
  - Optimization
  - nonconvex optimization
  - reduction mapping
  - convergence rate
  - manifold structure
  - over-parameterization
date: 2026-05-08
content_hash: 247d9ea01ca09e94
---

# Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings

**Conference**: NeurIPS 2025 (**Spotlight**)  
**arXiv**: [2506.08428](https://arxiv.org/abs/2506.08428)  
**Code**: None  
**Area**: Optimization  
**Keywords**: nonconvex optimization, reduction mapping, convergence rate, manifold structure, over-parameterization

## TL;DR
This paper proposes the Reduction Mapping framework, which exploits the manifold structure of the optimal solution set (arising from over-parameterization or symmetry) to reparameterize the optimization problem, and proves that this improves curvature properties and theoretically accelerates the convergence of gradient-based methods.

## Background & Motivation
**Background**: In high-dimensional nonconvex optimization problems such as deep learning training, the set of minimizers typically forms a smooth manifold, arising from model over-parameterization or symmetry.

**Limitations of Prior Work**: Standard nonconvex optimization analyses focus on general convergence guarantees and overlook the acceleration potential offered by solution structure.

**Key Challenge**: Algorithms that exploit structural information empirically converge faster in practice, yet a unified theoretical explanation remains lacking.

**Key Insight**: When the geometry of the optimal solution set is known, at least locally, a reduction mapping can reparameterize the problem to eliminate redundant directions.

## Method

### Overall Architecture
Consider the optimization problem $\min_x f(x)$, whose optimal solution set $\mathcal{M}^*$ forms a smooth manifold. A reduction mapping $\phi: \mathbb{R}^d \to \mathbb{R}^d$ maps a portion of the parameter space onto the solution manifold, yielding a reduced objective $\tilde{f}(z) = f(\phi(z))$.

### Key Designs
1. **Definition of Reduction Mapping**

    - Function: The reparameterization ensures that part of the optimization variables automatically lies on the solution manifold.
    - Formal definition: $\phi$ satisfies $\phi(z^*) \in \mathcal{M}^*$ for the desired $z^*$.
    - Origin: Arises naturally from inner optimization problems (e.g., a single step of alternating optimization).
    - Design Motivation: Eliminate degeneracy along the normal directions of the solution manifold.

2. **Curvature Improvement Theorem**

    - Core Idea: A well-designed reduction mapping improves the curvature of the objective function near the solution.
    - Specifically: The smallest positive eigenvalue of the Hessian of the reduced objective $\tilde{f}$ at the optimum is larger.
    - Formally: $\lambda_{\min}^+(\nabla^2 \tilde{f}) \geq \lambda_{\min}^+(\nabla^2 f)$, with strict inequality in most cases.
    - Significance: Better condition number $\to$ faster convergence.

3. **Convergence Acceleration Analysis**

    - Gradient descent achieves superior convergence rates on the reduced objective.
    - The unified framework subsumes alternating minimization, block coordinate descent, and reparameterization techniques.
    - Explicit bounds on the reduction in iteration count are provided.

### Theoretical Results
- **Theorem 1 (Curvature Improvement)**: If $\phi$ is a proper reduction mapping, then $\kappa(\tilde{f}) \leq \kappa(f)$ (the condition number does not increase).
- **Theorem 2 (Convergence Acceleration)**: The number of iterations for GD to reach an $\epsilon$-critical point on $\tilde{f}$ is $\mathcal{O}(\kappa(\tilde{f})/\epsilon^2)$, compared to $\mathcal{O}(\kappa(f)/\epsilon^2)$ for the original problem.

## Key Experimental Results

### Experiment 1: Matrix Factorization

| Method | Condition Number | Iterations to $\epsilon=10^{-6}$ |
|--------|-----------------|----------------------------------|
| Standard GD | 245 | 12,340 |
| GD + Reduction | **38** | **1,890** |
| Adam | 245 | 5,670 |
| Adam + Reduction | **38** | **980** |

### Experiment 2: Neural Network Training (Over-parameterized 2-Layer Network)

| Method | Iterations to Convergence | Final Loss |
|--------|--------------------------|------------|
| SGD | 5,000 | 1.2e-4 |
| SGD + Block Reduction | **2,100** | **8.5e-5** |
| Adam | 3,200 | 9.8e-5 |
| Adam + Block Reduction | **1,400** | **7.1e-5** |

### Experiment 3: Comparison of Reduction Strategies

| Reduction Type | Effective Condition Number | Speedup |
|----------------|---------------------------|---------|
| No Reduction (baseline) | 245 | 1× |
| Alternating Minimization | 82 | 2.9× |
| Orthogonal Reduction | 45 | 5.2× |
| **Optimal Reduction** | **38** | **6.5×** |

### Key Findings
- Condition number improvements are consistent with theoretical predictions.
- Acceleration is more pronounced under severe over-parameterization.
- The framework provides a unified explanation for the empirical speedups of alternating minimization and reparameterization techniques.
- 12 experimental figures clearly validate the theory.

## Highlights & Insights
- **Spotlight Paper**: The theoretical contribution is recognized as important and technically refined.
- **Unified Framework**: Disparate optimization techniques are subsumed under a single theoretical framework.
- **Practical Guidance**: The work clarifies when and how solution structure can be exploited to accelerate optimization.
- A 43-page comprehensive paper with complete proofs is provided.

## Limitations & Future Work
- The approach requires the solution manifold structure to be known, at least locally; methods for automatically discovering such structure remain to be investigated.
- The analysis for stochastic optimization (SGD) is less complete than for the deterministic setting.
- Precise characterization of the solution manifold in deep networks remains an open problem.
- The interaction effects with adaptive learning rate methods require further investigation.

## Related Work & Insights
- Over-parameterization theory (Allen-Zhu et al. 2019, Du et al. 2019)
- Manifold optimization (Absil et al. 2008)
- Implicit regularization and solution structure
- Insight: Combining NTK theory to analyze reduction mappings in deep networks is a promising direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified framework explaining the acceleration of classical techniques
- Experimental Thoroughness: ⭐⭐⭐⭐ Theory validated with 12 figures
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically clear with complete proofs
- Value: ⭐⭐⭐⭐⭐ Spotlight; foundational contribution

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Personalized Collaborative Learning with Affinity-Based Variance Reduction](../../ICLR2026/optimization/personalized_collaborative_learning_with_affinity-based_variance_reduction.md)

<!-- RELATED:END -->
