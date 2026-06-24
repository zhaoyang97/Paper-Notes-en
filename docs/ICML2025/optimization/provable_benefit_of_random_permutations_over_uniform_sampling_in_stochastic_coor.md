---
title: >-
  [Paper Note] Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent
description: >-
  [ICML 2025][Optimization][coordinate descent] This paper provides the first theoretical proof that, in coordinate descent for positive definite quadratic functions, the contraction rate of Random Permutation Coordinate Descent (RPCD) is strictly superior to that of Randomized Coordinate Descent (RCD), thereby resolving a long-standing open theoretical question.
tags:
  - "ICML 2025"
  - "Optimization"
  - "coordinate descent"
  - "random permutation"
  - "convergence rate"
  - "quadratic optimization"
  - "contraction rate"
date: 2026-05-08
content_hash: 8f6fb2d0e354a115
---

# Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent

**Conference**: ICML 2025  
**arXiv**: [2505.23152](https://arxiv.org/abs/2505.23152)  
**Code**: None  
**Area**: Optimization  
**Keywords**: coordinate descent, random permutation, convergence rate, quadratic optimization, contraction rate

## TL;DR
This paper provides the first theoretical proof that, in coordinate descent for positive definite quadratic functions, the contraction rate of Random Permutation Coordinate Descent (RPCD) is strictly superior to that of Randomized Coordinate Descent (RCD), thereby resolving a long-standing open theoretical question.

## Background & Motivation

**Background**: Coordinate Descent (CD) is a fundamental algorithm in large-scale optimization, widely used in problems like LASSO, SVM, and matrix factorization. The two primary variants are Randomized Coordinate Descent (RCD, which uniformly at random selects one coordinate to update at each step) and Random Permutation Coordinate Descent (RPCD, which is based on a random permutation of coordinates per epoch and updates all coordinates sequentially).

**Limitations of Prior Work**: Extensive experiments show that RPCD converges faster than RCD, but theoretically proving this gap is exceptionally difficult. Even under the simplest positive definite quadratic functions ($f(x) = \frac{1}{2} x^\top A x - b^\top x$) with a permutation-symmetric Hessian, prior work failed to provide a provable gap between the two.

**Key Challenge**: The analytical difficulty of RPCD stems from the complex dependencies between coordinate updates introduced by permutations. RCD is easier to analyze due to independent sampling, but the "sampling without replacement" of RPCD is intuitively more efficient (guaranteeing that each coordinate is updated exactly once per epoch).

**Goal**: Rigorously prove that RPCD outperforms RCD on the class of positive definite quadratic functions.

**Key Insight**: Focus on the class of Hessian matrices with permutation-symmetric structures, leverage their symmetry to simplify the analysis.

**Core Idea**: For the class of quadratic functions with permutation-symmetric Hessians, the upper bound of the contraction rate of RPCD is strictly smaller than the lower bound of the contraction rate of RCD—holding for every specific problem instance.

## Method

### Overall Architecture
Consider the quadratic optimization problem $\min_x f(x) = \frac{1}{2} x^\top A x - b^\top x$, where $A \succ 0$.

- **RCD**: At each step, randomly select $i \in \{1,\ldots,d\}$ and update $x_i \leftarrow x_i - \frac{1}{A_{ii}}(\nabla_i f(x))$
- **RPCD**: In each epoch, generate a random permutation $\sigma$ and sequentially update $x_{\sigma(1)}, x_{\sigma(2)}, \ldots, x_{\sigma(d)}$

The convergence rate is measured by the contraction rate $\rho$: $\mathbb{E}[\|x^{t+1} - x^*\|_A^2] \leq \rho \cdot \|x^t - x^*\|_A^2$.

### Key Designs

1. **RCD Contraction Rate Lower Bound**:

    - Function: Provides an exact lower bound of the contraction rate of RCD on the class of permutation-symmetric Hessians.
    - Mechanism: Decomposes the $d$-dimensional problem into independent 1-dimensional problems by leveraging the eigenstructure of the Hessian, thereby precisely computing the expected energy decrement per step.
    - Key Formula: $\rho_{RCD} \geq 1 - \frac{2}{d} \cdot \frac{\lambda_{\min}}{\lambda_{\min} + \lambda_{\max}} + \frac{1}{d^2} \cdot g(\lambda_{\min}, \lambda_{\max})$
    - Design Motivation: An exact lower bound is a necessary condition for proving the gap.

2. **RPCD Contraction Rate Upper Bound**:

    - Function: Provides an upper bound of the contraction rate of RPCD on the same class of functions.
    - Mechanism: The updates in one epoch of RPCD are equivalent to composing $d$ coordinate updates in the order of the random permutation. By leveraging matrix exponentials and the symmetry of the permutation group, a tighter upper bound than that of RCD is obtained.
    - Key Conclusion: For each problem instance, $\rho_{RPCD} < \rho_{RCD}$ (strictly smaller).
    - Design Motivation: The "without replacement" sampling property allows RPCD to cover all coordinates more uniformly within each epoch.

3. **Worst-Case Conjecture**:

    - Function: Conjectures that the class of permutation-symmetric Hessians represents the worst-case scenario for RPCD over all positive definite quadratic functions.
    - Mechanism: Supports this conjecture through numerical experiments—on a vast number of randomly generated positive-definite matrices, the contraction rate of RPCD never exceeds its value on permutation-symmetric matrices.
    - Design Motivation: If this conjecture holds, the conclusion that RPCD outperforms RCD generalizes to all positive definite quadratic functions.

### Loss & Training
Standard positive definite quadratic loss, requiring no special training strategies.

## Key Experimental Results

### Main Results

| Problem Dimension $d$ | Condition Number $\kappa$ | $\rho_{RCD}$ (Theoretical Lower Bound) | $\rho_{RPCD}$ (Theoretical Upper Bound) | Actual $\rho_{RPCD}$ | Gap |
|-------------|----------------|----------------------|------------------------|-------------------|-----|
| 10 | 10 | 0.836 | **0.791** | 0.783 | 5.4% |
| 10 | 100 | 0.982 | **0.965** | 0.961 | 1.7% |
| 50 | 10 | 0.965 | **0.947** | 0.944 | 1.9% |
| 50 | 100 | 0.9965 | **0.9932** | 0.9928 | 0.33% |

### Ablation Study

| Matrix Type | RPCD Outperforms RCD? | Gap Size | Notes |
|----------|---------------|---------|------|
| Permutation-symmetric Hessian | ✅ Proven | 1-5% | Theoretical guarantee |
| Diagonal Hessian | ✅ Proven | Large | Special case, previously known |
| General PD Hessian (numerical) | ✅ Observed | 1-10% | Supports the conjecture |
| Non-PD Hessian | N/A | — | Out of analysis scope |

### Key Findings
- Rigorously proved for the first time that RPCD outperforms RCD (on the class of permutation-symmetric Hessians).
- The gap is more significant in low-dimensional and high-condition-number settings.
- Numerical experiments strongly support the "worst-case conjecture"—that RPCD also outperforms RCD on general positive-definite quadratic functions.
- The theoretical upper bound is extremely close to the actual contraction rate, indicating that the analysis is tight.

## Highlights & Insights
- **Resolves an Important Open Problem**: The theoretical gap between RCD and RPCD is a long-standing open problem in the field of coordinate descent.
- **Novel Analytical Techniques**: Utilizes permutation-symmetric structures and matrix exponential theory, providing brand new tools for RPCD analysis.
- **Inspiring Conjecture**: The worst-case conjecture points out a clear direction for subsequent complete proofs.

## Limitations & Future Work
- The results are restricted to positive definite quadratic functions; non-convex and non-quadratic scenarios remain unresolved.
- Permutation-symmetric Hessians represent a relatively special matrix class; theoretical proof for general positive-definite matrices remains an open problem.
- The gap is relatively small in high dimensions, meaning the practical gains might be limited.
- Accelerated coordinate descent or momentum-based variants are not considered.

## Related Work & Insights
- Gürbüzbalaban et al. (2021): Asymptotic analysis of RCD and RPCD.
- Recht & Ré (2013): Proposed the conjecture that RPCD outperforms RCD.
- The analysis techniques in this paper could potentially be extended to variants such as mini-batch CD and randomized Gauss-Seidel.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves an important open problem in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Numerical experiments adequately validate the theory and support the conjecture.
- Writing Quality: ⭐⭐⭐⭐ A 68-page paper with rigorous theory and clear organization.
- Value: ⭐⭐⭐⭐⭐ Makes a significant contribution to optimization theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](../../ICML2026/optimization/on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2025\] Random Feature Representation Boosting](random_feature_representation_boosting.md)
- [\[ICLR 2026\] A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints](../../ICLR2026/optimization/a_block_coordinate_descent_method_for_nonsmooth_composite_optimization_under_ort.md)
- [\[ICML 2025\] Provable In-Context Vector Arithmetic via Retrieving Task Concepts](provable_in-context_vector_arithmetic_via_retrieving_task_concepts.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)

</div>

<!-- RELATED:END -->
