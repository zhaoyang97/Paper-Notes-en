---
title: >-
  [Paper Note] Perturbation Bounds for Low-Rank Inverse Approximations under Noise
description: >-
  [NeurIPS 2025][Optimization][Low-Rank Approximation] First non-asymptotic spectral norm perturbation bound for low-rank inverse approximation $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ under additive noise, derived via contour integration with sharp bounds depending on eigengap, spectral decay, and noise alignment — improving classical full-inverse bounds by up to $\sqrt{n}$.
tags:
  - NeurIPS 2025
  - Optimization
  - Low-Rank Approximation
  - Pseudoinverse
  - Perturbation Bound
  - Spectral Norm
  - Eigengap
date: 2026-05-08
content_hash: 27fef7dbcc9efd93
---
## TL;DR
This paper provides the first non-asymptotic spectral norm perturbation bound for low-rank inverse approximations $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ under additive noise. Using contour integration techniques, it derives sharp bounds depending on the eigenvalue gap, spectral decay, and noise alignment, improving upon classical full-inverse bounds by up to a factor of $\sqrt{n}$.

## Background & Motivation
**Background**: Low-rank pseudoinverses are widely used to accelerate inverse matrix approximations in ML, optimization, and scientific computing. Classical perturbation theory provides bounds only for the full inverse $\|A^{-1} - \tilde{A}^{-1}\|$.

**Key Challenge**: Classical bounds ignore the structure introduced by truncation to rank $p$, overlook the interaction between noise and eigenvalue gaps, and frequently yield overly pessimistic estimates.

**Core Idea**: Contour integration is applied to perform a localized resolvent expansion for the non-entire function $f(z) = 1/z$, precisely controlling the perturbation of the Riesz projection near the smallest eigenvalue.

## Method

### Problem Formulation
Given an $n \times n$ real symmetric positive definite matrix $A$ with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n > 0$ and a symmetric perturbation $E$, let $\tilde{A} = A + E$. The central quantity of interest is $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$, where $A_p^{-1} = \sum_{i=n-p+1}^{n} \lambda_i^{-1} u_i u_i^\top$ is the best rank-$p$ approximation to $A^{-1}$.

### Key Results
1. **Main Theorem (Theorem 2.1)**: Under the condition $4\|E\| \leq \min\{\lambda_n, \delta_{n-p}\}$:
$$\|(\tilde{A}^{-1})_p - A_p^{-1}\| \leq 5\left(\frac{\|E\|}{\lambda_n^2} + \frac{\|E\|}{\lambda_{n-p} \delta_{n-p}}\right)$$
   where $\delta_{n-p} = \lambda_{n-p} - \lambda_{n-p+1}$ denotes the eigenvalue gap.

2. **Comparison with Classical Bounds**: The classical EYM-N bound takes the form $\frac{8\|E\|}{3\lambda_n^2} + \frac{2}{\lambda_{n-p}}$, whose second term does not vanish as $\|E\| \to 0$—yielding nonzero error even in the noiseless case. The proposed bound correctly tends to zero as $\|E\| \to 0$.

### Technical Contributions
1. **Extension of Contour Integration**: Extended from entire functions (e.g., the matrix exponential $e^z$) to the non-entire function $f(z) = 1/z$.
2. **Localized Resolvent Expansion**: A Cauchy contour is constructed around the smallest eigenvalue, exploiting local spectral structure.
3. **Riesz Projection Perturbation Control**: Precise control over the effect of noise on eigenspace projections associated with low-curvature subspaces.
4. **Extension to General Symmetric Matrices**: Covers rank-deficient cases via pseudoinverses.

### Application: Improved PCG Convergence Rates
The proposed bounds yield tighter condition number estimates for PCG methods with low-rank-plus-regularization preconditioners, provably improving iteration count guarantees by a factor of $n^{1/4}$.

## Key Experimental Results

### Main Results

| Matrix Type | Tracking Error (Ours) | Tracking Error (Classical) |
|-------------|----------------------|---------------------------|
| Sample covariance | Small constant factor | 1–2 orders of magnitude overestimated |
| Discretized elliptic operator | Small constant factor | 1–2 orders of magnitude overestimated |
| BCSSTK09 stiffness | Small constant factor | 1–2 orders of magnitude overestimated |

### Noise Condition Feasibility
Validated on the 1990 US Census covariance matrix and the BCSSTK09 stiffness matrix; safety margins comfortably exceed typical noise levels encountered in differential privacy and structural engineering applications.

## Highlights & Insights
- Low-rank inverse approximations can be more robust than full inverses when the eigenvalue gap is sufficiently large—a counterintuitive finding, as truncation is typically assumed to introduce additional error.
- Spectrally aware perturbation bounds are far more informative in practice than worst-case bounds, as they exploit the spectral structure of the matrix.
- PCG convergence rates can be improved by a factor of $n^{1/4}$ using the proposed bounds, yielding direct algorithmic benefits.
- Extending contour integration from entire to non-entire functions constitutes a non-trivial mathematical contribution.

## Limitations & Future Work
- The noise condition $4\|E\| < \min\{\lambda_n, \delta_{n-p}\}$ may be restrictive in certain settings.
- Only the spectral norm is considered; Frobenius norm bounds remain to be developed.
- Structured noise (e.g., low-rank perturbations) may admit tighter bounds.

## Related Work & Insights
- **vs. Classical Neumann Series Bounds**: Classical bounds do not account for low-rank truncation, and the second term $2/\lambda_{n-p}$ does not vanish as $\|E\| \to 0$.
- **vs. Schatten Norm Results**: Existing work does not provide spectral norm guarantees.
- **vs. Sherman–Morrison–Woodbury**: Applicable only to structured low-rank updates, not to general additive noise.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First non-asymptotic spectral norm bound for low-rank inverse perturbations, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on diverse real and synthetic matrices with high tracking accuracy.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and fluent exposition with clearly structured proofs.
- Value: ⭐⭐⭐⭐⭐ Foundational contribution to numerical computing and ML optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[NeurIPS 2025\] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs](learning_sparse_approximate_inverse_preconditioners_for_conjugate_gradient_solve.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)

</div>

<!-- RELATED:END -->
