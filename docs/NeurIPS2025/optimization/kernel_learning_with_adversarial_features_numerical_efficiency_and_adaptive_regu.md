---
title: >-
  [Paper Note] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization
description: >-
  [NeurIPS 2025][Optimization][adversarial training] This paper proposes a new paradigm that relocates adversarial perturbations from the input space to the feature space within reproducing kernel Hilbert spaces (RKHS)…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "adversarial training"
  - "kernel methods"
  - "RKHS"
  - "adaptive regularization"
  - "multiple kernel learning"
date: 2026-05-08
content_hash: a7ed645d271cd727
---

# Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization

**Conference**: NeurIPS 2025
**arXiv**: [2510.20883](https://arxiv.org/abs/2510.20883)
**Code**: [antonior92/adversarial_training_kernel](https://github.com/antonior92/adversarial_training_kernel)
**Area**: Optimization
**Keywords**: adversarial training, kernel methods, RKHS, adaptive regularization, multiple kernel learning

## TL;DR
This paper proposes a new paradigm that relocates adversarial perturbations from the input space to the feature space within reproducing kernel Hilbert spaces (RKHS), enabling exact closed-form solutions to the inner maximization problem. The outer minimization is solved efficiently via iterative reweighted kernel ridge regression, and the resulting adaptive regularization matches cross-validation performance without any hyperparameter tuning.

## Background & Motivation
**Background**: Adversarial training is a central technique for improving model robustness, but classical input-space adversarial training (e.g., PGD) requires solving an intractable min-max problem at significant computational cost.

**Limitations of Prior Work**: (a) The inner maximization under input-space perturbations is non-concave and typically requires multi-step gradient approximations; (b) while the equivalence between adversarial training and parameter shrinkage (ridge/lasso) in linear models has been extensively studied, analogous theory in infinite-dimensional kernel spaces remains undeveloped.

**Key Challenge**: How can adversarial training in kernel methods be made both efficiently solvable and endowed with adaptive regularization properties, while preserving adversarial robustness guarantees?

**Key Insight**: Adversarial perturbations are applied in the RKHS feature space rather than the input space, exploiting the linear structure of the feature space to obtain closed-form solutions.

## Method

### Core Idea: Feature-Space Perturbations

Classical input-space adversarial training solves:

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \max_{\|\Delta x_i\| \le \delta} \ell(y_i, f(x_i + \Delta x_i))$$

This paper relocates perturbations to the feature space $\mathcal{H}$:

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \max_{\|d\|_\mathcal{H} \le \delta} (y_i - \langle f, \phi(x_i) + d \rangle)^2$$

**Key Theorem (Proposition 2, closed-form inner solution)**: For $\Omega_\mathcal{H} = \{d: \|d\|_\mathcal{H} \le \delta\}$,

$$\max_{d \in \Omega_\mathcal{H}} (y - \langle f, \phi(x) + d \rangle)^2 = (|y - f(x)| + \delta\|f\|_\mathcal{H})^2$$

The original problem is thus equivalent to:

$$\min_{f \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n (|y_i - f(x_i)| + \delta \|f\|_\mathcal{H})^2$$

This closely resembles kernel ridge regression, except that the regularization term appears inside rather than outside the squared loss.

### Relaxation Guarantee (Proposition 1)

Feature-space perturbations serve as an upper-bound relaxation of input-space perturbations: for $\Omega_\mathcal{X} = \{\Delta x: D_\mathcal{H}(x, x+\Delta x) \le \delta\}$,

$$\max_{d \in \Omega_\mathcal{H}} (y - \langle f, \phi(x) + d \rangle)^2 \ge \max_{\Delta x \in \Omega_\mathcal{X}} (y - f(x + \Delta x))^2$$

Correspondences between input-space and feature-space perturbation radii are established for Gaussian, Matérn, polynomial, and other kernels.

### Optimization Algorithm: Iterative Reweighted Kernel Ridge Regression (Algorithm 1)

An $\eta$-trick variational reformulation decomposes the loss into a weighted least-squares term plus a regularization term, enabling alternating optimization:

1. **Solve weighted kernel ridge regression**: $\hat{f} = \arg\min_f \frac{1}{n}\sum_i w_i (y_i - f(x_i))^2 + \lambda \|f\|_\mathcal{H}^2$
2. **Update weights**: $w_i = 1/\eta_i^0$, $\lambda = \frac{1}{n}\sum_i \delta^2/\eta_i^1$
3. Here $\eta_i^0, \eta_i^1$ are determined by the ratio of current residuals to norms
4. Repeat until convergence (typically only a few iterations suffice)

The computational complexity matches that of kernel ridge regression, with $O(n^3)$ dominated by kernel matrix decomposition; Nyström approximations are compatible.

### Generalization Bound

**Theorem 1 (Excess risk for adversarial kernel training)**: Let $R = \|f^*\|_\mathcal{H}$, $\sigma$ denote the noise level, and $\delta$ the adversarial radius. The excess risk satisfies $\mathcal{E}(\hat{f} - f^*) \le \min(\mathcal{B}_\gamma^{\rm adv}, \mathcal{B}_\beta^{\rm adv})$, where:

- $\mathcal{B}_\gamma^{\rm adv} = O(\sigma R \gamma)$ (with $\delta \propto \gamma$), and $\gamma$ is the Gaussian complexity
- $\mathcal{B}_\beta^{\rm adv} = O(\sigma^2 \beta^2)$ (with sufficiently small $\delta$), and $\beta$ is the local Gaussian complexity

**Core Advantage**: With the default $\delta$, adversarial training achieves a near-optimal rate adaptively without knowledge of the noise level $\sigma$, whereas kernel ridge regression requires $\lambda \propto \sigma/R$.

For Gaussian noise and translation-invariant kernels, $\gamma = O(1/\sqrt{n})$, yielding $\mathcal{B}_\gamma^{\rm adv} = O(\sigma R / \sqrt{n})$ with no dimension dependence.

### Multiple Kernel Learning Extension

The framework is extended to a multi-kernel space $\bar{\mathcal{H}} = \bigoplus_{j=1}^D \mathcal{H}_j$, where the perturbation set is taken as the intersection of balls in each kernel subspace, leading to the equivalent optimization:

$$\min \frac{1}{n}\sum_{i=1}^n \left(|y - \sum_j f_j(x)| + \delta \sum_j \|f_j\|_{\mathcal{H}_j}\right)^2$$

This form closely mirrors classical multiple kernel learning and is similarly solvable via the iterative reweighted algorithm.

## Key Experimental Results

### Clean Data Performance

| Dataset | Adversarial Kernel Training $R^2$ | Kernel Ridge Regression (CV) $R^2$ |
|--------|:---:|:---:|
| Abalone | **0.57** | 0.57 |
| Average across datasets | ≈ on par or better | Requires CV tuning |

Adversarial kernel training operates over a smaller hyperparameter space (only $\gamma$ needs tuning; $\delta$ uses the default value), yet achieves performance on par with or better than cross-validated kernel ridge regression.

### Adversarial Robustness (Abalone Dataset)

| Training Method | No Attack | $\ell_2 \le 0.01$ | $\ell_2 \le 0.1$ | $\ell_\infty \le 0.01$ | $\ell_\infty \le 0.1$ |
|----------|:---:|:---:|:---:|:---:|:---:|
| Adv Kern ($\delta \propto n^{-1/2}$) | 0.57 | 0.55 | **0.39** | 0.54 | **0.13** |
| Ridge Kernel (CV) | 0.57 | 0.55 | 0.26 | 0.52 | -0.16 |
| Adv Kern ($\|d\| \le 0.01$) | 0.56 | 0.55 | **0.40** | 0.53 | **0.18** |
| Adv Input ($\ell_2 \le 0.1$) | 0.57 | 0.55 | 0.27 | 0.52 | -0.14 |

**Key Findings**: Feature-space adversarial training outperforms direct input-space adversarial training in defending against input-space attacks, with particularly significant advantages under large attack budgets ($\ell_\infty \le 0.1$).

### Adaptive Regularization

Across target functions of varying smoothness and multiple kernels (Matérn-1/2, Matérn-3/2, Gaussian), adversarial kernel training with the default $\delta \propto 1/\sqrt{n}$ exhibits test MSE decay rates that closely match those of cross-validated kernel ridge regression, demonstrating automatic adaptation to the regularity of the target function.

## Highlights & Insights
1. The **closed-form inner solution** renders adversarial training computationally equivalent to standard kernel ridge regression, eliminating the need for multi-step PGD.
2. The **adaptive regularization** property requires no hyperparameter tuning—$\delta$ is independent of the noise level.
3. The generalization bound rigorously establishes the relaxation relationship between feature-space and input-space perturbations, providing theoretical guarantees.
4. The multiple kernel learning extension enables the framework to handle multimodal settings.

## Limitations & Future Work
1. Computational complexity remains $O(n^3)$; large-scale data requires Nyström approximation, which is not elaborated in the paper.
2. Theoretical analysis focuses on fixed-design regression; random-design and classification settings require further investigation.
3. Experiments are limited to moderate-scale UCI datasets; validation on large-scale or high-dimensional data is lacking.
4. No direct comparison with neural network adversarial training is provided.

## Related Work & Insights
- **vs. Linear model adversarial training** (Xing et al. 2021): This paper generalizes linear-model results to infinite-dimensional RKHS.
- **vs. PGD adversarial training** (Madry et al. 2018): The closed-form inner solution avoids multi-step approximations, yielding more efficient optimization.
- **vs. TRADES** (Zhang et al. 2019): TRADES is an approximate trade-off scheme; this paper provides an exact solution within the kernel framework.
- **vs. Robust kernel regression (M-estimators)**: The objectives differ—this paper targets robustness to adversarial perturbations rather than robustness to outliers.

The adaptive regularization property can inspire the design of regularization strategies that require no hyperparameter search. The multi-kernel adversarial learning framework is directly applicable to multimodal learning scenarios. The theory is extensible to the neural tangent kernel (NTK), offering theoretical guidance for adversarial training of deep networks.

## Rating
- ⭐ Novelty: 4/5 — The feature-space perturbation perspective is novel, and the closed-form solution provides a substantial computational advantage.
- ⭐ Experimental Thoroughness: 3/5 — Coverage via synthetic and UCI datasets is adequate, but high-dimensional and large-scale validation is absent.
- ⭐ Writing Quality: 4/5 — Theoretical derivations are clear and the structure is complete.
- ⭐ Value: 4/5 — Theoretical contributions are rigorous and the method is practically useful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](efficient_adaptive_experimentation_with_noncompliance.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)

</div>

<!-- RELATED:END -->
