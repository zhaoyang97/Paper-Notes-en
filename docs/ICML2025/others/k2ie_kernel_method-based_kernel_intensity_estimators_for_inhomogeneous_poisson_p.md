---
title: >-
  [Paper Note] K²IE: Kernel Method-based Kernel Intensity Estimators for Inhomogeneous Poisson Processes
description: >-
  [ICML2025][Poisson processes] K²IE is proposed as an RKHS least-squares regularized kernel intensity estimator. By proving that the dual coefficients in its representer theorem are identically 1, the authors theoretically unify classical kernel intensity estimation (KIE) with modern kernel methods, combining the computational efficiency of KIE with the boundary correction advantages of kernel methods.
tags:
  - "ICML2025"
  - "Poisson processes"
  - "kernel intensity estimation"
  - "reproducing kernel Hilbert space"
  - "least squares loss"
  - "equivalent kernel"
date: 2026-05-08
content_hash: e12f8c824d56592e
---

# K²IE: Kernel Method-based Kernel Intensity Estimators for Inhomogeneous Poisson Processes

**Conference**: ICML2025  
**arXiv**: [2505.24704](https://arxiv.org/abs/2505.24704)  
**Code**: [HidKim/K2IE](https://github.com/HidKim/K2IE)  
**Area**: Kernel Methods / Point Processes  
**Keywords**: Poisson processes, kernel intensity estimation, reproducing kernel Hilbert space, least squares loss, equivalent kernel

## TL;DR

K²IE is proposed as an RKHS least-squares regularized kernel intensity estimator. By proving that the dual coefficients in its representer theorem are identically 1, the authors theoretically unify classical kernel intensity estimation (KIE) with modern kernel methods, combining the computational efficiency of KIE with the boundary correction advantages of kernel methods.

## Background & Motivation

**Intensity function estimation** of inhomogeneous Poisson processes is a core task in point process modeling, widely applied in fields such as seismology, epidemiology, and reliability engineering. Existing non-parametric schemes can be categorized into two paradigms, each with its limitations:

**Classical Kernel Intensity Estimation (KIE)**: Represents the intensity function as a sum of smoothing kernels. It is computationally efficient and theoretically transparent, but suffers from limited boundary correction performance in high-dimensional or bounded domain settings, and its cross-validation relies on Monte Carlo integration.

**Flaxman's Kernel Method Estimator (FIE)**: Models the intensity function as the square of an RKHS function, $\lambda(\mathbf{x})=f^2(\mathbf{x})$. While it naturally handles boundary effects using equivalent RKHS kernels, it requires gradient descent to fit the dual coefficients $\boldsymbol{\alpha}$, incurring a computational cost of $\mathcal{O}(qMN)$.

Although both share the word "kernel", they are established on different theoretical foundations. The core motivation of this paper is: **Is it possible to find an estimator that possesses the computational efficiency of KIE (without needing to optimize dual coefficients) while inheriting the adaptive capability of kernel methods for boundary effects?**

## Method

### Core Idea: RKHS Regularization Based on Least Squares Loss

The authors introduce the least squares loss for Poisson processes (derived from the empirical risk minimization principle):

$$-2\sum_{n=1}^{N}\lambda(\mathbf{x}_n) + \int_{\mathcal{X}}\lambda(\mathbf{x})^2\,d\mathbf{x}$$

Instead of constraining $\sqrt{\lambda}$ as in FIE, the intensity function $\lambda$ is directly constrained to the RKHS $\mathcal{H}_k$, formulating the regularized optimization problem:

$$\min_{\lambda\in\mathcal{H}_k}\left\{-2\sum_{n=1}^{N}\lambda(\mathbf{x}_n) + \int_{\mathcal{X}}\lambda(\mathbf{x})^2\,d\mathbf{x} + \frac{1}{\gamma}\|\lambda\|_{\mathcal{H}_k}^2\right\}$$

### Theorem 1: Representer Theorem with Dual Coefficients Identically One

By utilizing the path integral representation to write the RKHS norm in functional form as $\|\lambda\|_{\mathcal{H}_k}^2 = \iint k^*(\mathbf{x},\mathbf{s})\lambda(\mathbf{x})\lambda(\mathbf{s})\,d\mathbf{x}\,d\mathbf{s}$, and setting the variational derivative of the objective functional to zero, the optimal solution is proved to be:

$$\hat{\lambda}(\mathbf{x}) = \sum_{n=1}^{N} h(\mathbf{x}, \mathbf{x}_n)$$

where $h(\cdot,\cdot)$ is the equivalent RKHS kernel, satisfying the Fredholm integral equation of the second kind: $\frac{1}{\gamma}h(\mathbf{x},\mathbf{x}') + \int_{\mathcal{X}}k(\mathbf{x},\mathbf{s})h(\mathbf{s},\mathbf{x}')\,d\mathbf{s} = k(\mathbf{x},\mathbf{x}')$.

**Key Finding**: The functional form of K²IE is identical to that of KIE (with dual coefficients $\alpha_n\equiv 1$), but it utilizes the equivalent RKHS kernel instead of traditional smoothing kernels. This not only builds a theoretical bridge between KIE and kernel methods but also eliminates the computational overhead of gradient descent optimization in FIE.

### Construction of Equivalent Kernels

- **Infinite Domain** $\mathcal{X}=\mathbb{R}^d$: Solved analytically via Fourier transform, $h = \mathcal{F}^{-1}[\tilde{k}(\omega)/(\gamma^{-1}+\tilde{k}(\omega))]$.
- **Bounded Domain** (union of hyper-rectangles): Approximate the RKHS kernel using Random Fourier Features $k(\mathbf{x},\mathbf{x}') \approx \boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{x}')$, yielding the equivalent kernel $h(\mathbf{x},\mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^\top(\gamma^{-1}\mathbf{I}_{2M}+\mathbf{A})^{-1}\boldsymbol{\phi}(\mathbf{x}')$, where the matrix $\mathbf{A}$ can be computed in **closed form** using sinc functions without requiring Nyström approximation.

### Computational Advantage in Cross-Validation

The least-squares cross-validation of K²IE only requires computing $\int_{\mathcal{X}}[\sum_n h(\mathbf{x},\mathbf{x}_n)]^2\,d\mathbf{x} = \boldsymbol{\xi}^\top\mathbf{A}\boldsymbol{\xi}$, with a complexity of $\mathcal{O}(M^2+MN)$, avoiding the need for Monte Carlo integration (as in KIE) or solving a dual problem (as in FIE).

## Key Experimental Results

Evaluated against KIE and FIE on 1D and 2D synthetic datasets under metrics including integrated squared error $L^2$, integrated absolute error $|L|$, the proportion of times outperforming KIE $\rho$, and CPU time.

### 1D Synthetic Data (100 Trials)

| Dataset | Method | $L^2$ ↓ | $|L|$ ↓ | $\rho$ ↑ | CPU (s) ↓ |
|--------|------|---------|---------|-----------|-----------|
| $\lambda^1_{1D}$ (N≈46) | KIE | **0.09** | **0.23** | – | – |
| | FIE | 0.11 | 0.24 | 0.34 | 1.06 |
| | K²IE | 0.12 | 0.26 | 0.26 | **0.01** |
| $10\times\lambda^1_{1D}$ (N≈466) | KIE | 1.43 | 0.87 | – | – |
| | FIE | 1.74 | 0.93 | 0.49 | 1.77 |
| | K²IE | **1.67** | **0.92** | 0.49 | **0.01** |

### 2D Synthetic Data (100 Trials, N≈543)

| Domain Proportion | Method | $L^2$ ↓ | $|L|$ ↓ | $\rho$ ↑ | CPU (s) ↓ |
|------------|------|---------|---------|-----------|-----------|
| $p=1.0$ | KIE | 63.3 | 6.36 | – | – |
| | FIE | 56.5 | **5.38** | 0.80 | 1.54 |
| | K²IE | **53.0** | 5.54 | **0.97** | **0.16** |
| $p=0.8$ | KIE | 64.5 | 6.34 | – | – |
| | FIE | 62.3 | **5.64** | 0.64 | 1.50 |
| | K²IE | **57.9** | 5.77 | **0.85** | **0.13** |

**Key Findings**:
- In 1D low-dimensional scenarios, KIE holds a slight advantage, but the gap essentially vanishes as the data volume increases.
- **In 2D scenarios, K²IE consistently outperforms both FIE and KIE in terms of $L^2$**, while being more than **10 times faster** than FIE in CPU time.
- The CPU time of K²IE is virtually independent of the data volume.

## Highlights & Insights

1. **Elegant Theoretical Unification**: Proves that classical KIE and modern kernel methods can be bridged fundamentally via least squares loss. The result that the dual coefficients are identically 1 is an elegant and surprising conclusion.
2. **Optimization-Free Inference**: Once the kernel hyperparameters are given, K²IE requires no iterative optimization and constructs the estimator directly from data points and equivalent kernels.
3. **Closed-Form Boundary Correction**: The equivalent kernel implicitly performs a more conservative boundary correction on bounded observation domains, showing substantial benefits in higher dimensions.
4. **Highly Efficient Cross-Validation**: The quadratic nature of the least squares loss allows hyperparameter selection to be performed fully analytically, bypassing Monte Carlo integration.
5. **Flexible Domain Support**: The framework can handle unions of hyper-rectangular regions, rendering it suitable for disconnected or irregular observation domains.

## Limitations & Future Work

1. **No Guarantee of Non-negativity**: K²IE directly models $\lambda\in\mathcal{H}_k$ rather than $\lambda=f^2$. Consequently, estimated values can be negative (particularly in data-sparse regions); the authors use truncation $\max(u,0)$ as a mitigation strategy.
2. **Synthetic-Only Evaluation in Main Text**: The main text only presents synthetic data results, while real-world dataset evaluations are relegated to the appendix, slightly weakening the empirical strength of the paper.
3. **RKHS Kernels Restricted to Translation-Invariant Kernels**: The current construction relies on Fourier transforms and random features; extending support to non-translation-invariant kernels remains an open avenue.
4. **Approximation Error of Random Features**: The accuracy of the equivalent kernel on bounded domains depends on the number of random Fourier features $2M$. The authors fix $M=250$ without thoroughly discussing the optimal choice.
5. **Insufficient Comparison with Bayesian Approaches**: Bayesian methods such as Gaussian Cox processes can provide uncertainty quantification, whereas K²IE, as a point estimation method, has natural limitations in this regard.

## Related Work & Insights

- **Flaxman et al. (2017)** proposed the RKHS-based Flaxman Intensity Estimator (FIE), proving the representer theorem via Mercer expansion.
- **Kim (2021)** provided the path-integral representation, which serves as a critical mathematical tool for the variational analysis in this paper.
- **Walder & Bishop (2017)** extended Flaxman's model to the Bayesian permanental process.
- **Rahimi & Recht (2007)**'s Random Fourier Features (RFF) form the foundation for constructing equivalent kernels on bounded domains.
- The application of least squares loss to point processes originates from **Hansen et al. (2015)**.

## Rating
- Novelty: ⭐⭐⭐⭐ — Achieving a representer theorem with dual coefficients of 1 by changing the loss function provides deep theoretical insights.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic experiments are well-designed, but there is a lack of real-world data validation in the main text.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, with a well-structured paper.
- Value: ⭐⭐⭐⭐ — Offers a novel paradigm for kernel intensity estimation that balances both efficiency and accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](../../ICLR2026/others/probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICML 2026\] New Bounds for Kernel Sums via Fast Spherical Embeddings](../../ICML2026/others/new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)
- [\[ICLR 2026\] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](../../ICLR2026/others/characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)
- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](../../ICLR2026/others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)

</div>

<!-- RELATED:END -->
