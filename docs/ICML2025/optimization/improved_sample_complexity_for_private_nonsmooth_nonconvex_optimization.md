---
title: >-
  [Paper Note] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization
description: >-
  [ICML2025][Optimization][Differential Privacy] This work studies nonsmooth nonconvex (NSNC) stochastic optimization under differential privacy constraints. By improving the effective sensitivity of the gradient estimator, it reduces the best-known sample complexity by a factor of $\Omega(\sqrt{d})$, and proves for the first time that Goldstein stationarity generalizes from empirical loss to population loss.
tags:
  - "ICML2025"
  - "Optimization"
  - "Differential Privacy"
  - "Nonsmooth Nonconvex Optimization"
  - "Goldstein Stationarity"
  - "Sample Complexity"
  - "Zeroth-Order Optimization"
  - "Variance Reduction"
date: 2026-05-08
content_hash: fe3a945078888724
---

# Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization

**Conference**: ICML2025  
**arXiv**: [2410.05880](https://arxiv.org/abs/2410.05880)  
**Code**: None (theoretical work)  
**Area**: Optimization  
**Keywords**: Differential Privacy, Nonsmooth Nonconvex Optimization, Goldstein Stationarity, Sample Complexity, Zeroth-Order Optimization, Variance Reduction

## TL;DR

This work studies nonsmooth nonconvex (NSNC) stochastic optimization under differential privacy constraints. By improving the effective sensitivity of the gradient estimator, it reduces the best-known sample complexity by a factor of $\Omega(\sqrt{d})$, and proves for the first time that Goldstein stationarity generalizes from empirical loss to population loss.

## Background & Motivation

- **Problem Setting**: Consider the stochastic/empirical objective function $F(x) = \mathbb{E}_{\xi \sim \mathcal{P}}[f(x;\xi)]$, where the component functions $f(\cdot;\xi)$ are neither smooth nor convex, which is extremely common in deep learning (e.g., ReLU, MaxPool).
- **Stationarity Concept**: In nonsmooth settings, minimizing the gradient norm suffers from lower bounds with exponential dimension dependence (Kornowski & Shamir, 2022). Thus, **Goldstein stationarity** is adopted—$x$ is an $(\alpha,\beta)$-stationary point if the norm of the convex combination of gradients in the $\alpha$-neighborhood of $x$ is $\leq \beta$.
- **Privacy Requirements**: Privacy protection of model training data is increasingly critical, with differential privacy (DP) being the standard formal framework. The core question is: what is the minimum sample size $n$ required to find an $(\alpha,\beta)$-stationary point under $(\varepsilon,\delta)$-DP constraints?
- **Prior Results**: Zhang et al. (2024) provided the only existing result for NSNC DP optimization, with a sample complexity of $n = \widetilde{\Omega}\!\left(\frac{d}{\alpha\beta^3} + \frac{d^{3/2}}{\varepsilon\alpha\beta^2}\right)$, where the privacy cost term contains $d^{3/2}$, leading to a relatively high dimension dependence.
- **Motivation**: Can the dimension dependence be reduced? Can empirical risk minimization (ERM) generalize to the population loss? Can first-order information be leveraged to reduce oracle complexity?

## Method

### Core Idea: Improving the Sensitivity of the Gradient Estimator

The key insight of this work is that the **effective sensitivity** of the zeroth-order gradient estimator can be significantly reduced. The standard estimator is:

$$\widetilde{\nabla} f_\alpha(x;\xi) = \frac{1}{m}\sum_{j=1}^{m} \frac{d}{2\alpha}\bigl(f(x+\alpha y_j;\xi) - f(x-\alpha y_j;\xi)\bigr) y_j$$

where $y_j \sim \text{Unif}(\mathbb{S}^{d-1})$. Zhang et al. choose $m=d$, yielding a mini-batch sensitivity upper bound of $\frac{Ld}{B}$. However, this work observes that when $m$ is sufficiently large, sub-Gaussian concentration inequalities guarantee that $\widetilde{\nabla} f_\alpha \approx \nabla f_\alpha$, meaning the sensitivity can be reduced to $\frac{L}{B}$, which is a factor of $d$ smaller than before. This implies a substantial reduction in the noise required for perturbation.

### Overall Architecture: O2NC + Tree Mechanism

The algorithm is based on the **Online-to-Non-Convex (O2NC)** conversion framework proposed by Cutkosky et al. (2023):

1. Maintain the increment $\Delta_t$ and the iterate $x_t = x_{t-1} + \Delta_t$
2. Query the gradient oracle at a random interpolation point $z_t = x_{t-1} + s_t \Delta_t$
3. Update using clipping: $\Delta_{t+1} = \min\{1, D/\|\Delta_t - \eta \tilde{g}_t\|\} \cdot (\Delta_t - \eta \tilde{g}_t)$
4. Output a random selection from $K$ windowed averages

Privacy is guaranteed via the **Tree Mechanism**—replacing independent noise with correlated Gaussian noise, reducing the error from $O(\sqrt{\Sigma})$ to $O(\log \Sigma)$.

### Contribution 1: Improved One-Pass Algorithm (Theorem 3.1)

Single pass over the dataset, $(\varepsilon,\delta)$-DP, outputting an $(\alpha,\beta)$-stationary point:

$$n = \widetilde{\Omega}\!\left(\frac{\sqrt{d}}{\alpha\beta^3} + \frac{d}{\varepsilon\alpha\beta^2}\right)$$

Both terms are reduced by a factor of $\Omega(\sqrt{d})$ compared to Zhang et al. Notably, the non-private term is $\sqrt{d}/(\alpha\beta^3)$, exhibiting a sublinear dimension dependence (which was previously incorrectly claimed to be impossible).

### Contribution 2: Multi-Pass ERM Algorithm (Theorem 4.1)

Allowing multiple passes over the data, the authors design an $(\varepsilon,\delta)$-DP ERM algorithm that outputs an $(\alpha,\beta)$-stationary point of the empirical loss:

$$n = \widetilde{\Omega}\!\left(\frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}\right)$$

This is the first private ERM algorithm for NSNC objectives possessing a sample complexity with sublinear dimension dependence.

### Contribution 3: Generalization from ERM to Population (Proposition 5.1)

For the first time, it is proven that Goldstein stationarity generalizes from the empirical loss to the population loss: an $(\alpha,\hat\beta)$-stationary point of the empirical loss is with high probability an $(\alpha,\beta)$-stationary point of the population loss, where $\beta = \hat\beta + \widetilde{O}(\sqrt{d/n})$. Combined with the ERM algorithm, the sample complexity for the stochastic objective is:

$$n = \widetilde{\Omega}\!\left(\frac{d}{\beta^2} + \frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}\right)$$

### Contribution 4: First-Order Algorithm to Reduce Oracle Complexity (Theorem 6.1)

Replacing the zeroth-order (function value) oracle with a first-order (gradient) oracle maintains the same sample complexity but reduces the number of oracle queries by a factor of $\widetilde{\Omega}(d^2)$.

## Main Results

| Method | Type | Empirical Sample Complexity | Stochastic Sample Complexity |
|------|------|---------------|---------------|
| Zhang et al. (2024) | One-pass | $\frac{d}{\alpha\beta^3} + \frac{d^{3/2}}{\varepsilon\alpha\beta^2}$ | Same as left |
| **Ours Thm 3.1** | One-pass | $\frac{\sqrt{d}}{\alpha\beta^3} + \frac{d}{\varepsilon\alpha\beta^2}$ | Same as left |
| **Ours Thm 4.1** | Multi-pass | $\frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}$ | $\frac{d}{\beta^2} + \frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}$ |

> Lipschitz constant $L$, initialization gap $\Phi$, and logarithmic factors are omitted in the table.

## Highlights & Insights

- **Sensitivity Improvement as the Core**: Reducing the effective sensitivity of the zeroth-order gradient estimator from $O(Ld/B)$ to $O(L/B)$ relies on increasing the sample size $m$ and leveraging sub-Gaussian concentration. A tighter sensitivity bound is obtained through high-probability clipping. This observation is simple yet powerful.
- **Correcting Prior Erroneous Claims**: Prior work claimed that the non-private term could not go below a linear dimension dependence $d$. This work demonstrates that it can achieve $\sqrt{d}$.
- **First Generalization Guarantee for Goldstein Stationarity**: Previous NSNC literature lacked generalization results from ERM to the population. This work fills this gap, making the ERM framework practical in private scenarios.
- **Theoretical Completeness**: Provides a systematic set of improvements, from one-pass to multi-pass, from zeroth-order to first-order, and from empirical to population.

## Limitations & Future Work

- **Purely Theoretical Work**: No experimental validation is provided, and all results hold only at the theoretical level. The effectiveness in practical deep learning scenarios remains unknown.
- **Constants and Logarithmic Factors**: Dependencies on poly-log factors, the Lipschitz constant $L$, and the initial gap $\Phi$ are hidden in the sample complexity, which may result in a large actual required sample size.
- **Lack of Lower Bounds**: No matching lower bounds are provided, leaving it unknown whether the current results are close to optimal.
- **Multi-Pass Algorithm Requires Multiple Data Accesses**: The ERM algorithm requires polynomial passes over the data, which may be infeasible in certain privacy-sensitive scenarios (e.g., federated learning, streaming data).
- **Only Lipschitz Assumption**: Although smoothness is not required, a global Lipschitz condition is still assumed, which may not hold for some practical models.
- **Practical Significance of Goldstein Stationarity**: Compared to the gradient norm in smooth cases, $(\alpha,\beta)$-Goldstein stationarity has weaker interpretability in practice.

## Related Work & Insights

- **Nonsmooth Nonconvex Optimization**: Zhang et al. (2020) pioneered the Goldstein stationarity perspective; Cutkosky et al. (2023) proposed the O2NC framework; Davis et al. (2022), Kornowski & Shamir (2024), among others, further developed this line of work.
- **Differentially Private Optimization**: Abundant literature exists for convex/smooth cases (Bassily et al., 2014, 2019; Feldman et al., 2020); for the NSNC case, only Zhang et al. (2024) provided results.
- **Tree Mechanism**: Classical privacy mechanism proposed by Dwork et al. (2010) and Chan et al. (2011) for privatizing cumulative sums.

## Rating

- Novelty: ⭐⭐⭐⭐ — The insight on sensitivity improvement is elegant and effective, and the generalization results fill a theoretical gap.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical work without experiments.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, progressing systematically from one-pass to multi-pass, with good notation consistency.
- Value: ⭐⭐⭐⭐ — Systematically improves the sample complexity for NSNC DP optimization, providing solid theoretical contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)
- [\[ICML 2025\] Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](../../NeurIPS2025/optimization/private_zeroth-order_optimization_with_public_data.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles](../../ICML2026/optimization/lower_complexity_bounds_for_nonconvex-strongly-convex_bilevel_optimization_with_.md)

</div>

<!-- RELATED:END -->
