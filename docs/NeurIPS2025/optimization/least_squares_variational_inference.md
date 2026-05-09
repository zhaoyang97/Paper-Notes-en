---
title: >-
  [Paper Note] Least Squares Variational Inference
description: >-
  [NeurIPS 2025][Optimization][variational inference] This paper proposes LSVI (Least Squares Variational Inference), a gradient-free variational inference method based on ordinary least squares regression. Within the exponential family, LSVI iteratively solves for the optimal variational approximation by performing OLS regression on a tempered log-target, admitting efficient $O(d^3)$ (full-covariance) or $O(d)$ (mean-field) implementations for the Gaussian family.
tags:
  - NeurIPS 2025
  - Optimization
  - variational inference
  - natural gradient descent
  - exponential family
  - least squares
  - gradient-free
date: 2026-05-08
content_hash: 20402dbba876556e
---

# Least Squares Variational Inference

**Conference**: NeurIPS 2025
**arXiv**: [2502.18475](https://arxiv.org/abs/2502.18475)
**Code**: [https://github.com/ylefay/LSVI](https://github.com/ylefay/LSVI)
**Area**: Optimization
**Keywords**: variational inference, natural gradient descent, exponential family, least squares, gradient-free

## TL;DR

This paper proposes LSVI (Least Squares Variational Inference), a gradient-free variational inference method based on ordinary least squares regression. Within the exponential family, LSVI iteratively solves for the optimal variational approximation by performing OLS regression on a tempered log-target, admitting efficient $O(d^3)$ (full-covariance) or $O(d)$ (mean-field) implementations for the Gaussian family.

## Background & Motivation

Variational inference (VI) is a core tool in probabilistic machine learning, aiming to find the distribution within a parametric family $\mathcal{Q}$ that minimizes the KL divergence to a target distribution $\pi$. Mainstream methods rely on gradient-based optimization (SGD or natural gradient descent, NGD), as implemented in software such as STAN, NumPyro, and PyMC3.

**Limitations of Prior Work**:

**Gradient dependence**: Standard methods require $\log \pi$ to be automatically differentiable or rely on the reparameterization trick, which is unavailable in many important settings: discrete distributions, non-differentiable $\pi$, and cases where the likelihood cannot be analytically computed (e.g., likelihood-free inference).

**Variance issues**: Without the reparameterization trick, gradient estimates via the log-derivative trick suffer from extremely high variance.

**Tuning difficulty**: SGD converges slowly and requires careful step-size tuning; naïve implementations of NGD require expensive Fisher information matrix inversion.

**Fisher matrix scale**: In the Gaussian family, the Fisher matrix scales quadratically with dimension $d$, making direct inversion $O(m^3)$ where $m = O(d^2)$.

**Core Idea**: Exploiting the mathematical structure of the exponential family, the first-order optimality condition for minimizing the unnormalized KL (uKL) is reformulated as a fixed-point equation $\eta = \phi(\eta)$, where $\phi(\eta)$ is precisely the OLS regression coefficient of $f(X)$ on $s(X)$ with $X \sim q_\eta$. Each iteration thus requires only a single ordinary least squares regression, completely eliminating the need for gradients of the target.

## Method

### Overall Architecture

The LSVI iterative procedure:
1. Draw samples $X_1, \ldots, X_N$ from the current approximation $q_{\hat\eta_t}$
2. Compute Monte Carlo estimates $\hat{F}$ and $\hat{z}$
3. Solve OLS to obtain $\hat\eta'_{t+1} = \hat{F}^{-1} \hat{z}$
4. Apply momentum relaxation: $\hat\eta_{t+1} = \varepsilon_t \hat\eta'_{t+1} + (1 - \varepsilon_t) \hat\eta_t$

### Key Designs

1. **Exact LSVI Map and Fixed-Point Iteration**:

    - **Function**: Reformulates VI as a fixed-point iteration problem.
    - **Mechanism**: The first-order condition for uKL minimization is equivalent to $\{\mathbb{E}_\eta[ss^\top]\}\eta = \mathbb{E}_\eta[fs]$, i.e., $\eta = F_\eta^{-1} z_\eta$. This is precisely the OLS solution with $s(X)$ as the regressor and $f(X)$ as the response variable.
    - **Momentum relaxation**: The update $\eta_{t+1} = \varepsilon_t \phi(\eta_t) + (1-\varepsilon_t) \eta_t$ is adopted to prevent iterates from leaving the natural parameter space $\mathcal{V}$. The relaxation coefficient $\varepsilon_t$ corresponds to regressing on the tempered density $q_{\eta_t}^{1-\varepsilon_t} \pi^{\varepsilon_t}$.
    - **Design Motivation**: When the target lies within the variational family, $\phi$ recovers the exact solution in a single step—an elegant property not shared by other methods.

2. **Equivalence of LSVI to Natural Gradient / Mirror Descent**:

    - **Function**: Establishes theoretical convergence guarantees.
    - **Mechanism**: LSVI iteration (5) is shown to be equivalent to natural gradient descent in the natural parameter space, $\eta_{t+1} = \eta_t - \varepsilon_t F_{\eta_t}^{-1} \nabla_\eta l(\eta_t) / Z_{\eta_t}$, and also equivalent to mirror descent in the moment parameter space.
    - **Convergence rate**: Under $L$-smoothness and $\mu$-strong convexity, the convergence rate is $O(k^{-\mu/\alpha}) + O(N^{-1})$, where $k$ is the iteration count and $N$ is the sample size. The optimal rate $O(k^{-1}) + O(N^{-1})$ is attained when $\alpha = \mu$.

3. **Efficient Reparameterization for the Gaussian Family**:

    - **Function**: Eliminates Fisher matrix inversion and substantially reduces computational complexity.
    - **Mechanism**: For full-covariance Gaussians, the regression of $f(X)$ on $s(X)$ is reparameterized as a regression of $f(\mu + CZ)$ on $t(Z)$ with $Z \sim N(0,I)$, where $C = \text{Chol}(\Sigma)$. The statistic $t(z)$ is carefully constructed so that $\mathbb{E}[t(Z)t(Z)^\top] = I$, reducing the OLS estimator to $\hat\gamma = N^{-1} \sum_i t(Z_i) f(\mu + CZ_i)$ without any matrix inversion.
    - **Complexity**: $O(d^3)$ for full covariance (dominated by Cholesky decomposition) and $O(d)$ for mean field.
    - **Theorem 4.1** provides an explicit recursive formula mapping $\gamma$ to $\eta$.

4. **Adaptive Step-Size Selection**:

    - **Function**: Automatically determines an appropriate relaxation step size $\varepsilon_t$.
    - **Mechanism**: The step size $\varepsilon$ is observed to reduce the regression residual variance by a factor of $\varepsilon^2$. Given a residual variance upper bound $u^2$, the step size is set as $\varepsilon \leq u/v$ where $v$ is the current residual standard deviation, combined with backtracking to ensure iterates remain within the parameter space.
    - **Design Motivation**: Smoothness and strong convexity parameters are typically unknown; fixed step sizes are either unstable or overly conservative.

### Loss & Training

- The unnormalized KL divergence (uKL) is used as the optimization objective; its minimizer coincides with that of the standard KL (Proposition 2.2).
- Per-iteration cost: $O(m^3 + m^2 N)$ for generic LSVI; $O(d^3 + dN)$ for Gaussian LSVI-FC; $O(d + dN)$ for Gaussian LSVI-MF.

## Key Experimental Results

### Main Results

**Logistic regression (Pima dataset, full covariance)**:

| Method | Convergence Speed | Notes |
|--------|-------------------|-------|
| LSVI (Algorithm 1) | ~1 step | Converges in essentially one step, but requires Fisher matrix inversion |
| LSVI-FC (Algorithm 3) | <100 steps | Efficient, $O(d^3)$ |
| NGD | ~100 steps | Requires automatic differentiation |
| ADVI (pyMC3/Blackjax) | >100 steps | Requires step-size tuning |
| GMMVI | ~100 steps | Gradient-free but limited to low dimensions |

**MNIST logistic regression (mean field)**: LSVI-MF outperforms ADVI and NGD in time efficiency.

**Variable selection (discrete distribution, Bernoulli family)**:

| Method | Applicability | Result |
|--------|---------------|--------|
| LSVI (Algorithm 1) | Applicable to discrete families | Posterior marginal probabilities agree with SMC exact inference |
| ADVI | Not applicable (requires reparameterization) | — |
| SGD | Not applicable (requires gradients) | — |

This constitutes the first demonstration of variational inference on a Bernoulli product family.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| LSVI with linearly decaying step size | KL ~$O(1/k)$ | Standard convergence |
| LSVI with adaptive step size | Faster convergence | Residual control strategy is effective |
| Two independent sample sets vs. OLS | OLS has lower variance | Joint OLS estimation is more efficient |
| LSVI-FC vs. generic LSVI | LSVI-FC superior in high dimensions | Avoids Fisher matrix inversion |

**Bayesian Synthetic Likelihood (BSL, toad displacement model)**:
- The variational approximation from LSVI-FC closely matches the MCMC posterior.
- CPU cost is substantially lower than MCMC, as multiple simulator runs are not required.

### Key Findings

- LSVI recovers the exact solution in one step when the target distribution belongs to the variational family.
- Compared to NGD/ADVI, LSVI exhibits lower iteration noise, as OLS is an optimal estimator.
- In gradient-free settings (discrete distributions, BSL), LSVI is the only viable VI approach.

## Highlights & Insights

1. **Reformulating VI as a regression problem**: This perspective is highly elegant, mapping a complex optimization problem onto the classical least squares framework.
2. **Theoretical completeness**: The paper establishes a full equivalence among LSVI, NGD, and mirror descent, with conditional convergence rates.
3. **Efficient implementation for the Gaussian family**: Eliminating Fisher matrix inversion via reparameterization is a key practical contribution.
4. **Gradient-free and gradient-compatible**: LSVI operates in gradient-free settings and can also be combined with techniques such as subsampling for large-scale data.

## Limitations & Future Work

- The current framework is restricted to the exponential family; extensions to mixture exponential families remain to be explored.
- When the posterior is strongly non-Gaussian in certain directions, Gaussian approximations may be inadequate.
- The independence assumption in discrete exponential families can potentially be relaxed via tree-structured dependencies.
- Strong convexity and smoothness assumptions do not always hold in practice, though local convexity is typically sufficient.

## Related Work & Insights

- **OLS perspective of Salimans & Knowles (2013)**: This work provides the theoretical foundation; the present paper further establishes equivalence with NGD and derives efficient implementations.
- **Bayesian learning rule of Khan & Rue (2023)**: LSVI falls within the broader natural gradient VI framework, with emphasis on gradient-free settings.
- **Insights**: For Bayesian inference with intractable likelihoods (e.g., simulator-based models), LSVI provides a practical solution with theoretical guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐ The OLS-VI perspective is not entirely new, but the efficient implementation and complete theoretical analysis represent important contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers differentiable, discrete, and non-differentiable settings, though large-scale experiments are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, the structure is clear, and theory is well integrated with practice.
- Value: ⭐⭐⭐⭐ Provides an elegant and practical solution for gradient-free variational inference.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Brain-like Variational Inference](brain-like_variational_inference.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

<!-- RELATED:END -->
