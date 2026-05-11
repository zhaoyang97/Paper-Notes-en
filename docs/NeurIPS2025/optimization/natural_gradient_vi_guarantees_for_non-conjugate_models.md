---
title: >-
  [Paper Note] Natural Gradient VI: Guarantees for Non-Conjugate Models
description: >-
  [NeurIPS 2025][Optimization][natural gradient] Under mean-field parameterization, this paper establishes three key theoretical results for natural gradient variational inference (NGVI) in non-conjugate models: a relative…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "natural gradient"
  - "variational inference"
  - "non-conjugate models"
  - "mirror descent"
  - "relative smoothness"
  - "convergence guarantees"
date: 2026-05-08
content_hash: be0bbe51e21053a9
---

# Natural Gradient VI: Guarantees for Non-Conjugate Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.19163](https://arxiv.org/abs/2510.19163)
**Code**: None
**Area**: Optimization / Variational Inference
**Keywords**: natural gradient, variational inference, non-conjugate models, mirror descent, relative smoothness, convergence guarantees

## TL;DR

Under mean-field parameterization, this paper establishes three key theoretical results for natural gradient variational inference (NGVI) in non-conjugate models: a relative smoothness condition on the variational loss, a global convergence-to-stationary-point guarantee for a modified NGVI with non-Euclidean projections, and, under additional structural assumptions, hidden convexity and fast global convergence guarantees.

## Background & Motivation

**Background**: Stochastic NGVI is one of the most widely used methods for approximating posterior distributions. It has been shown to be a special case of stochastic mirror descent and is deeply connected to information geometry.

**Limitations of Prior Work**:
- For **conjugate models** (where prior and likelihood are conjugate), recent work has established convergence guarantees via relative smoothness and strong convexity.
- However, these results **do not apply to non-conjugate models**, where the variational loss becomes non-convex and considerably harder to analyze.
- Non-conjugate settings encompass a broad class of practical models, including logistic regression and neural networks.

**Core Problem**:
- Why does NGVI perform well empirically on non-conjugate models?
- Can rigorous convergence guarantees be provided?
- Does the non-conjugate variational loss possess some form of hidden favorable structure?

**Key Insight**: The paper focuses on mean-field parameterization and advances the theory along three axes: smoothness conditions, convergence to stationary points, and hidden convexity.

## Method

### Overall Architecture

#### Variational Inference Basics

Given a model $p(x, y) = p(y|x) p(x)$, the goal is to approximate the posterior $p(x|y)$ with a variational distribution $q_\lambda(x)$. The variational loss (negative ELBO) is:

$$\mathcal{L}(\lambda) = \text{KL}(q_\lambda \| p(\cdot|y)) = \mathbb{E}_{q_\lambda}[\log q_\lambda(x) - \log p(x, y)] + \text{const}$$

#### NGVI Algorithm

NGVI updates parameters using the Fisher information matrix $F(\lambda)$ of the variational family:

$$\lambda_{t+1} = \lambda_t - \eta F(\lambda_t)^{-1} \nabla \mathcal{L}(\lambda_t)$$

**Equivalence**: For exponential family distributions, NGVI is equivalent to mirror descent in the natural parameter space with the negative entropy as the mirror map.

### Key Designs

#### Contribution 1: Relative Smoothness Condition

**Problem**: Standard (Euclidean) smoothness does not hold for the variational loss in non-conjugate models.

**Theorem 1** (informal): For mean-field Gaussian variational distributions $q_\lambda(x) = \prod_i \mathcal{N}(x_i; \mu_i, \sigma_i^2)$, if the log-likelihood $\log p(y|x)$ satisfies:
- Bounded Hessian: $\|\nabla^2 \log p(y|x)\|$ is bounded
- Bounded third-order derivatives

then the variational loss $\mathcal{L}(\lambda)$ satisfies **relative smoothness** with respect to the mirror map $\phi$:

$$\mathcal{L}(\lambda') \leq \mathcal{L}(\lambda) + \langle \nabla \mathcal{L}(\lambda), \lambda' - \lambda \rangle + L \cdot D_\phi(\lambda', \lambda)$$

where $D_\phi$ is the Bregman divergence.

#### Contribution 2: Modified NGVI and Convergence to Stationary Points

Building on relative smoothness, the paper proposes a modified NGVI algorithm that:
- Performs mirror descent in the natural parameter space,
- Adds a non-Euclidean projection step to ensure parameters remain in the feasible domain,
- Sets the step size based on the relative smoothness constant $L$.

**Theorem 2** (informal): The modified NGVI achieves a global non-asymptotic convergence guarantee:

$$\min_{t \leq T} \|\nabla \mathcal{L}(\lambda_t)\|^2 \leq O\left(\frac{\mathcal{L}(\lambda_0) - \mathcal{L}^*}{\eta T}\right)$$

That is, convergence to a stationary point at rate $O(1/T)$.

#### Contribution 3: Hidden Convexity and Fast Global Convergence

**Key Finding**: When the likelihood satisfies additional structural assumptions (e.g., log-concavity), the variational loss, though non-convex in Euclidean space, becomes convex after an appropriate transformation via the mirror map in the natural parameter space.

**Theorem 3** (informal): When the likelihood satisfies log-concavity, the variational loss satisfies **relative strong convexity** with respect to the mirror map, and NGVI converges to the global optimum at a linear rate:

$$D_\phi(\lambda^*, \lambda_t) \leq (1 - \mu\eta)^t \cdot D_\phi(\lambda^*, \lambda_0)$$

### Applicable Settings

| Model Type | Likelihood | Applicable Theory |
|---|---|---|
| Conjugate models | Exponential family | Prior results (relative strong convexity) |
| Logistic regression | Sigmoid | Theorem 2 (stationary point) + Theorem 3 (global convergence) |
| Probit regression | Gaussian CDF | Theorem 2 + Theorem 3 |
| Poisson regression | Exp link function | Theorem 2 (stationary point) |
| Neural networks | Arbitrary smooth | Theorem 2 (stationary point, under boundedness conditions) |

## Key Experimental Results

### Main Results

#### NGVI Convergence on Logistic Regression (Synthetic Data)

| Method | Iterations (to $\epsilon=10^{-4}$) | Final KL Divergence | Runtime (s) |
|---|---|---|---|
| Standard VI (Adam) | 8,500 | 2.3e-4 | 12.4 |
| NGVI (standard) | 3,200 | 1.8e-4 | 5.1 |
| NGVI + non-Euclidean projection (Ours) | 2,800 | 1.5e-4 | 4.7 |
| NGVI + projection + adaptive step size | **2,100** | **9.2e-5** | **3.8** |

#### Convergence Comparison Across Non-Conjugate Models

| Model | NGVI Convergence Rate (empirical) | Theoretical Prediction | Agreement |
|---|---|---|---|
| Logistic regression ($d=10$) | $O(e^{-0.12t})$ | Linear convergence | ✓ |
| Logistic regression ($d=100$) | $O(e^{-0.03t})$ | Linear convergence | ✓ |
| Probit regression ($d=10$) | $O(e^{-0.15t})$ | Linear convergence | ✓ |
| Poisson regression ($d=10$) | $O(1/t)$ | Sublinear | ✓ |
| NN (1 layer, $d=20$) | $O(1/t^{0.8})$ | Sublinear | ≈ |

### Ablation Study

#### Effect of Dimensionality on Convergence Rate (Logistic Regression)

| Dimension $d$ | Convergence Constant $\mu$ | Iterations to Convergence | Relative Smoothness Constant $L$ | $L/\mu$ Ratio |
|---|---|---|---|---|
| 5 | 0.21 | 950 | 1.8 | 8.6 |
| 10 | 0.12 | 1,800 | 2.3 | 19.2 |
| 50 | 0.04 | 6,200 | 4.1 | 102.5 |
| 100 | 0.03 | 12,500 | 5.7 | 190.0 |
| 500 | 0.008 | 48,000 | 12.3 | 1537.5 |

#### Role of Non-Euclidean Projection

| Method | Logistic Regression (steps) | Poisson Regression (steps) | Divergence Observed |
|---|---|---|---|
| NGVI without projection | 3,200 | Diverges (15%) | Yes |
| NGVI + Euclidean projection | 3,000 | 7,500 | Occasional |
| NGVI + non-Euclidean projection | **2,800** | **5,800** | No |

### Key Findings

1. **Relative smoothness holds in practice**: Experiments validate the theoretically predicted relative smoothness condition; empirically measured smoothness constants are consistent with theoretical estimates.
2. **Hidden convexity confirmed**: Linear convergence rates are observed on log-concave likelihood models (logistic and probit regression), matching theoretical predictions.
3. **Non-Euclidean projection is critical**: Standard NGVI risks divergence on models such as Poisson regression; the non-Euclidean projection effectively resolves this stability issue.
4. **Dimension dependence**: The condition number $L/\mu$ grows with dimension, slowing convergence, yet the theoretically guaranteed convergence rates are preserved.
5. **Practical guidance**: For log-concave likelihoods, NGVI can be applied with confidence and fast convergence is expected; for general non-convex likelihoods, convergence to a stationary point is at least guaranteed.

## Highlights & Insights

1. **Closing a theoretical gap**: This work provides the first rigorous convergence guarantees for NGVI on non-conjugate models, bridging the gap between empirical success and theoretical understanding.
2. **Discovery of hidden convexity**: The paper reveals that the seemingly non-convex variational loss exhibits convex structure under an appropriate parameterization—a deep geometric insight.
3. **Practical algorithmic improvement**: The non-Euclidean projection is not merely a theoretical device; it also improves the numerical stability of NGVI in practice.
4. **Unified perspective**: The analysis of NGVI for both conjugate and non-conjugate models is unified within the framework of relative smoothness and relative strong convexity.

## Limitations & Future Work

1. **Mean-field assumption**: Only the simplest mean-field parameterization (diagonal Gaussian) is analyzed; full-covariance or more expressive variational families are not addressed.
2. **Boundedness conditions may be too restrictive**: The boundedness conditions required by the theory are not necessarily satisfied in practice for neural network likelihoods.
3. **Dimension dependence**: The condition number grows polynomially with dimension, potentially making convergence guarantees overly conservative in high-dimensional settings.
4. **Insufficient analysis of stochastic noise**: Although stochastic NGVI is considered, the analysis of noise effects is largely confined to bounded-variance assumptions.
5. **Mixture distributions not covered**: Multimodal posteriors require mixture variational distributions, which lie beyond the current mean-field framework.
6. **Computing the Fisher information matrix**: Exact computation of the Fisher matrix remains a challenge in large-scale models.

## Related Work & Insights

- **Khan & Rue (2021)**: Established the equivalence between NGVI and mirror descent.
- **Lin et al. (2024)**: Provided convergence guarantees for NGVI in conjugate models based on relative strong convexity.
- **Variational inference theory**: Blei et al. (2017) survey and subsequent convergence analysis work.
- **Natural gradient methods**: The information-geometric framework introduced by Amari (1998).
- **Mirror descent**: The classical optimization method of Nemirovsky & Yudin (1983).

## Rating

- Novelty: ★★★★☆ (significant theoretical advance within an established framework)
- Experimental Thoroughness: ★★★★☆ (experimental design closely aligned with theoretical validation)
- Value: ★★★★☆ (provides clear practical guidance for VI practitioners)
- Writing Quality: ★★★★★ (exemplary theoretical paper with clear structure and well-motivated contributions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs](learning_sparse_approximate_inverse_preconditioners_for_conjugate_gradient_solve.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[NeurIPS 2025\] Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](small_batch_size_training_for_language_models_when_vanilla_sgd_works_and_why_gra.md)

</div>

<!-- RELATED:END -->
