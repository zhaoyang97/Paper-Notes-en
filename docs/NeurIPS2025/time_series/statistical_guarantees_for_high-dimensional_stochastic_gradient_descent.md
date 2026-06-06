---
title: >-
  [Paper Note] Statistical Guarantees for High-Dimensional Stochastic Gradient Descent
description: >-
  [NeurIPS 2025][Time Series][stochastic gradient descent] This work introduces coupling techniques from high-dimensional nonlinear time series into online learning…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "stochastic gradient descent"
  - "high-dimensional statistics"
  - "constant learning rate"
  - "geometric moment contraction"
  - "concentration inequalities"
date: 2026-05-08
content_hash: 45a5ae8b848faf12
---

# Statistical Guarantees for High-Dimensional Stochastic Gradient Descent

**Conference**: NeurIPS 2025
**arXiv**: [2510.12013](https://arxiv.org/abs/2510.12013)  
**Code**: None  
**Area**: Time Series
**Keywords**: stochastic gradient descent, high-dimensional statistics, constant learning rate, geometric moment contraction, concentration inequalities

## TL;DR

This work introduces coupling techniques from high-dimensional nonlinear time series into online learning, providing the first rigorous moment convergence bounds and high-probability concentration inequalities—under $\ell^s$ and $\ell^\infty$ norms—for constant learning rate SGD and its Ruppert–Polyak averaged variant (ASGD) in high dimensions.

## Background & Motivation

**Background**: SGD is a cornerstone of large-scale machine learning, widely used in high-dimensional and overparameterized settings. In practice, **constant (large) learning rates** are commonly adopted to accelerate convergence, yet the theoretical understanding remains severely lagging.

**Limitations of Prior Work**:
   - Classical SGD theory primarily targets **decaying learning rates** and cannot characterize the behavior under constant learning rates.
   - Existing convergence analyses are largely confined to the $\ell^2$ norm and low-dimensional settings.
   - Constant learning rate SGD iterates do not converge to a single point; instead, they oscillate around a stationary distribution with an irreducible $O(\alpha)$ bias.
   - High-dimensional sparse/structured models typically require $\ell^\infty$ norm control, yet this norm is non-differentiable, making gradient-based tools difficult to apply.

**Key Challenge**: Constant learning rate SGD works well in practice, yet rigorous high-dimensional guarantees—particularly for higher-order moments and general norms—are absent from the theory.

**Goal**: To establish rigorous statistical guarantees for high-dimensional constant learning rate SGD/ASGD: moment convergence bounds, high-probability concentration inequalities, and computational complexity bounds.

**Key Insight**: The SGD iterates are viewed as a **nonlinear autoregressive process**, and coupling techniques (functional dependence measures) from high-dimensional nonlinear time series analysis are imported into online learning.

**Core Idea**: By establishing geometric moment contraction of SGD under the $\ell^s$ norm (with $s \approx \log d$), the non-differentiability of $\ell^\infty$ is circumvented, yielding a complete theoretical framework for high-dimensional SGD.

## Method

### Overall Architecture

Consider the optimization problem $\boldsymbol{\beta}^* \in \arg\min_{\boldsymbol{\beta}} G(\boldsymbol{\beta})$, where $G(\boldsymbol{\beta}) = \mathbb{E}_{\boldsymbol{\xi} \sim \Pi} g(\boldsymbol{\beta}, \boldsymbol{\xi})$.

SGD iterate: $\boldsymbol{\beta}_k = \boldsymbol{\beta}_{k-1} - \alpha \nabla g(\boldsymbol{\beta}_{k-1}, \boldsymbol{\xi}_k)$, with constant learning rate $\alpha > 0$.

ASGD iterate: $\bar{\boldsymbol{\beta}}_k = \frac{1}{k} \sum_{i=1}^k \boldsymbol{\beta}_i$ (Ruppert–Polyak averaging).

### Key Designs

#### 1. $\ell^s$-$\ell^\infty$ Norm Bridge

**Function**: Circumvents the non-differentiability of the $\ell^\infty$ norm.

**Mechanism**: Choosing $s_d = 2\min\{\ell \in \mathbb{N}: 2\ell > \log(d)\}$ yields $|\boldsymbol{x}|_\infty \leq |\boldsymbol{x}|_{s_d} \leq e|\boldsymbol{x}|_\infty$, so the $\ell^{s_d}$ norm is equivalent to $\ell^\infty$ while being differentiable (even power), enabling gradient-based analysis.

**Design Motivation**: Directly analyzing $|\boldsymbol{\beta}_k - \boldsymbol{\beta}^*|_\infty$ is highly challenging because the non-differentiability of $\ell^\infty$ blocks standard gradient tools. This bridging technique serves as the technical foundation of the entire paper.

#### 2. Geometric Moment Contraction (GMC, Theorem 1)

**Function**: Proves that constant learning rate SGD forgets its initialization at an exponential rate.

**Core Result**: Under learning rate $0 < \alpha < \frac{2\mu}{\max\{q,s\} L_{s,q}^2}$, two SGD sequences sharing the same noise but starting from different initializations satisfy:

$$\||\ \boldsymbol{\beta}_k - \boldsymbol{\beta}'_k|_s\|_q \leq r_{\alpha,s,q}^k |\boldsymbol{\beta}_0 - \boldsymbol{\beta}'_0|_s$$

where $r_{\alpha,s,q} = 1 - 2\mu\alpha + \max\{q,s\} L_{s,q}^2 \alpha^2 < 1$.

**Significance**: Guarantees the existence and uniqueness of a stationary distribution $\pi_\alpha$ with $\boldsymbol{\beta}_k \Rightarrow \pi_\alpha$.

#### 3. High-Dimensional Moment Inequality (Lemma 2, Rio-type)

**Function**: Provides sharper upper bounds on high-dimensional norm operations than the standard triangle inequality.

**Core Formula**: For any $q \geq 2$, even integer $s \geq 2$, and $d$-dimensional random vectors $\boldsymbol{x}, \boldsymbol{y}$:

$$\||\boldsymbol{x} + \boldsymbol{y}|_s\|_q^2 \leq \||\boldsymbol{x}|_s\|_q^2 + 2\||\boldsymbol{x}|_s\|_q^{2-q} \mathbb{E}(|\boldsymbol{x}|_s^{q-s} \sum_j x_j^{s-1} y_j) + (\max\{q,s\}-1)\||\boldsymbol{y}|_s\|_q^2$$

In particular, when $\mathbb{E}[\boldsymbol{y}|\boldsymbol{x}] = 0$, the cross term vanishes.

#### 4. Three-Term Decomposition for ASGD (Theorem 3)

**Core Formula**: $\||\bar{\boldsymbol{\beta}}_k - \boldsymbol{\beta}^*|_\infty\|_q$ decomposes as:

$$\underbrace{O\left(\sqrt{\frac{c_q s_d}{k}} M_{s_d,q}\right)}_{\text{stochastic variance}} + \underbrace{O\left(\frac{\|\boldsymbol{\beta}_0 - \boldsymbol{\beta}_0^\circ\|}{k(1-r)}\right)}_{\text{initialization bias}} + \underbrace{O\left(M_{s_d,q}^2 \max\{q,s_d\} \alpha d^{q/(q-1) \cdot (1-2/s_d)}\right)}_{\text{constant learning rate bias}}$$

#### 5. Fuk–Nagaev-type High-Probability Bound (Theorem 4)

**Function**: Provides high-probability tail bounds for ASGD.

For any $z > 0$:

$$\mathbb{P}(|\bar{\boldsymbol{\beta}}_k - \boldsymbol{\beta}^*|_\infty > z) \lesssim \frac{\text{initialization term}}{(k\alpha z)^q} + \frac{\text{polynomial remainder}}{z^q k^{q-1}} + \exp\left(-\frac{Ckz^2 \alpha^{1-2/q}}{M_{s_d,q}^2 \log d}\right)$$

### Theoretical Assumptions

- **Assumption 1**: Coercivity condition on the loss function $G$.
- **Assumption 2**: Strong convexity under the $\ell^s$ norm with parameter $\mu > 0$.
- **Assumption 3**: Stochastic Lipschitz continuity with constant $L_{s,q}$.

## Key Experimental Results

### Main Results

This is a purely theoretical paper with no numerical experiments. The core contributions are theorems and corollaries.

### Key Theoretical Results

| Result | Content | Norm | Learning Rate Requirement |
|------|------|------|-----------|
| Theorem 1 | SGD geometric moment contraction | $\ell^s$ | $\alpha < 2\mu/[\max\{q,s\}L_{s,q}^2]$ |
| Theorem 2 | SGD moment convergence rate | $\ell^s, \ell^\infty$ | Same as above |
| Theorem 3 | ASGD moment convergence rate | $\ell^\infty$ | Same as above |
| Proposition 2 | ASGD complexity bound | $\ell^\infty$ | $O(1/\varepsilon^2)$ |
| Theorem 4 | ASGD high-probability concentration | $\ell^\infty$ | Same as above |
| Theorem 5 | Gaussian approximation | $\ell^2$ | $d = o(T^{1/4-\zeta})$ |

### Key Findings

1. Under a constant learning rate, high-dimensional SGD admits a unique stationary distribution, and the effect of initialization is forgotten at a geometric rate.
2. The complexity bound for ASGD is $O(1/\varepsilon^2)$, consistent with the Q-learning result of Wainwright (2019).
3. The learning rate upper bound scales as $\alpha \propto 1/(d^2 \log d)$ with dimension (in the linear regression setting where $L_{s_d,q} \asymp d$).
4. The polynomial decay rate $k^{1-q}$ in the high-probability tail bound is unimprovable (matching the Nagaev lower bound).
5. The Gaussian approximation requires $d = o(T^{1/4})$, revealing a fundamental relationship between dimensionality and sample size.

## Highlights & Insights

- **Methodological Innovation**: Coupling techniques from time series analysis are systematically imported into online optimization theory, opening a new analytical avenue.
- **Generality**: The analysis is not restricted to the $\ell^2$ norm; it is the first to handle general $\ell^s$ and $\ell^\infty$ norms.
- **Practical Relevance**: Constant learning rates are precisely what practitioners use; theory now catches up with practice.
- **The Rio-type high-dimensional moment inequality** (Lemma 2) has independent value and can be applied to analyze other high-dimensional learning algorithms.
- **Complete theoretical chain**: From stationarity → moment convergence → high probability → complexity → Gaussian approximation, presented in a unified framework.

## Limitations & Future Work

1. Strong convexity (under the $\ell^s$ norm) is required; the results do not directly apply to non-convex objectives.
2. The learning rate upper bound $\alpha \propto 1/(d^2 \log d)$ may be overly conservative in extremely high dimensions.
3. Adaptive learning rate methods (Adam, AdaGrad, etc.) are not addressed.
4. No numerical experiments are conducted to verify the tightness of the theoretical predictions.
5. Beyond linear regression, the dimension dependence of $L_{s,q}$ and $M_{s,q}$ requires case-by-case analysis for specific problems.

## Related Work & Insights

- This work is complementary to Dieuleveut et al. (2020), which treats constant step-size SGD as a Markov chain, but provides sharper non-asymptotic bounds.
- The technique of bridging $\ell^\infty$ via $\ell^s$ norms is instructive for other problems requiring control of the maximum coordinate error.
- The framework is extensible to SGD with dropout regularization (cf. the low-dimensional version in Li et al. 2024c).
- Future work may explore applying this technique to federated learning, distributed optimization, and other high-dimensional settings.

## Rating

⭐⭐⭐⭐

A rigorous, purely theoretical contribution that fills an important gap in the theory of high-dimensional constant learning rate SGD. The technical depth is high; however, the absence of numerical experiments and the limited direct guidance for practical algorithm design are notable shortcomings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[ICML 2026\] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](../../ICML2026/time_series/helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)
- [\[NeurIPS 2025\] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)

</div>

<!-- RELATED:END -->
