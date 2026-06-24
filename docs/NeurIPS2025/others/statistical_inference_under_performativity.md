---
title: >-
  [Paper Note] Statistical Inference Under Performativity
description: >-
  [NeurIPS 2025][performative prediction] This paper establishes the first complete end-to-end statistical inference framework for performative prediction, deriving a central limit theorem and data-driven covariance estimation for repeated risk minimization (RRM) algorithms, and extending prediction-powered inference (PPI) to the dynamic performative setting to obtain tighter confidence intervals.
tags:
  - "NeurIPS 2025"
  - "performative prediction"
  - "statistical inference"
  - "central limit theorem"
  - "prediction-powered inference"
  - "confidence intervals"
date: 2026-05-08
content_hash: 80e628b979776a23
---

# Statistical Inference Under Performativity

**Conference**: NeurIPS 2025
**arXiv**: [2505.18493](https://arxiv.org/abs/2505.18493)  
**Code**: None  
**Area**: Other
**Keywords**: performative prediction, statistical inference, central limit theorem, prediction-powered inference, confidence intervals

## TL;DR

This paper establishes the first complete end-to-end statistical inference framework for performative prediction, deriving a central limit theorem and data-driven covariance estimation for repeated risk minimization (RRM) algorithms, and extending prediction-powered inference (PPI) to the dynamic performative setting to obtain tighter confidence intervals.

## Background & Motivation

**Background**: Performative prediction describes a widely observed phenomenon in which decisions made based on predictions in turn influence the predicted target itself. For example, loan approval policies alter borrowers' spending habits and thereby affect repayment ability. Perdomo et al. (2020) formalized this problem, and subsequent work has focused primarily on efficiently locating performative stable points ($\theta_{\text{PS}}$) and analyzing convergence rates.

**Limitations of Prior Work**: The existing literature has almost entirely neglected the problem of statistical inference under performative settings. In many contexts, the parameter $\theta$ represents a concrete policy (e.g., a tax rate or credit-score threshold), and it is insufficient to know merely that an estimator converges to $\theta_{\text{PS}}$—policymakers require confidence intervals and hypothesis tests. The only directly related work (Cutler et al., 2024) establishes asymptotic normality only for single-sample online gradient updates, assumes all structural information is known, and provides no data-driven covariance estimation.

**Key Challenge**: In the performative setting, the data distribution changes with each parameter update, violating the fixed-distribution assumption underlying classical statistical inference. Standard CLTs do not apply directly because samples at each iteration are drawn from a different distribution that depends on the preceding (noisy) parameter estimate.

**Goal**: (1) Establish a central limit theorem for batch RRM; (2) provide data-driven covariance estimation without assuming structural information is known; (3) extend PPI to the performative setting to obtain improved inference using a small amount of labeled data together with a large amount of unlabeled data.

**Key Insight**: The paper focuses on batch RRM rather than single-sample online updates, which is more representative of realistic policy-making scenarios. Novel gradient-free score matching and policy perturbation techniques are introduced to estimate the gradient of the distribution map.

**Core Idea**: A CLT for the dynamic setting is established through recursive analysis of error propagation. Policy perturbation combined with score matching circumvents the unobservability of the distribution map gradient, enabling end-to-end statistical inference.

## Method

### Overall Architecture

The framework proceeds along two main lines: (1) **basic inference**—establishing a CLT for the RRM estimator $\hat\theta_t$, deriving its asymptotic variance $V_t$, proposing data-driven variance estimation, and constructing confidence intervals for $\theta_{\text{PS}}$; (2) **enhanced inference**—incorporating PPI into the performative setting, exploiting unlabeled data and a labeling model at each iteration to reduce variance and tighten confidence intervals.

### Key Designs

1. **Central Limit Theorem in the Dynamic Setting**

    - **Function**: Establishes the asymptotic distribution of the RRM estimator $\hat\theta_t$, enabling confidence interval construction.
    - **Mechanism**: The central challenge is error propagation—the error in $\hat\theta_t$ is transmitted through the distribution map $\mathcal{D}(\cdot)$ to the next iteration. The theorem proves $\sqrt{n}(\hat\theta_t - \theta_t) \xrightarrow{D} \mathcal{N}(0, V_t)$, where the asymptotic variance is $$V_t = \sum_{i=1}^t \prod_{k=i}^{t-1}\nabla G(\theta_k) \cdot \Sigma_{\theta_{i-1}}(\theta_i) \cdot \prod_{k=i}^{t-1}\nabla G(\theta_k)^\top.$$ This is a recursively accumulated variance in which the error at each step is propagated through subsequent steps via the Jacobian $\nabla G(\theta_k)$ of the map $G$.
    - **Design Motivation**: Unlike static CLTs, this analysis must simultaneously account for per-step estimation error and inter-step error propagation. The assumption $\varepsilon < \gamma/\beta$ (distribution sensitivity smaller than the ratio of strong convexity to smoothness) ensures that errors are not amplified.

2. **Policy-Perturbation-Based Score Matching**

    - **Function**: Data-driven estimation of the distribution map gradient $\nabla G(\theta_k)$, which is the key unknown quantity in constructing $V_t$.
    - **Mechanism**: The expression for $\nabla G(\theta_k)$ involves the score function $\nabla_\theta \log p(z, \theta)$, but the distribution $p(z,\theta)$ is unknown. A parametric model $M(z,\theta;\psi)$ is trained to approximate $p$ via a score matching objective. The additional difficulty is that differentiation with respect to $\theta$ (rather than $z$) is required. This is addressed through a policy perturbation technique: in addition to sampling at $\hat\theta_t$, samples are also collected at $\hat\theta_t + \eta e_i$ (small perturbations along each coordinate direction), and finite differences are used to approximate the $\theta$-partial derivative inside the integral. Since the policy dimension $d$ is typically low, the cost of $d$ additional perturbation sample sets is acceptable.
    - **Design Motivation**: This is a gradient-free estimation approach that requires no knowledge of the functional form of the distribution map, only the ability to sample observations under multiple policy values.

3. **PPI under Performativity**

    - **Function**: Uses a large volume of unlabeled data together with a labeling model $f$ to reduce the variance of the RRM estimator.
    - **Mechanism**: At each iteration, in addition to a small labeled dataset $\{(x_i,y_i)\}_{i=1}^n$, a large unlabeled dataset $\{x_i^u\}_{i=1}^N$ ($N \gg n$) is used to obtain pseudo-labels via the labeling model $f$. The PPI estimator is $$\hat\theta_{t+1}^{\text{PPI}}(\lambda) = \arg\min_\theta \frac{\lambda}{N}\sum \ell(x_i^u, f(x_i^u);\theta) + \frac{1}{n}\sum\left[\ell(x_i,y_i;\theta) - \lambda\ell(x_i,f(x_i);\theta)\right].$$ The weight $\lambda_t$ is selected via greedy step-wise optimization that minimizes a scalar function of the asymptotic variance.
    - **Design Motivation**: Policy feedback data (e.g., survey responses) are often scarce and costly to collect. PPI allows researchers to augment inference precision using large quantities of easily accessible unlabeled data.

### Loss & Training

Each RRM step employs empirical risk minimization with a strongly convex loss function. The score matching model is trained via the Hyvärinen modified score matching objective, supporting both Gaussian parametric models and deep neural networks. The PPI hyperparameter $\lambda_t$ is adaptively selected by minimizing a scalar function of the asymptotic variance (e.g., Trace or $\mathbf{1}^\top V \mathbf{1}$).

## Key Experimental Results

### Main Results

| Confidence Interval Method | Coverage ($t=4$, $n=1000$) | Width | Notes |
|---|---|---|---|
| $\lambda=0$ (labeled data only) | ~90% | Widest | No unlabeled data used |
| $\lambda=1$ (full weight) | ~90% | Medium | Fixed weight |
| $\lambda=\hat\lambda_t$ (optimized) | ~90% | **Narrowest** | Adaptive selection |

### Ablation Study

| Score Matching Model | Training Loss $J(\psi)$ | Variance Estimation Error $\|\hat{V}_t - V_t\|$ |
|---|---|---|
| Gaussian parametric model | <0.05 | Decreases as $n$ grows |
| DNN (2 layers × 128 units) | <0.05 | Decreases as $n$ grows |

### Key Findings

- The proposed confidence interval width is $O(n^{-1/2})$, improving upon the $O(n^{-1/m})$ bound of Perdomo et al. (2020) (where $m \geq 2$ is the data dimension), with the advantage being especially pronounced in high-dimensional settings.
- Q–Q plots validate the practical utility of the CLT: the empirical distribution of $\hat{V}_t^{-1/2}\sqrt{n}(\hat\theta_t - \theta_t)$ closely matches the standard normal.
- Adaptive selection of $\lambda$ in PPI consistently yields the narrowest confidence intervals while maintaining the nominal coverage level.
- In bias-aware inference for $\theta_{\text{PS}}$, the gap between confidence intervals for $\theta_{\text{PS}}$ and $\theta_t$ decays exponentially in $t$, demonstrating that tight inference for $\theta_{\text{PS}}$ is achievable after only a small number of iterations.
- Both score matching implementations (parametric Gaussian and DNN) achieve training loss $J(\psi) < 0.05$, with negligible variance estimation error.

## Highlights & Insights

- The key breakthrough enabling **end-to-end inference for the first time** lies in covariance estimation. Prior work either assumed structural information was known or established only asymptotic normality without providing usable confidence intervals. The policy perturbation combined with score matching elegantly resolves the unobservability of the distribution map gradient at low practical cost, since the policy dimension $d$ is typically small.
- The **integration of PPI with performativity** is both natural and practically motivated: policy feedback data (e.g., survey responses) are typically scarce and subject to low response rates. Leveraging ML models to generate pseudo-labels for large volumes of unlabeled data in order to enhance inference is a direction of significant applied importance.
- Improving confidence interval width from $O(n^{-1/m})$ to $O(n^{-1/2})$ eliminates dependence on data dimensionality, which is particularly consequential in high-dimensional policy spaces.

## Limitations & Future Work

- The current framework employs bias-aware inference for $\theta_{\text{PS}}$, whose confidence interval width contains a bias term that decays exponentially but remains nonzero. Developing inference methods that target $\theta_{\text{PS}}$ directly, without routing through $\theta_t$, is an important future direction.
- Policy perturbation requires sampling under multiple neighboring policies, which may be infeasible in certain practical settings (e.g., when multiple policies cannot be run simultaneously).
- Experiments are conducted on relatively low-dimensional synthetic data ($d=2$); performance in realistic high-dimensional policy spaces remains to be verified.
- The score matching approach depends on the expressive capacity of the model $M(z,\theta;\psi)$, and model misspecification may introduce inference bias.

## Related Work & Insights

- **vs. Perdomo et al. (2020)**: Foundational work that formalized performative prediction, but provides only an $O(n^{-1/m})$ nonparametric convergence bound. The proposed CLT achieves a tighter $O(n^{-1/2})$ rate and supports confidence interval construction.
- **vs. Cutler et al. (2024)**: Also establishes asymptotic normality for single-sample online updates, but assumes all structural information is known and provides no data-driven covariance estimation. The present paper addresses the batch setting and delivers complete end-to-end inference.
- **vs. PPI++ (Angelopoulos et al., 2023)**: PPI++ combines ML predictions with a small labeled dataset under a static distribution. The present paper extends this to the dynamic performative setting, addressing the additional challenge that the distribution changes as parameters are updated across iterations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First paper to establish a complete statistical inference framework under performative prediction, with substantive contributions spanning CLT derivation, covariance estimation, and PPI extension.
- **Experimental Thoroughness**: ⭐⭐⭐ — Synthetic data validation is thorough (CLT verification, PPI comparison, score matching evaluation), but real-data experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical development is rigorous and clear, though notation density is high and the accessibility threshold is considerable.
- **Value**: ⭐⭐⭐⭐ — Provides theoretical foundations for quantifying uncertainty in policy-making contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales](scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)

</div>

<!-- RELATED:END -->
