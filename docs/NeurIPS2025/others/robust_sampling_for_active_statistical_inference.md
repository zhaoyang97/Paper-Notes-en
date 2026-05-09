---
title: >-
  [Paper Note] Robust Sampling for Active Statistical Inference
description: >-
  [NeurIPS 2025][Active Inference] This paper proposes a robust sampling strategy based on budget-preserving paths that optimally interpolates between uniform sampling and active sampling, ensuring the resulting estimator's variance is never worse than either baseline. This addresses the performance degradation caused by inaccurate uncertainty estimation in active statistical inference.
tags:
  - NeurIPS 2025
  - Active Inference
  - Robust Sampling
  - Prediction-Powered Inference
  - Uncertainty Estimation
  - Inverse Probability Weighting
date: 2026-05-08
content_hash: 701c1a14602e3cb0
---

# Robust Sampling for Active Statistical Inference

**Conference**: NeurIPS 2025
**arXiv**: [2511.08991](https://arxiv.org/abs/2511.08991)
**Code**: [Available](https://github.com/lphLeo/Robust-Active-Statistical-Inference)
**Area**: Statistical Inference / Active Sampling
**Keywords**: Active Inference, Robust Sampling, Prediction-Powered Inference, Uncertainty Estimation, Inverse Probability Weighting

## TL;DR
This paper proposes a robust sampling strategy based on budget-preserving paths that optimally interpolates between uniform sampling and active sampling, ensuring the resulting estimator's variance is never worse than either baseline. This addresses the performance degradation caused by inaccurate uncertainty estimation in active statistical inference.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Active statistical inference leverages uncertainty scores from AI models to prioritize labeling of high-uncertainty samples, thereby improving estimation accuracy. However, when uncertainty estimates are of poor quality (e.g., overconfident LLMs, miscalibrated models under distribution shift), active sampling may yield high-variance estimates that are even worse than simple uniform sampling. The core problem addressed in this paper is: can one design a robust strategy that is simultaneously guaranteed to be no worse than either uniform sampling or the initial active sampling scheme?

### Mechanism

**Goal**: ### Overall Architecture
Given an initial sampling rule $\pi$ and uniform sampling $\pi^{\text{unif}}$, a budget-preserving path $\pi^{(\rho)}$ connects the two, and the parameter $\rho$ is optimally selected to minimize variance.

## Method

### Overall Architecture
Given an initial sampling rule $\pi$ and uniform sampling $\pi^{\text{unif}}$, a budget-preserving path $\pi^{(\rho)}$ connects the two, and the parameter $\rho$ is optimally selected to minimize variance. Robust optimization is further introduced to guard against misspecification of the error function.

### Key Designs

**Budget-Preserving Path**: A continuous path $\pi^{(\rho)}$, $\rho \in [0,1]$, satisfying $\pi^{(0)}=\pi$, $\pi^{(1)}=\pi^{\text{unif}}$, and $\mathbb{E}[\pi^{(\rho)}(X)] = \mathbb{E}[\pi(X)]$. Three instantiations are considered:
- **Linear path**: $\pi^{(\rho)} = (1-\rho)\pi + \rho\pi^{\text{unif}}$
- **Geometric path**: $\pi^{(\rho)} \propto \pi^{1-\rho}(\pi^{\text{unif}})^\rho$ (recommended default)
- **Hellinger path**: $\pi^{(\rho)} \propto ((1-\rho)\sqrt{\pi} + \rho\sqrt{\pi^{\text{unif}}})^2$

**Optimal $\rho$ Estimation**: The error function $\hat{e}^2(\cdot) \approx \mathbb{E}[(Y-f(X))^2|X]$ is fitted, and variance is minimized via grid search over its empirical approximation.

**Robust Optimization**: A minimax problem over a misspecification set $\mathcal{C}$ for the error function is formulated: $\rho_{\text{robust}} = \arg\min_\rho \max_{\epsilon \in \mathcal{C}} \frac{1}{n}\sum_{i=1}^n \frac{\hat{e}^2(X_i)+\epsilon_i}{\pi^{(\rho)}(X_i)}$. The default constraint is $\ell_2$-bounded: $\|\epsilon\|_2 \leq c$, where $c$ is selected via cross-validation.

### Loss & Training
- **Burn-in phase**: Collect initial labeled data to fit the error function $\hat{e}$.
- **Main phase**: Execute sampling using the robust sampling rule $\pi^{(\rho_{\text{robust}})}$.
- **Theoretical guarantee**: When $\hat{\rho}$ consistently estimates $\rho^*$, $\sqrt{n}(\hat{\theta}^{\pi^{(\hat{\rho})}} - \theta^*) \xrightarrow{d} \mathcal{N}(0, \sigma_{\rho^*}^2)$, with $\sigma_{\rho^*}^2 \leq \min\{\sigma_0^2, \sigma_1^2\}$.

## Key Experimental Results

### Main Results (Pew Post-Election Survey — Presidential Approval Estimation)

| Sampling Method | Effective Sample Size Trend | Coverage |
|---|---|---|
| Uniform Sampling | Baseline | ~90% |
| Active Sampling (poor error estimate) | **Below uniform** | Large coverage bias |
| **Robust Active** | **Never below either** | ~90% |

### Effect of Burn-in Size

| Burn-in Ratio | Robust $\rho$ | Strategy |
|---|---|---|
| Very small | ~1 (near uniform) | Automatically degrades to uniform sampling |
| Moderate | Intermediate value | Balances the two |
| Sufficient | ~0 (near active) | Fully exploits uncertainty |

### LLM Computational Social Science Experiments

| Task | Uniform | Active | Robust |
|---|---|---|---|
| Political Bias Detection | Baseline | High variance | ≥ Baseline |
| Politeness Analysis | Baseline | Unstable | ≥ Baseline |
| Misinformation Detection | Baseline | High variance | **Significantly outperforms both** |

### Key Findings
- The geometric path performs best across all tests.
- The robust constraint set $\mathcal{C}$ is automatically tuned in tightness via cross-validation.
- When error estimation quality is poor, $\rho_{\text{robust}}$ approaches 1 (uniform); when quality is high, it approaches 0 (active).
- LLM verbatim confidence scores are often overconfident, leading to extremely small sampling probabilities and explosive inverse probability weights.

## Highlights & Insights
1. **Theoretical elegance**: The geometric path and robust optimization jointly guarantee asymptotic optimality and a variance lower bound.
2. **Practically motivated**: The approach directly addresses the widespread problem of unreliable LLM uncertainty estimates.
3. **Adaptivity**: As error estimation quality improves, the strategy automatically transitions from uniform to active sampling.
4. **Generality**: The framework extends from mean estimation to general convex M-estimation (linear regression, logistic regression, etc.).

## Limitations & Future Work
- The choice of constraint set $\mathcal{C}$ is sensitive to performance; automatic selection of $\mathcal{C}$ remains an open problem.
- Data-driven selection of the optimal path is insufficiently explored.
- The theoretical guarantees are asymptotic; finite-sample guarantees are validated only empirically.
- Online and sequential settings are not addressed.

## Related Work & Insights
- The paper extends the frameworks of prediction-powered inference and active statistical inference.
- It is closely related to AIPW estimation in semiparametric inference, missing data, and causal inference.
- The robust optimization perspective can be applied to other active methods involving uncertainty estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ (robust path interpolation is a genuine innovation)
- Technical Depth: ⭐⭐⭐⭐⭐ (theoretical derivations are rigorous and complete)
- Experimental Thoroughness: ⭐⭐⭐⭐ (validated across multiple domains and scenarios)
- Value: ⭐⭐⭐⭐⭐ (directly addresses a practical problem of LLM uncertainty in annotation pipelines)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)

</div>

<!-- RELATED:END -->
