---
title: >-
  [Paper Note] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression
description: >-
  [ICML2025][Prediction Powered Inference] Reinterprets the tuning of $\lambda$ in PPI++ as a post-hoc regression and proposes two improved methods, Ridge-PPI and Sigmoid-PPI. These methods significantly reduce the variance of mean estimation in few-label scenarios ($n < 50$), outperforming both classical estimation and PPI++.
tags:
  - "ICML2025"
  - "Prediction Powered Inference"
  - "few-label evaluation"
  - "regression coefficients"
  - "variance reduction"
  - "LLM evaluation"
date: 2026-05-08
content_hash: f6a14eeaa5ce3a5e
---

# Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression

**Conference**: ICML2025  
**arXiv**: [2411.12665](https://arxiv.org/abs/2411.12665)  
**Code**: [ppi_py](https://github.com/aangelopoulos/ppi_py) (baseline implementation)  
**Area**: Other  
**Keywords**: Prediction Powered Inference, few-label evaluation, regression coefficients, variance reduction, LLM evaluation

## TL;DR

Reinterprets the tuning of $\lambda$ in PPI++ as a post-hoc regression and proposes two improved methods, Ridge-PPI and Sigmoid-PPI. These methods significantly reduce the variance of mean estimation in few-label scenarios ($n < 50$), outperforming both classical estimation and PPI++.

## Background & Motivation

In the LLM era, model evaluation faces scalability challenges. Traditional evaluation relies heavily on human annotation, which is expensive and struggles to keep pace with model iterations. Although auto-evaluation (substituting human labels with LLM predictions) is efficient, predictions suffer from systematic bias, leading to unreliable evaluation results.

The **Prediction Powered Inference (PPI)** framework constructs unbiased, low-variance estimators by combining a small number of ground-truth labels $\mathcal{D}_n$ and a large number of pseudolabels $\mathcal{D}_N$. Its core estimation formula is:

$$\hat{\mu}_{PPI} = \frac{1}{N}\sum_{i=1}^{N}\lambda f(X_i^u) + \frac{1}{n}\left(\sum_{i=1}^{n}h(X_i) - \lambda f(X_i)\right)$$

where $h(X)$ is the ground-truth label function, $f(X)$ is the prediction model output, and $\lambda$ is a tuning parameter.

**Core Problem**: Existing PPI++ works are evaluated primarily under label sizes of $n \geq 50$ (typically over 200), whereas annotation resources in practice are extremely scarce. The authors discover that when $n$ is very small, PPI++ can actually **underperform classical sample mean estimation**—a critical defect that was previously understudied.

## Method

### Key Insight: $\lambda$ as a Regression Coefficient

The authors prove that choosing the optimal $\lambda$ is equivalent to solving a univariate OLS regression:

$$\lambda_{Opt} = \frac{Cov[h(X), f(X)]}{Var[f(X)]}$$

This implies that $\lambda$ is not simply a $[0, 1]$ interpolation parameter, but rather a **post-hoc regression coefficient** from $f(X)$ to $h(X)$. When $n$ is small, the OLS estimation itself has a large variance, leading to unstable estimation of $\lambda$, which in turn degrades the PPI estimation.

### Method 1: Ridge-PPI (Ridge Regression Regularization)

Replace OLS with ridge regression to estimate $\lambda$ by introducing an L2 penalty term $\alpha$:

$$\hat{\lambda}_{\alpha} = \frac{\hat{Cov}[h(X), f(X)]}{\hat{Var}[f(X)] + \alpha}$$

- $\alpha > 0$ shrinks $\lambda$ toward zero, reducing overfitting on small sample sizes.
- $\alpha$ is selected via cross-validation on the labeled data.
- The scaling factor $(1 + n/N)^{-1}$ remains consistent with PPI++.

### Method 2: Sigmoid-PPI (Nonlinear Regression)

For binary labels $h(X) \in \{0, 1\}$, linear regression is unnatural. Instead, a class of sigmoid functions is used:

$$g(f(X)) = \frac{1}{1 + \exp(-\alpha f(X) + \beta)}$$

$$\hat{\mu}_{PPI_g} = \frac{1}{N}\sum_{i=1}^{N}g(f(X_i^u)) + \frac{1}{n}\left(\sum_{i=1}^{n}h(X_i) - g(f(X_i))\right)$$

Parameters $\alpha, \beta$ are learned via cross-validation with L2 regularization. In large data scenarios, a scaling factor $\frac{1}{1+n/N}$ must be added to prevent divergence.

### Theoretical Analysis: Stochastic $\lambda$ Framework

By treating $\lambda$ as a random variable, the excess variance of PPI can be decomposed as:

$$Var[\hat{\mu}_{PPI}] - Var[\hat{\mu}_h] = \mathbb{E}[\lambda]^2 \cdot (\tfrac{1}{N}+\tfrac{1}{n}) Var[f(X)] + Var[\lambda] \cdot (2\mathbb{E}[f(X)]^2 + \ldots) - \frac{2\mathbb{E}[\lambda]}{n}Cov(h(X),f(X))$$

Key finding: The variance of PPI++ is **inversely proportional** to $Var[f(X)]$. The smaller $Var[f(X)]$ is, the more unstable the estimation of $\lambda$ becomes, making PPI++ more prone to failure. This explains why Ridge-PPI (which reduces $Var[\lambda]$) improves performance.

## Key Experimental Results

### Datasets

| Dataset | Task | Source |
|--------|------|------|
| Research datasets (8) | Rate estimation for galaxy morphology, forest cover, censuses, etc. | Angelopoulos et al. (2023a) |
| LLM Refusal Dataset | LLM refusal rate estimation | 50,000+ prompt-answer pairs |

### Main Results

| Method | Few-Label MAE (Relative to Classical Estimation) | Characteristics |
|------|---------------------------|------|
| Classical | 1.00 (Baseline) | Labeled data only |
| PPI++ | > 1.00 in some scenarios (worse) | Fails when $Var[f(X)]$ is small |
| **Ridge-PPI** | **≤ 0.75** (25%+ reduction) | Robust, adapts to various $n$ |
| **Sigmoid-PPI** | **≤ 0.75** (25%+ reduction) | Optimal at small $n$ |

### Key Findings

- On datasets such as plankton, alphafold, and forest, PPI++ underperforms classical estimation under multiple settings of $n$, whereas Ridge-PPI and Sigmoid-PPI consistently remain competitive or superior.
- LLM Refusal experiment: On sub-distributions with small $Var[f(X)]$, PPI++ shows 20% more error than classical estimation, whereas Ridge/Sigmoid-PPI exhibit 10% less error.
- The Pearson correlation coefficient between $Var[f(X)]$ and the advantage of PPI++ is $r = -0.69$, validating theoretical predictions.
- The most significant improvement occurs around $n \approx 20$; as $n$ increases, Ridge-PPI converges to the performance of PPI++.

## Highlights & Insights

1. **Elegant and Powerful Theoretical Insight**: Reinterpreting the tuning of $\lambda$ as a regression problem bridges the gap between the regression literature and the PPI framework with a single shift in perspective.
2. **Practical Discovery**: Unveils the phenomenon where PPI++ can actually perform worse in few-label scenarios and provides the conditions for this (small $Var[f(X)]$).
3. **Lightweight Methods**: Ridge-PPI only requires adding a regularization term in the denominator, involving near-zero implementation cost while delivering significant improvements.
4. **Stochastic $\lambda$ Analysis**: First to analyze PPI variance by treating $\lambda$ as a random variable, revealing previously overlooked sources of estimation error.
5. **Natural LLM Evaluation Scenarios**: Data generated by the generative models themselves can serve directly as unlabeled data pools, making the PPI framework highly compatible.

## Limitations & Future Work

1. **Focus on Mean Estimation Only**: Does not extend to more general database inference tasks such as quantiles or confidence intervals.
2. **Binary Classification Assumption**: The methods primarily target $h(X) \in \{0,1\}$, with insufficient verification in non-binary scenarios.
3. **Asymptotic Bias of Sigmoid-PPI**: Potential asymptotic bias may exist when $n$ is large, requiring the manual addition of a scaling factor.
4. **Difficult-to-Estimate Theoretical Optimal $\alpha$**: $\alpha^* = \frac{n(1+n/N)V}{Cov(f(X),h(X))^2}$ relies on cross-validation in practice.
5. **Distribution Shift Unacknowledged**: Assumes that labeled and unlabeled data sharing the same distribution, which is often violated in reality.
6. **Limited Experimental Scale**: LLM evaluation was only tested on a single metric (refusal rate).

## Related Work & Insights

- **PPI Framework**: Founded by Angelopoulos et al. (2023a,b); extended by Zrnic & Candès (2024a,b) to active sampling and cross-fitting.
- **Auto-Evaluation**: Boyeau et al. (2024) introduce PPI to LLM evaluation.
- **Control Variates**: Zhang et al. (2019); South et al. (2023) utilize control variates to reduce variance in semi-supervised learning.
- **Insight**: For any scenario leveraging surrogate models for evaluation (such as LLM-as-judge), close attention should be paid to the impact of surrogate prediction variance on estimation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective shift is simple and elegant, though the method itself is a direct application of known regression techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple datasets alongside theoretical analysis, though LLM evaluation is limited to a single metric.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear theoretical derivations, thoroughly articulated motivation, and intuitive chart designs.
- Value: ⭐⭐⭐⭐ — Offers direct practical value for few-label auto-evaluation, which is a highly active area in LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[ICML 2025\] Gradient Aligned Regression via Pairwise Losses](gradient_aligned_regression_via_pairwise_losses.md)
- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](prediction_via_shapley_value_regression.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](curvature_enhanced_data_augmentation_for_regression.md)

</div>

<!-- RELATED:END -->
