---
title: >-
  [Paper Note] Learning Survival Distributions with the Asymmetric Laplace Distribution
description: >-
  [ICML2025][Social Computing][Survival Analysis] This paper proposes a parametric survival analysis method based on the asymmetric Laplace distribution (ALD). By using a neural network to learn the three parameters of the ALD (location, scale, and asymmetry), it achieves continuous, closed-form estimation of the survival distribution, comprehensively outperforming existing parametric and non-parametric approaches in both discriminative and calibration performance.
tags:
  - "ICML2025"
  - "Social Computing"
  - "Survival Analysis"
  - "Asymmetric Laplace Distribution"
  - "Parametric Models"
  - "Quantile Regression"
  - "Maximum Likelihood Estimation"
date: 2026-05-08
content_hash: 84bfea35391304c8
---

# Learning Survival Distributions with the Asymmetric Laplace Distribution

**Conference**: ICML2025  
**arXiv**: [2505.03712](https://arxiv.org/abs/2505.03712)  
**Code**: The paper includes code in supplementary materials  
**Area**: Survival Analysis / Probabilistic Modeling  
**Keywords**: Survival Analysis, Asymmetric Laplace Distribution, Parametric Models, Quantile Regression, Maximum Likelihood Estimation

## TL;DR

This paper proposes a parametric survival analysis method based on the asymmetric Laplace distribution (ALD). By using a neural network to learn the three parameters of the ALD (location, scale, and asymmetry), it achieves continuous, closed-form estimation of the survival distribution, comprehensively outperforming existing parametric and non-parametric approaches in both discriminative and calibration performance.

## Background & Motivation

Survival analysis (also known as time-to-event analysis) aims to predict the event time distribution given covariates, and is widely applied in fields such as healthcare, finance, and engineering. The core challenge lies in **right-censoring**: for some samples, the event has not occurred by the end of the observation period.

Existing methods are categorized into three classes, each with limitations:

- **Parametric methods** (Exponential, Weibull, Log-normal): Assume a fixed distributional form, lacking flexibility and suffering performance degradation when real data deviates from the assumptions.
- **Semi-parametric methods** (Cox proportional hazards model, DeepSurv): Rely on the proportional hazards assumption, with reduced reliability under high censoring rates.
- **Non-parametric methods** (DeepHit, CQRNN): Flexible but produce discrete or piecewise-constant estimates, making it difficult to extract continuous distribution summaries (such as mean, median, and quantiles) and incurring high computational overhead.

The core motivation of this paper is: **Is it possible to find a distribution that offers both closed-form solutions (the advantage of parametric models) and sufficient flexibility (the advantage of non-parametric models)?** The asymmetric Laplace distribution (ALD) perfectly meets this demand.

## Method

### Asymmetric Laplace Distribution (ALD) Definition

The ALD is controlled by three parameters $(\theta, \sigma, \kappa)$, corresponding to location, scale, and asymmetry, respectively:

$$f_{\text{ALD}}(y;\theta,\sigma,\kappa) = \frac{\sqrt{2}}{\sigma}\frac{\kappa}{1+\kappa^2} \begin{cases} \exp\left(\frac{\sqrt{2}\kappa}{\sigma}(\theta-y)\right), & y \geq \theta \\ \exp\left(\frac{\sqrt{2}}{\sigma\kappa}(y-\theta)\right), & y < \theta \end{cases}$$

Its CDF also has a closed-form expression, allowing the mean, median, mode, variance, and arbitrary quantiles to be calculated **analytically**.

Key property: Through the reparameterization $q = \kappa^2/(\kappa^2+1)$, the ALD can be naturally linked to quantile regression.

### Neural Network Architecture

The model adopts an architecture consisting of a shared encoder and three independent prediction heads:

- **Shared Encoder**: Fully connected layers with ReLU activation to extract covariate features.
- **Three Output Heads**: Predict $\theta$, $\sigma$, and $\kappa$ respectively, all utilizing an Exp activation to ensure non-negativity.
- **Residual Connections**: Enhance gradient flow and training stability.

### Maximum Likelihood Learning

The likelihood is constructed based on whether the event is observed:

$$-\mathcal{L}_{\text{ALD}} = \sum_{n \in \mathcal{D}_O} \log f_{\text{ALD}}(y_n|\mathbf{x}_n) + \sum_{n \in \mathcal{D}_C} \log S_{\text{ALD}}(y_n|\mathbf{x}_n)$$

where $\mathcal{D}_O$ represents the set of observed event times ($e=1$), $\mathcal{D}_C$ represents the set of censored times ($e=0$), and $S_{\text{ALD}} = 1 - F_{\text{ALD}}$ is the survival function.

**Key advantages over CQRNN**:
- CQRNN requires setting pseudo-values $y^* = 1.2 \max_i y_i$ (which are data-sensitive) and defining an approximate censoring quantile $q_c$ (where precision is limited by grid granularity).
- The proposed method directly maximizes the survival probability without requiring extra hyperparameters.
- It yields a continuous parametric distribution rather than discrete quantile points.

### Comparison with CQRNN

CQRNN is based on the pinball loss of the Portnoy estimator, which requires predefining a quantile grid $q=\{0.1, 0.2, \ldots, 0.9\}$ and estimating $\theta_q$ individually. In contrast, the proposed method only needs to learn three parameters to obtain the complete distribution, bypassing the difficulties of quantile grid selection and pseudo-value tuning.

## Key Experimental Results

### Dataset Configurations

The model is evaluated on 14 synthetic datasets and 7 real-world datasets, covering different censoring ratios (0.20 to 0.80) and feature dimensions (1 to 14). The real-world datasets span multiple domains such as oncology and cardiology:

| Dataset | No. of Features | Train Set | Test Set | Censoring Ratio |
|--------|--------|--------|--------|----------|
| METABRIC | 9 | 1523 | 381 | 0.42 |
| WHAS | 6 | 1310 | 328 | 0.57 |
| SUPPORT | 14 | 7098 | 1775 | 0.32 |
| GBSG | 7 | 1785 | 447 | 0.42 |
| TMBImmuno | 3 | 1328 | 332 | 0.49 |
| BreastMSK | 5 | 1467 | 367 | 0.77 |
| LGGGBM | 5 | 510 | 128 | 0.60 |

### Overall Performance Summary (21 datasets × 9 metrics = 189 comparisons)

| Baseline | Significantly Win | Significantly Lose | Draw |
|----------|----------|----------|------|
| vs. DeepHit | **113** | **22** | 54 |
| vs. LogNorm MLE | **113** | **6** | 70 |
| vs. DeepSurv | **77** | **27** | 85 |
| vs. CQRNN | **43** | **22** | 124 |

### Key Findings

- **IBS Metric**: ALD **wins in all cases** against LogNorm, DeepSurv, and DeepHit across the 21 datasets, and achieves 19 wins and 1 loss against CQRNN.
- **Calibration Metrics**: In terms of slope/intercept calibration for $S(t|\mathbf{x})$ and $f(t|\mathbf{x})$, ALD comprehensively outperforms traditional parametric and non-parametric methods.
- **Discriminative Metrics** (C-Index): The primary advantage of ALD is demonstrated in comparison with DeepHit (15 wins, 0 losses), while showing comparable performance to CQRNN.
- All experiments were replicated 10 times, with significance determined using the Wilcoxon signed-rank test.

## Highlights & Insights

1. **Elegant Mathematical Framework**: By leveraging the closed-form PDF/CDF of the ALD, the proposed work returns survival analysis to the clean form of classic parametric models while maintaining sufficient flexibility.
2. **Practical Value of Continuous Distribution**: Unlike DeepHit (which discretizes) and CQRNN (which outputs quantile points), ALD yields a continuous distribution, allowing direct calculation of the mean, median, variance, and arbitrary quantiles.
3. **Minimal Hyperparameters**: Eliminates the need to tune extra parameters such as quantile grids or pseudo-values, significantly reducing tuning difficulty.
4. **Outstanding Calibration**: Performance is particularly robust on distribution-level calibration metrics, indicating that ALD indeed fits the shape of real survival distributions more accurately.
5. **Unified Treatment of Censoring Types**: Can be adapted to other censoring types, such as left-censoring, by simply modifying the likelihood function.

## Limitations & Future Work

1. **Inherent Distribution Assumption**: Although ALD is more flexible than Normal or Weibull distributions, it still belongs to the family of unimodal distributions, which may feel inadequate when facing multimodal survival distributions.
2. **ALD Support Range Includes Negative Numbers**: $t < 0$ is meaningless in survival analysis; although the authors claim this rarely occurs in practice, it lacks theoretical guarantees.
3. **Evaluation Limited to Structured Data**: Has not been applied to high-dimensional unstructured covariates such as imaging or time series.
4. **Unstable MAE Metric**: Performance is mixed in terms of MAE compared to CQRNN/DeepSurv, indicating that the advantage in point prediction is less pronounced.
5. **Extension to Mixture Distributions**: Modeling a mixture of ALDs could count as a direction to handle more complex survival distributions.
6. **Competing Risks Scenario**: The current framework only processes single-event scenarios and has not been extended to multi-event competing risks settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying ALD to survival analysis is a novel entry point, though the link between ALD itself and quantile regression has been extensively studied.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Solid experimental design with 21 datasets, 9 metrics, 10 replications, and statistical testing.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations and an in-depth, thorough comparison with CQRNN.
- Value: ⭐⭐⭐⭐ — Provides a practical and elegant new parametric option for survival analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels](../../ECCV2024/social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](../../NeurIPS2025/social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[CVPR 2025\] Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)

</div>

<!-- RELATED:END -->
