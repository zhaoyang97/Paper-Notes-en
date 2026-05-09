---
title: >-
  [Paper Note] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition
description: >-
  [AAAI2026][Time Series] This work is the first to introduce Deep Evidential Regression (DER) with a Normal-Inverse-Gamma prior into a time series foundation model architecture, enabling epistemic-aleatoric uncertainty decomposition in a single forward pass. The practical value of uncertainty-aware trading strategies is validated on cryptocurrency forecasting.
tags:
  - AAAI2026
  - Time Series
  - foundation model
  - uncertainty quantification
  - deep evidential regression
  - financial forecasting
date: 2026-05-08
content_hash: ecb029db8d1fc5e7
---

# ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition

**Conference**: AAAI2026  
**arXiv**: [2601.10591](https://arxiv.org/abs/2601.10591)  
**Code**: To be confirmed  
**Area**: Time Series  
**Keywords**: time series, foundation model, uncertainty quantification, deep evidential regression, financial forecasting

## TL;DR
This work is the first to introduce Deep Evidential Regression (DER) with a Normal-Inverse-Gamma prior into a time series foundation model architecture, enabling epistemic-aleatoric uncertainty decomposition in a single forward pass. The practical value of uncertainty-aware trading strategies is validated on cryptocurrency forecasting.

## Background & Motivation
- Time Series Foundation Models (TSFMs) demonstrate strong zero-shot forecasting performance, yet lack principled uncertainty quantification in high-stakes scenarios such as finance.
- Limitations of prior work:
    - Mixture models (MOIRAI): rely on predefined distributional components and cannot distinguish epistemic from aleatoric uncertainty.
    - Student-t distribution (Lag-Llama): impose strong distributional assumptions that may not generalize across diverse time series characteristics.
    - Conformal Prediction (TimeGPT): post-hoc calibration that is not integrated into the learning process.
- Architectural heterogeneity makes it difficult to attribute performance gains to uncertainty quantification methods versus architectural advantages.

## Core Problem
1. How to achieve principled epistemic-aleatoric uncertainty decomposition within TSFMs?
2. How to provide complete uncertainty quantification without sacrificing point forecast accuracy?
3. How to fairly evaluate the contribution of uncertainty quantification strategies in isolation from architectural differences?

## Method

### Overall Architecture
ProbFM = Adaptive Patching + Transformer Backbone + DER Head, comprising six components: input processing, Transformer representation learning, DER uncertainty estimation, composite loss, single-stage training with evidence annealing, and single-pass inference.

### Key Designs

**1. Normal-Inverse-Gamma (NIG) Prior**

- Models the parameters of the predictive distribution rather than directly parameterizing the distribution: $p(\mu, \sigma^2) = \text{NIG}(\mu, \lambda, \alpha, \beta)$
- Explicit uncertainty decomposition:
    - Aleatoric: $\mathbb{U}_{\text{aleatoric}} = \frac{\beta}{\alpha - 1}$ (intrinsic data noise)
    - Epistemic: $\mathbb{U}_{\text{epistemic}} = \frac{\beta}{(\alpha-1)\lambda}$ (model uncertainty, reducible with more data)

**2. DER Head Parameter Projection**

- The Transformer output $h$ is projected to four NIG parameters: $\mu$ is unconstrained; $\lambda, \beta$ are kept positive via Softplus + $\epsilon$; $\alpha$ is constrained to $> 1$ via Softplus + 1 + $\epsilon$.

**3. Augmented Loss Function**

- Evidential loss: $\mathcal{L}_{\text{EDL}} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{evd}} \mathcal{L}_{\text{reg}}$
- Coverage loss: $\mathcal{L}_{\text{coverage}} = |\text{PICP}_{\text{target}} - \text{PICP}_{\text{actual}}|$, directly optimizing prediction interval coverage rate.
- Full objective: $\mathcal{L}_{\text{ProbFM}} = \mathcal{L}_{\text{EDL}} + \lambda_{\text{coverage}} \cdot \mathcal{L}_{\text{coverage}} + \lambda_{\text{wd}} \|\theta\|_2^2$

**4. Evidence Annealing**

- $\text{evidence\_scale}(t) = \min(1.0, t / T_{\text{anneal}})$, preventing overconfidence in early training.
- Unlike the KL regularization annealing of Sensoy et al., this approach directly controls the evidence accumulation process.

**5. Controlled Experimental Design**

- All methods uniformly employ a 1-layer LSTM (32 hidden dims) as the backbone.
- Only the loss function and output head are varied, isolating the contribution of uncertainty quantification strategies.

## Key Experimental Results

| Method | RMSE | MAE | Characteristics |
|--------|------|-----|-----------------|
| MSE Baseline | 0.044 | 0.030 | No probabilistic output |
| Gaussian NLL | 0.044 | 0.029 | Total variance only |
| Student-t NLL | 0.045 | 0.030 | Heavy-tail modeling |
| Quantile Loss | 0.044 | 0.029 | Quantile-based intervals |
| **Evidential (ProbFM)** | **0.045** | **0.030** | **Epistemic + aleatoric decomposition** |

- Forecast accuracy: DER achieves parity with other methods (RMSE 0.045 vs. baseline 0.044), demonstrating that uncertainty quantification does not sacrifice accuracy.
- Uncertainty-aware trading: filtering high-uncertainty predictions via epistemic/aleatoric thresholds improves risk-adjusted returns.
- Portfolio optimization: uncertainty-based position sizing outperforms the equal-weight baseline.

## Highlights & Insights
- First application of DER + NIG prior to the TSFM architecture, filling the gap in uncertainty decomposition for time series foundation models.
- The controlled experimental design (fixed LSTM backbone) rigorously isolates the contribution of uncertainty quantification methods.
- Coverage loss directly optimizes prediction interval coverage without requiring post-hoc calibration.
- Complete uncertainty quantification is obtained in a single forward pass, offering high computational efficiency.
- Financial application validation (trading filtering + portfolio optimization) demonstrates practical decision-making value.

## Limitations & Future Work
- Validation is limited to cryptocurrency daily return data; multi-domain (energy, transportation, weather) and multi-frequency experiments are absent.
- The controlled experiment uses a 1-layer LSTM (32 dims) with minimal model capacity; validation at true foundation model scale has not been conducted.
- Only univariate single-step forecasting is supported; multi-step and multivariate extensions (NIW prior) are mentioned only as future work.
- Although evidence annealing mitigates the evidence collapse problem in DER, theoretical guarantees remain insufficient.
- End-to-end comparisons against real TSFMs such as MOIRAI and Lag-Llama are not performed.

## Related Work & Insights
- vs **MOIRAI**: MOIRAI employs a 4-component mixture distribution requiring multi-component sampling; ProbFM uses a single forward pass but lacks MOIRAI's multivariate multi-step capability.
- vs **Lag-Llama**: Lag-Llama assumes a Student-t distribution as a single distributional family; ProbFM learns a distribution over distribution parameters via NIG.
- vs **TimeGPT**: TimeGPT's conformal prediction performs post-hoc calibration; ProbFM integrates uncertainty into the training process.
- vs **Standard Bayesian / MC Dropout**: ProbFM uses a single forward pass, avoiding the overhead of multiple sampling runs.

### Further Implications
- The epistemic-aleatoric decomposition of DER has a natural application in active learning: samples with high epistemic uncertainty should be prioritized for annotation.
- The coverage loss approach is generalizable to calibration in any probabilistic forecasting model.
- The evidence annealing strategy provides a useful reference for other evidential learning tasks (classification, object detection).
- The direction of TSFM + uncertainty decomposition warrants further exploration at larger scales.

## Rating
- Novelty: ⭐⭐⭐⭐ (First application of DER to TSFMs, though DER itself is not a novel method)
- Experimental Thoroughness: ⭐⭐⭐ (Controlled experimental design is sound, but data and model scale are insufficient)
- Writing Quality: ⭐⭐⭐⭐ (Methodological exposition is clear with a solid theoretical foundation)
- Value: ⭐⭐⭐ (The direction is meaningful, but the experimental scale limits persuasiveness)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [\[AAAI 2026\] Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths](interpreting_fedspeak_with_confidence_a_llm-based_uncertainty-aware_framework_gu.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](../../ICLR2026/time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](../../ICLR2026/time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)

<!-- RELATED:END -->
