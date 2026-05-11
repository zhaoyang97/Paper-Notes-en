---
title: >-
  [Paper Note] Power Ensemble Aggregation for Improved Extreme Event AI Prediction
description: >-
  [NeurIPS 2025][Time Series][extreme event prediction] This paper proposes an adaptive ensemble aggregation method based on the power mean. By applying nonlinear aggregation (power exponent $p>1$) to the score of ensemble…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "extreme event prediction"
  - "ensemble aggregation"
  - "power mean"
  - "heat wave classification"
  - "climate prediction"
date: 2026-05-08
content_hash: 8ef595897a647126
---

# Power Ensemble Aggregation for Improved Extreme Event AI Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2511.11170](https://arxiv.org/abs/2511.11170)
**Code**: Unavailable
**Area**: Time Series
**Keywords**: extreme event prediction, ensemble aggregation, power mean, heat wave classification, climate prediction

## TL;DR

This paper proposes an adaptive ensemble aggregation method based on the power mean. By applying nonlinear aggregation (power exponent $p>1$) to the score of ensemble members from generative weather prediction models, the method significantly improves classification performance for extreme high-temperature events, with greater gains at higher quantile thresholds.

## Background & Motivation

Deep learning weather forecasting models (e.g., GraphCast, PanguWeather) have surpassed physics-based models on routine weather prediction, yet **extreme event prediction remains a critical weakness**. The root causes are as follows:

**Rarity**: Extreme events are, by definition, low-probability occurrences, occupying a negligible fraction of training data; models consequently tend to predict outcomes near the mean.

**Inherent bias of mean aggregation**: When ensemble models are used, the standard practice of averaging member predictions naturally **suppresses extreme values**—even if a minority of members correctly predict an extreme event, their signals are diluted by the conservative majority.

**Lack of adaptive mechanisms**: Extreme events of varying intensity require different degrees of "bias toward extremes," yet traditional aggregation methods (mean or maximum) lack this flexibility.

Core motivation: **Between the mean (overly conservative) and the maximum (overly aggressive), does an optimal intermediate strategy exist?** The power mean provides a continuously tunable compromise.

## Method

### Overall Architecture

The system consists of three components: (1) a U-Net-based deterministic weather prediction model employing a cubed-sphere grid to avoid polar singularities; (2) Perlin noise injection that converts the model into a generative one, producing $n=50$ ensemble members; and (3) power mean aggregation that transforms ensemble member scores into extreme event classification probabilities.

### Key Designs

1. **Extreme Event Definition and Classification Framework**

   Extreme events are defined using local climatology: for each location and season, the local climatological mean and standard deviation of surface air temperature are computed, and temperature is converted into a local anomaly $x$. A threshold $q$ is defined such that a temperature anomaly $x$ satisfying $\Phi(x) \geq q$ is classified as an extreme event, where $\Phi$ is the standard normal CDF.

   This is a critical design choice—using a local rather than a global threshold avoids the limitation of detecting only perennially hot regions such as the Sahara Desert.

2. **Power Mean Aggregation**

   Given local anomaly predictions $\{\hat{x}_i\}_{i=1}^n$ from $n$ ensemble members, the score for each member is first computed as $\hat{s}_i = \Phi(\hat{x}_i)$, and power mean aggregation is then applied:

   $$\hat{s} = \left(\frac{1}{n}\sum_{i=1}^n \hat{s}_i^p\right)^{1/p}$$

   When $p=1$, this reduces to the arithmetic mean; as $p \to \infty$, it approaches the maximum. The parameter $p \geq 1$ controls the degree to which the aggregation is skewed toward extreme-predicting members.

   Note that the power mean is applied to the scores $\hat{s}_i$ (which are positive) rather than to the anomalies $\hat{x}_i$ (which may be negative), as the power operation requires positive inputs.

3. **Generative Model Construction**

   Perlin noise is injected into the inputs of the deterministic U-Net baseline to introduce ensemble diversity. Perlin noise offers a spatial coherence advantage over white noise—perturbations to weather fields should be spatially smooth rather than pixel-wise independent.

   Specific improvements include: generating Perlin noise on a 3D cube $[0,1]^3$ and taking the 2D slice corresponding to the Earth's surface to ensure global continuity; randomizing gradient magnitudes via a log-normal distribution to better capture extreme values (standard Perlin noise is constrained to $[-1,1]$); and combining noise at different frequencies into fractal noise via amplitude modulators learned through convolutional layers.

### Loss & Training

The Continuous Ranked Probability Score (CRPS) is used as the training loss. Training data consists of ERA5 reanalysis from 1990–2010, resampled to 1.5° spatial resolution and daily temporal resolution. A cubed-sphere grid ($6\times48\times48$) is used to avoid polar singularities. The model is trained on a single 16 GB GPU within a few hours.

## Key Experimental Results

### Main Results — AUC Comparison Across Quantiles

| Quantile $q$ | Lead Time | Mean Aggregation AUC | Power Mean AUC | $p_{opt}$ | Relative Improvement RI |
|---|---|---|---|---|---|
| 0.80 | 7 days | Baseline | Marginally better | ~2 | ~0.5% |
| 0.90 | 7 days | Baseline | Better | ~5 | ~1.0% |
| 0.98 | 7 days | Baseline | Significantly better | ~20 | ~2.5% |
| 0.80 | 12 days | Baseline | Better | ~2 | ~1% |
| 0.98 | 12 days | Baseline | Substantially better | ~20 | ~4% |

### Comparison with GraphCast (Test Set: 2018)

| Method | $q=0.80$ | $q=0.90$ | $q=0.98$ |
|---|---|---|---|
| Persistence model | Low | Low | Low |
| GraphCast (deterministic) | Good | Good | Limited |
| Ensemble mean | Good | Good | Good |
| **Ensemble power mean** | **Best** | **Best** | **Best** |

At high quantiles ($q=0.98$) and long lead times, power mean aggregation applied to a simple generative model even surpasses deterministic GraphCast.

### Key Findings

1. **The optimal power exponent $p_{opt}$ scales exponentially with quantile $q$**: $\log(p_{opt}) = f(q)$ is nearly perfectly linear, providing a simple extrapolation rule across quantiles.
2. **Improvement increases with event extremity**: The relative improvement RI grows as the quantile threshold increases, demonstrating that the method is most effective for the most extreme events.
3. **Improvement increases with lead time**: Greater uncertainty at longer forecast horizons amplifies the benefit of biasing toward extreme ensemble members via the power mean.
4. The $p_{opt}$ optimized on the validation set transfers directly to different forecast lead times, indicating good robustness.

## Highlights & Insights

1. **Elegant simplicity**: Introducing a single hyperparameter $p$ achieves a continuously tunable aggregation strategy spanning from the mean to the maximum.
2. **Exponential scaling law of $p_{opt}$**: Provides an off-the-shelf guide for selecting $p$ at different extremity thresholds.
3. **Model-agnostic applicability**: Power mean aggregation can be applied to any generative forecasting model without modifying the model architecture.
4. **Demonstrating that simplicity can outperform complexity**: A simple generative model combined with power mean aggregation surpasses the sophisticated GraphCast on extreme events.

## Limitations & Future Work

- **Simplified extreme event definition**: Only a single variable (surface air temperature) and a static climatology are used.
- **Climate change not considered**: Anomalies defined relative to a fixed climatology may be unreliable under nonstationary climate conditions.
- **Limitations of the AUC metric**: Application-oriented evaluation incorporating socioeconomic loss is absent.
- **Only tested on a simple self-built model**: Whether the approach is effective for stronger baselines (e.g., GenCast) remains unverified.
- Multivariate extreme events (e.g., compound extreme weather) are not addressed.

## Related Work & Insights

- **GraphCast**: DeepMind's deterministic weather forecasting model, used as the comparison baseline in this paper.
- **WeatherBench2**: A general-purpose weather forecasting benchmark and data source.
- **Perlin noise**: A coherent noise method from computer graphics, creatively repurposed here for meteorological ensemble perturbation.

## Rating

- Novelty: ⭐⭐⭐☆☆ — The power mean has precedents; the contribution lies in its systematic validation in the climate domain.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Multi-quantile and multi-lead-time comparisons are thorough, though stronger baseline models are not evaluated.
- Writing Quality: ⭐⭐⭐⭐☆ — Concise and focused.
- Value: ⭐⭐⭐⭐☆ — The method is simple and general, with practical relevance to extreme event prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[AAAI 2026\] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs](../../AAAI2026/time_series/gaico_a_deployed_and_extensible_framework_for_evaluating_diverse_and_multimodal_.md)
- [\[AAAI 2026\] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](../../AAAI2026/time_series/detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)

</div>

<!-- RELATED:END -->
