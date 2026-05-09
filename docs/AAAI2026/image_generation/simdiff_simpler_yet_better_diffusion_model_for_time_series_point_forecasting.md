---
title: >-
  [Paper Note] SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting
description: >-
  [AAAI 2026][Image Generation][Time Series Forecasting] This paper proposes SimDiff — the first purely end-to-end diffusion model to achieve state-of-the-art performance on time series point forecasting. A unified Transformer network serves simultaneously as denoiser and predictor. Combined with Normalization Independence for distribution shift handling and a Median-of-Means ensemble strategy that converts probabilistic samples into precise point predictions, SimDiff achieves 1st place on 6 and 2nd place on 3 out of 9 benchmarks.
tags:
  - AAAI 2026
  - Image Generation
  - Time Series Forecasting
  - Diffusion Models
  - Transformer
  - Normalization Independence
  - Median-of-Means
  - End-to-End
date: 2026-05-08
content_hash: 8a7807d7dee02af9
---

# SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting

**Conference**: AAAI 2026
**arXiv**: [2511.19256](https://arxiv.org/abs/2511.19256)
**Code**: [Available](https://github.com/Dear-Sloth/SimDiff)
**Area**: Diffusion Models / Time Series Forecasting / Point Forecasting
**Keywords**: Time Series Forecasting, Diffusion Models, Transformer, Normalization Independence, Median-of-Means, End-to-End

## TL;DR

This paper proposes SimDiff — the first purely end-to-end diffusion model to achieve state-of-the-art performance on time series point forecasting. A unified Transformer network serves simultaneously as denoiser and predictor. Combined with Normalization Independence for distribution shift handling and a Median-of-Means ensemble strategy that converts probabilistic samples into precise point predictions, SimDiff achieves 1st place on 6 and 2nd place on 3 out of 9 benchmarks.

## Background & Motivation

Diffusion models have demonstrated strong potential for probabilistic modeling in time series forecasting, yet they consistently lag behind regression-based methods in **point prediction accuracy**. Two core contradictions are identified:

**Insufficient contextual bias**: Distribution drift commonly exists between the historical and future portions of a time series. Pure likelihood-based methods (e.g., TimeGrad, CSDI) cannot track this drift, leading to unstable training and exploding sampling variance — "diversity so high as to be useless."

**Diversity–accuracy trade-off**: TimeDiff and mr-Diff stabilize training by providing initial trajectories via pre-trained regressors, but this rigidifies the diffusion process and constrains generative flexibility, effectively reducing these models to regression approaches.

Key question: **Can a purely end-to-end diffusion model be designed that neither relies on external pre-trained models nor sacrifices point prediction accuracy, and that instead leverages the intrinsic diversity of diffusion models to improve it?**

## Method

### Overall Architecture

SimDiff is a single-stage end-to-end framework with three core components:
- **Patch-based Transformer denoising network**: serves simultaneously as denoiser and predictor
- **Normalization Independence (N.I.)**: past and future are normalized independently during training; only past statistics are used at inference
- **Median-of-Means (MoM) ensemble**: robust point predictions are obtained from multiple inference samples via the MoM estimator

### Key Design 1: Normalization Independence

Conventional approaches use statistics $(\mu_X, \sigma_X)$ from the historical sequence $\mathbf{X}$ to normalize both $\mathbf{X}$ and $\mathbf{Y}$. Under distribution shift between past and future:

$$\mathbf{Y}_{\text{norm}} = \frac{\mathbf{Y} - \mu_X}{\sigma_X} = \frac{\sigma_Y \cdot \mathbf{Y}_{\text{real\_norm}}}{\sigma_X} + \frac{\mu_Y - \mu_X}{\sigma_X}$$

The second term $(\mu_Y - \mu_X)/\sigma_X$ constitutes an irremovable bias.

N.I. addresses this as follows:
- **During training**: $\mathbf{X}$ is normalized via learnable affine parameters $(\gamma, \beta)$; $\mathbf{Y}$ is independently normalized using its **own** statistics $(\mu_Y, \sigma_Y)$
- **During inference**: denoising starts from standard Gaussian noise, and the output is denormalized using $(\mu_X, \sigma_X, \gamma, \beta)$

This eliminates the bias term at training time, aligning the learned denoising target more closely with the standard Gaussian prior. The learnable parameters $(\gamma, \beta)$ enable the model to infer scale and shift changes in the future from the historical sequence.

### Key Design 2: Transformer Denoising Network

A carefully designed lightweight Transformer with several key choices:
- **Patch tokenization**: the time series is divided into overlapping patches as tokens to capture local dependencies
- **RoPE positional encoding**: encodes relative positional information, enhancing attention to temporal patterns; ablation studies demonstrate improvements of 8.3% on NorPool and 1.7% on ETTh1
- **Channel independence**: each channel is processed independently, increasing effective data volume and simplifying learning
- **No skip connections**: unlike U-ViT, skip connections are found to amplify noise and disturb the diffusion distribution in the time series setting

### Key Design 3: Median-of-Means Ensemble

Diffusion models naturally produce diverse probabilistic samples, but extreme values are unavoidable. The MoM estimator proceeds as follows:
1. Draw $n$ samples, partition them into $K$ groups of $B = n/K$ each
2. Compute the mean within each group: $\hat{\mu}_1, \ldots, \hat{\mu}_K$
3. Take the median of these $K$ group means
4. Repeat the shuffle-and-group procedure $R$ times, and average the $R$ resulting medians

$$\hat{\mu}_{\text{MoM}} = \frac{1}{R}\sum_{r=1}^{R} \text{median}(\hat{\mu}_1^{(r)}, \ldots, \hat{\mu}_K^{(r)})$$

Compared to simple averaging, MoM preserves subtle temporal patterns rather than smoothing out high-frequency details, while being more robust to outliers. Theoretically, MoM provides tighter finite-sample concentration bounds.

### Loss & Training

A weighted MAE loss (rather than standard MSE) is used, normalized by the cumulative noise schedule:

$$L(\theta) = \min_\theta \mathbb{E}_{Y^0, \epsilon, k} \left| \frac{Y^0 - Y_\theta(Y^k, k|c)}{\sqrt{1 - \alpha_{\text{cumprod}}[k]}} \right|$$

The denominator $\sqrt{1 - \alpha_\text{cumprod}[k]}$ ensures that higher-noise timesteps contribute stronger learning signals.

## Key Experimental Results

### Main Results: Multivariate Point Forecasting MSE (Selected from Table 2)

| Method | NorPool | Electricity | Traffic | ETTh1 | ETTm1 | Avg. Rank |
|--------|---------|-------------|---------|-------|-------|-----------|
| **SimDiff** | **0.534**(1) | **0.145**(1) | 0.383(2) | **0.394**(1) | **0.322**(1) | **1.33** |
| PatchTST | 0.547(2) | 0.147(2) | 0.385(3) | 0.405(2) | 0.337(3) | 3.22 |
| mr-Diff | 0.645(4) | 0.155(5) | 0.474(8) | 0.411(5) | 0.340(4) | 4.00 |
| TimeDiff | 0.665(6) | 0.193(7) | 0.564(10) | 0.407(3) | 0.336(2) | 5.67 |
| TMDM | 0.681(8) | 0.267(14) | 0.513(9) | 0.535(13) | 0.436(14) | 12.00 |
| TimeGrad | 1.152(22) | 0.736(23) | 1.745(24) | 0.993(24) | 0.874(23) | 21.89 |

SimDiff achieves an average rank of **1.33** among 25 methods, ranking 1st on 6 of 9 datasets.

### Ablation Study: Component Contributions (Tables 4 & 5)

| Component | ETTh1 | Weather | NorPool | Impact |
|-----------|-------|---------|---------|--------|
| Full SimDiff | 0.394 | 0.299 | 0.534 | — |
| w/o N.I. | 0.405(+2.8%) | 0.328(+9.7%) | 0.555(+3.9%) | Significant |
| w/o RoPE | 0.401(+1.8%) | 0.310(+3.7%) | 0.582(+9.0%) | Significant |
| 1 inference (no ensemble) | 0.408 | 0.317 | 0.548 | MoM reduces error by 3.4–6% |
| Simple average (non-MoM) | 0.398 | 0.305 | — | MoM outperforms averaging |

### Key Findings

1. **Overwhelming inference efficiency**: SimDiff requires only 0.22–0.46 ms per inference on ETTh1, compared to 67–380 ms for CSDI and 295–2312 ms for TimeGrad — **100–5000× faster**
2. **Sound distributional modeling**: SimDiff also achieves state-of-the-art probabilistic metrics (CRPS: Electricity 0.22, Traffic 0.16), indicating that its point prediction capability stems from high-quality distributional modeling
3. **N.I. yields the largest gains on datasets with severe distribution shift** (Weather, NorPool), consistent with the design motivation
4. **MoM vs. simple averaging**: MoM consistently outperforms simple averaging across all datasets, as averaging smooths out high-frequency temporal patterns
5. **End-to-end vs. pre-trained conditioning**: SimDiff exhibits higher but more meaningful sampling variance — pre-trained conditioning models constrain the exploration space of diffusion

## Highlights & Insights

1. **First purely end-to-end diffusion model to achieve SOTA on time series point forecasting**: challenges the prevailing paradigm that diffusion-based time series forecasting necessarily requires a pre-trained regressor
2. **N.I. is simple yet effective**: requires only a lightweight affine layer with negligible computational overhead, yet substantially improves adaptation to distribution shift
3. **MoM bridges probabilistic and point forecasting**: elegantly transforms the probabilistic sampling advantage of diffusion models into precise point predictions with theoretical guarantees
4. **Decisive speed advantage**: single-pass inference is over 1000× faster than TimeGrad; even with 100 samples plus MoM aggregation, SimDiff remains faster

## Limitations & Future Work

1. **Multiple inference passes required**: although each forward pass is extremely fast, tens to hundreds of samples are typically needed for robust MoM estimation
2. **Channel independence assumption**: may discard inter-variable correlations, potentially underperforming in strongly coupled multivariate settings
3. **Removal of skip connections**: while validated for time series, this limits the model's ability to capture multi-scale features
4. **Unevaluated on other modalities**: the framework is designed for time series; generalization to spatial data, images, and other modalities remains unexplored
5. **MoM hyperparameter sensitivity**: the optimal choices of group count $K$ and repetition count $R$ still require manual tuning

## Related Work & Insights

| Method | Stage | Pre-training Required | Inference Speed | Point Quality | Probabilistic Quality |
|--------|-------|-----------------------|-----------------|---------------|-----------------------|
| TimeGrad | 1 | ✗ | Very slow | Poor | Moderate |
| CSDI | 1 | ✗ | Very slow | Poor | Moderate |
| TimeDiff | 2 | ✓ (DLinear) | Moderate | Good | — |
| mr-Diff | 2 | ✓ (DLinear) | Moderate | Good | — |
| TMDM | 2 | ✓ (Autoformer) | Slow | Moderate | Good |
| **SimDiff** | **1** | **✗** | **Very fast** | **Best** | **Good** |

SimDiff is the only diffusion model that achieves SOTA point forecasting performance without relying on any pre-trained model.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (unique combination of end-to-end diffusion point forecasting, N.I., and MoM)
- **Technical Contribution**: ⭐⭐⭐⭐ (multiple carefully designed components, each supported by ablation studies)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (25 baselines, 9 datasets, dual evaluation of probabilistic and point forecasting)
- **Writing Quality**: ⭐⭐⭐⭐ (problem motivation is clear, though the main text is somewhat redundant)
- **Practical Impact**: ⭐⭐⭐⭐ (establishes a new paradigm for diffusion models in time series forecasting)
- **Overall Recommendation**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective](tsgdiff_rethinking_synthetic_time_series_generation_from_a_pure_graph_perspectiv.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)

</div>

<!-- RELATED:END -->
