---
title: >-
  [Paper Note] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales
description: >-
  [NeurIPS 2025][Time Series][Weather Forecasting] OmniCast is proposed as a weather forecasting method that combines a masked generative framework with a latent diffusion model. By jointly generating future weather sequen…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Weather Forecasting"
  - "Latent Diffusion Model"
  - "Masked Generative Modeling"
  - "Subseasonal Forecasting"
  - "VAE"
date: 2026-05-08
content_hash: 7de08abcff94baf6
---

# OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales

**Conference**: NeurIPS 2025
**arXiv**: [2510.18707](https://arxiv.org/abs/2510.18707)  
**Code**: [GitHub](https://github.com/tung-nd/omnicast)  
**Area**: Weather Forecasting, Generative Models
**Keywords**: Weather Forecasting, Latent Diffusion Model, Masked Generative Modeling, Subseasonal Forecasting, VAE

## TL;DR

OmniCast is proposed as a weather forecasting method that combines a masked generative framework with a latent diffusion model. By jointly generating future weather sequences rather than iterating autoregressively, it mitigates error accumulation, achieves state-of-the-art performance at the subseasonal-to-seasonal (S2S) scale, remains competitive for medium-range forecasting, and offers inference speeds 10–20× faster.

## Background & Motivation

Weather forecasting spans multiple time scales. Medium-range forecasting (~2 weeks) has been substantially advanced by deep learning methods (e.g., PanguWeather, GraphCast surpassing the IFS numerical system), yet **subseasonal-to-seasonal (S2S, 2–6 weeks)** forecasting remains a major challenge.

S2S forecasting is difficult for three reasons:

- **Error accumulation**: Existing methods are predominantly autoregressive; errors amplify progressively over short time-step iterations, and multi-step fine-tuning incurs prohibitive computational costs for the long sequences involved in S2S.
- **Initial conditions vs. boundary conditions**: Short-range forecasting relies primarily on initial conditions, whereas S2S must also account for boundary conditions (e.g., sea surface temperature, soil moisture) that autoregressive short-step training cannot capture.
- **Uncertainty quantification**: S2S forecasting inherently requires probabilistic approaches, yet ensemble system size is constrained by computational cost.

## Method

### Overall Architecture

OmniCast employs a two-stage training procedure:

1. **Stage 1 — VAE Encoder**: Compresses raw weather data $X \in \mathbb{R}^{V \times H \times W}$ (with $V$ physical variables) into a continuous low-dimensional latent token map of size $h \times w$, encoding each frame independently.
2. **Stage 2 — Masked Generative Transformer**: Models the conditional distribution $p(\mathbf{x} | \mathbf{c})$ of future token sequences in the latent space, using a diffusion head to handle continuous tokens.

### Key Designs

1. **Continuous VAE over Discrete VQ-VAE**: Weather data contains hundreds of physical variables; discrete quantization at a compression ratio of ~3938× causes severe reconstruction error. The continuous VAE uses $D=16$-dimensional vectors at a compression ratio of only 100×, substantially reducing information loss. Spatial downsampling is 16× ($128 \times 256 \to 8 \times 16$).

2. **Masked Generative Modeling**: During training, a random subset of future tokens is masked, and the model learns to recover masked tokens conditioned on the context tokens (initial states) and visible tokens. The masking ratio is sampled as $\gamma \sim \mathcal{U}[0.5, 1.0]$. During inference, the model begins from a fully masked state and progressively reveals tokens according to a cosine schedule. This joint generation strategy avoids autoregressive error accumulation and enables the model to capture long-range spatiotemporal dependencies.

3. **Per-Token Diffusion Head**: The Transformer backbone outputs a conditioning vector $z_i$ at each position; a lightweight MLP diffusion network conditioned on $z_i$ estimates the token distribution. The diffusion head uses 6 residual blocks (width 2048) with AdaLN to incorporate diffusion step embeddings. Training uses a 1000-step linear noise schedule; inference uses 100 steps via rescheduling. A key efficiency advantage is that the backbone requires only a single forward pass, with diffusion steps handled by the lightweight MLP.

4. **Auxiliary Deterministic Objective**: An MSE loss with exponentially decaying weights is applied to the first 10 frames, reflecting the fact that weather dynamics remain largely deterministic within 10 days. Beyond this horizon, weather becomes chaotic and imposing MSE is detrimental. The total objective is $\mathcal{L} = \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{deter}}$.

### Implementation Details

- Transformer: MAE encoder-decoder architecture, each with 16 layers × 16 heads and hidden dimension 1024.
- Training: 32 × A100 GPUs for 4 days (far less than Gencast's 32 TPUv5e × 5 days).
- S2S setting: $T=44$ steps (days 1–44), $\tau=1.3$, 50-member ensemble.
- Medium-range setting: 2-step prediction (12h interval), autoregressive sampling, $\tau=1.0$.

## Key Experimental Results

### S2S Deterministic Metrics (ERA5, 1.4° resolution, test year 2022)

| Method | Type | T850 RMSE Trend | Z500 RMSE Trend | Bias |
|--------|------|----------------|----------------|------|
| PanguWeather | DL (autoregressive) | Good short-term; sharp degradation >15 days | Same | Large bias |
| GraphCast | DL (autoregressive) | Good short-term; degradation >10 days | Same | Large bias |
| ECMWF-ENS | Numerical ensemble | Strong throughout | Strong throughout | Moderate bias |
| **OmniCast** | DL (masked diffusion) | Slightly weaker short-term; matches ECMWF >10 days | Same | **Lowest bias, near-zero** |

OmniCast matches or surpasses ECMWF-ENS at lead times >10 days and is the only method maintaining near-zero bias throughout the full forecast horizon.

### S2S Physical Consistency Metrics

| Method | Spectral Divergence (SDIV) | Spectral Residual (SRES) |
|--------|--------------------------|--------------------------|
| PanguWeather | Poor (severe spectral distortion) | Poor |
| GraphCast | Poor | Poor |
| ECMWF-ENS | Moderate | Moderate |
| **OmniCast** | **Best** (best spectral preservation) | **Best** |

OmniCast significantly outperforms other DL methods in physical consistency, frequently surpassing all numerical baselines as well.

### Inference Efficiency

| Method | Hardware | 15-day forecast (0.25°) | 15-day forecast (1.0°) |
|--------|----------|------------------------|------------------------|
| Gencast | TPUv5 | 480 s | 224 s |
| **OmniCast** | A100 | **29 s** | **11 s** |
| IFS-ENS | CPU cluster | ~hours | — |

OmniCast achieves inference speeds **10–20× faster** than Gencast, on less powerful hardware.

### Ablation Study

| Ablation | Short-term RMSE | S2S RMSE | CRPS | SSR |
|----------|----------------|---------|------|-----|
| Full OmniCast | Moderate | **Best** | **Best** | **Best** |
| w/o MSE objective | Poor (noticeable short-term degradation) | Poor | Poor | Near |
| MSE on all frames | Good (short-term) | Poor (harmful deterministic forcing) | Poor | Poor |
| Short-sequence training ($T<44$) | **Best** (short-term) | Poor (error accumulation) | Poor | Poor |
| Autoregressive unmasking | Near | Near | Near | Poor (under-dispersed) |
| $\tau=1.0$ | Near | Near | Near | Poor (under-dispersed) |
| $\tau=1.3$ | Near | **Best** | **Best** | **Best** |

### Key Findings

- **Full-sequence training is critical for S2S performance**: Short-sequence training yields better medium-range results but degrades substantially on S2S due to autoregressive error accumulation.
- **MSE is beneficial only for the first 10 frames**: This is consistent with the ~10-day deterministic predictability limit of atmospheric dynamics.
- **Random unmasking order outperforms autoregressive and frame-level ordering**: Additional stochasticity yields more diverse ensemble members, with SSR closer to 1.
- **Diffusion temperature $\tau=1.3$ offers the best trade-off**: Lower values lead to under-dispersion; higher values deviate from mean predictions.
- **Stable 100-year rollout**: OmniCast can produce stable simulated outputs over a 100-year horizon.

## Highlights & Insights

- Masked generative modeling (MaskGIT) and video generation (MAR) paradigms from computer vision are successfully transferred to weather forecasting.
- The combination of a continuous VAE and a diffusion head elegantly addresses the encoding challenge posed by high-dimensional weather data (100+ variables).
- The low-dimensional latent space combined with a lightweight diffusion MLP achieves order-of-magnitude inference acceleration.
- A single model unifies medium-range and S2S forecasting, remaining competitive at both time scales.
- Near-zero bias and strong spectral preservation are critically important properties for physical science applications.

## Limitations & Future Work

- Short- and medium-range forecasting still lags behind specialized autoregressive methods (e.g., Gencast), revealing a trade-off between short/medium-range and long-range performance.
- S2S experiments are conducted at relatively low resolution (1.4°); whether the advantage holds at higher resolutions requires verification.
- The trade-off between VAE reconstruction quality and Transformer modeling capacity has not been thoroughly investigated.
- Direct comparison with Gencast and NeuralGCM on S2S was not possible due to their prohibitive computational requirements.
- Temporal compression was not applied (preliminary experiments showed no clear benefit); further exploration is warranted.

## Related Work & Insights

- Autoregressive methods such as FourCastNet, GraphCast, and Stormer have achieved breakthroughs in medium-range forecasting but are limited at S2S scales due to error accumulation.
- ChaosBench provides a systematic benchmarking framework for S2S comparisons.
- The success of masked generative models (MaskGIT, MAR) in image and video generation offers cross-domain methodological inspiration for weather forecasting.
- Inference efficiency limitations in Gencast's diffusion approach constitute the primary motivation for OmniCast's improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The masked latent diffusion architecture unifying medium-range and S2S forecasting represents a genuinely novel approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers medium-range/S2S, deterministic/probabilistic/physical metrics, comprehensive ablations, and efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear motivation, complete methodological derivation, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Makes an important contribution to the meteorology and AI for Science communities, with open-source code and models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rotary Masked Autoencoders are Versatile Learners](rotary_masked_autoencoders_are_versatile_learners.md)
- [\[ICML 2026\] Latent Laplace Diffusion for Irregular Multivariate Time Series](../../ICML2026/time_series/latent_laplace_diffusion_for_irregular_multivariate_time_series.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)

</div>

<!-- RELATED:END -->
