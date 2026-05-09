---
title: >-
  [Paper Note] Competition-Aware CPC Forecasting with Near-Market Coverage
description: >-
  [CVPR 2026][Time Series][CPC forecasting] This paper reframes cost-per-click (CPC) forecasting in paid search advertising as a **partial competition observability** problem. By constructing three families of competition proxy signals — semantic neighborhood, DTW behavioral neighborhood, and geographic intent — and integrating them with temporal foundation models (Chronos-2/TimeGPT/Moirai) and spatiotemporal GNNs, the proposed framework achieves significant improvements in medium-to-long-term forecasting accuracy over 1,811 keyword time series.
tags:
  - CVPR 2026
  - Time Series
  - CPC forecasting
  - competition proxy
  - temporal foundation models
  - spatiotemporal graph neural networks
  - partial observability
date: 2026-05-08
content_hash: a3fb4ccff38b6043
---

# Competition-Aware CPC Forecasting with Near-Market Coverage

**Conference**: CVPR 2026
**arXiv**: [2603.13059](https://arxiv.org/abs/2603.13059)
**Code**: None
**Area**: Time Series Forecasting / Advertising Auctions
**Keywords**: CPC forecasting, competition proxy, temporal foundation models, spatiotemporal graph neural networks, partial observability

## TL;DR

This paper reframes cost-per-click (CPC) forecasting in paid search advertising as a **partial competition observability** problem. By constructing three families of competition proxy signals — semantic neighborhood, DTW behavioral neighborhood, and geographic intent — and integrating them with temporal foundation models (Chronos-2/TimeGPT/Moirai) and spatiotemporal GNNs, the proposed framework achieves significant improvements in medium-to-long-term forecasting accuracy over 1,811 keyword time series.

## Background & Motivation

In paid search advertising, CPC is an auction-generated price determined jointly by competitor bids, platform quality signals, and query-specific conditions. Advertisers face a fundamental **partial observability** problem:

- Advertisers can observe their own CPC, clicks, impressions, and spend.
- However, they **cannot** directly observe competitors' bids, budget constraints, or quality scores.
- As a result, CPC series contain only indirect, noisy traces of the competitive environment.

Pure autoregressive forecasting methods perform poorly over medium-to-long horizons, when competitive dynamics shift and demand patterns change. While the auction mechanism literature is extensive, it offers limited guidance on forecasting under **indirect competitive observability**. This paper proposes constructing "competition proxies" from observable signals to approximate latent competitive states.

## Method

### Overall Architecture

The paper constructs a **competition-aware forecasting design space** that operationalizes latent competition through three families of proxy signals, evaluated along two pathways: (1) as exogenous covariates fed directly into forecasting models; and (2) as relational priors encoded via semantic adjacency matrices into spatiotemporal graph neural networks. The forecasting targets are weekly CPC at horizons $h \in \{1, 6, 12\}$ weeks.

### Key Designs

1. **Semantic Neighborhood and Semantic Graph**: Each keyword $k_i$ is encoded into an embedding $e_i \in \mathbb{R}^{384}$ using a pretrained Transformer (all-MiniLM-L6-v2), and semantically related keywords are identified via cosine similarity. The semantic neighborhood serves a dual purpose: providing neighborhood competition covariates and defining a fixed semantic keyword graph $A^{sem} \in \mathbb{R}^{N \times N}$ ($k=10$ nearest neighbors, row-normalized).

2. **DTW Behavioral Neighborhood**: Textual similarity does not exhaust competitive relatedness — keywords with weak lexical overlap may exhibit similar CPC dynamics due to shared demand shocks. **Dynamic Time Warping (DTW)** with a Sakoe-Chiba band constraint is used to measure CPC trajectory similarity, constructing behavioral neighborhoods and deriving leakage-free behavioral competition features.

3. **Geographic Intent Proxy**: In the car rental market, auction pressure is highly geographically localized. Geographic intent (continent, country, and city levels) is extracted from keyword text, and each keyword is assigned structured location indicators. Multi-level geographic resolution is retained rather than assuming finer granularity is always superior.

### Forecasting Architectures

Three broad model families are evaluated:

- **Classical/Neural Baselines**: SARIMAX, XGBoost, Random Forest, LightGBM, MLP, LSTM, GRU, TabPFN
- **Covariate-Augmented Temporal Foundation Models (TSFMs)**: Chronos-2, TimeGPT, Moirai — with competition proxies as exogenous covariates
- **Spatiotemporal Graph Neural Networks (STGNNs)**: DCRNN (diffusion convolutional RNN), GConvLSTM (graph convolutional LSTM), GraphWaveNet (adaptive graph structure) — consuming the semantic graph $A^{sem}$

STGNN inputs are spatiotemporal tensors $X \in \mathbb{R}^{N \times T \times F}$, with multi-step predictions $\hat{Y}_{t+h} \in \mathbb{R}^{N}$, trained with MAE loss for robustness to heavy-tailed distributions.

### Loss & Training

- Target variable definition: $\text{cpc\_week}_{k,t} = \frac{\text{adcost\_sum}_{k,t}}{\text{adclicks\_sum}_{k,t}}$
- Strict temporal split: the last 20% of data is held out as the test set to prevent temporal leakage.
- All neighborhood features are constructed from historical data in a leakage-free manner.
- Evaluation metrics: sMAPE (primary) and RMSE (secondary).

## Key Experimental Results

### Main Results (6-Week Horizon)

| Model Family | Architecture | Competition Augmentation | sMAPE (%) | RMSE |
|---|---|---|---|---|
| Statistical/ML Baselines | SARIMAX | Univariate lags | 43.93±23.55 | 1.660±1.759 |
| Statistical/ML Baselines | XGBoost | Core operational features | 36.64±17.51 | 1.301±1.119 |
| Statistical/ML Baselines | TabPFN | Core operational features | 35.04±17.77 | 1.250±1.133 |
| Covariate TSFM | Moirai | Leakage-free lags + calendar stabilization | 30.14±18.24 | 1.000±0.970 |
| Covariate TSFM | TimeGPT | Calendar conditioning + growth clamping | 29.29±17.07 | 1.002±1.008 |
| Covariate TSFM | **Chronos-2** | **Geographic intent covariates** | **27.14±15.04** | **0.841±0.846** |
| Spatiotemporal GNN | GraphWaveNet | Semantic graph + search mix | 30.57±20.57 | 1.005±0.941 |
| Spatiotemporal GNN | GConvLSTM | Semantic graph + continent geography | 30.69±20.42 | 1.001±0.955 |
| Spatiotemporal GNN | DCRNN | Semantic graph + geo + semantic CPC | 30.42±20.42 | 1.000±0.926 |

### Cross-Horizon Summary

| Model Family | 1-Week sMAPE | 6-Week sMAPE | 12-Week sMAPE |
|---|---|---|---|
| Best Baseline | 30.42 | 35.04 | 40.23 |
| Best Covariate TSFM | 27.94 | **27.14** | **29.14** |
| Best Spatiotemporal GNN | **25.82** | 30.42 | 37.46 |

### Key Findings

- **Optimal method varies by horizon**: At 1 week, STGNNs perform best (25.82%); at 6 and 12 weeks, covariate-augmented TSFMs dominate (27.14% / 29.14%).
- **Coarse-grained geography is the most robust competition prior**: Continent-level encoding consistently improves stability across all backbones and horizons; finer country/city-level encoding disperses the signal.
- **Feature stacking is harmful**: At the 6-week horizon, naively stacking all proxies yields the worst performance (34.0% sMAPE), 3.3 percentage points below the optimal selective configuration.
- **Competitive frontier analysis**: Improvement is most pronounced on high-CPC, high-volatility keywords (402 keywords), where the Core + Geo + Sem CPC configuration reduces error by 1.3 percentage points in this regime.

## Highlights & Insights

- **Problem reframing**: Recasting CPC forecasting from "time series extrapolation" to "forecasting under partial competitive observability" constitutes the paper's central conceptual contribution.
- **Selectivity over exhaustiveness**: The value of competition proxies lies in selective combination rather than indiscriminate stacking — consistent with general principles of feature engineering.
- **Foundation models + domain priors**: Chronos-2, augmented only with coarse geographic intent covariates, outperforms all methods at medium-to-long horizons, underscoring the substantial potential of combining pretrained large models with domain-specific priors.
- The notion of the **competitive frontier** is highly instructive: attention to the **distribution** of forecasting errors, rather than average performance alone, reveals meaningful structure.

## Limitations & Future Work

- Validation is limited to a single vertical domain (the European car rental market) with high competitive concentration; generalizability requires further investigation.
- The semantic graph is static and cannot capture dynamic evolution of keyword relationships, including market entry and exit.
- Data reflects a single advertiser's perspective; "near-market coverage" remains substantially below full market observability.
- Dynamic graph construction and richer competitor-side signals remain unexplored.
- Weekly aggregation likely smooths intraday bidding dynamics.

## Related Work & Insights

- **Chronos-2 [Amazon 2024]**: A foundation model for general-purpose time series forecasting, extending beyond univariate settings.
- **TimeGPT [Garza et al. 2024]**: A zero-shot temporal foundation model.
- **DCRNN [Li et al. 2018]**: Diffusion convolutional RNN; a seminal STGNN for traffic forecasting.
- **GraphWaveNet [2019]**: Spatiotemporal forecasting with adaptive and fixed graph structures.
- Insight: The competition proxy construction methodology is transferable to forecasting in any multi-party strategic interaction setting, such as ride-hailing platform pricing and e-commerce auction bidding.

## Rating

| Dimension | Rating |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[CVPR 2026\] PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)
- [\[CVPR 2026\] L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanations.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[CVPR 2026\] Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)

</div>

<!-- RELATED:END -->
