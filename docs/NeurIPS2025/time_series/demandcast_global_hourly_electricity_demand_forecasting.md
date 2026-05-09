---
title: >-
  [Paper Note] DemandCast: Global hourly electricity demand forecasting
description: >-
  [NeurIPS 2025][Time Series][Electricity demand forecasting] DemandCast is an open-source machine learning framework that leverages XGBoost to integrate historical electricity demand, ERA5 temperature data, and socioeconomic features for hourly electricity demand forecasting across 56 countries/regions worldwide. By normalizing the target variable as a fraction of annual demand, the framework achieves cross-country comparability and attains a MAPE of 9.2% on a temporally held-out test set.
tags:
  - NeurIPS 2025
  - Time Series
  - Electricity demand forecasting
  - XGBoost
  - global scale
  - ERA5 meteorological data
  - normalized target
date: 2026-05-08
content_hash: 49b42fcbc401a576
---

# DemandCast: Global hourly electricity demand forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2510.08000](https://arxiv.org/abs/2510.08000)
**Code**: [GitHub](https://github.com/open-energy-transition/demandcast)
**Area**: Time Series Forecasting / Energy
**Keywords**: Electricity demand forecasting, XGBoost, global scale, ERA5 meteorological data, normalized target

## TL;DR

DemandCast is an open-source machine learning framework that leverages XGBoost to integrate historical electricity demand, ERA5 temperature data, and socioeconomic features for hourly electricity demand forecasting across 56 countries/regions worldwide. By normalizing the target variable as a fraction of annual demand, the framework achieves cross-country comparability and attains a MAPE of 9.2% on a temporally held-out test set.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: The global energy transition (decarbonization) requires accurate spatiotemporal forecasting of electricity demand to support renewable energy integration and grid management. Future electricity consumption patterns, however, are subject to substantial uncertainty driven by population growth, economic development, urbanization, and technological change. This uncertainty is particularly pronounced in Global South countries, which face the dual challenge of expanding electricity access while reducing carbon emissions.

Shortcomings of existing methods: (1) most are limited to a single country or a small number of regions; (2) data coverage is temporally narrow (e.g., using only 2015 data); (3) end-to-end open-source reproducible pipelines are lacking.

**Key Insight**: DemandCast constructs a large-scale forecasting framework covering 56 countries/regions over the period 2000–2025, employing a normalization design that enables the model to generalize across countries with heterogeneous data.

## Method

### Overall Architecture

DemandCast is a modular, end-to-end pipeline comprising: (1) data collection and cleaning (electricity demand, meteorological, and socioeconomic data); (2) feature engineering (temperature features, temporal features, socioeconomic features); (3) XGBoost model training and inference; and (4) post-processing to recover absolute demand values.

### Key Designs

1. **Normalized Target Variable**:
    - **Function**: Normalizes hourly electricity demand as a fraction of annual total demand.
    - **Mechanism**: $D_n(t) = \frac{D(t)}{D_Y} \cdot \frac{\sum_Y H_{\text{avail}}}{\sum_Y H}$; the model predicts the temporal distribution profile of demand rather than its absolute magnitude.
    - **Design Motivation**: Absolute demand levels differ by orders of magnitude across countries, making direct modeling of absolute values impractical for cross-country generalization. Normalization allows the model to focus on capturing diurnal, weekly, and seasonal patterns.

2. **Multi-source Feature Fusion**:
    - **Function**: Integrates ERA5 reanalysis temperature data with socioeconomic indicators such as GDP and population density.
    - **Mechanism**: Temperature features are extracted from the 1–3 most densely populated grid points; monthly mean temperatures and monthly rankings are constructed to capture seasonal variation. Per-capita GDP and per-capita electricity consumption serve as country-level features.
    - **Design Motivation**: Electricity demand is strongly correlated with temperature (cooling/heating loads), while socioeconomic factors determine the baseline demand level.

3. **Temporal Split Evaluation Strategy**:
    - **Function**: For each region, the final year is reserved as the test set, the penultimate year as the validation set, and the remainder as the training set.
    - **Mechanism**: Strict chronological partitioning (train 81.25% / val 9.84% / test 8.91%) ensures that evaluation measures genuine out-of-sample extrapolation to future periods.
    - **Design Motivation**: Prevents data leakage, a common issue in time series evaluation.

### Loss & Training

- Model: XGBoost (gradient-boosted trees)
- Training observations: 6,041,222 hourly records
- Validation observations: 731,538
- Test observations: 662,369

## Key Experimental Results

### Main Results

| Metric | Value |
|--------|-------|
| Mean MAPE (test set) | **9.2%** |
| Countries/regions covered | 56 |
| Time span | 2000–2025 |
| Training samples | 6,041,222 |
| Prior work MAPE | ~8% (smaller scale) |

### Performance on Selected Regions

| Region | MAPE (test) | Region | MAPE (test) |
|--------|------------|--------|------------|
| Spain (ES) | 2.34% | Germany (DE) | 6.74% |
| Brazil South (BR_S) | 8.00% | Japan Kanto (JP) | 8.53% |
| South Korea (KR) | 8.00% | Italy (IT) | 10.19% |
| Albania (AL) | 15.39% | Mexico North (MX_NOR) | 18.90% |

### Key Findings

- Forecasting errors are consistently lower for developed countries (with richer data) than for developing countries.
- Temperature features are the most important predictors, particularly in regions with strong heating or cooling demand.
- Higher MAPE values in certain regions (e.g., Albania 15%, Alberta, Canada 19%) are attributable primarily to insufficient training data or abrupt structural shifts in demand patterns.

## Highlights & Insights

- **Scale Novelty**: The first open-source framework for global hourly electricity demand forecasting covering 56 countries/regions over a 25-year span.
- **Elegant Normalization Design**: Recasting the problem from "predicting absolute demand" to "predicting the temporal distribution of demand" substantially improves cross-country generalization.
- **Fully Open and Reproducible**: The end-to-end pipeline—from data collection to forecasting—is openly available, offering direct practical value to the energy planning community.

## Limitations & Future Work

- Prediction uncertainty is not quantified, yet energy planning decisions require confidence intervals around forecasts.
- XGBoost is not benchmarked against deep learning methods (e.g., Transformers, N-BEATS).
- Data availability is highly uneven across regions; some regions have only a few years of training data.
- Demand-side management and the effects of extreme weather events are not accounted for.
- Systematic hyperparameter optimization has not been performed.

## Related Work & Insights

- **GlobalEnergyGIS (Mattsson et al.)**: Prior work covering 44 countries but relying solely on 2015 data.
- **ERA5 Reanalysis Dataset**: Provides globally consistent, high-resolution meteorological data.
- **Implications for the Energy Transition**: DemandCast can support planning for renewable energy integration and grid capacity expansion.

## Rating

- Novelty: ⭐⭐⭐ Methodologically straightforward (XGBoost), but novel in scale and open-source framework contribution.
- Experimental Thoroughness: ⭐⭐⭐ Full coverage across 56 countries, but lacks model comparison baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise with well-presented data.
- Value: ⭐⭐⭐⭐ A practical open-source tool for the energy planning community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs](nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](../../CVPR2026/time_series/stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[NeurIPS 2025\] BubbleFormer: Forecasting Boiling with Transformers](bubbleformer_forecasting_boiling_with_transformers.md)

<!-- RELATED:END -->
