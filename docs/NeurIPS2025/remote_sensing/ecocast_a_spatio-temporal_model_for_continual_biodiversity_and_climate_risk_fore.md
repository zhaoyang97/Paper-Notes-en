---
title: >-
  [Paper Note] EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting
description: >-
  [NeurIPS 2025][Remote Sensing][Biodiversity Forecasting] This paper proposes EcoCast, a Transformer-based spatio-temporal sequence model that integrates satellite remote sensing (Sentinel-2), climate reanalysis (ERA5)…
tags:
  - "NeurIPS 2025"
  - "Remote Sensing"
  - "Biodiversity Forecasting"
  - "Species Distribution Modeling"
  - "Transformer"
  - "Continual Learning"
  - "EWC"
date: 2026-05-08
content_hash: a6d111cfd3c2c90f
---

# EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2512.02260](https://arxiv.org/abs/2512.02260)  
**Code**: None  
**Area**: Time Series / Remote Sensing / Ecological Conservation
**Keywords**: Biodiversity Forecasting, Species Distribution Modeling, Transformer, Continual Learning, EWC

## TL;DR

This paper proposes EcoCast, a Transformer-based spatio-temporal sequence model that integrates satellite remote sensing (Sentinel-2), climate reanalysis (ERA5), and citizen science observations (GBIF). The model predicts next-month species occurrence probabilities from 12-month environmental feature sequences. On a five-species African bird distribution prediction task, the macro-average F1 score improves from 0.31 (Random Forest) to 0.65. An EWC-based continual learning framework is also designed to accommodate data updates.

## Background & Motivation

**Background**: Species Distribution Modeling (SDM) is a core tool in conservation biology. However, traditional SDMs are inherently static—they fit present-day environment–species relationships and rely on future climate scenario projections (e.g., RCP4.5/8.5) for decadal-scale predictions, making them incapable of providing operational forecasts at monthly to seasonal timescales.

**Limitations of Prior Work**: (1) Traditional SDMs cannot track rapidly changing environmental conditions, limiting their timeliness for conservation decision-making. (2) Methods such as Random Forest treat each (location, month) observation as an independent sample, failing to capture temporal autocorrelation, lagged environmental responses, and seasonal periodicity. (3) Monitoring data in biodiversity hotspots such as Africa are updated infrequently, causing deployed models to become outdated rapidly.

**Key Challenge**: Conservation managers require timely biodiversity risk forecasts analogous to weather predictions, yet existing tools can only provide long-term static estimates dependent on climate projections.

**Key Insight**: The paper draws on the paradigm of operational short-range forecasting in meteorology—directly predicting the near future from observational history—and applies Transformer-based modeling to environmental time series. ERA5 climate data become available in a preliminary version within five days of observation, making monthly forecast updates feasible.

**Core Idea**: A Transformer is applied to model multi-source environmental feature sequences spanning 12 months to predict next-month species occurrence probabilities, thereby eliminating dependence on future climate scenario assumptions.

## Method

### Overall Architecture

**Input**: A sequence of environmental feature vectors for each 0.1° grid cell over 12 consecutive months, $\mathbf{x}_{t-11:t} \in \mathbb{R}^{12 \times F}$, comprising Sentinel-2 band statistics and ERA5 climate variables. **Output**: Next-month species occurrence probability $y_{t+1}$. Training covers 2016–2021, fine-tuning uses 2022, and evaluation is performed on 2023.

### Key Designs

1. **Sequence-to-Point Transformer Forecasting Architecture**:

    - **Function**: Predicts next-month species occurrence from a 12-month environmental time series.
    - **Mechanism**: A Transformer encoder processes monthly environmental vector sequences of length $L=12$. Multi-head self-attention automatically learns seasonal vegetation cycles, lagged climate effects (e.g., changes in food availability 2–4 months after rainfall), and annual migratory cycles. Training objective: $\min_\theta \sum \mathcal{L}(f_\theta(\mathbf{x}_{t-L+1:t}), y_{t+1})$, where $\mathcal{L}$ is a class-imbalance-robust loss.
    - **Design Motivation**: Unlike the independent-observation assumption of Random Forest, self-attention explicitly models temporal dependencies, which is essential for capturing avian phenological patterns.

2. **Multi-Source Data Fusion and Preprocessing**:

    - Sentinel-2 monthly composite imagery: band statistics and vegetation indices (EVI, NDWI, NDMI, NBR) computed within 0.1° grids.
    - ERA5 climate variables: temperature, relative humidity, total precipitation, wind speed, and surface pressure.
    - GBIF bird observations: aggregated monthly to 0.1° grids (~10 km); spatial thinning applied to reduce clustering bias.
    - Pseudo-absence generation (presence-only data) and sampling effort covariates to distinguish true absences from undetected occurrences.

3. **EWC Continual Learning Framework**:

    - Constrained update upon arrival of new data: $\min_{\theta'} \sum_k \mathcal{L}(f_{\theta'}(x_{t+k}), y_{t+k}) + \lambda \Omega_{\text{EWC}}(\theta', \theta)$
    - EWC regularization penalizes deviations from important parameters, preventing catastrophic forgetting.
    - A fixed-size replay buffer is maintained for rehearsal.
    - Rolling-origin validation is used in current evaluations to simulate operational conditions.
    - **Design Motivation**: Ecological environments are non-stationary; the model must adapt to new data without forgetting historical patterns.

### Loss & Training

A class-imbalance-robust loss is employed. Stratified mini-batches balance presence/pseudo-absence samples per species. Spatio-temporal block cross-validation (spatial blocks × held-out years) prevents data leakage. F1 thresholds are selected per species on the validation set.

## Key Experimental Results

### Main Results (2023 holdout, 5 African bird species)

| Model | F1 macro | PR AUC macro |
|-------|----------|-------------|
| RF-ROE (Random Forest + Rolling-Origin Evaluation) | 0.31 | 0.29 |
| **EcoCast (t+1 forecast)** | **0.65** | **0.72** |

F1 improves by +34 percentage points; PR-AUC improves by +43 percentage points.

### Ablation Study

| RF Limitation | Corresponding EcoCast Design |
|--------------|------------------------------|
| Independent monthly predictions; no temporal autocorrelation | 12-month sequence input; self-attention models temporal dependencies |
| Cannot capture lagged environmental responses | Sequence modeling automatically learns 2–4 month lag effects |
| Requires manual construction of seasonal features | Positional encoding automatically encodes seasonal periodicity |
| Per-species independent training | Joint multi-label training shares cross-species ecological signals |

### Key Findings

- The primary driver of the doubled F1 score is explicit temporal modeling, not additional data.
- The 12-month window covers a full annual cycle, which is critical for avian phenology.
- Joint multi-label training outperforms single-species models, as different species share underlying environmental response patterns.
- The five focal species span distinct ecological niches: endangered species (African Grey Parrot), migratory species (African Pitta), and widespread species (Green Wood Hoopoe), among others.

## Highlights & Insights

- **Transfer of the Operational Forecasting Paradigm**: Introducing the meteorological concept of "predicting the near future from observational history" into ecology constitutes an important complement to the traditional SDM paradigm of "long-term estimation based on climate projections." This approach is extensible to invasive species spread, disease transmission risk, and related domains.
- **Natural Fit of Transformers for Ecological Time Series**: Seasonality, migratory patterns, and environmental lag effects in avian phenology are precisely the long-range dependencies that self-attention mechanisms are well-suited to capture.
- **Unique Value of the African Perspective**: Most ecological AI work focuses on European or North American datasets; this paper specifically targets Africa, a biodiversity hotspot with limited data availability.

## Limitations & Future Work

- **Limited Experimental Scale**: Only a pilot study covering five bird species; statistical significance is insufficient.
- **Weak Baselines**: Comparison is limited to Random Forest; spatio-temporal deep learning baselines such as ConvLSTM and EarthFormer are absent.
- **Continual Learning Not Empirically Evaluated**: The EWC framework is designed but not tested in an incremental data scenario.
- **Coarse Spatial Resolution**: The 0.1° grid (~10 km) cannot capture microhabitat variation.
- **Presence-Only Data Bias**: Pseudo-absence strategies may be unreliable in observation-sparse regions.
- Detailed ablation studies (e.g., removing individual data sources, varying window length) are lacking.

## Related Work & Insights

- **vs. Traditional SDMs (MaxEnt/GAM)**: Static models vs. dynamic time series; EcoCast requires no climate projections, but traditional SDMs offer stronger interpretability.
- **vs. EarthFormer**: Also uses Transformers for spatio-temporal data modeling but targets meteorological forecasting; EcoCast extends this to ecological prediction.
- **vs. ConvLSTM**: Better suited for capturing spatial structure; EcoCast currently relies on tabular grid statistics and does not exploit spatial neighborhood information.

## Rating

- Novelty: ⭐⭐⭐ Transformer + EWC is not novel, but introducing the operational forecasting paradigm into ecology is a meaningful contribution.
- Experimental Thoroughness: ⭐⭐ Only 5 bird species, only RF as baseline, continual learning not evaluated.
- Writing Quality: ⭐⭐⭐ Motivation is clear, but method and experiment descriptions are overly brief.
- Value: ⭐⭐⭐ The direction is meaningful, but the work resembles a proof-of-concept rather than a mature system.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD](../../AAAI2026/remote_sensing/spatio-temporal_context_learning_with_temporal_difference_convolution_for_moving.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)

</div>

<!-- RELATED:END -->
