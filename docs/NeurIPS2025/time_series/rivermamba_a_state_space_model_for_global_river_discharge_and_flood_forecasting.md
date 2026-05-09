---
title: >-
  [Paper Note] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
description: >-
  [NeurIPS 2025][Time Series][River discharge forecasting] The first deep learning model capable of 7-day river discharge forecasting on a 0.05° (~5.5 km) global grid — global grid points are serialized via space-filling curves into 3D spatiotemporal point sequences fed into bidirectional Mamba blocks, driven by ECMWF HRES meteorological forecasts, achieving F1 = 0.459 on flood detection across 1.5–500-year return periods, surpassing LSTM (0.358) and the physical model GloFAS.
tags:
  - NeurIPS 2025
  - Time Series
  - River discharge forecasting
  - flood prediction
  - Mamba
  - spatiotemporal modeling
  - global scale
  - GloFAS
date: 2026-05-08
content_hash: 9fbc2065e386007f
---

# RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2505.22535](https://arxiv.org/abs/2505.22535)
**Code**: [Project](https://hakamshams.github.io/RiverMamba)
**Area**: Time Series
**Keywords**: River discharge forecasting, flood prediction, Mamba, spatiotemporal modeling, global scale, GloFAS

## TL;DR
The first deep learning model capable of 7-day river discharge forecasting on a 0.05° (~5.5 km) global grid — global grid points are serialized via space-filling curves into 3D spatiotemporal point sequences fed into bidirectional Mamba blocks, driven by ECMWF HRES meteorological forecasts, achieving F1 = 0.459 on flood detection across 1.5–500-year return periods, surpassing LSTM (0.358) and the physical model GloFAS.

## Background & Motivation
**Background**: Physics-based hydrological models (GloFAS by ECMWF) remain the standard for global flood forecasting, but are computationally prohibitive (requiring full hydrological cycle simulations on supercomputers). AI methods have made recent progress: Google's ED-LSTM performs well at the local catchment scale, and the NeuralHydrology line of work has established the effectiveness of LSTMs in hydrology.

**Limitations of Prior Work**: (a) LSTMs model each catchment independently, ignoring spatial relationships between catchments (upstream–downstream interactions); (b) GNNs capture spatial relationships on graphs but do not scale to global coverage (>100K active grid points); (c) Transformer's $O(n^2)$ complexity is infeasible for a global 0.05° grid (hundreds of thousands of grid points × multiple days).

**Key Challenge**: Global flood forecasting simultaneously requires modeling (i) long time series (30+ days of history), (ii) global spatial relationships (river network topology), and (iii) meteorological forcing (weather forecasts as future conditional inputs), yet no existing method addresses all three.

**Goal**: Construct the first AI river discharge forecasting system with global 0.05° resolution and a 7-day lead time.

**Key Insight**: Mamba's selective scan mechanism provides $O(n)$ linear complexity for processing very long sequences; space-filling curves losslessly encode 2D spatial grid points into 1D sequences for Mamba processing.

**Core Idea**: Serialize global grid points via space-filling curves → encode spatiotemporal context with bidirectional Mamba → drive day-by-day forecasts with ECMWF meteorological predictions.

## Method

### Overall Architecture
A two-stage Encoder–Decoder: (1) **Hindcast layers** (encoder): aggregate past $T=4$ days of ERA5 reanalysis data into a spatiotemporal representation; (2) **Forecast layers** (decoder): ingest ECMWF HRES meteorological forecasts day by day to autoregressively generate $L=7$ days of lead-time predictions.

### Key Designs

1. **Space-Filling Curve Serialization**:

    - Function: Converts global 3D spatiotemporal grid points $(lat, lon, time)$ into a 1D sequence via a bijective function $\Phi: \mathbb{Z}^3 \to \mathbb{N}$
    - Mechanism: Employs a Sweep + Gilbert composite curve, with different Mamba blocks using different curves — ensuring all spatial neighborhood relationships are covered
    - Design Motivation: Different curves preserve different spatial localities — Hilbert curves preserve 2D proximity, while Sweep curves preserve continuity along latitude/longitude directions

2. **Bidirectional Mamba + LOAN**:

    - Function: Bidirectional Mamba blocks perform bidirectional state-space modeling over serialized spatiotemporal features
    - LOAN (Location-Aware Adaptive Normalization): Injects static river attributes (catchment area, slope, etc.) into the normalization step of each layer via a FiLM mechanism
    - Design Motivation: River attributes are invariant physical priors (catchment size and slope gradient affect hydrological response time) and should serve as conditioning signals rather than input features

3. **Flood-Weighted Loss**:

    - Function: $\mathcal{L} = \text{MSE} \times w_{\text{flood}}$, weighted by return period — rare flood events receive substantially higher weights than normal flow conditions
    - Data Processing: Log transformation to handle the large dynamic range of discharge values (<1 m³/s to >10,000 m³/s)
    - Prediction Target: The model predicts increments $\Delta X$ rather than absolute values — mitigating distributional shift issues

### Loss & Training
Training data: GloFAS reanalysis from 1979–2018. Validation: 2019–2020. Testing: 2021–2024. GRDC in-situ observations are used for additional validation.

## Key Experimental Results

### Main Results (GloFAS Reanalysis Test Set 2021–2024)

| Metric | RiverMamba | LSTM | GloFAS (Physical) |
|--------|-----------|------|------------------|
| R² | **0.873 ± 0.001** | 0.849 ± 0.002 | — |
| KGE | **0.913 ± 0.001** | 0.892 ± 0.003 | — |
| F1 (Flood) | **0.459 ± 0.008** | 0.358 ± 0.006 | — |

### GRDC In-Situ Validation

| Metric | RiverMamba | LSTM |
|--------|-----------|------|
| R² | **0.506 ± 0.003** | 0.462 ± 0.004 |
| F1 (Flood) | **0.243 ± 0.011** | 0.148 ± 0.001 |

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|------------|-------|
| Lead time >48h | RiverMamba advantage far exceeds LSTM | Long-range forecasting benefits from spatial modeling |
| Return period 1.5–500 yr | Outperforms LSTM / GloFAS at all return periods | Critical improvement in extreme flood detection |
| LOAN vs. no static attributes | LOAN improves F1 | Physical priors provide additional value |
| Flood-weighted vs. uniform loss | Weighted loss significantly improves F1 | Rare events require additional attention |
| Sweep+Gilbert vs. single curve | Composite is superior | Different curves are complementary |

### Key Findings
- Flood F1 improves from 0.358 (LSTM) to 0.459 (+28%), representing the best reported result for global-scale AI flood forecasting
- Advantages are pronounced at lead times >48h — spatial modeling enables RiverMamba to exploit upstream information for longer-range predictions
- GRDC in-situ validation yields R² of only 0.506 — because in-situ observations are influenced by anthropogenic regulation (dams, reservoirs) that the model does not explicitly represent

## Highlights & Insights
- **Elegant application of Mamba to scientific computing**: Leveraging Mamba's $O(n)$ complexity to process global-scale spatiotemporal sequences is a highly natural design choice. The introduction of space-filling curves addresses the problem of encoding 2D spatial structure into 1D sequences.
- **Integration of physical priors**: LOAN injects static river attributes as normalization conditions rather than concatenating them to the input — a more principled approach than simple concatenation, since physical attributes govern the distribution of features rather than their values.
- **Practical significance of global coverage**: Seven-day global flood forecasting at 0.05° resolution is particularly valuable for developing countries lacking dense observation networks.

## Limitations & Future Work
- Anthropogenic water management (dam and reservoir operations) is not modeled, leading to noticeable performance degradation on GRDC in-situ data
- ERA5 reanalysis data has a 5-day latency; near-real-time data would be required for operational deployment
- Predictions of increments $\Delta X$ depend on the quality of the previous day's observed values
- The selection of space-filling curves is largely heuristic; the optimal combination has not been thoroughly explored
- Extreme flood events (500-year return period) are severely underrepresented in training data, and model uncertainty in the tail of the distribution remains substantial

## Related Work & Insights
- **vs. Google ED-LSTM (Nearing et al., 2024)**: ED-LSTM outperforms GloFAS at the local catchment scale but lacks global spatial modeling; RiverMamba is the first AI model to incorporate spatially correlated global modeling
- **vs. GloFAS (ECMWF)**: The physical model offers complete physical constraints but is computationally prohibitive; RiverMamba achieves inference several orders of magnitude faster
- **vs. ClimaX / Pangu-Weather**: These are global weather forecasting models; RiverMamba is a global hydrological forecasting model that uses weather forecasts as driving inputs

## Rating
- Novelty: ⭐⭐⭐⭐ First global AI river discharge forecasting model; the combination of Mamba and space-filling curves is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ GloFAS reanalysis + GRDC in-situ + multiple return periods + ablation study
- Writing Quality: ⭐⭐⭐⭐ Hydrological background is clearly presented; methodology is described systematically
- Value: ⭐⭐⭐⭐⭐ Direct and significant practical value for global flood early warning

2. **LOAN**:
    - Function: Injects static river attributes into feature representations
    - Mechanism: $LOAN(X) = (X-\mu)/\sigma + GELU(Linear(X_{static}))$

3. **Flood Severity-Weighted Loss**:
    - Function: Directs model attention toward rare but critical extreme flood events
    - Mechanism: Return-period weighting (100-year events weighted ×500) + exponential weighting by lead time

### Loss & Training
- Weighted MSE + log transformation; two-stage training (GloFAS pretraining + GRDC fine-tuning)

## Key Experimental Results

### Main Results

| Model | R² | KGE | F1 (Flood) |
|-------|----|-----|-----------|
| Persistence | Moderate | Moderate | Rapid decay |
| LSTM (Google) | Good | Good | Good |
| **RiverMamba** | **Best** | **Best** | **Best** |

Advantages are pronounced at lead times >48h.

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| w/o spatial modeling | Significant degradation | Spatiotemporal relationships are critical |
| w/o flood weighting | Extreme flood F1 drops | Weighting is essential for rare events |
| Transformer vs. Mamba | Mamba is faster and more accurate | Linear vs. quadratic complexity |

### Key Findings
- Spatiotemporal modeling is the primary source of advantage
- Successful forecasting of the 2024 German flood event (case study)

## Highlights & Insights
- First global 0.05° AI discharge forecasting model
- The space-filling curve approach is generalizable to other geoscience tasks

## Limitations & Future Work
- Only forecasts riverine flooding; does not cover pluvial flooding or storm surges
- Errors in HRES meteorological forecasts propagate into hydrological predictions

## Related Work & Insights
- **vs. Google ED-LSTM**: A local model. RiverMamba performs global spatiotemporal modeling
- **vs. GloFAS**: A physical model with high computational cost. RiverMamba achieves faster inference

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First global AI flood forecasting model + innovative application of Mamba in hydrology
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Global evaluation + 3,366 GRDC stations
- Writing Quality: ⭐⭐⭐⭐ Methodology described systematically
- Value: ⭐⭐⭐⭐⭐ Direct application value for disaster prevention and mitigation

### Supplementary Technical Details
- During training, $P$ points are sampled globally; during inference, arbitrary point sets or dense global grids can be used
- Discharge values are transformed via $sign(\Delta\hat{x})\log(1+|\Delta\hat{x}|)$ to handle the multi-order-of-magnitude dynamic range
- LOAN layers encode static river attributes (catchment morphology, slope, etc.) as location-aware biases
- Hindcast blocks use 3 layers ($T=4$, each with temporal downsampling ×2), with encoding dimension $K=192$
- Forecast blocks process $L=7$ days of HRES forecasts sequentially, with the $l$-th block processing the $l$-th day
- The Flash-Attention baseline is inferior to Mamba in both inference speed and accuracy (verified in the appendix)
- Training period: 1979–2018; validation: 2019–2020; testing: 2021–2024 (temporal out-of-distribution evaluation)
- Results at non-gauged locations are also provided in the appendix
- Flood return periods range from 1.5 to 500 years, covering the full spectrum from common to extreme flooding
- Comparison with the reforecast version of Google LSTM is also provided in the appendix

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)
- [\[NeurIPS 2025\] DemandCast: Global hourly electricity demand forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)
- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

<!-- RELATED:END -->
