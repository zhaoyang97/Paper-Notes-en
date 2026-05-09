---
title: >-
  [Paper Note] AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting
description: >-
  [AAAI2026][Time Series][air quality forecasting] The first framework to introduce Neural Delay Differential Equations (NDDE) into air quality forecasting. By incorporating a memory-augmented attention module and a physics-guided delay evolution function, it models delay effects in the continuous-time propagation of pollutants, achieving an average MAE reduction of 8.79% across three datasets.
tags:
  - AAAI2026
  - Time Series
  - air quality forecasting
  - neural delay differential equations
  - physics-guided
  - spatiotemporal graph
date: 2026-05-08
content_hash: 5f6ed315c09ef7e4
---

# AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting

**Conference**: AAAI2026  
**arXiv**: [2603.17529](https://arxiv.org/abs/2603.17529)  
**Code**: [github.com/w2obin/airdde-aaai](https://github.com/w2obin/airdde-aaai)  
**Area**: Time Series  
**Keywords**: air quality forecasting, neural delay differential equations, physics-guided, spatiotemporal graph  

## TL;DR

The first framework to introduce Neural Delay Differential Equations (NDDE) into air quality forecasting. By incorporating a memory-augmented attention module and a physics-guided delay evolution function, it models delay effects in the continuous-time propagation of pollutants, achieving an average MAE reduction of 8.79% across three datasets.

## Background & Motivation

Air quality forecasting is critical for public health and environmental sustainability, yet the complex dynamics of pollutants make accurate prediction highly challenging. Existing methods suffer from two key limitations:

1. **Discrete-time modeling**: Conventional STGNNs and attention-based methods model pollutant dynamics as discrete-time processes, failing to capture the continuous-time evolution of real-world pollutants.
2. **Neglect of propagation delays**: Even recent Neural ODE-based methods (e.g., AirPhyNet, AirDualODE) that advance modeling to continuous time still adopt an "instantaneous assumption"—system evolution depends only on the current state—ignoring the non-zero time delay required for pollutants to travel from a source to downstream regions.

In practice, delay effects are ubiquitous: **pollutants emitted at one location may take several hours to be transported by wind to a downstream area**, creating a non-negligible time lag between source and observed impact. Furthermore, delays are spatially heterogeneous—inter-station delays are jointly modulated by meteorological factors (wind speed and direction) and geographic distance. Existing NDDE methods can only model a globally uniform delay and cannot characterize this location-specific behavior.

## Core Problem

How to effectively model **heterogeneous delay effects modulated by multiple factors** within a continuous-time pollutant evolution framework? Specifically, three challenges must be addressed:

- Delays are dynamically modulated by wind fields, geographic distance, and other factors, rather than being fixed constants.
- The concentration at a given location and time is the cumulative superposition of pollutants arriving from multiple surrounding locations with different delays.
- This spatiotemporal accumulation effect is rooted in atmospheric dynamical processes, making it difficult for purely data-driven methods to capture effectively.

## Method

The overall architecture of AirDDE consists of four core components:

### 1. Spatiotemporal Encoder

A GNN-GRU architecture is adopted: an adaptive adjacency matrix $\boldsymbol{A}$ is constructed via learnable node embeddings $\boldsymbol{E}_1, \boldsymbol{E}_2$, and the MLPs in the GRU gates are replaced by GNNs to incorporate graph topology into temporal updates:

$$\boldsymbol{h}_e^t = \text{GNN-GRU}(\boldsymbol{X}^t, \boldsymbol{h}_e^{t-1}, \boldsymbol{A})$$

### 2. Diffusion-Advection Graph Construction

- **Diffusion graph** $\boldsymbol{A}_{\text{diff}}$: Built from Haversine geographic distances between stations with Gaussian kernel normalization, modeling diffusive transport under calm or weak-wind conditions.
- **Advection graph** $\boldsymbol{A}_{\text{adv}}^t$: **Dynamically constructed at each time step**—a directed edge is established if the wind speed and direction at location $j$ at time $t_2$ can transport an air mass to location $i$ by time $t_1$. The lag $\tau = t_1 - t_2$ corresponds to the pollutant transport time. This formulation is more flexible than the global uniform delay assumption.

### 3. Memory-Augmented Attention (MAA) Module

A dual-attention mechanism is designed to capture the dual-scale historical patterns of pollutant propagation:

- **Global memory modeling**: Learnable global memory units $\boldsymbol{M}_g \in \mathbb{R}^{m \times d_e}$ are introduced; the current feature $\boldsymbol{h}_e^t$ adaptively retrieves global historical patterns (e.g., persistently high PM2.5 regions) via an attention mechanism.
- **Local memory modeling**: A dynamic neighborhood $\mathcal{N}(i)^t$ is defined based on the advection graph; for each location $i$, attention-based aggregation is applied over the historical features of itself and its neighbors within the lag window $\tau$, capturing local transient events (e.g., abrupt AQI spikes caused by dust storms).

The encoder features, global memory features, and local memory features are concatenated and passed through an MLP to produce the delay-aware initial state $\boldsymbol{h}_m^t$.

### 4. Physics-guided Delay Evolution Function (PDE)

Using the diffusion-advection equation as a physical prior, the evolution function models three terms:

$$\frac{d\boldsymbol{h}^t}{dt} = \underbrace{D \cdot \text{GNN}_{\text{diff}}(\boldsymbol{A}_{\text{diff}}, \boldsymbol{h}^t)}_{\text{Diffusion term}} + \underbrace{\text{GNN}_{\text{adv}}(\boldsymbol{A}_{\text{adv}}^t, \boldsymbol{h}^{t-\tau})}_{\text{Delayed advection term}} + \underbrace{f(\boldsymbol{h}^t || \boldsymbol{M})}_{\text{Source/sink term}}$$

- **Diffusion term**: Approximates Chebyshev polynomial-based message passing via GNN on the diffusion graph.
- **Delayed advection term**: The key innovation—uses the **historical state** $\boldsymbol{h}^{t-\tau}$ rather than the current state, propagating it on the advection graph to explicitly model transport delay.
- **Source/sink term**: An MLP learns implicit sources (e.g., wind-driven inflow) and sinks (e.g., precipitation scavenging) from multifactor features.

The DDE solver maintains a historical state buffer and applies fourth-order Runge-Kutta integration to solve for future states. A GNN-GRU decoder then produces the final predictions. The training loss is Huber Loss, which is robust to outliers.

## Key Experimental Results

### Datasets

| Dataset | # Factors | # Stations | Time Range | Granularity |
|---------|-----------|------------|------------|-------------|
| KnowAir | 18 | 184 | 2015–2018 | 3h |
| China-AQI | 8 | 209 | 2017–2019 | 1h |
| US-PM | 8 | 175 | 2020–2021 | 1h |

### Main Results (vs. 19 baselines)

| Dataset | AirDDE MAE | 2nd-best MAE | MAE Reduction |
|---------|-----------|--------------|---------------|
| KnowAir | **16.92** | 18.64 (AirDualODE) | −9.23% |
| China-AQI | **17.03** | 18.89 (AirDualODE) | −9.85% |
| US-PM | **3.53** | 3.81 (PDFormer) | −7.3% |

The largest improvement is observed on China-AQI—the dataset with the finest temporal granularity, highest pollution levels, and most stations, yielding the most complex dynamics, where AirDDE's delay modeling advantage is most pronounced.

### Ablation Study (KnowAir, 3-day average MAE)

| Variant | AVG MAE | vs. Full Model |
|---------|---------|----------------|
| Full AirDDE | **16.92** | — |
| w/o MAA module | 19.16 | +13.2% |
| w/o global memory | 17.80 | +5.2% |
| w/o local memory | 17.44 | +3.1% |
| w/o PDE function | 19.39 | +14.6% |
| w/o source/sink term | 18.18 | +7.4% |
| Replace physics prior with attention | 18.78 | +11.0% |

Both the PDE function and the MAA module are critical components; the physics-guided variant consistently outperforms the purely data-driven variant.

### Robustness (KnowAir)

As data quality degrades (missing rate 10%→50%, noise 80dB→40dB), the MAE improvement of AirDDE over the second-best method expands from 9.23% to 15.42%, demonstrating that global memory and physics-guided evolution provide stronger recovery and denoising capabilities under degraded data conditions.

### Efficiency

On China-AQI, AirDDE requires 9.24 min/epoch training time and 10.46 GB GPU memory, both superior to the competing method AirDualODE (10.09 min, 11.14 GB), while reducing MAE by 9.85%. Compared to AirFormer and PDFormer, training time is slightly longer but MAE is reduced by 13.11% and 10.70%, respectively.

## Highlights & Insights

1. **Pioneering contribution**: The first work to introduce NDDE into air quality forecasting combined with physical guidance, advancing delay modeling from "globally uniform" to "location- and time-specific."
2. **Strong physical interpretability**: The PDE function directly corresponds to the three terms of the diffusion-advection equation, each with clear physical meaning.
3. **Completeness of delay modeling**: MAA captures the delay-aware initial state modulated by multiple factors, while PDE maintains delay awareness throughout the evolution process, forming a closed loop.
4. **Comprehensive experimentation**: 19 baselines, 3 datasets, and ablation/long-term/robustness/efficiency/hyperparameter/case study experiments with excellent coverage.
5. **Intuitive case analysis**: City-level and regional-level advection delay cases clearly demonstrate the model's ability to capture wind-driven delayed transport.

## Limitations & Future Work

1. **Efficiency of delay state maintenance**: The DDE solver must maintain a historical state buffer, incurring increasing computational overhead as the number of stations and the time window grow.
2. **Deterministic delay modeling**: Wind fields are inherently stochastic; future work could incorporate uncertainty quantification into delay estimation.
3. **Unmodeled composite delays**: In practice, pollutants may transit through intermediate regions, forming multi-hop composite transport paths that the current framework does not explicitly model.
4. **Advection graph quality depends on wind field data**: If meteorological data are missing or inaccurate, the quality of the dynamic advection graph degrades accordingly.
5. **Discrete lag $\tau$ as a hyperparameter**: Selected from {0, 1, 2, 3}; future work could explore continuous or adaptive delay learning.

## Related Work & Insights

| Method | Temporal Modeling | Delay Modeling | Physics-guided | Multifactor |
|--------|------------------|----------------|----------------|-------------|
| STGNN-based (GAGNN, etc.) | Discrete | ✗ | ✗ | Partial |
| AirFormer | Discrete | ✗ | ✗ | ✓ |
| PDFormer | Discrete | Uniform delay | ✗ | ✗ |
| AirPhyNet | Continuous (NODE) | ✗ | ✓ | ✓ |
| AirDualODE | Continuous (NODE) | ✗ | ✓ | ✓ |
| STDDE | Continuous (NDDE) | Uniform delay | ✗ | ✗ |
| **AirDDE** | **Continuous (NDDE)** | **Heterogeneous delay** | **✓** | **✓** |

AirDDE is the only method that simultaneously achieves continuous-time modeling, heterogeneous delay, physical guidance, and multifactor fusion.

The combination of **NDDE + physical priors** represents a valuable extension direction for the Neural ODE community and can be generalized to spatiotemporal forecasting tasks with delay effects, such as traffic flow prediction and infectious disease spreading. The dynamic advection graph construction approach (conditional edges based on wind speed, direction, and distance) is transferable to other scenarios requiring propagation path modeling. The dual-scale design of global memory + local memory offers insights for handling missing/noisy data and can be integrated with other spatiotemporal forecasting frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ (First to apply NDDE + physical guidance to air quality forecasting; heterogeneous delay modeling is original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (19 baselines + 6 categories of experiments + case studies; exceptionally comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; physical motivation is well articulated)
- Value: ⭐⭐⭐⭐ (Methodological contribution to delay modeling in spatiotemporal forecasting; strong generalizability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](../../NeurIPS2025/time_series/in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[AAAI 2026\] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting](sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](../../ICLR2026/time_series/rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)
- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)

</div>

<!-- RELATED:END -->
