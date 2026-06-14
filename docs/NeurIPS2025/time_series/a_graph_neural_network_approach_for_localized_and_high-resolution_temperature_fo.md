---
title: >-
  [Paper Note] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting
description: >-
  [NeurIPS 2025][Time Series][GNN] This paper proposes a GCN-GRU hybrid framework for community-scale (2.5 km) high-resolution temperature forecasting (1–48 hours), validated across three regions in southwestern Ontario…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "GNN"
  - "temperature forecasting"
  - "high resolution"
  - "heat wave warning"
  - "climate equity"
date: 2026-05-08
content_hash: b363e83ccd7fe7d6
---

# A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2512.00546](https://arxiv.org/abs/2512.00546)  
**Code**: None  
**Area**: Time Series
**Keywords**: GNN, temperature forecasting, high resolution, heat wave warning, climate equity

## TL;DR
This paper proposes a GCN-GRU hybrid framework for community-scale (2.5 km) high-resolution temperature forecasting (1–48 hours), validated across three regions in southwestern Ontario, Canada. The largest region achieves an average MAE of 1.93°C and a 48-hour MAE of 2.93°C. The work explores ClimateBERT language model embeddings as a standardized input scheme, and provides a transferable lightweight forecasting framework targeting data-scarce regions in the Global South.

## Background & Motivation

**Background**: Approximately 490,000 deaths per year are attributed to heat-wave-related causes globally. Current operational weather forecasting systems operate at 10–30 km resolution, insufficient to capture microscale extreme events such as urban heat islands. ML weather models (GraphCast, FourCastNet) focus on global scales, leaving local high-resolution forecasting underdeveloped.

**Limitations of Prior Work**: NWP models are computationally expensive and resolution-limited; global ML models are not applicable at community scale (2.5 km); marginalized communities most in need of fine-grained warnings fall precisely in the blind spots of coarse-resolution models.

**Key Challenge**: Climate equity demands fine-grained forecasting, yet the most vulnerable regions in the Global South have the least data and the most limited computational resources.

**Goal**: Construct a lightweight, high-resolution (2.5 km, hourly) temperature forecasting model to serve as a foundation for localized heat wave warnings.

**Key Insight**: GNNs are naturally suited to capturing neighborhood interactions over spatial grids; GRUs capture temporal dependencies. NOAA URMA 2.5 km analysis data are used as inputs.

**Core Idea**: A compact regional GCN-GRU model can deliver accurate community-scale temperature forecasts, and can be extended to data-scarce regions via transfer learning.

## Method

### Overall Architecture
Regional meteorological grid points serve as graph nodes (features: temperature, wind speed, pressure, dew point, terrain elevation). Graph convolutional layers model spatial interactions, and GRU layers model temporal dependencies. The model takes a 24-hour historical window as input and produces multi-step forecasts at horizons of 1–48 hours, trained with MSE loss.

### Key Designs

1. **GCN-GRU Hybrid Network**:

    - **Function**: Spatial features are extracted via GCN; temporal sequences are modeled via GRU.
    - **Mechanism**: Each grid point is a graph node; edges connect neighboring points. After GCN aggregates neighborhood information, the GRU generates multi-step predictions along the temporal dimension.
    - **Design Motivation**: GNNs handle irregular grids and multi-scale spatial relationships more flexibly than CNNs.

2. **Multi-Region Experimental Design**:

    - **Function**: Three nested regions (A: 44×33 km, B: 111×163 km, C: 333×243 km) are used to validate scalability.
    - **Mechanism**: Hyperparameter search is performed on the smallest window, then validated on larger regions. Larger regions provide richer spatial context and yield better performance.
    - **Design Motivation**: Demonstrates the spatial scaling property of GNNs — larger graphs provide more neighborhood information.

3. **ClimateBERT Embedding Scheme**:

    - **Function**: Meteorological observations are converted into natural language descriptions, encoded into 768-dimensional vectors by ClimateBERT, dimensionality-reduced via PCA, and then fed into the GNN.
    - **Mechanism**: Standardizes heterogeneous input formats to accommodate incomplete or non-standard data.
    - **Design Motivation**: Data formats in the Global South are inconsistent; language model embeddings provide a unified representation to facilitate transfer learning.

### Loss & Training
- MSE loss, AdamW optimizer
- Forecast time steps: 1, 6, 12, 18, 24, 36, 48 h
- A 6-hour downsampled variant retains most accuracy (MAE degradation of only +0.46°C) while substantially reducing computational overhead.

## Key Experimental Results

### Main Results

| Region | Avg. MAE (°C) | MAE@48h (°C) | RMSE@48h (°C) |
|--------|--------------|-------------|--------------|
| A (44×33 km) | 2.55 | 3.78 | 4.84 |
| B (111×163 km) | 2.48 | 3.73 | 4.84 |
| C (333×243 km) | **1.93** | **2.93** | **3.90** |

### Ablation Study

| Configuration | Avg. MAE | MAE@48h | Notes |
|---------------|---------|---------|-------|
| Region C, 1 h sampling | 1.93 | 2.93 | Best |
| Region C, 6 h sampling | 2.39 | 3.15 | Substantially reduced compute |
| Region A + ClimateBERT | 3.34 | 4.34 | Slightly worse but provides standardized path |
| Region A, random weights (control) | 9.11 | 8.89 | Confirms embeddings carry meaningful signal |

### Key Findings
- **Larger spatial context yields better performance**: Region C > B > A; larger graphs capture richer mesoscale context.
- **6-hour downsampling incurs low cost**: MAE increases by only 0.46°C while computational demand drops substantially, suitable for resource-constrained settings.
- **ClimateBERT embeddings are lossy but viable**: MAE increases by approximately 0.8°C, yet provide a standardized pathway for handling heterogeneous or missing data.
- **Resolution mismatch complicates fair comparison**: 2.5 km hourly vs. existing models at 10–50 km and 3–6 h intervals.

## Highlights & Insights
- **An AI perspective on climate equity**: The work explicitly links technical contributions to social equity — marginalized communities most in need of fine-grained forecasts are precisely those least served. The transferability of the lightweight model is the core value proposition.
- **Creative use of ClimateBERT embeddings**: Meteorological observations are "translated" into natural language before re-encoding, accommodating non-standardized data. Although performance is degraded, this provides a unified interface for cross-region transfer.
- **6-hour downsampling as a practical compute–accuracy trade-off**: Directly applicable for resource-constrained deployment scenarios.

## Limitations & Future Work
- **Validation limited to a single region (Ontario)**: Transfer learning to data-scarce Global South regions has not been empirically tested.
- **Temperature-only forecasting**: Heat wave definitions typically require compound indicators including humidity and wind speed.
- **No fair baseline**: The absence of comparable baselines at 2.5 km resolution makes it difficult to assess relative advantages.
- **ClimateBERT embeddings underperform direct inputs**: Better embedding or fusion schemes are needed.
- **Small model scale and relatively simple methodology**: GCN+GRU is a basic combination; more modern architectures such as Transformers and attention mechanisms remain unexplored.

## Related Work & Insights
- **vs. GraphCast**: GraphCast operates globally at 28 km resolution; this work targets community scale at 2.5 km — different objectives but complementary.
- **vs. Li et al. GNN heat wave classification**: Li et al. use GNNs for regional heat wave event classification; this work performs continuous temperature forecasting with post-processing for heat wave identification, offering greater flexibility.
- **vs. NWP models**: Traditional NWP is computationally expensive and resolution-limited; the proposed GNN approach is lightweight and achieves higher resolution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The methodology is relatively basic (GCN+GRU); the embedding scheme is creative but of limited effect.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three regions are evaluated, but all within the same geographic area; no cross-region transfer experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Climate equity motivation is clearly articulated; the paper is concise.
- **Value**: ⭐⭐⭐⭐ — The research direction is important, but the current work requires further validation and extension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)
- [\[NeurIPS 2025\] Statistical Guarantees for High-Dimensional Stochastic Gradient Descent](statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)

</div>

<!-- RELATED:END -->
