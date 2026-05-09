---
title: >-
  [Paper Note] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics
description: >-
  [NeurIPS 2025][Time Series][Ionospheric forecasting] This paper proposes IonCast, a GraphCast-inspired graph neural network framework that integrates multi-source heterogeneous physics-driven data to achieve high-accuracy spatiotemporal forecasting of global Total Electron Content (TEC).
tags:
  - NeurIPS 2025
  - Time Series
  - Ionospheric forecasting
  - graph neural networks
  - spatiotemporal modeling
  - TEC
  - GraphCast
date: 2026-05-08
content_hash: 4dd48337d29bcc50
---

# IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics

**Conference**: NeurIPS 2025  
**arXiv**: [2511.15004](https://arxiv.org/abs/2511.15004)  
**Code**: [GitHub](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere)  
**Area**: Time Series Forecasting / Space Weather  
**Keywords**: Ionospheric forecasting, graph neural networks, spatiotemporal modeling, TEC, GraphCast

## TL;DR

This paper proposes IonCast, a GraphCast-inspired graph neural network framework that integrates multi-source heterogeneous physics-driven data to achieve high-accuracy spatiotemporal forecasting of global Total Electron Content (TEC).

## Background & Motivation

**State of the Field**: The ionosphere (approximately 50–1500 km altitude) is a critical component of near-Earth space, whose variability directly affects GNSS navigation accuracy, high-frequency communications, and aviation operations. JPL provides global ionospheric TEC maps (GIM) at a 15-minute temporal resolution.

**Limitations of Prior Work**: Traditional empirical models such as IRI and physics-based models such as GITM have well-documented inherent limitations; existing ML approaches are largely based on XGBoost, MLP, or BiLSTM architectures constrained to local regions, and are unable to handle multi-source heterogeneous data at a global scale.

**Root Cause**: Accurately integrating heterogeneous multi-source data (solar wind, geomagnetic indices, orbital mechanics, etc.) at a global scale while performing reliable short-to-long-term forecasting exceeds the capacity of traditional methods and simple ML models.

**Paper Goals**: To construct an advanced ML framework capable of handling heterogeneous multi-source data, operating at a global scale, and performing reliable nowcasting and forecasting of ionospheric dynamics.

**Starting Point**: The GraphCast architecture from the numerical weather prediction domain is adopted and adapted to the task of ionospheric dynamics forecasting.

**Core Idea**: A GraphCast-inspired spherical graph network encodes global TEC maps alongside multi-source physics-driven data, enabling autoregressive prediction of ionospheric evolution.

## Method

### Overall Architecture

IonCast comprises two model variants:
- **IonCast LSTM**: CNN encoder-decoder with an LSTM temporal bottleneck
- **IonCast GNN**: A GraphCast-based Encoder-Processor-Decoder architecture on a spherical mesh

Both models predict TEC autoregressively, using a context window of 8 time steps (2 hours) to forecast the next 15-minute step, with the target formulated as residual prediction: $x_{T+1} = x_T + \hat{x}_{predicted}$.

### Key Designs

1. **Multi-Source Heterogeneous Data Integration**: The framework unifies 2D vertical TEC global maps (JPL GIM, 15-minute intervals) with 1D driver time series (solar wind/geomagnetic parameters: SYM-H, IMF $B_{xyz}$, $V_{sw}$, activity indices $K_p$/$A_p$, solar irradiance proxy F10.7, etc.), as well as auxiliary spatial features (quasi-dipole magnetic coordinates, orbital mechanics data including solar/lunar zenith angles, etc.). To the authors' knowledge, this constitutes one of the most comprehensive datasets assembled for this type of analysis.

2. **IonCast LSTM Architecture**: A six-layer convolutional LSTM with circular padding downsamples $180 \times 360$ ionospheric maps to a 128-dimensional embedding. The decoder reconstructs the original resolution via bilinear upsampling and transposed convolutions. Training settings: batch size 4, dropout 0.15, learning rate 2e-4, JPLD loss weight 20.

3. **IonCast GNN Architecture**: Implemented in PyTorch using NVIDIA PhysicsNeMo as a GraphCast backbone, following an Encoder-Processor-Decoder structure:

    - **Encoder**: Message passing maps latitude-longitude grids to a spherical icosahedral mesh
    - **Processor**: Message passing on multi-resolution icosahedral meshes for feature learning
    - **Decoder**: Maps learned representations back to the latitude-longitude grid
    - Uses 6 multi-mesh layers and 6 processor layers with 32-hop neighborhood message passing
    - Key innovation: explicit distinction between **forcing features** (orbital mechanics quantities whose future values can be analytically computed) and **non-forcing features** (TEC, driver parameters, etc., predicted autoregressively)

4. **Residual Target Prediction**: The model predicts the residual $\hat{x}_{predicted}$ rather than absolute values; ablation experiments confirm this substantially improves model performance.

### Loss & Training

- MSE between predictions and ground truth is minimized across all target channels (excluding forcing features)
- Training data spans 2010-05-13 to 2024-08-01, sampling one 2-hour window per 256 sequences
- 10% of geomagnetic storm events at each intensity level are held out for test evaluation

## Key Experimental Results

### Main Results

Comparison against the persistence baseline and the IRI empirical model:

| Model | G0 Event (1h/6h/12h RMSE) | G2 Event (1h/6h/12h RMSE) | G4 Event (1h/6h/12h RMSE) |
|------|--------------------------|--------------------------|--------------------------|
| IRI | 5.32 / - / - | 6.24 / - / - | 16.59 / - / - |
| IonCast LSTM | 4.39 / 8.94 / 8.26 | 6.01 / 11.82 / 11.50 | 14.28 / 27.06 / 25.72 |
| IonCast GNN | **1.34** / **3.44** / 10.89 | **2.36** / **5.72** / 17.14 | **5.82** / **13.3** / 36.21 |

IonCast GNN comprehensively outperforms both IRI and LSTM within 6-hour forecast horizons.

### Ablation Study

Effect of different input features on 12-hour forecasting (IonCast GNN):

| Input Features | RMSE (TECU) |
|---------|-------------|
| JPLD only | $22.4 \pm 3.2$ |
| JPLD + F10.7 | $23.9 \pm 10.2$ |
| JPLD + solar irradiance combination | $13.3 \pm 4.4$ |
| JPLD + $A_p$ & $K_p$ | $12.7 \pm 3.2$ |
| JPLD + solar wind parameters | $15.5 \pm 12.6$ |
| JPLD + orbital mechanics + quasi-dipole | **$9.2 \pm 4.5$** |
| JPLD + all features (non-residual) | $18.8 \pm 10.7$ |
| JPLD + all features (residual) | $10.7 \pm 4.5$ |

### Key Findings

- Orbital mechanics and quasi-dipole magnetic coordinates are the most important auxiliary features, providing context for the apparent motion of TEC as the Earth rotates
- Models that exclude these channels exhibit spatial drift for forecast horizons exceeding 4 hours
- F10.7, due to its daily update frequency, offers limited benefit for 15-minute-resolution predictions at 12-hour horizons
- Models trained on solar maximum data achieve performance comparable to those trained on the full dataset, while models trained on solar minimum data generalize poorly

## Highlights & Insights

- The first global ML ionospheric forecasting framework to jointly integrate heterogeneous solar/geomagnetic driver data with TEC observations
- The GraphCast architecture naturally accommodates spherical data, avoiding the polar distortion inherent in CNN-based approaches
- The distinction between forcing and non-forcing features is an elegant design choice: features with known future values are excluded from loss optimization
- The residual prediction strategy is further validated as effective in geophysical modeling

## Limitations & Future Work

- At long forecast horizons (>6 h), GNN RMSE exceeds that of IRI, indicating that accumulated autoregressive error remains a key challenge
- As a workshop paper, space constraints limit the breadth of ablations; a detailed stratified analysis of performance across different geomagnetic activity levels is absent
- Model performance on extreme G5 events is not discussed (only one sample available)
- Computational cost and inference latency are not reported in detail

## Related Work & Insights

- **GraphCast** (Lam et al., 2023) and its success in numerical weather prediction directly inspired the graph network design adopted in this work
- The approach offers methodological reference for spatiotemporal forecasting in other geophysical domains (e.g., oceanography, magnetosphere)
- The paradigm of residual prediction combined with multi-source data fusion is transferable to other space weather applications

## Rating

- Novelty: ⭐⭐⭐⭐ GraphCast is successfully adapted to ionospheric forecasting with comprehensive data fusion, though architectural innovation is incremental
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies and multi-level storm evaluation, constrained by workshop page limits
- Writing Quality: ⭐⭐⭐⭐ Well-structured with sufficient physical background
- Value: ⭐⭐⭐⭐ Directly applicable to the space weather forecasting community, with open-source code and data

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)

<!-- RELATED:END -->
