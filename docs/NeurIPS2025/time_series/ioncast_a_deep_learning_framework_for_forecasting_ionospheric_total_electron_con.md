---
title: >-
  [Paper Note] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics
description: >-
  [NeurIPS 2025][Time Series][Ionospheric forecasting] This paper proposes IonCast, a framework comprising a GraphCast-based GNN model and a ConvLSTM baseline that integrates multi-source heterogeneous space weather data (…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Ionospheric forecasting"
  - "TEC"
  - "Graph Neural Network"
  - "GraphCast"
  - "Spatiotemporal prediction"
  - "Space weather"
date: 2026-05-08
content_hash: 6391b39e5e7a19f4
---

# IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2511.15004](https://arxiv.org/abs/2511.15004)
**Code**: [GitHub](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere)
**Area**: Time Series / Space Weather
**Keywords**: Ionospheric forecasting, TEC, Graph Neural Network, GraphCast, Spatiotemporal prediction, Space weather

## TL;DR
This paper proposes IonCast, a framework comprising a GraphCast-based GNN model and a ConvLSTM baseline that integrates multi-source heterogeneous space weather data (TEC maps, solar wind, geomagnetic indices, orbital mechanics, etc.) for global spatiotemporal forecasting of ionospheric total electron content (TEC). IonCast outperforms persistence baselines and the IRI empirical model under geomagnetic storm conditions.

## Background & Motivation

1. **Background**: The ionosphere (~50–1500 km altitude) is a critical region of near-Earth space whose disturbances directly degrade GNSS accuracy, high-frequency communications, and aviation operations. As societal dependence on space-based infrastructure deepens, accurate ionospheric forecasting becomes increasingly important.

2. **Limitations of Prior Work**:
    - Empirical models such as IRI and physics-based models such as GITM have well-documented inherent limitations.
    - Existing ML approaches are largely confined to classical methods (XGBoost, MLP) or cover only narrow geographic regions (e.g., BiLSTM applied solely to the Chinese region).
    - No advanced ML architecture simultaneously handles heterogeneous multi-source data, operates at global scale, and delivers reliable long-horizon forecasts.

3. **Key Challenge**: The rapid growth of high-quality ionospheric observations stands in contrast to the absence of ML frameworks capable of genuinely exploiting multi-source heterogeneous data for global forecasting.

4. **Key Insight**: Drawing on the success of GraphCast in numerical weather prediction, the paper adapts graph neural network architectures to the task of ionospheric dynamics forecasting.

5. **Core Idea**: A GraphCast-style GNN operates on a spherical mesh to fuse multi-source space weather data and autoregressively forecast global TEC.

## Method

### Overall Architecture
Multi-source data acquisition and alignment → Unified representation of 2D TEC maps and 1D driving time series → Encoder–Processor–Decoder GNN learning on a spherical mesh → Autoregressive multi-step global TEC forecasting.

### Key Designs

1. **Multi-Source Heterogeneous Data Integration**:
    - 2D data: JPL Global Ionosphere Maps (GIM), 15-minute temporal resolution, 180×360 spatial resolution.
    - 1D drivers: Solar wind and geomagnetic parameters (SYM-H/ASY-D, IMF $B_{xyz}$, $V_{sw}$), planetary activity indices (Kp, Ap), and solar irradiance proxies (F10.7, S10.7/M10.7/Y10.7).
    - Auxiliary spatial features: Quasi-dipole magnetic coordinates, orbital mechanics data (solar/lunar ephemerides, zenith angle, Earth–Sun/Moon distances).
    - **Design Motivation**: Ionospheric state is jointly driven by solar radiation, magnetospheric convection, and thermospheric dynamics, necessitating full-chain data coverage.

2. **IonCast GNN (Core Model)**:
    - Based on Google's GraphCast architecture, implemented with NVIDIA PhysicsNeMo.
    - **Encoder**: Message passing from the latitude–longitude grid to a spherical icosahedral mesh.
    - **Processor**: Learns dynamics via message passing on multi-resolution icosahedral meshes, with 6 processor layers and 6 multigrid levels.
    - **Decoder**: Maps from the icosahedral mesh back to the latitude–longitude grid.
    - Distinguishes *forcing* variables (analytically computable quantities such as orbital mechanics, provided as ground truth at all time steps) from *non-forcing* variables (quantities to be predicted, such as TEC).
    - **Autoregressive inference**: Starting from a context window of 8 steps (2 hours), TEC is predicted step by step into the future.

3. **IonCast LSTM (Baseline Model)**:
    - Convolutional encoder–decoder with an LSTM bottleneck.
    - A CNN encoder downsamples 180×360 TEC maps to 128-dimensional embeddings.
    - Six convolutional LSTM layers process the temporal embeddings, with circular padding to handle longitudinal continuity.
    - A CNN decoder recovers the original resolution via bilinear upsampling and transposed convolutions.

4. **Residual Target Strategy**:
    - The model predicts residual increments: $x_{T+1} = x_T + \hat{x}_{\text{predicted}}$.
    - Ablation experiments confirm that the residual target formulation substantially improves performance.

### Training Details
- Data range: 2010-05-13 to 2024-08-01, at 15-minute intervals.
- Training sampling: one sequence per 256 (~2.66-day interval) to improve computational efficiency.
- Context window: 8 steps (2 hours) → 1-step prediction (15 minutes); autoregressive unrolling at inference.
- Test set composition: stratified by NOAA geomagnetic storm level (G0–G5), with 10% of storm events held out per level.
- GNN hyperparameters: batch size 1, dropout 0.15, lr $3\times10^{-4}$, 32-hop neighbor message passing.
- LSTM hyperparameters: batch size 4, dropout 0.15, lr $2\times10^{-4}$.

## Key Experimental Results

### Main Results: RMSE Comparison across Forecast Horizons
Evaluated on G2-level moderate geomagnetic storms:
- IonCast GNN outperforms the persistence baseline at nearly all forecast horizons, with the advantage growing with lead time.
- IonCast LSTM exhibits a positive TEC overestimation bias at long horizons; the GNN's RMSE stabilizes in the 6–12 hour range.

### Comparison with the IRI Empirical Model (RMSE, TECU)

| Storm Level | IRI | LSTM 1h | GNN 1h | LSTM 6h | GNN 6h | LSTM 12h | GNN 12h |
|-------------|-----|---------|--------|---------|--------|----------|---------|
| G0 | 5.32 | 4.39 | **1.34** | 8.94 | **3.44** | 8.26 | 10.89 |
| G2 | 6.24 | 6.01 | **2.36** | 11.82 | **5.72** | 11.50 | — |

- The GNN significantly outperforms IRI within 6-hour forecast horizons.
- At 1-hour lead time, the GNN's RMSE is only 25–38% of IRI's.

### Ablation Study: Impact of Input Data Sources on GNN Performance

| Input Features | RMSE (TECU) |
|----------------|-------------|
| JPLD TEC only | 22.4±3.2 |
| JPLD + F10.7 | 23.9±10.2 |
| JPLD + multiple solar irradiance indices | 13.3±4.4 |
| JPLD + Ap & Kp | 12.7±3.2 |
| JPLD + solar wind IMF/velocity | 15.5±12.6 |
| **JPLD + orbital mechanics + quasi-dipole coordinates** | **9.2±4.5** |
| JPLD + all features (non-residual target) | 18.8±10.7 |
| JPLD + all features (residual target) | 10.7±4.5 |

### Key Findings
- **Orbital mechanics and magnetic coordinates are the most important inputs**: They encode the apparent motion of TEC induced by Earth's rotation, yielding the lowest RMSE (9.2).
- Using only orbital mechanics and quasi-dipole coordinates even outperforms using all data sources (9.2 vs. 10.7), possibly because forcing channels are excluded from the loss, allowing the model to focus on optimizing the TEC channel.
- F10.7, with its daily temporal resolution, contributes negligibly (and slightly degrades performance) for 15-minute to 12-hour forecasting.
- Models without orbital mechanics exhibit spatial drift at lead times exceeding 4 hours.
- Residual vs. non-residual target: 10.7 vs. 18.8 RMSE — a substantial improvement.

## Highlights & Insights
- **Successful transfer of GraphCast to the ionospheric domain**: From numerical weather prediction to space weather, this work validates the generality of spherical graph network architectures.
- **Elegant forcing/non-forcing distinction**: Analytically computable orbital mechanics quantities are injected as forcing at all time steps, while non-forcing variables such as TEC are predicted autoregressively, respecting the underlying physical constraints.
- **Dominant contribution of orbital mechanics**: The ablation study reveals a counterintuitive finding — solar wind and geomagnetic indices are less informative than simple solar/lunar position data, because the latter directly encodes the spatial evolution of TEC with Earth's rotation.
- **Critical role of residual learning**: Switching from predicting absolute values to predicting incremental changes nearly halves the RMSE.

## Limitations & Future Work
- As a workshop paper, the experimental scale is limited, with quantitative results reported for only a subset of storm events.
- Sparse training sampling (one sequence per ~2.66 days) may cause rapid transient events to be underrepresented.
- At the 12-hour G0 horizon, the GNN's RMSE (10.89) exceeds that of the LSTM (8.26), indicating that long-horizon forecasting under quiet conditions remains to be improved.
- Complete results for extreme storms (G4/G5) are not evaluated.
- Computational costs are not reported in detail; GraphCast-style models typically require substantial GPU resources.
- The framework could be extended to higher spatial resolutions and regionally refined forecasting.

## Related Work & Insights
- **vs. GraphCast (weather forecasting)**: IonCast directly adapts GraphCast's encoder–processor–decoder architecture while accommodating ionosphere-specific data structures and the forcing concept.
- **vs. Connecting the Dots (same conference)**: That dataset paper originates from the same NASA Heliolab project; IonCast builds its forecasting model on the corresponding dataset.
- **vs. conventional BiLSTM methods**: IonCast extends from regional to global-scale prediction and from single-source to multi-source heterogeneous inputs.
- **Broader inspiration**: The spherical graph network + forcing injection framework is transferable to other geophysical forecasting tasks, such as thermospheric density forecasting and magnetospheric dynamics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First adaptation of the GraphCast architecture to ionospheric TEC forecasting; the forcing/non-forcing distinction is a creative design choice.
- **Experimental Thoroughness**: ⭐⭐⭐ Workshop paper scope; ablation studies are valuable but scenario coverage is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed methodology descriptions and intuitive figures.
- **Value**: ⭐⭐⭐⭐ Significant reference value for the space weather ML community; demonstrates the feasibility of graph networks in this domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)

</div>

<!-- RELATED:END -->
