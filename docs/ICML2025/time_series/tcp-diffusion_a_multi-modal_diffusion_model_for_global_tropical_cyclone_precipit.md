---
title: >-
  [Paper Note] TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness
description: >-
  [ICML 2025][Time Series][Tropical Cyclone Precipitation] This paper proposes TCP-Diffusion, a conditional diffusion model that integrates historical precipitation, multimodal meteorological variables, and NWP forecasts. By predicting precipitation changes rather than absolute values through an Adjacent Residual Prediction (ARP) mechanism, it outperforms authoritative NWP methods such as ECMWF in global tropical cyclone precipitation forecasting.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Tropical Cyclone Precipitation"
  - "Diffusion Models"
  - "Multimodal Meteorological Data"
  - "Adjacent Residual Prediction"
  - "NWP Fusion"
date: 2026-05-08
content_hash: 3e05b789d541c99f
---

# TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness

**Conference**: ICML 2025  
**arXiv**: [2410.13175](https://arxiv.org/abs/2410.13175)  
**Code**: [https://github.com/Zjut-MultimediaPlus/TCP-Diffusion](https://github.com/Zjut-MultimediaPlus/TCP-Diffusion)  
**Area**: Diffusion Models / Weather Forecasting  
**Keywords**: Tropical Cyclone Precipitation, Diffusion Models, Multimodal Meteorological Data, Adjacent Residual Prediction, NWP Fusion

## TL;DR
This paper proposes TCP-Diffusion, a conditional diffusion model that integrates historical precipitation, multimodal meteorological variables, and NWP forecasts. By predicting precipitation changes rather than absolute values through an Adjacent Residual Prediction (ARP) mechanism, it outperforms authoritative NWP methods such as ECMWF in global tropical cyclone precipitation forecasting.

## Background & Motivation
**Background**: Although deep learning methods for general precipitation forecasting have progressed (U-Net, GAN, diffusion models), tropical cyclone (TC) precipitation forecasting has not been systematically investigated—specifically, the prediction target area moves with the TC, and the spatiotemporal structures of TC precipitation are more complex.

**Limitations of Prior Work**: (a) Directly predicting absolute precipitation values suffers from cumulative errors and lacks physical consistency; (b) relying solely on precipitation data is insufficient to capture complex TC precipitation patterns; (c) deep learning (DL) methods do not fully utilize the physical information provided by Numerical Weather Prediction (NWP) models.

**Key Challenge**: TC precipitation is highly dynamic (the target area moves along with the TC and is influenced by multiple factors like wind fields), and purely data-driven methods lack sufficient information.

**Goal**: (a) Reduce cumulative prediction errors; (b) extract richer meteorological information; (c) integrate NWP forecasts as guidance.

**Key Insight**: Drawing inspiration from predicting change metrics in NWP + using multiple encoders to extract heterogeneous meteorological data + leveraging NWP forecasts as conditions.

**Core Idea**: Adjacent Residual Prediction (ARP) for precipitation changes + multimodal conditional encoding + a spatiotemporal diffusion model integrated with NWP.

## Method

### Overall Architecture
TCP-Diffusion is built upon a spatiotemporal 3D Unet diffusion model. The input consists of two parts: (1) historical observational data (precipitation, $\Delta$Rainfall, ERA5 environmental variables, and TC scalar variables); (2) NWP future prediction data. ARP is used to shift the prediction target from absolute precipitation values to precipitation residuals between adjacent time steps.

### Key Designs

1. **Adjacent Residual Prediction (ARP)**:

    - **Function**: Change training targets from absolute precipitation values to differences between adjacent steps
    - **Mechanism**: $\Delta_x^t = X_{\text{rain}}^t - X_{\text{rain}}^{t-1}$ where the model predicts the future $\hat{\Delta}_y$, ultimately yielding $\hat{y}_{n+t} = X_{\text{rain}}^n + \sum_{z=1}^t \hat{\Delta}_y^{n+z}$
    - **Design Motivation**: Draw inspiration from incremental prediction in NWP to reduce cumulative errors and ensure the physical consistency of precipitation trends

2. **Historical Data Encoder (Multimodal)**:

    - **Function**: Encode 2D meteorological fields (precipitation + environmental variables) and 1D scalars (TC intensity, track)
    - **Mechanism**: Use 3D CNNs to encode spatiotemporal features of 2D data, and use an MLP + Transformer to encode chemical/temporal dependencies of 1D scalar variables
    - **Design Motivation**: Different data modalities have different dimensions, necessitating dedicated encoders

3. **Future Prediction Data Encoder**:

    - **Function**: Encode future predictions from the NWP model (ERA5-IFS) as conditions
    - **Mechanism**: Use a modified ResNet-18 to extract features from future predictions and inject them into the 3D UNet as guidance
    - **Design Motivation**: NWP predictions contain information encoded by physical equations, which can assist the DL model in making more accurate predictions

4. **EA-3DUNet (Core Denoising Network)**:

    - **Function**: Fuse all encoded features for spatiotemporal denoising
    - **Mechanism**: Process spatiotemporal data with a 3D UNet, where encoder outputs, scalar features, and NWP conditions are injected into different layers respectively
    - **Design Motivation**: Simultaneously capture spatial structures (rainband shapes) and temporal evolution (strengthening/weakening trends of precipitation)

### Loss & Training
- Standard diffusion denoising loss: $L(\theta) = \|r_s - \hat{r}_s\|_2$
- The dataset covers 1,877 TCs (1980–2020), with a train/val/test split of 1,751/87/126
- Inferences are averaged over 8 sampling runs to generate probabilistic predictions

## Key Experimental Results

### Main Results (vs. DL Methods)

| Method | ETS-6 ↑ | ETS-24 ↑ | ETS-60 ↑ | TPMAE ↓ |
|------|---------|----------|----------|---------|
| U-Net | 0.442 | 0.106 | 0 | 0.475 |
| PreDiff | 0.385 | 0.119 | 0.004 | 0.536 |
| NowcastNet | 0.422 | 0.090 | 0.000 | 0.570 |
| **TCP-Diffusion** | **0.438** | **0.147** | **0.006** | **0.423** |

### vs. NWP Methods

| Method | ETS-6 ↑ | ETS-24 ↑ | ETS-60 ↑ | TPMAE ↓ |
|------|---------|----------|----------|---------|
| ERA5-IFS | 0.202 | 0.016 | 0 | 0.511 |
| ECMWF-IFS | 0.302 | 0.083 | 0.003 | 0.507 |
| **TCP-Diffusion** | **0.412** | **0.128** | **0.004** | **0.474** |

### Ablation Study

| Component | ETS-6 ↑ | Relative Gain |
|------|---------|---------|
| Baseline (3D DM) | 0.391 | - |
| + ARP | 0.406 | +3.8% |
| + ARP + Multimodal | 0.429 | +9.7% |
| + ARP + Multimodal + NWP | **0.438** | +12.0% |

### Key Findings
- TCP-Diffusion is the only DL method that outperforms the Persistence baseline on heavy precipitation (ETS-60).
- Precipitation frequency distribution plots show that DM-based methods (TCP-Diffusion, PreDiff) perform significantly better than non-DM methods.
- The combination of low-cost NWP (ERA5-IFS) + DL can outperform high-cost NWP (ECMWF-IFS).

## Highlights & Insights
- **First Global TC Precipitation DL Forecasting System**: Covers 1,877 TCs across six major ocean basins.
- **Bridge between DL and NWP**: Ingeniously leverages low-resolution NWP forecasts as conditional inputs for the DL model.
- **Generality of the ARP Mechanism**: The concept of predicting changes is transferable to other time-series forecasting tasks.

## Limitations & Future Work
- The temporal resolution is only 3 hours; higher resolution (e.g., 1 hour) has not been verified.
- The probabilistic forecasting strategy, which averages 8 diffusion sampling runs, is relatively simple.
- Real-time forecasting capabilities for future TCs have not been evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of ARP and multimodal conditional diffusion in TC precipitation forecasting
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies with systematic comparisons against both DL and NWP methods
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and good physical intuition
- Value: ⭐⭐⭐⭐⭐ Holds significant practical value for AI in meteorology

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](../../NeurIPS2025/time_series/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[AAAI 2026\] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints](../../AAAI2026/time_series/mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)
- [\[ICLR 2026\] ICDiffAD: Implicit Conditioning Diffusion Model for Time Series Anomaly Detection](../../ICLR2026/time_series/icdiffad_implicit_conditioning_diffusion_model_for_time_series_anomaly_detection.md)
- [\[ICLR 2026\] TEN-DM: Topology-Enhanced Diffusion Model for Spatio-Temporal Event Prediction](../../ICLR2026/time_series/ten-dm_topology-enhanced_diffusion_model_for_spatio-temporal_event_prediction.md)
- [\[ICLR 2026\] MMPD: Diverse Time Series Forecasting via Multi-Mode Patch Diffusion Loss](../../ICLR2026/time_series/mmpd_diverse_time_series_forecasting_via_multi-mode_patch_diffusion_loss.md)

</div>

<!-- RELATED:END -->
