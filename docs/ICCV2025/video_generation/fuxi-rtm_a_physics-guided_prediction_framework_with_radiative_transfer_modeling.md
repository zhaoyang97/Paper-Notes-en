---
title: >-
  [Paper Note] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling
description: >-
  [ICCV 2025][Video Generation][Physics-guided deep learning] This paper proposes FuXi-RTM, the first hybrid physics-guided weather forecasting framework that integrates a deep learning radiative transfer model (DLRTM) as…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Physics-guided deep learning"
  - "radiative transfer modeling"
  - "weather forecasting"
  - "spatiotemporal sequence prediction"
  - "physical consistency"
date: 2026-05-08
content_hash: 85de239dc125209d
---

# FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling

**Conference**: ICCV 2025  
**arXiv**: [2503.19940](https://arxiv.org/abs/2503.19940)  
**Code**: None  
**Area**: Weather Forecasting  
**Keywords**: Physics-guided deep learning, radiative transfer modeling, weather forecasting, spatiotemporal sequence prediction, physical consistency

## TL;DR

This paper proposes FuXi-RTM, the first hybrid physics-guided weather forecasting framework that integrates a deep learning radiative transfer model (DLRTM) as a differentiable physical regularizer, outperforming the unconstrained baseline on 88.51% of variable–lead-time combinations.

## Background & Motivation

- **Background**: Deep learning weather forecasting models (e.g., FuXi, Pangu-Weather) have surpassed ECMWF HRES in forecast accuracy.
- **Limitations of Prior Work**: These models **lack explicit physical constraints** and may produce non-physical outputs (e.g., negative humidity), a problem particularly pronounced in radiative process modeling.
- **Key Challenge**: Radiative transfer is the primary energy driver of Earth's weather and climate system, governing temperature gradients, atmospheric pressure patterns, and wind circulation. Traditional NWP models simulate this via parameterization schemes at prohibitive computational cost.
- **Goal**: Embed radiative transfer capability into a weather forecasting framework by using a pretrained DLRTM as a frozen differentiable regularizer, providing physical constraint signals during training without additional training overhead.

## Method

### Overall Architecture

FuXi-RTM adopts an encoder–processor–decoder paradigm with two core components:
- **FuXi-base**: The trainable primary forecasting model (1.1B parameters, 30-layer Swin Transformer V2) that takes two consecutive atmospheric states as input and predicts the next time step.
- **DLRTM**: A pretrained and frozen deep learning radiative transfer surrogate model based on a Bi-LSTM architecture, processing atmospheric column data to produce radiative fluxes.

During training, outputs from FuXi-base are passed through the DLRTM to generate radiative fluxes, which are compared against ground-truth values computed by the RRTMG conventional model to produce a physics constraint loss.

### Key Designs

1. **DLRTM Radiative Transfer Surrogate**:

    - Architecture: 3-layer Bi-LSTM (forward 96-dim + backward 128-dim), processing each grid column independently.
    - Input: $\mathbf{Y}_t \in \mathbb{R}^{1 \times 71 \times H \times W}$, comprising 13 pressure levels × 11 variables (5 upper-air variables + 6 single-level variables).
    - Output: 4 radiative flux types (SWUFLX, SWDFLX, LWUFLX, LWDFLX) × 13 levels.
    - **Ghost Level Dynamic Masking**: Handles terrain elevation differences by masking to 0 when $P_{level} > P_{surface}$, excluding non-physical levels.
    - **Design Motivation**: Since DLRTM processes each atmospheric column independently, it natively supports global parallel computation, reducing RRTMG's 22-minute runtime (8 CPUs) to ~3 seconds (1 H100 GPU).

2. **Sunlit Region-Centered (SRC) Sampling Strategy**:

    - **Problem**: Global random sampling includes large regions without sunlight, which are meaningless for shortwave radiation.
    - **Solution**: Dynamically select a sunlit location and take a surrounding 250×250 grid patch as the loss computation region.
    - **Design Motivation**: Concentrating gradient computation in regions with the most significant solar radiation interactions, preserving local contextual relationships and reducing interference from irrelevant background signals.

3. **Physics-Guided Training Strategy**:

    - DLRTM parameters are frozen during FuXi-RTM training; only FuXi-base is updated.
    - Constraints focus on surface shortwave upwelling/downwelling fluxes (SWUFLX/SWDFLX).
    - Surface radiation constraints implicitly propagate vertical atmospheric interaction information through backpropagation.
    - **Design Motivation**: Surface radiation constraints already encode full vertical atmospheric column information; explicit supervision across all layers would introduce information redundancy.

### Loss & Training

Total loss: $L_{total} = L_{forecast} + L_{reg}$

- **Forecast Loss**: Latitude-weighted Charbonnier L1 loss:
$$L_{forecast} = \frac{1}{C \times H \times W} \sum_c \sum_i \sum_j \alpha_i \sqrt{(\hat{X}_{c,i,j} - X_{c,i,j})^2 + \epsilon^2}$$
where $\alpha_i = H \times \frac{\cos\Phi_i}{\sum_i \cos\Phi_i}$ is the latitude weighting factor.

- **Physical Regularization Loss**: Computed over the SRC sampled region:
$$L_{reg} = \frac{1}{R' \times H' \times W'} \sum_r \sum_i \sum_j \alpha_i (\lambda \sqrt{(\hat{Y}^{DLRTM} - Y^{DLRTM})^2 + \epsilon^2})$$
where $\lambda = 10^{-3}$ balances physical constraints against direct forecasting.

- Training configuration: 4× H100 GPUs, 60,000 iterations, AdamW ($\beta_1=0.9, \beta_2=0.95$), lr=2.5e-4.

## Key Experimental Results

### Main Results

Evaluated on a 5-year test set (2018–2022), initialized at 00/12 UTC daily with 6-hour intervals up to 10 days:

| Metric | FuXi-RTM vs. FuXi-base |
|------|----------------------|
| Overall advantage ratio | FuXi-RTM superior in 88.51% of 3,320 combinations |
| Cloud cover (CC) advantage ratio | 95.38% |
| Specific humidity (Q) advantage ratio | 93.46% |
| CLWC (upper levels) improvement | nRMSE difference > 2% |
| Surface albedo (FAL) | nRMSE improvement > 7% across all 10-day forecasts |
| Radiative fluxes | FuXi-RTM superior across 100% of lead-time combinations |
| ISSRD improvement | Close to 100% |
| DLRTM speedup | 22 min (8 CPUs) → 3 s (1 GPU) |

### Ablation Study

| Model Variant | Q50 | Q500 | CLWC500 | CC150 | TCC | TTR | TP | FAL |
|---------|-----|------|---------|-------|-----|-----|-----|-----|
| FuXi-RTM-Random | 0.1857 | 0.7459 | 0.0226 | 0.1725 | 0.3193 | 159.25 | 2.4327 | 0.02515 |
| FuXi-RTM-13level | 0.1697 | 0.7409 | 0.0226 | 0.1732 | 0.3184 | 159.23 | 2.4418 | 0.02313 |
| FuXi-RTM-13levelSW | 0.1762 | 0.7409 | 0.0224 | 0.1714 | 0.3183 | 158.42 | 2.4200 | 0.02335 |
| FuXi-RTM-GSW | 0.1668 | 0.7567 | 0.0226 | 0.1735 | 0.3207 | 160.95 | 2.4564 | 0.02263 |
| FuXi-RTM-ISSRD | 0.1741 | 0.7441 | 0.0226 | 0.1719 | 0.3191 | 159.14 | 2.4299 | 0.02379 |
| **FuXi-RTM** | **0.1546** | **0.7388** | **0.0223** | **0.1705** | **0.3179** | **158.13** | **2.4092** | **0.02300** |
| FuXi-base | 0.1735 | 0.7453 | 0.0226 | 0.1720 | 0.3196 | 159.14 | 2.4127 | 0.02553 |

### Key Findings

- **SRC outperforms random sampling**: Concentrating gradient computation in sunlit regions significantly outperforms global random sampling; spatially coherent gradient computation enhances feature learning.
- **Surface constraints outperform full-level constraints**: Optimizing across all pressure levels (13level) actually degrades performance; surface radiation constraints implicitly capture vertical interactions through backpropagation.
- **Constraining fundamental physical quantities outperforms derived ones**: Directly constraining SW fluxes outperforms constraining derived metrics such as ISSRD or GSW.
- **Energy conservation validation**: FuXi-RTM demonstrates better conservation of global total atmospheric energy in forecasts beyond 10 days.

## Highlights & Insights

- **First DL weather forecasting framework to explicitly integrate physical process modeling**: Distinct from prior approaches that apply ODE solvers to primitive equations.
- **Frozen DLRTM as a differentiable regularizer**: Extends radiative transfer capability without additional training, achieving extremely high computational efficiency.
- **Information completeness of surface constraints**: A seemingly simplified design choice (constraining only surface SW fluxes) in practice encodes complete physical information.
- **Analogy to video generation**: Weather forecasting is fundamentally a multi-channel spatiotemporal sequence prediction task, yet requires physical consistency across dozens of interrelated variables.

## Limitations & Future Work

- Wind components (u, v) are currently excluded; incorporating them may further improve performance at increased computational cost.
- Only radiative transfer is integrated; other critical physical processes such as convection, planetary boundary layer dynamics, and cloud microphysics are not covered.
- Certain variables (e.g., 1000 hPa Q) underperform the baseline in short-range forecasts, requiring 1.25 days before surpassing it.
- The SRC sampling strategy applies only to shortwave radiation; different strategies may be needed for longwave radiation.

## Related Work & Insights

- The autoregressive forecasting paradigm of the FuXi model series provides a mature framework for spatiotemporal sequence prediction.
- Yao et al. found Bi-LSTM to be most effective for radiative modeling, establishing the foundation for the DLRTM design.
- The approach of replacing NWP parameterization schemes with DL surrogate models is generalizable to other physical processes.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First work to integrate differentiable physical process modeling into DL weather forecasting; a pioneering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five-year test set, 3,320-combination evaluation, comprehensive ablation studies, and physical conservation validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with thorough explanations of physical background and technical details.
- **Value**: ⭐⭐⭐⭐⭐ Paves the way for the next generation of physically consistent weather forecasting systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics](ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](../../NeurIPS2025/video_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](../../CVPR2026/video_generation/phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

</div>

<!-- RELATED:END -->
