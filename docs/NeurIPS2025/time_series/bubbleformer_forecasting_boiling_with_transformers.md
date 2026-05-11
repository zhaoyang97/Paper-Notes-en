---
title: >-
  [Paper Note] BubbleFormer: Forecasting Boiling with Transformers
description: >-
  [NeurIPS 2025][Time Series][Boiling prediction] This paper proposes BubbleFormer, a Transformer architecture based on decomposed spatiotemporal attention for forecasting boiling dynamics—including the notoriously difficu…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Boiling prediction"
  - "Transformer"
  - "BubbleML"
  - "spatiotemporal decomposed attention"
  - "spontaneous nucleation"
date: 2026-05-08
content_hash: f908173d352f101d
---

# BubbleFormer: Forecasting Boiling with Transformers

**Conference**: NeurIPS 2025  
**arXiv**: [2507.21244](https://arxiv.org/abs/2507.21244)  
**Code**: Available  
**Area**: Time Series  
**Keywords**: Boiling prediction, Transformer, BubbleML, spatiotemporal decomposed attention, spontaneous nucleation

## TL;DR
This paper proposes BubbleFormer, a Transformer architecture based on decomposed spatiotemporal attention for forecasting boiling dynamics—including the notoriously difficult spontaneous bubble nucleation events—accompanied by the BubbleML 2.0 dataset (160+ high-fidelity simulations), achieving accurate spatiotemporal boiling predictions across diverse fluids, geometries, and wall conditions.

## Background & Motivation

### State of the Field

**Background**: Boiling heat transfer is one of the most complex multiphase flow phenomena in engineering (nuclear cooling, electronics thermal management, chemical processes). Traditional numerical simulations (e.g., VOF/Level-Set) are computationally prohibitive—a single simulation requires supercomputing-level resources—severely limiting design space exploration.

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional CFD simulations require days to weeks per run, precluding real-time use or rapid design iteration.

### Root Cause

**Key Challenge**: Deep learning surrogate models (FNO, U-Net, etc.) are effective for single-phase or simple multiphase flows, but spontaneous bubble nucleation in boiling—where new bubbles appear at random locations and times—remains extremely difficult to predict.

### Mechanism

**Mechanism**: Existing surrogate models are mostly designed for short-term extrapolation and exhibit poor long-term stability.

**Key Challenge**: Boiling involves both continuous flow field dynamics (amenable to PDE learning) and discrete stochastic events (bubble birth/merging/departure). The coupling of these two regimes causes standard spatiotemporal extrapolation to fail.

**Goal**: Design a spatiotemporal prediction model capable of simultaneously handling continuous physical fields and discrete stochastic events (spontaneous nucleation).

**Key Insight**: FiLM conditioning (Feature-wise Linear Modulation) is used to inject physical parameters; decomposed spatiotemporal attention reduces computational complexity; frequency-aware scaling addresses multi-scale physics.

**Core Idea**: Decomposed spatiotemporal attention + FiLM physical conditioning + BubbleML 2.0 high-fidelity data = boiling prediction including spontaneous nucleation.

## Method

### Overall Architecture
Input: multi-field data over $T$ historical time steps (temperature field, phase field, velocity field) + physical parameters (fluid type, wall temperature, contact angle, etc.) → BubbleFormer → predicted fields over the next $K$ steps. Autoregressive prediction enables long-term extrapolation.

### Key Designs

1. **Decomposed Spatiotemporal Attention**:

    - **Function**: Decomposes full spatiotemporal attention into alternating applications of temporal-axis attention and spatial-axis attention.
    - **Mechanism**: Spatially, each location attends to all other spatial locations (with time fixed); temporally, each time step attends to all other time steps (with spatial location fixed); these are stacked in alternation.
    - **Design Motivation**: Full spatiotemporal attention has complexity $O((HWT)^2)$, which is infeasible at high resolutions; decomposition reduces this to $O((HW)^2 \cdot T + HW \cdot T^2)$.

2. **FiLM Physical Conditioning**:

    - **Function**: Injects physical parameters (fluid type, wall temperature, contact angle) into the network.
    - **Mechanism**: Physical parameters are passed through an MLP to generate scale factor $\gamma$ and shift $\beta$, which apply an affine transformation $\gamma \cdot h + \beta$ to the features.
    - **Design Motivation**: Boiling behavior varies dramatically across physical conditions (water vs. refrigerant, high vs. low heat flux); conditioning allows a single model to generalize across multiple operating regimes.

3. **Frequency-Aware Scaling**:

    - **Function**: Handles the multi-scale physics present in boiling (thermal boundary layer vs. bulk flow).
    - **Mechanism**: Multi-scale resolutions are used during patch embedding; low frequencies capture global thermal distributions while high frequencies capture bubble interfaces and thermocapillary effects.
    - **Design Motivation**: In boiling, boundary layer thickness (~micrometers) and domain scale (~millimeters) differ by several orders of magnitude.

### Loss & Training
- MSE loss with optional physical constraints (energy/mass conservation).
- Trained on 160+ simulations from BubbleML 2.0.
- Covers multiple fluids (water, FC-72, refrigerants) and geometries (flat plate, microchannel, cylinder).

## Key Experimental Results

### Main Results
Prediction errors across BubbleML 2.0 operating conditions:

| Task | Method | Temperature RMSE | Phase Accuracy |
|------|--------|-----------------|----------------|
| Pool boiling prediction | U-Net | Baseline | Baseline |
| Pool boiling prediction | FNO | Moderate | Moderate |
| Pool boiling prediction | **BubbleFormer** | **Lowest** | **Highest** |
| Spontaneous nucleation prediction | U-Net | Poor | Poor |
| Spontaneous nucleation prediction | **BubbleFormer** | **Significantly better** | **Predictable** |

### Ablation Study

| Configuration | Temperature Error | Notes |
|--------------|------------------|-------|
| Full model | Lowest | Decomposed attention + FiLM + frequency scaling |
| w/o FiLM | Increased | Cannot adapt to varying physical conditions |
| w/o frequency scaling | Increased | Multi-scale information is lost |
| Full spatiotemporal attention | Comparable but slower | Similar accuracy with much higher compute |

### Key Findings
- **Spontaneous nucleation is the key differentiator**: Model differences are modest on continuous flow prediction, but BubbleFormer substantially outperforms U-Net/FNO in predicting stochastic bubble nucleation events.
- **FiLM conditioning enables a single model to generalize across operating conditions**: No need to train separate models for each fluid or geometry.
- **Long-term autoregressive stability**: Error growth remains relatively controlled over 100+ autoregressive steps.

## Highlights & Insights
- **BubbleML 2.0 is itself a major contribution**: 160+ high-fidelity simulations spanning multiple fluids and geometries provide a standardized benchmark for AI-driven boiling research.
- The proposed framework for handling the coupling of **discrete stochastic events and continuous physics** is transferable to other multi-physics problems (e.g., solidification nucleation in metallurgy, cloud formation in atmospheric science).
- The combination of decomposed attention and FiLM conditioning constitutes an effective paradigm for scientific computing Transformers.

## Limitations & Future Work
- Training data is simulation-based rather than experimental; a sim-to-real gap persists.
- Simulations are primarily 2D; extending to 3D poses significant computational challenges.
- Predictions of spontaneous nucleation remain probabilistic; deterministic prediction is fundamentally impossible.
- The BubbleML 2.0 simulation code itself involves simplifications in the underlying physical models.

## Related Work & Insights
- **vs. FNO (Li et al.)**: FNO performs well on smooth PDEs but cannot handle the discontinuities introduced by phase change.
- **vs. DeepONet**: An operator learning framework that is theoretically general but offers no specialized treatment of discrete events in boiling.
- **vs. Traditional CFD**: After training, BubbleFormer achieves inference speeds thousands of times faster than conventional CFD.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of Transformer with spontaneous nucleation prediction in boiling forecasting.
- Experimental Thoroughness: ⭐⭐⭐⭐ 160+ simulations, multiple fluids and geometries, complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ Physical motivation is clearly articulated; method description is rigorous.
- Value: ⭐⭐⭐⭐⭐ BubbleML 2.0 dataset and BubbleFormer represent important contributions to the engineering scientific computing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)
- [\[ICLR 2026\] Relational Feature Caching for Accelerating Diffusion Transformers](../../ICLR2026/time_series/relational_feature_caching_for_accelerating_diffusion_transformers.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] DemandCast: Global hourly electricity demand forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)

</div>

<!-- RELATED:END -->
