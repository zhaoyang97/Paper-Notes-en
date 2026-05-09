---
title: >-
  [Paper Note] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations
description: >-
  [NeurIPS 2025][Scientific Computing][plasma turbulence] This work presents GyroSwin, the first scalable 5D neural surrogate model for gyrokinetic plasma turbulence. It extends the Swin Transformer to the 5D gyrokinetic phase space, employs cross-attention for 3D↔5D interaction, and adopts channelwise mode separation to capture zonal flows. GyroSwin achieves higher accuracy than conventional quasilinear methods while being three orders of magnitude faster than the numerical solver GKW.
tags:
  - NeurIPS 2025
  - Scientific Computing
  - plasma turbulence
  - gyrokinetics
  - 5D surrogate model
  - Swin Transformer
  - nuclear fusion
date: 2026-05-08
content_hash: 797851184007cbc6
---

# GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations

**Conference**: NeurIPS 2025
**arXiv**: [2510.07314](https://arxiv.org/abs/2510.07314)
**Code**: [ml-jku/neural-gyrokinetics](https://github.com/ml-jku/neural-gyrokinetics)
**Area**: Scientific Computing
**Keywords**: plasma turbulence, gyrokinetics, 5D surrogate model, Swin Transformer, nuclear fusion

## TL;DR

This work presents GyroSwin, the first scalable 5D neural surrogate model for gyrokinetic plasma turbulence. It extends the Swin Transformer to the 5D gyrokinetic phase space, employs cross-attention for 3D↔5D interaction, and adopts channelwise mode separation to capture zonal flows. GyroSwin achieves higher accuracy than conventional quasilinear methods while being three orders of magnitude faster than the numerical solver GKW.

## Background & Motivation

**Nuclear Fusion and Plasma Turbulence**: Nuclear fusion is a critical pathway to clean energy, yet plasma turbulence in tokamak devices remains a central challenge limiting fusion efficiency. Understanding and predicting turbulent transport is essential for reactor design.

**Gyrokinetic Equations**: Plasma turbulence is governed by 5D gyrokinetic equations, where the distribution function $f(k_x, k_y, s, v_\parallel, \mu)$ is defined over the 2D wavenumber space $(k_x, k_y)$, the 1D field-line direction $s$, and the 2D velocity space $(v_\parallel, \mu)$. Direct numerical solvers such as GKW are computationally prohibitive, requiring substantial CPU hours per simulation.

**Limitations of Quasilinear (QL) Methods**: The dominant engineering approach reduces the problem to 3D by discarding velocity-space dimensions and augmenting with empirical saturation rules to approximate nonlinear effects. However, quasilinear methods fundamentally omit nonlinear physical processes such as zonal flows ($k_y=0$ modes), leading to limited predictive accuracy.

**Importance of Zonal Flows**: Zonal flows are large-scale structures ($k_y=0$ modes) self-organized by plasma turbulence that significantly suppress turbulent transport. Quasilinear methods cannot capture this key physical mechanism, causing systematic overestimation of the heat flux $Q$.

**Shortcomings of Existing ML Approaches**: Prior neural surrogate models have targeted low-dimensional (2D/3D) PDEs or single physical fields; no prior work directly operates on the 5D phase space. Standard 3D Transformers face quadratic memory and computational scaling that is prohibitive for 5D data.

**Goal**: The authors propose GyroSwin, the first natively 5D neural surrogate model, which retains the full 5D physical information while controlling computational complexity via local window attention, and introduces dedicated 3D↔5D interaction mechanisms and a zonal-flow separation strategy to fill the gap in 5D gyrokinetic surrogate modeling.

## Method

### Overall Architecture

GyroSwin adopts a UNet-style encoder–decoder architecture with the following core components:

- **Input**: 5D distribution function field $f \in \mathbb{R}^{N_{k_x} \times N_{k_y} \times N_s \times N_{v_\parallel} \times N_\mu}$ at resolution $(32 \times 8 \times 16 \times 85 \times 32)$
- **Encoder path**: Multiple 5D Swin Transformer blocks with progressive downsampling for multi-scale feature extraction
- **Decoder path**: Upsampling with skip connections to restore resolution and predict the next-timestep 5D distribution function $\hat{f}^{(t+1)}$
- **3D branch**: A Latent Integrator module that extracts the 3D electrostatic potential $\hat{\phi}$ and the scalar heat flux $\hat{Q}$ from 5D features
- **Autoregressive inference**: At test time, the model output $\hat{f}^{(t+1)}$ is fed back as input for multi-step rollout prediction

### Key Designs

**(1) 5D Shifted Window Attention (5DWA)**

The 2D shifted window mechanism of the Swin Transformer is generalized to five dimensions. Self-attention is computed within local 5D windows, and information propagates across windows via alternating shifts. This ensures linear complexity with respect to sequence length, avoiding the intractable $O(N^2)$ cost of global attention on 5D data.

**(2) Latent Integrator**

Physically, the mapping from the 5D distribution function $f$ to the 3D electrostatic potential $\phi$ corresponds to integration over velocity space $(v_\parallel, \mu)$. The authors design a cross-attention-based Latent Integrator:

- Learnable 1D query vectors serve as the "integration kernel"
- Velocity-space slices of the 5D features serve as keys and values
- The cross-attention output constitutes the "integrated" 3D representation

This design elegantly emulates the physical integration while remaining end-to-end differentiable.

**(3) 5D↔3D Mixing Layers**

To enable information sharing between 5D and 3D prediction targets during multi-task training:

- The Latent Integrator extracts a 3D representation from 5D encoder features
- The 3D representation is processed independently and then injected back into the 5D features via an inverse mapping
- This bidirectional information flow allows the model to leverage 3D electrostatic potential constraints when predicting the 5D distribution function, and vice versa

**(4) Channelwise Mode Separation**

Zonal flows ($k_y=0$ modes) are physically distinct from turbulent modes ($k_y \neq 0$):

- The zonal-flow component at $k_y=0$ is separated from the input and concatenated as additional channels to the network input
- This enables the network to naturally distinguish zonal flows from turbulent modes without needing to discover the separation through learning
- Ablation studies demonstrate that this design is critical for the stability of long-horizon rollout predictions

### Loss & Training

A multi-task weighted loss is employed:

$$\mathcal{L} = w_f \mathcal{L}_f + w_\phi \mathcal{L}_\phi + w_Q \mathcal{L}_Q$$

- $\mathcal{L}_f$: prediction error (MSE) on the 5D distribution function
- $\mathcal{L}_\phi$: prediction error on the 3D electrostatic potential field
- $\mathcal{L}_Q$: prediction error on the scalar heat flux
- Weights $w_f, w_\phi, w_Q$ balance tasks of different units and scales

The benefit of multi-task training is that the 3D and scalar objectives supply additional physical constraint signals, acting as regularization that guides the 5D prediction toward greater physical consistency.

## Key Experimental Results

### Main Results

**Dataset**: 255 GKW numerical simulations with varying plasma parameters per run. Resolution $(32 \times 8 \times 16 \times 85 \times 32)$, adiabatic electron approximation.

**Core metrics**: Correlation Time (the number of steps for which the model prediction remains highly correlated with the ground-truth trajectory; higher is better) and RMSE of the time-averaged heat flux $\bar{Q}$ (lower is better).

| Method | Training Data | Corr. Time (ID) | Corr. Time (OOD) | $\bar{Q}$ RMSE (ID) | $\bar{Q}$ RMSE (OOD) |
|--------|--------------|-----------------|------------------|---------------------|----------------------|
| Quasilinear (QL) | — | — | — | 89.53 | 95.22 |
| ViT | 48 sims | 16.8 | 19.2 | — | — |
| Transolver | 48 sims | 9.8 | 10.8 | — | — |
| **GyroSwin** | **48 sims** | **26.5** | **28.6** | **67.68** | **70.48** |
| **GyroSwin (scaled)** | **241 sims** | **110.33** | **111.80** | **18.35** | **26.43** |

Key findings:
- GyroSwin (48 sims) improves correlation time by 58% over ViT and 170% over Transolver
- Scaling training data to 241 simulations reduces $\bar{Q}$ RMSE from 67.68 to 18.35 (ID), substantially surpassing the QL baseline of 89.53
- After scaling, correlation time exceeds 110 steps, indicating stable autoregressive prediction over more than 100 time steps

### Ablation Study

| Component | Corr. Time (ID) | Corr. Time (OOD) | Contribution |
|-----------|-----------------|------------------|--------------|
| Full GyroSwin | 26.5 | 28.6 | baseline |
| w/o mode separation | ~20 | ~22 | zonal-flow separation contributes ~25% stability |
| w/o 3D↔5D mixing | ~22 | ~24 | multi-scale interaction improves long-term prediction |
| w/o multi-task loss | ~21 | ~23 | physics-constraint regularization is significant |
| Global attention replacing 5DWA | OOM | OOM | 5D global attention is infeasible |

### Key Findings

1. **Scaling Law Validation**: Experiments spanning from small models to 1B parameters reveal that performance improves with both parameter count and data volume following a power-law trend, analogous to the scaling laws observed in LLMs.
2. **Three Orders of Magnitude Faster than GKW**: The trained GyroSwin achieves inference speeds approximately 1000× faster than the GKW numerical solver, reducing simulations that previously required hours to seconds.
3. **OOD Generalization**: The model generalizes robustly to out-of-distribution plasma parameters, with small gaps between ID and OOD performance, suggesting that physically meaningful features are learned rather than overfitting to specific parameter regimes.
4. **Zonal Flow Capture**: Through channelwise mode separation, the model successfully learns the nonlinear suppression of turbulent transport by zonal flows—precisely the key physics that quasilinear methods cannot capture.

## Highlights & Insights

1. **Pioneering Contribution**: GyroSwin is the first neural surrogate model to natively operate on the 5D gyrokinetic phase space, directly addressing the long-standing curse of dimensionality in this domain.
2. **Physics-Driven Architecture Design**: Each core component has a clear physical correspondence—the Latent Integrator corresponds to velocity-space integration, mode separation corresponds to zonal-flow decomposition, and the multi-task loss corresponds to multi-physics consistency constraints. This "physics-inspired architecture" paradigm is transferable to other scientific computing domains.
3. **Engineering Practicality**: The three-order-of-magnitude speedup makes it feasible to integrate GyroSwin into the iterative optimization workflows for fusion reactor design (e.g., the JINTRAC workflow) as a replacement for existing quasilinear approximations.
4. **Scaling Laws in Scientific ML**: The verification of LLM-like scaling laws in scientific surrogate models provides confidence for the "large model + scientific simulation" research direction.
5. **Success of Local Attention**: The effectiveness of 5D shifted window attention demonstrates that, for high-dimensional scientific data, local attention is not merely a computational compromise but a physically justified inductive bias, as plasma turbulence itself exhibits locality.

## Limitations & Future Work

1. **Adiabatic Electron Approximation**: The current dataset considers only adiabatic electrons, neglecting electron kinetic effects. Handling fully kinetic electrons would require extending the dimensionality to include an electron species, doubling computational and memory overhead.
2. **Single Ion Species**: Only a single ion species is treated; realistic fusion plasmas (e.g., deuterium–tritium mixtures) involve multiple species requiring additional species dimensions.
3. **Magnetic Geometry Constraints**: Training data are based on specific magnetic field configurations (e.g., standard tokamak geometry); generalization to different configurations such as stellarators has not been validated.
4. **Fixed Time Resolution**: The model performs autoregressive prediction at a fixed time step and cannot adaptively adjust step size to handle efficiency differences between fast- and slow-varying regimes.
5. **Absence of Uncertainty Quantification**: As a deterministic model, no prediction confidence intervals are provided. For fusion engineering applications, uncertainty quantification is a necessary component of safety assessment.
6. **Data Generation Cost**: Despite three-order-of-magnitude inference speedup, training still requires 255 GKW simulations. Achieving comparable accuracy with fewer simulations—via, e.g., active learning strategies—is a direction worth exploring.

## Related Work & Insights

- **Swin Transformer** (Liu et al., 2021): The 2D→5D window attention extension proposed here is generalizable to other high-dimensional scientific computing problems (e.g., the 6D Boltzmann equation, high-dimensional Vlasov–Poisson systems).
- **Transolver** (Wu et al., 2024): A physics-slice attention-based PDE solver used as a baseline in this work; it underperforms GyroSwin in the 5D setting, possibly because its design favors continuous domains over discrete modal spaces.
- **FourCastNet / Pangu-Weather**: Transformer-based surrogate models for meteorology that inspired the "large-scale pretraining + scientific simulation" paradigm. GyroSwin demonstrates that this paradigm is equally viable in plasma physics.
- **QuaLiKiz / TGLF**: Existing quasilinear plasma transport models whose dominance in JINTRAC workflows is directly challenged by GyroSwin's results.
- **Neural Operators** (FNO, etc.): Spectral-domain operator learning methods suited for low-dimensional PDEs but difficult to apply directly to the heterogeneous 5D hybrid space (where some dimensions are physical-space coordinates and others are velocity-space coordinates).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First 5D gyrokinetic neural surrogate model, with high originality in both problem formulation and architectural design
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Physics-inspired architectural design is rigorous; 5DWA, Latent Integrator, and mode separation all have solid physical motivation
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablations are comprehensive, scaling law validation and baseline comparisons are thorough, though additional physical diagnostics (e.g., modal spectral analysis) are absent
- **Writing Quality**: ⭐⭐⭐⭐ — The paper provides clear background for both physics and ML audiences, though implementation details of 5D attention could be more elaborated
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses a core need in fusion engineering; three-order-of-magnitude speedup has the potential to transform plasma transport modeling workflows
- **Overall**: ⭐⭐⭐⭐⭐ — Important problem, novel methodology, and significant results; a landmark contribution at the intersection of scientific ML and fusion plasma physics

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)
- [\[NeurIPS 2025\] Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios](bayesian_surrogates_for_risk-aware_pre-assessment_of_aging_bridge_portfolios.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[NeurIPS 2025\] Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)

<!-- RELATED:END -->
