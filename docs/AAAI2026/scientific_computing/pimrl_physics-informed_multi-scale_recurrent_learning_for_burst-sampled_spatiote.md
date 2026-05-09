---
title: >-
  [Paper Note] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics
description: >-
  [AAAI 2026][Scientific Computing][physics-informed learning] This paper proposes PIMRL, a framework for learning from burst-sampled (short high-frequency bursts followed by long intervals) sparse spatiotemporal data. It features a dual-module architecture combining macro-scale latent-space reasoning and micro-scale physics correction, integrated via cross-scale message passing, achieving up to 80% error reduction across 5 PDE benchmarks.
tags:
  - AAAI 2026
  - Scientific Computing
  - physics-informed learning
  - multi-scale spatiotemporal dynamics
  - burst sampling
  - PDE solving
  - recurrent networks
date: 2026-05-08
content_hash: 4da1521851a1b7e7
---

# PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics

**Conference**: AAAI 2026
**arXiv**: [2503.10253](https://arxiv.org/abs/2503.10253)
**Code**: None
**Area**: Scientific Computing / Physics-Informed Learning
**Keywords**: physics-informed learning, multi-scale spatiotemporal dynamics, burst sampling, PDE solving, recurrent networks

## TL;DR
This paper proposes PIMRL, a framework for learning from burst-sampled (short high-frequency bursts followed by long intervals) sparse spatiotemporal data. It features a dual-module architecture combining macro-scale latent-space reasoning and micro-scale physics correction, integrated via cross-scale message passing, achieving up to 80% error reduction across 5 PDE benchmarks.

## Background & Motivation

**State of the Field**: Mainstream approaches for learning spatiotemporal PDE dynamics (FNO, DeepONet, PeRCNN, etc.) predominantly rely on densely and uniformly sampled data. However, data in practical settings (mobile sensing, physical experiments) is often burst-sampled — short high-frequency segments followed by long idle intervals.

**Limitations of Prior Work**: (a) Pure data-driven methods (FNO) require large amounts of high-quality data; (b) physics-informed methods (PeRCNN, PINN) depend on small time steps for local updates, leading to error accumulation in long-term prediction; (c) existing methods assume uniform temporal sampling and handle burst-sampled, multi-scale irregular data poorly.

**Root Cause**: In burst-sampled data, the micro-scale captures high-frequency physical details over short, sparse segments, while the macro-scale spans long continuous intervals but loses transient dynamics. The challenge is to simultaneously exploit the complementary information from both scales.

**Paper Goals**: Accurately learn long-term spatiotemporal dynamics from multi-scale, sparsely burst-sampled data.

**Starting Point**: Design a dual-module framework — a micro-scale module that extracts fine-grained dynamics from high-frequency bursts using physical priors, and a macro-scale module that performs efficient large-step inference in latent space — fused via message passing.

**Core Idea**: Micro-scale physics correction + macro-scale latent-space reasoning + cross-scale message passing, working in concert to enable accurate long-term prediction from burst-sparse data.

## Method

### Overall Architecture
PIMRL consists of two modules and a message passing mechanism (Figure 2):
- **Micro-scale module** (PeRCNN-based): small time step $\delta t$, captures fine-grained dynamics via physics-constrained convolution
- **Macro-scale module** (residual ConvLSTM autoencoder): large time step $\Delta t = k \cdot \delta t$, performs efficient long-range inference in latent space
- **Cross-scale message passing**: micro-scale outputs correct macro-scale states; macro-scale outputs guide micro-scale iterations

### Key Designs

1. **Micro-scale Module (PeRCNN Π-block)**:

    - **Function**: Learns fine-grained physical dynamics from high-frequency burst segments.
    - **Mechanism**: Adopts the PeRCNN Π-block architecture with forward Euler discretization $\mathbf{u}_{(k+1)\delta t} = \hat{\mathcal{F}}(\mathbf{u}_{k\delta t}) \cdot \delta t + \mathbf{u}_{k\delta t}$, where $\hat{\mathcal{F}}$ is approximated by products of multi-channel convolutions: $\hat{\mathcal{F}}(\mathbf{u}) = \sum_c W_c \cdot [\prod_l (K_{c,l} \star \mathbf{u} + b_l)]$. Convolutional kernels for known physical terms (e.g., the Laplacian) are directly initialized from finite difference stencils.
    - **Design Motivation**: Hard-coding known physical priors (physics-based Conv layers) enables accurate dynamics learning from limited data, while the Π-block captures unknown residual terms.

2. **Macro-scale Module (Residual ConvLSTM Autoencoder)**:

    - **Function**: Performs large-step inference in latent space, bridging long temporal gaps between bursts.
    - **Mechanism**: An encoder maps the physical space to a compact latent space; a ConvLSTM advances the latent state at step size $\Delta t$; a decoder reconstructs physical-space predictions. Crucially, the module periodically receives correction messages from the micro-scale module to anchor physical consistency.
    - **Design Motivation**: Direct large-step inference in physical space degrades quality (losing transient dynamics), whereas latent-space inference is more efficient and expressive. Micro-scale corrections prevent latent trajectories from drifting.

3. **Cross-scale Message Passing**:

    - **Function**: Fuses information from the micro-scale and macro-scale modules.
    - **Mechanism**: A three-level nested loop — (a) the micro-scale module performs $k$-step small-step rollout to produce physics correction messages $\mathbf{u}_{t+k\delta t}^{micro}$; (b) these messages are passed to the macro-scale module for state update $\hat{\mathbf{u}}_{t+2k\delta t} = F_{macro}(\mathbf{u}_{t+k\delta t}^{micro})$; (c) after $N-1$ autonomous macro-scale rollout steps, the micro-scale correction is received again, completing a $2N$-step prediction cycle.
    - **Design Motivation**: Micro-scale correction is activated only when burst data is available; during other intervals, the macro-scale module infers autonomously — perfectly matching the sparse pattern of burst sampling.

### Loss & Training
- RMSE loss computed on macro-scale observations is used to train the overall framework.
- The micro-scale module is pre-trained separately on high-frequency burst segments.
- Only macro-scale module outputs participate in final prediction and loss computation.
- Training requires very few trajectories (2–13 per dataset).

## Key Experimental Results

### Main Results
Five PDE systems (1D KdV, 2D Burgers, 2D FN, 2D GS, 3D GS):

| Dataset | PIMRL RMSE | Prev. SOTA RMSE | Gain |
|--------|------------|----------------|------|
| KdV | **0.0457** | 0.0942 (PeRCNN) | 51.5% |
| Burgers | **0.0068** | 0.0075 (PeRCNN) | 9.3% |
| FitzHugh-Nagumo | **0.1349** | 0.1591 (PeRCNN) | 15.2% |
| 2D Gray-Scott | **0.0133** | 0.0455 (PeRCNN) | 70.8% |
| 3D Gray-Scott | **0.0116** | 0.0532 (PeRCNN) | 78.2% |

On the most challenging 3D Gray-Scott benchmark, error is reduced by nearly 80%.

### Ablation Study

| Configuration | Outcome |
|------|------|
| Micro-scale only (PeRCNN) | Long-term error accumulation; cannot bridge large intervals |
| Macro-scale only (ConvLSTM) | Loss of physical accuracy; coarse predictions |
| w/o message passing | Macro-scale trajectory drift |
| **Full PIMRL** | Best performance |

### Key Findings
- FNO diverges (NaN) in multiple settings, demonstrating that purely data-driven methods are unreliable under data scarcity and burst sampling.
- PeRCNN is the strongest baseline — physics hard-coding yields high data efficiency, but large time-step extrapolation is poor.
- PIMRL achieves 0.0133 RMSE on 2D Gray-Scott with only 2 training trajectories (vs. 0.0455 for PeRCNN).
- PIMRL also leads comprehensively on the HCT (High Correction Time) metric, confirming stability not only in the short term but also over long horizons.

## Highlights & Insights
- **Burst sampling is a severely underexplored practical problem**: this paper is the first to systematically address this data pattern, filling an important gap.
- The **cross-scale message passing** design is highly intuitive — the micro-scale provides physical anchors, the macro-scale enables efficient inference, and message-passing timing naturally aligns with burst data availability windows.
- **Minimal data requirements** (2–13 trajectories) demonstrate the substantial advantage of physics-informed frameworks in data-scarce scenarios.
- Validated across 1D to 3D settings, confirming the generality of the framework.

## Limitations & Future Work
- The micro-scale module, based on PeRCNN, requires partial PDE prior knowledge (e.g., the form of the diffusion term); adaptation is needed for completely unknown systems.
- The macro-scale ConvLSTM may face efficiency limitations for very long sequences or high-dimensional spaces.
- $k$ (micro/macro step ratio) and $N$ (message-passing period) are hyperparameters requiring tuning.
- Validation is conducted solely on synthetic PDE data; real experimental data with noise and measurement errors has not been tested.

## Related Work & Insights
- **vs. PeRCNN**: PIMRL's micro-scale module is essentially PeRCNN, yet the addition of the macro-scale module and message passing yields substantial performance gains (2D GS: 0.0133 vs. 0.0455), demonstrating the value of the multi-scale framework.
- **vs. FNO**: FNO frequently diverges under burst-sampled data, while PIMRL remains stable and reliable under identical conditions.
- **vs. CROP**: CROP addresses discretization invariance but not burst sampling; PIMRL clearly outperforms CROP on FN and GS benchmarks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic treatment of PDE learning under burst sampling; cross-scale message passing is a genuinely novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets (1D–3D), multiple baselines, complete ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and the architecture diagram is intuitive, though some equations and loop descriptions are convoluted.
- **Value**: ⭐⭐⭐⭐ Provides an effective solution for modeling physical systems under sparse and multi-scale sampling regimes.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/scientific_computing/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[AAAI 2026\] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids](phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/scientific_computing/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](../../NeurIPS2025/scientific_computing/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)

<!-- RELATED:END -->
