---
title: >-
  [Paper Note] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling
description: >-
  [CVPR 2026][Autonomous Driving][Federated Learning] This paper proposes F3DGS, the first method to apply a federated learning framework to 3DGS, enabling decentralized multi-agent 3D reconstruction through frozen geometry and visibility-aware aggregation without sharing raw data.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Federated Learning
  - 3D Gaussian Splatting
  - Multi-Agent
  - Distributed Reconstruction
  - Visibility-Weighted Aggregation
date: 2026-05-08
content_hash: 69712a22147fd28d
---

# F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling

**Conference**: CVPR 2026
**arXiv**: [2604.01605](https://arxiv.org/abs/2604.01605)
**Code**: Coming soon (including dataset and development toolkit)
**Area**: Autonomous Driving / Multi-Agent 3D Reconstruction
**Keywords**: Federated Learning, 3D Gaussian Splatting, Multi-Agent, Distributed Reconstruction, Visibility-Weighted Aggregation

## TL;DR

This paper proposes F3DGS, the first method to apply a federated learning framework to 3DGS, enabling decentralized multi-agent 3D reconstruction through frozen geometry and visibility-aware aggregation without sharing raw data.

## Background & Motivation

**Background**: 3DGS has achieved state-of-the-art performance in novel view synthesis and is widely applied in robotics, autonomous driving, and embodied AI.

**Limitations of Prior Work**: All existing 3DGS methods assume centralized data access—all observations must be jointly optimized on a single machine. This poses three constraints in multi-agent distributed scenarios:
   - **Communication Overhead**: Bandwidth and storage for aggregating high-resolution images scale linearly with the number of agents.
   - **Data Privacy**: In multi-operator or multi-organization settings, raw sensor data is private and cannot be shared directly.
   - **Scalability**: The computational demands of joint optimization are tied to the total inference size, creating a centralized bottleneck.

**Key Challenge**: Federated learning can address the above issues by sharing only model updates, but directly applying FedAvg to 3DGS introduces two domain-specific challenges—**geometry drift** (inconsistencies caused by independently optimized positional parameters) and **partial observability** (each client observes only a subset of Gaussians).

**Goal**: Achieve unified multi-agent collaborative 3DGS reconstruction under federated constraints (zero raw image exchange).

**Key Insight**: Exploit the explicit separability of 3DGS parameters (position, covariance, and color are independent tensors) to decouple geometry from appearance.

**Core Idea**: Freeze the shared geometric skeleton (fixed positions) and federally optimize only appearance attributes, resolving partial observability via visibility-weighted aggregation.

## Method

### Overall Architecture

F3DGS operates in three stages:
1. **Shared Geometric Skeleton Construction**: Initialize a global Gaussian model by fusing LiDAR point clouds from all clients.
2. **Federated Optimization**: Fix positions; each client optimizes only appearance attributes.
3. **Visibility-Aware Aggregation**: Aggregate client updates weighted by visibility frequency.

### Key Designs

1. **Metric-Scale Pose Construction**

   **Function**: Align each client's camera trajectory to a global LiDAR anchor.

   **Mechanism**: Apply Umeyama $\mathcal{S}im(3)$ estimation to compute scale $s_k$, rotation $R_k$, and translation $\mathbf{t}_k$:

   $s_k, R_k, \mathbf{t}_k = \arg\min_{s,R,\mathbf{t}} \sum_j \|\mathbf{p}_j^{\text{anchor}} - (sR\mathbf{p}_j^{\text{client}} + \mathbf{t})\|^2$

   To prevent boundary discontinuities, exponentially decaying SE(3) corrections are applied for smoothing: $T_t^{\text{smooth}} = \text{Exp}(\beta(t) \cdot \text{Log}(\Delta T)) \cdot T_t^{\text{aligned}}$

   **Design Motivation**: Pose estimates from individual agents in a multi-agent system may employ different methods, leading to inconsistent scales and coordinate frames. A global LiDAR anchor provides a unified reference, while boundary smoothing eliminates stitching artifacts.

2. **Federated Optimization with Frozen Geometry**

   **Function**: Update only appearance parameters during local training while keeping positions fixed.

   **Mechanism**:
   $\mu_i^{(k)} = \mu_i \quad \forall k, \forall \text{steps}$

   Only appearance parameters $\theta_{\text{app}} = \{s_i, q_i, \alpha_i, \mathbf{c_i}\}_{i=1}^M$ (scale, rotation, opacity, and spherical harmonic coefficients) receive gradient updates. A visibility counter $v_{k,i}$ tracks how many times Gaussian $i$ is rasterized during training on client $k$.

   **Design Motivation**: Unlike neural network weights, Gaussian positions explicitly encode 3D coordinates in a smooth parameter space. Independently optimizing positions leads to geometry drift—averaging drifted positions produces incoherent reconstructions. Freezing positions eliminates this problem fundamentally.

3. **Visibility-Aware Federated Aggregation**

   **Function**: Aggregate updates from each client weighted by their observation frequency for each Gaussian.

   **Mechanism**: Aggregation weights are proportional to visibility:

   $\alpha_{k,i} = \frac{v_{k,i}}{\sum_{j=1}^K v_{j,i} + \epsilon}$

   The global attribute is the weighted combination: $a_i = \sum_{k=1}^K \alpha_{k,i} a_{k,i}$

   Quaternion parameters require special treatment (sign alignment followed by averaging and normalization). Gaussians with zero total visibility retain the previous round's global values.

   **Design Motivation**: Uniform averaging blends well-observed appearance estimates with random values from clients that never observed the corresponding Gaussian, diluting information quality. Visibility weighting ensures that clients with more observations contribute more to the aggregated result.

### Loss & Training

Each federated round consists of $T=1000$ local optimization steps over $R=7$ rounds. The rendering loss combines L1 and SSIM:

$$\mathcal{L}_k = \sum_{t \in \mathcal{I}_k} [(1-\lambda)\|I_t - \hat{I}_t\|_1 + \lambda(1 - \text{SSIM}(I_t, \hat{I}_t))]$$

where $\lambda = 0.2$. Adaptive density control and any form of post-aggregation alignment are disabled.

## Key Experimental Results

### Main Results (MeanGreen Indoor Dataset)

| Sequence | Frames | Clients | Local PSNR↑ | Global PSNR↑ | Local SSIM↑ | Global SSIM↑ |
|----------|--------|---------|-------------|--------------|-------------|--------------|
| 05 | 543 | 2 | 23.71 | 22.65 | 0.818 | 0.808 |
| 06 | 1042 | 3 | 25.66 | 22.66 | 0.831 | 0.803 |
| 08 | 718 | 2 | 24.52 | 23.94 | 0.853 | 0.832 |
| 11 | 552 | 2 | 24.01 | 22.77 | 0.827 | 0.808 |

### Ablation Study (Communication Rounds vs. Quality Trade-off)

| Sequence | Rounds R | Local Steps T | Local PSNR | Global PSNR | Trend |
|----------|----------|---------------|------------|-------------|-------|
| 07 | 1 | 7000 | 23.90 | 21.95 | Highest local but lowest global |
| 07 | 7 | 1000 | 23.64 | 22.74 | Balanced |
| 07 | 14 | 500 | 23.57 | **22.84** | Highest global |
| 08 | 7 | 1000 | 24.52 | 23.94 | Good balance |

### Key Findings

- The PSNR gap between the global and local models remains within 2 dB for most sequences.
- Increasing the number of communication rounds slightly reduces local performance but improves global consistency.
- Sequence 03 exhibits the largest global performance drop (18.82 dB), indicating that aggregation is sensitive to inconsistencies introduced by temporal segmentation.
- More federated aggregation rounds contribute to improved cross-client consistency.

## Highlights & Insights

- **Clear Problem Formulation**: This work is the first to formally define the 3DGS training problem under federated constraints, filling an important gap.
- **Clever Exploitation of Geometry–Appearance Decoupling**: The explicit representation of 3DGS (separable position, color, etc.) enables the "freeze geometry, federate appearance" strategy, which is infeasible in implicit NeRF representations.
- **Practical Application Orientation**: Zero raw data exchange satisfies privacy requirements; communication involves only model parameter updates.
- **Custom Dataset**: The MeanGreen multimodal dataset (RGB + LiDAR + IMU) serves as the evaluation platform.

## Limitations & Future Work

- The global model suffers significant performance degradation on complex sequences (e.g., sequence 03); the aggregation strategy is sensitive to data distribution.
- The current formulation assumes a fixed number of Gaussian primitives ($6\times10^5$) and does not support adaptive density control.
- Validation is limited to indoor corridor environments; outdoor and large-scale scene testing is absent.
- Pose construction relies on LiDAR, restricting applicability to scenarios where all agents are equipped with LiDAR.
- Overall PSNR values are relatively low (18–25 dB), potentially due to limited viewpoint diversity from forward-facing cameras.

## Related Work & Insights

- **FedNeRF**: Applies federated learning to NeRF, but geometry and appearance are entangled in the shared MLP parameters of NeRF's implicit representation, precluding selective freezing.
- **Fed3DGS**: Updates the server model via distillation but cannot prevent cross-client geometry drift.
- **CoSurfGS**: Distributed Gaussian reconstruction using a device–edge–cloud hierarchy, but assumes cooperative model sharing.
- **Insight**: The "separability" of 3DGS's explicit representation is a unique advantage for distributed scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Federated 3DGS is a new direction; the geometry-freezing strategy is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐ Validation is limited to a custom dataset; centralized training baselines are absent.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation and method description are clear.
- Value: ⭐⭐⭐⭐ Addresses practical needs for multi-agent collaboration with broad application potential.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Efficient Equivariant Transformer for Self-Driving Agent Modeling](efficient_equivariant_transformer_for_self-driving_agent_modeling.md)
- [\[CVPR 2026\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lrsgs_robust_lidarreflectanceguided_salient_gaussi.md)
- [\[NeurIPS 2025\] Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems](../../NeurIPS2025/autonomous_driving/regret_lower_bounds_for_decentralized_multi-agent_stochastic_shortest_path_probl.md)
- [\[CVPR 2026\] U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences](u4d_uncertainty-aware_4d_world_modeling_from_lidar_sequences.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](../../ICCV2025/autonomous_driving/3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)

<!-- RELATED:END -->
