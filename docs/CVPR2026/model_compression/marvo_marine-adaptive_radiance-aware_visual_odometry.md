---
title: >-
  [Paper Note] MARVO: Marine-Adaptive Radiance-aware Visual Odometry
description: >-
  [CVPR 2026][Model Compression][underwater visual odometry] MARVO is an underwater visual odometry framework that embeds a Physics-Aware Radiance Adapter (PARA) into the LoFTR feature matcher to compensate for wavelength-…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "underwater visual odometry"
  - "physics-aware feature matching"
  - "factor graph optimization"
  - "reinforcement learning pose graph optimization"
  - "multi-sensor fusion"
date: 2026-05-08
content_hash: 3de92974fbebb23a
---

# MARVO: Marine-Adaptive Radiance-aware Visual Odometry

**Conference**: CVPR 2026
**arXiv**: [2511.22860](https://arxiv.org/abs/2511.22860)
**Code**: N/A
**Area**: Model Compression
**Keywords**: underwater visual odometry, physics-aware feature matching, factor graph optimization, reinforcement learning pose graph optimization, multi-sensor fusion

## TL;DR

MARVO is an underwater visual odometry framework that embeds a Physics-Aware Radiance Adapter (PARA) into the LoFTR feature matcher to compensate for wavelength-dependent attenuation, integrates GTSAM multi-sensor factor graph fusion, and employs reinforcement learning-based pose graph optimization (RL-PGO), achieving robust localization in underwater scenes.

## Background & Motivation

Underwater visual localization presents unique challenges: light scattering, **wavelength-dependent attenuation**, and strong non-Gaussian noise lead to severe contrast degradation, unstable features, and long-term pose estimation inconsistency. The failure of conventional VO/SLAM underwater stems from two levels:

**Perception level**: The physical image formation process underwater (chromatic channel attenuation, backscatter) is not corrected, causing feature descriptors to fail in turbid regions. Standard LoFTR suffers significant matching quality degradation under spectral degradation.

**Optimization level**: Standard least-squares solvers (Gauss-Newton/LM) get trapped in local optima on high-noise, visually degraded trajectories, especially when loop closure constraints are sparse.

The core philosophy of MARVO is that robust underwater VO requires both (i) a perception module that explicitly compensates for radiometric distortion and (ii) a global optimizer capable of escaping local optima.

## Method

### Overall Architecture

Three modules in cascade:
1. **Front-end perception**: PARA-augmented LoFTR feature matching → physically corrected semi-dense correspondences
2. **Back-end estimation**: GTSAM factor graph fusing visual-inertial-barometric constraints → real-time VO
3. **Offline optimization**: RL-PGO on SE(2) via reinforcement learning → globally consistent trajectory

### Key Designs

#### 1. Physics-Aware Radiance Adapter (PARA)

- **Function**: A lightweight module inserted between LoFTR's CNN encoder and Transformer layers that explicitly corrects underwater optical degradation.
- **Mechanism**: Based on a revised underwater image formation model:

$$I_c(x) = J_c(x) e^{-\beta_c(x)z(x)} + B_\infty^c(x)(1 - e^{-\beta_c(x)z(x)})$$

PARA employs a three-branch prediction head to estimate per-pixel attenuation coefficients $\hat{\boldsymbol{\beta}} \in \mathbb{R}^{H \times W \times 3}$, asymptotic backscatter $\hat{\mathbf{B}}_\infty \in \mathbb{R}^{H \times W \times 3}$, and a depth proxy $\hat{\mathbf{z}} \in \mathbb{R}^{H \times W \times 1}$ from shared features. Inverting the physical model yields a radiometrically corrected estimate, producing a scalar correction mask:

$$\Gamma(x) = \frac{1}{3}\sum_{c \in \{R,G,B\}} \frac{\hat{J}_c(x)}{I_c(x) + \epsilon}$$

Applied to encoder features as: $\tilde{\mathbf{F}}(x) = \text{LN}(\Gamma(x) \odot \mathbf{F}(x))$

- **Design Motivation**: Rather than image pre-processing, physical correction is directly embedded into the feature pipeline. PARA adds fewer than 5% additional parameters while substantially improving descriptor consistency. Ablations confirm that physics-based supervision—not simple CNN modulation—is the key to robustness.

#### 2. Multi-Sensor Factor Graph Fusion

- **Function**: A fixed-lag smoother is constructed in GTSAM to fuse three types of constraints.
- **Mechanism**:
    - **IMU pre-integration factor**: Provides scale constraints and short-term motion via standard GTSAM pre-integration.
    - **MARVO visual factor**: Relative poses estimated from PARA-LoFTR semi-dense matches, with covariance inversely proportional to inlier count and spatial coverage, allowing high-visibility frames to automatically dominate optimization.
    - **Barometric depth prior**: A unary depth factor that eliminates the vertical drift common in monocular underwater VO.
- **Design Motivation**: Barometric sensors are extremely low-cost yet highly effective against vertical drift. Adaptive covariance allows the system to automatically down-weight degraded frames.

#### 3. Reinforcement Learning Pose Graph Optimization (RL-PGO)

- **Function**: An offline RL policy refines the pose graph on SE(2), surpassing the local optima of classical least-squares methods.
- **Mechanism**:
  1. SE(3) is projected onto SE(2) (AUV/ROV roll and pitch are stable; yaw is the primary rotational degree of freedom; depth is fixed by barometry).
  2. A GNN encoder aggregates edge residuals to generate the state representation.
  3. A recurrent SAC agent selects edges and outputs SE(2) retraction actions.
  4. After refinement, poses are re-embedded into SE(3), followed by a final LM pass for fine-tuning.
- **Key Innovation — Log-Weighted Orientation Cost**:

$$OC_{\text{log}} = \sqrt{\sum_{(i,j) \in E} w_{ij} \|R_i R_{ij} - R_j\|_F^2}$$

$$w_{ij} = 1 + \beta \log\left(\frac{\|\mathbf{t}_{ij}\|}{\bar{t}} + \epsilon\right)$$

The logarithmic sub-linear weighting emphasizes long-range constraints without allowing extremely long noisy edges to dominate. Setting $\beta=0$ reduces to uniform weighting.

- **Design Motivation**: Visual degradation underwater causes classical PGO to suffer from poor initialization and convergence to local optima.

### Loss & Training

Joint front-end loss: $\mathcal{L} = \lambda_{\text{match}}\mathcal{L}_{\text{match}} + \lambda_{\text{photo}}\mathcal{L}_{\text{photo}} + \lambda_{\text{phys}}\mathcal{L}_{\text{phys}}$

- $\mathcal{L}_{\text{match}} = \|\hat{\mathbf{P}} - \mathbf{P}^*\|_1$: geometric consistency of matched points
- $\mathcal{L}_{\text{photo}} = 1 - \text{SSIM}(I'_A, I'_B)$: view consistency after radiometric correction
- $\mathcal{L}_{\text{phys}} = \|\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}_{\text{gt}}\|_1 + \|\hat{\mathbf{B}}_\infty - \mathbf{B}_{\infty,\text{gt}}\|_1$: L1 supervision on physical parameters

Two-stage training: pre-training on ~120k synthetic underwater pairs (ScanNet/TartanAir/Hypersim rendered via SyreaNet) → fine-tuning on ~12k real frames (10% KITTI + in-house data). Mixed-precision training on 4×A100.

## Key Experimental Results

### Main Results

Real-world underwater VO performance (Scale Aligned):

| Method | ATE (m)↓ | RPE (deg/m)↓ | Drift (%)↓ |
|--------|---------|-------------|-----------|
| ORB-SLAM3 | 4.12 | 0.92 | 3.8 |
| LIBVISO2 | 3.47 | 0.85 | 3.1 |
| MAST3R-SLAM | 2.52 | 0.58 | 2.2 |
| VGGT-SLAM | 2.41 | 0.56 | 2.1 |
| **MARVO (Ours)** | **1.73** | **0.34** | **1.2** |

Synthetic underwater feature matching (Pose AUC):

| Method | @5° | @10° | @20° |
|--------|-----|------|------|
| SP+SuperGlue | 25.4 | 42.2 | 59.7 |
| LoFTR | 42.9 | 59.5 | 68.2 |
| **MARVO** | **49.7** | **62.9** | **71.3** |

### Ablation Study

| Configuration | AUC @10°↑ | ATE (m)↓ | Drift (%)↓ |
|---------------|----------|---------|-----------|
| Full MARVO | **0.92** | **1.73** | **1.2** |
| w/o PARA module | 0.81 | 2.24 | 1.9 |
| Replace w/ vanilla LoFTR | 0.76 | 2.47 | 2.3 |
| Classical PGO instead of RL-PGO | 0.84 | 2.05 | 1.7 |
| w/o physical radiance normalization | 0.73 | 2.68 | 2.6 |

### Key Findings

1. **Physical radiance normalization is the core component**: Removing it reduces AUC to 0.73 (largest drop), confirming that physics-based supervision rather than CNN modulation is essential.
2. ATE is reduced by 58% and drift by 68% compared to ORB-SLAM3.
3. RL-PGO reduces ATE from 2.05m (classical PGO) to 1.73m, with particularly pronounced gains in sparse loop closure scenarios.
4. Even compared to the recent VGGT-SLAM, ATE is reduced by 28% and drift by 43%.

## Highlights & Insights

1. **Physical model directly embedded in the deep learning pipeline**: PARA performs physical correction in feature space rather than image space, preserving end-to-end differentiability.
2. The **barometric depth prior** is an elegant design: a low-cost unary factor that entirely eliminates vertical drift.
3. **SE(2)-reduced RL-PGO** cleverly exploits AUV/ROV kinematic constraints, reducing 6-DoF optimization to 3-DoF.
4. Adaptive covariance allows the system to automatically rely on inertial/barometric constraints during visual degradation.

## Limitations & Future Work

1. **Lack of standardized underwater VO datasets**: Evaluation relies on synthetic rendering and COLMAP alignment, providing insufficient statistical significance.
2. The sim-to-real domain gap is addressed only by fine-tuning on 10% real data, offering limited robustness guarantees.
3. RL-PGO operates solely on SE(2); the assumption of coupled roll/pitch may not hold for certain AUV platforms.
4. 3D mapping (TSDF/MVS) is not integrated, and real-time metrics (frame rate/latency) are absent.
5. The experimental scale is small, with no large-scale multi-sequence long-duration evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining physical models with Transformer-based matching is a clear contribution; RL-PGO adapted for underwater scenarios is novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited by the scarcity of underwater datasets; experiments are small-scale and lack error bars and multi-sequence statistics.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is thorough, system design logic is clear, and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to underwater robotics; the physics-aware paradigm is transferable to fog/rain/nighttime localization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[CVPR 2026\] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)

</div>

<!-- RELATED:END -->
