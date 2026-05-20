---
title: >-
  [Paper Note] Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization
description: >-
  [AAAI 2026][Human Understanding][3D human pose estimation] To address error accumulation in online test-time adaptation (TTA) for 3D human pose estimation…
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "3D human pose estimation"
  - "test-time adaptation"
  - "motion discretization"
  - "error accumulation"
  - "personalized adaptation"
  - "soft reset"
  - "self-replay"
date: 2026-05-08
content_hash: a5333e353fb2fb62
---

# Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization

**Conference**: AAAI 2026
**arXiv**: [2511.18851](https://arxiv.org/abs/2511.18851)  
**Code**: To be confirmed  
**Area**: Human Understanding
**Keywords**: 3D human pose estimation, test-time adaptation, motion discretization, error accumulation, personalized adaptation, soft reset, self-replay

## TL;DR

To address error accumulation in online test-time adaptation (TTA) for 3D human pose estimation, this paper proposes a framework combining motion discretization (an anchor motion set obtained via unsupervised clustering), a self-replay mechanism, and a soft reset strategy. The approach enables robust long-term continuous adaptation by leveraging subject-specific body shape and habitual motion patterns, outperforming all existing online TTA methods on Ego-Exo4D and 3DPW.

## Background & Motivation

Pre-trained 3D human pose estimators suffer significant performance degradation when deployed in real-world scenarios outside the training domain. Online TTA mitigates this by continuously updating the model in a self-supervised manner on unlabeled streaming video. However, existing methods exhibit two critical limitations:

**Error accumulation**: Self-supervised signals derived from imperfect 2D detections and 3D estimations cause prediction errors to compound over time, leading to performance degradation after prolonged adaptation.

**Underutilization of personal characteristics**: A given subject has consistent body shape and habitual motion patterns; continuous observation should capture these personalized features to improve estimation accuracy, yet error accumulation makes sustained adaptation risky.

Specifically, BOA/DynaBOA suffer from inaccurate depth estimation due to insufficient 3D guidance; CycleAdapt over-relies on imperfect estimates as pseudo-labels, falling into a vicious cycle under long-term self-supervision.

## Method

### Overall Architecture

The framework consists of two alternately adapted components:
- **Pose estimator F**: ResNet-50 backbone that regresses SMPL parameters (pose $\theta$, shape $\beta$, translation $\psi$) from images.
- **Motion denoising network M**: Autoencoder architecture (Encoder $E$ + Decoder $D$ + Codebook $\mathcal{C}$) that processes continuous pose sequences of 16 frames at 15 fps.

Both components are alternately updated over 12 cycles per batch (160 frames at 30 fps): $F$ produces pose estimates, $M$ denoises and discretizes them to yield anchor and denoised motions, which in turn guide the adaptation of $F$.

### Key Design 1: Motion Discretization

During pre-training of $M$, the latent space is subjected to **unsupervised clustering** to construct a residual codebook $\mathcal{C} = \{C^i \in \mathbb{R}^{N_c \times d} | i=1,...,k\}$ ($k=3$ layers $\times$ 512 codewords $\times$ 512 dimensions), where each layer recursively retrieves the nearest codeword and subtracts the residual.

At test time:
- The output $\theta_{1:t}$ of $F$ is encoded into a latent vector $z = E(\theta_{1:t})$ and quantized to obtain $c = \sum_{i=1}^k c_i$.
- The anchor motion $\theta^* = D(c)$ is decoded: discretization **filters high-frequency noise** while preserving core motion patterns.
- The anchor loss $L_{ach} = ||\theta - \text{sg}(\theta^*)||$ regularizes the updates of $F$.

**Key insight**: Discretization acts as an information bottleneck — while the denoised motion $\theta'$ may be contaminated by imperfect self-supervision, the anchor motion $\theta^*$ effectively removes errors through codebook quantization, providing a reliable regularization signal.

### Key Design 2: Self-Replay

Online adaptation of $M$ induces **representation drift** — latent codewords gradually lose the ability to decode consistent and regular anchor motions. To address this, a self-replay mechanism is designed:

1. Randomly sample codewords $\bar{c}$ from the pre-trained codebook $\bar{\mathcal{C}}$.
2. Decode replay motions $\bar{\theta}_{1:t} = \bar{D}(\bar{c})$ using the pre-trained decoder $\bar{D}$.
3. Jointly update $M$ using reconstruction losses on both replay motions and test-time estimates:

$$L_M = ||\bar{\theta}'_{1:t} - \text{sg}(\bar{\theta}_{1:t})|| + ||\theta'_{1:t} - \text{sg}(\theta_{1:t})||$$

The codebook is simultaneously updated via EMA (decay 0.999) of replay latent vectors to keep it synchronized with the evolving latent space. **No access to the original pre-training data is required**, addressing privacy concerns.

### Key Design 3: Soft Reset

After adapting each batch, an EMA reset is applied to $F$:

$$F \leftarrow \mu_F \cdot F_{pre} + (1 - \mu_F) \cdot F$$

where $\mu_F = 0.95$. This reduces the impact of noisy updates on individual batches while retaining key personalized features learned from historical adaptation. Compared to full reset ($\mu_F=1$, discarding all adaptation) and no reset ($\mu_F=0$, noise accumulation), this achieves the optimal balance.

### Loss & Training

The total loss for pose estimator $F$:

$$L_F = L_p + \lambda_1 L_s + \lambda_2 L_{2D} + \lambda_3 L_{ach}$$

where $L_p$ is the denoised motion pseudo-label loss, $L_s$ is the shape consistency loss, $L_{2D}$ is the 2D reprojection error, and $L_{ach}$ is the anchor motion loss ($\lambda_1=0.001, \lambda_2=0.1, \lambda_3=0.3$).

## Key Experimental Results

### Main Results: Comparison with Online TTA Methods

| Method | Ego-Exo4D MPJPE↓ | Ego-Exo4D PA↓ | 3DPW MPJPE↓ | 3DPW PA↓ | 3DPW MPVPE↓ |
|------|:-:|:-:|:-:|:-:|:-:|
| Pre-trained F | 205.8 | 116.5 | 230.3 | 123.4 | 253.4 |
| BOA† | 135.1 | 70.0 | 98.2 | 55.8 | 114.2 |
| DynaBOA† | 153.3 | 71.6 | 139.7 | 63.8 | 155.1 |
| CycleAdapt | 145.0 | 80.5 | 141.0 | 79.6 | 155.6 |
| **Ours (OpenPose)** | **121.5** | **68.1** | **83.9** | **51.6** | **100.3** |
| **Ours (ViTPose)** | **116.4** | **60.8** | **85.0** | **53.3** | **100.4** |

†Uses original pre-training data. MPJPE in mm, lower is better.

### Ablation Study: Ego-Exo4D Full Scenario

| Soft Reset | Anchor Loss | Self-Replay | MPJPE↓ | PA↓ |
|:------:|:------:|:------:|:------:|:---:|
| ✗ | ✗ | ✗ | 144.3 | 80.6 |
| ✓ | ✗ | ✗ | 122.9 | 69.2 |
| ✗ | ✓ | ✓ | 138.2 | 74.8 |
| ✓ | ✓ | ✓ | **121.5** | **68.1** |

| Soft Reset Decay $\mu_F$ | MPJPE↓ |
|:-:|:-:|
| 0 (no reset) | 138.2 |
| 0.9 | 122.7 |
| **0.95** | **121.5** |
| 1.0 (full reset) | 125.3 |

### Key Findings

1. **Synergy of motion discretization, self-replay, and soft reset**: Soft reset alone reduces MPJPE from 144.3 to 122.9; adding motion discretization (with self-replay) further reduces it to 121.5; using the anchor loss alone (without self-replay) increases error due to representation drift.
2. **Sustained adaptation is effective**: Compared to resetting to pre-trained weights at each step (125.3), continuous adaptation achieves 121.5, demonstrating successful capture of personalized features.
3. **Competitive with domain generalization methods**: The proposed method with a ResNet-50 backbone surpasses HMR-2.0b (125.2 MPJPE) on Ego-Exo4D, which uses a ViT-H/16 backbone with more training data.
4. **Basketball scene challenges**: Distortion from fisheye cameras limits the effectiveness of continuous adaptation for some participants, revealing boundary conditions of the method.
5. **Runtime efficiency**: Processing 160 frames (5.3 seconds of video) takes only 6.8 seconds, approaching real-time performance.

## Highlights & Insights

- The design concept of **motion discretization as an information bottleneck** is elegant — codebook quantization automatically filters self-supervised noise without the need to manually design thresholds or quality metrics.
- The **self-replay mechanism** is notable for requiring no access to original training data; it "recalls" regular motions solely via the pre-trained codebook and decoder.
- The proposed **personalized TTA paradigm** is conceptually insightful — rather than treating each video clip independently, it leverages long-term observations of the same subject.
- The work balances theoretical rigor and practical validation: statistical significance testing (Wilcoxon test) confirms the consistent advantages of the method.

## Limitations & Future Work

1. The ResNet-50 backbone is used to align with prior work; the effectiveness of modern backbones such as ViT remains unverified.
2. Adaptation to severe distortion (edge regions after fisheye camera rectification) is limited.
3. The quality of replay motions is constrained by the diversity of pre-training data and codebook capacity.
4. Validation is conducted only on third-person/exocentric views; egocentric/first-person views with severe self-occlusion are not explored.
5. Loss weights and EMA decay coefficients are fixed hyperparameters without adaptive adjustment.

## Related Work & Insights

- **BOA / DynaBOA** (Guan et al., 2021/2022): Online bilevel adaptation with GT 2D keypoints and source-domain exemplars; lacks 3D guidance.
- **CycleAdapt** (Nam et al., 2023): Cyclically adapts $F$ and $M$, but does not address error accumulation.
- **TokenHMR** (Dwivedi et al., 2024): ViT-H backbone with quantized pose representations for domain generalization.
- **VQ-VAE motion generation** (Zhang et al., 2023; Guo et al., 2024): Discrete motion tokens for GPT-style generation.

## Rating

- Novelty: ⭐⭐⭐⭐ (The idea of suppressing error accumulation via motion discretization is novel; the personalized TTA paradigm is forward-looking.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two major datasets + 30 participants + per-scenario ablations + statistical tests + runtime analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; problem motivation is well articulated.)
- Value: ⭐⭐⭐⭐ (Long-term online adaptation is a critical issue for practical deployment; the solution is practical and efficient.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization](../../NeurIPS2025/human_understanding/cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] Face Time Traveller: Travel Through Ages Without Losing Identity](../../CVPR2026/human_understanding/face_time_traveller_travel_through_ages_without_losing_identity.md)
- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](../../ICCV2025/human_understanding/perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](../../ICCV2025/human_understanding/bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
