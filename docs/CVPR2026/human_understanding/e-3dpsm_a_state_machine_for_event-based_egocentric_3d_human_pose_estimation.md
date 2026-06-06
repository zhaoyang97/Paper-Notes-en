---
title: >-
  [Paper Note] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation
description: >-
  [CVPR 2026][Human Understanding][event camera] This paper proposes E-3DPSM, an event-camera-based egocentric 3D human pose state machine that formulates pose estimation as a continuous-time state evolution process. It in…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "event camera"
  - "egocentric pose estimation"
  - "state space model"
  - "3D human pose"
  - "temporal consistency"
date: 2026-05-08
content_hash: 20394dcc23bbc87f
---

# E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2604.08543](https://arxiv.org/abs/2604.08543)  
**Code**: [https://4dqv.mpi-inf.mpg.de/E-3DPSM/](https://4dqv.mpi-inf.mpg.de/E-3DPSM/)  
**Area**: Human Understanding
**Keywords**: event camera, egocentric pose estimation, state space model, 3D human pose, temporal consistency

## TL;DR

This paper proposes E-3DPSM, an event-camera-based egocentric 3D human pose state machine that formulates pose estimation as a continuous-time state evolution process. It integrates bidirectional SSM temporal modeling with a learnable Kalman-style fusion module to combine direct and incremental pose predictions, achieving real-time inference at 80Hz with a 19% reduction in MPJPE and a 2.7× improvement in temporal stability.

## Background & Motivation

Egocentric 3D human pose estimation is a core capability for VR/AR applications—including real-time avatar control, fitness tracking, and telepresence—yet faces several critical challenges:

**Inherent limitations of RGB cameras**: High noise in low-light conditions, motion blur from rapid head movements, and the bandwidth/power burden of high-resolution video streams on wearable devices.

**Advantages of event cameras**: Millisecond-level temporal resolution, high dynamic range, and near-zero motion blur, making them naturally suited for fast motion and self-occlusion scenarios.

Existing event-camera methods, EventEgo3D and EventEgo3D++, suffer from the following issues:

- **Architectures not fully adapted to event stream characteristics**: Only a frame buffer storing the previous frame's information is used, failing to exploit the asynchronous, continuous, and change-driven nature of event data.
- **Reliance on 2D heatmaps**: Introduces quantization errors.
- **Requirement for predicted segmentation masks**: Adds an additional source of error.
- **Temporal jitter and drift**: Insufficient 3D accuracy under self-occlusion.

Three core insights motivate this work:
- Events naturally encode 2D spatial changes and should correspond to 3D spatial changes (delta poses).
- The continuous nature of event streams is well-suited to modeling as a continuous process (SSM).
- Intermediate supervision via 2D heatmaps and segmentation masks can be eliminated.

## Method

### Overall Architecture

E-3DPSM processes input in three stages:

1. **Event stream preprocessing**: Raw events are converted into LNES frames $\{\mathbf{L}_t\}_{t=1}^N$ (20ms window, 192×256×2).
2. **Spatiotemporal Pose Encoding Module (SPEM)**: Extracts temporally-aware joint features.
3. **Pose Regression Module (PRM)**: Predicts a direct pose $\mathbf{P}_t^D$ and an incremental pose $\mathbf{P}_t^\Delta$, which are fused via a learnable module to produce the final 3D pose $\mathbf{P}_t$.

### Key Designs

1. **Spatiotemporal Pose Encoding Module (SPEM)**:

    - **Multi-stage convolutional encoding**: A 4-level hierarchical structure, each level containing two residual blocks followed by a downsampling convolution, progressively extracting spatial features.
    - **Deformable Attention**: Appended at the end of each stage, adaptively focusing on pose-critical regions to handle fisheye lens distortion and self-occlusion.
    - **Bidirectional SSM temporal modeling**: Event-specific S5 layers are inserted at stages 2 and 4, independently aggregating long-range temporal context at each spatial location. During training, bidirectional operation leverages full context; during inference, the model can be switched to causal mode for real-time deployment.
    - **Joint query decoder**: 16 learnable joint query embeddings $\mathbf{U}=\{\mathbf{u}_1,\ldots,\mathbf{u}_{16}\}$ interact with the last-stage encoder features via a Transformer Decoder, producing joint-aware features $\mathbf{F}_t \in \mathbb{R}^{16 \times 192}$.

2. **Pose Regression Module (PRM)**: Three components:

    - **Direct pose regression**: An MLP maps joint query features to $\mathbf{P}_t^D \in \mathbb{R}^{16 \times 3}$, serving as a global anchor to prevent drift.
    - **Incremental pose regression**: Current features are concatenated with the previous frame's pose embedding to predict the inter-frame displacement $\mathbf{P}_t^\Delta \in \mathbb{R}^{16 \times 3}$. Since event streams naturally encode changes, incremental regression is easier to learn than absolute position regression.
    - **Learnable Kalman fusion**: The core innovation. An internal state $\mathbf{X}_t$ and covariance $\Sigma_t$ are maintained, adaptively fusing predictions via a motion update step (using delta pose) and a measurement update step (using direct pose). The process noise $\mathbf{Q}$ and observation noise $\mathbf{R}$ are learnable parameters, trained end-to-end and fixed at deployment.

3. **Design Motivation**:

    - Naive summation $\mathbf{P}_t = \mathbf{P}_{t-1}^D + \mathbf{P}_t^\Delta$ leads to cumulative drift.
    - Post-processing Kalman filtering lacks task adaptability.
    - Learnable noise covariances allow the system to automatically balance trust between incremental updates and direct predictions.

### Loss & Training

Multiple losses are jointly supervised with weights $\lambda_{3D} = \lambda_\Delta = \lambda_{2D} = 0.01$ and $\lambda_{BL} = \lambda_{BA} = 10^{-3}$:

$$\mathcal{L}_{total} = \lambda_{3D}\mathcal{L}_{3D} + \lambda_\Delta\mathcal{L}_\Delta + \lambda_{2D}\mathcal{L}_{2D} + \lambda_{BL}\mathcal{L}_{BL} + \lambda_{BA}\mathcal{L}_{BA}$$

- $\mathcal{L}_{3D}$: MSE on 3D joint positions.
- $\mathcal{L}_\Delta$: MSE between predicted incremental poses and ground-truth inter-frame displacements.
- $\mathcal{L}_{2D}$: 2D projection error.
- $\mathcal{L}_{BL}$: Bone length L1 loss to preserve body proportions.
- $\mathcal{L}_{BA}$: Bone direction cosine loss to maintain anatomical plausibility.

Training details:
- **No synthetic data pretraining required** (prior methods require pretraining on EE3D-S).
- Adam optimizer, batch size 32.
- Trained for 15 epochs on EE3D-R ($\eta=10^{-3}$), fine-tuned for 10 epochs on EE3D-W ($\eta=10^{-4}$).
- Trained on 4 × A40 GPUs for 34 hours.

## Key Experimental Results

### Main Results

| Method | EE3D-R MPJPE↓ | EE3D-R PA-MPJPE↓ | EE3D-R $e_{smooth}$↓ | EE3D-W MPJPE↓ | EE3D-W PA-MPJPE↓ | EE3D-W $e_{smooth}$↓ |
|------|-------------|-----------------|---------------------|-------------|-----------------|---------------------|
| EgoPoseFormer | 151.66 | 96.99 | 66.50 | 220.40 | 130.45 | 79.23 |
| EventEgo3D | 110.39 | 84.52 | 27.06 | 195.50 | 108.20 | 45.29 |
| EventEgo3D++ | 103.28 | 77.06 | 22.93 | 172.43 | 98.41 | 40.87 |
| **Ours (Causal)** | **84.45** | **62.64** | **8.40** | **158.86** | **93.46** | **23.57** |
| **Ours (Non-Causal)** | **81.32** | **60.21** | **6.65** | **155.82** | **90.85** | **22.65** |

**Key gains**: ~19% reduction in MPJPE on EE3D-R; 2.7× improvement in $e_{smooth}$ (from 22.93 to 8.40).

**Occluded joint evaluation**:

| Method | EE3D-R Occ MPJPE↓ | EE3D-R Occ PA-MPJPE↓ |
|------|-------------------|---------------------|
| EventEgo3D++ | 88.43 | 49.53 |
| **Ours (Causal)** | **67.49** | **41.85** |

### Ablation Study

**SPEM module ablation**:

| Configuration | MPJPE↓ | PA-MPJPE↓ | $e_{smooth}$↓ | Notes |
|------|--------|----------|-------------|------|
| w/o SSM Blocks | 118.53 | 87.48 | 16.94 | No temporal modeling; severe degradation |
| Single SSM (Stage 4) | 90.18 | 66.06 | 7.84 | Early-stage temporal context is important |
| w/o Deformable Attn | 88.27 | 64.98 | 7.71 | Spatial adaptability is beneficial |

**PRM module ablation**:

| Configuration | MPJPE↓ | PA-MPJPE↓ | $e_{smooth}$↓ | Notes |
|------|--------|----------|-------------|------|
| w/o Fusion (naive sum) | 141.22 | 84.17 | 10.14 | Severe drift |
| Direct Pose Only | 91.26 | 65.43 | 17.22 | High jitter |
| Static Fusion | 88.31 | 65.63 | 9.93 | Inferior to adaptive fusion |
| **Full model** | **84.45** | **62.64** | **8.40** | Best |

### Key Findings

1. **Causal mode approaches non-causal performance**: Switching to causal inference after non-causal training still yields competitive results, indicating strong model generalization.
2. **Strong real-time capability**: 80Hz on an A6000 GPU; 52Hz on a 3050Ti.
3. **No synthetic data pretraining required**: A simpler pipeline than prior methods.
4. **Bone and direction losses are important**: They preserve anatomical plausibility and prevent non-physical predictions.
5. Adding post-processing Kalman smoothing to the baseline still underperforms the proposed method, confirming that the gains stem from architectural design rather than smoothing tricks.

## Highlights & Insights

1. **Deep exploitation of event stream characteristics**: The natural correspondence between "events encoding changes" and "3D delta poses" represents an elegant modality-task alignment.
2. **SSM for event cameras**: This is the first application of S5 layers to egocentric 3D pose estimation; the continuous state evolution of SSMs is inherently compatible with the asynchronous nature of event streams.
3. **Learnable Kalman fusion**: Outperforms both naive summation and fixed-parameter filtering by allowing the model to automatically learn which signal source to trust.
4. **Elimination of intermediate supervision**: Removing 2D heatmaps and segmentation masks simplifies the pipeline and eliminates potential error sources.

## Limitations & Future Work

1. **Severe occlusion combined with high dynamics remains challenging**: The paper acknowledges room for improvement in these extreme scenarios.
2. **Validation limited to fisheye egocentric viewpoints**: Generalizability to third-person perspectives is unknown.
3. **EE3D dataset is relatively small**: Real-world data from laboratory capture is limited; in-the-wild data presents greater difficulty.
4. **16-joint body representation**: Hand and facial joints are not covered; full-body pose estimation requires more joints.
5. **Mamba as an alternative to S5**: Newer SSM architectures may yield further improvements.

## Related Work & Insights

- **EventEgo3D/EventEgo3D++**: Direct predecessors and the primary targets of improvement.
- **S5 / Mamba**: Successful applications of SSMs in the event camera domain.
- **EgoPoseFormer**: An RGB-based method adapted to event input, used as a comparison baseline.
- Insights: The "change encoding" property of event cameras has broader potential in 3D vision (e.g., scene flow, depth estimation); the delta prediction + fusion paradigm is worth generalizing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Formulating pose estimation as a continuous state machine with an elegant delta+direct fusion design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations on two benchmarks, though datasets are relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete derivations, and intuitive figures.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for event-camera-based 3D vision; 80Hz real-time performance has practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](../../ICCV2025/human_understanding/high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)
- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)
- [\[CVPR 2026\] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)

</div>

<!-- RELATED:END -->
