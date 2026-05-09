---
title: >-
  [Paper Note] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][World Model] This paper proposes DriveX, a self-supervised world model framework that learns transferable general scene representations in a BEV latent space via Omni Scene Modeling (OSM)—jointly supervising 3D point cloud prediction, 2D semantic representation, and image generation. A Future Spatial Attention (FSA) paradigm is designed to seamlessly integrate predicted future states into downstream tasks such as occupancy prediction, flow estimation, and end-to-end driving, achieving state-of-the-art performance across multiple tasks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - World Model
  - Self-supervised Learning
  - BEV Representation
  - Point Cloud Prediction
  - End-to-End Driving
date: 2026-05-08
content_hash: 5c4030c5a7e1a159
---

# DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2505.19239](https://arxiv.org/abs/2505.19239)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: World Model, Self-supervised Learning, BEV Representation, Point Cloud Prediction, End-to-End Driving

## TL;DR

This paper proposes DriveX, a self-supervised world model framework that learns transferable general scene representations in a BEV latent space via Omni Scene Modeling (OSM)—jointly supervising 3D point cloud prediction, 2D semantic representation, and image generation. A Future Spatial Attention (FSA) paradigm is designed to seamlessly integrate predicted future states into downstream tasks such as occupancy prediction, flow estimation, and end-to-end driving, achieving state-of-the-art performance across multiple tasks.

## Background & Motivation

### Core Problem

World models predict future environment states from historical observations and ego-vehicle actions, and are regarded as a key direction for improving the safety and robustness of autonomous driving. The central question is: **how to construct a unified world model framework that extracts generalizable world representations while simultaneously benefiting diverse downstream driving tasks?**

### Limitations of Prior Work

Existing world model approaches fall into three categories:

**Pre-training strategies** (e.g., ViDAR): world modeling is used as pre-training followed by fine-tuning, but the general knowledge is destroyed during fine-tuning.

**Auxiliary supervision strategies**: world modeling is treated as an auxiliary head, which also fails to preserve general world knowledge.

**Video generation strategies**: video generation techniques are used to predict future frames, but lack 3D spatial information and spatiotemporal consistency.

The root cause is that prior methods either focus solely on geometric cues (point cloud prediction) or rely on costly semantic annotations (occupancy prediction), making it impossible to simultaneously achieve scalable training and comprehensive scene understanding.

### Core Idea

Decouple world representation learning from future state decoding. Rich world knowledge is encoded in a BEV latent space via multi-modal self-supervised signals (depth + semantics + color), enabling the model to retain strong task transfer capability even with frozen parameters.

## Method

### Overall Architecture

DriveX involves two-stage training:
1. **World representation learning**: a World Encoder is trained to encode multi-view images into BEV features, supervised by the OSM module with multi-modal signals.
2. **Latent future decoding**: the World Encoder is frozen; a Future Decoder is trained to predict future states in the BEV latent space.

Downstream applications adopt the Future Spatial Attention (FSA) paradigm, where task-specific queries aggregate information from predicted future BEV features.

### Key Designs

#### 1. Omni Scene Modeling (OSM)

- **Function**: Unifies multi-modal supervision signals to train the World Encoder.
- **Mechanism**: BEV features $B_t$ are transformed into voxel features $F_t$ via a Channel-to-Height module. Ray sample points are drawn along ray paths, and volume rendering predicts depth, semantics, and color:
  $$\alpha_i = \sum_{j=1}^{n} \tau_j (1-\exp(-\sigma_{i,j}\delta_j))\alpha_{i,j}$$
  Camera-view rays optimize semantic and color losses; LiDAR-view rays optimize depth loss.
- **Design Motivation**: A single supervision signal (e.g., depth only) cannot capture complete scene information. Semantic labels are obtained from foundation models (Grounded SAM, OpenSeeD), enabling fully self-supervised training.

#### 2. Decoupled Latent World Modeling

- **Function**: Separates representation learning and temporal dynamics modeling into distinct training stages.
- **Mechanism**:
    - The World Encoder is first trained (40 epochs) to obtain high-quality BEV representations.
    - The Future Decoder is then trained (24 epochs) to model temporal evolution in the frozen BEV space.
    - The Future Decoder employs a flow-based strategy to directly predict future states (non-autoregressively), avoiding error accumulation:
    $$g_f' = T_t^{t+k} g_f$$
    - Predicted BEV features are converted to a grid representation via distance-weighted interpolation.
- **Design Motivation**: Joint training causes mutual interference between representation learning and dynamics modeling; decoupling reduces Chamfer Distance by 0.44 m².

#### 3. Dynamic-aware Ray Sampling

- **Function**: Prioritizes sampling rays in motion-salient regions during scene modeling at future time steps.
- **Mechanism**: An offline tracker identifies regions of interest (RoI) for moving objects, and additional rays are sampled within these regions.
- **Design Motivation**: The majority of rays correspond to static background, causing dynamic objects to be underrepresented.

#### 4. Future Spatial Attention (FSA)

- **Function**: A unified paradigm for integrating world model predictions into downstream tasks.
- **Mechanism**: Task queries $q$ aggregate information from BEV features across multiple future time steps via spatial attention:
  $$q := q + \sum_{k=1}^{K}\sum_{j=1}^{J} A_{kj} W \hat{B}_{t+k}[T_t^{t+k}(p + \Delta p_{kj})]$$
- **Design Motivation**: Different tasks can adapt by adjusting sampling offsets $\Delta p_{kj}$ without modifying existing architectures.

### Loss & Training

- Representation learning: $\mathcal{L}_{scene}^{camera} = \mathcal{L}_{sem} + \mathcal{L}_{rgb}$, $\mathcal{L}_{scene}^{LiDAR} = \mathcal{L}_{depth}$
- Future decoding: $\mathcal{L}_{future} = \sum_{k=1}^{F}(\omega_l \mathcal{L}_{latent}^k + \omega_s \mathcal{L}_{scene}^k)$, where $\omega_l=1.0, \omega_s=0.5$
- Both stages are fully self-supervised.

## Key Experimental Results

### Main Results

**Point Cloud Prediction (nuScenes, Chamfer Distance m² ↓)**

| Method | Modality | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s |
|--------|----------|------|------|------|------|------|------|
| 4D-Occ | L | 0.91 | 1.13 | 1.30 | 1.53 | 1.72 | 2.11 |
| ViDAR | C | 1.01 | 1.12 | 1.25 | 1.38 | 1.54 | 1.73 |
| HERMES | C | - | 0.78 | - | 0.95 | - | 1.17 |
| **DriveX-B** | C | **0.55** | **0.66** | **0.75** | **0.86** | **0.97** | **1.10** |

**End-to-End Driving (NAVSIM test, PDMS ↑)**

| Method | NC↑ | DAC↑ | TTC↑ | EP↑ | PDMS↑ |
|--------|-----|------|------|-----|-------|
| TransFuser | 97.4 | 92.8 | 92.4 | 79.0 | 83.8 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 79.3 | 84.0 |
| **DriveX-S** | 97.5 | **94.0** | **93.0** | **79.7** | **84.5** |

### Ablation Study

**OSM Multi-modal Supervision Component Ablation**

| Depth | Semantics | Color | mIoU | IoU_geo | mAVE↓ |
|-------|-----------|-------|------|---------|-------|
| ✓ | | | 3.96 | 59.7 | 1.388 |
| ✓ | ✓ | | 42.53 | 73.1 | 0.396 |
| ✓ | ✓ | ✓ | **43.47** | **73.44** | **0.385** |

**Decoupled Training Strategy Ablation**

| Training Strategy | 1.0s CD↓ | 2.0s CD↓ | 3.0s CD↓ | Avg↓ |
|-------------------|----------|----------|----------|------|
| Joint training | 1.09 | 1.41 | 1.75 | 1.41 |
| Decoupled training | 0.83 | 1.03 | 1.31 | 1.06 |
| Decoupled + dynamic sampling | **0.80** | **0.99** | **1.28** | **1.02** |

### Key Findings

- With frozen parameters, DriveX drops only 0.1 PDMS on NAVSIM, whereas ViDAR drops 6.6, demonstrating that genuinely generalizable world knowledge has been learned.
- Direct prediction outperforms autoregressive decoding (Avg CD: 1.02 vs. 1.32), avoiding error accumulation.
- Scaling data from 50% to 100% yields a substantial improvement of 0.51 m², demonstrating favorable scaling properties.
- On occupancy prediction, mIoU improves by 0.93 and mAVE decreases by 0.027, confirming that FSA effectively transfers predictive information.

## Highlights & Insights

1. **Frozen-parameter evaluation is a critical metric for validating general representations**: The experimental design in Table 1 is highly convincing—near-lossless performance with frozen world model parameters indicates that the BEV features encode task-agnostic general world knowledge.
2. **Complementarity of multi-modal self-supervision**: Depth-only supervision yields a mIoU of only 3.96 (almost no semantic information); adding semantics raises it to 42.53, and color provides an additional gain of 0.94—all three signals are indispensable.
3. **Simplicity and efficiency of decoupled training**: Using the same total training budget, decoupled training substantially outperforms joint training, representing a practical and generalizable design strategy.
4. **Unifying nature of the FSA paradigm**: A single world model simultaneously serves occupancy prediction, flow estimation, and planning, requiring only adjustment of sampling offsets.

## Limitations & Future Work

1. **Validation limited to nuScenes and NAVSIM**: Generalization to larger-scale datasets (e.g., Waymo, Argoverse) has not been verified.
2. **Inference latency increases by approximately 42–48 ms**: Further optimization is needed for scenarios with stringent real-time requirements.
3. **Semantic label quality depends on foundation models**: Errors from Grounded SAM and OpenSeeD propagate into the training process.
4. **Multi-modal sensor input unexplored**: The current framework uses only cameras and LiDAR, without considering other sensors such as radar.

## Related Work & Insights

- ViDAR pioneered the pre-training paradigm for vision-only point cloud prediction, but the learned knowledge is easily overwritten during fine-tuning.
- HERMES represents the previous state-of-the-art in point cloud prediction; DriveX substantially surpasses it at all time horizons.
- DiffusionDrive achieves strong performance in end-to-end driving; DriveX further improves upon it via FSA.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of multi-modal self-supervision, decoupled training, and FSA is innovative, though each individual component is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across four tasks (point cloud prediction, occupancy prediction, flow estimation, end-to-end driving) with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear; method and experiments are well-organized.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a unified framework for autonomous driving world models; the frozen-parameter and scaling experiments demonstrate strong practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[ICCV 2025\] Passing the Driving Knowledge Test](passing_the_driving_knowledge_test.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](../../CVPR2026/autonomous_driving/vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)

<!-- RELATED:END -->
