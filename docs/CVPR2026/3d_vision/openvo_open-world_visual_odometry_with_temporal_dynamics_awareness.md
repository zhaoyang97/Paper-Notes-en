---
title: >-
  [Paper Note] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness
description: >-
  [CVPR2026][3D Vision][Visual Odometry] This paper proposes OpenVO, an open-world monocular visual odometry framework that achieves robust metric-scale ego-motion estimation under uncalibrated cameras and varying frame ra…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "Visual Odometry"
  - "Temporal Dynamics Awareness"
  - "Camera-Free"
  - "3D Flow Field"
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 01ee8b3a51b3bfa6
---

# OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness

**Conference**: CVPR2026  
**arXiv**: [2602.19035](https://arxiv.org/abs/2602.19035)  
**Code**: [openvo.github.io](https://openvo.github.io/)  
**Area**: 3D Vision  
**Keywords**: Visual Odometry, Temporal Dynamics Awareness, Camera-Free, 3D Flow Field, Autonomous Driving

## TL;DR

This paper proposes OpenVO, an open-world monocular visual odometry framework that achieves robust metric-scale ego-motion estimation under uncalibrated cameras and varying frame rates, via a time-aware flow encoder and a geometry-aware context encoder. OpenVO achieves over 20% ATE improvement across datasets and reduces error by 46%–92% under variable frame-rate settings.

## Background & Motivation

1. **Dashcam data is abundant but hard to exploit**: Dashcam videos on platforms such as YouTube contain rare driving events (e.g., collisions) and are valuable for trajectory dataset construction, yet they are typically monocular and uncalibrated, with large variation in camera parameters and frame rates.
2. **Existing VO methods assume fixed frame rates**: Methods such as TartanVO, XVO, and ZeroVO are trained and evaluated at fixed frame rates (e.g., 10 Hz, 12 Hz), entirely ignoring temporal dynamics, which leads to severe performance degradation under frame-rate mismatch.
3. **Classical methods require camera calibration**: Geometry-based methods such as ORB-SLAM and DSO require known camera intrinsics and cannot handle uncalibrated open-world observations.
4. **Learning-based methods lack cross-domain generalization**: Early learning-based methods are trained and tested under similar conditions and lack explicit modeling of varying camera geometries, resulting in poor cross-domain performance.
5. **Temporal overfitting is overlooked**: The reinforcement learning and world model communities have demonstrated that training at a fixed sampling rate leads to temporal overfitting, yet this issue has rarely been explored in the VO community.
6. **Scale ambiguity in monocular VO**: Monocular VO is inherently scale-ambiguous; recovering metric-scale from appearance alone is infeasible without geometric priors.

## Method

### Overall Architecture

OpenVO adopts a two-frame pose regression architecture that takes two consecutive frames of dashcam video as input and outputs the SE(3) relative camera pose. The framework consists of three core modules:

- **Time-Aware Flow Encoder**: Encodes frame-rate information into optical flow features.
- **Geometry-Aware Context Encoder**: Fuses depth and camera intrinsic priors.
- **World-Coordinate Egomotion Decoder**: Regresses translation and rotation.

### Time-Aware Flow Encoder

**Time Condition Layers**: The frame rate $f$ is mapped to a temporal interval $\Delta t = 1/f$, expanded into a high-dimensional embedding $\text{PE}(\Delta t)$ via sinusoidal positional encoding, and then passed through two linear layers to produce affine transformation parameters $\alpha, \beta$ that modulate the optical flow correlation features:

$$\tilde{F^c} = (1 + \alpha) \odot F^c + \beta$$

The modulated features are refined through 4 layers of self-attention to capture spatial correlations, enabling the network to reason about motion structure while remaining aware of temporal dynamics.

**Differentiable 2D-Guided 3D Flow**: Using 2D optical flow (MaskFlowNet) and metric depth (Metric3Dv2), pixels are back-projected into 3D points via perspective unprojection: $P_1 = D_1 \cdot K^{-1} p_1$. Optical flow warps pixels to sub-pixel locations in the second frame, where depth is bilinearly sampled and back-projected to obtain $P_2$, yielding dense 3D flow $(P_2 - P_1)$. The entire process is fully differentiable and end-to-end trainable. The 3D flow passes through 4 self-attention layers and is fused with the temporally modulated optical flow features to form the **Time-Aware Flow Feature**.

### Geometry-Aware Context Encoder

- **Camera Tokenizer**: WildCamera is used to estimate camera intrinsics $K$, constructing a normalized ray field $r(u,v) = K^{-1}[u,v,1]^\top$ that encodes the 3D viewing direction of each pixel.
- **Depth Tokenizer**: Metric3Dv2 estimates metric depth $D$; ray directions are scaled by depth values as $M(u,v) = D(u,v) \cdot r(u,v)$ to obtain metric-scale 3D point distributions.
- The concatenation $[r, M, D]$ is assembled into a token set and fed into an 8-layer self-attention encoder to produce a unified geometric embedding.

### Decoder and Loss

The Time-Aware Flow Feature and Geometry-Aware Context Feature are concatenated and passed through two MLP branches to regress:

- **Rotation**: A Fisher matrix $\mathcal{F} \in \mathbb{R}^{3\times3}$ is predicted and mapped to SO(3) via the Matrix Fisher distribution, modeling directional uncertainty.
- **Translation**: A metric-scale regression module directly predicts world-coordinate displacement $t_i \in \mathbb{R}^3$.

**Multi-Temporal-Scale Training**: Videos at original frame rate $f_0$ are temporally subsampled by a factor $k$ to generate training samples at $f_0/k$ (e.g., 12 Hz → 6 Hz / 4 Hz), exposing the model to diverse temporal scales. Gradient clipping is applied during training for stability.

## Key Experimental Results

### Cross-Dataset Generalization (trained only on nuScenes Singapore-OneNorth)

| Method | KITTI ATE | nuScenes ATE | Argoverse2 ATE |
|--------|-----------|-------------|-----------------|
| TartanVO | 103.07 | 6.26 | 7.03 |
| ZeroVO‡ | 123.42 | 8.40 | 5.71 |
| XVO | 168.43 | 8.30 | 5.70 |
| **OpenVO✓** | **93.23** | **5.91** | **2.39** |

OpenVO achieves a 24% improvement over ZeroVO‡ on KITTI ATE and 58% on Argoverse2.

### Variable Frame-Rate Robustness (selected from Tab. 4)

| Setting | OpenVO ATE | ZeroVO‡ ATE | Improvement |
|---------|-----------|-------------|-------------|
| KITTI 2.5Hz | 368.47 | 553.52 | −33% |
| nuScenes 6Hz | 6.07 | 21.55 | −72% |
| Argoverse2 20Hz | 6.47 | 36.14 | −82% |

OpenVO reduces error by 46%–92% across all variable frame-rate settings.

### Ablation Study

- **Temporal encoding dimension**: $K=8$ (PE dimension 17) yields the best performance; smaller values underfit temporal variation while larger values introduce high-frequency oscillation.
- **Training frequency combination**: \{12/6/4\} Hz is optimal; removing the Time Condition Layers increases KITTI ATE from 93.23 to 152.42 (+64%), confirming the necessity of explicit temporal awareness.
- **Differentiable vs. non-differentiable 3D flow**: The differentiable variant reduces KITTI ATE from 109.01 to 93.23, providing more consistent trajectory predictions.

## Highlights & Insights

- **First temporal dynamics modeling in VO**: Injecting frame-rate information into optical flow features via sinusoidal positional encoding and affine modulation is an elegant and effective solution to temporal overfitting.
- **Fully differentiable 2D→3D flow construction**: Unifying 2D optical flow, metric depth, and estimated intrinsics in an end-to-end differentiable pipeline yields significantly better results than non-differentiable counterparts.
- **Calibration-free open-world VO**: By leveraging foundation model priors from WildCamera and Metric3Dv2, metric-scale recovery is achieved without ground-truth intrinsics.
- **Multi-temporal-scale training strategy**: Frame-skipping augmentation exposes the model to diverse frame rates; combined with Time Condition Layers, this enables strong generalization to unseen frame rates.
- **Dominant advantage under variable frame rates**: Compared to ZeroVO, OpenVO reduces error by up to 92% in variable frame-rate evaluations, demonstrating strong practical value.

## Limitations & Future Work

- **Independent depth and intrinsic estimation**: Metric3Dv2 and WildCamera operate independently; errors may propagate in cascade to the final VO output without joint optimization.
- **Multi-temporal-scale schedule is empirically chosen**: The \{12/6/4\} Hz training frequency combination is manually specified; an adaptive sampling strategy may be more principled.
- **Mixed-frequency training introduces inconsistent gradients**: Local trajectory segment errors ($t_{err}$, $r_{err}$) on KITTI are slightly higher than some baselines, attributed to inconsistent parameter updates from multi-frequency training.
- **High training cost**: 96 GPU hours on A6000, which is resource-intensive for constrained settings.
- **Extreme scenarios not validated**: Cases such as very low frame rates (<2 Hz), severe occlusion, and dense dynamic scenes remain untested.

## Related Work & Insights

| Method | Requires Calibration | Temporal Awareness | 3D Geometric Prior | Extra Data |
|--------|---------------------|-------------------|-------------------|------------|
| ORB-SLAM3 | ✓ | ✗ | ✗ | ✗ |
| TartanVO | ✓ (GT intrinsics) | ✗ | ✗ | ✗ |
| XVO | ✗ | ✗ | ✗ | YouTube pseudo-labels |
| ZeroVO | ✗ | ✗ | ✓ (3D flow + language) | YouTube + text |
| **OpenVO** | **✗** | **✓** | **✓ (differentiable 3D flow)** | **✗** |

OpenVO is the only calibration-free VO method that simultaneously incorporates temporal dynamics awareness and geometric priors without relying on additional data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first work to introduce temporal dynamics awareness into VO; the affine modulation design of Time Condition Layers is concise and elegant; the differentiable 2D→3D flow is also a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three major autonomous driving benchmarks with both standard and variable frame-rate evaluations and comprehensive ablations; however, more extreme frame rates and other sensor types are not tested.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, method description is fluent, figures and tables are informative, and the problem formulation is well-defined.
- **Value**: ⭐⭐⭐⭐ — Addresses practical challenges in open-world dashcam trajectory reconstruction, with direct applicability to autonomous driving data collection and YouTube-scale video analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space](raynova_scale-temporal_autoregressive_world_modeling_in_ray_space.md)
- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[CVPR 2026\] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI](wanderland_geometrically_grounded_simulation_for_open-world_embodied_ai.md)
- [\[ICCV 2025\] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness](../../ICCV2025/3d_vision/ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)
- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](../../NeurIPS2025/3d_vision/towards_3d_objectness_learning_in_an_open_world.md)

</div>

<!-- RELATED:END -->
