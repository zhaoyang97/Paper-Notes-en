---
title: >-
  [Paper Note] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance
description: >-
  [NeurIPS 2025][Autonomous Driving][3D Semantic Scene Completion] This paper proposes FlowScene, which leverages optical flow to guide temporal feature aggregation and employs occlusion masks for voxel refinement. Using only 2 historical frames as input, FlowScene achieves state-of-the-art performance on the SemanticKITTI and SSCBench-KITTI-360 benchmarks (mIoU 17.70 / 20.81).
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "3D Semantic Scene Completion"
  - "Optical Flow Guidance"
  - "Temporal Modeling"
  - "Occlusion Awareness"
  - "Voxel Refinement"
date: 2026-05-08
content_hash: 6f60ff555281c81e
---

# FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance

**Conference**: NeurIPS 2025
**arXiv**: [2502.14520](https://arxiv.org/abs/2502.14520)  
**Authors**: Meng Wang, Fan Wu, Ruihui Li, Yunchuan Qin, Zhuo Tang, Kenli Li (Hunan University)
**Code**: [https://github.com/willemeng/FlowScene](https://github.com/willemeng/FlowScene)  
**Area**: Autonomous Driving / 3D Semantic Scene Completion
**Keywords**: 3D Semantic Scene Completion, Optical Flow Guidance, Temporal Modeling, Occlusion Awareness, Voxel Refinement

## TL;DR

This paper proposes FlowScene, which leverages optical flow to guide temporal feature aggregation and employs occlusion masks for voxel refinement. Using only 2 historical frames as input, FlowScene achieves state-of-the-art performance on the SemanticKITTI and SSCBench-KITTI-360 benchmarks (mIoU 17.70 / 20.81).

## Background & Motivation

3D Semantic Scene Completion (SSC) is a core perception task in autonomous driving, aiming to jointly infer 3D geometric structure and semantic labels from sparse observations. Existing methods exhibit two categories of limitations:

**Single-frame methods** (MonoScene, CGFormer, etc.): Rely solely on the current frame's limited observations to recover 3D geometry and semantics, leading to insufficient information.

**Existing temporal methods** (VoxFormer-T, HTCL, etc.): Naively stack historical frame features or align features via estimated camera poses, ignoring scene motion context and failing to achieve temporal consistency.

**Core Problem**: How to accurately identify correspondences between historical and current frames to effectively guide temporal SSC modeling?

Optical flow naturally encodes pixel-level inter-frame motion correspondences, capturing motion, viewpoint variation, occlusion, and deformation. This paper proposes to leverage optical flow to guide temporal modeling, injecting motion-aware and occlusion information into the SSC pipeline.

## Method

### Overall Architecture

FlowScene comprises four core modules:

1. **Image Encoder**: RepViT + FPN to extract current frame features $F_t$ and historical frame features $F_{temp}$
2. **Optical Flow Estimation (OFE)**: Pre-trained GMFlow generates bidirectional optical flow; forward-backward consistency check yields occlusion mask $M$
3. **Flow-Guided Temporal Aggregation (FGTA)**: Aligns and aggregates temporal features in 2D feature space using optical flow
4. **Occlusion-Guided Voxel Refinement (OGVR)**: Adaptively fuses voxel features in 3D voxel space using the occlusion mask

### Optical Flow Estimation and Occlusion Detection

**Flow-Guided Warping**: GMFlow estimates the flow field $Flow^{t \to t-i}$ from the current frame to historical frames; bilinear interpolation warps historical frame features to the current frame coordinate system:

$$F_{warp}^{t-i \to t} = \mathcal{W}arp(F_{t-i}, Flow^{t \to t-i})$$

**Occlusion Detection**: Classical forward-backward consistency check is employed. For each pixel, the forward flow maps it to the historical frame, and the backward flow maps it back. If the round-trip residual exceeds threshold $\tau$, the pixel is marked as occluded:

$$M(x) = \begin{cases} 1 & \text{if } \|\Delta(x)\| > \tau \quad \text{(occluded)} \\ 0 & \text{otherwise} \end{cases}$$

This detection approach requires zero parameters and zero training overhead, yet effectively identifies occluded regions.

### Flow-Guided Temporal Aggregation (FGTA)

The FGTA module performs temporal feature alignment and aggregation in 2D image feature space, consisting of two sub-modules:

**(1) Temporal Aggregation**: For warped historical frame features, spatial position weights are computed via cosine similarity:

$$w_{t-i \to t}(P) = \text{similarity}(F_{warp}^{t-i \to t}(P), F_t(P))$$

Weighted aggregation yields the aggregated feature $F_{agg} = \sum_{i=0}^{t} w_{t-i \to t} \cdot F_{warp}^{t-i \to t}$. Intuitively, historical features more similar to the current frame after warping receive higher weights, achieving motion-aware feature fusion.

**(2) Occlusion Cross-Attention**: A Neighborhood Cross-Attention (NCA) mechanism selectively supplements current frame features with information from **non-occluded regions** of historical frames:

$$F_t = \mathcal{NCA}(F_t, (1 - M) \cdot F_{warp})$$

Here the current frame features serve as queries, while warped features from non-occluded historical regions serve as keys/values. This prevents unreliable occluded region information from being injected into the current frame, while exploiting texture and contextual information that is visible in historical frames but occluded in the current frame.

### Occlusion-Guided Voxel Refinement (OGVR)

After FGTA completes temporal fusion in 2D space, explicit geometric modeling in 3D space is still lacking. The OGVR module elevates occlusion awareness to the 3D voxel space.

First, via LSS (Lift-Splat-Shoot) view transformation, $F_t$, $F_{agg}$, and $M$ are projected into 3D voxel space to obtain $V_t$, $V_{agg}$, and $V_{mask}$, respectively.

The two voxel features are then adaptively fused using the occlusion mask:

$$V_{fine} = \frac{(1 - V_{mask}) \cdot V_{agg} + V_t}{(1 - V_{mask}) + 1}$$

Design rationale:
- **Non-occluded regions**: Prioritize aggregated features $V_{agg}$ (fusing multi-frame information)
- **Occluded regions**: Use current frame features $V_t$ (historical frames are unreliable in these areas)
- **Normalization**: Ensures smooth feature transitions at occlusion boundaries, avoiding abrupt changes

The OGVR module introduces **zero additional parameter overhead**, achieving significant gains solely through occlusion-mask-based weighting.

### Loss & Training

The overall loss function comprises four terms:

$$\mathcal{L} = \lambda_{sem}\mathcal{L}_{scal}^{sem} + \lambda_{geo}\mathcal{L}_{scal}^{geo} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_d\mathcal{L}_d$$

- $\mathcal{L}_{scal}$: Scene-class affinity loss from MonoScene (semantic + geometric)
- $\mathcal{L}_{ce}$: Class-frequency-weighted cross-entropy loss
- $\mathcal{L}_d$: Binary cross-entropy supervision on depth distribution (using LiDAR projection)

## Key Experimental Results

### SemanticKITTI Test Set

| Method | Input | IoU(%) | mIoU(%) |
|--------|-------|--------|---------|
| MonoScene | S | 34.16 | 11.08 |
| CGFormer | S | 44.41 | 16.63 |
| VoxFormer-T | T(5 frames) | 43.21 | 13.41 |
| HTCL | T(3 frames) | 44.23 | 17.09 |
| **FlowScene** | **T(2 frames)** | **45.20** | **17.70** |

Using only 2 historical frames, FlowScene surpasses HTCL (3 frames) by +0.61% mIoU and +0.97% IoU, and outperforms the best single-frame method CGFormer by +1.07% mIoU.

### SSCBench-KITTI-360 Test Set

| Method | IoU(%) | mIoU(%) |
|--------|--------|---------|
| CGFormer | 48.07 | 20.05 |
| Symphonies | 44.12 | 18.58 |
| **FlowScene** | **46.95** | **20.81** |

Advantages are particularly pronounced on dynamic objects (car, truck, person, etc.), with dynamic object mIoU reaching 14.87%, far exceeding CGFormer's 11.37%.

### Performance at Different Distance Ranges (SemanticKITTI Validation Set)

| Method | 12.8m | 25.6m | 51.2m |
|--------|-------|-------|-------|
| VLScene | 26.51 | 24.37 | 17.83 |
| **FlowScene** | **27.63** | **24.65** | **18.13** |

FlowScene consistently outperforms existing methods across all distance ranges.

### Efficiency Comparison

| Method | mIoU(%) | Inference Time (s) | Parameters (M) |
|--------|---------|--------------------|----------------|
| HTCL | 17.13 | 0.297 | 181.4 |
| BRGScene | 15.43 | 0.285 | 161.4 |
| **FlowScene** | **18.13** | 0.301 | **52.4** |

FlowScene has only 52.4M parameters, far fewer than HTCL (181.4M), with comparable inference speed.

### Ablation Study

| Configuration | IoU(%) | mIoU(%) |
|---------------|--------|---------|
| Baseline (naive stacking) | 43.98 | 15.89 |
| + Flow Warping | 44.13 | 16.21 |
| + Occlusion Detection | 44.38 | 16.43 |
| + Temporal Aggregation | 44.63 | 17.23 |
| + Occlusion Cross-Attention | 44.42 | 17.08 |
| Full (+ OGVR) | **45.01** | **18.13** |

Each module contributes clearly, with a total gain of +2.24% mIoU and +1.03% IoU.

**Temporal Frame Count Ablation**: 2 frames is the optimal trade-off. As the frame count increases, optical flow prediction quality degrades (due to large inter-frame intervals), and performance actually decreases with 4–5 frames.

**Optical Flow Network Ablation**: GMFlow > FlowFormer > RAFT > PWC-Net, with GMFlow achieving the best performance at the smallest parameter count (4.7M).

**Backbone Ablation**: RepViT-M2.3 > EfficientNetB7 > ResNet50, offering the best performance with the fewest parameters.

## Highlights & Insights

### Novelty

1. **Optical flow-guided temporal SSC**: The first work to systematically introduce optical flow into SSC, moving beyond naive frame stacking or pose-based alignment.
2. **Occlusion-aware dual-space modeling**: Occlusion information is exploited in both 2D (FGTA) and 3D (OGVR) spaces, forming a complete occlusion-handling pipeline.
3. **Efficient design**: Achieves SOTA with only 2 historical frames and 52.4M parameters; OGVR introduces zero additional parameters.

### Limitations & Future Work

1. Relies on a pre-trained optical flow model (GMFlow); estimation errors propagate downstream.
2. Evaluation is limited to KITTI-series datasets; experiments on multi-camera settings such as nuScenes are absent.
3. The number of usable historical frames is constrained by optical flow quality — flow estimates for temporally distant frames are unreliable.

## Personal Reflections

- The intuition behind introducing optical flow into SSC is clear — optical flow encodes precise pixel-level motion correspondences, preserving temporal consistency better than naive stacking or pose estimation.
- The use of occlusion masks is elegantly designed: in 2D space, they filter out unreliable features; in 3D space, they enable adaptive fusion — the two modules are complementary.
- The OGVR weighting formula (Eq. 7) is simple yet effective, with the core strategy being a divide-and-conquer approach: use historical information in non-occluded regions, and current-frame information in occluded regions.
- A noteworthy trend: compared to methods requiring 5 input frames, FlowScene achieves superior performance with only 2 frames, suggesting that "quality > quantity" — high-quality temporal alignment matters more than stacking more frames.
- Future directions include combining optical flow guidance with BEV perception (e.g., BEVFormer) or occupancy networks (e.g., Occ3D).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion](../../ECCV2024/autonomous_driving/hierarchical_temporal_context_learning_for_camera-based_semantic_scene_completio.md)
- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](../../AAAI2026/autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[AAAI 2026\] Unleashing Semantic and Geometric Priors for 3D Scene Completion](../../AAAI2026/autonomous_driving/unleashing_semantic_and_geometric_priors_for_3d_scene_completion.md)
- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)

</div>

<!-- RELATED:END -->
