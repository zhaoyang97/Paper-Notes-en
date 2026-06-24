---
title: >-
  [Paper Note] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction
description: >-
  [AAAI 2026 Oral][Autonomous Driving][Collaborative Perception] This paper proposes the first vision-only semantic occupancy prediction framework that uses sparse 3D semantic Gaussian primitives as the communication medium for collaborative perception. Through ROI cropping, rigid transformation of Gaussians, and a neighborhood fusion module to suppress noise and redundancy, the method achieves +8.42 mIoU over the single-agent baseline and +3.28 mIoU over the baseline collabora…
tags:
  - "AAAI 2026 Oral"
  - "Autonomous Driving"
  - "Collaborative Perception"
  - "3D Gaussian Splatting"
  - "Semantic Occupancy"
  - "V2X Communication"
  - "Vision-Only"
date: 2026-05-08
content_hash: fa7c23b12aec7bfe
---

# Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction

**Conference**: AAAI 2026 Oral  
**arXiv**: [2508.10936](https://arxiv.org/abs/2508.10936)  
**Code**: [GitHub](https://github.com/ChengChen2020/VOGS-CP)  
**Area**: Autonomous Driving / Collaborative Perception
**Keywords**: Collaborative Perception, 3D Gaussian Splatting, Semantic Occupancy, V2X Communication, Vision-Only

## TL;DR

This paper proposes the first vision-only semantic occupancy prediction framework that uses sparse 3D semantic Gaussian primitives as the communication medium for collaborative perception. Through ROI cropping, rigid transformation of Gaussians, and a neighborhood fusion module to suppress noise and redundancy, the method achieves +8.42 mIoU over the single-agent baseline and +3.28 mIoU over the baseline collaborative method.

## Background & Motivation

- **Background**: Collaborative perception extends single-agent perception range via V2X communication. 3D semantic occupancy prediction provides finer-grained scene understanding than BEV or 3D object detection.
- **Limitations of Prior Work**: Existing collaborative occupancy methods (e.g., CoHFF) rely on tri-plane features requiring depth supervision, multi-stage training, and complex cross-agent alignment; dense voxel feature transmission incurs high communication costs.
- **Key Challenge**: Fine-grained 3D representations demand large data transmission, which conflicts with limited V2X communication bandwidth.
- **Goal**: Design a communication-efficient, end-to-end trainable collaborative semantic occupancy prediction framework.
- **Key Insight**: 3D Gaussians are sparse, rigidly transformable representations that simultaneously encode geometry and semantics.
- **Core Idea**: Replace voxel/plane features with 3D Gaussian primitives as the V2X communication medium, which naturally supports rigid alignment and sparse transmission.

## Method

### Overall Architecture

Single agent: Image-to-Gaussian module (randomly initialized Gaussians → multi-scale image feature-guided refinement) → Gaussian-to-voxel splatting → occupancy prediction. Collaborative: Gaussian packaging (rigid transformation + ROI cropping) → transmission → cross-agent Gaussian fusion → splatting.

### Key Designs

**Design 1: Gaussian Primitives as Communication Medium**
- **Function**: Transmit 3D Gaussians (mean + scale + rotation + opacity + semantics) as V2X messages.
- **Mechanism**: Gaussians are closed under rigid transformation (mean rotation + translation, rotation quaternion multiplication, scale/opacity/semantics unchanged); ROI cropping transmits only Gaussians within the ego agent's region of interest.
- **Design Motivation**: Sparser than voxel features, better preserves 3D structure than plane features, and alignment requires only a simple rigid transformation.

**Design 2: Cross-Agent Gaussian Fusion Module**
- **Function**: Fuse Gaussian primitives from multiple agents while suppressing noise and redundancy.
- **Mechanism**: Neighborhood-conditioned proposal → cross-neighborhood pooling → attribute blending with ego Gaussians. Unlike GaussianFormer's single-agent refinement, this module specifically addresses cross-agent inconsistencies.
- **Design Motivation**: Gaussians from different agents may be redundant or conflicting due to occlusion-induced noise, necessitating learned fusion.

**Design 3: End-to-End Single-Stage Training**
- **Function**: The entire pipeline is trained end-to-end without separate depth estimation or multi-stage schedules.
- **Mechanism**: GaussianFormer-based Image-to-Gaussian + Gaussian-to-voxel, jointly optimized with the fusion module.
- **Design Motivation**: CoHFF requires two-stage training and an independent depth network, increasing overall complexity.

### Loss & Training

Standard semantic occupancy loss (CE + Lovász loss), single-stage end-to-end training.

## Key Experimental Results

### Main Results

| Method | IoU↑ | mIoU↑ |
|--------|------|-------|
| Single-Agent GSFormer | 67.76 | 29.20 |
| CoHFF (Collaborative) | 50.46 | 34.16 |
| Zero-Shot Stacking | 67.88 | 30.54 |
| Naive Fusion | 70.10 | 36.02 |
| **Learned Fusion** | **72.87** | **37.44** |

### Ablation Study

| Communication Volume | mIoU |
|----------------------|------|
| 100% Gaussians | 37.44 |
| 34.6% Gaussians | 36.06 (+1.9 vs. single agent) |

### Key Findings

1. Even zero-shot stacking (without joint training) improves collaborative perception, validating the advantage of explicit representations.
2. Using only 34.6% of the communication budget still surpasses the single-agent baseline by +1.9 mIoU, demonstrating high communication efficiency.
3. Learned fusion improves over naive fusion by +1.4 mIoU, confirming the effectiveness of the neighborhood fusion module.

## Highlights & Insights

1. This work is the first to introduce 3D Gaussian splatting into collaborative perception, establishing a pioneering research direction.
2. The closure of Gaussians under rigid transformation makes cross-agent alignment trivial.
3. Sparse, explicit, and interpretable representations are better suited to communication-constrained scenarios than implicit features.

## Limitations & Future Work

1. Experiments are conducted only on simulated data; validation on real-world V2X datasets is absent.
2. Gaussian initialization remains random; better initialization strategies warrant exploration.
3. Communication latency and asynchrony are not addressed.

## Related Work & Insights

- GaussianFormer applies Gaussians to single-agent occupancy prediction; this work extends the approach to collaborative settings.
- CoHFF is the first collaborative occupancy framework but relies on tri-plane features and depth supervision; this work substantially simplifies the pipeline.
- Insight: Choosing an appropriate representation can simultaneously address communication efficiency and alignment difficulty.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★★ |
| Practicality | ★★★★☆ |
| Experimental Thoroughness | ★★★☆☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](../../ECCV2024/autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](../../ICCV2025/autonomous_driving/semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[CVPR 2026\] ParkGaussian: Surround-view 3D Gaussian Splatting for Autonomous Parking](../../CVPR2026/autonomous_driving/parkgaussian_surround-view_3d_gaussian_splatting_for_autonomous_parking.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](../../CVPR2025/autonomous_driving/occmamba_semantic_occupancy_prediction_with_state_space_models.md)

</div>

<!-- RELATED:END -->
