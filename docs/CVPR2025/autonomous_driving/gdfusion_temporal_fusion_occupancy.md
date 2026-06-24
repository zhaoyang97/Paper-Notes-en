---
title: >-
  [Paper Note] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction
description: >-
  [CVPR 2025][Autonomous Driving][3D Semantic Occupancy Prediction] GDFusion is proposed, which reinterprets RNN as gradient descent on the feature space to uniformly fuse four types of heterogeneous temporal information (voxel-level, scene-level, motion, geometry) in VisionOcc, achieving a 1.4%-4.8% mIoU improvement on Occ3D while reducing GPU memory by 27%-72%.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Semantic Occupancy Prediction"
  - "Temporal Fusion"
  - "Gradient Descent RNN"
  - "Motion Compensation"
  - "Scene Adaptation"
date: 2026-05-08
content_hash: 5d0d1c4e5eb9c0de
---

# GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction

**Conference**: CVPR 2025  
**arXiv**: [2504.12959](https://arxiv.org/abs/2504.12959)  
**Code**: [https://cdb342.github.io/GDFusion](https://cdb342.github.io/GDFusion)  
**Area**: Autonomous Driving / Occupancy Prediction  
**Keywords**: 3D Semantic Occupancy Prediction, Temporal Fusion, Gradient Descent RNN, Motion Compensation, Scene Adaptation

## TL;DR

GDFusion is proposed, which reinterprets RNN as gradient descent on the feature space to uniformly fuse four types of heterogeneous temporal information (voxel-level, scene-level, motion, geometry) in VisionOcc, achieving a 1.4%-4.8% mIoU improvement on Occ3D while reducing GPU memory by 27%-72%.

## Background & Motivation

**Background**: Temporal information is increasingly important in vision-based 3D语义占用预测 (VisionOcc), but existing methods only focus on voxel-level feature fusion.

**Limitations of Prior Work**: Three temporal cues are neglected: scene-level consistency priors (unchanged weather/lighting in the short term), historical motion information to correct ego-motion alignment errors of the current frame, and historical geometric information to complement depth estimation of the current frame.

**Key Challenge**: The four types of temporal information have completely different representations (3D feature maps, network parameters, 3D flow fields, and probabilistic point clouds), making them difficult to fuse in a unified manner.

**Core Idea**: Reinterpreting the vanilla RNN update $h^t = Ah^{t-1} + Bx^t$ as a gradient descent step that minimizes $||Ah^{t-1} - Bx^t||^2$, thereby designing specific loss functions to fuse heterogeneous representations in a unified framework.

## Method

### Key Designs

1. **Scene-level Temporal Fusion**: Encodes scene information into trainable network parameters $\mathbf{S}^t$ (including scale/shift of LayerNorm and linear layers), and updates the parameters frame-by-frame during inference via self-supervised reconstruction loss to adapt to the current scene.

2. **Motion Temporal Fusion**: Learns displacement offsets $\mathbf{M}^t$ to compensate for dynamic object motion and ego-motion estimation errors, where historical motion gradients correct the current frame predictions.

3. **Geometric Temporal Fusion**: Fuses historical depth probability distributions (geometric priors from 2D-to-3D lifting) with the current frame to enhance depth estimation quality.

### Loss & Training

Each temporal fusion process is unified into a gradient descent formulation: computing the discrepancy loss between the current frame representation and the historical state, and then adding the gradient as a temporal residual to the current representation. The entire process is highly differentiable and only maintains a historical state equivalent to the size of a single frame.

## Key Experimental Results

### Main Results

| Baseline | Original mIoU | +GDFusion mIoU | Memory Savings |
|------|----------|---------------|---------|
| FB-Occ | 39.2 | 40.6 (+1.4) | -27% |
| COTR | 42.4 | 44.8 (+2.4) | -72% |
| SurroundOcc | 20.6 | 34.6 (+14.0) | - |

### Key Findings

- The four temporal cues make complementary contributions.
- The gradient descent perspective enables the fusion of heterogeneous representations.
- Memory efficiency is significantly superior to methods like SOLOFusion.
- Achieves a massive 14.0% mIoU improvement on SurroundOcc (from 20.6% to 34.6%), demonstrating the significant headroom temporal fusion provides for weak baselines.
- Also achieves a 6.3% mIoU improvement on OpenOccupancy with almost negligible inference overhead.

## Highlights & Insights

- The reinterpretation of RNN as gradient descent is highly elegant.
- Plug-and-play, applicable to various VisionOcc baselines.
- The "test-time adaptation" concept of scene-level fusion is highly novel - maintaining only a single-frame-sized historical state, resulting in a memory efficiency far superior to SOLOFusion, which requires storing multi-frame features.
- By designing specific loss functions to quantify the discrepancies between different temporal representations and the current frame, it achieves an elegant and unified fusion.

## Limitations & Future Work

- The self-supervised task design for scene-level fusion is relatively simple.
- There is no explicit supervision for motion information, relying on indirect learning.
- The representational capacity of scene-adaptive parameters (LayerNorm scale/shift + linear layers) is limited, which may not capture complex scene changes.
- Geometric temporal fusion depends on the quality of historical depth estimation; errors may accumulate when depth estimates are incorrect across consecutive multiple frames.
- Currently only validated on the nuScenes dataset; its performance on larger and more diverse datasets (such as Waymo) remains to be confirmed.
- The theoretical elegance of the gradient descent perspective might be affected by the choice of learning rates in practice.
- In scenarios with dramatic weather changes (e.g., heavy rain, dense fog), the adaptation speed of scene-level fusion may not be sufficiently fast.
- The relationship with world-model-based methods (e.g., OccWorld) is worth further exploration.

## Rating

- Novelty: 9/10 — Theoretical contribution of unifying RNN via gradient descent
- Technical Depth: 9/10 — Theoretical derivation + four types of fusion
- Experimental Thoroughness: 8/10 — Multiple baselines on three benchmarks
- Writing Quality: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [\[CVPR 2025\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)

</div>

<!-- RELATED:END -->
