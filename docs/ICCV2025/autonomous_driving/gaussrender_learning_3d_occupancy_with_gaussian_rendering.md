---
title: >-
  [Paper Note] GaussRender: Learning 3D Occupancy with Gaussian Rendering
description: >-
  [ICCV 2025][Autonomous Driving][3D Occupancy Prediction] This paper proposes GaussRender, a plug-and-play differentiable Gaussian rendering module that projects predicted and ground-truth 3D occupancy onto 2D views and enforces semantic and depth consistency constraints, thereby eliminating visual artifacts such as floating voxels. The approach achieves significant improvements in geometric fidelity across multiple benchmarks, with particularly pronounced gains on surface-sen…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D Occupancy Prediction"
  - "Gaussian Splatting"
  - "Differentiable Rendering"
  - "Projective Consistency"
date: 2026-05-08
content_hash: c91806b2e7148f97
---

# GaussRender: Learning 3D Occupancy with Gaussian Rendering

**Conference**: ICCV 2025
**arXiv**: [2502.05040](https://arxiv.org/abs/2502.05040)  
**Code**: [https://github.com/valeoai/GaussRender](https://github.com/valeoai/GaussRender)  
**Area**: Autonomous Driving
**Keywords**: 3D Occupancy Prediction, Gaussian Splatting, Differentiable Rendering, Projective Consistency, autonomous driving

## TL;DR

This paper proposes GaussRender, a plug-and-play differentiable Gaussian rendering module that projects predicted and ground-truth 3D occupancy onto 2D views and enforces semantic and depth consistency constraints, thereby eliminating visual artifacts such as floating voxels. The approach achieves significant improvements in geometric fidelity across multiple benchmarks, with particularly pronounced gains on surface-sensitive metrics such as RayIoU.

## Background & Motivation

3D occupancy prediction is a core task in autonomous driving perception, requiring the inference of three-dimensional geometric and semantic structures from multi-view camera images. Existing methods (e.g., SurroundOcc, TPVFormer) are typically trained with per-voxel losses such as cross-entropy, Dice, or Lovász, which treat all voxels equally and **disregard spatial consistency among neighboring voxels**.

This leads to a critical issue: while models may perform well on voxel-wise IoU metrics, their predictions exhibit numerous **visual artifacts**—floating voxels, discontinuous surfaces, and boundary misalignments. Although these artifacts have minimal impact on voxel segmentation losses (which assign equal weight to all voxels), they can severely affect downstream tasks such as free-space estimation and motion planning.

The central insight of this paper is that projecting 3D predictions onto 2D views makes such physically implausible spatial arrangements apparent. Consequently, **incorporating projective consistency into the training objective** encourages the model to learn coherent and physically plausible geometric structures. Compared to NeRF-based rendering approaches, this work leverages Gaussian Splatting for efficient rendering without requiring temporal supervision or LiDAR reprojection.

## Method

### Overall Architecture

The GaussRender pipeline proceeds as follows: (1) both predicted and ground-truth voxel grids are "Gaussianized"—each voxel is converted into a spherical Gaussian primitive; (2) virtual cameras are placed in the scene, including fixed bird's-eye-view (BEV) cameras and dynamically randomized cameras; (3) 3D Gaussians are projected onto 2D views via Gaussian Splatting to produce semantic and depth renderings; (4) the rendered outputs from predictions and ground truth are compared via an L1 loss. The entire module operates only during training and introduces no additional computation at inference.

The total loss is: $L = L_{3D} + \lambda L_{2D}$, where $L_{3D}$ is the standard per-voxel loss and $L_{2D}$ is the rendering consistency loss.

### Key Designs

1. **Voxel Gaussianization**:

    - Function: Converts each voxel into a simplified spherical Gaussian primitive.
    - Mechanism: The Gaussian parameters are highly simplified—the position $\mu$ is fixed at the voxel center, the scale $S = \text{Diag}(s)$ is fixed based on voxel size, the rotation $R = I$ (no orientation needed for spherical primitives), the semantic "color" $c$ is taken from the final predicted logits, and only the opacity $o$ is learned from voxel features (or derived from the empty-class logit).
    - Design Motivation: The extreme simplification reduces the learning burden. Unlike GaussianOcc and GaussTR, which couple Gaussian representations with the model architecture, GaussRender operates entirely at the prediction level and imposes no constraint on the underlying 3D representation.

2. **Camera Placement Strategy**:

    - Function: Flexibly places virtual rendering cameras within the 3D scene.
    - Mechanism: Two types of virtual cameras are employed: (a) fixed orthographic BEV cameras providing a global top-down constraint, and (b) dynamic "elevated + translated" cameras that are raised along the z-axis and randomly shifted within a small range in the xy-plane to expand coverage of occluded regions.
    - Design Motivation: Conventional methods are constrained by sensor camera viewpoints or LiDAR reprojection and cannot supervise occluded regions. GaussRender renders both predictions and ground truth simultaneously, relying neither on RGB images nor on LiDAR pseudo-labels, and can render from arbitrary viewpoints. The elevated cameras can "see" regions that are horizontally occluded by ground-level objects.

3. **Gaussian Rendering & $L_{2D}$ Loss**:

    - Function: Projects 3D Gaussians onto 2D and computes semantic and depth rendering losses.
    - Mechanism: The 3D covariance matrix is projected onto the image plane via camera parameters: $\Sigma_{2D} = J \cdot W \cdot \Sigma_{3D} \cdot W^T \cdot J^T$. For each pixel $p$, the rendered semantic value $C_p = \sum_i T_i \alpha_i \mathbf{c}_i$ and depth $D_p = \sum_i T_i \alpha_i \mathbf{d}_i$ are computed, where $T_i = \prod_{j<i}(1-\alpha_j)$ is the accumulated transmittance. The loss for each virtual camera is: $L_{2D}^* = L_{depth}^* + L_{sem}^*$, and the total rendering loss is $L_{2D} = L_{2D}^{bev} + L_{2D}^{cam}$.
    - Design Motivation: The semantic rendering loss reinforces local semantic consistency, while the depth rendering loss penalizes artifacts that violate occlusion relationships. The depth loss is normalized by $d_{range}^*$ to ensure scale consistency across scenes.

### Loss & Training

- The semantic loss compares predicted and ground-truth semantic renderings via L1 distance: $L_{sem}^* = \|I_{sem}^* - \tilde{I}_{sem}^*\|_1$
- The depth loss applies L1 distance with normalization: $L_{depth}^* = \frac{1}{d_{range}^*}\|I_{depth}^* - \tilde{I}_{depth}^*\|_1$
- During ground-truth rendering, occupied voxels are assigned opacity 1 and empty voxels opacity 0.
- The module is computed in parallel with the standard 3D loss during training, requiring no modification to the model architecture.
- The rendering module is entirely removed at inference, introducing zero additional overhead.

## Key Experimental Results

### Main Results

| Dataset / Model | Metric | Baseline | +GaussRender | Gain |
|---|---|---|---|---|
| SurroundOcc-nuSc / TPVFormer | IoU / mIoU | 30.86 / 17.10 | 32.05 / 20.85 | +1.19 / +3.75 |
| SurroundOcc-nuSc / SurroundOcc | IoU / mIoU | 31.49 / 20.30 | 32.61 / 20.82 | +1.12 / +0.52 |
| Occ3D-nuSc / TPVFormer | mIoU | 27.83 | 30.48 | +2.65 |
| Occ3D-nuSc / SurroundOcc | mIoU | 29.21 | 30.38 | +1.17 |
| SSCBench-KITTI360 / Symphonies | IoU / mIoU | 43.40 / 17.82 | 44.08 / 18.11 | +0.68 / +0.29 |
| Occ3D-nuSc / TPVFormer | RayIoU | 37.2 | 38.3 | +1.1 |
| Occ3D-nuSc / SurroundOcc | RayIoU | 35.5 | 37.5 | +2.0 |

### Ablation Study

| Configuration | IoU | mIoU | Notes |
|------|-----|------|------|
| Cam semantic | 26.3 | 14.3 | Camera semantic rendering loss only |
| + Cam depth | 26.8 | 15.1 | Adding camera depth loss |
| + BEV semantic | 27.2 | 15.6 | Adding BEV semantic rendering |
| + BEV depth (full) | 27.5 | 16.4 | All loss components |
| Camera strategy: Sensor (2D+3D) | - | 25.9 | Sensor positions |
| Camera strategy: Elevated+Around (2D+3D) | - | 26.3 | Best strategy |
| Camera strategy: Fully Random (2D+3D) | - | 25.4 | Random viewpoints underperform |

### Key Findings

- GaussRender yields consistent improvements across three datasets and three distinct architectures, validating its generalizability.
- Gains are more pronounced on the surface-sensitive RayIoU metric (SurroundOcc +2.0), demonstrating that rendering constraints effectively eliminate floating artifacts.
- Under a 2D-supervision-only setting (without 3D GT), GaussRender achieves 25.3 mIoU, surpassing all existing rendering-based methods including RenderOcc (23.9), which uses temporal frames.
- The elevated + surrounding camera strategy is optimal in 2D+3D training, while sensor positions perform best under pure 2D training—indicating that virtual camera placement should be adapted to the supervision signal.

## Highlights & Insights

- **Plug-and-play design**: No model architecture modifications are required; the rendering loss is added solely during training with zero inference overhead. This enables an older architecture (TPVFormer) to outperform newer methods (GaussianFormerV2), suggesting that well-designed training strategies can matter more than architectural complexity.
- **LiDAR-free supervision**: Unlike RenderOcc and GSRender, GaussRender renders both predictions and ground truth simultaneously, requiring no LiDAR reprojection or pseudo-labels.
- **Arbitrary-viewpoint rendering**: Overcomes the restriction of existing methods to sensor viewpoints or temporal frames.

## Limitations & Future Work

- The virtual camera placement strategy is currently rule-based (elevation + random translation) and does not adapt to scene complexity.
- The effects of rendering resolution and Gaussian parameter choices remain insufficiently explored.
- Temporal information is not integrated—combining dynamic view synthesis with temporal sequences could further improve occlusion reasoning.
- The approach has not been extended to open-vocabulary scene understanding.

## Related Work & Insights

- **RenderOcc / GSRender**: Apply NeRF or Gaussian rendering for 2D supervision but rely on LiDAR reprojection and temporal frames; GaussRender eliminates both dependencies.
- **GaussianOcc / GaussTR**: Embed Gaussian representations into the model architecture; GaussRender operates at the prediction level, offering greater flexibility.
- **SparseOcc**: Proposes the RayIoU metric; GaussRender achieves the largest gains on this metric, validating the value of projective constraints.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)
- [\[CVPR 2025\] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](../../CVPR2025/autonomous_driving/gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](../../CVPR2025/autonomous_driving/gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
