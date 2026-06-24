---
title: >-
  [Paper Note] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
description: >-
  [3D Vision] FLARE proposes a cascade learning paradigm that uses camera poses as a bridge to decompose 3D reconstruction into four progressive stages: pose estimation → local geometry → global geometry → Gaussian appearance. It achieves high-quality camera pose estimation, geometric reconstruction, and novel view synthesis from 2-8 uncalibrated sparse images within 0.5 seconds.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: d02c71add88b9eef
---

# FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views

## TL;DR

FLARE proposes a cascade learning paradigm that uses camera poses as a bridge to decompose 3D reconstruction into four progressive stages: pose estimation → local geometry → global geometry → Gaussian appearance. It achieves high-quality camera pose estimation, geometric reconstruction, and novel view synthesis from 2-8 uncalibrated sparse images within 0.5 seconds.

## Background & Motivation

- **Core Problem**: Reconstructing 3D scenes from multi-view images is a fundamental problem in computer vision. Traditional SfM + MVS pipelines degrade severely under sparse views.
- **Limitations of Prior Work**:
    - **Optimization-based methods** (BARF, NeRF--): Require good initialization and exhibit poor generalization capabilities.
    - **DUSt3R/MASt3R**: Only handle pairwise matching for two views followed by post-processed global alignment, which is slow and yields suboptimal results.
    - **PF-LRM**: Feed-forward, but the tri-plane representation limits performance in complex scenes.
    - **NoPoSplat/Splatt3R**: Rely on the imperfect geometric estimation of DUSt3R.
- **Key Insights**:
    - Camera pose serves as a "bridge" connecting 2D images and 3D structures—even imperfect poses can provide effective geometric priors.
    - Directly optimizing all parameters jointly easily traps the optimization in local minima; thus, a **progressive decomposition** is preferable.
    - Learning local geometry first in the camera coordinate system and then projecting it into the global coordinate system converges much more easily than directly predicting global geometry.

## Method

### Overall Architecture

Input uncalibrated sparse views → Four-stage cascade:
1. **Neural Pose Predictor**: Estimates coarse camera poses.
2. **Camera-centric Geometry**: Predicts local point clouds in each camera's coordinate system.
3. **Global Geometry Projection**: Projects local geometry into the global coordinate system.
4. **3D Gaussian Head**: Predicts Gaussian parameters on the global point cloud to achieve photorealistic rendering.

### Key Designs

#### 1. Neural Pose Predictor

- Models pose estimation as a direct transformation problem from image space to camera space, completely discarding feature matching.
- Concatenates image patch tokens with a learnable camera latent $\mathcal{Q}_c$ into a 1D sequence.
- Directly regresses 7-dimensional poses (3D translation + normalized quaternion) using a small decoder-only transformer $F_p$.
- Key finding: Poses do not need to be highly accurate—approximating the ground-truth distribution is sufficient to provide effective priors for subsequent stages.

#### 2. Two-stage Geometry Learning

**Stage 1 — Camera-centric Geometry Estimation**:
- Learns geometry under the local coordinate system of each camera, which aligns with the physical image formation process (each view directly observes local geometry).
- Feeds image tokens and pose tokens into a transformer $F_l$, employing self-attention for multi-view association.
- Upsamples the output using a DPT decoder to obtain local point clouds $\mathcal{G}_l$ and confidence maps $\mathcal{C}_l$.
- Simultaneously introduces an additional pose token $\mathcal{Q}_f$ to refine poses (leveraging complementary supervision of multi-task learning).
- **Pose Augmentation during Training**: Adds Gaussian noise perturbation to the predicted poses, making the network robust to inaccurate poses during inference.

**Stage 2 — Global Geometry Projection**:
- Instead of using rigid geometric transformation (which is unreliable due to imprecise poses), a **neural scene projector** $F_g$ is learned.
- Conditioned on local point tokens $\mathcal{T}_l$ and refined poses $\mathcal{P}_f$, tokens are transformed into global coordinates via a transformer.
- A DPT decoder is then used to generate the global point cloud $\mathcal{G}_g$.

#### 3. 3D Gaussian Appearance Modeling

- Uses the global point cloud as Gaussian centers to predict opacity, rotation, scale, and SH coefficients.
- Introduces a pre-trained VGG network to extract image appearance features $\mathcal{V}$, which are fused with geometric features to regress Gaussian parameters via a CNN decoder.
- Normalization is applied to address the scale inconsistency between the estimated geometry and the ground-truth (GT) geometry.
- Utilizes a differentiable Gaussian rasterizer $R(\cdot)$ for end-to-end rendering.

### Loss & Training

$$\mathcal{L}_{total} = \lambda_{pose}\mathcal{L}_{pose} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{splat}\mathcal{L}_{splat}$$

- **Pose Loss**: Huber loss is utilized to supervise both coarse and refined poses.
- **Geometry Loss**: Confidence-weighted 3D regression loss in both local and global coordinates:
  $$\mathcal{L}_{geo} = \sum_i \sum_j \mathbf{C}_{i,j}^{camera}\ell_{regr}^{camera} - \alpha\log\mathbf{C}_{i,j}^{camera} + \mathbf{C}_{i,j}^{global}\ell_{regr}^{global} - \alpha\log\mathbf{C}_{i,j}^{global}$$
- **Rendering Loss**: L2 loss + VGG perceptual loss + monocular depth loss.

## Key Experimental Results

### Pose Estimation (Tab. 1 — RealEstate10K)

| Method | RRA@5°↑ | RTA@5°↑ | AUC@30°↑ |
|------|---------|---------|----------|
| DUSt3R (Optimization) | 0.83 | 0.37 | 54.9 |
| MASt3R (Optimization) | 0.87 | 0.45 | 61.1 |
| COLMAP | 0.63 | 0.07 | 16.0 |
| VGGSfM | - | - | 72.1 |
| **FLARE (Ours)** | **0.92** | **0.56** | **76.8** |

### Novel View Synthesis (RealEstate10K & ACID)

Compared with DUSt3R + 3DGS, NoPoSplat, and MASt3R on RealEstate10K, FLARE achieves a comprehensive lead in PSNR, SSIM, and LPIPS.

### Key Findings

1. The feed-forward pose estimation of FLARE outperforms optimization-based DUSt3R/MASt3R, with AUC@30° improving from 61.1 to 76.8.
2. Two-stage geometry learning (local → global) converges faster and exhibits fewer geometric distortions than directly predicting global geometry.
3. The pose augmentation strategy makes the model more robust to pose errors during inference.
4. The overall inference time is below 0.5 seconds, which is one to two orders of magnitude faster than optimization-based methods (which require global alignment like DUSt3R).
5. Highly robust generalization capability is demonstrated in real-world scenarios (such as casual captures of indoor bedrooms).

## Highlights & Insights

1. **Cascade Decomposition Philosophy**: Decomposes the challenging joint optimization problem into progressive stages, where the output of each stage conditions the learning of the next. This "simplicity-over-complexity" concept is both elegant and effective.
2. **Pose as a Bridge**: Even imprecise poses can significantly reduce the complexity of subsequent geometry learning—a crucial practical insight.
3. **Local-to-Global Geometry Strategy**: Learning local geometry in the camera coordinate system aligns with physical intuition (each viewpoint observes local structures), leaving the global projection to a learned module.
4. **Pose Noise Augmentation**: A simple yet highly efficient strategy for improving robustness.

## Limitations & Future Work

1. **GPU Memory Constraints**: The self-attention in transformers scales quadratically with the sequence length (number of patches over all views), which may lead to memory bottlenecks when scaling to more than 8 views.
2. **Generalization to Large-scale Scenes**: It is primarily validated on indoor and object-centric scenes; its performance on large-scale outdoor scenes remains to be evaluated.
3. **Textureless Regions**: Geometry estimation under sparse views in weak-texture regions remains challenging.
4. **Dynamic Scenes**: It currently assumes static scenes and cannot handle moving objects.

## Related Work & Insights

- **DUSt3R/MASt3R**: Point map representation + pairwise view matching → Ours extends this to multi-view feed-forward + global consistency.
- **PF-LRM**: 4-view feed-forward reconstruction → Ours utilizes point maps + cascade learning to achieve better generalization.
- **VGGSfM**: Differentiable bundle adjustment → Ours directly regresses poses using a transformer, which is faster.
- **Insights**: The "coarse-to-fine" cascade strategy in 3D reconstruction is a universal paradigm, where pose estimation and geometric reconstruction can mutually benefit each other.

## Rating

⭐⭐⭐⭐ — The method is meticulously designed, and the cascade learning paradigm is highly efficient and elegant. Completing the joint inference of pose, geometry, and appearance from uncalibrated sparse views within 0.5 seconds holds outstanding practical value. It comprehensively outperforms existing methods across multiple tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] Multi-View Pose-Agnostic Change Localization with Zero Labels](multi-view_pose-agnostic_change_localization_with_zero_labels.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)
- [\[CVPR 2025\] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)

</div>

<!-- RELATED:END -->
