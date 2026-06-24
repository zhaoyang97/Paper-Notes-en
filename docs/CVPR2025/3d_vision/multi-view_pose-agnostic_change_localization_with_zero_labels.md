---
title: >-
  [Paper Note] Multi-View Pose-Agnostic Change Localization with Zero Labels
description: >-
  [3D Vision] This paper proposes the first label-free, pose-agnostic multi-view change detection method. By embedding a change channel into 3D Gaussian Splatting (change-aware 3DGS), it fuses multi-view feature-aware and structure-aware change masks to achieve SOTA performance gains of 1.7× mIoU and 1.5× F1 in complex multi-object scenes, and enables change mask generation for unseen views.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: e6a1b05c78623cb4
---

# Multi-View Pose-Agnostic Change Localization with Zero Labels

## TL;DR

This paper proposes the first label-free, pose-agnostic multi-view change detection method. By embedding a change channel into 3D Gaussian Splatting (change-aware 3DGS), it fuses multi-view feature-aware and structure-aware change masks to achieve SOTA performance gains of 1.7× mIoU and 1.5× F1 in complex multi-object scenes, and enables change mask generation for unseen views.

## Background & Motivation

Autonomous agents need to detect environmental changes to update maps and replan tasks. However, existing change detection methods face three major challenges:

1. **Pose Constraints**: Traditional methods require precise alignment of images before and after changes (e.g., fixed cameras, planar scenes), which is impractical for scenarios where robots collect data along random trajectories.
2. **Label Dependency**: Supervised learning methods require expensive change annotation datasets, and their performance drops sharply under distribution shifts.
3. **Vulnerability of Single-View Methods**: Existing pose-agnostic methods (OmniPoseAD, SplatPose) render reference views using NeRF/3DGS for frame-by-frame comparison, but are highly susceptible to view-dependent false positives (such as reflections, shadows, and occlusions).

Key Insight: Single-view methods generate a significant number of false positives due to perspective changes. Leveraging multi-view information and fusing change cues in the 3D scene can effectively suppress these view-dependent noises.

## Method

### Overall Architecture

The overall pipeline consists of five steps:
1. Construct the 3DGS reference scene representation using reference scene images.
2. Register the query scene images into the same coordinate system as the reference scene.
3. Render reference images aligned with the query views to generate feature-aware + structure-aware candidate change masks.
4. Embed a change channel into the query scene 3DGS to learn multi-view consistent change representations.
5. Render the final multi-view change masks from the change-aware 3DGS.

### Key Designs

#### 1. Feature-Aware + Structure-Aware Candidate Change Masks

**Feature-Aware Change Mask**:
- Pre-trained DINOv2 is used to extract dense features from both the rendered and query images.
- Feature differences are calculated as: $D^k = \sum_{j=1}^d |f_{ren}^{k,j} - f_{inf}^{k,j}|$
- The difference map is normalized and thresholded at 0.5 to filter out minor false changes.

**Structure-Aware Change Mask**:
- SSIM (Structural Similarity Index) is used to compare the rendered and query images.
- Low SSIM areas are binarized as: $M_S^k = \mathbf{1}(\text{SSIM}(I_{ren}^k, I_{inf}^k) \leq 0.5)$

**Joint Mask**: Achieves "dual-voting" filtering via element-wise multiplication: $M_{F,S}^k = M_F^k \cdot M_S^k$

Key Insight: Feature-level (DINOv2) and pixel-level (SSIM) change detection are complementary—the former is sensitive to semantics but robust to illumination, whereas the latter is highly sensitive to structure and illumination changes.

#### 2. Change-Aware 3D Gaussian Splatting

Core Contribution—Adding two change parameters per Gaussian in 3DGS:
- **Change magnitude $\tilde{c}$**: Represents the degree of change captured by the Gaussian in the scene.
- **Change opacity $\tilde{\alpha}$**: Controls the contribution of the Gaussian to the change mask rendering.

Key Design Decision: The change magnitude utilizes **zero-order spherical harmonics coefficients** (SH degree=0), instead of the standard third-order coefficients in standard 3DGS.

Reasoning: Scene changes are view-independent, whereas most view-dependent changes (reflections, shadows, minor alignment errors) represent false positives. Utilizing low-order SH allows the model to:
- Learn true change regions leveraging multi-view consistency.
- Avoid overfitting to view-dependent false positives.

The reference scene 3DGS is used to initialize the query scene's change-aware 3DGS, which is then jointly optimized with the query images and candidate change masks.

#### 3. Data Augmentation Strategy

Leveraging the reverse operation of the change-aware 3DGS:
- Render reference-view images from the query scene 3DGS.
- Compare the rendered images with the original reference images to generate additional change masks.
- Merge the old and new masks to re-optimize the change channel.

This is equivalent to a "bidirectional detection" scheme—discovering changes from both ends, which enhances training data for learning the change-aware channel.

### Loss & Training

The standard 3DGS optimization loss ($L1 + \text{D-SSIM}$ loss) is extended to the change channel:
- RGB reconstruction loss is used to update scene appearance parameters.
- $L1 + \text{D-SSIM}$ loss of the change channel is utilized to learn $\tilde{c}$ and $\tilde{\alpha}$.

## Key Experimental Results

### Main Results

**MAD-Real Dataset (Single Object)**:

| Method | mIoU↑ | F1↑ | AUROC↑ |
|------|-------|-----|--------|
| OmniPoseAD | 0.064 | 0.115 | 0.937 |
| SplatPose | 0.077 | 0.123 | 0.898 |
| Feature Diff. | 0.052 | 0.089 | 0.967 |
| **Ours** | **0.132** | **0.210** | 0.953 |

**PASLCD Dataset (Multi-Object Scenes, Average)**:

| Method | mIoU↑ | F1↑ |
|------|-------|-----|
| OmniPoseAD | 0.168 | 0.262 |
| SplatPose | 0.173 | 0.281 |
| CYWS-2D | 0.273 | 0.398 |
| Feature Diff. | 0.264 | 0.386 |
| **Ours** | **0.461** | **0.612** |

### Ablation Study

**Effect of Spherical Harmonics Degree**:

| SH Degree | mIoU↑ | F1↑ | FP↓ |
|-----------|-------|-----|-----|
| 0 | Optimal | Optimal | Lowest |
| 3 | Lower | Lower | ~3× |

Using SH degree=0 reduces false positives by approximately 70%.

**Impact of Query View Counts**:
- Using only 5 query images achieves a 1.8× mIoU improvement over Feature Diff.
- Performance scales up with an increasing number of views (5→10→15→25).

### Key Findings

1. **Multi-View Methods Significantly Outperform Single-View**: Achieving 1.7× mIoU and 1.5× F1 improvements on PASLCD.
2. **Capable of Generating Change Masks for Unseen Views**: This represents a novel capability that existing frame-by-frame methods cannot achieve.
3. **SH degree=0 is Crucial**: Changes should be modeled as view-independent; low-order SH effectively suppresses false positives.
4. **CYWS-2D + Change-3DGS Improves by 44%**: Demonstrating that change-aware 3DGS can serve as a multi-view plugin/extension for any change masking method.
5. **1.7× Improvement on ChangeSim**: Demonstrating equal effectiveness in industrial simulation environments.

## Highlights & Insights

1. **Elevating Change Detection from 2D "image-pair" to 3D "scene"**: By embedding change channels into 3DGS, change detection evolves from frame-by-frame comparison to scene-level understanding.
2. **Ingenious Design of SH degree=0**: Grounded on the assumption that "real changes are view-independent, while false positives are view-dependent", denoising is achieved by restricting the SH degree—elegantly capitalizing on the expressiveness of 3DGS.
3. **Dual Change Detection of Feature + Structure**: The complementarity between DINOv2 features and SSIM is clearly validated, where the former is robust to lighting variations and the latter captures fine structural differences.
4. **Curation of a Real-World Dataset, PASLCD**: Comprising 10 real-world scenes (indoor/outdoor), 500 annotated masks, and 91 change instances, significantly accelerating research in this field.

## Limitations & Future Work

1. **Difficulty in Detecting Color-Level Surface Changes**: For changes that do not alter the 3D structure (e.g., liquid spills, color replacements), DINOv2 features can be insensitive.
2. **Small Objects are Easily Missed in Large-scale Scenes**: Detection performance deteriorates when encountering minute changes in spacious environments like the Playground or Lunch Room.
3. **Overestimation of Change Masks**: Due to patch-to-pixel interpolation of feature maps, the boundaries of the estimated change masks are typically larger than the ground truth.
4. **Failure under COLMAP Registration Failure**: The pipeline breaks if query scenes are extremely dark or have insufficient overlap with reference scenes for registration.
5. **Computational Cost**: Training two 3DGS models (reference + query) alongside DINOv2 feature extraction demands considerable compute resources.

## Related Work & Insights

- **3D Gaussian Splatting** [Kerbl et al., 2023]: This work extends 3DGS from a rendering framework to a change detection pipeline.
- **OmniPoseAD / SplatPose**: Prior endeavors leveraged 3D representations for pose-agnostic anomaly detection, but were confined to single-view comparisons.
- **DINOv2**: Acting as a robust backbone for feature-aware change masking, its zero-shot generalization capabilities across diverse scenes form the bedrock of this method's success.
- **SSIM**: An established image quality metric is innovatively adapted for change detection, serving as a powerful complement to deep learning features.
- Insight: 3D representations can serve not only rendering and reconstruction purposes but also act as a vehicle for multi-view information fusion.

## Rating

⭐⭐⭐⭐ (8.5/10)

- Novelty: ⭐⭐⭐⭐⭐ — The first multi-view label-free pose-agnostic change detection framework; the change-aware 3DGS formulation is highly novel.
- Practicality: ⭐⭐⭐⭐ — Highly practical for robotic environment understanding and infrastructure monitoring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Benchmarked across three datasets against multiple baselines, with detailed ablation and view-count analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Structurally clear and well-written methodology, complemented by excellent figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)
- [\[CVPR 2025\] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)

</div>

<!-- RELATED:END -->
