---
title: >-
  [Paper Note] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos
description: >-
  [ECCV 2024][3D Vision] Proposes TRAM, a two-stage method that restores metric-scale camera motion via robustified SLAM and regresses camera-frame human motion using a video Transformer (VIMO), combining both to achieve accurate 3D global trajectory and motion reconstruction of humans in world coordinates.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 5772adb4a8287302
---

# TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos

**Conference**: ECCV 2024  
**arXiv**: [2403.17346](https://arxiv.org/abs/2403.17346)  
**Code**: [Project Page](https://yufu-wang.github.io/tram4d/)  
**Area**: 3D Vision

## TL;DR

Proposes TRAM, a two-stage method that restores metric-scale camera motion via robustified SLAM and regresses camera-frame human motion using a video Transformer (VIMO), combining both to achieve accurate 3D global trajectory and motion reconstruction of humans in world coordinates.

## Background & Motivation

### Background

**Background**: Recovering the full motion of humans in world space (global trajectory + local pose) from in-the-wild videos is crucial yet highly challenging.

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional SLAM assumes static environments, where moving humans degrade estimation accuracy; monocular SLAM only recovers scale-ambiguous trajectories.

### Key Challenge

**Key Challenge**: Methods like GLAMR and SLAHMR rely on motion priors learned from MoCap to infer trajectory scales, exhibiting poor generalization in complex scenes (e.g., climbing stairs, parkour).

### Proposed Solution

**Proposed Solution**: WHAM achieves promising results by directly regressing trajectories, but its reliance on MoCap data limits its prediction of novel trajectories.

### Supplementary Notes

**Supplementary Notes**: *Core Insight*: If the camera trajectory can be accurately localized (at metric scale) and human motion can be estimated within the camera-frame, then human motion in world coordinates = camera motion $\circ$ relative motion.

## Method

### Overall Architecture

1. **Masked DROID-SLAM**: Dual masking makes SLAM robust to dynamic humans.
2. **Metric Scale Estimation**: Aligns SLAM depth with metric depth predictions from ZoeDepth to recover the true scale.
3. **VIMO**: A video Transformer model based on HMR2.0 that regresses SMPL poses and positions in the camera coordinate frame.

### Key Designs

**Dual-Masked DROID-SLAM**:
- Mask input images: Reduce interference from dynamic objects on global features.
- Mask flow confidence in Dense Bundle Adjustment (DBA): Equivalent to removing dynamic region coordinates from the reprojection error calculation.
- Detect and segment dynamic objects using YOLOv7+SAM.

**Metric Scale Estimation**:
- Independently solve $\alpha \cdot d_{\text{SLAM}} \approx D_{\text{ZoeDepth}}$ for each frame using the German-McClure robust loss.
- Compute the median over the entire sequence to eliminate the influence of outlier frames.
- Exclude distant areas (such as the sky and other regions with inaccurate depth estimation).

**VIMO Video Transformer**:
- Freeze the ViT-H backbone of HMR2.0 and introduce two new temporal Transformers.
- First temporal Transformer: Performs cross-time attention on each spatial-position patch token of ViT (factorized spatiotemporal model).
- Second temporal Transformer: Directly performs encoding-decoding on SMPL pose sequences (learning motion priors in the pose space instead of latent space).
- Fully "transformerized" design, trained end-to-end from video.

### Loss & Training

$$\mathcal{L} = \lambda_{2D}\mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D} + \lambda_{SMPL}\mathcal{L}_{SMPL} + \lambda_{V}\mathcal{L}_{V}$$

Applies constraints to 2D joint reprojection, 3D joints, SMPL parameters, and vertices, respectively.

## Key Experimental Results

### Main Results

Evaluation of global human trajectory on the EMDB 2 dataset:

| Method | PA-MPJPE↓ | WA-MPJPE100↓ | W-MPJPE100↓ | RTE(%)↓ | ERVE↓ |
|------|-----------|--------------|-------------|---------|-------|
| GLAMR | 56.0 | 280.8 | 726.6 | 11.4 | 18.0 |
| SLAHMR | 61.5 | 326.9 | 776.1 | 10.2 | 19.7 |
| WHAM | 38.2 | 133.3 | 343.9 | 4.6 | 14.7 |
| **TRAM** | **38.1** | **76.4** | **222.4** | **1.4** | **10.3** |

### Ablation Study

Camera trajectory evaluation (EMDB 2, ATE in m):

| Method | Short(5) | Medium(10) | Long(10) | Average |
|------|----------|------------|----------|---------|
| DROID-SLAM | 0.40 | 2.55 | 3.31 | 2.42 |
| DROID+Mask Image | 0.36 | 0.63 | 2.74 | 1.42 |
| DROID+Mask DBA | 0.45 | 0.42 | 1.63 | 0.91 |
| **Masked DROID** | **0.32** | **0.20** | **0.44** | **0.32** |

### Key Findings

- The global trajectory RTE error is reduced by approximately 70% (4.6% $\rightarrow$ 1.4%), validating the superiority of the scene-centric approach.
- Dual masking reduces the ATE of DROID-SLAM from 2.42m to 0.32m, showing significant effectiveness especially in long sequences.
- WHAM fails on complex trajectories (large curves, walking up/down stairs), whereas TRAM generalizes well because it does not rely on MoCap motion priors.
- Metric scale estimation adds only about 30cm of error compared to the ground-truth scale (0.32m $\rightarrow$ 0.66m).
- VIMO achieves a PA-MPJPE of 34.1mm on 3DPW, outperforming HMR2.0 and WHAM.

## Highlights & Insights

- Clever two-stage decoupled design: Leverages the scene background (rather than human motion models) to estimate the metric scale, bypassing the generalization bottleneck of MoCap priors.
- Performing temporal modeling in the SMPL pose space (rather than the latent feature space) is more direct and effective.
- Masked DROID is a practical engineering contribution, resolving the robustness issue of visual SLAM in scenes with large-area dynamic objects.
- The insight that the background scene provides scale information is highly inspiring ("seeing Spider-Man swinging between skyscrapers is enough to determine the scale").

## Limitations & Future Work

- SLAM fails to work when the camera is completely static, making it inapplicable to tripod-mounted camera scenes.
- ZoeDepth depth predictions are inaccurate in certain scenes, which degrades scale estimation.
- VIMO's training data is limited (3DPW + H36M + BEDLAM); scaling up with more video data would further improve performance.

## Rating

- Novelty: ⭐⭐⭐⭐
- Effectiveness: ⭐⭐⭐⭐⭐ — Large reduction in global trajectory error
- Practicality: ⭐⭐⭐⭐ — Applicable to in-the-wild videos
- Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)

</div>

<!-- RELATED:END -->
