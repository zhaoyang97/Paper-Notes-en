---
title: >-
  [Paper Note] Multi-View Pose-Agnostic Change Localization with Zero Labels
description: >-
  [CVPR 2025][3D Vision][Multi-view change detection] This paper proposes the first zero-label, pose-agnostic multi-view change detection method. By constructing a change-aware 3DGS representation to fuse multi-view change information, it improves mIoU by 1.7 times compared to the baseline and is capable of generating change masks for unseen views.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-view change detection"
  - "3D Gaussian Splatting"
  - "Zero-label"
  - "Pose-agnostic"
  - "DINOv2"
date: 2026-05-08
content_hash: d228d7bf146b6bd9
---

# Multi-View Pose-Agnostic Change Localization with Zero Labels

**Conference**: CVPR 2025  
**arXiv**: [2412.03911](https://arxiv.org/abs/2412.03911)  
**Code**: [https://MV-3DCD.github.io](https://MV-3DCD.github.io)  
**Area**: 3D Vision / Change Detection  
**Keywords**: Multi-view change detection, 3D Gaussian Splatting, Zero-label, Pose-agnostic, DINOv2

## TL;DR

This paper proposes the first zero-label, pose-agnostic multi-view change detection method. By constructing a change-aware 3DGS representation to fuse multi-view change information, it improves mIoU by 1.7 times compared to the baseline and is capable of generating change masks for unseen views.

## Background & Motivation

**Background**: Change detection typically relies on precisely aligned pre- and post-event image pairs, which limits its application scenarios. A few methods support inconsistent poses but require labeled training data.

**Limitations of Prior Work**: Existing label-free methods (such as OmniPoseAD and SplatPose) only compare image pairs on a single-view basis, which is severely affected by view-dependent pseudo-changes (e.g., reflections and shadows).

**Key Challenge**: Single-view comparisons tend to yield many false positives, whereas multi-view fusion lacks an effective 3D representation.

**Goal**: To build a 3D change representation leveraging multi-view information to suppress view-dependent false positives.

**Key Insight**: Embedding change channels into 3DGS and modeling view-independent changes using zero-order spherical harmonic coefficients.

**Core Idea**: Learning additional change channels (change magnitude + change opacity) within the 3DGS of the evaluation scene to fuse multi-view feature- and structure-aware change masks.

## Method

### Overall Architecture

(1) Reconstruct 3DGS_ref using reference scene images; (2) Render reference scene images from the viewpoints of the evaluation scene; (3) Compare the rendered images with actual images to generate feature- and structure-aware change masks; (4) Embed change information into Change-3DGS_inf of the evaluation scene; (5) Render multi-view change masks from arbitrary novel viewpoints.

### Key Designs

1. **Feature- and Structure-Aware Change Mask**:

    - Function: Detect candidate change regions from a single viewpoint.
    - Mechanism: Extract feature differences using DINOv2 to obtain the feature-aware mask $M_F^k$; compute SSIM to acquire the structure-aware mask $M_S^k$; perform element-wise multiplication of both to obtain the combined mask $M_{F,S}^k$.
    - Design Motivation: Feature and structural information are complementary—DINOv2 captures semantic changes, while SSIM captures pixel-level changes.

2. **3DGS Change Channel Embedding**:

    - Function: Encode change information within the 3D representation to achieve multi-view fusion.
    - Mechanism: Append two parameters—change magnitude $\tilde{c}$ and change opacity $\tilde{\alpha}$—to each Gaussian point, model the change magnitude using zero-order spherical harmonic coefficients (view-independent), and supervise change channel rendering with L1 + D-SSIM loss.
    - Design Motivation: Zero-order spherical harmonic coefficients model the changes as view-independent, effectively suppressing view-dependent false positives such as reflections and shadows.

3. **Data Augmentation Strategy**:

    - Function: Increase the number of training masks used for learning change channels.
    - Mechanism: Use Change-3DGS_inf to render evaluation scene images from reference scene viewpoints, backward-calculate change masks, and merge them with forward masks to augment training data.
    - Design Motivation: The number of evaluation scene images can be very small (as few as 5 images); data augmentation improves the quality of change channel learning.

### Loss & Training

Change channel learning utilizes L1 + D-SSIM loss. The final change mask is filtered via the alpha channel to exclude unseen regions: $M^k = M_{ren}^k \cdot \mathbf{1}(A_{ren}^k \geq 0.5)$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | SplatPose | Gain |
|--------|------|------|-----------|------|
| MAD-Real | mIoU | 0.132 | 0.077 | 1.7× |
| MAD-Real | F1 | 0.210 | 0.123 | 1.7× |
| ChangeSim | mIoU(C) | 0.407 | - | 1.7× vs CSCDNet |
| PASLCD | Mean mIoU | Highest | Second Highest | Leading in all scenes |

### Key Findings

- Only 5 evaluation scene images are required to learn effective change channels.
- Able to generate change masks for novel viewpoints that are unseen in both the evaluation and reference scenes.
- Zero-order spherical harmonic coefficients perform better than higher-order counterparts, verifying the assumption that changes are view-independent.

## Highlights & Insights

- Lifts change detection to the 3D representation level for the first time, achieving genuine multi-view fusion.
- Contributes the PASLCD dataset containing 10 real-world scenes.
- The method can serve as a multi-view extension for any single-view change detection method.

## Limitations & Future Work

- Relies on COLMAP to register evaluation scene images to the reference scene.
- Extreme appearance changes (such as complete darkness) may lead to COLMAP registration failure.
- Training the change channels requires scene-by-scene optimization.

## Rating

- Novelty: 9/10 — For embedding change channels in 3DGS for the first time.
- Technical Depth: 8/10 — The design of modeling change with zero-order spherical harmonics is theoretically sound.
- Experimental Thoroughness: 8/10 — Three datasets plus a newly contributed dataset.
- Writing Quality: 8/10 — Clear description of the methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [\[CVPR 2025\] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)
- [\[CVPR 2025\] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)

</div>

<!-- RELATED:END -->
