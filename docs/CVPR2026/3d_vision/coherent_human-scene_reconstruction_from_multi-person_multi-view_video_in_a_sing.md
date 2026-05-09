---
title: >-
  [Paper Note] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass
description: >-
  [CVPR 2026][3D Vision][human-scene joint reconstruction] CHROMM is proposed as a unified framework that jointly estimates camera parameters, scene point clouds, and human body meshes (SMPL-X) from multi-person multi-view video in a single forward pass, without external modules or preprocessed data. It achieves competitive performance on global human motion estimation and multi-view pose estimation tasks while being more than 8× faster than optimization-based methods.
tags:
  - CVPR 2026
  - 3D Vision
  - human-scene joint reconstruction
  - multi-person multi-view
  - SMPL-X
  - 3D foundation model
  - feed-forward inference
date: 2026-05-08
content_hash: f35d1c26da95dcee
---

# Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass

**Conference**: CVPR 2026
**arXiv**: [2603.12789](https://arxiv.org/abs/2603.12789)
**Code**: [https://nstar1125.github.io/chromm](https://nstar1125.github.io/chromm)
**Area**: 3D Vision
**Keywords**: human-scene joint reconstruction, multi-person multi-view, SMPL-X, 3D foundation model, feed-forward inference

## TL;DR
CHROMM is proposed as a unified framework that jointly estimates camera parameters, scene point clouds, and human body meshes (SMPL-X) from multi-person multi-view video in a single forward pass, without external modules or preprocessed data. It achieves competitive performance on global human motion estimation and multi-view pose estimation tasks while being more than 8× faster than optimization-based methods.

## Background & Motivation
Joint 3D human-scene reconstruction is a core problem in computer vision with applications in robotics, autonomous driving, and AR/VR. **State of the Field and Limitations of Prior Work**: (1) **Monocular methods** such as UniSH and Human3R have made progress but do not support multi-view input; (2) **Multi-view methods** such as HSfM and HAMSt3R rely on external modules (2D keypoint detectors, cross-view ReID) or require iterative optimization, resulting in high system complexity and computational cost; (3) Appearance-based cross-view person re-identification (ReID) is unreliable in visually similar scenes such as those with uniforms.

**Root Cause**: How to jointly reconstruct cameras, scenes, and human bodies across multiple persons and views in a unified manner without relying on external modules, preprocessed data, or iterative optimization? **Starting Point**: The paper integrates priors from Pi3X (near-metric scale scene reconstruction) and Multi-HMR (multi-person whole-body mesh recovery) into a unified network, introduces a scale alignment module to bridge the scale gap between the two, and designs a test-time multi-view fusion strategy along with a geometry-based rather than appearance-based multi-person association method.

## Method

### Overall Architecture
Multi-view multi-timestep RGB images $\{I^v_t\}$ are flattened into a sequence $\{I_n\}$ (exploiting the permutation equivariance of Pi3). A dual encoder extracts scene features $F^{scene}$ (Pi3X) and human features $F^{human}$ (Multi-HMR). Scene features are decoded by the Pi3X decoder to reconstruct point maps and camera parameters. Detected head tokens from the human features are fused with scene tokens and then regressed to SMPL-X parameters. At test time, multi-person association, multi-view fusion, and scale adjustment are applied.

### Key Designs
1. **Dual-Feature Encoding**:

    - Function: Extract dedicated feature representations for scene and human body respectively.
    - Mechanism: The Pi3X encoder extracts global 3D geometric features $F^{scene}$, and the Multi-HMR encoder extracts human-specific features $F^{human}$. The two feature streams are **not fused early** — scene features are passed to the Pi3X decoder, and human features are passed directly to the human reconstruction head.
    - Design Motivation: Experiments show that altering the input distribution to the decoder (even with frozen weights) degrades geometric reconstruction performance. Preserving the input distribution of the Pi3X decoder is essential to fully exploit its pretrained priors.

2. **Head-Pelvis Scale Adjustment Module**:

    - Function: Resolve the scale mismatch between the near-metric scale scene predicted by Pi3X and the metric-scale SMPL body.
    - Mechanism: The 2D head-pelvis distance in the image $\ell^{\text{img}}$ is compared with the projected SMPL head-pelvis distance $\ell^{\text{smpl}}$ to compute a global adjustment ratio $r = \frac{1}{|\mathcal{S}|}\sum \frac{\ell^{\text{smpl}}}{\ell^{\text{img}}}$, yielding the corrected scale $s^* = r \cdot s$.
    - Pelvis localization adopts a **coarse-to-fine strategy**: a coarse pelvis position is first estimated from the head token, then a patch is sampled around that position for fine-grained offset prediction.
    - Design Motivation: The scene scale from Pi3X may be too small (causing SMPL to penetrate the ground) or too large (causing SMPL to float). The head-pelvis distance serves as a stable body proportion reference.

3. **Test-Time Multi-View Fusion**:

    - Function: Aggregate per-view estimates into a unified global representation without optimization.
    - Mechanism: SMPL parameters are divided into view-invariant and view-dependent categories:
        - **View-invariant** (shape $\beta$, canonical pose $\theta$): averaged directly across views.
        - **View-dependent** (root rotation $R$, head translation $\tau$): transformed to world coordinates using estimated camera extrinsics; rotations are averaged as quaternions, and translations are resolved via **multi-view ray triangulation**.
    - Design Motivation: Token-level max-pooling mixes view-dependent features and degrades view-invariant parameter estimation; explicit separation yields more principled aggregation.

4. **Geometry-Based Multi-Person Association**:

    - Function: Establish cross-view person identity correspondences.
    - Mechanism: Within a single view, human token L2 distance combined with Sinkhorn optimal transport is used for temporal tracking, with 3D joint displacement thresholds filtering outliers. Across views, the matching cost $\mathcal{C}(a,b) = \lambda_p\|\mathcal{J}^a - \mathcal{J}^b\|_2 + \lambda_\theta\|\mathcal{J}^{a,\text{canon}} - \mathcal{J}^{b,\text{canon}}\|_2$ (with $\lambda_p=0.8, \lambda_\theta=0.2$) is used with Hungarian matching and threshold filtering.
    - Design Motivation: Appearance-based ReID fails in visually similar scenes; geometric features based on 3D position and pose are more robust.

### Loss & Training
- **Two-stage training**: Stage 1 freezes the encoders and trains the SMPL decoder, fusion/mask/pelvis detection MLPs on BEDLAM (20 epochs); Stage 2 trains only the pelvis detection MLP on a mixed dataset (10 epochs).
- Stage 1: geometric losses (3D vertices + joints + 2D projection) + parameter losses (pose/shape/translation) + detection losses (BCE for mask/head/pelvis).
- Stage 2: pelvis detection + 2D reprojection + Chamfer distance loss (visible SMPL vertices vs. point map depth).
- Training runs on 4× A100 for approximately 2 days.

## Key Experimental Results

### Main Results

| Dataset | Metric | CHROMM-multi | CHROMM-mono | Human3R | UniSH |
|--------|------|-------------|-------------|---------|-------|
| EMDB-2 | WA-MPJPE↓ | - | **102.6** | 112.2 | 118.5 |
| EMDB-2 | W-MPJPE↓ | - | **255.0** | 267.9 | 270.1 |
| RICH | WA-MPJPE↓ | **53.1** | 87.5 | 110.0 | 118.1 |
| RICH | W-MPJPE↓ | **79.0** | 138.3 | 184.9 | 183.2 |

Multi-view pose estimation (EgoHumans):

| Method | W-MPJPE↓ | GA-MPJPE↓ | PA-MPJPE↓ | Time↓ |
|------|----------|-----------|-----------|-------|
| **CHROMM** | **0.51** | **0.15** | **0.05** | ~4s |
| HSfM | 1.04 | 0.21 | 0.05 | ~118s |
| HAMSt3R | 3.80 | 0.42 | 0.14 | ~32s |

### Ablation Study

| Configuration | WA-MPJPE↓ | W-MPJPE↓ | Note |
|------|-----------|----------|------|
| w/o scale adj. | 169.7 | 447.9 | Scale mismatch severely degrades global accuracy |
| w/ scale adj. | **102.6** | **255.0** | Scale adjustment is critical |

Multi-view fusion strategy (RICH): Only Avg. 69.3 → Max-Pool+Tri. 63.2 → **Avg.+Tri. 53.1**

Multi-person association: Position+Pose combined accuracy **91.3%** vs. Pose-only 70.6% vs. Position-only 91.1%

### Key Findings
- The scale adjustment module reduces WA-MPJPE from 169.7 to 102.6 (−39.5%).
- Explicitly separating view-invariant/dependent attributes with triangulation outperforms implicit max-pooling by 16 mm.
- Geometry-based multi-person association achieves 91.3% accuracy; Pose-only precision is only 48.5% (severe over-matching).
- Runtime of ~4s vs. HAMSt3R ~32s vs. HSfM ~118s (8× speedup).

## Highlights & Insights
- **System-level innovation in unified feed-forward inference**: The first joint reconstruction framework for multi-person multi-view settings that requires no external modules, no preprocessing, and no optimization.
- **Elegant Head-Pelvis scale ratio design**: Exploits stable body proportion relationships to resolve the scale gap between Pi3X and SMPL.
- **Decision against early feature fusion**: Counterintuitive yet effective — preserving the input distribution of pretrained models is more important than forcing feature fusion.
- **ReID insight on geometry vs. appearance**: 3D position and pose are more robust than appearance features for person association.

## Limitations & Future Work
- Heavy reliance on head tokens leads to performance degradation under severe head occlusion.
- Fully frozen encoders limit the model's ability to adapt to novel scenes.
- Geometry-based association requires at least partial frame-level visibility overlap.
- Future directions include integrating the dual encoders into a unified encoder and enabling robust reconstruction under head occlusion.

## Related Work & Insights
- The Pi3X + Multi-HMR integration paradigm is generalizable to other "Foundation Model A + Specialized Model B" fusion scenarios.
- The test-time multi-view fusion strategy (view-invariant/dependent separation) is worth adopting in other multi-view estimation tasks.
- The geometry-based ReID idea has broader implications for multi-person multi-view tracking.

## Rating
- Novelty: ⭐⭐⭐⭐ System-level integration innovation with concise and practical module designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, two tasks, multiple ablations, and runtime comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-formatted figures and tables, contributions stated explicitly.
- Value: ⭐⭐⭐⭐⭐ First unified feed-forward reconstruction framework for multi-person multi-view settings with high practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[CVPR 2026\] MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction](mv-roma_from_pairwise_matching_into_multi-view_track_reconstruction.md)
- [\[CVPR 2026\] PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)

<!-- RELATED:END -->
