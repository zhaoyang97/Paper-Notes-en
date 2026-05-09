---
title: >-
  [Paper Note] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass
description: >-
  [CVPR 2026][3D Vision][Multi-view human reconstruction] CHROMM is a unified framework that integrates the geometric prior of Pi3X and the human prior of Multi-HMR into a single feed-forward network, enabling joint reconstruction of cameras, scene point clouds, and SMPL-X human meshes from multi-person multi-view video in a single pass—without external modules, preprocessing, or iterative optimization. It achieves a multi-view WA-MPJPE of 53.1 mm on RICH and runs more than 8× faster than HAMSt3R.
tags:
  - CVPR 2026
  - 3D Vision
  - Multi-view human reconstruction
  - multi-person scene
  - SMPL-X
  - 3D foundation model
  - scale alignment
date: 2026-05-08
content_hash: 59c35c530825cadf
---

# Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass

**Conference**: CVPR 2026
**arXiv**: [2603.12789](https://arxiv.org/abs/2603.12789)
**Code**: [Project Page](https://nstar1125.github.io/chromm)
**Area**: 3D Vision / Joint Human-Scene Reconstruction
**Keywords**: Multi-view human reconstruction, multi-person scene, SMPL-X, 3D foundation model, scale alignment

## TL;DR

CHROMM is a unified framework that integrates the geometric prior of Pi3X and the human prior of Multi-HMR into a single feed-forward network, enabling joint reconstruction of cameras, scene point clouds, and SMPL-X human meshes from multi-person multi-view video in a single pass—without external modules, preprocessing, or iterative optimization. It achieves a multi-view WA-MPJPE of 53.1 mm on RICH and runs more than 8× faster than HAMSt3R.

## Background & Motivation

**Background**: Joint 3D human-scene reconstruction is a core problem in computer vision, with applications in robotics, autonomous driving, and AR/VR. Recent 3D foundation models (DUSt3R, VGGT, Pi3X) have advanced scene reconstruction, while Multi-HMR has enabled multi-person human mesh recovery.

**Limitations of Prior Work**:

1. Monocular methods such as UniSH and Human3R cannot exploit multi-view information, limiting their accuracy.
2. Multi-view methods such as HSfM and HAMSt3R rely on additional modules (2D keypoint detectors, cross-view ReID modules) or require iterative optimization, resulting in high system complexity.
3. Appearance-based ReID methods fail in visually similar scenarios (e.g., uniformed subjects).
4. The near-metric scale output by Pi3X is misaligned with the true metric scale of SMPL—causing human meshes to penetrate the ground or float above it.

**Key Challenge**: Simultaneous reconstruction of the scene and multiple humans is hindered by scale inconsistency between the two, difficulty in cross-view person association, and the desire to avoid reliance on external preprocessing.

**Goal**: To build a unified feed-forward framework that requires no external modules or preprocessed data and performs joint multi-person multi-view human-scene reconstruction in a single pass.

**Key Insight**: Fuse the scene prior of Pi3X with the human prior of Multi-HMR, design a scale adjustment module to bridge the two, and replace appearance-based matching with geometric cues for cross-view association.

**Core Idea**: Dual-encoder late fusion + head-pelvis ratio scale adjustment + view-invariant/view-dependent decomposition fusion + geometry-driven multi-person association.

## Method

### Overall Architecture

Input multi-view multi-person video $\{I_t^v\}$ → Dual encoders (Pi3X for scene features; Multi-HMR for human features) → Pi3X decoder reconstructs point maps and cameras → Head detection extracts human tokens, which are fused with scene tokens → SMPL decoder regresses pose/shape/translation → At test time: per-view tracking → geometry-driven cross-view multi-person association → view-invariant/view-dependent decomposition fusion → scale adjustment module aligns scene and human.

### Key Designs

1. **Dual-Encoder Late Fusion Architecture**

    - The Pi3X encoder captures global 3D geometry; the Multi-HMR encoder is optimized for human body representation.
    - Key design decision: **early fusion is avoided**—feeding human tokens into the Pi3X decoder disrupts the input distribution and degrades scene reconstruction.
    - Human tokens are fused with scene tokens only after decoding, via an MLP: $H_n = \text{MLP}_{\text{fuse}}([Z_n^{\text{scene}} | Z_n^{\text{human}}])$

2. **Depth-Residual Translation Estimation**

    - Rather than directly regressing 3D head translation, the method exploits the depth prior provided by Pi3X point maps.
    - A residual relative to the scene depth map is predicted: $d_n^m = d_{n,m}^{\text{coarse}} + \Delta d_n^m$, which is back-projected to a 3D position using the 2D head keypoint and camera intrinsics.
    - Ablation: depth residual (107.5 mm) vs. direct depth (133.8 mm) vs. direct translation regression (196.4 mm)—the differences are substantial.

3. **Head-Pelvis Ratio Scale Adjustment**

    - Problem: The near-metric scale $s$ output by Pi3X may be underestimated (human penetrates the ground) or overestimated (human floats above it).
    - Solution: Compute the ratio between the 2D head-pelvis distance in the image $\ell^{\text{img}}$ and the projected SMPL head-pelvis distance $\ell^{\text{smpl}}$, then average over all frames and persons to obtain a global adjustment factor $r = \frac{1}{|\mathcal{S}|}\sum \frac{\ell^{\text{smpl}}}{\ell^{\text{img}}}$, yielding the corrected scale $s^* = r \cdot s$.
    - Coarse-to-fine pelvis localization: the head token estimates a coarse position → the corresponding patch regresses an offset → the coarse position is used as fallback if the pelvis is out of bounds.
    - Ablation: scale adjustment reduces WA-MPJPE from 169.7 to 102.6 mm (−39.5%).

4. **Multi-View Fusion (Test Time, Optimization-Free)**

    - View-invariant quantities (shape $\beta$, pose $\theta$): direct parameter averaging, which outperforms implicit token max-pooling.
    - View-dependent quantities (rotation $R$, translation $\tau$): transformed to world coordinates, then averaged via quaternion averaging and multi-view ray triangulation, respectively.
    - Ablation: Avg+Tri (53.1) > MaxPool+Tri (63.2) > Only Avg (69.3).

5. **Geometry-Based Multi-Person Association**

    - Per-view tracking: head token L2 distance for inter-frame matching; Sinkhorn optimal transport handles unmatched detections.
    - Cross-view association cost: $\mathcal{C}(a,b) = 0.8 \cdot \|3\text{D position difference}\| + 0.2 \cdot \|\text{canonical pose difference}\|$, solved with the Hungarian algorithm for one-to-one matching.
    - Ablation: position alone 91.1% precision vs. pose alone 70.6%; combined 91.3%.

### Loss & Training

- **Two-stage training**: Stage 1 freezes the Pi3X and Multi-HMR encoders and trains new modules including the SMPL decoder (20 epochs, BEDLAM, lr = 5e-5; scale adjustment disabled for the first 10 epochs).
- Stage 2 unfreezes only the pelvis detection MLP (10 epochs, mixed 3DPW + MPII + COCO + BEDLAM, lr = 1e-4).
- Stage 1 losses: 3D vertex/joint L1 ($\lambda = 5.0$) + 2D reprojection L1 + SMPL parameter L1 + detection BCE + pelvis BCE.
- Stage 2 adds: Chamfer distance (visible SMPL vertices vs. predicted depth map).
- Training hardware: 4 × A100, approximately 2 days.

## Key Experimental Results

### Main Results (Global Human Motion Estimation)

| Method | Multi-View | No External Modules | EMDB-2 WA-MPJPE↓ (mm) | RICH WA-MPJPE↓ (mm) | RICH W-MPJPE↓ (mm) |
|--------|-----------|--------------------|-----------------------|---------------------|---------------------|
| JOSH3R | ✗ | ✗ | 220.0 | — | — |
| UniSH | ✗ | ✗ | 118.5 | 118.1 | 183.2 |
| Human3R | ✗ | ✓ | 112.2 | 110.0 | 184.9 |
| CHROMM-mono | ✗ | ✓ | **102.6** | 87.5 | 138.3 |
| CHROMM-multi | ✓ | ✓ | — | **53.1** | **79.0** |

### Multi-View Pose Estimation

| Method | No ReID | No Optimization | EgoHumans W-MPJPE↓ (m) | EgoHumans GA-MPJPE↓ (m) | EgoExo4D W-MPJPE↓ (m) |
|--------|---------|----------------|------------------------|--------------------------|------------------------|
| HSfM | ✗ | ✗ | 1.04 | 0.21 | 0.56 |
| HAMSt3R | ✓ | △ | 3.80 | 0.42 | 0.51 |
| **CHROMM** | ✓ | ✓ | **0.51** | **0.15** | **0.26** |

### Runtime

| Method | Per-Frame Inference Time (3 persons, 4 views) |
|--------|----------------------------------------------|
| HSfM | ~118 s |
| HAMSt3R | ~32 s |
| **CHROMM** | **~4 s** (8×+ speedup) |

### Key Findings

- Multi-view fusion yields substantial gains: RICH WA-MPJPE drops from 87.5 mm (monocular) to 53.1 mm (multi-view), a 39.3% improvement.
- Scale adjustment is the most critical module: removing it raises WA-MPJPE from 102.6 to 169.7 mm (+65.5%).
- The depth-residual strategy outperforms direct translation regression by 89 mm (107.5 vs. 196.4 mm).
- Geometry-based association (91.3% accuracy) substantially outperforms pose-only matching (70.6%).
- CHROMM is 29× faster than HSfM and 8× faster than HAMSt3R, while requiring no ReID.

## Highlights & Insights

- **First end-to-end unified framework for multi-person multi-view human-scene reconstruction**: requires no external modules, preprocessing, or optimization.
- **Head-pelvis ratio scale adjustment**: bridges the scale gap between scene and human using anatomical proportions—simple yet highly effective.
- **View-invariant/view-dependent decomposition fusion**: explicit parameter averaging combined with triangulation outperforms implicit token aggregation.
- **Geometry-driven cross-view association**: avoids the failure of appearance matching in uniformed scenarios; the combination of 3D position and canonical pose is an elegant design choice.

## Limitations & Future Work

- Heavy reliance on head tokens for human detection—performance degrades when heads are occluded or invisible.
- The dual encoders are not unified into a single encoder—there remains room to improve scene-human interaction modeling.
- Extreme close-up shots (head filling the image) and close-range interpersonal interactions are typical failure cases.
- Scale adjustment depends on pelvis visibility—it degrades under full-body occlusion.

## Related Work & Insights

- **vs. Human3R**: CHROMM extends to multi-view without external modules, achieving 9.6 mm improvement on EMDB-2 and 57 mm on RICH.
- **vs. HSfM**: CHROMM is 29× faster, with EgoHumans W-MPJPE of 0.51 m vs. 1.04 m (50% improvement).
- **vs. HAMSt3R**: CHROMM is 8× faster and supports multi-person association without external ReID.
- Insights: integrating 3D foundation models with human body priors is an emerging trend; scale alignment is a central engineering challenge; the view-invariant/view-dependent decomposition is generalizable to other multi-view estimation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — First unified multi-person multi-view framework free of external dependencies; scale adjustment and geometric association are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, monocular and multi-view comparisons, comprehensive ablations, and runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ — Contributions are clearly articulated; each design decision is validated experimentally.
- Value: ⭐⭐⭐⭐ — Fast inference and preprocessing-free operation are practically significant for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction](mv-roma_from_pairwise_matching_into_multi-view_track_reconstruction.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[CVPR 2026\] PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)

</div>

<!-- RELATED:END -->
