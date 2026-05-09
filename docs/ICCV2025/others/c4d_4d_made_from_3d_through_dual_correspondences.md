---
title: >-
  [Paper Note] C4D: 4D Made from 3D through Dual Correspondences
description: >-
  [ICCV 2025][4D reconstruction] This paper proposes C4D, a framework that upgrades existing 3D reconstruction paradigms to full 4D reconstruction by jointly capturing dual temporal correspondences — short-term optical flow and dynamic-aware long-term point tracking (DynPT) — on top of DUSt3R's 3D pointmap predictions. Motion masks are generated to separate static and dynamic regions. Three optimization objectives are introduced: camera motion alignment, camera trajectory smoothing, and point trajectory smoothing. The resulting system produces per-frame point clouds, camera parameters, and 2D/3D trajectories, achieving competitive performance across depth estimation, pose estimation, and point tracking tasks.
tags:
  - ICCV 2025
  - 4D reconstruction
  - temporal correspondence
  - point tracking
  - motion segmentation
  - DUSt3R
date: 2026-05-08
content_hash: b716084ab1990fa7
---

# C4D: 4D Made from 3D through Dual Correspondences

**Conference**: ICCV 2025
**arXiv**: [2510.14960](https://arxiv.org/abs/2510.14960)
**Code**: [https://littlepure2333.github.io/C4D](https://littlepure2333.github.io/C4D)
**Area**: Other
**Keywords**: 4D reconstruction, temporal correspondence, point tracking, motion segmentation, DUSt3R

## TL;DR
This paper proposes C4D, a framework that upgrades existing 3D reconstruction paradigms to full 4D reconstruction by jointly capturing dual temporal correspondences — short-term optical flow and dynamic-aware long-term point tracking (DynPT) — on top of DUSt3R's 3D pointmap predictions. Motion masks are generated to separate static and dynamic regions. Three optimization objectives are introduced: camera motion alignment, camera trajectory smoothing, and point trajectory smoothing. The resulting system produces per-frame point clouds, camera parameters, and 2D/3D trajectories, achieving competitive performance across depth estimation, pose estimation, and point tracking tasks.

## Background & Motivation
Pointmap-based methods such as DUSt3R achieve strong results in static scene 3D reconstruction, but fail in dynamic scenes where moving objects violate multi-view geometric constraints. Existing 4D approaches either require fine-tuning the model (e.g., MonST3R) or rely on complex NeRF/3DGS optimization pipelines. The central question is: how can temporal correspondence information be leveraged to upgrade 3D reconstruction to 4D without modifying pretrained weights?

## Core Problem
How can temporal correspondences (optical flow + point tracking) be exploited to distinguish static from dynamic regions, improve camera pose estimation, and achieve temporally smooth 4D reconstruction?

## Method

### Overall Architecture
Monocular video → DUSt3R/MASt3R/MonST3R predicts pointmaps + DynPT predicts long-term trajectories and dynamic scores + optical flow estimates short-term correspondences → correspondence-guided motion mask prediction (static-point fundamental matrix + epipolar error) → multi-objective optimization (GA + CMA + CTS + PTS) → 4D output (per-frame point clouds / depth / camera poses / intrinsics / motion masks / 2D+3D trajectories).

### Key Designs
1. **DynPT (Dynamic-aware Point Tracker)**: Built on the CoTracker architecture, augmented with a 3D-aware ViT encoder (frozen DUSt3R pretrained encoder) and a CNN dual-feature extractor. A Transformer iteratively updates each tracked point's position, confidence, visibility, and **mobility**. Trained on Kubric; mobility ground truth is generated via positional difference thresholding.
2. **Correspondence-Guided Motion Mask**: Static points predicted by DynPT are used to sample static correspondences from optical flow → LMedS estimates the fundamental matrix (reflecting camera motion only) → Sampson distance identifies dynamic regions violating epipolar constraints → masks from multiple frames are fused via union. This yields more accurate motion masks than MonST3R.
3. **Correspondence-Assisted Optimization**:
   - CMA (Camera Motion Alignment): constrains the ego-motion field to be consistent with optical flow in static regions.
   - CTS (Camera Trajectory Smoothing): penalizes abrupt changes in rotation and translation between adjacent frames.
   - PTS (Point Trajectory Smoothing): adaptively weighted smoothing on sparse tracked points → linear blending displacement (LBD) propagated to dense points.
4. **Plug-and-Play**: No modifications to DUSt3R/MASt3R/MonST3R weights; new objectives are introduced only at the optimization stage.

### Training / Optimization Details
- DynPT: trained for 50K steps, batch size 32, AdamW + OneCycle scheduler, lr $5 \times 10^{-4}$.
- Optimization is conducted in two stages: (1) GA + CMA + CTS optimize depth, pose, and intrinsics; (2) pose is fixed and PTS optimizes depth only.
- Each stage runs for 300 iterations, Adam lr $0.01$.

## Key Experimental Results

### Camera Pose Estimation (ATE ↓)

| Method | Sintel | TUM-dyn | ScanNet |
|--------|--------|---------|---------|
| MonST3R+GA | 0.158 | 0.099 | 0.075 |
| **C4D-M** | **0.103** | **0.071** | **0.061** |
| DROID-SLAM† | 0.175 | — | — |
| LEAP-VO† | 0.089 | 0.068 | 0.070 |

- Compared to MonST3R+GA: Sintel ATE reduced by 35%; RPE_rot reduced from 1.924 to 0.705.
- Competitive with dedicated VO methods that require GT intrinsics.

### Video Depth Estimation (AbsRel ↓, scale-only alignment)

| Method | Sintel | Bonn | KITTI |
|--------|--------|------|-------|
| MonST3R | 0.345 | 0.065 | 0.159 |
| **C4D-M** | **0.338** | **0.063** | **0.091** |
| DepthCrafter | 0.692 | 0.217 | 0.141 |

Under scale-only alignment, KITTI AbsRel improves from 0.159 to 0.091 (43% relative gain over MonST3R).

### Point Tracking (TAP-Vid DAVIS AJ ↑)

| Method | AJ | $\delta_\text{avg}$ | OA |
|--------|----|---------------------|----|
| CoTracker | 61.8 | 76.1 | 88.3 |
| **DynPT** | **61.6** | 75.4 | 87.4 |

Competitive with SOTA, with the additional capability of predicting mobility (D-ACC: MOVi-E 87.9%, Pan.MOVi-E 94.1%).

### Ablation Study (Sintel)

| Variant | ATE ↓ | RPE_t ↓ | RPE_r ↓ |
|---------|-------|---------|---------|
| w/o CMA | 0.140 | 0.051 | 0.905 |
| w/o CTS | 0.131 | 0.058 | 1.348 |
| w/o PTS | 0.103 | 0.040 | 0.705 |
| C4D (full) | 0.103 | 0.040 | 0.705 |

CTS has the largest impact on RPE_rot (0.705 → 1.348 when removed).

## Highlights & Insights
- **Plug-and-play 4D upgrade**: The 3D-to-4D transition requires no fine-tuning of 3D model weights; it is achieved solely through new optimization objectives and temporal correspondences.
- **Mobility prediction in DynPT**: A key innovation — distinguishing whether a point's motion originates from camera movement or object movement, enabling more accurate motion mask prediction.
- **LMedS + fundamental matrix for motion segmentation**: An elegant solution — the fundamental matrix is estimated using only static points, and any region violating epipolar constraints is classified as dynamic.
- **Multi-frame motion mask fusion**: Addresses cases where temporarily stationary dynamic objects (e.g., a pedestrian's feet while standing) appear static between adjacent frame pairs.

## Limitations & Future Work
- DynPT is trained on synthetic Kubric data; the domain gap may affect mobility prediction in real-world dynamic scenes.
- The optimization stage is relatively slow (2 × 300 iterations).
- PTS provides marginal improvement on quantitative depth metrics, though temporal smoothness is substantially enhanced (best assessed via visual evaluation).

## Related Work & Insights
- **vs. MonST3R**: MonST3R fine-tunes the DUSt3R decoder; C4D leaves model weights unchanged and upgrades reconstruction through optimization. C4D produces more accurate motion masks (Fig. 6).
- **vs. Shape-of-Motion / GFlow**: These methods require NeRF/3DGS optimization; C4D is more lightweight by operating directly on pointmaps.
- **vs. DROID-SLAM / LEAP-VO**: These methods require GT intrinsics; C4D operates from monocular video alone.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — DynPT's mobility prediction and correspondence-guided motion masks are the primary contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 3D/4D comparisons, three downstream tasks, six+ datasets, ablations, and motion segmentation evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams are clear; method descriptions are complete.
- **Value**: ⭐⭐⭐⭐ — Provides both 4D reconstruction methodology and an important extension of the DUSt3R ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization](iap_invisible_adversarial_patch_attack_through_perceptibility-aware_localization.md)
- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](lacoot_layer_collapse_through_optimal_transport.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[NeurIPS 2025\] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](../../NeurIPS2025/others/4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)

</div>

<!-- RELATED:END -->
