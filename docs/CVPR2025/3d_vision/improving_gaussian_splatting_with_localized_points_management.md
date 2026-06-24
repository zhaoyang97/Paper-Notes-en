---
title: >-
  [Paper Note] Improving Gaussian Splatting with Localized Points Management
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper proposes a Localized Points Management (LPM) strategy that localizes error-prone 3D regions using multi-view geometric constraints. Within these regions, targeted point densification and opacity resetting are executed. As a plug-and-play module, it improves the reconstruction quality of various 3DGS models while maintaining real-time rendering speed.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Point Management"
  - "Adaptive Density Control"
  - "Novel View Synthesis"
  - "Geometric Calibration"
date: 2026-05-08
content_hash: 0f224bc89e2deaa3
---

# Improving Gaussian Splatting with Localized Points Management

**Conference**: CVPR 2025  
**arXiv**: [2406.04251](https://arxiv.org/abs/2406.04251)  
**Code**: [https://happy-hsy.github.io/projects/LPM/](https://happy-hsy.github.io/projects/LPM/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Point Management, Adaptive Density Control, Novel View Synthesis, Geometric Calibration

## TL;DR

This paper proposes a Localized Points Management (LPM) strategy that localizes error-prone 3D regions using multi-view geometric constraints. Within these regions, targeted point densification and opacity resetting are executed. As a plug-and-play module, it improves the reconstruction quality of various 3DGS models while maintaining real-time rendering speed.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) achieves real-time, high-quality novel view synthesis by initializing 3D Gaussian points via SfM and optimizing them through differentiable rasterization. During training, Adaptive Density Control (ADC) is used to manage point distribution—densifying points (clone/split) based on view-average gradient thresholds, pruning based on opacity thresholds, and periodically resetting opacity globally.

**Limitations of Prior Work**: ADC has three key limitations: (1) global gradient thresholds fail to identify all under-optimized regions that need densification, as large Gaussian points often have low average gradients and are easily overlooked; (2) when the SfM initialization points are sparse, ADC struggles to add sufficient reliable new points in points-deficient regions; (3) there is a lack of mechanism to deal with "ill-conditioned points"—incorrect high-opacity points block valid points, leading to inaccurate depth estimation (e.g., misaligned Gaussian points in window areas floating in front).

**Key Challenge**: The global threshold strategy of ADC cannot adapt to local variations in scene geometric complexity, and only focuses on point densification (where more points are needed) while ignoring geometric calibration (which points have errors).

**Goal**: (1) How to accurately localize 3D regions that cause rendering errors in 3D space? (2) How to perform reasonable point operations within these regions—both densifying missing points and correcting harmful ill-conditioned points?

**Key Insight**: Rendering error maps indicate where anomalies occur on 2D images. Through multi-view geometric constraints (feature matching + double cone projection intersection), 2D errors can be back-traced to the causative 3D regions. Targeted local operations can be performed within these localized regions instead of relying on global thresholds.

**Core Idea**: Use rendering error maps and multi-view geometric constraints to locate 3D error-source regions, and execute localized point densification and opacity resetting of front ill-conditioned points within these regions.

## Method

### Overall Architecture

LPM runs in parallel with the original ADC as an auxiliary point management module during 3DGS training. The pipeline consists of three steps: (1) generating the rendering error map of the current view; (2) localizing the 3D error-source region through cross-view feature matching and bi-cone projection intersection; (3) performing local point densification, opacity resetting of front points, and density-adaptive pruning within the localized regions. The entire process does not modify the 3DGS architecture, allowing it to integrate into any 3DGS-based method as a plug-and-play plugin.

### Key Designs

1. **Error-Contributed 3D Region Localization**:

    - **Function**: Accurately trace 2D rendering errors back to source regions in 3D space.
    - **Mechanism**: Implemented in two steps: first perform cross-view feature matching via LightGlue to map the error region $R_e$ of the current view to the corresponding region $R_e'$ of adjacent reference views. Then, cast conical rays from both views toward $R_e$ and $R_e'$ (the cone radius equals the minimum bounding circle radius of the error region), compute the intersection of the two cones, and define the error-source 3D region $R_{zone}$ as the minimum bounding sphere containing these intersections.
    - **Design Motivation**: Relying solely on 2D error maps cannot determine which points in 3D are faulty (a single pixel can correspond to multiple Gaussians at different depths). Multi-view geometric constraints provide accurate 2D-to-3D projection, and feature matching is performed offline to save computation.

2. **Localized Point Operations (Densification + Resetting)**:

    - **Function**: Perform targeted point operations within the localized error-source regions.
    - **Mechanism**: Handled in two cases within $R_{zone}$: (1) if there are Gaussian points inside the region, perform local densification with a lower gradient threshold (cloning small Gaussians and repositioning them along the gradient direction, splitting large Gaussians) to enhance geometric details; (2) if no points exist in the region (point sparsity), directly add new Gaussian points at the center of the region. Meanwhile, perform opacity resetting on high-opacity points located in front of $R_{zone}$—these points may block valid points behind them, and resetting grants them a chance to be re-optimized.
    - **Design Motivation**: The global threshold of standard ADC either misses under-optimized regions or, if lowered, generates a massive number of redundant points in already dense regions. LPM densifies with a low threshold only in "most needed" regions, which is precise and efficient. Resetting front points addresses the geometric calibration problem that is entirely neglected by ADC.

3. **Density-Adaptive Pruning**:

    - **Function**: Control model inflation and maintain real-time rendering speed.
    - **Mechanism**: Adaptively prune points from lowest to highest opacity based on the point density inside the region. The amount of pruned points is determined by the regional density—higher density leads to more pruning.
    - **Design Motivation**: LPM adds new points in error regions, which significantly increases the total point count if left uncontrolled. Density-adaptive pruning ensures that the growth of total points is controlled, maintaining training and rendering efficiency.

### Loss & Training

LPM does not alter the original 3DGS loss functions and training strategies, using the same training schedules and hyperparameters as original 3DGS and SpaceTimeGS. Point matching (LightGlue) is extracted offline to save computational cost. All experiments are conducted on an RTX 3090, training for 30k iterations.

## Key Experimental Results

### Main Results

| Dataset | Metric | 3DGS* + LPM | 3DGS* | PixelGS* + LPM | PixelGS* |
|--------|------|-------------|-------|----------------|----------|
| Mip-NeRF 360 | PSNR↑ | 27.59 | 27.47 | **27.80** | 27.54 |
| Tanks&Temples | PSNR↑ | 23.83 | 23.67 | **24.02** | 23.75 |
| Deep Blending | PSNR↑ | 29.76 | 29.55 | 29.65 | 29.58 |
| Neural 3D Video (Full) | PSNR↑ | **32.40** | 31.99 | - | - |
| Neural 3D Video (Flame Salmon) | PSNR↑ | 29.84 | 29.48 | - | - |

LPM as a plugin consistently improves all baseline methods (3DGS, 2DGS, MipGS, PixelGS, SpaceTimeGS), with the most significant improvements on Tanks&Temples (challenging scenes such as transparent/reflective surfaces).

### Ablation Study

| Configuration | PSNR (PlayRoom) | LPIPS | PSNR (Truck) | Description |
|------|-----------------|-------|--------------|------|
| Full LPM | **30.22** | 0.241 | **25.61** | Full model |
| w/o Point Densification | 30.10 | 0.241 | 25.43 | PSNR drops by 0.12/0.18 without densification |
| w/o Reset | 30.07 | 0.243 | 25.52 | PSNR drops by 0.15/0.09 without resetting |

| Method | PSNR (Truck) | No. of Gaussians | Training Time |
|------|-------------|---------|---------|
| 3DGS* | 25.42 | 257k | 19min |
| 3DGS* (Low Threshold) | 25.45 | 635k | 35min |
| GaussianPro | 25.40 | 312k | 36min |
| PixelGS | 25.51 | 518k | 37min |
| **3DGS + LPM** | **25.61** | 265k | 21min |

### Key Findings

- Point densification and opacity resetting both provide positive contributions and complement each other—densification supplements missing geometric details, while resetting corrects occlusion errors.
- Simply lowering the global gradient threshold causes the number of Gaussian points to double (257k -> 635k) and actually decreases PSNR (producing redundant points in already dense regions). Powered by precise localization, LPM achieves better performance while only slightly increasing the point count (265k).
- Compared to GaussianPro and PixelGS, LPM requires shorter training time (21min vs 36-37min) because feature matching is fast and operations are performed only in required regions.
- Integrating LPM into SpaceTimeGS on dynamic scenes (Neural 3D Video) achieves SOTA performance (32.40 PSNR), demonstrating that LPM is equally effective for 4D scenes.

## Highlights & Insights

- **Strong generality of "error back-tracing" concept**: The idea of projecting 2D rendering errors back to 3D space via multi-view geometry is not limited to 3DGS and can be extended to any scene reconstruction method requiring localizing problematic 3D regions.
- **Plug-and-play design**: LPM does not modify any underlying architecture and can be seamlessly integrated into multiple methods like 3DGS, 2DGS, MipGS, PixelGS, and SpaceTimeGS, proving its versatility.
- **Resetting front points is a key innovation**: Previous methods (GaussianPro, PixelGS) only address densification. LPM is the first to propose resetting ill-conditioned high-opacity points in front of error regions, offering a chance to correct erroneous geometry.

## Limitations & Future Work

- LPM still follows the original split/clone densification rules of 3DGS, which might not be the optimal operation strategy for under-optimized points.
- Feature matching relies on the quality of LightGlue, which may fail in textureless or repetitive texture regions.
- The cone intersection of error regions uses a minimum bounding sphere approximation, which might not be sufficiently accurate in complex geometry (e.g., narrow slits/occlusion boundaries).
- Point management is only improved during training, and optimizations for the final rendering pipeline have not been explored.

## Related Work & Insights

- **vs GaussianPro**: GaussianPro propagates points directly in sparse regions via multi-view stereo, but nearly doubles training time (36min vs 21min) and causes OOM on PlayRoom.
- **vs PixelGS**: PixelGS suppresses close-to-camera artifacts via gradient scaling, but takes a long training time (37min) and results in a high point count (518k). LPM achieves better results with fewer points and less time.
- **vs Standard ADC**: ADC uses a global threshold strategy, whereas LPM adopts a localized strategy. They are complementary—ADC handles regular regions, while LPM focuses on difficult regions.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of error back-tracing localization + localized operation is clear and practical. Resetting front points is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on static 3D and dynamic 4D scenes, with plugin validation across 5 baseline methods, accompanied by comprehensive ablation and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Exquisite illustrations (e.g., the pipeline diagram in Fig.2) and in-depth problem analysis.
- Value: ⭐⭐⭐⭐ Holds broad application value as a universal plugin for the 3DGS ecosystem, though the depth of the core innovation is slightly limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Seurat: From Moving Points to Depth](seurat_from_moving_points_to_depth.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)

</div>

<!-- RELATED:END -->
