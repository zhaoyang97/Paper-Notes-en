---
title: >-
  [Paper Note] Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps
description: >-
  [ICCV 2025][3D Vision][3DGS SLAM] This paper proposes S3PO-GS, a monocular RGB-only outdoor SLAM system that anchors pose estimation to 3DGS-rendered pointmaps for scale self-consistency, and employs a patch-based dynamic mapping mechanism, achieving high-accuracy localization without cumulative scale drift and high-fidelity novel view synthesis.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3DGS SLAM"
  - "Monocular Vision"
  - "Scale Consistency"
  - "Pointmap"
  - "Outdoor Scenes"
date: 2026-05-08
content_hash: 9372fd3e0fb8dbbc
---

# Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps

**Conference**: ICCV 2025
**arXiv**: [2507.03737](https://arxiv.org/abs/2507.03737)  
**Code**: [Project Page](https://3dagentworld.github.io/S3PO-GS/)  
**Area**: 3D Vision
**Keywords**: 3DGS SLAM, Monocular Vision, Scale Consistency, Pointmap, Outdoor Scenes

## TL;DR

This paper proposes S3PO-GS, a monocular RGB-only outdoor SLAM system that anchors pose estimation to 3DGS-rendered pointmaps for scale self-consistency, and employs a patch-based dynamic mapping mechanism, achieving high-accuracy localization without cumulative scale drift and high-fidelity novel view synthesis.

## Background & Motivation

3D Gaussian Splatting (3DGS) has become a popular choice for SLAM due to its high-fidelity real-time rendering, but faces two core challenges in **RGB-only outdoor scenes**:

**Lack of geometric priors**: Tracking methods based on differentiable rendering pipelines (e.g., MonoGS) tend to fall into local optima in complex outdoor environments and suffer from convergence difficulties.

**Scale drift**: Introducing independent tracking modules (e.g., Photo-SLAM, OpenGS-SLAM) supplements geometric constraints but requires maintaining scale alignment between the external module and the 3DGS map. Under large rotations and translations, cumulative errors lead to severe scale drift.

**Core Insight**: Pre-trained pointmap models (e.g., MASt3R) can provide geometric priors, but their scale is inconsistent with the 3DGS scene. The authors propose **not involving the pre-trained model in pose estimation itself**, but using it solely to establish pixel-level 2D–3D correspondences — with 3D coordinates derived entirely from 3DGS-rendered pointmaps — fundamentally eliminating the scale alignment problem.

## Method

### Overall Architecture

S3PO-GS comprises three core modules: 3DGS scene representation, self-consistent 3DGS Pointmap Tracking, and Patch-based Dynamic Mapping. The system initializes the 3DGS map using MASt3R (optimized for 1000 steps), then performs tracking and mapping for each new frame.

### Key Designs

1. **Pointmap-Anchored Pose Estimation (PAPE)**:

    - A 3DGS depth map $D_{ak}$ is rendered from the viewpoint of an adjacent keyframe; the **rendered pointmap** $X_{ak}^r$ is obtained by back-projecting through camera intrinsics.
    - A pre-trained model (MASt3R/DUSt3R) generates **pre-trained pointmaps** $X_{ak}^p, X_n^p$ for the current frame and keyframe; inter-frame pixel correspondences are established via nearest-neighbor matching.
    - The correspondence chain $X_{ak}^r \leftrightarrow I_{ak} \leftrightarrow I_n$ is propagated to yield **2D–3D correspondences**.
    - RANSAC + PnP is applied to solve the relative pose $\mathbf{T}_n^{rel}$.
    - **Key advantage**: 3D coordinates originate from 3DGS rendering and are inherently scale-consistent with the scene; the pre-trained model serves only as a "bridge" without participating in pose computation.

2. **Pose Refinement**:

    - Starting from the PnP-initialized pose, the photometric loss $L_{pho} = \|I(\mathcal{G}, T) - \bar{I}\|_1$ is minimized through the 3DGS differentiable rendering pipeline.
    - The pose $T \in SE(3)$ is linearized in the Lie algebra $\mathfrak{se}(3)$, with gradients computed explicitly within the CUDA pipeline.
    - Only **5 iterations** are required to achieve accuracy comparable to 100 iterations in conventional methods.

3. **Patch-Level Scale Alignment**:

    - The rendered pointmap $X^r$ and pre-trained pointmap $X^p$ are divided into $P \times P$ patches.
    - Patches with similar statistical distributions (mean/standard deviation differences below thresholds) are selected for normalization.
    - A set of "correct points" $CP$ is identified in the normalized space, and the scaling factor is computed as: $\sigma' = \frac{\mu(X^r[CP])}{\mu(X^p[CP])}$
    - Iterative alignment continues until the scaling factor stabilizes, yielding $\hat{X}^p = \sigma \times X^p$.
    - If insufficient correct points are found, a remedial scaling factor is computed using the aligned pointmap of an adjacent keyframe.

4. **Pointmap Replacement Mechanism**:

    - When inserting new Gaussians at keyframes, the aligned pre-trained pointmap $\hat{X}^p$ is used to detect "erroneous points" in the rendered pointmap $X^r$.
    - Points with deviation exceeding the threshold are replaced: $\hat{X}^r(x) = \hat{X}^p(x)$ if $|X^r(x) - \hat{X}^p(x)| > \epsilon_m \times \hat{X}^p(x)$
    - Random sparse downsampling controls the number of Gaussians.

### Loss & Training

The total loss jointly optimizes poses and the Gaussian map over a keyframe window with three terms:

$$\min_{T_k, \mathcal{G}} \sum_{k \in \mathcal{W}} \alpha L_{pho}^k + (1-\alpha) L_{geo}^k + \lambda_{iso} L_{iso}$$

- **Photometric loss** $L_{pho}$: L1 rendering reconstruction loss
- **Geometric loss** $L_{geo} = \|X^r - \hat{X}^p\|_1$: pointmap alignment supervision
- **Isotropy regularization** $L_{iso}$: prevents excessive elongation of Gaussian ellipsoids

## Key Experimental Results

### Main Results

Comprehensive evaluation on three outdoor datasets — Waymo, KITTI, and DL3DV:

| Dataset | Metric | S3PO-GS (Ours) | OpenGS-SLAM | MonoGS | GlORIE-SLAM |
|---------|--------|----------------|-------------|--------|-------------|
| Waymo | ATE↓ | **0.622** | 0.839 | 8.529 | 0.589 |
| Waymo | PSNR↑ | **26.73** | 23.99 | 21.80 | 18.83 |
| KITTI | ATE↓ | **1.048** | 3.224 | 9.493 | 1.134 |
| KITTI | PSNR↑ | **20.03** | 15.61 | 14.78 | 15.49 |
| DL3DV | ATE↓ | **0.032** | 0.141 | 0.274 | 0.492 |
| DL3DV | PSNR↑ | **29.97** | 24.75 | 24.99 | 16.20 |

NVS PSNR improvements across all datasets are +2.73/+4.42/+4.98; tracking achieves best performance on KITTI and DL3DV, and is comparable to GlORIE-SLAM on Waymo.

### Ablation Study

| Configuration | ATE↓ | PSNR↑ | Note |
|---------------|------|-------|------|
| w/o pose refinement | 1.79 | 24.45 | Tracking degrades without refinement |
| w/o scale alignment | 3.50 | 23.49 | Unaligned pointmaps introduce erroneous supervision |
| w/o point replacement | 1.35 | 25.59 | Incorrect Gaussian insertion degrades map quality |
| w/o $L_{geo}$ | 3.73 | 25.70 | Lack of geometric supervision reduces translation awareness |
| **Full model** | **0.62** | **26.73** | — |

Convergence iteration ablation (Waymo 405841 scene, ATE):

| Iterations | MonoGS | OpenGS-SLAM | S3PO-GS |
|------------|--------|-------------|---------|
| 5 | 12.6 | 4.17 | **0.55** |
| 50 | 2.98 | 0.85 | **0.49** |
| 100 | 1.70 | 0.80 | **0.46** |

### Key Findings

- The PAPE module enables pose estimation to converge in only 5 iterations (MonoGS requires 50+).
- Directly incorporating pre-trained pointmaps into MonoGS without the proposed processing leads to severe geometric blurring due to scale drift.
- Patch-level local alignment is more robust than global alignment, avoiding contamination from outliers.

## Highlights & Insights

- **Elegant design**: The pre-trained model is used solely as a "matchmaker" (establishing correspondences) without directly participating in pose computation, eliminating scale alignment issues at the architectural level.
- **Plug-and-play**: The approach generalizes to other 3DGS SLAM systems that need to integrate external geometric priors.
- **Rapid convergence**: Stable accuracy is achieved in just 5 iterations, making it highly compatible with real-time systems.

## Limitations & Future Work

- Loop closure detection is not incorporated; drift may still accumulate over long sequences.
- Dependence on the inference speed of pre-trained pointmap models may constrain real-time applicability.
- Evaluation is limited to static scenes; dynamic objects are not addressed.

## Related Work & Insights

- **MASt3R/DUSt3R**: Provide powerful pre-trained pointmap priors, but direct application to SLAM introduces scale inconsistency.
- **MonoGS**: A pioneering 3DGS SLAM system, but suffers from convergence difficulties in outdoor scenes.
- **GlORIE-SLAM**: Achieves high tracking accuracy via inter-frame relationships, but does not support NVS.

## Rating

- Novelty: ⭐⭐⭐⭐ (Scale self-consistent design is conceptually novel)
- Technical Depth: ⭐⭐⭐⭐ (Complete tracking-mapping pipeline)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets + thorough ablations)
- Practical Value: ⭐⭐⭐⭐ (Directly targets outdoor scenarios such as autonomous driving)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings](../../CVPR2026/3d_vision/sce-slam_scale-consistent_monocular_slam_via_scene_coordinate_embeddings.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](../../CVPR2025/3d_vision/wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[ICCV 2025\] 4D Gaussian Splatting SLAM](4d_gaussian_splatting_slam.md)
- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)

</div>

<!-- RELATED:END -->
