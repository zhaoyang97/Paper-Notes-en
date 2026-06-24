---
title: >-
  [Paper Note] 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians
description: >-
  [CVPR 2025][3D Vision][4D SLAM] This paper presents the first 4D tracking and mapping method (4DTAM) based on differentiable rendering and 2D Gaussian surface primitives. By jointly optimizing camera poses, scene geometry, appearance, and dynamic deformation fields, 4DTAM achieves real-time reconstruction of non-rigid dynamic scenes from monocular RGB-D video streams. It also releases a novel synthetic 4D dataset, Sim4D, for evaluation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D SLAM"
  - "non-rigid reconstruction"
  - "2D Gaussian Splatting"
  - "deformation field"
  - "dynamic scenes"
date: 2026-05-08
content_hash: 1382ac3fc3416331
---

# 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians

**Conference**: CVPR 2025  
**arXiv**: [2505.22859](https://arxiv.org/abs/2505.22859)  
**Code**: [https://muskie82.github.io/4dtam/](https://muskie82.github.io/4dtam/)  
**Area**: 3D Vision  
**Keywords**: 4D SLAM, non-rigid reconstruction, 2D Gaussian Splatting, deformation field, dynamic scenes

## TL;DR
This paper presents the first 4D tracking and mapping method (4DTAM) based on differentiable rendering and 2D Gaussian surface primitives. By jointly optimizing camera poses, scene geometry, appearance, and dynamic deformation fields, 4DTAM achieves real-time reconstruction of non-rigid dynamic scenes from monocular RGB-D video streams. It also releases a novel synthetic 4D dataset, Sim4D, for evaluation.

## Background & Motivation
Visual SLAM technology has made massive progress over the past two decades, but the vast majority of methods assume static scenes. The real world is full of moving elements—rivers, trees, pedestrians, etc. While dynamic objects can be detected and filtered out to focus on the static parts, doing so discards the spatio-temporal information of the scene. True 4D reconstruction (3D space + time) requires handling both camera motion and non-rigid scene deformation simultaneously. This is an extremely difficult problem because the optimization space is extremely high-dimensional and observations from a single camera are highly sparse.

Existing non-rigid SLAM methods (such as the DynamicFusion series) typically rely on TSDF volume representations and specific deformation models, which limits their reconstruction accuracy. Recently, 3D Gaussian Splatting (3DGS) has shown strong capabilities in static SLAM, but its blob-like representation is not suitable for accurate surface reconstruction. On the other hand, most dynamic reconstruction methods require known camera poses or multi-camera systems, which limits their practical application.

The core insight of this paper is to utilize 2D Gaussian Splatting (2DGS) as surface primitives, combining them with an MLP deformation field to model non-rigid motion. Simultaneously, it derives analytical camera pose Jacobians to achieve efficient pose estimation, thereby building a complete 4D SLAM system.

## Method

### Overall Architecture
4DTAM adopts a classic tracking-mapping dual-thread architecture. The tracking module is responsible for fast online pose estimation; the mapping module jointly optimizes camera poses, canonical space Gaussians, and deformation field parameters within a sliding window. The input is a monocular RGB-D video stream, and the output is a complete 4D spatio-temporal reconstruction model.

### Key Designs

1. **2DGS Surface Primitive Representation**:

    - Unlike 3DGS, 2DGS constrains each Gaussian onto a 2D tangent plane, which naturally possesses a well-defined surface normal direction.
    - Each 2D Gaussian is represented by a 3D mean position, a rotation matrix (decomposed into two tangent vectors and a normal vector), color, opacity, and a 2D scaling vector.
    - Efficient rendering is achieved through ray-splat intersection, avoiding numerically unstable matrix inversions.
    - This surface representation better utilizes depth signals, which is crucial for monocular non-rigid reconstruction.

2. **Analytical Camera Pose Jacobians**:

    - This is a major technical contribution of the paper: deriving complete analytical pose gradients for 2DGS.
    - Using Lie algebra to parameterize the $SE(3)$ pose, the partial derivative of the transformation matrix $M^T$ with respect to the pose parameter $\tau$ is derived.
    - The Jacobian of the rendered surface normals with respect to the camera pose is also derived.
    - Implemented via CUDA kernels, this retains the real-time rendering advantages of Gaussian splatting.
    - This formulation has wide applicability beyond SLAM.

3. **MLP Deformation Field (Warp Field)**:

    - A compact MLP network is used as the deformation field to map Gaussians from the canonical space to the deformed space at each time step.
    - The input consists of frequency positional encodings of time $t$ and the Gaussian center position $x$, and the output features displacement $\delta x$, rotational offset $\delta r$, and scaling offset $\delta s$.
    - The continuity of the MLP naturally provides a smoothness prior for motion.
    - A CUDA-optimized MLP implementation (tiny-cuda-nn) ensures execution efficiency.

4. **Tracking Module**:

    - Minimizes photometric and depth errors between the current frame and the rendered deformed Gaussian model.
    - Key design: Camera poses are estimated relative to the deformed Gaussians at the timestamp of the latest keyframe, under the assumption that the deformed scene changes continuously over time.
    - Every $N$ frames, a keyframe is selected and sent to the mapping module.

5. **Mapping Module and Regularization**:

    - When a new keyframe arrives, new canonical space Gaussians are generated by back-projecting RGB-D observations.
    - Innovatively, finite differences of the depth map are used to compute surface normals to initialize the 2DGS normals, which performs better than random initialization.
    - A sensor-normal-based supervision loss is introduced, avoiding the high computational cost of calculating rendered depth differential normals at each step in the original 2DGS method.
    - As-Rig-As-Possible (ARAP) regularization constrains relative displacements between neighboring Gaussians to remain rigid.
    - An innovative normal rigidity loss: constrains the relative relationship of normals of neighboring Gaussians to remain similar across different time steps.

### Loss & Training
Total loss function: $L_{total} = \lambda_p L_p + \lambda_g L_g + \lambda_n L_n + \lambda_{iso} L_{iso} + L_{ARAP} + L_{ARAP\_n}$

- $L_p$: Photometric rendering loss (L1)
- $L_g$: Depth rendering loss (L1)
- $L_n$: Normal consistency loss (based on sensor measurements)
- $L_{iso}$: Isotropic loss
- $L_{ARAP}$: Positional rigidity regularization
- $L_{ARAP\_n}$: Normal rigidity regularization (newly proposed)

Global optimization stage: After tracking is completed, poses and the number of Gaussians are fixed, and keyframes are selected randomly for global optimization, taking about 1 minute on an RTX 4090.

## Key Experimental Results

### Main Results

| Dataset | Metric | 4DTAM | SurfelWarp | Gain |
|--------|------|-------|------------|------|
| Sim4D (curtain) | ATE RMSE (cm) | **0.25** | 6.10 | 24x |
| Sim4D (flag) | ATE RMSE (cm) | **1.00** | 31.9 | 32x |
| Sim4D (mercedes) | PSNR (dB) | **32.13** | 25.7 | +6.4 |
| Sim4D (shoe_rack) | L1 Depth (cm) | **0.99** | 4.25 | 4x |
| Sim4D (Average) | ATE RMSE (cm) | **~0.35** | ~7.0 | ~20x |

### Ablation Study

| Configuration | ATE (cm) | Depth L1 (cm) | F1 (%) | Description |
|------|---------|---------------|--------|------|
| MonoGS (3DGS) | 0.59 | 4.52 | 31.9 | Baseline static SLAM |
| MonoGS-2D (Ours 2DGS) | **0.36** | **0.54** | **88.8** | 2DGS surface primitives significantly improve geometric reconstruction |

Offline non-rigid reconstruction ablation:

| Configuration | Metric | Ours | Morpheus | Description |
|------|------|------|----------|------|
| iPhone Dataset | Depth L1 (cm) | **0.57** | 2.4 | More accurate geometry |
| iPhone Dataset | LPIPS | **0.26** | 0.63 | Significantly improved rendering quality |

### Key Findings
- Using 2DGS as a SLAM surface representation yields a qualitative leap in geometric reconstruction compared to 3DGS (F1 score increases from 31.9% to 88.8%).
- Normal initialization is crucial for 2DGS; initialization based on depth sensors is far superior to random initialization.
- Normal rigidity regularization effectively prevents surface tearing during non-rigid deformation.
- The camera pose estimation speed is around 1.5 fps.

## Highlights & Insights
- First to introduce 2DGS to SLAM and derive complete analytical pose Jacobians, providing reference value for the entire GS-SLAM field.
- The proposed normal rigidity loss is clever—it utilizes the surface normal characteristics of 2DGS to constrain the local rigidity of deformation, which is impossible with 3DGS.
- The construction approach of the Sim4D dataset is worth learning from: utilizing large-scale open-source 3D models and animations, rendered through Blender to generate 4D data with complete annotations.
- High engineering completeness, ranging from mathematical derivations to CUDA implementations.

## Limitations & Future Work
- Tracking speed is only 1.5 fps, which is still far from real-time (30 fps).
- Global optimization still requires an extra 1 minute of post-processing.
- Currently relies on RGB-D input; pure RGB extension is only briefly demonstrated in the supplementary materials.
- The Sim4D dataset mainly features dynamics of a single object, not yet covering large-scale multi-object dynamic scenes.
- Topology changes (such as object splitting) are not yet handled effectively.

## Related Work & Insights
- The DynamicFusion series pioneered non-rigid SLAM based on TSDF, but geometric accuracy is limited by voxel resolution.
- MonoGS validated the feasibility of Gaussian Splatting in SLAM, which this work extends to 2DGS and dynamic scenes.
- The MLP representation of deformation fields draws inspiration from D-NeRF and Nerfies, but associates it with 2DGS and SLAM frameworks for the first time.
- Compared to DyNoMo, this work supports both pose optimization and high-quality geometric reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 2DGS-based 4D SLAM, with multiple technical innovations (analytical pose Jacobians, normal rigidity loss).
- Experimental Thoroughness: ⭐⭐⭐⭐ New dataset + multi-perspective ablation studies, though real-world evaluation is mostly qualitative.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete mathematical derivations and clear paper structure.
- Value: ⭐⭐⭐⭐⭐ Opens up new research directions for modern 4D SLAM; the dataset and evaluation protocols will facilitate subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GauSTAR: Gaussian Surface Tracking and Reconstruction](gaustar_gaussian_surface_tracking_and_reconstruction.md)
- [\[CVPR 2026\] RINO: Rotation-Invariant Non-Rigid Correspondences](../../CVPR2026/3d_vision/rino_rotation-invariant_non-rigid_correspondences.md)
- [\[CVPR 2025\] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)
- [\[CVPR 2025\] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)

</div>

<!-- RELATED:END -->
