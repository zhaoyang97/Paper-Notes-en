---
title: >-
  [Paper Note] IM360: Large-scale Indoor Mapping with 360 Cameras
description: >-
  [ICCV 2025][3D Vision][indoor 3D mapping] This paper presents IM360, a 3D mapping pipeline for large-scale indoor environments captured under sparse scanning conditions. By deeply integrating a spherical camera model int…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "indoor 3D mapping"
  - "360° panoramic cameras"
  - "spherical SfM"
  - "texture optimization"
  - "large-scale indoor scenes"
date: 2026-05-08
content_hash: 0963ed78ceaad507
---

# IM360: Large-scale Indoor Mapping with 360 Cameras

**Conference**: ICCV 2025
**arXiv**: [2502.12545](https://arxiv.org/abs/2502.12545)  
**Code**: [https://jdk9405.github.io/IM360/](https://jdk9405.github.io/IM360/) (project page available)  
**Area**: 3D Vision
**Keywords**: indoor 3D mapping, 360° panoramic cameras, spherical SfM, texture optimization, large-scale indoor scenes

## TL;DR

This paper presents IM360, a 3D mapping pipeline for large-scale indoor environments captured under sparse scanning conditions. By deeply integrating a spherical camera model into every stage of SfM—combined with dense feature matching and differentiable rendering-based texture optimization—IM360 achieves substantially superior camera localization accuracy and rendering quality on Matterport3D and Stanford2D3D compared to existing methods (PSNR gain of 3.5).

## Background & Motivation

Indoor 3D mapping and photo-realistic rendering are core technologies for AR/VR, robot navigation, and related applications. Conventional image acquisition pipelines rely on perspective cameras with limited field of view, requiring dense capture to achieve scene coverage, which is both time-consuming and labor-intensive.

**Advantages and challenges of 360° panoramic cameras**: A single panoramic image covers the full spherical field of view. Matterport3D and Stanford2D3D are captured at approximately 0.23 images/m², whereas ScanNet, using a perspective camera, requires 72.35 images/m². Nevertheless, panoramic cameras present two major challenges for 3D reconstruction:

**Challenge 1: SfM failure under sparse viewpoints.** Large-scale indoor scenes contain extensive texture-less regions (white walls, floors) and frequent occlusions. Traditional SfM systems (COLMAP, OpenMVG) rely on keypoint detection and sparse matching, which degrade severely in such regions. Even with state-of-the-art matchers (SuperPoint+SuperGlue), nearly half of images on Matterport3D fail to register. No complete SfM pipeline specifically designed for the spherical camera model currently exists.

**Challenge 2: Poor neural rendering quality under sparse viewpoints.** NeRF and 3DGS optimization depend on densely overlapping views; in sparse panoramic scanning scenarios, these methods tend to overfit training views and produce low-quality novel-view synthesis. Methods designed specifically for ERP images (e.g., OmniSDF) fail to converge on large-scale scenes, and Gaussian splatting approaches generate severe floater artifacts.

**Core Starting Point**: Integrating the spherical camera model (unit bearing vector representation) into every step of SfM—from dense matching and two-view geometry to triangulation and Bundle Adjustment—and combining classical texture mapping with differentiable rendering fine-tuning to achieve high-quality reconstruction. A key observation is that, in sparse indoor scenes, a hybrid approach of classical texture mapping plus neural optimization outperforms purely neural rendering methods.

## Method

### Overall Architecture

IM360 consists of three stages: (1) Spherical SfM — estimating camera poses and sparse 3D points from sparse panoramic images; (2) Geometry Reconstruction — converting ERP images to cubemaps and reconstructing meshes via neural SDF; (3) Texture Optimization — initializing with classical texture mapping followed by differentiable rendering fine-tuning of diffuse and specular textures.

### Key Designs

1. **Spherical Dense Matching SfM**

    - Function: Constructs the first complete spherical SfM pipeline, with all steps operating on the spherical manifold.
    - Mechanism:
        - **Feature matching**: Uses EDM, a dense matching method designed for 360° images, to extract dense correspondences directly on ERP images. This is more efficient and accurate than converting ERP to cubemap and performing sparse matching across 36 perspective image pairs.
        - **Spherical two-view geometry**: Image points are represented as unit bearing vectors $u \in \mathbb{S}^2$. The classical epipolar constraint is extended to $u_1^T E u_2 = 0$. The essential matrix $E$ is estimated via an 8-point method within the DLT framework, with additional normalization to improve numerical stability of SVD.
        - **Spherical Bundle Adjustment**: Observations are represented as bearing vectors $u$; reprojection error is defined as $L = \sum_i \sum_j \rho(\|\Pi(P_j; R_i, t_i) - u_{ij}\|^2)$, where $\Pi: \mathbb{R}^3 \mapsto \mathbb{S}^2$ is the spherical projection.
        - Quantization, rounding, and merging strategies are applied to dense matching results to ensure multi-view matching consistency.
    - Design Motivation: Existing COLMAP and OpenMVG provide only fragmented support for panoramic images, lacking an integrated spherical pipeline from matching to optimization. The core advantage of spherical representation is the large-viewpoint overlap afforded by the 360° field of view, which is critical for sparse scanning.

2. **Geometry Reconstruction**

    - Function: Reconstructs high-quality 3D meshes from panoramic images.
    - Mechanism: ERP images are projected to cubemaps to generate 6 perspective views. Monocular depth and normal priors are estimated using the pretrained Omnidata model, which are fed into a DebSDF network to learn a signed distance field; meshes are then extracted via Marching Cubes.
    - Design Motivation: Methods trained directly on ERP images (OmniSDF) fail to converge due to severe distortion. Gaussian splatting-based surface reconstruction methods (SuGaR, VCR-GauS) produce fragmented results in sparse scenes. Converting to cubemaps enables reuse of mature perspective-based methods and monocular prior models.

3. **Texture Optimization**

    - Function: Generates high-quality texture maps while modeling both diffuse reflectance and view-dependent specular effects.
    - Mechanism:
        - **Initialization**: The classical texture mapping method TexRecon selects the optimal color image for each mesh face and generates an initial texture atlas $K_d$.
        - **Diffuse texture optimization**: The texture atlas is parameterized and rendered via the differentiable rasterizer nvdiffrast; an L1+SSIM photometric loss against real images is used to optimize the texture.
        - **Specular modeling**: A specular feature $K_s \in \mathbb{R}^3$ is additionally initialized, and a compact MLP $f_s$ is trained as a fragment shader, taking the specular feature and viewing direction as input and outputting the specular color. The final rendered color is $\hat{I} = \hat{I_d} + \hat{I_s}$.
        - Loss function: $L_{photo} = (1-\alpha)\|\hat{I} - I\| + \alpha(1 - SSIM(\hat{I}, I))$, with $\alpha=0.2$.
    - Design Motivation: Classical texture mapping suffers from seams and color inconsistencies, while purely neural rendering severely overfits in sparse scenes. The hybrid scheme uses classical methods for stable initialization and differentiable rendering for fine-tuning, balancing robustness and quality. Adding the specular component yields an additional 0.8 PSNR over diffuse-only optimization.

### Loss & Training

- SfM stage: Spherical BA uses a soft-L1 robust loss for local BA; no robust kernel is applied in global BA.
- Geometry reconstruction: DebSDF trains an 8-layer MLP (hidden dimension 256) with input images at 384×384; Adam optimizer with lr=5e-4 and exponential decay.
- Texture optimization: Specular MLP is 2-layer with 32 hidden units; Adam lr=5e-4, trained for 7,000 steps at 512×512 rendering resolution.

## Key Experimental Results

### Main Results

Camera localization performance on Matterport3D:

| Method | Registration Rate (6 scenes overall) | AUC@5° (Mean) |
|--------|--------------------------------------|---------------|
| OpenMVG | Very low (<50% in multiple scenes) | ~12 |
| SPSG COLMAP | Moderate | ~29 |
| DKM COLMAP | High | ~30 |
| SphereGlue COLMAP | High | ~31 |
| **IM360 (Ours)** | **100% (all 6 scenes fully registered)** | **~57** |

Rendering quality comparison on Matterport3D:

| Method | Rendering | PSNR (Mean) | SSIM (Mean) | LPIPS (Mean) |
|--------|-----------|-------------|-------------|--------------|
| ZipNeRF | Volume | 13.9 | 0.51 | 0.68 |
| 3DGS | Splat | 13.4 | 0.47 | 0.55 |
| SparseGS | Splat | 14.3 | 0.46 | 0.53 |
| TexRecon | Mesh | 15.9 | 0.54 | 0.43 |
| **IM360 (Ours)** | **Mesh** | **19.4** | **0.67** | **0.37** |

### Ablation Study

Contribution of texture optimization components (Matterport3D PSNR Mean):

| Configuration | PSNR | Notes |
|---------------|------|-------|
| TexRecon (no optimization) | 15.9 | Classical texture mapping baseline |
| IM360* (diffuse only) | 18.6 | Differentiable rendering fine-tunes diffuse texture, +2.7 |
| IM360 (diffuse + specular) | **19.4** | Specular MLP added, additional +0.8 |

Registration rate comparison on Stanford2D3D:

| Method | area 3 (85 imgs) | area 4 (258 imgs) | area 5a (143 imgs) |
|--------|------------------|-------------------|--------------------|
| OpenMVG | 6/85 | 17/258 | 8/143 |
| SPSG COLMAP | 28/85 | 73/258 | 54/143 |
| **IM360** | **85/85** | **258/258** | **138/143** |

### Key Findings

- IM360 achieves 100% image registration across all Matterport3D scenes, whereas OpenMVG registers fewer than 10% of images in multiple scenes.
- Neural rendering methods (NeRF, 3DGS) underperform classical texture mapping (TexRecon) in sparse panoramic scenes, demonstrating that sparse viewpoints are the core bottleneck.
- Texture optimization contributes a total of 3.5 PSNR improvement: 2.7 from diffuse fine-tuning and 0.8 from specular modeling.
- Spherical dense matching shows a significant advantage over sparse/dense matching on cubemaps in scenes with heavy occlusion.

## Highlights & Insights

- This is the first complete spherical SfM pipeline, with all steps from feature matching to BA performed on the spherical manifold, filling a critical gap in the toolchain for panoramic camera 3D reconstruction.
- A key observation: in sparse scanning scenarios, mesh-based texture mapping combined with differentiable rendering outperforms purely implicit or Gaussian-based methods—an important finding for the community.
- The diffuse+specular texture parameterization is simple yet effective; a minimal MLP (2 layers, 32 units) suffices to model view-dependent effects.
- High practical value: 360° cameras are already widely used in real estate scanning and similar applications, making this method directly applicable to such use cases.

## Limitations & Future Work

- The current pipeline relies on manually defined image pairs (i.e., knowing which images belong to the same room); automatic image retrieval is not yet implemented.
- Cubemap projection combined with neural SDF-based geometry reconstruction degrades in quality as scene scale increases beyond a certain range.
- No dedicated monocular depth and normal estimation model exists for ERP images; the pipeline currently relies on converting to cubemaps and applying perspective-view priors.
- Texture optimization is trained for a fixed 7,000 steps; longer training or adaptive scheduling may yield further improvements.

## Related Work & Insights

- COLMAP/OpenMVG: Classical SfM frameworks with limited support for panoramic cameras and low registration rates in large-scale indoor scenes.
- EDM: The first dense matching method for 360° images, serving as the feature matching backbone for this paper's SfM pipeline.
- DebSDF: A neural SDF method leveraging monocular geometric priors, more robust than alternatives in sparse scenes.
- TMO: A pioneering texture map optimization method, but limited to diffuse modeling; this paper extends it to diffuse+specular.
- Insight: In data-sparse scenarios, a hybrid approach of classical methods plus neural fine-tuning may be more practical than end-to-end neural methods.

## Rating

- Novelty: ⭐⭐⭐⭐ (primarily system-level integration innovation; individual components are moderately novel but the overall integration is complete)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (two large-scale datasets, comprehensive evaluation of SfM and rendering, extensive multi-method comparison)
- Writing Quality: ⭐⭐⭐⭐ (pipeline description is clear, though some details are dispersed across supplementary material)
- Value: ⭐⭐⭐⭐⭐ (fills a critical gap in the panoramic camera indoor mapping toolchain with high practical applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](../../CVPR2026/3d_vision/indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)
- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
