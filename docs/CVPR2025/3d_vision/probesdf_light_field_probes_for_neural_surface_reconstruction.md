---
title: >-
  [Paper Note] ProbeSDF: Light Field Probes for Neural Surface Reconstruction
description: >-
  [CVPR 2025][3D Vision][Neural Surface Reconstruction] ProbeSDF redesigns the appearance model of SDF-based neural surface reconstruction by decoupling and storing spatial and angular features in voxel grids of different resolutions. Utilizing minimal parameters (4 per voxel) and a tiny MLP, it achieves superior geometry and image quality. Training takes only 1-2 minutes and supports real-time rendering.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Neural Surface Reconstruction"
  - "Light Field Probes"
  - "SDF"
  - "Appearance Modeling"
  - "Spherical Harmonics"
date: 2026-05-08
content_hash: 0e52473ca5564a3b
---

# ProbeSDF: Light Field Probes for Neural Surface Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2412.10084](https://arxiv.org/abs/2412.10084)  
**Code**: [https://gitlab.inria.fr/projects-morpheo/ProbeSDF](https://gitlab.inria.fr/projects-morpheo/ProbeSDF)  
**Area**: 3D Vision  
**Keywords**: Neural Surface Reconstruction, Light Field Probes, SDF, Appearance Modeling, Spherical Harmonics

## TL;DR

ProbeSDF redesigns the appearance model of SDF-based neural surface reconstruction by decoupling and storing spatial and angular features in voxel grids of different resolutions. Utilizing minimal parameters (4 per voxel) and a tiny MLP, it achieves superior geometry and image quality. Training takes only 1-2 minutes and supports real-time rendering.

## Background & Motivation

**Background**: SDF-based neural surface reconstruction methods (e.g., NeuS, Voxurf, NeuS2) have become the mainstream approach to obtaining high-quality 3D models from multi-view images. These methods typically use an MLP to jointly decode spatial features and view directions into colors.

**Limitations of Prior Work**: Existing methods co-locate color encoding and angular features in the same resolution grid. However, spatial textures (albedo) require high-frequency details, while angular variations (lighting reflections) vary slowly over space. This co-location leads to: (1) spatial features needing extra dimensions to encode lighting information, which increases the grid capacity; (2) requiring larger MLPs to decode, reducing efficiency; and (3) local lighting effects being difficult to model accurately with a global MLP.

**Key Challenge**: Spatial features require high-resolution encoding for texture details, while angular features require spatial smoothness (slowly changing lighting). Hybrid storage is memory-wasteful and overburdens the MLP.

**Goal**: Decouple the spatial and angular components of the radiance field into grids of different resolutions, achieving better reconstruction quality and faster speeds with minimal parameters and a tiny MLP.

**Key Insight**: Drawing inspiration from the concept of "light field probes" in real-time rendering, where incoming radiance is precomputed at sparse 3D locations and interpolated between probes during rendering.

**Core Idea**: High-resolution grids store spatial features (texture), while a probe grid at 1/16 resolution stores angular features (lighting). The two are decoded via a tiny MLP integrated with a Fresnel view-dependency term.

## Method

### Overall Architecture

The 3D scene is divided into 16x16x16 tiles. Within each tile, high-resolution spatial features are stored using planar decomposition, while light field probes (spherical harmonic coefficients) are stored at the 8 corners of the tile to encode angular features. Given a spatial point and a view direction, spatial features are retrieved via planar interpolation, and angular features are obtained by tri-linearly interpolating the spherical harmonics of the 8 probes evaluated in the reflection direction. Both features, along with a Fresnel term, are fed into a 2-layer, 32-neuron MLP to output color. The SDF is stored directly in a grid and volume-rendered using NeuS formulations.

### Key Designs

1. **Decoupled Spatial-Angular Feature Parameterization**:

    - **Function**: Encodes texture and lighting/reflection information at different resolutions, respectively.
    - **Mechanism**: Spatial features within a tile obtain high-resolution encoding using VM decomposition (three sets of 16x16 orthogonal planes). Angular features are tri-linearly interpolated from the probes at the 8 corner vertices; its spatial resolution is only 1/16 but is sufficient to encode slowly varying lighting. Color is decoded from both features and the Fresnel term by a tiny MLP.
    - **Design Motivation**: Probes take up only 1/6 of the memory of spatial features. The MLP does not need to decrypt lighting information from compressed features, thus allowing it to be extremely small (2 layers of 32 neurons).

2. **Light Field Probes**:

    - **Function**: Encodes spatially varying angular information on a low-resolution grid.
    - **Mechanism**: Each probe stores a vector of spherical harmonic (SH) coefficients. The angular feature is the SH evaluation along the reflection direction using the tri-linearly interpolated coefficients. Key difference: Unlike 3DGS, which uses SH to encode RGB directly, the probes encode abstract features processed non-linearly by the MLP. Sampling in the reflection direction instead of the view direction allows surface normals to participate in appearance modeling.
    - **Design Motivation**: Spherical harmonics pre-structure the angular information before feeding it into the MLP, reducing the learning difficulty. Spatial interpolation of probes naturally provides smooth transitions in lighting.

3. **Fresnel View-Dependency Term**:

    - **Function**: Handles the physical effect of reflection enhancement at grazing angles.
    - **Mechanism**: Feeds $(1 - \mathbf{n} \cdot \mathbf{v})^k$ ($k=0...5$) as an extra input to the MLP, allowing the MLP to learn the coefficients of the Fresnel polynomial without explicitly estimating base reflectivity.
    - **Design Motivation**: Two different views can yield the same reflection direction but different incident angles; the Fresnel term resolves this ambiguity. Removing the Fresnel term increases Chamfer by 13%.

### Loss & Training

6 losses: SDF smoothness, Eikonal regularization, normal smoothness, feature smoothness, probe smoothness, and photometric loss. Training progresses from coarse to fine, gradually increasing resolution and SH degree. An optional camera bias vector can be trained per camera for certain datasets.

## Key Experimental Results

### Main Results

MVMannequins (14 mannequins, 68 cameras):

| Method | Chamfer (mm) | PSNR | Training Time |
|------|-------------|------|---------|
| MMH | 1.14 | 36.33 | 2-3min |
| Voxurf | 1.59 | 35.51 | 15min |
| NeuS2 | 2.13 | 34.22 | 5min |
| 2DGS | 3.35 | 34.89 | >1h |
| **ProbeSDF** | **1.04** | **36.81** | **1min** |

DTU:

| Method | Chamfer | PSNR | Training Time |
|------|---------|------|---------|
| Voxurf | 0.73 | 37.08 | 15min |
| NeuS2 | 0.80 | 36.00 | 5min |
| 2DGS | 0.76 | 36.03 | 12min |
| **ProbeSDF (4,4,4)** | **0.68** | 37.03 | **150s** |

### Ablation Study

| Configuration | Chamfer (mm) | PSNR | Description |
|------|-------------|------|------|
| (4,4,4) Full | 1.04 | 36.81 | MVMann full config |
| w/o probes smoothing | 1.09 | 36.91 | Remove probe smoothing |
| w/o Fresnel | 1.18 | 36.79 | Remove Fresnel term |
| (4,4,1) BMVS | 2.58 | 34.44 | Constant SH only |
| (4,4,4) BMVS | 2.37 | 35.19 | Order-4 SH |
| (8,8,4) BMVS | 2.22 | 35.89 | More feature dimensions |

### Key Findings

- 1-minute training on MVMannequins outperforms all baselines, being 2-3x faster than MMH.
- Only 4 parameters are needed per voxel, resulting in a 30MB model (vs 500MB for Voxurf).
- The Fresnel term has a significant impact on geometric quality (Chamfer decreases from 1.18 to 1.04 when included).
- Spherical harmonics degree $l=4$ is sufficient for materials with common roughness.
- On the high-resolution ActorsHQ dataset, it achieves 37.48 dB PSNR in only 4 minutes.
- Rendering at 200-400 Hz supports real-time applications.

## Highlights & Insights

- **Extreme Parameter Efficiency**: 4 parameters per voxel; probes take 1/6 of the spatial feature memory; 30MB model. This stems from a deep physical understanding of light fields.
- **Tiny MLP Design**: A 2-layer, 32-neuron MLP is fused into a single CUDA kernel, as spatial and angular information are already well-structured before entering the MLP.
- **Concept Transfer of Light Field Probes from Rendering to Reconstruction**: Reverses the concept of precomputed lighting in real-time rendering—learning probe parameters directly from observed images.

## Limitations & Future Work

- Lighting is encoded locally in space, leading to limited extrapolation capability—shading artifacts may appear in regions with sparse camera coverage.
- Does not support lighting/material decomposition; relighting or material editing is not possible.
- Assumes isotropic materials.
- No direct comparison with 3DGS regarding NVS quality.

## Related Work & Insights

- **vs Voxurf**: Directly improves the appearance model, reducing training from 15 minutes to 150 seconds while achieving better quality.
- **vs NeuS2**: NeuS2 uses hash grid + large MLP, which takes 5 minutes of training but lags in quality, whereas explicit probes are more efficient.
- **vs 3DGS/Plenoxels**: Directly encoding RGB with spherical harmonics requires 48 parameters, whereas ProbeSDF only requires 4 parameters to encode abstract features.
- **vs NeRFactor/TensoIR**: Full decomposition requires 5+ hours of training, whereas ProbeSDF achieves better fitting in minutes.

## Rating

- Novelty: ⭐⭐⭐⭐ Spatial-angular decoupling is intuitive and physically motivated, and the transfer of light field probes is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets covering objects and humans, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-refined physical motivations, intuitive diagrams, and interpretable parameters.
- Value: ⭐⭐⭐⭐⭐ Fast training, fast rendering, high quality, and low parameters—suitable as a drop-in replacement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](../../ICLR2026/3d_vision/lito_surface_light_field_tokenization.md)
- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[CVPR 2025\] 4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)
- [\[CVPR 2025\] GauSTAR: Gaussian Surface Tracking and Reconstruction](gaustar_gaussian_surface_tracking_and_reconstruction.md)

</div>

<!-- RELATED:END -->
