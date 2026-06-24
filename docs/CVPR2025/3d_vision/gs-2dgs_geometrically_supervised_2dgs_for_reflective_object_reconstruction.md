---
title: >-
  [Paper Note] GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction
description: >-
  [CVPR 2025][3D Vision][2D Gaussian Splatting] By introducing depth/normal pseudo-label supervision from foundation models (Marigold + Depth Pro) and a physically-based rendering (PBR) pipeline with deferred shading on top of 2DGS, this approach significantly outperforms existing GS methods and matches the performance of SDF methods on reflective object reconstruction while being an order of magnitude faster.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "reflective object"
  - "PBR"
  - "deferred shading"
  - "foundation model"
  - "normal estimation"
date: 2026-05-08
content_hash: f8217c8b7ee25418
---

# GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2506.13110](https://arxiv.org/abs/2506.13110)  
**Code**: [GitHub](https://github.com/hirotong/GS2DGS)  
**Area**: 3D Vision  
**Keywords**: 2D Gaussian Splatting, reflective object, PBR, deferred shading, foundation model, normal estimation

## TL;DR

By introducing depth/normal pseudo-label supervision from foundation models (Marigold + Depth Pro) and a physically-based rendering (PBR) pipeline with deferred shading on top of 2DGS, this approach significantly outperforms existing GS methods and matches the performance of SDF methods on reflective object reconstruction while being an order of magnitude faster.

## Background & Motivation

**Background**: 3D modeling of highly reflective objects is a long-standing challenge since specular reflections heavily depend on the viewpoint, violating the multi-view consistency assumption. SDF-based methods (such as NeRO, TensoSDF) can extract high-quality meshes but require hours of training; 3DGS achieves fast rendering but yields noisy surface extraction.

**Limitations of Prior Work**:
- GS-based methods (e.g., GShader, GS-IR, R3DG) only partially solve the rendering/relighting problem, while geometric reconstruction remains coarse.
- The appearance of reflective objects is jointly determined by surface properties (material + geometry) and environmental illumination, which is a highly ill-posed problem.
- Merely combining PBR and geometric constraints is insufficient, as normals and depth of reflective surfaces cannot be reliably estimated from multi-view stereo.

**Key Challenge**: The need to simultaneously solve geometric estimation and material/lighting decomposition, despite reflective surfaces rendering traditional multi-view methods ineffective.

**Key Insight**: Leveraging foundation models to predict normals and depth from single views. Since these models are trained on massive datasets, they do not rely on multi-view consistency and are therefore insensitive to reflective surfaces.

## Method

### Overall Architecture

A two-stage training workflow:
1. **Stage 1 (30K iter)**: Leverages normal loss $\mathcal{L}_n$ and depth loss $\mathcal{L}_d$ from foundation models on top of original 2DGS to optimize the geometry.
2. **Stage 2 (10K iter)**: Enables the PBR pipeline, assigning albedo, metallic, and roughness parameters to each 2D Gaussian, and jointly optimizes geometry, material, and environment lighting.

### Key Designs

**1. Geometric Supervision from Foundation Models**
- **Function**: Uses Marigold (for normal estimation) and Depth Pro (for depth estimation) to predict pseudo-GT normals $\tilde{N}$ and depth $\tilde{D}$ for each input image.
- **Normal Loss**: $\mathcal{L}_n = \|\hat{N} - \tilde{N}\|_1 + (1 - \hat{N}^T \tilde{N})$ (a combination of L1 and cosine similarity).
- **Depth Loss**: Scale-invariant depth loss, aligning rendered depth with predicted depth by solving scale $\omega$ and shift $b$ via least squares.
- **Design Motivation**: Foundation models infer geometry from single views based on prior knowledge gained from massive training datasets, making them immune to the effects of reflective surfaces.

**2. Deferred Shading**
- **Function**: Divides rendering into a geometry pass (rendering G-buffer: depth, normals, and PBR parameters) and a shading pass (PBR shading based on G-buffer).
- **Mechanism**: 
    - Forward Shading calculates radiance and performs alpha-blending for each Gaussian independently → varying normals of multiple Gaussians along a ray lead to inaccurate shading.
    - Deferred Shading performs shading only once at the final synthesized surface point → yielding more accurate normals and positions.
- **Design Motivation**: For reflective objects, accurate normal directions of the shading points are crucial. Deferred shading also reduces computational load (shading once instead of per-Gaussian).

**3. PBR + Environment Lighting Decomposition**
- **Function**: Decomposes the rendering equation into diffuse and specular terms based on the Cook-Torrance BRDF.
- **Mechanism**: 
    - Each 2D Gaussian learns three PBR parameters: albedo $\mathbf{a}$, metallic $m$, and roughness $\rho$.
    - A learnable HDR cube map represents the environment lighting.
    - The specular term is approximated using the split-sum method.
- **Design Motivation**: PBR decomposition separates material properties from lighting, enabling relighting.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{GS} + \lambda_n \mathcal{L}_n + \lambda_d \mathcal{L}_d + \lambda_{light} \mathcal{L}_{light} + \lambda_{pbr} \mathcal{L}_{pbr}$$

- $\mathcal{L}_{GS}$: Original 2DGS loss (RGB reconstruction + normal consistency).
- $\mathcal{L}_{light}$: Natural lighting regularization $\|\mathbf{L} - \bar{\mathbf{L}}\|^2$ (constraint on the mean of three channels).
- $\mathcal{L}_{pbr}$: Smoothness regularization for PBR parameters $\|\nabla \mathbf{X}\| \exp(-\|\nabla \mathbf{C}_{gt}\|)$.
- Weights: $\lambda_n=0.5$, $\lambda_d=0.05$, $\lambda_{light}=0.002$.

## Key Experimental Results

### Main Results (Glossy Blender Reconstruction Quality Chamfer-L1↓)

| Method | Type | Average Chamfer-L1 | Training Time |
|---|---|---|---|
| NeRO | SDF | 0.0042 | 12h |
| TensoSDF | SDF | 0.0106 | 6h |
| GShader | GS | 0.0169 | 0.5h |
| R3DG | GS | 0.0303 | 1h |
| GS-IR | GS | 0.0553 | 0.5h |
| **GS-2DGS (Ours)** | **GS** | **0.0068** | **0.7h** |

Ours is the best among GS methods and close to the SDF SOTA (NeRO 0.0042) while being 17x faster to train.

### Relighting Quality (Glossy Blender PSNR↑/SSIM↑)

| Method | PSNR | SSIM | FPS |
|---|---|---|---|
| GShader | 14.96 | 0.811 | 50 |
| GS-IR | 17.11 | 0.811 | 214 |
| R3DG | 19.19 | 0.837 | 1.5 |
| **Ours** | **19.56** | **0.856** | **160** |

### Ablation Study

| Configuration | Chamfer-L1↓ | PSNR↑ |
|---|---|---|
| 2DGS baseline | 0.0481 | 26.23 |
| + Geometric supervision | 0.0084 | 25.52 |
| + PBR | 0.0074 | 25.86 |
| + Deferred Shading (Full) | **0.0068** | **26.76** |

### Key Findings

1. **Geometric supervision contributes the most**: Chamfer-L1 decreases from 0.0481 to 0.0084 (an 82% reduction), serving as the core of the performance improvement.
2. **Deferred shading improves environment map estimation**: Compared to forward shading, deferred shading estimates environment lighting more accurately, boosting PSNR by 0.9 dB.
3. **PBR balances reconstruction and rendering**: Adding PBR further decreases Chamfer-L1 while recovering PSNR.
4. **Practical training efficiency**: The total 40K iterations take about 42 minutes, which is far faster than the 6-12 hours required by SDF-based methods.

## Highlights & Insights

- Leveraging foundation models to remedy the blind spots of geometric estimation on reflective surfaces is an elegant and generalizable idea.
- Deferred shading is introduced to 2DGS for reflective objects for the first time, backed by clear theoretical analysis.
- The two-stage training strategy balances geometric accuracy and material decomposition.
- This is the first time a GS-based method has approached the reconstruction quality of SDF-based methods while preserving real-time rendering speeds.

## Limitations & Future Work

- There remains a small gap compared to SDF (e.g., NeRO), as GS lacks inherent geometric smoothness assumptions.
- Errors in depth/normals predicted by foundation models may still exist and could introduce hallucinations.
- Transparent objects or subsurface scattering are not handled.
- The model is only verified on object-level datasets and has not been extended to scene-level scenarios.
- Relying on two external foundation models increases preprocessing time.

## Related Work & Insights

- NeRO and TensoSDF established the baseline for reflective object reconstruction using SDF + PBR.
- R3DG proposed point-based ray tracing but suffers from slow inference (1.5 FPS).
- The emergence of foundation models like Marigold and Depth Pro provides a new signal source for geometric supervision.
- Insight: Foundation models acting as pseudo-GT providers will play an active role in resolving more 3D problems.

## Rating

⭐⭐⭐⭐ — Well-designed methodology with remarkable performance, representing a significant advancement in GS reflective object reconstruction. The combination of foundation models and deferred shading effectively addresses the core issue with high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GS-ASM: 2DGS-Supervised Active Stereo Matching](../../CVPR2026/3d_vision/gs-asm_2dgs-supervised_active_stereo_matching.md)
- [\[CVPR 2025\] Ref-GS: Directional Factorization for 2D Gaussian Splatting](ref-gs_directional_factorization_for_2d_gaussian_splatting.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
