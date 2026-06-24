---
title: >-
  [Paper Note] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering
description: >-
  [ICCV 2025][3D Vision][Inverse Rendering] This paper proposes GeoSplatting, which differentiably generates surface-aligned Gaussians from an optimizable explicit mesh to provide accurate geometric guidance for 3DGS, achieving state-of-the-art inverse rendering performance (material–lighting decomposition) with training times of only 10–15 minutes.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Inverse Rendering"
  - "3D Gaussian Splatting"
  - "Material Decomposition"
  - "Environment Lighting"
  - "Mesh-Guided"
date: 2026-05-08
content_hash: 91fc5e37d4a0b6d9
---

# GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering

**Conference**: ICCV 2025
**arXiv**: [2410.24204](https://arxiv.org/abs/2410.24204)  
**Code**: [Project Page](https://pku-vcl-geometry.github.io/GeoSplatting/)  
**Area**: 3D Vision
**Keywords**: Inverse Rendering, 3D Gaussian Splatting, Material Decomposition, Environment Lighting, Mesh-Guided

## TL;DR

This paper proposes GeoSplatting, which differentiably generates surface-aligned Gaussians from an optimizable explicit mesh to provide accurate geometric guidance for 3DGS, achieving state-of-the-art inverse rendering performance (material–lighting decomposition) with training times of only 10–15 minutes.

## Background & Motivation

Inverse rendering aims to recover material properties (albedo, roughness, metalness) and environment lighting from multi-view images, which is critical for downstream applications such as games, film, and AR/VR. The core challenge lies in accurately modeling light transport.

Existing methods can be categorized by their scene representation:

**Implicit field methods** (TensoIR, NeRO): Accurate normals are obtained via SDF gradients, making them suitable for inverse rendering; however, dense sampling in volume rendering leads to training times of several hours.

**Mesh-based methods** (NVdiffrec, NVdiffrecmc): Naturally support PBR pipelines and ray tracing, but differentiable rendering only produces gradients at triangle edges, making optimization difficult.

**3DGS-based methods** (R3DG, GS-IR, GS-Shader): Offer high rendering efficiency but suffer from two fundamental limitations:
   - **Inaccurate normals**: Gaussian normals are approximated through implicit geometric constraints (e.g., depth-normal regularization), yielding insufficient accuracy.
   - **Non-opaque surfaces**: Gaussians are inherently semi-transparent, preventing precise definition of light–surface intersection points.

The authors' core insight is that accurate light transport modeling requires two conditions: (1) precise normal directions (determining light propagation) and (2) opaque surfaces (defining light–surface intersection points). **Explicit meshes naturally satisfy both conditions.** By "bridging" 3DGS with an explicit mesh, one can simultaneously leverage the geometric accuracy of meshes and the rendering efficiency of 3DGS.

## Method

### Overall Architecture

The GeoSplatting pipeline proceeds as follows: scalar field $\boldsymbol{\zeta}$ → FlexiCubes triangle mesh extraction $\mathbf{M}$ → MGadapter generating surface-aligned Gaussians → multi-resolution hash grid querying PBR attributes → PBR rendering equation computing per-Gaussian PBR colors → 3DGS rasterization producing the final image. The entire pipeline is fully differentiable and supports end-to-end training.

### Key Designs

1. **Mesh-to-Gaussian Adapter (MGadapter)**:

    - **Function**: Differentiably generates structured Gaussians from each face of the triangle mesh.
    - **Mechanism**: For each triangular face $\mathbf{P}$, $K=6$ Gaussians are generated and placed according to predefined patterns in barycentric coordinate space. Positions $\boldsymbol{\mu}_i$ and normals $\mathbf{n}_i$ are computed via barycentric interpolation, while scales $\mathbf{S}_i$ and rotations $\mathbf{R}_i$ are determined by the orientation and shape of the triangle. Opacity is fixed to 1 to reflect the opaque nature of the mesh surface.
    $$\{(\boldsymbol{\mu}_i, \mathbf{S}_i, \mathbf{R}_i, \mathbf{n}_i) \mid i=1,\ldots,K\} = \mathcal{T}(\mathbf{P})$$
    - **Design Motivation**: Since the shape parameters of each Gaussian are fully determined by the corresponding triangle, shape consistency between the mesh and 3DGS is guaranteed—mesh normals equal Gaussian normals, eliminating the need for additional normal learning or regularization.

2. **Physics-Based Gaussian Rendering**:

    - **Function**: Computes physically interpretable PBR colors for each Gaussian.
    - **Mechanism**: The spherical harmonics of vanilla 3DGS are replaced by a PBR rendering equation based on the GGX microfacet model:
    $$\mathbf{L}_o(\mathbf{x}, \boldsymbol{\omega}_o) = \int_{\mathcal{H}^2} \mathbf{f}_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \mathbf{L}_i(\mathbf{x}, \boldsymbol{\omega}_i) |\mathbf{n} \cdot \boldsymbol{\omega}_i| \mathrm{d}\boldsymbol{\omega}_i$$
    Material attributes (albedo $\mathbf{a}$, roughness $\rho$, metalness $m$) are queried from multi-resolution hash grids $\mathcal{E}_d, \mathcal{E}_s$. The rendering equation integral is evaluated via Monte Carlo sampling.
    - **Design Motivation**: Physically interpretable material representations support realistic relighting, which spherical harmonics cannot achieve.

3. **Mesh-Based Efficient Light Transport Modeling**:

    - **Function**: Leverages the explicit mesh for efficient self-occlusion evaluation and indirect illumination modeling.
    - **Mechanism**: Incident light is decomposed into direct $\mathbf{L}_{\text{dir}}$ and indirect $\mathbf{L}_{\text{ind}}$ components, weighted by an occlusion factor $O(\mathbf{x}, \boldsymbol{\omega}_i)$. The key innovation is replacing the continuous Gaussian occlusion $O_{\text{3dgs}} \in [0,1]$ with binary mesh occlusion $O_{\text{mesh}} \in \{0,1\}$, enabling efficient occlusion evaluation via BVH-accelerated mesh ray tracing.
    $$\mathbf{L}_i(\mathbf{x}, \boldsymbol{\omega}_i) = (1-O)\mathbf{L}_{\text{dir}}(\boldsymbol{\omega}_i) + O \cdot \mathbf{L}_{\text{ind}}(\mathbf{x}, \boldsymbol{\omega}_i)$$
    - **Design Motivation**: The MGadapter guarantees shape consistency between the mesh and 3DGS, so $O_{\text{mesh}} \approx O_{\text{3dgs}}$, making the substitution introduce negligible error. BVH-accelerated mesh ray tracing is far more efficient than accumulating opacity over 3D Gaussians.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{\text{img}} + \lambda_{\text{entropy}}\mathcal{L}_{\text{entropy}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{light}}\mathcal{L}_{\text{light}}$

where $\mathcal{L}_{\text{img}} = \mathcal{L}_1 + \lambda_{\text{ssim}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$.

**Two-stage training strategy**: The initial stage uses the Split-Sum approximation (no self-occlusion, fast precomputation) as a warm-up; once geometry stabilizes, training switches to Monte Carlo sampling for full light transport modeling. Optionally, Deferred Shading can be applied at the end of training for appearance refinement, improving high-frequency specular effects.

Owing to geometry guidance, additional regularization terms such as dist loss or pseudo depth-normal loss are not required.

## Key Experimental Results

### Main Results

| Dataset | Metric | GeoSplatting | R3DG | TensoIR | NVdiffrecmc | GS-IR |
|--------|------|-------------|------|---------|-------------|-------|
| Synthetic4Relight | Relighting PSNR↑ | **34.10** | 31.00 | 29.94 | 30.23 | 23.81 |
| TensoIR Synthetic | Relighting PSNR↑ | **29.95** | 28.55 | 28.51 | 26.51 | 24.35 |
| TensoIR Synthetic | Albedo PSNR↑ | **29.41** | 28.74 | 28.35 | 27.71 | 26.80 |
| Shiny Blender | NVS PSNR↑ | **31.14** | 28.83 | 27.89 | 28.03 | 27.01 |
| Training Time (min) | - | **14** | ~110 | ~270 | 82 | 20 |

**Normal Quality (MAE↓)**:

| Dataset | GeoSplatting | R3DG | TensoIR | GS-IR | NVdiffrecmc |
|--------|-------------|------|---------|-------|-------------|
| TensoIR Synthetic | **4.08** | 5.45 | 4.10 | 5.41 | 4.81 |
| Shiny Blender | **2.15** | 7.04 | 4.42 | 4.42 | 9.76 |

### Ablation Study

| Configuration | NVS PSNR↑ | Relighting PSNR↑ | Albedo PSNR↑ | Normal MAE↓ |
|------|----------|-----------------|-------------|------------|
| w/o shape alignment | 35.95 | 26.39 | 26.72 | 8.29 |
| w/o appearance refinement | 35.07 | 27.95 | 29.31 | 4.42 |
| w/o occlusion modeling | 35.87 | 27.36 | 27.80 | 6.17 |
| w/o indirect illumination | 36.01 | 28.92 | 29.18 | 4.97 |
| Full model | **36.45** | **29.95** | **29.41** | **4.08** |

### Key Findings

1. Shape alignment (MGadapter) is the single largest contributor to performance—removing it degrades normal MAE from 4.08 to 8.29.
2. Occlusion modeling has limited impact on NVS but is critical for decomposition—albedo PSNR drops from 29.41 to 27.80 without it.
3. GeoSplatting trains in only 14 minutes, an order of magnitude faster than implicit field methods (TensoIR ~270 min, NeRO ~800 min).
4. The advantage is most pronounced on reflective surfaces (Shiny Blender), where normal MAE is 69% lower than R3DG.

## Highlights & Insights

1. **Best of both explicit and implicit representations**: The mesh provides precise geometry, 3DGS provides efficient rendering, and MGadapter seamlessly bridges the two. This hybrid representation outperforms either alone.
2. **No normal learning required**: The method fundamentally eliminates the dependence on normal approximation that plagues prior 3DGS inverse rendering methods, resolving the material decomposition noise caused by inaccurate normals.
3. **14-minute training**: State-of-the-art performance is achieved at the fastest training speed, making the method highly practical for iterative design workflows.
4. The two-stage strategy of Split-Sum warm-up followed by MC sampling balances early-stage stability with late-stage accuracy.

## Limitations & Future Work

1. Reliance on isosurface extraction (FlexiCubes) constrains mesh resolution, making it difficult to handle thin structures and complex geometry.
2. Training requires object masks, limiting direct applicability to unmasked scenes.
3. The level of detail in forward shading (per-Gaussian shading) is bounded by Gaussian density; while Deferred Shading partially alleviates this, it adds pipeline complexity.
4. The method currently operates at the object level; extension to scene-scale reconstruction is an important future direction.

## Related Work & Insights

- **R3DG**: Learns additional normal attributes and regularizes with depth maps → GeoSplatting directly uses mesh normals, achieving higher accuracy.
- **NVdiffrecmc**: Also employs mesh + Monte Carlo rendering → GeoSplatting integrates 3DGS for faster training and improved rendering quality.
- **2DGS/SuGaR**: Geometry-enhanced 3DGS → GeoSplatting extends geometry enhancement to inverse rendering tasks.
- **Insight**: Explicit geometry guidance is the key pathway toward physically accurate rendering with 3DGS.

## Rating

- **Novelty**: ⭐⭐⭐⭐ MGadapter is an elegant bridging design, though the mesh+3DGS hybrid concept is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets covering diffuse, specular, and real-world scenes with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear figures, thorough motivation, and complete technical details.
- **Value**: ⭐⭐⭐⭐⭐ State-of-the-art inverse rendering in 14 minutes offers high practical value; the hybrid representation paradigm is broadly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](../../CVPR2025/3d_vision/svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)
- [\[CVPR 2026\] IR-HGP: Physically-Aware Gaussian Inverse Rendering for High-Illumination Scenes via Generative Priors](../../CVPR2026/3d_vision/ir-hgp_physically-aware_gaussian_inverse_rendering_for_high-illumination_scenes_.md)
- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](../../CVPR2026/3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)

</div>

<!-- RELATED:END -->
