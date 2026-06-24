---
title: >-
  [Paper Note] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation
description: >-
  [ECCV 2024][3D Vision][NeRF] Proposes Mesh2NeRF, which directly constructs GT radiance fields from textured meshes via analytical solutions, modeling the density field with an occupancy function and the color field with a reflection model, providing accurate 3D point-wise supervision for NeRF representation and generation tasks.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF"
  - "Mesh Supervision"
  - "Radiance Field Generation"
  - "3D Generation"
  - "Diffusion Models"
date: 2026-05-08
content_hash: d0bff5c7dd1c145c
---

# Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.19319](https://arxiv.org/abs/2403.19319)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: NeRF, Mesh Supervision, Radiance Field Generation, 3D Generation, Diffusion Models

## TL;DR

Proposes Mesh2NeRF, which directly constructs GT radiance fields from textured meshes via analytical solutions, modeling the density field with an occupancy function and the color field with a reflection model, providing accurate 3D point-wise supervision for NeRF representation and generation tasks.

## Background & Motivation

**Background**: NeRF has been widely used as a representation for 3D generation, but training generative NeRFs requires GT radiance field samples. The current way to obtain GT is to render multi-view images from a mesh and then fit them with NeRF.

**Limitations of Prior Work**: 
   - The pipeline of multi-view rendering $\rightarrow$ NeRF fitting is redundant and introduces information loss (occlusions, underfitting).
   - The pixel-level rendering loss of NeRF is a weak supervision: the color of each pixel must simultaneously supervise the density and color of all sampled points along the ray.
   - When view coverage is uneven, this weak supervision easily leads to inaccurate radiance fields.

**Key Challenge**: Despite possessing precise 3D mesh data, taking a detour through 2D rendering to reconstruct back to 3D wastes the precise geometric information of the mesh and introduces unnecessary errors.

**Goal**: Skip the multi-view rendering phase and directly derive the density and color values of the radiance field analytically from the textured mesh to serve as point-wise 3D supervision for NeRF.

**Key Insight**: Starting from the mathematical derivation of volume rendering, approximate the Dirac delta density distribution using a top-hat function to encode the mesh surface as an occupancy-based alpha value.

**Core Idea**: Convert meshes into radiance fields using analytical solutions to achieve direct 3D supervision per sampled point.

## Method

### Overall Architecture

Mesh2NeRF consists of two parts: (1) analytically deriving the radiance field (density field + color field) from the textured mesh; (2) using the derived radiance field as a direct supervision signal to train single-scene NeRF fitting or NeRF generative models.

### Key Designs

1. **Density Field Modeling — Occupancy-based Alpha**:

    - Ideally, the density of the mesh surface is a Dirac delta function (infinite density only at the surface), but this is infeasible for neural networks.
    - Approximate the Dirac delta function using a top-hat function $\Delta_n(t)$:

    $\Delta_n(t) = \begin{cases} n/2, & \text{if } |t| < 1/n \\ 0, & \text{otherwise} \end{cases}$

    - Key insight: instead of directly using density $\sigma$ (the value of which tends to infinity), use the alpha value $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$.
    - For large $n$, the alpha value can be simplified to a distance-based occupancy function:

    $\alpha = \begin{cases} 1, & \text{if } d < h \\ 0, & \text{otherwise} \end{cases}$

      where $d$ is the distance to the mesh surface, and $h$ is the surface half-thickness.
    - Physical meaning: The first sample point $i_m$ on the ray that intersects the surface has $\alpha_{i_m}=1$, with its color corresponding to the color of the textured mesh intersection point, while other points have $\alpha=0$.

2. **Color Field Modeling — BRDF Reflection Model**:

    - The color of all sample points on a ray is defined as the color $\mathbf{c}_i$ of the first intersection point of the ray with the mesh surface.
    - Use the Phong reflection model to calculate view-dependent colors (can also be replaced with any BRDF).
    - Takes into account the combined influence of mesh geometry (normals), texture, and ambient lighting.
    - The volume rendering result is $\hat{C}(\mathbf{y}) = \alpha_{i_m} \mathbf{c}_{i_m}$, which is highly consistent with the GT mesh rendering.

3. **Mesh2NeRF as NeRF Supervision**:

    - Provides point-wise 3D supervision, instead of traditional pixel-wise 2D supervision.
    - Alpha loss: $\mathcal{L}_{alpha} = \sum_{i=1}^{N} |\hat{\alpha}_i - \alpha_i|^2$
    - Color loss: $\mathcal{L}_{color} = \sum_{i=1}^{N} \|\hat{\mathbf{c}}_i - \mathbf{c}_i\|_2^2$
    - Integral loss (optional): $\mathcal{L}_{integral}$ constrains the ray color integration.
    - Total loss: $\mathcal{L} = \mathcal{L}_{alpha} + w_{color}\mathcal{L}_{color} + w_{integral}\mathcal{L}_{integral}$

4. **Efficient Sampling Strategy**:

    - Use the BVH acceleration structure of the Embree library for ray-mesh intersection.
    - For rays intersecting the mesh: perform stratified sampling in the empty scene space and inside the narrow band near the surface (within distance $h$).
    - For rays not intersecting the mesh: sample randomly along the ray.

### Application in NeRF Generation Tasks

- Based on the SSDNeRF framework (triplane NeRF auto-decoder + triplane latent diffusion model).
- Replace the rendering loss $\mathcal{L}_{rend}$ in SSDNeRF with the Mesh2NeRF loss $\mathcal{L}$.
- Use direct 3D supervision of Mesh2NeRF during the training phase, while still using rendering loss during the conditional generation inference phase (since there is no mesh during inference).
- Total training objective: $\mathcal{L}_{ssdnerf} = w_{rend}\mathcal{L} + w_{diff}\mathcal{L}_{diff}$

### Loss & Training

- Single-scene fitting: Supports three encoding methods NeRF/TensoRF/Instant NGP, corresponding to Mesh2NeRF NeRF/TensoRF/NGP respectively.
- Generation tasks: Uses the official SSDNeRF implementation, keeping the same number of views and training settings, only replacing the loss function to ensure a fair comparison.
- No extra ray sampling overhead.

## Key Experimental Results

### Single-scene Fitting — ABO & Poly Haven

| Method | ABO PSNR↑ | ABO SSIM↑ | ABO LPIPS↓ | Poly Haven PSNR↑ |
|------|-----------|-----------|------------|-------------------|
| NeRF | 25.09 | 0.882 | 0.137 | 21.24 |
| TensoRF | 31.33 | 0.944 | 0.032 | 23.32 |
| Instant NGP | 30.16 | 0.928 | 0.039 | 24.39 |
| Mesh2NeRF NeRF | 32.40 | 0.942 | 0.044 | 22.75 |
| Mesh2NeRF TensoRF | 32.00 | 0.957 | 0.024 | 23.97 |
| **Mesh2NeRF NGP** | **33.28** | **0.969** | **0.018** | **25.30** |

### Conditional Generation — ShapeNet Cars

| Method | 1-view PSNR↑ | 2-view PSNR↑ | 3-view PSNR↑ | 4-view PSNR↑ |
|------|-------------|-------------|-------------|-------------|
| SSDNeRF | 21.09 | 24.67 | 25.71 | 26.54 |
| **Ours** | **21.78** | **24.98** | **25.89** | **26.51** |

ShapeNet Chairs achieves the most significant improvement under the 2-view condition: PSNR increases from 19.65$\rightarrow$22.22 (+2.57dB).

### Key Findings

- Single-scene fitting: Mesh2NeRF NGP improves upon Instant NGP by $+3.12$ dB PSNR (on the ABO dataset), proving that point-wise 3D supervision is far superior to pixel-level 2D supervision.
- Conditional generation: The advantages are especially pronounced on unconventional shapes and real data (KITTI)—where SSDNeRF fails when the car body color is close to the background, Mesh2NeRF can still reconstruct it.
- Unconditional generation (Objaverse Mugs): When extracting meshes from NeRF, the geometric quality of Mesh2NeRF is far superior to SSDNeRF (e.g., whether the mug is sealed/hollow).
- 3D supervision in generative models learns correct geometric priors better than 2D supervision.

## Highlights & Insights

- **From First Principles**: Rather than heuristic designs, the conversion from mesh to radiance field is mathematically derived from volume rendering formulas, providing theoretical guarantees.
- **Eliminating Redundancy**: Eliminates the indirect path of "mesh $\rightarrow$ render $\rightarrow$ fit NeRF" and directly maps "mesh $\rightarrow$ radiance field", reducing information loss.
- **Plug-and-Play**: Can replace the loss function in any NeRF method that uses rendering loss, and is compatible with various encodings such as NeRF/TensoRF/Instant NGP.
- **3D Point-wise Supervision vs 2D Pixel Supervision**: Experiments extensively demonstrate the superiority of point-wise supervision, especially under occlusion and sparse viewpoints.

## Limitations & Future Work

1. Color modeling uses the Phong model, which bakes illumination into the appearance, and does not support relighting.
2. The sampling strategy relies on the ray casting of rendered images and does not fully utilize the known mesh geometry for more efficient sampling.
3. The generation task is only validated on synthetic datasets such as ShapeNet/Objaverse, lacking large-scale validation on real-world scenes.
4. Integration with more advanced 3D generation frameworks (such as 3DGS-based methods) has not been explored.
5. The choice of the surface thickness parameter $h$ affects the results and requires tuning.

## Related Work & Insights

- **NeRF2Mesh** (reverse direction): Reconstructs meshes from NeRF $\leftrightarrow$ Mesh2NeRF constructs NeRF from meshes; they are inverse processes of each other.
- **SSDNeRF**: Serves as the generative framework baseline; the plug-and-play nature of Mesh2NeRF is validated here.
- **depth-supervised NeRF**: Depth priors provide a weak form of geometric supervision $\rightarrow$ Mesh2NeRF provides a strong form (complete occupancy + color).
- **Insight**: When accurate 3D data is available, its precision should be fully utilized for direct supervision instead of dimensionality reduction to 2D for indirect learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (An elegant approach that establishes an analytical mapping from meshes to radiance fields via mathematical derivation)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Validated across single-scene, conditional generation, unconditional generation, and real-world data)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear mathematical derivation with a logical causal chain)
- **Value**: ⭐⭐⭐⭐ (Provides a new paradigm for NeRF applications using mesh-based 3D datasets)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] Lagrangian Hashing for Compressed Neural Field Representations](lagrangian_hashing_for_compressed_neural_field_representations.md)
- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)

</div>

<!-- RELATED:END -->
