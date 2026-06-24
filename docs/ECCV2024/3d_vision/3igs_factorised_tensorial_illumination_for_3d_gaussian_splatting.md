---
title: >-
  [Paper Note] 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] 3iGS replaces the independently optimized spherical harmonics (SH) coefficients of each Gaussian in 3DGS with a continuous incident illumination field based on tensor decomposition. Combined with learnable BRDF features and a lightweight neural renderer to model the outgoing radiance, it significantly improves the rendering quality of view-dependent effects such as specular reflections while maintaining real-time rendering speeds.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Illumination Modeling"
  - "BRDF"
  - "Tensor Decomposition"
  - "View-Dependent Effects"
date: 2026-05-08
content_hash: 1419ffc2a9055cce
---

# 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org/abs/2408.03753)  
**Authors**: Zhe Jun Tang, Tat-Jen Cham (NTU S-Lab)  
**Code**: [https://github.com/TangZJ/3iGS](https://github.com/TangZJ/3iGS)  
**Area**: 3D Vision / Novel View Synthesis  
**Keywords**: 3D Gaussian Splatting, Illumination Modeling, BRDF, Tensor Decomposition, View-Dependent Effects  

## TL;DR
3iGS replaces the independently optimized spherical harmonics (SH) coefficients of each Gaussian in 3DGS with a continuous incident illumination field based on tensor decomposition. Combined with learnable BRDF features and a lightweight neural renderer to model the outgoing radiance, it significantly improves the rendering quality of view-dependent effects such as specular reflections while maintaining real-time rendering speeds.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the mainstream method for real-time reconstruction and rendering of 3D scenes from multi-view images. It represents the outgoing radiance by independently optimizing spherical harmonic (SH) coefficients for each Gaussian, combined with tile-based rasterization to achieve real-time rendering.

**Limitations of Prior Work**: 3DGS **independently** optimizes SH coefficients for each Gaussian to describe outgoing radiance, completely ignoring light interactions between adjacent Gaussians in the scene: the color of a Gaussian should be affected by light reflected from surrounding objects, but the current framework cannot model such scene-level indirect illumination. This leads to sub-optimal rendering performance in scenes with specular reflections and glossy surfaces, where reflective highlights lack realism and surface colors vary inaccurately with the viewing perspective.

**Key Challenge**: On one hand, there is a desire to accurately model BRDF and illumination like PBR to generate realistic view-dependent effects; on the other hand, accurately estimating physical material parameters (roughness, albedo, metallicity, etc.) from multi-view images is a severely ill-posed inverse rendering problem. Existing methods (such as GaussianShader, which directly predicts Cook-Torrance BRDF parameters) have limited accuracy, which instead introduces errors.

**Goal**: How to introduce scene-level illumination modeling to 3DGS to improve view-dependent effects without relying on accurate physical parameter estimation?

**Key Insight**: Inspired by irradiance volumes and light probes in game engines—which do not solve physical equations precisely but instead precompute/store illumination information in space and quickly query it via interpolation during rendering. Similarly, a continuous illumination field can be learned, allowing each Gaussian to query "incident illumination features" and combine them with its own BRDF features to output color.

**Core Idea**: Utilizing a factorized tensorial continuous illumination field + learnable BRDF features + a neural renderer to replace the independently optimized spherical harmonics coefficients in 3DGS, thereby modeling view-dependent effects without solving inverse rendering.

## Method

### Overall Architecture
The pipeline of 3iGS: Given the 3D position of a Gaussian $\mathbf{x}_i$ as input, it first interpolates the incident illumination feature $L_i$ from the factorized tensorial illumination field $\mathcal{G}_l$. Then, $L_i$, the Gaussian's own BRDF feature $\rho_i$, and the view direction $\omega_o$ (encoded via IDE) are passed to a lightweight MLP $\mathcal{F}$ to output the specular color $\mathbf{c_s}$. Finally, it is linearly added to the diffuse color $\mathbf{c_d}$ to obtain the final outgoing radiance:

$$\mathbf{c}(\omega_o) = \mathbf{c_d} + \mathbf{c_s}(\omega_o)$$

$$\mathcal{F}: \{\rho_i, L_i, \omega_o\} \mapsto \mathbf{c_s}$$

### Key Designs

1. **Factorised Tensorial Illumination Field**:

    - **Function**: Learn a continuous 3D illumination field covering the entire scene, from which each Gaussian can query the incident illumination feature at its position.
    - **Mechanism**: Employing the VM (Vector-Matrix) decomposition of TensoRF, which decomposes the 3D voxel grid of illumination into a compact sum of outer products of vectors and matrices. Given the Gaussian position $\mathbf{x}_i$, the illumination feature $L_i$ is obtained from the factorized tensor via trilinear interpolation:
    $\mathcal{G}_l = \sum_{r=1}^{R_L} \mathbf{A}_{L,r}^{X} \circ \mathbf{b}_{3r-2} + \mathbf{A}_{L,r}^{Y} \circ \mathbf{b}_{3r-1} + \mathbf{A}_{L,r}^{Z} \circ \mathbf{b}_{3r}$
    - **Design Motivation**: (1) A continuous illumination field allows all Gaussians to share scene-level illumination information, breaking the constraints of independent optimization; (2) VM decomposition is extremely compact ($150^3$ voxels, which is 87.5% smaller than the original TensoRF), and querying only requires interpolation, having almost no impact on rendering speed; (3) Shrinking the bounding box and resampling during training further improves efficiency.

2. **Gaussian BRDF Features**:

    - **Function**: Each Gaussian carries a learnable BRDF feature vector $\rho_i$ that describes its surface reflection properties.
    - **Mechanism**: Repurpose the original 16x3 SH coefficients of 3DGS as BRDF feature channels, and add 4 extra parameters (base color + roughness) for IDE view encoding. The key is to **not enforce physical interpretability**—unlike GaussianShader, which predicts physical parameters like roughness/albedo/metallicity, this method treats BRDF features as a set of weights to modulate the incident illumination field within the neural renderer.
    - **Design Motivation**: Inverse rendering estimation of physical material parameters is a severely ill-posed problem, so it is better to let the network learn a set of "generalized BRDF" features. Inspired by the SH convolution theory ($B_{lm} = \Lambda_l \rho_l L_{lm}$), the BRDF essentially acts as a filter of the incident illumination.

3. **Neural Renderer and Shading**:

    - **Function**: A small MLP maps illumination features, BRDF features, and viewing direction to the specular color.
    - **Mechanism**: The viewing direction is encoded using Integrated Directional Encoding (IDE, from Ref-NeRF). IDE requires a roughness parameter to modulate the encoding frequency—preserving low frequencies for rough surfaces and high frequencies for smooth surfaces.
    - **Difference from GaussianShader**: GaussianShader uses a global differentiable environment cube map + analytical GGX BRDF calculation, whereas 3iGS uses a local continuous illumination field + neural network prediction, eliminating the reliance on specific rendering equations.

### Loss & Training
- Identical loss function as 3DGS: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$, with $\lambda=0.2$.
- **Progressive Training**: Only the diffuse color $\mathbf{c_d}$ is trained in the first 3000 steps, and the specular color $\mathbf{c_s}$ is added afterward to improve training stability.
- **Illumination Field Boundary Shrinking**: During training, the illumination grid is shrunk to the actual bounding box of the Gaussians and resampled with the same voxel resolution.
- Employs the same adaptive density control as 3DGS.

## Key Experimental Results

### Main Results

**NeRF Synthetic Dataset (Average of 8 Scenes)**:

| Method | PSNR | SSIM | LPIPS | Real-time Rendering? |
|------|------|------|-------|----------|
| NeRF | 31.01 | 0.947 | 0.081 | No |
| Ref-NeRF | 31.29 | 0.947 | 0.058 | No |
| 3DGS | 33.30 | 0.969 | 0.030 | Yes |
| GaussianShader | 33.38 | 0.968 | 0.029 | Yes (6.3x slower) |
| **3iGS (Ours)** | **33.64** | **0.970** | **0.029** | Yes (2.0x slower) |

**Tanks and Temples Dataset (Real Scenes, Average of 5 Scenes)**:

| Method | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 3DGS | 29.61 | 0.950 | 0.060 |
| GaussianShader (reproduced) | 28.46 | 0.938 | 0.077 |
| **3iGS (Ours)** | **30.20** | **0.953** | **0.058** |

**Shiny Blender Dataset**: 3iGS achieves a PSNR of 30.77 vs 3DGS 30.37 vs GaussianShader 31.94. GaussianShader performs better on this dataset because the Shiny Blender scenes have simple geometry (single object) dominated by direct illumination, which is perfectly suited for GaussianShader's global environment map + GGX BRDF. In contrast, in NeRF Synthetic scenes with multiple objects and complex indirect illumination, 3iGS's local continuous illumination field shows a clear advantage.

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS | Explanation |
|------|------|------|-------|------|
| Direct prediction of outgoing radiance field | 32.38 | 0.965 | 0.035 | Directly predict radiance from position like NeRF, without BRDF decomposition |
| Without roughness parameter (no IDE) | 33.26 | 0.967 | 0.031 | Remove the roughness parameter and replace IDE with standard Fourier positional encoding |
| **Full Model** | **33.64** | **0.970** | **0.029** | All components |

### Key Findings
- **Illumination field is the core contribution**: The PSNR gap between direct prediction of outgoing radiance and the full model is 1.26 dB, demonstrating that the continuous illumination field is crucial for modeling view-dependent effects.
- **IDE view encoding is effective**: IDE controlled by the roughness parameter outperforms standard Fourier encoding by 0.38 dB PSNR.
- **Speed advantages**: The rendering speed of 3iGS is only 2.0x slower than 3DGS, and the training is 3.2x slower; in contrast, GaussianShader is 6.3x slower in rendering and 12.1x slower in training. This shows that the factorized tensorial illumination field is much more efficient than GaussianShader's ray-tracing approach.
- **Good generalization to real scenes**: 3iGS significantly outperforms GaussianShader on Tanks and Temples (+1.74 dB PSNR), showing that continuous local illumination modeling is more effective than global environment maps in complex real-world scenes.

## Highlights & Insights
- **Design philosophy borrowed from game engines**: Transferring the idea of irradiance volumes/light probes to differentiable rendering—instead of solving for exact physical parameters, the network learns in the illumination feature space, avoiding the ill-posed nature of inverse rendering while maintaining efficiency.
- **Compactness of tensor decomposition**: $150^3$ voxels + VM decomposition results in minimal storage overhead (far less than the parameter size of millions of Gaussians), and querying requires only trilinear interpolation, which is virtually "free."
- **Plug-and-play design**: The method is highly compatible with the 3DGS framework. It only replaces the color modeling component (SH -> illumination + BRDF + MLP), while the loss functions, adaptive control, and other components are completely reused.
- **Transferable "soft constraint" BRDF concept**: Learning neural BRDF features instead of enforcing rigid physical parameters can be extended to other tasks requiring material modeling.

## Limitations & Future Work
- **Limited to bounded scenes**: The factorized tensorial illumination grid requires a predefined bounding box, making it unable to directly handle unbounded outdoor large-scale scenes without scene warping techniques.
- **High VRAM requirements**: Inheriting a large number of Gaussians from 3DGS along with the additional illumination grid requires a high-VRAM GPU.
- **Unimproved geometric quality**: Inherits the difficulty of 3DGS in generating precise scene geometry—it only improves the appearance, while normal/geometry estimation remains inaccurate.
- **Single illumination field assumption**: Currently, there is only a single global illumination field, which might require partitioned modeling for dynamic lighting or extremely large-scale scenes.
- **Possible future directions**: Combining unbounded scene warping (such as contraction in mip-NeRF 360) + multi-resolution illumination fields.

## Related Work & Insights
- **vs 3DGS**: 3DGS independently optimizes each Gaussian's SH without scene-level illumination information; 3iGS introduces a shared illumination field so that Gaussians can "see" their surroundings, resulting in more realistic specular effects.
- **vs GaussianShader**: GaussianShader uses a global environment cube map + Cook-Torrance/GGX BRDF physical model; 3iGS uses a local continuous illumination field + neural BRDF, avoiding the ill-posed physical parameter estimation. GaussianShader excels in simple single-object scenes, while 3iGS is superior in complex multi-object scenes.
- **vs TensoRF**: TensoRF uses VM decomposition to model the entire radiance field (as a NeRF alternative); 3iGS only uses VM decomposition to model the illumination field, while the radiance field is still carried by Gaussians. The two are complementary.
- **vs Ref-NeRF**: Ref-NeRF introduces IDE encoding in the NeRF framework to improve reflection modeling; 3iGS adapts IDE into the Gaussian framework and combines it with the illumination field for better results.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing the illumination volume concept from game engines into the 3DGS framework, utilizing tensor decomposition for efficient illumination modeling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on both synthetic and real datasets, clear ablation studies, and convincing speed comparisons.
- Writing Quality: ⭐⭐⭐⭐ Smooth motivation derivation, with a clear logical chain from rendering equations to design choices.
- Value: ⭐⭐⭐⭐ Provides an efficient solution for view-dependent effects in 3DGS, and the concept of factorized tensorial illumination fields has practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints](surface_reconstruction_from_3d_gaussian_splatting_via_local_structural_hints.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)

</div>

<!-- RELATED:END -->
