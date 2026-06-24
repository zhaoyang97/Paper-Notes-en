---
title: >-
  [Paper Note] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering
description: >-
  [CVPR 2025][3D Vision][Inverse Rendering] This paper proposes the SVG-IR framework, which introduces a Spatially-Varying Gaussian (SVG) representation to allow a single Gaussian primitive to possess spatially-varying material and normal parameters. Combined with a physically-based indirect illumination model, SVG-IR maintains real-time rendering speeds while outperforming NeRF-based methods by 2.5 dB and existing Gaussian-based methods by 3.5 dB in relighting quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Inverse Rendering"
  - "3D Gaussian Splatting"
  - "Spatially-Varying Materials"
  - "Physically-Based Indirect Illumination"
  - "Relighting"
date: 2026-05-08
content_hash: 2363a8eeca43d292
---

# SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering

**Conference**: CVPR 2025  
**arXiv**: [2504.06815](https://arxiv.org/abs/2504.06815)  
**Code**: [https://github.com/learner-shx/SVG-IR](https://github.com/learner-shx/SVG-IR)  
**Area**: 3D Vision  
**Keywords**: Inverse Rendering, 3D Gaussian Splatting, Spatially-Varying Materials, Physically-Based Indirect Illumination, Relighting

## TL;DR

This paper proposes the SVG-IR framework, which introduces a Spatially-Varying Gaussian (SVG) representation to allow a single Gaussian primitive to possess spatially-varying material and normal parameters. Combined with a physically-based indirect illumination model, SVG-IR maintains real-time rendering speeds while outperforming NeRF-based methods by 2.5 dB and existing Gaussian-based methods by 3.5 dB in relighting quality.

## Background & Motivation

**Background**: Reconstructing 3D assets from multi-view images (inverse rendering) is a long-standing task in computer graphics and vision. 3D Gaussian Splatting (3DGS) has shown outstanding performance in novel view synthesis (NVS) with fast training and rendering speeds. Several methods extend it to relighting by decomposing radiance into BRDF material parameters and illumination.

**Limitations of Prior Work**: Existing Gaussian-based inverse rendering methods (e.g., GS-IR, Relightable 3DGS) exhibit significant artifacts during relighting. The root cause is that each Gaussian possesses constant material parameters and normals, resulting in a uniform color for a given view and light direction. However, in practice, a single Gaussian might cover multiple pixels that should have different colors. Furthermore, these methods lack physical constraints when modeling indirect illumination, utilizing unconstrained spherical harmonics to learn residual lighting, which leads to unnatural indirect illumination that does not change under new light sources.

**Key Challenge**: The fundamental conflict between the "constant property" assumption of individual Gaussians and the reality of spatially-varying materials in real-world scenes; the unconstrained learning of indirect light cannot support realistic relighting under novel light sources.

**Goal**: (1) Design a more powerful Gaussian representation to support spatially-varying materials; (2) Introduce a physically constrained indirect illumination model to achieve realistic relighting effects.

**Key Insight**: Inspired by the evolution from flat shading to interpolated shading in computer graphics, multiple "Gaussian vertices" are defined on a Gaussian to allow materials and normals to vary spatially within the Gaussian.

**Core Idea**: Replace constant-parameter Gaussians with spatially-varying Gaussians containing multiple vertices, coordinate this with an SVG splatting pipeline similar to vertex/fragment shading in triangle rendering, and compute physical indirect illumination via ray tracing within the radiance field.

## Method

### Overall Architecture

SVG-IR is a multi-stage inverse rendering framework. First, standard Gaussian Surfels are used to initialize Gaussian properties (geometry, color, opacity). Then, Spatially-Varying Gaussians (SVG) are constructed, which inherit the pre-trained properties and add M=4 Gaussian vertices to each Gaussian, where each vertex possesses independent albedo, roughness, and normal offsets. The framework renders images via SVG splatting, using a physically-based lighting model (comprising direct and indirect illumination), and optimizes the properties of the Gaussian vertices using loss functions.

### Key Designs

1. **Spatially-Varying Gaussian Representation (SVG)**:

    - **Function**: Break the limitation of each Gaussian only having constant materials, allowing a single Gaussian primitive to have internal variations in materials and normals.
    - **Mechanism**: Define $M$ representative positions (Gaussian vertices) for each Gaussian in the tangent space based on 2D Gaussian surfels, where each vertex has an independent albedo $\boldsymbol{a}_i$, roughness $r_i$, and normal offset $\Delta N_i$. When querying properties at any position within a Gaussian, bilinear interpolation is performed among the vertices. Constant Gaussians represent a special case of SVG ($M=1$).
    - **Design Motivation**: As shown in Fig. 2, constant Gaussians require multiple Gaussians to fit a single BRDF distribution, whereas SVG requires only one to complete the fitting through different BRDF lobes at different positions, yielding stronger expressiveness.

2. **SVG Splatting (Gaussian Vertex/Fragment Shading)**:

    - **Function**: Achieve efficient rendering of spatially-varying Gaussians, supporting per-pixel material interpolation.
    - **Mechanism**: Analogous to the triangle rendering pipeline, this is divided into vertex shading and fragment shading stages. Vertex shading stage: calculate the radiance of each Gaussian vertex using the physical illumination model to obtain $M$ colors $\mathbf{c}^{\{M\}_i}$. Fragment shading stage: for each camera ray, compute the corresponding $(u, v)$ coordinates of the pixel in the Gaussian tangent space, obtain the color at this pixel via bilinear interpolation, and then blend the contributions of all Gaussians through alpha blending. The key formula is $\mathbf{c}_i = \text{BilinearInterp}(\mathbf{c}_i^{\{M\}}, u, v)$.
    - **Design Motivation**: Directly leveraging the mature vertex/fragment shading paradigm from computer graphics is both compatible with the rasterization pipeline of 3DGS and efficient in handling spatially-varying properties.

3. **Physically-Based Indirect Illumination Model**:

    - **Function**: Explicitly model the light transport process by ray tracing within the radiance field to achieve physically plausible indirect illumination.
    - **Mechanism**: For each Gaussian, uniformly sample $K$ directions in the upper hemisphere, and build a BVH by treating Gaussians as ellipses to accelerate ray tracing. Accumulate transmittance and radiance along each ray to obtain the indirect incident radiance $L_i^{ind}$. Incident light is decomposed as: $L_i(x, \omega_i) = L_i^{dir}(x, \omega_i)V(x, \omega_i) + L_i^{ind}(x, \omega_i)$, where $V$ is visibility, determined by a transmittance threshold. During relighting, the indirect illumination learned during training is replaced with one-bounce indirect illumination, achieved by querying secondary Gaussians and computing their outgoing radiance under the new light source to enable dynamic indirect illumination.
    - **Design Motivation**: Existing methods learn indirect illumination using unconstrained spherical harmonics, which, while improving NVS quality, results in unnatural indirect illumination that cannot change with new light sources. Physical constraints make the decoupling of material and illumination more reasonable, supporting realistic indirect illumination under novel light sources.

### Loss & Training

The total loss consists of multiple terms: $\mathcal{L} = \lambda_1\mathcal{L}_1 + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{rc}\mathcal{L}_{rc} + \lambda_n\mathcal{L}_n + \lambda_{s,a}\mathcal{L}_{s,a} + \lambda_{s,r}\mathcal{L}_{s,r} + \lambda_{reg,n}\mathcal{L}_{reg,n}$.

Where $\mathcal{L}_1$ and $\mathcal{L}_{ssim}$ are the L1 and SSIM losses between the rendered image and the ground truth; $\mathcal{L}_{rc}$ is the radiance consistency loss, which supervises the outgoing radiance of secondary Gaussians using the indirect radiance obtained by ray tracing; $\mathcal{L}_n$ is the normal consistency loss; $\mathcal{L}_{s,a}$ and $\mathcal{L}_{s,r}$ are the total variation smoothness regularization for albedo and roughness; $\mathcal{L}_{reg,n}$ is the L2 regularization term for normal offsets.

## Key Experimental Results

### Main Results

| Method | TensoIR Average | ADT Average | Type |
|------|------------|---------|------|
| MII | 27.76 | 29.74 | NeRF |
| TensoIR | 28.53 | 30.58 | NeRF |
| GS-Shader | 19.89 | 23.04 | GS |
| GS-IR | 24.69 | 26.95 | GS |
| RelightGS | 27.60 | 32.84 | GS |
| **Ours** | **31.10** | **34.69** | GS |

Outperforms the best NeRF method by 2.5 dB and the best GS method by 3.5 dB, while the training time is only 1 hour (compared to 5-6h for NeRF methods).

### Ablation Study

| SVG | PBI | Relighting PSNR | NVS PSNR |
|-----|-----|-----------|---------|
| ✗ | ✗ | 28.614 | 34.640 |
| ✓ | ✗ | 29.447 | 35.794 |
| ✓ | ✓ | **31.087** | **36.709** |

### Key Findings

- The SVG representation alone can improve the relighting PSNR by approximately 0.8 dB, and further combining it with physical indirect illumination yields a 1.6 dB improvement.
- The indirect illumination of Relightable 3DGS remains unchanged under different environment maps (leading to abnormally bright regions in dark lighting), whereas this method dynamically adapts to novel light sources.
- On the NVS task, the proposed method also outperforms all baseline methods, achieving a PSNR of 36.71 on TensoIR Synthetic (vs 35.17 for TensoIR).
- GPU memory overhead increases by 50%-80% (compared to 3DGS/2DGS) and 10%-20% (compared to Relightable 3DGS).

## Highlights & Insights

- **Drawing Inspiration from Classical Graphics Paradigms**: Transferring the evolution of flat shading $\rightarrow$ interpolated shading to 3DGS to achieve spatially-varying materials within a single Gaussian, which is conceptually simple and highly efficient.
- **A Practical Solution for Physical Indirect Illumination**: Obtaining indirect illumination through ray tracing on top of the existing radiance field, which avoids the need for training extra parameters and supports dynamic updating under novel light sources.
- **Clever Utilization of Radiance Consistency Loss**: Using indirect radiance as supervision for the outgoing radiance of secondary Gaussians, providing guidance from additional views.

## Limitations & Future Work

- Lacks geometric priors (such as SDF), resulting in suboptimal recovery for highly specular objects.
- Gaussian vertices incur additional GPU memory overhead.
- Indirect illumination only considers single-bounce reflections, which may not be precise enough for complex multi-bounce scenes.
- Future work could integrate GS-ROR or similar methods to introduce SDF to improve geometric accuracy.

## Related Work & Insights

- NeRF-based inverse rendering (e.g., TensoIR, InvRender) yields high quality but is slow, while 3DGS-based methods are fast but constrained by the constant-material assumption.
- The SVG concept in this paper is inspiring for all Gaussian-based methods: any task that requires representing spatially-varying properties within a Gaussian can refer to this vertex-based interpolation design.
- The one-bounce scheme for physical indirect illumination achieves a good balance between real-time performance and quality.

## Rating

- **Novelty**: 9/10 — The concept of Spatially-Varying Gaussian is novel and elegantly aligns with graphics paradigms.
- **Experimental Thoroughness**: 8/10 — Comprehensive evaluation on synthetic and real datasets with clear ablations.
- **Writing Quality**: 8/10 — Fluent logic with intuitive diagrams.
- **Value**: 9/10 — Significantly pushes forward the field of 3DGS inverse rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[CVPR 2026\] IR-HGP: Physically-Aware Gaussian Inverse Rendering for High-Illumination Scenes via Generative Priors](../../CVPR2026/3d_vision/ir-hgp_physically-aware_gaussian_inverse_rendering_for_high-illumination_scenes_.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)

</div>

<!-- RELATED:END -->
