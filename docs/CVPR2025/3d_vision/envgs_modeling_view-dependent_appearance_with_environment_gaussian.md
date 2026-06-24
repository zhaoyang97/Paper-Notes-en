---
title: >-
  [Paper Note] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper proposes EnvGS, which utilizes a set of environment Gaussian primitives as an explicit 3D representation to capture scene reflections. By jointly optimizing environment Gaussians and base Gaussians through a GPU RT Core-based differentiable ray-tracing renderer, it achieves real-time (26+ FPS) and high-quality specular reflection novel view synthesis in real-world scenes for the first time…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Reflection Modeling"
  - "Ray Tracing"
  - "2D Gaussian"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: d4539cd26145622a
---

# EnvGS: Modeling View-Dependent Appearance with Environment Gaussian

**Conference**: CVPR 2025  
**arXiv**: [2412.15215](https://arxiv.org/abs/2412.15215)  
**Code**: [https://zju3dv.github.io/envgs](https://zju3dv.github.io/envgs)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Reflection Modeling, Ray Tracing, 2D Gaussian, Novel View Synthesis

## TL;DR
This paper proposes EnvGS, which utilizes a set of environment Gaussian primitives as an explicit 3D representation to capture scene reflections. By jointly optimizing environment Gaussians and base Gaussians through a GPU RT Core-based differentiable ray-tracing renderer, it achieves real-time (26+ FPS) and high-quality specular reflection novel view synthesis in real-world scenes for the first time, significantly outperforming all real-time methods.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) achieves real-time novel view synthesis via explicit Gaussian primitives and rasterization. However, using Spherical Harmonics (SH) to model view-dependent effects limits representation capability, leading to poor results under strong specular reflections.

2. **Limitations of Prior Work**: Methods like GaussianShader and 3DGS-DR introduce environment maps to enhance reflection modeling, but they suffer from two fundamental limitations: (a) environment maps assume far-field illumination, failing to model near-field reflections (e.g., reflections of objects on a table); and (b) low-frequency environment maps lack the capacity to capture high-frequency reflection details.

3. **Key Challenge**: There is a need for a representation that can model reflections of arbitrary complexity (near-field + far-field + high-frequency) while maintaining real-time rendering speed. NeRF-based methods (e.g., NeRF-Casting) offer high quality but are too slow (<0.1 FPS), whereas Gaussian-based methods are fast but produce poor reflection quality.

4. **Goal**: (1) How to model complex reflections using explicit 3D representations? (2) How to efficiently render queries along reflection directions? (3) How to jointly optimize reflection representations and scene geometry?

5. **Key Insight**: Instead of projectively representing environmental lighting with 2D environment maps, another set of Gaussian primitives can be used to explicitly represent reflection content in 3D space. Gaussian primitives naturally support high-frequency details and near-field positional information.

6. **Core Idea**: Substitute environment maps with environment Gaussians to explicitly model 3D reflections, combined with differentiable ray tracing based on OptiX RT Core for efficient rendering.

## Method

### Overall Architecture
The scene is represented by two sets of 2D Gaussian primitives: base Gaussians $\mathbf{P}_{base}$ modeling geometry and diffuse appearance, and environment Gaussians $\mathbf{P}_{env}$ modeling reflections. The rendering pipeline consists of three steps: (1) rasterizing base Gaussians to obtain surface normals $\mathbf{n}$, base colors $\mathbf{c}_{base}$, blend weights $\beta$, and surface positions $\mathbf{x}$; (2) tracing rays from surface point $\mathbf{x}$ along the reflection direction $\mathbf{d}_{ref}$ to render the environment Gaussians via ray tracing, yielding reflection colors $\mathbf{c}_{ref}$; (3) blending to produce the final color $\mathbf{c} = (1-\beta) \cdot \mathbf{c}_{base} + \beta \cdot \mathbf{c}_{ref}$.

### Key Designs

1. **Environment Gaussian Representation**:

    - **Function**: Explicitly model the reflection content of the scene in 3D space.
    - **Mechanism**: Shares the same 2DGS parameterization with base Gaussians (center position, opacity, tangent vectors, scaling, SH coefficients), but represents environmental/reflection information instead of scene geometry. Each base Gaussian additionally learns a blend weight $\beta$ to control the diffuse/specular mixing ratio. Environment Gaussians are initialized by partitioning the scene bounding box into $32^3$ voxels and randomly sampling 5 primitives per voxel, being jointly optimized with the base Gaussians after the bootstrapping stage.
    - **Design Motivation**: Compared to environment maps, Gaussian primitives possess precise 3D positional information, naturally supporting near-field reflections. Furthermore, the number of Gaussian primitives can adaptively grow (via densification), theoretically allowing the representation of arbitrary high-frequency reflection details.

2. **Differentiable Gaussian Tracer**:

    - **Function**: Efficiently render the environmental Gaussian colors seen from each surface point along the reflection direction.
    - **Mechanism**: Encloses each 2D Gaussian within two triangles (transforming 4 vertices to world space) to build a BVH acceleration structure. OptiX's programmable `raygen` and `anyhit` entry points are used for chunk-by-chunk rendering. `anyhit` collects the $k=16$ nearest intersections and sorts them, while `raygen` integrates volume rendering in front-to-back depth order. It terminates when the accumulated transmittance falls below a threshold or when there are no more intersections. Backpropagation re-emits rays in the same front-to-back order as the forward pass to compute gradients, focusing on key computations $\partial \mathcal{L}/\partial \mathbf{o}$ and $\partial \mathcal{L}/\partial \mathbf{d}$ (gradients with respect to ray origin and direction), enabling joint optimization of base and environment Gaussians. This achieves up to 30 FPS for 2 million 2DGS on an RTX 4095.
    - **Design Motivation**: In reflection rendering, each pixel corresponds to a unique reflection ray (with different origins and directions), preventing the use of standard rasterization. Leveraging modern GPU RT Core hardware acceleration for ray-triangle intersection is the only viable choice that meets both speed and flexibility requirements.

3. **Monocular Normal Supervision and Joint Optimization**:

    - **Function**: Resolve geometric ambiguity in reflective/refractive scenes, ensuring that base Gaussians reconstruct correct surface normals.
    - **Mechanism**: In addition to the standard normal consistency constraint $\mathcal{L}_{norm}$ (rendered normals vs. depth-map gradient normals), predictions $\mathbf{N}_m$ from a pre-trained monocular normal estimation model are introduced as auxiliary supervision: $\mathcal{L}_{mono} = \frac{1}{N_p} \sum (1 - \mathbf{n}_i^\top \mathbf{N}_m)$. During joint optimization, gradients from ray tracing backpropagate through the reflection directions to the normal parameters of the base Gaussians, establishing a closed-loop geometry-appearance optimization.
    - **Design Motivation**: Normals of reflective/refractive surfaces are highly sensitive to reflection directions; minor normal errors can cause massive deviations in reflection paths. Monocular normals serve as view-independent priors, preventing Gaussians from distorting geometry (the "foggy" geometry issue) to fit reflections.

### Loss & Training
The loss function is defined as $\mathcal{L} = \mathcal{L}_{rgb} + 0.04 \cdot \mathcal{L}_{norm} + 0.01 \cdot \mathcal{L}_{mono} + 0.01 \cdot \mathcal{L}_{perc}$, where $\mathcal{L}_{rgb}$ is $0.8 \cdot L_1 + 0.2 \cdot \text{D-SSIM}$, and $\mathcal{L}_{perc}$ is the VGG-16 perceptual loss. First, base Gaussians are trained alone (bootstrapping), followed by initializing environment Gaussians to begin joint optimization. Adaptive density control from 3DGS is combined with the normal propagation and color corruption strategies from 3DGS-DR.

## Key Experimental Results

### Main Results

| Dataset | Method Type | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS |
|--------|---------|------|-------|-------|--------|-----|
| Ref-Real | Real-time | 3DGS | 23.70 | 0.641 | 0.262 | 182 |
| Ref-Real | Real-time | 2DGS | 23.80 | 0.654 | 0.281 | 159 |
| Ref-Real | Real-time | GaussianShader | 22.88 | 0.622 | 0.314 | 28 |
| Ref-Real | Real-time | **EnvGS** | **24.62** | **0.671** | **0.241** | 26 |
| Ref-Real | Non-real-time | NeRF-Casting | 24.67 | 0.659 | 0.246 | <0.1 |
| NeRF-Cast Scenes | Real-time | **EnvGS** | **30.44** | **0.886** | **0.148** | 26 |
| NeRF-Cast Scenes | Non-real-time | NeRF-Casting | 31.02 | 0.889 | 0.128 | <0.1 |

### Ablation Study

| Configuration | PSNR | SSIM | Description |
|------|------|------|------|
| Full EnvGS | 24.62 | 0.671 | Full model |
| w/ environment map instead of EnvGS | - | - | Fails to capture near-field reflections |
| w/o monocular normal constraint | - | - | High geometric noise, inaccurate reflection reconstruction |
| w/o joint optimization | - | - | Inaccurate normals cause shifts in reflection directions |

### Key Findings
- EnvGS ranks first among all real-time methods, outperforming the runner-up 2DGS on Ref-Real by ~0.8 dB in PSNR, with a 14% improvement in LPIPS.
- Its performance is highly comparable to the non-real-time SOTA NeRF-Casting (PSNR difference of only 0.05), while being over 100x faster.
- The improvements of EnvGS are even more pronounced in foreground and near-field areas (foreground PSNR 33.30 vs 31.68, near-field PSNR 46.39 vs 44.16).
- Joint optimization is crucial: without it, normal inaccuracy leads to shifts in reflection directions.
- Training takes approximately 2.5 hours on an RTX 4090, which is over 20x faster than NeRF-Casting (>47 hours).

## Highlights & Insights
- **Representing the Environment with Gaussians instead of Environment Maps**: This conceptual paradigm shift is highly intuitive but effective—the same representation is used for both the scene and its environment, allowing unified optimization and natural support for near-field reflections.
- **Differentiable Ray Tracing Accelerated by OptiX RT Cores**: Introducing hardware-accelerated ray tracing to Gaussian rendering is a substantial technical contribution. The chunk-by-chunk forward rendering and re-projection backpropagation strategy elegantly resolves memory and footprint speed trade-offs.
- **A Milestone in Real-time Reflection Rendering**: To the best of the authors' knowledge, this is the first method to realize real-time, high-quality specular reflection synthesis in real-world scenes, providing direct value to AR/VR applications.

## Limitations & Future Work
- The FPS is around 26. Although real-time, it is significantly slower than standard 3DGS (182 FPS) without reflection modeling, still presenting a gap for performance-critical applications.
- Environment Gaussians are initialized via uniform grid sampling, making them sensitive to scene scale. Tuning may be required for extremely large or small scenes.
- It only addresses specular reflections, leaving out more complex optical phenomena such as scattering and refraction.
- It relies on modern GPUs (specifically the RT Cores of an RTX 4090), and performance might degrade severely on older GPUs.

## Related Work & Insights
- **vs GaussianShader / 3DGS-DR**: These methods employ 2D environment maps to model reflections, which can only handle far-field lighting. EnvGS replaces them with 3D Gaussians, naturally accommodating both near-field and far-field effects.
- **vs NeRF-Casting**: NeRF-Casting performs ray marching and MLP decoding along reflection directions, yielding excellent quality but at an extremely slow speed. EnvGS leverages explicit Gaussians and hardware-accelerated ray tracing to achieve comparable quality while being 100x faster.
- **vs 3iGS**: 3iGS models the illumination field using tensor decomposition, but is restricted to bounded scenes. EnvGS’s explicit Gaussians naturally support unbounded scenes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of environment Gaussians is novel and elegant, and the differentiable ray-tracing renderer possesses significant technical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-dataset comparisons, ablations, and specialized evaluations of foreground/near-field areas are provided.
- Writing Quality: ⭐⭐⭐⭐⭐ The methodology is clearly described, diagrams are highly informative, and the structure is well-organized.
- Value: ⭐⭐⭐⭐⭐ A landmark breakthrough in real-time reflection rendering, holding substantial practical value for 3D reconstruction and the VR/AR domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[CVPR 2025\] ReCap: Better Gaussian Relighting with Cross-Environment Captures](recap_better_gaussian_relighting_with_cross-environment_captures.md)
- [\[ICML 2025\] LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](../../ICML2025/3d_vision/tackling_view-dependent_semantics_in_3d_language_gaussian_splatting.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2025\] Learnable Infinite Taylor Gaussian for Dynamic View Rendering](learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)

</div>

<!-- RELATED:END -->
