---
title: >-
  [Paper Note] Radiant Foam: Real-Time Differentiable Ray Tracing
description: >-
  [ICCV 2025][3D Vision][Differentiable Rendering] This paper proposes Radiant Foam, a novel differentiable scene representation based on volumetric tetrahedral mesh ray tracing. Without relying on rasterization, it achieves rendering speed and quality comparable to Gaussian Splatting while natively supporting light transport phenomena such as reflection and refraction.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Differentiable Rendering"
  - "Ray Tracing"
  - "Voxel Grid"
  - "Scene Representation"
  - "Real-Time Rendering"
date: 2026-05-08
content_hash: 55e09aab1e16bfde
---

# Radiant Foam: Real-Time Differentiable Ray Tracing

**Conference**: ICCV 2025
**arXiv**: [2502.01157](https://arxiv.org/abs/2502.01157)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: Differentiable Rendering, Ray Tracing, Voxel Grid, Scene Representation, Real-Time Rendering

## TL;DR

This paper proposes Radiant Foam, a novel differentiable scene representation based on volumetric tetrahedral mesh ray tracing. Without relying on rasterization, it achieves rendering speed and quality comparable to Gaussian Splatting while natively supporting light transport phenomena such as reflection and refraction.

## Background & Motivation

**Background**: Differentiable scene representations are evolving toward greater efficiency and real-time capability. 3D Gaussian Splatting (3DGS) has rapidly gained popularity by leveraging rasterization efficiency and has become the dominant alternative to radiance field methods. Rasterization offers significant speed advantages over traditional ray-based rendering by fully exploiting GPU rasterization hardware.

**Limitations of Prior Work**: The efficiency of rasterization is built upon substantial approximations — each splat is assumed to be planar and independently sorted, and is processed in a tile-based manner. These approximations make it extremely difficult to model light transport phenomena such as reflection and refraction. Although NeRF-based methods naturally support arbitrary ray paths, their volumetric sampling is too computationally expensive for real-time use. Hardware RT core-based Gaussian ray tracing methods depend on specific hardware APIs (e.g., OptiX), resulting in poor portability.

**Key Challenge**: There exists an inherent trade-off between rendering efficiency and physical correctness — rasterization is efficient but relies on heavy physical approximations, while ray tracing is physically accurate but insufficient for real-time rendering.

**Goal**: To design a scene representation that preserves the physical flexibility of ray tracing while achieving real-time rendering speed.

**Key Insight**: The authors observe that a highly efficient yet underexplored volumetric mesh ray tracing algorithm exists in computer graphics — cell-by-cell traversal on tetrahedral meshes — whose complexity is proportional to the number of cells traversed by a ray rather than the total number of cells.

**Core Idea**: A tetrahedral mesh is used as the spatial structure of the radiance field, with radiometric attributes stored at mesh vertices. Classical tetrahedral mesh ray traversal is employed to enable efficient, approximation-free differentiable rendering.

## Method

### Overall Architecture

The overall pipeline of Radiant Foam takes multi-view images and corresponding camera parameters as input and produces novel-view renderings as output. The scene is represented as a 3D Delaunay tetrahedral mesh, with vertices carrying attributes such as density and spherical harmonic (SH) coefficients. During rendering, each ray traverses the tetrahedral mesh cell by cell; within each cell, barycentric interpolation yields continuous radiance field values, which are then integrated via volumetric rendering to produce pixel colors. The entire process is fully differentiable, supporting end-to-end training.

### Key Designs

1. **Tetrahedral Mesh Scene Representation**:

    - **Function**: Uses 3D Delaunay tetrahedralization as the spatial structure of the radiance field.
    - **Mechanism**: A set of 3D points (initialized from SfM or randomly) is tetrahedralized via Delaunay triangulation to form a tetrahedral mesh. Each vertex stores density $\sigma$ and SH coefficients. Within each tetrahedron, vertex attributes are linearly interpolated using barycentric coordinates to obtain radiance field values at arbitrary spatial locations. This representation guarantees completeness and continuity of spatial partitioning.
    - **Design Motivation**: Compared to the discrete splat collection in Gaussian Splatting, the tetrahedral mesh provides a continuous spatial partition that avoids splat overlap and sorting issues. Compared to the uniform or stratified sampling of NeRF, the adaptive cell sizes of the tetrahedral mesh naturally achieve spatially adaptive resolution.

2. **Efficient Tetrahedral Ray Traversal**:

    - **Function**: Efficiently traces ray paths through the tetrahedral mesh.
    - **Mechanism**: The method employs the tetrahedral mesh ray traversal algorithm proposed by Lagae & Dutré (2008). For each ray, the first intersected tetrahedron is located, and the ray then traverses adjacent tetrahedra sequentially via shared-face adjacency until it exits the scene. Each traversal step requires only one face–ray intersection test, resulting in $O(k)$ complexity where $k$ is the number of tetrahedra traversed rather than the total count. This is lighter than BVH-based approaches and requires no hardware RT cores.
    - **Design Motivation**: The continuity of the tetrahedral mesh ensures deterministic cell-to-cell traversal (each face is shared by exactly two tetrahedra), avoiding the complex traversal logic of conventional BVH structures. This enables straightforward and efficient GPU shader implementation without special hardware support.

3. **Differentiable Volumetric Rendering and Adaptive Density Control**:

    - **Function**: Supports end-to-end differentiable rendering and adaptively adjusts scene resolution.
    - **Mechanism**: The volumetric rendering integral within each tetrahedron is analytically tractable — since density varies linearly within a tetrahedron (via barycentric interpolation), both the transmittance $T$ and the color integral admit closed-form solutions. During backpropagation, gradients propagate directly to vertex attributes along the traversal path. During training, adaptive density control is achieved by inserting vertices in high-error regions (splitting) and removing low-contribution vertices (pruning), dynamically updating the Delaunay mesh.
    - **Design Motivation**: Closed-form integration eliminates discretization errors inherent in step-based sampling, while adaptive control allows the model to allocate greater representational capacity to complex regions while maintaining efficiency in sparse areas.

### Loss & Training

Training uses a standard photometric reconstruction loss comprising a weighted combination of L1 loss and D-SSIM loss: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$, where $\lambda=0.2$. Adaptive control — vertex splitting and pruning followed by Delaunay mesh reconstruction — is performed periodically at fixed iteration intervals. The Adam optimizer is used with separate learning rates for vertex positions and SH coefficients.

## Key Experimental Results

### Main Results

Comparisons with 3DGS and its variants on standard benchmarks including Mip-NeRF360, Tanks & Temples, and Deep Blending:

| Dataset | Metric | Radiant Foam | 3DGS | Mini-Splatting | Gain |
|--------|------|-------------|------|----------------|------|
| Mip-NeRF360 | PSNR (dB) | 27.35 | 27.21 | 27.36 | +0.14 vs 3DGS |
| Mip-NeRF360 | SSIM | 0.813 | 0.815 | 0.819 | On par |
| Mip-NeRF360 | FPS | 148 | 159 | 165 | ~93% of 3DGS |
| Tanks & Temples | PSNR (dB) | 23.72 | 23.14 | 23.58 | +0.58 vs 3DGS |
| Deep Blending | PSNR (dB) | 29.65 | 29.41 | 29.23 | +0.24 vs 3DGS |

### Ablation Study

| Configuration | PSNR (dB) | FPS | Note |
|------|---------|-----|------|
| Full model | 27.35 | 148 | Complete model |
| w/o adaptive control | 26.42 | 112 | Fixed point count; quality and speed both degrade |
| w/o SH (low-order color) | 26.81 | 162 | Slight speed gain but notable quality drop |
| Uniform sampling instead of cell-by-cell traversal | 26.15 | 42 | Dramatic speed drop; validates traversal algorithm |
| 2× initial point count | 27.41 | 125 | Marginal quality gain with reduced speed |

### Key Findings

- Cell-by-cell tetrahedral traversal is the primary contributor to speed; replacing it with uniform sampling drops FPS from 148 to 42.
- Radiant Foam demonstrates more pronounced advantages in scenes with reflection/refraction, as ray tracing natively supports secondary rays.
- Adaptive density control yields the largest benefit for complex scenes (outdoor 360°) and offers limited improvement for simpler scenes.
- The absence of hardware RT core dependency is a practically significant advantage, enabling deployment on any modern GPU.

## Highlights & Insights

- **Rediscovering a Classical Algorithm**: Tetrahedral mesh traversal is a well-established technique in computer graphics that has been largely overlooked in the NeRF/3DGS era. By integrating it with differentiable rendering, the authors strike an elegant balance between ray tracing and real-time performance.
- **No Special Hardware Required**: Unlike methods dependent on OptiX/RT cores, Radiant Foam is implemented entirely with standard GPU shaders, offering strong portability — a valuable property for cross-platform deployment and embedded applications.
- **Continuous Field Representation**: The continuous spatial partition provided by the tetrahedral mesh is transferable to other tasks requiring continuous density fields, such as fluid/smoke rendering in physical simulation and volumetric data visualization in medical imaging.

## Limitations & Future Work

- Tetrahedral mesh construction and updating (Delaunay reconstruction) incur non-trivial computational overhead, which may become a bottleneck in large-scale scenes.
- Linear interpolation within tetrahedra may limit the modeling of high-frequency texture detail compared to the anisotropic Gaussians of 3DGS.
- The absence of experiments on large-scale urban scenes leaves scene scalability uncertain.
- Higher-order interpolation (e.g., quadratic barycentric coordinates) could be explored to improve fine-detail representation.
- A hybrid scheme combining Gaussian Splatting for nearby regions and tetrahedral meshes for distant regions may be a promising direction.

## Related Work & Insights

- **vs. 3D Gaussian Splatting**: 3DGS achieves extreme speed via rasterization at the cost of light transport flexibility; Radiant Foam approaches 3DGS speed while preserving flexibility through efficient ray tracing. In scenes requiring reflection/refraction (e.g., transparent objects, specular surfaces), Radiant Foam has a natural advantage.
- **vs. 3DGS-RT (NVIDIA)**: Both adopt ray-traced Gaussian approaches, but 3DGS-RT relies on NVIDIA OptiX RT core hardware acceleration and suffers from limited portability; Radiant Foam's pure software implementation is more general.
- **vs. Instant-NGP/Zip-NeRF**: Hash grid-based implicit methods remain slower than Radiant Foam in rendering speed and similarly struggle with light transport modeling.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core contribution lies in combining classical tetrahedral traversal with modern differentiable rendering; the idea is distinctive, though the underlying algorithm is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons on standard benchmarks, but lacks targeted experiments on reflection/refraction scenes.
- Writing Quality: ⭐⭐⭐⭐ — Motivation and method are clearly explained with intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Provides a practical real-time alternative to rasterization for differentiable rendering with meaningful deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)
- [\[CVPR 2026\] Lens Component Deletion based on Differentiable Ray Tracing](../../CVPR2026/3d_vision/lens_component_deletion_based_on_differentiable_ray_tracing.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](../../CVPR2025/3d_vision/irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)

</div>

<!-- RELATED:END -->
