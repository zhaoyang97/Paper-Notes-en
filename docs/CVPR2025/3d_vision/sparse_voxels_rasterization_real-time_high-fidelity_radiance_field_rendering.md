---
title: >-
  [Paper Note] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering
description: >-
  [CVPR 2025][3D Vision][Sparse Voxels] This paper proposes SVRaster, an efficient radiance field rendering method that requires no neural networks or 3D Gaussians. By utilizing an adaptive multi-level sparse voxel representation and a customized rasterizer based on direction-dependent Morton sorting, it achieves artifact-free, real-time, high-fidelity rendering.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse Voxels"
  - "Rasterization"
  - "Radiance Field Rendering"
  - "Neural-Network-Free"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 846195ddeb454d24
---

# Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering

**Conference**: CVPR 2025  
**arXiv**: [2412.04459](https://arxiv.org/abs/2412.04459)  
**Code**: [https://github.com/NVlabs/svraster](https://github.com/NVlabs/svraster)  
**Area**: 3D Vision  
**Keywords**: Sparse Voxels, Rasterization, Radiance Field Rendering, Neural-Network-Free, Novel View Synthesis

## TL;DR

This paper proposes SVRaster, an efficient radiance field rendering method that requires no neural networks or 3D Gaussians. By utilizing an adaptive multi-level sparse voxel representation and a customized rasterizer based on direction-dependent Morton sorting, it achieves artifact-free, real-time, high-fidelity rendering.

## Background & Motivation

**Background**: In the field of novel view synthesis, there are two mainstream directions: ray-tracing-based NeRF methods, which employ dense sampling for volume rendering and are physically accurate but slow; and 3DGS, which leverages the rasterization of Gaussian primitives for fast rendering but suffers from popping artifacts due to inaccurate sorting.

**Limitations of Prior Work**: (1) 3DGS sorts primitives based on their centers, which cannot guarantee the correct depth order, leading to sudden color jumps (popping artifacts) when the view changes; (2) the volumetric density of overlapping 3D Gaussians is ill-defined, making surface reconstruction difficult; (3) existing voxel-based methods (e.g., Plenoxels) use uniform resolutions or rely on dense 3D data structures, which limits their efficiency and quality.

**Key Challenge**: The efficiency of rasterization and the physical correctness of volume rendering seemingly require a trade-off. However, as grid primitives, voxels inherently possess clear ordering and volumetric definitions, acting as an ideal bridge to connect the two.

**Goal**: To use sparse voxels as scene representations and design an efficient rasterization algorithm that simultaneously achieves the speed of 3DGS and the physical correctness of volume rendering.

**Key Insight**: Morton encoding (Z-order curve) provides a natural spatial ordering for voxels in an octree layout. The rendering order can be guaranteed simply by choosing the correct Morton arrangement according to the ray direction.

**Core Idea**: (1) Use adaptive multi-level sparse voxels (octree layout, with a maximum resolution of $65536^3$) to explicitly store density and spherical harmonics coefficients without neural networks; (2) Design a direction-dependent Morton sorting scheme to guarantee the correct rendering order of voxels within a rasterization framework, thereby eliminating popping artifacts.

## Method

### Overall Architecture

The scene is represented by a set of sparse voxels, with each voxel storing 8 corner density values (continuous density field obtained via trilinear interpolation) and spherical harmonics coefficients (view-dependent colors). During rendering, voxels are projected onto the image space, allocated to tiles, sorted according to the direction-dependent Morton order, and finally blended into pixel colors using standard alpha blending.

### Key Designs

1. **Adaptive Multi-Level Sparse Voxel Representation**:

    - **Function**: To faithfully represent different levels of detail in the scene at varying resolutions.
    - **Mechanism**: Voxels are allocated under an octree layout with a maximum hierarchy level $L=16$ (corresponding to a resolution of $(2^{16})^3 = 65536^3$). Each voxel is defined by its index $v=\{i,j,k\}$ and level $l$, with its size calculated as $\mathbf{v}_s = \mathbf{w}_s \cdot 2^{-l}$. The density field is represented by 8 corner parameters $\mathbf{v}_{\text{geo}} \in \mathbb{R}^{2 \times 2 \times 2}$, and adjacent voxels share corner points to ensure a continuous density field. The alpha value is obtained by numerical integration over $K$ points uniformly sampled along the ray-voxel intersection segment: $\alpha = 1 - \exp(-\frac{l}{K}\sum_k \text{explin}(\text{interp}(\mathbf{v}_{\text{geo}}, \mathbf{q}_k)))$. No dense 3D data structures are used, and voxels are stored in a 1D array.
    - **Design Motivation**: Uniform-resolution voxel methods (such as Plenoxels) either waste memory or lose details in scenes with varying detail density. The adaptive hierarchy allows coarse regions to use large voxels and detailed regions to use small voxels, substantially improving representation efficiency.

2. **Direction-Dependent Morton Sorting Rasterizer**:

    - **Function**: To guarantee a correct rendering sequence for voxels of arbitrary sizes and eliminate popping artifacts.
    - **Mechanism**: Under an octree layout, sorting voxels by Morton codes (bit-interleaving operations) guarantees a correct depth order, provided that the Morton sorting scheme aligns with the ray direction. There are 8 different Morton arrangements in 3D space (determined by the signs of the three components of the ray direction), and the algorithm selects the corresponding Morton arrangement for sorting based on the current ray direction. The entire rasterization pipeline consists of: (a) projecting the 8 corners of voxels to image space; (b) allocating voxels to covered tiles; (c) sorting them in direction-dependent Morton order; (d) performing forward/backward alpha blending.
    - **Design Motivation**: Determining sorting order based on primitive centers in 3DGS is merely an approximation; mixed-sized Gaussians lead to incorrect orderings (as shown in Figure 4a). Voxels in an octree layout naturally support Morton sorting, completely avoiding sorting approximation issues.

3. **Progressive Scene Optimization Strategy**:

    - **Function**: To adaptively increase voxel resolution in a coarse-to-fine manner.
    - **Mechanism**: At the beginning of training, voxels are initialized at coarse levels. Every fixed number of steps, a "growing" operation is performed, wherein voxels with large rendering gradients are subdivided into 8 sub-voxels (octree splitting). Simultaneously, low-density voxels are pruned to reduce redundancy. The density activation function uses $\text{explin}(x)$ instead of softplus, which is linear and more efficient for large values. COLMAP sparse points are not required for initialization.
    - **Design Motivation**: The coarse-to-fine strategy avoids requiring a large number of voxels at the outset, and progressive growth stabilizes optimization. Eliminating the need for sparse point initialization makes the method more versatile.

### Loss & Training

The method uses L1 + SSIM color loss, along with a density distillation loss and normal smoothness regularization. The SH order progressively increases from 0 to 3. Both training and rendering are implemented using custom CUDA kernels. Voxel properties (color, normals) are calculated once during the preprocessing phase and shared among all pixels.

## Key Experimental Results

### Main Results

Novel view synthesis on the MipNeRF-360 dataset:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|---|---|---|---|---|
| iNGP | 25.59 | 0.699 | 0.331 | 9.43 |
| Plenoxels | 23.08 | 0.626 | 0.463 | 6.79 |
| 3DGS | 27.49 | 0.815 | 0.214 | 134 |
| 2DGS | 26.76 | 0.805 | 0.230 | 117 |
| **SVRaster** | **27.30** | **0.813** | **0.218** | **142** |

### Ablation Study

| Configuration | PSNR↑ | FPS↑ | Details |
|---|---|---|---|
| Uniform Voxels | ~23-24 | ~10 | Plenoxels level |
| Adaptive Voxels + Naive Sorting | — | — | popping artifacts |
| Adaptive Voxels + Morton Sorting | **27.30** | **142** | Full Method |

Comparison with Plenoxels:

| Method | PSNR↑ | FPS↑ |
|---|---|---|
| Plenoxels | 23.08 | 6.79 |
| **SVRaster** | **27.30** | **142** |

Gain: +4.22 dB PSNR, 20× FPS acceleration.

### Key Findings

- SVRaster is comparable to 3DGS in terms of PSNR/SSIM/LPIPS (27.30 vs 27.49) while delivering a higher FPS (142 vs 134).
- Compared with Plenoxels, a similar neural-network-free voxel-based method, SVRaster achieves a gain of over 4 dB PSNR and a 20× FPS acceleration.
- The direction-dependent Morton sorting completely eliminates popping artifacts, which remains an unresolved issue in 3DGS.
- Sparse voxels are naturally compatible with classical 3D algorithms like Volume Fusion and Marching Cubes, enabling direct mesh extraction and fusion with semantic feature fields.

## Highlights & Insights

1. **Reverse thinking on "returning to voxels"**: Re-evaluating voxels when Gaussian Splatting is at its peak reveals their inherent advantages in ordering correctness and volumetric definition.
2. **Clever application of Morton order**: Utilizing the mathematical properties of octree structures and Morton encoding to resolve rendering sequence issues yields a simple, efficient, and correct solution.
3. **Seamless compatibility with classical 3D processing**: Sparse voxels can directly undergo Volume Fusion, Voxel Pooling, and Marching Cubes, facilitating downstream applications of radiance fields like semantic understanding and mesh extraction.

## Limitations & Future Work

- Voxel representations are inherently discrete and may be inferior to continuous representations on extremely smooth surfaces.
- The $K$-point sampling is an approximation inside the voxels, limiting integration accuracy within large voxels.
- Scalability has not been validated on extremely large-scale (city-level) scenes.
- Future work could integrate SDF instead of density fields to achieve superior surface reconstruction.

## Related Work & Insights

- **vs 3DGS**: 3DGS employs Gaussian primitives + rasterization, which is fast but suffers from popping artifacts and volumetric ambiguity; SVRaster utilizes voxels + custom rasterization, achieving comparable speed without artifacts.
- **vs Plenoxels**: Both are neural-network-free voxel methods, but SVRaster's adaptive hierarchy and rasterizer are much more efficient than Plenoxels' uniform voxels and ray casting.
- **vs iNGP**: iNGP uses hash grids + a small MLP, whereas SVRaster is completely neural-network-free and renders faster.
- **Insight**: Scene representations do not necessarily have to be novel—classical voxels paired with an excellent rendering algorithm can still achieve SOTA performance.

## Rating

- **Novelty**: 8/10 — Pioneering "voxel + rasterization" direction, with an elegant Morton sorting scheme.
- **Experimental Thoroughness**: 8/10 — Full coverage of major datasets, comprehensive comparisons with SOTA, and demonstrations of downstream applications.
- **Writing Quality**: 9/10 — Clear methodology descriptions, intuitive illustrations, and an highly educational analysis of the sorting problems.
- **Value**: 9/10 — Provides a new paradigm for radiance field rendering that balances speed, quality, and physical correctness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[CVPR 2025\] Volumetrically Consistent 3D Gaussian Rasterization](volumetrically_consistent_3d_gaussian_rasterization.md)
- [\[CVPR 2025\] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [\[ECCV 2024\] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](../../ECCV2024/3d_vision/citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)

</div>

<!-- RELATED:END -->
