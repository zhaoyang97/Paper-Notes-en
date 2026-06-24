---
title: >-
  [Paper Note] Geometry Field Splatting with Gaussian Surfels
description: >-
  [CVPR 2025][3D Vision][Gaussian Surfel] This paper introduces the Geometry Field theory into the Gaussian Surfel framework, deriving an efficient and near-exact differentiable rendering algorithm for opaque surface reconstruction. It simultaneously resolves the loss discontinuity issue when surfels aggregate and employs a latent representation based on reflection vectors to better handle specular surfaces.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gaussian Surfel"
  - "Geometry Field"
  - "Surface Reconstruction"
  - "Differentiable Rendering"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: f36ba77e62cd6616
---

# Geometry Field Splatting with Gaussian Surfels

**Conference**: CVPR 2025  
**arXiv**: [2411.17067](https://arxiv.org/abs/2411.17067)  
**Code**: None  
**Area**: 3D Vision / Surface Reconstruction  
**Keywords**: Gaussian Surfel, Geometry Field, Surface Reconstruction, Differentiable Rendering, Novel View Synthesis

## TL;DR
This paper introduces the Geometry Field theory into the Gaussian Surfel framework, deriving an efficient and near-exact differentiable rendering algorithm for opaque surface reconstruction. It simultaneously resolves the loss discontinuity issue when surfels aggregate and employs a latent representation based on reflection vectors to better handle specular surfaces.

## Background & Motivation

**Background**: Reconstructing the geometry of opaque surfaces from multi-view images is a classic problem in computer vision. Recently, NeRF and 3D Gaussian Splatting (3DGS) have revolutionized volume rendering and radiance fields—3DGS achieves rapid novel view synthesis through the splatting of Gaussian kernels, while 2DGS/Gaussian Surfel methods degenerate 3D Gaussians into 2D planar surfels to better represent surfaces.

**Limitations of Prior Work**: (1) 3DGS and its variants use volume density to model opaque surfaces, but volume density is a "smoke" model, which is only an approximation for opaque solids and leads to noise and artificial thickness in extracted surfaces; (2) current surfel rendering algorithms involve several approximations—such as Taylor series expansion and ignoring self-attenuation—which degrade gradient accuracy; (3) when multiple surfels cluster near the same surface, the gradient of the rendered color with respect to surfel colors exhibits discontinuous jumps, causing optimization instability.

**Key Challenge**: Opaque surfaces require "surfaces" rather than "volumes"—volume density models are inherently unsuitable for accurate surface reconstruction; while surfel methods point in the right direction, current rendering approximations and discontinuities limit reconstruction accuracy.

**Goal**: To integrate the Geometry Field theory with Gaussian Surfels to derive an accurate, gradient-continuous differentiable rendering algorithm, thereby achieving high-quality opaque surface reconstruction.

**Key Insight**: The authors start from the Geometry Field (a recently proposed theoretical framework for stochastic opaque surface modeling). The geometry field represents opaque surfaces as a special form of volume density, which can be converted to volume density but models surfaces more accurately. The key observation is that using Gaussian Surfels to parameterize the geometry field (instead of volume density) naturally yields a surface-oriented inductive bias.

**Core Idea**: Splat the geometry field instead of volume density using Gaussian Surfels, and derive efficient, accurate rendering formulations to resolve approximation errors and gradient discontinuities.

## Method

### Overall Architecture
Given multi-view images, the model parameterizes the scene's geometry field using a set of Gaussian Surfels (2D Gaussian planar elements). Each surfel is characterized by attributes such as position, normal, scale, and color/appearance. The image is rendered from an arbitrary viewpoint using the geometry field splatting formulation, compared with ground-truth images to calculate losses, and the surfel parameters are optimized via backpropagation. Finally, the surface is extracted via the zero level-set of the geometry field.

### Key Designs

1. **Geometry Field Splatting Formula**:

    - Function: Integrates the geometry field theory with the Gaussian surfel representation, deriving an accurate and efficient differentiable rendering algorithm.
    - Mechanism: Traditional 3DGS methods use alpha-compositing to accumulate the contributions of Gaussian kernels, where the alpha values are based on an approximation of volume density. This paper derives rendering equations directly from the geometry field. A geometry field $G(\mathbf{x})$ is defined in 3D space, where its value indicates the probability of point $\mathbf{x}$ being inside the surface. For a geometry field parameterized by Gaussian surfels, the geometry field values along a ray can be computed analytically. When rendering pixel colors, the derivative of the geometry field along the ray is integrated: $C = \int_0^{\infty} G'(t) \cdot c(t) dt$, where $G'(t)$ is the derivative of the geometry field (i.e., the probability density of the surface) at ray parameter $t$, and $c(t)$ is the color at that position. This formula can be efficiently computed for Gaussian Surfels and eliminates Taylor expansion approximations as well as the assumption of ignoring self-attenuation present in current methods.
    - Design Motivation: Exact rendering formulas imply more accurate gradient signals, which directly improve optimization convergence speed and final reconstruction quality. Removing approximations also eliminates surface noise caused by approximation errors.

2. **Color Continuity Guarantee**:

    - Function: Ensures that the rendered color is a continuous function of surfel colors when surfels aggregate close to the surface, avoiding optimization jumps.
    - Mechanism: When two surfels are extremely close along a ray, their depth order can flip due to tiny perturbations, causing a sudden jump in the rendered color (since alpha-compositing is order-dependent). By analytically translating the geometry field rendering integral, the authors prove that rendered colors are guaranteed to be continuous with respect to surfel color attributes under the geometry field framework. Specifically, the method uses a weighted average instead of hard sorting for the color contributions of overlapping surfels along the ray direction: $c_{\text{blend}} = \sum_i w_i c_i$, where the weights $w_i$ are determined by the geometry field gradient (instead of order-dependent alpha) and are therefore continuous.
    - Design Motivation: Discontinuities in loss functions are highly detrimental to gradient optimization, causing oscillations and non-convergence. Ensuring continuity stabilizes the optimization process, especially in detailed surface regions with high surfel density.

3. **Latent Reflection Vector Representation**:

    - Function: Better models the view-dependent appearance of specular/reflective surfaces.
    - Mechanism: Traditional 3DGS uses spherical harmonics (SH) to encode view-dependent appearance, but SH is expanded on spherical coordinates (view direction). This paper instead uses the reflection vector as the parameterization basis for SH—the reflection direction of the view line is calculated using the surfel normal as $\mathbf{r} = 2(\mathbf{n} \cdot \mathbf{v})\mathbf{n} - \mathbf{v}$, and the SH is expanded over this reflection vector. Furthermore, instead of directly predicting RGB colors, the method predicts a latent feature vector, which is then decoded into the final color via a small MLP.
    - Design Motivation: Directional features of specular reflections correlate much more strongly with the reflection vector than with the view direction. High-order SH is required on the view direction to fit sharp specular highlights, whereas low-order is sufficient on the reflection direction. The latent representation further enhances representation capability.

### Loss & Training
- L1 color reconstruction loss + SSIM loss: $L = (1-\lambda)L_1 + \lambda L_{\text{SSIM}}$
- Normal consistency regularization: encouraging adjacent surfels to share consistent normal directions.
- Depth regularization (optional): used when depth priors are available.
- Adaptive density control strategy identical to 3DGS (clone/split/prune).

## Key Experimental Results

### Main Results
**DTU Dataset (surface reconstruction quality, Chamfer Distance mm)**:

| Method | CD↓ | PSNR↑ | SSIM↑ |
|------|-----|-------|-------|
| NeuS | 0.89 | 31.0 | 0.95 |
| 3DGS | 1.52 | 34.2 | **0.97** |
| 2DGS | 0.76 | 32.8 | 0.96 |
| GOF | 0.72 | 33.1 | 0.96 |
| **GFSplatting (Ours)** | **0.63** | **33.5** | 0.96 |

**Mip-NeRF 360 Dataset (novel view synthesis quality)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS | 27.4 | 0.81 | 0.22 |
| 2DGS | 27.0 | 0.80 | 0.23 |
| **GFSplatting** | **27.6** | **0.82** | **0.21** |

### Ablation Study

| Configuration | DTU CD↓ | PSNR↑ | Description |
|------|---------|-------|------|
| Full model | 0.63 | 33.5 | Full model |
| Using traditional volume density splatting | 0.76 | 33.0 | Degrades to a method similar to 2DGS |
| w/o color continuity | 0.70 | 33.1 | Reconstruction degrades at surface details |
| w/o reflection vector representation | 0.65 | 32.8 | Rendering quality drops on specular surfaces |
| SH color (non-latent) | 0.64 | 33.0 | Latent representation provides additional expressiveness |

### Key Findings
- **Geometry field splatting is the largest contributor**: Compared to traditional volume density splatting (2DGS), Chamfer Distance on DTU drops from 0.76 to 0.63, a 17% improvement.
- **Color continuity guarantee** yields noticeable improvements in high-density surfel regions (e.g., fine geometric structures); removing it increases the CD by 11%.
- **Reflection vector representation** is primarily advantageous on specular objects, contributing minimally to diffuse surfaces.
- The overall method significantly outperforms existing GS methods in surface reconstruction quality (CD) while maintaining competitive rendering quality.

## Highlights & Insights
- **From theory to practice**—The geometry field theory is inherently an elegant mathematical framework, which this paper successfully instantiates into a practical differentiable rendering algorithm. Theoretical exactness directly translates to empirical gains.
- **In-depth continuity analysis**—Pointing out and resolving the optimization issues caused by surfel ordering discontinuities—a previously overlooked but practically significant problem.
- **Use of reflection vectors** is not entirely new, but its combination with a latent representation is a first in the Gaussian Splatting framework, making it easily adoptable by other GS methods.

## Limitations & Future Work
- Although surface reconstruction quality surpasses 2DGS/GOF, the improvement in novel view synthesis quality is limited (PSNR increases by only ~0.5dB).
- Geometry field theory currently assumes opaque surfaces, making it inapplicable to semi-transparent objects (like smoke or glass).
- Computational overhead is slightly larger than 3DGS (due to more precise rendering formulas), though real-time rendering rates are still maintained.
- The accuracy of normal estimation directly impacts the performance of the reflection vector representation; inaccurate normals lead to inaccurate reflection directions.

## Related Work & Insights
- **vs 2DGS**: 2DGS also uses surfels but is based on volume density splatting. This paper is theoretically more exact (geometry field vs volume density) and demonstrates superior reconstruction quality experimentally.
- **vs GOF (Gaussian Opacity Field)**: GOF also explores methods to extract better surfaces from 3DGS but remains within the volume density framework. GFSplatting's geometry field formulation is better suited for opaque surfaces.
- **vs NeuS/VolSDF**: These SDF-based implicit methods are also designed for surface reconstruction but are much slower to compute. GFSplatting inherits the fast rendering advantages of GS.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces geometry field theory into the GS framework, yielding solid theoretical contributions and in-depth rendering formula derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Standard evaluation on DTU + Mip-NeRF 360, with ablations covering all modules.
- Writing Quality: ⭐⭐⭐⭐ High mathematical rigor and clear derivations, though the theoretical portions have a high entry barrier.
- Value: ⭐⭐⭐⭐ Advances the theoretical foundation of GS surface reconstruction, though the empirical gains are moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2025\] 3D-HGS: 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)

</div>

<!-- RELATED:END -->
