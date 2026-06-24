---
title: >-
  [Paper Note] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views
description: >-
  [CVPR 2025][3D Vision][Sparse-view reconstruction] MAtCha Gaussians is proposed to model scene surfaces as an atlas of charts on 2D manifolds rendered via 2D Gaussian Surfels. By leveraging monocular depth initialization, a lightweight neural deformation model, and structure-preserving loss, it achieves high-quality surface mesh reconstruction and photorealistic novel view synthesis simultaneously in minutes under only 3-10 sparse views.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse-view reconstruction"
  - "3D Gaussian Splatting"
  - "surface reconstruction"
  - "monocular depth prior"
  - "manifold representation"
date: 2026-05-08
content_hash: d60a222e5ea3b3cb
---

# MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views

**Conference**: CVPR 2025  
**arXiv**: [2412.06767](https://arxiv.org/abs/2412.06767)  
**Code**: [https://anttwo.github.io/matcha/](https://anttwo.github.io/matcha/)  
**Area**: 3D Reconstruction / Novel View Synthesis  
**Keywords**: Sparse-view reconstruction, 3D Gaussian Splatting, surface reconstruction, monocular depth prior, manifold representation

## TL;DR

MAtCha Gaussians is proposed to model scene surfaces as an atlas of charts on 2D manifolds rendered via 2D Gaussian Surfels. By leveraging monocular depth initialization, a lightweight neural deformation model, and structure-preserving loss, it achieves high-quality surface mesh reconstruction and photorealistic novel view synthesis simultaneously in minutes under only 3-10 sparse views.

## Background & Motivation

**Background**: NeRF and 3D Gaussian Splatting achieve excellent novel view synthesis under dense views. However, the learned geometric representations are inherently optimized for 2D rendering, yielding rough surface quality where multiple Gaussians float around the surface to mimic the appearance. Extracting explicit geometry (TSDF/Marching Cubes) from volumetric representations is a post-processing step that inevitably loses high-frequency details.

**Limitations of Prior Work**: (1) Existing methods (NeRF/3DGS) require dense view sampling, whereas geometric reconstruction theoretically needs only a few views; (2) The low-pass filtering nature of volume rendering fails to recover sharp corners and edges; (3) Existing sparse-view methods (SparseNeus, Spurfies) rely on feed-forward networks trained on limited datasets, making them difficult to generalize to unbounded scenes; (4) Although 2D Gaussian Surfel methods fit surfaces better, their massive degrees of freedom lead to geometric divergence under sparse views.

**Key Challenge**: How to simultaneously achieve photorealism (requiring the flexibility of volume rendering) and accurate geometry (requiring explicit surface constraints) in a single model from an extremely small number of images?

**Key Insight**: Explicitly model the surface manifold using 2D charts (i.e., 3D point clouds back-projected from monocular depth). Gaussian Surfels are instantiated on these charts on-the-fly for differentiable rendering. Charts constrain the Gaussians from diverging, while Gaussian rendering provides optimization gradients to the charts—creating a mutually beneficial cycle.

## Method

### Overall Architecture

The input consists of $N$ sparse RGB images and camera parameters estimated by MASt3R-SfM. The pipeline consists of three steps: (1) Initialize $n \leq N$ charts using a monocular depth model (DepthAnythingV2); (2) Align the charts to SfM point clouds via a neural deformation model while preserving the high-frequency structures of the depth maps; (3) Instantiate 2D Gaussian Surfels on the charts for differentiable rendering refinement. Finally, a unified mesh is extracted from the Gaussians using an improved adaptive tetrahedralization method.

### Key Designs

1. **Lightweight Neural Chart Deformation Model (Chart Encodings + Depth Encodings)**:
    - Each chart maintains a sparse 2D feature grid $E_i \in \mathbb{R}^{rh \times rw \times d}$ (where $r$ is the downscaling factor) along with a 1D feature $z_i(d(u))$ along the depth axis.
    - Deformation field: $\Delta_i(u) = f_{\theta_i}[E_i(u) + z_i(d(u))]$, decoded into 3D displacements by a small MLP.
    - **2D grid sparsity** ensures that the deformation contains only low-frequency components, thereby preserving the high-frequency details of the monocular depth.
    - **Depth encoding** allows points at different depths to deform independently, handling scale inconsistencies between foreground and background objects.
    - **Design Motivation**: Global affine scaling is too coarse (different objects have different scales), while pixel-wise scaling is over-parameterized under sparse views, leading to a loss of high frequencies.

2. **Multi-Loss Guided Chart Alignment**:
    - **Fitting Loss $\mathcal{L}_{fit}$**: Encourages charts to fit SfM point clouds, accompanied by a learnable confidence map to handle SfM outliers.
    - **Structure Loss $\mathcal{L}_{struct}$**: Constrains the normals and mean curvatures of the deformed charts to match those of the initial depth maps, preserving high-frequency structures: $(1 - N_i \cdot N_i^{(0)}) + \frac{1}{4}\|M_i - M_i^{(0)}\|_1$.
    - **Mutual Alignment Loss $\mathcal{L}_{align}$**: Encourages overlapping regions of different charts to align, forming a coherent manifold.
    - **Design Motivation**: The three losses address three distinct geometric demands: "matching observations", "preserving details", and "multi-view consistency".

3. **Gaussian Surfel Rendering on Charts and Adaptive Mesh Extraction**:
    - Generate 2D Gaussian Surfels on-the-fly on the charts (where positions/covariances are determined by the vertices, and colors/opacities are learnable textures).
    - The support region of a Gaussian Surfel is larger than a triangle, enabling adaptive Gaussian-blur-like gradient propagation, which is superior to triangle rasterization.
    - Mesh extraction: An improved version of GOF's adaptive tetrahedralization is used, defining a binary opacity field via depth maps + adaptive dilation to prevent geometric erosion.
    - **Design Motivation**: Gaussian Surfel rendering provides better gradients than triangle rendering under sparse views. Tetrahedralization recovers more complete foreground and background structures than TSDF.

### Loss & Training

- **Alignment Phase**: $\mathcal{L} = \mathcal{L}_{fit} + \lambda_{struct}\mathcal{L}_{struct} + \lambda_{align}\mathcal{L}_{align}$ (with $\lambda_{struct}=4, \lambda_{align}=5$).
- **Rendering Refinement Phase**: Photometric loss (L1 + SSIM) + 2DGS regularization + confidence-weighted structure loss.

## Key Experimental Results

### Main Results

**DTU Surface Reconstruction (3 views, Chamfer Distance mm ↓):**

| Method | Scan 21 | Scan 24 | Scan 37 | Scan 110 | Mean |
|------|---------|---------|---------|----------|------|
| Spurfies | 2.36 | 1.12 | 2.39 | 1.14 | 1.36 |
| 2DGS+MASt3R-SfM | 1.43 | 1.29 | 2.79 | 2.26 | 1.79 |
| **MAtCha (Ours)** | **1.27** | **0.88** | **1.89** | **0.87** | **1.04** |

- Using only 3 views achieves a 24% improvement over SOTA (Spurfies) (1.04 vs 1.36).

**Mip-NeRF 360 Novel View Synthesis (5 views):**

| Method | 10%Q PSNR | Avg PSNR |
|------|-----------|----------|
| 2DGS+MASt3R-SfM | 15.37 | 20.84 |
| GOF+MASt3R-SfM | 15.78 | 21.24 |
| **MAtCha (Ours)** | **18.18** | **21.90** |

### Ablation Study

| Configuration | DTU CD↓ | MipNeRF360 PSNR↑ |
|------|---------|-------------------|
| No Charts Encodings | 2.693 | 16.37 |
| No Depth Encodings | 1.601 | 17.38 |
| No $\mathcal{L}_{struct}$ | 1.716 | 17.00 |
| No $\mathcal{L}_{align}$ | 1.565 | 17.33 |
| **Full Model** | **1.04** | **17.59** |

- Charts Encodings is the most critical component (excluding it worsens CD by $2.6\times$).
- The structure loss is crucial for both geometry and rendering quality.

### Key Findings

- Under sparse views, MAtCha achieves SOTA performance in both surface reconstruction and novel view synthesis, simultaneously realizing these two seemingly contradictory goals.
- Reconstruction time: $<3$ min for alignment, 5-10 min for refinement, which is significantly faster than other methods that require hours.
- Adaptive tetrahedralization vs. TSDF: TSDF tends to erode geometry, producing holes and "disk-like aliasing" artifacts, while tetrahedralization recovers complete foreground and background surfaces.
- Convincing geometry and rendering results can be generated with only 3 views.

## Highlights & Insights

- **Representation Innovation**: Explicitly modeling scene surfaces as an atlas of 2D manifold charts is an elegant geometric design that turns high-precision monocular depth information into an asset rather than a burden.
- **Exquisite Dual-Encoding Deformation Model**: A sparse 2D grid controls low-frequency deformation to preserve high frequencies, while a 1D depth encoding handles discontinuities between foreground and background. This achieves extremely high parameter efficiency and excellent generalization.
- **Mutually Beneficial Architecture**: Charts constrain Gaussians from drifting (solving the Gaussian divergence problem under sparse views), while Gaussian rendering provides gradients for the charts (avoiding gradient issues inherent in triangle rendering)—each acting as the solution for the other.
- **Physical Intuition of Structure Loss**: Preserving normals and curvatures corresponds to maintaining local geometric invariants, which is more reasonable than directly constraining depth values.

## Limitations & Future Work

- The 2D Gaussian rasterizer requires the assumption of a centered optical center, which necessitates image cropping in some scenes, resulting in incomplete reconstructions.
- Chart alignment may fail when complex occlusion relationships occur at the foreground/background boundaries.
- Dynamic scenes and deformable objects are not yet supported.
- The performance relies on the quality of camera poses estimated by MASt3R-SfM.

## Related Work & Insights

- **Relationship with 2DGS/GOF**: 2DGS provides the Gaussian Surfel representation, and GOF introduces adaptive tetrahedralization. MAtCha integrates these into a chart framework and resolves the core issues under sparse views.
- **Monocular Depth Distillation**: DepthAnythingV2 provides high-frequency geometric priors, but scaling inconsistencies across multiple views must be resolved—the neural deformation model in this work offers a general solution.
- **Inspiration for 3D Reconstruction Pipelines**: Modularizing the workflow into three phases—initialization (depth prior) $\to$ coarse alignment (SfM points) $\to$ refinement (differentiable rendering)—using the most suitable method for each phase.

## Rating

⭐⭐⭐⭐⭐ — Representation innovation (mutually beneficial architecture of manifold charts + Gaussian Surfels), complete methodology (neural deformation model + three-loss alignment + differentiable refinement). It achieves SOTA geometric reconstruction and SOTA novel view synthesis simultaneously under extremely sparse views (3 images), with training completing in minutes. It holds the potential to become a new benchmark method for sparse-view 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_sparse_view_reconstruction.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [\[ECCV 2024\] Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction](../../ECCV2024/3d_vision/deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)

</div>

<!-- RELATED:END -->
