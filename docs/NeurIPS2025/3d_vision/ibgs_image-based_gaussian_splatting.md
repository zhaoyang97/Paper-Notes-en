---
title: >-
  [Paper Note] IBGS: Image-Based Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] This paper proposes Image-Based Gaussian Splatting (IBGS), which enhances standard 3DGS rendering quality by learning color residuals from neighboring training images. The method significantly improves the modeling of high-frequency details and view-dependent effects without introducing additional storage overhead.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Novel View Synthesis
  - Image-Based Rendering
  - Color Residual
  - View-Dependent Effects
date: 2026-05-08
content_hash: 23abec463aaa0c4d
---

# IBGS: Image-Based Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2511.14357](https://arxiv.org/abs/2511.14357)
**Code**: [GitHub](https://hoangchuongnguyen.github.io/ibgs)
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, Novel View Synthesis, Image-Based Rendering, Color Residual, View-Dependent Effects

## TL;DR

This paper proposes Image-Based Gaussian Splatting (IBGS), which enhances standard 3DGS rendering quality by learning color residuals from neighboring training images. The method significantly improves the modeling of high-frequency details and view-dependent effects without introducing additional storage overhead.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the dominant approach for novel view synthesis (NVS), attracting widespread attention for its high-quality rendering and fast optimization. However, each Gaussian can only represent a single color per viewpoint, and low-order spherical harmonics (SH) struggle to capture complex view-dependent effects.

**Limitations of Prior Work**: Existing improvements either employ global texture mapping (which fails in complex scenes) or per-Gaussian texture mapping (where storage cost grows quadratically with texture resolution), and neither adequately handles view-dependent effects.

**Key Challenge**: How can one simultaneously model high-frequency details and view-dependent effects without significantly increasing storage?

**Goal**: Leverage the high-frequency detail and viewpoint information already present in training images to enhance rendering.

**Key Insight**: Inspired by traditional image-based rendering (IBR) techniques, this work integrates 3DGS with image-based rendering.

**Core Idea**: Pixel color = base color from standard 3DGS + color residual learned from neighboring viewpoint images.

## Method

### Overall Architecture

IBGS models the final color of each pixel as the sum of two components:

$$\mathbf{c}^{\text{final}}(\mathbf{p}) = \underbrace{\sum_{i=1}^{N} w_i \Psi_l(\mathbf{h}_i, \mathbf{v}_i)}_{\text{Base color } \mathbf{c}(\mathbf{p})} + \underbrace{\mathcal{F}(\mathbf{c}(\mathbf{p}), \mathbf{d}(\mathbf{p}), \{\Delta\mathbf{c}_m\}_{m=1}^{M}, \{\Delta\mathbf{d}_m\}_{m=1}^{M})}_{\text{Color residual } \Delta\mathbf{c}(\mathbf{p})}$$

The base color is obtained from standard 3DGS rasterization, while the color residual is predicted by a lightweight network from neighboring source views.

### Key Designs

1. **Source View Feature Extraction**:

    - For target pixel $\mathbf{p}$, compute the intersection $\mathbf{x}_i(\mathbf{p})$ between the ray and each Gaussian.
    - The intersection is computed as the point where the ray meets the plane defined by the Gaussian center and normal:
    $\mathbf{x}_i(\mathbf{p}) = \mathbf{o} + \frac{\mathbf{n}_i^T(\boldsymbol{\mu}_i - \mathbf{o})}{\mathbf{n}_i^T \mathbf{d}(\mathbf{p})} \mathbf{d}(\mathbf{p})$
    - Only the $K$ median intersections — where cumulative transmittance is close to 0.5 — are projected, filtering out floater Gaussian noise.
    - These intersections are projected onto neighboring source views to obtain colors; the weighted average warped color minus the base color yields:
    $\Delta\mathbf{c}_m(\mathbf{p}) = \mathbf{c}_m^{\text{warp}}(\mathbf{p}) - \mathbf{c}(\mathbf{p})$

2. **Color Residual Prediction Network**:

    - A **PointNet-style** per-pixel feature extractor processes per-source-view color differences and camera difference features.
    - Multi-view features are aggregated via max-pooling to produce a feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times 32}$.
    - A **9-layer $3\times3$ convolutional decoder** predicts the color residual map $\Delta\mathbf{C}$.
    - The network is extremely lightweight and does not impact rendering speed.

3. **Exposure Correction Module**:

    - Addresses cross-view brightness inconsistencies caused by automatic exposure in modern cameras.
    - Assuming similar lighting conditions at nearby positions, an affine transformation matrix is fitted via least squares:
    $\mathbf{A}^{\star} = \arg\min_{\mathbf{A}} \sum_{\mathbf{p}} \left\| \mathbf{A} \begin{bmatrix} \mathbf{c}(\mathbf{p}) \\ 1 \end{bmatrix} - \mathbf{c}_1^{\text{warp}}(\mathbf{p}) \right\|_2^2$
    - Key advantage: generalizes to arbitrary novel views, overcoming the limitation of prior methods that can only correct training views.

4. **Visibility-Based Source View Selection**: Source views where the target point is occluded are excluded via depth consistency checking:
    $\frac{|z(\mathbf{x}(\mathbf{p})) - z(\mathbf{x}_s^{\text{warp}}(\mathbf{p}))|}{z(\mathbf{x}(\mathbf{p})) + z(\mathbf{x}_s^{\text{warp}}(\mathbf{p}))} \leq \tau$

### Loss & Training

The total loss consists of three terms:
$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_1 \mathcal{L}_{\text{photo}} + \lambda_2 \mathcal{L}_{\text{normal}}$$

- **Color rendering loss** $\mathcal{L}_{\text{rgb}}$: L1 + SSIM loss applied to both the base image and the final image, with weight $\gamma$ annealed from 1.0 to 0.5.
- **Multi-view photometric consistency loss** $\mathcal{L}_{\text{photo}}$: Enforces consistency between warped colors and ground-truth images, encouraging accurate pixel matching.
- **Normal consistency loss** $\mathcal{L}_{\text{normal}}$: Improves geometric quality.
- For the first 7,000 iterations, only the RGB loss is used; afterwards $\lambda_1=0.3$ and $\lambda_2=0.03$ are activated.
- SH degree $l=2$, median intersection count $K=4$, candidate source views $S=4$, visible source views $M=3$.

## Key Experimental Results

### Main Results

Comparisons on three standard NVS benchmarks: Mip-NeRF360, Tanks and Temples (TNT), and Deep Blending:

| Method | Mip-NeRF360 PSNR↑ | TNT PSNR↑ | Deep Blending PSNR↑ | TNT #Gauss(M) | TNT Mem(MB) |
|------|-------------------|-----------|---------------------|---------------|-------------|
| 3DGS | 27.69 | 23.11 | 29.53 | 1.75 | 415 |
| SuperGauss | 27.31 | 23.72 | 28.83 | 1.50 | 502 |
| TexturedGauss | 27.35 | 24.26 | 28.33 | - | - |
| **IBGS (Ours)** | **28.33** | **24.84** | **30.12** | **0.75** | **143** |

Results on the challenging Shiny dataset (specular highlights, reflections, CD diffraction):

| Scene | 3DGS PSNR | SuperGauss PSNR | **IBGS PSNR** |
|------|----------|-----------------|--------------|
| Guitars (specular highlights) | 29.37 | 30.43 | **35.65** |
| Lab (reflections) | 29.17 | 29.38 | **35.06** |
| CD (diffraction) | 29.10 | 29.49 | **35.23** |

PSNR improvements on the Shiny dataset exceed **5.2 dB**, with fewer Gaussians.

### Ablation Study

| Setting | TNT PSNR↑ | Mip-NeRF360 PSNR↑ |
|------|----------|-------------------|
| Full | **24.84** | **28.33** |
| Base color only (no residual) | 23.06 | 27.08 |
| Without photometric consistency loss | 24.70 | 28.31 |
| Full warped color instead of difference as input | 24.61 | 28.21 |
| Without exposure correction | 24.28 | - |

### Key Findings

1. The color residual module provides approximately **1.8 dB** PSNR gain on TNT and is the most critical component.
2. Using the difference $\Delta\mathbf{c}_m$ rather than the full warped color as network input yields better performance.
3. Under aggressive opacity pruning (threshold 0.05), IBGS suffers almost no quality loss while 3DGS degrades significantly — indicating that IBGS Gaussians are more concentrated near actual surfaces.
4. On Mip-NeRF360 and TNT, IBGS reduces the number of Gaussians by at least **62%** and storage by at least **42%**, while achieving superior quality.

## Highlights & Insights

- The decomposition into **base color + residual** is elegant: the base color handles the majority of appearance, while the residual compensates for details that SH cannot capture.
- Projecting only the median intersections cleverly filters noise from floater Gaussians while encouraging Gaussians to align with real surfaces.
- Exposure correction generalizes to novel views, resolving the limitation of existing methods that can only correct training views.
- The residual network — a 9-layer lightweight CNN combined with PointNet-style aggregation — achieves an excellent balance between efficiency and quality.
- The substantial 5+ dB gain on the Shiny dataset demonstrates a fundamental advantage in modeling view-dependent effects.

## Limitations & Future Work

1. Performance may degrade in **sparse-view settings**, as dense pixel correspondences are required to predict residuals.
2. Additional rendering computation results in **lower rendering speed** than vanilla 3DGS, with higher runtime memory usage.
3. The method relies on training images as a "texture source" and may degrade in regions not covered by the training viewpoints.
4. Can be combined with Gaussian compression/quantization methods to further reduce storage.

## Related Work & Insights

- **Traditional IBR** (Light Field, IBRNet): IBGS elegantly combines IBR's "pixel borrowing" concept with 3DGS's efficient rasterization.
- **2DGS**: IBGS adopts the median intersection and normal consistency loss designs from this work.
- **TexturedGaussian / SuperGaussian**: Per-Gaussian texture mapping methods with large storage overhead and inability to handle view-dependent effects.
- Insight: Leveraging existing observation images as a "texture library" in scene representation is a more efficient strategy than learning texture mappings.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of IBR and 3DGS is novel, and the color residual design is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets with detailed ablations and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with complete derivations.
- Value: ⭐⭐⭐⭐⭐ Achieves state-of-the-art results on multiple benchmarks with high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Anti-Aliased 2D Gaussian Splatting](anti-aliased_2d_gaussian_splatting.md)
- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)

<!-- RELATED:END -->
