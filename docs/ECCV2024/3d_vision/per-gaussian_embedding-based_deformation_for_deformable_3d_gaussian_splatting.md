---
title: >-
  [Paper Note] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][Deformable 3DGS] This paper proposes a deformation representation method based on Per-Gaussian Embedding, which defines deformation as a function of per-Gaussian latent embeddings and temporal embeddings. Combined with coarse-to-fine deformation decomposition and local smoothness regularization, it achieves comprehensive advantages in quality, speed, and model capacity across multiple dynamic scene datasets.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Deformable 3DGS"
  - "Dynamic Scene Reconstruction"
  - "Per-Gaussian Embedding"
  - "Coarse-to-Fine Deformation Decomposition"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: a80563f2147dd434
---

# Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2404.03613](https://arxiv.org/abs/2404.03613)  
**Code**: [Project Page](https://jeongminb.github.io/e-d3dgs/)  
**Area**: 3D Vision  
**Keywords**: Deformable 3DGS, Dynamic Scene Reconstruction, Per-Gaussian Embedding, Coarse-to-Fine Deformation Decomposition, Novel View Synthesis  

## TL;DR

This paper proposes a deformation representation method based on Per-Gaussian Embedding, which defines deformation as a function of per-Gaussian latent embeddings and temporal embeddings. Combined with coarse-to-fine deformation decomposition and local smoothness regularization, it achieves comprehensive advantages in quality, speed, and model capacity across multiple dynamic scene datasets.

## Background & Motivation

1. 3D Gaussian Splatting (3DGS) has achieved real-time, high-quality novel view synthesis leveraging differentiable rasterization.
2. A natural extension of 3DGS to dynamic scenes is to define a canonical 3DGS combined with a deformation field that deforms them to each frame.
3. Existing methods (4DGaussians, D3DGS) model the deformation field as a coordinate-based function $f(x, y, z, t)$.

**Core Problem**: 3DGS is inherently a mixture of multiple Gaussians rather than a single continuous field. It is unreasonable to use a single coordinate network to predict deformations for all Gaussians.
5. Coordinate-based methods are limited by the resolution and model capacity of feature grids; doubling the grid resolution yields only minor improvements.
6. In complex dynamic scenes (e.g., fast-moving hands, reflective surfaces), existing methods produce blurry or distorted results.

## Method

### Overall Architecture

The E-D3DGS framework comprises three core components:

1. **Per-Gaussian Embedding**: Assigns an independent learnable latent embedding vector to each 3D Gaussian.
2. **Coarse-to-Fine Deformation Decomposition**: Models slow/large-scale motion and fast/fine-grained motion separately using two decoders.
3. **Local Smoothness Regularization**: Encourages neighboring Gaussians to have similar embeddings, thereby yielding similar deformations.

### Key Designs

#### Module 1: Embedding-Based Deformation

Departing from coordinate-based deformation fields, a deformation function is introduced that takes the per-Gaussian embedding $\mathbf{z}_g \in \mathbb{R}^{32}$ and temporal embedding $\mathbf{z}_t \in \mathbb{R}^{256}$ as inputs:

$$\mathcal{F}_\theta: (\mathbf{z}_g, \mathbf{z}_t) \rightarrow (\Delta\mathbf{x}, \Delta\mathbf{r}, \Delta\mathbf{s}, \Delta\sigma, \Delta Y)$$

Where $\Delta\mathbf{x}$ is the position offset, $\Delta\mathbf{r}$ is the change in rotation quaternion, $\Delta\mathbf{s}$ is the scale change, $\Delta\sigma$ is the opacity change, and $\Delta Y$ is the change in Spherical Harmonics coefficients. $\mathcal{F}_\theta$ is implemented as a shallow MLP followed by parameter-specific MLP heads.

The parameters of each Gaussian at frame $t$ are obtained by adding the predicted deformation to the canonical parameters. The key advantage of this design is that each Gaussian possesses an independent embedding to encode its temporal variation characteristics without relying on spatial coordinates.

#### Module 2: Coarse-to-Fine Deformation Decomposition

Different parts of a dynamic scene exhibit motion with varying velocities (e.g., a slowly moving body vs. a quickly stirring arm). To address this, temporal embeddings are split into coarse and fine scales:

**Temporal Embedding Grid**: Starts as a 1D feature grid $Z \in \mathbb{R}^{N \times 256}$ ($N$ frames).

- **Fine Deformation Temporal Embedding**: $\mathbf{z}_t^f = \text{interp}(Z, t)$ using the original resolution.
- **Coarse Deformation Temporal Embedding**: Linearly downsamples $Z$ by 5 times to remove high frequencies, then interpolates to obtain $\mathbf{z}_t^c$.

The final deformation is the sum of the outputs from two decoders:

$$\Delta = \mathcal{F}_{\theta_c}(\mathbf{z}_g, \mathbf{z}_t^c) + \mathcal{F}_{\theta_f}(\mathbf{z}_g, \mathbf{z}_t^f)$$

- **Coarse Deformation Decoder** $\mathcal{F}_{\theta_c}$: Responsible for large-scale/slow motion (e.g., head and torso movement).
- **Fine Deformation Decoder** $\mathcal{F}_{\theta_f}$: Responsible for fast/fine-grained motion (e.g., arms, tongs, shadows).

During early training, $Z_t^f$ starts with the same $N/5$ resolution as $Z_t^c$ and progressively increases to resolution $N$ over 10K iterations.

#### Module 3: Local Smoothness Regularization

Neighboring Gaussians forming dynamic objects should exhibit locally similar deformations. This is implicitly enforced by constraining the local smoothness of per-Gaussian embeddings:

$$\mathcal{L}^{\text{emb\_reg}} = \frac{1}{k|\mathcal{S}|} \sum_{i \in \mathcal{S}} \sum_{j \in \text{KNN}_{i;k}} w_{i,j} \|\mathbf{z}_{g_i} - \mathbf{z}_{g_j}\|_2$$

Where the weight factor $w_{i,j} = \exp(-\lambda_w \|\mu_j - \mu_i\|_2^2)$, $\lambda_w = 2000$, and $k = 20$. The KNN sets are only updated during densification to reduce computational overhead.

Unlike methods that directly constrain physical properties (rigidity, rotation), this work implicitly constrains deformation consistency through smoothness in the embedding space, which allows for better capture of the textures and details of dynamic objects.

### Loss & Training

- **Rendering Loss**: L1 + periodic DSSIM.
- Opacity regularization weight is set to $1.0 \times 10^{-4}$, and the original opacity reset of 3DGS is disabled.
- Decoder MLP: 128 hidden units, 2 layers.
- Decoder learning rate exponentially decays from $1.6 \times 10^{-4}$ to $1.6 \times 10^{-5}$.
- Per-Gaussian embedding learning rate is set to $2.5 \times 10^{-3}$.
- **Efficient Training Strategy**: Error-based frame sampling + uniform camera view coverage + periodic DSSIM densification.

## Key Experimental Results

### Main Results

**Neural 3D Video Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Model Size↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| DyNeRF | 29.58 | - | 0.083 | 0.01 | 56 MB |
| K-Planes | 30.86 | 0.939 | 0.096 | 0.13 | 309 MB |
| 4DGS | 31.19 | 0.940 | 0.051 | 33.7 | 8700 MB |
| 4DGaussians | 30.71 | 0.935 | 0.056 | 51.9 | 59 MB |
| **Ours** | **31.31** | **0.945** | **0.037** | **74.5** | **35 MB** |

**Technicolor Light Field Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Model Size↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| DyNeRF | 31.80 | - | 0.140 | 0.02 | 0.6 MB |
| HyperReel | 32.32 | 0.899 | 0.118 | 0.91 | 289 MB |
| 4DGaussians | 29.62 | 0.844 | 0.176 | 34.8 | 51 MB |
| **Ours** | **33.24** | **0.907** | **0.100** | **60.8** | **77 MB** |

**HyperNeRF Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Model Size↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| HyperNeRF | 22.29 | 0.598 | 0.153 | <1 | 15 MB |
| D3DGS | 22.40 | 0.598 | 0.275 | 6.95 | 309 MB |
| 4DGaussians | 25.03 | 0.682 | 0.281 | 96.3 | 60 MB |
| **Ours** | **25.43** | **0.697** | **0.231** | **139.3** | **33 MB** |

### Ablation Study

**Ablation of Coarse-to-Fine Deformation Decomposition**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| **Ours (Coarse+Fine)** | **29.70** | **0.933** | **0.041** |
| Coarse Only | 29.48 | 0.931 | 0.044 |
| Fine Only | 29.23 | 0.932 | 0.043 |
| Ours + Coord Injection | 29.60 | 0.931 | 0.045 |

**Ablation of Local Smoothness Regularization**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| w/o Embedding Reg | 32.26 | 0.951 | 0.037 |
| **+ Embedding Reg (Ours)** | **32.34** | **0.952** | **0.036** |
| + Physical Constraint Reg | 32.08 | 0.950 | 0.036 |

### Key Findings

1. **Coordinate Injection is Harmful**: Injecting Gaussian coordinates $\mathbf{x}$ as additional inputs to the decoder degrades quality (29.60 vs 29.70), validating the argument that coordinate dependency should be discarded.
2. **Coarse and Fine Decompositions are Complementary**: Using only the coarse or only the fine decoder results in blurry dynamic regions; utilizing both yields the best performance.
3. **Embedding Regularization Outperforms Physical Regularization**: The implicit embedding smoothness constraint (+0.08 PSNR) outperforms direct physical rigidity constraints (-0.18 PSNR), as the latter excessively restricts deformation flexibility.
4. **Minor Impact of Embedding Dimensions**: Varying $\mathbf{z}_g$ from 16 to 64 dimensions, and $\mathbf{z}_t$ from 128 to 512 dimensions, results in performance variations of less than 0.18 PSNR.
5. **Comprehensive Advantages**: Simultaneously achieves the best trade-off among quality, rendering speed, and model size across three datasets.

## Highlights & Insights

- Grounded in the essence of 3DGS (a mixture of Gaussians rather than a single continuous field), defining independent deformations for each Gaussian provides a clear and logical design.
- Coarse-to-fine deformation decomposition is achieved via simple downsampling of the temporal grid, avoiding complex multi-scale architectures.
- The model requires only 35MB (on Neural 3D Video), which is significantly smaller than the 8700MB of 4DGS, demonstrating the high efficiency of embedding representations.
- The rendering speed reaches 74.5 FPS, far exceeding NeRF baselines and several 3DGS baselines.

## Limitations & Future Work

- Performance remains sub-optimal on casually captured monocular videos (e.g., the challenging setups in HyperNeRF).
- The memory footprint of per-Gaussian embeddings scales linearly with the number of Gaussians.
- The KNN neighbor search is only updated during densification, which may cause inconsistencies in fast-deforming scenes.
- Initial point clouds reconstructed via COLMAP are required, indicating a dependency on initialization quality.

## Related Work & Insights

- **4DGaussians**: Uses HexPlane to decode features for temporal deformation; a representative coordinate-dependent method.
- **D3DGS**: Uses implicit functions to handle time and position; also a deformable 3DGS method.
- **Nerfies / HyperNeRF**: Employs frame-by-frame deformation codes, but developed for NeRF instead of 3DGS.
- **Dynamic 3D Gaussians**: Introduces local rigidity regularization, which inspired the local smoothness regularization in this work.
- **Insights**: The concept of per-element embedding can be extended to other discrete element-based representations (e.g., 3D mesh deformation).

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)

</div>

<!-- RELATED:END -->
