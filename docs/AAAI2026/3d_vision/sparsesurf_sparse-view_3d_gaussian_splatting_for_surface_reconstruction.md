---
title: >-
  [Paper Note] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction
description: >-
  [AAAI 2026][3D Vision][Sparse-view] Proposes SparseSurf, which enhances geometric consistency under sparse views through Stereo Geometry-Texture Alignment and Pseudo-Feature Enhanced Geometry Consistency, simultaneously achieving high-precision surface reconstruction and high-quality novel view synthesis, achieving SOTA on DTU, BlendedMVS, and Mip-NeRF360 datasets.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Sparse-view"
  - "Surface reconstruction"
  - "Gaussian Splatting"
  - "Stereo matching"
  - "Multi-view consistency"
date: 2026-05-08
content_hash: 37a63ad50e032701
---

# SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction

**Conference**: AAAI 2026  
**arXiv**: [2511.14633](https://arxiv.org/abs/2511.14633)  
**Code**: [Project Page](https://miya-oi.github.io/SparseSurf-project)  
**Area**: 3D Vision  
**Keywords**: Sparse-view, Surface reconstruction, Gaussian Splatting, Stereo matching, Multi-view consistency

## TL;DR

Proposes SparseSurf, which enhances geometric consistency under sparse views through Stereo Geometry-Texture Alignment and Pseudo-Feature Enhanced Geometry Consistency, simultaneously achieving high-precision surface reconstruction and high-quality novel view synthesis, achieving SOTA on DTU, BlendedMVS, and Mip-NeRF360 datasets.

## Background & Motivation

3D Gaussian Splatting (3DGS) efficiently reconstructs high-quality surfaces under dense views, but easily overfits under sparse views, leading to a severe degradation in reconstruction quality. Existing methods face two key challenges:

**Challenge 1: Flattened Gaussians exacerbate overfitting**
- To better fit surface geometry, recent methods (FatesGS, Sparse2DGS) employ flattened 2D Gaussian primitives.
- However, flattening increases anisotropy, which instead exacerbates the risk of overfitting under sparse views.
- While no issues are apparent from the training views, the rendering quality under novel views significantly degenerates.

**Challenge 2: Limitations of monocular depth priors**
- Existing methods utilize monocular depth estimation as geometric constraints.
- However, monocular depth suffers from scale ambiguity and lacks confidence estimation.
- Under sparse views, the multi-view inconsistency introduced by noise becomes more severe.

The authors' core insight is to leverage stereo matching to provide metric-level supervision, and alleviate overfitting through multi-view feature consistency, thereby enabling mutual reinforcement between surface reconstruction and novel view synthesis.

## Method

### Overall Architecture

SparseSurf is based on flattened 3DGS (similar to PGSR/GaussianSurfels) and contains two core modules:
1. **Stereo Geometry-Texture Alignment**: Renders stereo view pairs to obtain metric-level depth priors via a pre-trained stereo matching network.
2. **Pseudo-Feature Enhanced Geometry Consistency**: Combines multi-view feature consistency across both training views and pseudo-unseen views.

### Key Designs

#### 1. **Stereo Geometry-Texture Alignment**: Connecting rendering quality and geometric estimation

The core idea is to leverage the excellent interpolation rendering capability of 3DGS to render stereo view pairs and obtain accurate metric-level geometric priors via a pre-trained stereo matching network.

**Stereo Prior Estimation**:
- For each training camera pose $\mathbf{P}_i$, generate a stereo view at a horizontal baseline $b$.
- Render stereo view images to form a stereo pair, and input them into a pre-trained stereo matching network to obtain a disparity map.
- Convert the disparity to depth $\mathcal{D}^*$ using the known baseline and focal length.
- Calculate normals $\mathcal{N}^*$ from the depth map.
- Generate a reliability mask $\mathcal{M}^*$ through stereo view consistency checks to filter out unreliable pixels.
- Periodically (every 300 iterations) re-render and update the priors during training.

**Stereo Geometry Supervision**:
$$\mathcal{L}_{depth} = \mathcal{L}_1(D, \mathcal{D}^*)$$
$$\mathcal{L}_{normal} = 1 - \mathcal{C}osine(N, \mathcal{N}^*)$$
$$\mathcal{L}_{nd} = 1 - \mathcal{C}osine(N_d, \mathcal{N}^*)$$

Additionally, introduce an edge-aware Laplacian smoothing loss:
$$\mathcal{L}_{smooth} = \mathcal{S}mooth(N, \mathcal{N}^*) + \mathcal{S}mooth(N_d, \mathcal{N}^*)$$

Total stereo loss:
$$\mathcal{L}_{stereo} = (\lambda_d \mathcal{L}_{depth} + \lambda_n \mathcal{L}_{normal} + \lambda_{nd} \mathcal{L}_{nd})\mathcal{M}^* + \lambda_s \mathcal{L}_{smooth}$$

**Design Motivation**: As training progresses, rendering quality improves $\rightarrow$ more accurate stereo depth priors $\rightarrow$ better geometric supervision $\rightarrow$ further improves rendering quality, forming a positive feedback loop.

#### 2. **Pseudo-Feature Enhanced Geometry Consistency**: Mitigating Overfitting

Includes two sub-modules:

**Pseudo-view Feature Consistency**:
- Appends feature attributes to each Gaussian primitive, learning multi-view feature representations from a frozen feature extraction model via feature distillation.
- Feature distillation loss: $\mathcal{L}_f = 1 - \mathcal{C}osine(F, \mathcal{F}^*)$
- Render feature maps at random pseudo-views, compute feature differences through bidirectional warping, and generate confidence masks.
- Adopt patch-level cosine similarity to avoid pixel-level noise contamination:

$$\mathcal{L}_{pseudo} = \sum_{i,j} \mathcal{M}_{feat}^{(i,j)} [1 - \mathcal{C}osine(\bar{\mathcal{F}}_{p2t}^{(i,j)}, \bar{\mathcal{F}}_r^{(i,j)})]$$

**Train-view Feature Alignment**:
- Utilize high-confidence features from training views to enforce multi-view consistency at the pixel level.
- $\mathcal{L}_{train} = 1 - \mathcal{C}osine(\mathcal{F}_{s2t}, \mathcal{F}_s)$

This joint constraint of "sparse training views + pseudo-unseen views" effectively mitigates the overfitting problem of flattened Gaussians under sparse views.

#### 3. **Multi-View Feature Representation**: Efficient Feature Distillation

Uses Vis-MVSNet to extract 8-dimensional multi-view features. The key design is to encode features into Gaussian attributes to avoid the computational overhead of re-extracting pseudo-view features at each iteration, keeping the entire pipeline efficient.

### Loss & Training

The total training loss includes the rendering loss, stereo loss, and feature consistency loss. The stereo prior is introduced from the 500th iteration and updated every 300 iterations to achieve progressive geometric guidance.

## Key Experimental Results

### Main Results (DTU Surface Reconstruction — Chamfer Distance↓)

| Method | Little-overlap Setup | Large-overlap Setup | Category |
|------|-------------------|-------------------|------|
| COLMAP | 2.61 | 1.52 | MVS |
| NeuSurf | 1.35 | 0.99 | Neural Implicit |
| FatesGS | 1.37 | 0.92 | GS Surface Reconstruction |
| 2DGS | 2.52 | 1.69 | GS Surface Reconstruction |
| Sparse2DGS | — | 1.13 | GS Surface Reconstruction |
| **SparseSurf** | **1.05** | **0.89** | GS Surface Reconstruction |

Achieves the optimal Chamfer Distance under both sparse-view settings on DTU.

### DTU Novel View Synthesis

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | AVGE↓ |
|------|-------|-------|--------|-------|
| CoR-GS | 19.21 | 0.853 | 0.119 | 0.082 |
| Binocular3DGS | 20.71 | 0.862 | 0.111 | — |
| NexusGS | 20.21 | 0.869 | 0.102 | 0.071 |
| **SparseSurf** | **21.31** | **0.886** | **0.089** | **0.067** |

Also achieves comprehensive superiority on novel view synthesis, demonstrating that surface reconstruction and rendering quality can be mutually enhanced.

### Ablation Study

| Configuration | Accuracy↓ | Completion↓ | Average CD↓ | Description |
|------|-----------|-------------|-------------|------|
| Baseline (without modules) | 1.318 | 2.302 | 1.810 | Baseline |
| + $L_{stereo}$ | 0.822 | 1.612 | 1.217 | Stereo constraint significantly improves performance |
| + $L_{stereo}$ + $L_{pseudo}$ | 0.610 | 1.327 | 0.969 | Pseudo-views further improve performance |
| + All ($L_{train}$) | **0.533** | **1.239** | **0.886** | Training view alignment provides additional gains |

### Key Findings

1. The stereo prior is the largest contributor to performance (CD decreases from 1.810 to 1.217, a 33% reduction).
2. Pseudo-view feature consistency effectively mitigates overfitting (CD drops from 1.217 to 0.969).
3. Train-view feature alignment provides additional robustness gains (0.969 $\rightarrow$ 0.886).
4. Patch-level feature consistency is more robust than pixel-level, preventing noise propagation.

## Highlights & Insights

1. **Synergy between Surface Reconstruction and Rendering**: Breaks the traditional trade-off of "better surface fitting $\rightarrow$ worse rendering".
2. **Positive Loop Design of Stereo Prior**: Improved rendering quality $\rightarrow$ better stereo prior $\rightarrow$ better geometry $\rightarrow$ further improved rendering, achieving self-reinforcement.
3. **Feature-level Supervision for Pseudo-views**: Compared to previous works that only use RGB or monocular depth to supervise pseudo-views, multi-view feature consistency constraints are more effective.
4. **Computational Efficiency Considerations**: Encoding features into Gaussian attributes avoids the overhead of re-extracting features for pseudo-views each time.
5. **Moderate Use of Flattened Gaussians**: Recognizes the overfitting risk brought by flattening and mitigates it using consistency constraints.

## Limitations & Future Work

1. Relies on the quality of the pre-trained stereo matching network, which may provide noisy priors in the early stages of training when rendering quality is poor.
2. The pseudo-view generation strategy is relatively simple (based on the vicinity of training cameras); more intelligent view selection can be explored.
3. Computational overhead: Requires additional stereo matching inference and feature extraction.
4. Not specifically optimized for large-scale scenes (such as Mip-NeRF360 outdoor scenes).
5. The sparse setting of 3 views is fixed, without exploring performance under different levels of sparsity.

## Related Work & Insights

- **GS2Mesh**: Most related work, uses stereo matching to extract meshes from 3DGS, but performs poorly under sparse views.
- **FatesGS/Sparse2DGS**: Surface reconstruction methods with flattened Gaussians, where SparseSurf points out their overfitting issues.
- **DNGaussian**: A depth-regularization method, but its geometric constraints are too loose to reconstruct accurate surfaces.
- **Insight**: Stereo matching is a promising direction as geometric supervision for 3DGS.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel design of self-reinforcing stereo priors and feature-level pseudo-view consistency.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, two sparse settings, detailed ablation, and comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Thorough motivation analysis, clear methodology derivation.
- **Value**: ⭐⭐⭐⭐ — High demand for sparse-view surface reconstruction applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](../../CVPR2026/3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[CVPR 2026\] SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting](../../CVPR2026/3d_vision/sv-gs_sparse_view_4d_reconstruction_with_skeleton-driven_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
