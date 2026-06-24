---
title: >-
  [Paper Note] SAGS: Structure-Aware 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This work proposes SAGS, which implicitly encodes scene geometry using a local-global graph representation and graph neural networks. It improves the rendering quality of 3DGS, reduces storage requirements (up to 24× compression), and significantly suppresses floater artifacts while maintaining real-time rendering.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "graph neural network"
  - "structure-aware"
  - "novel view synthesis"
  - "model compression"
date: 2026-05-08
content_hash: 162e77cf737fc103
---

# SAGS: Structure-Aware 3D Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2404.19149](https://arxiv.org/abs/2404.19149)  
**Code**: [Available](https://eververas.github.io/SAGS/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, graph neural network, structure-aware, novel view synthesis, model compression

## TL;DR

This work proposes SAGS, which implicitly encodes scene geometry using a local-global graph representation and graph neural networks. It improves the rendering quality of 3DGS, reduces storage requirements (up to 24× compression), and significantly suppresses floater artifacts while maintaining real-time rendering.

## Background & Motivation

### Proposed Solution

**Goal**: **Background**: 3D-GS optimizes each Gaussian kernel independently in a geometry-agnostic manner, ignoring the intrinsic 3D structure of the scene. This leads to Gaussian kernels deviating significantly from their initial positions, producing floater artifacts and poor depth map quality. Existing compression methods (such as codebook quantization) also fail to utilize structural information. This paper introduces the idea of graph networks from point cloud analysis, allowing adjacent Gaussian kernels to share information and learn topology-preserving displacements.

## Method

### Overall Architecture

1. **Curvature-aware densification**: Estimating the Gaussian curvature of point clouds and increasing point density in low-curvature regions using midpoint interpolation.
2. **Structure-aware encoder**: Constructing a $k$-NN graph and using a GNN to aggregate local and global features.
3. **Refinement network**: Decoding color, opacity, covariance, and displacement using four independent MLPs.

### Key Designs

**Curvature-Aware Densification**: COLMAP often suffers from insufficient point sampling in textureless planar regions. By estimating curvature via local PCA, midpoints are inserted in low-curvature regions to supplement the point cloud density.

**Structure-Aware GNN Encoder**:
$$\Phi(\mathbf{p}_i, \mathbf{f}_i) = \phi\left(\sum_{j \in \mathcal{N}(i)} w_{ij} h_\Theta(\gamma(\mathbf{p}_j), \mathbf{f}_j - \mathbf{f}_i, \mathbf{g})\right)$$
Utilizing relative features $\mathbf{f}_j - \mathbf{f}_i$, positional encoding $\gamma(\mathbf{p})$, and global features $\mathbf{g}=\max(\mathbf{f})$, neighborhood information is aggregated through inverse distance weighting.

**Displacement Prediction**: Gaussian positions are modeled as the initial COLMAP positions plus displacements $\Delta \mathbf{p}$ predicted by the MLP, constraining small displacements to preserve scene topology.

**SAGS-Lite**: Training the network only on keypoints and obtaining midpoint attributes through interpolation, achieving extreme lightweight performance without any compression techniques.

### Loss & Training

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{SSIM}, \quad \lambda=0.2$$

## Key Experimental Results

### Main Results

**Rendering Quality on Mip-NeRF360 / Tanks & Temples / Deep Blending**:

| Method | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DB PSNR/SSIM/LPIPS |
|------|---------------------------|---------------------|---------------------|
| 3D-GS | 28.69/0.870/0.182 | 23.14/0.841/0.183 | 29.41/0.903/0.243 |
| Scaffold-GS | 28.84/0.848/0.220 | 23.96/0.853/0.177 | 30.21/0.906/0.254 |
| **SAGS** | **29.65/0.874/0.179** | **24.88/0.866/0.166** | **30.47/0.913/0.241** |

**Storage Compression Ratio (Compared to 3D-GS)**:

| Method | MipNeRF360 (MB) | T&T (MB) | Deep Blending (MB) |
|------|-----------------|----------|---------------------|
| 3D-GS | 693 | 411 | 676 |
| Scaffold-GS | 252 (2.8×↓) | 87 (4.7×↓) | 66 (10.2×↓) |
| **SAGS** | **135 (5.1×↓)** | **75 (5.5×↓)** | **58 (11.7×↓)** |
| **SAGS-Lite** | **76 (9.1×↓)** | **35 (12×↓)** | **28 (24×↓)** |

### Ablation Study

| Ablation Item | DB PSNR | T&T PSNR |
|--------|---------|----------|
| w/o Curvature Densification | 29.87% | 23.97% |
| w/o GNN | 29.94% | 24.19% |
| w/o Positional Encoding | 30.21% | 24.31% |
| w/o Global Features | 30.17% | 24.42% |
| w/o View-Dependent Positions | 30.07% | 24.37% |
| **Full SAGS** | **30.47%** | **24.88%** |

### Key Findings

- Structure awareness constrains Gaussian kernel displacements within a small range, suppressing floater artifacts.
- The depth maps generated by SAGS are significantly superior to those of 3D-GS and Scaffold-GS, capturing sharp edges and flat surfaces.
- SAGS-Lite achieves 24× storage reduction without using compression techniques, while maintaining rendering quality close to 3D-GS.

## Highlights & Insights

1. **First structure-aware 3DGS method**: Bridging the fields of point cloud analysis and 3DGS.
2. **Displacement prediction paradigm**: Constraining Gaussian kernels to stay near the initial geometry, implicitly preserving scene topology.
3. **Extremely lightweight SAGS-Lite**: The midpoint interpolation scheme is simple and effective, achieving 24× compression without quantization.
4. Achieving superior rendering quality and smaller model sizes simultaneously.

## Limitations & Future Work

- GNN inference introduces extra computational overhead.
- It heavily relies on the quality of initial COLMAP point clouds.
- It is not applicable to highly dynamic scenes.

## Related Work & Insights

- Scaffold-GS introduces a hierarchical structure but still employs structureless optimization.
- Concepts from point cloud analysis GNNs (e.g., DGCNN, PointNet++) can be transferred to 3DGS.
- Insight: Scene structure priors are crucial for reducing redundant Gaussian kernels and improving rendering quality.

## Rating

- Novelty: ★★★★★ First work to introduce GNN to 3DGS, with a novel structure-aware concept.
- Practicality: ★★★★☆ Real-time rendering and substantial compression hold great value for VR/AR applications.
- Experimental Quality: ★★★★★ Fully evaluated on 13 scenes across 3 datasets, with comprehensive ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)
- [\[ECCV 2024\] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](hac_hash-grid_assisted_context_for_3d_gaussian_splatting_compression.md)

</div>

<!-- RELATED:END -->
