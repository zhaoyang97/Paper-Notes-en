---
title: >-
  [Paper Note] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction
description: >-
  [3D Vision] This paper proposes UrbanGS, a scalable 3DGS reconstruction framework for urban-scale scenes that simultaneously improves geometric accuracy, rendering quality…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 179103ee676077fe
---

# UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction

- **Conference**: ICLR 2026
- **arXiv**: [2602.02089](https://arxiv.org/abs/2602.02089)
- **Code**: Not released
- **Area**: 3D Vision / Large-Scale Scene Reconstruction
- **Keywords**: 3D Gaussian Splatting, Large-Scale Reconstruction, Depth-Normal Regularization, Gaussian Pruning, Urban Scene

## TL;DR

This paper proposes UrbanGS, a scalable 3DGS reconstruction framework for urban-scale scenes that simultaneously improves geometric accuracy, rendering quality, and memory efficiency through depth-consistent D-Normal regularization, spatially adaptive Gaussian pruning (SAGP), and a unified partitioning strategy.

## Background & Motivation

3DGS performs well in bounded scenes, but scaling to large urban environments poses three major challenges:

**Poor geometric consistency**: Supervising only rendered normals updates rotation parameters but not positional parameters, resulting in inaccurate surface reconstruction.

**Low memory efficiency**: Homogeneous regions (sky, distant building facades) generate large numbers of redundant Gaussian primitives.

**Poor computational scalability**: Partitioning schemes introduce boundary discontinuities, and processing irrelevant viewpoints wastes computation.

## Method

### Overall Architecture

UrbanGS comprises three core modules:

1. **Depth-consistent D-Normal Regularization** (geometric accuracy)
2. **Spatially Adaptive Gaussian Pruning (SAGP)** (memory efficiency)
3. **Unified partitioning and view assignment** (scalability)

### 1. Depth-Consistent D-Normal Regularization

**Problem**: Directly supervising rendered normals $\hat{N}$ with pseudo-normals $N$ updates only rotation parameters $R$ via gradients, and cannot effectively update positional parameters $u$.

**Solution**: D-Normals $\bar{N}_d$ are derived from the rendered depth map:

$$\bar{N}_d(n,p) = \frac{\nabla_v d(n,p) \times \nabla_h d(n,p)}{|\nabla_v d \times \nabla_h d|}$$

where $d$ denotes 3D coordinates obtained by back-projecting the depth map. The D-Normal regularization loss is:

$$\mathcal{L}_{dn} = \|\bar{N}_d - N\|_1 + (1 - \bar{N}_d \cdot N)$$

By linking geometric constraints to depth through D-Normals, both positional and rotation parameters are updated simultaneously.

### Depth Consistency Regularization

To ensure multi-view depth consistency, an inverse depth loss and adaptive confidence weighting are introduced:

**Inverse depth loss**:

$$\mathcal{L}_{id}(u,v) = |\hat{D}^{-1}(u,v) - D_{ext}^{-1}(u,v)|$$

**Geometry-aware confidence**:

$$w_d = \exp\left(\frac{\cos\phi - 1}{0.01}\right) \cdot \exp\left(-\frac{\epsilon_d}{0.1}\right)$$

where $\cos\phi$ measures depth gradient consistency and $\epsilon_d$ measures normalized inverse depth deviation.

**Total loss**:

$$\mathcal{L}_{total} = \mathcal{L}_{RGB} + \lambda_1 \mathcal{L}_n + \lambda_2 \mathcal{L}_{dn} + \lambda_3 (w_d \cdot \mathcal{L}_{id})$$

### 2. Spatially Adaptive Gaussian Pruning (SAGP)

**Scene partitioning**: The scene is divided into voxel cells with characteristic length correlated to global Gaussian density:

$$\ell = \lambda \left(\frac{\mathcal{V}_{scene}}{\mathcal{N}}\right)^{1/3}$$

**Local volume normalization** (sublinear transform to suppress oversized primitives):

$$w_{v,i} = \left(\min\left(\frac{v_i}{\vartheta_{local}^{(t)}}, 1\right)\right)^{\kappa}$$

With $\kappa=0.5$ (square root), the importance of fine-grained structures is amplified.

**Importance score** (product of three factors):

$$S_i = \phi_i \cdot \tau_i \cdot w_{v,i}$$

- $\phi_i$: normalized ray intersection frequency
- $\tau_i$: opacity mapped through Sigmoid
- $w_{v,i}$: sublinear volume weight

A Gaussian is retained only when it simultaneously exhibits high visibility, frequent observation, and appropriate geometric scale.

### 3. Partitioning Strategy

Building upon CityGS with the following improvements:
- Global coarse 3DGS is first pruned via SAGP to reduce redundant Gaussians that attract irrelevant views.
- Sub-block boundaries retain shared Gaussian primitives to avoid geometric discontinuities.
- Camera view assignment is based on geometry and SSIM.

## Key Experimental Results

### Datasets

- **Mill19**: Building, Rubble (aerial scenes)
- **UrbanScene3D**: Residence, Sci-Art (urban scenes)

### Main Results (Rendering Quality)

| Method | Building PSNR | Rubble PSNR | Residence PSNR | Sci-Art PSNR |
|--------|--------------|-------------|----------------|-------------|
| 3DGS | 22.53 | 25.51 | 22.36 | 24.13 |
| CityGS-v2 | - | - | - | - |
| VCR-GauS | - | - | - | - |
| **UrbanGS** | **Best** | **Best** | **Best** | **Best** |

UrbanGS achieves state-of-the-art or near-state-of-the-art SSIM, PSNR, and LPIPS across all datasets.

### Geometric Accuracy

Qualitative comparison of rendered depth maps shows:
- UrbanGS produces smoother object surfaces.
- CityGS-v2 and VCR-GauS exhibit distortions on distant buildings and in complex regions.

### Memory Efficiency

SAGP achieves significant model compression (see ablation for specific ratios) while maintaining rendering quality. VCR-GauS fails with out-of-memory errors on an A5000 GPU, whereas UrbanGS runs without issue.

### Ablation Study

| Ablation | Effect |
|----------|--------|
| w/o D-Normal regularization | Positional parameters cannot be effectively updated; surfaces appear rough |
| w/o depth consistency | Multi-view depth misalignment |
| w/o confidence weighting | Unreliable depth predictions interfere with optimization |
| w/o SAGP | Gaussian count explodes; out-of-memory failure |
| Global vs. adaptive pruning | Adaptive pruning preserves more detail |

## Highlights & Insights

1. **D-Normal regularization** elegantly resolves the problem that normal supervision cannot update positional parameters.
2. The **combined depth and normal supervision** is theoretically well-motivated and mathematically justified.
3. **SAGP** is the first pruning framework specifically designed for urban-scale 3DGS.
4. The approach offers a **systematic solution** balancing geometric accuracy, memory efficiency, and scalability.
5. Large-scale scene reconstruction is demonstrated on consumer-grade GPUs such as the A5000.

## Limitations & Future Work

1. The method depends on the quality of external depth estimators (DepthAnything-v2) and normal estimators.
2. Hyperparameters of SAGP ($\lambda, t, \kappa$) require tuning.
3. The partitioning strategy is largely inherited from CityGS, offering limited novelty.
4. Evaluation is restricted to aerial/urban scenes; applicability to large-scale indoor scenes remains unverified.
5. The inverse depth loss may over-smooth nearby objects.

## Related Work & Insights

- **Large-scale 3DGS**: VastGaussian (Lin et al., 2024) employs block partitioning but suffers boundary inconsistencies; CityGaussian (Liu et al., 2024a) requires time-consuming post-processing; CityGS-v2 (Liu et al., 2024b) adopts 2DGS but at the cost of rendering quality.
- **Geometric optimization**: 2DGS (Huang et al., 2024a) and VCR-GauS (Chen et al., 2024b) introduce depth/normal regularization but fail to sufficiently update positional parameters.
- **Gaussian pruning**: Fan et al. (2023) apply simple global-metric-based pruning that oversimplifies large-scale scenes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both D-Normal regularization and SAGP represent targeted contributions.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly addresses real-world pain points in urban-scale reconstruction.
- **Clarity**: ⭐⭐⭐⭐ — Methods are systematically described with sufficient theoretical analysis.
- **Significance**: ⭐⭐⭐⭐ — Provides a complete solution for large-scale 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](../../ICCV2025/3d_vision/s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] Universal Beta Splatting](universal_beta_splatting.md)

</div>

<!-- RELATED:END -->
