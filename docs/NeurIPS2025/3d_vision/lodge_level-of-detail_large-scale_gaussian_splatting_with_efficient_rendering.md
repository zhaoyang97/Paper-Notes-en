---
title: >-
  [Paper Note] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering
description: >-
  [NeurIPS 2025][3D Vision][Gaussian Splatting] This paper proposes LODGE, which manages 3D Gaussian Splatting at multiple scales through a hierarchical Level-of-Detail (LOD) strategy. By dynamically selecting Gaussian representations of appropriate granularity based on camera distance, LODGE enables high-quality real-time rendering of large-scale scenes.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Gaussian Splatting
  - Level-of-Detail
  - Large-Scale Scene
  - Efficient Rendering
  - LOD
date: 2026-05-08
content_hash: de4afdc91b3d5c2d
---

# LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering

**Conference**: NeurIPS 2025
**arXiv**: [2505.23158](https://arxiv.org/abs/2505.23158)
**Code**: Available
**Area**: 3D Vision / Large-Scale Scene Rendering
**Keywords**: Gaussian Splatting, Level-of-Detail, Large-Scale Scene, Efficient Rendering, LOD

## TL;DR

This paper proposes LODGE, which manages 3D Gaussian Splatting at multiple scales through a hierarchical Level-of-Detail (LOD) strategy. By dynamically selecting Gaussian representations of appropriate granularity based on camera distance, LODGE enables high-quality real-time rendering of large-scale scenes.

## Background & Motivation

### State of the Field

**Background**: 3DGS performs well on small scenes but faces an explosion in the number of Gaussians when scaled to large scenes at the city level.

**Limitations of Prior Work**: Millions of Gaussians lead to (1) GPU memory overflow; (2) enormous rasterization overhead; (3) wasteful computation on distant fine details.

**Key Challenge**: High quality requires dense Gaussians, whereas efficiency demands sparse representations.

**Key Insight**: LOD techniques from traditional computer graphics — distant objects are represented at low resolution, and nearby objects at high resolution.

**Core Idea**: Construct a multi-level LOD pyramid of Gaussians and select the appropriate level based on distance at render time.

### Mechanism

**Goal**: ### Overall Architecture

Scene partitioning → LOD pyramid construction per block (fine to coarse) → distance-based level selection at render time → hybrid rasterization.

## Method

### Overall Architecture

Scene partitioning → LOD pyramid construction per block (fine to coarse) → distance-based level selection at render time → hybrid rasterization.

### Key Designs

1. **LOD Pyramid Construction**

    - Function: Progressively merges fine-grained Gaussians into coarser representations level by level.
    - Mechanism: Spatial clustering with attribute fusion — neighboring Gaussians are merged into larger Gaussians with weighted averaging of color and opacity.
    - Design Motivation: At large distances, small Gaussians are covered by individual pixels; merging them yields equivalent visual results with significantly fewer primitives.

2. **Distance-Adaptive Level Selection**

    - Function: Selects the appropriate LOD level for each scene block based on camera distance.
    - Mechanism: $\text{LOD}(d) = \lfloor \log_2(d / d_{min}) \rfloor$, i.e., one coarser level per doubling of distance.
    - Design Motivation: Human vision is insensitive to distant detail; Screen Space Error is used to govern visual quality.

3. **Cross-Level Smooth Transition**

    - Function: Prevents abrupt visual changes (popping artifacts) at LOD level boundaries.
    - Mechanism: Gaussians from two adjacent levels are blended in boundary regions using linear interpolation weights.
    - Design Motivation: Sudden level switching produces visible flickering artifacts.

### Loss & Training

Each level is trained independently: $\mathcal{L} = \lambda_1 \mathcal{L}_{photometric} + \lambda_2 \mathcal{L}_{SSIM}$. Training begins at the finest level, with coarser levels constructed by progressive merging.

## Key Experimental Results

### Main Results

| Method | Mill19 PSNR↑ | Render FPS↑ | GPU Memory↓ | # Gaussians↓ |
|------|------------|---------|---------|-------------|
| 3DGS | 26.8% | 15 | 24GB | 45M |
| Mega-NeRF | 25.1% | 0.5 | 8GB | N/A |
| **LODGE** | **26.5%** | **45** | **8GB** | **12M (at render)** |

### Ablation Study

| Configuration | PSNR | FPS | Notes |
|------|------|-----|------|
| Single level (finest) | 26.8% | 15 | Memory overflow |
| LOD w/o smooth transition | 26.1% | 42 | Popping artifacts |
| **LOD + smooth transition** | **26.5%** | **45** | **Best** |

### Key Findings

- LODGE achieves a 3× FPS improvement (15→45) with only a 0.3 dB PSNR drop.
- GPU memory is reduced from 24 GB to 8 GB, enabling large-scale scene rendering on consumer-grade hardware.
- Smooth transition contributes +0.4 PSNR and effectively eliminates popping artifacts.

## Highlights & Insights

- **Revival of Classic LOD in 3DGS**: A well-established graphics technique demonstrates renewed vitality under a novel representation paradigm.
- **Practicality**: LODGE makes large-scale 3DGS feasible on consumer hardware, substantially lowering the barrier to deployment.

## Limitations & Future Work

- LOD merging incurs information loss — certain fine details may degrade under close-up observation.
- LOD update strategies for dynamic scenes are not addressed.
- Seam artifacts at block boundaries require further handling.

## Related Work & Insights

- **vs. Mega-NeRF**: Mega-NeRF performs block-based decomposition on NeRF; LODGE achieves greater efficiency on 3DGS.
- **vs. CityGaussian**: CityGaussian also targets large-scale 3DGS but lacks an LOD strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ The LOD concept is mature, but its adaptation to 3DGS introduces meaningful innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on large-scale scenes.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-organized.
- Value: ⭐⭐⭐⭐⭐ A practical solution for large-scale 3DGS.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[ICCV 2025\] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](../../ICCV2025/3d_vision/resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)
- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](../../ICCV2025/3d_vision/vertexregen_mesh_generation_with_continuous_level_of_detail.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](../../ICCV2025/3d_vision/occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[NeurIPS 2025\] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)

<!-- RELATED:END -->
