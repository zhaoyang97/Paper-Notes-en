---
title: >-
  [Paper Note] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes an occlusion-aware scene partitioning strategy and region-based rendering technique. By clustering a camera co-visibility graph, it achieves partitions aligned with the scene layout, significantly improving reconstruction quality and rendering speed for large-scale 3DGS.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Large Scene Reconstruction
  - Occlusion-Aware
  - Scene Partitioning
  - Rendering Acceleration
date: 2026-05-08
content_hash: 11ca1a38fe1eb578
---

# OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering

**Conference**: ICCV 2025
**arXiv**: [2503.16177](https://arxiv.org/abs/2503.16177)
**Code**: [Project Page](https://occlugaussian.github.io)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Large Scene Reconstruction, Occlusion-Aware, Scene Partitioning, Rendering Acceleration

## TL;DR

This paper proposes an occlusion-aware scene partitioning strategy and region-based rendering technique. By clustering a camera co-visibility graph, it achieves partitions aligned with the scene layout, significantly improving reconstruction quality and rendering speed for large-scale 3DGS.

## Background & Motivation

Large scene reconstruction is critical for autonomous driving, cultural heritage preservation, and VR/AR applications. 3D Gaussian Splatting (3DGS) faces scalability challenges in large scenes due to its memory-intensive representation, and thus commonly adopts a **divide-and-conquer** strategy to partition the scene into smaller sub-regions for independent reconstruction.

However, existing scene partitioning methods share a common flaw: **occlusion unawareness**. They primarily partition based on camera positions or point clouds with uniform subdivision, neglecting scene layout and occlusion relationships. This leads to:

**Severe occlusion within regions** — cameras separated by walls or buildings are grouped into the same region

**Low inter-camera correlation** — cameras within a region share little co-visible content, resulting in low average contribution

**Degraded reconstruction quality** — training resources are dispersed across unrelated regions

This issue is particularly pronounced in ground-level captured scenes, where frequent occluders such as walls and buildings are common. Furthermore, after reconstruction, large scenes contain a massive number of Gaussian primitives, making rendering speed a critical bottleneck.

## Method

### Overall Architecture

OccluGaussian introduces two core innovations:
1. **Occlusion-Aware Scene Partitioning** — a camera partitioning strategy based on attributed graph clustering
2. **Region-Based Rendering Acceleration** — culling occluded Gaussians invisible to the current viewpoint

### Occlusion-Aware Scene Partitioning

#### Attributed View Graph Construction

An undirected attributed graph $\mathcal{G}=(\mathcal{V},\mathcal{E},X)$ is constructed as follows:
- **Nodes**: each node corresponds to a camera
- **Edges**: an edge is established between two cameras if they share visible content, with edge weight equal to the number of matched feature points, yielding adjacency matrix $A \in \mathbb{R}^{n \times n}$
- **Features**: the 3D coordinates of each camera are position-encoded as node features $X \in \mathbb{R}^{n \times d}$

Cameras that are occluded from each other or spatially distant typically share very few overlapping views and can thus be effectively distinguished in the graph.

#### Graph Clustering

An attributed graph clustering algorithm is applied, first performing graph convolution to produce smoothed features, followed by spectral clustering:

$$L_s = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$

Graph convolution filter: $G = (I - \frac{1}{2}L_s)^r$

Filtered features are obtained as $\bar{X} = GX$. A similarity matrix is then computed, and spectral clustering is applied to group cameras that share substantial overlapping views or are spatially proximate into the same region.

#### Adaptive Cluster Number Determination

Starting from an initial cluster count $K$, the algorithm recursively refines partitions by:
- Splitting clusters that contain too many cameras
- Ignoring clusters with too few cameras or those whose convex hull is entirely covered by another cluster
- Iterating recursively until all clusters reach a balanced size

### Region Reconstruction

Three types of training cameras are selected for each region:
- **Base set**: cameras located within the region
- **Extension set**: cameras outside the region that capture sufficient visible content within it
- **Boundary set**: cameras facing the region but occluded, used to constrain Gaussian primitives near the boundary

### Region-Based Rendering Acceleration

For each region, the 3D Gaussians visible to all training cameras within that region are recorded. During rendering, only the Gaussians recorded for the region containing the current viewpoint are processed, effectively culling occluded and invisible Gaussians. Regions can be further subdivided into smaller sub-regions for additional acceleration.

## Key Experimental Results

### Main Results — OccluScene3D Dataset

| Scene | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|-------|--------|-------|-------|--------|------|
| Gallery | VastGaussian | 25.09 | 0.903 | 0.095 | 215 |
| Gallery | CityGaussian | 21.98 | 0.808 | 0.294 | 120 |
| Gallery | Hierarchical-GS | 22.23 | 0.800 | 0.182 | 216 |
| Gallery | **OccluGaussian** | **25.81** | **0.903** | **0.094** | **289** |
| Canteen | VastGaussian | 24.60 | 0.890 | 0.105 | 211 |
| Canteen | **OccluGaussian** | **25.25** | **0.900** | **0.100** | **312** |
| ClassBuilding | VastGaussian | 24.05 | 0.884 | 0.111 | 270 |
| ClassBuilding | **OccluGaussian** | **25.33** | **0.921** | **0.083** | **340** |

### Comparison on Zip-NeRF Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| MERF | 23.49 | 0.747 | 0.445 |
| SMERF | 27.28 | 0.829 | 0.340 |
| Zip-NeRF | 27.37 | 0.836 | — |

### Key Findings

1. OccluGaussian achieves the best reconstruction quality across all scenes, with PSNR improvements of 1–3 dB
2. Rendering speed is significantly improved, with FPS increasing from 200+ to 280–340; region-based rendering effectively eliminates redundant computation on invisible Gaussians
3. The occlusion-aware partitioning strategy yields stronger inter-camera correlation and higher average contribution within each region

## Highlights & Insights

1. **Precise problem formulation** — Reframing scene partitioning from the perspective of camera co-visibility precisely identifies the root cause of quality degradation due to occlusion
2. **Effective reuse of SfM information** — The view graph is built directly from SfM matching results with no additional computational overhead
3. **Unified rendering acceleration and partitioning** — The occlusion-aware region partitioning naturally supports region-based rendering culling, achieving two goals simultaneously
4. **Strong applicability to ground-level scenes** — Particularly well-suited for indoor and urban environments with frequent occlusions

## Limitations & Future Work

- Relies on camera poses and matching information provided by SfM
- Primarily targets ground-level captured scenes; improvements are limited for aerial or open-world scenarios
- The adaptive cluster number determination requires recursive refinement, which may incur non-trivial overhead for extremely large scenes

## Related Work & Insights

- **Large scene reconstruction**: BlockNeRF, Mega-NeRF, VastGaussian, CityGaussian, and other divide-and-conquer approaches
- **Camera clustering**: COLMAP's Metis graph partitioning, Out-of-Core-BA
- **Rendering acceleration**: Octree-GS, LightGaussian, Hierarchical-3DGS, and other LoD-based methods

## Rating

- Novelty: ⭐⭐⭐⭐ (The unified design of occlusion-aware partitioning and region-based rendering is novel)
- Technical Depth: ⭐⭐⭐⭐ (Graph clustering methodology is complete and well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-dataset validation with comprehensive quantitative and qualitative evaluation)
- Practical Value: ⭐⭐⭐⭐⭐ (Directly addresses practical pain points in large scene reconstruction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)

</div>

<!-- RELATED:END -->
