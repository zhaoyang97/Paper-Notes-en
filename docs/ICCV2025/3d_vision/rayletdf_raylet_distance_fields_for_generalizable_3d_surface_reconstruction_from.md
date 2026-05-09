---
title: >-
  [Paper Note] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians
description: >-
  [ICCV 2025][3D Vision][Raylet Distance Field] This paper proposes RayletDF, a generalizable 3D surface reconstruction method based on a "raylet" (ray segment) distance field. Through three modules — a raylet feature extractor, a distance field predictor, and a multi-raylet mixer — RayletDF directly predicts surface points from point clouds or 3D Gaussians, achieving high-accuracy cross-dataset generalization via a single forward pass on unseen datasets.
tags:
  - ICCV 2025
  - 3D Vision
  - Raylet Distance Field
  - Surface Reconstruction
  - Point Cloud
  - 3DGS
  - Generalization
date: 2026-05-08
content_hash: e749d2dc5de5efb0
---

# RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians

**Conference**: ICCV 2025
**arXiv**: [2508.09830](https://arxiv.org/abs/2508.09830)
**Code**: [https://github.com/vLAR-group/RayletDF](https://github.com/vLAR-group/RayletDF)
**Area**: 3D Vision / Surface Reconstruction / Generalizable Representation
**Keywords**: Raylet Distance Field, Surface Reconstruction, Point Cloud, 3DGS, Generalization

## TL;DR
This paper proposes RayletDF, a generalizable 3D surface reconstruction method based on a "raylet" (ray segment) distance field. Through three modules — a raylet feature extractor, a distance field predictor, and a multi-raylet mixer — RayletDF directly predicts surface points from point clouds or 3D Gaussians, achieving high-accuracy cross-dataset generalization via a single forward pass on unseen datasets.

## Background & Motivation

Recovering 3D surfaces from RGB/D images or point clouds is a fundamental requirement for applications such as mixed reality and embodied AI. Existing methods each suffer from distinct limitations:

- **Coordinate-based methods** (OF, SDF, NeRF): require dense sampling and network evaluation to extract explicit surfaces, incurring high computational cost.
- **3DGS**: enables real-time RGB rendering but produces poor depth quality and fails to capture fine surface geometry.
- **Ray-based methods** (DRDF, PRIF, RayDF): are efficient but constrained by Plücker/spherical ray parameterizations, limited to object-level surfaces, and require per-scene optimization.

**Core Insight**: Existing ray-based methods use complete rays as input, preventing them from capturing local geometric patterns. By instead using **local ray segments (raylets)** — unit ray segments whose starting points are sampled near the surface — the method can focus on fine-grained local surface patterns that are **generalizable** across different shapes.

## Method

### Core Concepts: Raylet and Raylet Distance

- **Raylet** $\mathbf{l}$: a unit segment of a ray, with its starting point sampled near the shape surface, parameterized as a 6D vector (starting point xyz + unit direction vector).
- **Raylet Distance** $d_l$: the signed distance between the surface hit point and the raylet starting point. Positive values indicate the hit point lies in front of the starting point; negative values indicate behind.
- Key advantage: multiple raylets can be sampled on both sides of the surface along a single ray, with each raylet focusing on local surface patterns.

### Three-Module Pipeline

**Module 1: Raylet Feature Extractor**
- Extracts per-point features $\mathbf{F} \in \mathbb{R}^{N \times 32}$ from the input scene (point cloud or 3D Gaussians) using SparseConv.
- For a query raylet $\mathbf{l}$, the $K$ nearest points are retrieved via KNN, and neighborhood information is aggregated as:
$$\hat{\mathbf{f}}_l^k = \left(\mathbf{p}_l^k \oplus \frac{(\mathbf{p}_l^k - \mathbf{p}_l)}{\|\mathbf{p}_l^k - \mathbf{p}_l\|} \oplus \|\mathbf{p}_l^k - \mathbf{p}_l\|\right) \oplus \mathbf{f}_l^k$$
- Key insight: the extracted features preserve local geometric patterns near the surface, enabling the learned representation to generalize across scenes.

**Module 2: Raylet Distance Field Predictor**
- An 8-layer MLP (256 hidden units per layer) takes the raylet position, direction, and features as input, and outputs a distance value and a confidence score:
$$(d_l, s_l) = MLPs(\mathbf{p}_l \oplus \mathbf{u}_l \oplus \mathbf{f}_l)$$
- No dense coordinate sampling along the ray is required; the surface distance is predicted in a single pass.

**Module 3: Multi-Raylet Mixer**
- $T$ raylets are sampled along the same ray (same direction, different starting points), with distances predicted in parallel.
- Predictions are fused via softmax-weighted aggregation:
$$D = \sum_{t=1}^T \hat{s}_{l_t}\left(\|\mathbf{p}_{cam} - \mathbf{p}_{l_t}\| + d_{l_T}\right), \quad \hat{s}_{l_t} = \frac{e^{s_{l_t}}}{\sum_{t=1}^T e^{s_{l_t}}}$$
- Multi-raylet fusion improves generalizability and robustness.

### Raylet Sampling Strategy
- For point clouds: a virtual sphere (radius = distance to the nearest point) is constructed for each point; the scene surface is bounded by the union of all virtual spheres. Ray–sphere intersections are projected onto the ray, and the top-$T$ intersection points with the smallest perpendicular distances are selected as raylet starting points.
- For 3DGS: ray–Gaussian intersections are computed, and the top-$T$ points are selected based on alpha blending contribution.

### Training Loss
- An $\ell_1$ loss supervises the predicted distance $D$; ground-truth values are converted from depth maps.

## Key Experimental Results

### Main Results: Cross-Dataset Generalization (Trained on ARKitScenes)

| Method | Type | ARKitScene ADE↓ | ScanNet/++ ADE↓ | MultiScan ADE↓ |
|--------|------|----------------|----------------|---------------|
| 3DGS | Per-scene | 0.268 | 0.321 | 0.431 |
| PGSR | Per-scene | 0.219 | 0.202 | 0.315 |
| DepthAnythingV2 | Aligned | 0.206 | 0.168 | 0.228 |
| Pointersect | Generalizable | 0.286 | 0.366 | 0.266 |
| RayDF | Generalizable | 0.183 | 0.227 | 0.326 |
| **RayletDF** | **Generalizable** | **0.115** | **0.175** | **0.216** |

### Ablation Study: Impact of Key Components ($\delta$ Metric)

| Ablation | ARKit $\delta$↑ | ScanNet++ $\delta$↑ |
|----------|----------------|-------------------|
| RayletDF (full) | **0.928** | **0.894** |
| w/o multi-raylet mixing | 0.908 | 0.847 |
| w/o confidence score | 0.921 | 0.882 |
| K=4 (vs. K=16) | 0.916 | 0.870 |

**Key Findings**:
- RayletDF reduces ADE on ARKitScene by 37% over RayDF (0.183→0.115), with particularly strong cross-dataset generalization.
- Even when trained solely on ARKitScenes, the method significantly outperforms all generalizable baselines on fully unseen ScanNet++ and MultiScan datasets.
- Multi-raylet mixing is critical (removing it drops $\delta$ by ~2%); confidence weighting further improves accuracy.
- The method supports reconstruction from both point cloud and 3DGS inputs within the same pipeline.
- Gaussian data for 7,770 3D scenes (ScanNet/++, ARKitScenes, MultiScan) will be publicly released.

## Highlights & Insights
1. **Elegant raylet concept**: Decomposing rays into segments that focus on local patterns is the key to achieving generalization — local geometric patterns are shared across diverse scenes.
2. **No dense sampling required**: Unlike SDF/OF methods that require dense coordinate sampling along rays, RayletDF predicts surface distances in a single forward pass.
3. **Unified input pipeline**: The same pipeline seamlessly handles both point cloud and 3DGS inputs.
4. **Closed-form surface normal derivation**: The ray-based formulation allows surface normals to be derived analytically without an additional network.

## Limitations & Future Work
- The SparseConv backbone requires voxelization, and memory consumption grows with scene scale.
- No prediction can be produced for query rays far from the point cloud surface (such rays are discarded).
- Cross-dataset generalization still exhibits a non-trivial accuracy gap on MultiScan.
- The use of surface normals as additional regularization or for outlier filtering has not been explored.

## Related Work & Insights
- Coordinate-based methods (OF, SDF, UDF) require dense sampling.
- Ray-based methods (RayDF, PRIF) are limited to object-level shapes.
- Depth estimation (DepthAnythingV2) produces high-quality results but lacks cross-frame consistency.

## Rating
- **Novelty**: ★★★★★ — The raylet distance field is an elegant and effective new representation.
- **Practicality**: ★★★★☆ — Generalizable reconstruction offers substantial value for downstream applications (AR/robotics).
- **Experimental Thoroughness**: ★★★★★ — Comprehensive evaluation across multiple datasets with thorough ablation studies.
- **Writing Quality**: ★★★★☆ — Concepts are clearly articulated with intuitive illustrations.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[NeurIPS 2025\] AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians](../../NeurIPS2025/3d_vision/atlasgs_atlanta-world_guided_surface_reconstruction_with_implicit_structured_gau.md)
- [\[ICCV 2025\] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing](svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](../../AAAI2026/3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)

<!-- RELATED:END -->
