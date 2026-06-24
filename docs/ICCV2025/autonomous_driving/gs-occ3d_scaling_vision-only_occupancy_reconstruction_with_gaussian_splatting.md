---
title: >-
  [Paper Note] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting
description: >-
  [ICCV 2025][Autonomous Driving][occupancy reconstruction] This paper proposes GS-Occ3D, a scalable vision-only occupancy reconstruction framework that achieves full-dataset auto-labeling on Waymo through Octree-based Gaussian Surfel representation and a three-layer decomposed modeling of ground, static background, and dynamic objects. The resulting labels enable downstream occupancy prediction models to achieve zero-shot generalization comparable to or better than LiDAR-based…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "occupancy reconstruction"
  - "Gaussian splatting"
  - "vision-only"
  - "auto-labeling"
  - "3D reconstruction"
date: 2026-05-08
content_hash: 1e9ea6291630c537
---

# GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2507.19451](https://arxiv.org/abs/2507.19451)  
**Code**: [Project Page](https://github.com/baijunye/GS-Occ3D)  
**Area**: Autonomous Driving
**Keywords**: occupancy reconstruction, Gaussian splatting, vision-only, auto-labeling, 3D reconstruction

## TL;DR

This paper proposes GS-Occ3D, a scalable vision-only occupancy reconstruction framework that achieves full-dataset auto-labeling on Waymo through Octree-based Gaussian Surfel representation and a three-layer decomposed modeling of ground, static background, and dynamic objects. The resulting labels enable downstream occupancy prediction models to achieve zero-shot generalization comparable to or better than LiDAR-based annotations.

## Background & Motivation

3D occupancy prediction is a critical foundation for autonomous driving perception and planning, yet existing approaches face a severe **scalability bottleneck**:

**High cost of LiDAR annotation**: Mainstream occupancy labels (e.g., Occ3D) rely on LiDAR point clouds, requiring expensive specialized survey vehicles and making it difficult to leverage large-scale crowdsourced data.

**Challenges of vision-only reconstruction**: Sparse viewpoints, dynamic occlusions, texture-less regions, and long-range trajectories lead to geometric degradation.

**Existing methods are ill-suited for large-scale labeling**:
   - NeRF-based methods produce over-smoothed geometry or require full volumetric processing, limiting scalability.
   - Gaussian Splatting methods optimize for rendering quality rather than geometric accuracy, producing fragmented geometry when applied directly to occupancy reconstruction.
   - Mesh representations require extensive post-processing and are unsuitable for automated pipelines.

**Dynamic objects are neglected**: Most prior methods handle only static scenes and cannot model the occupancy of moving objects.

The core goal is to **construct a vision-only occupancy annotation pipeline that requires no LiDAR, no geometric priors, and can reconstruct the entire Waymo dataset.**

## Method

### Overall Architecture

A three-stage pipeline:

1. **Geometric Reconstruction**: The scene is decomposed into ground, static background, and dynamic objects, each modeled separately.
2. **Label Curation**: Per-frame partitioning → multi-frame aggregation → ray casting.
3. **Downstream Training**: Occupancy prediction models are supervised using the generated vision-only labels.

### Key Designs

1. **Octree-based Gaussian Surfel**: Sparse point clouds from SfM serve as the initial skeleton to construct a dynamic octree structure. Each voxel generates $m$ Gaussian Surfel primitives as local surface approximations. The number of hierarchy levels $K$ is adaptively determined by the distance distribution from camera centers to the point cloud:

    $K = \lfloor \log_2(d_{max}/d_{min}) \rceil + 1$

   Voxel centers at each level are computed via hierarchical quantization: $\mathbf{V}_L = \{\lfloor \mathbf{P}/(\epsilon/2^L) \rceil \cdot (\epsilon/2^L)\}$

   **Design Motivation**: The octree maintains memory efficiency while coarse levels model global structures (roads, walls) and fine levels capture high-frequency details (vegetation, boundaries). The structure can adaptively expand or contract during training. Cumulative LOD is used instead of a single LOD to enhance cross-scale geometric completeness.

2. **Ground Gaussians**: Texture-less ground is a major challenge for vision-only reconstruction. Camera poses are projected onto the $xy$-plane to initialize ground surfels, with $z$-coordinates adjusted by a fixed height offset from the nearest camera, and orientations inherited from camera rotations. **Design Motivation**: Explicitly modeling the ground — a dominant structural element — ensures large-area consistency, particularly in uphill/downhill scenarios. A planar regularization loss is applied to maintain flatness.

3. **Dynamic Object Reconstruction**: RGB-based 3D object tracking initializes bounding box poses $(\mathbf{R}_t, \mathbf{t}_t)$ for each dynamic vehicle, with a fixed number of points sampled within each box. To mitigate initial pose noise, learnable corrections are introduced:

    $\mathbf{R}_t' = \mathbf{R}_t \Delta\mathbf{R}_t, \quad \mathbf{t}_t' = \mathbf{t}_t + \Delta\mathbf{t}_t$

4. **Label Generation Pipeline**:

    - **Per-frame Partitioning**: A perception range centered at the camera pose is defined, and points are uniformly sampled to form per-frame point clouds.
    - **Multi-frame Aggregation**: Point clouds of dynamic objects are aggregated across frames (transformed into the box coordinate system and concatenated) to address sparsity.
    - **Ray-casting Voxelization**: Rays are cast from each camera to each occupied voxel; only the first hit voxel is marked as "observed," and all others are marked as "unobserved," explicitly handling occlusion.

### Loss & Training

$$L = L_{rgb} + \lambda_{geo}L_{geo} + \lambda_{obj}L_{obj} + \lambda_{road}L_{road} + \lambda_{sky}L_{sky}$$

- $L_{rgb}$: L1 + D-SSIM reconstruction loss.
- $L_{geo} = \lambda_s L_s + \lambda_d L_d + \lambda_n L_n$: Surfel regularization + depth distortion + depth-normal consistency.
- $L_{obj}$: Entropy loss on object opacity maps to encourage clear foreground/background decoupling.
- $L_{road}$: Regularization on height variation among neighboring surfels.
- $L_{sky}$: BCE loss on rendered opacity in sky regions.

## Key Experimental Results

### Geometric Reconstruction (Waymo Static-32)

| Method | CD ↓ | PSNR ↑ | Memory (GB) | Training Time |
|--------|------|--------|-------------|---------------|
| NeuS (w/ LiDAR) | 0.76 | 13.24 | 31 | 5.0h |
| StreetSurf (w/ LiDAR) | 0.90 | 26.85 | 21 | 1.5h |
| 2DGS | 1.23 | 25.60 | 15 | 1.0h |
| GVKF | 0.82 | 25.87 | 24 | 2.0h |
| **GS-Occ3D (vision-only)** | **0.56** | **26.89** | **10** | **0.8h** |

- The vision-only method **surpasses all LiDAR-assisted baselines** (NeuS, StreetSurf), achieving a CD of only **0.56**.

### Downstream Occupancy Prediction

| Training Labels | Eval Set | IoU ↑ | F1 ↑ | Prec. ↑ | Rec. ↑ |
|----------------|----------|-------|------|---------|--------|
| Ours (Waymo) | Occ3D-Val (Waymo) | 44.7 | 61.8 | 58.2 | 65.9 |
| Occ3D (Waymo) | Occ3D-Val (Waymo) | 57.4 | 73.0 | 62.9 | 87.0 |
| **Ours (Waymo)** | **Occ3D-Val (nuScenes)** | **33.4** | **50.1** | **62.5** | **41.8** |
| Occ3D (Waymo) | Occ3D-Val (nuScenes) | 31.4 | 47.8 | 38.8 | 62.1 |

- In-domain Waymo performance is moderately lower than LiDAR labels (44.7 vs. 57.4), within a reasonable margin.
- **Zero-shot generalization to nuScenes surpasses LiDAR labels** (33.4 vs. 31.4 IoU) with substantially higher precision (62.5 vs. 38.8).

### Ablation Study

| Configuration | CD ↓ | PSNR ↑ | Notes |
|--------------|------|--------|-------|
| 5-cam input (Ours) | 0.56 | 26.89 | Best |
| 3-cam input (Ours) | 0.66 | 26.96 | Still outperforms other methods |
| 5-cam GVKF | 0.82 | 25.87 | Baseline |
| w/o Ground Gaussians | — | — | Holes and abnormal protrusions appear |

### Key Findings

- Direct point cloud representation outperforms mesh conversion: mesh introduces post-processing artifacts (holes, sky-enclosure artifacts).
- Ground Gaussians are critical for texture-less regions, eliminating ground holes and distortions.
- Vision-only labels can cover 66 semantic categories (vs. 16 in Occ3D), including motorcycles and lane markings that LiDAR tends to miss.
- Five cameras benefit this method more substantially (0.56 vs. 0.66), whereas other methods degrade due to forward multi-view ambiguity.

## Highlights & Insights

- **Paradigm shift**: From "LiDAR annotation → train model" to "vision-only reconstruction → auto-generate labels → train model," reducing cost by orders of magnitude.
- **Full Waymo dataset reconstruction**: The first vision-only method to achieve this, demonstrating true engineering scalability.
- **Four aspects where vision-only labels outperform LiDAR**: broader coverage, stronger zero-shot generalization, richer and cheaper semantics, and potential advantages in adverse weather conditions.

## Limitations & Future Work

- Cameras cover only the front and side views; missing rear-view information is unavoidable.
- Nighttime conditions and exposure issues reduce the effective visual range.
- The vision-only approach fails in ego-static scenarios (vehicle standstill).
- Only geometric labels are currently provided; joint semantic and geometric reconstruction is identified as a key future direction.

## Related Work & Insights

- The Octree-GS framework has potential applications in other large-scale scene tasks, including urban reconstruction and indoor modeling.
- The generalization advantage of vision-only labels suggests that training data diversity may matter more than annotation precision.
- The ground-specific modeling approach can be extended to other dominant structural elements such as ceilings and walls.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Vision-only occupancy reconstruction with full Waymo dataset annotation.
- **Technical Depth**: ⭐⭐⭐⭐ — Octree Surfel + three-layer decomposition + complete labeling pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Reconstruction, downstream tasks, generalization, ablations, and comparison with LiDAR; very comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Directly supports large-scale autonomous driving data annotation with high practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[AAAI 2026\] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction](../../AAAI2026/autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)

</div>

<!-- RELATED:END -->
