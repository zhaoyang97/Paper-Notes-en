---
title: >-
  [Paper Note] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective
description: >-
  [CVPR 2026 (Oral)][Autonomous Driving][Semantic Scene Completion] OccuFly introduces the first real-world camera-based Semantic Scene Completion (SSC) benchmark from the aerial perspective, comprising 20,000+ samples across 21 semantic categories, spanning multi-season and multi-altitude urban, industrial, and rural scenes. It further reveals fundamental limitations of current visual foundation models in aerial settings.
tags:
  - CVPR 2026 (Oral)
  - Autonomous Driving
  - Semantic Scene Completion
  - Aerial Perspective
  - UAV
  - Benchmark Dataset
  - Depth Estimation
date: 2026-05-08
content_hash: 89a88d80b65d78ab
---

# OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective

**Conference**: CVPR 2026 (Oral)
**arXiv**: [2512.20770](https://arxiv.org/abs/2512.20770)
**Code**: [https://github.com/markus-42/occufly](https://github.com/markus-42/occufly) (available, to be released)
**Area**: Autonomous Driving / 3D Vision
**Keywords**: Semantic Scene Completion, Aerial Perspective, UAV, Benchmark Dataset, Depth Estimation

## TL;DR

OccuFly introduces the first real-world camera-based Semantic Scene Completion (SSC) benchmark from the aerial perspective, comprising 20,000+ samples across 21 semantic categories, spanning multi-season and multi-altitude urban, industrial, and rural scenes. It further reveals fundamental limitations of current visual foundation models in aerial settings.

## Background & Motivation

**State of the Field**: Semantic Scene Completion (SSC) is a core task in 3D perception, aiming to jointly estimate the occupancy state and semantic category of dense voxels from partial observations. SSC has been extensively studied in ground-level autonomous driving, with benchmarks such as SemanticKITTI and nuScenes-Occ driving substantial methodological progress.

**Limitations of Prior Work**: (1) SSC research has been almost exclusively focused on ground-vehicle perspectives, leaving aerial (UAV) scenarios largely unexplored and limiting progress on downstream UAV tasks such as obstacle avoidance, path planning, and 3D mapping. (2) Acquiring aerial SSC data faces unique challenges: most UAVs cannot carry LiDAR due to flight regulations and payload constraints, and high-altitude LiDAR point clouds are extremely sparse. (3) Existing SSC datasets rely heavily on LiDAR and cannot be directly adapted to camera-only aerial settings.

**Root Cause**: Autonomous UAVs require 3D scene understanding for self-navigation, yet no suitable SSC benchmark or methodology exists for aerial perspectives. The large viewpoint discrepancy between aerial (top-down) and ground-level (forward-facing) views renders direct transfer of ground-based SSC data and models infeasible.

**Paper Goals**: (1) Design a LiDAR-free, camera-only data generation pipeline for automated aerial SSC annotation. (2) Release the first aerial SSC benchmark, OccuFly, based on this pipeline. (3) Benchmark existing SSC and depth estimation methods on OccuFly to expose the unique challenges of aerial scenes.

**Starting Point**: The authors propose leveraging classical 3D reconstruction (SfM + MVS) to obtain dense point clouds from aerial image sequences, then lifting semantic labels from sparsely annotated images (<10%) into 3D point clouds via projection, thereby automatically generating semantic voxel annotations without any LiDAR dependency.

**Core Idea**: Replace LiDAR with camera-based 3D reconstruction for SSC annotation generation, combined with a sparse label propagation strategy, to construct the first aerial SSC benchmark and systematically evaluate existing methods in aerial scenarios.

## Method

### Overall Architecture

The OccuFly construction pipeline consists of three stages: (1) **Data Acquisition** — UAVs capture image sequences across low, medium, and high altitudes, all four seasons, and urban, industrial, and rural environments. (2) **3D Reconstruction and Annotation** — SfM and MVS reconstruct dense point clouds from the image sequences; pixel-level semantic annotations are provided for fewer than 10% of images, which are then projected to lift 2D labels into 3D point clouds and voxelized into semantic occupancy grids. (3) **Benchmarking** — Multiple SSC and monocular depth estimation methods are evaluated on OccuFly.

### Key Designs

1. **LiDAR-Free Camera-Based Data Generation Framework**:

    - **Function**: Automatically generates SSC annotation data from aerial images without LiDAR.
    - **Mechanism**: Structure-from-Motion (SfM) estimates camera poses, and Multi-View Stereo (MVS) produces dense point clouds. The key innovation lies in semantic label propagation: 2D semantic segmentation annotations are provided for fewer than 10% of images, and pixel labels are back-projected into 3D point clouds using known intrinsic and extrinsic camera parameters. For each 3D point, the final semantic label is determined by majority voting over all projected labels. The annotated semantic point cloud is then voxelized into a standard occupancy grid.
    - **Design Motivation**: Full 3D annotation is prohibitively expensive, whereas mature and efficient 2D annotation tools are widely available. The sparse label propagation strategy achieves complete 3D semantic coverage with less than one-tenth of the annotation effort, substantially reducing dataset construction cost.

2. **Multi-Altitude, Multi-Season Data Collection Strategy**:

    - **Function**: Ensures that the dataset captures the diversity of challenges inherent to aerial scenes.
    - **Mechanism**: Data are collected at three flight altitudes (low, medium, high), covering different viewing ranges and ground resolutions. All four seasons are represented, capturing variations in vegetation, illumination, and weather. Three typical environment types—urban, industrial, and rural—are included. The resulting dataset contains 20,000+ samples across 21 semantic categories.
    - **Design Motivation**: The most critical distinction between aerial and ground-level perspectives is the dramatic change in observation distance and coverage area with altitude. Seasonal variations cause significant appearance changes (e.g., winter snow, dense summer vegetation), which are essential for evaluating model robustness.

3. **Standardized Data Organization and Evaluation Protocol**:

    - **Function**: Ensures seamless integration with the existing SSC research ecosystem.
    - **Mechanism**: OccuFly adopts a data organization format compatible with SemanticKITTI, providing standardized image–voxel–depth triplets. Unified evaluation metrics are defined for SSC (IoU, mIoU) and depth estimation (AbsRel, RMSE, $\delta$ thresholds), along with official train/val/test splits.
    - **Design Motivation**: Compatibility with existing formats allows researchers to directly evaluate and train on OccuFly using established codebases, minimizing the migration barrier.

### Loss & Training

OccuFly is a benchmark dataset and does not propose new training methods. In benchmark experiments, SSC models are trained with per-voxel cross-entropy loss, and depth estimation models are trained with the standard scale-invariant logarithmic loss.

## Key Experimental Results

### Main Results (SSC Benchmark)

| Method | Type | Geometric IoU | Semantic mIoU | Notes |
|--------|------|--------------|--------------|-------|
| MonoScene | Monocular SSC | ~15 | ~5 | Large drop when directly transferred from ground to aerial |
| VoxFormer | Monocular SSC | ~18 | ~6 | Slightly better than MonoScene |
| TPVFormer | Multi-view SSC | ~20 | ~7 | Multi-view input provides modest gains |
| OccFormer | Multi-view SSC | ~22 | ~8 | Best performance on aerial benchmark |

### Depth Estimation Benchmark

| Method | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ | Notes |
|--------|---------|------|---------|-------|
| Depth Anything v2 | ~0.25 | ~8.5 | ~65% | Visual foundation model degrades significantly in aerial settings |
| Metric3D v2 | ~0.22 | ~7.8 | ~70% | Metric depth model slightly better |
| ZoeDepth | ~0.28 | ~9.2 | ~60% | Poor transfer from indoor pretraining |
| Ground-truth upper bound | 0 | 0 | 100% | Reference |

### Key Findings

- All SSC methods that perform well on ground-level data suffer large performance drops in aerial scenarios, with mIoU consistently below 10%.
- Current visual foundation models (e.g., Depth Anything v2) perform substantially worse on aerial depth estimation than in ground-level settings, with AbsRel increasing approximately 2–3×.
- Data collected at high altitude is more challenging than at low altitude, due to smaller ground targets and larger depth ranges.
- Seasonal variation—particularly winter snow cover—has a significant impact on both SSC and depth estimation methods.
- OccuFly reveals a critical gap: existing methods have severely insufficient capacity for modeling top-down aerial geometry.

## Highlights & Insights

- **Filling the data gap in aerial 3D perception**: OccuFly is the first aerial SSC benchmark, with major implications for UAV autonomous flight, urban mapping, and related applications. The dataset construction methodology is itself a contribution—enabling SSC annotation from pure camera imagery without LiDAR.
- **The sparse label propagation strategy** is highly practical: annotating fewer than 10% of images yields complete 3D semantic labels. This idea generalizes to 3D annotated dataset construction in any new domain.
- Systematic benchmarking reveals the fundamental limitations of current methods and charts a clear research agenda for aerial 3D perception. Its CVPR 2026 Oral recognition reflects the community's appreciation for its pioneering role in opening a new research area.

## Limitations & Future Work

- Code and data are not yet fully publicly available, limiting community validation and extension.
- The SfM+MVS-based annotation generation relies on high-quality multi-view reconstruction, which may introduce annotation noise in texture-poor or repetitively textured regions (e.g., farmland, water surfaces).
- The 21 semantic categories may lack sufficient granularity; for instance, different building types or vehicle classes are not distinguished.
- The dataset covers a geographically limited set of scenes; generalization to different climate zones and urban morphologies remains to be validated.
- Annotation quality for dynamic objects (pedestrians, vehicles) may be affected by motion blur and reconstruction failures.

## Related Work & Insights

- **vs. SemanticKITTI**: SemanticKITTI is the de facto ground-level SSC benchmark but is entirely LiDAR-driven and ground-perspective. OccuFly extends SSC to the aerial perspective without LiDAR.
- **vs. nuScenes-Occ**: Also a ground-level benchmark with a surround-camera configuration. OccuFly's cameras are mounted on UAVs, resulting in entirely different viewpoints and scene characteristics.
- **vs. ScanNet / Matterport3D**: Indoor 3D reconstruction benchmarks using RGB-D sensors. OccuFly's pure-RGB reconstruction faces substantially greater challenges in large-scale outdoor environments.
- The LiDAR-free data generation framework presented in this paper is also applicable to satellite remote sensing 3D reconstruction, robotic exploration, and related domains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First aerial SSC benchmark; a pioneering work in this research direction (Oral).
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmarks cover both SSC and depth estimation tasks with multi-method comparisons.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline is clearly presented; challenge analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Establishes a benchmark for aerial 3D perception; highly valuable to the UAV research community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](../../AAAI2026/autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[AAAI 2026\] Unleashing Semantic and Geometric Priors for 3D Scene Completion](../../AAAI2026/autonomous_driving/unleashing_semantic_and_geometric_priors_for_3d_scene_completion.md)
- [\[AAAI 2026\] HD2-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving](../../AAAI2026/autonomous_driving/hd2-ssc_high-dimension_high-density_semantic_scene_completion_for_autonomous_dri.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)

<!-- RELATED:END -->
