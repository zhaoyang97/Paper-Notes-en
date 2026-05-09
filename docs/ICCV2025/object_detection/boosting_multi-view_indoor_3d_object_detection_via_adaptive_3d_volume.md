---
title: >-
  [Paper Note] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction
description: >-
  [ICCV 2025][Object Detection][Multi-view 3D detection] This paper proposes SGCDet, a framework that achieves efficient and accurate multi-view indoor 3D object detection without relying on ground-truth scene geometry, via a Geometry and Context-Aware aggregation module (adaptive feature lifting) and a sparse voxel construction strategy (coarse-to-fine adaptive voxel selection).
tags:
  - ICCV 2025
  - Object Detection
  - Multi-view 3D detection
  - indoor scene
  - sparse voxel construction
  - deformable attention
  - occupancy prediction
date: 2026-05-08
content_hash: ec9dea15e42ead9d
---

# Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction

**Conference**: ICCV 2025
**arXiv**: [2507.18331](https://arxiv.org/abs/2507.18331)
**Code**: [Available](https://github.com/RM-Zhang/SGCDet)
**Area**: 3D Vision
**Keywords**: Multi-view 3D detection, indoor scene, sparse voxel construction, deformable attention, occupancy prediction

## TL;DR

This paper proposes SGCDet, a framework that achieves efficient and accurate multi-view indoor 3D object detection without relying on ground-truth scene geometry, via a Geometry and Context-Aware aggregation module (adaptive feature lifting) and a sparse voxel construction strategy (coarse-to-fine adaptive voxel selection).

## Background & Motivation

Multi-view indoor 3D object detection reconstructs a 3D voxel representation from multiple posed images and detects objects therein. Existing methods suffer from two major bottlenecks:

1. **Limited feature lifting**: Prior methods (ImVoxelNet, NeRF-Det, etc.) project voxel centers onto fixed image locations for single-point sampling, resulting in a limited receptive field. This strategy ignores contextual information, over-relies on geometric estimation accuracy, and cannot effectively resolve occlusion during projection.
2. **Inefficient dense voxel construction**: 3D scenes are inherently sparse, yet existing methods construct high-resolution dense 3D voxels, incurring substantial redundant computation in free space.
3. **Dependence on GT geometry**: Certain methods (ImGeoNet, CN-RMA) require ground-truth scene geometry (e.g., TSDF, depth maps) during training, limiting applicability to datasets without GT geometry annotations.

SGCDet aims to address all three issues simultaneously, enabling adaptive and efficient voxel construction.

## Method

### Overall Architecture

SGCDet consists of three components:
- **Image backbone**: ResNet-50 + FPN to extract 2D features (80×60 resolution)
- **View transformation module**: Lifts 2D features into a 3D voxel representation; the core includes DepthNet, sparse voxel construction, and Geometry and Context-Aware aggregation
- **Detection head**: Anchor-free head predicting 3D bounding boxes

### Key Designs

**1. Geometry and Context-Aware Aggregation (GCA)**

Unlike single-point sampling, the GCA module achieves adaptive feature aggregation in two steps:

- **Intra-view feature sampling**: The 2D features are first outer-producted with a depth distribution to produce 3D pixel-space features $F_n^{3D} \in \mathbb{R}^{H \times W \times D \times C}$. Deformable attention is then applied in this 3D space, using the sampled feature at the projected location as a query to aggregate geometry and context information within a deformable neighborhood. This allows each voxel to adaptively integrate information from its surroundings rather than being restricted to a fixed projection point.
- **Inter-view feature fusion**: A multi-view attention mechanism is introduced, using the average feature across all views as queries and per-view features as keys/values, to dynamically weight each view's contribution to the final voxel feature.

**2. Sparse Voxel Construction Strategy**

A coarse-to-fine pipeline is adopted to construct 3D voxels:
- A low-resolution coarse voxel grid (e.g., 10×10×4) is first constructed.
- Over $L$ stages, the grid is progressively upsampled: each stage applies 2× upsampling → predicts occupancy probability → selects the top-$k$% high-probability voxels for feature refinement.
- The expensive GCA feature lifting is applied only to voxels likely to contain objects, substantially reducing redundant computation in free space.

**3. DepthNet**

A lightweight network that fuses multi-view depth features (cost volume built via plane sweeping) with monocular depth features to provide a depth distribution prior for 2D-to-3D projection.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{det} + \lambda \mathcal{L}_{occ}$

- **Detection loss** $\mathcal{L}_{det}$: centerness cross-entropy + IoU regression + classification focal loss
- **Occupancy loss** $\mathcal{L}_{occ}$: pseudo-labels generated from 3D bounding boxes (a voxel center is labeled 1 if it falls inside any GT box); BCE loss applied to occupancy predictions at each stage. Weight $\lambda=0.5$
- Key contribution: occupancy supervision requires only 3D bounding boxes, eliminating the need for GT geometry

Training uses AdamW + cosine decay; 40 training images / 100 test images; trained for 12 epochs on ScanNet.

## Key Experimental Results

### Main Results

**ScanNet comparison (methods without GT geometry):**

| Method | Voxel Res. | mAP@0.25 | mAP@0.50 | Train Mem. (GB) | Train Time (h) | Infer. Mem. (GB) | FPS |
|--------|-----------|----------|----------|----------------|----------------|-----------------|-----|
| ImVoxelNet | 40³×16 | 46.7 | 23.4 | 11 | 13 | 9 | 2.60 |
| NeRF-Det | 40³×16 | 53.5 | 27.4 | 13 | 14 | 12 | 1.30 |
| MVSDet | 40³×16 | 56.2 | 31.3 | 35 | 36 | 28 | 0.87 |
| **SGCDet** | 40³×16 | **61.2** | **35.2** | 20 | 19 | 14 | 1.46 |

**ARKitScenes dataset:**

| Method | mAP@0.25 | mAP@0.50 |
|--------|----------|----------|
| MVSDet | 60.7 | 40.1 |
| **SGCDet** | **62.3** | **44.7** |
| **SGCDet-L** | **70.4** | **57.0** |

### Ablation Study

**Geometry and Context-Aware Aggregation ablation:**

| Setting | 2D Deformable | 3D Deformable | Multi-view Attn. | mAP@0.25 | mAP@0.50 |
|---------|--------------|--------------|-----------------|----------|----------|
| (a) Baseline | | | | 56.0 | 29.8 |
| (b) | ✓ | | | 56.2 | 30.5 |
| (c) | | ✓ | | 59.5 | 34.1 |
| (d) Full | | ✓ | ✓ | **61.2** | **35.2** |

**Sparse voxel construction ablation (selection ratio):**

| Setting | Selection Ratio | mAP@0.25 | Train Mem. (GB) | FPS |
|---------|----------------|----------|----------------|-----|
| (a) No sparsity | 100% | 61.0 | 31 | 1.33 |
| (e) SGCDet | 25% | 61.2 | 20 | 1.46 |
| (f) Over-sparse | 10% | 57.0 | 19 | 1.53 |

### Key Findings

- 3D deformable attention outperforms 2D deformable attention (+3.3/3.6 mAP) by jointly integrating geometry and contextual information.
- A 25% selection ratio preserves accuracy while significantly reducing computation (−35.5% memory, +10% FPS).
- SGCDet improves mAP@0.5 by 3.9 over MVSDet while reducing training memory by 42.9% and training time by 47.2%.
- SGCDet surpasses ImGeoNet, which requires GT geometry, without using any GT geometry.

## Highlights & Insights

1. **Adaptivity as a unifying principle**: Feature aggregation regions are adaptive (deformable attention), view contribution weights are adaptive (multi-view attention), and voxel selection is adaptive (occupancy probability).
2. **Elegant pseudo-label strategy**: Occupancy pseudo-labels are generated solely from 3D bounding boxes, elegantly circumventing the dependency on GT geometry and broadening applicability.
3. **Unified feature representation in 3D pixel space**: Outer-producting 2D features with depth distributions and operating in 3D pixel space enables deformable attention to flexibly sample in both spatial and depth dimensions simultaneously.

## Limitations & Future Work

1. The top-$k$ hard-threshold selection in sparse voxel construction may miss small objects near boundaries.
2. DepthNet uses a fixed $K=2$ neighboring views to build the cost volume, without accounting for view coverage diversity.
3. Pseudo occupancy labels are based on axis-aligned boxes, which may be inaccurate for non-axis-aligned objects (e.g., rotated boxes in ARKitScenes).
4. Multi-scale feature fusion and higher-resolution feature strategies remain unexplored.

## Related Work & Insights

- The 3D deformable attention concept from DFA3D is borrowed and improved upon, transitioning from view-agnostic to view-specific queries.
- The sparse design draws inspiration from DETR-style query proposals and occupancy prediction methods.
- Compared to MVSDet's 3DGS self-supervised strategy, the pseudo-label scheme proposed here is simpler and more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combined design of the aggregation module and sparse strategy is original, though individual contributions are incremental)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three datasets, detailed ablations, efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Significant SOTA improvement with substantial efficiency gains; high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SGCDet: Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume_constructi.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[ICCV 2025\] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)

</div>

<!-- RELATED:END -->
