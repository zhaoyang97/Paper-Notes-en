---
title: >-
  [Paper Note] SGCDet: Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction
description: >-
  [ICCV 2025][Object Detection][Multi-view 3D detection] SGCDet achieves efficient and accurate multi-view indoor 3D object detection through adaptive sparse 3D voxel construction and geometry-context-aware aggregation, surpassing existing methods without requiring ground-truth geometric supervision.
tags:
  - ICCV 2025
  - Object Detection
  - Multi-view 3D detection
  - indoor scene understanding
  - sparse voxel construction
  - deformable attention
  - occupancy prediction
date: 2026-05-08
content_hash: 2b0ee91cd32e8bc3
---

# SGCDet: Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction

**Conference**: ICCV 2025  
**arXiv**: [2507.18331](https://arxiv.org/abs/2507.18331)  
**Code**: [GitHub](https://github.com/RM-Zhang/SGCDet)  
**Area**: 3D Vision  
**Keywords**: Multi-view 3D detection, indoor scene understanding, sparse voxel construction, deformable attention, occupancy prediction

## TL;DR

SGCDet achieves efficient and accurate multi-view indoor 3D object detection through adaptive sparse 3D voxel construction and geometry-context-aware aggregation, surpassing existing methods without requiring ground-truth geometric supervision.

## Background & Motivation

Multi-view indoor 3D object detection is a fundamental task in embodied AI, AR/VR, and robotics. Existing methods face three core challenges:

**Limited receptive field**: Prior methods (ImVoxelNet, NeRF-Det, etc.) restrict each voxel's receptive field to a fixed location on the image, acquiring features via single-point sampling while ignoring surrounding contextual information. This single-point sampling strategy not only limits a voxel's ability to perceive visual information but also amplifies its dependence on precise geometric information.

**Inefficiency of dense construction**: Existing methods construct high-resolution dense 3D voxel grids, yet the majority of voxels in a 3D scene correspond to free space. Such dense representations fail to account for the inherent sparsity of 3D scenes and introduce unnecessary computational overhead.

**Reliance on geometric supervision**: Several high-performing methods (CN-RMA, ImGeoNet) require ground-truth scene geometry as supervision, limiting their applicability in practical settings where precise geometric data is unavailable.

## Method

### Overall Architecture

SGCDet consists of three core components:
- **Image backbone**: ResNet-50 + FPN for extracting multi-view 2D features
- **View transformation module**: Lifting 2D features into a 3D volumetric representation, comprising sparse voxel construction and geometry-context-aware aggregation
- **Detection head**: Anchor-free design for predicting 3D bounding boxes

### Key Design 1: Sparse Volume Construction

The core idea is a coarse-to-fine progressive 3D volume construction:

1. **Coarse-level construction**: An initial low-resolution 3D volume (e.g., $10\times10\times4$) is first constructed.
2. **Occupancy probability prediction**: A lightweight prediction head estimates the occupancy probability of each voxel.
3. **Adaptive refinement**: The top-$k$ (25%) voxels by occupancy probability are selected for feature refinement.
4. **Iterative optimization**: After $L=2$ refinement stages, the target resolution ($40\times40\times16$) is obtained.

**Innovation in occupancy supervision**: Rather than relying on ground-truth scene geometry, pseudo-labels derived from 3D detection boxes are used to supervise occupancy probability via binary cross-entropy. A voxel is labeled 1 if it falls inside any detection box, and 0 otherwise.

### Key Design 2: Geometry and Context Aware Aggregation

This module consists of intra-view feature sampling and inter-view feature fusion:

**Intra-view Feature Sampling**:
- 2D features are lifted into 3D pixel space via an outer product with the depth distribution.
- A 3D deformable attention mechanism is introduced, which uses the projected point as a query and adaptively aggregates geometric and contextual information within a deformable neighborhood.
- Deformable attention with $M=4$ sampling points is used; offsets and weights are generated from the query feature via a linear layer.

**Inter-view Feature Fusion**:
- A multi-view attention mechanism dynamically adjusts the contribution weights of different views.
- The average-pooled feature across all views serves as the query, while each view's feature serves as key and value.
- Views whose projections fall outside the image boundary are discarded.

### Depth Estimation Network (DepthNet)

Multi-view and monocular depth features are fused as follows:
- The $K=2$ nearest neighboring views are selected to construct a cost volume via plane sweeping.
- Depth range: $[0.2\text{m},\ 5\text{m}]$ with 12 depth bins.
- The multi-view branch provides geometric cues via feature matching; the monocular branch provides detailed structural information.
- Features from both branches are concatenated and passed through a depth decoder to produce the depth distribution.

### Loss & Training

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{det} + \lambda \cdot \mathcal{L}_\text{occ}, \quad \lambda = 0.5$$

- **Detection loss**: Cross-entropy (centerness) + IoU (localization) + Focal loss (classification)
- **Occupancy loss**: Sum of binary cross-entropy losses across all refinement stages

## Key Experimental Results

### Main Results: ScanNet Dataset

| Method | Voxel Res. | mAP@0.25 | mAP@0.50 | Train Mem. | Train Time | Infer. Mem. | FPS |
|--------|-----------|----------|----------|------------|------------|-------------|-----|
| ImVoxelNet | $40^3$ | 46.7 | 23.4 | 11GB | 13h | 9GB | 2.60 |
| NeRF-Det | $40^3$ | 53.5 | 27.4 | 13GB | 14h | 12GB | 1.30 |
| MVSDet | $40^3$ | 56.2 | 31.3 | 35GB | 36h | 28GB | 0.87 |
| ImGeoNet† | $40^3$ | 54.8 | 28.4 | 13GB | 16h | 11GB | 2.50 |
| CN-RMA† | $256^3$ | 58.6 | 36.8 | 43GB | 242h | 12GB | 0.26 |
| **SGCDet** | $40^3$ | **61.2** | **35.2** | 20GB | 19h | 14GB | 1.46 |

†Requires ground-truth geometric supervision. SGCDet uses no ground-truth geometry, achieving +5.0 mAP@0.25 and +3.9 mAP@0.50 over MVSDet, while reducing training memory by 42.9%, training time by 47.2%, inference memory by 50%, and inference time by 40.8%.

### ARKitScenes Dataset

| Method | mAP@0.25 | mAP@0.50 |
|--------|----------|----------|
| MVSDet | 60.7 | 40.1 |
| **SGCDet** | 62.3 | 44.7 |
| **SGCDet-L** ($80^3$) | **70.4** | **57.0** |
| CN-RMA† ($192^3$) | 67.6 | 56.5 |

SGCDet-L even surpasses CN-RMA, which employs higher resolution and ground-truth geometric supervision.

### Ablation Study: Geometry and Context Aware Aggregation

| Setting | 2D Deform | 3D Deform | MV Attn | mAP@0.25 | mAP@0.50 |
|---------|-----------|-----------|---------|----------|----------|
| Baseline | | | | 56.0 | 29.8 |
| +2D Deformable | ✓ | | | 56.2 | 30.5 |
| +3D Deformable | | ✓ | | 59.5 | 34.1 |
| +3D + MV Attn | | ✓ | ✓ | **61.2** | **35.2** |

3D deformable attention yields substantially larger gains than its 2D counterpart (+3.3/+3.6 vs. +0.2/+0.7), as it simultaneously integrates geometric and contextual information.

### Ablation Study: Sparse Volume Construction

| Resolution Strategy | mAP@0.25 | mAP@0.50 | Train Mem. | Infer. Mem. |
|--------------------|----------|----------|------------|-------------|
| Dense $40^3$ (100%) | 61.0 | 36.0 | 31GB | 22GB |
| 10→20→40 (25%) | 61.2 | 35.2 | 20GB | 13GB |
| 10→20→40 (10%) | 57.0 | 31.7 | 19GB | 13GB |

The 25% selection ratio best balances accuracy and efficiency, reducing memory by approximately 40% with negligible performance degradation. Removing the occupancy loss results in a 6.7-point drop in mAP@0.25.

## Highlights & Insights

1. **Adaptive sparse construction**: By refining only high-probability voxels via occupancy prediction, the method reduces computation by over 40% compared to dense construction without performance loss — an effective engineering realization of the intuition that "3D scenes are inherently sparse."
2. **Deformable attention in 3D pixel space**: Operating in the depth-outer-product 3D space rather than directly on 2D feature maps enables simultaneous capture of geometric consistency and semantic context, with ablations demonstrating a significant performance gap.
3. **Pseudo-labels as a substitute for ground-truth geometry**: Using detection boxes as occupancy pseudo-labels completely eliminates dependence on ground-truth geometry, representing an important step toward practical applicability.
4. **View-specific queries**: Unlike the view-agnostic queries in DFA3D, view-specific features are used as queries to accommodate the large camera pose variations characteristic of indoor scenes.

## Limitations & Future Work

1. The top-$k$ selection ratio in sparse construction requires manual tuning; an overly small ratio (10%) leads to significant performance degradation.
2. Occupancy pseudo-labels are derived from detection boxes and cannot effectively model discrete surfaces such as walls and floors.
3. Depth estimation still relies on multi-view matching and may fail in textureless or repetitively textured regions.
4. The method requires 40 images during training and 100 images during inference, imposing non-trivial demands on input image count.

## Related Work & Insights

- **Voxel-based 3D detection**: ImVoxelNet → ImGeoNet → NeRF-Det → MVSDet → SGCDet, progressing from dense to sparse representations and from fixed sampling to adaptive aggregation.
- **Sparse design**: DETR-series sparse queries; occupancy prediction methods still rely on precise geometric supervision.
- **Deformable attention**: DFA3D applies it to autonomous driving BEV settings; this work introduces view-specific query improvements tailored to indoor scenes.

## Rating

- **Novelty**: 7/10 — The combination of sparse construction and 3D deformable aggregation is original, though individual modules are not entirely new.
- **Technical Depth**: 8/10 — The method is systematically designed, with ablations thoroughly validating each component's contribution.
- **Experimental Thoroughness**: 9/10 — Three datasets, extensive ablations, computational cost comparisons, and visualizations.
- **Value**: 8/10 — No ground-truth geometric supervision required, with substantial reductions in computational cost.
- **Overall**: 8/10

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[ICCV 2025\] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)

</div>

<!-- RELATED:END -->
