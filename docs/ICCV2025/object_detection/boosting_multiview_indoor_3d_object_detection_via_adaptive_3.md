---
title: >-
  [Paper Note] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction
description: >-
  [ICCV 2025][Object Detection][multi-view 3D object detection] SGCDet achieves state-of-the-art performance in multi-view indoor 3D object detection without ground-truth geometric supervision, through a geometry- and context-aware aggregation module (3D deformable attention + multi-view attention fusion) and an occupancy-probability-based sparse voxel construction strategy, while substantially reducing computational overhead.
tags:
  - ICCV 2025
  - Object Detection
  - multi-view 3D object detection
  - indoor scene
  - sparse voxel construction
  - deformable attention
  - occupancy prediction
date: 2026-05-08
content_hash: 0942c3b80b1af135
---

# Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction

**Conference**: ICCV 2025
**arXiv**: [2507.18331](https://arxiv.org/abs/2507.18331)
**Code**: [https://github.com/RM-Zhang/SGCDet](https://github.com/RM-Zhang/SGCDet)
**Area**: 3D Vision
**Keywords**: multi-view 3D object detection, indoor scene, sparse voxel construction, deformable attention, occupancy prediction

## TL;DR

SGCDet achieves state-of-the-art performance in multi-view indoor 3D object detection without ground-truth geometric supervision, through a geometry- and context-aware aggregation module (3D deformable attention + multi-view attention fusion) and an occupancy-probability-based sparse voxel construction strategy, while substantially reducing computational overhead.

## Background & Motivation

Indoor 3D object detection is a core capability for embodied AI and AR/VR applications. Traditional methods rely on expensive 3D sensors to acquire point clouds; recent work has shifted toward multi-view image-based 3D detection. The central challenge lies in constructing high-quality 3D voxel representations from 2D images.

Prior methods exhibit two critical bottlenecks:

1. **Limited feature sampling**: Methods such as ImVoxelNet project each voxel onto the image for single-point sampling, resulting in an extremely limited receptive field and inability to handle occlusion. Subsequent methods (CN-RMA, MVSDet) introduce explicit geometric constraints but either rely on GT geometry or incur prohibitive computational cost.
2. **Dense voxel redundancy**: Existing methods construct complete dense 3D voxel grids, yet indoor scenes consist largely of free space, leading to severe computational waste.

## Core Problem

How can the quality of 2D-to-3D feature projection be improved (addressing single-point sampling and occlusion) while reducing redundant computation in the 3D volume representation, without relying on ground-truth scene geometry?

## Method

### Overall Architecture

SGCDet comprises three components: (1) an image backbone (ResNet-50 + FPN) for 2D feature extraction; (2) a view transformation module that lifts 2D features into 3D voxels; and (3) a detection head for 3D bounding box prediction. The core innovations reside in the view transformation module and consist of two key designs.

### Key Designs

**Design 1: Geometry- and Context-Aware Aggregation (GCA)**

Rather than projecting voxel centers onto the image for single-point sampling, SGCDet performs adaptive aggregation in two steps:

- **Intra-view Feature Sampling**: A DepthNet first estimates the depth distribution; 2D features are lifted into a 3D pixel-space representation via outer product, $\mathbf{F}_n^{3D} = \mathbf{F}_n^{2D} \otimes \mathbf{D}_n$. Instead of sampling at the projected location directly, the projected-point feature serves as a query, and 3D deformable attention aggregates geometric and contextual information within a local neighborhood. Ablation studies show that 3D deformable attention substantially outperforms its 2D counterpart, since 2D deformable attention suffers from depth ambiguity.

- **Inter-view Feature Fusion**: Appearance and scale vary considerably across viewpoints, making simple averaging suboptimal. SGCDet uses the mean-pooled feature over all views as the query and each view's feature as key/value, dynamically weighting each view's contribution via standard attention.

Compared to the view-agnostic queries in DFA3D, SGCDet uses view-specific queries for intra-view aggregation, which is better suited to the large camera pose variation characteristic of indoor scenes.

**Design 2: Sparse Volume Construction**

A coarse-to-fine strategy is adopted to progressively upsample voxels:

1. A low-resolution (e.g., $10\times10\times4$) coarse voxel grid is constructed first.
2. Over $L$ stages, the resolution is doubled at each stage:
    - A lightweight occupancy prediction head estimates the occupancy probability of each voxel.
    - Only the top-$k$% (default 25%) voxels by occupancy probability are selected for GCA feature refinement.
    - A residual connection is applied: $\mathbf{V}_l = \mathbf{V}_l^{init} + \mathcal{P}(\mathbf{P}_l, \{\mathbf{F}_n^{2D}\}, \{\mathbf{D}_n\})$

**Occupancy supervision design**: Rather than requiring GT scene geometry, pseudo-labels are generated from 3D bounding boxes — voxels inside boxes are labeled 1, otherwise 0. Although these pseudo-labels are noisy (not all space inside a box is occupied), the top-25% selection strategy at inference is sufficient to cover regions with actual objects.

**DepthNet**: Multi-view depth features (built via plane sweep cost volume) and monocular depth features (capturing image-level detail) are concatenated and decoded to produce the depth distribution, with the two branches complementing each other.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{det} + 0.5 \cdot \mathcal{L}_{occ}$$

- $\mathcal{L}_{det}$: anchor-free detection loss = centerness CE loss + IoU loss + classification focal loss
- $\mathcal{L}_{occ}$: sum of BCE losses on occupancy probabilities across all stages

Training configuration: AdamW optimizer, lr = 0.0002, cosine decay; 12 epochs on ScanNet/ARKitScenes, 30 epochs on ScanNet200; 40 images during training, 100 images during testing.

## Key Experimental Results

| Dataset | Metric | SGCDet | Prev. SOTA (MVSDet) | Gain |
|---------|--------|--------|---------------------|------|
| ScanNet | mAP@0.25 | 61.2 | 56.2 | **+5.0** |
| ScanNet | mAP@0.50 | 35.2 | 31.3 | **+3.9** |
| ARKitScenes | mAP@0.25 | 62.3 | 60.7 | +1.6 |
| ARKitScenes | mAP@0.50 | 44.7 | 40.1 | **+4.6** |
| ScanNet200 (SGCDet-L) | mAP@0.25 | 28.9 | ImGeoNet 22.3 | **+6.6** |

Computational efficiency vs. MVSDet: training memory ↓42.9% (20 vs. 35 GB), training time ↓47.2% (19 vs. 36 h), inference memory ↓50% (14 vs. 28 GB), FPS 1.46 vs. 0.87 (↑67.8%).

SGCDet-L achieves 70.4/57.0 on ARKitScenes, surpassing even CN-RMA (67.6/56.5), which uses GT geometric supervision.

### Ablation Study

1. **3D vs. 2D deformable attention**: The 3D variant yields +3.5/+4.3 mAP gain, while the 2D variant contributes only +0.2/+0.7, confirming that performing deformable attention in 3D pixel space jointly captures geometry and context.
2. **Multi-view attention**: Adds +1.7/+1.1 on top of 3D deformable attention, validating the value of dynamic view weighting.
3. **Selection ratio**: 25% vs. 100% yields nearly identical performance (61.2 vs. 61.0) while reducing memory from 31 to 20 GB. 10% is too aggressive and causes a large performance drop (−4.2 mAP@0.25).
4. **Occupancy loss is essential**: Removing it causes a sharp drop of −6.7/−6.2, demonstrating that explicit occupancy supervision is critical for sparse construction.
5. **Depth quality upper bound**: Adding depth supervision yields 62.2/37.1; GT depth yields 64.3/42.3, suggesting that improved depth estimation can further boost performance.
6. **Robustness to annotation noise**: With 15% random box dropping and 15% random scaling, SGCDet drops only 0.5/1.6, compared to 0.8/2.2 for ImGeoNet.

## Highlights & Insights

1. **Using bounding boxes as occupancy pseudo-labels is a clever trick**: It avoids dependency on GT geometry while providing sufficiently informative occupancy supervision. The top-$k$ selection strategy at inference further compensates for the imprecision of pseudo-labels.
2. **3D deformable attention substantially outperforms its 2D counterpart**: While performing deformable sampling in 2D feature maps may seem more natural, experiments demonstrate that adaptive sampling along the depth dimension is necessary to genuinely resolve depth ambiguity.
3. **Sparsification yields across-the-board gains**: A 25% selection ratio dramatically reduces resource consumption while preserving accuracy — highly valuable for real-world deployment.
4. **DepthNet's dual-branch design**: The multi-view branch provides geometrically consistent cross-view constraints, while the monocular branch preserves fine image-level structural detail; the two branches are mutually complementary.

## Limitations & Future Work

1. **Depth estimation remains a significant bottleneck**: The GT depth upper bound (64.3/42.3 vs. 61.2/35.2) indicates that the model is limited by depth estimation accuracy; incorporating stronger depth estimation modules or pretrained depth foundation models is a promising direction.
2. **Coarseness of pseudo-labels**: Much of the space inside bounding boxes is actually free; generating more accurate pseudo-occupancy labels via shape priors or self-supervised signals could yield further improvements.
3. **Fixed top-$k$ selection**: The 25% ratio is fixed, yet object density varies considerably across scenes; an adaptive selection strategy may be more appropriate.
4. **Larger backbones and pretrained weights unexplored**: The current model uses ResNet-50; Vision Transformers or larger pretrained models may provide additional gains in feature representation capacity.

## Related Work & Insights

| Method | GT Geometry Required | Feature Lifting Strategy | Efficiency |
|--------|---------------------|--------------------------|------------|
| ImVoxelNet | ✗ | Ray-based average | High, but low accuracy |
| ImGeoNet | ✓ | Ray + opacity post-processing | Medium |
| NeRF-Det | ✗ | NeRF ray + opacity post-processing | Medium |
| CN-RMA | ✓ | TSDF-guided + multi-stage | Low (243 h training) |
| MVSDet | ✗ | MVS depth + 3DGS self-supervision | Low (35 GB memory) |
| **SGCDet** | **✗** | **3D deform. attn + sparse construction** | **High (20 GB, 1.46 FPS)** |

SGCDet comprehensively outperforms all methods that do not use GT geometry, and its efficiency approaches or surpasses methods that do.

## Connection to My Research

- The occupancy prediction and sparsification paradigm is transferable to autonomous driving 3D occupancy prediction (e.g., open-vocabulary 3D occupancy prediction), and the pseudo-label generation strategy offers a reference for scenarios lacking dense GT annotations.
- The design philosophy of 3D deformable attention — performing adaptive sampling in the lifted 3D space rather than on the 2D plane — is applicable to any task requiring modeling of depth uncertainty.
- The multi-view + monocular dual-branch fusion pattern in DepthNet constitutes a general depth estimation paradigm reusable in other multi-view 3D understanding tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of 3D deformable attention and bbox-based pseudo-occupancy labels is sufficiently novel; the removal of GT geometry dependency has clear engineering and academic value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, detailed ablations (five groups), computational cost analysis, robustness experiments, and visualizations — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, well-motivated, with intuitive figures and rigorous mathematical notation.
- **Value**: ⭐⭐⭐ The occupancy prediction and sparsification strategies are transferable, though the connection to the primary current research direction is moderate.

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
