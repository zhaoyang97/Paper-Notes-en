---
title: >-
  [Paper Note] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection
description: >-
  [CVPR 2025][3D Vision][3D object detection] Proposes CPDet3D, the first unified indoor-outdoor sparsely-supervised 3D object detection method. It mines the categories of unlabeled objects through class-aware prototype clustering (cross-scene Sinkhorn-Knopp optimal transport matching) and recovers missed detections using multi-label cooperative refinement (pseudo labels + prototype labels). It achieves 78% of fully-supervised performance on ScanNet V2, 90% on SUN RGB-D…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D object detection"
  - "sparse supervision"
  - "prototype learning"
  - "optimal transport"
  - "self-training"
date: 2026-05-08
content_hash: 789609fbcdca6214
---

# Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.21099](https://arxiv.org/abs/2503.21099)  
**Code**: [GitHub](https://github.com/zyrant/CPDet3D)  
**Area**: 3D vision  
**Keywords**: 3D object detection, sparse supervision, prototype learning, optimal transport, self-training

## TL;DR

Proposes CPDet3D, the first unified indoor-outdoor sparsely-supervised 3D object detection method. It mines the categories of unlabeled objects through class-aware prototype clustering (cross-scene Sinkhorn-Knopp optimal transport matching) and recovers missed detections using multi-label cooperative refinement (pseudo labels + prototype labels). It achieves 78% of fully-supervised performance on ScanNet V2, 90% on SUN RGB-D, and 96% on KITTI using only 1 annotation per scene.

## Background & Motivation

**Background**: 3D object detection heavily relies on precise annotations, which are highly expensive. Sparse supervision (annotating only a few objects per scene) is a key direction to reduce annotation costs, but existing methods are only applicable to outdoor autonomous driving scenarios.

**Limitations of Prior Work**:
1. **SS3D / CoIn**: Rely on ground-truth (GT) sampling strategies (copying annotated objects to other scenes) to ensure a single scene covers all classes before mining unlabeled objects.
2. **Infeasibility of GT sampling indoors**: Indoor objects have scene-specific context (e.g., toilets cannot be placed in living rooms) and cannot be simply copied.
3. **Semi-supervised methods**: Domain gaps exist between annotated and unannotated scenes, and annotating entire scenes is time-consuming.

**Core Motivation**: Can a method be designed **without relying on GT sampling** that mines unlabeled objects by learning class-aware representations across scenes, thereby unifying indoor and outdoor scenarios?

## Method

### Overall Architecture

Two-stage training paradigm:
1. **Stage 1**: Train the initial detector + prototype miner module with sparse annotations.
2. **Stage 2**: Generate pseudo labels with the initial model $\to$ prototype mining + multi-label cooperative refinement $\to$ self-training.

### Key Designs

#### 1. Prototype-based Object Mining

**Class-aware Prototype Clustering:**
- Each class maintains $O$ prototypes $\bm{P}_k \in \mathbb{R}^{O \times C}$ (default $O=10$) to capture intra-class feature diversity.
- In each forward pass: detector features are projected via MLP $\to$ class-aware masks extract annotated object features $\to$ **Sinkhorn-Knopp optimal transport** computes the feature-prototype assignment matrix.
- Momentum update of prototypes: $\bm{p}'_{k,i} \leftarrow \mu \bm{p}_{k,i} + (1-\mu)\frac{1}{N_k}\sum \bm{F}_{k,i}$.
- Set a 1000-iteration warm-up phase to ensure prototypes gain class discriminativeness before matching.

**Prototype Label Matching:**
- Calculate the affinity matrix between all features and prototypes: $\bm{A} = \bm{F}^\top \bm{P}$.
- Propagation probability: $\bm{W} = \bm{S} \odot \bm{A}'$ (classification score $\times$ optimal prototype affinity).
- Assign classes $\bm{C}_f = \arg\max_{k} \bm{W}$ $\to$ filter background/annotated/out-of-bound regions $\to$ obtain **prototype labels**.

#### 2. Multi-label Cooperative Refinement

**Iterative Pseudo Label Generation:**
- Score Filter: Filter low-confidence predictions with a classification score threshold $\alpha_{cls}=0.2$.
- IoU Filter: $\alpha_{iou}=0.5$ to remove overlapping pseudo labels.
- Collision Filter: $\alpha_{col}=0.2$ to avoid conflict with ground-truth sparse labels.

**Prototype Label Cooperation:**
- Pseudo labels use high thresholds to guarantee quality $\to$ but lead to missed detections.
- Leverage prototype labels to **fill in foreground regions missed by pseudo labels**.
- Triple label cooperation: sparse ground-truth labels + pseudo labels + prototype labels $\to$ maximize recall.

#### 3. Prototype-Feature Contrastive Loss

Info-NCE loss $\mathcal{L}_{pcon}$ pulls same-class prototype-feature pairs together and pushes different classes apart, enhancing clustering quality in the feature space.

### Loss & Training

- **Stage 1**: $\mathcal{L}_{stage1} = \mathcal{L}_{det} + \mathcal{L}_{pcon} + \mathcal{L}_{pcls}$
    - $\mathcal{L}_{det}$: TR3D detection loss
    - $\mathcal{L}_{pcon}$: Prototype-feature contrastive loss (Info-NCE)
    - $\mathcal{L}_{pcls}$: Prototype classification loss (Focal Loss)
- **Stage 2**: $\mathcal{L}_{stage2} = \mathcal{L}_{stage1} + \mathcal{L}_{ref}$ (pseudo-label detection loss)

## Key Experimental Results

### Main Results

**Indoor Scenarios (1 object/scene, TR3D backbone)**:

| Method | ScanNet V2 mAP@0.25 | mAP@0.5 | SUN RGB-D mAP@0.25 | mAP@0.5 |
|------|---------------------|---------|---------------------|---------|
| TR3D (Sparse) | 37.6 | 21.8 | 53.9 | 36.3 |
| SparseDet (ICCV) | 46.0 | 28.2 | 56.7 | 38.8 |
| **CPDet3D (Ours)** | **56.1** | **40.8** | **60.2** | **43.3** |
| TR3D (Full Sup.) | 72.0 | 57.4 | 66.3 | 49.6 |

Shows an improvement of **+10.1 mAP@0.25** over SparseDet on ScanNet V2, reaching **78%** of the fully-supervised performance.

**Outdoor Scenarios (KITTI, 2% annotation, Voxel-RCNN backbone)**:

| Method | Easy | Moderate | Hard |
|------|------|----------|------|
| Voxel-RCNN (Sparse) | 72.5 | 54.9 | 44.8 |
| CoIn++ | 92.0 | 79.5 | 71.5 |
| **CPDet3D** | **94.1** | **82.2** | **72.6** |
| Voxel-RCNN (Full) | 92.3 | 85.2 | 82.8 |

Shows an improvement of **+2.7 AP** over CoIn++ on Moderate, reaching **96%** of fully-supervised performance.

### Ablation Study

**Label Recall Statistics (ScanNet V2)**:

| Label Type | Sparse | Prototype | Pseudo | mAR |
|---------|--------|-----------|--------|-----|
| Sparse Only | ✓ | | | 8.3 |
| + Prototype Label | ✓ | ✓ | | 47.8 |
| + Pseudo Label | ✓ | | ✓ | - |
| Cooperation of Three | ✓ | ✓ | ✓ | Best |

Prototype labels boost recall from 8.3 to 47.8, demonstrating the effectiveness of cross-scene prototype mining.

### Key Findings

- **Setting the number of prototypes to 10 per class** is optimal; too few fail to capture intra-class diversity, while too many introduce noise.
- Prototypes exhibit class discriminativeness only after a **1000-iteration warm-up** (validated via t-SNE visualization).
- A momentum coefficient of $\mu=0.9$ is optimal; smaller values lead to excessively fast updates and unstable prototypes.
- Comparison with semi-supervised methods (under the same annotation budget): CPDet3D (54.6 mAP@0.25) > DQS3D (49.2), proving that sparse supervision combined with prototype mining outperforms the semi-supervised paradigm.

## Highlights & Insights

1. **First unified indoor-outdoor sparsely-supervised 3D detection method**: Bypass the limitation of GT sampling via prototype learning, marking a significant paradigm breakthrough.
2. **Optimal transport + momentum prototypes**: Sinkhorn-Knopp matching avoids degenerate solutions (e.g., mapping all features to the same prototype), while momentum updates ensure training stability.
3. **Triple label cooperation**: Smart exploitation of the complementarity among three types of labels (ground-truth, pseudo, and prototype labels), using high-threshold pseudo labels to guarantee accuracy and prototype labels to improve recall.
4. **Extremely low annotation cost**: Achieves ~80%+ of fully-supervised performance using only 1 annotation per scene, demonstrating outstanding annotation efficiency.

## Limitations & Future Work

1. Prototype labels **only provide class targets** without bounding box regression information, which limits their contribution to localization accuracy.
2. Using a fixed number of prototypes ($O=10$) per class might be sub-optimal for datasets with large intra-class variations (e.g., ScanNet with 18 classes).
3. The two-stage training increases implementation complexity; **end-to-end training** is worth exploring.
4. Prototype transfer in **open-vocabulary / zero-shot generalization** scenarios remains unaddressed.

## Related Work & Insights

- **SS3D** (Liu et al., CVPR): Pioneering work in outdoor sparsely-supervised detection, relying on GT sampling.
- **TR3D**: A simple and efficient 3D detector, used as the indoor baseline in this paper.
- **ProtoSeg / ContrastiveSeg**: Applications of 2D prototype learning in semantic segmentation, which inspired introducing prototypes into 3D detection.
- **Insight**: The paradigm of combining prototypes and optimal transport for cross-scene mining can be extended to other sparsely-supervised tasks like 3D instance segmentation and 3D semantic segmentation.

## Rating

⭐⭐⭐⭐ — Valuable problem formulation (first unified indoor-outdoor framework), complete method design (prototype mining + multi-label refinement), and comprehensive experiments (3 datasets + multiple baselines).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[CVPR 2025\] FSHNet: Fully Sparse Hybrid Network for 3D Object Detection](fshnet_fully_sparse_hybrid_network_for_3d_object_detection.md)
- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](../../ICCV2025/3d_vision/unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[CVPR 2025\] MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)

</div>

<!-- RELATED:END -->
