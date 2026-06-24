---
title: >-
  [Paper Note] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][Multi-Dataset Training] Proposes PoseBH, which achieves unified training across datasets with different skeleton definitions (e.g., humans, animals, hands) via non-parametric keypoint prototypes (Sinkhorn-Knopp online clustering) and cross-type self-supervision (CSS). It improves upon ViTPose++ by 11.2 AP on the APT-36K animal video dataset, demonstrating the effectiveness of cross-type knowledge transfer.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Multi-Dataset Training"
  - "Keypoint Prototype"
  - "Cross-Skeleton Transfer"
  - "Sinkhorn Clustering"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: 15f1a740a52ee49b
---

# PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2505.17475](https://arxiv.org/abs/2505.17475)  
**Code**: [https://github.com/uyoung-jeong/PoseBH](https://github.com/uyoung-jeong/PoseBH)  
**Area**: Human Understanding / Pose Estimation  
**Keywords**: Multi-Dataset Training, Keypoint Prototype, Cross-Skeleton Transfer, Sinkhorn Clustering, Self-Supervised Learning

## TL;DR

Proposes PoseBH, which achieves unified training across datasets with different skeleton definitions (e.g., humans, animals, hands) via non-parametric keypoint prototypes (Sinkhorn-Knopp online clustering) and cross-type self-supervision (CSS). It improves upon ViTPose++ by 11.2 AP on the APT-36K animal video dataset, demonstrating the effectiveness of cross-type knowledge transfer.

## Background & Motivation

### Background

**Background**: Pose estimation datasets often have different skeleton definitions—COCO has 17 human keypoints, AP-10K has animal keypoints, and InterHand has hand keypoints. The standard approach is to train models independently for each skeleton, which discards shared knowledge across datasets.

**Limitations of Prior Work**: Multi-dataset joint training faces two critical challenges: (1) different skeletons have varying numbers and semantics of keypoints, making it impossible to share a prediction head; (2) keypoints labeled in one dataset are unlabeled in others, leading to missing labels.

**Key Challenge**: Skeletons of different entities appear completely different, yet joint types share substantial commonalities—for example, "bending joints" and "end joints" exist across humans, animals, and hands.

**Key Insight**: Prototype learning can be utilized to discover shared prototypes across skeletons in the embedding space, letting clustering automatically find correspondences without manual definition.

**Core Idea**: Non-parametric keypoint prototypes + cross-type self-supervision = unified pose estimation across all skeleton types.

### Mechanism

**Goal**: ### Key Designs

1. **Non-parametric keypoint prototype**:

    - Function: Learn keypoint representations shared across datasets in the embedding space
    - Mechanism: Maintain a $J \times M \times F$ prototype matrix ($J$ keypoints $\times$ $M$ prototypes $\times$ $F$-dimensional features) for each keypoint type, and update prototypes via online Sinkhorn-Knopp clustering.


## Method

### Key Designs

1. **Non-parametric keypoint prototype**:

    - Function: Learn keypoint representations shared across datasets in the embedding space
    - Mechanism: Maintain a $J \times M \times F$ prototype matrix ($J$ keypoints $\times$ $M$ prototypes $\times$ $F$-dimensional features) for each keypoint type, and update prototypes via online Sinkhorn-Knopp clustering. Classification is performed during prediction using the distance between pixel features and prototypes.
    - Design Motivation: Eliminates the need to manually define relationships like "human elbow = cat front knee." Prototypes cluster automatically during training.

2. **Cross-type Self-Supervision (CSS)**:

    - Function: Perform self-supervised learning using predictions of unlabeled keypoint types.
    - Mechanism: For samples in each mixed batch, predictions are made using both the keypoint head and the embedding head. Highly confident predictions from one head serve as pseudo-labels for the other head via weighted averaging.
    - Design Motivation: Animal data is unlabeled in human datasets, but the model can make predictions based on the "joint" concepts learned from human training, and vice versa.

### Loss & Training

$\mathcal{L}_{MDT} = \mathcal{L}_{KPL} + \mathcal{L}_{CSS}$. The keypoint loss includes the pixel-prototype contrastive loss ($\mathcal{L}_{PPC}$) and the pixel-prototype distance loss ($\mathcal{L}_{PPD}$). Three-stage progressive training is used.

## Key Experimental Results

### Main Results

| Dataset | PoseBH (ViT-B) | ViTPose++ | Gain |
|--------|---------------|-----------|------|
| COCO | 77.3 AP | 76.5% | +0.8 |
| AP-10K Animal | 75.0 AP | 74.1% | +0.9 |
| **APT-36K Video Animal** | **87.2 AP** | **76.0%** | **+11.2** |
| InterHand Hand | 87.1 AUC | 86.2% | +0.9 |

### Key Findings
- **APT-36K achieves the largest gain (+11.2)**—being a video dataset, multi-dataset pretraining provides superior temporal understanding.
- Prototypical clustering successfully discovers shared structures across different types.
- CSS contributes +0.2 to the average score (totaling +2.4 compared to the baseline).

## Highlights & Insights
- **Philosophy of "Beyond Human Pose Estimation"**—Not just a better human pose estimator, but a unified joint detector for all species/objects.
- **Substantial gain of +11.2 on APT-36K**—Demonstrates that cross-type knowledge transfer holds massive value for data-scarce domains like animal video.

## Limitations & Future Work
- Few-shot prototype learning is still required; fully zero-shot cross-skeleton transfer is not yet feasible.
- CSS requires similar dataset distributions.
- The 3D domain remains unexplored.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel prototype-driven cross-skeleton unified learning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across four categories of datasets: human, animal, hand, and video.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Provides a scalable framework for unified pose estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](../../ECCV2024/human_understanding/worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2025\] Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)

</div>

<!-- RELATED:END -->
