---
title: >-
  [Paper Note] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation
description: >-
  [CVPR 2025][Autonomous Driving][Semi-supervised learning] This paper proposes the AIScene framework, which leverages intra-scene consistency (point erasure strategy) and inter-scene affinity (MixPatch + InsFill cross-scene augmentation) to improve semi-supervised LiDAR segmentation by 1.9 mIoU on SemanticKITTI using only 1% labels.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Semi-supervised learning"
  - "LiDAR segmentation"
  - "point erasure"
  - "cross-scene augmentation"
  - "pseudo-labeling"
date: 2026-05-08
content_hash: 490e554d1f35a4ae
---

# Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2408.11280](https://arxiv.org/abs/2408.11280)  
**Code**: [https://github.com/azhuantou/AIScene](https://github.com/azhuantou/AIScene)  
**Area**: Autonomous Driving  
**Keywords**: Semi-supervised learning, LiDAR segmentation, point erasure, cross-scene augmentation, pseudo-labeling

## TL;DR
This paper proposes the AIScene framework, which leverages intra-scene consistency (point erasure strategy) and inter-scene affinity (MixPatch + InsFill cross-scene augmentation) to improve semi-supervised LiDAR segmentation by 1.9 mIoU on SemanticKITTI using only 1% labels.

## Background & Motivation

**Background**: Semi-supervised LiDAR semantic segmentation trains models using a small amount of labeled data and a large amount of unlabeled data. Prevailing methods adopt a teacher-student framework, where the teacher model generates pseudo-labels to train the student.

**Limitations of Prior Work**: (1) Intra-scene inconsistency—the forward pass processes all points, whereas the backward pass only computes losses for points with pseudo-labels, resulting in asymmetric information flow. (2) Data augmentations that simply concatenate two scenes offer limited semantic diversity and cannot cover complex scene combinations.

**Key Challenge**: The asymmetric information flow of forward and backward propagation under pseudo-labeling strategies—the forward pass perceives the complete scene, while the backward pass only sees local regions covered by pseudo-labels, forcing the model to learn inconsistent representations.

**Goal**: To improve semi-supervised LiDAR segmentation from the perspectives of both intra-scene consistency and inter-scene diversity.

**Key Insight**: Point erasure: removing points lacking pseudo-labels in the forward pass to keep forward and backward propagation consistent; patch/instance-level cross-scene augmentation: mixing patches and instances from multiple scenes to provide richer semantic combinations.

**Core Idea**: Ensuring intra-scene consistency by erasing points without pseudo-labels and enhancing inter-scene diversity through multi-scene patch/instance hybrid augmentation, cooperatively boosting semi-supervised LiDAR segmentation.

## Method

### Overall Architecture
Teacher-Student EMA framework $\rightarrow$ teacher generates pseudo-labels (threshold $\tau = 0.9$) $\rightarrow$ Point Erasure: forward + backward passes after erasing low-confidence points $\rightarrow$ MixPatch: sampling BEV patches from a scene pool to replace parts of the current scene $\rightarrow$ InsFill: sampling object instances from an instance pool to fill gaps in the scene.

### Key Designs

1. **Point Erasure**:

    - **Function**: Eliminating asymmetric information flow in forward and backward propagation.
    - **Mechanism**: Only retaining points whose pseudo-label confidence exceeds the threshold $\tau_s=0.9$ for forward propagation: $\hat{x}_i^u = \{x_i^u | \Phi_s(x_i^u) \geq \tau_s\}$. This ensures both forward and backward passes process only points with pseudo-labels, maintaining consistency.
    - **Design Motivation**: This strategy is plug-and-play and can be applied to any semi-supervised LiDAR framework, contributing approximately 1% mIoU under 1% annotations.

2. **MixPatch Cross-Scene Patch Augmentation**:

    - **Function**: Mixing BEV patches from multiple scenes to increase semantic diversity.
    - **Mechanism**: Dividing the BEV space into regular patch grids and uniformly sampling patches from both the labeled pool and the pseudo-labeled pool to replace corresponding positions in the current scene. Unlike two-scene concatenation, MixPatch can blend patches from $N$ different scenes.
    - **Design Motivation**: Concatenating only two scenes provides limited combinations, whereas multi-scene patch mixing offers exponentially more semantic combinations.

3. **InsFill Instance-Level Augmentation**:

    - **Function**: Sampling 3D object instances from an instance pool to fill empty spaces in scenes.
    - **Mechanism**: Maintaining an instance pool (storing point cloud instances extracted from all scenes categorized by class), randomly selecting instances to place into the scene during augmentation, and checking for occlusions and contextual plausibility.
    - **Design Motivation**: Patch-level augmentation modifies background semantics, while instance-level augmentation increases the diversity of foreground objects, serving as complementary components.

### Loss & Training
Standard cross-entropy + pseudo-label consistency loss. Teacher EMA $\alpha = 0.99$. The labeled pool is persistent, while the pseudo-labeled pool is updated epoch by epoch. Backbone: MinkowskiNet / Cylinder3D.

## Key Experimental Results

### Main Results

| Method | SemanticKITTI 1% | 10% | 50% | nuScenes 1% | 10% |
|------|-----------------|------|------|-------------|------|
| DDSemi | 59.3 | 65.1 | 67.0 | 58.1 | 70.2 |
| **AIScene** | **61.2** | **66.3** | **67.9** | **60.2** | **72.3** |
| Gain | +1.9 | +1.2 | +0.9 | +2.1 | +2.1 |

### Ablation Study

| Component | SemanticKITTI 1% mIoU |
|------|---------------------|
| Baseline | ~59 |
| + Point Erasure | +1.0 |
| + MixPatch | +1.5 |
| + InsFill | +1.9 |

### Key Findings
- **Greatest improvement achieved under 1% annotation** (+1.9/+2.1 mIoU), demonstrating that the proposed method yields the highest value when annotations are extremely scarce.
- **Point Erasure acts as a general plug-and-play component**: It can be directly integrated into any teacher-student framework to yield consistent performance gains.
- **Multi-scene hybrid > Two-scene concatenation**: The order-of-magnitude difference in semantic diversity leads to qualitative performance improvements.

## Highlights & Insights
- **The idea of Point Erasure is extremely simple yet effective**—a single-line code modification is sufficient to improve standard forward and backward propagation consistency.
- **The multi-pool strategy (labeled pool + pseudo-labeled pool + instance pool)** provides rich augmentation materials for semi-supervised scenarios.

## Limitations & Future Work
- The pseudo-label threshold $\tau = 0.9$ is fixed; an adaptive threshold might yield better performance.
- BEV patch mixing may introduce unnatural boundary effects.
- Validated only on outdoor driving scenes; the effectiveness on indoor 3D scenes remains unexplored.

## Related Work & Insights
- **vs LaserMix / CPS**: Traditional two-scene mixing augmentations. AIScene's multi-scene + multi-granularity (patch + instance) augmentation provides significantly more combinations.
- **vs DDSemi**: The current SOTA semi-supervised method. AIScene surpasses it across all settings, and the methods are orthogonal and complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of Point Erasure is simple and novel, and the multi-scene hybrid augmentation is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, three annotation ratios, and two backbones are evaluated.
- Writing Quality: ⭐⭐⭐⭐ The analysis framework for intra-/inter-scene and consistency mechanisms is clearly structured.
- Value: ⭐⭐⭐⭐ Directly applicable and holds practical value for semi-supervised 3D segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](../../ECCV2024/autonomous_driving/ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [\[CVPR 2025\] Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)
- [\[CVPR 2025\] 3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)
- [\[CVPR 2025\] A Dataset for Semantic Segmentation in the Presence of Unknowns](a_dataset_for_semantic_segmentation_in_the_presence_of_unknowns.md)
- [\[CVPR 2025\] VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow](voteflow_enforcing_local_rigidity_in_self-supervised_scene_flow.md)

</div>

<!-- RELATED:END -->
