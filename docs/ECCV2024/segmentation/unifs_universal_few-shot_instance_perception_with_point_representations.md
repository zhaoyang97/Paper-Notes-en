---
title: >-
  [Paper Note] UniFS: Universal Few-Shot Instance Perception with Point Representations
description: >-
  [ECCV 2024][Segmentation][Few-Shot Learning] This paper proposes UniFS, the first universal few-shot instance perception model. By unifying object detection, instance segmentation, pose estimation, and object counting into a dynamic point representation learning paradigm and introducing a structure-aware point learning (SAPL) loss to capture high-order structural relations among points, UniFS achieves performance close to expert models under minimal task hypotheses.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Few-Shot Learning"
  - "Unified Model"
  - "Point Representation"
  - "Instance Perception"
  - "Multi-Task Learning"
date: 2026-05-08
content_hash: ce45546d92d7fdb7
---

# UniFS: Universal Few-Shot Instance Perception with Point Representations

**Conference**: ECCV 2024  
**arXiv**: [2404.19401](https://arxiv.org/abs/2404.19401)  
**Code**: [https://github.com/jin-s13/UniFS](https://github.com/jin-s13/UniFS)  
**Area**: Segmentation  
**Keywords**: Few-Shot Learning, Unified Model, Point Representation, Instance Perception, Multi-Task Learning

## TL;DR

This paper proposes UniFS, the first universal few-shot instance perception model. By unifying object detection, instance segmentation, pose estimation, and object counting into a dynamic point representation learning paradigm and introducing a structure-aware point learning (SAPL) loss to capture high-order structural relations among points, UniFS achieves performance close to expert models under minimal task hypotheses.

## Background & Motivation

Instance perception tasks (object detection, instance segmentation, pose estimation, object counting) are crucial in industrial applications. Supervised learning methods are limited by high annotation costs, prompting the emergence of few-shot learning methods.

However, the core dilemma of existing few-shot methods is **task fragmentation**:

**Data discrepancy**: Detection datasets mostly consist of scene images (multi-object), whereas pose estimation datasets mostly consist of cropped single-object images.

**Feature granularity discrepancy**: Detection requires global semantic features, segmentation requires fine-grained semantic features, and pose estimation requires both semantic and localization features simultaneously.

**Output structure discrepancy**: Detection outputs bounding box coordinates, segmentation outputs pixel-level masks, and pose estimation outputs Gaussian heatmaps.

Each task relies on separate methods (DeFRCN, DCFS, SAFECount, etc.), independent datasets, and distinct evaluation metrics, lacking a unified framework. The authors pursue an ultimate goal: **solving multiple instance perception tasks with a single model using a minimal number of exemplars.**

## Method

### Overall Architecture

UniFS consists of three fully shared components (no task-specific designs):
1. **Feature Extractor**: ResNet-101 backbone, extracting support and query image features.
2. **Point Decoder**: Transformer decoder, enhancing point features via self-attention and cross-attention.
3. **Point Head**: MLP predicting point coordinate offsets.

Core idea: Unifying the output space of all tasks into a point set representation. By providing different types of point annotations on the support image, the model automatically learns to predict corresponding points on the query image.

### Key Designs

1. **Unified Point Representation**:

    - **Object Detection**: Representing a bounding box by uniformly sampling 16 points along the box edges.
    - **Instance Segmentation**: Uniformly sampling 32 points on the instance mask contour (following Deep Snake, ensuring clockwise order starting from the leftmost point).
    - **Pose Estimation**: Each semantic keypoint naturally corresponds to a point, where the number and definition of keypoints can vary across different categories.
    - **Object Counting**: Predicting the center point (bounding box center) of each object; counting is achieved by counting these points.
    - This unified representation offers four major advantages: a task-agnostic architecture, fully shared parameters, generalizability to new tasks, and cross-task knowledge sharing.

2. **Point Decoder (Transformer Architecture)**:

    - $L=2$-layer Transformer decoder.
    - **Self-Attention**: Exchanges information among support point features to model the context of the point sequence (especially task-related information).
    - **Cross-Attention**: Support point features (query) $\times$ query image RoI features (key/value), bridging the representation gap between support and query.
    - Outputs enhanced point features $\{\widehat{S_i}\}_{i \in [1,K]}$.
    - Finally, an MLP predicts the offsets relative to the RPN anchor center: $P_{xi} = A_{cx} + \Delta x_i \times A_w$.

3. **Structure-Aware Point Learning (SAPL)**:

    - Core problem: $L_1$/$L_2$ losses only focus on single-point errors. A predicted point falling on a diamond or circle centered at the ground truth receives the same loss, creating ambiguity.
    - Solution: Supervise the angular relationships between a point and its neighboring points additionally.
    - $\theta_i^{(n)}$ denotes the angle formed by three points: $i-n$, $i$, and $i+n$.
    - SAPL loss: $L_{SAPL} = \frac{1}{N} \sum_{n=1}^{N} L_1(\sin(\frac{\hat{\theta}_i^{(n)}}{2}), \sin(\frac{\theta_i^{(n)}}{2}))$
    - Using the $\sin(\theta/2)$ transformation: Amplifies gradients at sharp corners and dampens gradients in flat areas, capturing detailed shape information.
    - The optimal N-hop is set to $N=2$: 1-hop is sensitive to noise, while 3/4-hop is overly smooth.

### Loss & Training

- **Total Loss**: $L_{point} = |P_i - \hat{P}_i| + L_{SAPL}$, objective combined with RPN and classification losses.
- **Two-stage Transfer Learning**:
    - Base class training: Joint training of detection, segmentation, and pose estimation on 60 base classes (counting task is held out without training).
    - Novel class finetuning: Fine-tuning on $K$ samples with the learning rate reduced to 0.01.
- **Training Configuration**: batch=32 (4/GPU $\times$ 8 GPUs), lr=0.028, SGD, up to 55K iterations.
- **Results are averaged over 10 seeds** to ensure reliability.

## Key Experimental Results

### Main Results (COCO-UniFS Benchmark)

| Model | Type | Det. AP(1-shot) | Det. AP(5-shot) | Seg. AP(1-shot) | Seg. AP(5-shot) | Kpt. AP(1-shot) | Kpt. AP(5-shot) | Cnt. MSE(1-shot)↓ | Cnt. MSE(5-shot)↓ |
|------|------|----------------|----------------|----------------|----------------|----------------|----------------|------------------|------------------|
| FRCN-ft | Expert | 1.0 | 4.0 | — | — | — | — | — | — |
| TFA | Expert | 4.4 | 7.7 | — | — | — | — | — | — |
| FADI | Expert | 5.7 | 10.1 | — | — | — | — | — | — |
| DCFS | Expert | 8.1 | 16.4 | 7.2 | 13.5 | — | — | — | — |
| MPSR | Expert | 5.1 | 8.7 | — | — | — | — | 1.42† | 1.40† |
| Mask-RCNN-ft | Universal | 2.4 | 6.9 | 2.0 | 5.5 | 2.3 | 6.7 | 1.48† | 1.45† |
| **UniFS** | **Universal** | **12.7** | **18.2** | **8.6** | **11.5** | **12.2** | **22.1** | **1.38‡** | **1.32‡** |

†: Task seen during training | ‡: Zero-shot generalization to unseen tasks

### Ablation Study

**SAPL Effect (COCO-UniFS val)**:

| Loss Function | Det. AP(1/5-shot) | Seg. AP(1/5-shot) | Kpt. AP(1/5-shot) |
|---------|------------------|------------------|------------------|
| L2 only | 10.9 / 16.1 | 6.1 / 7.7 | 9.1 / 19.5 |
| L1 only | 10.6 / 16.2 | 7.2 / 8.7 | 12.0 / 21.0 |
| L1 + 1-hop SAPL | 12.6 / 17.9 | 8.4 / 11.3 | 12.3 / 21.8 |
| **L1 + 2-hop SAPL** | **12.7 / 18.2** | **8.6 / 11.5** | **12.2 / 22.1** |
| L1 + 3-hop SAPL | 12.6 / 17.7 | 8.2 / 11.0 | 12.4 / 21.6 |
| L1 + 4-hop SAPL | 12.7 / 17.8 | 8.4 / 11.3 | 12.2 / 20.9 |

**Multi-Task Learning Effect**:

| Training Tasks | Det. AP(1/5) | Seg. AP(1/5) | Kpt. AP(1/5) |
|---------|-------------|-------------|-------------|
| Det. only | 12.2 / 17.9 | — | — |
| Det.+Seg. | 12.6 / 17.6 | 8.5 / 11.2 | — |
| Det.+Seg.+Kpt. | 12.7 / 18.2 | 8.6 / 11.5 | 12.2 / 22.1 |

### Key Findings

1. **Universal model comprehensively outperforms multi-task baselines**: UniFS significantly outperforms Mask-RCNN-ft on all tasks (e.g., 1-shot detection 12.7 vs. 2.4).
2. **Competitive with expert models**: UniFS outperforms all expert methods in detection and pose estimation, while being only slightly inferior to DCFS in 5-shot segmentation (11.5 vs. 13.5).
3. **Strong generalization ability to unseen tasks**: Even though never trained on the counting task, UniFS achieves a 1.38 MSE, which outperforms all baselines that observed the counting task during training (MPSR 1.42, FSDetView 1.42).
4. **More pronounced advantage in 1-shot scenarios**: The advantage of UniFS is larger in extremely low-shot settings, indicating that the unified representation yields better priors.
5. **Significant contribution of SAPL**: Segmentation AP improves from 7.2 to 8.6 (+19.4%), confirming that structural constraints are particularly important for shape-sensitive tasks.
6. **L1 consistently outperforms L2**: L1 outperforms L2 robustly in both segmentation and pose estimation, likely because L1 is less sensitive to outliers.
7. **Synergistic multi-task benefits**: Adding more tasks does not degrade performance on existing tasks; instead, it brings slight gains.

## Highlights & Insights

1. **Point representation as a natural unified language**: bbox = points uniformly sampled along edges, mask = points uniformly sampled on the contour, keypoint = semantic points, counting = center point; this concise mapping is niezwykle elegant (extremely elegant).
2. **Geometric intuition of SAPL**: Traditional L1/L2 losses treat points as independent coordinate prediction tasks, ignoring the structural relationships among them. SAPL introduces the concept of "shape" via angular constraints.
3. **Zero task assumptions**: The model behavior is agnostic to the current task — it only knows "where the given support points are, and where the corresponding query points should be."
4. **Contribution of the COCO-UniFS benchmark**: It unifies datasets, splits, and evaluation protocols across four tasks, providing a fair evaluation platform for future research.

## Limitations & Future Work

1. **Potential quantization error in point sampling**: Representing mask contours with a finite number of points introduces quantization errors, especially for complex shapes (32 points might be insufficient).
2. **Unimproved classification capabilities**: Point representation focuses heavily on localization; the few-shot ability for image-level classification has not been specifically improved.
3. **Limited to 2D tasks**: It has not been extended to 3D perception or temporal inputs (e.g., video tracking).
4. **Reliance on RPN**: The architecture still relies on an RPN to generate region proposals, which is inherently not very few-shot friendly.
5. **Scalable future directions**: Increasing the point count, introducing dynamic/variable point numbers, and supporting 3D or video tasks.

## Related Work & Insights

- Works like Painter use dense maps to unify different tasks, but fail to handle instance-level tasks.
- Works like Pix2Seq use textual sequence representations but suffer from slow inference speeds.
- CenterNet, RepPoints, etc., have already utilized point representations for detection/segmentation, but are limited to single-task training and evaluation.
- UniFS generalizes point representations to a few-shot multi-task unified framework for the first time, proving that "less is more" in terms of design philosophy.
- The contour point representation of Deep Snake provides the technical foundation for converting masks into points.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to unify four instance perception tasks into a few-shot point learning paradigm, with an ingeniously designed SAPL loss.
- Experimental Thoroughness: ⭐⭐⭐⭐ — COCO-UniFS benchmark is comprehensive, but evaluated only on ResNet-101 without verifying with stronger backbones.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem definition, intuitive methodology, and elegant geometric explanation of SAPL.
- Value: ⭐⭐⭐⭐⭐ — Pioneering work; unifying problem definitions and benchmarks holds significant value for the development of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Eliminating Feature Ambiguity for Few-Shot Segmentation](eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [\[ECCV 2024\] GiT: Towards Generalist Vision Transformer through Universal Language Interface](git_towards_generalist_vision_transformer_through_universal_language_interface.md)
- [\[CVPR 2026\] The Missing Point in Vision Transformers for Universal Image Segmentation](../../CVPR2026/segmentation/the_missing_point_in_vision_transformers_for_universal_image_segmentation.md)
- [\[ECCV 2024\] Point-Supervised Panoptic Segmentation via Estimating Pseudo Labels from Learnable Distance](point-supervised_panoptic_segmentation_via_estimating_pseudo_labels_from_learnab.md)
- [\[ECCV 2024\] You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception](you_only_learn_one_query_learning_unified_human_query_for_single-stage_multi-per.md)

</div>

<!-- RELATED:END -->
