---
title: >-
  [Paper Note] Sketchy Bounding-Box Supervision for 3D Instance Segmentation
description: >-
  [CVPR 2025][3D Vision][Weakly-supervised 3D instance segmentation] The Sketchy-3DIS framework is proposed, introducing inaccurate ("sketchy") 3D bounding box annotations to weakly-supervised 3D instance segmentation for the first time. Through joint training of an adaptive box-to-point pseudo-label generator and a coarse-to-fine instance segmenter, it achieves SOTA performance on ScanNetV2 and S3DIS, even outperforming some fully-supervised methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Weakly-supervised 3D instance segmentation"
  - "inaccurate bounding boxes"
  - "pseudo-labeling"
  - "coarse-to-fine segmentation"
  - "point cloud understanding"
date: 2026-05-08
content_hash: 8350d60b15835eb8
---

# Sketchy Bounding-Box Supervision for 3D Instance Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2505.16399](https://arxiv.org/abs/2505.16399)  
**Code**: [https://github.com/dengq7/Sketchy-3DIS](https://github.com/dengq7/Sketchy-3DIS)  
**Area**: 3D Vision  
**Keywords**: Weakly-supervised 3D instance segmentation, inaccurate bounding boxes, pseudo-labeling, coarse-to-fine segmentation, point cloud understanding

## TL;DR

The Sketchy-3DIS framework is proposed, introducing inaccurate ("sketchy") 3D bounding box annotations to weakly-supervised 3D instance segmentation for the first time. Through joint training of an adaptive box-to-point pseudo-label generator and a coarse-to-fine instance segmenter, it achieves SOTA performance on ScanNetV2 and S3DIS, even outperforming some fully-supervised methods.

## Background & Motivation

**Background**: 3D instance segmentation is a core task in point cloud scene understanding. Current mainstream methods (e.g., SPFormer, Mask3D) rely on point-wise annotations. However, annotating a scene in ScanNet takes about 22.3 minutes, whereas annotating bounding boxes requires only 1.93 minutes. Consequently, a vast body of recent work has adopted bounding boxes as weak supervision signals.

**Limitations of Prior Work**: Existing box-supervised methods (e.g., Box2Mask, GaPro, BSNet) all assume that the annotated bounding boxes are accurate and tight. However, in practical annotation, obtaining completely accurate 3D bounding boxes is highly challenging—annotators often introduce scaling, translation, and rotation biases. Experiments demonstrate that GaPro suffers from severe performance degradation when using scaled sketchy boxes.

**Key Challenge**: Bounding boxes in real-world scenarios cannot be perfectly accurate, yet existing methods are highly sensitive to box precision. Inaccurate bounding boxes cause a large number of points to be incorrectly assigned to neighboring instances, producing noisy pseudo-labels that severely impair segmentation quality.

**Goal**: To design a weakly-supervised 3D instance segmentation framework robust to bounding box noise, capable of (1) generating high-quality pseudo-labels from inaccurate bounding boxes, and (2) training a high-precision segmenter based on these pseudo-labels.

**Key Insight**: The authors observe that the core issue of inaccurate bounding boxes lies in point assignment within overlapping regions. If the similarity between points and bounding boxes can be learned, points in overlapping regions can be adaptively assigned to the correct instances.

**Core Idea**: Jointly train an adaptive pseudo-label generator (converting sketchy boxes into tight boxes and generating point-level pseudo-labels) and a coarse-to-fine instance segmenter (gradually refining instances via multi-level attention). The two components mutually benefit each other, progressively improving segmentation quality.

## Method

### Overall Architecture

Input is a 3D point cloud scene and its corresponding set of sketchy bounding box annotations. First, point features are extracted through a 3D U-Net backbone, which then feed into two parallel branches: (1) an adaptive box-to-point pseudo-label generator that converts coarse bounding box annotations into fine, point-level instance labels; (2) a coarse-to-fine instance segmenter that predicts instances using a Transformer query mechanism. Finally, bipartite matching is employed to pair pseudo-labels with predicted instances for joint training of both components. During inference, only the backbone and the segmenter are required.

### Key Designs

1. **Sketchy Bounding Box Generation**:

    - Function: Simulate inaccurate bounding boxes in practical annotations.
    - Mechanism: Apply three types of perturbation to the ground-truth bounding boxes—scaling ($\alpha=5\%$), translation ($\beta=5\%$), and rotation ($\gamma=5°$). By combining these three basic operations, four types of bounding boxes with varying degrees of "sketchiness" ($S_1$ to $S_4$) are generated. The perturbation magnitudes are set based on reasonable ranges of practical annotation biases.
    - Design Motivation: To provide a controllable experimental framework for investigating the impact of annotation noise on weakly-supervised methods.

2. **Adaptive Box-to-Point Pseudo-Label Generator**:

    - Function: Convert inaccurate bounding box annotations into high-quality point-level instance pseudo-labels.
    - Mechanism: Process different types of points in three steps. (a) Points outside the bounding boxes are directly labeled as background. (b) For points inside only a single bounding box, background points are filtered out through the product of cosine similarity in feature space and distance in coordinate space: $s_{p,B} = \cos(f_p, f_B) \times e^{-|c_B c_p|}$. (c) For points within the overlapping regions of multiple bounding boxes, the overlapping part is first removed to obtain reliable points (points belonging to only one box). The features of these reliable points are used to represent the box features, and an MLP is then utilized to learn point-to-box assignment probabilities, supervised by cross-entropy loss $L_{pl}$.
    - Design Motivation: The primary issue with inaccurate bounding boxes is the expansion of overlapping regions, leading to more points being misassigned. By learning similarities instead of hard rules, different levels of inaccurate annotations can be adaptively handled.

3. **Coarse-to-Fine Instance Segmenter (Multi-level Attention Block)**:

    - Function: Progressively refine instance predictions from global to local levels.
    - Mechanism: Instance queries first perform global cross-attention with the full-scene point features to obtain coarse instances, and then perform local attention with reliable region features ($F^{rel}$, weighted by the intersection of predicted box and mask box) and core region features ($F^{un}$, weighted by the intersection of the scaled core box and mask box). A six-layer stacked Multi-level Attention Block progressively refines the predictions. The reliable region is weighted by the IoU of the predicted box and mask box: $F^{rel} = \sigma(F, M \odot e^{IoU(B_{pred}, B_{mask})})$.
    - Design Motivation: The coarse labels in weak supervision lead to inaccurate initial predictions. A hierarchical global-to-core attention mechanism allows queries to progressively focus on the target core regions, thereby improving boundary accuracy.

### Loss & Training

The total loss is $L = L_{pl} + L_{seg}$, where the pseudo-label loss $L_{pl}$ is the cross-entropy loss for reliable point assignment, and the instance segmentation loss $L_{seg}$ comprises classification loss (cross-entropy), mask loss (BCE + Dice), and box loss (L1 + core-box MSE). Hungarian matching is used to establish correspondences between pseudo-labels and predicted instances. Training incorporates the AdamW optimizer with a learning rate of 0.0002 and weight decay of 0.05, conducted on a single RTX 3090. The joint training strategy allows pseudo-label quality and segmenter performance to mutually reinforce each other.

## Key Experimental Results

### Main Results

| Dataset | Method | Supervision | AP50 | AP25 |
|--------|------|------|------|------|
| ScanNetV2 Val | GaPro+SPFormer | Accurate box S0 | 70.4 | 79.9 |
| ScanNetV2 Val | BSNet+SPFormer | Accurate box S0 | 72.7 | 83.4 |
| ScanNetV2 Val | **Sketchy-3DIS** | **Accurate box S0** | **68.8** | **83.6** |
| ScanNetV2 Val | GaPro+SPFormer | Sketchy S1 | 53.5 | 72.2 |
| ScanNetV2 Val | **Sketchy-3DIS** | **Sketchy S1** | **65.8** | **83.1** |
| S3DIS Area 5 | GaPro+ISBNet | Accurate box S0 | 61.2 | - |
| S3DIS Area 5 | BSNet+ISBNet | Accurate box S0 | 64.3 | - |
| S3DIS Area 5 | **Sketchy-3DIS** | **Accurate box S0** | **69.1** | - |
| S3DIS Area 5 | ISBNet (Fully-supervised) | Mask | 65.8 | - |

### Ablation Study

| Configuration | AP | AP50 | AP25 | Description |
|------|-----|------|------|------|
| No pseudo-label generator (Partition only) | 15.9 | 32.2 | 58.5 | Spatial assignment only, heavy label noise |
| +Assign (Overlap region assignment) | 41.8 | 64.8 | 72.3 | Overlap resolution is critical |
| +Similarity (Background filtering) | 45.2 | 67.3 | 83.4 | Filtering non-target points further improves |
| Full model | 46.0 | 68.8 | 83.6 | Full model |
| Disjoint training (Non-joint) | 45.3 | 60.4 | 70.0 | Joint training significantly outperforms disjoint training |
| Joint training | 53.4 | 69.1 | 77.5 | Pseudo-labels + segmenter mutually reinforce |

### Key Findings

- **Point assignment in overlapping regions** is the most critical module for performance improvement, boosting AP50 from 32.2 to 64.8 (+32.6% AP50), as overlapping regions are greatly expanded due to sketchy boxes.
- On S3DIS, Sketchy-3DIS exceeds the fully-supervised ISBNet by +3.3 AP50 (69.1 vs 65.8) even when utilizing inaccurate annotations (S0), indicating that carefully designed weakly-supervised methods can surpass fully-supervised counterparts.
- Across different levels of sketchy boxes from S1 to S4, the AP50 drops only from 65.8 to 62.5 on ScanNetV2, exhibiting solid robustness.
- The gap between joint and disjoint training is substantial: AP50 of 69.1 vs 60.4, validating that pseudo-label generation and the segmenter must be optimized cooperatively.
- In the Multi-level Attention Block, all four types of attention (Scene/Coarse/Core/Self) contribute, with the combination of Coarse and Core being the most crucial.

## Highlights & Insights

- **First study on the sketchy box setting**: While prior box-supervised methods assume accurate boxes, this paper is the first to systematically investigate the impact of inaccurate annotations, which closely aligns with real-world scenarios. This problem setting itself holds significant value.
- **Mutual promotion mechanism of joint training**: The pseudo-label generator and the segmenter share backbone features, achieving a virtuous cycle through joint optimization—better features produce better pseudo-labels, and better pseudo-labels train a better segmenter.
- **Separation strategy of reliable vs. unreliable points**: For overlapping regions, the model learns assignment rules using only reliable points (non-overlapping parts) and then generalizes them to unreliable points, avoiding the influence of noisy labels. This methodology can be transferred to other weakly-supervised tasks.

## Limitations & Future Work

- The model may perform suboptimally in scenes with large variations in object sizes, as the perturbation ratio of the sketchy box is globally uniform.
- The experiments only studied simulated sketchy boxes, without validation on actual inaccurate annotations generated by human annotators.
- Inference speed and computational overhead are not reported; joint training of two branches may lead to higher training costs.
- Incorporating 2D foundation models like SAM could be considered to help resolve ambiguities in overlapping regions.
- Extending the framework to large-scale point clouds in outdoor environments (e.g., autonomous driving) is a highly promising direction.

## Related Work & Insights

- **vs GaPro**: GaPro uses Gaussian processes to generate reliable pseudo-labels but relies on precise boxes. Under S1 sketchy boxes, its AP50 plummets from 70.4 to 53.5. Sketchy-3DIS still maintains 65.8 under S1.
- **vs BSNet**: BSNet enhances data through a mean teacher and scene synthesis, performing stronger under precise boxes (72.7 vs 68.8 AP50), but its robustness under inaccurate annotations has not been validated.
- **vs Box2Mask**: As the first box-supervised method, Box2Mask is conceptually simpler and only achieves 52.4 AP50 under sketchy boxes.

## Rating

- Novelty: ⭐⭐⭐⭐ First to study the sketchy box setting; the problem definition is valuable, though specific methods combine existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two datasets, multiple sketchiness levels, comprehensive ablation studies, and qualitative visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and architecture diagrams, though some detailed descriptions could be more concise.
- Value: ⭐⭐⭐⭐ Highly practical, since inaccurate annotations reflect real-world needs; the method is effective, and the code is open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)
- [\[CVPR 2025\] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation](relation3d_enhancing_relation_modeling_for_point_cloud_instance_segmentation.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](../../AAAI2026/3d_vision/retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](../../ICCV2025/3d_vision/cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[CVPR 2026\] EvObj: Learning Evolving Object-centric Representations for 3D Instance Segmentation without Scene Supervision](../../CVPR2026/3d_vision/evobj_learning_evolving_object-centric_representations_for_3d_instance_segmentat.md)

</div>

<!-- RELATED:END -->
