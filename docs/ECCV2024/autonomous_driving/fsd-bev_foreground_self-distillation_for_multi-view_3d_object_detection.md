---
title: >-
  [Paper Note] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection
description: >-
  [ECCV 2024][Autonomous Driving][BEV 3D detection] This paper proposes a Foreground Self-Distillation (FSD) framework which constructs teacher-student branches sharing image features within the same model, effectively avoiding the distribution discrepancy challenge in cross-modal distillation. Combined with point cloud intensification and multi-scale foreground enhancement modules, it achieves SOTA performance on nuScenes.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "BEV 3D detection"
  - "self-distillation"
  - "foreground enhancement"
  - "point cloud enhancement"
  - "multi-view perception"
date: 2026-05-08
content_hash: b183b5306349f766
---

# FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2407.10135](https://arxiv.org/abs/2407.10135)  
**Code**: [GitHub](https://github.com/CocoBoom/fsd-bev)  
**Area**: Autonomous Driving  
**Keywords**: BEV 3D detection, self-distillation, foreground enhancement, point cloud enhancement, multi-view perception

## TL;DR

This paper proposes a Foreground Self-Distillation (FSD) framework which constructs teacher-student branches sharing image features within the same model, effectively avoiding the distribution discrepancy challenge in cross-modal distillation. Combined with point cloud intensification and multi-scale foreground enhancement modules, it achieves SOTA performance on nuScenes.

## Background & Motivation

Multi-view camera-based BEV 3D object detection is a cost-effective and deployable solution in autonomous driving, but still exhibits a performance gap compared to LiDAR-based methods. Existing cross-modal distillation approaches (e.g., BEVDistill, UniDistill, DistillBEV) attempt to transfer knowledge from a LiDAR teacher model to a camera student model, but face the following core issues:

**Distribution Discrepancy**: The teacher and student BEV features originate from different modalities (LiDAR vs. camera) and different network architectures, leading to a large gap in feature distribution.

**Extra Pre-training Overhead**: A separate LiDAR teacher model needs to be trained.

**Complex Distillation Strategies**: To bridge the distribution discrepancy, tedious adaptation modules and distillation losses must be designed.

**Background Interference**: Aligning extensive background areas in BEV features is unhelpful or even harmful to detection accuracy.

The core insight of this paper is: if the teacher and student BEV features share image features from the same source, the distribution discrepancy will be significantly reduced, naturally improving the distillation efficacy.

## Method

### Overall Architecture

FSD-BEV adopts a unified framework consisting of three core components:

1. **Foreground Self-Distillation (FSD)**: Adds an auxiliary teacher branch within the same model, where the teacher and student share image features but use different depth/foreground labels to generate BEV features.
2. **Point Cloud Intensification (PCI)**: Compensates for point cloud sparsity via frame combination and pseudo point assignment.
3. **Multi-Scale Foreground Enhancement (MSFE)**: Extracts multi-scale foreground features using elliptical Gaussian heatmaps.

After extracting features from the backbone and FPN, MSFE enhances foreground information. The View Transformation Module generates student and teacher BEV features separately. These are concatenated along the batch dimension and passed through a shared BEV Encoder and detection heads for joint training and distillation.

### Key Designs

#### Foreground Self-Distillation (FSD)

**Student BEV Generation**: DepthNet is used to predict depth maps $D$ and foreground segmentation $S$. Combined with context features $C$, the student BEV features containing only foreground information are generated via SA-BEVPool: $B_s = \text{SA-BEVPool}(C, D, S)$.

**Teacher BEV Generation**: Utilizing the hard labels (ground truth depth map and foreground segmentation) generated from LiDAR point clouds, combined with the soft labels predicted by the student to fill in the missing parts of the hard labels. The formulation of combined labels is:

- Depth: $$\bar{D} = M \odot \hat{D} + (1-M) \odot D$$
- Foreground: $$\bar{S} = M \odot \hat{S} + (1-M) \odot S$$

where $M$ is the validity mask of the hard label (1 if hard label is available, 0 otherwise). The teacher BEV is likewise generated via SA-BEVPool. The key advantage of this design is that the teacher and student share image features $C$, drastically reducing the distribution gap.

**Cooperative Training**: The teacher and student BEV features are concatenated along the batch dimension and jointly pass through the BEV Encoder. The BEV Encoder naturally serves as an adaptation module without extra parameters. Both branches are supervised by detection loss simultaneously; the teacher branch has a smaller loss and does not heavily disrupt student training.

#### Point Cloud Intensification (PCI)

**Frame Combination**: Point clouds of static foreground objects (such as parked cars, riderless bicycles, traffic cones) from adjacent frames are transformed into the current frame coordinate system and combined to increase point cloud density. Only static objects are selected to avoid errors introduced by dynamic objects.

**Pseudo Point Assignment**: For objects still lacking point cloud coverage after frame combination, pseudo points are assigned at the centers of their 2D projection boxes. The depth of the pseudo point is taken as the minimum depth of the eight corners of the 3D bounding box (close to the object surface depth). Assignment conditions: no real points inside the box after frame combination, depth within the perception range, and visibility level is good (set to 3 or 4 in nuScenes).

#### Multi-Scale Foreground Enhancement (MSFE)

Utilizing multi-scale features $F_4$, $F_8$, and $F_{16}$ output by the FPN, foreground segmentation $S_4$ is predicted on the high-resolution feature $F_4$. Since LiDAR labels are extremely sparse under high resolution (around 80% missing), an **elliptical Gaussian heatmap** is adopted as the label (differing from the circular Gaussian in CenterNet, the ellipse better fills 2D boxes), trained with Focal Loss.

The enhanced feature is aggregated via foreground segmentation weighting and downsampling:
$$F_{16\_MSFE} = F_{16} + \text{DS2}(F_8 \odot \text{DS2}(S_{4\_f})) + \text{DS4}(F_4 \odot S_{4\_f})$$

### Loss & Training

**Distillation Loss**: L2 loss is applied to the normalized high-level BEV features of the teacher and the student. Normalization prevents the branches from taking a shortcut by simply shrinking the magnitude rather than truly aligning the features.

**Total Loss**: Detection loss (both teacher and student branches participate) + distillation loss + depth supervision loss + Focal Loss of MSFE.

**Training Configuration**: 8x RTX 3090, AdamW (lr=$2\times 10^{-4}$), mixed precision training, 24 epochs, CBGS strategy, 1 past frame (0.5s interval).

## Key Experimental Results

### Main Results

**nuScenes val set (ResNet50, 256x704, 2 frames)**:

| Method | mAP | NDS | mATE | mASE | mAOE |
|------|-----|-----|------|------|------|
| BEVDepth | 0.351 | 0.475 | 0.639 | 0.267 | 0.479 |
| BEVStereo | 0.372 | 0.500 | 0.598 | 0.270 | 0.438 |
| SA-BEV | 0.387 | 0.512 | 0.613 | 0.266 | 0.352 |
| **FSD-BEV** | **0.403** | **0.526** | **0.576** | **0.259** | **0.362** |
| FSD-BEV (256 BEV) | **0.412** | **0.538** | **0.527** | **0.256** | **0.363** |

**nuScenes val set (ResNet101, 512x1408, 2 frames)**:

| Method | Frames | mAP | NDS |
|------|------|-----|-----|
| BEVFormer* | 4 | 0.416 | 0.517 |
| TiG-BEV | 2 | 0.440 | 0.544 |
| StreamPETR* | 8 | 0.504 | 0.592 |
| **FSD-BEV** | 2 | 0.488 | 0.589 |
| **FSD-BEV*** | 2 | 0.500 | **0.596** |

**nuScenes test set**: FSD-BEV (V2-99) achieves 54.3% mAP / 63.3% NDS, outperforming SOLOFusion (17 frames) and SA-BEV.

### Ablation Study

**Module Combinations**:

| FSD | PCI | MSFE | mAP | NDS |
|-----|-----|------|-----|-----|
| - | - | - | 0.363 | 0.486 |
| Y | - | - | 0.394 | 0.516 |
| Y | Y | - | 0.400 | 0.516 |
| Y | Y | Y | **0.403** | **0.526** |

**Impact of Foreground Segmentation on Teacher Branch**:

| Foreground Seg. | Branch | mAP |
|---------|------|-----|
| w/o | Teacher | 0.468 |
| w/ | Teacher | 0.584 |
| w/o | Student | 0.372 |
| w/ | Student | 0.393 |

### Key Findings

1. Self-distillation significantly outperforms the baseline without distillation (+4% mAP), without requiring an additionally pre-trained teacher model.
2. Foreground segmentation is extremely critical for improving the performance of the teacher branch (+11.6% mAP), validating the necessity of foreground-only distillation.
3. Point cloud intensification mainly boosts the detection capability for distant or sparse objects.
4. Compelling performance is achieved with only 2 frames, competing with or even surpassing approaches using 4-8 frames.

## Highlights & Insights

- **Self-distillation vs. Cross-modal Distillation**: By sharing the image feature source between the teacher and the student, the most troublesome distribution discrepancy issue in cross-modal distillation is elegantly evaded.
- **Complementarity of Soft and Hard Labels**: The soft labels from the student fill in the sparse vacancies of the teacher's hard labels, establishing a beneficial synergistic growth mechanism.
- **Distillation with No Extra Parameters**: The BEV Encoder naturally acts as an adaptation module, and L2 loss is sufficient to achieve effective distillation.
- **Novel Pseudo Point Assignment**: Assigning pseudo points to objects without point cloud coverage presents a practical solution for point cloud data augmentation.

## Limitations & Future Work

1. Still relies on LiDAR point clouds to provide depth GT and foreground segmentation GT; although not required during inference, they are necessary during training.
2. Pseudo point assignment employs heuristic rules (box center + minimum corner depth); more precise position estimation could yield further improvements.
3. Only utilizes 1 history frame, failing to fully exploit temporal information.
4. The elliptical Gaussian heatmap label of MSFE is still a coarse approximation; actual foreground shapes can be more complex.

## Related Work & Insights

- **SA-BEV**: The direct baseline of FSD-BEV, which provides the SA-BEVPool method for foreground BEV generation.
- **BEVDistill / UniDistill / DistillBEV**: Representatives of cross-modal distillation methods, facing distribution discrepancy challenges.
- **StreamPETR**: A strong attention-based baseline that uses more historical frames.
- The self-distillation concept can be generalized to other BEV tasks requiring knowledge distillation.

## Rating

- **Novelty**: 4/5 - The self-distillation framework is cleverly designed, with the soft-hard label complementarity being a highlight.
- **Experimental Thoroughness**: 4/5 - Sufficient ablation experiments and validation on multi-scale backbones.
- **Writing Quality**: 4/5 - Clear motivation and detailed methodology description.
- **Value**: 4/5 - No additional computational cost during inference, and no extra pre-training required during training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[ECCV 2024\] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)
- [\[ECCV 2024\] Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)
- [\[ECCV 2024\] Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)

</div>

<!-- RELATED:END -->
