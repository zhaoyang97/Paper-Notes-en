---
title: >-
  [Paper Note] MITracker: Multi-View Integration for Visual Object Tracking
description: >-
  [CVPR 2025][Autonomous Driving][Multi-View Object Tracking] This paper proposes a multi-view object tracking dataset MVTrack (234K frames, 27 target classes) and a method named MITracker. By projecting 2D features into 3D feature volumes and compressing them into a BEV (Bird's-Eye View) plane for cross-view fusion, combined with spatially-augmented attention to refine individual view tracking results, the method achieves rapid tracking recovery from occlusions.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multi-View Object Tracking"
  - "3D Feature Volumes"
  - "Bird's-Eye View"
  - "Spatially-Augmented Attention"
  - "Multi-view Dataset"
date: 2026-05-08
content_hash: de3a6c3630cc9059
---

# MITracker: Multi-View Integration for Visual Object Tracking

**Conference**: CVPR 2025  
**arXiv**: [2502.20111](https://arxiv.org/abs/2502.20111)  
**Code**: [https://mii-laboratory.github.io/MITracker](https://mii-laboratory.github.io/MITracker)  
**Area**: Autonomous Driving  
**Keywords**: Multi-View Object Tracking, 3D Feature Volumes, Bird's-Eye View, Spatially-Augmented Attention, Multi-view Dataset

## TL;DR

This paper proposes a multi-view object tracking dataset MVTrack (234K frames, 27 target classes) and a method named MITracker. By projecting 2D features into 3D feature volumes and compressing them into a BEV (Bird's-Eye View) plane for cross-view fusion, combined with spatially-augmented attention to refine individual view tracking results, the method achieves rapid tracking recovery from occlusions.

## Background & Motivation

Multi-view object tracking (MVOT) addresses challenging issues such as occlusion and target loss in single-view tracking by utilizing complementary views, but its development is constrained by the following factors:

1. **Dataset Scarcity**: Existing multi-view datasets are limited to specific categories (e.g., pedestrians, birds), and most of them are evaluation sets lacking training data. GMTD contains only 18K frames and does not provide a training set.
2. **Methodological Limitations**: Existing MVOT methods are primarily based on the detection-and-reidentification paradigm, which is designed for specific classes and cannot perform class-agnostic tracking.
3. **Difficult Cross-View Fusion**: Simple post-processing fusion (projecting to the ground and then reprojecting back to 2D) performs poorly due to significant distribution gaps.

**Goal**: Construct a large-scale multi-view tracking dataset and design an end-to-end tracking method that genuinely utilizes multi-view geometric information.

## Method

### Overall Architecture

MITracker consists of two primary modules: (1) View-Specific Feature Extraction Module, which uses a ViT encoder to process video streams from each view independently to generate single-view tracking results and 2D feature maps; (2) Multi-View Integration Module, which projects multi-view 2D features into 3D feature volumes, aggregates them under the guidance of BEV, and refines the tracking results of each view through spatially-augmented attention.

### Key Designs

1. **Streaming View-Specific Encoder**: Utilizing ViT (DINOv2-base) as the backbone, the input includes the search frame $S$, the reference frame $R$, and two temporal tokens. The temporal token design borrows from ODTrack: a learnable token $T_t$ for the current frame and $T_{t-1}$ propagated from the previous frame to ensure temporal continuity across frames. Among the output tokens, the attention weights between $T_t'$ and the search frame token $I_S'$ are calculated to focus on the target region: $I_U = I_S' \cdot (I_S' \times (T_t')^\top)$. Meanwhile, $I_U$ is mapped to a pixel-level 2D feature map $F_{2D} \in \mathbb{R}^{32 \times H_s \times W_s}$ to establish the pixel correspondence between the features and the search image.

2. **3D Feature Volume Construction and BEV Aggregation**: Based on the camera intrinsic matrix $C_K$, rotation $C_R$, and translation $C_t$, the 2D features $F_{2D}^k$ from each view are back-projected into a unified 3D feature volume $F_{3D} \in \mathbb{R}^{32 \times X \times Y \times Z}$ ($X=Y=200, Z=3$). The multi-view features are fused by averaging within the 3D voxels. Then, 1D convolution along the Z-axis is applied to aggregate and compress the volume into a BEV representation $F_{3D}' \in \mathbb{R}^{32 \times X \times Y}$. A classification head is trained to predict the BEV score map as a supervisory signal to constrain the cross-view feature fusion.

3. **Spatially-Augmented Attention**: BEV guidance provides only implicit constraints, which is insufficient to directly correct tracking failures. The aggregated $F_{3D}'$ is compressed via convolution into a 3D-aware token $T_{3D} \in \mathbb{R}^{1 \times D}$, which is then concatenated with the unrefined features $I_U^k$ of each view and fed into Transformer blocks for attention interaction. In this way, the tracking results of each view can be refined using the fused 3D spatial information, which especially helps recover tracking when the target is occluded by leveraging information from other visible views.

### Loss & Training

$$L_{track} = L_{cls} + \lambda_{giou}L_{giou} + \lambda_{L_1}L_1 + \lambda_{bev}L_{bev}$$

where $\lambda_{giou}=5, \lambda_{L_1}=2, \lambda_{bev}=0.1$. $L_{bev}$ uses focal loss to constrain the BEV score map.

Two-stage training:
- **Stage 1**: Only the view-specific feature extraction module is trained, using single-view data from GOT-10K and MVTrack. Each sample contains 1 reference frame and 2 search frames (spaced 200 frames apart) to facilitate temporal information propagation.
- **Stage 2**: The encoder is fine-tuned, and the complete framework is trained using the multi-view data of MVTrack, randomly selecting 2 to 4 views each time. Trained on 2×A100 80GB GPUs.

## Key Experimental Results

### Main Results

| Method | MVTrack Multi-View AUC/PNorm/P | MVTrack Single-View AUC | GMTD Single-View AUC |
|------|------|------|------|
| ODTrack | - (Single-view 63.36/82.25/74.46) | 63.36 | 61.43 |
| OSTrack | - (post-fusion 49.10/65.19/67.34) | 60.04 | 58.44 |
| **MITracker** | **71.13/91.87/83.95** | **68.57** | **65.96** |

Under the multi-view setting, MITracker improves the PNorm by approximately 26% compared to the post-fusion OSTrack. Under the single-view setting, it also outperforms ODTrack by about 5% AUC.

### Ablation Study

| Configuration | AUC (%) | PNorm (%) | P (%) | Note |
|------|---------|------|------|------|
| Baseline (w/o BEV/Spatial) | 63.99 | 82.82 | 75.00 | Single-view only |
| + BEV Loss | 69.64 | 89.85 | 82.01 | Implicit spatial awareness +5.65 AUC |
| + BEV Loss + Spatial Attention | 71.13 | 91.87 | 83.95 | Explicit 3D refinement additional +1.49 AUC |

### Key Findings

- **Significantly Improved Recovery**: The recovery rate within 10 frames after target disappearance is improved from 56.7% (SAM2Long) to 79.2% (+22.5%).
- **Longer Continuous Tracking**: The maximum number of continuously tracked frames is nearly 100 frames more than ODTrack, with fewer restarts.
- **Strong Generalization**: The method also achieves SOTA on the GMTD dataset, which was not involved in training, proving that the multi-view training strategy enhances the spatial understanding capability of the model.
- **Poor Post-Processing Fusion**: The performance of all single-view methods actually drops after post-processing multi-view fusion, indicating that simple geometric projection cannot bridge the feature distribution gaps across views.

## Highlights & Insights

- **The MVTrack dataset fills an important gap**: With 234K frames, 27 target classes, 3-4 views, including missing annotations and calibration info, it is the first large-scale multi-view tracking dataset providing both training and evaluation sets.
- **The 2D $\rightarrow$ 3D $\rightarrow$ BEV feature fusion path** is clear and natural, borrowing the concept of BEV perception from the autonomous driving field and applying it to general object tracking.
- **The design of the 3D-aware token** ingeniously compresses multi-view spatial information into a single token, achieving cross-view information transfer with minimal overhead.
- **Recovery capability is the core value of multi-view tracking**: when one view is occluded, information from other views can assist in recovery.

## Limitations & Future Work

- The dataset only contains indoor scenes; generalization to outdoor scenarios still needs validation.
- It relies on precise camera calibration parameters for 3D projection, making it difficult to apply in uncalibrated scenarios (e.g., handheld cameras).
- The fixed size of the 3D feature volume (200 $\times$ 200 $\times$ 3) may be insufficient for large-scale outdoor scenes.
- Currently, it only supports 3-4 views, and scalability when extending to more views needs further study.
- The dataset labeling adopts a semi-automatic approach, leaving room for optimization in annotation accuracy and efficiency.

## Related Work & Insights

- RTracker uses a tree-structured memory to detect and recover from target loss, but it is complex and relies on category-specific detectors; MITracker solves the recovery problem more naturally through multi-view fusion.
- Methods in autonomous driving, such as BEVFormer, project multi-view images into BEV for perception; this paper is the first to systematically apply this approach to general object tracking.
- GMT attempts to utilize multi-view information within a single-view training framework but fails to model actual multi-view relationships effectively.
- The 3D feature volume construction method can be borrowed by other multi-view vision tasks (e.g., multi-view action recognition).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of the dataset and the method is valuable. Although the BEV-guided multi-view fusion idea has precedents in other fields, it is novel in the tracking domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across dual datasets (MVTrack + GMTD), including ablation, recovery capability, and visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The dataset and method are described clearly with intuitive illustrations, though some variable notations require cross-reference to fully understand.
- **Value**: ⭐⭐⭐⭐ The long-term value of the dataset is higher than the method itself, providing the first complete training and evaluation infrastructure for multi-view tracking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SceneCrafter: Controllable Multi-View Driving Scene Editing](scenecrafter_controllable_multi-view_driving_scene_editing.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](../../ICCV2025/autonomous_driving/evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ECCV 2024\] Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](../../ECCV2024/autonomous_driving/weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)
- [\[ECCV 2024\] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](../../ECCV2024/autonomous_driving/open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)
- [\[CVPR 2025\] ZeroVO: Visual Odometry with Minimal Assumptions](zerovo_visual_odometry_with_minimal_assumptions.md)

</div>

<!-- RELATED:END -->
