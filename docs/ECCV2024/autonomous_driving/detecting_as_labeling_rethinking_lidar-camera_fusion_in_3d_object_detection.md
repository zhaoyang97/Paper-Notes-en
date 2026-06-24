---
title: >-
  [Paper Note] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR-Camera Fusion] This paper summarizes the fundamental rule from the data labeling process that "image features should not be used for regression tasks" and proposes the DAL paradigm. DAL analogizes the detection process to the labeling process, using LiDAR features independently to complete regression predictions and fused features for classification predictions. Combined with a simplified training pipeline…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR-Camera Fusion"
  - "3D Object Detection"
  - "Data Labeling Paradigm"
  - "Overfitting Suppression"
  - "Speed-Accuracy Trade-off"
date: 2026-05-08
content_hash: ea7611d3335ce5d7
---

# Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2311.07152](https://arxiv.org/abs/2311.07152)  
**Code**: [https://github.com/HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet)  
**Area**: Autonomous Driving / 3D Object Detection  
**Keywords**: LiDAR-Camera Fusion, 3D Object Detection, Data Labeling Paradigm, Overfitting Suppression, Speed-Accuracy Trade-off

## TL;DR

This paper summarizes the fundamental rule from the data labeling process that "image features should not be used for regression tasks" and proposes the DAL paradigm. DAL analogizes the detection process to the labeling process, using LiDAR features independently to complete regression predictions and fused features for classification predictions. Combined with a simplified training pipeline, DAL substantially refreshes the SOTA on nuScenes with 74.0 NDS (val) and 74.8 NDS (test).

## Background & Motivation

**Background**: LiDAR-camera fusion for 3D object detection is a core task in autonomous driving perception. In recent years, a large number of fusion methods (TransFusion, BEVFusion, CMT, etc.) have emerged, competing fiercely on the nuScenes leaderboard. However, these methods commonly suffer from overfitting and rely on complex multi-stage pre-training and specialized learning rate schedules to mitigate this issue.

**Limitations of Prior Work**: (1) All existing methods involve image features in regression tasks (such as predicting the center, size, and orientation of 3D boxes), which violates the basic rules of data labeling. (2) Due to the inherently ill-posed nature of monocular depth estimation, image features are not robust enough when regressing geometric attributes, leading to model overfitting. (3) To combat overfitting, existing methods adopt complex training pipelines—multi-stage pre-training (on multiple datasets such as ImageNet, nuScenes, and nuImages) and customized learning rate strategies, introducing additional costs and uncertainty. (4) The involvement of the image branch in regression also limits the range of image space data augmentation, as consistency between image features and target predictions must be maintained.

**Key Challenge**: Images and LiDAR play different roles in 3D detection. LiDAR point clouds are precise "rulers" capable of accurately locating the boundaries of 3D boxes; images are "experienced gamblers", excellent at recognition and classification but unreliable for geometric regression. Existing methods fail to distinguish the distinct roles of these two modalities across target sub-tasks.

**Key Insight**: The authors draw inspiration from the data labeling workflow, where annotators follow two rules: (A) combine images and point clouds to search for target candidates and determine categories; (B) annotate 3D boxes based solely on point clouds. Existing algorithms violate rule B. DAL builds a detection pipeline by mimicking this labeling process.

**Core Idea**: Analogize the detection process to the data labeling process, utilizing only LiDAR features for regression tasks and fused features for classification tasks, fundamentally eliminating the source of overfitting.

## Method

### Overall Architecture

DAL adopts a dense-to-sparse paradigm. Dense perception stage: An image encoder and a LiDAR encoder extract features $F_I$ and $F_P$, respectively. The image features are projected to the BEV space via LSS, and merged to generate a dense heatmap, from which the top-$K$ candidates are selected. Sparse perception stage: For each candidate, its LiDAR features are fed through an FFN to predict regression targets (center, size, orientation, velocity), while a classification prediction is made by fusing image features, image BEV features, and LiDAR BEV features. The key is that the regression branch does not use image features at all.

### Key Designs

1. **Modality-specific Task Assignment**:

    - **Function**: Fundamentally eliminate the overfitting caused by involving image features in regression.
    - **Mechanism**: In the sparse perception phase, regression targets (center, size, orientation, velocity) are predicted solely by LiDAR features via a simple FFN. The classification task is completed by fusing image features, image BEV features, and LiDAR BEV features. In the dense perception phase, BEV features from both modalities are fused to generate a heatmap for candidate search. The key differences from BEVFusion are: (1) delayed fusion—fusion occurs after the BEV encoder instead of before; (2) removal of attention between sparse instances and BEV features; (3) regression uses point cloud features exclusively.
    - **Design Motivation**: Mimic rule B of data labeling—geometric attributes of 3D boxes should be determined solely based on point clouds. The ill-posed nature of monocular depth estimation in image features introduces systematic noise into regression.

2. **Simplified Training Pipeline**:

    - **Function**: Eliminate reliance on complex pre-training and customized learning rate strategies.
    - **Mechanism**: Only the ImageNet pre-trained image backbone weights are loaded, followed by end-to-end training for 20 epochs using CBGS data sampling and a cyclic learning rate schedule (initial value $2.0 \times 10^{-4}$). No pre-training of the LiDAR backbone is required on datasets like nuScenes or nuImages. The total loss is $L_{\text{DAL}} = L_{\text{aux}} + L_{\text{TransFusion}}$, where $L_{\text{aux}}$ is the auxiliary classification head loss based on image features.
    - **Design Motivation**: Since the regression task does not involve image features, the gradients of the image branch are no longer affected by imprecise depth estimation, making simple end-to-end training feasible. This also enables large-scale image resize augmentation (as there is no longer a need to maintain consistency between image size and regression targets).

3. **Velocity Augmentation**:

    - **Function**: Resolve the extreme imbalance of velocity distribution in the training data.
    - **Mechanism**: Most vehicle instances in nuScenes are stationary, resulting in a severely skewed velocity distribution. A predefined velocity is randomly assigned to some stationary targets, and the positions of their multi-frame point clouds are adjusted accordingly to create a "motion" effect. This augmentation is only applied to stationary targets because their complete point clouds can be precisely obtained through annotated bounding boxes.
    - **Design Motivation**: The imbalanced velocity distribution degrades the model's performance on velocity prediction. Velocity prediction is crucial for the planning module of autonomous driving. Ablation studies show that velocity augmentation reduces the AVE metric by approximately 25%.

### Loss & Training

DAL shares the target design and loss function formulations of TransFusion and BEVFusion, adding an extra auxiliary classification head—extracting sparse image features based on the annotated target center of gravity for classification. Its loss is directly added to the total loss without re-weighting. The auxiliary classification head compensates for the lack of supervision on the image branch in the dense and sparse perception stages.

## Key Experimental Results

### Main Results

| Dataset | Metric | DAL-Large | Prev. SOTA (UniTR) | Prev. SOTA (CMT) | Gain |
|--------|------|-----------|----------|------|------|
| nuScenes val | NDS | 74.0 | 73.3 | 72.9 | +0.7 |
| nuScenes val | mAP | 71.5 | 70.9 | 72.0 | +0.6 (vs CMT) |
| nuScenes test | NDS | 74.8 | 74.5 | 74.1 | +0.3 |
| nuScenes test | mAP | 72.0 | 70.5 | 72.0 | +1.5 (vs UniTR) |

DAL-Tiny achieves 71.3 NDS at 16.55 FPS, making it faster and more accurate than CMT-R50 with similar speed (14.2 FPS, 70.8 NDS).

### Ablation Study

| Config | Pipeline | Auxiliary Class. | Image Resize Range | Velocity Aug. | mAP | NDS |
|------|----------|---------|------------|---------|-----|-----|
| A (LiDAR only) | BEVFusion | - | - | - | 63.67 | 69.00 |
| B | BEVFusion | ✗ | 0.36-0.55 | ✗ | 63.59 | 68.71 |
| F | DAL | ✓ | 0.36-0.55 | ✗ | 64.16 | 69.52 |
| G | DAL | ✓ | 0.36-0.88 | ✗ | 68.07 | 70.87 |
| H | DAL | ✓ | 0.36-0.88 | ✓ | 68.50 | 71.94 |

### Key Findings

- When BEVFusion utilizes DAL's simplified training pipeline (config B), its performance falls short of the LiDAR-only baseline (config A), demonstrating its reliance on complex pre-training to leverage the image modality.
- DAL's pipeline enables large-scale resize augmentation (config F$\rightarrow$G), resulting in a +3.91 mAP improvement.
- Velocity augmentation reduces mAVE from 25.80 to 19.31, a reduction of approximately 25%.
- DAL recommends using a small image branch + large LiDAR branch configuration, as classification tasks place lower demands on the image branch.

## Highlights & Insights

- **Deriving algorithm design from labeling rules**: Elevating industry standards of data labeling to algorithmic design principles, offering a unique and compelling perspective.
- **Simple yet powerful**: Reaches SOTA using only the most classical components (ResNet + VoxelNet + FPN + SECOND) without any attention mechanism.
- **Minimalist training pipeline**: Requires only an ImageNet pre-trained image backbone and a one-stage end-to-end training process, eliminating the need for customized learning rate schedules.
- **Pareto optimality of speed and accuracy**: Provides a superior speed-accuracy trade-off compared to existing methods across various configurations.

## Limitations & Future Work

- Objects outside the LiDAR range are not considered (as these objects are not annotated in nuScenes).
- Since nuScenes has only 10 classes, simple classification tasks cannot fully leverage the capabilities of advanced image backbones (such as SwinTransformer).
- DAL currently uses an attention-free pipeline, and attention mechanisms like DSVT or DETR could be introduced to enhance performance in the future.
- Generalizability has not been verified on other datasets like Waymo.

## Related Work & Insights

- **BEVFusion (MIT/ADLab)**: A representative method of BEV space fusion, serving as the baseline for DAL.
- **TransFusion**: A Transformer-based fusion method from which DAL shares target and loss design.
- **CMT / UniTR**: SOTA methods based on attention mechanisms.
- **Insight**: Algorithm design should respect the basic rules of data generation; "less is more" holds true under correct design principles.

## Rating

- Novelty: ⭐⭐⭐⭐ (The perspective of deriving design from labeling rules is highly novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Detailed ablations, speed-accuracy analyses, and comparisons across multiple configurations)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐ (Provides clear design principles and a strong baseline for LiDAR-camera fusion)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](../../CVPR2025/autonomous_driving/racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[ECCV 2024\] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)

</div>

<!-- RELATED:END -->
