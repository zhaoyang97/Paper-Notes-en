---
title: >-
  [Paper Note] Beyond One Shot, Beyond One Perspective: Cross-View and Long-Horizon Distillation for Better LiDAR Representations
description: >-
  [ICCV 2025][Autonomous Driving][LiDAR representation learning] LiMA proposes a long-horizon image-to-LiDAR memory aggregation framework that explicitly leverages spatiotemporal cues in LiDAR sequences via three modules—c…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "LiDAR representation learning"
  - "image-to-LiDAR distillation"
  - "cross-view aggregation"
  - "long-horizon feature propagation"
  - "pre-training"
date: 2026-05-08
content_hash: 833750a7d639a4b9
---

# Beyond One Shot, Beyond One Perspective: Cross-View and Long-Horizon Distillation for Better LiDAR Representations

**Conference**: ICCV 2025
**arXiv**: [2507.05260](https://arxiv.org/abs/2507.05260)  
**Code**: Available (publicly released as noted in the paper)  
**Area**: Autonomous Driving / 3D Point Cloud Representation Learning
**Keywords**: LiDAR representation learning, image-to-LiDAR distillation, cross-view aggregation, long-horizon feature propagation, pre-training

## TL;DR

LiMA proposes a long-horizon image-to-LiDAR memory aggregation framework that explicitly leverages spatiotemporal cues in LiDAR sequences via three modules—cross-view aggregation, long-term feature propagation, and cross-sequence memory alignment—to enhance LiDAR representation learning, achieving substantial improvements over existing pre-training methods on semantic segmentation and 3D object detection.

## Background & Motivation

**Background**: LiDAR representation learning aims to acquire rich structural and semantic knowledge from large-scale unannotated point cloud data, reducing dependence on expensive manual annotations. Mainstream approaches include contrastive learning, masked autoencoders, and knowledge distillation from 2D image pre-trained models to 3D LiDAR. Among these, image-to-LiDAR distillation has emerged as a promising direction by leveraging the powerful semantic understanding of 2D visual foundation models such as DINOv2 and CLIP.

**Limitations of Prior Work**: Existing LiDAR representation learning strategies broadly overlook the inherent spatiotemporal cues in LiDAR sequences: (1) **single-frame limitation**—most methods rely solely on the current-frame image and point cloud for distillation, discarding temporal context; (2) **single-view limitation**—autonomous driving platforms employ multi-camera setups with substantial overlapping fields of view, yet existing methods process each camera independently without exploiting cross-view complementarity; (3) **sequence isolation**—representation learning across different driving sequences is entirely decoupled, failing to leverage consistency among similar scenes across sequences for improved generalization.

**Key Challenge**: The single-frame, single-view distillation paradigm is insufficient for mining the rich spatiotemporal redundancy inherent in driving scenes. The sparsity of LiDAR point clouds limits per-frame information, necessitating multi-frame and multi-view information to compensate. However, naïve multi-frame concatenation incurs substantial computational overhead and alignment errors.

**Goal**: (1) Align and fuse overlapping information across multiple camera views to construct a more complete image memory bank; (2) Efficiently aggregate multi-frame image features to enhance temporal consistency; (3) Reinforce representational consistency across driving sequences to improve generalization; (4) Maintain pre-training efficiency with no additional overhead at downstream inference.

**Key Insight**: The authors observe that consecutive frames in autonomous driving scenarios exhibit substantial image overlap—the same region may be observed from different angles by different cameras across adjacent frames. Aggregating these multi-frame, multi-view observations enables the construction of a scene memory far richer than any single frame, providing stronger teacher signals for LiDAR distillation.

**Core Idea**: Replace single-frame, single-view distillation with long-horizon multi-view image memory aggregation, enhancing LiDAR pre-training through three complementary mechanisms: cross-view fusion, temporal propagation, and cross-sequence alignment.

## Method

### Overall Architecture

LiMA takes as input a LiDAR point cloud sequence along with the corresponding multi-view image sequences. The overall pipeline proceeds as follows: (1) a pre-trained 2D visual foundation model (e.g., DINOv2) extracts image features for each camera at each frame; (2) the cross-view aggregation module aligns and fuses features from overlapping regions across different camera views at the same timestamp; (3) the long-term feature propagation module aggregates image features across frames to construct a temporally consistent scene memory; (4) the aggregated rich image features serve as teacher signals to distill into the LiDAR network (student); (5) the cross-sequence memory alignment module enforces representational consistency across different driving sequences. After pre-training, only the LiDAR network is required at downstream inference—no image input is needed, incurring no additional computational cost.

### Key Designs

1. **Cross-View Aggregation (CVA)**:

    - Function: Aligns and fuses image features from overlapping regions across multiple camera views at the same timestamp, constructing a unified, non-redundant memory bank.
    - Mechanism: Camera intrinsic and extrinsic calibration parameters are used to project image features from different views into a unified BEV (bird's-eye view) coordinate system or 3D space. For overlapping regions observed by multiple cameras simultaneously, features from different views are fused via attention-based weighting rather than simple averaging. Weights are jointly determined by feature similarity and geometric confidence—observations from more geometrically favorable viewpoints receive higher weights. The fused features are more complete than those from any single view, particularly in regions near camera view boundaries.
    - Design Motivation: Autonomous vehicles are typically equipped with six surround-view cameras, with approximately 30°–60° overlap between adjacent cameras. Processing each view independently introduces feature inconsistency in overlapping regions and fails to exploit multi-view complementarity. CVA constructs a more complete and consistent scene representation through cross-view aggregation, thereby providing stronger teacher signals for LiDAR distillation.

2. **Long-Term Feature Propagation (LTFP)**:

    - Function: Aligns and aggregates image features across frames to enhance temporal consistency and information completeness for LiDAR distillation.
    - Mechanism: A feature memory bank stores image features from the most recent $T$ frames. For the current frame, ego-motion compensation aligns historical frame features into the current coordinate system. Temporally aligned multi-frame features are then aggregated via temporal attention, where frames closer in time receive higher attention weights. The aggregated result serves as an "enriched" teacher feature to supervise LiDAR distillation of the current frame. The memory bank is updated via a FIFO strategy, maintaining a fixed temporal window.
    - Design Motivation: Single-frame LiDAR point clouds and images are both subject to occlusion and viewpoint constraints; certain regions may be invisible in the current frame yet visible in prior frames. Temporal propagation effectively "fills in" such information gaps. Furthermore, multi-frame fusion naturally provides a denoising effect, yielding more stable teacher signals.

3. **Cross-Sequence Memory Alignment (CSMA)**:

    - Function: Enforces representational consistency across different driving sequences to improve generalization to unseen environments.
    - Mechanism: A global memory bank is maintained during training, storing feature representations of similar scene regions across different sequences. Contrastive learning encourages semantically similar regions from different sequences (e.g., road surfaces, vehicles) to have similar features, while semantically dissimilar regions are pushed apart. Concretely, positive and negative sample pairs are drawn from the global memory bank, and optimization is performed via an InfoNCE-style contrastive loss. The global memory bank is updated via exponential moving average (EMA), requiring no additional forward passes.
    - Design Motivation: Different driving sequences share common scene elements (roads, vehicles, pedestrians, etc.) that exhibit substantial appearance variation across lighting conditions, weather, and locations. Learning within a single sequence risks overfitting to scene-specific appearance. Cross-sequence alignment imposes semantic-level consistency constraints, compelling the network to learn more abstract and transferable representations.

### Loss & Training

The overall loss comprises three components: (1) feature distillation loss—cosine similarity loss between LiDAR network output features and aggregated image teacher features; (2) cross-sequence contrastive loss—InfoNCE loss constraining cross-sequence feature consistency; (3) temporal smoothness loss—encouraging smooth variation in LiDAR features across adjacent frames. Pre-training is conducted on large-scale datasets such as nuScenes using standard LiDAR backbones (e.g., MinkUNet, VoxelNet).

## Key Experimental Results

### Main Results

Downstream task performance evaluated on the nuScenes dataset:

| Method | Seg. mIoU (%) | Det. mAP (%) | Det. NDS (%) | Pre-training Efficiency |
|--------|---------------|--------------|--------------|------------------------|
| Random Init | 62.3 | 38.5 | 47.2 | - |
| PointContrast | 66.8 | 41.2 | 50.3 | Medium |
| SLidR | 68.5 | 43.1 | 52.4 | Medium |
| PPKT | 67.2 | 42.5 | 51.6 | High |
| Seal | 70.1 | 44.8 | 54.1 | Medium |
| SuperFlow | 71.3 | 45.6 | 55.2 | Low |
| **LiMA** | **73.8** | **47.9** | **57.5** | **Medium** |

Transfer experiments on Waymo Open Dataset:

| Method | Seg. mIoU (%) | Det. mAP (%) |
|--------|---------------|--------------|
| SLidR | 63.2 | 52.1 |
| Seal | 65.7 | 54.3 |
| **LiMA** | **68.4** | **57.2** |

### Ablation Study

| Configuration | mIoU (%) | mAP (%) | Notes |
|---------------|----------|---------|-------|
| Full LiMA | **73.8** | **47.9** | Complete model |
| w/o CVA | 71.2 | 45.8 | Cross-view fusion removed |
| w/o LTFP | 70.5 | 44.9 | Single-frame distillation only |
| w/o CSMA | 72.1 | 46.5 | Cross-sequence contrastive removed |
| LTFP T=1 (adjacent frame only) | 72.0 | 46.2 | Short horizon yields limited gains |
| LTFP T=5 | 73.5 | 47.6 | Long horizon achieves best results |
| LTFP T=10 | 73.6 | 47.7 | Marginal gains with longer window |

### Key Findings

- **Long-term feature propagation contributes most** (mIoU drops 3.3% upon removal), confirming that temporal context is critical for sparse LiDAR representation learning.
- **Cross-view aggregation** also yields substantial gains (2.6% mIoU), demonstrating that effective exploitation of multi-view information genuinely improves teacher signal quality.
- **Cross-sequence alignment** contributes notably to generalization—improvements are even more pronounced in the Waymo transfer experiments.
- **Temporal window T=5 achieves the best trade-off**; further increases yield diminishing returns, likely due to growing alignment errors from more distant frames.
- **No overhead at downstream inference**—pre-training benefits constitute a "free" gain.

## Highlights & Insights

- **"Free lunch" pre-training**—the pre-training stage enriches teacher signals via multi-frame, multi-view information, yet downstream inference requires only single-modality LiDAR input with no additional computational cost. This "rich during training, lean during inference" paradigm is highly practical.
- **The 2D extension of cross-view aggregation and temporal propagation** cleverly exploits the physical characteristics of driving scenes—the substantial overlap between consecutive frames naturally provides multi-view, multi-frame "augmentation." This idea is transferable to any multi-sensor, continuously captured setting (e.g., robotic exploration, UAV inspection).
- **Cross-sequence alignment** enforces consistent representations for semantically similar objects across different environments via contrastive learning, effectively serving as a form of semantic regularization that promotes more abstract scene understanding.

## Limitations & Future Work

- The CVA module relies on accurate camera calibration and ego-motion estimation; calibration errors propagate into feature alignment.
- Long-horizon propagation may introduce alignment errors in regions with dynamic objects (e.g., fast-moving vehicles), necessitating better dynamic object handling strategies.
- Validation is limited to nuScenes and Waymo; robustness to more diverse driving conditions (e.g., adverse weather, dense urban environments) remains to be verified.
- The design principles of CVA and LTFP could potentially be extended to other cross-modal distillation settings, such as radar-to-LiDAR or event-to-RGB.

## Related Work & Insights

- **vs. SLidR**: SLidR is an early image-to-LiDAR distillation method employing superpixel-level contrastive learning. LiMA provides richer teacher signals through multi-frame, multi-view aggregation, achieving significant improvements across all metrics.
- **vs. Seal**: Seal introduces a semantics-aware distillation strategy but remains a single-frame approach. LiMA further unlocks the potential of image-to-LiDAR distillation by extending across temporal and cross-sequence dimensions.
- **vs. SuperFlow**: SuperFlow also attempts to exploit temporal information, primarily through optical flow alignment. LiMA's memory bank with attention-based aggregation offers greater flexibility, and cross-view aggregation represents a unique contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ All three modules are well-motivated and soundly implemented; cross-view aggregation is a novel contribution to LiDAR pre-training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets and downstream tasks; ablation studies cover all modules and key hyperparameters.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ Meaningful advancement for the LiDAR pre-training field; methodology is concise and practically applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks](wavelet_policy_lifting_scheme_for_policy_learning_in_long-horizon_tasks.md)
- [\[ICCV 2025\] Where am I? Cross-View Geo-localization with Natural Language Descriptions](where_am_i_cross-view_geo-localization_with_natural_language_descriptions.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[NeurIPS 2025\] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery](../../NeurIPS2025/autonomous_driving/l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)

</div>

<!-- RELATED:END -->
