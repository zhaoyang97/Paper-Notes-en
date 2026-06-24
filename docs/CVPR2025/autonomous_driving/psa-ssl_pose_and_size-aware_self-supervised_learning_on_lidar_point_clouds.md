---
title: >-
  [Paper Note] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds
description: >-
  [CVPR 2025][Autonomous Driving][Self-Supervised Learning] PSA-SSL is proposed to preserve object pose and size information by incorporating a self-supervised bounding box regression pre-training task into contrastive learning, integrated with LiDAR Pattern Augmentation to achieve cross-sensor generalization, significantly outperforming SOTA self-supervised methods on 3D semantic segmentation and object detection.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Self-Supervised Learning"
  - "Point Cloud Representation"
  - "LiDAR Cross-Sensor"
  - "Bounding Box Regression"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 242a0a0d2df1adcb
---

# PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds

**Conference**: CVPR 2025  
**arXiv**: [2503.13914](https://arxiv.org/abs/2503.13914)  
**Code**: [GitHub](https://github.com/TRAILab/PSA-SSL)  
**Area**: Autonomous Driving  
**Keywords**: Self-Supervised Learning, Point Cloud Representation, LiDAR Cross-Sensor, Bounding Box Regression, Contrastive Learning

## TL;DR

PSA-SSL is proposed to preserve object pose and size information by incorporating a self-supervised bounding box regression pre-training task into contrastive learning, integrated with LiDAR Pattern Augmentation to achieve cross-sensor generalization, significantly outperforming SOTA self-supervised methods on 3D semantic segmentation and object detection.

## Background & Motivation

Self-Supervised Learning (SSL) on LiDAR point clouds shows promise in learning feature representations transferable to various downstream tasks and sensors. However, existing contrastive learning-based SSL methods suffer from a fundamental limitation: the contrastive loss maximizes the feature similarity of the same instance under different geometric transformations (rotation, translation, scaling), resulting in learned features that are invariant to these transformations—meaning the pose and size information of objects is lost.

This is detrimental to downstream localization and geometry-sensitive 3D scene understanding tasks (e.g., semantic segmentation, object detection). Furthermore, existing methods are typically pre-trained and fine-tuned on the same dataset, showing limited generalization capability across different LiDAR sensors.

Key Finding: Contrastive learning makes features invariant to geometric transformations, whereas downstream tasks precisely require geometry-aware features. The authors explicitly encode pose and size information using a bounding box regression pre-training task, which is complementary to the contrastive loss—where the contrastive loss learns category-discriminative features and the regression head preserves geometric information.

## Method

### Overall Architecture

PSA-SSL consists of three stages: (1) **Preprocessing**—separating ground points using Patchwork++, clustering non-ground points with HDBSCAN, and fitting bounding boxes as regression targets; (2) **Pre-training**—jointly optimizing contrastive loss and bounding box regression loss; (3) **Fine-tuning**—fine-tuning on downstream segmentation or detection tasks. The method is model-agnostic and can be applied to any contrastive learning-based SSL method.

### Key Design 1: Self-Supervised Bounding Box Regression Pre-training Task

**Function**: Preserve object pose and size information on top of contrastive learning.

**Mechanism**: In the preprocessing phase, non-ground points are clustered using HDBSCAN to obtain object clusters, and an upright bounding box is fitted using an off-the-shelf algorithm. In the pre-training phase, a fully-connected regression head (2 layers with 256 dimensions) is added to the concatenated features output by the query and momentum encoders, predicting the offset of each clustered point to a fixed-size anchor box. A smooth L1 loss is used to calculate the regression loss only for the clustered points. Total loss: $\mathcal{L} = \beta_1 \mathcal{L}_{con} + \beta_2 \mathcal{L}_{reg}$.

**Design Motivation**: While contrastive loss makes features invariant to rotation, translation, and scaling, the regression task forces features to retain this geometric information. The two are complementary—contrastive learning provides category discriminativeness, while regression provides geometry awareness. The regression target is automatically obtained through unsupervised clustering and box fitting, ensuring true self-supervision.

### Key Design 2: LiDAR Pattern Augmentation (LPA)

**Function**: Learn generalized features across different LiDAR sensors.

**Mechanism**: During the data augmentation stage, the input point cloud is transformed into beam patterns of different LiDAR configurations (Velodyne-32/64, Ouster-64, etc.) via spherical projection, and then projected back to the point cloud. When contrastive loss maximizes feature similarity under different beam patterns, the model learns representations invariant to LiDAR sparsity patterns. Two variants are proposed: Single Pattern (transforming to a single random configuration) and PolarMix (mixing azimuth crops of multiple configurations).

**Design Motivation**: Existing SSL methods employ random point dropping and cuboid cropping as augmentations, which fail to simulate realistic differences in LiDAR beam patterns. By employing realistic sensor utility parameter transformations (FOV, channels, resolution), a single pre-trained model can be successfully transferred to various LiDARs.

### Key Design 3: Universality Design of the Framework

**Function**: Plug-and-play improvement for any contrastive learning-based SSL method.

**Mechanism**: The PSA extension can be applied to contrastive learning methods of different granularities, such as DepthContrast (scene-level) and SegContrast (region-level). The regression head operates as a parallel branch and does not increase pre-training time (actually reducing it by 33% due to faster convergence). The query/momentum encoders from the MoCo framework are used.

**Design Motivation**: Avoid designing a new SSL framework; instead, elevate existing methods as a general plug-in. Different granularities of contrastive learning (scene/region/point-level) have their respective pros and cons, and the PSA extension is effective across all granularities.

### Loss & Training

$\mathcal{L} = \beta_1 \mathcal{L}_{con} + \beta_2 \mathcal{L}_{reg}$, where $\mathcal{L}_{con}$ is the InfoNCE contrastive loss, and $\mathcal{L}_{reg}$ is the smooth L1 bounding box regression loss. $\beta_1 = 1.0$, $\beta_2 = 0.5$.

## Key Experimental Results

### Main Results: Pre-trained on Waymo → Fine-tuned on Different Datasets (1% Label, mIoU)

| Method | Waymo | nuScenes | SemanticKITTI |
|------|-------|----------|---------------|
| No pretraining | 49.34 | 35.71 | 41.95 |
| DepthContrast | 50.36 | 35.32 | 48.77 |
| **PSA-DC (Ours)** | **53.02** (+2.66) | **37.75** (+2.43) | **49.92** (+1.15) |
| SegContrast | 53.50 | 36.01 | 49.72 |
| **PSA-SC (Ours)** | **54.36** (+0.86) | **37.89** (+1.87) | **52.11** (+2.39) |

### Comparison with SOTA (SemanticKITTI Pre-trained & Fine-tuned)

| Method | 1% | 10% | 50% | 100% |
|------|-----|-----|-----|------|
| BEVContrast | 44.1 | 56.3 | 62.2 | 63.1 |
| MAELi | 41.1 | 56.3 | - | 63.5 |
| **PSA-SC (Ours)** | **45.9** | **58.5** | **63.3** | **64.2** |

### Cross-LiDAR Generalization (SemanticKITTI → nuScenes, 1% Label)

| Method | mIoU |
|------|------|
| TARL | 25.9 |
| **PSA-SC (Ours)** | **29.5** (+3.56) |

### Key Findings

- The PSA extension yields significant improvements across all baseline methods and datasets, with the most pronounced enhancement observed under extremely low label ratios (1%).
- PSA-DC closes the performance gap between scene-level and region-level methods, indicating that bounding box regression assists scene-level methods in learning finer-grained features.
- Obtaining the same performance as BEVContrast with 100% labels requires only 10% of SemanticKITTI labels (a 10x savings in annotation).
- A single model pre-trained on Waymo and directly transferred to nuScenes/SemanticKITTI significantly outperforms other methods, demonstrating robust cross-sensor generalization.
- Pre-training time does not increase and is even reduced by 33%, as bounding box regression accelerates the learning of useful features.

## Highlights & Insights

1. **Identifying fundamental limitations of contrastive learning**: Points out the overlooked issue of geometric invariance brought by contrastive loss, and provides a simple yet effective solution.
2. **Self-supervised bounding box regression**: Introduces bounding box regression from supervised learning into SSL pre-training for the first time, leveraging unsupervised clustering to generate regression targets.
3. **Cross-sensor generalization**: Achieves robust transfer across different sensors through realistic LiDAR utility parameter transformations (instead of simple random point dropping).

## Limitations & Future Work

- Preprocessing relies on the clustering quality of HDBSCAN, which may suffer from over- or under-segmentation.
- Bounding box fitting assumes objects are upright, which is not applicable to tilted objects.
- LiDAR augmentation only supports dense-to-sparse transformations, rendering it inapplicable when pre-training on sparse LiDARs (e.g., nuScenes).
- Regression targets are based on clustered point groups. Ground points and extremely large clusters are excluded, which may miss some scene info.

## Related Work & Insights

- **DepthContrast / SegContrast**: Two main baseline methods. The PSA extension significantly improves the performance of both in a plug-and-play manner.
- **BEVContrast**: Prev. SOTA, which compares BEV features. PSA-SC outperforms it across all label ratios.
- **LiDomAug**: A LiDAR beam augmentation method. Ours introduces it from domain adaptation to SSL and validates its effectiveness.

## Rating

⭐⭐⭐⭐ — Identifies the fundamental limitation of contrastive learning, with a simple yet effective solution (only adding a lightweight regression head). The 10x label savings result is highly practical. Cross-sensor transferability is a practical and essential demand in the autonomous driving domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[CVPR 2025\] Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)
- [\[CVPR 2025\] VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow](voteflow_enforcing_local_rigidity_in_self-supervised_scene_flow.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)

</div>

<!-- RELATED:END -->
