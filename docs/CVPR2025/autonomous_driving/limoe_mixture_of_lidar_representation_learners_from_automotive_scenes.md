---
title: >-
  [Paper Note] LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes
description: >-
  [CVPR 2025][Autonomous Driving][LiDAR Self-Supervised Learning] This paper proposes LiMoE, which fuses three complementary LiDAR representations (range images, sparse voxels, and raw point clouds) using a Mixture of Experts (MoE) mechanism. Through three-stage training (image-to-LiDAR pre-training -> contrastive mixture learning -> semantic mixture supervision), it achieves 51.4% mIoU on nuScenes segmentation and generalizes across 7 datasets.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "LiDAR Self-Supervised Learning"
  - "Mixture of Experts"
  - "Multi-Representation Fusion"
  - "Contrastive Learning"
  - "Point Cloud Segmentation"
date: 2026-05-08
content_hash: 136e03c9b68dfe06
---

# LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes

**Conference**: CVPR 2025  
**arXiv**: [2501.04004](https://arxiv.org/abs/2501.04004)  
**Code**: [https://github.com/Xiangxu-0103/LiMoE](https://github.com/Xiangxu-0103/LiMoE)  
**Area**: Autonomous Driving / Point Cloud Learning  
**Keywords**: LiDAR Self-Supervised Learning, Mixture of Experts, Multi-Representation Fusion, Contrastive Learning, Point Cloud Segmentation

## TL;DR

This paper proposes LiMoE, which fuses three complementary LiDAR representations (range images, sparse voxels, and raw point clouds) using a Mixture of Experts (MoE) mechanism. Through three-stage training (image-to-LiDAR pre-training -> contrastive mixture learning -> semantic mixture supervision), it achieves 51.4% mIoU on nuScenes segmentation and generalizes across 7 datasets.

## Background & Motivation

### Background

**Background**: LiDAR point clouds can be converted into multiple representations: range images (preserving raw sensor structure), sparse voxels (3D spatial structure), and raw point clouds (fine-grained details). Each representation has its own advantages and offers complementary information.

**Limitations of Prior Work**: Existing self-supervised methods only use a single representation (e.g., UniPAD uses voxels, SLidR uses range images), neglecting complementary information. Simple concatenation or average fusion yields suboptimal results, as different queries require varying contributions from different representations.

**Key Challenge**: The feature spaces and information densities of the three representations are entirely different (range images are dense 2D, voxels are sparse 3D, and point clouds are unordered), making direct fusion challenging.

**Key Insight**: Utilizing the gating mechanism of MoE allows the network to automatically select the most appropriate combination of representations for each query, enabling dynamic weighting instead of static fusion.

**Core Idea**: Image-distillation pre-training of three LiDAR encoders -> MoE dynamic fusion -> semantic mixture supervision = unified powerful point cloud representation.

### Goal

**Goal**: ### Key Designs

1. **Stage 1: Image-to-LiDAR Knowledge Distillation**: Distill features from a 2D image backbone into three LiDAR encoders, inheriting the semantic understanding capabilities of the image model.

2. **Stage 2: Contrastive Mixture Learning (CML)**: The MoE layer dynamically activates features of the three representations using gating and noise injection, and a contrastive loss distills the mixed features into a unified student encoder.

3. **Stage 3: Semantic Mixture Supervision (SMS)**: Extend MoE.

## Method

### Key Designs

1. **Stage 1: Image-to-LiDAR Knowledge Distillation**: Distill features from a 2D image backbone into three LiDAR encoders, inheriting the semantic understanding capabilities of the image model.

2. **Stage 2: Contrastive Mixture Learning (CML)**: The MoE layer dynamically activates the features of the three representations via a gating mechanism and noise injection, distilling the mixed features into a unified student encoder using contrastive loss.

3. **Stage 3: Semantic Mixture Supervision (SMS)**: Extend MoE to downstream segmentation, where each of the three representations predicts semantic logits, which are then fused via MoE weighting.

### Loss & Training

Contrastive: $\mathcal{L}_{con} = -\frac{1}{S}\sum_i \log \frac{e^{\langle k_i, q_i\rangle/\tau}}{\sum_{j\neq i} e^{\langle k_i, q_j\rangle/\tau}}$. Segmentation: CE + Lovász-Softmax + boundary loss.

## Key Experimental Results

| Dataset | LiMoE (ViT-L) | Best Single Rep. | Gain |
|--------|-------------|-----------|------|
| nuScenes seg. | **51.4%** mIoU | ~46-48 | +3-5 |
| SemanticKITTI (1%) | **44.85%** | ~40 | +4-5 |
| nuScenes-C Robustness | **mCE 88.43** | — | Best |

### Ablation Study
- MoE significantly outperforms concatenation/average fusion—since different regions require contributions from different representations.
- Three distinct representations > three identical representations—complementarity is key.
- Each representation exhibits unique activation patterns: range images focus on middle beams/dynamic objects, voxels focus on the upper layer/static background, and point clouds focus on the lower layer/fine details.

### Key Findings
- **Complementarity analysis is highly insightful**: Visualizations show that range images, voxels, and point clouds have distinctly different contribution patterns across different beam numbers, distances, and semantic categories.
- Cross-domain generalization to 7 datasets is consistently effective.

## Highlights & Insights
- **Dynamic fusion of MoE > static fusion**: Allows each query point to autonomously select the most useful representation.
- **Quantitative complementarity analysis**: Visualizing the "expertise area" of each representation provides valuable insights into understanding LiDAR perception.

## Limitations & Future Work
- High computational overhead during the SMS stage (85.8M parameters, 8.3 FPS).
- Range images are unavailable in some cross-domain datasets due to unknown FoV parameters.
- Only 2D representation (range image) was explored; there might be more choices for 3D representations.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel application of MoE in LiDAR multi-representation fusion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ nuScenes + KITTI + 7 cross-domain datasets + robustness tests.
- Writing Quality: ⭐⭐⭐⭐ Clear complementarity analysis.
- Value: ⭐⭐⭐⭐ Provides a unified framework for LiDAR representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 4D Contrastive Superflows are Dense 3D Representation Learners](../../ECCV2024/autonomous_driving/4d_contrastive_superflows_are_dense_3d_representation_learners.md)
- [\[CVPR 2025\] Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [\[CVPR 2025\] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)

</div>

<!-- RELATED:END -->
