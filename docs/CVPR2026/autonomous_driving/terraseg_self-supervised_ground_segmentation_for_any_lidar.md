---
title: >-
  [Paper Note] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR
description: >-
  [CVPR 2026][Autonomous Driving][Ground Segmentation] This paper proposes TerraSeg, the first self-supervised, domain-agnostic LiDAR ground segmentation model. By constructing the large-scale unified OmniLiDAR dataset (12 public benchmarks, 15 sensor types, ~22 million scans) and a novel PseudoLabeler self-supervised pseudo-label generation module, TerraSeg achieves state-of-the-art performance on nuScenes, SemanticKITTI, and Waymo without any human annotation.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Ground Segmentation
  - Self-Supervised Learning
  - Cross-Sensor Generalization
  - LiDAR Perception
  - Pseudo Labels
date: 2026-05-08
content_hash: 215eea8130c63215
---

# TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR

**Conference**: CVPR 2026
**arXiv**: [2603.27344](https://arxiv.org/abs/2603.27344)
**Code**: Public (Apache 2.0)
**Area**: Autonomous Driving / 3D Point Cloud Segmentation
**Keywords**: Ground Segmentation, Self-Supervised Learning, Cross-Sensor Generalization, LiDAR Perception, Pseudo Labels

## TL;DR

This paper proposes TerraSeg, the first self-supervised, domain-agnostic LiDAR ground segmentation model. By constructing the large-scale unified OmniLiDAR dataset (12 public benchmarks, 15 sensor types, ~22 million scans) and a novel PseudoLabeler self-supervised pseudo-label generation module, TerraSeg achieves state-of-the-art performance on nuScenes, SemanticKITTI, and Waymo without any human annotation.

## Background & Motivation

**Background**: LiDAR ground segmentation is a foundational task in the autonomous driving perception stack, serving object discovery, free-space estimation, and localization/mapping. Existing methods fall into two categories: hand-crafted geometric methods (e.g., RANSAC, PatchWork++) and supervised learning methods (e.g., GndNet).

**Limitations of Prior Work**: Hand-crafted methods are fast and annotation-free but rely on simplistic terrain assumptions (e.g., global planarity) and sensor-specific parameter tuning, requiring re-tuning for new environments or sensors and offering poor generalization. Supervised learning methods generalize better but depend on expensive per-point manual annotations, making them highly unscalable.

**Key Challenge**: Fast, annotation-free hand-crafted methods lack generalization, while generalizable learning-based methods require costly annotations. The ideal solution should simultaneously be annotation-free, achieve zero-shot cross-sensor generalization, and run in real time.

**Goal**: (1) How to train a high-quality ground segmentation model without any human annotation; (2) How to enable a single model to generalize across different sensors, scenes, and weather conditions.

**Key Insight**: Inspired by the success of large-scale pre-training in NLP/CV, but rather than pursuing a multi-task general-purpose system, this work follows a single-task, domain-agnostic approach—training self-supervisedly on highly diverse geometric data to achieve zero-shot cross-domain transfer.

**Core Idea**: Aggregate ~22 million scans from 12 datasets and 15 sensor types to construct OmniLiDAR, and use self-supervised pseudo labels (PseudoLabeler) to train a domain-agnostic ground segmentation model based on Point Transformer v3.

## Method

### Overall Architecture

The TerraSeg framework consists of three core components: (1) the OmniLiDAR dataset—unifying and standardizing raw LiDAR scans from 12 public driving datasets; (2) the PseudoLabeler—generating per-frame ground/non-ground pseudo labels via self-supervised runtime optimization; and (3) the TerraSeg model—a real-time, domain-agnostic ground segmentation network based on Point Transformer v3, trained on pseudo labels. The input is raw 3D point cloud coordinates; the output is per-point ground/non-ground confidence scores.

### Key Designs

1. **OmniLiDAR Unified Dataset**:

    - **Function**: Provides unprecedented sensor diversity to support cross-domain generalization training.
    - **Mechanism**: Aggregates ~22 million raw scans from 12 public driving datasets—including nuScenes, SemanticKITTI, Waymo, and Argoverse 2—covering 15 distinct LiDAR hardware configurations. All data undergo a three-step normalization: (1) downsampling to 0.2 Hz to ensure diversity; (2) aligning coordinate frames via sensor-specific transforms such that $z=0$ approximates ground level and the $x$-axis points forward; (3) removing ego-vehicle points according to sensor model. Only normalized $(x, y, z)$ coordinates and metadata labels are retained.
    - **Design Motivation**: The sensor and scene diversity of any single dataset is insufficient to train a generalizable model; large-scale aggregation is required to cover diverse terrain, weather, and sensor configurations.

2. **PseudoLabeler Self-Supervised Pseudo Label Generation**:

    - **Function**: Generates high-quality ground/non-ground labels for each raw LiDAR scan without human annotation.
    - **Mechanism**: Ground segmentation is reformulated as a surrogate task of bird's-eye-view elevation map estimation. An MLP parameterizes an elevation map $g_\theta: \mathbb{R}^2 \to \mathbb{R}$, and the vertical residual for each point is computed as $\Delta d_i = z_i - g_\theta(x_i, y_i)$. Leveraging the geometric prior that the ground is the lowest continuous surface in the scene, an asymmetric loss is designed: points below the predicted surface incur a quadratic penalty, while points above incur a Huber loss. The pipeline consists of three stages: preprocessing (removing negative noise) → runtime optimization (SiLU activation + AdamW + EMA early stopping) → postprocessing (columnar refinement to recover object base points misclassified as ground).
    - **Design Motivation**: The simple yet powerful geometric prior that "the ground is the lowest surface" enables fully self-supervised training. Carefully designed pre- and post-processing steps address practical issues such as multipath reflection noise and vehicle tire misclassification.

3. **TerraSeg Domain-Agnostic Model Design**:

    - **Function**: Real-time inference with zero-shot cross-sensor ground segmentation.
    - **Mechanism**: Built on the Point Transformer v3 backbone with three key domain-agnostic adaptations: (1) dataset-specific normalization is disabled, forcing the model to learn universal geometric priors; (2) Batch Normalization is replaced with Group Normalization to handle distributional instability in mixed-sensor batches; (3) only three input features are used—a constant feature of 1, normalized height, and normalized horizontal distance—with raw coordinates used solely for voxel grid construction. Both Base (accurate) and Small (efficient) variants are provided.
    - **Design Motivation**: By constraining input features and disabling domain-specific modules, the model is forced to learn only universal geometric criteria rather than sensor-specific artifact patterns.

### Loss & Training

The training loss is Binary Cross-Entropy combined with symmetric Lovász-Softmax loss: $\mathcal{L} = \mathcal{L}_{BCE} + \lambda \mathcal{L}_{Lovász}$ ($\lambda = 1.0$). The BCE term uses dynamic positive class weights (tracked via EMA of ground/non-ground point ratios) to adaptively handle class imbalance across scenes. The AdamW optimizer is used (lr = 2e-3, weight decay = 5e-3) with linear warm-up followed by cosine decay. Voxelization resolution is 0.05, effective batch size is 256, and a custom epoch length of 20,000 frames is employed.

## Key Experimental Results

### Main Results

Results on the nuScenes validation set (trained without human annotation):

| Method | Annotation | Ground IoU | Non-Ground IoU | mIoU | Throughput (Hz) |
|--------|-----------|-----------|---------------|------|----------------|
| RANSAC | None | 89.14 | 83.97 | 86.55 | 255.0 |
| PatchWork++ | None | 86.19 | 81.42 | 83.80 | 30.0 |
| TRAVEL | None | 89.76 | 87.16 | 88.46 | 365.7 |
| GndNet | **Yes** | 82.54 | 78.72 | 80.62 | 484.3 |
| **TerraSeg-B** | **None** | **93.50** | **91.45** | **92.47** | 28.0 |
| **TerraSeg-S** | **None** | 92.40 | 90.49 | 91.45 | 49.8 |
| Supervised Upper Bound | Yes | 95.96 | 94.65 | 95.31 | 28.0 |

### Ablation Study

Ablation of PseudoLabeler components (nuScenes):

| Configuration | Effect |
|---------------|--------|
| Without preprocessing | Negative noise causes elevation map to sink, leading to over-segmentation of ground |
| Without postprocessing | Vehicle bases/tires are misclassified as ground |
| Full PseudoLabeler | mIoU 90.63 (TerraSeg trained on these labels achieves 92.47) |

### Key Findings

- **Self-supervised surpasses supervised baselines**: TerraSeg-B (no annotation) achieves mIoU 92.47, far exceeding GndNet trained with annotations (80.62).
- **Near supervised upper bound**: The gap with the fully supervised variant (95.31) is only ~3 percentage points.
- **Cross-dataset consistency**: State-of-the-art results are achieved consistently across nuScenes, SemanticKITTI, and Waymo.
- **Real-time inference**: TerraSeg-S reaches ~50 Hz and TerraSeg-B ~28 Hz, satisfying online deployment requirements.
- **Student surpasses teacher**: The TerraSeg model (mIoU 92.47) outperforms its pseudo-label source, PseudoLabeler (90.63), demonstrating the model's ability to generalize and denoise.

## Highlights & Insights

- **Large-scale aggregation strategy**: Unifying 12 datasets across 15 sensor types is a substantial engineering contribution; OmniLiDAR itself is a significant artifact.
- **Elegant self-supervised formulation**: Recasting ground segmentation as elevation map estimation is a clever use of a simple geometric prior to enable annotation-free training.
- **Student-surpasses-teacher phenomenon**: This demonstrates that large-scale diverse data combined with neural network generalization can effectively filter noise from pseudo labels.
- **High practical value**: Requires no annotation data, supports any LiDAR sensor, runs in real time, and is open-sourced—making this an immediately deployable contribution.

## Limitations & Future Work

- Only single-frame point clouds are processed; temporal information is not exploited, which may limit performance on complex terrain.
- Pseudo-label quality may degrade in extreme scenarios such as steep slopes or dense vegetation.
- Only binary classification (ground/non-ground) is produced, with no semantic information (e.g., traversability level).
- Extension to semantic ground segmentation (distinguishing road, grass, dirt, etc.) is a natural future direction.

## Related Work & Insights

- The concentric zone polar grid concept from PatchWork++ could be combined with learning-based methods.
- The self-supervised runtime optimization of Chodosh et al. is the direct predecessor of PseudoLabeler; this work introduces key improvements upon that foundation.
- The successful application of Point Transformer v3 as a backbone validates its general applicability to point cloud tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both the OmniLiDAR dataset and the self-supervised domain-agnostic training paradigm are pioneering contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across three mainstream benchmarks, detailed ablation studies, and complete baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated relationships among the three core components.
- **Value**: ⭐⭐⭐⭐⭐ — Extremely high practical value, directly addressing annotation bottlenecks and sensor generalization challenges in autonomous driving ground segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)
- [\[CVPR 2026\] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)
- [\[AAAI 2026\] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning](../../AAAI2026/autonomous_driving/dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)
- [\[CVPR 2026\] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)

</div>

<!-- RELATED:END -->
