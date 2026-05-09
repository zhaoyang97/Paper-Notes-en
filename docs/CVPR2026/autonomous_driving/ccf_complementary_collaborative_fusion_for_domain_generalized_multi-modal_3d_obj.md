---
title: >-
  [Paper Note] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection
description: >-
  [CVPR 2026][Autonomous Driving][multi-modal 3D detection] To address the modality imbalance problem in dual-branch multi-modal 3D detectors under domain shift, this paper proposes the CCF framework, which systematically improves camera query utilization and cross-domain robustness through three components: Query Decoupled Loss, LiDAR-Guided Depth Prior, and Complementary Cross-Modal Masking.
tags:
  - CVPR 2026
  - Autonomous Driving
  - multi-modal 3D detection
  - domain generalization
  - modality imbalance
  - LiDAR-camera fusion
  - cross-domain robustness
date: 2026-05-08
content_hash: 289e7744ad04a9eb
---

# CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.23276](https://arxiv.org/abs/2603.23276)
**Code**: [GitHub](https://github.com/IMPL-Lab/CCF.git)
**Area**: Autonomous Driving
**Keywords**: multi-modal 3D detection, domain generalization, modality imbalance, LiDAR-camera fusion, cross-domain robustness

## TL;DR

To address the modality imbalance problem in dual-branch multi-modal 3D detectors under domain shift, this paper proposes the CCF framework, which systematically improves camera query utilization and cross-domain robustness through three components: Query Decoupled Loss, LiDAR-Guided Depth Prior, and Complementary Cross-Modal Masking.

## Background & Motivation

**State of the Field**: Multi-modal 3D detection (LiDAR + Camera) has achieved strong performance on standard benchmarks, but suffers significant degradation under domain shift scenarios such as adverse weather and lighting changes.

**Limitations of Prior Work**: (a) Under conditions such as rain or nighttime, different modalities degrade at different rates—rainy conditions lead to sparse LiDAR point clouds, while nighttime conditions deteriorate camera image quality; (b) in dual-branch detectors, the LiDAR branch dominates the detection process, causing the semantic information from the camera branch to be systematically underutilized.

**Root Cause**: Pilot study analysis reveals that during training, the matching ratio between 3D queries and 2D queries reaches 37.5:1, leaving 2D queries with almost no supervision signal. Even when 2D detector proposal quality remains relatively high under cross-domain conditions (2D AP outperforms 3D projection), the 3D mAP of 2D queries is only 18.44% (vs. 67.75% for 3D queries).

**Paper Goals**: Rebalance modality utilization in dual-branch detectors so that the camera branch can contribute more effectively when LiDAR quality degrades.

**Starting Point**: The problem is approached from three dimensions: supervision imbalance, inaccurate depth initialization, and over-reliance on LiDAR during the fusion stage.

**Core Idea**: Systematically improve the competitiveness of 2D queries through decoupled supervision, geometric prior enhancement, and a complementary masking strategy.

## Method

### Overall Architecture

Built upon the MV2DFusion dual-branch detection framework, CCF consists of three complementary components: Query Decoupled Loss (QDL) for balanced supervision, LiDAR-Guided Depth Prior (LGDP) for improved spatial initialization, and Complementary Cross-Modal Masking (CCMM) for adaptive fusion.

### Key Designs

1. **Query Decoupled Loss (QDL)**: The decoder is executed in parallel three times (with shared weights): using only 2D queries, only 3D queries, and fused queries, respectively. Hungarian matching and loss computation are performed independently for each:
    $\mathcal{L}_{total} = \mathcal{L}_{2d} + \mathcal{L}_{3d} + \mathcal{L}_{fused}$
   **Design Motivation**: In standard training, 3D queries monopolize Hungarian matching due to superior localization quality (37.5:1 ratio), leaving 2D queries with almost no gradient updates. Three parallel executions rather than post-hoc separation after a single decoding step are used to prevent 2D queries from exploiting 3D queries as shortcuts via self-attention (shortcut learning). At inference, only the fused branch is used, incurring no additional computational cost.

2. **LiDAR-Guided Depth Prior (LGDP)**: For each 2D proposal, a learned depth distribution $\mathbf{d}_i^{2d} \in \mathbb{R}^D$ is obtained from the image branch, and a geometric prior distribution $\mathbf{d}_i^{3d} \in \mathbb{R}^D$ (a depth histogram of LiDAR points within the view frustum) is obtained from the LiDAR point cloud. A confidence network predicts the fusion weight $\lambda_i \in [0,1]$:
    $\mathbf{d}_i^{fused} = \sigma(\lambda_i \cdot \log(\mathbf{d}_i^{2d}) + (1-\lambda_i) \cdot \log(\mathbf{d}_i^{3d}))$
   This is a Product-of-Experts-style fusion in log space. **Design Motivation**: Pure image-based depth prediction yields a MAE of 1.78m on the source domain, which degrades further under domain shift (Rain: 3.01m). Directly incorporating LiDAR geometric information substantially improves the 3D localization of 2D queries. The adaptive weight accommodates cases where LiDAR is sparse at long range or noisy in rainy conditions.

3. **Complementary Cross-Modal Masking (CCMM)**: GridMask is applied to images, while a complementary mask is applied to LiDAR (retaining LiDAR points at image-masked locations, and vice versa). Curriculum learning is adopted, with the masking probability linearly increasing from 0 to $p=0.7$.
   **Design Motivation**: This simulates the asynchronous modality degradation observed in real-world domain shift (poor LiDAR but available camera in rain, and the opposite at night), forcing the decoder to adaptively select queries based on modality reliability during fusion rather than fixating on LiDAR. Unlike the complete dropout strategy in CMT, complementary masking keeps both modalities simultaneously available but with complementary visibility.

### Loss & Training

- Classification loss: Focal Loss; Regression loss: L1 Loss
- Two-stage training: Stage 1 independently pre-trains the 2D/3D detectors; Stage 2 freezes the 3D detector and trains the fusion decoder
- AdamW optimizer, initial LR 4e-4, cosine annealing, 24 epochs

## Key Experimental Results

### Main Results

| Method | Source mAP | Rain mAP | Night mAP | Boston mAP | Avg mAP |
|--------|-----------|----------|-----------|------------|---------|
| FSDv2 (LiDAR-only) | 59.6 | 23.4 | 36.6 | 28.2 | 29.4 |
| ISFusion | 66.3 | 39.8 | 41.8 | 45.4 | 42.3 |
| Baseline | 68.4 | 41.9 | 42.9 | 47.4 | 44.1 |
| **CCF (Ours)** | **68.2** | **44.7** | **44.2** | **50.6** | **46.5** |
| CCF (Oracle) | 73.6 | 72.9 | 46.9 | 73.6 | 64.5 |

### Ablation Study

| DL | DP | CM | Rain mAP | Night mAP | Boston mAP |
|----|----|----|----------|-----------|------------|
| ✗ | ✗ | ✗ | 41.9 | 42.9 | 47.4 |
| ✓ | ✗ | ✗ | 42.8 | 42.1 | 48.1 |
| ✗ | ✗ | ✓ | 44.5 | 43.4 | 49.6 |
| ✓ | ✓ | ✗ | 44.7 | 42.2 | 50.0 |
| ✓ | ✓ | ✓ | 44.7 | 44.2 | 50.6 |

### Key Findings

1. CCF consistently improves performance across all three target domains: Rain +2.8, Night +1.3, Boston +3.2 mAP, while preserving source domain performance (68.2 vs. 68.4).
2. Complementary masking (CM) is the most effective individual component; CM alone yields gains of 2.6/0.5/2.2 on Rain/Night/Boston, respectively.
3. Complementary GridMask significantly outperforms consistent GridMask (Rain: 44.3 vs. 42.8), validating the importance of the complementary design.
4. Curriculum learning improves training stability: with vs. without curriculum learning shows a NDS gap of 56.9 vs. 55.1 on Boston.

## Highlights & Insights

- The pilot study is thorough, systematically demonstrating the existence of modality imbalance from three perspectives: 2D AP, matching ratio, and depth error.
- The "three parallel decoding" design in QDL elegantly avoids shortcut learning with no additional inference overhead.
- The complementary masking design is inspired by the asynchronous modality degradation patterns observed in the real world, endowing it with strong physical intuition.

## Limitations & Future Work

- Experiments are conducted only on nuScenes; generalization to larger-scale datasets such as Waymo remains unverified.
- The GridMask pattern in complementary masking is fixed; learning adaptive masking patterns could be explored.
- The 2D proposal generator (Faster R-CNN) is relatively dated; replacing it with a stronger 2D detector may further unlock potential.
- Temporal information is not considered; multi-frame fusion could potentially further improve cross-domain robustness.

## Related Work & Insights

- Unlike missing-modality methods such as MetaBEV and UniBEV, CCF targets scenarios where "modalities are available but differ in reliability."
- The complementary masking concept is generalizable to other multi-modal tasks (e.g., complementary text-image augmentation in VLMs).
- Product-of-Experts-style depth fusion represents an elegant paradigm for cross-modal information integration.

## Rating

- Novelty: ⭐⭐⭐⭐ — Clear problem formulation with a systematic solution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Pilot study + main experiments + multi-dimensional ablations + oracle upper bound
- Writing Quality: ⭐⭐⭐⭐⭐ — Highly coherent logic, seamlessly connecting problem discovery to solution
- Value: ⭐⭐⭐⭐ — Significant practical relevance for domain generalization in autonomous driving multi-modal detection

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](../../ICCV2025/autonomous_driving/evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi_modal_learning_in_3d_human_pose_estimation.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2026\] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection](look_before_you_fuse_2d-guided_cross-modal_alignment_for_robust_3d_detection.md)
- [\[CVPR 2026\] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)

<!-- RELATED:END -->
