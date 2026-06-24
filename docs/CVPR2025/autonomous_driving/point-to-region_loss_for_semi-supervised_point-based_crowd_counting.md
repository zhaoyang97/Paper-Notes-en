---
title: >-
  [Paper Note] Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting
description: >-
  [CVPR 2025][Autonomous Driving][Semi-supervised crowd counting] This paper identifies that point-to-point (P2P) matching in semi-supervised crowd counting leads to model over-activation on unlabeled data (visualized via PSAM gradient diagnosis). To address this, the authors propose point-to-region (P2R) matching, which expands each GT/pseudo-labeled point into a local region and propagates confidence. On ShanghaiTech-A with 5% labeled data, it achieves an MAE of 69.9 (vs. pre…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Semi-supervised crowd counting"
  - "point-to-region matching"
  - "pseudo-label"
  - "gradient diagnosis"
  - "P2R loss"
date: 2026-05-08
content_hash: 7793e3fb485f2cc9
---

# Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting

**Conference**: CVPR 2025  
**arXiv**: [2505.21943](https://arxiv.org/abs/2505.21943)  
**Code**: [https://github.com/Elin24/P2RLoss](https://github.com/Elin24/P2RLoss)  
**Area**: Autonomous Driving / Crowd Counting  
**Keywords**: Semi-supervised crowd counting, point-to-region matching, pseudo-label, gradient diagnosis, P2R loss

## TL;DR

This paper identifies that point-to-point (P2P) matching in semi-supervised crowd counting leads to model over-activation on unlabeled data (visualized via PSAM gradient diagnosis). To address this, the authors propose point-to-region (P2R) matching, which expands each GT/pseudo-labeled point into a local region and propagates confidence. On ShanghaiTech-A with 5% labeled data, it achieves an MAE of 69.9 (vs. prev. SOTA of 83.7) while running 68 times faster than P2P.

## Background & Motivation

### Background

**Background**: Point-based crowd counting annotates each head as a point and conducts counting by learning point classification maps. Semi-supervised methods use a teacher model to generate pseudo-labels for training the student model.

**Limitations of Prior Work**: P2P matching (the Hungarian algorithm matching predicted points to pseudo-labeled points one-to-one) fails in semi-supervised settings. Background points in pseudo-labels lack negative sample supervision (due to uncertainty of true background locations), causing the model to output high responses at all locations and leading to an explosion in counting.

**Key Challenge**: P2P only supervises foreground points (matched ones) and does not supervise the background. The student model receives no "no person here" signal on unlabeled data, leading to a "rather over-detect than miss" behavior.

**Key Insight**: (1) PSAM (Point-Specific Activation Map, a Jacobian-based gradient visualization) is leveraged to diagnose the cause of failure, confirming the vanishing background gradient; (2) points are expanded to regions to involve all pixels inside the region in supervision, allowing background pixels to receive confidence-weighted negative sample signals.

**Core Idea**: Point-to-region expansion + intra-region confidence propagation = restoration of background gradients, solving the over-activation of semi-supervised P2P.

### Mechanism

**Goal**: ### Key Designs

1. **PSAM Gradient Diagnosis**: $H[q] = \max(\sum_k \nabla p[q] \odot F, 0)$—representing Jacobian visualization of predicted points.


## Method

### Key Designs

1. **PSAM Gradient Diagnosis**: $H[q] = \max(\sum_k \nabla p[q] \odot F, 0)$—representing Jacobian visualization of predicted points. After unlabeled training with P2P, PSAM values are large and diffused (the model perceives human heads everywhere), whereas the supervised model has PSAM focused tightly on human heads.

2. **P2R Matching**: A region is formed within a radius of μ around each labeled point. Matching changes from point-to-point to region-to-region, where all pixels within the region receive confidence weights based on their distance to the center. This is 68 times faster than P2P (requiring no Hungarian algorithm).

3. **Pseudo-label Confidence Propagation**: $Z = \text{diag}[M_{st}\zeta + (1_n - \beta)]$—confidence of pseudo-labels propagates from points to the entire region, enabling strong supervision for high-confidence regions and weak supervision for low-confidence regions.

### Loss & Training

Labeled: P2P BCE. Unlabeled: P2R BCE + confidence weighting. Region radius μ ~ 10-15 pixels.

## Key Experimental Results

| Dataset (5% Labeled) | P2R | OT-M (Prev. SOTA) | Labeled Only |
|-----------------|-----|-------------|--------|
| SHA MAE↓ | **69.9%** | 83.7% | 93.7% |
| SHB MAE↓ | **9.1%** | 12.6% | — |
| QNRF MAE↓ | **100.1** | 118.4 | — |
| P2R Inference Time | **0.0064s** | — | P2P: 0.4307s |

### Ablation Study
- PSAM diagnosis clearly demonstrates the background over-activation issue of P2P.
- P2R restores the gradient flow of background pixels, allowing the model to relearn "where there is no person".
- The region radius μ controls the intensity of background suppression (10-15 pixels is empirically optimal).

### Key Findings
- **P2P fundamentally fails in semi-supervised training**—this is not due to poor pseudo-label quality, but a structural flaw in the loss function.
- **68× Acceleration**—P2R eliminates the need for Hungarian matching, performing direct regional assignment.
- **PSAM is a generic diagnostic tool**—applicable to failure analysis in any point-detection task.

## Highlights & Insights
- **Complete logical chain of PSAM diagnosis $\rightarrow$ P2R solution**—understanding the root cause of failure before designing a targeted solution.
- **Elegant extension from point to region**—retains the cost-efficiency of point annotations while achieving dense supervision effects via region expansion.

## Limitations & Future Work
- No offset regression (classification only).
- The region radius μ needs to be manually set.
- Assumes local regions are approximately circular.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of PSAM diagnosis and the P2R solution is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets + computational efficiency comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear causal reasoning chain.
- Value: ⭐⭐⭐⭐ Addresses a fundamental flaw in semi-supervised crowd counting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](unlocking_generalization_power_in_lidar_point_cloud_registration.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[CVPR 2025\] SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization](superpc_a_single_diffusion_model_for_point_cloud_completion_upsampling_denoising.md)

</div>

<!-- RELATED:END -->
