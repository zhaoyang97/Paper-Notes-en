---
title: >-
  [Paper Note] SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens
description: >-
  [CVPR 2025][3D Vision][Multi-Person 3D Mesh Estimation] This paper proposes SAT-HMR, a DETR-based real-time multi-person 3D human mesh estimation framework. By introducing scale-adaptive tokens—utilizing high-resolution tokens for small-scale humans, low-resolution tokens for large-scale humans, and pooled/compressed tokens for the background—it improves inference speed to 24 FPS while maintaining the accuracy of high-resolution inputs, achieving an optimal balance between pr…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-Person 3D Mesh Estimation"
  - "Scale-Adaptive"
  - "Efficient ViT"
  - "DETR"
  - "Real-Time Inference"
date: 2026-05-08
content_hash: caed68afc7ef3f2d
---

# SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens

**Conference**: CVPR 2025  
**arXiv**: [2411.19824](https://arxiv.org/abs/2411.19824)  
**Code**: [Project Page](https://ChiSu001.github.io/SAT-HMR/)  
**Area**: 3D Vision / Human Body Understanding  
**Keywords**: Multi-Person 3D Mesh Estimation, Scale-Adaptive, Efficient ViT, DETR, Real-Time Inference

## TL;DR

This paper proposes SAT-HMR, a DETR-based real-time multi-person 3D human mesh estimation framework. By introducing scale-adaptive tokens—utilizing high-resolution tokens for small-scale humans, low-resolution tokens for large-scale humans, and pooled/compressed tokens for the background—it improves inference speed to 24 FPS while maintaining the accuracy of high-resolution inputs, achieving an optimal balance between precision and speed.

## Background & Motivation

- **Challenges in Multi-Person 3D Mesh Estimation**: Estimating SMPL parameters for all individuals from a single RGB image requires both local details (joint poses) and global context (relative positions, occlusion relations).
- **Multi-Stage vs. One-Stage**: Multi-stage methods perform detection first followed by person-by-person cropping and estimation, which achieves high accuracy but loses global context and struggles with occlusions. One-stage methods (e.g., ROMP, BEV) process the entire image based on CNNs, but low-resolution inputs limit their representation capability.
- **The Cost of High Resolution**: Emerging DETR-style methods (e.g., AiOS, Multi-HMR) achieve SOTA performance by using high-resolution inputs (1288 resolution), but their inference speed is only around 5 FPS. Key observation: **High resolution primarily benefits small-scale people (long distance/children/crouching poses)**—the error drops by 35mm in the 0-10% scale range, while improvements in the 30%+ scale range are negligible.
- **Core Idea**: Allocating high-resolution tokens to large-scale persons (close to the camera, occupying a large image area) is computationally wasteful because they are already represented by a sufficient number of tokens. High-resolution computational resources should be concentrated on small-scale persons who truly need them.
- **Compressible Background Regions**: Background tokens provide useful contextual information (and should not be completely discarded) but can be further compressed via spatial pooling.

## Method

### Overall Architecture

SAT-HMR adopts a DETR-style pipeline: (1) extracting tokens from low-resolution images and encoding them with a shallow Transformer encoder $\rightarrow$ (2) predicting a patch-level scale map $\mathbf{S}$ using a scale head $\rightarrow$ (3) classifying tokens into background, small-scale, and large-scale based on the scale map $\rightarrow$ (4) replacing small-scale tokens with their high-resolution counterparts, pooling and compressing background tokens, and keeping large-scale tokens unchanged $\rightarrow$ (5) concatenating these tokens to obtain the scale-adaptive tokens $\mathcal{T}_{\text{SA}}$ $\rightarrow$ (6) feeding them into a subsequent Transformer encoder, decoder, and prediction head to regress SMPL parameters.

### Key Designs

**Design 1: Patch-Level Scale Map Prediction**
- **Function**: Determines whether each patch covers a person and predicts the relative scale of that person.
- **Mechanism**: The scale map $\mathbf{S}(i,j) = (c, s)$ contains two values: $c$ is the person confidence (0 indicates background), and $s = \min(d_{\text{bb}} / S_{\text{hr}}, 1)$ represents the person scale (the ratio of the bounding box diagonal to the maximum edge of the image). When a patch overlaps with multiple people, the scale of the closest person is taken. It is predicted from low-resolution tokens via a scale head consisting of $N_{\text{lr}}$ Transformer layers and an MLP.
- **Design Motivation**: The scale definition directly reflects the proportion of a person in the image, which is strongly correlated with whether that region requires high-resolution tokens. Using a lightweight prediction head introduces minimal computational overhead.

**Design 2: Scale-Adaptive Token Selection and Replacement**
- **Function**: Dynamically adjusts the token resolution of each region.
- **Mechanism**: The tokens are divided into three groups based on thresholds $\alpha_c$ (confidence) and $\alpha_s$ (scale): (1) Background $\mathcal{T}_B$: every 4 adjacent tokens are pooled into 1 to obtain $\mathcal{T}_B'$; (2) Small-scale $\mathcal{T}_{\text{SMALL}}$: pruned and replaced by corresponding tokens from the high-resolution image, where $k_{\text{hr}} = 4 k_{\text{small}}$; (3) Large-scale $\mathcal{T}_{\text{LARGE}}$: kept at low resolution. Finally, they are concatenated as $\mathcal{T}_{\text{SA}} = \{\mathcal{T}_B', \mathcal{T}_{\text{LARGE}}, \mathcal{T}_{\text{HR}}\}$.
- **Design Motivation**: Small-scale persons lack adequate features at low resolutions (which dominates the gains brought by high resolution), and replacing them with 4x resolution tokens compensates for this deficiency; large-scale persons are already covered by enough tokens; background is kept and pooled rather than discarded to preserve contextual information.

**Design 3: Dual-Resolution Encoder Alignment**
- **Function**: Ensures that low-resolution and high-resolution tokens can be concatenated within the same feature space.
- **Mechanism**: The low-resolution and high-resolution branches are processed by their respective shallow Transformer encoders ($N_{\text{lr}} = N_{\text{hr}} = 3$ layers), which share the same pretrained DINOv2 weights. After independent encoding in both branches, feature space alignment is performed at the concatenation point, followed by processing through a unified Transformer encoder with $N_{\text{sa}} = 9$ layers.
- **Design Motivation**: Having the same number of layers ensures consistent feature abstraction levels, facilitating seamless processing of mixed-resolution tokens by the subsequent unified encoder.

### Loss & Training

The total loss is a weighted sum of multiple loss terms: $\mathcal{L} = \lambda_{\text{map}} \mathcal{L}_{\text{map}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{pose}} \mathcal{L}_{\text{pose}} + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}} + \lambda_{\text{j3d}} \mathcal{L}_{\text{j3d}} + \lambda_{\text{j2d}} \mathcal{L}_{\text{j2d}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{det}} \mathcal{L}_{\text{det}}$. Specifically, $\mathcal{L}_{\text{map}}$ consists of a focal loss and an L1 loss for the scale map, $\mathcal{L}_{\text{depth}}$ is the normalized depth L1 loss, $\mathcal{L}_{\text{det}}$ represents the detection focal loss, and the rest employ L1 distance.

## Key Experimental Results

### Main Results: AGORA Test Set

| Method | Resolution | Time (ms) | MACs (G) | F1 ↑ | MPJPE ↓ | MVE ↓ |
|------|-------|----------|---------|------|--------|------|
| ROMP | 512 | 38.7 | 43.6 | 0.91 | 108.1 | 103.4 |
| BEV | 512 | 50.6 | 48.9 | 0.93 | 105.3 | 100.7 |
| AiOS | 1333 | 405.2 | 314.5 | 0.94 | 63.9 | 57.5 |
| Multi-HMR | 1288 | 231.7 | 6104.6 | 0.95 | 65.3 | 61.1 |
| **SAT-HMR** | **644*** | **42.0** | **133.1** | **0.95** | **67.9** | **63.3** |

### Generalization on Other Datasets

| Method | 3DPW PA-MPJPE ↓ | MuPoTS PCK All ↑ | CMU Panoptic Avg ↓ |
|------|----------------|------------------|-------------------|
| Multi-HMR | 41.7 | 85.0 | - |
| BEV | 46.9 | 70.2 | 109.5 |
| **SAT-HMR** | **41.6** | **89.0** | **84.2** |

### Ablation Study: Background Token Strategy

| Strategy | 0-20% MVE | 80%+ MVE | Avg MVE |
|------|----------|---------|---------|
| Discard All | 59.9 | 70.8 | 57.2 |
| No Pooling | 60.3 | 64.1 | 56.1 |
| Pooling ×2 | 60.7 | 66.5 | 56.3 |
| **Pooling ×1 (Ours)** | **60.0** | **62.7** | **56.0** |

### Key Findings

1. SAT-HMR achieves comparable accuracy to Multi-HMR (4 FPS) at 24 FPS (42ms), rendering a speedup of approximately 5.5x.
2. MACs are reduced from 6104.6G (Multi-HMR) to 133.1G, a decrease of 97.8%.
3. Completely discarding background tokens increases the error for large-scale persons from 62.7 to 70.8, demonstrating the importance of background context.
4. On the CMU Panoptic dataset, the error drops from 109.5 (BEV) to 84.2 (a 23.1% improvement), demonstrating outstanding generalization ability.

## Highlights & Insights

- **Precise Insights**: The empirical observation that "high resolution primarily benefits small-scale persons" directly guides the methodology design.
- **Intelligent Allocation of Computational Resources**: Precious high-resolution tokens are concentrated in the areas where they are most needed, while background tokens are compressed instead of discarded.
- **First Real-Time SOTA**: Achieves real-time (24 FPS) SOTA performance on the AGORA leaderboard, offering extremely high practical value.
- The method design is simple and elegant; instead of introducing complex, new modules, it achieves its effectiveness through token-level reallocation.

## Limitations & Future Work

- The definition of scale does not consider human height information (e.g., the bounding boxes of a crouching adult and a standing child can be similar), which may lead to depth estimation biases.
- Currently, it only estimates the SMPL body mesh, without extending to the SMPL-X whole-body mesh (including hands and face).
- The scale threshold $\alpha_s$ is a fixed hyperparameter, which may require adaptive adjustment under highly varying scenes.
- It requires processing both high-resolution and low-resolution images, leaving room for further optimization in memory consumption.

## Related Work & Insights

- Compared to efficient ViT methods such as FlexiViT/TORE, the token reallocation strategy in SAT-HMR is more task-adaptive.
- The concept of scale-adaptation can be transferred to other dense prediction tasks (e.g., panoptic segmentation, depth estimation).
- The design of pooling the background rather than discarding it serves as a valuable reference for detection and estimation tasks requiring global context.

## Rating

⭐⭐⭐⭐ — Accurate problem insights, simple method design, and thorough experimentation achieve outstanding real-time SOTA results. It provides a highly valuable reference for applying efficient Vision Transformers to human understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](../../ECCV2024/3d_vision/multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[CVPR 2025\] MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)

</div>

<!-- RELATED:END -->
