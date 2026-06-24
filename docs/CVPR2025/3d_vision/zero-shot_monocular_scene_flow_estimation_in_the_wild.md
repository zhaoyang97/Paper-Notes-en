---
title: >-
  [Paper Note] Zero-Shot Monocular Scene Flow Estimation in the Wild
description: >-
  [CVPR 2025][3D Vision][Scene Flow] Proposes the first monocular scene flow estimation method capable of zero-shot generalization in the wild. By jointly predicting geometry and motion, constructing a diverse training dataset of over one million samples, and employing a pointmap + 3D offset parameterization, it comprehensively outperforms existing methods in 3D endpoint error.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Scene Flow"
  - "Zero-Shot Generalization"
  - "Monocular Estimation"
  - "Pointmaps"
  - "Large-Scale Training Data"
date: 2026-05-08
content_hash: 45172c044a438769
---

# Zero-Shot Monocular Scene Flow Estimation in the Wild

**Conference**: CVPR 2025  
**arXiv**: [2501.10357](https://arxiv.org/abs/2501.10357)  
**Code**: [Project Page](https://research.nvidia.com/labs/lpr/zero_msf/)  
**Area**: 3D Vision / Scene Flow  
**Keywords**: Scene Flow, Zero-Shot Generalization, Monocular Estimation, Pointmaps, Large-Scale Training Data

## TL;DR

Proposes the first monocular scene flow estimation method capable of zero-shot generalization in the wild. By jointly predicting geometry and motion, constructing a diverse training dataset of over one million samples, and employing a pointmap + 3D offset parameterization, it comprehensively outperforms existing methods in 3D endpoint error.

## Background & Motivation

Scene flow captures the geometric transformations of dynamic scenes through the motion of 3D points, holding broad application potential in AR, autonomous driving, and robotics, but is rarely used in practice due to:
- **Poor Generalization of Existing Methods**: Most methods focus solely on autonomous driving scenarios, with performance dropping significantly on out-of-distribution data.
- **Entanglement of Geometry and Motion**: 2D displacement is a combined effect of depth and motion; estimating them separately is inherently inaccurate.
- **Scarcity of Labeled Data**: Scene flow is harder to measure than depth and optical flow, and different datasets exhibit scale inconsistencies (some are metric, others are relative scales).
- **Sensitivity to Parameterization**: Parameterizations using "depth + optical flow" or "3D endpoint differences" degrade the quality of the results.
- Large models such as DUSt3R/MASt3R perform exceptionally well in static scene geometry estimation but cannot handle motion.
- A unified approach is required to simultaneously handle geometry estimation and motion estimation with in-the-wild generalization capabilities.

## Method

### Overall Architecture

Based on the ViT architecture of CroCoV2, a dual-branch weight-sharing encoder and cross-attention decoder are used. Three DPT prediction heads respectively output the pointmaps of two frames $\hat{X_1}, \hat{X_2} \in \mathbb{R}^{H \times W \times 3}$ (in the $C_1$ coordinate system) and the scene flow offset $\hat{S} \in \mathbb{R}^{H \times W \times 3}$. The geometry estimation component is initialized with pre-trained weights from DUSt3R/MASt3R.

### Key Designs

#### 1. Joint Geometry-Motion Prediction

**Function**: Simultaneously predicts scene geometry (pointmaps) and 3D motion (scene flow) in a unified network, resolving the entanglement of geometry and motion.

**Mechanism**: Utilizing an information-sharing ViT backbone, the encoder possesses a dual-branch weight-sharing structure $\text{Enc}_v$, and the decoder $\text{Dec}_v$ performs inter-frame information exchange via cross-attention. Three dedicated prediction heads $\text{H}_{X_1}, \text{H}_{X_2}, \text{H}_S$ respectively predict two-frame pointmaps and the scene flow offset. Due to the shared internal representation, the 3D geometric prior assists motion estimation, while the temporal correspondences learned by the motion estimation head in turn improve the geometry prediction.

**Design Motivation**: The entanglement of geometry and motion means they must be reasoned about jointly. Experiments demonstrate that joint training not only improves scene flow accuracy but also enhances depth estimation in dynamic scenes.

#### 2. Million-scale Multi-domain Training Data Recipe

**Function**: Constructs a training dataset of over one million samples by integrating 6 synthetic datasets to cover diverse scenarios.

**Mechanism**: Integrates 6 datasets including SHIFT (driving), Dynamic Replica (indoor), Virtual KITTI 2 (driving), MOVi-F (Kubric), PointOdyssey (indoor+Kubric), and Spring (animation). Different datasets have different annotation types (optical flow, scene flow, depth), and for data with only depth annotations, the projection of scene flow in image space is supervised by optical flow. The key innovation is cross-dataset **scale-adaptive optimization**: metric and relative scale datasets are processed separately, and a scale-alignment mechanism is used for unified training.

**Design Motivation**: The scarcity of scene flow annotations is a bottleneck in the field. By combining multiple datasets, annotation types, and scale alignment, data diversity is maximized without sacrificing quality.

#### 3. Pointmap + 3D Offset Parameterization

**Function**: Selects the most suitable scene flow representation for learning, avoiding quality degradation caused by the choice of parameterization.

**Mechanism**: Systematically compares three parameterizations: (1) depth + optical flow (traditional decomposition); (2) the difference between pointmaps of two frames ($X_2 - X_1$); (3) pointmap + independent 3D offset (ours). Scheme (3) allows the scene flow head to directly learn the 3D offset $S = X_{1,t_2} - X_1$ instead of relying on the difference between two pointmaps, thereby avoiding error accumulation. The three output maps $\hat{X_1}, \hat{X_2}, \hat{S}$ respectively encode the geometry and motion transformations of $(C_1,t_1)$ and $(C_1,t_2)$.

**Design Motivation**: Experiments reveal that both schemes (1) and (2) significantly degrade result quality. Independent offset prediction prevents geometric errors from directly propagating to scene flow estimation.

### Loss & Training

Regression loss combination: pointmap regression loss (L1/L2) + optical flow projection supervision loss + confidence weighting + scale-adaptive alignment factor. Different datasets utilize different combinations of supervision signals depending on their annotation types.

## Key Experimental Results

### Main Results: Cross-Dataset Zero-Shot Generalization

| Method | KITTI SF EPE3D ↓ | Spring SF EPE3D ↓ | DAVIS Generalization | RoboTAP Generalization |
|------|-----------------|-------------------|-----------|-------------|
| **Ours** | **Best** | **Best** | ✓ Zero-shot | ✓ Zero-shot |
| Self-Mono-SF | Second Best | Poor | ✗ | ✗ |
| Depth+Flow baseline | Poorer | Poorer | Partial | Partial |
| MonST3R (concurrent) | No motion | No motion | Partial | Partial |

### Ablation Study: Parameterization Selection

| Parameterization | SF EPE3D ↓ | Depth Accuracy |
|-----------|-----------|---------|
| Pointmap + Independent 3D Offset (Ours) | **Best** | **Best** |
| Depth + Optical Flow | Poorer | Poorer |
| Difference of Two Pointmaps | Poorer | Moderate |

### Key Findings

- Jointly training motion estimation **also improves depth estimation of dynamic scenes**, further confirming the entanglement of geometry and motion.
- Cross-domain diversity in the million-scale data recipe is key to zero-shot generalization.
- Shows strong zero-shot generalization on unseen DAVIS (everyday videos) and RoboTAP (robotic manipulation) data.
- The choice of parameterization has a significant impact on final quality, where independent offset prediction outperforms difference-based or decomposed prediction.

## Highlights & Insights

- **First Zero-Shot Scene Flow Model in the Wild**: Fills the gap in large-scale generalization for scene flow estimation.
- **Bidirectional Benefits of Joint Training**: The observation that motion estimation helps depth estimation is valuable.
- **Practical Data Engineering**: The scale-alignment mechanism enables mixing metric and relative datasets.

## Limitations & Future Work

- Training data is entirely synthetic; the synthetic-to-real domain gap still exists.
- Only handles scene flow between two frames, without scaling up to multi-frame or video-level.
- Downstream applications of scene flow (e.g., AR rendering, collision detection) remain insufficiently validated.
- Performance can be further enhanced in the future by incorporating ongoing improvements of DUSt3R/MASt3R.

## Related Work & Insights

- Builds upon the success of DUSt3R/MASt3R, demonstrating the effective extension of large-scale geometric pre-trained models to dynamic scenes.
- Compared to MonST3R, this method explicitly models the motion field instead of just enhancing robustness.
- The ideas from the data recipe can be applied to other low-level vision tasks with scarce data.

## Rating

⭐⭐⭐⭐ — Systematically addresses the three major challenges of monocular scene flow estimation (joint prediction, data scarcity, and parameterization), achieving zero-shot generalization in the wild for the first time. The method design and data engineering are solid, significantly driving the scene flow field forward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2025\] Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)
- [\[CVPR 2025\] SCFlow2: Plug-and-Play Object Pose Refiner with Shape-Constraint Scene Flow](scflow2_plug-and-play_object_pose_refiner_with_shape-constraint_scene_flow.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
