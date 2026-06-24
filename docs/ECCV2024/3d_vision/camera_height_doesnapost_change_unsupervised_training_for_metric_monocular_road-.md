---
title: >-
  [Paper Note] Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation
description: >-
  [ECCV 2024][3D Vision][Monocular depth estimation] Proposes the FUMET training framework, which leverages vehicle size priors detected on the road to aggregate camera height estimates and utilizes the invariance of camera height within the same video sequence as metric scale supervision, enabling any monocular depth network to learn absolute scale without auxiliary sensors.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Monocular depth estimation"
  - "metric scale"
  - "self-supervised learning"
  - "camera height invariance"
  - "vehicle size prior"
date: 2026-05-08
content_hash: a7ef26b3bab88b69
---

# Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation

**Conference**: ECCV 2024  
**arXiv**: [2312.04530](https://arxiv.org/abs/2312.04530)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Monocular depth estimation, metric scale, self-supervised learning, camera height invariance, vehicle size prior

## TL;DR

Proposes the FUMET training framework, which leverages vehicle size priors detected on the road to aggregate camera height estimates and utilizes the invariance of camera height within the same video sequence as metric scale supervision, enabling any monocular depth network to learn absolute scale without auxiliary sensors.

## Background & Motivation

Monocular depth estimation (MDE) is crucial for autonomous driving. Self-supervised methods reduce reliance on GT depth but suffer from the **scale ambiguity** problem. Existing solutions require auxiliary sensors: velocity (GPS), IMU/gravity, or camera height (manually annotated), making it impossible to utilize the vast amount of driving videos available on the internet.

**Key Insight**: Vehicles on the road are rigid objects; their actual sizes remain constant and are unique to each make and model. FUMET aggregates vehicle size clues across different frames into the **camera height** and leverages the fact that it remains invariant within the same sequence as a supervision signal.

## Method

### Overall Architecture

On top of standard self-supervised MDE training, FUMET introduces metric scale learning, consisting of a depth/pose network, camera height estimation, a Silhouette Projector, and a Learned Size Prior (LSP).

### Key Designs

#### 1. Scale-aware Self-supervised Learning

For road-region pixels, the pixel-wise camera height is obtained by calculating normal vectors from the depth map, with the median value taken as the frame-level estimate. The scale factor is obtained via the Silhouette Projector to yield the scaled camera height. Across epochs, pseudo-labels are optimized using a weighted moving average to make the supervision signal increasingly accurate.

#### 2. Silhouette Projector

Robustly estimates the scale factor by leveraging two facts: the height of an object's silhouette projected onto a plane perpendicular to the ground does not change with pose; and it can be calculated even under partial occlusion as long as the top is visible. Pipeline: Depth reconstruction to point cloud $\to$ Orthographic projection $\to$ Silhouette height $\to$ Comparison with LSP to obtain the scale factor. An outlier filtering threshold of $T=0.2$ is used.

#### 3. Learned Size Prior (LSP)

Predicts 3D dimensions (height + width + length) from vehicle mask images. Training data is crawled from the web without manual annotation. Rich data augmentation is used to simulate occlusions and truncations. Predicting width/length helps improve the accuracy of height estimation.

### Loss & Training

Total loss = Reconstruction loss (SSIM+L1) + Smoothness loss + Camera height loss + Auxiliary coarse geometry loss.

Key strategy: Logarithmic dynamic weight scheduling—the auxiliary loss weight decreases from 1, while the camera height loss weight increases from 0, becoming fixed after the mid-epoch. This is because depth is unreliable in the early stages of training, making over-reliance on camera height loss unstable; in the later stages of training, the imprecise planar assumption of the auxiliary loss degrades accuracy. $\alpha=0.01$, $\beta=1.0$, 50 epochs.

## Key Experimental Results

### Main Results: KITTI Eigen Test Set (640x192)

| Method | Supervision | AbsRel↓ | SqRel↓ | RMSE↓ | δ<1.25↑ |
|------|----------|---------|--------|-------|---------|
| G2S | GPS | 0.109 | 0.860 | 4.855 | 0.865 |
| PackNet-SfM | Velocity | 0.111 | 0.829 | 4.788 | 0.864 |
| VADepth | Camera Height (GT) | 0.120 | 0.975 | 4.971 | 0.867 |
| DynaDepth | IMU+V+G | 0.109 | 0.787 | 4.705 | 0.869 |
| **FUMET** | **None** | **0.108** | **0.785** | **4.736** | **0.871** |
| VADepth+FUMET | None | 0.108 | 0.809 | 4.572 | 0.883 |

### Cityscapes Dataset

| Method | AbsRel↓ | RMSE↓ | δ<1.25↑ |
|------|---------|-------|---------|
| G2S | 4.156 | 58.89 | 0.046 |
| VADepth | 0.363 | 11.95 | 0.295 |
| **FUMET** | **0.125** | **6.359** | **0.858** |

Weakly supervised methods degrade significantly due to reliance on unreliable sensor data, whereas FUMET is robust as it only relies on RGB videos.

### Mixed Dataset Training (Argoverse2+Lyft+A2D2+DDAD)

| Training Data | AbsRel↓ | RMSE↓ | δ<1.25↑ |
|----------|---------|-------|---------|
| KITTI | 0.103 | 4.708 | 0.903 |
| Mixed | 0.113 | 5.009 | 0.883 |
| **Mixed+KITTI** | **0.082** | **4.307** | **0.923** |

### Ablation Study

- Camera height loss contributes more than auxiliary geometry loss.
- Cross-frame height optimization is more stable than using priors independently frame-by-frame.
- Using both losses jointly with dynamic weight scheduling achieves the best results (AbsRel 0.108).
- Offline pre-computed fixed camera height is slightly better than online optimization.
- Offline pre-training + online fine-tuning achieves the highest accuracy.

### Key Findings

1. The simplest Monodepth2 + FUMET outperforms weakly supervised methods that require GT scale labels.
2. FUMET not only learns the metric scale but also improves geometric accuracy (improvements remain even after median scaling).
3. VADepth requires GT camera height but its accuracy is actually lower than FUMET, indicating that accurately measuring camera height itself is difficult.

## Highlights & Insights

1. **Elegant Core Insight**: Camera height invariance aggregates scattered vehicle size clues into a stable supervision signal.
2. **Architecture Agnostic**: Can be plugged into any monocular depth network.
3. **True Unsupervised Metric Depth**: Requires only monocular driving videos and camera intrinsics.
4. **Mixed Dataset Training**: Datasets with different camera heights can be trained jointly under a unified framework.
5. **Zero Inference Overhead**: The computational cost is exactly the same as the original MDE model.

## Limitations & Future Work

1. **Dependency on Vehicle Detection**: May fail in vehicle-free scenes.
2. **LSP Generalizability**: May be inaccurate for uncommon vehicle models.
3. **Limited to Driving Scenes**: Assumes road scenes with vehicles on a ground plane.
4. Can be extended to other objects of known sizes in the future (e.g., pedestrians, traffic signs).

## Related Work & Insights

- The fundamental difference from weakly supervised methods is that auxiliary sensors are not required.
- The weighted moving average optimization strategy can be generalized to other cross-frame consistency tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Camera height invariance as a supervision signal is extremely clever.
- Value: ⭐⭐⭐⭐⭐ — Truly achieves metric depth without auxiliary sensors.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated across multiple datasets and architectures with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with well-motivated components.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[ECCV 2024\] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](../../CVPR2025/3d_vision/depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)

</div>

<!-- RELATED:END -->
