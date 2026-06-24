---
title: >-
  [Paper Note] MonoSOWA: Scalable Monocular 3D Object Detector Without Human Annotations
description: >-
  [ICCV2025][Autonomous Driving][Monocular 3D Detection] This paper proposes the first monocular 3D object detection method that requires no human annotations of any kind (neither 2D nor 3D). A novel Local Object Motion Model (LOMM) is introduced to disentangle inter-frame motion sources, enabling auto-labeling at a speed ~700× faster than prior work. A Canonical Object Space (COS) is further proposed to enable multi-dataset training across heterogeneous camera configurations.
tags:
  - "ICCV2025"
  - "Autonomous Driving"
  - "Monocular 3D Detection"
  - "Weak Supervision"
  - "Auto-Labeling"
  - "Pseudo-LiDAR"
  - "Annotation-Free"
date: 2026-05-08
content_hash: a2cb79e6184dc8b9
---

# MonoSOWA: Scalable Monocular 3D Object Detector Without Human Annotations

**Conference**: ICCV2025
**arXiv**: [2501.09481](https://arxiv.org/abs/2501.09481)  
**Code**: [github.com/jskvrna/MonoSOWA](https://github.com/jskvrna/MonoSOWA)  
**Area**: Autonomous Driving
**Keywords**: Monocular 3D Detection, Weak Supervision, Auto-Labeling, Pseudo-LiDAR, Annotation-Free

## TL;DR

This paper proposes the first monocular 3D object detection method that requires no human annotations of any kind (neither 2D nor 3D). A novel Local Object Motion Model (LOMM) is introduced to disentangle inter-frame motion sources, enabling auto-labeling at a speed ~700× faster than prior work. A Canonical Object Space (COS) is further proposed to enable multi-dataset training across heterogeneous camera configurations.

## Background & Motivation

- Monocular 3D object detection is a critical component in autonomous driving; conventional methods depend on LiDAR sensors and extensive manual 3D annotations.
- The annotation process is extremely time-consuming and expensive, limiting training data diversity; any change in camera setup necessitates re-collection and re-annotation.
- Existing weakly supervised methods still require 2D instance masks (e.g., WeakMono3D, VSRD) or LiDAR data (e.g., WeakM3D, Autolabels).
- VSRD requires ~15 minutes per frame for annotation; annotating Waymo's 158K frames would take approximately 4 years, rendering it practically unscalable.
- Both WeakMono3D and VSRD fail to handle moving objects—they either discard them or assign low confidence, wasting a large amount of training signal.
- **Goal**: Completely eliminate dependence on human annotations and LiDAR, so that data from commodity vehicles equipped only with monocular cameras can be directly exploited.

## Method

### Overall Architecture

The method requires only: **image sequences + ego-motion data (GPS/IMU) + known camera intrinsics and extrinsics**. No human annotations are needed, not even 2D labels. Two off-the-shelf pretrained models are used:
1. **2D Object Detector**: MViT2-Huge (Detectron2 framework, COCO pretrained) — provides instance segmentation masks.
2. **Monocular Depth Estimator**: Metric3D v2 — selected for its best zero-shot metric depth generalization.

The auto-labeling pipeline decomposes the 7-DOF 3D bounding box estimation problem into three sequential sub-problems: orientation, position, and dimensions, avoiding the difficulty of direct joint optimization.

### Step 1: Pseudo-LiDAR Aggregation

- Metric depth maps are inferred per frame and back-projected into 3D point clouds.
- Instance segmentation masks from the 2D detector are used to extract per-object point clouds $P_{i,j}$.
- Object tracking is performed in 3D world coordinates:
    - The median of the object point cloud approximates spatial position.
    - A physics-based motion model predicts the position in the next frame.
    - Nearest-neighbor matching with a distance threshold is applied.
- Multi-frame point clouds are transformed into a reference frame coordinate system using ego-motion data.

### Step 2: Local Object Motion Model (LOMM)

This is the core contribution of the paper, addressing a key challenge: since the ego-vehicle is also moving, all objects undergo relative displacement between frames. The question is how to identify which objects are truly in motion.

- **Core Idea**: Decouple inter-frame object position changes into two sources — **ego-vehicle motion** and **object self-motion**.
- **Classification**: For each object instance, compute the frame-wise position difference sequence $\Delta_{i,j} = L_{i,j} - L_{i-1,j}$:
    - Mean $\mu_j = \frac{1}{l-k} \sum \Delta_{i,j}$, standard deviation $\sigma_j$ (normalized by $\sqrt{2}$).
    - Ratio $z_j = \|\mu_j\|_2 / \|\sigma_j\|_2$.
    - Stationary objects: displacement variation stems from pseudo-LiDAR noise jitter (akin to a random walk), so $\|\mu\|_2 \ll \|\sigma\|_2$.
    - Moving objects: persistent directional motion yields $\|\mu\|_2 \gg \|\sigma\|_2$.
    - Decision criterion: $z > T_z = 0.2$ and net displacement $> T_m = 5\text{m}$.
- **For stationary objects**: All frames' point clouds are directly aggregated as $A_j = \{P_{i,j}\}$ (after transformation to the reference frame), yielding a denser representation with recovery from occlusion.
- **For moving objects**: The known trajectory is leveraged to directly compute orientation from physical constraints (vehicles travel in the direction of motion).
- In contrast to prior methods (WeakMono3D, VSRD) that discard or downweight moving objects, LOMM **exploits information from both stationary and moving objects simultaneously**.

### Step 3: Sequential Auto-Labeling

The 7-DOF pose estimation is decomposed into three independent sub-problems:

**Orientation Estimation**:
- Moving objects: orientation is computed directly from consecutive frame positions as $\theta = \text{atan2}(\Delta z / \Delta x)$; the median yaw over the preceding and following 5 frames is used for robustness.
- Stationary objects: all angles $\theta \in [0, \pi/2)$ are iterated in BEV, and the optimal angle is selected using the proposed **Saturated Closeness Criterion (SCC)**:
    - Improves upon Zhang's Closeness Criterion: (1) applies sigmoid saturation $\sigma(\alpha \cdot x)$ ($\alpha=10$) to suppress outlier influence; (2) replaces min/max with the 10th/90th percentiles as boundary reference points.
    - The criterion accumulates, for each point, the minimum distance to the nearest boundary across two perpendicular axes.
    - Since the algorithm cannot distinguish front from back, two orientation hypotheses (differing by $\pi$) are produced.

**Dimension Estimation**:
- Dimensions are obtained from the SCC algorithm output; outliers (outside typical vehicle size ranges) are replaced by prior dimensions.
- View-dependent unobservable dimensions are detected: when the angle between vehicle heading and viewing direction is near $0, \pi/2, \pi, 3\pi/2$, prior dimensions are substituted.

**Position Refinement**:
- Given the estimated orientation and coarse position, small perturbations (≤2m) are applied along the x and z axes.
- A Template Fitting Loss (TFL) using a canonical vehicle template point cloud selects the optimal position; TFL is more robust to outliers than Chamfer Distance.
- The front/back orientation ambiguity is resolved by testing both $\theta$ and $\theta + \pi$ and selecting the hypothesis with lower TFL.

### Step 4: Canonical Object Space (COS)

- **Problem**: Under different camera focal lengths, the same object at the same distance appears at different image sizes; in extreme cases, the network must predict very different distances for objects with identical pixel sizes.
- **Solution**: A canonical focal length $f^C = 750$ is chosen, and only the 3D label coordinates are scaled: $\omega_i = f^C / f_i$, $(x^C, y^C, z^C) = (x \cdot \omega_i, y \cdot \omega_i, z \cdot \omega_i)$.
- During training the model learns in COS; during inference the predictions are inverse-transformed back to world coordinates using the target frame's $\omega_j$.
- Data augmentation (e.g., image scaling) requires synchronized adjustment of the perceived focal length.
- Inspired by Metric3D and Omni3D, but only label coordinates are transformed rather than entire images or depth maps, making the design minimally invasive.
- This allows **a single model to train and infer across heterogeneous camera configurations** without per-camera retraining.

### Implementation Details

- Detector: MonoDETR is used as the final 3D detection model, optimized with AdamW (lr=2e-4, wd=1e-4).
- Aggregation window: 100 frames (up to 50 frames before and after each object instance).
- Thresholds: $T_z = 0.2$ (motion/static classification), $T_m = 5$m (minimum motion distance), $\alpha = 10$ (SCC steepness).
- Identical hyperparameters are used across all three datasets, demonstrating the generalizability of the method.

## Key Experimental Results

### KITTI-360 Test Set

| Method | Human Annotation | AP_BEV@0.5 (Easy/Hard) | AP_3D@0.5 (Easy/Hard) | Labeling Speed |
|--------|-----------------|------------------------|-----------------------|----------------|
| MonoFlex (Fully Supervised) | 3D boxes | 50.82/41.78 | 43.11/34.43 | - |
| MonoDETR (Fully Supervised) | 3D boxes | 47.21/36.05 | 41.01/30.38 | - |
| Autolabels | LiDAR+masks | 20.18/14.33 | 4.69/2.79 | 6s/frame |
| VSRD | Masks | 29.07/22.83 | 21.77/16.46 | 15min/frame |
| **MonoSOWA (No Annotation)** | **None** | **38.41/35.26** | **29.98/27.56** | **1.3s/frame** |

### Waymo Validation Set (Level 2)

| Method | AP_BEV@0.5 All | AP_3D@0.5 All |
|--------|----------------|---------------|
| MonoDETR (Fully Supervised) | 23.63 | 21.41 |
| **MonoSOWA (No Annotation)** | **18.98** | **13.46** |

### Pseudo-Label Pretraining + Fine-tuning with Limited Human Annotations (KITTI)

| Pretraining | Human Annotation Ratio | AP_BEV@0.7 Easy | AP_3D@0.7 Easy |
|-------------|------------------------|-----------------|----------------|
| None | 25% | 31.72 | 21.76 |
| **MonoSOWA** | **25%** | **39.99** | **32.64** |
| None | 100% | 37.99 | 29.36 |

- MonoSOWA pretraining with only 25% human annotations surpasses fully supervised training with 100% human annotations.
- 15% human annotations + MonoSOWA pretraining ≈ 100% human annotation performance → 85% annotation cost reduction.

### Cross-Dataset Training

| Training Data | AP_BEV@0.5 (Easy) | AP_3D@0.5 (Easy) |
|--------------|--------------------|------------------|
| KITTI pseudo-labels | 61.24 | 53.22 |
| K360 pseudo-labels | 57.62 | 47.39 |
| KITTI+K360 pseudo-labels | **64.29** | **58.97** |
| KITTI human labels | 67.44 | 65.09 |

Joint training with multi-dataset pseudo-labels approaches human-label performance. On Hard@0.3, it even surpasses human-labeled training.

### Ablation Study

| LOMM | SCC | AP_BEV@0.5 Easy |
|------|-----|-----------------|
| ✗ | ✗ | 20.41 |
| ✗ | ✓ | 20.37 |
| ✓ | ✗ | 35.89 |
| ✓ | ✓ | **39.22** |

LOMM is the key factor driving performance gains; SCC further improves by ~3.3 AP on top of LOMM.

## Highlights & Insights

**Strengths**:
- First fully annotation-free monocular 3D detection system (requires neither 2D nor 3D labels).
- Labeling speed of ~1.3s/frame, approximately 700× faster than VSRD.
- LOMM is the first approach to exploit temporal information from moving objects (prior methods could only discard them).
- COS enables a single model to train and infer across datasets with different camera configurations.
- As a pretraining tool, it reduces human annotation costs by 85%.

## Limitations & Future Work

- Detection accuracy is lower for distant objects (a few pixels in height), which is an inherent limitation of monocular detection rather than the labeling pipeline.
- Performance depends on the zero-shot depth estimation quality of Metric3D.
- AP at IoU=0.3 still lags behind VSRD on KITTI-360, because KITTI-360 human labels are amodal (including occluded parts), which inherently favors VSRD's use of human masks; when given identical inputs, MonoSOWA consistently outperforms VSRD.
- Only the vehicle (car) category is validated; non-rigid or small object categories such as pedestrians and cyclists are not addressed.

## Rating

- Novelty: ⭐⭐⭐⭐ First fully annotation-free monocular 3D detection system; both LOMM and SCC present meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated on three large-scale datasets with thorough ablations; pretraining and cross-dataset experiments are highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed method descriptions and logically coherent pipeline steps.
- Value: ⭐⭐⭐⭐⭐ Significant practical impact on annotation cost reduction in autonomous driving; the 700× speedup makes large-scale application feasible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](../../ECCV2024/autonomous_driving/monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[ECCV 2024\] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](../../ECCV2024/autonomous_driving/monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)
- [\[AAAI 2026\] Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection](../../AAAI2026/autonomous_driving/difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)
- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)

</div>

<!-- RELATED:END -->
