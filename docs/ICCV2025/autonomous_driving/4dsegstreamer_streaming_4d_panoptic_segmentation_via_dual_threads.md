---
title: >-
  [Paper Note] 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads
description: >-
  [ICCV 2025][Autonomous Driving][4D panoptic segmentation] This paper proposes 4DSegStreamer, a streaming 4D panoptic segmentation framework built upon a dual-thread system (predictive thread + inference thread). It achieves real-time, high-quality 4D panoptic segmentation through geometric and motion memory maintenance, ego-pose prediction, and inverse forward flow iteration.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 4D panoptic segmentation
  - streaming perception
  - dual-thread system
  - motion alignment
  - point cloud sequences
date: 2026-05-08
content_hash: b2d4ad023a23c56b
---

# 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads

**Conference**: ICCV 2025
**arXiv**: [2510.17664](https://arxiv.org/abs/2510.17664)
**Code**: [https://github.com/llada60/4DSegStreamer](https://github.com/llada60/4DSegStreamer)
**Area**: Autonomous Driving
**Keywords**: 4D panoptic segmentation, streaming perception, dual-thread system, motion alignment, point cloud sequences

## TL;DR

This paper proposes 4DSegStreamer, a streaming 4D panoptic segmentation framework built upon a dual-thread system (predictive thread + inference thread). It achieves real-time, high-quality 4D panoptic segmentation through geometric and motion memory maintenance, ego-pose prediction, and inverse forward flow iteration.

## Background & Motivation

4D Panoptic Segmentation aims to perform instance-level and semantic-level dense perception over continuous point cloud sequences. In highly dynamic scenes such as autonomous driving, each incoming frame must be processed within a strict time budget, i.e., under the **streaming perception** setting.

Core challenges faced by existing methods:

**Computational latency**: Existing 4D segmentation methods (Mask4Former, Eq-4D-StOP, etc.) are computationally heavy and cannot satisfy real-time constraints, leading to severe performance degradation under streaming settings.

**Insufficient granularity**: Prior streaming perception research has focused primarily on 2D/3D object detection and tracking, providing bounding-box-level outputs that are insufficient for downstream decision-making (e.g., recognizing construction zones and sidewalks).

**Inadequate handling of dynamic objects**: Naive temporal feature fusion fails to handle moving objects in highly dynamic scenes, with performance deteriorating significantly at high frame rates.

The authors argue that wrapping existing segmentation methods into a streaming-compatible system is more efficient and flexible than training a dedicated real-time model from scratch.

## Method

### Overall Architecture

4DSegStreamer adopts a **Dual-Thread System** that categorizes streaming frames into keyframes and non-keyframes:

- **Predictive Thread**: Extracts geometric and motion features on keyframes, continuously updates memory, and leverages historical information to predict future dynamics.
- **Inference Thread**: For each incoming non-keyframe, rapidly retrieves features via geometric alignment with the latest memory, enabling real-time inference.

The two threads run in parallel and share memory; inference latency is dominated by the lightweight inference thread.

### Key Designs

**1. Geometric Memory Update**

A sparse ConvGRU mechanism is employed to update geometric memory. Upon the arrival of a new keyframe:
- The previous memory state is first transformed into the current frame's coordinate system via motion alignment.
- The memory is then updated using the current frame features through a GRU gating mechanism (comprising an update gate $z_t$ and a reset gate $r_t$).

This design supports seamless integration with existing 3D and 4D segmentation backbones.

**2. Ego-pose Future Alignment**

Two settings are supported:
- **Known pose**: Relative poses are used directly for alignment.
- **Unknown pose**: Suma++ estimates relative poses between keyframes; an LSTM maintains a pose memory, and a multi-head predictor forecasts future poses.

**3. Dynamic Object Future Alignment**

- **Future flow prediction**: FastNSF is used during training to obtain ground-truth supervisory flow; at inference, the lightweight zeroFlow estimates inter-keyframe optical flow, which is then fed into an LSTM to predict future flow.
- **Inverse Forward Flow Iteration**: The core technical contribution. Directly warping memory with forward flow requires rebuilding a KD-Tree, which is costly; directly predicting backward flow is inaccurate because future point positions are unknown. The paper proposes an iterative approach: for a query point $y$, the method iterates $x_{n+1} = y - \text{flow}(x_n)$, which converges to a fixed point under Lipschitz continuity conditions.

### Loss & Training

- Training proceeds in two stages: the backbone segmentation model is first trained normally, after which the backbone is frozen and the pose prediction, flow prediction, and memory aggregation modules are trained.
- The maximum number of iterations for the inverse flow iteration is set to 10, with a convergence threshold to control accuracy.
- A moving mask is applied to moving objects, assigning non-zero flow only to dynamic instances.

## Key Experimental Results

### Main Results

SemanticKITTI, unknown-pose streaming setting:

| Method | sLSTQ | sPQ | sPQ_d | sPQ_s |
|--------|-------|-----|-------|-------|
| StreamYOLO | 0.415 | 0.373 | 0.429 | 0.371 |
| Mask4Former | 0.515 | 0.485 | 0.571 | 0.413 |
| PTv3 | 0.536 | 0.567 | 0.638 | 0.464 |
| **4DSegStreamer (M4F)** | **0.688** | **0.634** | **0.744** | **0.486** |

nuScenes, known-pose streaming setting:

| Method | sLSTQ | sPQ | sPQ_d | sPQ_s |
|--------|-------|-----|-------|-------|
| Eq-4D-StOP | 0.695 | 0.673 | 0.654 | 0.693 |
| **4DSegStreamer (M4F)** | **0.765** | **0.751** | **0.734** | **0.786** |

### Ablation Study

Incremental component ablation (SemanticKITTI, unknown-pose):

| Configuration | sLSTQ | sLSTQ_d | sLSTQ_s |
|---------------|-------|---------|---------|
| P3Former baseline | 0.304 | 0.265 | 0.357 |
| + Memory | 0.349 | 0.292 | 0.408 |
| + Memory + Pose | 0.497 | 0.488 | 0.501 |
| + Memory + Pose + Flow | 0.591 | 0.667 | 0.514 |
| + Memory + Pose + Moving Flow | **0.613** | **0.682** | **0.516** |

Comparison of flow prediction strategies:

| Method | sLSTQ | sLSTQ_d | sLSTQ_s |
|--------|-------|---------|---------|
| Backward flow | 0.565 | 0.637 | 0.483 |
| Forward flow | 0.589 | 0.667 | 0.497 |
| Inverse flow iteration | **0.613** | **0.682** | **0.516** |

### Key Findings

1. Compared to PTv3, the proposed method achieves a 15.2% gain in sLSTQ (0.536 → 0.688) on the SemanticKITTI unknown-pose streaming setting, representing a substantial improvement.
2. Gains on dynamic object perception (sPQ_d) are particularly pronounced, demonstrating the effectiveness of motion alignment for dynamic instances.
3. Inverse forward flow iteration shows clear advantages over direct forward/backward flow prediction, achieving a favorable accuracy–efficiency trade-off.
4. The framework is general-purpose and can be seamlessly integrated with multiple backbones including P3Former, Mask4Former, and Eq-4D-StOP.
5. The method also outperforms the runner-up by 6.6% sLSTQ on the indoor dataset HOI4D.

## Highlights & Insights

- **Engineering elegance of the dual-thread design**: Offloading heavy computation to the predictive thread while keeping the inference thread lightweight makes the system naturally suited for real-time deployment.
- **Inverse forward flow iteration** is the paper's most significant technical contribution, elegantly resolving the dilemma between forward flow (requiring expensive data structure reconstruction) and backward flow (suffering from poor prediction accuracy for unknown future positions).
- **Plug-and-play capability**: The framework can endow any existing 3D/4D segmentation method with streaming capability, offering high practical value.
- This work is the first to systematically define and evaluate the streaming 4D panoptic segmentation task.

## Limitations & Future Work

1. Pose prediction relies on Suma++, which may be inaccurate under extreme motion conditions.
2. The zeroFlow model used for flow estimation is a distilled model, imposing an accuracy ceiling.
3. Convergence of the inverse flow iteration depends on Lipschitz continuity, which may not hold under extremely fast motion.
4. Validation is limited to LiDAR point clouds; extension to camera-based methods has not been explored.
5. Memory size grows over time; long-term scenarios may require a memory eviction strategy.

## Related Work & Insights

- The dual-thread design shares conceptual similarities with the fast–slow system in DriveVLM-Dual, but 4DSegStreamer unifies both components into a single pipeline.
- The memory mechanism (ConvGRU) draws inspiration from NSM4D and MemorySeg.
- The streaming perception paradigm follows the same thread as 2D streaming methods (StreamYOLO, DAMO-StreamNet), but extends to the more challenging task of dense panoptic segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual-thread system and inverse flow iteration are novel; the overall framework is an elegant combination of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, multiple backbones, detailed ablations, and multiple evaluation settings.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ — First work to define streaming 4D panoptic segmentation; high practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[ICCV 2025\] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)
- [\[ICCV 2025\] ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models](eta_efficiency_through_thinking_ahead_a_dual_approach_to_self-driving_with_large.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)

<!-- RELATED:END -->
