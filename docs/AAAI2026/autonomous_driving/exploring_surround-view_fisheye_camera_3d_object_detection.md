---
title: >-
  [Paper Note] Exploring Surround-View Fisheye Camera 3D Object Detection
description: >-
  [AAAI 2026][Autonomous Driving][fisheye camera] This paper systematically investigates 3D object detection with surround-view fisheye cameras. It constructs the Fisheye3DOD benchmark dataset containing both pinhole and f…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "fisheye camera"
  - "3D object detection"
  - "BEV perception"
  - "spherical representation"
  - "surround-view perception"
date: 2026-05-08
content_hash: ac7601fb360361d7
---

# Exploring Surround-View Fisheye Camera 3D Object Detection

**Conference**: AAAI 2026
**arXiv**: [2511.18695](https://arxiv.org/abs/2511.18695)
**Code**: [https://github.com/weiyangdaren/Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD)
**Area**: 3D Vision / Autonomous Driving
**Keywords**: fisheye camera, 3D object detection, BEV perception, spherical representation, surround-view perception

## TL;DR

This paper systematically investigates 3D object detection with surround-view fisheye cameras. It constructs the Fisheye3DOD benchmark dataset containing both pinhole and fisheye camera data, and proposes two frameworks—FisheyeBEVDet and FisheyePETR—that embed fisheye geometric modeling into mainstream detection paradigms via spherical feature representations, achieving up to 6.2 FDS improvement over rectification-based baselines.

## Background & Motivation

360° surround-view perception is critical for autonomous driving. Current mainstream solutions rely on multi-pinhole camera arrays (e.g., 6 cameras in nuScenes, 8 in Tesla), whereas fisheye cameras, with their ultra-wide field of view (>180° FoV), can achieve full coverage with only 4 cameras. They offer three key advantages:

**Hardware availability**: U.S. regulations since 2018 mandate rear-view fisheye lenses to prevent backup accidents, making them standard equipment on mass-produced vehicles (e.g., BMW), enabling direct utilization without hardware modification.

**Physical redundancy**: Overlapping FoVs naturally provide multi-perspective coverage, offering robustness against sensor failures.

**Compact deployment**: The ultra-wide FoV suits space-constrained or cost-sensitive scenarios (indoor robots, surveillance systems).

However, the nonlinear projection of fisheye lenses causes severe pixel compression—experiments show that objects in fisheye images occupy only approximately 15% of the pixel area compared to pinhole images. This information loss is irreversible; rectification cannot recover the original information. This motivates two core research questions:
- **RQ1**: How much accuracy is lost when transferring a pinhole detector to fisheye images?
- **RQ2**: How can such a transfer be made more effective?

Existing fisheye datasets do not provide paired pinhole–fisheye data from the same scenes, making systematic answers to these questions infeasible.

## Method

### Overall Architecture

The methodology proceeds in three steps: (1) constructing the Fisheye3DOD dataset to provide a fair comparison benchmark; (2) quantitatively answering RQ1 on this dataset; and (3) proposing FisheyeBEVDet and FisheyePETR to address RQ2. Both frameworks share a core design principle: introducing spherical/equirectangular projection at the feature level to map fisheye image features into a spherical coordinate system, which is then integrated with BEV-based and query-based detection paradigms respectively.

### Key Designs

1. **Fisheye3DOD Dataset**:

    - **Function**: A synchronized multi-view dataset constructed using the CARLA simulator.
    - **Mechanism**: 144 driving sequences covering urban/suburban environments, multiple lighting conditions (noon/sunset/night), and weather (clear/cloudy/rainy). Each scene is captured at 10 Hz × 50 s = 500 frames. Each scene is equipped with 6 pinhole cameras and 4 fisheye cameras (FoV = 220°), with 3D bounding box annotations.
    - **Design Motivation**: Since CARLA lacks native fisheye support, fisheye distortion is mathematically simulated via the Kannala–Brandt projection model ($r(\theta) = k_0\theta + k_1\theta^3 + k_2\theta^5 + k_3\theta^7 + k_4\theta^9$). The dual-camera system within the same scene ensures fair comparison.

2. **Spherical Feature Representation (Shared Foundation)**:

    - **Function**: Projects 2D features from fisheye images onto a spherical equirectangular representation.
    - **Mechanism**: Given backbone features $\mathbf{F}^{2d}$ from a fisheye image, a differentiable warp operation is applied via a precomputed sampling grid $\mathbf{G}_{sph}$, yielding $\mathbf{F}^{proj} = \mathbf{F}^{2d} \circ \mathbf{G}_{sph}$. The sampling grid is derived by mapping spherical direction vectors $\bar{\mathbf{p}} = [\cos\theta\cos\phi, \sin\theta, \cos\theta\sin\phi]^T$ to image coordinates through the calibrated projection function.
    - **Design Motivation**: Spherical coordinates naturally match the radial projection geometry of fisheye lenses, and the equirectangular representation's uniform angular sampling in the vertical direction is superior to cylindrical projection.

3. **FisheyeBEVDet (BEV Paradigm)**:

    - **Function**: Constructs BEV space under a spherical coordinate system.
    - **Mechanism**: Concentric spherical shells replace the parallel-plane depth discretization used in LSS. D radial depths $r_d$ are uniformly sampled along each spherical direction $\bar{\mathbf{p}}$, giving 3D points $\mathbf{p}^{cam}_{d,h,w} = r_d \times \bar{\mathbf{p}}_{h,w}$. A fully connected layer predicts depth probability distribution $\alpha$ and context vector $\mathbf{c}$, with $\mathbf{c}_d = \alpha_d \cdot \mathbf{c}$; the lifted feature volume is then projected into BEV space.
    - **Design Motivation**: The LSS assumption of perspective projection is incompatible with the nonlinear distortion of fisheye lenses. Spherical shell discretization aligns with the camera ray directions.

4. **FisheyePETR (Query Paradigm)**:

    - **Function**: Replaces the position encoding based on perspective projection with spherical coordinate-based position encoding.
    - **Mechanism**: A quadratically increasing depth interval is adopted ($r_d = r_{min} + \frac{r_{max}-r_{min}}{D(D+1)} \times d(d+1)$). Spherical coordinate encoding is applied to the projected features, which then interact with object queries via multi-head cross-attention.
    - **Design Motivation**: The 3D position encoding in PETR relies on the perspective projection assumption; applying it directly to fisheye images results in incorrect spatial position encoding.

### Loss & Training

Training is conducted on a single NVIDIA A6000 GPU. The first 70% of frames per scene are used for training and the remaining 30% for testing, sampled at 2 Hz. Models are trained for 20 epochs with batch size 4, AdamW optimizer (lr = 0.0002, weight_decay = 0.01), with 500-step linear warmup followed by cosine annealing. Detection range is $[-48, 48] \times [-48, 48] \times [-5, 5]$ m. CBGS class-balanced sampling is applied to mitigate data imbalance. Evaluation follows the nuScenes protocol (mAP + mATE/mASE/mAOE → composite FDS score).

## Key Experimental Results

### Main Results

| Method | Camera | Rectification | FDS↑ | mAP↑ | mATE↓ | mASE↓ | mAOE↓ |
|--------|--------|---------------|------|------|-------|-------|-------|
| BEVDet | 6×P | — | 0.563 | 0.506 | 0.458 | 0.161 | 0.520 |
| BEVDet | 4×F | Perspective | 0.440 | 0.304 | 0.588 | 0.177 | 0.505 |
| FisheyeBEVDet | 4×F | Equirect. | **0.485** | **0.382** | 0.591 | 0.164 | 0.480 |
| PETR | 6×P | — | 0.553 | 0.482 | 0.580 | 0.120 | 0.430 |
| PETR | 4×F | Perspective | 0.408 | 0.274 | 0.783 | 0.161 | 0.433 |
| FisheyePETR | 4×F | Equirect. | **0.470** | **0.374** | 0.727 | 0.142 | 0.434 |

### Ablation Study (Sensor Layout & Robustness)

| Configuration | FDS↑ | mAP↑ | Note |
|---------------|------|------|------|
| BEVDet 4×P (w/o front & rear) | 0.370 | 0.206 | Removing front/rear pinhole cameras creates severe blind spots |
| FisheyeBEVDet 2×F (front & rear) | 0.454 | 0.324 | Two fisheye cameras still provide omnidirectional coverage |
| FisheyeBEVDet 2×F (left & right) | 0.431 | 0.315 | Left-right layout underperforms front-rear |
| FisheyeBEVDet 4×F | 0.485 | 0.382 | Full coverage achieves best performance |
| PETR 4×P (w/o front & rear) | 0.321 | 0.142 | Pinhole degrades more severely |
| FisheyePETR 2×F (front & rear) | 0.421 | 0.289 | Fisheye degrades more gracefully |

### Key Findings

- **Answer to RQ1**: Direct transfer from pinhole to fisheye causes a drop of more than 12 FDS points (BEVDet: 0.563 → 0.440), primarily due to pixel compression (object area is ~15% of that in pinhole images).
- **Answer to RQ2**: Spherical feature modeling recovers approximately half of the performance gap (FisheyePETR +6.2 FDS over the perspective baseline).
- **RF1**: Fisheye systems are naturally robust to sensor failures—removing front and rear cameras causes only a 4–5% drop for fisheye versus 19+% for pinhole.
- **RF2**: Front-rear layout outperforms left-right layout, as most traffic participants appear along the longitudinal axis.
- **RF3**: Fisheye FDS within 30 m (0.586) approaches the pinhole FDS over the full 48 m range (0.563), making it well-suited for low-speed near-field scenarios.
- **RF4**: Small objects such as pedestrians and cyclists are most severely affected by fisheye pixel compression.

## Highlights & Insights

- This is the first work to systematically quantify the performance gap between pinhole and fisheye 3D detection, providing quantitative evidence for engineering trade-off decisions.
- The spherical representation approach is concise and general, and can be embedded into any BEV-based or query-based detection framework.
- The value of fisheye cameras in low-speed scenarios (automated parking, warehouse robots, sidewalk delivery robots) is quantitatively validated.
- The inclusion of both pinhole and fisheye data in the same dataset is a significant contribution that fills a gap in evaluation benchmarks.

## Limitations & Future Work

- The dataset is synthetic (CARLA), introducing a domain gap with real fisheye images, particularly due to insufficient texture richness that makes small object detection more challenging.
- Pixel compression in fisheye images is a fundamental challenge; geometric modeling alone cannot fully compensate for the information loss.
- Hybrid fisheye–pinhole camera configurations are not explored.
- The comparison of 4× fisheye vs. 6× pinhole involves different numbers of cameras, making it not entirely fair.
- Temporal information (e.g., BEVDet4D) and LiDAR fusion are not considered.

## Related Work & Insights

- Plaut et al. represent the only prior work on fisheye 3D object detection, but only handle single-view scenarios without comparing against pinhole cameras.
- The spherical representation approach generalizes to 3D perception for panoramic (360°) imagery.
- This work complements near-field perception studies on fisheye BEV segmentation (e.g., F2BEV, FisheyeBEVSeg) by providing a detection counterpart.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel problem formulation; methodology constitutes a well-reasoned engineering adaptation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Structured RQ+RF analysis with comprehensive multi-perspective ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Problem-driven narrative with clear structure)
- Value: ⭐⭐⭐⭐ (Dataset and framework make important contributions to the fisheye 3D detection community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](fq-petr_fully_quantized_position_embedding_transformation_fo.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[AAAI 2026\] DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving](driveflow_rectified_flow_adaptation_for_robust_3d_object_detection_in_autonomous.md)
- [\[CVPR 2026\] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](../../CVPR2026/autonomous_driving/coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)

</div>

<!-- RELATED:END -->
