---
title: >-
  [Paper Note] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency
description: >-
  [CVPR 2026][Autonomous Driving][Monocular depth estimation] This paper proposes PTC-Depth, a monocular depth estimation framework that combines optical flow triangulation with wheel odometry. It tracks the metric scale of a depth foundation model via recursive Bayesian updates, achieving temporally consistent metric depth prediction with strong generalization across KITTI, TartanAir, and thermal infrared datasets.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Monocular depth estimation
  - temporal consistency
  - Bayesian scale fusion
  - optical flow triangulation
  - wheel odometry
date: 2026-05-08
content_hash: 391d466c4e2babe4
---

# PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency

**Conference**: CVPR 2026
**arXiv**: [2604.01791](https://arxiv.org/abs/2604.01791)
**Code**: [https://ptc-depth.github.io](https://ptc-depth.github.io)
**Area**: Autonomous Driving / Depth Estimation
**Keywords**: Monocular depth estimation, temporal consistency, Bayesian scale fusion, optical flow triangulation, wheel odometry

## TL;DR

This paper proposes PTC-Depth, a monocular depth estimation framework that combines optical flow triangulation with wheel odometry. It tracks the metric scale of a depth foundation model via recursive Bayesian updates, achieving temporally consistent metric depth prediction with strong generalization across KITTI, TartanAir, and thermal infrared datasets.

## Background & Motivation

1. **Background**: Monocular depth estimation (MDE) is widely applied in autonomous driving and mobile robotics. Depth foundation models (e.g., Depth Anything v2) have achieved remarkable zero-shot generalization, but most predict only relative depth, lacking absolute metric scale.

2. **Limitations of Prior Work**: (a) Single-frame depth estimation suffers from severe temporal inconsistency (jitter and abrupt changes) across consecutive frames; (b) video depth models (e.g., VDA) improve consistency but still do not provide metric depth; (c) depth completion methods (e.g., OGNI-DC) require additional LiDAR input and are unsuitable for camera-only + odometry settings.

3. **Key Challenge**: Relative depth models preserve structure well and generalize broadly, but lack metric scale; metric depth models provide absolute scale but generalize poorly (e.g., UniDepth degrades significantly in out-of-distribution scenarios). The strengths of both are difficult to combine naively.

4. **Goal**: Using only a monocular camera and wheel odometry (no LiDAR or depth sensors), convert the relative depth output of a depth foundation model into temporally consistent metric depth.

5. **Key Insight**: The observation that the metric baseline provided by wheel odometry, combined with optical flow, jointly constrains the metric scale of depth. Sparse metric depth is obtained via triangulation between consecutive frames, and a recursive Bayesian framework tracks global/local scale factors.

6. **Core Idea**: Model the conversion from relative to metric depth as a Bayesian recursive estimation problem over a scale field $S$, with superpixel segmentation enabling local scale adaptation.

## Method

### Overall Architecture

The input consists of consecutive video frames and wheel odometry data. The pipeline comprises four steps: (1) compute optical flow between consecutive frames; (2) estimate camera pose via RANSAC using optical flow and relative depth, with wheel odometry providing the metric baseline; (3) triangulate sparse metric depth from pose and optical flow; (4) fuse the triangulated depth with the prior depth propagated from the previous frame via recursive Bayesian updates to produce the final metric depth map.

### Key Designs

1. **Robust Pose Estimation from Motion Fields**:

    - **Function**: Recover camera rotation $\boldsymbol{\Omega}$ and translation direction $\hat{\boldsymbol{T}}$ from optical flow and relative depth.
    - **Mechanism**: Applies the Longuet-Higgins motion field equation to decompose optical flow into a rotational term $\mathbf{B}\boldsymbol{\Omega}$ and a translational term $\frac{1}{\alpha d^{rel}}\mathbf{A}\boldsymbol{T}$. Assuming relative depth $d^{rel}$ is converted to metric depth via a single scale factor $\alpha$, pose recovery is formulated as an overdetermined linear system. RANSAC with stratified sampling (the image is divided into a grid with equal sampling per cell) rejects optical flow outliers from dynamic objects; IRLS with Huber weights further refines the solution.
    - **Design Motivation**: Optical flow from dynamic objects does not reflect camera motion and must be excluded. Stratified sampling ensures RANSAC hypotheses cover the entire field of view, preventing bias toward specific regions.

2. **Triangulation Quality Assessment via Sampson Residuals**:

    - **Function**: Assign reliability weights to each triangulated depth point.
    - **Mechanism**: Metric depth $z^{tri}$ is obtained by triangulating each optical flow correspondence, while the Sampson residual $\rho$ measures how well the correspondence satisfies the epipolar constraint. A small Sampson residual indicates a reliable match and accurate triangulation; a large residual flags the point as unreliable. This per-pixel reliability score is used directly as the observation uncertainty in Bayesian fusion: $V^{obs} = \sigma^2 \frac{\rho}{f_x f_y}$.
    - **Design Motivation**: Triangulation can fail due to optical flow errors, dynamic objects, or inaccurate pose; per-pixel reliability measures are necessary rather than a global threshold.

3. **Recursive Bayesian Scale Fusion (Core)**:

    - **Function**: Fuse sparse metric depth from triangulation with the prior propagated from the previous frame to produce temporally consistent metric depth.
    - **Mechanism**: Rather than fusing directly in depth space, the method estimates a latent scale field $S$ such that $Z = S \cdot d^{rel}$. A prior scale $S^{prior} = Z^{prior}/d^{rel}$ is propagated from the previous frame, and an observation scale $S^{obs} = Z^{tri}/d^{rel}$ is obtained from triangulation. A per-pixel Kalman update is performed: the normalized innovation $\gamma$ is computed for outlier detection (chi-square test), and a consistency-constrained Kalman gain $\kappa$ fuses prior and observation. Additionally, when frame-level geometric quality is poor (large median Sampson residual), the prior variance is adaptively inflated.
    - **Design Motivation**: Operating in scale space rather than depth space preserves the structural coherence of $d^{rel}$, avoiding smoothing artifacts from direct fusion. The constrained gain $\kappa$ prevents over-updating when prior and observation are weakly consistent.

4. **Superpixel-Level Scale Integration**:

    - **Function**: Address the shift component of affine-invariant depth models.
    - **Mechanism**: Felzenszwalb segmentation partitions the image into superpixels whose boundaries follow the geometric structure of $d^{rel}$. Within each superpixel $\Lambda_\ell$, the median posterior scale $\bar{s}_\ell$ is used as the unified scale for that region; regions with low fitting error use the local scale, while others fall back to a global scale estimate. The final metric depth is $Z^{post} = S^{seg} \cdot d^{rel}$.
    - **Design Motivation**: A single global scale cannot fully compensate for the shift component of affine-invariant models; local scale estimation better accommodates scale variation across different depth regions in the scene.

### Loss & Training

This method is a **training-free** inference framework with no neural network training or fine-tuning. The depth foundation model (Depth Anything v2) is used in a frozen manner; all computations are analytic (optical flow, RANSAC, Bayesian updates).

## Key Experimental Results

### Main Results

Full-range (0–80 m) depth estimation:

| Dataset | Method | AbsRel ↓ | δ<1.25 ↑ | TAE ↓ |
|--------|------|----------|----------|-------|
| KITTI | UniDepth | **0.047** | **0.977** | 4.34 |
| KITTI | **Ours** | 0.137 | 0.877 | 5.35 |
| TartanAir | **Ours** | **0.427** | **0.688** | 5.42 |
| TartanAir | UniDepth | 0.503 | 0.176 | 11.11 |
| Roadside | **Ours** | **0.309** | **0.725** | 5.27 |
| Roadside | UniDepth | 0.465 | 0.201 | 11.92 |
| MS2 (Thermal IR) | **Ours** | 0.247 | 0.700 | 5.29 |
| MS2 (Thermal IR) | DA v2 metric | 0.405 | 0.187 | 4.87 |

Short-range (0–20 m) depth estimation:

| Dataset | Method | AbsRel ↓ | δ<1.25 ↑ |
|--------|------|----------|----------|
| TartanAir | **Ours** | **0.339** | **0.712** |
| TartanAir | UniDepth | 0.485 | 0.202 |
| Roadside | **Ours** | **0.165** | **0.860** |
| Roadside | UniDepth | 0.432 | 0.241 |

### Ablation Study: Triangulation Pose Source Comparison

| Method | Pose Source | KITTI AbsRel | TartanAir AbsRel | Roadside δ1 |
|------|----------|-------------|-------------------|-------------|
| MADPose | UniDepth metric depth | 0.115 | 0.481 | 0.222 |
| **Ours** | Odometry + optical flow | **0.115** | **0.239** | **0.649** |
| GT Pose | Ground-truth pose | 0.130 | 0.168 | - |

### Key Findings

- UniDepth is strongest on KITTI (in-distribution), but the proposed method significantly outperforms it on all out-of-distribution datasets (TartanAir AbsRel 0.427 vs. 0.503; Roadside 0.309 vs. 0.465).
- MADPose relies on UniDepth's generalization capability; its triangulation accuracy degrades substantially on OOD datasets (TartanAir AbsRel 0.481), whereas the proposed method depends only on odometry for scale recovery, maintaining consistently high accuracy.
- Short-range (0–20 m) triangulation performs best due to sufficiently large baselines that yield favorable triangulation geometry; at longer ranges (20–80 m), reduced parallax causes triangulation to degrade—an inherent limitation of all geometry-based approaches.
- VDA achieves good temporal consistency (low TAE) but its metric accuracy degrades severely in OOD scenarios (Roadside AbsRel 2.198), illustrating that consistently erroneous predictions can also yield low TAE.

## Highlights & Insights

- **Training-free general framework**: Requires no dataset-specific training—only a frozen relative depth model and wheel odometry—and operates on both RGB and thermal infrared inputs. This plug-and-play design is well-suited for robotic deployment.
- **Scale-space fusion rather than depth-space fusion**: Performing Bayesian fusion in the $S = Z/d^{rel}$ space preserves the boundary sharpness and spatial structure predicted by the foundation model. This insight transfers to any problem requiring conversion from relative to absolute predictions.
- **Sampson residuals as per-pixel reliability measures**: Rather than training a dedicated confidence network, the degree to which geometric constraints are satisfied is used directly as a weighting signal—an elegant and efficient design choice.

## Limitations & Future Work

- Triangulation accuracy beyond 20 m is limited by vanishing parallax at short baselines, an inherent constraint of geometric methods.
- The approach has a degree of dependence on wheel odometry accuracy; odometry errors on uneven terrain or under wheel slip propagate into scale estimation.
- Performance on the MS2 dataset is limited by optical flow quality on thermal infrared images and odometry synchronization accuracy.
- The parameters of Felzenszwalb superpixel segmentation (scale threshold) may require tuning for different scenes.
- Only the scale component of affine-invariant models is addressed; local compensation for the shift component depends on superpixel granularity.

## Related Work & Insights

- **vs. UniDepth**: UniDepth is strongest within its training domain (KITTI) but generalizes poorly. The proposed method avoids dependence on metric depth training by exploiting geometric constraints, yielding superior generalization.
- **vs. VDA (Video Depth Anything)**: VDA is a video depth model with good temporal consistency but no metric scale, and degrades severely in OOD settings (Roadside AbsRel 2.198). The proposed method provides metric depth while maintaining reasonable temporal consistency.
- **vs. MADPose**: MADPose uses UniDepth for metric pose estimation and inherits UniDepth's generalization bottleneck. The proposed method recovers scale solely from odometry, completely decoupling the pipeline from dependence on a metric depth model.

## Rating

- **Novelty**: ⭐⭐⭐ The Bayesian scale fusion framework is a careful combination of established techniques. The core idea (odometry-constrained scale with Kalman fusion) is not entirely novel, but superpixel-level local scale estimation and Sampson residual weighting are nice design contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers RGB and thermal infrared, real and synthetic data, multiple OOD scenarios; triangulation and depth estimation are evaluated separately with in-depth near/far range analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and complete; the architecture diagram is intuitive; experimental analysis is well-structured.
- **Value**: ⭐⭐⭐⭐ High practical value for robotic and autonomous driving deployment—training-free, no additional depth sensors required, and cross-modal capable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[CVPR 2026\] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)
- [\[ICCV 2025\] DCHM: Depth-Consistent Human Modeling for Multiview Detection](../../ICCV2025/autonomous_driving/dchm_depth-consistent_human_modeling_for_multiview_detection.md)
- [\[CVPR 2026\] WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation](walkgpt_grounded_vision-language_conversation_with_depth-aware_segmentation_for_.md)

</div>

<!-- RELATED:END -->
