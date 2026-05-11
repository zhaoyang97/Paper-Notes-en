---
title: >-
  [Paper Note] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction
description: >-
  [ICCV 2025][3D Vision][Bundle Adjustment] This paper proposes BA-Track, a framework that leverages a 3D point tracker to decompose observed motion into camera motion and object motion…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Bundle Adjustment"
  - "Dynamic Scene"
  - "Motion Decoupling"
  - "3D Tracking"
  - "Depth Refinement"
date: 2026-05-08
content_hash: 5074e510676dd61f
---

# Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2504.14516](https://arxiv.org/abs/2504.14516)
**Code**: [https://wrchen530.github.io/projects/batrack](https://wrchen530.github.io/projects/batrack)
**Area**: 3D Vision / Dynamic SLAM
**Keywords**: Bundle Adjustment, Dynamic Scene, Motion Decoupling, 3D Tracking, Depth Refinement

## TL;DR

This paper proposes BA-Track, a framework that leverages a 3D point tracker to decompose observed motion into camera motion and object motion, enabling classical Bundle Adjustment to jointly handle static and dynamic scene elements for accurate camera pose estimation and temporally consistent dense reconstruction.

## Background & Motivation

Traditional SLAM/SfM systems rely on epipolar constraints, which inherently assume a static scene. When moving objects are present, these constraints are violated and the system fails. Existing strategies each carry notable drawbacks:

**Filtering dynamic elements**: Detecting and removing moving objects prior to BA → incomplete reconstruction, loss of dynamic object information

**Independent motion modeling**: Separately estimating camera and object motion → motion estimates tend to be inconsistent

**Monocular depth regression**: Per-frame reconstruction using depth priors → depth scale inconsistency across frames, difficult to align globally

The core insight of BA-Track: rather than discarding dynamic points, the framework **infers the camera-induced motion component of dynamic points**. Dynamic points become "pseudo-static" in their local reference frame, making epipolar constraints applicable to all points.

## Method

### Overall Architecture

BA-Track consists of three stages:
1. **Motion-decoupled 3D tracker** (front-end): decomposes observed motion into a static component (camera motion) and a dynamic component (object motion)
2. **Bundle Adjustment** (back-end): performs BA over all points (including dynamic ones) using the static component to recover camera poses and sparse 3D structure
3. **Global refinement**: aligns dense monocular depth maps using sparse depth estimates from BA

### Key Designs

1. **Motion-Decoupled 3D Tracker**

    - Employs a dual-network architecture instead of a single network:
        - Tracker $\mathcal{T}$ (6-layer Transformer): predicts **total motion** $X_{total}$, visibility $v$, and static/dynamic labels $m$
        - Dynamic tracker $\mathcal{T}_{dyn}$ (3-layer Transformer, lightweight): predicts the **dynamic component** $X_{dyn}$
    - Motion decoupling formula: $X_{static} = X_{total} - m \cdot X_{dyn}$
    - $m$ acts as a gating factor: when $m=0$ (static point), the static component equals total motion; when $m=1$ (dynamic point), the object motion is subtracted
    - Inputs include both RGB features and depth features (from the monocular depth model ZoeDepth) to enhance 3D reasoning
    - Design motivation: experiments show that a single network jointly learning visual tracking and motion patterns is suboptimal; task decomposition via dual networks is more effective

2. **RGB-D Bundle Adjustment**

    - Extends the DPVO framework to RGB-D BA
    - A sliding window extracts 3D trajectories of query points, yielding a local trajectory tensor $\mathbf{X} \in \mathbb{R}^{L \times N \times S \times 3}$
    - Reprojection error:
    $\arg\min_{\{\mathbf{T}_t\},\{\mathbf{Y}\}} \sum_{|i-j|\leq S} \sum_n W_n^i(j) \|\mathcal{P}_j(\mathbf{x}_n^i, y_n^i) - X_n^t(j)\|_\rho + \alpha \|y_n^i - d(\mathbf{X}_n^i)\|^2$
    - Confidence weight $W_n^i(j) = v_n^i(j) \cdot (1 - m_n^i)$: dynamic points receive lower weight, and pose recovery relies primarily on static points
    - Solved efficiently via Gauss-Newton with Schur complement decomposition

3. **Global Depth Refinement**

    - BA only adjusts the depth of sparse query points; the remaining points are not optimized
    - A 2D scale grid $\theta_t \in \mathbb{R}^{H_g \times W_g}$ (lower resolution than the original image) is introduced to apply per-pixel scaling to the depth map:
    $\hat{D}_t[\mathbf{x}] = \theta_t[\mathbf{x}] \cdot D_t[\mathbf{x}]$
    - Depth consistency loss: aligns dense depth with sparse BA trajectories
    - Scene rigidity loss: enforces that 3D distances between static points remain constant across frames
    $\mathcal{L}_{rigid} = \sum_{|i-j|<S} \sum_{(a,b) \in N} W_{static}(\|P_a^i(j) - P_b^i(j)\| - \|P_a^i - P_b^i\|)$

### Loss & Training

Tracker training loss:
$$\mathcal{L}_{total} = \mathcal{L}_{3D} + w_1 \mathcal{L}_{vis} + w_2 \mathcal{L}_{dyn}$$

- $\mathcal{L}_{3D}$: L1 loss on total motion and static component, with exponential decay weight $\gamma^{K-k}$ at each iteration
- $\mathcal{L}_{vis}$ / $\mathcal{L}_{dyn}$: binary cross-entropy for visibility and dynamic labels
- Training data: TAP-Vid-Kubric (11,000 sequences); static trajectory GT is generated by unprojecting camera poses
- $w_1 = w_2 = 5$, $K = 4$ iterative updates, window size $S = 12$

## Key Experimental Results

### Main Results (Camera Pose Evaluation — ATE)

| Method | MPI Sintel | AirDOS Shibuya | Epic Fields |
|--------|-----------|----------------|-------------|
| DROID-SLAM | 0.175 | 0.256 | 1.424 |
| DPVO | 0.115 | 0.146 | 0.394 |
| LEAP-VO | 0.089 | 0.031 | 0.486 |
| MonST3R | 0.108 | (0.512) | — |
| **BA-Track** | **0.034** | **0.028** | **0.385** |

ATE on Sintel is reduced from 0.089 to 0.034, a relative improvement of **62%**.

Depth evaluation (Abs Rel ↓):

| Method | MPI Sintel | AirDOS Shibuya | Bonn |
|--------|-----------|----------------|------|
| ZoeDepth | 0.467 | 0.571 | 0.087 |
| MonST3R | 0.335 | (0.208) | 0.063 |
| **BA-Track** | **0.408** | **0.299** | **0.084** |

### Ablation Study

Comparison of motion decoupling strategies (Sintel ATE):

| Setting | Trajectory Type | Dynamic Mask | ATE |
|---------|----------------|--------------|-----|
| (a) Total motion only | Total | — | 0.137 |
| (b) Total motion + mask | Total | ✓ | 0.047 |
| (c) Direct static prediction | Static* | — | 0.091 |
| (e) Motion decoupling | Total-Dynamic | — | 0.065 |
| **(f) Decoupling + mask** | Total-Dynamic | ✓ | **0.034** |

Depth refinement ablation (Bonn crowd2):

| $\mathcal{L}_{depth}$ | $\mathcal{L}_{rigid}$ | Abs Rel | $\delta<1.25$ |
|---|---|---------|---------------|
| ✗ | ✗ | 0.121 | 89.6% |
| ✓ | ✗ | 0.103 | 94.8% |
| ✗ | ✓ | 0.117 | 88.4% |
| ✓ | ✓ | **0.089** | **95.0%** |

### Key Findings

- Motion decoupling reduces ATE from 0.137 to 0.065 (a relative reduction of 53%); adding the dynamic mask further brings it down to 0.034
- Directly predicting the static component with a single network (Static*) underperforms the dual-network decoupling approach
- The two depth refinement losses are complementary: the depth consistency loss contributes more, while the rigidity constraint provides additional gain
- MonST3R can process at most 90 frames on a 48 GB GPU, whereas BA-Track is significantly more memory-efficient

## Highlights & Insights

- **Conceptual elegance**: rather than "removing" dynamic elements, the framework "transforms" them into pseudo-static points, restoring the applicability of classical BA
- The **dual-network decoupling design** is well-motivated and thoroughly validated through ablation, demonstrating the suboptimality of a single network learning both visual tracking and motion patterns
- A strong exemplar of **hybrid approaches**: organically combining classical optimization (BA) with learned priors (3D tracker)
- Global refinement employs a lightweight scale grid rather than a neural network, keeping parameter count low and inference efficient

## Limitations & Future Work

- The depth refinement uses a simple scale grid; more expressive deformation models (e.g., neural networks) may yield further improvements
- Performance depends on the quality of the monocular depth prior (ZoeDepth/UniDepth)
- The tracker is trained on synthetic data (Kubric), and domain transfer to real-world scenes may incur a performance gap
- Integration with novel representations such as 3D Gaussian Splatting remains unexplored

## Related Work & Insights

- Relationship to DROID-SLAM and DPVO: BA-Track extends traditional BA frameworks to support dynamic scenes
- Comparison with MonST3R: MonST3R filters dynamic regions using optical flow masks, whereas BA-Track actively exploits information from dynamic regions
- The motion decoupling idea generalizes to other tasks requiring handling of mixed motion (e.g., static/dynamic separation in autonomous driving)
- The results validate the considerable potential of the "classical optimization + learned prior" hybrid paradigm for dynamic scene understanding

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Motion decoupling enables BA to handle dynamic scenes with a novel and intuitively clear formulation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, comprehensive evaluation of camera pose, depth, and reconstruction, with thorough ablations
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear and technical descriptions are detailed
- Value: ⭐⭐⭐⭐⭐ Significant contribution to dynamic scene SLAM and reconstruction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](../../CVPR2026/3d_vision/vggt-slam.md)
- [\[ICCV 2025\] Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)
- [\[ICCV 2025\] Proactive Scene Decomposition and Reconstruction](proactive_scene_decomposition_and_reconstruction.md)

</div>

<!-- RELATED:END -->
