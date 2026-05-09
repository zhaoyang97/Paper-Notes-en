---
title: >-
  [Paper Note] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras
description: >-
  [ICCV 2025][3D Vision][Rolling shutter] This paper proposes a rolling shutter relative pose estimation method that requires no explicit camera motion modeling. It recovers camera pose solely from the intersections of line projections with a single selected scanline per image, and develops multiple minimal solvers for special configurations such as parallel lines and known gravity direction.
tags:
  - ICCV 2025
  - 3D Vision
  - Rolling shutter
  - relative pose estimation
  - minimal solvers
  - motion artifact
  - line field motion
date: 2026-05-08
content_hash: 5f9f0d518b37b2be
---

# Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras

**Conference**: ICCV 2025
**arXiv**: [2506.22069](https://arxiv.org/abs/2506.22069)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: Rolling shutter, relative pose estimation, minimal solvers, motion artifact, line field motion

## TL;DR

This paper proposes a rolling shutter relative pose estimation method that requires no explicit camera motion modeling. It recovers camera pose solely from the intersections of line projections with a single selected scanline per image, and develops multiple minimal solvers for special configurations such as parallel lines and known gravity direction.

## Background & Motivation

**Background**: Relative pose estimation is a fundamental problem in computer vision, widely used in SfM, SLAM, multi-view stereo, and visual odometry. Most methods assume a global shutter (GS) camera model where all pixels are exposed simultaneously. However, consumer devices (smartphones, action cameras) commonly use rolling shutter (RS) sensors that capture images row by row — when the camera moves during exposure, RS effects introduce image distortion.

**Limitations of Prior Work**: Nearly all existing RS-aware methods rely on parametric motion models (e.g., SLERP, Cayley transform, linearized rotation, affine motion) to describe the camera trajectory during scanning. These approaches suffer from: (1) motion model assumptions that may deviate from the true trajectory, introducing errors; (2) the need for prior knowledge to select an appropriate model; and (3) increased solver complexity due to complex motion parameterizations.

**Key Challenge**: Accurate RS relative pose estimation seems to require knowledge of the camera's motion state at each scanline — yet the motion model itself is a strong assumption that may introduce systematic bias.

**Goal**: Can RS relative pose estimation be performed entirely without a motion model, estimating the pose of each scanline independently? If so, this would provide a model-agnostic initialization scheme for RS SfM.

**Key Insight**: The paper exploits the projection properties of 3D lines onto RS images — by selecting one scanline per image and detecting intersections of line projections with that scanline, pose constraints can be established without any motion model.

**Core Idea**: The RS relative pose estimation problem is reformulated as a geometric problem of recovering camera pose from intersections of line projections with a single scanline per image. New algebraic constraints are derived and minimal solvers are developed accordingly.

## Method

### Overall Architecture

Given $m$ 3D lines observed by $n$ RS cameras, one scanline $y_i$ is selected from each image, and the intersections $\mathbf{p}_{i,j}$ of line projections with that scanline are detected. The goal is to recover the camera pose $(\mathbf{R}_i(y_i), \mathbf{C}_i(y_i))$ at each scanline solely from these intersection points. The core constraint follows from geometry: the ray back-projected from $\mathbf{p}_{i,j}$ must intersect the corresponding 3D line $\mathbf{L}_j$, yielding a scalar triple product constraint equal to zero.

### Key Designs

1. **Single-Scanline Geometric Constraint**:

    - Function: Establish pose constraints independent of any motion model.
    - Mechanism: The 3D line $\mathbf{L}_j$ is parameterized by a point $\mathbf{L}_{0,j}$ and direction $\mathbf{L}_{d,j}$. The camera pose at scanline $y_i$ is $(\mathbf{R}_i(y_i), \mathbf{C}_i(y_i))$. The intersection condition between the back-projected ray and the 3D line is equivalent to linear dependence of three vectors: $\mathbf{p}_{i,j}^T \mathbf{R}_i(y_i) [\mathbf{L}_{d,j}]_\times (\mathbf{L}_{0,j} - \mathbf{C}_i(y_i)) = 0$. Each intersection point yields one constraint with no dependence on any camera motion function.
    - Design Motivation: Conventional RS methods must encode a motion model into the constraints. By using only the instantaneous pose at a single scanline, the method discretizes the continuous motion problem into an isolated pose problem.

2. **Minimal Solvers for the Parallel-Line Case**:

    - Function: Exploit the parallel-line prior to substantially reduce problem complexity.
    - Mechanism: When all 3D lines are parallel (e.g., vertical lines in architectural scenes), they share direction $\mathbf{L}_d = \mathbf{e}_2$, and each line has only 2 degrees of freedom. The constraint simplifies to $\mathbf{u}_{i,j}^T \mathbf{A}_i \mathbf{L}_{h,j} = 0$, where $\mathbf{A}_i \in \mathbb{R}^{2 \times 3}$ encodes the pose. This is equivalent to 2D structure recovery from $n$ uncalibrated 1D cameras. The (B,3,7) problem (3 cameras, 7 lines) can be solved linearly via the trifocal tensor, with an average runtime of only 6.92 μs.
    - Design Motivation: Minimal problems in the general case have degree exceeding 40k, making them unsuitable for RANSAC. The parallel-line assumption is naturally satisfied in urban and architectural scenes, and reduces the degree to 2, enabling linear or low-degree polynomial solvers.

3. **Minimal Solvers with Gravity Prior**:

    - Function: Further simplify the problem and resolve projective ambiguity using a known vertical direction.
    - Mechanism: With a known gravity direction (i.e., the vanishing point of vertical lines in the image), camera rotation has only 1 degree of freedom (rotation about the vertical axis). For the vertical-line + gravity setting, the problem becomes equivalent to 2D structure recovery from calibrated 1D cameras. Two solvers are developed: (E,3,5) and (E,4,4) with degrees 16 and 32 respectively, along with a (D,3,7) parallel-line + gravity solver requiring homotopy continuation (degree 48).
    - Design Motivation: Gravity direction is available in many scenarios via IMU or scene priors (or simply assumed as $[0,1,0]^T$) at negligible cost, yet significantly reduces problem complexity and ambiguity.

### Loss & Training

This work is a purely geometric method with no learning component. Solvers are derived using algebraic techniques (SVD, Gröbner bases, homotopy continuation). Within the RANSAC framework, reprojection error is used to select inliers and the best model.

## Key Experimental Results

### Synthetic Experiments: Numerical Stability

Solver accuracy evaluated on $10^5$ noise-free synthetic instances:

| Solver | Median Rotation Error | Median Translation Error | Runtime |
|--------|----------------------|--------------------------|---------|
| (B,3,7) parallel lines | ~$10^{-11}$ rad | ~$10^{-11}$ rad | 6.92 μs |
| (E,3,5) vertical + gravity | ~$10^{-12}$ rad | ~$10^{-12}$ rad | 9.16 μs |
| (E,4,4) vertical + gravity | ~$10^{-11}$ rad | ~$10^{-10}$ rad | 70.84 μs |
| (D,3,7) parallel + gravity | ~$10^{-8}$ rad | ~$10^{-8}$ rad | 19089 μs |

### Real-World Experiments on Fastec Dataset

| Setting | Solver | ≥1 frame <5° | ≥1 frame <10° | ≥1 frame <20° | ≥1 frame <30° |
|---------|--------|--------------|---------------|---------------|---------------|
| Multi-view | (E,3,5) | 1/19 | 3/19 | 10/19 | 15/19 |
| Multi-view | (E,4,4) | 5/19 | 8/19 | 11/19 | 14/19 |
| Multi-view | (D,3,7) | 6/19 | 10/19 | 15/19 | 17/19 |
| Single-view | (E,3,5) | 3/19 | 3/19 | 5/19 | 10/19 |
| Single-view | (E,4,4) | 2/19 | 4/19 | 7/19 | 13/19 |
| Single-view | (D,3,7) | 4/19 | 4/19 | 7/19 | 9/19 |

### Key Findings

- **The (D,3,7) solver performs best in multi-view settings**: 17/19 sequences have at least one frame with error below 30°, sufficient for SfM initialization.
- **Limited information from a single scanline**: Absolute per-frame accuracy is modest, but for SfM initialization it suffices to find a reasonable initial pose.
- Single-view mode (selecting multiple scanlines from the same RS image) is also viable, though slightly less accurate than multi-view mode.
- (B,3,7) recovers only a projective reconstruction (projective ambiguity unresolved) and requires additional constraints for disambiguation.
- On a custom dataset with severe RS distortion, the proposed method achieves 30.6% of relative pose errors below 10°, far outperforming the five-point method at 10.7%.

## Highlights & Insights

- **Paradigm shift: motion-model-free RS pose estimation**: This is the first RS relative pose method requiring no assumption about the camera motion model. For scenes with complex motion (e.g., rapid rotation of a handheld device), this approach is theoretically more robust than methods tied to specific motion parameterizations.
- **Theoretical connection between 1D cameras and RS cameras**: The paper establishes an equivalence between the RS scanline problem and 1D camera pose estimation, an insight that not only simplifies the solvers but also bridges two research communities.
- **Positioning as an SfM building block**: Rather than pursuing high per-frame accuracy, the method is positioned as an initialization module for SfM — a pragmatic and well-motivated design choice.

## Limitations & Future Work

- RS distortion causes line projections to become curves; the method assumes reliable detection of line segments and their intersections with the chosen scanline, requiring robust line detection and matching.
- Solvers are only applicable to special scene configurations (parallel lines, vertical lines + gravity); the general minimal problem has degree exceeding 40k and is currently intractable.
- Pose can only be recovered up to a translational ambiguity along the line direction; additional constraints are needed for disambiguation.
- Per-frame accuracy is limited, necessitating RANSAC combined with subsequent refinement.
- Future directions include developing curve detectors and matchers, integrating the method into a complete RS SfM pipeline, and incorporating motion priors to improve accuracy.

## Related Work & Insights

- **vs. Dai et al. [2016] RS Essential Matrix**: Their work defines a RS essential matrix under a specific motion model; the present paper eliminates the motion model assumption entirely, at the cost of being restricted to line feature scenarios.
- **vs. Hahn et al. [2024] Order-one RS**: They classify RS relative pose minimal problems under an order-one motion model; the present paper serves as a complementary, motion-model-free alternative.
- **vs. PLMP (Duff et al.)**: Both are minimal problem classification works, but PLMP targets GS cameras with mixed point-line features. The methodology for classifying minimal problems in the present paper is conceptually similar.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First motion-model-free RS relative pose estimation method; strong theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive synthetic and real-data experiments with thorough comparison across multiple solvers.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous mathematical derivations and clear problem classification.
- Value: ⭐⭐⭐⭐ — Provides important theoretical foundations and practical tools for RS SfM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RePoseD: Efficient Relative Pose Estimation with Known Depth Information](reposed_efficient_relative_pose_estimation_with_known_depth_information.md)
- [\[ICCV 2025\] BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation](boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)
- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[NeurIPS 2025\] SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference](../../NeurIPS2025/3d_vision/singref6d_monocular_novel_object_pose_estimation_with_a_single_rgb_reference.md)
- [\[ICCV 2025\] IM360: Large-scale Indoor Mapping with 360 Cameras](im360_large-scale_indoor_mapping_with_360_cameras.md)

</div>

<!-- RELATED:END -->
