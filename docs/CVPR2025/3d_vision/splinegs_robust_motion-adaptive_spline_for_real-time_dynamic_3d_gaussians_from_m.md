---
title: >-
  [Paper Note] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video
description: >-
  [CVPR 2025][3D Vision][Dynamic Scene Reconstruction] SplineGS proposes a dynamic 3DGS framework based on cubic Hermite splines. By modeling the continuous trajectories of dynamic Gaussians through Motion-Adaptive Spline (MAS) and Motion-Adaptive Control Point pruning (MACP) while jointly optimizing camera parameters, it achieves SOTA dynamic novel view synthesis and real-time rendering without requiring COLMAP.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dynamic Scene Reconstruction"
  - "3D Gaussian Splatting"
  - "Spline Curves"
  - "Monocular Video"
  - "COLMAP-free"
date: 2026-05-08
content_hash: 3b1f2a6ea55a4d43
---

# SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video

**Conference**: CVPR 2025  
**arXiv**: [2412.09982](https://arxiv.org/abs/2412.09982)  
**Code**: [https://kaist-viclab.github.io/splinegs-site/](https://kaist-viclab.github.io/splinegs-site/)  
**Area**: 3D Vision  
**Keywords**: Dynamic Scene Reconstruction, 3D Gaussian Splatting, Spline Curves, Monocular Video, COLMAP-free

## TL;DR

SplineGS proposes a dynamic 3DGS framework based on cubic Hermite splines. By modeling the continuous trajectories of dynamic Gaussians through Motion-Adaptive Spline (MAS) and Motion-Adaptive Control Point pruning (MACP) while jointly optimizing camera parameters, it achieves SOTA dynamic novel view synthesis and real-time rendering without requiring COLMAP.

## Background & Motivation

**Background**: Novel view synthesis of dynamic scenes is a core challenge in 3D vision. Existing dynamic 3DGS methods model deformation using MLPs (D3DGS, slow), space-time grids (4DGS, resolution-limited), or polynomial trajectories (STGS, fixed order), each with its own limitations.

**Limitations of Prior Work**: MLPs severely slow down rendering speed; grid methods face bottlenecks in capturing fine dynamic details; fixed-order polynomials cannot adapt to the varying motion complexities across the scene (e.g., static background vs. highly dynamic balloons). Furthermore, most methods rely on COLMAP, which frequently fails on dynamic monocular videos.

**Key Challenge**: There is a need for a deformation representation that can accurately model continuous motion trajectories, adaptively adjust motion complexity, and maintain real-time rendering speed.

**Goal**: To design a COLMAP-free dynamic 3DGS framework that accurately represents the continuous trajectories of dynamic Gaussians using minimal parameters while supporting adaptability to motion complexity.

**Key Insight**: Spline curves are classic tools in computer graphics for modeling continuous curves—representing smooth piecewise cubic curves with a few control points, which is both flexible and efficient.

**Core Idea**: Utilizing cubic Hermite splines to represent the position trajectories of dynamic 3D Gaussians, where the number of control points is automatically determined through motion-adaptive pruning.

## Method

### Overall Architecture

SplineGS divides 3D Gaussians into static and dynamic sets. The position of a dynamic Gaussian $\mu(t) = S(t, \mathbf{P})$ is defined by a cubic Hermite spline. The training is split into two stages: (1) Warm-up, which optimizes only camera parameters, and (2) Main training, which jointly optimizes Gaussian attributes, spline control points, and camera parameters.

### Key Designs

1. **Motion-Adaptive Spline (MAS)**:

    - **Function**: Models time-continuous position trajectories for each dynamic 3D Gaussian.
    - **Mechanism**: A cubic Hermite spline generates a C1-continuous piecewise cubic curve between adjacent control points, with tangent vectors approximated via finite differences. Control point initialization is achieved by back-projecting 2D trajectories into 3D using 2D tracking from CoTracker and metric depth from UniDepth, followed by least-squares fitting.
    - **Design Motivation**: Unlike polynomials, splines are defined piecewise, avoiding the numerical instability (Runge's phenomenon) of high-degree polynomials. Computation only requires locating adjacent control points and performing interpolation, which is extremely fast (5.63 ns vs. 149 ns for MLPs).

2. **Motion-Adaptive Control Point pruning (MACP)**:

    - **Function**: Automatically determines the optimal number of control points for each Gaussian.
    - **Mechanism**: Every 100 iterations, the method attempts to approximate the trajectory with a new spline having one fewer control point using least squares. If the projection error in image space is below a threshold, the pruning is accepted. Simple motions end up using 2-3 control points, while complex motions utilize 6-8.
    - **Design Motivation**: A fixed number of control points is redundant for simple motion (overfitting and latency increase) and insufficient for complex motion (underfitting and quality degradation).

3. **COLMAP-free Camera Parameter Estimation**:

    - **Function**: Estimates camera intrinsic and extrinsic parameters from purely monocular videos.
    - **Mechanism**: A shallow MLP predicts rotation and translation for each frame, while the focal length is shared across all frames. Joint optimization is performed through photometric consistency (aligning colors across frames on static regions) and geometric consistency (aligning 3D points), using a motion mask to exclude dynamic regions.
    - **Design Motivation**: COLMAP frequently fails on dynamic monocular videos (for instance, it completely fails to work on the DAVIS dataset).

### Loss & Training

Warm-up loss: photometric consistency + geometric consistency. Main training loss consists of 6 components: RGB L1 loss, depth loss, motion mask Dice loss, photometric consistency loss, rendered depth photometric consistency loss, and geometric consistency loss. Training employs a 1K warm-up followed by 20K main training iterations.

## Key Experimental Results

### Main Results

Novel view synthesis on the NVIDIA dataset (COLMAP-free):

| Method | PSNR | LPIPS | FPS |
|------|------|-------|-----|
| RoDynRF | 25.38 | 0.079 | 0.45 |
| MoSca | 26.61 | 0.069 | N/A |
| **SplineGS** | **27.21** | **0.053** | **400** |

Novel view + novel time synthesis (a more challenging setting):

| Method | PSNR | LPIPS | tOF |
|------|------|-------|-----|
| DynNeRF | 23.36 | 0.219 | 0.921 |
| RoDynRF | 21.58 | 0.221 | 2.138 |
| **SplineGS** | **25.92** | **0.098** | **0.703** |

### Ablation Study

| Deformation Model | PSNR | LPIPS | Latency (ns) |
|---------|------|-------|----------|
| MLP | 23.51 | 0.125 | 149.41 |
| Grid | 25.48 | 0.090 | 98.89 |
| Poly (3rd) | 25.14 | 0.111 | 1.80 |
| Poly (10th) | 24.38 | 0.120 | 7.71 |
| Bezier | 27.19 | 0.060 | 8.78 |
| **MAS (ours)** | **27.21** | **0.053** | 5.63 |

| MACP Configuration | PSNR | LPIPS | Latency |
|-----------|------|-------|------|
| w/o MACP (Nc=4) | 26.62 | 0.065 | 5.34 |
| w/o MACP (Nc=Nf) | 27.08 | 0.054 | 6.11 |
| **Full (MACP)** | **27.21** | **0.053** | 5.63 |

### Key Findings

- MAS achieves the best quality among all deformation models while keeping a speed close to the fastest polynomial (5.63 vs. 1.80 ns), which is significantly faster than MLP (149 ns).
- Higher-degree polynomials (10th) perform worse than lower-degree ones (3rd), validating the numerical instability issue.
- The advantage is even more pronounced (+2.5 dB) in novel time synthesis tasks, as the continuity of splines naturally supports temporal interpolation.
- Removing photometric consistency causes the PSNR to drop to 17.49, proving it to be the cornerstone of camera estimation.

## Highlights & Insights

- **A Return to Classical Graphics Tools**: Amid dynamic modeling dominated by MLPs and grids, returning to splines offers a simple and efficient alternative. Splines naturally possess continuity, local controllability, and high computational efficiency.
- **Adaptive Strategy of MACP**: It acts as "intelligent compression" to automatically allocate motion complexity budgets for each Gaussian. The Nc heatmap intuitively demonstrates the motion complexity of different regions.
- **Practical Value of Being COLMAP-free**: COLMAP fails completely on in-the-wild videos such as DAVIS. SplineGS is capable of recovering camera parameters from scratch.

## Limitations & Future Work

- Still relies on CoTracker and UniDepth as priors.
- Only position trajectories are modeled; rotation and scale are parameterized per time step.
- Extreme motion or topological change scenarios have not been verified.

## Related Work & Insights

- **vs D3DGS**: MLP-based deformation is slow and yields poor quality (23.02 vs. 27.21). SplineGS demonstrates that explicit parameterization significantly outperforms implicit methods.
- **vs STGS**: Fixed-order polynomials are the fastest but deliver limited quality; splines offer a superior trade-off between quality and speed.
- **vs RoDynRF**: Both are COLMAP-free, but RoDynRF is NeRF-based and extremely slow. SplineGS is 890 times faster and delivers better quality.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of spline trajectories and MACP is clever, with a relatively intuitive core idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two datasets covering NVS + NVT, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology, rigorous mathematical formulations, and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Offers COLMAP-free operation, real-time rendering, and SOTA quality, providing highly practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](../../AAAI2026/3d_vision/mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)
- [\[CVPR 2025\] MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)
- [\[CVPR 2025\] SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens](sat-hmr_real-time_multi-person_3d_mesh_estimation_via_scale-adaptive_tokens.md)
- [\[CVPR 2025\] MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)

</div>

<!-- RELATED:END -->
