---
title: >-
  [Paper Note] PRaDA: Projective Radial Distortion Averaging
description: >-
  [CVPR 2025][3D Vision][Camera Calibration] PRaDA proposes a radial distortion calibration method operating entirely in projective space. By performing weighted averaging of distortion estimates from multiple image pairs in the function space, it achieves high-precision distortion correction without requiring 3D point reconstruction or camera pose estimation, significantly outperforming traditional methods such as COLMAP and GLOMAP on multiple datasets with severe distortion.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Camera Calibration"
  - "Radial Distortion"
  - "Projective Geometry"
  - "Distortion Averaging"
  - "Self-Calibration"
date: 2026-05-08
content_hash: 19967eca9a53e261
---

# PRaDA: Projective Radial Distortion Averaging

**Conference**: CVPR 2025  
**arXiv**: [2504.16499](https://arxiv.org/abs/2504.16499)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Camera Calibration, Radial Distortion, Projective Geometry, Distortion Averaging, Self-Calibration

## TL;DR

PRaDA proposes a radial distortion calibration method operating entirely in projective space. By performing weighted averaging of distortion estimates from multiple image pairs in the function space, it achieves high-precision distortion correction without requiring 3D point reconstruction or camera pose estimation, significantly outperforming traditional methods such as COLMAP and GLOMAP on multiple datasets with severe distortion.

## Background & Motivation

**Background**: Accurate camera models are fundamental to geometric vision algorithms such as SfM, SLAM, and novel view synthesis. Current distortion calibration primarily follows three routes: (1) SfM-based methods (e.g., COLMAP/GLOMAP) that jointly estimate distortion parameters in bundle adjustment; (2) learning-based methods (e.g., GeoCalib, DeepCalib) that directly regress parameters from a single image; (3) N-point solvers that estimate the fundamental matrix and distortion on image pairs.

**Limitations of Prior Work**: SfM-based methods couple distortion estimation with 3D reconstruction, risking convergence failure without a good initialization, especially in scenes with severe distortions such as fisheye. Learning-based methods lack accuracy and robustness, and their training data is deficient in diverse fisheye distortion samples. Two-view solvers are typically inaccurate across the entire image domain because the corresponding points may not cover the full region.

**Key Challenge**: Accurately estimating distortion parameters either requires solving the complete SfM problem (complex and prone to failure) or relies on learning-based methods (insufficient accuracy). Is it possible to bypass the complexity of SfM methods while retaining their accuracy?

**Goal**: De-couple distortion calibration from 3D reconstruction to achieve high-precision distortion estimation in projective space solely using two-view relations.

**Key Insight**: In projective space, the geometric relationship is described by the fundamental matrix, which can absorb all camera parameters other than distortion (such as focal length). This implies that distortion optimization can be performed without estimating focal length, 3D points, or poses. Meanwhile, learning-based matchers are already robust enough to handle distorted images.

**Core Idea**: Independently estimate distortion parameters from each image pair, and then fuse these inconsistent estimates by weighted averaging in the function space to obtain a single, consistent camera model.

## Method

### Overall Architecture

The pipeline of the method consists of four steps: (1) obtain keypoint correspondences for each image pair using feature matching, and estimate a one-parameter distortion model via a robust solver; (2) perform non-linear refinement for each pair to upgrade to a higher-order polynomial model; (3) perform distortion averaging over all two-view estimates of the same camera to fuse them into a single model; (4) perform global refinement across all image pairs to optimize the Sampson error. The entire process requires no 3D points or camera poses.

### Key Designs

1. **Two-view Initialization and Non-linear Refinement**:

    - **Function**: Obtain initial distortion estimates from each image pair and upgrade them to higher-order models
    - **Mechanism**: Use LO-RANSAC + F10 solver to obtain an initial estimate of the single-parameter division model $d_\lambda(r) = 1/(1+\lambda r^2)$ while performing inlier filtering. Then, non-linear refinement is conducted by minimizing the Sampson error: $\text{argmin}_{F,\theta} \sum_l r^2_{\text{sampson}}(p_l, q_l, F, \theta_1, \theta_2)$, and upgrading the model to a k-th order polynomial $h_\theta(r) = \sum_{i=0}^k \theta_i r^i$. The fundamental matrix is parameterized using the minimal 7-DoF parameterization of $SO(3) \times S^1 \times SO(3)$.
    - **Design Motivation**: Although the single-parameter model has an efficient minimal solver, its expressiveness is insufficient. Upgrading to higher-order models via non-linear optimization allows capturing more complex real-world distortion patterns. Sampson error depends solely on epipolar constraints and does not require 3D points.

2. **Distortion Averaging in Function Space**:

    - **Function**: Fuse multiple inconsistent distortion estimates of the same camera into a single consistent model
    - **Mechanism**: Each two-view estimate is only reliable in the regions covered by the corresponding points and may be inconsistent in other regions. Formulate the averaging problem as a weighted least squares: $\bar{\theta} = \text{argmin}_\theta \sum_i \omega_i \int_0^R \|1/h_\theta(r) - 1/h_{\theta_i}(r)\|^2 r^3 dr$, solved numerically after initializing with the weighted average $\bar{\theta} = \sum \omega_i \theta_i / \sum \omega_i$. Radial symmetry simplifies the integration from the 2D image domain to the 1D radial domain.
    - **Design Motivation**: This is the core innovation of the paper. Distortion estimation from a single image pair is limited by the distribution of correspondences. Fusing multiple pair estimates is necessary to cover the entire image and obtain a globally consistent model. This also enables the generation of polynomials of arbitrary orders.

3. **Polynomial Regularization and Global Refinement**:

    - **Function**: Constrain distortion behavior in regions uncovered by correspondences, and optimize globally and consistently across all images
    - **Mechanism**: Regularization ensures monotonicity by minimizing the rate of change of the undistortion function $\min \int_0^R \|dU_\theta(r)/dr\|^2 dr$. Global refinement jointly optimizes the robust Sampson error of all image pairs: $\text{argmin}_{\{F_{ij}\}, \{\theta_k\}} \sum_{l,i,j} \rho(r_{\text{sampson}}(\cdot))$, using the Cauchy loss function to handle outliers.
    - **Design Motivation**: Higher-order polynomials can behave unreasonably (e.g., oscillation) in unconstrained regions. Regularization ensures physical consistency. Global refinement allows the same camera to share parameters across all image pairs, further improving consistency.

### Loss & Training

A non-learning method, with the core metric being the Sampson error (an approximation of the minimal pixel adjustment under epipolar constraints). Global refinement uses the Cauchy robust loss. Pixel coordinates are normalized by the image diagonal to improve numerical stability.

## Key Experimental Results

### Main Results

Focal-Length Adjusted Reprojection Error (FA-RE, pixels):

| Method | ScanNet++ Mean | ETH3D cam4 Mean | ETH3D cam5 Mean | KITTI-360 cam2 Mean | KITTI-360 cam3 Mean |
|------|----------------|-----------------|-----------------|---------------------|---------------------|
| COLMAP | 2.0 | 26.0 | 25.1 | 125.5 | 112.4 |
| GLOMAP | 1.8 | 18.4 | 19.6 | 122.0 | 113.3 |
| DroidCalib | 1.2 | 36.3 | 46.4 | 102.2 | 128.1 |
| GeoCalib | 4.6 | 35.8 | 34.6 | 125.5 | 123.1 |
| DeepCalib | 10.8 | 20.9 | 18.2 | 160.5 | 153.2 |
| **Ours (SIFT)** | **0.6** | **5.3** | **14.4** | **44.8** | **50.2** |

WoodScape (180° fisheye): Front 51.2 vs DeepCalib 9.7 / GeoCalib 98.0

### Ablation Study

PRaDA initialization + GLOMAP vs GLOMAP alone (ScanNet++ sparse test set):

| Metric | PRaDA + GLOMAP | GLOMAP |
|------|---------------|--------|
| Rotation Error Min/Mean/Max | 0.18/4.51/44.56 | 0.25/28.99/118.6 |
| Translation Direction Error Min/Mean/Max | 0.26/8.70/81.07 | 0.33/27.39/95.76 |

### Key Findings

- On heavily distorted datasets like KITTI-360, PRaDA's error is roughly 1/3 to 1/2 of COLMAP/GLOMAP.
- On ScanNet++, sub-pixel error concentration is much higher than SfM methods.
- After providing initialization for GLOMAP through decoupled calibration, the average rotation error drops from 28.99° to 4.51°, significantly improving 3D reconstruction.
- Forward/backward-facing cameras represent degenerate cases (epipolar lines are straight lines), where distortion estimation is poorer.
- Learning-based matchers (LOFTR) can still provide effective correspondences on highly distorted WoodScape images.

## Highlights & Insights

- Decoupling distortion in projective space is highly elegant: the fundamental matrix absorbs parameters like focal length, making distortion estimation an independent problem.
- Distortion averaging in function space is an innovative and mathematically beautiful design, fusing multiple locally accurate estimates into a globally consistent model.
- It does not require point tracks across images, reducing data requirements.
- Methodologically, it demonstrates that "calibrating distortion accurately first, then running SfM" is more reliable than "joint estimation within SfM".

## Limitations & Future Work

- It assumes radially symmetric distortion and cannot handle non-radially symmetric distortion models (e.g., GT models of ETH3D/KITTI-360).
- It relies on the quality of the matcher; if the matcher is primarily trained under pinhole settings, it may introduce non-Gaussian errors in heavily distorted areas.
- It uses a predefined RANSAC threshold; the authors suggest that future work could use the σ++ consensus method to completely avoid thresholding.
- Focal length is not estimated, requiring an extra step to obtain the complete camera model.

## Related Work & Insights

- Fitzgibbon's [12] division model and Kukelova's [30] F10 solver are the cornerstones of this method.
- It shares similarities with the global SfM approach of GLOMAP [41], but PRaDA does not require 3D points.
- Insights: Geometric principles can be precisely handled using traditional methods without relying on neural networks to relearn them—this is consistent with the views of Sarlin et al.

## Rating

- **Novelty**: 9/10 — Distortion averaging under the projective framework is a brand-new concept, and the mathematical derivation is elegant.
- **Experimental Thoroughness**: 8/10 — Evaluated on four datasets with severe distortions, with a comprehensive comparison against five baselines.
- **Writing Quality**: 8/10 — Mathematical derivations are clear with well-defined motivation, though some notations are dense.
- **Value**: 8/10 — Provides a new high-precision tool for self-calibration, with high practical value as an SfM preprocessing step.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Deformable Radial Kernel Splatting](deformable_radial_kernel_splatting.md)
- [\[ICCV 2025\] PLMP -- Point-Line Minimal Problems for Projective SfM](../../ICCV2025/3d_vision/plmp_-_point-line_minimal_problems_for_projective_sfm.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2026\] Hermite Radial Basis Function for Surface Reconstruction via Differentiable Rendering](../../CVPR2026/3d_vision/hermite_radial_basis_function_for_surface_reconstruction_via_differentiable_rend.md)
- [\[CVPR 2025\] Volumetrically Consistent 3D Gaussian Rasterization](volumetrically_consistent_3d_gaussian_rasterization.md)

</div>

<!-- RELATED:END -->
