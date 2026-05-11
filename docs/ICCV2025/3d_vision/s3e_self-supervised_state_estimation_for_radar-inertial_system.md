---
title: >-
  [Paper Note] S3E: Self-Supervised State Estimation for Radar-Inertial System
description: >-
  [ICCV 2025][3D Vision][millimeter-wave radar] S3E is proposed as the first method to achieve complementary self-supervised state estimation from radar signal spectra and inertial data…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "millimeter-wave radar"
  - "IMU fusion"
  - "self-supervised learning"
  - "state estimation"
  - "radar spectrum"
date: 2026-05-08
content_hash: 6f373660a1bcb66f
---

# S3E: Self-Supervised State Estimation for Radar-Inertial System

**Conference**: ICCV 2025
**arXiv**: [2509.25984](https://arxiv.org/abs/2509.25984)
**Code**: Not released
**Area**: 3D Vision
**Keywords**: millimeter-wave radar, IMU fusion, self-supervised learning, state estimation, radar spectrum

## TL;DR

S3E is proposed as the first method to achieve complementary self-supervised state estimation from radar signal spectra and inertial data, leveraging a rotation-based cross-fusion technique to enhance spatial structural information under limited angular resolution.

## Background & Motivation

### State of the Field

Millimeter-wave radar offers unique reliability under adverse conditions (fog/rain/snow), yet existing approaches face the following challenges:

**Sparse point clouds**: Point clouds extracted by CFAR detectors are sparse and contaminated by ghost points.

**Multipath effects**: High reflective power generates false-positive "ghost points."

**Limited angular resolution**: Single-chip radar antennas have a small antenna count.

Key motivations:

### Limitations of Prior Work

The richer range-azimuth spectrum (RAS) should replace sparse point clouds as the primary representation.

### Root Cause

Radar provides exteroceptive sensing (compensating IMU drift), while IMU provides proprioceptive sensing (distilling motion-consistent landmarks).

### Starting Point

The rotational component manifests as a linear shift along the azimuth axis in the RAS.

## Method

### Rotation-based Cross Fusion

A core finding is that the displacement of the dominant energy between consecutive RAS frames is determined by the rotational component. Shifting the peak power of frame $k$ linearly along the azimuth by the rotation angle $\vartheta$ aligns its peak with that of frame $k+1$.

The inter-frame rotation $\boldsymbol{q}_{k+1}^k$ is obtained via IMU pre-integration and used to augment the RAS:

$$\boldsymbol{M}_{k+1}^k = \text{Softmax}\left(-\frac{(\mathbf{1}\boldsymbol{\eta}^T - \vartheta\mathbf{11}^T - \boldsymbol{\eta}\mathbf{1}^T)^2}{\kappa}\right) \cdot \boldsymbol{M}_k$$

$$\boldsymbol{M}_{k+1}' = \boldsymbol{M}_{k+1}^k \oplus \boldsymbol{M}_{k+1}$$

### Consistent Landmark Extractor

A U-Net-based multi-head architecture is employed:
- **Location head**: Outputs detection scores $L \in \mathbb{R}^{H \times W}$ and extracts sub-pixel positions.
- **Score head**: Produces normalized confidence weights to suppress ghost point influence.
- **Descriptor head**: Generates 248-dimensional features for cross-frame landmark matching.

Sub-pixel positions are computed via Softmax weighting:
$$u_k = \sum_{(i,j) \in \mathcal{U}_k} u_{ij}[\text{Softmax}(L_{\mathcal{U}_k})]_{ij}$$

### Differentiable Velocity Estimation

An overdetermined system is established using the cosine constraint between the Doppler velocity of static landmarks and the vehicle velocity:

$$-\begin{bmatrix} v_1^r \\ \vdots \\ v_N^r \end{bmatrix} = \begin{bmatrix} \cos\alpha_1 & \sin\alpha_1 \\ \vdots & \vdots \\ \cos\alpha_N & \sin\alpha_N \end{bmatrix} \begin{bmatrix} v^R\cos\beta \\ v^R\sin\beta \end{bmatrix}$$

The system is solved via differentiable least squares: $(\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{B}$

### Self-Supervised Loss

Three constraints are jointly optimized:
- **Geometric constraint** $\mathcal{L}_1$: Landmark pairs satisfy the IMU transformation matrix.
- **Kinematic constraint** $\mathcal{L}_2$: Static landmarks satisfy the Doppler cosine constraint.
- **Velocity alignment** $\mathcal{L}_3$: Velocities observed via IMU and radar are consistent with the transformation-derived velocity.

$$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_3$$

## Key Experimental Results

### ColoRadar Dataset

### Main Results

| Method | longboard Trans.↓ | edgar_classroom Trans.↓ | outdoors Trans.↓ |
|--------|-------------------|------------------------|-----------------|
| EKF-RIO | - | 5.32 | 4.65 |
| PG-RIO | - | 2.61 | 7.67 |
| Milliego | 9.14 | 2.32 | 2.02 |
| **S3E** | **5.69** | best | best |

S3E achieves state-of-the-art or highly competitive performance across most scenarios.

### Generalization on Self-Collected Dataset

Evaluated on unseen scenes, S3E as a self-supervised method demonstrates superior generalization compared to the supervised Milliego.

## Highlights & Insights

1. **First radar spectrum + IMU fusion**: Bypasses sparse point clouds by directly utilizing the information-rich RAS.
2. **Self-supervised without localization ground truth**: Supervision is derived from complementary radar–IMU constraints.
3. **Rotation–spectrum correspondence**: Elegantly exploits the physical property that rotation manifests as a linear shift in the RAS.
4. **Differentiable velocity estimation**: End-to-end training is enabled by a differentiable solver from Doppler observations to ego-velocity.

## Limitations & Future Work

- Applicable only to single-chip low-resolution radar; 4D imaging radar may not require such complex processing.
- IMU pre-integration accuracy degrades in high-speed scenarios.
- The static landmark assumption may be insufficient in highly dynamic environments.
- Large-scale long-term SLAM evaluation has not been conducted.

## Related Work & Insights

- EKF-RIO, PG-RIO: Model-based radar odometry methods.
- Milliego: Supervised learning-based radar–IMU fusion.
- RadarHD: Radar spectrum enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first self-supervised radar spectrum + IMU fusion)
- Technical Depth: ⭐⭐⭐⭐⭐ (cross-fusion + differentiable velocity estimation + triple self-supervision)
- Experimental Thoroughness: ⭐⭐⭐⭐ (public + self-collected datasets)
- Value: ⭐⭐⭐⭐ (practical solution for navigation under adverse conditions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[ICCV 2025\] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)

</div>

<!-- RELATED:END -->
