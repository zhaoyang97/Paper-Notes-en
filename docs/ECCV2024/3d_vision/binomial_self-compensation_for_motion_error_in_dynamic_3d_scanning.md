---
title: >-
  [Paper Note] Binomial Self-compensation for Motion Error in Dynamic 3D Scanning
description: >-
  [ECCV 2024][3D Vision][Phase-shifting profilometry] This work proposes a binomial self-compensation (BSC) algorithm. By performing a weighted sum of motion-affected phase sequences based on binomial coefficients, the algorithm exponentially eliminates motion errors in four-step phase-shifting profilometry without requiring any intermediate variables, thereby achieving high-precision dynamic 3D scanning at the same frame rate as the camera.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Phase-shifting profilometry"
  - "dynamic 3D scanning"
  - "motion error compensation"
  - "binomial self-compensation"
  - "structured light"
date: 2026-05-08
content_hash: 91c101e0f35070a2
---

# Binomial Self-compensation for Motion Error in Dynamic 3D Scanning

**Conference**: ECCV 2024  
**arXiv**: [2404.06693](https://arxiv.org/abs/2404.06693)  
**Code**: [Available](https://github.com/GeyouZhang/BSC)  
**Area**: 3D Vision  
**Keywords**: Phase-shifting profilometry, dynamic 3D scanning, motion error compensation, binomial self-compensation, structured light

## TL;DR

This work proposes a binomial self-compensation (BSC) algorithm. By performing a weighted sum of motion-affected phase sequences based on binomial coefficients, the algorithm exponentially eliminates motion errors in four-step phase-shifting profilometry without requiring any intermediate variables, thereby achieving high-precision dynamic 3D scanning at the same frame rate as the camera.

## Background & Motivation

Phase-shifting profilometry (PSP) is widely favored in high-precision 3D scanning due to its high accuracy, robustness, and pixel-wise characteristics. However, the core assumption of PSP is that the measured object remains stationary. In dynamic measurements, this assumption is violated, and object motion leads to ripple-like errors in the reconstructed point cloud.

Existing motion compensation methods are mainly divided into three categories:

**Motion-induced phase shift estimation**: These methods estimate motion-induced phase shift values by analyzing the affected phase frames, but their performance is limited when motion information cannot be accurately predicted.

**Target tracking**: These methods track object positions using markers or feature matching, but they only apply to 2D motion perpendicular to the line of sight and perform poorly on textureless objects.

**Self-compensation**: These methods apply specific operators to images or phase frames to directly obtain error-free phases, but introducing non-pixel-wise operators destroys the pixel-wise advantage of PSP.

All existing methods rely on certain intermediate variables (such as motion phase shift values, rigid-body transformation matrices, or Hilbert/differential transform results). The **Core Problem** of this paper is: **Can motion errors be compensated using only the motion-affected phase sequence itself, without relying on any intermediate variables?**

## Method

### Overall Architecture

The BSC method consists of three core components:
1. **Near-axial Binocular Structured Light System**: Utilizes a primary camera, an auxiliary camera, and a projector. The baseline between the two cameras is extremely short to achieve the phase uniqueness constraint, while the primary camera-projector baseline is sufficiently long to guarantee reconstruction accuracy.
2. **Cyclic Projection Strategy**: High-frequency fringe patterns ($\pi/2$ phase shift) are cyclically projected to compute the phase frame-by-frame from continuously captured images.
3. **Binomial Self-Compensation**: Performs a weighted sum of consecutive phase frames based on binomial coefficients to automatically eliminate motion errors.

### Key Designs

#### 1. Near-axial Binocular Structured Light System

This system transfers the constraint on the camera-projector baseline in traditional SPU (Stereo Phase Unwrapping) to the camera-camera baseline. The uniqueness constraint ensures that the phase remains single-valued within the feasible stereo matching range of the auxiliary camera, thereby allowing the use of high-frequency fringes. In the practical system, the two cameras are placed closely together to achieve the shortest baseline.

#### 2. Mathematical Modeling of Motion Error

In cycling projection of fringe patterns with a $\pi/2$ phase shift, the $i$-th captured image frame contains an unknown phase shift caused by motion. For the motion-affected phase calculated by four-step PSP, the error comprises:
- **DC Component**: Manifests as point cloud latency, is independent of the phase, and does not generate ripples.
- **Harmonic Component**: Its frequency is twice that of the wrapped phase, serving as the root cause of ripple errors.

The error of three-step PSP contains both $\cos(2\phi)$ and $\sin(2\phi)$ terms, whereas the error of four-step PSP only contains the $\cos(2\phi)$ term, making four-step PSP easier to compensate.

#### 3. Core Principle of Binomial Self-Compensation

Key finding: The trigonometric coefficients of two adjacent phase frames exhibit opposite signs due to the $\pi/2$ phase shift. Summing consecutive phase frames pairwise yields higher-order difference terms in the harmonic coefficients, naturally forming the structure of Pascal's triangle.

After performing a weighted sum of $K+1$ consecutive phase frames based on $K$-th order binomial coefficients, the harmonic amplitude contains a factor of $2^{-(K+2)}$ and a $(K+1)$-th order difference term. Increasing $K$ yields two effects:
- The exponential factor causes the harmonic amplitude to decay exponentially.
- The higher-order difference terms approach zero.

Together, these two effects suppress the motion error exponentially.

#### 4. Phase Jump Processing

In the wrapped phase jump regions, a special addition operator is defined to handle ambiguity during phase sequence summation: when the difference between two phases is greater than $\pi$, they are directly averaged; otherwise, they are averaged and then added by $\pi$. The actual implementation is performed in a pyramid-like layer-by-layer summation.

### Loss & Training

BSC is a non-learning-based method and does not involve loss functions or training. The core hyperparameter is the binomial order $K$: $K=0$ represents the original phase, and larger $K$ yields higher accuracy but requires more frames. Experiments demonstrate that $K=4$ (8 frames) achieves an optimal balance between accuracy and efficiency.

## Key Experimental Results

### Main Results: Absolute Accuracy Comparison

| Method | Frame Count | Type | Mean Error ($\mu$m) |
|------|------|------|-------------|
| Traditional 4-step PSP | 4 | Baseline | 324.2 |
| HTC | 4 | Self-Compensation | Higher |
| $\mu$-FTP | 4 | Fourier Transform | Higher |
| PFD | 8 | Phase Frame Difference | Moderate |
| PFS | 5 | Phase Frame Summation | Moderate |
| **BSC (Ours)** | **8** | **Self-Compensation** | **54.78** |

Measurement conditions: flat plate distance 500 mm, motion range [400, 500] mm, speed [-150, 150] mm/s. At 88 mm/s, BSC reduces the error from 324.2 $\mu$m to 54.78 $\mu$m.

### Robustness in Depth-discontinuous Scenes

| Method | Pixel-wise | RMSE at Depth Jumps |
|------|--------|---------------|
| $\mu$-FTP | No | Significantly Larger |
| HTC | No | Significantly Larger |
| **BSC** | **Yes** | **Minimum** |

### Temporal Resolution

| Method | 3D Point Cloud Frame Rate |
|------|-----------|
| $\mu$-FTP | 30 fps |
| **BSC** | **90 fps (= Camera Frame Rate)** |

### Ablation Study

- $K=0$: Original phase, maximum motion error.
- $K=1 \sim 4$: Error decreases exponentially, and $K=4$ reaches a practical balance.
- Applicable speed range: Based on the small phase-shift assumption, residual errors rise as inter-frame errors increase, which can be mitigated by reducing the fringe frequency.

### Key Findings

1. The error distribution after BSC compensation follows a normal distribution, indicating that motion errors are almost completely eliminated.
2. BSC simultaneously maintains two key characteristics: pixel-wise accuracy and frame-by-frame cyclic reconstruction, which is achieved for the first time among existing methods.
3. The error of four-step PSP only contains the $\cos(2\phi)$ term, which is easier to compensate than three-step PSP.

## Highlights & Insights

1. **Mathematical Elegance**: Utilizes the properties of Pascal's triangle and binomial coefficients, transforming motion compensation into a concise weighted sum.
2. **Plug-and-Play**: BSC can serve as an enhanced version of the traditional four-step PSP, directly applicable to dynamic scanning.
3. **Best of Both Worlds**: Achieves both pixel-wise precision and frame-level cyclicity for the first time.
4. **Deployment-Friendly**: Only requires adding an auxiliary camera and simple post-processing on top of the traditional PSP setup.

## Limitations & Future Work

1. **High-Contrast Textures**: BSC assumes low-frequency textures; objects with high-contrast textures may generate artifacts.
2. **Phase Unwrapping Errors**: Notable phase unwrapping errors may occur on complex curved surfaces during block-matching correspondence.
3. **Speed Range Limitations**: Based on the assumption of small phase shifts, high-speed objects require reducing the fringe frequency.
4. **System Complexity**: Requires a hardware system composed of binocular cameras and a projector.

## Related Work & Insights

- Compared to HTC and $\mu$-FTP, the advantage of BSC lies in preserving pixel-wise characteristics.
- The cyclic projection strategy serves as the foundation for BSC to achieve frame-level cyclicity.
- The connection between binomial coefficients and finite differences provides a new perspective for motion error compensation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The binomial self-compensation idea is unique, and the mathematical derivation is elegant.
- Practicality: ⭐⭐⭐⭐⭐ — Plug-and-play, open-sourced, with clear prospects for industrial application.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensively validated through multiple sets of experiments, but lacks testing on large-scale complex scenes.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivation and rational experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)
- [\[CVPR 2026\] ReFlow: Self-correction Motion Learning for Dynamic Scene Reconstruction](../../CVPR2026/3d_vision/reflow_self-correction_motion_learning_for_dynamic_scene_reconstruction.md)
- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[ECCV 2024\] Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation](open_vocabulary_3d_scene_understanding_via_geometry_guided_self-distillation.md)

</div>

<!-- RELATED:END -->
