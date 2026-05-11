---
title: >-
  [Paper Note] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion
description: >-
  [ICCV 2025][Autonomous Driving][Structure-from-Motion] This paper proposes MGSfM, a global Structure-from-Motion (SfM) framework for multi-camera systems. By exploiting multi-camera rigid constraints through two core mod…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Structure-from-Motion"
  - "multi-camera systems"
  - "global motion averaging"
  - "rotation averaging"
  - "translation averaging"
  - "3D reconstruction"
date: 2026-05-08
content_hash: 1c67fafbe213db2c
---

# MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion

**Conference**: ICCV 2025
**arXiv**: [2507.03306](https://arxiv.org/abs/2507.03306)
**Code**: [3dv-casia/MGSfM](https://github.com/3dv-casia/MGSfM)
**Area**: Autonomous Driving
**Keywords**: Structure-from-Motion, multi-camera systems, global motion averaging, rotation averaging, translation averaging, 3D reconstruction, autonomous driving

## TL;DR

This paper proposes MGSfM, a global Structure-from-Motion (SfM) framework for multi-camera systems. By exploiting multi-camera rigid constraints through two core modules — **Decoupled Multi-camera Rotation Averaging (DMRA)** and **Multi-camera Geometry driven Position estimation (MGP)** — MGSfM achieves accuracy comparable to or better than incremental SfM on large-scale scenes while being approximately 10× faster.

## Background & Motivation

### Importance of Multi-Camera Systems in Autonomous Driving

Autonomous driving and robotics platforms increasingly adopt multi-camera systems (stereo cameras, surround-view rigs, etc.) for environment perception. The **rigid relative pose constraints** among these cameras — i.e., the relative pose between cameras on the same rigid body remains constant across all frames — constitute a strong geometric prior that can substantially improve the accuracy and robustness of SfM.

### Limitations of Prior Work

- **Incremental SfM** (e.g., COLMAP, MCSfM): Cameras and 3D points are added frame by frame, yielding high accuracy but at large computational cost. Moreover, **scale drift** causes errors to accumulate monotonically with sequence length, particularly in large-scale scenes without loop closure.
- **Global SfM** (e.g., GLOMAP): All camera poses are estimated jointly, yielding more uniform error distribution, but robustness is insufficient — it is susceptible to outlier matches, especially when relying solely on camera-to-point constraints.
- **Multi-camera global methods** (e.g., MMA): Multi-camera constraints are used for translation averaging, but only relative translation directions are exploited, discarding the rich information embedded in feature tracks.

### Core Problem

How can a global SfM framework simultaneously exploit: (1) multi-camera rigid constraints to resolve scale ambiguity; and (2) complementary camera-to-camera and camera-to-point constraints to improve robustness?

## Method

MGSfM is built upon GLOMAP, taking a COLMAP database as input and producing COLMAP-compatible sparse reconstruction outputs. The framework consists of two core stages.

### 1. Decoupled Multi-camera Rotation Averaging (DMRA)

Conventional rotation averaging solves all camera rotations jointly. However, in multi-camera systems, the relative rotations among cameras within the same rigid unit should remain consistent. DMRA adopts a **hierarchical strategy**:

1. **Intra-rig rotation estimation**: Redundant multi-frame observations are exploited to robustly estimate the relative rotations $\{R_{ij}^{rig}\}$ among cameras within the same rigid unit via median rotation averaging.
2. **Global rig rotation estimation**: Each multi-camera unit is treated as a single rigid-body node. The results from step 1 are used to convert all inter-camera relative rotations into inter-rig relative rotations, which are then solved via standard global rotation averaging.
3. **Back-projection**: Per-camera global rotations are recovered from the global rig rotations and the intra-rig relative rotations.

The advantage of this decoupled strategy is twofold: intra-rig rotation estimation benefits from multi-frame redundancy and is robust to outliers; the global rotation averaging problem is substantially reduced in scale (the number of nodes equals the number of rigid units rather than the total number of cameras).

### 2. Multi-camera Geometry Driven Position Estimation (MGP)

Translation averaging is the core challenge in global SfM, as relative translations carry only directional information (no scale). Multi-camera systems provide natural scale constraints. MGP integrates two complementary types of constraints.

#### Camera-to-Camera Constraints (Relative Translations)

Relative translation directions $\hat{t}_{ij}$ decomposed from essential matrices between frame pairs provide directional constraints on inter-frame motion. In multi-camera systems, overlapping image pairs from different cameras within the same frame enable recovery of relative scale between rig units.

#### Camera-to-Point Constraints (Feature Tracks)

Each feature track associates a 3D point with multiple camera centers, providing additional angular constraints. Feature tracks are abundant and offer wide coverage, but are susceptible to outlier contamination.

#### Two-Stage Optimization

1. **Initialization stage**: Camera-to-camera relative translation constraints are used to solve for an initial estimate of camera positions and 3D points via a **convex distance-based objective function**. Convex optimization guarantees convergence to the global optimum, avoiding local minima.
2. **Refinement stage**: Starting from the initialized solution, an **unbiased non-bilinear angle-based objective** that fuses camera-to-camera and camera-to-point constraints is constructed, and all camera positions and 3D points are refined via nonlinear optimization.

Key properties of the angle-based objective:

- **Unbiasedness**: Invariant to the distance between 3D points and cameras, preventing nearby points from dominating optimization.
- **Non-bilinearity**: Unlike bilinear formulations, no auxiliary variables are required, leading to more efficient optimization.
- Improved robustness to outlier feature tracks.

### 3. Overall Pipeline

1. Feature extraction and matching (COLMAP)
2. DMRA for global rotation estimation
3. MGP for global translation estimation
4. Bundle Adjustment for joint refinement

## Key Experimental Results

### KITTI Odometry (Outdoor Stereo Camera)

| Method | Type | Multi-Camera | Key Performance |
|--------|------|-------------|-----------------|
| COLMAP | Incremental | ✗ | Baseline; significant scale drift |
| GLOMAP | Global | ✗ | Uniform error distribution but high overall error |
| MMA | Global | ✓ | Uses relative translations only; better than GLOMAP |
| MCSfM | Incremental | ✓ | High accuracy but slow |
| **MGSfM** | **Global** | **✓** | **Best accuracy; far faster than incremental methods** |

- On the challenging sequence 08 (no complete loop closure), MGSfM produces trajectories closest to ground truth.
- On sequence 01 (sparse features, high outlier ratio), methods relying solely on feature tracks tend to fall into local minima; MGSfM's hybrid strategy demonstrates robust performance.

### KITTI-360 (Large-Scale Outdoor Multi-Camera)

- MGSfM is approximately **10× faster** than MCSfM.
- Reconstruction quality surpasses COLMAP, GLOMAP, and MMA.
- Intra-rig camera pose estimation accuracy is comparable to MCSfM results after bundle adjustment, validating the robustness of DMRA.

### ETH3D-SLAM (Indoor Scenes)

| Scene | GLOMAP AUC@0.1m | MGSfM AUC@0.1m | GLOMAP Time (s) | MGSfM Time (s) |
|-------|-----------------|----------------|-----------------|----------------|
| ceiling_1 | 18.5 | **59.7** | 240 | **34** |
| desk_3 | 86.4 | **95.8** | 462 | **89** |
| large_loop_1 | 70.0 | **87.9** | 250 | **30** |
| motion_1 | 25.7 | **46.5** | 885 | **158** |
| reflective_1 | 79.1 | **91.3** | 3239 | **335** |
| repetitive | 66.9 | **91.4** | 90 | **23** |

MGSfM substantially outperforms GLOMAP on all scenes, with accuracy gains of 10–40 percentage points and speedups of 4–10×.

### In-House Datasets

- **CAMPUS** (29,000+ images, 520,000 m²): MGSfM takes 66 minutes vs. COLMAP 1,588 minutes, GLOMAP 580 minutes, and MCSfM 401 minutes. COLMAP, GLOMAP, and MMA produce incorrect reconstructions; MCSfM yields locally erroneous structures.
- **STREET** (surround-view camera, 12,000+ images, 500,000 m²): Only MGSfM correctly reconstructs the road trajectory.

### Ablation Study

Six configurations are compared on KITTI Odometry (relative translations only / feature tracks only / hybrid, bilinear / non-bilinear):

- The hybrid strategy consistently outperforms single-constraint methods, with particularly pronounced advantages on sequences with high outlier ratios.
- The non-bilinear objective achieves better robustness and accuracy than the bilinear formulation given proper initialization.
- MGSfM (Hybrid-Non-Bilinear) is the fastest among all configurations except "relative translations only," demonstrating that high-quality initialization is critical to efficiency.

## Highlights & Insights

1. **Hierarchical decoupled rotation averaging**: Decomposing multi-camera rotation estimation into intra-rig and inter-rig levels leverages multi-frame redundancy while reducing the scale of the global optimization problem — an elegant engineering design.
2. **Hybrid-constraint translation averaging**: Camera-to-camera constraints (sparse but robust) and camera-to-point constraints (dense but outlier-prone) are unified within an angle-based framework, with two-stage optimization (convex initialization + nonlinear refinement) balancing robustness and accuracy.
3. **Multi-camera rigid constraints resolve scale ambiguity**: This is the most important geometric prior of multi-camera systems — redundant observations from multiple cameras on the same rigid body constrain the relative scale between adjacent frames, fundamentally alleviating scale drift in monocular/stereo SfM.
4. **High practical value**: Built upon GLOMAP with COLMAP-compatible input/output, the code is open-sourced and supports both single-camera and multi-camera configurations.

## Limitations & Future Work

1. **Single rig configuration only**: The current implementation assumes all frames originate from the same multi-camera system; mixed configurations with heterogeneous camera rigs are not supported (the authors note multi-rig support as future work on GitHub).
2. **Reliance on COLMAP feature matching**: Feature extraction and matching still follow the conventional COLMAP pipeline; learned features (e.g., SuperPoint + LightGlue) are not integrated, and matching quality in texture-poor or repetitive-texture scenes remains a bottleneck.
3. **Sequential image assumption**: The framework is primarily designed for temporally ordered image sequences (e.g., autonomous driving video), and its applicability to unordered image collections has not been thoroughly validated.
4. **Limited gains on small indoor scenes**: Although ETH3D-SLAM results are competitive, the advantages of multi-camera rigid constraints are less pronounced in small-scale scenes compared to large-scale environments.
5. **No comparison with end-to-end learning-based methods**: No evaluation against recent deep learning-based SfM approaches (e.g., DUSt3R, MASt3R) is provided.

## Related Work & Insights

- **COLMAP** [Schönberger & Frahm, CVPR 2016]: The standard incremental SfM method; high accuracy but slow.
- **GLOMAP** [Pan et al., ECCV 2024]: Global SfM; the codebase upon which MGSfM is built.
- **MMA** [Cui et al.]: Global translation averaging using multi-camera geometry; inspires MGSfM's hybrid strategy.
- **MCSfM**: Incremental multi-camera SfM; the primary accuracy baseline for MGSfM.
- **HETA** [Tao et al., CVPR 2024]: A prior work by the same authors, revisiting global translation estimation using feature tracks.

**Research Implications**: The rigid-constraint paradigm of multi-camera systems is generalizable to other multi-sensor fusion scenarios. The two-stage optimization pattern (convex initialization + non-convex refinement) is a general-purpose paradigm for non-convex optimization problems.

## Rating

- Novelty: ⭐⭐⭐⭐ — The hierarchical decoupled rotation and hybrid translation global framework is a genuinely novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple datasets, ablation studies, and comprehensive quantitative and qualitative evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with complete mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ — Open-sourced and COLMAP-compatible; directly applicable to autonomous driving 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [\[ICCV 2025\] Language Driven Occupancy Prediction (LOcc)](language_driven_occupancy_prediction.md)
- [\[ICCV 2025\] SAM4D: Segment Anything in Camera and LiDAR Streams](sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)

</div>

<!-- RELATED:END -->
