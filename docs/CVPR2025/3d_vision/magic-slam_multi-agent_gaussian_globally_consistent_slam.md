---
title: >-
  [Paper Note] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM
description: >-
  [CVPR 2025][3D Vision][Multi-Agent SLAM] This paper proposes MAGiC-SLAM, a multi-agent SLAM system based on a rigidly deformable 3D Gaussian scene representation. Through a novel tracking and map fusion mechanism, and DinoV2-based loop closure detection, it achieves a processing speed 24 times faster than CP-SLAM with 7 times lower GPU usage, along with more accurate trajectory estimation and high-fidelity novel view rendering.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-Agent SLAM"
  - "3D Gaussian Splatting"
  - "Loop Closure Detection"
  - "Map Fusion"
  - "Novel View Synthesis"
  - "DinoV2"
date: 2026-05-08
content_hash: 46a53277f4f42636
---

# MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM

**Conference**: CVPR 2025  
**arXiv**: [2411.16785](https://arxiv.org/abs/2411.16785)  
**Code**: None  
**Area**: Autonomous Driving / 3D Vision  
**Keywords**: Multi-Agent SLAM, 3D Gaussian Splatting, Loop Closure Detection, Map Fusion, Novel View Synthesis, DinoV2

## TL;DR

This paper proposes MAGiC-SLAM, a multi-agent SLAM system based on a rigidly deformable 3D Gaussian scene representation. Through a novel tracking and map fusion mechanism, and DinoV2-based loop closure detection, it achieves a processing speed 24 times faster than CP-SLAM with 7 times lower GPU usage, along with more accurate trajectory estimation and high-fidelity novel view rendering.

## Background & Motivation

### Background

**Background**: SLAM systems with novel view synthesis (NVS) capabilities are widely used in augmented reality, robotics, and autonomous driving. 3D Gaussian Splatting (3DGS) has shown advantages in speed and rendering quality over NeRF in single-agent SLAM. However, the multi-agent NVS-SLAM domain remains mostly unexplored.

**Limitations of Prior Work**:

### Limitations of Prior Work

**Limitations of Prior Work**: Existing multi-agent NVS-SLAM methods like CP-SLAM utilize distributed neural scene representations, resulting in extremely slow processing speeds (tracking at 3.36s/frame) and supporting only up to two agents.

### Key Challenge

**Key Challenge**: Neural scene representations inherently do not support rigid body transformations, making effective map correction and fusion impossible.

### Core Idea

**Core Idea**: Existing methods exhibit extremely poor rendering quality on real-world data (e.g., CP-SLAM achieves only ~10 dB PSNR on AriaMultiagent) and fail to build accurate maps.

### Supplementary Notes

**Supplementary Notes**: In multi-agent scenarios, trajectory drift and differences in cross-agent observations make globally consistent reconstruction particularly challenging.

**Key Challenge**: Multi-agent SLAM must simultaneously achieve high-precision tracking, a globally consistent map, and high-quality novel view synthesis, but existing neural representation methods fail to meet the requirements in speed, scalability, and rendering quality.

**Goal**: Build an efficient SLAM system that supports an arbitrary number of agents and enables globally consistent 3D reconstruction and high-fidelity novel view synthesis.

**Key Insight**: Leveraging 3D Gaussians as the scene representation (which inherently support rigid body transformations), combined with a submap strategy, a hybrid tracking paradigm, and foundation-model-based loop closure detection.

**Core Idea**: 3D Gaussians naturally support rigid body transformations, enabling simple and efficient map correction and fusion in multi-agent environments.

## Method

### Overall Architecture

MAGiC-SLAM adopts a centralized architecture: each agent independently processes RGB-D streams to perform local mapping and tracking, and sends submaps and image features to a central server. The server is responsible for loop closure detection, pose graph optimization, and global map fusion.

### Key Designs

1. **Submap Strategy and Efficient Mapping**: Each agent segments the scene into submaps based on a fixed number of frames (50 frames for Replica, 20 frames for Aria), with each submap represented by 3D Gaussians. A key innovation is that only Gaussians with non-zero rendered opacity in the current camera frustum are maintained in active memory, which significantly reduces disk storage (by over 3x compared to Gaussian-SLAM) and accelerates map fusion. Spherical harmonics are not optimized to decrease memory footprint and improve tracking precision.

2. **Hybrid Implicit Tracking Mechanism**: This combines the advantages of both frame-to-frame and frame-to-model paradigms. It first initializes the relative pose using deterministic frame-to-frame dense registration (multi-scale ICP + joint color/geometric residuals), then refines it using a frame-to-model re-rendering loss (with soft alpha masks and error masks). It is found that implicit tracking is more accurate than explicit tracking when robustly initialized.

3. **DinoV2-Based Loop Closure Detection and Loop Closing**: DinoV2 ViT-small is used as the feature extractor instead of NetVLAD, leveraging its generalization capability from large-scale pre-training. Loop closure constraints are estimated through local FPFH global registration + ICP refinement, registering with the input point clouds rather than Gaussian means (due to high Gaussian distribution discrepancy across different agents). After pose graph optimization, Gaussian parameters (means and covariances) of submaps are directly updated via rigid body transformations.

## Key Experimental Results

- **Tracking Accuracy**: Average ATE RMSE of 0.25 cm on ReplicaMultiagent, which is a 2.6x improvement over CP-SLAM (0.95 cm); 0.90 cm versus 3.03 cm for CP-SLAM on the real-world AriaMultiagent dataset.
- **Rendering Quality**: PSNR of 34.26 dB on ReplicaMultiagent versus CP-SLAM's 22.71 dB (an 11.5 dB gain); 22.61 dB versus 9.06 dB on AriaMultiagent.
- **Speed**: Tracking takes 0.69s per frame versus CP-SLAM's 3.36s (4.9x faster), and mapping takes 0.71s versus 16.95s (24x faster).
- **Resources**: Peak GPU of only 1.12 GiB versus 7.70 GiB (an 86% reduction), and map fusion takes 167s versus 1448s.
- **Loop Closure Detection**: DinoV2 reduces ATE on AriaMultiagent by 34% (0.90 vs 1.36) compared to NetVLAD with faster inference speed (0.028s vs 0.045s).

### Key Findings

- Pose initialization is crucial for implicit tracking (without initialization, ATE increases from 0.37 cm to 0.82 cm), whereas it can sometimes be counter-productive for explicit tracking.
- Registration using input point clouds performs far better than using Gaussian means (0.98 cm vs 5.62 cm ATE).
- Loop closure detection significantly improves trajectory consistency in multi-agent scenarios (ATE dropped from 0.50 cm to 0.25 cm).

## Highlights & Insights

- **Inherent Advantage of 3DGS**: Native support for rigid body transformations makes map updates after pose correction remarkably simple and efficient—only requiring translation and rotation of means, and rotation of covariances, without the need for retraining.
- **Exquisite Submap Caching Strategy**: Caching only unseen Gaussians avoids expensive intersection checks at submap boundaries while drastically reducing storage footprints.
- **Coarse-to-Fine Map Fusion**: Performing coarse stitching followed by joint optimization and pruning effectively eliminates visual and geometric artifacts at submap boundaries.
- The system imposes no hard limit on the number of agents, constrained only by server capacity.

## Limitations & Future Work

- The current system runs slightly above 1 FPS because implicit tracking requires multiple iterations to converge, which is still some distance from real-time applications.
- It only supports RGB-D inputs and does not explore pure RGB or monocular scenarios.
- Pose graph optimization is performed in one go after execution finishes, without supporting online incremental loop closure correction.
- Dynamic objects are not handled, requiring manual filtering of segments without dynamic objects in the AriaMultiagent dataset.
- The centralized architecture has limited applicability in network-restricted environments, and communication overhead is not compared against distributed architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots](mne-slam_multi-agent_neural_slam_for_mobile_robots.md)
- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[CVPR 2025\] Volumetrically Consistent 3D Gaussian Rasterization](volumetrically_consistent_3d_gaussian_rasterization.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)

</div>

<!-- RELATED:END -->
