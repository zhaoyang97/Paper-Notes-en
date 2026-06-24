---
title: >-
  [Paper Note] Deep Patch Visual SLAM
description: >-
  [ECCV 2024][3D Vision][Visual SLAM] Based on the DPVO visual odometry system, this work extends it into a complete SLAM system, DPV-SLAM, by introducing efficient proximity loop closure and classical loop closure mechanisms, achieving real-time, high-precision, and low-memory monocular visual SLAM on a single GPU.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Visual SLAM"
  - "Loop Closure Detection"
  - "Sparse Optical Flow"
  - "Deep Networks"
  - "Monocular Vision"
date: 2026-05-08
content_hash: c8385d1f6e5c19ee
---

# Deep Patch Visual SLAM

**Conference**: ECCV 2024  
**arXiv**: [2408.01654](https://arxiv.org/abs/2408.01654)  
**Code**: [https://github.com/princeton-vl/DPVO](https://github.com/princeton-vl/DPVO)  
**Area**: 3D Vision  
**Keywords**: Visual SLAM, Loop Closure Detection, Sparse Optical Flow, Deep Networks, Monocular Vision

## TL;DR

Based on the DPVO visual odometry system, this work extends it into a complete SLAM system, DPV-SLAM, by introducing efficient proximity loop closure and classical loop closure mechanisms, achieving real-time, high-precision, and low-memory monocular visual SLAM on a single GPU.

## Background & Motivation

**Background**: Deep-network-based visual SLAM (such as DROID-SLAM) has significantly surpassed traditional methods in accuracy and is widely used as a subsystem in downstream tasks like monocular depth estimation, view synthesis, and 3D human pose estimation.

**Limitations of Prior Work**:
   - **Huge VRAM Overhead**: Methods like DROID-SLAM require 24GB of VRAM because they need to store dense feature maps for all frames.
   - **Inability to Achieve Real-Time Performance on a Single GPU**: The front end and back end of deep SLAM compete for GPU resources. Since CUDA operations are essentially executed serially, the frame rate drops precipitously from 30Hz to less than 1Hz.
   - **Poor Cross-Domain Generalization**: Traditional methods work well indoors but fail outdoors, whereas deep learning methods exhibit the opposite behavior.

**Key Challenge**: Implementing loop closure detection in deep SLAM systems requires global optimization, which requires storing a large volume of deep features. This leads to linear growth in memory usage and blocks front-end inference.

**Goal**: Build a monocular deep SLAM system that runs efficiently on a single GPU and is robust across different domains.

**Key Insight**: DPVO replaces dense correspondence with sparse optical flow, drastically reducing the cost per frame. The authors observe that the direction of edges in the patch graph can be flipped arbitrarily without affecting the optimization, which allows clever control over which frames must store dense features.

**Core Idea**: Minimize feature storage through a unidirectional edge patch graph design, and combine proximity loop closures and classical loop closures into the same optimization to achieve real-time deep SLAM on a single GPU.

## Method

### Overall Architecture

DPV-SLAM is built upon the DPVO visual odometry system and introduces two loop closure detection mechanisms: (1) camera proximity-based loop closure, which detects revisited locations and optimizes them via global bundle adjustment; and (2) classical loop closure, based on image retrieval and pose graph optimization. Both mechanisms run on a single-process, single-GPU setup, where the former shares the scene graph with the odometry, and the latter runs in parallel on the CPU.

### Key Designs

1. **Patch Graph Scene Representation**:

    - **Function**: Represent the scene using sparse $p \times p$ patches instead of dense depth maps.
    - **Mechanism**: Each frame $i$ contains several patches $\mathbf{P}_{ik} = (\mathbf{x}, \mathbf{y}, \mathbf{1}, \mathbf{d})^T$, where $\mathbf{d}$ is the inverse depth estimation. The patches are connected to other frames via directed edges and reprojected as $\mathbf{P}'_{ikj} = \Pi[G_j^{-1} \cdot G_i \cdot \Pi^{-1}(\mathbf{P}_{ik})]$. The optimization objective is to minimize the reprojection error:
    $$\arg\min_{G,\mathbf{d}} \sum_i \sum_k \sum_j \|\Pi[G_j^{-1} \cdot G_i \cdot \Pi^{-1}(\mathbf{P}_{ik})] - \mathcal{I}_{ikj}\|^2_{\Sigma_{ikj}}$$
    - **Design Motivation**: Sparse patches drastically save storage and computation compared to dense optical flow.

2. **Proximity Loop Closure**:

    - **Function**: Detect when the camera revisits prior locations, insert long-range edges, and perform global bundle adjustment.
    - **Mechanism**: Leverage the property that edge directions in the patch graph can be flipped—the correlation operation for each edge $\mathbf{C}(u,v,\alpha,\beta) = \langle \mathbf{g}(u,v), \mathbf{f}(\mathbf{P}'(u,v) + \Delta_{\alpha\beta}) \rangle$ only requires storing the dense feature map of the target frame. Utilizing this, creating unidirectional edges pointing from old frame patches to new frames allows storing only the patch features of all historical frames permanently (approx. 0.6GB / 1K frames) without needing dense feature maps.
    - **Efficient Global Optimization**: Contributes a CUDA-accelerated block-sparse bundle adjustment implementation, mixing odometry factors and loop factors in the same optimization. Each loop closure detection takes only 0.1–0.18s, which is significantly faster than DROID-SLAM's 0.5–5s.
    - **Design Motivation**: Avoid the need for dual GPUs, leveraging the unique property that edge directions do not affect the optimization results to minimize storage.

3. **Classical Loop Closure**:

    - **Function**: Detect and correct scale drift through image retrieval and $Sim(3)$ pose-graph optimization.
    - **Mechanism**:
        - Uses dBoW2 for image retrieval (ORB features), where extraction and retrieval are performed in parallel on an independent CPU process.
        - Uses off-the-shelf keypoint detectors and matchers to estimate 2D correspondences, performs 3D point cloud alignment via RANSAC + Umeyama after triangulating depth, and estimates the 7-DoF drift $\Delta S^{loop}_{jk} \in Sim(3)$.
        - Optimizes the pose-graph objective: $\arg\min_{S_1,...S_N} \sum_i \|r_i\|^2 + \sum_{(j,k)} \|r_{jk}\|^2$, where the smoothness term is $r_i = \log_{Sim(3)}(\Delta S_{(i,i+1)}^{-1} \cdot S_i^{-1} \cdot S_{i+1})$, and the loop closure term is $r_{jk} = \log_{Sim(3)}(\Delta S^{loop}_{jk} \cdot S_j^{-1} \cdot S_k)$.
    - **Design Motivation**: Proximity detection is insufficient for loop closure when scale drift is present (e.g., in long outdoor sequences), thus requiring appearance-based detection as a complement.

### Loss & Training

- Inherits the training strategy of DPVO, trained solely on the TartanAir synthetic dataset.
- Employs a differentiable bundle adjustment layer to learn outlier rejection end-to-end by supervising camera poses.
- Keypoints are selected randomly (without a detector), which works surprisingly well.

## Key Experimental Results

### Main Results (EuRoC-MAV Dataset - ATE Metric)

| Method | Mean ATE↓ | Frame Rate (FPS) | VRAM |
|------|----------|-----------|-------------|
| DPV-SLAM | **0.024** | **50** | **5.0G** |
| DROID-SLAM | 0.022 | 20 | 20G |
| DPVO (odometry only) | 0.105 | 60 | 4.0G |
| GO-SLAM | 0.035 | 6.4 | 7.2G |

### KITTI Dataset (ATE Metric, Unit: meters)

| Method | Mean ATE↓ | FPS |
|------|----------|-----|
| DPV-SLAM++ | **25.76** | 39 |
| DPVO | 53.61 | 48 |
| DROID-SLAM | - (failed on multiple sequences) | 17 |
| LDSO | 22.42 | 49 |

### TUM-RGBD Dataset

| Method | Mean ATE↓ | Failed Sequences |
|------|----------|-----------|
| DPV-SLAM++ | 0.054 | 0 |
| DROID-SLAM | 0.038 | 0 |
| ORB-SLAM3 | - | 5/9 failed |
| DeFlowSLAM | 0.114 | 0 |

### Key Findings

- DPV-SLAM achieves accuracy comparable to DROID-SLAM on EuRoC (0.024 vs 0.022) while being 2.5 times faster and requiring only 1/4 of the VRAM.
- Compared to the base DPVO system, the error is reduced by a factor of 4 (0.105 $\rightarrow$ 0.024), with only a slight increase in speed and VRAM overhead.
- **Outstanding Cross-Domain Robustness**: Zero catastrophic failures, showing excellent performance in both indoor and outdoor environments, which is challenging for other methods to achieve.
- DROID-SLAM is excellent indoors but fails multiple times outdoors on KITTI; traditional methods show the opposite behavior, while DPV-SLAM++ achieves both.
- Proximity loop closure takes only 0.1–0.18s per run, whereas the DROID-SLAM back end requires 0.5–5s.

## Highlights & Insights

- **Insight on Edge Direction Flipping**: The direction of edges in the patch graph can be adjusted arbitrarily with virtually no impact on optimization results (since each factor constrains both source and target camera poses simultaneously). This is a unique property of the DPVO patch representation, which cleverly solves the VRAM issue.
- **Single-Process Single-GPU Design Philosophy**: Avoids the complexity of multi-GPU or multi-process systems while maintaining real-time performance, representing an elegant combination of engineering and algorithms.
- **Effectiveness of Random Keypoint Selection**: DPV-SLAM inherits DPVO's strategy of randomly selecting keypoints without utilizing a feature detector. Paradoxically, the points with the highest confidence are often located in low-texture regions.

## Limitations & Future Work

- Only monocular video is supported; stereo or inertial information is not utilized.
- Proximity loop closure has limited efficacy in the presence of severe scale drift, necessitating the complement of classical loop closure.
- Although generalization is good when trained on synthetic data, performance might not match specialized, fine-tuned methods in specific domains such as autonomous driving.
- Positioned as a systems paper, its core contributions lie in engineering optimization, with relatively limited novelty.

## Related Work & Insights

- **vs DROID-SLAM**: A sister system. DPV-SLAM replaces dense optical flow with sparse patches, significantly improving VRAM and speed, although its accuracy is slightly lower under optimal conditions.
- **vs DPVO**: DPV-SLAM is a loop-closure extension of DPVO, achieving a 4-fold error reduction, which demonstrates the critical value of loop closure detection for odometry systems.
- **vs ORB-SLAM3**: Traditional methods often fail indoors due to rapid camera motion but perform well outdoors on KITTI. DPV-SLAM++ achieves all-scene robustness by combining deep learning with classical methods.
- **vs GO-SLAM / DeFlowSLAM**: Other extensions based on DROID but with higher VRAM demands (7-8GB) and lower frame rates.

## Rating

- Novelty: ⭐⭐⭐ Primarily focused on engineering optimization and system integration; the insight on edge direction flipping is innovative, but the core idea originates from DPVO.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 datasets (indoor/outdoor), multiple comparisons on speed and memory, and evaluations against more than 10 methods.
- Writing Quality: ⭐⭐⭐⭐ The system paper has clear logic and precise problem definitions, and the project is fully open-sourced.
- Value: ⭐⭐⭐⭐ Holds high practical value for downstream applications requiring real-time camera poses, such as view synthesis and 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)
- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[ECCV 2024\] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [\[CVPR 2026\] Dynamic Visual SLAM using a General 3D Prior](../../CVPR2026/3d_vision/dynamic_visual_slam_using_a_general_3d_prior.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](../../ICCV2025/3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)

</div>

<!-- RELATED:END -->
