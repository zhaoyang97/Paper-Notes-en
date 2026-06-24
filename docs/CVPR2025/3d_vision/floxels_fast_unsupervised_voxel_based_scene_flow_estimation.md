---
title: >-
  [Paper Note] Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation
description: >-
  [CVPR 2025][3D Vision][Scene flow estimation] Floxels is proposed to replace MLPs with a simple voxel grid as an implicit representation of scene flow. Combined with multi-scan distance transform loss and cluster consistency constraints, it achieves 2nd place among unsupervised methods on the Argoverse 2 benchmark (behind EulerFlow) while reducing runtime from 24 hours to 10 minutes (60-140x speedup).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Scene flow estimation"
  - "voxel grid"
  - "unsupervised optimization"
  - "point clouds"
  - "autonomous driving"
date: 2026-05-08
content_hash: 932c2a8eee132907
---

# Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.04718](https://arxiv.org/abs/2503.04718)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Scene flow estimation, voxel grid, unsupervised optimization, point clouds, autonomous driving

## TL;DR
Floxels is proposed to replace MLPs with a simple voxel grid as an implicit representation of scene flow. Combined with multi-scan distance transform loss and cluster consistency constraints, it achieves 2nd place among unsupervised methods on the Argoverse 2 benchmark (behind EulerFlow) while reducing runtime from 24 hours to 10 minutes (60-140x speedup).

## Background & Motivation

1. **Background**: Scene flow estimation (estimating 3D motion fields from consecutive LiDAR scans) is dominated by two paradigms: (1) supervised methods which are fast but require a large amount of annotated data and are sensitive to domain shift; (2) optimization-based unsupervised methods (e.g., NSFP, EulerFlow) which are robust to domain shift but computationally expensive.
2. **Limitations of Prior Work**: EulerFlow yields excellent results but requires around 24 hours to process a single sequence (on a V100 GPU), making it impractical. NSFP/FNSF are lightweight but produce poor performance, suffering from "windmill artifacts" (motion predicted incorrectly in empty regions), occluded shadow regions biased towards near-point matching, and insufficient accuracy due to slow MLP convergence.
3. **Key Challenge**: MLPs as implicit representations converge slowly and suffer from "homogeneous motion" bias—gradients propagate to global parameters, making it difficult to capture local details. Meanwhile, matching point clouds across only two frames struggles with occlusion changes, leading to incorrect point correspondences.
4. **Goal**: (1) How to accelerate convergence while maintaining high quality? (2) How to address incorrect correspondences caused by occlusion? (3) How to eliminate windmill artifacts and errors in shadow regions?
5. **Key Insight**: Inspired by how replacing MLPs with voxel grids in NeRF significantly accelerates convergence, the same idea is applied to scene flow estimation. The framework is extended to multi-frame to handle occlusions, and cluster consistency is used to encourage neighboring points to exhibit similar motions.
6. **Core Idea**: Use a voxel grid to explicitly parameterize the 3D motion field in place of MLPs, combined with multi-scan distance transform constraints and a cluster consistency loss, achieving scene flow quality close to EulerFlow at an extremely low computational cost.

## Method

### Overall Architecture
The input is a sequence of multi-frame LiDAR point clouds (default of 5 frames), and the output is the 3D motion vector of each point in the reference frame. The method first constructs a 3D voxel grid covering the scene, where each grid vertex stores a learnable 3D flow vector. The scene flow of any point is obtained via trilinear interpolation of adjacent vertices. After applying the flow to points in the reference frame, matching errors are evaluated using precomputed multi-scan distance transforms. Concurrently, DBSCAN clustering is employed to encourage points within the same cluster to have consistent motion. The entire optimization is performed via gradient descent for at most 500 epochs with early stopping.

### Key Designs

1. **Voxel Grid replacing MLP**:

    - **Function**: Serves as the explicit parameterized representation of the 3D scene flow field.
    - **Mechanism**: A 3D grid covering the scene is constructed, in which each vertex learns a 3D flow vector $f_t \in \mathbb{R}^3$. The flow of any arbitrary point is interpolated trilinearly. Compared to MLPs, gradients on voxel grids only affect neighboring grid points, providing inherent local smoothness regularization. The default voxel size is set to 0.5 meters, and the learning rate is 0.05.
    - **Design Motivation**: MLPs converge slowly because the gradients of each point propagate to all parameters. Voxel grids localize gradients, enabling fast convergence and naturally eliminating windmill artifacts—empty regions receive no gradient signals, so their flows remain at the initialized zero values.

2. **Multi-Scan Distance Transform Loss**:

    - **Function**: Employs multiple neighboring scans to constrain flow estimation, resolving incorrect correspondences caused by occlusions.
    - **Mechanism**: Assuming constant velocity motion, reference frame points are projected to the precomputed distance transforms of preceding and succeeding $m$ frames using the estimated flow, and matching distance is calculated as: $\ell_d = \sum_{t \neq 0} \frac{\lambda(t)}{N} D(\mathcal{S}_0 + f_0 \Delta_t, S_t)$, where $\lambda(t) = 1/t^2$ decays the weights of distant frames. Distances exceeding 5 meters in the distance transform are truncated for robustness. Only $2(m-1)$ distance transforms need to be precomputed.
    - **Design Motivation**: Under two-frame settings, if a target point is occluded in the second frame, it cannot be matched correctly. Under a multi-frame scheme, even if a correspondence is missing in one frame, it is likely present in other adjacent frames, providing proper constraint signals.

3. **Cluster Consistency Loss**:

    - **Function**: Encourages spatially neighboring points to have similar motions to prevent incorrect associations between dynamic and static objects.
    - **Mechanism**: The reference point cloud is clustered using DBSCAN ($\epsilon=0.5$, min_points=4). This ensures over-segmentation (each object might correspond to multiple clusters). The loss then constrains flow consistency within each cluster: $\ell_C = \frac{1}{N}\sum_i ||f_t^i - f_{C_i}||_2$, where $f_{C_i}$ is the average flow of cluster $C_i$.
    - **Design Motivation**: Over-segmentation is safe (it does not cause catastrophic failures), whereas grouping two differently moving objects into the same cluster results in severe errors. This constraint effectively combats incorrect matching when dynamic objects are close to static ones.

### Loss & Training
- Final loss: $\ell = \lambda_d \ell_d + (2m-1)(\lambda_C \ell_C + \lambda_\gamma \gamma)$
- $\gamma = ||f_t||_2$ is the flow magnitude regularization, suppressing noisy flow in static regions.
- The $(2m-1)$ factor scales the clustering and regularization terms linearly with the number of frames to balance with the distance transform term.
- Adam optimizer is used with a learning rate of 0.05, running for at most 500 epochs with an early stopping patience of 250 steps.

## Key Experimental Results

### Main Results (Argoverse 2 Scene Flow Challenge Test Set)

| Method | Type | mdnEPE↓ | Runtime |
|------|------|---------|---------|
| NSFP | Unsupervised optimization | ~0.55 | ~63s/frame |
| FNSF | Unsupervised optimization | ~0.50 | ~21s/frame |
| EulerFlow | Unsupervised optimization | **Best** | ~24 hr/sequence |
| Flow4D | Supervised | Near optimal | Fast |
| **Floxels (13 frames)** | **Unsupervised optimization** | **2nd Place** | **~24 min/sequence** |

### Ablation Study (nuScenes mini, dynamic points)

| Configuration | EPE↓ | Acc5↑ | Acc10↑ | Description |
|------|------|-------|--------|------|
| Floxels (Full) | **0.085** | **0.537** | **0.833** | Full model |
| - flow norm | 0.084 | 0.528 | 0.833 | Flow regularization has minor impact |
| - cluster loss | 0.201 | 0.133 | 0.413 | Remove cluster loss, performance drops sharply |
| - cluster + flow norm | 0.206 | 0.123 | 0.401 | Remove both |

| Frames | EPE↓ | Acc5↑ | Time (s)↓ |
|------|------|-------|---------|
| 3 frames | 0.095 | 0.468 | 2.47 |
| 5 frames | 0.085 | 0.537 | 3.52 |
| 9 frames | 0.078 | 0.516 | 5.69 |
| 11 frames | 0.076 | 0.486 | 6.72 |

### Key Findings
- **Dominant contribution of cluster consistency loss**: Removing it causes EPE to surge from 0.085 to 0.201 and Acc5 to collapse from 0.537 to 0.133. This is the most critical component.
- **Voxel grids completely eliminate windmill artifacts**: MLPs predict incorrect flow (windmill shape) in empty regions without objects. Voxel grids naturally preserve zero flow in empty regions since no gradient signal is generated there.
- **Excellent speed vs. quality trade-off**: Floxels with 5 frames takes only 3.5 seconds per frame, which is 4.7x faster than FNSF and 14.4x faster than NSFP, while significantly exceeding both in accuracy. Compared to EulerFlow, it delivers a 60-140x speedup with closely comparable accuracy.
- **Robustness to voxel size**: EPE varies marginally within the 0.3m to 2.0m range, with only pedestrian categories decreasing slightly at 2.0m.

## Highlights & Insights
- **Analogy transfer from NeRF to scene flow**: The insight in NeRF that replacing MLPs with voxel grids (Plenoxels/DVGO) accelerates convergence is successfully transferred to scene flow estimation. The same "explicit vs. implicit" trade-off manifests in different tasks, proving it to be a general paradigm.
- **Over-segmentation is safer than under-segmentation**: The clustering strategy with over-segmentation is a key engineering insight—merging objects with different motions into the same cluster fails catastrophically, while over-segmentation at most loses some local consistency.
- **Time-conditioned MLP is unnecessary**: EulerFlow's time-conditioned MLP learns flow variations across timesteps but is extremely slow. Floxels demonstrates that a simple constant-velocity assumption combined with multi-scan constraints is sufficient, significantly simplifying the problem.

## Limitations & Future Work
- **Constant-velocity assumption**: The multi-scan loss assumes constant-velocity motion, which may be inaccurate for variable-speed scenarios such as acceleration, deceleration, or turning.
- **Voxel resolution constraints**: A 0.5m voxel size cannot capture the fine-grained motion of very small objects (e.g., pedestrian arm swings).
- **LiDAR-only application**: The framework is tailored for LiDAR point clouds and does not immediately adapt to RGB scene flow.
- **Potential extensions**: Introducing multi-resolution voxel grids to handle motion at different scales, using deformable voxels to adapt to non-uniform scene distributions, and extending to image-domain scene flow.

## Related Work & Insights
- **vs. EulerFlow**: EulerFlow implements a time-conditioned MLP + full sequence Euler integration for peak accuracy but requires 24 hours. Floxels secures comparable accuracy while running 60-140x faster with a voxel grid + 5-scan constant-velocity assumption, highlighting the utility of a "good enough, simple method".
- **vs. NSFP/FNSF**: NSFP uses MLPs to describe scene flow whereas FNSF uses distance transforms for speed. Floxels analyzes their failure modes (windmill artifacts, shadow-region bias, near-point bias) and addresses them systematically with a voxel grid + clustering + multi-scan structure.
- **vs. DifFlow3D (Supervised)**: Supervised approaches are rapid but rely heavily on training data and suffer from domain shifts. On Argoverse 2, Floxels (unsupervised) performs comparably or even superiorly to DifFlow3D without any training.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of replacing MLPs with voxels is inspired by NeRF, yet its application and thorough analysis in scene flow are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, various baselines, intensive ablations, computational speed analysis, and failure case visualizations.
- Writing Quality: ⭐⭐⭐⭐ In-depth analysis of issues, clear diagrams, and a cohesive narrative.
- Value: ⭐⭐⭐⭐⭐ Extremely practical—Floxels stands as the premier choice among current unsupervised methods for latency-sensitive scenarios like autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[NeurIPS 2025\] Flux4D: Flow-based Unsupervised 4D Reconstruction](../../NeurIPS2025/3d_vision/flux4d_flow-based_unsupervised_4d_reconstruction.md)
- [\[CVPR 2025\] SCFlow2: Plug-and-Play Object Pose Refiner with Shape-Constraint Scene Flow](scflow2_plug-and-play_object_pose_refiner_with_shape-constraint_scene_flow.md)
- [\[CVPR 2025\] Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](flow-nerf_joint_learning_of_geometry_poses_and_dense_flow_within_unified_neural_.md)
- [\[CVPR 2025\] ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)

</div>

<!-- RELATED:END -->
