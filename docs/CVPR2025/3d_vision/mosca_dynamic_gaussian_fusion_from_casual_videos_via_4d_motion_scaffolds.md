---
title: >-
  [Paper Note] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds
description: >-
  [CVPR 2025][3D Vision][4D reconstruction] Proposes the 4D Motion Scaffold (MoSca) representation, which compactly encodes scene motion using a sparse 6-DoF trajectory graph. Combined with 2D foundation model priors and physical regularization, it achieves fully automatic 4D scene reconstruction from pose-free, casual monocular videos.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D reconstruction"
  - "dynamic scene"
  - "Gaussian splatting"
  - "motion scaffold"
  - "deformation graph"
  - "pose-free"
date: 2026-05-08
content_hash: d72b3a747dbd5007
---

# MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds

**Conference**: CVPR 2025  
**arXiv**: [2405.17421](https://arxiv.org/abs/2405.17421)  
**Code**: [https://www.cis.upenn.edu/~leijh/projects/mosca](https://www.cis.upenn.edu/~leijh/projects/mosca)  
**Area**: 3D Vision  
**Keywords**: 4D reconstruction, dynamic scene, Gaussian splatting, motion scaffold, deformation graph, pose-free

## TL;DR

Proposes the 4D Motion Scaffold (MoSca) representation, which compactly encodes scene motion using a sparse 6-DoF trajectory graph. Combined with 2D foundation model priors and physical regularization, it achieves fully automatic 4D scene reconstruction from pose-free, casual monocular videos.

## Background & Motivation

**Background**: Novel view synthesis of dynamic scenes is a key capability for building AGI datasets, spatial computing content creation, and embodied AI. Robust 4D reconstruction from monocular casual videos (the most common data format) is highly challenging because multi-view stereo cues are extremely limited.

**Limitations of Prior Work**:
1. **per-frame methods** suffer from insufficient information: local depth warping methods fail directly under large off-axis test views, resulting in large missing regions.
2. **Local temporal fusion methods** (e.g., PGDVS, Gaussian Marbles) only fuse small temporal windows, leaving occluded regions uncompleted.
3. **Dense Gaussian methods** (e.g., 4D-GS) rely heavily on strong multi-view stereo cues, failing in monocular casual video scenarios.
4. **Over-parameterized deformation representations**: Most methods use MLPs to learn the deformation field, which creates an excessively large solution space and makes the optimization prone to degeneration.
5. **Reliance on external pose estimation**: Most methods require COLMAP to pre-estimate camera poses, which often fails in dynamic scenes.

**Core Motivation**: To leverage strong priors from 2D foundation models combined with physics-inspired low-rank motion representations to globally fuse full-sequence observations, building a fully automatic system from pose-free RGB videos to renderable 4D scenes.

## Method

### Overall Architecture

The four-step system pipeline:
1. **2D Foundation Model Inference**: Obtains depth estimation, long-term 2D trajectories, and foreground/background separation.
2. **Camera Initialization**: Solves focal length and camera poses using bundle adjustment based on static tracklets.
3. **MoSca Geometric Optimization**: Lifts 2D priors to 3D and optimizes the motion graph with ARAP regularization.
4. **Photometric Optimization**: Fuses Gaussians across all timesteps to the query time, and optimizes via Gaussian Splatting rendering.

### Key Designs

#### Module 1: Motion Scaffold (MoSca) Deformation Representation

Core Innovation — Encoding motion through a sparse, structured trajectory graph:
- **Graph Nodes** $v^{(m)}$: Each node is a 6-DoF trajectory $[Q_1^{(m)}, ..., Q_T^{(m)}]$, with a control radius $r^{(m)}$.
- **Graph Topology**: Uses curve distance (the maximum spatial-temporal distance between trajectories) to construct a KNN graph, naturally handling topological changes (e.g., doors opening without connecting the door and the wall).
- **Deformation Interpolation**: Dual Quaternion Blending (DQB) interpolates multiple rigid body transformations on the $SE(3)$ manifold, avoiding artifacts from linear blend skinning.
- **Weight Computation**: RBF kernel $w_i(x,t) = \exp(-\|x - t_{t}^{(i)}\|^2 / 2r^{(i)})$.

The number of MoSca nodes $M$ is far fewer than the number of points $N$ (as shown in Tab. 7), leveraging the physical prior of real-world motion being low-rank and smooth.

#### Module 2: 2D Foundation Model Prior Fusion and Camera Solving

- **Depth**: Uses pre-trained monocular depth estimation (e.g., Metric3D, DepthAnything).
- **Long-term Trajectories**: Uses BootsTAPIR/CoTracker to obtain dense 2D pixel trajectories.
- **Dynamic/Static Segmentation**: Separates foreground and background using epipolar error maps calculated via RAFT optical flow.
- **Camera BA**: Filters static trajectories with low epipolar errors to jointly optimize camera poses and focal length, incorporating reprojection error $\mathcal{L}_{proj}$ and scale-invariant depth alignment loss $\mathcal{L}_z$.

#### Module 3: Global Gaussian Fusion and Rendering

- **Full-timestep Fusion**: Initializes Gaussians from depth points back-projected from all timesteps, which are then fused after being transformed to the query timestep via the MoSca deformation field.
- **Learnable Skinning Correction**: Each Gaussian learns an additional skinning weight correction $\Delta w_j$.
- **Node Control**: Similar to the densification/pruning strategy in 3DGS — adding nodes in high tracking-loss gradient regions and pruning low-contribution nodes.

### Loss & Training

**Bundle Adjustment**: $\mathcal{L}_{BA} = \lambda_{proj}\mathcal{L}_{proj} + \lambda_z\mathcal{L}_z$

**Geometric Optimization**: $\mathcal{L}_{geo} = \lambda_{arap}\mathcal{L}_{arap} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{vel}\mathcal{L}_{vel}$
- ARAP Loss: Preserves local distances and local coordinates between neighbors.
- Velocity/Acceleration Regularization: Ensures temporal smoothness.

**Photometric Optimization**: $\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{dep}\mathcal{L}_{dep} + \lambda_{track}\mathcal{L}_{track} + \lambda_{arap}\mathcal{L}_{arap} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{vel}\mathcal{L}_{vel}$

where $\mathcal{L}_{track}$ supervises 2D trajectory consistency by rendering XYZ coordinate maps.

## Key Experimental Results

### Main Results

**DyCheck Dataset (Most challenging, average over 7 scenes)**:

| Method | Pose | mPSNR↑ | mSSIM↑ | mLPIPS↓ |
|------|------|--------|--------|---------|
| HyperNeRF | Known | 16.81 | 0.569 | 0.332 |
| Shape-of-Motion | Known | 17.32 | 0.598 | 0.296 |
| **MoSca** | **Known** | **19.32** | **0.706** | **0.264** |
| RobustDynRF | Unknown | 17.10 | 0.534 | 0.517 |
| **MoSca** | **Unknown** | **18.84** | **0.676** | **0.289** |
| **MoSca (w/ focal)** | **Unknown** | **19.02** | **0.683** | **0.279** |

**NVIDIA Dataset**: PSNR 26.72, LPIPS 0.070, outperforming all baseline methods.

**Camera Pose Accuracy**: Sintel ATE 0.090 (outperforming DROID-SLAM and MonST3R), TUM-dynamics ATE 0.031 (SOTA).

### Ablation Study

| Component | mPSNR | mSSIM | mLPIPS |
|------|-------|-------|--------|
| Full model | 19.32 | 0.706 | 0.264 |
| No geometric optimization | 18.85 | 0.693 | 0.287 |
| No multi-level topology | 19.14 | 0.701 | 0.270 |
| No dual quaternion blending | 19.18 | 0.701 | 0.276 |
| Only fuse 4 neighboring frames | 16.96 | 0.663 | 0.344 |
| Only fuse 8 neighboring frames | 17.26 | 0.664 | 0.346 |

### Key Findings

1. **Global Fusion is Crucial**: Fusing only a 4-frame neighborhood vs. the full sequence results in a 2.36 dB gap in mPSNR, validating the core value of global aggregation.
2. **Geometric Optimization Stage** contributes significantly (+0.47 dB), with the ARAP prior effectively propagating motion information to occluded regions.
3. **DQB Outperforms Linear Blend Skinning**: Interpolating on the $SE(3)$ manifold avoids the degeneration associated with linear blending.
4. **Correspondence Tracking**: The reconstructed MoSca tracking accuracy (PCK-T 0.824) outperforms the original BootsTAPIR (0.779), showing that safety-guided optimization improves the initial priors.
5. **Pose-free Setting** results in only about a 0.5 dB loss, demonstrating the system's robustness to unknown camera parameters.

## Highlights & Insights

1. **Elegant Combination of Physical Priors and Learned Priors**: MoSca's ARAP regularization encodes a "rigidity-oriented" motion prior, and the 2D foundation models provide initialization — the two are complementary.
2. **Full-Temporal Global Fusion**: Unlike per-frame or sliding window methods, it truly achieves information aggregation across all frames; occluded regions in a single frame can be completed from other frames.
3. **System Completeness**: A fully automatic pipeline from raw RGB video to renderable 4D scenes, requiring no COLMAP or any external tool.
4. **In-the-wild Generalization**: Works robustly on movie clips, web videos, and SORA-generated videos.

## Limitations & Future Work

1. MoSca node initialization depends heavily on the quality of 2D trackers; heavy occlusion or rapid motion may cause tracklet breaks.
2. Static/dynamic segmentation is based on epipolar error thresholds, which may fail in scenes with minimal camera motion (e.g., surveillance videos).
3. Execution time is not reported, and the multi-step pipeline may be slow.
4. Complex optical phenomena such as dynamic lighting changes and reflective surfaces are not handled.

## Related Work & Insights

- **Gaussian Marbles**: A similar idea but uses unstructured per-Gaussian motion and only performs local fusion $\to$ MoSca uses structured graphs for global fusion.
- **Shape-of-Motion**: An important concurrent work; MoSca leads significantly on DyCheck (+2.0 dB).
- **Embedded Deformation Graph**: A classic deformation graph method; MoSca's core innovation lies in fusing it with 2D foundation model priors and Gaussian Splatting.
- **Inspiration**: This direction can be further explored by introducing language priors into dynamic scene reconstruction (e.g., combining SAM segmentation for semantic 4D reconstruction).

## Rating

⭐⭐⭐⭐⭐ — Elegant method design (combining physical priors and foundation models), exceptionally high system coverage (fully automatic pose-free), substantial performance lead (mPSNR +2.0 dB), and strong in-the-wild generalization. It represents a significant breakthrough in the field of dynamic scene reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](../../NeurIPS2025/3d_vision/orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)

</div>

<!-- RELATED:END -->
