---
title: >-
  [Paper Note] PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences
description: >-
  [CVPR 2026][3D Vision][point cloud scene flow] PCSTracker is the first end-to-end framework for long-term scene flow estimation on point cloud sequences. Through iterative joint geometry-motion optimization, spatiotemporal trajectory updates, and an overlapping sliding window strategy, it reduces EPE_3D by 57.9% on the synthetic dataset PointOdyssey3D while running in real time at 32.5 FPS.
tags:
  - CVPR 2026
  - 3D Vision
  - point cloud scene flow
  - long-term trajectory estimation
  - spatiotemporal Transformer
  - sliding window
  - 3D motion analysis
date: 2026-05-08
content_hash: b39dcf8f8a1daf4a
---

# PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences

**Conference**: CVPR 2026
**arXiv**: [2603.19762](https://arxiv.org/abs/2603.19762)
**Code**: [https://github.com/MinLin2022/PCSTracker](https://github.com/MinLin2022/PCSTracker)
**Area**: 3D Vision / Scene Flow Estimation
**Keywords**: point cloud scene flow, long-term trajectory estimation, spatiotemporal Transformer, sliding window, 3D motion analysis

## TL;DR

PCSTracker is the first end-to-end framework for long-term scene flow estimation on point cloud sequences. Through iterative joint geometry-motion optimization, spatiotemporal trajectory updates, and an overlapping sliding window strategy, it reduces EPE_3D by 57.9% on the synthetic dataset PointOdyssey3D while running in real time at 32.5 FPS.

## Background & Motivation

1. **State of the Field**: Understanding fine-grained long-term 3D motion from point cloud sequences is critical for autonomous driving, robotic navigation, and AR/VR. Existing approaches fall into two lines: object tracking (focusing on object-level motion, unable to recover fine-grained motion) and scene flow estimation (limited to adjacent frame pairs, unable to maintain temporal consistency over long sequences).

2. **Limitations of Prior Work**: Naively chaining short-term methods over long sequences (tens to hundreds of frames) leads to catastrophic errors:
    - Viewpoint changes and object deformations cause temporal drift in point features, breaking point correspondence consistency.
    - Frequent occlusions and out-of-bound motion interrupt point correspondences.
    - Small errors inevitably accumulate over time, ultimately causing severe drift.

3. **Root Cause**: Per-frame scene flow methods lack the ability to model geometric evolution, handle occlusions, and suppress error accumulation over long time spans, while object tracking methods cannot provide point-level fine-grained motion.

4. **Paper Goals**: How to robustly and efficiently predict long-term scene flow (a complete $T \times 3$ 3D trajectory matrix) directly from raw point cloud sequences, while addressing the three key challenges of geometric change, occlusion, and error accumulation.

5. **Starting Point**: Extending scene flow estimation from two frames to long sequences can be viewed as a point-level refinement of object tracking—combining the fine-grained motion estimation of scene flow with the long-term temporal modeling of object tracking.

6. **Core Idea**: End-to-end long-term point cloud scene flow estimation is achieved through three dedicated designs: iterative joint geometry-motion optimization to handle geometric variation, a spatiotemporal Transformer to infer occluded point positions, and an overlapping sliding window strategy to suppress error accumulation.

## Method

### Overall Architecture

PCSTracker takes a point cloud sequence $\mathbf{S} = \{S_t\}_{t=1}^T$ (with $N_t$ points per frame) and the initial coordinates $P_{xyz}$ of $N$ query points, and outputs the complete $T \times N \times 3$ trajectory. The pipeline consists of four steps: (1) PointConv feature extraction and KNN trajectory initialization; (2) Iterative Geometry-Motion Optimization (IGMO) to compute local geometric similarity and update trajectories; (3) Spatiotemporal Trajectory Update (STTU) to model global spatiotemporal dependencies and estimate residual motion; (4) overlapping sliding window inference for long sequences. The entire process is iterated $K$ times.

### Key Designs

1. **Iterative Geometry-Motion Optimization Module (IGMO)**:

    - **Function**: Explicitly models the temporal evolution of query point features to maintain reliable correspondences under dynamic geometric changes.
    - **Mechanism**: At each iteration, the module computes local geometric similarity $C_g^k$ between the current trajectory feature $Q_{feat}^{k-1}$ and the precomputed feature map $\mathbf{F}$. The top-M highest-correlation entries are selected to construct a truncated correlation volume. A dual-branch correlation module is employed: the **point correlation branch** selects KNN neighbors and aggregates similarity and relative position offsets $C_{point}^k = \max(\text{MLP}(\text{concat}(C^k(\mathcal{N}_{M_k}), \mathcal{N}_{M_k} - Q_{xyz}^k)))$; the **voxel correlation branch** discretizes the local space into $a \times a \times a$ cubes of varying sizes and averages point correlations within each sub-cube to build multi-scale long-range features $C_{voxel}^{k,r}$. The two branches are fused to jointly update both motion and geometry features.
    - **Design Motivation**: Unlike per-frame scene flow, query point geometry in long sequences changes substantially over time; accurate matching cannot be maintained without updating features. The dual-branch design simultaneously captures fine-grained local and long-range spatial correlations.

2. **Spatiotemporal Point Trajectory Update Module (STTU)**:

    - **Function**: Leverages broad temporal context to infer plausible positions of occluded points in intermediate frames, ensuring motion continuity.
    - **Mechanism**: Motion tokens are first constructed by concatenating the fused correlation volume $C_{fuse}^k$, trajectory feature $Q_{feat}^{k-1}$, and sinusoidally encoded optical flow information into a motion feature, which is then combined with positional encoding $\eta_p(Q_{xyz}^{k-1})$ and timestamp encoding $\eta_t(t)$. The motion tokens are passed through $2 \times M$ Transformer blocks with alternating inter-frame (temporal) and intra-frame (spatial) self-attention. A predictor $\Psi$ then estimates residual motion and feature updates: $({\Delta Q_{xyz}^k}, {\Delta Q_{feat}^k}) = \Psi(\mathbf{F}_{token}^o)$, which are accumulated onto the previous iteration's results.
    - **Design Motivation**: Frequent occlusions severely ambiguate point correspondences. Jointly estimating the complete motion of all query points within the temporal window—rather than independently estimating frame by frame—enables the model to infer positions in invisible frames from sparse visible timesteps and maintain physical consistency.

3. **Overlapping Sliding Window Inference Strategy**:

    - **Function**: Processes long sequences in segments while preserving cross-window temporal consistency to suppress error accumulation.
    - **Mechanism**: A sequence of total length $T'$ is divided into $W_{all} = \lceil 2T'/T - 1 \rceil$ sub-sequences of length $T$, with adjacent windows overlapping by $T/2$. Each window is initialized with the trajectory estimate from the previous window and then iteratively optimized for $K$ steps within the current window. Windows are processed sequentially, with overlap propagation and intra-window optimization alternating.
    - **Design Motivation**: Processing hundreds of frames at once is computationally infeasible, while naive non-overlapping segmentation produces discontinuities at window boundaries. The $T/2$ overlap ensures every timestep is covered by at least two windows, effectively propagating context and suppressing error propagation.

### Loss & Training

- Supervised loss: $Loss = \sum_{w=0}^{W_{all}} \sum_{t=1}^{T} \sum_{k=1}^{n} \gamma^{n-k} \|Q_{xyz}^{k,t,w} - Q_{xyz}^{GT}\|_2$
- Exponentially decaying weight $\gamma = 0.8$; later iteration steps receive higher weights.
- Trained on PointOdyssey3D for 200K steps with batch size 4; each sample contains 24 frames, 256 query points, and 8192 points per frame.
- AdamW optimizer with OneCycle learning rate scheduling; initial lr = 2e-4.
- During inference, local auxiliary points (KNN) and global auxiliary points (FPS/random sampling) are supported; 1024 auxiliary points are added by default.

## Key Experimental Results

### Main Results

PointOdyssey3D dataset (synthetic):

| Method | Input | EPE_3D↓ | δ_3D^avg↑ | Survival_3D^0.50↑ |
|--------|-------|---------|-----------|-------------------|
| SpatialTracker | RGB-D | 0.924 | 42.25 | 49.54 |
| SceneTracker | RGB-D | 0.204 | 79.48 | 87.98 |
| SF-baseline | Point | 0.330 | 61.65 | 77.78 |
| **PCSTracker** | **Point** | **0.133** | **86.37** | **93.65** |

ADT3D dataset (real-world):

| Method | Input | EPE_3D↓ | δ_3D^avg↑ | Survival_3D^0.50↑ |
|--------|-------|---------|-----------|-------------------|
| SceneTracker | RGB-D | 0.601 | 68.99 | 80.40 |
| SF-baseline | Point | 0.945 | 40.49 | 51.61 |
| **PCSTracker** | **Point** | **0.372** | **74.44** | **87.74** |

### Ablation Study

| Experiment | Variable | EPE_3D↓ | δ_3D^avg↑ |
|------------|----------|---------|-----------|
| Geometry feature update | w/o | 0.202 | 75.85 |
| Geometry feature update | w/ | **0.133** | **86.37** |
| Window size | T=2 | 0.206 | 78.33 |
| Window size | T=8 | 0.166 | 83.06 |
| Window size | T=16 | **0.133** | **86.37** |
| Transformer blocks | 6×1 (temporal only) | 0.202 | 75.84 |
| Transformer blocks | 3×2 (spatiotemporal alternating) | **0.133** | **86.37** |
| Auxiliary points (single query) | None | 0.852 | 47.49 |
| Auxiliary points (single query) | KNN+FPS | **0.119** | **87.64** |

### Key Findings

- **Geometry feature update is critical**: Removing it increases EPE_3D by 34.2% (0.133→0.202), demonstrating that long sequences require explicit modeling of the temporal evolution of features.
- **Longer temporal context consistently helps**: Extending the window from 2 to 16 frames reduces EPE_3D by 35.4% (0.206→0.133).
- **Spatial attention is indispensable**: Using temporal attention alone (6×1) causes a large performance drop; spatiotemporal alternation (3×2) is optimal.
- **Large benefit from auxiliary points**: In single-query-point mode, adding KNN+FPS auxiliary points reduces EPE_3D from 0.852 to 0.119 (an 86% reduction); FPS global sampling outperforms random sampling.
- **Temporal drift analysis**: At 40 frames, PCSTracker achieves EPE 0.205 vs. SF-baseline's 0.543, with a substantially slower error growth rate.
- **Efficiency**: Only 3.48M parameters vs. SceneTracker's 24.2M and SpatialTracker's 34.0M; inference speed of 32.5 FPS (fastest among compared methods).

## Highlights & Insights

- **Pioneering problem definition**: As the first work to systematically study long-term scene flow estimation on point clouds, this paper explicitly identifies three core challenges (geometric variation, occlusion, error accumulation) and provides targeted solutions. Using only point clouds, the proposed method achieves superior 3D motion understanding compared to RGB-D approaches.
- **Dual-branch correlation volume design**: The point branch captures fine-grained local matching while the voxel branch captures multi-scale long-range structure, yielding strong complementarity. Inherited from PV-RAFT, this design is validated in the long-sequence setting.
- **Practical value of the auxiliary point strategy**: Since the irregular, discrete nature of point clouds precludes regular-grid auxiliary points, the KNN+FPS combination is a concise and effective solution with highly significant gains in single-point tracking scenarios.
- **Dataset contribution**: The paper introduces PointOdyssey3D (synthetic, for training) and ADT3D (real-world, for evaluation), filling a data gap in this research direction.

## Limitations & Future Work

- The model is sensitive to geometric scale and scene distance variation; performance may degrade when transferring from synthetic data to real scenes with different spatial distributions (e.g., autonomous driving).
- Training exclusively on synthetic data means that noise and sparsity in real-world point clouds may pose additional challenges.
- The current setting of 8192 points per frame may limit computational efficiency for dense point clouds (e.g., LiDAR with tens of thousands of points).
- Future directions include: introducing scene-specific data or adaptive training strategies to mitigate distribution shift; exploring more efficient correlation volume computation and Transformer attention mechanisms; extending the framework to large-scale outdoor scenes.

## Related Work & Insights

- **vs. SceneTracker (RGB-D)**: SceneTracker is the strongest RGB-D baseline, with EPE 0.204 vs. PCSTracker's 0.133 on PointOdyssey3D, demonstrating that point-cloud-only methods have a natural advantage in 3D motion understanding (unconstrained by 2D appearance-driven frameworks).
- **vs. PV-RAFT (SF-baseline)**: The dual-branch correlation volume design of PV-RAFT is inherited, yet the simple chaining strategy without long-sequence-specific design yields EPE as high as 0.330 (+148%), clearly demonstrating the necessity of dedicated long-sequence designs.
- **vs. SpatialTracker/DELTA**: These RGB-D methods are limited by 2D appearance features and perform poorly on 3D trajectory recovery (EPE 0.924/0.780), highlighting the clear advantage of rich 3D geometric information.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to systematically define and address long-term scene flow estimation on point clouds; the three design modules are highly targeted and tightly integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets (synthetic + real-world), multi-dimensional ablations, temporal drift analysis, and efficiency comparisons — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, method hierarchy is well-structured, and experimental analysis is in-depth.
- Value: ⭐⭐⭐⭐⭐ A pioneering contribution with dataset releases and real-time inference, making a significant impact on the field of 3D motion analysis.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)
- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[CVPR 2026\] Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences](neu-pig_neural_preconditioned_grids_for_fast_dynamic_surface_reconstruction_on_l.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](../../NeurIPS2025/3d_vision/rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[AAAI 2026\] Class-Partitioned VQ-VAE and Latent Flow Matching for Point Cloud Scene Generation](../../AAAI2026/3d_vision/class-partitioned_vq-vae_and_latent_flow_matching_for_point_cloud_scene_generati.md)

<!-- RELATED:END -->
