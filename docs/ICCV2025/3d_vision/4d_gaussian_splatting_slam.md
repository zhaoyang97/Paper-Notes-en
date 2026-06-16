---
title: >-
  [Paper Note] 4D Gaussian Splatting SLAM
description: >-
  [ICCV 2025][3D Vision][4D Gaussian] This paper presents the first complete 4D Gaussian Splatting SLAM system capable of simultaneously performing camera pose tracking and 4D Gaussian radiance field reconstruction in dyna…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D Gaussian"
  - "Dynamic Scene SLAM"
  - "Optical Flow Rendering"
  - "Sparse Control Points"
  - "RGB-D"
date: 2026-05-08
content_hash: 0094e89ed30328a4
---

# 4D Gaussian Splatting SLAM

**Conference**: ICCV 2025
**arXiv**: [2503.16710](https://arxiv.org/abs/2503.16710)  
**Code**: [https://github.com/yanyan-li/4DGS-SLAM](https://github.com/yanyan-li/4DGS-SLAM)  
**Area**: 3D Vision
**Keywords**: 4D Gaussian, Dynamic Scene SLAM, Optical Flow Rendering, Sparse Control Points, RGB-D

## TL;DR
This paper presents the first complete 4D Gaussian Splatting SLAM system capable of simultaneously performing camera pose tracking and 4D Gaussian radiance field reconstruction in dynamic scenes. Gaussian primitives are partitioned into static and dynamic sets; dynamic object motion is modeled via sparse control points and an MLP; and a novel 2D optical flow map rendering algorithm is introduced to supervise dynamic Gaussian motion learning.

## Background & Motivation
Existing GS-SLAM methods (SplaTAM, MonoGS, Gaussian-SLAM) predominantly assume static scenes. In dynamic environments, the dominant strategy is to detect and remove dynamic objects (e.g., via semantic segmentation) and reconstruct only the static background. This approach introduces two fundamental problems: (1) regions occupied by dynamic objects leave "holes," preventing complete scene reconstruction; and (2) dynamic information is entirely discarded, precluding downstream interaction tasks (e.g., robotic grasping of moving objects). Although dynamic Gaussian methods such as D3DGS and SC-GS can model motion, they require pre-given camera poses and are thus unsuitable for online incremental SLAM. Consequently, achieving accurate pose estimation and high-quality 4D Gaussian radiance field reconstruction simultaneously from RGB-D sequences in unknown dynamic environments remains an unresolved core challenge.

## Method

### Overall Architecture
The system comprises three core modules: (1) **Initialization**: YOLOv9 generates motion masks to partition Gaussians into a static set $\mathcal{G}_{st}$ and a dynamic set $\mathcal{G}_{dy}$, with sparse control points initialized in dynamic regions; (2) **Tracking**: pose estimation is performed using only static Gaussian rendering, excluding interference from dynamic objects; (3) **4D Mapping**: Gaussian attributes, camera poses, and the dynamic deformation network are jointly optimized, with optical flow constraints guiding dynamic motion learning. Each Gaussian is augmented with an attribute $dy$ indicating whether it belongs to the dynamic set, yielding the representation $\mathcal{G}=[\Sigma\;\mu\;\alpha\;\mathbf{c}\;dy]$.

### Key Designs
1. **Static/Dynamic Gaussian Separation and Keyframe Strategy**: During tracking, only static Gaussians are rendered, eliminating interference from dynamic objects. During mapping, static reconstruction and dynamic motion are optimized separately. Keyframe selection additionally accounts for changes in the motion mask — even when the camera is nearly stationary, a new keyframe is inserted if dynamic object motion is significant (or at least once every 5 frames). Newly inserted keyframes initialize only new static Gaussians, without adding dynamic ones.

2. **Sparse Control Points + MLP Deformation Network**: Inspired by SC-GS, sparse control points are initialized in dynamic regions. Unlike SC-GS, which requires lengthy pre-training, the proposed method initializes control points directly from motion regions in the first frame. An MLP $\Psi(P_k, t) \to [R_t, T_t]$ predicts the time-varying 6-DoF transformation for each control point. KNN search identifies the $K$ nearest control points for each dynamic Gaussian, and dense transformations are obtained via Gaussian RBF interpolation (Linear Blend Skinning), updating position $\mu$, rotation $R$, and scale $S$ simultaneously. This avoids the high parameter overhead of per-Gaussian motion learning.

3. **2D Optical Flow Map Rendering Supervision (Core Innovation)**: Dynamic Gaussians are projected onto the current camera plane at adjacent timesteps $t$ and $t-1$, yielding two sets of 2D coordinates. The displacement $d_x$ is rendered into an optical flow map via alpha-blending: $F(p)=\sum_i d_x \alpha_i \prod_j^{i-1}(1-\alpha_j)$. Both forward and backward optical flows are computed and supervised against RAFT-estimated flows within motion mask regions using an $L_1$ loss. This provides temporally consistent motion-geometric constraints and is the key factor driving significant improvements in dynamic reconstruction quality.

### Loss & Training
- **Tracking loss**: $L_t = \sum_p \mathcal{M}(\lambda O(p) L_1(C(p)) + (1-\lambda) L_1(D(p)))$, where the motion mask $\mathcal{M}$ filters out dynamic regions so that the loss is computed only over static areas. Color loss is applied only to pixels whose gradient exceeds threshold $\sigma$; depth loss requires $O(p)>0.95$ and $d(p)>0$.
- **Mapping loss**: $L_{mapping} = \lambda L_1(C) + (1-\lambda) L_1(D) + \lambda_{flow}\mathcal{L}_{flow} + W_1 E_{ARAP} + W_2 E_{iso}$, where $E_{iso}$ penalizes non-uniform stretching of Gaussian ellipsoids.
- **Two-stage mapping strategy**: Stage 1 freezes Gaussian parameters and optimizes only poses and the dynamic network (with doubled weight on dynamic regions); Stage 2 jointly optimizes all components (3 window frames + 5 overlap frames + 2 global random frames).
- **Global color refinement**: A final 1500-iteration pass randomly selects 10 frames per iteration and optimizes using $0.2\text{D-SSIM}+0.8L_1(C)+0.1L_1(D)+W_1 E_{ARAP}+W_2 E_{iso}$.
- **Implementation**: PyTorch + CUDA, single RTX 3090 Ti; hyperparameters $\lambda=0.9$, $\lambda_{flow}=3$, $W_1=1e\text{-}4$, $W_2=10$.

## Key Experimental Results

### Main Results

**Pose Estimation ATE (cm)↓ — BONN Dataset (9-sequence average)**:

| Method | balloon | ps_track | sync | p_no_box | Avg (9 seq) |
|--------|---------|----------|------|----------|-------------|
| MonoGS | 29.6 | 54.5 | 68.5 | 71.5 | 33.1 |
| SplaTAM | 32.9 | 77.8 | 59.5 | 91.9 | 56.8 |
| Gaussian-SLAM | 66.9 | 107.2 | 111.8 | 69.9 | 84.3 |
| RoDyn-SLAM | 7.9 | 14.5 | 1.3 | 4.9 | 7.9 |
| **Ours** | **2.4** | **8.9** | **2.8** | **1.8** | **3.6** |

**Rendering Quality — BONN Dataset (9-sequence average)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| MonoGS | 21.06 | 0.780 | 0.342 |
| SplaTAM | 19.34 | 0.757 | 0.233 |
| SC-GS | 21.63 | 0.724 | 0.461 |
| **Ours** | **23.66** | **0.852** | **0.241** |

**TUM RGB-D Dataset (6-sequence average)**: ATE 1.8 cm (Ours) vs. 5.1 cm (RoDyn-SLAM) vs. 15.8 cm (MonoGS); PSNR 22.46 (Ours) vs. 20.78 (SC-GS) vs. 17.74 (MonoGS).

### Ablation Study

**Effect of Optical Flow Loss and Gaussian Separation (BONN synchronous sequence, PSNR↑)**:

| Flow Loss | Separate Gaussians | syn PSNR | syn2 PSNR |
|-----------|--------------------|----------|-----------|
| ✗ | ✗ | 18.37 | 22.11 |
| ✗ | ✓ | 22.87 | 24.84 |
| ✓ | ✗ | 17.40 | 21.03 |
| ✓ | ✓ | **23.25** | **25.42** |

**Mapping Frame Selection Strategy**: The combination of 3 window frames + 5 overlap frames + 2 global random frames achieves the best reconstruction quality in both static and dynamic regions. Other combinations either cause blurring in dynamic regions (too many global frames) or catastrophic forgetting in static regions (too many window frames).

### Key Findings
- Both components are indispensable: applying the optical flow loss alone without Gaussian separation (17.40) yields worse results than using neither (18.37), because static Gaussians are erroneously supervised by dynamic optical flow signals.
- Static GS-SLAM methods exhibit 10–50× ATE degradation in highly dynamic scenes — e.g., on BONN *sit_half*, MonoGS achieves 54.5 cm vs. 8.9 cm for the proposed method.
- The proposed method slightly underperforms MonoGS on the TUM *sit* sequence (small motion), but demonstrates a decisive advantage on *walk* (large motion): 2.1 cm vs. 30.7 cm.

## Highlights & Insights
- **First complete 4D GS-SLAM system**: Rather than discarding dynamic objects, the system simultaneously tracks and reconstructs the full 4D scene, filling an important gap in the literature.
- **Optical flow rendering supervision** is the core innovation — 2D optical flow is naturally derived from 3D Gaussian motion and cross-validated against RAFT estimates, forming a triple constraint of geometry, appearance, and motion.
- ATE reaches 3.6 cm on BONN and 1.8 cm on TUM, substantially outperforming all static GS-SLAM and NeRF-based dynamic SLAM baselines.
- Ablations demonstrate that the optical flow loss and Gaussian separation must operate synergistically; applying optical flow supervision alone is harmful — a counter-intuitive finding with important methodological implications.

## Limitations & Future Work
- Dependency on YOLOv9 for motion mask generation may cause failure on dynamic objects of unknown categories; integration of unsupervised cues such as optical flow warrants investigation.
- Certain sequences require manual specification of the dynamic initialization frame (e.g., *placing_nonobstructing_box*); fully automatic detection remains to be improved.
- Validation is limited to indoor RGB-D scenes; extension to outdoor or monocular settings requires addressing depth unavailability, potentially via monocular depth estimation.
- The number of dynamic Gaussians is fixed after initialization, precluding handling of newly appearing or disappearing dynamic objects.
- The 1500-step global color refinement may be suboptimal in efficiency; adaptive optimization scheduling merits exploration.

## Related Work & Insights
- **vs. MonoGS / SplaTAM / Gaussian-SLAM**: These static GS-SLAM methods suffer severe pose drift in dynamic scenes; the proposed method addresses this through static/dynamic separation and masking.
- **vs. RoDyn-SLAM**: A NeRF-based dynamic SLAM with comparable pose accuracy, but inferior rendering quality and efficiency relative to Gaussian-based approaches.
- **vs. DGS-SLAM / DG-SLAM**: These GS-SLAM methods remove dynamic objects and reconstruct only the static scene; the proposed method explicitly models dynamics and renders the complete 4D scene.
- **vs. SC-GS / D3DGS**: Dynamic Gaussian methods that require pre-given poses; the proposed method estimates poses online incrementally, making it applicable to real SLAM settings.
- The idea of deriving optical flow from Gaussian motion can be extended to motion constraints in video generation and editing.
- The static/dynamic separation framework is applicable to semantic SLAM in dynamic environments and moving-object perception in embodied navigation.

## Rating
- Novelty: ⭐⭐⭐⭐ — 4D GS-SLAM is a natural and necessary extension; optical flow rendering supervision is a genuine contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual datasets (TUM + BONN), multiple baselines, and comprehensive ablations; large-scale/outdoor evaluation is absent.
- Writing Quality: ⭐⭐⭐ — Method description is clear, but overall organization has room for improvement.
- Value: ⭐⭐⭐⭐ — Fills the 4D GS-SLAM gap with significant implications for dynamic scene understanding and robotics applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM](../../CVPR2026/3d_vision/flow4dgs-slam_optical_flow-guided_4d_gaussian_splatting_slam.md)
- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [\[ICCV 2025\] Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps](outdoor_monocular_slam_with_global_scale-consistent_3d_gaussian_pointmaps.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[CVPR 2026\] AERGS-SLAM: Auto-Exposure-Robust Stereo 3D Gaussian Splatting SLAM](../../CVPR2026/3d_vision/aergs-slam_auto-exposure-robust_stereo_3d_gaussian_splatting_slam.md)

</div>

<!-- RELATED:END -->
