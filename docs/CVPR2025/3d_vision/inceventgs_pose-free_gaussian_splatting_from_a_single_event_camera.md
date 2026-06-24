---
title: >-
  [Paper Note] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera
description: >-
  [CVPR 2025][3D Vision][Event Camera] This paper proposes IncEventGS, the first method for incremental 3D Gaussian Splatting scene reconstruction using only a monocular event camera without known poses. By adopting a tracking-mapping SLAM paradigm to jointly optimize camera motion and scene representation, it outperforms existing methods in both novel view synthesis and pose estimation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Event Camera"
  - "3D Gaussian Splatting"
  - "Incremental Reconstruction"
  - "Visual Odometry"
  - "SLAM"
date: 2026-05-08
content_hash: 55f344bd175755d0
---

# IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera

**Conference**: CVPR 2025  
**arXiv**: [2410.08107](https://arxiv.org/abs/2410.08107)  
**Code**: [https://github.com/WU-CVGL/IncEventGS](https://github.com/WU-CVGL/IncEventGS)  
**Area**: 3D Vision  
**Keywords**: Event Camera, 3D Gaussian Splatting, Incremental Reconstruction, Visual Odometry, SLAM

## TL;DR

This paper proposes IncEventGS, the first method for incremental 3D Gaussian Splatting scene reconstruction using only a monocular event camera without known poses. By adopting a tracking-mapping SLAM paradigm to jointly optimize camera motion and scene representation, it outperforms existing methods in both novel view synthesis and pose estimation.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) and NeRF have achieved massive progress in novel view synthesis, typically requiring RGB or RGB-D cameras combined with COLMAP to provide initial camera poses and point clouds. Event cameras, as bio-inspired sensors, offer high temporal resolution, high dynamic range, low latency, and low power consumption, performing far better than frame-based cameras in rapid motion and low-light conditions.

**Limitations of Prior Work**: (1) Existing event-based NeRF methods (e.g., E-NeRF, EventNeRF, Robust e-NeRF) all require ground-truth poses, which severely limits their practical application; poses are typically computed from paired RGB images via COLMAP or provided by motion capture systems. (2) Event data is asynchronous and sparse, which cannot be directly utilized by traditional 3D reconstruction algorithms. (3) Existing event-based SLAM methods typically retrieve only sparse depth maps (2.5D) rather than dense 3D scenes. (4) Two-stage methods (e.g., E2VID translation -> COLMAP pose estimation -> 3DGS) introduce significant errors during both event-to-image translation and pose estimation.

**Key Challenge**: A gap exists between the asynchronous nature of event camera data and the synchronized intensity maps and accurate poses required by 3DGS. Moreover, the pure event stream lacks absolute intensity information, making initialization and pose estimation exceptionally difficult.

**Goal**: How to incrementally and simultaneously reconstruct 3D Gaussian scene representation and camera motion trajectory starting solely from a monocular event stream without any prior poses?

**Key Insight**: The approach borrows the tracking-mapping paradigm from traditional SLAM but substitutes the underlying scene representation with 3DGS. The key observation is that the event stream can be differentiably linked to the intensity maps rendered by 3DGS through an intensity change model, allowing joint optimization of poses and the scene. To address the lack of initial poses and point clouds, a monocular depth estimation-guided initialization is designed.

**Core Idea**: Combining the tracking-mapping paradigm of event-based SLAM with 3D Gaussian Splatting, and formulating a differentiable loss through the event intensity change model for joint bundle adjustment.

## Method

### Overall Architecture

IncEventGS divides the event stream into chunks with fixed time windows, where each chunk is associated with a continuous-time trajectory parameterization (linear interpolation in the $\mathfrak{se}(3)$ space). The system alternates between two phases: (1) **Tracking**: The 3DGS is fixed, and only the camera trajectory parameters of the latest chunk are optimized; (2) **Mapping**: Using a sliding window bundle adjustment, the 3DGS parameters and trajectory parameters of the most recent $n_w=20$ chunks are jointly optimized. As the camera moves, Gaussian points are incrementally added in newly explored regions.

### Key Designs

1. **Event-3DGS Differentiable Relationship**:

    - **Function**: Establish a differentiable loss between the event stream and 3DGS rendering.
    - **Mechanism**: Within each chunk, two close timestamps $t_k$ and $t_{k+\Delta t}$ are randomly sampled. The corresponding poses $T_k$ and $T_{k+\Delta t}$ are interpolated from the trajectory parameterization, and two intensity maps $\hat{I}_k$ and $\hat{I}_{k+\Delta t}$ are rendered from 3DGS. The synthesized intensity change is computed as $\hat{E}(\mathbf{x}) = \log(\hat{I}_{k+\Delta t}) - \log(\hat{I}_k)$ , and the measured intensity change $E(\mathbf{x})$ is accumulated from the event stream of the corresponding time interval. Minimizing $\|E - \hat{E}\|_2$ serves as the core loss.
    - **Design Motivation**: Event cameras record intensity changes rather than absolute intensity, and the logarithmic difference model naturally corresponds to the event triggering mechanism. Rendering intensity maps at two timestamps makes both 3DGS parameters and camera poses differentiable, enabling joint optimization.

2. **Incremental Tracking and Mapping**:

    - **Function**: Achieve incremental 3D reconstruction without prior poses.
    - **Mechanism**: The **tracking phase** only optimizes $T_{start}$ and $T_{end}$ of the latest chunk, using the reconstructed 3DGS scene as a reference to minimize the event loss. The **mapping phase** utilizes a sliding window ($n_w=20$) to perform bundle adjustment, jointly optimizing the trajectories of all chunks in the window and the 3DGS parameters. Simultaneously, new Gaussian points are added in newly explored regions: depth maps are rendered using the existing 3DGS and back-projected into 3D space to determine positions of new points, while a visibility mask $V < \lambda_V$ ensures points are only added in uncovered regions.
    - **Design Motivation**: The tracking-mapping paradigm is well-established (e.g., ORB-SLAM) but needs to be adapted for 3DGS and event data. Sliding window BA balances global consistency and computational efficiency. Incremental point addition combined with transparent Gaussian pruning ensures gradual scene expansion.

3. **Depth-Guided Initialization**:

    - **Function**: Resolve the difficulty of 3DGS initialization under pure event streams.
    - **Mechanism**: The first $m=3$ chunks are used to train the initial 3DGS and poses (poses are initialized near the identity matrix, and point clouds are randomly sampled within a bounding box). After a certain number of training iterations, a monocular depth estimation network predicts a dense depth map from the rendered intensity map. The pixel depths are then back-projected to obtain new Gaussian center locations for re-initialization, followed by retraining.
    - **Design Motivation**: Ablation studies demonstrate that depth initialization drops the ATE drastically from 1.534cm to 0.046cm. The quality of initial 3D structures from short-baseline event data is extremely poor, and the depth estimation network provides a vital prior from 2D to 3D. Without a good initial structure, subsequent incremental expansion will drift.

### Loss & Training

The tracking phase only uses the event L2 loss (Eq.10). The mapping phase uses a combination of event L2 loss and SSIM structural similarity loss ($\mathcal{L} = (1-\lambda)\mathcal{L}_{event} + \lambda\mathcal{L}_{ssim}$, where $\lambda=0.05$). The time window of each chunk is 50ms. The training consists of 4500 steps for initialization, 200 steps for tracking, and 1500 steps for mapping. The event stream is divided equally into $n_{seg}=100$ segments based on event count for timestamp sampling, with a sampling window size of $n_{low}=400k$, $n_{up}=500k$ (for synthetic datasets).

## Key Experimental Results

### Main Results

| Scene (Replica) | Metric | IncEventGS (No Pose) | Robust e-NeRF (GT Pose) | E2VID+COLMAP+3DGS |
|----------------|------|---------------------|------------------------|-------------------|
| room0 | PSNR↑ | **24.31** | 17.26 | 14.45 |
| room2 | PSNR↑ | **23.75** | 16.43 | 15.74 |
| office0 | PSNR↑ | **25.64** | 18.93 | 18.91 |
| office2 | PSNR↑ | **21.74** | 16.81 | 14.03 |
| office3 | PSNR↑ | **21.18** | 19.22 | 13.25 |

Even without using GT poses, IncEventGS significantly outperforms NeRF-based methods using GT poses (averaging +5-7 dB PSNR).

| Dataset | Method | ATE (cm)↓ |
|--------|------|-----------|
| Replica room0 | DEVO | 0.289 |
| Replica room0 | **IncEventGS** | **0.046** |
| TUM-VIE desk | DEVO | 0.732 |
| TUM-VIE desk | **IncEventGS** | **0.231** |
| TUM-VIE 6dof | DEVO | 2.93 |
| TUM-VIE 6dof | **IncEventGS** | **0.251** |

Pose estimation accuracy completely outperforms the SOTA event-based visual odometry DEVO.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | ATE (cm)↓ | Description |
|------|-------|-------|--------|-----------|------|
| Full model | 21.74 | 0.82 | 0.23 | 0.046 | Full model |
| w/o Depth Initialization | 17.80 | 0.76 | 0.26 | 1.534 | ATE increases by 33x |

| Event Window ($n_{low}$-$n_{up}$) | PSNR↑ | ATE↓ | Description |
|------|-------|------|------|
| 1k-10k | 16.07 | 0.167 | Window too small, lacking information |
| 400k-500k | **21.74** | **0.046** | Optimal configuration |
| 600k-700k | 18.06 | 0.214 | Window too large, leading to high memory and excessive motion |

### Key Findings

- Depth initialization is a decisive factor for the system's success - removing it leads to a 4 dB drop in PSNR and a 33x spike in ATE, indicating that 3D reconstruction from short-baseline event data heavily depends on a good initialization.
- The event window size must be carefully chosen (400k-500k is optimal); too small lacks information, while too large introduces excessive motion, making matching difficult.
- Even without GT poses, the combination of 3DGS representation and event-based BA still performs far better than NeRF methods using GT poses, demonstrating the superiority of the 3DGS representation.
- On the TUM-VIE real-world dataset, the pose estimation of IncEventGS even outperforms the stereo event-based method ESVO2, proving the power of monocular + 3DGS BA.

## Highlights & Insights

- **First pose-free 3DGS reconstruction from pure events**: This fills an important research gap. Previous event-based NeRF methods relied on GT poses, where acquiring poses in practical applications required frame-based cameras in the first place.
- **The advantages of the 3DGS representation are amplified in event reconstruction**: The 5-7 dB improvement over NeRF methods suggests that the explicit representation and differentiable rendering of 3DGS are easier to optimize under weak oversight signals (where the event stream only provides intensity changes).
- **The clever use of a depth estimation network as an initialization prior**: The depth network is not required continuously; utilizing it once during the bootstrap phase significantly improves all subsequent optimizations.

## Limitations & Future Work

- Performance on synthetic data is much better than on real-world data, where noise and contrast threshold estimation errors of real-world event cameras have a substantial impact.
- The sliding window BA only considers local consistency, and accumulation drift may exist in long trajectories (loop closure detection was not reported).
- The event stream lacks absolute intensity information, and NVS metrics utilize linear color correction, suggesting that the recovered absolute intensity might be biased.
- Initialization relies on the quality of the monocular depth estimation network, which may fail in scenes where the network generalizes poorly.
- Comparison is missing with concurrent works such as EvGGS and Event3DGS under identical settings (as those methods use GT poses).

## Related Work & Insights

- **vs E-NeRF / EventNeRF / Robust e-NeRF**: These methods adopt NeRF representation and require GT poses. IncEventGS utilizes 3DGS and does not require poses, completely dominating in NVS metrics.
- **vs DEVO**: DEVO is a SOTA event-based visual odometry that only estimates poses without reconstructing scenes. IncEventGS simultaneously accomplishes pose estimation and dense reconstruction with higher pose accuracy (room0: 0.046 vs 0.289 cm).
- **vs MonoGS / GS-SLAM**: These are frame-based camera 3DGS SLAM methods. IncEventGS inherits similar concepts transferred to the event-camera domain, successfully tackling challenges specific to event data like asynchrony and lack of absolute intensity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first pose-free event-camera 3DGS reconstruction; the problem formulation holds significant practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on synthetic + real-world datasets, dual metrics of NVS + pose, thorough ablation studies, but lacks direct comparison with concurrent tasks.
- Writing Quality: ⭐⭐⭐⭐ Clean pipeline diagram with a systematic and complete method description.
- Value: ⭐⭐⭐⭐⭐ Provides significant advancements to the event-camera 3D reconstruction field; removing pose dependance substantially increases the potential for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] EventFly: Event Camera Perception from Ground to the Sky](eventfly_event_camera_perception_from_ground_to_the_sky.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
