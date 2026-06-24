---
title: >-
  [Paper Note] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video
description: >-
  [CVPR 2025][3D Vision][NeRF] Modeling camera motion as time-continuous angular and linear velocities, this work avoids direct optimization of large-range camera-to-world transformations through velocity integration. Combined with a time-dependent NeRF and SDF flow constraints, camera poses and scene geometry are jointly optimized from monocular video without depth priors.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "NeRF"
  - "camera pose estimation"
  - "continuous motion"
  - "SDF"
  - "velocity integration"
  - "prior-free"
date: 2026-05-08
content_hash: 03550f8ba17ca472
---

# Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video

**Conference**: CVPR 2025  
**arXiv**: [2504.19819](https://arxiv.org/abs/2504.19819)  
**Code**: [HoangChuongNguyen/cope-nerf](https://github.com/HoangChuongNguyen/cope-nerf)  
**Area**: 3D Vision  
**Keywords**: NeRF, camera pose estimation, continuous motion, SDF, velocity integration, prior-free

## TL;DR

Modeling camera motion as time-continuous angular and linear velocities, this work avoids direct optimization of large-range camera-to-world transformations through velocity integration. Combined with a time-dependent NeRF and SDF flow constraints, camera poses and scene geometry are jointly optimized from monocular video without depth priors.

## Background & Motivation

**Background**: NeRF performs excellently in novel view synthesis, but training requires precise pre-computed camera poses (typically from COLMAP), which incurs extra computational cost and lacks differentiability.

**Limitations of Prior Work**:
- Joint optimization methods such as NeRFmm and BARF are limited to forward-facing scenes or require good initialization.
- NoPe-NeRF relies on pre-trained depth networks as priors.
- All methods that optimize poses independently frame-by-frame struggle in scenes with large-range camera motion, as the camera-to-world transformations can be extremely large.

**Key Challenge**: Optimizing poses independently frame-by-frame requires direct estimation of large camera-to-world transformations, which easily falls into local optima during the early stages of optimization; furthermore, the temporal continuity information of the video is underutilized.

**Goal**: Without any priors (no depth, no initial poses), jointly optimize camera motion and 3D scenes solely from monocular RGB video and camera intrinsics.

**Key Insight**: Transform discrete pose optimization into continuous motion estimation—using an MLP to predict the angular and linear velocities at each timestamp, and obtaining the relative transformations between any two frames through integration.

## Method

### Overall Architecture

1. **Motion network** $\phi_v(t)$ predicts the angular velocity $\boldsymbol{\omega}(t)$ and linear velocity $\mathbf{v}(t)$ at each timestamp $t$.
2. **Time-dependent NeRF** $(\phi_g, \phi_c)$ represents the local scene geometry (SDF) and appearance at each timestamp $t$.
3. Relative transformation $\mathbf{P}_{t_1 \to t_2}$ between any two frames is computed via velocity integration (Euler method).
4. Multiple consistency losses constrain the consistency of scene geometry and camera motion.
5. In the late stage, poses are fixed, and a complete scene NeRF is trained in the world frame.

### Key Designs

**1. Continuous Camera Motion Modeling**
- **Function**: Maps timestamp $t$ to $(\boldsymbol{\omega}(t), \mathbf{v}(t)) \in \mathbb{R}^6$ using an MLP, and integrates them via the Euler method to obtain rotation matrices and translation vectors.
- **Mechanism**: $\mathbf{R}_{t\to t+l} = \prod_{u=0}^{U-1} \psi(\boldsymbol{\omega}(t+u\Delta t)\Delta t)$, $\mathbf{t}_{t\to t+l} = \sum_{u=0}^{U-1} \mathbf{v}(t+u\Delta t)\Delta t$, step size $\Delta t = l/U$, using 10 sub-intervals.
- **Design Motivation**: Decomposing large motions into the accumulation of infinitesimal increments greatly simplifies the learning of large-range motions. Unlike frame-by-frame independent optimization or SLAM-style relative transformations between neighboring frames, continuous modeling naturally supports cross-frame constraints and smoothness.

**2. Time-Dependent NeRF**
- **Function**: NeRF takes $(\mathbf{x}, t)$ as input to predict SDF $s(\mathbf{x},t)$ and color $\mathbf{c}(\mathbf{x},t)$, defining the scene in the local camera coordinate system at each timestamp $t$.
- **Mechanism**: When pose noise is large in the early stages of training, the NeRF at each timestamp only needs to explain the local scene of neighboring frames—this is more stable than training with noisy poses in the global coordinate system.
- **Design Motivation**: Avoids accumulated errors caused by mapping 3D points to the global coordinate system under early stage noisy poses. The time-dependent design also enables SDF flow constraints.

**3. SDF Flow and Motion Consistency Constraint**
- **Function**: Utilizes the linear relationship between the SDF temporal derivative and camera motion to constrain the consistency of scene and motion: $\mathcal{L}_{flow} = \left|\frac{\partial s}{\partial t} + (\boldsymbol{\omega} \times \mathbf{x} + \mathbf{v})^T \mathbf{n}\right|$.
- **Mechanism**: For static scenes, the SDF of surface points satisfies rigid-body constraints under camera motion; the temporal rate of change of the SDF equals the projection of the camera velocity onto the surface normal.
- **Design Motivation**: This physical constraint forces the scene surface changes at each timestamp to be consistent with camera motion, preventing the time-dependent NeRF from independently "drifting" at each timestamp.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_{eik} + \lambda_2 \mathcal{L}_{flow} + \lambda_3 \mathcal{L}_{photo} + \lambda_4 \mathcal{L}_{sdf}$$

- $\mathcal{L}_{rgb}$: L2 loss for rendered color
- $\mathcal{L}_{eik}$: Eikonal regularization (SDF gradient norm of 1)
- $\mathcal{L}_{flow}$: SDF and motion consistency
- $\mathcal{L}_{photo}$: Photo-consistency (L1 difference of projected colors to neighboring frames)
- $\mathcal{L}_{sdf}$: Consistency between the SDF at each timestamp and the world frame SDF (weight is 0 for the first 200 epochs, then gradually increases)

Two-stage training: Jointly optimize motion + time-dependent NeRF in the early stage; fix poses in the late stage and train the full-scene NeRF solely in the world frame for 5000 epochs.

## Key Experimental Results

### Main Results — Depth Evaluation

| Method | Depth Prior | Co3D AbRel↓ | Co3D δ₁↑ | ScanNet AbRel↓ | ScanNet δ₁↑ |
|---|---|---|---|---|---|
| NeRFmm | No | 0.293 | 0.464 | 0.319 | 0.486 |
| NoPe-NeRF | Yes | 0.176 | 0.721 | 0.141 | 0.828 |
| CF3DGS | Yes | 0.211 | 0.732 | 0.157 | 0.803 |
| **Ours** | **No** | **0.031** | **0.975** | **0.063** | **0.952** |

Depth error decreases by more than 82% (Co3D) and 78% (ScanNet).

### Main Results — Pose Evaluation

| Method | Co3D RPE_t↓ | Co3D RPE_r↓ | Co3D ATE↓ |
|---|---|---|---|
| NeRFmm | 0.500 | 2.785 | 0.054 |
| NoPe-NeRF | 0.281 | 1.449 | 0.050 |
| CF3DGS | 0.097 | 0.402 | 0.011 |
| **Ours** | **0.024** | **0.064** | **0.002** |

Achieves the lowest pose error across all scenes.

### Ablation Study (Co3D Average)

| Setting | AbRel↓ | δ₁↑ | RPE_t↓ | PSNR↑ |
|---|---|---|---|---|
| Full | **0.031** | **0.975** | **0.023** | 27.49 |
| w/o $\mathcal{L}_{flow}$ | 0.084 | 0.877 | 0.114 | 26.31 |
| w/o $\mathcal{L}_{photo}$ | 0.399 | 0.386 | 0.216 | 21.79 |
| w/o Time-dependent NeRF | 0.287 | 0.504 | 0.299 | 24.08 |
| w/o Motion network (frame-by-frame optimization) | 0.080 | 0.912 | 0.028 | 26.23 |

### Key Findings

1. **Photometric consistency is most critical**: Removing $\mathcal{L}_{photo}$ deteriorates AbRel by 12.9× and drops PSNR by 5.7dB.
2. **Time-dependent NeRF is indispensable**: Directly training in the global coordinate system (removing the time dependency) causing pose error to surge by 13×.
3. **SDF flow constraint significantly improves pose accuracy**: Removing it increases RPE_t from 0.023 to 0.114 (5×).
4. **Continuous motion outperforms frame-by-frame optimization**: The motion network improves both depth and PSNR compared to frame-by-frame pose optimization.

## Highlights & Insights

- Continuous motion modeling fundamentally addresses the challenge of joint optimization under large-range motions—by decomposing "large jumps" into "small integrations".
- The combination of time-dependent NeRF and SDF flow constraints is elegant: the former provides local stability, and the latter ensures cross-temporal consistency.
- Completely prior-free, yet it significantly outperforms methods relying on depth priors in both pose and depth accuracy.
- The two-stage training strategy is reasonable: first learning accurate motion, then refining the full scene.

## Limitations & Future Work

- Only applicable to static scenes; dynamic objects will violate the rigid-body assumption.
- NeRF-based methods suffer from slow training speeds.
- Camera intrinsics are assumed to be known (fails to handle zoom, etc.).
- Lacks comparison with more recent 3DGS+pose methods (e.g., InstantSplat).
- Extending continuous motion modeling to the 3DGS framework could be explored for acceleration.

## Related Work & Insights

- **NoPe-NeRF**: A joint optimization baseline requiring depth priors; ours significantly outperforms it without any priors.
- **CF3DGS**: A 3DGS joint optimization method, which is less accurate in geometry compared to the proposed SDF-based scheme.
- **BARF / NeRFmm**: Early joint optimization works, which are limited to forward-facing scenes or require initialization.
- **CasualSAM (Li et al.)**: The theoretical source of the linear relationship between SDF flow and scene flow.

## Rating

⭐⭐⭐⭐ — The idea of continuous motion modeling + time-dependent NeRF shows strong originality, and the experimental results comprehensively outperform prior-based methods; however, being limited to static scenes and the NeRF framework, its utility has some constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)
- [\[CVPR 2026\] Evidential Neural Radiance Fields](../../CVPR2026/3d_vision/evidential_neural_radiance_fields.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2025\] RelationField: Relate Anything in Radiance Fields](relationfield_relate_anything_in_radiance_fields.md)

</div>

<!-- RELATED:END -->
