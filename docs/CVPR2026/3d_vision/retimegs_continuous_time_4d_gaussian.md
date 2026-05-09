---
title: >-
  [Paper Note] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][4D Gaussian Splatting] This paper proposes RetimeGS, which addresses temporal aliasing (ghosting) in 4DGS frame interpolation through regularized temporal opacity (dual-Sigmoid short-tailed distribution) and Catmull-Rom spline trajectories for modeling continuous Gaussian primitive motion, combined with bidirectional optical flow supervision, triple rendering, and dynamic stretching strategies. RetimeGS achieves 30.08 dB PSNR on the Stage-Capture dataset, surpassing the previous SOTA by 1.29 dB.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D Gaussian Splatting
  - continuous-time
  - optical flow
  - spline trajectory
  - temporal aliasing
date: 2026-05-08
content_hash: 9195a07890370297
---

# RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2603.13783](https://arxiv.org/abs/2603.13783)
**Code**: [Project Page](https://william-wang2.github.io/RetimeGS/)
**Area**: 3D Vision / Dynamic Scene Reconstruction
**Keywords**: 4D Gaussian Splatting, continuous-time, optical flow, spline trajectory, temporal aliasing

## TL;DR
This paper proposes RetimeGS, which addresses temporal aliasing (ghosting) in 4DGS frame interpolation through regularized temporal opacity (dual-Sigmoid short-tailed distribution) and Catmull-Rom spline trajectories for modeling continuous Gaussian primitive motion, combined with bidirectional optical flow supervision, triple rendering, and dynamic stretching strategies. RetimeGS achieves 30.08 dB PSNR on the Stage-Capture dataset, surpassing the previous SOTA by 1.29 dB.

## Background & Motivation

**Background**: 4D Gaussian Splatting (4DGS) methods extend 3D Gaussian primitives to the temporal dimension for high-fidelity reconstruction of dynamic scenes. Existing approaches fall into two categories based on temporal parameterization: deformation-field-based methods (modeling dynamics via deformation fields, control points, or physical constraints in canonical space) and 4D primitive-based methods (controlling primitive appearance and disappearance through temporal opacity).

**Limitations of Prior Work**: (a) Deformation-field-based methods assume dynamics arise primarily from geometric motion, struggling with scenes where object visibility or texture appearance changes over time, and producing unreliable correspondence estimation under large motion or limited inter-frame overlap. (b) In 4D primitive-based methods, temporal opacity is supervised only at integer timestamps and lacks regularization, causing overfitting to discrete frames (temporal aliasing) and ghosting during intermediate frame interpolation—semi-transparent overlapping structures from adjacent input frames. (c) Simply low-pass filtering temporal opacity (widening the temporal support of primitives) can resolve aliasing but requires accurate trajectory estimation across multiple frames, which, when it fails, produces a different form of ghosting.

**Key Challenge**: 4D primitives must dynamically appear and disappear to capture visibility changes, yet must simultaneously span the complete temporal interval between input frames; accurate continuous trajectories are required without relying on cross-frame correspondence estimation.

**Goal**: Design a 4DGS representation that enables primitives to produce ghost-free, temporally coherent renderings at arbitrary timestamps, particularly achieving high-quality continuous-time interpolation in low-frame-rate, large-motion scenarios.

**Key Insight**: Treating temporal aliasing as the fundamental problem in 4DGS (analogous to 3D Mip-Splatting addressing spatial aliasing), with three design principles: (i) primitives can dynamically appear and disappear; (ii) regularization prevents degeneration under sparse temporal sampling; (iii) accurate and consistent trajectories are maintained throughout primitive lifetimes.

**Core Idea**: Replace freely optimized temporal distributions with short-tailed temporal opacity, replace linear motion assumptions with Catmull-Rom splines, and provide explicit trajectory supervision via bidirectional optical flow.

## Method

### Overall Architecture
RetimeGS takes multi-view video and corresponding bidirectional optical flow (estimated by WAFT) as input to reconstruct 4D scenes. Each 4D Gaussian primitive's parameters are extended to $(\mu_\tau, \tau_l, \tau_r, \boldsymbol{\mu}, \boldsymbol{v}, \boldsymbol{s}, \boldsymbol{q}(t), \boldsymbol{h}, \sigma)$, where the new parameters control temporal opacity and spatial trajectory. The pipeline comprises four complementary training strategies: bidirectional optical flow trajectory supervision, triple rendering, dynamic stretching with periodic relocation, and flow-aware initialization. VGGT is used to estimate the initial point cloud, MCMC strategy controls primitive density, and all scenes are trained for 20,000 iterations.

### Key Designs

1. **Regularized Temporal Opacity**:

    - **Function**: Defines each primitive's visibility distribution along the temporal axis, controlling its appearance and disappearance.
    - **Mechanism**: Temporal opacity $\sigma_\tau(t)$ is formed by the product of two Sigmoid functions centered at left and right temporal boundaries $\mu_\tau - \tau_l$ and $\mu_\tau + \tau_r$. At initialization, $\mu_\tau = (t_i + t_{i+1})/2$ and $\tau_l = \tau_r = \Delta t / 2$ are set as non-optimizable parameters, ensuring each primitive group is centered and covers the complete interval between two adjacent input frames. At video boundaries, the Sigmoid is replaced by the constant 1 to avoid visibility degradation.
    - **Design Motivation**: The short-tailed distribution prevents primitives from degenerating to a single frame, while adjacent primitive groups blend in and out at input frames to ensure seamless transitions. Unlike stretched Gaussian distributions, short-tailed distributions do not require accurate trajectory estimation across multiple frames.

2. **Catmull-Rom Spline Trajectory**:

    - **Function**: Parameterizes the continuous spatial position $\boldsymbol{x}(t)$ of each primitive within the temporal interval $[t_i, t_{i+1}]$.
    - **Mechanism**: A Catmull-Rom spline is defined with 4 control points. Inner control points $\boldsymbol{p}_1, \boldsymbol{p}_2$ correspond to positions at frames $t_i, t_{i+1}$ (derived from pseudo-mean $\boldsymbol{\mu}$ and velocity $\boldsymbol{v}_2$), while outer control points $\boldsymbol{p}_0, \boldsymbol{p}_3$ are determined by velocities $\boldsymbol{v}_1, \boldsymbol{v}_3$ from adjacent temporal intervals for curvature. Formulas: $\boldsymbol{p}_{1} = \boldsymbol{\mu} - \frac{1}{2}\Delta t \cdot \boldsymbol{v}_2$, $\boldsymbol{p}_{2} = \boldsymbol{\mu} + \frac{1}{2}\Delta t \cdot \boldsymbol{v}_2$.
    - **Design Motivation**: Linear velocity assumptions produce piecewise-linear motion artifacts under sparse temporal sampling; splines enable smooth motion interpolation. Experiments show that optimizing the pseudo-mean and velocity converges more easily than directly optimizing control points.

3. **Bidirectional Flow Supervision**:

    - **Function**: Leverages optical flow to establish coarse inter-frame correspondences, providing explicit supervision for trajectory parameters.
    - **Mechanism**: For each input frame $t_i$, the 3D control point displacements of the preceding and following primitive groups are projected onto the 2D image plane and rasterized into forward/backward flow maps, which are supervised at the pixel level against ground-truth flows $\mathbf{F}^{\mathrm{fwd}}, \mathbf{F}^{\mathrm{bwd}}$. The optical flow learning rate decays exponentially from 0.5 to $10^{-6}$, gradually transitioning to RGB supervision as training stabilizes.
    - **Design Motivation**: RGB supervision alone cannot learn reliable correspondences from sparse temporal inputs; optical flow provides a signal that combines geometric constraints with motion supervision.

4. **Triple Rendering**:

    - **Function**: Addresses uneven coverage where two adjacent primitive groups each reconstruct only partial regions, with the complete scene only emerging after merging.
    - **Mechanism**: For each internal frame $t_i$, three images are rendered: a full rendering using all primitives, a separate rendering using only the preceding primitive group, and a separate rendering using only the following primitive group. All three images are supervised with ground-truth RGB.
    - **Design Motivation**: If only the merged rendering result is supervised, each primitive group may learn to reconstruct only a partial region, leading to under-reconstruction in intermediate frames (e.g., one group missing left-sleeve texture, the other missing right-sleeve texture).

### Loss & Training
- **RGB Loss**: Standard reconstruction loss applied to all three images from triple rendering.
- **Optical Flow Loss**: Pixel-level L2 loss on forward/backward optical flow, with weight decaying exponentially during training.
- **Regularization**: Opacity regularization (weight 0.01) + scale regularization (weight 0.1).
- **Dynamic Stretching**: Every 3,000 iterations, adjacent primitives with similar colors and near-zero velocities are identified and their temporal boundaries are stretched and merged to reduce redundant representation in static regions. Stretched primitives are pruned with probability $1 - 1/(k+1)$.
- **Weighted Relocation**: MCMC strategy executes every 100 iterations; relocation score $s = \sigma / (\tau_l + \tau_r)$ encourages allocation of primitives to dynamic regions.
- **Flow-Aware Initialization**: VGGT estimates the initial point cloud; 3D velocity initial values are estimated via multi-view optical flow back-projection.

## Key Experimental Results

### Main Results
Stage-Capture dataset (9 scenes, 32 synchronized 4K cameras, 22 FPS → 11 FPS for training), foreground region metrics:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Deform-GS | 28.45 | 0.867 | 0.0272 |
| STGS | 25.34 | 0.825 | 0.0357 |
| GaussianFlow | 25.91 | 0.825 | 0.0339 |
| Ex4DGS | 25.95 | 0.811 | 0.0379 |
| 2D Lifting (FILM+STGS) | 28.79 | 0.886 | 0.0267 |
| **RetimeGS** | **30.08** | **0.904** | **0.0225** |

Neural3DV dataset (Flame Steak + Flame Salmon, 30→3 FPS):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Deform-GS | 31.79 | 0.952 | 0.081 |
| STGS | 32.52 | 0.959 | 0.079 |
| 2D Lifting | 33.17 | 0.960 | 0.080 |
| **RetimeGS** | **33.22** | 0.959 | **0.074** |

### Ablation Study
Stage-Capture dataset (foreground region):

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| w/o flow initialization | 29.69 | 0.899 | 0.0227 |
| w/o flow supervision | 27.24 | 0.861 | 0.0282 |
| w/o triple rendering | 27.16 | 0.849 | 0.0319 |
| w/o dynamic stretching | 28.81 | 0.886 | 0.0247 |
| linear trajectory (no spline) | 28.50 | 0.884 | 0.0243 |
| **Full RetimeGS** | **30.08** | **0.904** | **0.0225** |

Optical flow estimator ablation (WAFT vs. SEA-RAFT):

| Flow Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|--------|--------|---------|
| WAFT | 30.08 | 0.904 | 0.0225 |
| SEA-RAFT | 29.73 | 0.898 | 0.0253 |

### Key Findings
- Triple rendering (−2.92 dB) and optical flow supervision (−2.84 dB) are the most critical components; removing either results in severe quality degradation.
- Dynamic stretching identifies approximately 9% of primitives as static across multiple frames under a 1M primitive budget, effectively reducing redundancy, with a temporal sum of approximately 2.26M—equivalent to a 2.26× reduction in primitive count.
- Spline trajectories show clear advantages over linear trajectories in nonlinear motion scenarios such as circular motion (+1.58 dB).
- Training time is approximately 3,794 seconds (vs. 1,407 seconds for STGS), with peak VRAM of 3.14 GB (vs. 2.47 GB for STGS); triple rendering and flow supervision incur additional training overhead.

## Highlights & Insights
- **Temporal aliasing is the fundamental problem in 4DGS**: Overfitting to discrete frame indices causes interpolation failure; the clear formulation of this problem and its systematic resolution represent the paper's most significant contribution.
- **Short-tailed temporal opacity is an elegant design**: It simultaneously allows primitives to dynamically appear and disappear while enforcing coverage of inter-frame intervals, avoiding the respective limitations of deformation-based and 4D primitive-based methods.
- **Optical flow provides a natural motion supervision signal**: Without additional annotation, optical flow serves as both a trajectory supervision signal and a geometric constraint, and integrates well with spline parameterization.
- **Triple rendering is a simple yet effective idea**: A straightforward supervision strategy resolves the uneven coverage issue across primitive groups; the −2.92 dB ablation result is impressive.

## Limitations & Future Work
- The method depends on optical flow estimation quality; when inter-frame motion exceeds approximately 50 pixels (at 1K resolution) or FPS is extremely low (<7.5), optical flow becomes unreliable and the method degrades.
- Training overhead is approximately 2.7× that of STGS (3,794s vs. 1,407s), primarily due to triple rendering and optical flow supervision.
- Discontinuities between adjacent primitive groups at input frames may cause mild flickering; addressing this with a unified 4D representation is a future direction.
- The hyperparameter $\gamma=0.005$ for temporal opacity is sensitive and may require scene-specific tuning.
- Validation is currently limited to stage-capture settings (multi-view synchronized cameras); generalization to in-the-wild monocular video remains unknown.

## Related Work & Insights
- **vs. STGS**: STGS models temporal opacity with a 1D Gaussian but lacks regularization, causing overfitting to discrete frames; RetimeGS addresses this with short-tailed Sigmoid distributions and non-optimizable initialization (+4.74 dB).
- **vs. GaussianFlow**: GaussianFlow introduces forward optical flow supervision but lacks temporal opacity regularization, allowing the optimizer to satisfy flow constraints by shortening primitive temporal support; RetimeGS applies both bidirectional flow supervision and temporal regularization simultaneously (+4.17 dB).
- **vs. Deform-GS**: Deformation-based methods represent all frames with a single set of primitives, capturing coarse global trajectories but failing to establish fine correspondences in regions with fast motion or visibility changes (+1.63 dB).
- **vs. SplineGS**: SplineGS uses splines to drive control points but targets monocular 4D reconstruction and is not suited for multi-view frame interpolation tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The clear formulation of temporal aliasing and its systematic solution are elegantly designed; the combination of short-tailed opacity, spline trajectories, and flow supervision is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Main experiments + complete ablations + per-scene analysis + flow estimator ablation + training efficiency analysis + failure case discussion; the experimental design is highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear, method description is detailed, and the logical chain between motivation and design is complete.
- **Value**: ⭐⭐⭐⭐ Addresses a key problem in 4DGS frame interpolation, representing a substantive advance for the dynamic scene reconstruction field.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[CVPR 2026\] EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy](emgauss_continuous_slice-to-3d_reconstruction_via_dynamic_gaussian_modeling_in_v.md)

<!-- RELATED:END -->
