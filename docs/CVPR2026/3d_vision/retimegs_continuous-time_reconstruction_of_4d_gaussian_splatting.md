---
title: >-
  [Paper Note] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting
description: >-
  [CVPR2026][3D Vision][4D Gaussian Splatting] RetimeGS is proposed to address ghosting artifacts and temporal aliasing in 4DGS during inter-frame interpolation, through regularized temporal opacity, Catmull-Rom spline trajectories, bidirectional optical flow supervision, and triple rendering, enabling artifact-free continuous-time 4D reconstruction at arbitrary timestamps.
tags:
  - CVPR2026
  - 3D Vision
  - 4D Gaussian Splatting
  - dynamic scene reconstruction
  - temporal interpolation
  - optical flow supervision
  - Catmull-Rom spline
  - temporal aliasing
date: 2026-05-08
content_hash: d4f2b511b191e355
---

# RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting

**Conference**: CVPR2026
**arXiv**: [2603.13783](https://arxiv.org/abs/2603.13783)
**Code**: None
**Area**: 3D Vision
**Keywords**: 4D Gaussian Splatting, dynamic scene reconstruction, temporal interpolation, optical flow supervision, Catmull-Rom spline, temporal aliasing

## TL;DR

RetimeGS is proposed to address ghosting artifacts and temporal aliasing in 4DGS during inter-frame interpolation, through regularized temporal opacity, Catmull-Rom spline trajectories, bidirectional optical flow supervision, and triple rendering, enabling artifact-free continuous-time 4D reconstruction at arbitrary timestamps.

## Background & Motivation

High-fidelity reconstruction of dynamic scenes is a fundamental problem in CV/CG. A key requirement is **retime control**—rendering dynamic scenes at arbitrary timestamps while maintaining temporal consistency—for applications such as slow-motion playback, high-frame-rate VR rendering, and bullet-time VFX effects. This inherently requires generating continuous intermediate frames between discrete input frames.

### Two Dominant Paradigms and Their Limitations

**Paradigm 1: Deformation-based methods** (Deform-GS, MotionGS, etc.) model geometry and appearance in a canonical space and capture dynamics via deformation fields, control points, or physical constraints:

- Assume dynamics arise purely from geometric motion; fail when **visibility or appearance changes over time**
- Rely on accurate point correspondence estimation, which becomes unreliable under **large motion or limited inter-frame overlap**
- Accumulate spatially misaligned signals on the same primitive due to erroneous correspondences, causing visual artifacts and incorrect trajectories

**Paradigm 2: 4D primitive methods** (STGS, Ex4DGS, etc.) represent dynamic scenes directly with 4D primitives, decomposing opacity into base opacity × spatial 3D Gaussian × temporal 1D Gaussian:

- Core issue: temporal opacity is only supervised at **discrete integer frames** with no regularization
- Learned opacity **overfits to discrete frames** (temporal aliasing: temporal support collapses to sub-frame level)
- Intermediate-frame rendering produces characteristic **ghosting artifacts**—semi-transparent overlapping structures from adjacent input frames statically superimposed
- Relatively benign for small motion or high frame-rate data, but **severe under large motion**

The intuitive fix is to apply low-pass filtering to temporal opacity (analogous to Mip-Splatting's spatial anti-aliasing), but stretched temporal distributions require accurate cross-frame trajectory estimation; otherwise, a different form of ghosting is introduced.

### Design Principles

Based on the above analysis, the representation in RetimeGS must satisfy three principles:

**Dynamic appearance/disappearance** — Capture changes in appearance and visibility, overcoming the limitations of deformation-based methods

**Regularization against collapse** — Prevent degenerate clustering at discrete frames under sparse temporal sampling

**Accurate and consistent trajectories** — Maintain smooth, accurate motion throughout a primitive's lifetime to avoid inconsistency-induced ghosting

## Method

### Overall Architecture

The input consists of multi-view video and corresponding bidirectional optical flow (precomputed by WAFT); the output is a 4D scene representation renderable at arbitrary time $t$. The core contributions lie in the 4D representation design and four training strategies.

### Key Design 1: 4D Primitive Representation

Building upon standard 3DGS parameters $(x, s, h, q, \sigma)$, each Gaussian primitive is extended to:

$$(\mu_\tau,\ \tau_l,\ \tau_r,\ \boldsymbol{\mu},\ \boldsymbol{v},\ \boldsymbol{s},\ \boldsymbol{q}(t),\ \boldsymbol{h},\ \sigma)$$

where the new parameters denote:

- $\mu_\tau$: temporal mean; $\tau_l, \tau_r$: left and right temporal boundary offsets, defining temporal opacity
- $\boldsymbol{\mu}$: pseudo spatial mean; $\boldsymbol{v} = (v_1, v_2, v_3)$: velocity components, jointly defining the spline trajectory with $\mu$
- Rotation $q(t)$ is modeled as a low-order polynomial in time

At any time $t$, the standard 3DGS parameters $(\boldsymbol{x}(t), \boldsymbol{s}, \boldsymbol{q}(t), \boldsymbol{h}, \sigma_\tau(t), \sigma)$ can be derived from these, and rendering proceeds via standard Gaussian Splatting projection, depth sorting, and alpha compositing.

### Key Design 2: Regularized Temporal Opacity (Short-Tailed Sigmoid Kernel)

**Initialization constraint**: Temporal mean and boundary offsets are initialized as non-learnable, set to the midpoint and half-interval of adjacent frame pairs:

$$\mu_\tau = \frac{t_i + t_{i+1}}{2}, \quad \tau_l = \tau_r = \frac{\Delta t}{2}$$

**Short-tailed temporal kernel**: Temporal opacity is defined as the product of two sigmoid functions, smoothly decaying at the left and right boundaries:

$$\sigma_\tau(t) = \tilde{\psi}_l\left(\frac{t - (\mu_\tau - \tau_l)}{\gamma}\right) \cdot \tilde{\psi}_r\left(\frac{(\mu_\tau + \tau_r) - t}{\gamma}\right)$$

At global boundaries (beginning and end of the video), the corresponding sigmoid is replaced by a constant 1 to prevent visibility degradation at the boundary. $\gamma=0.005$ ensures the short-tailed property.

**Design intuition**: Each group of primitives is centered to cover the interval between two adjacent input frames and is supervised by both frames. Near an input frame, two adjacent primitive groups smoothly blend in and out, ensuring seamless transitions.

### Key Design 3: Catmull-Rom Spline Spatial Trajectory

Regularizing temporal opacity alone is insufficient—under sparse temporal input, moving objects have almost no content overlap between adjacent frames, and RGB supervision cannot learn reliable correspondences. A linear velocity assumption produces piecewise-linear artifacts under large motion.

Therefore, **Catmull-Rom splines** are used to model the spatial mean $\boldsymbol{x}(t)$, with parameters explicitly supervised by bidirectional optical flow:

- For a primitive with temporal mean at $(t_i + t_{i+1})/2$:
    - $v_2$: linear velocity from frame $t_i$ to $t_{i+1}$ (3D correspondence)
    - $v_1$: velocity from frame $t_{i-1}$ to $t_i$; $v_3$: velocity from frame $t_{i+1}$ to $t_{i+2}$
    - $\boldsymbol{\mu}$ is the position at $\mu_\tau$ under linear motion assumption

The four spline control points are directly derived from these parameters:

- Inner control points (spline passes through exactly) = positions at frames $t_i$ and $t_{i+1}$: $p_1 = \mu - \frac{1}{2}\Delta t \cdot v_2$, $p_2 = \mu + \frac{1}{2}\Delta t \cdot v_2$
- Outer control points determine curvature at inner points: $p_0 = p_1 - \Delta t \cdot v_1$, $p_3 = p_2 + \Delta t \cdot v_3$

For static primitives, velocities are approximately zero; even when temporal support is stretched, extrapolation maintains a consistent static position. Experiments show that optimizing the pseudo-mean and velocity components is significantly easier than directly optimizing four control points, despite mathematical equivalence.

### Training Strategy 1: Bidirectional Optical Flow Trajectory Supervision

Coarse correspondences from forward and backward optical flow supervise the trajectory parameters $(\mu, v)$:

- At frame $t_i$, the 3D displacement between control points of two adjacent primitive groups is projected to 2D and rasterized into forward/backward flow maps
- During rasterization, temporal opacity is normalized by dividing by $\sigma_\tau(t_i)$ (since the two groups are rendered separately)
- Pixel-wise loss is computed against GT optical flow
- In later training stages, the optical flow learning rate is gradually decayed to zero, transitioning fully to RGB fine-tuning

### Training Strategy 2: Triple Rendering

**Problem**: Rendering all primitives jointly reconstructs the input frames, but each of two adjacent primitive groups individually covers only a disjoint spatial region, leading to under-reconstruction when rendered separately.

**Solution**: For each interior frame $t_i$, three images are rendered—(1) all primitives jointly; (2) the preceding group alone; (3) the following group alone—and all three are supervised against GT. Boundary frames have only one primitive group and are rendered once.

### Training Strategy 3: Dynamic Stretching and Periodic Relocation

- **Dynamic stretching**: After training stabilizes, the nearest-neighbor primitives in adjacent groups are inspected; if their base colors are similar and velocities near zero, $\tau_l$ and $\tau_r$ are stretched to cover a larger temporal range, and redundant primitives are pruned with probability $1 - 1/(k+1)$
- **Effect**: Static regions are represented with fewer primitives, freeing capacity under MCMC budget constraints for dynamic regions
- In experiments, approximately 9% of primitives are static long-duration primitives, reducing the effective primitive count by **2.26×**
- **Relocation scoring**: $s = \sigma / (\tau_l + \tau_r)$, weighting base opacity by temporal duration to encourage relocation toward dynamic regions

### Training Strategy 4: Flow-Aware Initialization

VGGT (without bundle adjustment) is used to coarsely estimate per-frame point clouds. 2D optical flow is back-projected to 3D across multiple views and averaged to obtain initial 3D velocity estimates. These velocities initialize all velocity components $v_1, v_2, v_3$, and displacement estimates initialize the pseudo-mean $\boldsymbol{\mu}$.

### Loss & Training

- RGB reconstruction loss + optical flow loss (learning rate decayed from 0.5 to $10^{-6}$ after 12K iterations) + opacity regularization (0.01) + scale regularization (0.1)
- MCMC relocation every 100 iterations (minimum opacity threshold 0.01); dynamic stretching every 3K iterations
- Total training: 20K iterations; learning rate decay applied to all attributes after 18K iterations
- Single RTX 4090D GPU; data downscaled to 1K resolution

## Key Experimental Results

### Datasets and Evaluation Setup

- **DNA-Rendering**: 10 scenes, 60 cameras at 4K/2K resolution, 15 FPS, 17 frames (qualitative evaluation)
- **Stage-Capture** (self-collected): 9 new scenes, 32 synchronized 4K cameras, 22 FPS → downsampled to effective 11 FPS by skipping frames; retained frames serve as intermediate-frame GT (quantitative evaluation)
- Metrics: foreground-region PSNR/SSIM + masked-background LPIPS

### Main Results

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:---|:---:|:---:|:---:|
| Deform-GS | 28.45 | 0.867 | 0.0272 |
| STGS | 25.34 | 0.825 | 0.0357 |
| GaussianFlow | 25.91 | 0.825 | 0.0339 |
| Ex4DGS | 25.95 | 0.811 | 0.0379 |
| 2D Lifting (FILM+STGS) | 28.79 | 0.886 | 0.0267 |
| **RetimeGS (Ours)** | **30.08** | **0.904** | **0.0225** |

RetimeGS achieves the best performance across all three metrics. Compared to the strongest baseline 2D Lifting, PSNR improves by **+1.29 dB**; compared to the analogous 4D primitive method STGS, PSNR improves by **+4.74 dB**.

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:---|:---:|:---:|:---:|
| w/o flow initialization | 29.69 | 0.899 | 0.0227 |
| w/o flow supervision | 27.24 | 0.861 | 0.0282 |
| w/o triple rendering | 27.16 | 0.849 | 0.0319 |
| w/o dynamic stretching | 28.81 | 0.886 | 0.0247 |
| Linear trajectory | 28.50 | 0.884 | 0.0243 |
| **Full RetimeGS** | **30.08** | **0.904** | **0.0225** |

### Key Findings

1. **Triple rendering** has the largest impact (−2.92 dB)—without it, each primitive group covers only a partial spatial region; ablation visualizations clearly show the preceding group missing right-side texture and the following group missing left-side texture
2. **Optical flow supervision** is second in importance (−2.84 dB)—removing it causes severe texture distortion for fast-moving objects
3. **Spline vs. linear trajectory** (−1.58 dB)—in circular-motion scenes, difference error maps show significantly reduced errors along object boundaries
4. **Dynamic stretching** (−1.27 dB)—88K out of 1M primitives are stretched static primitives, freeing capacity for dynamic regions
5. **Flow initialization** contributes the least (−0.39 dB), providing a reasonable starting point that accelerates convergence
6. GaussianFlow introduces forward optical flow trajectory supervision but no temporal regularization; the optimizer can still shrink temporal support while satisfying flow constraints, leaving ghosting artifacts intact

## Highlights & Insights

1. **Precise problem diagnosis**: Ghosting in 4D primitive methods is attributed to temporal aliasing, forming an analogous framework to the spatial aliasing addressed by Mip-Splatting
2. **Pseudo-mean + velocity parameterization**: Although mathematically equivalent to directly optimizing four control points, this parameterization yields a far more favorable optimization landscape—an excellent design choice
3. **Triple rendering is simple yet effective**: By requiring each primitive group to independently explain input frames, it fundamentally resolves uneven spatial coverage
4. **Dynamic stretching yields multiple benefits**: Reduces redundant primitives, releases budget for dynamic regions, and accumulates cross-frame supervision for static regions to reduce flickering
5. **Elegant use of optical flow**: Initialization + bidirectional supervision + automatic learning-rate decay in late training stages exploits optical flow coarse-to-fine without overfitting to noisy estimates

## Limitations & Future Work

1. **Failure under very low frame rates**: When inter-frame motion exceeds approximately 50 pixels (@1K), optical flow becomes unreliable and intermediate frames exhibit artifacts; fast dancing at 7.5 FPS already shows noticeable degradation
2. **Mild flickering**: The disjoint nature of adjacent primitive groups may still cause minor temporal discontinuities at input frames
3. **Dependency on precomputed optical flow**: The quality of WAFT flow directly affects results, increasing preprocessing complexity
4. **Appearance changes not modeled**: SH coefficients do not vary over time, potentially limiting performance in scenes with drastic illumination changes
5. **Future directions**: Incorporating video diffusion models as motion priors to handle extremely large motion; unifying the 4D representation to eliminate inter-group boundary discontinuities

## Related Work & Insights

- **Mip-Splatting**: Its spatial anti-aliasing solution inspires RetimeGS to extend analogous ideas to the temporal dimension, while noting that naive low-pass filtering is infeasible without accompanying trajectory design
- **GaussianFlow**: First introduced optical flow trajectory supervision but is shown to be insufficient alone; temporal opacity regularization is also necessary
- **STGS**: Canonical 4D primitive baseline; unconstrained temporal opacity leading to temporal aliasing is its archetypal failure mode
- **SplineGS**: Pioneer of spline-based trajectories, designed for monocular settings; RetimeGS generalizes this to multi-view scenarios with bidirectional flow constraints

## Rating

| Dimension | Score (1–10) |
|:---|:---:|
| Novelty | 7 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 9 |
| Value | 7 |
| **Overall** | **7.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)

</div>

<!-- RELATED:END -->
