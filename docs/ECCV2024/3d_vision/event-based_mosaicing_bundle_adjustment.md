---
title: >-
  [Paper Note] Event-based Mosaicing Bundle Adjustment
description: >-
  [ECCV 2024][3D Vision][Event Cameras] This work proposes EMBA, the first photometric Bundle Adjustment method for rotation-only event cameras. It formulates the problem as a regularized non-linear least squares optimization based on a linearized event generation model, designs an efficient solver by exploiting the block-diagonal sparse structure of the normal equation matrix, and simultaneously optimizes the camera rotation trajectory and the panoramic gradient map.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Event Cameras"
  - "Bundle Adjustment"
  - "Panoramic Mosaicing"
  - "Sparse Optimization"
  - "Gradient Map Reconstruction"
date: 2026-05-08
content_hash: 15873c62d9a233c0
---

# Event-based Mosaicing Bundle Adjustment

**Conference**: ECCV 2024  
**arXiv**: [2409.07365](https://arxiv.org/abs/2409.07365)  
**Code**: [Yes](https://github.com/tub-rip/emba)  
**Area**: Event Cameras / Computer Vision  
**Keywords**: Event Cameras, Bundle Adjustment, Panoramic Mosaicing, Sparse Optimization, Gradient Map Reconstruction

## TL;DR

This work proposes EMBA, the first photometric Bundle Adjustment method for rotation-only event cameras. It formulates the problem as a regularized non-linear least squares optimization based on a linearized event generation model, designs an efficient solver by exploiting the block-diagonal sparse structure of the normal equation matrix, and simultaneously optimizes the camera rotation trajectory and the panoramic gradient map.

## Background & Motivation

Event cameras are novel bio-inspired vision sensors that asynchronously detect brightness changes pixel-by-pixel. Compared to frame-based cameras, event cameras possess unique advantages in high dynamic range (HDR), low power consumption, and fast-motion scenarios. Bundle Adjustment (BA) is the core problem for simultaneously optimizing camera motion and scene maps, which is crucial in fields such as panoramic mosaicing, visual odometry, and SLAM.

**Limitations of Prior Work**:

**Lack of Backend Optimization**: Existing rotation estimation methods for event cameras (PF-SMT, RTPT, CMax-$\omega$, CMax-GAE) are all short-term front-end estimations, lacking a BA backend to improve accuracy and consistency.

**Indirect Methods Only**: Existing event BA (e.g., Chin et al., 2019) relies on indirect, feature-matching-based methods, which discard a massive amount of information contained in events and quantize the high temporal resolution.

**Poor Map Quality**: Although CMax-SLAM has a backend, its map is merely an image of warped events (IWE), which cannot recover a grayscale intensity panorama.

**Direct Photometric BA Blank**: Pure event-driven direct (photometric) BA remains a blank in the literature.

**Key Insight / Core Idea**: To exploit the natural property of events—where each event represents a relative brightness measurement—to design a direct photometric BA that simultaneously optimizes camera rotation and an intensity panorama, filling the research gap.

## Method

### Overall Architecture

The workflow of EMBA: (1) obtain initial camera rotations and gradient maps using front-end methods; (2) construct a photometric error objective function based on the Linearized Event Generation Model (LEGM); (3) iteratively optimize using a Levenberg-Marquardt solver exploiting the block-diagonal sparse structure; (4) recover the grayscale intensity panorama from the optimized gradient map via the Poisson equation.

### Key Designs

1. **Linearized Event Generation Model (LEGM) and Objective Function**: Modeling the correlation between events and scene gradients as the core constraint.

    - An event camera triggers an event when the log-brightness change at a pixel reaches a threshold $C$: $\Delta L = s_k C$
    - Linearization under the brightness constancy assumption: $\Delta L \approx \nabla M(\mathbf{p}(t_k)) \cdot \Delta \mathbf{p}(t_k) = s_k C$
    - where $\nabla M$ is the panoramic gradient map, and $\Delta \mathbf{p}$ is the map displacement caused by camera rotation.
    - The objective function is the sum of squared photometric errors: $\min_{\mathbf{P}} g(\mathbf{P}) = \sum_{k=1}^{N_e} (\hat{\Delta L}_k(\mathbf{P}) - \Delta L_k)^2$
    - **Design Motivation**: LEGM naturally establishes a one-to-one correspondence between each event and the gradient of a map point, avoiding explicit data association. Although choosing a linearized model introduces approximation errors, it provides the crucial block-diagonal sparse structure.

2. **Parameterization and Block-Diagonal Sparse Structure**: The key to achieving an efficient solver.

    - The parameters to be optimized $\mathbf{P}$ are divided into two parts: camera control poses $\boldsymbol{\alpha} \in \mathbb{R}^{3N_{\text{poses}}}$ and panoramic gradient map pixels $\boldsymbol{\beta} \in \mathbb{R}^{2N_p}$.
    - The camera trajectory is parameterized using linear interpolation splines, and the rotation is updated using the Lie Group LM method.
    - The normal equations naturally partition into blocks as: $\begin{pmatrix} A_{11} & A_{12} \\ A_{12}^\top & A_{22} \end{pmatrix} \begin{pmatrix} \Delta P_\alpha^* \\ \Delta P_\beta^* \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}$
    - **Key Findings**: Since each error term $(e)_k$ only depends on the gradient of a single map point, $A_{22}$ possesses a $2 \times 2$ block-diagonal structure. Its inversion complexity is only $O(N_p)$, allowing an efficient solution via the Schur complement.
    - **Design Motivation**: Directly storing and operating on the Jacobian matrix $J \in \mathbb{R}^{N_e \times (3N_\text{poses} + 2N_p)}$ is infeasible for millions of event data. Exploiting the sparse structure is key to achieving a practically viable system.

3. **Map Regularization**: Preventing optimization divergence.

    - Adding an $L^2$ regularization term: $\min_{\{R_i\}, \nabla M} \|e(\{R_i\}, \nabla M)\|^2 + \eta \|\nabla M\|^2$
    - Regularization only adds $\eta I$ to the diagonal of $A_{22}$, which does not destroy the block-diagonal structure.
    - Distinguishing "active pixels" (receiving >5 events) from "inactive pixels", where inactive pixels have their gradients set to zero purely via regularization.
    - **Design Motivation**: In pure photometric error optimization, gradients of certain pixels might grow excessively fast, inhibiting updates to other pixels. Regularization ensures stable convergence.

### Loss & Training

- Objective Function: Regularized non-linear least squares $\|e\|^2 + \eta \|\nabla M\|^2$
- Solver: Levenberg-Marquardt method + Schur complement
- Panoramic Image Reconstruction: Recovering the intensity map $M$ from the optimized gradient map $\nabla M$ via the Poisson equation.
- Default camera control pose frequency is 20 Hz, and the map size is $1024 \times 512$ px.

## Key Experimental Results

### Main Results

Photometric error on synthetic data ($\times 10^6$), using CMax-$\omega$ initialization:

| Scene | Before | After (EMBA) | Relative Reduction |
|------|--------|---------------|----------|
| playroom | 0.326 | **0.151** | **54.5%** |
| bicycle | 0.552 | **0.295** | 46.6% |
| city | 2.714 | **1.978** | 27.1% |
| street | 1.895 | **1.336** | 29.5% |
| town | 1.917 | **1.425** | 25.7% |
| bay | 2.303 | **1.827** | 20.7% |

Rotation error RMSE (°) on synthetic data, CMax-$\omega$ initialization:

| Scene | Before | After | Description |
|------|--------|--------|------|
| city | 1.532 | **0.973** | 36.5% reduction |
| town | 1.905 | **0.858** | 55.0% reduction |
| street | 0.965 | **0.744** | 22.9% reduction |

Photometric error on real data ($\times 10^5$), using CMax-$\omega$ initialization:

| Sequence | Before | After | Reduction |
|------|--------|--------|----------|
| shapes | 0.575 | **0.361** | 37.2% |
| poster | 4.368 | **2.579** | 40.9% |
| boxes | 3.921 | **2.250** | 42.6% |
| dynamic | 3.049 | **2.130** | 30.1% |

### Ablation Study

Runtime analysis (seconds), real data:

| Step | shapes | poster | boxes | dynamic |
|------|--------|--------|-------|---------|
| Objective Evaluation | 1.114 | 8.873 | 7.436 | 5.837 |
| Construct Normal Equations | 0.300 | 2.366 | 2.106 | 1.574 |
| Schur Solver | 0.429 | 2.013 | 2.006 | 1.656 |
| CG Solver (Comparison) | 0.267 | 3.127 | 3.561 | 2.056 |
| Active Pixels | 6,913 | 50,738 | 49,357 | 41,313 |
| Event Count | 1.78M | 12.59M | 10.76M | 8.80M |

Joint use with the CMax-SLAM backend: After initializing with EMBA, the rotation RMSE is further reduced from 0.470° to **0.377°**, indicating that the two methods are complementary.

### Key Findings

- EMBA improves the results under initializations from all four front-end methods (EKF-SMT, CMax-GAE, CMax-$\omega$, RTPT).
- Photometric error is reduced by 30%–54.5%, and the map quality is visually significantly improved: blurred regions become sharp, and hidden details are revealed.
- The Schur complement solver is faster than the conjugate gradient (CG) solver in large-scale scenarios.
- Even without an initial map, EMBA can recover high-quality panoramas from scratch (VGA/HD event camera experiments).

## Highlights & Insights

- The first purely event-driven direct (photometric) Bundle Adjustment method, filling an important research gap.
- Clear theoretical contribution: converts the structural properties of LEGM to the block-diagonal sparsity of $A_{22}$, designing an efficient solver with a complexity of $O(N_p)$.
- The reconstruction pipeline from the gradient map to the intensity panorama (via the Poisson equation) is simple and elegant.
- Produces impressive outdoor panoramas on VGA and HD (1.28 million pixels) event cameras.

## Limitations & Future Work

- Assumes purely rotational motion and a static scene; evaluation on real handheld sequences involving translation is challenging.
- Highly textured scenes that generate massive volumes of events can slow down the algorithm.
- The local convergence of the LM method means that the quality of initialization is crucial.
- The linearization (LEGM) introduces approximation errors; non-linear event generation models can be explored in the future.
- Can be extended to 6-DOF BA and dynamic scene handling.

## Related Work & Insights

- Compared to mature BA methods in frame-based cameras (e.g., DSO, ORB-SLAM), research on event BA is still in its infancy; this work opens up a new direction.
- CMax-SLAM provides a complementary front-end + back-end combination, paving the way for constructing a complete event SLAM system in the future.
- The utilization of the block-diagonal sparse structure can inspire the solver design for other optimization problems in event cameras.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First event-direct photometric BA with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic + real data + multiple front-ends + runtime analysis, but real-world evaluation is limited by the pure rotation assumption.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, compact structure.
- **Value**: ⭐⭐⭐⭐ — Lays the foundation for backend optimization in event-camera SLAM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)
- [\[CVPR 2026\] Parallel Rigidity Matters for Bundle Adjustment](../../CVPR2026/3d_vision/parallel_rigidity_matters_for_bundle_adjustment.md)
- [\[ICCV 2025\] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](../../ICCV2025/3d_vision/back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)
- [\[ECCV 2024\] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](bad-gaussians_bundle_adjusted_deblur_gaussian_splatting.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)

</div>

<!-- RELATED:END -->
