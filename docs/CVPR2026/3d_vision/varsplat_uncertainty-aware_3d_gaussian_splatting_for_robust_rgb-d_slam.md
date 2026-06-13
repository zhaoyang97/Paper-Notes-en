---
title: >-
  [Paper Note] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] This paper presents VarSplat, the first 3DGS-SLAM system that learns a **per-splat appearance variance** $\sigma^2$ and renders a **per-pixel uncertainty map** $V$ via the law…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "SLAM"
  - "uncertainty modeling"
  - "RGB-D"
  - "alpha compositing"
date: 2026-05-08
content_hash: 1bda8aaa170a2941
---

# VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

**Conference**: CVPR2026
**arXiv**: [2603.09673](https://arxiv.org/abs/2603.09673)  
**Code**: [Project Page](https://anhthuan1999.github.io/varsplat/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, SLAM, uncertainty modeling, RGB-D, alpha compositing

## TL;DR

This paper presents VarSplat, the first 3DGS-SLAM system that learns a **per-splat appearance variance** $\sigma^2$ and renders a **per-pixel uncertainty map** $V$ via the law of total variance. The uncertainty is uniformly applied to tracking, submap registration, and loop detection, achieving robust and state-of-the-art performance across four datasets.

## Background & Motivation

3DGS-SLAM achieves fast differentiable rendering via rasterization of anisotropic Gaussians, surpassing NeRF-SLAM in reconstruction quality and speed. However, existing methods share a critical limitation: **measurement reliability is not explicitly modeled**. In the presence of low-texture regions, transparent or reflective surfaces, or depth discontinuity boundaries, uniform photometric weighting leads to pose estimation drift.

Shortcomings of existing uncertainty modeling approaches:

- **Geometric uncertainty** (e.g., depth variance in CG-SLAM, per-pixel depth uncertainty in UncLe-SLAM): models only the geometric dimension, ignoring appearance instability.
- **Pretrained predictors** (e.g., WildGS-SLAM uses DINOv2 features to predict uncertainty maps): relies on external models and cannot be optimized end-to-end.
- **Ray termination probability** (e.g., Uni-SLAM's termination-probability field): uncertainty does not originate from the rasterizer itself.

VarSplat's core idea is to **directly learn a per-Gaussian appearance variance $\sigma_i^2$, propagate it to per-pixel uncertainty $V$ via the law of total variance and alpha compositing, all within a single rasterization pass**.

## Core Problem

1. How to explicitly model appearance uncertainty in 3DGS without introducing additional networks or pretrained models?
2. How to efficiently propagate per-splat variance into a per-pixel uncertainty map?
3. How to unify uncertainty across the three key components of SLAM: tracking, registration, and loop detection?

## Method

### 3.1 Per-Pixel Uncertainty Rendering

**Extended splat representation.** Building on standard 3DGS, each Gaussian $G_i$ is augmented with an additional **appearance variance parameter** $\sigma_i^2 \in \mathbb{R}^3$ (three channels) beyond the standard mean position $\mu_i$, opacity $\alpha_i$, scale $s_i$, covariance $\Sigma_i$, and spherical harmonic color $c_i$. This parameter encodes the degree of uncertainty around the mean color of each splat. Each submap is defined as:

$$P^s = \{G_i^s(\mu_i, \Sigma_i, \alpha_i, s_i, c_i, \sigma_i^2) \mid i=1,\ldots,N^s\}$$

**Intuition.** $\sigma_i^2$ is fundamentally different from the spatial covariance $\Sigma_i$ (which defines geometric extent) and the SH coefficients (which define mean appearance). Even when SH correctly models view-dependent mean color, small viewpoint changes near depth discontinuities, occlusion boundaries, or reflective surfaces can alter the visibility and alpha weights of overlapping splats, producing inconsistent color observations—situations in which $\sigma_i^2$ learns large values.

**Standard alpha compositing.** Transmittance and blending weights:

$$w_i = T_i \alpha_i, \quad T_i = \prod_{j}^{i-1}(1-\alpha_j)$$

Rendered color $C$ and depth $D$:

$$C = \sum_i w_i c_i, \quad D = \sum_i w_i z_i$$

**Variance rendering via the law of total variance.** For a random variable $X$ (pixel color) conditioned on $Z$ (3D Gaussians), the law of total variance decomposes the per-pixel variance into two terms:

$$\text{Var}[X] = \mathbb{E}[\text{Var}[X|Z]] + \text{Var}(\mathbb{E}[X|Z])$$

- **First term** (expected per-splat variance): obtained directly via alpha compositing as $\sum_i w_i \sigma_i^2$.
- **Second term** (variance of splat means): computed using the second-moment formula $\sum_i w_i c_i^2 - (\sum_i w_i c_i)^2$.

Combining these yields the final per-pixel uncertainty $V$:

$$V = \sum_i w_i(\sigma_i^2 + c_i^2) - \left(\sum_i w_i c_i\right)^2$$

**Key advantage.** The computation of $V$ shares the same single rasterization pass as color and depth rendering, requiring no additional forward passes or Monte Carlo sampling, thereby preserving real-time efficiency.

### 3.2 Mapping

**Submap management.** Following the submap strategy of LoopSplat/Gaussian-SLAM, a new submap is created when the camera moves beyond a spatial threshold from the submap centroid or when cumulative tracking uncertainty exceeds a preset limit. Gaussians are initialized via depth unprojection on the first frame; subsequent frames add new Gaussians in unobserved regions or merge overlapping ones.

**Mapping loss:**

$$\mathcal{L}_{\text{map}} = \lambda_{\text{color}} \cdot \mathcal{L}_{\text{color}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{depth}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}}$$

where:

- **Color loss**: weighted combination of L1 and SSIM: $\mathcal{L}_{\text{color}} = (1-\lambda_{\text{SSIM}})\|\hat{I}-I\|_1 + \lambda_{\text{SSIM}}(1-\text{SSIM}(\hat{I},I))$
- **Depth loss**: $\mathcal{L}_{\text{depth}} = \|\hat{D}-D\|_1$
- **Regularization**: controls Gaussian scale via $\mathcal{L}_{\text{reg}} = \|\hat{s}-s\|_1$
- **Variance loss**: based on the Gaussian negative log-likelihood

**Learning variance from scratch.** Inspired by the likelihood perspective of ActiveNeRF, the variance loss takes the form of a Gaussian negative log-likelihood:

$$\mathcal{L}_{\text{var}} = \frac{1}{2V}\left(\|\hat{I}-I\|_2^2 + \|\hat{D}-D\|_2^2\right) + \log(V)$$

Design considerations:
- **Squared L2 (MSE)** is used rather than L1 as the residual, since L1 corresponds to the scale parameter of a Laplace distribution, which would violate the Gaussian model assumption.
- Both color and depth residuals are incorporated, so that the variance reflects the combined reliability of geometry and appearance.
- Gradient analysis: $\frac{\partial \mathcal{L}_{\text{var}}}{\partial \sigma_i^2} = \frac{\partial \mathcal{L}_{\text{var}}}{\partial V} \cdot w_i$, i.e., variance is propagated to each splat through the alpha weight $w_i$.

### 3.3 Downstream Pose Estimation

**Uncertainty-normalized weights.** Median-centered log-scaling is applied to compute weights for pixel-level and submap-level variance respectively:

$$\widetilde{w}_p = \exp[-(\log V - \widetilde{V})/\tau], \quad \widetilde{V} = \text{median}_\Omega(\log V)$$

$$\widetilde{w}_s = \exp[-(\log \sigma^2 - \widetilde{\sigma^2})/\tau], \quad \widetilde{\sigma^2} = \text{median}(\log \sigma^2)$$

Pixels or splats with above-median variance receive attenuated weights, while reliable regions receive stronger supervision. $\tau > 0$ controls the sharpness of the weighting.

**Tracking.** Given an input frame $(I, D)$, the current pose $T_j$ is estimated. RGB images are more susceptible to viewpoint changes, low-texture regions, and occlusions; uncertainty weights are therefore used to adaptively constrain the optimization:

$$\mathcal{L}_{\text{track}} = \sum \lambda_c (\widetilde{w_p} \odot \|\hat{I}-I\|_1) + (1-\lambda_c)\|\hat{D}-D\|_1$$

Critically, during tracking the **variance parameters are frozen** and gradients through $\widetilde{w_p}$ are stopped, preventing conflicts with pose optimization.

**Loop Detection.** Per-splat variance $\sigma_i^2$ is used to modulate submap similarity. A variance-weighted opacity ratio is computed:

$$r = \frac{\sum_j \widetilde{w_s} \alpha_j}{\sum_j \alpha_j}, \quad \text{sim} = \text{cross\_sim} \odot (r_q \cdot r_{db})$$

This ratio encodes how much reliable appearance information remains in the submap, without requiring per-submap penalties.

**Registration.** Upon loop detection, the query keyframe is localized within the database submap using an uncertainty-weighted photometric loss:

$$\mathcal{L}_{\text{registration}} = \sum \widetilde{w_p} \odot \|\hat{I}-I\|_1 + \|\hat{D}-D\|_1$$

**Global merging.** All submaps are merged via TSDF fusion, with fused geometry used to initialize global Gaussian centers; the result is then refined using $\mathcal{L}_{\text{color}}$. Uncertainty weights are not applied at this stage, as unreliable regions have already been suppressed in the preceding steps.

## Key Experimental Results

### Tracking Performance (ATE RMSE ↓, cm)

| Dataset | Best Baseline | VarSplat | Gain |
|--------|--------------|----------|------|
| Replica (8-scene avg.) | LoopSplat: 0.26 | **0.23** | ~12% |
| ScanNet++ (5-scene avg.) | LoopSplat: 2.05 | **1.69** | ~18% |
| TUM-RGBD (5-scene avg.) | LoopSplat: 3.33 | **3.20** | ~4% |
| ScanNet (6-scene avg.) | Loopy-SLAM: 7.7 | **6.5** | ~16% |

### Rendering and Reconstruction Performance

| Metric | Dataset | VarSplat | Prev. SOTA (LoopSplat) |
|--------|---------|----------|------------------------|
| PSNR ↑ | Replica | 37.15 | 36.63 |
| SSIM ↑ | Replica | 0.986 | 0.985 |
| LPIPS ↓ | Replica | 0.109 | 0.112 |
| Depth L1 ↓ | Replica | 0.50 | 0.51 |
| F1 ↑ | Replica | 90.2% | 90.4% |
| NVS PSNR ↑ | ScanNet++ | **21.33** | 21.30 |

### Ablation Study

Incremental effect of enabling uncertainty in each component (ScanNet 6-scene avg. ATE RMSE):

- No uncertainty: 8.20 → +Tracking: 7.63 → +Loop: 7.49 → +Registration (all enabled): **6.53**, total gain ~**20%**

Runtime (Replica/Room0, A100 80GB): Mapping 1.9 s/frame, Tracking 2.0 s/frame, comparable to LoopSplat (1.2 s / 1.8 s).

## Highlights & Insights

1. **Mathematical elegance**: Per-splat variance is propagated to per-pixel uncertainty via the law of total variance without Monte Carlo sampling or additional forward passes—entirely within a single rasterization pass.
2. **End-to-end learning**: Variance $\sigma_i^2$ is a differentiable parameter jointly optimized with pose and Gaussian parameters, without relying on pretrained models.
3. **Unified uncertainty application**: The same variance signal drives tracking (pixel-level), loop detection (submap-level), and registration (pixel-level).
4. **Freezing strategy**: Variance parameters are selectively frozen during tracking and loop detection to avoid gradient conflicts—a principled design choice.
5. **Notable robustness gains**: Performance on real-world datasets (ScanNet / ScanNet++ / TUM-RGBD) is substantially more stable than baselines.

## Limitations & Future Work

1. **RGB-D only**: The method is not extended to monocular or stereo settings, limiting its applicability.
2. **Increased computational cost**: Mapping time increases from 1.2 s/frame (LoopSplat) to 1.9 s/frame (+58%), which may be prohibitive for latency-sensitive applications.
3. **Uncertainty discarded during merging**: The global refinement stage after TSDF fusion does not use $V$, potentially sacrificing final reconstruction quality.
4. **Variance modeling assumptions**: Variance is modeled as isotropic per-channel $\sigma_i^2 \in \mathbb{R}^3$; inter-channel covariance is not captured.
5. **Dynamic scenes**: Dynamic objects are not handled; the method may fail in non-static environments.

## Related Work & Insights

| Method | Uncertainty Type | Source | Online Learning | Single Pass |
|--------|----------------|--------|----------------|-------------|
| CG-SLAM | Depth variance | Geometry-driven | ✓ | ✓ |
| Uni-SLAM | Ray termination probability | Implicit field | ✓ | ✗ |
| WildGS-SLAM | DINOv2 feature map | Pretrained | ✗ | ✓ |
| ActiveNeRF | Per-pixel variance | Neural network | ✓ | ✗ |
| **VarSplat** | **Per-splat appearance variance** | **Law of total variance** | **✓** | **✓** |

VarSplat's core advantage lies in the fact that uncertainty originates directly from the 3DGS representation itself (rather than an external model), is propagated via a closed-form formula (rather than sampling), and is optimized online end-to-end (rather than as post-processing).

The decomposition $V = \mathbb{E}[\text{Var}] + \text{Var}(\mathbb{E})$ via the law of total variance is generalizable to uncertainty estimation for other Gaussian attributes (e.g., semantics, normals). The proposed **freezing strategy**—selectively freezing or training variance parameters at different stages to avoid gradient conflicts—offers broadly applicable guidance for multi-task joint optimization. The uncertainty-weighted supervision paradigm is also transferable to other 3DGS-based tasks, including active viewpoint selection for novel view synthesis, scene completion, and semantic segmentation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The derivation combining the law of total variance with alpha compositing is clean and elegant, representing a natural yet novel approach to uncertainty modeling in 3DGS.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets (synthetic + real-world), comparison against 12+ baselines, ablations covering three downstream tasks and variance training strategies.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, motivation is well-articulated, and experiments are logically organized.
- **Value**: ⭐⭐⭐⭐ — Provides an efficient and practical uncertainty modeling paradigm for 3DGS-SLAM systems, with strong methodological contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[ICCV 2025\] 4D Gaussian Splatting SLAM](../../ICCV2025/3d_vision/4d_gaussian_splatting_slam.md)
- [\[AAAI 2026\] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting](../../AAAI2026/3d_vision/splats_in_splats_robust_and_effective_3d_steganography_towards_gaussian_splattin.md)
- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](vggt-slam.md)

</div>

<!-- RELATED:END -->
