---
title: >-
  [Paper Note] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes BezierGS, a 3D Gaussian Splatting method that models dynamic object motion trajectories using learnable Bézier curves, eliminating reliance on precise bounding box annotations. The method achieves state-of-the-art performance on both dynamic and static scene reconstruction on the Waymo and nuPlan datasets.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - dynamic scene reconstruction
  - Bézier curve
  - novel view synthesis
  - autonomous driving
date: 2026-05-08
content_hash: 98b14a8b411669d7
---

# BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2506.22099](https://arxiv.org/abs/2506.22099)
**Code**: [github.com/fudan-zvg/BezierGS](https://github.com/fudan-zvg/BezierGS)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, dynamic scene reconstruction, Bézier curve, novel view synthesis, autonomous driving

## TL;DR

This paper proposes BezierGS, a 3D Gaussian Splatting method that models dynamic object motion trajectories using learnable Bézier curves, eliminating reliance on precise bounding box annotations. The method achieves state-of-the-art performance on both dynamic and static scene reconstruction on the Waymo and nuPlan datasets.

## Background & Motivation

Dynamic 3D street scene modeling is a fundamental requirement for autonomous driving, where high-quality scene reconstruction can provide simulation environments for closed-loop evaluation. Existing methods face the following challenges:

- **Annotation-based methods** (Street Gaussians, OmniRe, etc.) rely heavily on manually annotated object poses (including orientation and position), and annotation errors directly degrade reconstruction quality. When annotation precision is poor—as in the nuPlan dataset—the performance of such methods drops significantly.
- **Self-supervised methods** (S³Gaussian employs spatiotemporal decomposition networks for implicit motion modeling; PVG stitches trajectory segments via periodic vibration) avoid annotation dependency, but S³Gaussian's implicit modeling is difficult to optimize, PVG's periodic vibration assumption does not reflect realistic motion, and its segmented trajectories cannot exploit temporal consistency across the same object.

BezierGS addresses these issues by explicitly and learnably modeling motion trajectories with Bézier curves, combining both flexibility and accuracy.

## Method

### Overall Architecture

The scene consists of two components: static background Gaussians and dynamic foreground Gaussians. The static component is optimized globally using standard 3DGS; each Gaussian primitive in the dynamic component models its motion trajectory via a learnable Bézier curve. Given a timestamp $\tau$, the dynamic Gaussian positions are computed, merged with static Gaussians for rendering, and then composited with a sky cubemap to produce the final image.

### Key Designs

1. **Bézier Curve Trajectory Modeling**:

    - Each dynamic object has a center trajectory Bézier curve $\boldsymbol{\gamma}(t,g) = \sum_{i=0}^{n} b_{i,n}(t) \boldsymbol{p}_i^g$, defined by $n+1$ learnable control points.
    - Each Gaussian primitive additionally has an offset trajectory curve $\boldsymbol{\delta}(t)$, representing the displacement relative to the object center.
    - The final position is $\boldsymbol{\mu}(\tau,g) = \boldsymbol{\delta}(t) + \boldsymbol{\gamma}(t,g)$, where $t = f(\tau,g)$ is the mapping from timestamp to Bézier parameter.
    - Standard cubic Bézier curves ($n=3$) are used, whose effectiveness in trajectory modeling has been widely validated.
    - Core advantage: learnable control points can automatically correct annotation pose errors, and the explicit curve formulation facilitates temporal consistency.

2. **Time-to-Bézier Mapping**:

    - Object motion along a Bézier curve is non-uniform in speed, requiring a mapping from timestamp $\tau$ to Bézier parameter $t$.
    - Since the mapping differs across objects (due to varying speeds), it is also modeled as a Bézier curve $t = f(\tau,g)$.
    - This implicitly encodes per-object velocity information.

3. **Inter-Curve Consistency Loss**:

    - Dynamic Gaussian primitives with excessive degrees of freedom may drift away from their associated dynamic object, causing floaters and artifacts in novel views.
    - For rigid bodies (e.g., vehicles), the magnitude of motion deviation across different parts should remain constant.
    - This is enforced by constraining the norm of the offset curve $\boldsymbol{\delta}(t)$ to be consistent with the mean norm of its endpoint control points:
    $$\mathcal{L}_{icc} = \left\| \|\boldsymbol{\delta}(t)\| - \frac{\|\boldsymbol{p}_0\| + \|\boldsymbol{p}_n\|}{2} \right\|_1$$
    - This effectively suppresses excessive local geometric variation.

### Loss & Training

The total loss comprises multiple terms:

$$\mathcal{L} = (1-\lambda_r)\mathcal{L}_1 + \lambda_r\mathcal{L}_{ssim} + \lambda_o^{sky}\mathcal{L}_o^{sky} + \lambda_{icc}\mathcal{L}_{icc} + \lambda_{dr}\mathcal{L}_{dr} + \lambda_v\mathcal{L}_v + \lambda_d\mathcal{L}_d$$

- **Dynamic rendering loss $\mathcal{L}_{dr}$**: Dynamic object masks are obtained via Grounded-SAM; RGB and opacity supervision is applied to the separate rendering of dynamic Gaussians to enforce dynamic-static separation.
- **Velocity loss $\mathcal{L}_v$**: Exploiting the analytic differentiability of Bézier curves, Gaussian velocities are computed directly and rendered as velocity maps, ensuring that dynamic motion is confined to dynamic regions.
- **Depth loss $\mathcal{L}_d$**: Sparse inverse depth maps projected from LiDAR are used to enhance geometric awareness.
- **Sky opacity loss $\mathcal{L}_o^{sky}$**: Gaussian opacity is minimized in sky-masked regions.

Hyperparameter settings: $\lambda_r=0.2$, $\lambda_{icc}=0.01$, $\lambda_{dr}=0.1$, $\lambda_v=1.0$, $\lambda_d=1.0$. Training is conducted for 30k iterations on a single A6000 GPU.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Dyn-PSNR↑ |
|--------|------|-------|-------|--------|-----------|
| Waymo (NVS) | DeformableGS | 29.52 | 0.889 | 0.100 | 24.66 |
| Waymo (NVS) | Street Gaussians | 28.92 | 0.877 | 0.110 | 25.54 |
| Waymo (NVS) | OmniRe | 29.41 | 0.884 | 0.101 | 25.85 |
| Waymo (NVS) | PVG | 29.64 | 0.864 | 0.179 | 24.46 |
| Waymo (NVS) | **BezierGS** | **31.51** | **0.903** | **0.092** | **28.51** |
| nuPlan (NVS) | OmniRe | 26.01 | 0.819 | 0.173 | 23.90 |
| nuPlan (NVS) | PVG | 26.38 | 0.772 | 0.222 | 19.69 |
| nuPlan (NVS) | **BezierGS** | **29.42** | **0.860** | **0.133** | **25.12** |

On Waymo, PSNR improves by 1.87 dB and Dyn-PSNR by 2.66 dB; on nuPlan, PSNR improves by 3.04 dB and LPIPS decreases by 16.35%.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Dyn-PSNR↑ |
|------|-------|-------|--------|-----------|
| w/o $\mathcal{L}_{icc}$ | 30.83 | 0.900 | 0.096 | 26.15 |
| w/o $\mathcal{L}_{dr}$ | 30.99 | 0.891 | 0.099 | 28.07 |
| w/o $\mathcal{L}_v$ | 31.40 | 0.901 | 0.094 | 28.29 |
| w/o time-to-Bézier | 31.36 | 0.899 | 0.094 | 27.97 |
| w/ MLP trajectory (DeformableGS) | 29.58 | 0.898 | 0.087 | 24.78 |
| w/ sinusoidal trajectory (PVG) | 29.65 | 0.877 | 0.099 | 26.27 |
| **BezierGS (full)** | **31.51** | **0.903** | **0.092** | **28.51** |

### Key Findings

- The ICC loss contributes the most; removing it causes a 2.36 dB drop in Dyn-PSNR, indicating that inter-curve consistency is critical for dynamic object modeling.
- Bézier curves outperform MLP-based and sinusoidal trajectories by approximately 2 dB and 1.9 dB in PSNR, respectively.
- BezierGS demonstrates a more pronounced advantage on nuPlan, where annotation quality is poor, as it can automatically correct pose errors.

## Highlights & Insights

- As a parametric curve, Bézier curves are easier to optimize than MLPs and more physically plausible than periodic vibrations.
- Annotation-based bounding box methods can be viewed as a special case of BezierGS (with constant offsets in the object coordinate frame and fixed box poses), demonstrating the generality of the proposed framework.
- Velocity is analytically derivable via Bernstein basis functions, allowing velocity constraints to be naturally integrated into the framework.

## Limitations & Future Work

- Object-aware initialization (grouping relies on 3D detection or annotations) is still required.
- Cubic Bézier curves may require piecewise representations for highly complex trajectories such as U-turns.
- The sky model uses a simple cubemap, which may be insufficient for complex sky scenarios.

## Related Work & Insights

- The Bézier trajectory modeling concept can be extended to trajectory prediction tasks for robots or moving objects.
- The inter-curve consistency constraint can be applied to other tasks requiring geometric consistency of objects.

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling motion trajectories with Bézier curves is a natural and elegant choice that unifies annotation-based and self-supervised approaches.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two large-scale benchmarks, comprehensive ablations, and quantitative/qualitative comparisons against multiple state-of-the-art methods.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete mathematical derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Addresses the practical bottleneck of annotation quality limiting reconstruction performance, with direct applicability to autonomous driving simulation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)

<!-- RELATED:END -->
