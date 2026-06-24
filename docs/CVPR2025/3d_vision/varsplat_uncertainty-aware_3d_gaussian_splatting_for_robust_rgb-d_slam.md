---
title: >-
  [Paper Note] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] VarSplat learns the appearance variance $\sigma^2$ for each Gaussian splat within the 3DGS-SLAM framework. It derives a differentiable per-pixel uncertainty map $V$ via the law of total variance and applies it to tracking, loop detection, and registration, achieving more robust pose estimation and competitive reconstruction quality on Replica, TUM, ScanNet, and ScanNet++ datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "SLAM"
  - "uncertainty modeling"
  - "RGB-D"
  - "pose estimation"
date: 2026-05-08
content_hash: 60c49ea44e33973d
---

# VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

**Conference**: CVPR 2025  
**arXiv**: [2603.09673](https://arxiv.org/abs/2603.09673)  
**Code**: [https://anhthuan1999.github.io/varsplat/](https://anhthuan1999.github.io/varsplat/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, SLAM, uncertainty modeling, RGB-D, pose estimation

## TL;DR
VarSplat learns the appearance variance $\sigma^2$ for each Gaussian splat within the 3DGS-SLAM framework. It derives a differentiable per-pixel uncertainty map $V$ via the law of total variance and applies it to tracking, loop detection, and registration, achieving more robust pose estimation and competitive reconstruction quality on Replica, TUM, ScanNet, and ScanNet++ datasets.

## Background & Motivation
**Background**: 3DGS-SLAM systems (SplaTAM, Gaussian-SLAM, LoopSplat) have achieved promising results in dense RGB-D SLAM by rendering anisotropic Gaussians through fast differentiable rasterization.

**Limitations of Prior Work**: Existing methods implicitly model measurement reliability by using uniform photometric loss weights for all pixels. This causes pose estimation to easily drift in scenarios such as low-texture areas, reflective surfaces, and depth discontinuities.

**Key Challenge**: While uncertainty on the geometric side (depth variance, probabilistic filters) has been studied, appearance uncertainty—which directly reflects the rendering instability of 3DGS—has never been treated as a first-class citizen in online dense SLAM.

**Goal**: How to explicitly quantify appearance uncertainty in 3DGS-SLAM and utilize it to improve the robustness of pose estimation (tracking, loop detection, and registration)?

**Key Insight**: Add a learnable appearance variance parameter $\sigma_i^2$ for each splat, and propagate it through alpha compositing using the law of total variance to obtain a per-pixel uncertainty map.

**Core Idea**: Learn per-splat appearance variance and propagate it via alpha compositing into per-pixel uncertainty, serving as confidence weights for tracking, loop detection, and registration.

## Method

### Overall Architecture
Input RGB-D stream → Incremental submap construction (each submap contains 3D Gaussians with $\sigma^2$) → Differentiable rendering to obtain color $\hat{I}$, depth $\hat{D}$, and uncertainty map $V$ → Three downstream modules weighted using $V$ or $\sigma^2$: tracking (inter-frame pose), registration (submap alignment), and loop detection (long-range loop closure).

### Key Designs

1. **Per-splat Appearance Variance $\sigma_i^2$**:

    - **Function**: New 3D parameters $\sigma_i^2 \in \mathbb{R}^3$ are introduced for each Gaussian splat, representing the variance around its mean color.
    - **Mechanism**: They are optimized end-to-end along with positions $\mu_i$, covariances $\Sigma_i$, and SH colors $c_i$. The parameter $\sigma_i^2$ differs from the geometric covariance $\Sigma_i$ (which defines spatial extent) and SH coefficients (which define mean color), modeling the "fluctuation amplitude of the splat color under different viewpoints."
    - **Design Motivation**: At depth discontinuities, occlusion boundaries, and reflective/transparent surfaces, the splat's contribution becomes unstable due to severe fluctuations of alpha weights across different viewpoints; $\sigma^2$ naturally learns larger values in these areas.

2. **Per-pixel Uncertainty Rendering (Law of Total Variance)**:

    - **Function**: Propagate per-splat variance to per-pixel variance $V$.
    - **Mechanism**: Utilizing the law of total variance $\text{Var}[X] = \mathbb{E}[\text{Var}[X|Z]] + \text{Var}(\mathbb{E}[X|Z])$, letting $X$ be the pixel color and $Z$ be the Gaussian index. Derived via alpha compositing, it yields: $V = \sum_i w_i(\sigma_i^2 + c_i^2) - (\sum_i w_i c_i)^2$
    - **Design Motivation**: Variance rendering shares the same rasterization pass with color/depth, requiring no extra decoders or pre-trained models, thereby maintaining real-time performance.

3. **End-to-End Variance Learning (Gaussian NLL)**:

    - **Function**: Train the variance via Gaussian Negative Log-Likelihood loss $\mathcal{L}_{\text{var}}$.
    - **Mechanism**: $\mathcal{L}_{\text{var}} = \frac{1}{2V}(\|\hat{I}-I\|_2^2 + \|\hat{D}-D\|_2^2) + \log(V)$. Choosing MSE instead of L1 is mathematically consistent with the Gaussian assumption. The variance gradient is $\frac{\partial \mathcal{L}}{\partial \sigma_i^2} = \frac{\partial \mathcal{L}}{\partial V} \cdot w_i$, which naturally propagates via alpha weights.
    - **Design Motivation**: Avoid depending on pre-trained models (e.g., DINOv2), learning variance from scratch to match the current scene.

### Downstream Usage

- **Tracking**: Per-pixel weights $\tilde{w}_p = \exp[-(\log V - \tilde{V})/\tau]$ (median-centered log-scaling) weight the photometric loss. During tracking, the variance parameters are frozen.
- **Loop Detection**: Compute the submap opacity ratio $r = \frac{\sum_j \tilde{w}_s \alpha_j}{\sum_j \alpha_j}$ to modulate cross similarity, preventing false loop detection triggers in unreliable regions.
- **Registration**: Refine pose after submap matching by weighting the photometric loss with per-pixel weights.

### Loss & Training
$$\mathcal{L}_{\text{map}} = \lambda_{\text{color}} \cdot \mathcal{L}_{\text{color}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{depth}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}}$$

Where $\mathcal{L}_{\text{color}}$ dynamically combines L1 and SSIM, $\mathcal{L}_{\text{depth}}$ is the depth L1, and $\mathcal{L}_{\text{reg}}$ controls the Gaussian scale.

## Key Experimental Results

### Main Results

| Dataset | Metric (ATE RMSE ↓ cm) | VarSplat | LoopSplat | CG-SLAM | Gain |
|--------|----------------------|----------|-----------|---------|------|
| Replica (8 scenes) | ATE RMSE | **0.23** | 0.26 | 0.27 | -11.5% |
| ScanNet++ (5 scenes) | ATE RMSE | **1.69** | 2.05 | — | -17.6% |
| TUM-RGBD (5 scenes) | ATE RMSE | **3.20** | 3.33 | 4.0 | -4.0% |
| ScanNet (6 scenes) | ATE RMSE | **6.5** | 7.7 | 8.1 | -15.6% |

Rendering quality: PSNR is 37.15 on Replica (LoopSplat: 36.63), ScanNet++ NVS PSNR is 21.33 (LoopSplat: 21.30), and the reconstruction F1 score is 90.2%, which is on par with LoopSplat (90.4%).

### Ablation Study

| Configuration | ATE RMSE ↓ | Description |
|------|-----------|------|
| No uncertainty | 8.20 | Baseline, without uncertainty |
| +Tracking only | 7.63 | Adding tracking weight reduces ATE by 7% |
| +Tracking+Loop | 7.49 | Loop detection further reduces error |
| +Loop+Registration | 7.51 | Effective even without tracking |
| **Full (T+L+R)** | **6.53** | Best combination of all three, 20.4% reduction |

Ablation of variance training (ScanNet): freezing variance during tracking (7.55 → 6.53, 13.5% reduction), adding depth residual into variance loss (7.17 → 6.53, 8.9% reduction), using MSE instead of L1 (7.38 → 6.53, 11.5% reduction).

### Key Findings
- The effects of uncertainty across the three stages (tracking/loop/registration) are complementary, achieving the best performance when all are enabled.
- Freezing variance during tracking to avoid conflicts with pose optimization is a critical design choice.
- Runtime is comparable to LoopSplat (Mapping: 1.9s/fr vs 1.2s/fr, Tracking: 2.0s/fr vs 1.8s/fr).
- The improvement is most significant in scenarios with low texture and reflective surfaces.

## Highlights & Insights
- **Elegant combination of the law of total variance and alpha compositing**: Implements statistical derivation directly into the existing rendering pipeline without auxiliary networks or sampling, rendering the uncertainty map in a single rasterization pass, which is both theoretically elegant and practically efficient.
- **Decoupled strategy of freezing during tracking and training during mapping**: The variance is learned alongside other parameters during the mapping phase but is frozen during tracking and registration, preventing interference between the two objectives.
- **Intuitively correct uncertainty visualization**: High uncertainty naturally concentrates in intuitively unreliable areas such as depth discontinuities, occlusion boundaries, and reflective/transparent surfaces, demonstrating that the learned variance effectively reflects measurement uncertainty.

## Limitations & Future Work
- Variance rendering increases the Mapping time by approximately 60% (1.9s vs 1.2s per frame), which may pose a bottleneck in resource-constrained environments.
- It only models appearance variance without considering geometric (position/scale) uncertainty; jointly modeling both could yield a more complete confidence measure.
- The uncertainty weight is removed during the global refinement stage, which leaves room for improvement.
- Handling of dynamic scenes is not addressed—variance might be perturbed by moving objects.

## Related Work & Insights
- **vs LoopSplat [ECCV'24]**: Both are submap-based 3DGS-SLAM, but LoopSplat lacks uncertainty modeling. VarSplat incorporates variance learning into this framework, reducing ATE on ScanNet++ from 2.05 to 1.69.
- **vs CG-SLAM**: CG-SLAM models depth-driven geometric uncertainty, while VarSplat models appearance uncertainty, making the two approaches complementary.
- **vs Uni-SLAM**: Uni-SLAM utilizes the variance of ray-termination probability, whereas VarSplat's variance is learned end-to-end within the rasterizer and updated frame-by-frame.
- **Idea Transfer**: The per-splat variance approach can be extended to other 3D Gaussian applications (such as NeRF synthesis, dynamic scenes, and semantic segmentation) as a general form of uncertainty quantification.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of the law of total variance in 3DGS is novel, although the overall framework is built upon LoopSplat.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across four datasets with multiple baselines, detailed ablations, and runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, providing sound mathematical and intuitive explanations.
- Value: ⭐⭐⭐⭐ Presents a clean and effective uncertainty modeling solution for 3DGS-SLAM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[CVPR 2025\] MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[ECCV 2024\] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](../../ECCV2024/3d_vision/cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
