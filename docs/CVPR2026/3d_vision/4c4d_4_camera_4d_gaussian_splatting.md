---
title: >-
  [Paper Note] 4C4D: 4 Camera 4D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][4D Gaussian Splatting] This paper proposes the 4C4D framework, which employs a Neural Decaying Function to adaptively control Gaussian opacity decay…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "Sparse-View"
  - "Dynamic Scene Reconstruction"
  - "Neural Decaying Function"
  - "Geometry-Appearance Balance"
date: 2026-05-08
content_hash: b763f704be2f644e
---

# 4C4D: 4 Camera 4D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2604.04063](https://arxiv.org/abs/2604.04063)
**Code**: [Project Page](https://junshengzhou.github.io/4C4D)
**Area**: 3D Vision
**Keywords**: 4D Gaussian Splatting, Sparse-View, Dynamic Scene Reconstruction, Neural Decaying Function, Geometry-Appearance Balance

## TL;DR
This paper proposes the 4C4D framework, which employs a Neural Decaying Function to adaptively control Gaussian opacity decay, addressing the geometry–appearance learning imbalance in sparse-view (only 4 cameras) 4D Gaussian Splatting, and achieves state-of-the-art performance across multiple benchmarks.

## Background & Motivation
**Background**: Novel view synthesis for 4D dynamic scenes typically requires dense camera arrays (tens to hundreds of cameras), severely limiting practical deployment. 3DGS/4DGS perform well under dense-view settings.

**Limitations of Prior Work**: Under extremely sparse views (e.g., 4 cameras), 4DGS fails significantly. The root cause is an **optimization bias**: fitting appearance (color) is relatively easy, whereas recovering accurate geometry (depth) under insufficient supervision is extremely difficult. Existing Gaussian formulations cannot balance the two objectives.

**Key Challenge**: Insufficient spatial supervision under sparse views → inadequate geometry learning → overfitting to training-view appearance → severe artifacts in novel views.

**Key Observation**: 4DGS can accurately reproduce appearance at training viewpoints but produces severely degraded depth geometry (see Fig. 3), indicating that the problem lies in optimization bias rather than model capacity.

**Core Idea**: Introduce a learnable opacity decaying function to redirect optimization gradients toward geometry learning.

## Method

### Overall Architecture
4D Gaussian primitives + Neural Decaying Function $f_\theta$ + visibility-detection-based decoupled decaying strategy → jointly optimized via photometric rendering loss.

### Key Designs
1. **Neural Decaying Function**:
   A lightweight neural network that takes Gaussian attributes (position, opacity, rotation) as input and predicts a decaying factor $\tau$:
   $$\tau = f_\theta(x, y, z, o, r)$$
   The final opacity is:
   $$o(\tilde{t}) = \tau \cdot \exp\left(-\frac{1}{2}\frac{(\tilde{t} - \mu_t)^2}{\Sigma_{4,4}}\right) \cdot o$$
   - **Design Motivation**: Opacity is a critical parameter for geometry learning in 4DGS. By modulating opacity via a neural network, additional learnable degrees of freedom are introduced, causing gradients to flow more toward geometric parameters (position, scale, etc.) rather than simply minimizing appearance error. This rebalances the optimization of geometry and appearance.

2. **Visibility-Detection-Based Decoupled Decaying Strategy**:
   - **Key Issue**: Gradients only exist for Gaussians visible at the current viewpoint/timestep. Applying the same decay to invisible Gaussians would distort the optimization.
   - **Visibility Detection**: $G_m = Z_V(\tilde{v}, \sigma, Z_T(\tilde{t}, s_t, G))$
     - $Z_V$: Spatial visibility (filters out Gaussians whose centers fall outside the current viewpoint)
     - $Z_T$: Temporal visibility (filters out Gaussians whose temporal span does not include the current timestep)
   - **Decoupled Strategy**:
   $$\tau(g) = \begin{cases} f_\theta(x,y,z,o,r) & \text{if } g \in G_m \\ \beta=0.999 & \text{if } g \in G_m^* \end{cases}$$
   - **Design Motivation**: Visible Gaussians require precise decay learning; invisible Gaussians are stabilized with a small constant decay (consistent with observations in AbsGS).

### Loss & Training
- Standard photometric rendering loss (L1 + SSIM)
- Neural Decaying Function and 4D Gaussians are jointly optimized
- 4D Gaussians inherit temporal attributes ($\mu_t$, $s_t$) and 4D spherical harmonic coefficients from 4DGS

## Key Experimental Results

### Main Results

| Dataset | Metric | 4C4D | 4DGS | 4DGaussians | Ex4DGS | Gain (vs. best) |
|--------|------|------|------|-------------|--------|------|
| Neural3DV | PSNR↑ | **22.29** | 20.60 | 20.82 | 19.33 | +1.47 |
| Neural3DV | LPIPS↓ | **0.146** | 0.244 | 0.190 | 0.239 | +23.2% |
| ENeRF-Outdoor | PSNR↑ | **24.32** | 23.52 | 18.21 | 21.89 | +0.80 |
| ENeRF-Outdoor | LPIPS↓ | **0.121** | 0.151 | 0.456 | 0.263 | +19.9% |
| Mobile-Stage | PSNR↑ | **22.36** | 22.15 | 20.15 | 17.85 | +0.21 |
| Mobile-Stage | LPIPS↓ | **0.121** | 0.180 | 0.226 | 0.260 | +32.8% |

### Ablation Study

| Configuration | PSNR↑ | DSSIM1↓ | LPIPS↓ | Note |
|------|-------|---------|--------|------|
| w/o Neural Decaying | 22.60 | 0.097 | 0.147 | Fixed decay insufficient |
| w/o Visibility Detection | 24.49 | 0.075 | 0.127 | No visible/invisible separation |
| Full Model | **24.68** | **0.070** | **0.115** | Both components are complementary |
| Constant Decay | 24.31 | 0.075 | 0.125 | Inferior to learnable decay |
| Exponential Decay | 24.32 | 0.077 | 0.135 | Hand-crafted function suboptimal |
| Power Decay | 24.35 | 0.074 | 0.124 | Still inferior to neural network |
| **Neural Decay (Ours)** | **24.68** | **0.070** | **0.115** | Automatically learns optimal strategy |

### Key Findings
- High-fidelity 4D reconstruction is achievable with only 4 cameras, substantially lowering the acquisition barrier compared to dense-array methods
- Neural decay vs. constant decay: PSNR +0.37, LPIPS −0.010, demonstrating the advantage of adaptive over fixed decay
- Contribution of visibility decoupling: PSNR +0.19, LPIPS −0.012
- The most significant improvements are observed on the outdoor complex scene benchmark (ENeRF-Outdoor)
- The self-collected dataset Dyn4Cam (4 GoPro cameras, <$1,500) enables temporally consistent 4D dynamic capture

## Highlights & Insights
- **Rigorous problem analysis**: The paper clearly identifies the root cause of 4DGS failure under sparse views as a geometry–appearance optimization imbalance, rather than insufficient model capacity.
- **Elegant solution**: Without modifying the 3D representation, gradient flow is redirected solely through opacity modulation, yielding significant gains.
- **High practical value**: 4D capture with 4 GoPro cameras (<$1,500) is genuinely accessible to consumer-level users.
- **Decoupled treatment of visible and invisible Gaussians** is a subtle yet critically important design detail.

## Limitations & Future Work
- Four viewpoints still leave occlusion blind spots, making scenes with severe self-occlusion difficult to handle.
- The Neural Decaying Function introduces additional computational overhead.
- Integration with monocular depth priors (e.g., MiDaS) remains unexplored.
- Dyn4Cam does not support quantitative evaluation due to the absence of ground-truth test viewpoints.

## Related Work & Insights
- The constant decay for invisible Gaussians is consistent with observations reported in AbsGS.
- The view-inflated attention mechanism of MVDream may also be applicable to this setting.
- The "gradient redirection" paradigm could be extended to other imbalanced optimization problems (e.g., sparse-view NeRF).

## Rating
- Novelty: ⭐⭐⭐⭐ Core idea is clear but not fundamentally disruptive
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, detailed ablations, and a self-collected dataset — very solid
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is well-articulated and the method is presented concisely
- Value: ⭐⭐⭐⭐ Consumer-level 4D capture carries significant practical significance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] r4det 4d radar camera fusion 3d detection](r4det_4d_radar_camera_fusion_3d_detection.md)
- [\[NeurIPS 2025\] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation](../../NeurIPS2025/3d_vision/every_camera_effect_every_time_all_at_once_4d_gaussian_ray_tracing_for_physics-b.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
