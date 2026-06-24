---
title: >-
  [Paper Note] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This work introduces a physical motion blur imaging model into the 3D Gaussian Splatting framework, jointly optimizing scene Gaussian parameters and camera trajectories during exposure to restore sharp 3D scenes from blurry images and achieve real-time rendering.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Motion Deblurring"
  - "Bundle Adjustment"
  - "novel view synthesis"
  - "Camera Pose Optimization"
date: 2026-05-08
content_hash: 3318a386b0894e59
---

# BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2403.11831](https://arxiv.org/abs/2403.11831)  
**Code**: [lingzhezhao/BAD-Gaussians](https://github.com/WU-CVGL/BAD-Gaussians)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Motion Deblurring, Bundle Adjustment, novel view synthesis, Camera Pose Optimization

## TL;DR

This work introduces a physical motion blur imaging model into the 3D Gaussian Splatting framework, jointly optimizing scene Gaussian parameters and camera trajectories during exposure to restore sharp 3D scenes from blurry images and achieve real-time rendering.

## Background & Motivation

**NeRF and 3D-GS rely on sharp images**: Existing neural rendering methods (NeRF, 3D-GS) assume high-quality sharp images as input. However, motion-blurred images are highly common under low-light or long-exposure conditions in the real world. Directly training on blurry images severely degrades reconstruction quality.

**Blurry images lead to inaccurate pose estimation**: Camera poses recovered by COLMAP from blurry images exhibit poor accuracy because feature matching across multi-view images is challenging, which further exacerbates the initialization and optimization issues of 3D-GS.

**3D-GS initialization relies on sparse point clouds**: Blurry images result in fewer matched points from COLMAP, leading to degraded quality in the initialization of 3D Gaussians.

**Prior deblurring NeRF methods have limitations**: Deblur-NeRF and DP-NeRF train with fixed inaccurate poses and are based on implicit MLP representations, making it difficult to recover fine details and impossible to achieve real-time rendering. Although BAD-NeRF models the physical blurring process, it is constrained by NeRF's implicit representation, with rendering speeds below 1 FPS.

**Advantages of explicit representation remain unexploited**: The explicit point cloud representation of 3D-GS is naturally beneficial for differentiable rasterization and efficient rendering, but no prior work has combined it with motion deblurring.

**Need for joint optimization of poses and scenes**: An end-to-end framework is required to simultaneously optimize camera exposure trajectories and scene representation, enabling high-quality reconstruction without relying on accurate initial poses.

## Method

### Overall Architecture

BAD-Gaussians takes a set of motion-blurred images, along with their inaccurate poses and sparse point clouds estimated by COLMAP, as input. For each blurry image, the camera motion trajectory is parameterized by two poses at the start and end of the exposure. Through interpolation on $SE(3)$, $n$ virtual sharp views are generated. Each virtual sharp image is rendered using the differentiable rasterization of 3D-GS, and their average is taken to simulate the physical blurring process. By minimizing the photometric error between the synthesized blurry images and the real blurry images, the Gaussian parameters and camera trajectories are jointly optimized via backpropagation.

### Key Designs

#### 1. Physical Motion Blur Imaging Model

- **Function**: Models a blurry image as the discrete average of continuous virtual sharp images during exposure time: $\mathbf{B}(\mathbf{u}) \approx \frac{1}{n}\sum_{i=0}^{n-1}\mathbf{C}_i(\mathbf{u})$.
- **Mechanism**: Renders sharp images from $n$ uniformly sampled virtual camera poses and averages them to synthesize the blurry image, faithfully simulating the photon integration process of real camera sensors.
- **Design Motivation**: Unlike the deformable convolution kernels in Deblur-NeRF, directly modeling the physical process handles large-scale motion blur better and naturally integrates with the differentiable rasterization pipeline of 3D-GS.

#### 2. Camera Motion Trajectory Modeling on $SE(3)$

- **Function**: Learns the exposure start pose $\mathbf{T}_{\text{start}}$ and end pose $\mathbf{T}_{\text{end}}$ for each blurry image, and obtains intermediate virtual poses $\mathbf{T}_t$ through linear or cubic B-spline interpolation on the Lie group.
- **Mechanism**: $\mathbf{T}_t = \mathbf{T}_{\text{start}} \cdot \exp(\frac{t}{\tau} \cdot \log(\mathbf{T}_{\text{start}}^{-1} \cdot \mathbf{T}_{\text{end}}))$, performing smooth interpolation on the $SE(3)$ manifold to ensure geometric consistency of rotation and translation.
- **Design Motivation**: Exposure time is typically short, making linear interpolation sufficient to represent constant-velocity motion. For accelerated motion in real-world scenes, a cubic B-spline (with 4 control points) can be used to capture more complex trajectories. It has a small parameter size (adding only 12 or 24 learnable parameters per frame), and is efficient and differentiable.

#### 3. Analytical Gradient Propagation from Gaussians to Camera Poses

- **Function**: Derives the analytical Jacobian of the Gaussian parameters (mainly the mean position $\boldsymbol{\mu}'$) with respect to the camera poses $\mathbf{T}_i$, allowing photometric loss gradients to flow to pose parameters.
- **Mechanism**: Decomposes the gradient chain into $\frac{\partial \mathbf{C}_i}{\partial \boldsymbol{\mu}'} \cdot \frac{\partial \boldsymbol{\mu}'}{\partial \boldsymbol{\mu}} \cdot \frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{\mu}_c} \cdot \frac{\partial \boldsymbol{\mu}_c}{\partial \mathbf{T}_i}$, where the first term is computed by the 3D-GS CUDA backend, and the subsequent terms are analytically derived. $\frac{\partial \Sigma'}{\partial \mathbf{T}_i}$ is ignored for efficiency.
- **Design Motivation**: Compared to NeRF's implicit representation, the explicit Gaussian projection of 3D-GS natively supports analytical Jacobian computation, making joint pose optimization more stable and efficient.

#### 4. Adaptive Selection of the Number of Virtual Poses $n$ and Trajectory Representation

- **Function**: Determines $n=10$ as the optimal trade-off between performance and efficiency via ablation study; uses linear interpolation for synthetic data and cubic B-splines for real data.
- **Mechanism**: Larger $n$ improves recovery of severe blur but yields diminishing returns. Linear interpolation is sufficient for short exposure times, while real-world scenes with long exposure times require higher-order splines.
- **Design Motivation**: Avoids over-parameterization while ensuring effective modeling under various blur levels.

## Loss & Training

The loss function follows the combination used in 3D-GS: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$, where $\mathcal{L}_1$ is the L1 loss between synthesized and real blurry images, and $\mathcal{L}_{\text{D-SSIM}}$ is the structural similarity loss. Gaussian parameters are optimized using the Adam optimizer (learning rate consistent with original 3D-GS), and the camera pose learning rate decays exponentially from $1\times10^{-3}$ to $1\times10^{-5}$. Training takes approximately 30 minutes (RTX 4090), whereas baseline methods require over 10 hours.

## Experiments

### Synthetic Data Deblurring (Deblur-NeRF Dataset, Table 3)

| Method | Cozyroom PSNR | Tanabata PSNR | Trolley PSNR | Average PSNR Gain |
|------|:---:|:---:|:---:|:---:|
| NeRF | 26.13 | 20.57 | 21.71 | — |
| 3D-GS | 25.86 | 20.51 | 21.65 | — |
| Deblur-NeRF | 29.53 | 23.20 | 25.68 | — |
| DP-NeRF* (GT pose) | 30.77 | 25.27 | 26.99 | — |
| BAD-NeRF | 32.11 | 25.80 | 29.68 | — |
| **BAD-Gaussians** | **34.68** | **32.12** | **33.97** | **+3.6 dB vs 2nd place** |

BAD-Gaussians outperforms the second-best method, BAD-NeRF, by an average of 3.6 dB PSNR across 5 synthetic scenes, while achieving a rendering speed of >200 FPS (compared to <1 FPS for BAD-NeRF).

### Real-World New View Synthesis (Deblur-NeRF Real Dataset, Table 6)

| Method | Coffee PSNR | Heron PSNR | Stair PSNR | Average LPIPS |
|------|:---:|:---:|:---:|:---:|
| 3D-GS | 27.44 | 20.28 | 22.68 | Poor |
| Deblur-NeRF | 30.72 | 22.63 | 25.39 | ~0.19 |
| DP-NeRF | 31.35 | 22.79 | 25.53 | ~0.17 |
| BAD-NeRF | 29.08 | 21.81 | 25.64 | ~0.22 |
| **BAD-Gaussians** | **32.17** | **24.52** | **26.63** | **~0.10** |

Across 10 real-world scenes, BAD-Gaussians outperforms all baseline methods in PSNR, SSIM, and LPIPS, whereas BAD-NeRF degrades significantly on real-world data.

### Pose Estimation Accuracy (Table 7)

The Absolute Trajectory Error (ATE) of BAD-Gaussians is superior to COLMAP-blur and BAD-NeRF in most scenes, validating the effectiveness of the joint pose optimization.

## Highlights & Insights

1. **First 3D-GS-based motion deblurring framework**, introducing a physical blur imaging model into explicit Gaussian representations, opening up a new direction.
2. **Real-time rendering**: >200 FPS, whereas all prior NeRF deblurring methods run at <1 FPS, significantly enhancing practical applicability.
3. **Highly efficient training**: ~30 minutes vs >10 hours for other methods.
4. **Joint camera trajectory and scene optimization**: Does not rely on accurate pose priors, demonstrating robustness in scenes where COLMAP fails.
5. **Substantial performance gains**: An average boost of +3.6 dB PSNR on synthetic data, and approximately halving the LPIPS on real-world data.

## Limitations & Future Work

1. **Inferior performance in the Factory scene compared to BAD-NeRF**: 3D-GS has weaker capability in representing textureless regions like sky compared to NeRF's implicit continuous representation.
2. **Limitations of the linear trajectory assumption for complex motion**: Although a cubic B-spline option is provided, sharper non-uniform motions (e.g., sudden braking, sharp turns) may require higher-order modeling.
3. **Reliance on COLMAP initialization**: Still requires COLMAP to provide initial poses and sparse point clouds, which may completely fail under extreme blur conditions.
4. **Rolling shutter blur is not addressed**: Only models motion blur under global shutter, leaving rolling shutter effects (common in mobile phones) unaddressed.

## Related Work & Insights

- **Deblur-NeRF** (Ma et al., CVPR 2022): Models blur using a deformable sparse kernel, training with fixed COLMAP poses.
- **DP-NeRF** (Lee et al., CVPR 2023): Introduces physical priors based on Deblur-NeRF, but still keeps poses fixed.
- **BAD-NeRF** (Wang et al., CVPR 2023): The most direct predecessor of this work, which models the physical blur process during exposure time and jointly optimizes NeRF and camera poses. However, it is limited by implicit representations and cannot support real-time rendering. BAD-Gaussians transfers the same physical modeling concept to the explicit 3D-GS framework, achieving dual improvements in both quality and speed.
- **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023): The foundation of the scene representation used in this work, but the original 3D-GS cannot handle blurry inputs.
- **BARF / CamP**: Jointly optimizes NeRF and camera poses, but only processes sharp images.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce physical blur model + joint pose optimization into 3D-GS. The concept is straightforward but executed solidly.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic, real-world, and MBA-VO datasets, with complete ablation studies and comprehensive pose accuracy evaluations.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations, intuitive pipeline diagram, and detailed experimental tables.
- Value: ⭐⭐⭐⭐ — Solves the key challenge of 3D-GS handling motion blur. High-speed rendering makes it highly valuable for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Event-based Mosaicing Bundle Adjustment](event-based_mosaicing_bundle_adjustment.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)

</div>

<!-- RELATED:END -->
