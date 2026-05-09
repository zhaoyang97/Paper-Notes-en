---
title: >-
  [Paper Note] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] A unified framework is proposed that jointly models defocus blur and motion blur via learnable blur kernel convolution, combined with a dynamic Gaussian densification strategy and unseen-view constraints, enabling high-quality novel view synthesis of dynamic scenes from blurry monocular videos using 3DGS.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - dynamic scene reconstruction
  - defocus blur
  - motion blur
  - novel view synthesis
date: 2026-05-08
content_hash: 49c9027a1b8fb3ec
---

# Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos

**Conference**: NeurIPS 2025
**arXiv**: [2510.10691](https://arxiv.org/abs/2510.10691)
**Code**: [hhhddddddd/dydeblur](https://github.com/hhhddddddd/dydeblur)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, dynamic scene reconstruction, defocus blur, motion blur, novel view synthesis

## TL;DR

A unified framework is proposed that jointly models defocus blur and motion blur via learnable blur kernel convolution, combined with a dynamic Gaussian densification strategy and unseen-view constraints, enabling high-quality novel view synthesis of dynamic scenes from blurry monocular videos using 3DGS.

## Background & Motivation

Reconstructing dynamic scenes from monocular video and synthesizing novel views is a fundamental problem in 3D vision. Existing methods suffer from several core bottlenecks:

1. **Disjoint blur modeling**: Defocus blur and motion blur arise from entirely different physical causes — the former from depth-of-field limitations, the latter from relative motion during exposure. Existing methods address only one type.
2. **Difficulty in kernel estimation**: Although both blur types can be modeled as kernel convolution, accurately estimating per-pixel blur kernels from dynamic scenes is extremely challenging.
3. **Incomplete dynamic Gaussians**: Dynamic Gaussians initialized from tracked points are missing in occluded regions.

## Core Problem

How to design a unified framework that simultaneously handles defocus blur and motion blur in monocular videos, enabling high-fidelity sharp novel view synthesis?

## Method

### Dynamic Gaussian Representation and Densification

The Shape-of-Motion framework is adopted to model static and dynamic Gaussians separately. Dynamic Gaussian motion is represented as a linear combination of $\mathrm{SE}(3)$ motion bases:

$$\mathrm{T}_{t_0 \to t} = \sum_{b=0}^{N_b} \mathbf{w}^{(b)} \mathrm{T}_{t_0 \to t}^{(b)}$$

The transformed dynamic Gaussian positions and rotations are:

$$\mu_t = \mathrm{R}_{t_0 \to t} \mu_{t_0} + \mathrm{t}_{t_0 \to t}, \quad \mathrm{R}_t = \mathrm{R}_{t_0 \to t} \mathrm{R}_{t_0}$$

**Dynamic Gaussian Densification (DGD)**: Dynamic Gaussians are first initialized using tracked points visible in the canonical frame. Depth maps from all observed frames are then back-projected to supplement dynamic regions. Gaussians from observed frames are mapped back to the canonical frame via:

$$\mu_{t_0}^g = (\mathrm{R}_{t_0 \to t}^{G'})^{-1} (\mu_t^g - \mathrm{t}_{t_0 \to t}^{G'})$$

Densification is performed once after $N_d = 2500$ training iterations.

### Unified Blur Synthesis

Both defocus and motion blur are uniformly modeled as per-pixel kernel convolution:

$$\tilde{B}(x) = \sum_{x_i \in \mathcal{N}(x)} \tilde{I}(x_i) k_x(x_i), \quad \text{s.t.} \sum_{x_i} k_x(x_i) = 1$$

**Blur Prediction Network (BP-Net)**: A 4-layer CNN that takes as input the camera embedding $e(i)$, scene features $f_{\text{scene}}$ (encoded from the rendered image, depth, and motion mask), and pixel-coordinate positional encoding $p(x)$, and jointly predicts the blur kernel $k_x$ and blur intensity $m_x$:

$$k_x, m_x = F_\Theta(e(i), f_{\text{scene}}(x), p(x))$$

The output blurry image is obtained by blending the sharp and blurred images:

$$\hat{B}(x) = (1 - m_x) \cdot \tilde{I}(x) + m_x \cdot \tilde{B}(x)$$

**Blur-Aware Sparsity Constraint**: Prevents over-blurring in mildly blurred regions. The target center weight of the kernel is defined as:

$$c_x = \text{sigmoid}(\text{scale} \cdot (1 - \text{sg}(m_x)))$$

$$\mathcal{L}_{\text{spa}} = \mathcal{L}_1(c_x, k_x(c))$$

High blur intensity → low target center weight → dispersed kernel allowed; low blur intensity → high target center weight → concentrated kernel enforced.

### Unseen-View Constraints

Unseen views surrounding the training viewpoints are synthesized from geometric and appearance information to mitigate overfitting in monocular video:

$$p_t = K P_t^{-1} P_s D_s(p_s) K^{-1} p_s$$

Parallel unseen views (interpolated between adjacent training viewpoints) and perpendicular unseen views (offset along the normal of the camera trajectory) are generated and introduced every $N_u = 5$ iterations.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{smo}} + \mathcal{L}_{\text{spa}}$$

The reconstruction loss is $\mathcal{L}_{\text{rec}} = (1-\beta)\mathcal{L}_1(\hat{B}, B) + \beta \mathcal{L}_{\text{ssim}}(\hat{B}, B)$, with $\beta = 0.2$.

## Key Experimental Results

### D2RF (Defocus Blur) + DyBluRF (Motion Blur)

| Method | Defocus PSNR↑ | Defocus SSIM↑ | Defocus LPIPS↓ | Motion PSNR↑ | Motion SSIM↑ | Motion LPIPS↓ | Training Time |
|--------|--------------|--------------|---------------|-------------|-------------|--------------|---------------|
| D3DGS | 22.54 | 0.715 | 0.215 | 21.54 | 0.675 | 0.287 | 10 min |
| SoM | 28.32 | 0.784 | 0.164 | 26.21 | 0.823 | 0.109 | 10 min |
| D2RF | 27.04 | 0.808 | 0.128 | 23.67 | 0.745 | 0.120 | 48 hrs |
| DyBluRF | 26.24 | 0.788 | 0.159 | 24.53 | 0.864 | 0.079 | 48 hrs |
| De4DGS | 28.49 | 0.791 | 0.154 | 26.62 | 0.871 | 0.059 | 20 hrs |
| **Ours** | **29.39** | **0.859** | **0.078** | **27.01** | **0.876** | **0.056** | **1 hr** |

On the defocus dataset, PSNR exceeds De4DGS by 0.9 dB and LPIPS is reduced by 49%; training requires only 1 hour versus 20 hours.

### Ablation Study

| Configuration | Defocus PSNR | Motion PSNR |
|---------------|-------------|------------|
| w/o sparsity constraint | 29.03 | 26.63 |
| w/o Shortcut | 29.12 | 26.74 |
| w/o DGD | 29.19 | 26.53 |
| w/o unseen views | 29.09 | 26.66 |
| **Full method** | **29.39** | **27.01** |

Kernel size $K=9$ represents the optimal trade-off; gains beyond $K>9$ are marginal.

## Highlights & Insights

1. **First unified framework**: Simultaneously handles defocus and motion blur in dynamic scene reconstruction.
2. **Blur intensity–kernel sparsity coupling**: $m_x$ adaptively constrains kernel distribution in a physically grounded manner.
3. **High efficiency**: 1-hour training and 65 FPS rendering, 48× faster than NeRF-based approaches.
4. The unseen-view synthesis strategy effectively alleviates overfitting in monocular video reconstruction.

## Limitations & Future Work

- Relies on 2D priors (depth estimation, segmentation); errors in these priors propagate downstream.
- Fails on scenes with large non-rigid motion blur.
- Requires per-scene optimization and does not generalize to unseen scenes.
- Blur kernel size is fixed at $9 \times 9$, which may not cover large-scale blur.

## Related Work & Insights

- **vs. De4DGS / DyBluRF**: These methods simulate motion blur by weighting multiple frames within the exposure interval and cannot handle defocus blur; the proposed method unifies both via kernel convolution.
- **vs. D2RF**: D2RF handles defocus blur via layered depth-of-field volume rendering but cannot address motion blur and is extremely slow (48 hours).
- **vs. Shape-of-Motion**: SoM is the SOTA for sharp video; this work extends it with blur modeling to recover sharp scenes from blurry inputs.

The decoupled design of blur kernel prediction and blur intensity prediction is noteworthy — it explicitly models both "how blurry a pixel is" and "how it is blurred" as two separate dimensions. The unseen-view synthesis constraint is a general strategy for mitigating overfitting in monocular reconstruction.

## Rating

- ⭐ Novelty: 8/10 — Unifying defocus and motion blur in dynamic 3DGS is a first; the blur-aware sparsity constraint is elegantly designed.
- ⭐ Experimental Thoroughness: 8/10 — Two datasets covering both blur types with complete ablations.
- ⭐ Writing Quality: 7/10 — Structure is clear, but certain details (e.g., unseen-view generation) are described somewhat briefly.
- ⭐ Value: 8/10 — High practical value; the unified framework combined with high efficiency directly advances blurry video reconstruction.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](../../ICCV2025/3d_vision/comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[NeurIPS 2025\] DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)

<!-- RELATED:END -->
