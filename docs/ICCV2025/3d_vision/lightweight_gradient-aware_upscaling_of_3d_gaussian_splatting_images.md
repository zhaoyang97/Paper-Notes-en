---
title: >-
  [Paper Note] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] A lightweight image upscaling technique tailored for 3DGS is proposed, leveraging analytic image gradients from Gaussian primitives for gradient-aware bicubic spline interpolation. Without any deep learning inference, the method achieves 3–4× rendering acceleration while surpassing standard bicubic interpolation and DL-based upscaling in reconstruction quality.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - image upscaling
  - analytic gradients
  - spline interpolation
  - lightweight rendering
date: 2026-05-08
content_hash: d3e31d55c3286733
---

# Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images

**Conference**: ICCV 2025
**arXiv**: [2503.14171](https://arxiv.org/abs/2503.14171)
**Code**: Not released
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, image upscaling, analytic gradients, spline interpolation, lightweight rendering

## TL;DR

A lightweight image upscaling technique tailored for 3DGS is proposed, leveraging analytic image gradients from Gaussian primitives for gradient-aware bicubic spline interpolation. Without any deep learning inference, the method achieves 3–4× rendering acceleration while surpassing standard bicubic interpolation and DL-based upscaling in reconstruction quality.

## Background & Motivation

Despite continuous improvements in 3DGS for novel view synthesis, **high-end GPUs still struggle to maintain stable 30 FPS at high display resolutions**. The challenge is even more severe for VR applications, which require rendering two high-resolution images simultaneously. On mobile devices, performance drops to unacceptable levels.

**Limitations of existing upscaling approaches**:

**Traditional interpolation** (bicubic, Lanczos): introduces visible artifacts (ringing, staircase effects) at large upscaling factors, with notable quality degradation.

**Deep learning-based upscaling** (NinaSR, DLSS, SwinIR):
- High-quality methods are too slow for real-time use.
- DLSS requires dedicated hardware and additional auxiliary inputs (depth buffers, motion vectors) that 3DGS cannot provide.
- DL upscaling **introduces hallucinated artifacts** and **temporal inconsistency** (particularly problematic for VR users).
- PSNR may actually decrease (images appear sharper but are less faithful to the ground truth).

**Core Insight**: Gaussian primitives in 3DGS are **analytically differentiable** — the gradient of image color with respect to pixel coordinates can be computed exactly, without finite-difference approximations. Injecting these precise gradients into bicubic spline interpolation yields results substantially superior to traditional interpolation at negligible computational cost.

## Method

### Core Idea

Traditional bicubic interpolation estimates slopes at sample points via finite differences and then fits smooth curves. However, finite differences are merely approximations of the true gradient and can cause reconstruction errors in regions of rapid signal variation.

This paper directly replaces finite differences with **analytic gradients obtained at no extra cost** during the 3DGS rendering pass, enabling more accurate spline fitting.

### Analytic Image Gradient Computation

The pixel color rendered by 3DGS is $I(x,y) = \sum_{i=1}^N T_i(x,y) \alpha_i(x,y) c_i$.

Differentiating with respect to spatial coordinates:

$$\frac{\partial I(x,y)}{\partial x} = \sum_{i=1}^N c_i \left(\frac{\partial T_i}{\partial x} \alpha_i + T_i \frac{\partial \alpha_i}{\partial x}\right)$$

$$\frac{\partial I(x,y)}{\partial y} = \sum_{i=1}^N c_i \left(\frac{\partial T_i}{\partial y} \alpha_i + T_i \frac{\partial \alpha_i}{\partial y}\right)$$

The cross-derivative $\frac{\partial^2 I}{\partial x \partial y}$ is derived analogously. Derivatives of the transmittance $T_i$ are computed via a recursive iteration through $\alpha$-blending:

$$\frac{\partial A_i}{\partial x} = \frac{\partial A_{i-1}}{\partial x}(1 - \alpha_i) + (1 - A_{i-1})\frac{\partial \alpha_i}{\partial x}$$

Crucially, these computations can be **embedded directly into the 3DGS forward rendering pass** with only marginal overhead.

### Gradient-Aware Spline Interpolation

Bicubic interpolation solves $F = C A C^T$ to obtain polynomial coefficients $A \in \mathbb{R}^{4 \times 4}$:

$$F = \begin{bmatrix} f(0,0) & f(0,1) & f_y(0,0) & f_y(0,1) \\ f(1,0) & f(1,1) & f_y(1,0) & f_y(1,1) \\ f_x(0,0) & f_x(0,1) & f_{xy}(0,0) & f_{xy}(0,1) \\ f_x(1,0) & f_x(1,1) & f_{xy}(1,0) & f_{xy}(1,1) \end{bmatrix}$$

In conventional methods, $f_x, f_y, f_{xy}$ are estimated via finite differences. In the proposed method, they are provided exactly by the analytic gradients derived above.

Interpolation is then evaluated at arbitrary target pixel locations as:

$$p(x,y) = [1, x, x^2, x^3] A [1, y, y^2, y^3]^T$$

### Differentiable Backpropagation

The gradients are fully differentiable and can be integrated into 3DGS optimization: the model renders at low resolution, applies upscaling, computes the loss on the upscaled output, and backpropagates to optimize Gaussian primitive parameters.

$$\frac{\partial^2 B_i}{\partial x \partial \sigma_k} = \frac{\partial^2 B_i}{\partial \alpha_k \partial x} \frac{\partial \alpha_k}{\partial \sigma_k} + \frac{\partial B_i}{\partial \alpha_k} \frac{\partial^2 \alpha_k}{\partial \sigma_k \partial x}$$

## Key Experimental Results

### Rendering Quality Comparison (4× Upscaling)

| Method | Type | PSNR | Extra Cost |
|:---|:---:|:---:|:---|
| Bicubic interpolation | Traditional | Baseline | Negligible |
| Lanczos (FSR) | Traditional | Marginally better than bicubic | Moderate |
| DL-based (ESRGAN, etc.) | Deep Learning | PSNR may decrease | High (GPU inference) |
| **Gradient-aware upscaling** | **Ours** | **Best** | **Marginal** |

Key finding: DL-based upscaling appears sharper visually but introduces hallucinated details, causing PSNR to decrease.

### Speed Improvement

| Configuration | Method | Rendering Speed |
|:---|:---:|:---|
| Full resolution | Standard 3DGS | 1× (baseline) |
| 1/4 resolution + upscaling | Gradient-aware upscaling | **3–4× speedup** |

The overhead of gradient computation is fully offset by the savings from low-resolution rasterization.

### Training Integration

| Mode | Description | Quality |
|:---|:---|:---|
| Upscaling at inference only | Standard training + inference-time upscaling | Better than bicubic |
| **Upscaling during training and inference** | **Low-res training + upscaling + backprop** | **Further improved** |

Integrating upscaling into training allows the model to learn Gaussian primitive shapes optimized for the upscaled scene.

### Ablation Study: Gradient Type Comparison

| Gradient Type | Quality |
|:---|:---|
| Finite differences (standard bicubic) | Ringing artifacts, staircase effects |
| **Analytic gradients (Ours)** | **Smooth, high-fidelity** |

Qualitative comparisons on 2D interpolation benchmarks show that analytic gradients eliminate the ringing and staircase artifacts caused by finite differences.

## Highlights & Insights

1. **Training-free upscaling**: No neural network is introduced; no pretraining is required. The method exploits only the mathematical properties inherent to 3DGS.
2. **Elegant implementation**: Gradient computation is embedded into the existing rendering pipeline, enabling non-intrusive integration with any 3DGS implementation.
3. **Temporal stability**: Unlike DL-based upscaling, which introduces inter-frame inconsistency, analytic gradients guarantee deterministic outputs.
4. **VR-friendly**: High-resolution output can be obtained on lightweight GPUs, making the approach suitable for mobile VR devices.

## Limitations & Future Work

1. At very large upscaling factors (e.g., 8×+), interpolation accuracy remains limited by the sampling density of the low-resolution input.
2. High-frequency texture regions may still exhibit some degree of smoothing.
3. Synergistic effects with anti-aliasing methods such as Mip-Splatting require further investigation.

## Related Work & Insights

- **3DGS optimization**: Mip-Splatting, SRGS, gradient-guided densification
- **Image upscaling**: bicubic interpolation, Lanczos, ESRGAN, DLSS, GaussianSR
- **NeRF super-resolution**: NeRF-SR, RefSR-NeRF

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of leveraging analytic gradients for upscaling is concise and elegant.
- Technical Depth: ⭐⭐⭐⭐ — Complete differentiable derivations for both forward and backward passes; mathematically rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset, multi-baseline evaluation covering both speed and quality.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play, zero additional training, significant acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)

</div>

<!-- RELATED:END -->
