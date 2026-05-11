---
title: >-
  [Paper Note] TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration
description: >-
  [CVPR 2026][Image Generation][diffusion model acceleration] This paper proposes TC-Padé, a feature residual prediction framework based on Padé rational function approximation. Through adaptive coefficient modulation and…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "diffusion model acceleration"
  - "feature caching"
  - "Padé approximation"
  - "trajectory consistency"
  - "residual prediction"
date: 2026-05-08
content_hash: 08f1f45eed79be81
---

# TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration

**Conference**: CVPR 2026
**arXiv**: [2603.02943](https://arxiv.org/abs/2603.02943)
**Area**: Image Generation
**Keywords**: diffusion model acceleration, feature caching, Padé approximation, trajectory consistency, residual prediction

## TL;DR

This paper proposes TC-Padé, a feature residual prediction framework based on Padé rational function approximation. Through adaptive coefficient modulation and a stage-aware strategy, TC-Padé achieves trajectory-consistent acceleration in low-step (20–30 steps) diffusion sampling scenarios (2.88× on FLUX.1-dev, 1.72× on Wan2.1), significantly outperforming existing Taylor-expansion-based methods.

## Background & Motivation

Diffusion models have achieved state-of-the-art performance in image and video generation, but their iterative denoising process requires tens to hundreds of network forward passes, incurring substantial computational cost. Existing acceleration approaches fall into two categories:

**Reducing sampling steps**: solver-based methods such as DDIM and DPM-Solver, as well as distillation methods (consistency models, adversarial distillation).

**Reducing per-step computation**: model compression (pruning, quantization) and feature caching.

Feature caching has attracted attention due to its **training-free, plug-and-play** nature. However, existing methods suffer from critical limitations:

- **Reuse-based methods** (ToCa, Δ-DiT, TeaCache): perform reasonably well at higher step counts (50 steps), but when steps are reduced to 20–30, the larger time intervals between adjacent steps cause feature similarity to decay exponentially, and direct reuse leads to severe trajectory deviation.
- **Prediction-based methods** (TaylorSeer): employ polynomial extrapolation via Taylor series expansion, but the finite convergence radius of Taylor expansion causes approximation error to grow sharply as the interval increases.

The authors use PCA visualization to confirm that existing caching methods produce significant deviations from the ground-truth feature trajectory under 20-step sampling.

## Method

### Overall Architecture

TC-Padé partitions the sampling trajectory into cache intervals of length $\mathcal{N}$. Within each interval, only the first step performs a full forward pass; subsequent steps adaptively determine the computation mode via a **Trajectory Stability Indicator (TSI)**:

$$\text{TSI}(\mathcal{R}_{t+3}, \mathcal{R}_{t+2}, \mathcal{R}_{t+1}) = \frac{1}{2}\|\mathbf{u}_{t+1} - \mathbf{u}_{t+2}\|_2$$

where $\mathbf{u}_t = (\mathcal{R}_t - \mathcal{R}_{t+1}) / \|\mathcal{R}_t - \mathcal{R}_{t+1}\|_2$ is the normalized residual difference vector. When $\text{TSI} \geq \theta$, computation is skipped and the residual is predicted via Padé approximation; otherwise, full computation is performed to preserve generation quality.

### Key Design 1: Residual-Based Padé Approximation Prediction

**Why use residuals rather than raw features?** The authors observe that residuals (inter-layer increments $\mathcal{R}_t^{l:r} = x_t^r - x_t^l$) exhibit substantially higher temporal similarity than raw features. When TaylorSeer directly predicts raw features, cosine similarity drops below 0.5 as the step interval grows; by contrast, residual cosine similarity remains consistently high.

**Padé approximation vs. Taylor expansion**: Taylor series are polynomial approximations with a finite convergence radius; Padé approximation employs a rational function $P_m(x)/Q_n(x)$, which better captures asymptotic behavior and nonlinear transitions. A $[2/1]$-order Padé approximation ($k=3, m=1$) is adopted:

$$\mathcal{R}_{Pad\acute{e},t} = \frac{b_0 \mathcal{R}_{t+3} + b_1 \mathcal{R}_{t+2}}{1 + a_1 \mathcal{R}_{t+1}}$$

The output feature is reconstructed from the predicted residual as: $\bar{x}_t = x_{t+1} + \mathcal{R}_{Pad\acute{e},t}$

### Key Design 2: Adaptive Coefficient Modulation

Rather than solving coefficients analytically as in classical Padé approximation, coefficients are dynamically modulated by a stability factor $\sigma_{stab}$:

$$\sigma_{stab} = \exp\left(-\lambda \frac{\|\mathcal{R}_{t+1} - \mathcal{R}_{t+2}\|}{\|\mathcal{R}_{t+1} + \mathcal{R}_{t+2}\|}\right)$$

When the residual changes rapidly, $\sigma_{stab} \to 0$ and the coefficients adopt conservative values; when the residual is stable, $\sigma_{stab} \to 1$ and the prediction is fully utilized. The coefficients are set as:

$$b_0 = 2\sigma_{stab}, \quad b_1 = -\sigma_{stab}, \quad a_1 = \frac{1}{\lambda}\sigma_{stab}$$

### Key Design 3: Denoising Stage-Aware Strategy

The denoising process is divided into three stages with distinct residual update strategies:

- **Early stage** ($t > 0.7T$): Structure evolves rapidly; a weighted combination of the two most recent residuals is used directly: $\alpha_1 \mathcal{R}_{t+1} + \alpha_2 \mathcal{R}_{t+2}$ ($\alpha_1 + \alpha_2 = 1$).
- **Middle stage** ($0.2T \leq t \leq 0.7T$): The full Padé approximation $\mathcal{R}_{Pad\acute{e},t}$ is applied to capture long-range dependencies.
- **Late stage** ($t < 0.2T$): A first-order difference term $\beta(\mathcal{R}_{t+1} - \mathcal{R}_{t+2})$ is added on top of the Padé prediction to capture subtle velocity changes.

### Loss & Training

This is a **training-free** method and involves no loss function design. The core idea is to substitute Padé rational function approximation for full network computation during inference.

## Key Experimental Results

### Main Results: Text-to-Image Generation (FLUX.1-dev, 20 steps, COCO 2017)

| Method | Speedup | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|---------|------|-------|-------|-------|--------|
| FLUX.1-dev (baseline) | 1.00× | 23.38 | 32.10 | - | - | - |
| ToCa (N=5) | 1.81× | 24.18 | 31.48 | 17.29 | 0.613 | 0.481 |
| TeaCache (fast) | 2.15× | 24.11 | 31.50 | 18.02 | 0.690 | 0.419 |
| TaylorSeer (N=5) | 2.31× | †severe degradation | 31.52 | 17.46 | 0.525 | 0.616 |
| **TC-Padé (slow)** | **2.20×** | **23.85** | **31.90** | **24.67** | **0.861** | **0.144** |
| **TC-Padé (fast)** | **2.88×** | **24.14** | **31.82** | **21.96** | **0.782** | **0.290** |

### Main Results: Text-to-Video Generation (Wan2.1-1.3B, 20 steps, VBench-2.0)

| Method | Speedup | VBench-2.0↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|---------|-------------|-------|-------|--------|
| Wan2.1 (baseline) | 1.00× | 64.16% | - | - | - |
| TeaCache (slow) | 1.17× | 60.73% | 27.19 | 0.867 | 0.107 |
| TaylorSeer (N=4) | 1.66× | 54.50% | 14.93 | 0.353 | 0.586 |
| **TC-Padé (fast)** | **1.72×** | **60.38%** | **21.70** | **0.639** | **0.300** |

### Main Results: Class-Conditional Image Generation (DiT-XL/2, 20 steps, ImageNet)

| Method | Speedup | FID↓ | IS↑ | Precision↑ | Recall↑ |
|--------|---------|------|-----|-----------|---------|
| DiT-XL/2 (baseline) | 1.00× | 3.56 | 221.27 | 0.78 | 0.58 |
| ToCa (N=3) | 1.35× | 10.72 | 164.40 | 0.69 | 0.49 |
| TaylorSeer (N=4) | 1.51× | 7.86 | 175.11 | 0.71 | 0.53 |
| **TC-Padé (fast)** | **1.46×** | **6.93** | **185.12** | **0.72** | **0.54** |

### Ablation Study: Cache Residual Granularity (FLUX.1-dev)

| Granularity | Speedup | Aesthetic↑ | CLIP↑ | ImgRwd↑ |
|-------------|---------|-----------|-------|---------|
| Double-stream | 1.36× | 5.10 | 31.31 | 0.792 |
| Single-stream | 1.94× | 5.69 | 31.66 | 0.872 |
| **Entire Block** | **2.88×** | **5.76** | **31.83** | **0.918** |

### Ablation Study: Effect of TSI Threshold θ

| θ | Speedup | Aesthetic↑ | CLIP↑ | ImgRwd↑ |
|---|---------|-----------|-------|---------|
| 1.3 | 1.63× | 5.80 | 32.02 | 0.956 |
| 1.0 | 2.20× | 5.77 | 31.97 | 0.924 |
| 0.7 | 2.88× | 5.76 | 31.83 | 0.918 |

### Deployment Efficiency: Combination with Quantization

| Configuration | FID↓ | CLIP↑ | Aesthetic↑ |
|---------------|------|-------|-----------|
| FLUX.1-dev | 23.38 | 32.10 | 6.25 |
| TC-Padé | 24.14 | 31.82 | 6.11 |
| TC-Padé + Quantization | 24.31 | 31.08 | 6.01 |

TC-Padé combined with quantization reduces generation latency from 9s to 1.83s at batch=1 (approximately 6× acceleration), increasing throughput from 0.22 img/s to 0.54–0.57 img/s.

### Key Findings

- TC-Padé substantially outperforms all compared methods in PSNR/SSIM/LPIPS under the 20-step setting, indicating that its generated outputs closely match the full-step baseline.
- TaylorSeer suffers severe FID degradation on 20-step FLUX.1-dev (marked †), whereas TC-Padé incurs only approximately 3% FID loss.
- Combining TC-Padé with quantization achieves approximately 6× latency reduction with minimal quality degradation.

## Highlights & Insights

1. **Sound mathematical foundation**: The motivation for replacing Taylor polynomials with Padé rational functions is clear — rational functions can capture asymptotic behavior and poles, whereas polynomial expansions diverge at large intervals. This represents an elegant transfer from numerical analysis to deep learning.
2. **Residuals over raw features**: Predicting residuals (inter-layer increments) is more stable than predicting high-dimensional raw features — an observation with independent value beyond this work.
3. **Principled stage-aware strategy**: Conservative reuse in early stages, Padé prediction in the middle stage, and difference-term correction in the late stage align naturally with the dynamics of diffusion models at different denoising phases.
4. **Adaptive stability detection**: The TSI metric and adaptive coefficient design enable the method to detect abrupt trajectory changes and fall back to full computation when instability is detected.
5. **Orthogonal compatibility with quantization**: The demonstrated composability with quantization and other acceleration techniques underscores the method's practical utility.

## Limitations & Future Work

1. **Hyperparameter sensitivity**: Parameters including λ, θ, α, and β require tuning and may need different settings across models and tasks.
2. **Limitations of low-order approximation**: The $[2/1]$-order Padé is chosen for efficiency, which may yield insufficient accuracy in regions of rapid feature change.
3. **Validation limited to 20 steps**: Although the method targets low-step regimes, evaluation under more extreme settings (e.g., 8–10 steps) is absent.
4. **Moderate speedup ratios**: The method achieves only 1.46× on DiT-XL/2 and 1.72× on video generation, remaining well below distillation-based approaches.
5. **No direct comparison with distillation methods**: Comparisons are confined to feature-caching methods; performance relative to consistency models and similar approaches is not examined.
6. **Heuristic stage boundaries**: The stage partitioning thresholds (0.2T, 0.7T) in the stage-aware strategy are empirically chosen and lack theoretical justification.

## Rating

⭐⭐⭐⭐ (4/5)

The mathematical motivation is clear and the method design is elegant, with experiments covering both image and video generation comprehensively. The work represents a meaningful advance in the feature-caching acceleration track under low-step settings. Nevertheless, the core contribution leans toward engineering-level optimization, and there remains room for improvement in theoretical depth and generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration](lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)

</div>

<!-- RELATED:END -->
