---
title: >-
  [Paper Note] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers
description: >-
  [ICCV 2025][Image Generation][Diffusion model acceleration] This paper proposes TaylorSeer, which upgrades the feature caching paradigm for diffusion models from "cache-and-reuse" to "cache-and-forecast" — leveraging Tay…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "feature caching"
  - "Taylor expansion"
  - "training-free acceleration"
  - "DiT"
  - "FLUX"
  - "HunyuanVideo"
date: 2026-05-08
content_hash: f6ea46a956f00efa
---

# From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers

**Conference**: ICCV 2025
**Code**: [https://github.com/Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)  
**Area**: Image Generation
**Keywords**: Diffusion model acceleration, feature caching, Taylor expansion, training-free acceleration, DiT, FLUX, HunyuanVideo

## TL;DR

This paper proposes TaylorSeer, which upgrades the feature caching paradigm for diffusion models from "cache-and-reuse" to "cache-and-forecast" — leveraging Taylor series expansion with high-order finite differences over historical features to predict intermediate features at future timesteps. TaylorSeer achieves near-lossless 4.99× acceleration on FLUX and 5.00× on HunyuanVideo, entirely without additional training.

## Background & Motivation

Diffusion Transformers (DiT) have achieved revolutionary progress in high-fidelity image and video generation, yet their substantial computational demands remain the primary bottleneck for real-time applications.

**Fundamental limitations of feature caching**: Existing feature caching methods (e.g., FORA, Δ-DiT, TeaCache) follow a "cache-and-reuse" paradigm — directly reusing features computed at a previous timestep in subsequent steps. While effective for adjacent timesteps, this approach has a fundamental limitation: **feature similarity decays exponentially as the timestep gap increases**, causing the error introduced by direct reuse to grow sharply and significantly degrading generation quality. This confines feature caching methods to a narrow range of low speedup ratios.

**Key observation**: Through PCA visualization of features across different timesteps, the authors find that:
1. Features form **stable trajectories** across timesteps, indicating that future features are predictable.
2. The **derivatives** of features (i.e., velocities along the trajectory) exhibit high stability and continuity between adjacent timesteps.

This suggests that predicting future features is not a complex problem and can even be solved with non-parametric methods.

## Method

### Overall Architecture: From "Reuse" to "Forecast"

TaylorSeer upgrades diffusion model feature caching from direct copying to trajectory prediction based on Taylor series. The core idea is to use multi-step historical features to approximate derivatives of various orders, then apply Taylor expansion to predict features at future timesteps.

### Hierarchical Prediction Formula

For a feature function $\mathcal{F}(x_t^l)$ that is $(m+1)$-times differentiable, the feature at a future timestep $t-k$ can be expressed via Taylor expansion as:

$$\mathcal{F}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^{m} \frac{\mathcal{F}^{(i)}(x_t^l)}{i!}(-k)^i + R_{m+1}$$

To avoid explicit computation of high-order derivatives, **finite differences** are used for recursive approximation:

$$\Delta^i \mathcal{F}(x_t^l) = \Delta^{i-1}\mathcal{F}(x_{t+N}^l) - \Delta^{i-1}\mathcal{F}(x_t^l)$$

where the $i$-th order finite difference approximates the $i$-th order derivative scaled by $N^i$: $\Delta^i \mathcal{F}(x_t^l) \approx N^i \mathcal{F}^{(i)}(x_t^l)$

Substituting into the Taylor expansion yields the **order-$m$ prediction formula**:

$$\mathcal{F}_{\text{pred},m}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^{m} \frac{\Delta^i \mathcal{F}(x_t^l)}{i! \cdot N^i}(-k)^i$$

Only $(m+1)$ fully computed timesteps $\{t+mN, \dots, t+N, t\}$ are needed to predict intermediate timestep features.

### Unified Perspective

- **$m=0$**: Degenerates to naive feature caching (direct reuse).
- **$m=1$**: Linear prediction, using first-order finite differences to capture linear trends.
- **$m \geq 2$**: Higher-order prediction, capturing nonlinear trajectory dynamics and reducing long-range errors.

### Error Bound Analysis

The prediction error has a rigorous theoretical upper bound:

$$E_m(k) \leq \frac{M_{m+1}}{(m+1)!}|k|^{m+1} + \sum_{i=1}^{m}\frac{C_i}{i! \cdot |N|^{i-1}}|k|^i$$

This reveals the fundamental trade-off between order and error: higher orders effectively reduce the dominant error term but introduce additional finite-difference approximation errors.

## Key Experimental Results

### Main Results: FLUX Text-to-Image Generation (Image Reward)

| Method | Speedup | Image Reward ↑ | CLIP ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|
| FLUX Original (50 steps) | 1.00× | 0.9898 | 19.604 | — | — | — |
| Δ-DiT (N=3) | 1.95× | 0.8561 | 18.833 | 28.794 | 0.6665 | 0.4133 |
| FORA (N=3) | 2.82× | 0.9227 | 18.950 | 30.652 | 0.7666 | 0.2450 |
| DuCa (N=5) | 3.45× | 0.9896 | 19.595 | 29.413 | 0.7142 | 0.3082 |
| **TaylorSeer (N=3,O=2)** | **2.82×** | **1.0181** | **19.397** | **30.762** | **0.7818** | **0.2300** |
| FORA (N=6) | 4.99× | 0.7761 | 17.986 | 28.360 | 0.6001 | 0.5177 |
| DuCa (N=6) | 4.56× | 0.9470 | 19.082 | 28.672 | 0.6228 | 0.4182 |
| **TaylorSeer (N=6,O=2)** | **4.99×** | **1.0039** | **19.427** | **28.945** | **0.6556** | **0.4020** |

At 4.99× speedup, TaylorSeer's Image Reward **exceeds the original model**, while all competing methods suffer severe degradation.

### Ablation/Comparison: DiT-XL/2 Class-Conditional Image Generation (FID-50k)

| Method | Speedup | FID ↓ | sFID ↓ | IS ↑ |
|---|---|---|---|---|
| DDIM-50 steps | 1.00× | 2.32 | 4.32 | 241.25 |
| FORA (N=3) | 2.77× | 3.55 | 6.36 | 229.02 |
| DuCa (N=3) | 2.48× | 2.88 | 4.66 | 233.37 |
| **TaylorSeer (N=3,O=3)** | **2.77×** | **2.34** | **4.69** | **238.42** |
| FORA (N=5) | 4.53× | 6.58 | 11.29 | 193.01 |
| DuCa (N=5) | 3.78× | 6.06 | 6.72 | 198.46 |
| **TaylorSeer (N=5,O=3)** | **4.53×** | **2.65** | **5.36** | **231.59** |

At 4.53× speedup, TaylorSeer achieves an FID of only 2.65 (vs. 6.58 for FORA and 6.06 for DuCa), **3.41 lower than the previous SOTA**.

### HunyuanVideo Video Generation

| Method | Speedup | VBench ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| Original (50 steps) | 1.00× | 80.66 | — | — | — |
| FORA (N=5) | 5.00× | 78.83 | 16.072 | 0.6334 | 0.3457 |
| TeaCache (l=0.4) | 4.55× | 79.36 | 16.072 | 0.6216 | 0.4377 |
| **TaylorSeer (N=5,O=1)** | **5.00×** | **79.93** | **16.796** | **0.7039** | **0.2691** |

### Key Findings

1. **Absolute advantage at high speedup ratios**: At >4× speedup, all "cache-and-reuse" methods degrade noticeably, while TaylorSeer maintains near-original quality.
2. **Efficiency over quality loss**: Quality degradation is reduced by **36×** compared to the previous SOTA.
3. **Viable up to 6× speedup**: All prior methods fail completely at >6×, while TaylorSeer still produces acceptable results.
4. **Image Reward can surpass the original model**: In certain configurations (e.g., N=4, O=2), TaylorSeer's generation quality actually exceeds that of the unaccelerated original model.

## Highlights & Insights

1. **Paradigm innovation**: The shift from "cache-and-reuse" to "cache-and-forecast" is the paper's most significant contribution — not merely an incremental improvement, but the opening of a new research direction.
2. **Mathematical elegance**: Taylor series expansion provides a unified and elegant mathematical framework for feature prediction, subsuming direct caching and high-order prediction under a single formula.
3. **Fully training-free**: No search procedure or additional training cost is required; the method is plug-and-play.
4. **Validated on both images and video**: Near-lossless ~5× acceleration is achieved on both FLUX (images) and HunyuanVideo (video), demonstrating the generality of the approach.

## Limitations & Future Work

1. **Additional memory for caching**: Higher-order prediction requires storing features and finite differences from multiple timesteps; memory overhead grows with the order.
2. **Strength of Assumption 1**: The method relies on the assumption that the feature function is differentiable and that higher-order derivatives are bounded, which may break down when the diffusion process exhibits discontinuous changes at certain timesteps.
3. **Order selection requires tuning**: The optimal Taylor order varies across models and speedup ratios (e.g., O=2 for FLUX, O=3/4 for DiT).
4. **Not validated on high-resolution ultra-long video**: The HunyuanVideo experiments are limited in resolution and frame count.

## Related Work & Insights

- **Complementarity with ODE-solver acceleration**: TaylorSeer accelerates the computation of the denoising network and is orthogonal to methods that reduce the number of sampling steps (e.g., DPM-Solver); the two can be combined.
- **Continuity of feature trajectories**: The smooth variation of features across timesteps reflects a deep physical intuition — the SDE underlying diffusion models inherently defines smooth forward and reverse processes, of which feature continuity is a natural consequence.
- **Implications for future caching methods**: Future caching approaches may evolve from purely numerical methods to learned predictors (e.g., small MLPs predicting feature changes), further improving accuracy.

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — Paradigm-level innovation, elegant and concise.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers image (FLUX, DiT) and video (HunyuanVideo) across multiple speedup ratios and baselines.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free, plug-and-play, ~5× acceleration with near-lossless quality.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and figures are intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](../../AAAI2026/image_generation/specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)
- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](../../CVPR2026/image_generation/seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)

</div>

<!-- RELATED:END -->
