---
title: >-
  [Paper Note] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models
description: >-
  [CVPR2026][Image Generation][diffusion model acceleration] This paper proposes SeaCache, a training-free dynamic caching strategy based on a Spectral-Evolution-Aware (SEA) filter. By separating signal and noise component…
tags:
  - "CVPR2026"
  - "Image Generation"
  - "diffusion model acceleration"
  - "caching strategy"
  - "spectral evolution"
  - "frequency-domain filtering"
  - "training-free acceleration"
date: 2026-05-08
content_hash: 4b0d39054c7efdc0
---

# SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models

**Conference**: CVPR2026
**arXiv**: [2602.18993](https://arxiv.org/abs/2602.18993)  
**Code**: [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache)  
**Area**: Image Generation
**Keywords**: diffusion model acceleration, caching strategy, spectral evolution, frequency-domain filtering, training-free acceleration

## TL;DR

This paper proposes SeaCache, a training-free dynamic caching strategy based on a Spectral-Evolution-Aware (SEA) filter. By separating signal and noise components in the frequency domain to measure inter-timestep redundancy, SeaCache significantly improves the latency–quality trade-off in diffusion model inference.

## Background & Motivation

1. **Inference latency bottleneck**: Diffusion models and rectified flow models require tens to hundreds of iterative denoising steps, resulting in severe latency in user-facing applications.
2. **Limitations of existing acceleration methods**: Distillation, quantization, and efficient attention mechanisms, while effective, introduce additional training overhead and dependencies on specific tasks or data.
3. **Potential of caching-based acceleration**: Reusing intermediate features from adjacent timesteps reduces the number of forward passes without retraining, representing a complementary acceleration pathway.
4. **Static vs. dynamic scheduling**: Early methods such as DeepCache employ fixed-interval caching, which cannot adapt to input diversity; TeaCache and DiCache introduce dynamic scheduling but still measure distances in the raw feature space.
5. **Neglect of spectral evolution**: The diffusion denoising process exhibits well-defined spectral evolution—early timesteps establish low-frequency structure while later timesteps refine high-frequency details—yet existing caching strategies treat all spectral components uniformly.
6. **Entanglement of content and noise**: Raw feature distances conflate content-bearing signal components with stochastic noise components, causing caching decisions to be perturbed by high-frequency noise and thus deviating from optimal scheduling.

## Method

### Overall Architecture

SeaCache inserts a Spectral-Evolution-Aware (SEA) filtering step prior to the distance measurement used in existing dynamic caching strategies. Given input features $I_t$ and $I_{t+1}$ at adjacent timesteps, the features are first transformed to the frequency domain via FFT, multiplied by the timestep-dependent SEA filter $G_t^{\text{norm}}$, and then transformed back to the spatial domain via iFFT. The relative $\ell_1$ distance is subsequently computed on the filtered features. When the accumulated distance exceeds a threshold $\delta$, a refresh is triggered; otherwise, the cached output is reused.

### SEA Filter Design

- **Theoretical foundation**: Starting from the linear minimum mean squared error (MMSE) denoiser, the frequency response of the optimal linear denoising filter is derived as $G_t(f) = a_t S_x(f) / (a_t^2 S_x(f) + b_t^2)$, taking the form of a Wiener filter.
- **Spectral evolution modeling**: Assuming that the power spectrum of natural images follows a $1/f$ power-law distribution, the filter predominantly passes low frequencies at early timesteps and progressively incorporates high frequencies at later timesteps, consistent with the spectral evolution of diffusion denoising.
- **Gain normalization**: The average gain of the raw $G_t(f)$ varies across timesteps, rendering cross-timestep distance comparisons unreliable. A normalization factor $\nu_t$ is introduced so that $G_t^{\text{norm}}(f)$ achieves unit mean gain over radial frequencies, stabilizing the energy of the filtered features.
- **Filtering operation**: $\mathcal{P}(G_t^{\text{norm}}, I_t) = \text{iFFT}(G_t^{\text{norm}}(f) \odot \text{FFT}(I_t))$, applied channel-wise over spatial axes (images) or spatiotemporal axes (videos).

### Spectrally Aware Dynamic Caching

- **Input-side proxy**: Directly using filtered output features is infeasible as it requires a full forward pass; filtered input features $\mathcal{P}(G_t^{\text{norm}}, I_t)$ are therefore used as a proxy. Experiments confirm that the distance between SEA-filtered input features closely approximates that between SEA-filtered output features.
- **Distance metric**: $\widetilde{\Delta}_t = \text{L1}_{\text{rel}}(\mathcal{P}(G_t^{\text{norm}}, I_t), \mathcal{P}(G_{t+1}^{\text{norm}}, I_{t+1}))$
- **Cumulative threshold rule**: The cumulative distance refresh logic from TeaCache is retained unchanged; only the distance metric is replaced.
- **Plug-and-play**: No modifications to the network architecture or sampler are required; a single FFT–filter–iFFT step is inserted before distance computation.

## Key Experimental Results

### Text-to-Image (FLUX.1-dev, 50 steps, DrawBench 200 prompts)

| Method | Latency (s) | TFLOPs | PSNR↑ | LPIPS↓ | SSIM↑ |
|--------|-------------|--------|-------|--------|-------|
| Original | 20.9 | 2976 | – | – | – |
| TeaCache (δ=0.3) | 11.4 | 1547 | 20.76 | 0.211 | 0.810 |
| TaylorSeer (S=3) | 9.8 | 1191 | 22.78 | 0.163 | 0.828 |
| **SeaCache (δ=0.3)** | **9.4** | **1098** | **26.29** | **0.106** | **0.893** |
| TeaCache (δ=0.6) | 7.1 | 892 | 17.21 | 0.348 | 0.714 |
| TaylorSeer (S=5) | 7.5 | 834 | 19.97 | 0.236 | 0.762 |
| **SeaCache (δ=0.6)** | **6.4** | **773** | **21.33** | **0.226** | **0.798** |

### Text-to-Video (HunyuanVideo / Wan2.1 1.3B, 50 steps, VBench 944 prompts)

- **HunyuanVideo ~50%**: SeaCache PSNR 32.39 vs. TeaCache 23.40 (+9 dB), latency 90.8s vs. 98.5s
- **HunyuanVideo ~30%**: SeaCache PSNR 26.46 vs. TeaCache 20.42 (+6 dB), latency 58.1s vs. 64.4s
- **Wan2.1 ~50%**: SeaCache PSNR 26.60 vs. TeaCache 20.84 (+5.8 dB), latency 83.9s vs. 86.6s
- **Wan2.1 ~30%**: SeaCache PSNR 21.78 vs. TeaCache 18.88 (+2.9 dB), latency 56.6s vs. 63.6s

### Ablation Study

| Variant | Effect |
|---------|--------|
| SEA filter (full) | Optimal PSNR–refresh-rate trade-off |
| 1−SEA (complementary filter) | Slightly inferior; tracking noise components is less effective than tracking signal components |
| Without gain normalization | PSNR degradation; cross-timestep distance bias |
| Static low-pass filter (LPF 30%) | Substantially worse than SEA, demonstrating the importance of timestep-dependent spectral evolution |

## Highlights & Insights

- **Theory-driven design**: The timestep-dependent SEA filter is derived from the optimal Wiener filter, achieving a tight integration of theory and practice.
- **Plug-and-play**: The method replaces only one filtering step within the distance metric and can be directly embedded into existing caching methods such as TeaCache and DiCache.
- **Cross-model generalization**: Consistent improvements over baselines are demonstrated across FLUX (image), HunyuanVideo, and Wan2.1 (video).
- **Adaptive early refresh**: More compute budget is naturally allocated to earlier timesteps without requiring manual hyperparameters such as "compute the first N steps unconditionally."
- **Substantial PSNR gains**: The +9 dB PSNR improvement on HunyuanVideo is particularly notable.

## Limitations & Future Work

- **Linear denoiser assumption**: The SEA filter is derived under the assumption of an optimal linear denoiser, whereas actual diffusion models are highly nonlinear; the filter is therefore only an approximation.
- **Fixed power spectrum prior**: The method assumes a natural $1/f$ power spectrum, and its applicability to non-natural images (e.g., text, charts) remains to be verified.
- **Addresses only "when to reuse"**: Spectrally aware strategies for "how to reuse" (e.g., differential reuse across frequency bands) are not explored.
- **Reconstruction-centric evaluation**: Metrics such as PSNR, LPIPS, and SSIM measure deviation from a full-computation reference; reporting on downstream perceptual quality (FID, user preference) is relatively limited (only CycleReward is reported).
- **FFT overhead**: Although lightweight, the FFT/iFFT operations introduce additional computation at each timestep, which may become non-negligible in extreme acceleration scenarios.

## Related Work & Insights

| Method | Scheduling Type | Distance Space | Spectral Awareness | Training Required |
|--------|----------------|----------------|--------------------|-------------------|
| DeepCache | Static | – | No | No |
| PAB | Static (block-wise) | – | No | No |
| TeaCache | Dynamic | Raw features | No | No |
| TaylorSeer | Dynamic (Taylor expansion) | Raw features | No | No |
| DiCache | Dynamic (middle blocks) | Raw features | No | No |
| **SeaCache** | **Dynamic** | **SEA-filtered features** | **Yes** | **No** |

SeaCache is the first method to inject explicit frequency priors into cache reuse decisions. By reweighting features in the frequency domain to suppress noise and emphasize content, it enables caching schedules that more faithfully track the full-computation trajectory.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing spectral evolution priors into cache scheduling is a novel perspective; the theoretical derivation of the SEA filter is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both image and video generation across multiple models, with complete ablations and thorough plug-and-play validation.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, theoretical derivations are self-consistent, and figures are informative.
- Value: ⭐⭐⭐⭐ — The plug-and-play nature and substantial performance gains offer direct practical value for diffusion model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[ICCV 2025\] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](../../ICCV2025/image_generation/from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)
- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](../../AAAI2026/image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)

</div>

<!-- RELATED:END -->
