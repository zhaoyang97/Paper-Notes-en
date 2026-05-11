---
title: >-
  [Paper Note] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration
description: >-
  [CVPR 2026][Image Generation][Diffusion model acceleration] This paper proposes LESA, a framework that employs KAN (Kolmogorov-Arnold Network) as learnable temporal predictors…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "feature caching"
  - "KAN"
  - "stage-awareness"
  - "DiT"
  - "text-to-image"
  - "text-to-video"
date: 2026-05-08
content_hash: e5b300dba773be32
---

# LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration

**Conference**: CVPR 2026
**arXiv**: [2602.20497](https://arxiv.org/abs/2602.20497)
**Area**: Image Generation / Diffusion Model Acceleration
**Keywords**: Diffusion model acceleration, feature caching, KAN, stage-awareness, DiT, text-to-image, text-to-video

## TL;DR
This paper proposes LESA, a framework that employs KAN (Kolmogorov-Arnold Network) as learnable temporal predictors, combined with a multi-stage multi-expert architecture and a two-phase training strategy. LESA achieves 5× acceleration on FLUX with only 1.0% quality degradation, 6.25× acceleration on Qwen-Image with 20.2% quality improvement over TaylorSeer, and 5× acceleration on HunyuanVideo with a 24.7% PSNR gain.

## Background & Motivation

**Background**: The DiT architecture has demonstrated remarkable performance in image/video generation, yet the inference cost of deep Transformers combined with multi-step denoising is prohibitive. Feature caching, which exploits temporal redundancy between adjacent timesteps to accelerate inference, has emerged as a prominent research direction.

**Taxonomy of Existing Approaches**: (a) Cache-Then-Reuse — directly reusing features from the previous step (e.g., PAB, DBCache), which ignores temporal evolution and causes error accumulation; (b) Cache-Then-Forecast — predicting features via Taylor expansion (e.g., TaylorSeer, FORA), which assumes smooth and continuous feature evolution, yet the actual diffusion process is **stage-dependent**.

**Key Observation**: Through cosine similarity analysis and PCA trajectory visualization, the feature dynamics of the diffusion process exhibit three distinct stages: highly volatile and unstable dynamics in the high-noise stage, stable and continuous evolution in the intermediate stage, and fine-detail refinement in the low-noise stage. A single fixed prediction strategy cannot accommodate all three regimes.

**Core Idea**: Replace training-free polynomial predictors with **learnable** KANs, and assign **dedicated expert predictors** to each noise stage to explicitly model stage-dependent dynamics.

## Method

### Overall Architecture
During diffusion inference, a subset of timesteps are computed by the DiT model (anchor steps), while the remaining steps are skipped — their features predicted by LESA predictors based on the cached features from the preceding $K$ steps. A full DiT forward pass is performed every $N$ steps.

### Key Designs

1. **KAN-Based Temporal Modeling**:

    - A linear projection maps the cached $K$-step feature sequence into a feature representation: $\mathbf{z} = \mathbf{W}[\mathbf{h}_{t+K-1},...,\mathbf{h}_t] + b$
    - A KAN processes the relative timestep offsets $\Delta t$ and produces a scalar temporal modulation factor: $\alpha = f_{KAN}(\Delta t_{L-1},...,\Delta t_0) = \sum_{m=1}^{M} w_m \phi_m(a_m^\top \Delta \mathbf{t})$
    - Residual prediction: $\hat{\mathbf{h}}_{t-1} = \mathbf{h}_t + \alpha \mathbf{z}$
    - **Separation of Variables Principle**: the linear projection transforms features in the spatial domain, while the KAN learns a continuous scalar modulation in the temporal domain.

2. **Stage-Aware Multi-Expert Architecture**:

    - The denoising process is partitioned into three stages according to noise level, each assigned an independent expert predictor.
    - High-noise expert: handles the volatile early stage with a shorter history window $K=4$.
    - Mid-noise expert: manages the stable, continuous denoising phase.
    - Low-noise expert: responsible for fine-detail refinement with a longer history window $K=8$.
    - The substantial differences in feature dynamics across stages make a single predictor insufficient.

3. **Two-Phase Training Strategy**:

    - **GT-Guided Training**: the ground-truth intermediate features produced by the base model running without acceleration serve as supervision targets, with an L1 loss.
    - **CL-AR Training (Closed-Loop Autoregressive Training)**: the predictor takes its own historical predictions (including accumulated errors) as input, simulating the error accumulation that occurs during inference and building robustness to prediction drift.

### Parameter Efficiency
The KAN architecture introduces very few parameters; each expert consists only of a linear projection and a compact KAN module, with total additional parameters far smaller than the base DiT.

## Key Experimental Results

### Main Results: FLUX.1-dev Text-to-Image

| Method | FLOPs Speedup | ImageReward↑ | CLIP Score↑ | PSNR↑ | LPIPS↓ |
|------|----------|-------------|-------------|-------|--------|
| Original 50 steps | 1.00× | 0.99 | 32.64 | ∞ | 0.00 |
| TaylorSeer (N=6,O=2) | 4.99× | 1.02 | 32.53 | 28.94 | 0.40 |
| TeaCache (l=1.0) | 4.54× | 0.84 | 31.88 | 28.61 | 0.48 |
| **LESA (N=7)** | **5.00×** | **0.98** | **32.88** | **30.17** | **0.32** |
| TaylorSeer (N=9,O=2) | 6.24× | 0.86 | 32.04 | 28.38 | 0.51 |
| **LESA (N=10)** | **6.25×** | **0.91** | **32.65** | **29.65** | **0.40** |

### Main Results: Qwen-Image Text-to-Image

| Method | FLOPs Speedup | ImageReward↑ | PSNR↑ | LPIPS↓ |
|------|----------|-------------|-------|--------|
| TaylorSeer (N=6,O=2) | 5.00× | 1.01 | 28.58 | 0.46 |
| **LESA (N=7)** | **5.00×** | **1.15** | **30.18** | **0.25** |
| TaylorSeer (N=8,O=2) | 6.24× | 0.84 | 28.14 | 0.68 |
| **LESA (N=10)** | **6.25×** | **1.01** | **29.23** | **0.34** |

### Main Results: HunyuanVideo Text-to-Video

| Method | FLOPs Speedup | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|----------|-------|-------|--------|
| TaylorSeer (N=5,O=1) | 5.00× | 17.29 | 0.55 | 0.42 |
| TeaCache (l=0.4) | 4.55× | 18.25 | 0.61 | 0.38 |
| **LESA (N=7)** | **5.00×** | **21.43** | **0.72** | **0.29** |
| **LESA (N=8)** | **5.56×** | **21.05** | **0.70** | **0.32** |

### Ablation Study (FLUX, N=5)

| Stage Split | Timestep Module | PSNR↑ | LPIPS↓ |
|----------|----------|-------|--------|
| ✗ | MLP | 30.29 | 0.31 |
| ✗ | KAN | 30.77 | 0.25 |
| ✔ | MLP | 30.76 | 0.25 |
| **✔** | **KAN** | **30.96** | **0.24** |

### Key Findings
- At 5× acceleration on FLUX, ImageReward drops by only 1.0% (0.99→0.98), achieving **near-lossless acceleration**.
- At 6.25× acceleration on Qwen-Image, ImageReward surpasses TaylorSeer by 20.2% (0.84→1.01).
- At 5× acceleration on HunyuanVideo, PSNR exceeds TaylorSeer by 24.7% (17.29→21.43).
- The combination of KAN and stage partitioning yields the best performance; neither component alone is sufficient.
- The method remains effective on distilled models (FLUX-schnell, Qwen-Lightning).
- At high speedup ratios (6×+), training-free methods degrade severely (†), while LESA remains competitive.

## Highlights & Insights
- **KAN for Temporal Modeling**: the expressive function representation guaranteed by the Kolmogorov-Arnold representation theorem enables learnable temporal extrapolation that is more flexible than Taylor expansion and more parameter-efficient than MLP.
- **Stage-Aware Design** explicitly models the non-uniform dynamics of the diffusion process, aligning with intuition and validated empirically.
- **Closed-Loop Autoregressive Training** is an engineering cornerstone: models trained with GT supervision suffer rapid error accumulation at inference time, whereas closed-loop training substantially improves robustness.
- Strong generalization across models (FLUX/Qwen/HunyuanVideo) and tasks (T2I/T2V).

## Limitations & Future Work
- Collecting feature trajectories for training requires a preliminary run of the base model without acceleration.
- The predictor must be retrained for each new model, limiting plug-and-play usability.
- Stage boundary positions are set manually; adaptive partitioning remains unexplored.

## Rating
- **Novelty**: ⭐⭐⭐⭐ KAN combined with stage-aware predictors constitutes a novel combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 3 T2I models, 1 T2V model, distilled models, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous analysis with clear figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Substantial practical inference acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration](tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)

</div>

<!-- RELATED:END -->
