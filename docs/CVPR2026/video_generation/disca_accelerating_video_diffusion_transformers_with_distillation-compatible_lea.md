---
title: >-
  [Paper Note] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching
description: >-
  [CVPR2026][Video Generation][Video diffusion model acceleration] This paper proposes DisCa, the first framework to combine **learnable feature caching** with **step distillation** by replacing handcrafted caching strategies with lightweight neural predictors. It further introduces Restricted MeanFlow to stabilize large-scale video model distillation, achieving an **11.8× speedup** on HunyuanVideo with negligible quality degradation.
tags:
  - CVPR2026
  - Video Generation
  - Video diffusion model acceleration
  - feature caching
  - step distillation
  - MeanFlow
  - learnable predictor
  - GAN training
date: 2026-05-08
content_hash: 2b71f315d0c6d9c8
---

# DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching

**Conference**: CVPR2026
**arXiv**: [2602.05449](https://arxiv.org/abs/2602.05449)
**Code**: Coming soon
**Area**: Image Generation / Video Generation Acceleration
**Keywords**: Video diffusion model acceleration, feature caching, step distillation, MeanFlow, learnable predictor, GAN training

## TL;DR

This paper proposes DisCa, the first framework to combine **learnable feature caching** with **step distillation** by replacing handcrafted caching strategies with lightweight neural predictors. It further introduces Restricted MeanFlow to stabilize large-scale video model distillation, achieving an **11.8× speedup** on HunyuanVideo with negligible quality degradation.

## Background & Motivation

1. **Prohibitive computational cost of diffusion models**: State-of-the-art video diffusion models such as HunyuanVideo require dozens of iterative denoising steps, taking over 1,155 seconds to generate a 5-second 704×704 video, making deployment extremely costly.
2. **Training-free feature caching has an inherent ceiling**: Conventional training-free caching methods (e.g., direct reuse or interpolation) inevitably lose semantic and detail information at high compression ratios, with semantic scores dropping by 13–27%.
3. **Step distillation underperforms on video generation**: Methods such as MeanFlow excel at image generation but fail when directly applied to large-scale video models, as numerical errors in Jacobian-vector product (JVP) computations cause training divergence and severe generation artifacts.
4. **Incompatibility between caching and distillation**: Distilled models operate with sparse sampling steps, significantly increasing feature discrepancy between adjacent steps and invalidating traditional caching methods that rely on inter-step similarity.
5. **Limited capacity of handcrafted prediction functions**: Methods such as TaylorSeer predict feature evolution via Taylor expansion, but such manually designed functions with fixed prior assumptions struggle to capture complex high-dimensional feature dynamics.
6. **Memory inefficiency**: TaylorSeer, FORA, and similar methods maintain multi-order cached tensors per layer, consuming over 120 GB of VRAM for high-resolution long videos—beyond the capacity of even 4-way sequence parallelism.

## Method

### Overall Architecture

DisCa consists of two core modules: **Restricted MeanFlow** (for stable distillation) and **Learnable Feature Caching** (for accelerated inference). The overall pipeline first applies CFG distillation to HunyuanVideo (eliminating the dual forward pass of classifier-free guidance), then performs step distillation via Restricted MeanFlow to compress the number of sampling steps, and finally trains a lightweight predictor to enable feature-caching-based acceleration.

### Restricted MeanFlow

- **Problem**: The original MeanFlow targets one-step distillation with a mean-velocity sampling interval $\mathcal{I} = (t-r) \in [0,1]$, which is too aggressive for large-scale video models and leads to training instability at high compression ratios.
- **Solution**: A restriction factor $\mathcal{R} \in (0,1)$ is introduced to clip the sampling interval to $\mathcal{I} \in [0, \mathcal{R}]$, directly excluding overly aggressive high-compression scenarios from the MeanFlow training objective.
- **Effect**: At $\mathcal{R}=0.2$, the semantic score improves by 5.7% over the original MeanFlow at 20 steps and by 12.0% at 10 steps, effectively eliminating distortion and artifacts.

### Lightweight Learnable Predictor

- **Inference**: In every $N$-step cycle, the first step runs the full model $\mathcal{M}$ to compute and refresh the cache $\mathcal{C}$; the subsequent $N-1$ steps use the lightweight predictor $\mathcal{P}$ to rapidly estimate features from the cache.
- **Architecture**: The predictor is composed of a small stack of DiT Blocks, with a parameter count consistently below 4% of the full model.
- **Cache design**: Only a **single tensor** is maintained as a global cache (no per-layer multi-order caching), substantially reducing VRAM consumption.

### Loss & Training

- **MSE loss**: The predictor is supervised using the full model's output as ground truth: $\mathcal{L}(\theta_p) = \mathbb{E}\|\mathcal{M}(x_{t'}, r', t') - \mathcal{P}(\mathcal{C}, x_{t'}, r', t')\|_2^2$
- **GAN adversarial training**: A multi-scale discriminator with spectral normalization and Hinge Loss is introduced. The full model is used as a feature extractor $\mathcal{F}$, and adversarial training is conducted in the perceptual feature space to compensate for high-frequency detail and semantic structure loss.
- **Discriminator loss** $\mathcal{L}_\mathcal{D}$: Maximizes the margin between real and generated samples.
- **Predictor loss** $\mathcal{L}_\mathcal{P}$: MSE + $\lambda$ · adversarial loss ($\lambda=1.0$).

## Key Experimental Results

### Main Results: Acceleration Comparison on HunyuanVideo (VBench Evaluation)

| Method | Speedup | Peak VRAM | Semantic↑ | Quality↑ | Total↑ |
|--------|---------|-----------|-----------|----------|--------|
| Original 50-step (no CFG distill) | 1.0× | 99.2 GB | 73.5 | 81.5 | 79.9 |
| TeaCache (l=0.4) | 9.22× | 97.7 GB | 62.1 (−15.5%) | 78.7 | 75.4 |
| TaylorSeer (N=6, O=1) | 6.96× | 130.7 GB | 63.7 (−13.3%) | 79.9 | 76.7 |
| Restricted MeanFlow 9-step | 10.7× | 97.2 GB | 67.8 (−7.8%) | 81.0 | 78.4 |
| **DisCa (R=0.2, N=3)** | **8.84×** | **97.6 GB** | **70.3 (−4.4%)** | **81.8** | **79.5** |
| **DisCa (R=0.2, N=4)** | **11.8×** | **97.6 GB** | **69.3 (−5.7%)** | **81.1** | **78.8** |

- At 11.8× speedup, the total score drops by only 1.4%, semantic by 5.7%, and quality is nearly preserved (−0.5%).
- Compared to TaylorSeer, which loses 13.3% semantic score at a lower 6.96× speedup, DisCa performs substantially better at a higher compression ratio.
- DisCa's VRAM usage of 97.6 GB is significantly lower than TaylorSeer (130.7 GB) and FORA (124.6 GB).

### Ablation Study

| Configuration | Semantic | Quality | Total |
|---------------|----------|---------|-------|
| Full DisCa | 69.3 | 81.1 | 78.7 |
| w/o Restricted MeanFlow | 65.2 (−5.9%) | 80.3 | 77.3 |
| w/o Learnable Predictor (conventional cache) | 67.3 (−2.9%) | 80.5 | 77.9 |
| w/o GAN adversarial training | 68.5 (−1.2%) | 81.0 | 78.5 |

- Restricted MeanFlow contributes the most (removing it causes a 5.9% semantic drop), confirming that stable distillation is the foundation of quality preservation.
- The learnable predictor improves semantic score by 2.9% over conventional caching, and GAN training provides a further 1.2% gain.

## Highlights & Insights

- **Pioneering "learnable caching + distillation-compatible" paradigm**: Replacing handcrafted caching formulas with data-driven lightweight neural networks opens a new direction for feature caching research.
- **Restricted MeanFlow is simple yet effective**: A single restriction factor $\mathcal{R}$ suffices to clip aggressive compression scenarios and significantly stabilize the distillation of large-scale video models.
- **Memory efficient**: Only a single global cache tensor is maintained, and the predictor uses fewer than 4% of the full model's parameters, making it deployable in practical high-resolution long-video settings.
- **11.8× acceleration with near-lossless quality**: DisCa achieves the highest reported speedup on HunyuanVideo with a total score drop of only 1.4%, far surpassing all compared methods.

## Limitations & Future Work

- **Validated only on HunyuanVideo**: No experiments are conducted on other video diffusion models (e.g., CogVideoX, Wan), leaving generalizability to be verified.
- **Predictor requires additional training**: Although the parameter count is small, training (MSE + GAN) is still needed on the target model, making DisCa no longer a purely training-free approach.
- **Fixed caching interval $N$**: The number of steps between cache refreshes is fixed at inference time, lacking an adaptive dynamic scheduling mechanism.
- **GAN training stability**: Adversarial training is inherently prone to instability; while the paper reports smooth loss curves, robustness on other models and datasets remains unknown.
- **Manual tuning of $\mathcal{R}$ in Restricted MeanFlow**: The optimal $\mathcal{R}$ may vary across models and step counts, and no automatic selection strategy is provided.

## Related Work & Insights

| Category | Representative Methods | Characteristics | DisCa's Advantage |
|----------|----------------------|-----------------|-------------------|
| Direct cache reuse | Δ-DiT, PAB, FORA | Training-free, direct feature reuse | Completely collapses at high compression (semantic drop 20%+) |
| Adaptive caching | TeaCache, AdaCache | Adaptive decisions per timestep | Still limited by handcrafted strategies; large loss at high compression |
| Cache + prediction | TaylorSeer | Taylor-expansion feature prediction | Handcrafted function has limited capacity; high VRAM (130 GB) |
| Step distillation | MeanFlow, Shortcut | Compress sampling steps | Original MeanFlow is unstable on video models |
| **DisCa** | Ours | Learnable caching + restricted distillation | First to unify both paradigms; 11.8× near-lossless |

## Rating

- Novelty: ⭐⭐⭐⭐ — First to propose a learnable caching and distillation-compatible framework; Restricted MeanFlow is simple but effective
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dimensional VBench evaluation with complete ablations, though validated on a single model
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rich figures, and well-presented mathematical derivations
- Value: ⭐⭐⭐⭐⭐ — Introduces a highly practical new paradigm for video diffusion model acceleration; 11.8× speedup carries significant industrial value

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[CVPR 2026\] LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)
- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)

<!-- RELATED:END -->
