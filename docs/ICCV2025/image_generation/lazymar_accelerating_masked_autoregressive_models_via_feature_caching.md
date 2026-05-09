---
title: >-
  [Paper Note] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching
description: >-
  [ICCV 2025][Image Generation][MAR acceleration] LazyMAR addresses the inference efficiency bottleneck of Masked Autoregressive (MAR) models by exploiting two types of redundancy: token redundancy (most token features are highly similar across adjacent decoding steps) and condition redundancy (the residual between conditional and unconditional outputs in classifier-free guidance changes minimally between adjacent steps). Based on these observations, the paper proposes token cache and condition cache mechanisms, achieving a 2.83× speedup with negligible loss in generation quality.
tags:
  - ICCV 2025
  - Image Generation
  - MAR acceleration
  - feature caching
  - token redundancy
  - condition redundancy
  - plug-and-play
date: 2026-05-08
content_hash: d0ef989cc00ea0da
---

# LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching

**Conference**: ICCV 2025
**arXiv**: [2503.12450](https://arxiv.org/abs/2503.12450)
**Code**: [https://github.com/feihongyan1/LazyMAR](https://github.com/feihongyan1/LazyMAR)
**Area**: Image Generation
**Keywords**: MAR acceleration, feature caching, token redundancy, condition redundancy, plug-and-play

## TL;DR
LazyMAR addresses the inference efficiency bottleneck of Masked Autoregressive (MAR) models by exploiting two types of redundancy: token redundancy (most token features are highly similar across adjacent decoding steps) and condition redundancy (the residual between conditional and unconditional outputs in classifier-free guidance changes minimally between adjacent steps). Based on these observations, the paper proposes token cache and condition cache mechanisms, achieving a 2.83× speedup with negligible loss in generation quality.

## Background & Motivation

**State of the Field**: Autoregressive (AR) image generation models (e.g., VQGAN+Transformer) generate images token by token, but sequential dependencies limit efficiency. Masked Autoregressive (MAR) models (e.g., MaskGIT, MAR) accelerate generation by predicting multiple masked tokens in parallel, while maintaining or surpassing the generation quality of AR models.

**Limitations of Prior Work**: MAR models employ bidirectional attention, which allows parallel decoding of multiple tokens but requires simultaneous access to all tokens, making traditional KV Cache inapplicable. KV Cache works for AR models because, under causal attention, the KV representations of previously generated tokens remain unchanged. Under MAR's bidirectional attention, every token's representation may change as new tokens are decoded, rendering traditional KV Cache entirely ineffective. This leaves MAR's inference efficiency surprisingly below expectations.

**Root Cause**: The parallel decoding advantage of MAR is offset by cache invalidation under bidirectional attention, necessitating novel caching mechanisms tailored to MAR.

**Paper Goals**: Design caching mechanisms adapted to MAR models that substantially accelerate inference while preserving generation quality.

**Starting Point**: By analyzing computational redundancy in the MAR decoding process, the authors identify two key redundancies: (a) the features of most tokens remain nearly unchanged across adjacent decoding steps (token redundancy); (b) the residual between conditional and unconditional outputs in classifier-free guidance changes minimally between adjacent steps (condition redundancy).

**Core Idea**: Exploit token redundancy and condition redundancy in the MAR decoding process to design token cache and condition cache, enabling training-free, plug-and-play acceleration.

## Method

### Overall Architecture
LazyMAR introduces two caching mechanisms into MAR's iterative decoding process, combined with a periodic "cache-reuse-refresh" strategy. Full computation is performed in the initial steps to populate the cache. In subsequent steps, computation is skipped for similar tokens (token cache) and for the unconditional branch (condition cache). The cache is refreshed every $K$ steps to prevent error accumulation. The entire process requires no training and is plug-and-play.

### Key Designs

1. **Token Redundancy Analysis and Token Cache**:

    - Function: Cache and reuse features of tokens that remain stable across adjacent decoding steps.
    - Mechanism:
        - At each MAR decoding step, tokens can be categorized into four types (Fig. 1): currently decoded ($t$), decoded in the previous step ($t-1$), decoded in earlier steps ($<t-1$), and not yet decoded ($>t$).
        - Key finding: The features of tokens in the $<t-1$ and $>t$ categories exhibit near-zero cosine distance across adjacent steps (virtually unchanged), while only the $t-1$ tokens show significant variation.
        - Implementation: Full computation is performed in the initial steps and all token features are stored in the cache. In subsequent steps, all tokens are processed in the first 3 layers (as shallow-layer features change more). In deeper layers, cosine similarity between current features and cached features determines whether computation is skipped—tokens with high similarity reuse cached values directly.
        - Result: Approximately 84% of token computations can be skipped on average.
    - Design Motivation: Already-decoded stable tokens and undecoded mask tokens change negligibly between adjacent steps; recomputing them is wasteful.

2. **Condition Redundancy Analysis and Condition Cache**:

    - Function: Skip full computation of the unconditional branch in classifier-free guidance.
    - Mechanism:
        - Classifier-free guidance requires computing both conditional and unconditional outputs and mixing them, effectively doubling the computation cost.
        - Key finding (Fig. 2): Both conditional and unconditional outputs individually vary substantially across adjacent steps (large MSE), but their **residual** (conditional minus unconditional) changes minimally (approximately 35.5× smaller).
        - Implementation: The residual from the previous step is cached. In the current step, only the conditional branch is computed, and the cached residual approximates the unconditional branch: $uncond_t \approx cond_t - residual_{t-1}$.
        - Result: Skipping one branch reduces computation by 50%.
    - Design Motivation: Although both branches change in absolute terms, they change in highly consistent directions and magnitudes, leaving their difference nearly constant.

3. **Periodic Cache-Reuse-Refresh Strategy**:

    - Function: Periodically refresh the cache to prevent error accumulation.
    - Mechanism: Every $K$ decoding steps, the caching mechanisms are disabled and full computation is performed for all tokens and both branches, refreshing the cache with the computed results. This prevents approximation errors from accumulating indefinitely.
    - Design Motivation: Iterative cache reuse leads to exponential error amplification. Periodic refreshing keeps errors within an acceptable range.

### Loss & Training
- **No training required**: LazyMAR is entirely an inference-time acceleration strategy, involving no training or fine-tuning.
- Hyperparameters: cache refresh period $K$, cosine similarity threshold for skipping tokens, and the number of warmup steps at the beginning during which caching is disabled.

## Key Experimental Results

### Main Results

| Method | FID ↓ | IS ↑ | Speedup | Notes |
|--------|-------|------|---------|-------|
| MAR-Diffusion (original) | 1.55 | 303.7 | 1.0× | baseline |
| Direct step reduction | 1.88 | 289.2 | 2.83× | significant quality drop |
| **LazyMAR** | **1.65** | **302.9** | **2.83×** | FID increase of only 0.1 |

### Ablation Study

| Configuration | FID | IS | Speedup | Notes |
|--------------|-----|-----|---------|-------|
| Token Cache only | 1.58 | 303.2 | ~1.8× | token cache contributes primary speedup |
| Condition Cache only | 1.60 | 302.5 | ~1.5× | condition cache contributes 50% compute savings |
| Token + Condition | 1.65 | 302.9 | 2.83× | combined effect is multiplicative |
| w/o periodic refresh | 2.10 | 285.0 | higher | severe error accumulation without refresh |
| w/o warmup | 1.85 | 295.0 | slightly higher | full computation in initial steps is critical |

### Key Findings
- The speedup from token cache and condition cache is multiplicative: token cache skips 84% of token computations (~1.8×) and condition cache skips one branch (~1.5×), yielding a combined 2.83×.
- LazyMAR achieves a 2.83× speedup with only a 0.1 FID increase (1.55→1.65), whereas direct step reduction at the same speedup incurs a 0.33 FID increase.
- Periodic cache refreshing is critical for maintaining quality; removing it degrades FID from 1.65 to 2.10.
- Warmup steps with full computation are also essential, as these early steps determine the fundamental content of the generated image.
- LazyMAR generalizes to all MAR models (MaskGIT, MAGE, MAR-Diffusion, etc.) without model-specific adjustments.

## Highlights & Insights
- **Filling the MAR caching gap**: AR models have KV Cache and diffusion models have DeepCache/FasterDiffusion, but no caching acceleration scheme existed for MAR prior to this work. LazyMAR fills this gap.
- **Insightful discovery of condition redundancy**: While conditional and unconditional outputs individually vary substantially, their residual is remarkably stable—this finding is valuable in its own right and may generalize to all generative models using classifier-free guidance.
- **Training-free and plug-and-play**: No training or model modification is required, making the approach universally applicable to all MAR models with minimal deployment overhead.
- **Two orthogonal caches**: Token cache (spatial redundancy) and condition cache (guidance redundancy) are independent of each other, yielding multiplicative rather than additive speedup gains.

## Limitations & Future Work
- Although 2.83× is a meaningful speedup, MAR models typically use relatively few steps (~20), leaving limited room for absolute time savings.
- The similarity threshold requires tuning to balance speed and quality.
- Periodic refreshing introduces some computational overhead.
- Validation is currently limited to class-conditional generation; the redundancy characteristics under text-conditional generation may differ.
- Adaptive refresh schedules, rather than fixed periods, could be explored.

## Related Work & Insights
- **vs. KV Cache (AR models)**: KV Cache exploits the properties of causal attention and is inapplicable to MAR's bidirectional attention. LazyMAR designs a cache based on feature similarity that is compatible with bidirectional attention.
- **vs. DeepCache (diffusion models)**: DeepCache caches the output of UNet upsampling blocks and is tailored to the UNet architecture. LazyMAR is designed specifically for Transformer-based MAR models.
- **vs. direct step reduction**: Step reduction is the simplest acceleration strategy but incurs significant quality loss. LazyMAR achieves substantially better quality at the same speedup ratio.

## Rating
- Novelty: ⭐⭐⭐⭐ The identification of token and condition redundancy is insightful, though the method fundamentally follows the standard paradigm of feature caching.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are comprehensive, with comparisons across multiple baselines and MAR model variants.
- Writing Quality: ⭐⭐⭐⭐⭐ Analytical figures (Fig. 1, 2) are highly intuitive and the motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ Fills the MAR caching gap with strong practical utility as a plug-and-play solution.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[ICCV 2025\] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)

<!-- RELATED:END -->
