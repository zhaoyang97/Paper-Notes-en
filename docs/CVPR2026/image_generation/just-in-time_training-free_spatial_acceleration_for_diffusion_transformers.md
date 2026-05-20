---
title: >-
  [Paper Note] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer] This paper proposes the Just-in-Time (JiT) framework, which dynamically selects sparse anchor tokens in the spatial domain to drive generative ODE evolution…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Transformer"
  - "spatial acceleration"
  - "training-free"
  - "Flow Matching"
  - "token sparsification"
  - "ODE solving"
date: 2026-05-08
content_hash: a53a183a419c00b5
---

# Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers

**Conference**: CVPR 2026
**arXiv**: [2603.10744](https://arxiv.org/abs/2603.10744)  
**Code**: [Project Page](https://wenhao-sun77.github.io/JiT/)  
**Area**: Image Generation / Diffusion Model Acceleration
**Keywords**: Diffusion Transformer, spatial acceleration, training-free, Flow Matching, token sparsification, ODE solving

## TL;DR

This paper proposes the Just-in-Time (JiT) framework, which dynamically selects sparse anchor tokens in the spatial domain to drive generative ODE evolution, and introduces a deterministic micro-flow (DMF) mechanism to ensure seamless activation of newly included tokens. JiT achieves up to 7× acceleration on FLUX.1-dev with negligible quality degradation.

## Background & Motivation

1. **Computational bottleneck of DiTs**: The self-attention complexity of Diffusion Transformers is $\mathcal{O}(N^2)$, resulting in prohibitively high inference latency for high-resolution image and video generation, severely limiting real-time interaction and consumer-grade deployment.
2. **Limitations of temporal acceleration**: Existing acceleration methods primarily focus on the temporal domain (higher-order solvers, distilled few-step models), but suffer from significant quality degradation at ultra-low step counts, and distillation requires substantial retraining resources.
3. **Ceiling of caching methods**: Feature caching methods (TeaCache, TaylorSeer) reuse intermediate activations to reduce computation, but their quality upper bound is constrained by the baseline performance at the corresponding NFE, and they suffer from feature staleness.
4. **Overlooked spatial redundancy**: The diffusion generation process exhibits a progressive structure from low-frequency global layout to high-frequency details, yet existing methods apply uniform computation across all spatial regions — an unnecessary waste.
5. **Deficiencies of existing spatial methods**: Existing pyramid/hierarchical spatial acceleration methods rely on explicit upsampling and distribution correction, which readily introduce aliasing artifacts and information loss.
6. **Core insight**: Global structure is established in the early stages of generation; only a small number of key regions need to be computed to drive the evolution of the full latent state, while detail regions can be deferred.

## Method

### Overall Architecture

JiT is a training-free spatial-domain acceleration framework consisting of two core components:

- **SAG-ODE (Spatially Approximated Generative ODE)**: Computes the velocity field on sparse anchor tokens and extrapolates to the full spatial domain via an augmented lifting operator.
- **DMF (Deterministic Micro-Flow)**: At stage transitions, smoothly evolves newly activated tokens from their interpolated state to a statistically correct target state via a finite-time ODE.

### SAG-ODE Design

A nested chain of token subsets $\Omega_K \subset \Omega_{K-1} \subset \cdots \subset \Omega_0 = \{1,...,N\}$ is constructed, progressively expanding from the smallest subset.

The core equation is:

$$\frac{d\mathbf{y}(t)}{dt} = \mathbf{\Pi}_k \, \boldsymbol{u}_\theta(\mathbf{S}_k^\top \mathbf{y}(t), t)$$

where the augmented lifting operator $\mathbf{\Pi}_k$ consists of two components:
- **Embedding map** $\mathbf{S}_k \boldsymbol{u}_\theta$: places the exact velocities of anchor tokens back into their corresponding positions in the full space.
- **Interpolation operator** $\mathcal{I}_k(\boldsymbol{u}_\theta)$: provides spatial interpolation approximations for inactive tokens.

**Consistency guarantee**: $\mathbf{S}_k^\top(\mathbf{\Pi}_k \boldsymbol{u}_\theta) = \boldsymbol{u}_\theta$, meaning the dynamics of anchor tokens are exactly governed by the Transformer, so acceleration does not compromise quality in critical regions.

### DMF (Deterministic Micro-Flow)

At stage transitions, the target state for newly activated tokens is constructed as:

$$\mathbf{y}_k^\star = \mathbf{Q}_k \left( T_k \Phi_k(\mathbf{S}_k^\top \hat{\mathbf{y}}(1)) + (1-T_k)\epsilon \right)$$

Clean data is predicted via the Tweedie formula, combined with structural prior interpolation and the correct noise level, ensuring statistical consistency. A finite-time shooting ODE then drives newly activated tokens to precisely converge to the target within an extremely short interval.

### Importance-Guided Token Activation (ITA)

Rather than using a fixed grid pattern, an importance map is computed from the local variance of the velocity field:

$$\mathbf{I}(t) = \mathbb{E}_\mathcal{W}[\boldsymbol{u}_\theta \odot \boldsymbol{u}_\theta] - (\mathbb{E}_\mathcal{W}[\boldsymbol{u}_\theta])^{\odot 2}$$

Regions with the highest variance (most active during generation) are prioritized for activation, allocating compute to high-frequency detail regions.

## Key Experimental Results

### Main Results (FLUX.1-dev, Tab. 1)

| Method | NFE | Latency (s) | TFLOPs | Speedup | CLIP-IQA↑ | ImageReward↑ | HPSv2.1↑ | GenEval↑ | T2I-Comp↑ |
|--------|-----|-------------|--------|---------|-----------|-------------|----------|----------|-----------|
| FLUX.1-dev | 50 | 25.25 | 2991 | 1.0× | 0.6139 | 1.004 | 30.39 | 0.6565 | 0.4836 |
| TeaCache | 28 | 6.98 | 729 | 4.1× | 0.6003 | 0.964 | 29.68 | 0.6493 | 0.4849 |
| **JiT (Ours)** | 18 | **6.02** | **706** | **4.24×** | **0.6166** | **1.017** | 29.77 | **0.6540** | **0.4991** |
| TeaCache | 28 | 4.53 | 432 | 6.9× | 0.5183 | 0.773 | 27.86 | 0.5837 | 0.4625 |
| **JiT (Ours)** | 11 | **3.67** | **423** | **7.07×** | **0.5397** | **0.975** | **29.02** | **0.6457** | **0.4961** |

- At 4× speedup: JiT achieves top performance on CLIP-IQA, ImageReward, GenEval, and T2I-Comp, approaching the 50-NFE baseline.
- At 7× speedup: JiT substantially outperforms all competing methods, with ImageReward improving from 0.773 to 0.975.

### User Study

| Comparison | JiT Preference Rate |
|------------|-------------------|
| vs FLUX.1-dev (12 NFE) | 85.6% |
| vs Bottleneck (14 NFE) | 90.3% |
| vs FLUX.1-dev (7 NFE) | 93.1% |
| vs TaylorSeer (28 NFE) | 89.5% |

Twenty participants in 1,000 blind evaluations significantly preferred JiT-generated results.

### Ablation Study (T2I-CompBench complex compositions)

| Variant | HPSv2.1↑ | T2I-Comp↑ |
|---------|----------|-----------|
| Full JiT | **26.90** | **0.3727** |
| w/o SAG-ODE interpolation | 24.18 | 0.3414 |
| w/o ITA (fixed grid) | 26.51 | 0.3670 |
| w/o DMF target construction | 26.04 | 0.3602 |

Removing spatial interpolation causes catastrophic degradation (inactive regions degenerate to noise), validating the necessity of each component.

## Highlights & Insights

- **Fully training-free**: Directly applicable to pretrained DiT models without any retraining or fine-tuning.
- **Upsampling-free design**: Eliminates the dependence on explicit upsampling/downsampling found in conventional spatial acceleration methods, avoiding artifacts at the source.
- **Mathematical elegance**: SAG-ODE provides a consistency proof (anchor tokens are lossless); DMF offers rigorous convergence guarantees via shooting ODE.
- **Dynamic resource allocation**: ITA employs a content-aware strategy based on velocity field variance, which is more efficient than fixed patterns.
- **Quality preserved under extreme acceleration**: High-frequency details such as text rendering remain correct at 7× speedup, with advantages becoming more pronounced at the extreme end.

## Limitations & Future Work

- Validation is limited to a single model (FLUX.1-dev); generalization to other DiTs (SD3, PixArt, etc.) is not demonstrated.
- Stage scheduling ($\{T_k, m_k\}$) requires manual design; an adaptive scheduling mechanism is lacking.
- The interpolation operator $\mathcal{I}_k$ is relatively simple (spatial smooth interpolation), which may be insufficient for texture-rich regions.
- Only image generation is validated; extension to video generation (with far more tokens and potentially greater spatial redundancy) is not explored.
- The complementary potential of combining JiT with temporal acceleration methods (step distillation) — which are theoretically orthogonal — remains unexplored.
- The noise $\epsilon$ in DMF is resampled at each stage transition, which may introduce minor stochasticity.

## Related Work & Insights

| Category | Method | Comparison |
|----------|--------|------------|
| Spatial acceleration | RALU, Bottleneck Sampling | Rely on explicit upsampling and distribution correction, prone to artifacts; JiT uses an upsampling-free design. |
| Caching acceleration | TeaCache, TaylorSeer | Quality upper bound constrained by the low-NFE baseline; JiT is not subject to this limitation. |
| Subspace diffusion | Subspace Diffusion | Conceptually inspiring but restricted to low-dimensional subspaces; JiT's dynamic token subset operation is more flexible. |
| Pyramid methods | Pyramidal Flow | Progressive upsampling with correction; JiT achieves lossless dimensionality transitions via DMF. |

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of spatial sparse-token acceleration and micro-flow transitions is novel, with a clear mathematical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative, qualitative, user study, and ablation analyses are provided, though validation is limited to a single model.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clearly articulated, mathematical derivations are rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Training-free 7× acceleration offers high practical value, though generalizability and video extension remain to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[CVPR 2026\] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)

</div>

<!-- RELATED:END -->
