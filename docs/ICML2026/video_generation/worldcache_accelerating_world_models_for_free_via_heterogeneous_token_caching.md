---
title: >-
  [Paper Note] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching
description: >-
  [ICML 2026][Video Generation][Diffusion World Model] WorldCache addresses the issue of non-uniform evolution of multi-modal tokens (e.g., RGB/depth) in diffusion world models. By categorizing tokens into stable, linear…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Diffusion World Model"
  - "Feature Caching"
  - "Heterogeneous Token"
  - "Adaptive Skipping"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: f03ba873d1265625
---

# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

**Conference**: ICML 2026  
**arXiv**: [2603.06331](https://arxiv.org/abs/2603.06331)  
**Code**: https://github.com/FofGofx/WorldCache  
**Area**: Video Generation / World Model Acceleration  
**Keywords**: Diffusion World Model, Feature Caching, Heterogeneous Token, Adaptive Skipping, Inference Acceleration  

## TL;DR
WorldCache addresses the issue of non-uniform evolution of multi-modal tokens (e.g., RGB/depth) in diffusion world models. By categorizing tokens into stable, linear, and chaotic types based on curvature and adaptively triggering full forward passes, it achieves up to 3.65x to 3.7x end-to-end acceleration on models like HunyuanVoyager and Aether while maintaining the quality of world generation and 3D reconstruction.

## Background & Motivation
**Background**: Generative world models are evolving from pure video generation toward environmental dynamics simulation. Typical inputs include images, text, and camera trajectories, while outputs simultaneously contain RGB video, depth, or geometric information. Many high-quality world models are based on diffusion transformers, requiring dozens to hundreds of denoising steps. Each step calls a large backbone, leading to high costs for interactive use and long-horizon rollouts.

**Limitations of Prior Work**: Training-free feature caching is an attractive direction for diffusion model acceleration because it requires no retraining, simply reusing or predicting intermediate features during sampling. However, most caching strategies originate from single-modal image/video diffusion, assuming homogeneous token dynamics. When directly migrated to world models, token trajectories for RGB vs. depth, different spatial regions, and motion boundaries vary significantly. Uniform reuse, linear extrapolation, or fixed skipping intervals easily accumulate errors.

**Key Challenge**: Most tokens in a world model are smooth across many denoising steps and suitable for aggressive caching; however, a few key tokens exhibit non-linear directional mutations, which often determine whether geometric boundaries, depth discontinuities, and motion structures collapse. Global conservative strategies are slowed down by a few difficult tokens, while global aggressive strategies cause these difficult tokens to drift.

**Goal**: The authors aim to preserve the training-free advantage of feature caching while enabling the caching strategy to understand the token heterogeneity and temporal non-stationarity of world models. Specifically, the goal is to reduce sampling latency for diffusion world models like HunyuanVoyager and Aether on a single GPU while maintaining the quality of multi-modal rollouts (RGB/depth/camera pose).

**Key Insight**: The paper treats the denoising output of each token as a temporal trajectory, using the last three full forward passes to estimate velocity, acceleration, and curvature. Smaller curvature indicates a near-linear trajectory, while larger curvature signifies violent local directional changes, necessitating more cautious prediction and timely full recomputation.

**Core Idea**: Curvature-driven token-level heterogeneous prediction is combined with monitoring the normalized drift of only chaotic tokens to decide when to re-invoke the full backbone. This concentrates computational power on the few tokens and difficult time intervals that truly cause rollout collapse.

## Method

### Overall Architecture
During the diffusion sampling process, WorldCache alternates between two types of steps: FULL and CACHE. A FULL step normally calls the world model backbone to obtain token-space outputs and stores the last three FULL outputs in a history buffer. When sufficient history is available, the method calculates the curvature for each token and refreshes stable, linear, and chaotic masks. A CACHE step skips the backbone call, instead using different predictors based on token type to generate surrogate outputs, which are then passed to the original diffusion scheduler to update the latent.

Unlike conventional fixed-interval caching, WorldCache maintains an accumulated error signal $E_{acc}$. After each CACHE step, it calculates curvature-normalized feature drift only on the set of chaotic tokens and adds it to $E_{acc}$. When $E_{acc}$ exceeds a threshold $\eta$, the next step switches back to FULL to recalibrate curvature and historical outputs.

### Key Designs
1. **Curvature-guided Heterogeneous Token Prediction**:

	- **Function**: Assigns different caching rules based on token trajectory predictability, avoiding the use of a single predictor for all tokens.
	- **Mechanism**: For token $i$, velocity $v$ and acceleration $a$ are calculated using the last three FULL outputs. Curvature is defined as $\kappa_i=\|a_i\|_2/(\|v_i\|_2^2+\epsilon)$. Tokens are then partitioned into three groups by curvature quantiles: low-curvature tokens are directly reused, high-curvature tokens enter the chaotic group, and intermediate tokens use linear extrapolation.
	- **Design Motivation**: Token dynamics in world models are long-tailed. Most background or smooth regions do not warrant recomputation every step, whereas a few tokens related to boundaries, depth changes, and motion require finer control. Curvature reflects whether a token can be "predicted by the first order" better than raw feature differences.

2. **Damped update for chaotic tokens**:

	- **Function**: Reduces directional drift caused by single-step linear extrapolation in high-curvature tokens.
	- **Mechanism**: Stable tokens use $\tilde{y}_{t,i}=y_{t^\star,i}$, and linear tokens use $\tilde{y}_{t,i}=y_{t^\star,i}+k v_{t^\star,i}$. Chaotic tokens utilize a smoothstep weight to mix the current velocity with the previous FULL velocity: $v_i^{adapt}(k)=(1-\alpha_k)v_{t^\star,i}+\alpha_k v_{t^\star-1,i}$, becoming increasingly conservative as continuous caching steps increase.
	- **Design Motivation**: The latest tangent direction of high-curvature tokens may not be reliable. The longer the continuous skipping, the more likely they are to diverge along the wrong direction. Damped updates stabilize predictions with historical velocity, preserving some dynamics while reducing error accumulation in long cache streaks.

3. **Chaotic-prioritized Adaptive Skipping**:

	- **Function**: Determines when to switch from CACHE back to FULL, avoiding the non-robustness of fixed skipping across different time stages and modal scales.
	- **Mechanism**: The method constructs a dimensionless drift $e_i(t)=\kappa_i\|\tilde{y}_{t,i}-\tilde{y}_{t+1,i}\|_2$, averages it only over chaotic tokens to get $E(t)$, and accumulates it into $E_{acc}$. When $E_{acc}\geq\eta$, it indicates that the uncertainty of difficult tokens has accumulated enough to require correction by the full backbone.
	- **Design Motivation**: Feature norms in world models vary significantly across modalities and timesteps, making raw norms or global difference thresholds difficult to unify. Curvature-multiplied drift offsets global scale changes, while monitoring only chaotic tokens prevents easy tokens from diluting real risk signals.

### Loss & Training
WorldCache does not train the model nor change the diffusion scheduler. It is a pure inference-time caching strategy: each FULL step updates historical outputs and curvature groupings, and each CACHE step replaces backbone output with token-wise surrogates. Core hyperparameters include stable/chaotic quantile thresholds $(p_s,p_c)$, maximum cache streak $n_{max}$, and the CAS threshold $\eta$. Main experiments use stable quantile combinations and $\eta=0.20$ on Aether as a trade-off between speed and quality.

## Key Experimental Results

### Main Results
The main experiments cover image-to-world generation for HunyuanVoyager-13B, image-to-world generation for Aether-5B, and 3D reconstruction for Aether. The table below selects rows that best illustrate the speed-quality trade-off.

| Model/Task | Method | WorldScore Static | WorldScore Dynamic | PSNR | SSIM | LPIPS | Latency | Speed | Memory |
|-----------|------|-------------------|--------------------|------|------|-------|---------|-------|--------|
| HunyuanVoyager-13B world gen | Original | 66.28 | 46.40 | ∞ | 1.000 | 0.000 | 1053.7s | 1.00x | 50.44GB |
| HunyuanVoyager-13B world gen | TeaCache | 60.88 | 42.61 | 16.25 | 0.565 | 0.372 | 311.5s | 3.38x | 56.52GB |
| HunyuanVoyager-13B world gen | EasyCache | 64.16 | 44.91 | 21.76 | 0.737 | 0.208 | 294.5s | 3.58x | 50.98GB |
| HunyuanVoyager-13B world gen | WorldCache | 64.89 | 45.43 | 23.49 | 0.770 | 0.176 | 288.6s | 3.65x | 50.58GB |
| Aether-5B world generation | Original | 64.60 | 45.22 | ∞ | 1.000 | 0.000 | 179.7s | 1.00x | 46.58GB |
| Aether-5B world generation | EasyCache | 62.89 | 44.02 | 22.84 | 0.720 | 0.186 | 120.9s | 1.49x | 46.59GB |
| Aether-5B world generation | WorldCache | 63.68 | 44.72 | 31.87 | 0.924 | 0.066 | 107.2s | 1.68x | 46.59GB |

| 3D reconstruction Method | Abs Rel | $\delta<1.25$ | $\delta<1.25^2$ | ATE | RPE trans | RPE rot | Latency | Speed | Memory |
|------------------------|---------|--------|----------|-----|-----------|---------|---------|-------|--------|
| Aether Original | 0.340 | 0.502 | 0.738 | 0.177 | 0.068 | 0.780 | 55.42s | 1.00x | 50.19GB |
| TeaCache | 0.341 | 0.496 | 0.724 | 0.183 | 0.068 | 0.797 | 25.85s | 2.14x | 50.20GB |
| HERO | 0.347 | 0.490 | 0.716 | 0.181 | 0.071 | 0.861 | 27.44s | 1.96x | 61.56GB |
| WorldCache | 0.341 | 0.508 | 0.741 | 0.184 | 0.068 | 0.796 | 21.20s | 2.61x | 50.20GB |

### Ablation Study
Ablations primarily verify that token prediction must be heterogeneous by curvature and that skip triggering must focus on the normalized drift of chaotic tokens.

| Token Prediction Strategy | PSNR | SSIM | LPIPS | Latency | Note |
|----------------|------|------|-------|---------|------|
| Reuse | 22.74 | 0.714 | 0.336 | 86.32s | Cheap but fails on changing tokens |
| Linear | 18.01 | 0.537 | 0.396 | 87.07s | Linear extrapolation fails in high-curvature regions |
| Damped | 23.76 | 0.665 | 0.276 | 87.51s | Too conservative for easy tokens |
| Random Group | 22.59 | 0.710 | 0.314 | 86.98s | Diverse operations alone do not yield gains |
| CHTP | 25.76 | 0.791 | 0.227 | 86.94s | Curvature grouping provides best fidelity |

| Skipping Strategy | PSNR | SSIM | LPIPS | Note |
|----------|------|------|-------|------|
| Fixed Interval | 26.18 | 0.830 | 0.216 | Cannot adapt to difficult time periods |
| Difference Guided | 26.79 | 0.824 | 0.207 | Raw difference affected by scale |
| Norm Guided | 26.02 | 0.809 | 0.217 | Norm thresholds unstable across modalities |
| Curvature Guided | 25.87 | 0.788 | 0.236 | Looking only at difficulty causes over-recomputation |
| CAS | 27.10 | 0.881 | 0.198 | Combining curvature and actual drift is most stable |

### Key Findings
- On HunyuanVoyager, WorldCache is slightly faster than EasyCache, while PSNR improves from 21.76 to 23.49 and WorldScore Dynamic improves from 44.91 to 45.43, proving it does not gain speed by aggressively sacrificing quality.
- On Aether world generation, WorldCache's PSNR/SSIM/LPIPS significantly outperform other acceleration methods with almost no memory increase; this is critical for single-GPU resource-constrained scenarios.
- 3D reconstruction results show the caching strategy does not damage geometric capabilities. WorldCache's depth accuracy $\delta<1.25$ and $\delta<1.25^2$ are even slightly higher than the original model's measured values, and pose metrics are close to the no-cache baseline.
- Ablations of CHTP and CAS separately demonstrate the necessity of the design: a uniform predictor makes opposite errors on different token types, and global drift or curvature alone is inferior to the "difficult tokens + actual drift" combination.

## Highlights & Insights
- The paper advances diffusion caching from "which steps to skip" to "which tokens can be safely predicted in which steps." This granularity is better suited for multi-modal world models because real difficulties are often concentrated in small regions and specific physical structures.
- Curvature is a natural bridge: it explains why token prediction error increases with second-order changes and can be used to construct scale-normalized drift metrics. This design has more physical meaning than manually tuning raw norm thresholds.
- WorldCache is training-free and model-level, independent of modifying weights for HunyuanVoyager or Aether. For large world model deployment, this "plug-in" acceleration is more practical than retraining.
- Experiments go beyond visual metrics to include WorldScore, depth, camera pose, memory, and latency. Accelerating world models by looking only at single perceptual metrics can easily overlook geometric drift; this evaluation dimension is closer to the task's essence.

## Limitations & Future Work
- The method requires the last three FULL outputs to stably estimate curvature, so early steps still require full computation; gains may be limited for extremely short sampling processes or models with very few steps.
- While curvature and drift thresholds are ablated, they remain manual hyperparameters. Whether $p_s, p_c, \eta$ can be set automatically across different world models, resolutions, frame counts, or motion conditions requires further study.
- The paper primarily evaluates RGB/depth-coupled diffusion world models. For world simulators containing interactive actions, language feedback, occupancy grids, or multi-agent states, token heterogeneity may be more complex.
- Cache surrogates still approximate backbone outputs in feature space without directly constraining final physical consistency. Future work could integrate geometric constraints, camera trajectory errors, or 3D consistency signals into the trigger strategy.

## Related Work & Insights
- **vs TeaCache / EasyCache**: These model-wise caching methods use global signals for skipping, which are simple and low-memory; WorldCache further distinguishes token types and monitors only chaotic tokens, leading to a better quality-speed trade-off.
- **vs DuCa / ToCa / HiCache / TaylorSeer**: Layer-wise caching may save more intermediate features but incurs high memory overhead on world models, potentially requiring CPU offloading; WorldCache is a model-level surrogate with near-zero extra memory.
- **vs token-adaptive diffusion caching**: Existing token selection methods are mostly for single-modal image/video; WorldCache's novelty lies in linking token difficulty with the multi-modal/geometric dynamics of world models.
- **vs HERO**: HERO combines caching with token merging, which may accelerate but also damage details; WorldCache does not merge tokens but predicts token-space outputs, making it more suitable for preserving boundaries and depth structures.
- **Insight**: Future multi-modal generation acceleration could utilize "local dynamics" signals more often rather than relying solely on attention or global feature differences. For video generation, 3D reconstruction, and robot rollouts, state evolution metrics like curvature/drift are worth exploring.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Curvature-driven heterogeneous caching and chaotic-prioritized triggers fit world models well with clear logic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two main world models, world generation, 3D reconstruction, visual comparisons, component ablations, and extensive appendix evaluations.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and method are well-explained; symbols are somewhat numerous, but the overall structure is smooth.
- Value: ⭐⭐⭐⭐⭐ Very helpful for the practical inference cost of diffusion world models, especially in single-GPU and long-rollout scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](../../CVPR2026/video_generation/disca_accelerating_video_diffusion_transformers_wi.md)
- [\[ICML 2026\] Exploring Data-Free LoRA Transferability for Video Diffusion Models](exploring_data-free_lora_transferability_for_video_diffusion_models.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](../../NeurIPS2025/video_generation/training-free_efficient_video_generation_via_dynamic_token_carving.md)

</div>

<!-- RELATED:END -->
