---
title: >-
  [Paper Note] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models
description: >-
  [ICML2025][Image Generation][Diffusion model acceleration] The Morse dual-sampling framework is proposed, which learns residual feedback via a fast Dot model to compensate for the information loss in jump sampling of Dash (the original diffusion model), achieving 1.78×–3.31× lossless acceleration.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "dual-sampling framework"
  - "residual feedback"
  - "jump sampling"
  - "lossless acceleration"
date: 2026-05-08
content_hash: 7dace04f16fcaf58
---

# Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models

**Conference**: ICML2025  
**arXiv**: [2506.18251](https://arxiv.org/abs/2506.18251)  
**Code**: [deep-optimization/Morse](https://github.com/deep-optimization/Morse)  
**Authors**: Chao Li, Jiawei Fan, Anbang Yao  
**Area**: Image Generation  
**Keywords**: Diffusion model acceleration, dual-sampling framework, residual feedback, jump sampling, lossless acceleration

## TL;DR

The Morse dual-sampling framework is proposed, which learns residual feedback via a fast Dot model to compensate for the information loss in jump sampling of Dash (the original diffusion model), achieving 1.78×–3.31× lossless acceleration.

## Background & Motivation

Diffusion models (DMs) perform exceptionally well on tasks such as image generation and text-to-image generation, but the sampling process requires hundreds of iterative function evaluations, leading to massive inference overheads. Existing acceleration paths are divided into two categories:

**Improving Samplers**: DDIM, SDE/ODE solvers (DPM-Solver), etc., reduce the number of sampling steps through better step scheduling, but generation quality degrades significantly when the step count is too low.

**Knowledge Distillation**: Progressive Distillation, Consistency Distillation, etc., train student models to match teacher outputs with fewer steps, but usually at the cost of performance loss.

**Core Problem**: Given an arbitrary pre-trained diffusion model and an arbitrary sampler, can consistent acceleration be achieved across a wide range of step budgets (from a few steps to over a hundred steps) without sacrificing generation quality?

**Key Insight**: Mainstream DMs universally support **Jump Sampling (JS)**—only visiting a subsequence of time steps. JS makes sampling faster, but skipped steps cause information loss. The larger the step size, the more severe the quality degradation. If the information loss of JS can be compensated for efficiently, both speed and quality can be obtained simultaneously.

## Method

### Overall Architecture: Dash + Dot Dual-Model

Morse reformulates the iterative generation of a single model into the interaction of two models:

- **Dash**: The pre-trained DM to be accelerated, running in JS mode (jump sampling, reducing Dash invocations).
- **Dot**: A lightweight model that is $N$ times faster than Dash ($N \approx 5\text{–}10$). It is responsible for interpolating extra steps between adjacent Dash sampling points, compensating for JS information loss through residual feedback.

The two execute alternately over time, where Dash anchors and Dot fills.

### Core Formulae

Given a sequence of sampling steps $t_n > \cdots > t_0$, let $S$ be the set of steps assigned to Dash. The noise estimation for each step is:

$$
\mathbf{z}_{t_i} = \begin{cases}
\theta(\mathbf{x}_{t_i}, t_i) & t_i \in S \\
\mathbf{z}_{t_s} + \eta(\mathbf{x}_{t_s}, \mathbf{x}_{t_i}, \mathbf{z}_{t_s}, t_s, t_i) & t_i \notin S
\end{cases}
$$

- $\theta$: Dash model (pre-trained DM), which independently estimates noise.
- $\eta$: Dot model, which generates **residual feedback** based on observations at the current Dash trajectory point (input sample $\mathbf{x}_{t_s}$, output sample $\mathbf{x}_{t_i}$, noise estimation $\mathbf{z}_{t_s}$, and time steps $t_s, t_i$).
- Dot does not estimate noise independently; instead, it adds a residual correction on top of Dash's estimation $\mathbf{z}_{t_s}$ to approximate the estimation of Dash when not skipping steps.

### Acceleration Analysis

The standard process of $n$ steps requires $n$ LSD (Latency per Step of Dash). With Morse, within the same $n$ LSD budget, one can run $(n-k)$ steps of Dash + $Nk$ steps of Dot, totaling $(n - k + Nk)$ steps. The theoretical acceleration upper bound is:

$$\text{Speedup} = \frac{n - k + Nk}{n}$$

When $N = 5, k = n/2$, the theoretical acceleration is $3\times$.

### Weight Sharing and Dot Construction

The Dot model is not trained from scratch but is built based on a weight-sharing strategy from Dash:

1. Add $m$ lightweight down-sampling/up-sampling blocks (typically $m = 2$) to the top and bottom of Dash, reducing the input resolution by $4^m = 16$ times.
2. Share and freeze the pre-trained layer weights of Dash, training only the newly added blocks and LoRA modules.
3. Dot inherits most of the knowledge of Dash, making training extremely efficient.

### Training Objectives

The training objective of Dot is to make the residual feedback output $\mathbf{z}_{t_s} + \eta(\cdot)$ approximate the noise estimation $\theta(\mathbf{x}_{t_i}, t_i)$ of Dash at $t_{i}$ without jump sampling, which is a standard supervised regression loss.

## Training & Experimental Settings

- **Dot Training**: Follows the official training configuration of the corresponding Dash model, but with significantly reduced batch size and iteration steps.
- **Extra block count** $m = 2$, making Dot 5–10 times faster than Dash.
- **Hardware**: 8× NVIDIA RTX 3090.
- **Evaluation Metrics**: FID (primary), CLIP score (text-to-image).
- **Acceleration Metric**: LSD (Latency per Step of Dash) is defined as the time unit. Under the same FID, the LSD required with and without Morse is compared to calculate the average speedup ratio.
- **Stable Diffusion Experiment**: Dash is SD v1.4 (860M parameters), Dot has only 97.84M parameters. It is trained with 2M samples (only 0.1% of Dash's training data) and 190 A100-hours (only 0.1% of Dash's).

## Main Results

### Acceleration across Different Samplers (CIFAR-10)

| Sampler | Average Speedup |
|--------|---------|
| DDPM | 2.01× |
| DDIM | 2.94× |
| DPM-Solver (Discrete) | Significant Speedup |
| SDE | Consistent Speedup |
| DPM-Solver (SDE) | Consistent Speedup |

Morse is effective for both discrete and continuous-time methods and can even accelerate SOTA samplers that already utilize trajectory information, like DPM-Solver.

### Different Benchmarks (DDIM Sampler)

Approximate 2× speedup is obtained on CIFAR-10 (32²), ImageNet (64²), CelebA (64²), CelebA-HQ (256²), and LSUN-Church (256²). On CelebA, the speedup exceeds 4× under certain LSD budgets.

### Text-to-Image (Stable Diffusion)

| Method | FID@10 LSD | FID@50 LSD |
|------|-----------|-----------|
| SD (best scale) | 10.65 | 8.22 |
| SD + Morse (best scale) | 8.60 | 8.15 |

The average speedup is 2.29×, with the FID-CLIP curve outperforming the baseline comprehensively.

### Cross-Model Summary

Across 9 baseline diffusion models and 6 image generation tasks, the average lossless acceleration achieved is **1.78×–3.31×**.

### Accelerating LCM-SDXL (Distilled Model)

Morse can be stacked on top of Consistency Distillation to further accelerate LCM-SDXL, which is already accelerated, demonstrating complementarity with distillation methods.

## Ablation Study

- **Trajectory information** is key to the success of the Dot model—performance degrades significantly without trajectory information input.
- **Weight sharing** ensures that Dot inherits the generation knowledge of Dash while substantially reducing training costs.
- **LoRA fine-tuning** is more efficient than full fine-tuning with comparable results.

## Limitations & Future Work

1. **Requires training an additional Dot model**: Although the training cost is only ~0.1% of Dash, a separate Dot still needs to be trained for each new DM, limiting generalizability.
2. **Acceleration upper bound constrained by $N$**: The acceleration factor $N$ of Dot depends on the resolution down-sampling factor; setting it too high sacrifices accuracy.
3. **No coverage of modalities like video/3D**: Experiments focus on image generation, leaving validation on video or 3D generation to be addressed.
4. **GPU dependency**: The paper notes that $N$ may vary across different GPU configurations, so the actual speedup ratio may fluctuate in deployment.
5. **Limited evaluation metrics**: Mainly relies on FID and CLIP score, lacking human preference evaluation or fine-grained quality analysis.

## Reproducibility Key Points

- Code and models are open-sourced: [GitHub](https://github.com/deep-optimization/Morse)
- Extremely low training cost for Dot (190 A100-hours for SD scale), reproducible for typical academic labs.
- The framework is decoupled from the sampler, compatible with mainstream samplers such as DDPM/DDIM/DPM-Solver/SDE.
- Note that the choice of $m$ and $N$ needs to be adapted for specific architectures.

## Related Work & Insights

- **Relationship with DDIM/DPM-Solver**: Morse can be stacked on top of any fast sampler without conflict.
- **Relationship with Distillation Methods**: Distillation is "training a faster student," whereas Morse is "retaining the teacher + adding a lightweight assistant." The two are complementary and can be combined.
- **Residual Learning Concept**: Similar to ResNet's residual connection idea, Dot only needs to learn the "difference" instead of full noise estimation, lowering the learning difficulty.
- **Inspirations for Future Work**: This dual-model interaction paradigm can be extended to video diffusion, consistency models, and other scenarios.

## Personal Comments

The core contribution of this work is to break the dichotomy of "modifying samplers" or "distillation" for diffusion model acceleration, proposing a third path—dual-model collaboration. The design intuition of Dot as a lightweight residual compensator is exceptionally clean: since jump sampling loses information, use a fast model to compensate for it. The engineering design of weight sharing + LoRA is highly practical, reducing the training cost of Dot to only one-thousandth of Dash. The experiments cover 9 baseline models, 5 samplers, and 6 datasets, which are highly convincing. The main limitation is that the method still requires training a dedicated Dot for each new model; designing a universal Dot or a zero-shot transfer scheme would have even greater practical value.

## Rating
- Novelty: ⭐⭐⭐⭐ — The dual-sampling framework is novel, addressing JS information compensation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extensive validation across multiple models, samplers, and datasets.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with an engaging Dash/Dot analogy.
- Value: ⭐⭐⭐⭐ — Highly practical and complementary to existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SADA: Stability-guided Adaptive Diffusion Acceleration](sada_stability-guided_adaptive_diffusion_acceleration.md)
- [\[ICML 2026\] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](../../ICML2026/image_generation/speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)
- [\[ICML 2025\] Importance Sampling for Nonlinear Models](importance_sampling_for_nonlinear_models.md)
- [\[ICML 2025\] GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)

</div>

<!-- RELATED:END -->
