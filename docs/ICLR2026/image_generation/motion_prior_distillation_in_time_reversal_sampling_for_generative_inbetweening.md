---
title: >-
  [Paper Note] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening
description: >-
  [ICLR 2026][Image Generation][Generative Inbetweening] This paper proposes Motion Prior Distillation (MPD), an inference-time distillation method that distills motion residuals from the forward path into the backward path, fundamentally resolving the bidirectional motion prior conflict in time reversal sampling. MPD enables more coherent generative inbetweening without any additional training.
tags:
  - ICLR 2026
  - Image Generation
  - Generative Inbetweening
  - Motion Prior
  - Time Reversal Sampling
  - Inference-Time Distillation
  - SVD
date: 2026-05-08
content_hash: e2ef102cd542f2cc
---

# Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening

**Conference**: ICLR 2026
**arXiv**: [2602.12679](https://arxiv.org/abs/2602.12679)
**Code**: [https://vvsjeon.github.io/MPD/](https://vvsjeon.github.io/MPD/)
**Area**: Diffusion Models / Video Generation
**Keywords**: Generative Inbetweening, Motion Prior, Time Reversal Sampling, Inference-Time Distillation, SVD

## TL;DR

This paper proposes Motion Prior Distillation (MPD), an inference-time distillation method that distills motion residuals from the forward path into the backward path, fundamentally resolving the bidirectional motion prior conflict in time reversal sampling. MPD enables more coherent generative inbetweening without any additional training.

## Background & Motivation

**Background**: Advances in I2V diffusion models have extended frame interpolation beyond traditional optical flow methods to semantic-level "generative inbetweening."

**Mechanism of Time Reversal Sampling**: Start and end frames are used to condition forward and backward denoising paths respectively, whose intermediate frames are then merged.

**Motion Prior Conflict**: I2V models are trained to generate consecutive frames forward from a single frame. When conditioned on the end frame and run in reverse, the model still tends to "look forward" rather than trace back to earlier frames, introducing a forward generation bias.

**Off-Manifold Issue in Parallel Methods**: Linearly interpolating the two paths (e.g., TRF) pushes samples off the learned data manifold, causing oscillations and artifacts.

**Persistent Conflict in Sequential Methods**: Alternating denoising of forward and backward paths (e.g., ViBiDSampler) preserves manifold consistency but still suffers from conflicting motion priors between the two paths.

**Observed Symptoms**: These conflicts manifest as ghosting artifacts, reversed playback, and object disappearance, severely degrading generation quality.

## Method

### Overall Architecture

MPD augments the standard time reversal sampling framework with motion residual distillation for single-path alignment: during early denoising steps, only forward-path motion information is used, deliberately avoiding independent denoising of the backward path, thereby eliminating motion prior conflicts.

### Key Designs

#### Core Insight: Motion Residuals

The inter-frame residual estimated during forward denoising encodes motion information:

$$\Delta \hat{x}_{0,c_{\text{start}}}^{(i)} := \hat{x}_{0,c_{\text{start}}}^{(i)} - \hat{x}_{0,c_{\text{start}}}^{(i-1)}$$

The forward noise residual is defined as:

$$\Delta \epsilon_{\text{fwd}} = \frac{\Delta x_t - \Delta \hat{x}_{0,c_{\text{start}}}}{\sigma_t}$$

#### Key Step: Backward Path Reconstruction

1. **Initialization**: Initialize the first frame of the backward noise using the end frame $z_{\text{end}}$: $\epsilon_{\text{bwd}}^{(1)} = \frac{(x_t')^{(1)} - z_{\text{end}}}{\sigma_t}$

2. **Cumulative Distillation**: Reconstruct backward noise by cumulatively subtracting forward noise residuals:

$$\epsilon_{\text{bwd}}^{(i)} = \epsilon_{\text{bwd}}^{(1)} - \sum_{k=2}^{i} \Delta \epsilon_{\text{fwd}}^{(k)}$$

3. **Reconstruct the backward denoising estimate** (without $c_{\text{end}}$ conditioning):

$$\hat{x}_{0,c_{\text{start}}^*}' = x_t - \sigma_t \epsilon_{\text{bwd}}$$

4. **Fusion and Update**:

$$\tilde{x}_{0,c_{\text{start}}} = (1-\lambda) \hat{x}_{0,c_{\text{start}}} + \lambda (\hat{x}_{0,c_{\text{start}}^*}')'$$

$$x_{t-1} = \tilde{x}_{0,c_{\text{start}}} + \frac{\sigma_{t-1}}{\sigma_t}(x_t - \hat{x}_{0,\varnothing})$$

### Loss & Training

**Phased Application Strategy**:
- **Early steps** ($t > (1-\gamma)T$): Apply MPD with re-noising, imposing distillation during the global motion trajectory formation phase.
- **Late steps**: Revert to standard time reversal sampling (TRF or ViBiD) to enhance endpoint consistency and fine details.

**Loss Perspective**: MPD simplifies the original dual-path optimization objective:

$$\mathcal{L} = \frac{1}{\sigma_t^2} \|\hat{x}_{0,c_{\text{start}}} - (\hat{x}_{0,c_{\text{end}}}')\|_2^2$$

to a single-path objective:

$$\mathcal{L} = \frac{1}{\sigma_t^2} \|\hat{x}_{0,c_{\text{start}}} - (\hat{x}_{0,c_{\text{start}}^*}')\|_2^2$$

thereby avoiding the introduction of an independent end-frame motion prior.

## Key Experimental Results

### Main Results

**DAVIS Dataset**

| Method | LPIPS ↓ | FID ↓ | FVD ↓ | VBench ↑ | VBench++ ↑ |
|--------|---------|-------|-------|----------|------------|
| TRF | 0.3127 | 56.894 | 674.31 | 0.7618 | 0.9352 |
| GI | 0.2432 | 48.427 | 654.91 | 0.7747 | 0.9320 |
| FCVG | 0.2347 | 38.997 | 621.82 | 0.7904 | 0.9353 |
| ViBiD | 0.2492 | 39.883 | 559.49 | 0.7733 | 0.9387 |
| **Ours + TRF** | **0.2212** | **34.910** | 612.17 | **0.7992** | 0.9330 |
| **Ours + ViBiD** | 0.2220 | 37.241 | **527.05** | 0.7845 | **0.9474** |

**Pexels Dataset**

| Method | LPIPS ↓ | FID ↓ | FVD ↓ | VBench++ ↑ |
|--------|---------|-------|-------|------------|
| FCVG | 0.1160 | 35.269 | 525.08 | 0.9701 |
| **Ours + TRF** | 0.1149 | **34.470** | **460.99** | **0.9862** |
| **Ours + ViBiD** | **0.1028** | 34.775 | 412.66 | 0.9605 |

**User Study (30 Participants)**

| Method | Alignment Score ↑ | Artifact Rate ↓ | Unrealistic Motion ↓ |
|--------|-------------------|-----------------|----------------------|
| TRF | -0.3119 | 28.09% | 25.24% |
| ViBiD | -0.0678 | 28.10% | 25.24% |
| **Ours + TRF** | **0.3060** | 20.36% | 22.62% |
| **Ours + ViBiD** | 0.2440 | **8.93%** | **9.88%** |

## Highlights & Insights

1. **Precise Problem Identification**: The paper identifies motion prior conflict as the fundamental issue in time reversal sampling, rather than attributing it to path fusion strategies alone.
2. **Single-Path Design**: By deliberately avoiding denoising the backward path and instead reconstructing it via motion residual distillation, the second motion prior is completely eliminated.
3. **Plug-and-Play**: MPD can be directly applied on top of existing methods such as TRF (parallel) or ViBiD (sequential).
4. **Training-Free**: A purely inference-time method that runs on a single RTX 4090.
5. **Compelling User Study**: Artifact detection rate is reduced to 8.93% and unrealistic motion to 9.88%, substantially outperforming all baselines.

## Limitations & Future Work

1. Validated only on SVD; not extended to more recent video models such as Wan or CogVideoX.
2. The distillation ratio $\gamma$, re-noising step count $k$, and interpolation scale $\lambda$ require manual tuning.
3. The motion residual assumption may break down under large motions (e.g., scene cuts) or non-rigid deformations.
4. The number of generated frames is constrained by SVD (14–25 frames); long-video settings are unexplored.
5. Computational overhead: MPD requires additional forward passes during early steps to compute residuals.

## Related Work & Insights

- **TRF (Feng et al.)**: The first time reversal sampling method (parallel fusion); MPD can be directly stacked on top for improvement.
- **ViBiDSampler (Yang et al.)**: Sequential time reversal sampling with CFG++; MPD is equally compatible.
- **GI (Wang et al.)**: Fine-tunes reverse motion by rotating temporal self-attention, requiring training; MPD is training-free.
- **FCVG (Zhu et al.)**: Injects line correspondences as frame-level conditions, but the fundamental motion prior conflict remains unresolved.
- **Insight**: The inter-frame residuals of diffusion model denoising estimates carry rich motion semantics and may be applicable to other temporal consistency tasks such as video editing and video inpainting.

## Rating

- Novelty: ⭐⭐⭐⭐ — The motion residual distillation concept is original and well-analyzed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Quantitative evaluation, user study, and ablation study together provide strong evidence.
- Writing Quality: ⭐⭐⭐⭐ — Derivations are complete, figures are clear, and problem motivation is well articulated.
- Value: ⭐⭐⭐⭐ — Directly advances the generative inbetweening field with high practical utility due to its training-free nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[AAAI 2026\] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](../../AAAI2026/image_generation/pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)

</div>

<!-- RELATED:END -->
