---
title: >-
  [Paper Note] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment
description: >-
  [CVPR 2025][Image Generation][DDIM Inversion] Proposes InPO (Inversion Preference Optimization), which simplifies preference optimization from a long Markovian process requiring a full denoising chain to a single-step optimization using a reparameterized DDIM inversion technique, outperforming existing Diffusion-DPO methods in both training efficiency and generation quality.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "DDIM Inversion"
  - "Preference Optimization"
  - "Diffusion Model Alignment"
  - "Reparameterization"
  - "Efficient Training"
date: 2026-05-08
content_hash: b24dfd2c18283b33
---

# InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Model Alignment / Diffusion Models  
**Keywords**: DDIM Inversion, Preference Optimization, Diffusion Model Alignment, Reparameterization, Efficient Training

## TL;DR
Proposes InPO (Inversion Preference Optimization), which simplifies preference optimization from a long Markovian process requiring a full denoising chain to a single-step optimization using a reparameterized DDIM inversion technique, outperforming existing Diffusion-DPO methods in both training efficiency and generation quality.

## Background & Motivation

### Background

**Background**: Applying DPO (Direct Preference Optimization) to diffusion models has become an important direction for improving text-to-image (T2I) generation quality. However, the multi-step sampling nature of diffusion models poses unique challenges for preference optimization.

**Limitations of Prior Work**:

### Limitations of Prior Work

**Limitations of Prior Work**: The generation process of diffusion models is a long-chain Markov process (typically 20-50 steps). DPO requires computing the log-probabilities of the entire chain, which incurs enormous computational overhead.

### Key Challenge

**Key Challenge**: Existing Diffusion-DPO methods require complete forward/backward passes on preference samples during training, leading to low training efficiency.

### Proposed Approach

**Approach**: Due to the long chain and complex dependencies, gradient signals suffer from severe attenuation during backpropagation, leading to unstable optimization.

### Supplementary Notes

**Supplementary Notes**: Limited improvement in generation quality — noise accumulation in the long chain hinders the effective propagation of preference signals.

**Key Challenge**: DPO requires computing the log-probabilities of the entire generation trajectory, but the long-chain sampling of diffusion models makes this computation both expensive and unstable.

**Goal**: How to efficiently compute the preference probability of diffusion model generation trajectories, avoiding the computational bottleneck of full long-chain sampling.

**Key Insight**: Utilizing the deterministic inversion property of DDIM to transform the long-chain probability computation "from noise to image" into an inversion "from image to noise" plus single-step optimization.

**Core Idea**: Use DDIM inversion to map known preference images back to the noise space, and directly perform preference comparison in the inverted noise space, avoiding full forward chain sampling.

## Method

### Overall Architecture
InPO training pipeline: (1) Invert the win/lose images in the preference data to the noise space using DDIM inversion to obtain corresponding latent noise pairs; (2) Calculate the step-wise preference probability in the noise space using the reparameterized DDIM formulation; (3) Optimize model parameters with the DPO loss, without requiring the model itself to perform the complete sampling process.

### Key Designs

1. **DDIM Inversion Reparameterization**:
    - **Function**: Maps preference images to initial noise via DDIM inversion, establishing a deterministic correspondence between images and noise.
    - **Mechanism**: The deterministic sampling of DDIM implies that an image uniquely corresponds to an initial noise. Through inversion, preference comparison can be performed in the noise space rather than the image space.
    - **Design Motivation**: Comparison in the noise space avoids the computational bottleneck of requiring a full generation chain to evaluate probabilities.

2. **Single-step Preference Loss**:
    - **Function**: Independently computes the preference loss at each timestep $t$, rather than requiring the joint probability of the entire chain.
    - **Mechanism**: Utilizes DDIM reparameterization to decompose the log-probability of the entire chain into independent contributions from each step, allowing each step to be optimized individually.
    - **Design Motivation**: Independent step-level optimization avoids the long-chain gradient attenuation problem, leading to more stable optimization.

3. **Efficient Training Strategy**:
    - **Function**: Substantially reduces training overhead via pre-computing inversion noise and random timestep sampling.
    - **Mechanism**: Inversion only needs to be performed once and cached; during training, timesteps are randomly sampled instead of traversing all steps, reducing the computational complexity from $O(T)$ to $O(1)$.
    - **Design Motivation**: Gradient computation of the full chain requires $O(T)$ forward passes, which random step sampling reduces to a constant level.

## Key Experimental Results

### Main Results

| Method | Training Efficiency (GPU hr) | Aesthetic Score | Prompt Alignment | Generation Diversity |
|------|------------------|---------|---------|-----------|
| Diffusion-DPO | High overhead | Moderate gain | Maintained | Low |
| D3PO | High overhead | Moderate gain | Maintained | Moderate |
| **InPO** | **Low overhead** | **Significant gain** | **Improved** | **Maintained** |

### Key Findings
- The training time of InPO is reduced by approximately 50-70% compared to standard Diffusion-DPO, primarily due to bypassing full-chain sampling.
- Inversion reparameterization provides cleaner preference signals, leading to greater improvements in generation quality.
- Single-step independent optimization effectively alleviates the long-chain gradient decay problem.
- Pre-computing inversion noise can be completed offline, without affecting online training efficiency.

## Highlights & Insights
- **Inversion = Training without Sampling**: Ingeniously bypassing the computational bottleneck of the full sampling chain by leveraging the determinism of DDIM inversion is an elegant mathematical insight.
- **Substantial Improvement in Training Efficiency**: Rather than reducing computation at the engineering level, it reduces the required computation from the algorithmic level.
- **Deep Integration with DDIM Properties**: The effectiveness of the method is built on the deterministic inversion property of DDIM, reflecting a profound understanding of the sampling mechanism of diffusion models.

## Limitations & Future Work
- Relies on the deterministic inversion of DDIM, making it inapplicable to stochastic sampler algorithms (e.g., DDPM, Euler Ancestral).
- DDIM inversion itself suffers from approximation errors, leading to inaccurate inversion when the number of steps is small.
- Only validated on text-to-image generation; scenarios such as image editing and video generation remain to be explored.
- Future work can explore more efficient inversion optimization methods by combining with novel samplers such as Consistency Models.

## Related Work & Insights

- **vs Diffusion-DPO**: Diffusion-DPO computes preference probabilities directly on the full denoising chain, resulting in heavy training overhead and weak gradient signals. InPO transforms the problem into single-step optimization in the noise space via inversion reparameterization, achieving a win-win in both efficiency and performance.
- **vs D3PO**: D3PO optimizes diffusion models using online RL, which requires online sampling and reward model evaluation, incurring even greater computational costs. InPO is completely offline, requiring no additional reward models.
- **vs SPO (Step-aware Preference Optimization)**: SPO also attempts step-level preference optimization, but still requires forward sampling to obtain the outputs of each step. InPO directly obtains latent variables at each step through inversion, avoiding forward sampling.
- InPO elegantly combines the determinism of DDIM inversion with DPO's preference learning. This paradigm can be transferred to other diffusion model alignment tasks such as video generation and image editing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of DDIM inversion and preference optimization is ingenious, but the core components (DDIM inversion, DPO) are existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Achieving SOTA in just 400 fine-tuning steps shows significant efficiency gains, but lacks large-scale human preference evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations and highly convincing motivation.
- **Value**: ⭐⭐⭐⭐ Significantly reduces the training cost of diffusion model alignment, exerting a direct impact on the practical deployment of T2I models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](../../ICML2025/image_generation/smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)
- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)

</div>

<!-- RELATED:END -->
