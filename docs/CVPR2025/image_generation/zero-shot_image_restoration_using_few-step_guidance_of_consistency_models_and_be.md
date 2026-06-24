---
title: >-
  [Paper Note] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)
description: >-
  [CVPR 2025][Image Generation][Zero-shot image restoration] CM4IR proposes a zero-shot image restoration scheme based on Consistency Models (CM). By combining a novel noise injection mechanism (decoupled denoising/injected noise levels + randomized/estimated noise splitting) with back-projection guidance and improved initialization, it surpasses existing diffusion model methods that require 20–1000 steps using only 4 neural network evaluations (NFEs).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Zero-shot image restoration"
  - "consistency models"
  - "noise injection"
  - "back-projection guidance"
  - "few-step inference"
date: 2026-05-08
content_hash: 9e03c4c5288bdffa
---

# Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)

**Conference**: CVPR 2025  
**arXiv**: [2412.20596](https://arxiv.org/abs/2412.20596)  
**Code**: [https://github.com/tirer-lab/CM4IR](https://github.com/tirer-lab/CM4IR)  
**Area**: Image Generation  
**Keywords**: Zero-shot image restoration, consistency models, noise injection, back-projection guidance, few-step inference

## TL;DR
CM4IR proposes a zero-shot image restoration scheme based on Consistency Models (CM). By combining a novel noise injection mechanism (decoupled denoising/injected noise levels + randomized/estimated noise splitting) with back-projection guidance and improved initialization, it surpasses existing diffusion model methods that require 20–1000 steps using only 4 neural network evaluations (NFEs).

## Background & Motivation

**Background**: Zero-shot image restoration (restoring images without training specialized networks for specific degradation models) has become a major trend. Methods based on Diffusion Models (DMs) utilize pre-trained denoising models as signal priors and add data fidelity guidance at inference time to restore images. However, existing methods require at least 20+ NFEs (Neural Function Evaluations), e.g., DPS requires 1000 NFEs, and DiffPIR requires 20 NFEs.

**Limitations of Prior Work**: The high number of NFEs results in slow inference, which fundamentally stems from the iterative generation process of diffusion models. Although Consistency Models (CMs) can generate images in 1–2 NFEs, existing CM-guided restoration methods (the original CM paper requires 40 NFEs, and CoSIGN requires per-task fine-tuning rather than being zero-shot) still have notable limitations.

**Key Challenge**: While CMs can perform unconditional generation in extremely few steps, existing noise injection and guidance mechanisms in guided restoration scenarios do not adapt well to the few-step nature of CMs, preventing further reduction of step counts.

**Goal**: Design a truly zero-shot restoration scheme that utilizes pre-trained CMs to perform high-quality super-resolution, deblurring, and inpainting in only 4 NFEs.

**Key Insight**: The authors identify three critical aspects: (1) In restoration tasks, the denoising noise level and the injected noise level do not need to be identical; (2) The injected noise can be split into a random component and an estimated component, with the latter acting similarly to Polyak acceleration; (3) Initializing with the pseudo-inverse of the observed data yields better results than starting with pure noise.

**Core Idea**: Design a noise injection mechanism adapted to the few-step characteristics of CMs. By decoupling noise levels to grant the denoiser more "freedom" and using anti-correlated estimated noise to accelerate sampling, 4-step zero-shot restoration is achieved.

## Method

### Overall Architecture
Given the degraded image $\mathbf{y} = \mathbf{A}\mathbf{x}^* + \mathbf{e}$, a pre-trained CM is used as a prior. The pipeline consists of: (1) Initialization using $\mathbf{A}^\dagger \mathbf{y}$ (e.g., bicubic upsampling for super-resolution) followed by adding noise $\tau_N$; (2) In each step, denoising is first performed using the CM to obtain $\mathbf{x}_{0|\tau_n}$, followed by applying back-projection guidance to correct data fidelity, and finally generating the next step input using the new noise injection mechanism. The entire process takes only $N=4$ steps (4 NFEs).

### Key Designs

1. **Decoupled Noise Levels**:

    - **Function**: Grants the denoiser greater flexibility to modify the input in restoration tasks.
    - **Mechanism**: For a signal with an injected noise level of $\tau_n$, the denoising level of the CM is set to $(1+\delta)\tau_n$ ($\delta \geq 0$). This effectively signals the denoiser that "the input has more noise than in reality," prompting more drastic modifications. This design is motivated by two reasons: (1) The guidance step introduces extra noise from $\mathbf{y}$; (2) The difference between the estimated and true signals is large in early iterations, requiring more aggressive denoising.
    - **Design Motivation**: Standard CM sampling matches denoising and injection, but restoration tasks introduce extra errors from guidance. Decoupling allows the denoiser to better compensate for these errors.

2. **Randomized-Estimated Noise Splitting**:

    - **Function**: Introduces directional momentum into noise injection to accelerate convergence.
    - **Mechanism**: The reverse direction of the estimated noise is defined as $\hat{\mathbf{z}}^- = (\mathbf{x}_{0|\tau_n} - \mathbf{x}_{\tau_n})/\tau_n$. The injected noise is then split into two parts: $\sqrt{1-\eta^2}\tau_{n-1}\hat{\mathbf{z}}^- + \eta\tau_{n-1}\mathbf{z}$, where $\eta \in [0,1]$ controls the ratio between randomized and estimated noise. Since $\hat{\mathbf{z}}^-$ points in the denoising direction, it acts as a "momentum" to push the sampling process.
    - **Design Motivation**: This splitting preserves the marginal distribution properties in the unguided setting (with theoretical guarantees) while serving as a "noisy version" of Polyak acceleration, which significantly reduces the required iterations.

3. **Back-projection Guidance + Data-aware Initialization**:

    - **Function**: Ensures the restored results align with the observed data.
    - **Mechanism**: Back-projection guidance $\nabla_{\mathbf{x}}\ell_{BP} = \mathbf{A}^\dagger(\mathbf{A}\mathbf{x} - \mathbf{y})$ is employed instead of standard least-squares guidance to speed up convergence. The initialization is set as $\mathbf{x}_{init} = \mathbf{A}^\dagger \mathbf{y}$ (median initialization is used for inpainting tasks), with $\tau_N < T$ to prevent initialization details from being completely washed out by noise.
    - **Design Motivation**: Back-projection guidance is theoretically and empirically proven to require fewer iterations than least-squares guidance, and data-aware initialization further reduces the steps required compared to starting from pure noise.

### Loss & Training
No training is required—the pre-trained CM is used plug-and-play as a prior. Hyperparameters include step count $N=4$, noise levels $\{\tau_n\}$, guidance scales $\{\mu_n\}$, decoupling parameter $\delta$, and noise splitting ratio $\eta$.

## Key Experimental Results

### Main Results (ImageNet 1K Val, 256×256)

| Task/Method | NFE | PSNR↑ | LPIPS↓ | FID↓ |
|----------|-----|-------|--------|------|
| **Super-resolution ×4 (bicubic, σ=0.05)** |  |  |  |  |
| DPS | 1000 | 23.79 | 0.335 | 58.56 |
| DiffPIR | 20 | 24.19 | 0.310 | 48.37 |
| DDRM | 20 | 25.19 | 0.282 | 39.07 |
| **CM4IR** (Ours) | **4** | **25.38** | **0.264** | **35.93** |
| **Deblurring (Gaussian, σ=0.025)** |  |  |  |  |
| DPS | 1000 | 24.39 | 0.296 | 50.33 |
| DiffPIR | 20 | 26.55 | 0.208 | 37.54 |
| **CM4IR** (Ours) | **4** | **27.15** | **0.193** | **32.87** |

### Ablation Study

| Configuration | PSNR (SR×4) | LPIPS (SR×4) |
|------|-------------|--------------|
| Without decoupling ($\delta=0$) | 24.72 | 0.289 |
| Without noise splitting ($\eta=1$) | 24.95 | 0.281 |
| Pure noise initialization | 24.31 | 0.301 |
| **Full CM4IR** | **25.38** | **0.264** |

### Key Findings
- Each component contributes significantly to the performance: noise decoupling (+0.66 PSNR), noise splitting (+0.43 PSNR), and data-aware initialization (+1.07 PSNR).
- With only 4 NFEs, the method outperforms DiffPIR and DDRM (which require 20 NFEs) across three tasks.
- The noise injection technique is transferable to DM methods: while DiffPIR exhibits severe degradation when dropping from 20 NFEs to 4 NFEs, incorporating the proposed noise injection significantly restores performance.
- The optimal value of $\delta$ positively correlates with the noise level—larger noise requires the denoiser to have more "freedom" for adjustment.

## Highlights & Insights
- **5× Speedup in Zero-Shot Restoration**: 4 NFEs vs. 20 NFEs with superior performance, offering significant value for practical deployment. CM4IR is currently the zero-shot restoration method with the fewest steps.
- **Theoretical Insights on Noise Injection**: Linking the reverse direction of estimated noise with Polyak acceleration provides concrete theoretical motivations rather than purely heuristic designs.
- **Generality Beyond CM**: The noise injection technique helps existing DM methods retain performance in low-NFE regimes, demonstrating that the improvement is not exclusive to CMs.

## Limitations & Future Work
- The current generation quality based on CMs still lags behind state-of-the-art diffusion models (there remains a gap in generative fidelity).
- Hyperparameters ($\delta$, $\eta$, $\{\mu_n\}$) require tuning for different degradation types and noise levels.
- The method is only verified on linear degradation models; non-linear degradation (e.g., JPEG compression) remains to be explored.
- Future progress in CM technology (e.g., iCT) is expected to further raise the performance boundary.

## Related Work & Insights
- **vs. DPS**: DPS uses LS guidance with 1000 NFEs, whereas CM4IR employs BP guidance with CM in 4 NFEs, achieving a 250× speedup with superior quality.
- **vs. CoSIGN**: CoSIGN also utilizes CMs but requires per-task tuning (non-zero-shot) and degrades when noise assumptions are mismatched. CM4IR is entirely zero-shot.
- The noise decoupling concept can be extended to other iterative generative approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of the noise injection mechanism has theoretical support and delivers significant performance gains.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three tasks, with comprehensive ablation and cross-method transfer verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations and well-defined motivations.
- Value: ⭐⭐⭐⭐ Significantly lowers the inference cost of zero-shot restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[ICCV 2025\] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](../../ICCV2025/image_generation/ld-rps_zero-shot_unified_image_restoration_via_latent_diffusion_recurrent_poster.md)
- [\[CVPR 2025\] Z-Magic: Zero-shot Multiple Attributes Guided Image Creator](z-magic_zero-shot_multiple_attributes_guided_image_creator.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)

</div>

<!-- RELATED:END -->
