---
title: >-
  [Paper Note] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution
description: >-
  [CVPR 2025][Image Generation][image super-resolution] This paper proposes FaithDiff, which unleashes (fine-tunes) pre-trained diffusion model priors for image super-resolution for the first time. It designs an alignment module to bridge degraded image features and diffusion latent noise space, achieving faithful structural restoration through joint optimization of the encoder and diffusion model.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "image super-resolution"
  - "diffusion prior"
  - "alignment module"
  - "joint fine-tuning"
  - "faithful restoration"
date: 2026-05-08
content_hash: 0e7d7222a0607709
---

# FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2411.18824](https://arxiv.org/abs/2411.18824)  
**Code**: [https://jychen9811.github.io/FaithDiff_page/](https://jychen9811.github.io/FaithDiff_page/)  
**Area**: Image Generation  
**Keywords**: image super-resolution, diffusion prior, alignment module, joint fine-tuning, faithful restoration

## TL;DR

This paper proposes FaithDiff, which unleashes (fine-tunes) pre-trained diffusion model priors for image super-resolution for the first time. It designs an alignment module to bridge degraded image features and diffusion latent noise space, achieving faithful structural restoration through joint optimization of the encoder and diffusion model.

## Background & Motivation

**Background**: LDM-based image super-resolution methods (such as DiffBIR, SUPIR, SeeSR) have achieved significant progress, but they freeze the pre-trained diffusion models and only guide the diffusion process by improving the encoder to extract degradation-robust features.

**Limitations of Prior Work**:
- Freezing the diffusion model misinterprets any erroneous features extracted by the encoder as real image structures, leading to unfaithful restoration results (such as the distortion in the text regions in Figure 1).
- A significant gap exists between the LQ features of degraded images and the noise latent representations of the diffusion model, making direct concatenation ineffective.
- Optimizing the encoder and diffusion model separately limits the ability of each module to collaborate.

**Key Challenge**: Existing methods only rely on improved feature extraction to guide frozen diffusion models, but feature extraction from degraded images is inherently imperfect, and frozen diffusion models cannot distinguish between degradation artifacts and real structural information.

**Goal**: To enable LDMs to not only generate realistic textures but also restore faithful structures consistent with the input.

**Key Insight**: Unleashing the diffusion model (allowing fine-tuning) to let it learn to identify useful information from degraded inputs, while designing an alignment module to align encoder features with the progressive diffusion process.

**Core Idea**: Instead of freezing diffusion priors, unleash them, and allow the encoder and diffusion model to adapt to each other in joint optimization to achieve faithful super-resolution.

## Method

### Overall Architecture

1. Extract the second-to-last layer features $f^{LQ}$ (512 channels, instead of the conventional 8 channels) from the LQ image using the VAE encoder.
2. Merge $f^{LQ}$ with the noise latent representation $x_t^{HQ}$ at the current diffusion step $t$ using the alignment module to generate aligned features $f_t^a$.
3. Perform denoising with the diffusion model conditioned on $f_t^a$ and text embedding $c$.
4. Jointly fine-tune the VAE encoder, alignment module, and diffusion model.
5. Reconstruct the HQ image from the refined features using the frozen VAE decoder.

### Key Designs

**1. Deep LQ Feature Extraction**
- **Function**: Uses the second-to-last layer features (512 channels) of the VAE encoder instead of the last layer (8 channels) as the LQ feature $f^{LQ}$.
- **Mechanism**: The last layer severely compresses the channel dimension, failing to capture both degradation factors and structural details simultaneously. The second-to-last layer preserves richer information, which benefits the subsequent diffusion process.
- **Design Motivation**: Experiments demonstrate that the 512-channel feature outperforms the 8-channel feature across all metrics (validated by ablation studies). The increased information capacity is crucial for faithful restoration.

**2. Alignment Module**
- **Function**: Designs a Transformer-based alignment module to dynamically align the LQ feature $f^{LQ}$ with the noise latent representation $x_t^{HQ}$ at each diffusion step.
- **Mechanism**: Convolutions are applied to $x_t^{HQ}$ and $f^{LQ}$ separately, which are then concatenated and put through 2 Transformer blocks for interaction. An aligned feature $f_t^a$ is outputted via residual connection and a linear layer: $f_t^a = \text{Linear}(\text{Trans}(f_t^c) + f_t^x)$.
- **Design Motivation**: As the diffusion process progresses, $x_t^{HQ}$ gradually becomes clearer, while $f^{LQ}$ remains fixed. Direct addition would lead to persistent interference of degradation factors on generation. The alignment module adaptively extracts useful information from $f^{LQ}$ relevant to the current diffusion step through Transformer interactions.

**3. Unified Feature Optimization**
- **Function**: Jointly fine-tunes three components (the VAE encoder, alignment module, and diffusion model), keeping the VAE decoder and text encoder frozen.
- **Mechanism**: Two-stage training: (1) freeze the encoder and diffusion model first, pre-training only the alignment module; (2) jointly fine-tune the three components to allow mutual adaptation.
- **Design Motivation**: Separate optimization restricts the collaborative ability of the modules. Joint optimization enables the encoder to learn features better suited to the diffusion process, and the diffusion model to learn to distinguish between degradation artifacts and real structures from the degraded inputs.

### Loss & Training

- L1 noise prediction loss: $L = \|\epsilon - \hat{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} x_0^{HQ} + \sqrt{1-\bar{\alpha}_t} \epsilon, f^{LQ}, c, t)\|_1$
- Two-stage training: pre-train the alignment module first to establish connection, then jointly fine-tune the whole system.
- Use text embedding via cross-attention to assist structural feature extraction.
- Based on the Stable Diffusion architecture.

## Key Experimental Results

### Main Results (DIV2K-Val + LSDIR-Val, Three Degradation Levels)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | CLIPIQA+↑ |
|---|---|---|---|---|---|
| **Level-I (DIV2K)** | | | | | |
| Real-ESRGAN | 26.64 | 0.7737 | 0.1964 | 62.38 | 0.4649 |
| DiffBIR | 24.60 | 0.6595 | 0.2496 | 66.23 | 0.5407 |
| SUPIR | 25.09 | 0.7010 | 0.2139 | 65.49 | 0.5202 |
| SeeSR | 25.08 | 0.6967 | 0.2263 | 66.48 | 0.5336 |
| **FaithDiff** | 24.29 | 0.6668 | **0.2187** | **66.53** | **0.5432** |
| **Level-II (LSDIR)** | | | | | |
| SeeSR | 22.00 | 0.6026 | 0.2469 | 70.91 | 0.5837 |
| SUPIR | 21.30 | 0.5713 | 0.2733 | 70.59 | 0.5998 |
| **FaithDiff** | 20.88 | 0.5493 | **0.2469** | **71.15** | **0.6219** |

### Ablation Study

1. **Unleashed vs. Frozen Diffusion Model**: Unleashing yields MUSIQ +2.1, CLIPIQA+ +0.04.
2. **Joint vs. Separate Optimization**: Joint optimization consistently outperforms separate optimization in perceptual quality metrics.
3. **512 vs. 8-Channel LQ Features**: The 512-channel feature achieves comprehensive leadership on LPIPS, MUSIQ, and CLIPIQA+.
4. **Alignment Module vs. Direct Concatenation**: Transformer-based alignment significantly outperforms simple concatenation.

### Key Findings

1. **Frozen Diffusion Model is the Bottleneck**: Existing methods freeze the diffusion model, preventing it from distinguishing between degradation artifacts and real structures. Unleashing it allows the model to actively suppress reconstruction errors.
2. **Synergistic Effects of Joint Optimization**: The encoder learns to extract features matching the diffusion process, and the diffusion model learns to extract useful information from degraded features, synergistically improving faithfulness.
3. **Perceptual Metrics vs. Distortion Metrics Trade-off**: FaithDiff achieves SOTA in perceptual quality (MUSIQ, CLIPIQA+), but has slightly lower PSNR/SSIM. This is because the method prioritizes restoring realistic textures over pixel-wise fitting.
4. **Greater Advantage on Heavier Degradation**: On Level-III (severe degradation), the improvement of FaithDiff in CLIPIQA+ compared to DiffBIR is most significant.

## Highlights & Insights

- Challenges the "frozen diffusion prior" paradigm for the first time, revealing the critical role of unleashing the prior for faithful restoration.
- The design of the alignment module is simple yet effective, with the core idea being to dynamically adapt the LQ feature to the progressive denoising process.
- The joint optimization strategy fully leverages the complementarity of the three modules.
- The analysis of deep features (512 channels) vs. shallow features (8 channels) provides guidance for future work.

## Limitations & Future Work

- Pixel-level metrics such as PSNR/SSIM are inferior to conventional methods, which may not be applicable in scenarios requiring precise pixel reconstruction (such as medical imaging).
- Joint fine-tuning increases training costs and complexity.
- The quality of text descriptions affects the super-resolution outcomes.
- The alignment module uses a fixed 2-layer Transformer, which might not be the optimal architecture.
- The performance on larger-scale diffusion models (such as SDXL) has not been explored.

## Related Work & Insights

- The two-stage schemes of DiffBIR and SUPIR (degradation removal + detail generation) are limited by the precision of degradation removal; FaithDiff bypasses this bottleneck through end-to-end joint optimization.
- Unlike condition injection methods such as ControlNet, FaithDiff directly modifies the internals of the diffusion model to adapt to the SR task.
- Insight: Unleashing diffusion priors might also be effective in other low-level vision tasks (denoising, deblurring).

## Rating

⭐⭐⭐⭐ — The core insight of unleashing the diffusion prior is valuable and the joint optimization strategy is effective, but the regression in pixel-level metrics is a trade-off.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[ICLR 2026\] Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for Super-Resolution](../../ICLR2026/image_generation/bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r.md)

</div>

<!-- RELATED:END -->
