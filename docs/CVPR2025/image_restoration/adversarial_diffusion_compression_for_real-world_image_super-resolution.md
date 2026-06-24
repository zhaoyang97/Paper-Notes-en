---
title: >-
  [Paper Note] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution
description: >-
  [CVPR 2025][Image Restoration][Real-world ISR] An Adversarial Diffusion Compression (ADC) framework is proposed to distill the one-step diffusion model OSEDiff into a streamlined diffusion-GAN hybrid model. This achieves a 73% reduction in inference time, a 78% reduction in computational cost, and a 74% reduction in parameters while maintaining generative quality, reaching real-time super-resolution at 34.79 FPS.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Real-world ISR"
  - "diffusion compression"
  - "adversarial distillation"
  - "one-step diffusion"
  - "GAN"
  - "model pruning"
date: 2026-05-08
content_hash: 551b9c910ce6d012
---

# AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2411.13383](https://arxiv.org/abs/2411.13383)  
**Code**: [Guaishou74851/AdcSR](https://github.com/Guaishou74851/AdcSR)  
**Institution**: Peking University / The Hong Kong Polytechnic University / OPPO Research Institute  
**Area**: Image Restoration / Super-Resolution  
**Keywords**: Real-world ISR, diffusion compression, adversarial distillation, one-step diffusion, GAN, model pruning

## TL;DR
An Adversarial Diffusion Compression (ADC) framework is proposed to distill the one-step diffusion model OSEDiff into a streamlined diffusion-GAN hybrid model. This achieves a 73% reduction in inference time, a 78% reduction in computational cost, and a 74% reduction in parameters while maintaining generative quality, reaching real-time super-resolution at 34.79 FPS.

## Background & Motivation

**Background**: Based on Stable Diffusion (SD), real-world image super-resolution (ISR) methods have achieved significant success (StableSR, DiffBIR, SeeSR, etc.), but their multi-step inference hinders practical deployment. Recent one-step networks (OSEDiff, S3Diff) alleviate latency but still suffer from heavy computational overhead due to their reliance on large pre-trained SD models.

**Limitations of Prior Work**:
   - Multi-step diffusion methods (StableSR, etc.) suffer from extremely slow inference (dozens of iterations).
   - One-step diffusion methods (OSEDiff) still require the complete SD model: VAE encoder, prompt extractor, text encoder, and full UNet, totaling 1311M parameters and taking 0.07s per frame.
   - No existing method simultaneously addresses "maintaining quality" and "speeding up".

**Key Challenge**: Directly pruning or removing SD modules severely degrades generative capability, while retaining the entire model fails to meet real-time requirements.

**Key Insight**: By systematically analyzing each module of OSEDiff, they are classified into removable modules (VAE encoder, prompt extractor, text encoder, etc.) and prunable modules (UNet, VAE decoder). A two-stage scheme is designed for progressive compression.

**Core Idea**: Module removal + channel pruning + stage-wise adversarial distillation = real-time diffusion super-resolution.

## Method

### Adversarial Diffusion Compression (ADC) Framework

#### Module Classification
The modules of OSEDiff are classified into two categories:
- **Removable**: VAE encoder, prompt extractor (RAM/BLIP2), text encoder, cross-attention modules, and timestep embedding.
- **Prunable**: Denoising UNet (retaining the first 75% of channels) and VAE decoder (retaining the first 50% of channels).

#### Stage 1: Pre-training Pruned VAE Decoder
- A VAE decoder with 50% channel pruning is trained from scratch on OpenImage + LAION-Face + LAION-Aesthetic.
- Trained for 250K + 250K steps.
- Loss: L1 + LPIPS + patch-based adversarial loss.
- Restores the image decoding capability of the pruned VAE decoder.

#### Stage 2: Adversarial Distillation
- Jointly fine-tune the UNet with 25% channel pruning and the first-layer block of the Stage 1 pre-trained VAE decoder.
- Conducts adversarial training using a LoRA-adapted discriminator.
- Loss: $\mathcal{L} = \mathcal{L}_{distill} + \lambda_{adv} \mathcal{L}_{adv}$

### Key Technical Details

1. **Temporal Direction Alignment (TDA)**
    - After removing the text encoder and timestep embedding, the timestep is fixed to $t=1$ and the noise is set to $\epsilon = 0$.
    - A Straight-Through Estimator is used to handle the non-differentiable clipping operation.

2. **UNet-VAE Connection Optimization**
    - Connects the UNet output directly to the lowest-resolution first-layer block of the VAE decoder.
    - USes a PixelUnshuffle layer (scale factor of 2) to align the resolution of the latent space and LR image space.

3. **Discriminator Design**
    - A lightweight discriminator based on LoRA with a LoRA rank of 4.
    - Learning rate of 1e-6.

## Key Experimental Results

### Efficiency Comparison

| Method | Steps | Inference Time (s) | FLOPs (G) | Params (M) | FPS |
|------|------|-----------|----------|----------|-----|
| StableSR | 200 | 11.50 | — | — | — |
| DiffBIR | 50 | 2.72 | — | — | — |
| OSEDiff | 1 | 0.11 | 513 | 1311 | — |
| S3Diff | 1 | 0.28 | — | — | — |
| **AdcSR** | **1** | **0.03** | **111** | **340** | **34.79** |

Speedup ratio: vs. StableSR **383×**, vs. OSEDiff **3.7×**, vs. S3Diff **9.3×**.

### Restoration Quality Comparison (DRealSR)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|------|-------|-------|--------|--------|
| StableSR | 27.63 | — | 0.3317 | — |
| OSEDiff | 28.02 | — | 0.3087 | 0.2239 |
| **AdcSR** | **28.10** | — | **0.3046** | **0.2200** |

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | Params (M) | Time (s) |
|------|-------|--------|----------|---------|
| Keep VAE Encoder | 27.97 | 0.3077 | 490 | 0.05 |
| **Remove VAE Encoder** | **28.10** | **0.3046** | **456** | **0.03** |

| Configuration | FID↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|------|------|--------|---------|----------|
| Without connection optimization | 140.09 | 65.18 | 0.5807 | 0.6756 |
| **With connection optimization** | **134.05** | **66.26** | **0.5927** | **0.7049** |

## Loss & Training

### Hardware & Scale
- 8× NVIDIA A100 (80GB)
- Batch size of 96
- Stage 1: 250K steps on OpenImage + 250K steps on LAION-Face/Aesthetic
- Stage 2: 200K steps on LSDIR
- Real-ESRGAN degradation pipeline to synthesize LR-HR pairs

### Degradation Pipeline
Uses the high-order degradation model from Real-ESRGAN, including multiple stages of degradation such as blur, downsampling, noise, and JPEG compression.

## Highlights & Insights
- Systematic **module classification scheme**: Instead of mindless pruning, the role of each module is analyzed first before deciding on removal/pruning strategies.
- Removing the VAE encoder actually **improves quality** (PSNR +0.13), indicating that redundant components might introduce noise.
- **34.79 FPS** is the first real-time super-resolution speed achieved by a diffusion model.
- Key to the two-stage design: First restore VAE decoding capability, then conduct end-to-end distillation to avoid performance collapse caused by one-step-to-destination compression.
- The Straight-Through Estimator addresses the non-differentiability of clipping, which is a noteworthy engineering detail.
- Scores better in LPIPS/DISTS compared to the teacher OSEDiff, demonstrating a "the student surpasses the master" effect after distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improved Adversarial Diffusion Compression for Real-World Video Super-Resolution](../../ICLR2026/image_restoration/improved_adversarial_diffusion_compression_for_real-world_video_super-resolution.md)
- [\[CVPR 2025\] Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](iterative_predictor-critic_code_decoding_for_real-world_image_dehazing.md)
- [\[CVPR 2026\] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution](../../CVPR2026/image_restoration/one-step_diffusion_transformer_for_controllable_real-world_image_super-resolutio.md)
- [\[CVPR 2026\] Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution](../../CVPR2026/image_restoration/time-aware_one_step_diffusion_network_for_real-world_image_super-resolution.md)
- [\[CVPR 2025\] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution](pidsr_complementary_polarized_image_demosaicing_and_super-resolution.md)

</div>

<!-- RELATED:END -->
