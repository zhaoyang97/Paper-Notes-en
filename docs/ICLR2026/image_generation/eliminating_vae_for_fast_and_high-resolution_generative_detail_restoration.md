---
title: >-
  [Paper Note] Eliminating VAE for Fast and High-Resolution Generative Detail Restoration
description: >-
  [ICLR 2026][Image Generation][VAE elimination] By replacing the VAE encoder and decoder with ×8 pixel-(un)shuffle operations, this work converts latent-space diffusion super-resolution (GenDR) into pixel-space super-resolution (GenDR-Pix). Combined with multi-stage adversarial distillation and a PadCFG inference strategy, the method achieves 2.8× speedup and 60% memory reduction with negligible visual degradation, enabling 4K image restoration within 1 second using only 6 GB of VRAM for the first time.
tags:
  - ICLR 2026
  - Image Generation
  - VAE elimination
  - super-resolution
  - pixel-space diffusion
  - adversarial distillation
  - pixel-shuffle
date: 2026-05-08
content_hash: fe13f894ac921cff
---

# Eliminating VAE for Fast and High-Resolution Generative Detail Restoration

**Conference**: ICLR 2026
**arXiv**: [2602.10630](https://arxiv.org/abs/2602.10630)
**Code**: None (improvement upon GenDR)
**Area**: Image Generation
**Keywords**: VAE elimination, super-resolution, pixel-space diffusion, adversarial distillation, pixel-shuffle

## TL;DR

By replacing the VAE encoder and decoder with ×8 pixel-(un)shuffle operations, this work converts latent-space diffusion super-resolution (GenDR) into pixel-space super-resolution (GenDR-Pix). Combined with multi-stage adversarial distillation and a PadCFG inference strategy, the method achieves 2.8× speedup and 60% memory reduction with negligible visual degradation, enabling 4K image restoration within 1 second using only 6 GB of VRAM for the first time.

## Background & Motivation

Diffusion models have achieved breakthroughs in real-world super-resolution (Real-World SR), yet they face two major bottlenecks: **slow inference** and **high memory consumption**. Existing acceleration approaches (e.g., step distillation) reduce diffusion steps to one, but **memory constraints still limit the maximum processable resolution**, requiring tile-by-tile processing for high-resolution images.

**Key Finding: VAE is the bottleneck.** Profiling analysis reveals:
- For a 1024² input, VAE latency is 2.6× that of UNet (191ms vs. 73ms)
- VAE memory is 1.3× that of UNet (3.5 GB vs. 2.7 GB)
- For a 2880² input, VAE latency is 1.89× that of UNet (3228ms vs. 1710ms)
- At 4096², VAE directly causes OOM

**Fidelity bottleneck**: Even the 16-channel VAE achieves reconstruction PSNR below 35 dB, with lossy compression discarding high-frequency details.

**Key Challenge**: Existing methods (e.g., AdcSR) only simplify the VAE while still operating in latent space; the generation–reconstruction conflict introduced by simplification cannot be fundamentally resolved (e.g., removing decoder attention leads to texture loss).

**Core Idea**: pixel-(un)shuffle performs spatial scale transformations analogous to the VAE and can fully replace it, converting latent-space diffusion to pixel space. However, ×8 pixel-shuffle introduces checkerboard/repetitive-pattern artifacts, and no suitable discriminator exists for this setting.

## Method

### Overall Architecture

GenDR-Pix = GenDR − VAE + ×8 pixel-(un)shuffle + multi-stage adversarial distillation + PadCFG. The VAE encoder and decoder are progressively removed across two stages.

### Key Designs

1. **Stage I: Removing the Encoder**

    - Replace the VAE encoder with a ×8 pixel-unshuffle layer
    - Use GenDR's UNet as a feature extractor for the discriminator
    - Generator loss: $\mathcal{L}_{\mathcal{G}_1} = \|z_{\text{tea}} - z_{\text{stu}}\|_1 + \lambda_1 \cdot \text{softplus}(-\mathcal{D}(z_{\text{stu}}))$
    - Output model GenDR-Adc: low-quality pixels → high-quality latent

2. **Stage II: Removing the Decoder (Core Challenge)**

    - Replace the decoder with a ×8 pixel-shuffle layer
    - **Problem 1 – Repetitive pattern artifacts**: a single improper weight in ×8 upscaling propagates identical artifacts across all 8×8 patches
    - **Solution – Masked Fourier Space (MFS) Loss**:
        - Artifacts manifest as periodic highlights in the frequency domain, aligned with the pixel-shuffle scale factor
        - A band-reject filter mask $\mathcal{M}$ is designed to penalize anomalous amplitudes:
    $\mathcal{L}_{\mathcal{F}} = \|\mathcal{M} \cdot (|\mathcal{F}\{y_{\text{stu}}\}| - |\mathcal{F}\{y_{\text{tea}}\}|)\|_1$

    - **Problem 2 – No suitable discriminator**: latent-space discriminators cannot process pixel-shuffled features
    - **Solution – Use Stage I model as discriminator**: GenDR-Adc ($\mathcal{G}_1$) naturally encodes inputs via pixel-unshuffle

    - **Problem 3 – Discriminator collapse**: fixed pixel-unshuffle patterns cause the discriminator to focus on discrete distributional representations
    - **Solution – Random Padding (RandPad) augmentation**: randomly sample $p_h, p_w \in \{0,...,7\}$ and apply random padding to both SR and HQ images:
    $\text{randpad}(y) = \text{pad}(y, [p_h, 8-p_h, p_w, 8-p_w])$
      This encourages the discriminator to extract continuous representations, preventing pattern collapse

3. **PadCFG: Classifier-Free Guidance in Pixel Space**

    - Applying CFG directly in pixel space exacerbates artifacts
    - Self-ensemble (multiple paddings → fusion) is integrated with CFG:
    $\bar{y} = \omega \times \mathcal{G}_2(\text{pad}(x,[4,4,4,4]), c_{\text{pos}}) + (1-\omega) \times \mathcal{G}_2(\text{pad}(x,[3,5,3,5]), c_{\text{neg}})$
    - Requires only 2 forward passes (same as standard CFG) while incorporating the ensemble effect of different paddings

### Loss & Training

Stage II total generator loss:
$$\mathcal{L}_{\mathcal{G}_2} = \|y_{\text{tea}} - y_{\text{stu}}\|_1 + \lambda_1 \cdot \text{softplus}(-\mathcal{G}_1(y_{\text{stu}})) + \lambda_2 \mathcal{L}_{\mathcal{P}} + \lambda_3 \mathcal{L}_{\mathcal{F}}$$

Parameters $\lambda_{1,2,3} = 0.05, 1, 0.1$. Trained with AdamW, learning rate 1e-5, BFloat16, 8×A100 GPUs, DeepSpeed ZeRO2.

## Key Experimental Results

### Main Results

**ImageNet-Test Quantitative Comparison (×4 SR, 512² input)**:

| Method | Params | MACs | PSNR↑ | SSIM↑ | CLIPIQA↑ | MUSIQ↑ |
|--------|--------|------|-------|-------|----------|--------|
| StableSR-50 | 1410M | 79940G | 26.00 | 0.7317 | 0.5768 | 64.54 |
| DiffBIR-50 | 1717M | 24234G | 25.45 | 0.6651 | 0.7486 | 73.04 |
| OSEDiff-1 | 1775M | 2265G | 24.82 | 0.7017 | 0.6778 | 71.74 |
| GenDR-1 | 933M | 1637G | 24.14 | 0.6878 | 0.7395 | 74.68 |
| **GenDR-Pix** | **866M** | **344/744G** | **25.49** | **0.7286** | 0.7168 | 72.85 |

**Efficiency Comparison (4K output, A100)**:

| Model | VAE Status | Time | Memory | MUSIQ |
|-------|-----------|------|--------|-------|
| GenDR | Full VAE | 4.92s | 20.75 GB | 70.96 |
| GenDR-Adc | Encoder removed | 2.69s (−45%) | 17.75 GB (−14%) | 70.44 |
| **GenDR-Pix** | **VAE removed** | **1.75s (−64%)** | **8.01 GB (−61%)** | 70.23 |
| GenDR-Pix⋆ | No CFG | 0.87s (−82%) | 5.03 GB (−76%) | 68.64 |

### Ablation Study

**Discriminator Design**:

| Discriminator | RandPad | MUSIQ | CLIPIQA |
|--------------|---------|-------|---------|
| None | − | 60.95 | 0.4924 |
| GenDR | − | 61.07 | 0.5457 |
| GenDR-Adc | ✗ | 63.87 | 0.5764 |
| GenDR-Adc | ✓ | **63.89** | **0.5937** |

**MFS Loss**: Removing the frequency-domain loss degrades NIQE from 4.11 to 4.84 and LIQE from 3.21 to 2.91.

**PadCFG**: PadCFG(4,4)+(3,5) achieves MUSIQ +0.45 and CLIPIQA +0.0061 while maintaining the same latency as vanilla CFG.

### Key Findings

- The speedup from VAE removal substantially exceeds that of architectural pruning approaches
- Artifacts from ×8 pixel-shuffle can be effectively suppressed via frequency-domain loss and random padding
- GenDR-Pix is the only model capable of directly super-resolving to 8K without tiling
- User studies show that GenDR-Pix achieves perceptual quality comparable to the original GenDR

## Highlights & Insights

- The finding that **VAE is simultaneously an efficiency and fidelity bottleneck for one-step diffusion SR** has broad implications
- The analogy between pixel-(un)shuffle and VAE is elegant and motivates a novel acceleration direction
- The multi-stage distillation strategy of using the previous-stage model as the discriminator for the next stage is a clever design choice
- RandPad's solution to discriminator collapse is simple yet effective, inspired by E-LPIPS
- The practical performance of 1-second 4K SR with 6 GB VRAM has significant industrial deployment value

## Limitations & Future Work

- Validation is currently limited to GenDR (SD2.1 UNet); generalization to newer architectures such as SDXL and Flux remains unexplored
- Only ×4 SR is supported; adapting to other scale factors requires adjusting pixel-shuffle parameters
- The padding parameter selection in PadCFG is largely empirical and lacks theoretical justification
- Stage II training still requires the Stage I model as a discriminator, resulting in a relatively complex training pipeline
- For extreme degradations (e.g., severe compression artifacts), the performance gap with GenDR may widen

## Related Work & Insights

- **AdcSR**: replaces only the encoder with ×2 pixel-unshuffle; this work extends the approach to ×8, fully eliminating the VAE
- **PixelFlow**: explores pixel-space diffusion
- **GenDR / SiD / VSD**: foundational one-step diffusion SR methods
- **E-LPIPS**: perceptual loss with random transformations, which inspired the RandPad design
- Insight: similar VAE elimination strategies may be applicable to other diffusion-based applications that rely on VAEs, such as image editing and inpainting

## Rating

- Novelty: ⭐⭐⭐⭐ The direction of fully eliminating the VAE is novel; however, the core mechanism (pixel-shuffle substitution) is relatively intuitive, with the main contribution lying in overcoming engineering challenges
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multiple datasets, metrics, detailed ablations, user studies, and efficiency analyses
- Writing Quality: ⭐⭐⭐⭐ The roadmap-style presentation is clear and intuitive, though the density of equations is high
- Value: ⭐⭐⭐⭐⭐ Significant practical acceleration (2.8× speedup / 60% memory reduction); 4K in 1 second has clear industrial application value

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GenDR: Lighten Generative Detail Restoration](gendr_lighten_generative_detail_restoration.md)
- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](../../CVPR2026/image_generation/da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)

<!-- RELATED:END -->
