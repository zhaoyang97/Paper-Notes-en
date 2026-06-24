---
title: >-
  [Paper Note] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression
description: >-
  [CVPR 2025][Image Generation][Diffusion Transformer] DiT-IC adapts a pre-trained T2I diffusion Transformer into a one-step image compression reconstruction model. Operating in a 32x downsampled deep latent space, it achieves SOTA perceptual quality through three alignment mechanisms—variance-guided reconstruction flow, self-distillation alignment, and latent conditional guidance—while decoding 30x faster than existing diffusion codecs.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Transformer"
  - "Image Compression"
  - "One-step Inference"
  - "Variance-guided"
  - "Self-distillation"
date: 2026-05-08
content_hash: f77851165b2549c6
---

# DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression

**Conference**: CVPR 2025  
**arXiv**: [2603.13162](https://arxiv.org/abs/2603.13162)  
**Code**: [https://njuvision.github.io/DiT-IC/](https://njuvision.github.io/DiT-IC/)  
**Area**: Image Compression / Diffusion Models  
**Keywords**: Diffusion Transformer, Image Compression, One-step Inference, Variance-guided, Self-distillation

## TL;DR

DiT-IC adapts a pre-trained T2I diffusion Transformer into a one-step image compression reconstruction model. Operating in a 32x downsampled deep latent space, it achieves SOTA perceptual quality through three alignment mechanisms—variance-guided reconstruction flow, self-distillation alignment, and latent conditional guidance—while decoding 30x faster than existing diffusion codecs.

## Background & Motivation

**Background**: Diffusion-based image compression provides outstanding perceptual quality, but the sampling overhead (4-50 steps) and high memory footprint limit its practicality.

**Limitations of Prior Work**: Existing diffusion codecs are based on U-Net, forcing them to operate in an 8x shallow latent space. Traditional VAE codecs operate at 16x-64x.

**Key Challenge**: Compression reconstruction starts from structured latent variables (close to the data manifold), making multi-step denoising potentially redundant; however, directly fine-tuning reproductive generative models leads to manifold mismatch.

**Goal**: To enable one-step diffusion inference in a 32x deep latent space.

**Key Insight**: Three "alignment" mechanisms adapt the pre-trained T2I DiT (SANA) into a compression reconstruction model.

**Core Idea**: Threefold alignment from generation to reconstruction: variance-guided denoising, self-distillation from multi-step to one-step, and latent conditioning to replace text.

## Method

### Overall Architecture

ELIC encoder + SANA DiT reconstructor. The encoder produces quantized latent variables, and the DiT performs one-step variance-guided flow matching in the 32x space. LoRA is utilized for efficient fine-tuning.

### Key Designs

1. **Variance-Guided Reconstruction Flow**

    - **Function**: Folds multi-step diffusion into a one-step adaptive transformation.
    - **Mechanism**: Utilizes encoder variance as a measure of spatial uncertainty, mapping it to pixel-wise pseudo-timesteps. Regions with high variance receive stronger denoising.
    - **Design Motivation**: Compression noise is spatially heterogeneous, making a globally uniform timestep insufficient.

2. **Self-Distillation Alignment**

    - **Function**: Distills multi-step diffusion behavior into a single step.
    - **Mechanism**: Freezes the encoder, aligning the DiT output with the encoder's latent variables (cosine alignment + margin).
    - **Design Motivation**: There is no off-the-shell multi-step teacher in deep latent spaces.

3. **Latent Conditional Guidance**

    - **Function**: Replaces text conditioning with compressed representations.
    - **Mechanism**: A lightweight projection maps features to the same embedding space as the pre-trained text encoder; utilizes CLIP-style contrastive alignment.
    - **Design Motivation**: Text lacks sufficient detail for fine-grained spatial information and requires a heavy encoder.

### Loss & Training

Two-stage IBP: Stage 1 with 100K iterations (256), Stage 2 with 60K iterations (512). LoRA rank is set to VAE=32 and DiT=64.

## Key Experimental Results

### Main Results

| Method | Steps | Latency | LPIPS BD-rate | DISTS BD-rate |
|------|------|------|-------------|-------------|
| StableCodec | 1 | 0.34s | -79.19% | -83.95% |
| ResULIC | 4 | 0.83s | -62.27% | -65.64% |
| OSCAR | 1 | 0.32s | -19.04% | -58.38% |
| **DiT-IC** | **1** | **0.15s** | **-83.65%** | **-87.88%** |

### Ablation Study

| Configuration | LPIPS | DISTS |
|------|-------|-------|
| Full | 0.00% | 0.00% |
| Train from scratch | +22.00% | +32.45% |
| Full finetuning | +7.95% | +8.05% |

### Key Findings

- SOTA perceptual quality, and a decoding time of 0.15s, which is 80x faster than DiffEIC (12.4s).
- Capable of reconstructing 2048x2048 images on a 16GB laptop GPU.
- Pre-trained initialization is highly critical; training from scratch leads to a DISTS degradation of +32.45%.
- LoRA 32/64 outperforms full finetuning.

## Highlights & Insights

- First to apply DiT to image compression, demonstrating the feasibility of operating in deep latent spaces.
- Three alignment mechanisms build upon one another progressively.
- One-step + deep latent space = extremely low latency and memory footprint.

## Limitations & Future Work

- Insufficient information in latent variables at extremely low bitrates.
- The training data is limited to only 150K samples.
- Lack of comparison with concurrent works, such as OneDC.

## Related Work & Insights

- StableCodec pioneered one-step diffusion compression but is limited by the U-Net architecture and cannot operate in deep latent spaces.
- SANA’s linear-attention DiT provides the architectural foundation for efficient diffusion; this work is the first to adapt it for compression tasks.
- ResULIC performs 4-step diffusion in a 32x latent space, whereas this work showcases that a single step can achieve better performance within the same space.
- The self-distillation approach (using the encoder itself as the target) is more elegant than distilling from external teachers and can be generalized to other reconstruction tasks.
- The feature alignment paradigm of VA-VAE inspired the self-distillation design in this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply DiT to compression; each of the three alignment mechanisms has unique merits.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three test datasets + multiple baselines + detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architecture diagrams, with step-by-step ablation demonstrations.
- **Value**: ⭐⭐⭐⭐⭐ 30x speedup + SOTA quality = a critical step toward practical diffusion-based compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](lavin-dit_large_vision_diffusion_transformer.md)
- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] PICD: Versatile Perceptual Image Compression with Diffusion Rendering](picd_versatile_perceptual_image_compression_with_diffusion_rendering.md)
- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)

</div>

<!-- RELATED:END -->
