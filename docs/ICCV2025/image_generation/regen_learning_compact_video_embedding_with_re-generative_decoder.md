---
title: >-
  [Paper Note] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder
description: >-
  [Image Generation] REGEN replaces the conventional VAE decoder with a Diffusion Transformer (DiT) as a re-generative decoder for video, breaking the temporal compression bottleneck through a "generation rather than exact reconstruction" learning paradigm and achieving up to 32× temporal compression.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 4713577a86879074
---

# REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder

> **Conference**: ICCV 2025
> **arXiv**: [2503.08665](https://arxiv.org/abs/2503.08665)
> **Code**: [Project Page](https://bespontaneous.github.io/REGEN/)
> **Area**: Image Generation
> **Keywords**: video embedder, diffusion transformer, temporal compression, latent diffusion, video generation

## TL;DR

REGEN replaces the conventional VAE decoder with a Diffusion Transformer (DiT) as a re-generative decoder for video, breaking the temporal compression bottleneck through a "generation rather than exact reconstruction" learning paradigm and achieving up to 32× temporal compression.

## Background & Motivation

Current latent diffusion models (LDMs) for video generation rely heavily on video embedders to compress video into a latent space for modeling. State-of-the-art video embedders such as MAGVIT-v2 typically achieve 8× spatial compression but only 4× temporal compression. Higher temporal compression is critical for training and inference efficiency but faces a fundamental bottleneck.

**Key Challenge**: In traditional encoder-decoder architectures, increasing the compression ratio inevitably leads to information loss; the decoder cannot accurately reconstruct high-frequency details from sparse latents, creating a fundamental compression-reconstruction trade-off.

**Key Insight**: In the context of latent diffusion modeling, the core property of the latent space should be to **generate visually plausible content** rather than faithfully reconstruct the input video. This relaxed criterion makes it possible to substantially increase the compression ratio.

Based on this, REGEN transforms the conventional encoder-decoder into an **encoder-generator** paradigm: the encoder only needs to retain core semantic and structural information, while the DiT decoder is responsible for synthesizing realistic details.

## Method

### Overall Architecture

REGEN consists of two core modules (Fig. 2):

1. **Spatiotemporal Video Encoder**: Encodes the input video into a two-frame compact latent representation (content latent $z_c$ + motion latent $z_m$)
2. **DiT Generative Decoder**: Conditioned on the latent variables, re-synthesizes the video from noise via a diffusion process

### Key Design 1: Spatiotemporal Video Encoder

A MAGVIT-v2-style causal 3D convolutional encoder is adopted to encode $k+1$ frames of video into two frames of latent features:

$$z_c, z_m = E(x_{input})$$

- $z_c$: **Content latent frame**, which causally contains only the first-frame information
- $z_m$: **Motion latent frame**, encoding compressed motion information from the remaining frames
- Both use 8 latent channels, 8× spatial compression, and temporal compression up to 8×/16×/32×

### Key Design 2: Latent Conditioning Module (Content-Aware PE)

This is the core innovation of REGEN. Conventional DiTs use fixed positional encodings (PE) that generalize poorly to unseen resolutions and aspect ratios at training time. REGEN replaces positional encodings with **content-aware positional encodings** generated from the encoded latent variables:

$$C_e(x, y, t_f | [z_c, z_m]) = M_s\left(z_c(x,y) \oplus M_t(t_f | z_m(x,y))\right)$$

where $M_t$ is a SIREN network that maps temporal coordinates $t_f$ to feature vectors modulated by $z_m$, and $M_s$ is a linear projector. The resulting expanded latent $z_e$ is added to the DiT's token embeddings and timestep embeddings as input.

This design: (1) completely removes the original spatial/temporal PE from the DiT; (2) naturally supports arbitrary resolutions and aspect ratios; (3) enables temporal interpolation and extrapolation.

### Loss & Training

The encoder and DiT decoder are jointly trained end-to-end using the standard diffusion denoising objective:

$$\mathcal{L}(\theta) = \|\epsilon - \epsilon_\theta(x^t_{target}, [z_c, z_m])\|^2$$

### DiT Decoder Configuration

- 24 Transformer blocks, 16 heads, hidden dimension 2048
- Patch size = 8 (matching the spatial downsampling ratio)
- Full spatiotemporal self-attention

## Key Experimental Results

### Main Results: High-Compression Reconstruction Comparison (Tab. 1)

| Method | Compression | MCL-JCV PSNR | MCL-JCV rFVD ↓ | DAVIS PSNR | DAVIS rFVD ↓ |
|--------|-------------|-------------|----------------|-----------|-------------|
| MAGVIT-v2 | 8×8×8 | 29.14 | 72.07 | 24.75 | 125.03 |
| **REGEN** | 8×8×8 | **32.74** | **29.88** | **29.34** | **89.98** |
| MAGVIT-v2 | 8×8×16 | 26.62 | 185.69 | 21.21 | 417.43 |
| **REGEN** | 8×8×16 | **30.41** | **92.48** | **26.27** | **235.13** |
| MAGVIT-v2 | 8×8×32 | 22.97 | 536.01 | 18.23 | 1080.15 |
| **REGEN** | 8×8×32 | **28.71** | **224.56** | **23.49** | **522.20** |

REGEN outperforms MAGVIT-v2 across all compression ratios, with the margin increasing as compression ratio grows. At 32× temporal compression, REGEN's rFVD is approximately 50% that of MAGVIT-v2.

### Baseline 4× Compression Comparison (Tab. 2, 512×512)

| Method | PSNR | SSIM | rFVD ↓ |
|--------|------|------|--------|
| OmniTokenizer | 24.63 | 0.710 | 93.35 |
| WF-VAE | 31.00 | 0.804 | 55.01 |
| VidTok | 32.06 | 0.836 | 38.85 |
| MAGVIT-v2 | 31.49 | 0.829 | 28.63 |
| **REGEN** | **32.94** | **0.857** | **22.40** |

Even at the baseline 4× compression setting, REGEN surpasses all state-of-the-art methods specifically designed for this configuration.

### Ablation Study: Conditioning Mechanism (Tab. 3)

| Method | 192×320 PSNR | 384×640 PSNR | 384×640 rFVD ↓ |
|--------|-------------|-------------|----------------|
| In-context conditioning | 25.71 | 23.39 | 441.98 |
| **Ours (Content-Aware PE)** | **26.04** | **29.41** | **57.01** |

**Key Findings**: In-context conditioning exhibits severe grid-like artifacts at higher resolutions (rFVD spikes to 442), whereas REGEN's content-aware PE generalizes gracefully to unseen resolutions.

### Few-Step and Single-Step Sampling

The DiT decoder achieves high-quality reconstruction in a single sampling step without external distillation. PSNR improves slightly as the number of steps decreases (reduced sharpening), while rFVD increases marginally. This is attributed to the extremely strong conditioning signal provided by the encoded latent variables.

## Highlights & Insights

1. **Paradigm Shift**: "Encoder-generator" replaces "encoder-decoder," shifting from "exact reconstruction" to "plausible generation" and breaking the compression-reconstruction trade-off.
2. **Content-Aware PE** simultaneously addresses positional encoding generalization, conditioning injection, and arbitrary-resolution support.
3. The decoder supports **single-step inference** without distillation, significantly reducing practical deployment costs.
4. 32× temporal compression reduces the number of latent frames in text-to-video generation by ~5×, substantially lowering both training and inference costs.

## Limitations & Future Work

- Generative decoding introduces stochasticity, potentially causing minor variations across decoding runs.
- Temporal artifacts in high-motion regions persist under extreme compression.
- The computational overhead of the DiT decoder warrants further optimization.

## Related Work & Insights

- Video embedders: MAGVIT-v2, OmniTokenizer, VidTok, WF-VAE
- Diffusion autoencoders: DiffAE, PDAE
- Video LDMs: CogVideoX, HunyuanVideo

## Rating

- **Novelty**: ★★★★★ — The encoder-generator paradigm is pioneering
- **Technical Depth**: ★★★★☆ — Content-aware PE is elegantly designed with strong theory-practice integration
- **Experimental Thoroughness**: ★★★★★ — Comprehensive comparisons across multiple compression ratios and datasets with thorough ablations
- **Writing Quality**: ★★★★★ — Core insights are clearly articulated with well-structured argumentation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)

</div>

<!-- RELATED:END -->
