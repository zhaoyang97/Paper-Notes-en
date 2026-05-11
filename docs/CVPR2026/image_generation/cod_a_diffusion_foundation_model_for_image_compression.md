---
title: >-
  [Paper Note] CoD: A Diffusion Foundation Model for Image Compression
description: >-
  [CVPR 2026][Image Generation][Compression-oriented diffusion] This paper proposes CoD, the first diffusion foundation model designed for image compression. Trained from scratch for joint compression-generation optimization, CoD replaces Stable Diffusion in downstream diffusion codecs and achieves state-of-the-art performance at ultra-low bitrates (0.0039 bpp), with a training cost of only 0.3% of that required by SD.
tags:
  - CVPR 2026
  - Image Generation
  - Compression-oriented diffusion
  - foundation model
  - rectified flow
  - pixel-space diffusion
  - rate-distortion-perception
date: 2026-05-08
content_hash: 51db66487eb5fc1a
---

# CoD: A Diffusion Foundation Model for Image Compression

**Conference**: CVPR 2026
**arXiv**: [2511.18706](https://arxiv.org/abs/2511.18706)
**Code**: [GitHub](https://github.com/microsoft/GenCodec/tree/main/CoD)
**Area**: Image Compression / Diffusion Models
**Keywords**: Compression-oriented diffusion, foundation model, rectified flow, pixel-space diffusion, rate-distortion-perception

## TL;DR

This paper proposes CoD, the first diffusion foundation model designed for image compression. Trained from scratch for joint compression-generation optimization, CoD replaces Stable Diffusion in downstream diffusion codecs and achieves state-of-the-art performance at ultra-low bitrates (0.0039 bpp), with a training cost of only 0.3% of that required by SD.

## Background & Motivation

Existing diffusion codecs (PerCo, DiffEIC, OSCAR, etc.) are typically built upon Stable Diffusion to inherit its generative prior. However, text conditioning is suboptimal from a compression standpoint:

**Limited descriptive capacity**: Human-generated text cannot faithfully describe the spatial and textural details of natural images.

**Non-differentiable discrete vocabulary**: Text encoders (e.g., BLIP-2) and diffusion models (e.g., SD) cannot be jointly optimized end-to-end, precluding rate-distortion optimization.

**Empirical evidence**: Zero-shot experiments in DiffC demonstrate that text conditioning actually **degrades** compression performance at low bitrates.

Core insight: Viewing an image captioning model as an encoder and a diffusion model as a decoder constitutes a compression system in essence—yet text as an intermediate representation is inefficient. The correct direction is to replace text with native image tokens learned by neural networks, and to jointly train compression and generation end-to-end.

## Method

### Overall Architecture

CoD adopts a compact architecture: native image encoder → information bottleneck (vector quantization) → conditional decoder → diffusion model (DiT backbone + DDT head). Implementations are provided in both pixel space and latent space.

### Key Designs

1. **Conditional Encoding and Information Bottleneck**: The encoder compresses the image to 1/32 resolution using residual blocks and attention layers. The information bottleneck employs vector quantization (VQ) with codebook size $N = 2^4 = 16$, corresponding to an ultra-low bitrate of $4 \text{ bits} / (32 \times 32) = 0.0039 \text{ bpp}$. This forces the diffusion model to develop strong generative capacity to compensate for information loss. The conditional decoder reconstructs quantized tokens into intermediate conditions at 1/16 resolution.

2. **Unified Rectified Flow Training**: CoD predicts the velocity field $v_t = x - \epsilon$ under a linear interpolation schedule $x_t = t \cdot x + (1-t) \cdot \epsilon$, trained with the rectified flow loss. A key finding is that the standard RF loss **guarantees structural consistency but not color fidelity**. To address this, a unified training strategy is proposed: $\alpha\%$ of samples are trained with $t \in [0,1]$ (optimizing perception), while the remainder are trained with $t=0$. At $t=0$, the RF loss is equivalent to single-step reconstruction MSE:

    $\mathcal{L}_{\text{RF}}|_{t=0} = \text{MSE}(v_0, v_0^{\text{pred}}) = \text{MSE}(x, \hat{x}_0)$

   This naturally incorporates a distortion term within the RF framework, enabling joint rate-distortion-perception optimization.

3. **Pixel Space vs. Latent Space**: Latent-space CoD operates in the SD-VAE latent space (2×2 patch embedding → 1/16), and is constrained by the VAE reconstruction ceiling (~26 dB PSNR, ~0.6 bpp bitrate cap). Pixel-space CoD uses 16×16 patch embeddings to directly model raw pixels, where each DDT head feature predicts a neural-field reconstruction of a 16×16 patch. Without VAE constraints, pixel-space CoD covers a wide bitrate range of 0.0039–4 bpp, achieving PSNR up to ~47 dB approaching near-lossless quality.

4. **Zero-Shot Distortion-Perception Control**: The unified training endows CoD with the ability to directly control the distortion-perception trade-off via the number of sampling steps. 25 steps yields optimal perceptual quality; reducing to 1 step improves PSNR by 3.4 dB (16.2→19.6 dB), with smooth interpolation at intermediate step counts.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{RF}} + \lambda \cdot \mathcal{L}_{\text{REPA}} + \beta \cdot \mathcal{L}_C + \gamma \cdot \mathcal{L}_{\text{aux}}$$

where $\mathcal{L}_{\text{REPA}}$ is a DINOv2 feature alignment loss, $\mathcal{L}_C$ is the codebook commitment loss, and $\mathcal{L}_{\text{aux}}$ is an auxiliary head loss (reconstructing raw pixels and DINOv2 features). Training proceeds in two stages: 256×256 (400k steps) → 512×512 (150k steps), taking approximately 5 days on 4 A100 GPUs.

## Key Experimental Results

### Main Results

**Pixel-space comparison** (Kodak 512×512):

| Method | Bitrate (bpp) | PSNR↑ | FID↓ | Notes |
|--------|--------------|-------|------|-------|
| VTM | ~0.2 | Baseline | - | Traditional codec |
| Pixel-CoD+DiffC | ~0.2 | **≈VTM** | **Much better** | BD-Rate -2.1% vs VTM |
| MS-ILLM (GAN) | ~0.2 | Lower | Higher | Perceptual quality at cost of PSNR |
| HiFiC (GAN) | ~0.2 | Low | Medium | Same as above |

**Latent-space comparison** (ultra-low bitrate):

| Method | Bitrate (bpp) | Reconstruction Quality | Notes |
|--------|--------------|----------------------|-------|
| CoD (latent) + DiffC | <0.02 | **SOTA** | Significant advantage at ultra-low bitrate |
| SD-based DiffC | <0.02 | Poor | Text conditioning harmful at low bitrate |
| PerCo (SD) | 0.0036 | Medium | Relies on text+image conditioning |
| OSCAR | ~0.01 | Better | Text-free but still SD-based |

### Ablation Study / Scaling Law

| Model Scale (parameters) | Compression Performance | Notes |
|--------------------------|------------------------|-------|
| 49M CoD | Already outperforms MS-ILLM (181M) | GAN method has more parameters but worse performance |
| 114M CoD | Noticeably better | |
| 330M CoD | Further improved | Clear scaling law trend |

### Key Findings

- **The potential of pixel-space diffusion has been severely underestimated**: Pixel-space CoD simultaneously achieves VTM-level PSNR and perceptual quality surpassing GANs, constituting the first demonstration that a diffusion codec can win on both distortion and perception metrics.
- **Text conditioning is genuinely harmful**: Adding text conditioning to DiffC on SD degrades LPIPS at low bitrates, whereas CoD conditioning directly improves performance.
- **Extremely low training cost**: ~20 A100 GPU days vs. ~6,250 days for SD (0.3%), fully reproducible with open-source data.
- **A 49M-parameter model outperforms a 181M-parameter GAN codec**, demonstrating that performance gains stem from algorithmic improvements rather than model scale.

## Highlights & Insights

- The role of text conditioning in diffusion codecs is re-examined from the perspective of compression theory, yielding the counterintuitive conclusion that text is harmful, supported by a clear theoretical explanation.
- The unified RF training makes single-step reconstruction at $t=0$ equivalent to MSE distortion optimization, naturally incorporating rate-distortion-perception tri-objective optimization within a continuous flow framework.
- Controlling the distortion-perception trade-off via the number of sampling steps is a zero-cost auxiliary capability requiring no additional training.
- A comprehensive revival of pixel-space diffusion: whereas latent-space diffusion was previously considered universally superior, this work demonstrates that pixel-space diffusion offers irreplaceable advantages in high-bitrate and wide-range scenarios.

## Limitations & Future Work

- Currently limited to 512×512 resolution; scaling to 2K+ would require substantially greater computational cost.
- Like all diffusion codecs, inference speed does not meet real-time encoding requirements (though a single-step distillation variant approaches real-time performance).
- The minimum bitrate is fixed at 0.0039 bpp (constrained by VQ codebook size); flexible bitrate control requires additional design.
- Validation on video compression has not been conducted; temporal extension is a natural future direction.

## Related Work & Insights

- **DiffC** [Theis et al.] proposed a theoretical framework for zero-shot diffusion-based compression; CoD provides a more suitable foundation model for this framework.
- **PerCo** [Careil et al.] demonstrated the potential of diffusion models for extreme low-bitrate compression, but relies on text conditioning.
- **CDC** [Yang et al.] is an early exploration of pixel-space diffusion codecs, but requires perceptual losses and does not investigate scaling laws.
- The rate-distortion-perception trade-off theory [Blau & Michaeli] provides the theoretical basis for the optimization objectives of this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first diffusion foundation model designed for compression; both the unified training strategy and the revival of pixel-space diffusion represent significant contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual-track pixel/latent-space comparisons, multiple benchmarks, scaling law analysis, zero-shot control, and visual comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative from problem analysis to method design to experimental validation is tightly connected, with deep insights throughout.
- **Value**: ⭐⭐⭐⭐⭐ 0.3% training cost, fully open-source data, and SOTA performance collectively provide a foundational advancement for the field of diffusion-based image compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](ditic_aligned_diffusion_transformer_for_efficient.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](../../ICLR2026/image_generation/zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[AAAI 2026\] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](../../AAAI2026/image_generation/steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)

</div>

<!-- RELATED:END -->
