---
title: >-
  [Paper Note] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer
description: >-
  [ICCV 2025][Image Generation][Autoregressive image generation] This paper proposes DC-AR, a masked autoregressive text-to-image generation framework built upon a Deep Compression Hybrid Tokenizer (DC-HT…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Autoregressive image generation"
  - "image tokenizer"
  - "masked autoregressive"
  - "deep compression"
  - "text-to-image"
date: 2026-05-08
content_hash: 20fa19269dc0f3a4
---

# DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer

**Conference**: ICCV 2025
**arXiv**: [2507.04947](https://arxiv.org/abs/2507.04947)  
**Code**: [https://github.com/dc-ai-projects/DC-AR](https://github.com/dc-ai-projects/DC-AR)  
**Area**: Image Generation
**Keywords**: Autoregressive image generation, image tokenizer, masked autoregressive, deep compression, text-to-image

## TL;DR

This paper proposes DC-AR, a masked autoregressive text-to-image generation framework built upon a Deep Compression Hybrid Tokenizer (DC-HT, 32× spatial compression). Through a hybrid pipeline of discrete token generation for structure followed by residual token refinement, DC-AR achieves state-of-the-art gFID of 5.49 on MJHQ-30K while delivering 1.5–7.9× higher throughput than diffusion models.

## Background & Motivation

Autoregressive (AR) image generation is rapidly closing the gap with diffusion models, with masked autoregressive models (MaskGIT paradigm) enabling efficient generation via parallel decoding. However, the efficiency bottleneck of AR models lies in the **compression ratio of the image tokenizer**:

**The current standard is 8×/16× spatial compression**: a 256×256 image still requires 1024/256 tokens, and computational cost grows rapidly at higher resolutions.

**Continuous tokenizers have achieved 32× compression (DC-AE)**, but **discrete tokenizers cannot be directly applied**—experiments show that applying vector quantization directly to DC-AE yields severely degraded reconstruction quality.

**1D tokenizers (e.g., TiTok)** can achieve high compression ratios but sacrifice 2D spatial correspondence, preventing cross-resolution generalization and requiring retraining for different resolutions at considerable cost.

**Key Challenge**: How can a high-compression-ratio tokenizer for AR models be constructed while preserving 2D spatial structure to support cross-resolution generalization?

## Method

### Overall Architecture

DC-AR = DC-HT (tokenizer) + Hybrid Masked Autoregressive Generator

- **Tokenizer DC-HT**: Decomposes an image into discrete tokens ($\mathbf{Z}_q$, structural information) and residual continuous tokens ($\mathbf{Z}_r = \mathbf{Z} - \mathbf{Z}_q$, detail information), with 32× spatial compression.
- **Generator**: A masked autoregressive Transformer first predicts discrete tokens (12-step unmasking), then an MLP diffusion head conditioned on Transformer hidden states predicts residual tokens → the two are summed → decoded by the decoder to produce the final image.

### Key Designs

1. **DC-HT (Deep Compression Hybrid Tokenizer)**:

    - Based on the DC-AE-f32c32 architecture (CNN encoder + decoder), with 32× spatial compression and latent channel=32.
    - **Hybrid tokenization**: Simultaneously supports a discrete path ($\mathbf{Z}_q = \text{Quant}(\text{Enc}(\mathbf{I}))$) and a continuous path ($\mathbf{Z} = \text{Enc}(\mathbf{I})$), ensuring the decoder can effectively decode both token types.
    - Residual tokens are defined as $\mathbf{Z}_r = \mathbf{Z} - \mathbf{Z}_q$, compensating for quantization loss.
    - **Three-stage adaptation training strategy** (key contribution):
        - **Stage 1 – Continuous Warm-up**: Trains only the continuous path (briefly) to initialize encoder weights.
        - **Stage 2 – Discrete Learning**: Trains only the discrete path to learn a stable VQ codebook ($N=16384$).
        - **Stage 3 – Alternating Fine-tuning**: Freezes the encoder and quantizer; fine-tunes the decoder with 50% probability of selecting either the continuous or discrete path.
    - Outcome: rFID improves from 1.92→1.60; discrete-rFID improves from 6.18→5.13.
    - Key advantage: Preserves 2D spatial structure, supporting cross-resolution generalization (256→512 without retraining the tokenizer).

2. **Hybrid Masked Autoregressive Generation**:

    - **Transformer backbone**: PixArt-α architecture (28 layers, width=1152, 634M parameters); text is injected via cross-attention.
    - **During training**: Discrete tokens are randomly masked and predicted with cross-entropy loss; simultaneously, Transformer hidden states condition an MLP diffusion head to predict residual tokens with a diffusion loss.
    - **During inference**: Starting from fully masked tokens, 12-step progressive unmasking generates all discrete tokens → final hidden states condition the diffusion head to generate residual tokens via denoising → the two are summed → decoded to produce the image.
    - **Key design decision**: Only discrete tokens participate in the Transformer forward pass. MaskGIT requires only 8 steps to approach optimality, whereas MAR (continuous tokens) requires 64 steps. Residual tokens serve solely for refinement without altering the overall structure.

3. **Cross-Resolution Training Strategy**:

    - The resolution generalization property of the 2D tokenizer enables a "low-resolution pretraining + high-resolution fine-tuning" strategy.
    - 256×256 pretraining for 200K steps + 512×512 fine-tuning for 50K steps.
    - Saves 1.9× GPU hours compared to training from scratch at 512×512 (760 vs. 1440), while achieving better quality (gFID 5.50 vs. 6.64).

### Loss & Training

- Tokenizer: Reconstruction loss + GAN loss (standard VQGAN training).
- Generator: Cross-entropy (masked prediction of discrete tokens) + diffusion loss (residual token prediction).
- Text encoder: T5-base (109M parameters), relatively lightweight.
- Training data: JourneyDB + an internal MidJourney-style synthetic dataset with captions generated by VILA1.5-13B.
- Diffusion head: 6-layer MLP with only 37M parameters.

## Key Experimental Results

### Main Results — Text-to-Image Generation (MJHQ-30K & GenEval)

| Method | Type | Params | Resolution | Steps | gFID↓ | Latency (s) | Throughput (img/s) |
|--------|------|--------|------------|-------|-------|-------------|-------------------|
| SDXL | Diffusion | 2.6B | 1024² | 20 | 6.63 | 1.4 | 2.1 |
| PixArt-α | Diffusion | 630M | 512² | 20 | 6.14 | 1.2 | 1.7 |
| Sana-0.6B | Diffusion | 590M | 512² | 20 | 5.67 | 0.8 | 6.7 |
| Show-o | Mask. AR | 1.3B | 512² | 12 | 14.59 | 1.1 | 1.3 |
| TA-TiTok (KL) | Mask. AR | 602M | 256² | 32 | 7.24 | - | - |
| **DC-AR** | **Mask. AR** | **671M** | **512²** | **12** | **5.49** | **0.4** | **10.3** |

DC-AR outperforms all baselines in gFID, with a latency of only 0.4s (2× faster than Sana, 3.5× faster than SDXL) and a throughput of 10.3 img/s (7.9× higher than Show-o).

GenEval benchmark:

| Method | S.Obj | T.Obj | Count | Colors | Position | C.Attri. | Overall |
|--------|-------|-------|-------|--------|----------|----------|---------|
| Sana-0.6B | 0.99 | 0.76 | 0.64 | 0.88 | 0.18 | 0.39 | 0.64 |
| Show-o | 0.98 | 0.80 | 0.66 | 0.84 | 0.31 | 0.50 | 0.68 |
| **DC-AR** | **1.00** | **0.75** | 0.52 | **0.90** | **0.45** | 0.51 | **0.69** |

### Ablation Study

Effectiveness of hybrid design:

| Configuration | rFID↓ | gFID↓ | GenEval↑ | Throughput |
|---------------|-------|-------|----------|------------|
| DC-AR (hybrid) | 1.60 | 5.50 | 0.69 | 10.3 |
| Discrete-only | 5.13 | 6.71 | 0.66 | 11.4 |

With only 10% additional overhead, gFID improves by 1.21 and GenEval improves by 0.03.

Three-stage training strategy vs. alternatives:

| Strategy | Discrete-rFID↓ | rFID↓ |
|----------|----------------|-------|
| Discrete + Alternate Fine-tune | 5.93 | 1.76 |
| Continuous Warm-up + Alternate Train | 6.18 | 1.92 |
| **Three-stage adaptation** | **5.13** | **1.60** |

Tokenizer reconstruction quality (ImageNet 256², 64 tokens):

| Method | Type | rFID↓ | PSNR↑ | SSIM↑ | Cross-resolution? |
|--------|------|-------|-------|-------|------------------|
| TiTok | 1D-Discrete | 1.70 | 17.06 | 0.4021 | ✗ |
| TexTok* | 1D-Continuous | 1.53 | 20.10 | 0.5618 | ✗ |
| **DC-HT** | **2D-Hybrid** | **1.60** | **21.50** | **0.5676** | **✓** |

### Key Findings

- **A 2D tokenizer at 32× compression is for the first time competitive with 1D tokenizers**: DC-HT's rFID is only 0.07 higher than TexTok while supporting cross-resolution generalization.
- **Near-optimal generation in 12 steps**: The discrete-token-dominant design allows DC-AR to substantially reduce sampling steps compared to MAR (which requires 64 steps).
- **Three-stage training is critical**: Direct alternating training causes quality degradation due to conflicts between discrete and continuous representation spaces; stabilizing each component separately before joint fine-tuning is the key to success.
- **Cross-resolution training saves 1.9× GPU hours** while yielding better final quality.

## Highlights & Insights

- **Integration of engineering intuition and principled design**: The authors identify that directly applying VQ to DC-AE causes collapse, and elegantly resolve this through hybrid tokenization and three-stage training.
- **Substantial efficiency advantage**: DC-AR is the first AR method to surpass diffusion model quality while offering a several-fold speed advantage.
- **Clear design philosophy**: Discrete tokens handle structure (few steps); continuous residual tokens handle details (only requiring an MLP head)—each component serves a distinct role.
- **Preserving 2D spatial structure** is a pivotal design decision—trading a modest compression ratio for cross-resolution capability and training efficiency.

## Limitations & Future Work

- GenEval scores for Count and Color Attribution fall short of Show-o, indicating room for improvement in complex compositional semantic understanding.
- T5-base as the text encoder is relatively small (109M parameters), potentially limiting the ceiling of text-image alignment.
- Higher resolutions (e.g., 1024²) and video generation scenarios remain unexplored.
- The number of denoising steps in the diffusion head has not been subjected to detailed ablation.

## Related Work & Insights

- **Key distinction from HART**: HART employs a 16× multi-scale tokenizer with hybrid tokenization; DC-AR achieves 32× compression with a single-scale tokenizer and a three-stage training strategy.
- The efficiency advantage of the MaskGIT paradigm (parallel decoding) becomes more pronounced at higher compression ratios.
- **Broader implication**: The hybrid tokenization paradigm (discrete + continuous residual) is generalizable to other domains requiring efficient tokenization, such as video and 3D generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-stage training strategy and hybrid generation framework are innovative, though the core components are largely combinations of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual evaluation of tokenizer and generator, detailed efficiency analysis, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and excellent figures (particularly Fig. 1's efficiency comparison).
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical solution for efficient AR image generation; high engineering quality from NVIDIA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Spectral Image Tokenizer](spectral_image_tokenizer.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)

</div>

<!-- RELATED:END -->
