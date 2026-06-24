---
title: >-
  [Paper Note] FlexTok: Resampling Images into 1D Token Sequences of Flexible Length
description: >-
  [ICML 2025][Image Generation][Variable-length Tokenizer] FlexTok is proposed, a tokenizer that resamples 2D images into variable-length, ordered 1D discrete token sequences. It learns hierarchical encoding via nested dropout and utilizes a rectified flow decoder to generate high-quality reconstructions at any token count, achieving autoregressive image generation with FID < 2 using only 8 to 128 tokens on ImageNet.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Variable-length Tokenizer"
  - "1D Token Sequence"
  - "Nested Dropout"
  - "Rectified Flow Decoder"
  - "Coarse-to-fine Generation"
date: 2026-05-08
content_hash: 50ad4cddae3ea921
---

# FlexTok: Resampling Images into 1D Token Sequences of Flexible Length

**Conference**: ICML 2025  
**arXiv**: [2502.13967](https://arxiv.org/abs/2502.13967)  
**Code**: [https://flextok.epfl.ch/](https://flextok.epfl.ch/) (Project Page)  
**Area**: Image Tokenizer / Autoregressive Image Generation  
**Keywords**: Variable-length Tokenizer, 1D Token Sequence, Nested Dropout, Rectified Flow Decoder, Coarse-to-fine Generation

## TL;DR
FlexTok is proposed, a tokenizer that resamples 2D images into variable-length, ordered 1D discrete token sequences. It learns hierarchical encoding via nested dropout and utilizes a rectified flow decoder to generate high-quality reconstructions at any token count, achieving autoregressive image generation with FID < 2 using only 8 to 128 tokens on ImageNet.

## Background & Motivation

**Background**: Autoregressive (AR) image generation has emerged as a mainstream paradigm on par with diffusion models. The core technology is the image tokenizer, which encodes images into discrete token sequences, subsequently predicted by GPT-style Transformers.

**Limitations of Prior Work**:
   - **Redundancy of 2D Grid Tokenizers**: Traditional methods (VQGAN, LlamaGen) encode images into 2D token grids (e.g., 16×16=256 tokens), where many tokens carry highly redundant information (e.g., background regions).
   - **Fixed-length Limitation of TiTok**: TiTok demonstrates the viability of 1D tokenizers, but requires training different models for each compression rate, with the token count being fixed.
   - **Inability to Adapt to Image Complexity**: Simple images (e.g., an apple on a solid background) and complex images (e.g., crowded street scenes) use the same number of tokens, which is highly inefficient.

**Key Challenge**: The contradiction between fixed token counts and the variability of image complexity—wasting tokens on simple images, while lacking sufficient tokens for complex ones.

**Goal**: To design a single model capable of encoding images into arbitrary-length sequences from 1 to 256 tokens, while generating plausible reconstructions at all lengths.

**Key Insight**: Leveraging nested dropout to force the encoder to order tokens by importance (from semantics to details), and utilizing a rectified flow decoder to guarantee high-quality outputs under arbitrary token counts.

**Core Idea**: FlexTok compresses images into ordered 1D token sequences, forming a "visual vocabulary"—where fewer tokens capture coarse semantics and more tokens progressively add fine details.

## Method

### Overall Architecture

Three-stage pipeline:
- **Stage 0**: Train a VAE (similar to SDXL VAE) to compress the image into continuous 2D latent grids (8× downsampling).
- **Stage 1**: FlexTok tokenizer resamples the 2D VAE latents into 1D discrete token sequences.
- **Stage 2**: Train an autoregressive Transformer to generate token sequences, which are then reconstructed into images using the FlexTok decoder.

### Key Designs

1. **ViT Encoder + Register Token Bottleneck**:

    - The encoder is a Vision Transformer taking 2D VAE latent patches as input.
    - Uses 256 **register tokens** as a 1D bottleneck representation.
    - Applies **Finite Scalar Quantization (FSQ)** to the register tokens with levels=[8,8,8,5,5,5], resulting in an effective codebook size of 64,000.
    - The encoder and decoder utilize 2×2 patchification, combining with the 8× downsampling of the VAE to achieve a total downsampling of 16×.
    - **Design Motivation**: The register token mechanism, originating from ViT research, is naturally suited for a 1D information bottleneck. FSQ is more stable than VQ and eliminates concerns regarding codebook collapse.

2. **Nested Dropout for Ordered Encoding**:

    - During training, nested dropout is applied to the register tokens: the first $k$ tokens are randomly kept ($k$ is uniformly sampled from 1 to 256), and the remaining tokens are discarded.
    - This forces the encoder to package the most critical information into the leading tokens.
    - The first few tokens encode high-level semantics (e.g., "Golden Retriever"), while subsequent tokens progressively append finer details (e.g., fur texture, background).
    - **Design Motivation**: Nested dropout offers the most concise way to achieve token ordering—without explicitly defining "importance", the model automatically learns to prioritize global semantics.

3. **Rectified Flow Decoder**:

    - The decoder is not a simple deterministic decoder, but a **rectified flow model**.
    - Input: Noisy VAE latent patches + (randomly masked) register tokens.
    - Prediction: The flow from noise to clean latents.
    - Employs AdaLN-zero to condition both patches and registers on the timestep.
    - Additionally applies the **REPA inductive bias loss** (using DINOv2-L) to speed up convergence.
    - **Design Motivation**: Deterministic decoders output blurry average images when given extremely few tokens (e.g., 1-8). Rectified flow can "imagine" missing details, generating sharp and plausible images at any token count.

### Loss & Training

- **Rectified Flow Objective**: Standard flow matching loss.
- **REPA Loss**: Alignment loss between intermediate decoder features and DINOv2-L features, accelerating semantic learning.
- **Model Scale**: Encoder-decoder depth configurations of d12-d12, d18-d18, and d18-d28, with width=64d.
- Trained at a resolution of 256×256 on ImageNet-1k and DFN-2B.

## Key Experimental Results

### Main Results: ImageNet Class-Conditional Image Generation (1.3B AR Model)

| Method | Token Count | gFID ↓ | Traits |
|------|---------|--------|------|
| LlamaGen (2D grid) | 256 | ~2.2 | Fixed 256 tokens, raster scan |
| TiTok-S-128 | 128 | 1.97 | Fixed 128 tokens |
| TiTok-L-32 | 32 | 2.77 | Fixed 32 tokens, separate model |
| **FlexTok d18-d28** | **8** | **<2** | Single model, 8 tokens |
| **FlexTok d18-d28** | **32** | **<2** | Single model, 32 tokens |
| **FlexTok d18-d28** | **128** | **<2** | Single model, 128 tokens |

FlexTok achieves FID < 2 across the range of 8 to 128 tokens using a **single model**.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| w/o nested dropout | Unordered tokens, poor quality at low token counts | Fails to achieve variable length |
| Deterministic decoder | Extremely blurry at low token counts | Lacks generative capability to compensate for missing information |
| w/o REPA loss | 2-3× slower convergence | DINOv2 inductive bias accelerates semantic learning |
| Small model d12-d12 | Higher rFID | Model capacity affects reconstruction FID, but has little impact on MAE |
| Large model d18-d28 | Significant rFID improvement | A stronger generative decoder is most critical |

### Key Findings
- **Coarse-to-fine Visual Vocabulary**: The initial tokens capture high-level semantics (category, composition), whereas subsequent tokens append fine details (textures, colors).
- **Condition Complexity Dictates Token Count**: ImageNet class-conditional generation is satisfied with 16-32 tokens, whereas open-ended text prompts require up to 256 tokens.
- **Impact of AR Model Scale**: At low token counts (1-8), model scale has negligible impact as coarse semantics are easy to learn, whereas large models perform significantly better with high token counts (128+).
- **Comparison with 2D Tokenizers**: FlexTok performs on par with 2D methods at the 256-token limit, but is vastly more flexible—allowing adaptive reduction of token counts based on task demands.

## Highlights & Insights
- **Paradigm Shift**: Transitioning from "fixed-length raster scanning" to "variable-length coarse-to-fine generation" redefines the thinking behind AR image generation.
- **Single Model, Multi-resolution**: One tokenizer accommodates all compression ratios, radically reducing deployment and training costs.
- **Ingenious Use of Rectified Flow Decoder**: Elegantly converts uncertainty into generative diversity, avoiding the issue of blurry reconstructions.
- **64,000 "Image Prototypes"**: The very first token essentially clusterizes all possible images into 64,000 semantic groups.

## Limitations & Future Work
- Currently only supports 256×256 resolution; higher resolutions (512/1024) require further validation.
- The rectified flow decoder introduces additional inference overhead (requiring multi-step sampling), which is slower than deterministic decoders.
- Token ordering depends on the randomness of nested dropout, which may not be the optimal strategy for information sorting.
- Multimodal extensions such as video and audio have not yet been explored.

## Related Work & Insights
- **TiTok**: The pioneer of 1D tokenizers, but restricted to a fixed length. FlexTok can be regarded as a flexible-length generalization of TiTok.
- **ElasticTok/ALIT/One-D-Piece**: Concurrent, parallel works exploring similar variable-length ideas.
- **REPA**: A representation alignment technique that significantly accelerates tokenizer training.
- **Insights**: The concept of variable-length representation can be extended to video (having much higher temporal redundancy) and 3D generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of a variable-length 1D tokenizer and a rectified flow decoder is elegant and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both class-conditional and text-conditional generation, with comprehensive scaling analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Project page is excellent, with rich visualizations and clear explanations of concepts.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the field of AR image generation, paving the way for adaptive compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Any-Order Flexible Length Masked Diffusion](../../ICLR2026/image_generation/any-order_flexible_length_masked_diffusion.md)
- [\[ICML 2025\] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICML 2026\] Diffusion Differentiable Resampling](../../ICML2026/image_generation/diffusion_differentiable_resampling.md)
- [\[ICML 2025\] Synthetic Perception: Can Generated Images Unlock Latent Visual Prior for Text-Centric Reasoning?](synthetic_perception_can_generated_images_unlock_latent_visual_prior_for_text-ce.md)

</div>

<!-- RELATED:END -->
