---
title: >-
  [Paper Note] Lazy Diffusion Transformer for Interactive Image Editing
description: >-
  [ECCV 2024][Image Generation] Proposes LazyDiffusion, an asymmetric encoder-decoder Transformer architecture that compresses global information via a context encoder and executes diffusion denoising only on the masked region, achieving a 10× speedup with image quality comparable to full-image generation methods during interactive image editing.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: e174272985a44324
---

# Lazy Diffusion Transformer for Interactive Image Editing

**Conference**: ECCV 2024  
**arXiv**: [2404.12382](https://arxiv.org/abs/2404.12382)  
**Area**: Image Generation

## TL;DR

Proposes LazyDiffusion, an asymmetric encoder-decoder Transformer architecture that compresses global information via a context encoder and executes diffusion denoising only on the masked region, achieving a 10× speedup with image quality comparable to full-image generation methods during interactive image editing.

## Background & Motivation

Current diffusion-based inpainting methods suffer from severe computational wastage:

**RegenerateImage methods** (e.g., SDXL, SD Inpaint): Generate pixels for the entire image but discard everything except the masked region.

**RegenerateCrop methods** (e.g., cropping schemes commonly used in practical software): Only process a small cropped region around the mask. Although faster, this approach discards global context, leading to semantic inconsistency.

In interactive editing scenarios, user modifications typically cover only 10-20% of the image, making full-image regeneration highly wasteful. This paper introduces a "lazy" generation strategy—only generating the required pixels.

## Method

### Overall Architecture

LazyDiffusion decouples the generation process into two steps:

1. **Global Context Encoder $E$** (ViT): Processes the entire canvas and the mask to extract $N=4096$ tokens, then **retains only** the $N_{hole}$ tokens corresponding to the masked region as the compressed context. The encoder runs only once outside the diffusion loop.
2. **Incremental Diffusion Decoder $D$** (a variant of PixArt-α): At each denoising step, it only processes the tokens corresponding to the masked region, conditioned on both the compressed context and text prompts.

**Key Idea**: The self-attention mechanism in the encoder allows each token to encode global information; thus, discarding non-masked tokens still preserves complete semantic context.

### Key Designs

**Token Dropping**: The $N$ tokens output by the encoder are filtered via a max-pooling mask, retaining only the $N_{hole}$ tokens in the masked region. This creates an information bottleneck, encouraging the encoder to compress global context into the tokens at the masked locations.

**Context Conditioning**: The decoder is conditioned by concatenating the noise tokens $\mathcal{X}_{hole}^t$ and the context tokens $\mathcal{T}_{hole}$ along the hidden dimension: $\mathcal{X}_{hole}^{t-1} = D(\mathcal{X}_{hole}^t \oplus \mathcal{T}_{hole}; t, \mathbf{c})$

**Latent Space Operation**: Operating within a 4-channel latent space using Stable Diffusion's pretrained VAE (8× downsampling). The final output undergoes Poisson blending to eliminate seams.

**Training**: The encoder is trained from scratch, while the decoder is initialized from PixArt-α pretrained weights. They are jointly trained for 100K iterations on 56 A100 GPUs with a batch size of 224.

### Loss & Training

Trained using the Improved DDPM objective function to denoise and reconstruct latent pixels in the masked region.

## Key Experimental Results

### Main Results

Quantitative comparison on the OpenImages 10K dataset (at 1024×1024 resolution):

| Method | CLIP Score ↑ | FID ↓ | Remarks |
|---|---|---|---|
| SD2-crop | 0.21 | 6.95 | Reference, different architecture |
| SDXL | 0.21 | 6.88 | Reference, different architecture |
| RegenerateCrop (PixArt) | 0.19 | 9.35 | Cropping scheme |
| RegenerateImage (PixArt) | 0.19 | 7.38 | Full-image generation |
| **Ours**| **0.19** | **7.70** | Masked region only |

Runtime comparison (10% mask, single A100):

| Method | Execution Mechanism | Latency |
|---|---|---|
| RegenerateImage | Full-image 4096 tokens | ~374ms/step |
| RegenerateCrop | Fixed crop | Fixed |
| **Ours** | Masked tokens only | **~28ms/step (10% mask)** |

User study (1778 responses, 48 users):

| Alternative Method | User Preference for Ours |
|---|---|
| vs RegenerateCrop | **81.0%** |
| vs SD2-crop | **82.5%** |
| vs RegenerateImage | 46.1% |
| vs SDXL | 48.5% |

### Ablation Study

Encoder overhead analysis:

| Component | Time |
|---|---|
| Context Encoder | 73ms (runs only once) |
| Latent Space Encoder | 97ms |
| Latent Space Decoder | 176ms |
| T5 Text Encoder | 21ms |
| Diffusion Decoder (10% mask, single step) | 28ms |
| Diffusion Decoder (Full-image, single step) | 374ms |

### Key Findings

- Achieves a **10× speedup** at 10% mask size, matching the speed of RegenerateCrop at 25% mask size.
- FID increases by only 4% compared to full-image generation (7.70 vs 7.38), but is 26% lower than the cropping method (7.70 vs 9.35).
- Compressed context preserves critical semantic information—performing on par with full-image methods in scenarios demanding high semantic consistency (e.g., generating bread of the same style on a tray).
- The speedup advantage is more pronounced in scenarios with high diffusion steps and high resolutions.

## Highlights & Insights

1. **Precise Problem Definition**: Shifting from "generating full images and then cropping" to "generating only the required pixels," making computation proportional to the size of the edited region.
2. **Reverse MAE Design**: Opposite to Masked Autoencoder—the encoder processes all tokens, whereas the decoder only processes the masked tokens.
3. **Progressive Interaction**: Distributes image generation costs over multiple user interactions, enabling true interactive use of diffusion models.
4. **Orthogonal Contribution**: Orthogonal to fast sampling, distillation, and other diffusion acceleration methods, allowing for cumulative speedups.
5. **Support for Multimodal Conditioning**: In addition to text, it supports local conditioning such as sketch guidance (similar to SDEdit).

## Limitations & Future Work

- The encoder still needs to process the full image (quadratic complexity), which may become a bottleneck for ultra-high-resolution images.
- Subtle color shifts occasionally appear in the generated regions, requiring Poisson blending post-processing.
- The training data consists of an internal dataset (220M images), making it difficult to fully replicate.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The design of asymmetric encoder-decoding + token dropping is both elegant and effective.
- **Value**: ⭐⭐⭐⭐⭐ — The 10× speedup makes diffusion models practical for interactive editing pipelines for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative evaluation, user studies, and progressive generation demos are provided, though comparisons against open-source baselines are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear writing, rich illustrations, with rigorous logic connecting motivation, methodology, and experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)

</div>

<!-- RELATED:END -->
