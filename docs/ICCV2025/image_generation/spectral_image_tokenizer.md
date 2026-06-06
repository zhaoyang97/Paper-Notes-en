---
title: >-
  [Paper Note] Spectral Image Tokenizer
description: >-
  [ICCV 2025][Image Generation][image tokenizer] This paper proposes the Spectral Image Tokenizer (SIT), which tokenizes images in the frequency domain after converting them via the Discrete Wavelet Transform (DWT). The re…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "image tokenizer"
  - "discrete wavelet transform"
  - "autoregressive generation"
  - "coarse-to-fine"
  - "multiscale"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 14bd64b1c4a4e3f9
---

# Spectral Image Tokenizer

**Conference**: ICCV 2025
**arXiv**: [2412.09607](https://arxiv.org/abs/2412.09607)  
**Authors**: Carlos Esteves, Mohammed Suhail, Ameesh Makadia (Google Research)
**Area**: Image Generation
**Keywords**: image tokenizer, discrete wavelet transform, autoregressive generation, coarse-to-fine, multiscale, VQ-VAE

## TL;DR

This paper proposes the Spectral Image Tokenizer (SIT), which tokenizes images in the frequency domain after converting them via the Discrete Wavelet Transform (DWT). The resulting token sequence is naturally arranged in a coarse-to-fine order, enabling capabilities unavailable to conventional raster-scan tokenizers, including multi-resolution reconstruction, progressive generation, text-guided super-resolution, and image editing.

## Background & Motivation

### Core Problem with Existing Methods

Mainstream image tokenizers (e.g., VQGAN, ViT-VQGAN) partition images into spatial patches and arrange them into token sequences following a raster-scan order. This approach suffers from several fundamental problems:

**Unnatural sequence ordering**: Raster scan proceeds row by row from top-left to bottom-right. During autoregressive prediction, the "observed context" is a partial reconstruction of the upper portion of the image—inconsistent with human visual perception (global before local) and detrimental to conditional modeling.

**Fixed resolution**: Patch sizes in ViT-based tokenizers are fixed; doubling the resolution quadruples the sequence length, causing the computational cost of training and inference to escalate dramatically.

**No progressive decoding**: During generation, the first 50% of tokens can only reconstruct the top half of the image, with no coarse preview of the complete image available.

**Underutilization of frequency-domain priors**: Natural image energy is concentrated in low frequencies, and high-frequency details are inherently more compressible, yet spatial tokenizers treat all patches uniformly.

### Starting Point

The authors observe that the **multi-scale decomposition** property of wavelet transforms is naturally aligned with autoregressive generation: low-frequency approximation coefficients correspond to coarse image content, while successive high-frequency detail coefficients capture increasingly fine textures. Encoding DWT coefficients from low to high frequency as tokens makes autoregressive generation equivalent to "generating the global structure first, then progressively adding detail."

## Method

### Overall Architecture

The SIT pipeline is: **Input image → multi-level Haar DWT → per-scale patchification → per-scale linear embedding → Transformer encoder → vector quantization (dual codebook) → Transformer decoder → per-scale back-projection → IDWT reconstruction**.

The generative model AR-SIT appends an autoregressive Transformer on top of SIT to predict quantized discrete codes token by token, with the SIT decoder restoring the image.

### Key Designs

#### 1. Spectral Patchification

An $L$-level Haar DWT is applied to the input image, yielding one set of approximation coefficients and $L$ sets of detail coefficients (horizontal/vertical/diagonal). $S = L + 1$ scales are defined, each allocated a fixed $N$ tokens (with $N = 256$ in experiments):

- **Scale 1** (approximation): The coarsest low-frequency approximation coefficients are divided into $N$ patches, each approximately $32{\times}32{\times}3$.
- **Scale $s$** (details): The H/V/D detail coefficients at the corresponding level are concatenated along the channel dimension, then divided into $N$ patches.

Because higher-frequency scales have larger spatial resolution, the same number of tokens implies larger patches, meaning **high-frequency content is more heavily compressed**—consistent with the spectral statistics of natural images.

Compared to ViT-VQGAN: a $256{\times}256$ image with $8{\times}8$ patches yields 1024 tokens; SIT with 4 scales $\times$ 256 tokens/scale also yields 1024 tokens, but scaling to $512{\times}512$ requires only 1–2 additional scales (+256/+512 tokens) rather than a 4× increase.

#### 2. Approximation-Detail Transformer (ADTransformer)

Since approximation and detail coefficients follow very different distributions (approximations resemble natural images; details resemble zero-mean Gaussians), the Transformer uses **per-scale parameters** in its internal layers:

- Layer Norm and MLP layers use separate parameters for approximation and detail tokens.
- QKV projections in self-attention are shared across all scales to preserve cross-scale interaction.
- The parameter overhead is minimal, as MLP/LN parameters are far smaller than attention parameters.

#### 3. Scale-Causal Attention

A **scale-causal attention mask** is introduced: tokens at scale $s$ can only attend to tokens at scales 1 through $s$ (block lower-triangular form). This design enables:

- **Encoder SC**: encoding inputs at varying resolutions (low-resolution inputs activate only the first few scales).
- **Decoder SC**: decoding partial token sequences into coarse images (progressive decoding).

The mask can be applied independently to the encoder and decoder, supporting different applications:
- Multi-scale reconstruction: SC on both encoder and decoder.
- Coarse-to-fine generation: SC on decoder only (SIT-SCD).
- Image super-resolution: SC on encoder only (SIT-SCE).
- Image editing: SC on encoder only.

#### 4. Dual Codebook Quantization

Encoder output token features are mapped to discrete codes via vector quantization:

- **Approximation codebook** $Q_{\text{approx}}$: for scale-1 approximation tokens.
- **Detail codebook** $Q_{\text{details}}$: for scales 2 through $S$ detail tokens.

Codebook size and feature dimensionality are consistent with the ViT-VQGAN baseline (8192), but because the two codebooks are disjoint, the same code index carries different semantics depending on the scale.

### Loss & Training

The same loss combination as ViT-VQGAN is adopted, computed in the spatial domain (after IDWT reconstruction):

| Loss Term | Weight | Description |
|-----------|--------|-------------|
| L2 reconstruction | 1.0 | Pixel-level mean squared error |
| Perceptual | 0.1 | Feature-level loss from a pretrained network |
| Adversarial | 0.1 | Discriminator guidance for realistic textures |
| Commitment | 0.25 | Stabilizes codebook learning |

Key modifications: the logit-laplace loss is **removed** (subsequent work [Parti] demonstrated it to be harmful), and **spectral normalization** is introduced to address training instability caused by the adversarial loss.

## Key Experimental Results

### Main Results 1: Multi-Scale Image Reconstruction (ImageNet)

| Model | Resolution | LPIPS ↓ | PSNR ↑ | FID ↓ | IS ↑ |
|-------|------------|---------|--------|-------|------|
| ViT-VQGAN | 256² | 0.163 | 23.8 | 1.20 | 194.6 |
| **SIT-4** | 256² | **0.144** | 24.0 | **1.20** | **199.5** |
| **SIT-5** | 256² | **0.135** | **24.5** | **0.97** | **202.3** |
| ViT-VQGAN | 512² | 0.320 | 22.4 | 6.92 | 151.5 |
| **SIT-6** | 512² | **0.239** | **23.1** | **1.74** | **203.7** |

SIT-SC models handle multiple resolutions without retraining:

| Model | Resolution | LPIPS ↓ | PSNR ↑ | FID ↓ |
|-------|------------|---------|--------|-------|
| SIT-SC-5 | 128² | 0.159 | 27.1 | 2.13 |
| SIT-SC-5 | 64² | 0.111 | 31.3 | 1.39 |
| SIT-SC-5 | 32² | 0.029 | 36.8 | 0.31 |

### Main Results 2: Class-Conditional Image Generation (ImageNet 256²)

| Model | Parameters | FID ↓ | IS ↑ |
|-------|------------|-------|------|
| AR-ViT-VQGAN | 650M | 8.37 | 111.8 |
| **AR-SIT-4** | 650M | **6.95** | **138.3** |
| LlamaGen-L | 343M | 4.08 | 198.5 |
| VAR | 310M | 3.30 | 274.4 |
| **AR-SIT-4*** | 350M | **4.06** | 190.9 |

Under a fair comparison (identical architecture and training protocol), AR-SIT-4 reduces FID from 8.37 to 6.95. With improved hyperparameters, AR-SIT-4* achieves 4.06, on par with LlamaGen.

### Main Results 3: Text-Guided Generation (MS-COCO)

| Model | Resolution | FID ↓ | Throughput (imgs/s) ↑ | Memory Efficiency (imgs/GB) ↑ |
|-------|------------|-------|----------------------|-------------------------------|
| Parti350M | 256² | 12.4 | 7.8 | 12.0 |
| AR-SIT-SCD-4 | 256² | 12.6 | 6.5 | 8.0 |
| Parti350M | 64² | 10.5 | 7.6 | 12.0 |
| AR-SIT-SCD-4 | 64² | 11.4 | **24.5** | **16.0** |
| Parti350M | 32² | 5.8 | 7.7 | 7.7 |
| AR-SIT-SCD-4 | 32² | 7.6 | **74.7** | **28.0** |

At low resolutions, AR-SIT offers substantial advantages in throughput and memory efficiency (~10× throughput gain at 32²).

### Ablation Study

| Configuration | Description | FID ↓ |
|---------------|-------------|-------|
| SIT-4 (Haar) | Default | 1.20 |
| SIT-4 (LeGall 5/3) | More complex wavelet | Higher |
| SIT-4 (CDF 9/7) | JPEG2000 wavelet | Higher |
| w/o ADTransformer | All layers share parameters | Higher |
| w/o scale-causal | Dense attention | Slightly lower (but loses multi-scale capability) |

**Key Findings**:
- **Haar wavelet is optimal**: Despite being the simplest wavelet, it outperforms LeGall 5/3 and CDF 9/7 commonly used in compression; the authors attribute this to boundary leakage caused by larger filter supports in longer wavelets.
- Increasing codebook size or sequence length improves reconstruction but degrades generation quality (tokenizer–generator trade-off).
- Text-guided super-resolution FID drops from 12.6 (text only) to 6.2 (conditioned on a low-resolution image).

## Highlights & Insights

1. **Natural alignment between frequency domain and autoregressive generation**: The multi-scale nature of wavelet transforms perfectly matches the autoregressive paradigm of "predicting the unknown from the known"—low-to-high frequency corresponds to coarse-to-fine, which is far more natural than the row-by-row raster scan order.
2. **One tokenizer, multiple applications**: By flexibly combining scale-causal masks on the encoder and decoder, the same framework supports multi-scale reconstruction, progressive generation, super-resolution, and editing without retraining.
3. **Graceful sequence length scaling**: Doubling the resolution requires only one or two additional scales (+$N$ tokens) rather than a 4× increase, which is critical for high-resolution generation.
4. **Practical value of low-resolution previews**: In interactive generation scenarios, multiple coarse candidates can be generated rapidly using only the first 25% of tokens; users select a candidate before the remaining details are completed.
5. **Stable training at 512²**: ViT-VQGAN exhibits severe instability when trained at 512², whereas SIT trains successfully without additional tuning.

## Limitations & Future Work

1. **Text-to-image metrics do not clearly surpass the baseline**: Under fair comparison, AR-SIT text-to-image FID is on par with Parti350M; a better tokenizer does not necessarily yield a better generative model (a finding consistent with observations by Chang et al.).
2. **Small model scale**: Experiments are limited to AR models with 350M–650M parameters, while Parti scales up to 22B; performance at larger scales remains unknown.
3. **Limitations of the Haar wavelet**: Although Haar performs best in these experiments, it lacks desirable theoretical properties such as smoothness and higher-order vanishing moments, and may not be optimal at larger resolutions or in more complex scenarios.
4. **Gap to SOTA**: AR-SIT-4* achieves FID 4.06, approaching LlamaGen but falling short of VAR (3.30), which employs a convolutional tokenizer and additional techniques (AdaLN, attention normalization).
5. **Slightly lower full-resolution throughput**: Because each token must pass through IDWT, full-resolution throughput is lower than that of spatial methods.

## Related Work & Insights

- **VQ-VAE / VQGAN / ViT-VQGAN**: Direct baselines for SIT; the key upgrade is replacing spatial patchification with frequency-domain patchification.
- **VAR / RQ-VAE**: Also multi-scale in spirit, but operating on residuals in latent space rather than the input spectrum; SIT's advantage lies in genuine multi-resolution input/output capability.
- **Dieleman (2024) "Diffusion is spectral autoregression"**: Theoretically argues that diffusion models implicitly generate from low to high frequencies in the spectral domain; SIT achieves this in a literal, explicit sense.
- **LlamaGen / FAR**: More recent AR generation methods with different tokenizer designs; SIT's multi-scale capability is an orthogonal contribution that can be combined with these advances.

**Insight**: Frequency-domain representations offer an underappreciated inductive bias for visual generation. Future work could explore combining SIT with larger-scale AR models, more advanced codebook designs (e.g., FSQ), and video generation.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | Frequency-domain tokenizer is a fresh perspective; the unified multi-application framework is elegantly designed |
| Technical Depth | 4 | Integration of DWT and Transformer is complete and well-motivated; ablations are thorough |
| Experimental Thoroughness | 4 | Validated across multiple tasks (reconstruction/generation/super-resolution/editing) with fair baseline comparisons |
| Writing Quality | 4 | Motivation is clear, figures are well-crafted, problem formulation is precise |
| Value | 3.5 | Progressive generation and multi-resolution capability are practical, but absolute performance does not surpass SOTA |
| **Overall** | **4** | A solid methodology paper that introduces the frequency-domain prior as a new paradigm for autoregressive image generation |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[ICCV 2025\] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](../../ICML2026/image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)

</div>

<!-- RELATED:END -->
