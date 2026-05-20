---
title: >-
  [Paper Note] Generative Video Compression with One-Dimensional Latent Representation
description: >-
  [CVPR 2026][Model Compression][Video Compression] This paper proposes GVC1D, which for the first time replaces the 2D grid latent representation in video compression with a compact 1D token sequence. Combined with a 1D m…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Video Compression"
  - "1D Latent Representation"
  - "Generative Codec"
  - "Long-term Memory"
  - "Token Compression"
date: 2026-05-08
content_hash: 60ebf2edf0a929c7
---

# Generative Video Compression with One-Dimensional Latent Representation

**Conference**: CVPR 2026
**arXiv**: [2603.15302](https://arxiv.org/abs/2603.15302)  
**Code**: [https://gvc1d.github.io/](https://gvc1d.github.io/)  
**Area**: Model Compression
**Keywords**: Video Compression, 1D Latent Representation, Generative Codec, Long-term Memory, Token Compression

## TL;DR

This paper proposes GVC1D, which for the first time replaces the 2D grid latent representation in video compression with a compact 1D token sequence. Combined with a 1D memory module for modeling long-term temporal context, GVC1D achieves over 60% bitrate savings on perceptual quality metrics.

## Background & Motivation

Conventional and neural video codecs typically encode frames as **2D latent grids** (e.g., 2D feature maps or blocks). This paradigm suffers from two fundamental limitations:

**Irreducible spatial redundancy**: The rigid structure of 2D grids forces each image patch to correspond to a fixed number of tokens, allocating the same capacity to simple and complex regions alike, resulting in substantial redundancy.

**Limited temporal modeling**: 2D representations prioritize spatial variation over semantic dynamics, making it difficult to aggregate shared content across frames and limiting the exploitation of long-term context.

Generative video codecs (GVC) improve perceptual quality through powerful generative models but remain constrained by the above limitations of 2D representations. While 1D tokenization has demonstrated its potential for compact semantic compression in image generation (TiTok) and image compression (DLF), it has yet to be applied to video compression.

## Method

### Overall Architecture

GVC1D adopts an encoder–entropy model–decoder architecture. The core innovations are:
- **Encoder**: Encodes the current frame $x_t \in \mathbb{R}^{3 \times H \times W}$ into a small number of 1D latent tokens $y_t$.
- **Entropy model**: An autoregressive Transformer performs probabilistic modeling and arithmetic coding of 1D tokens.
- **Decoder**: Reconstructs the frame $\hat{x}_t$ from 1D tokens.
- **Context model**: Combines short-term context (previous frame features) and long-term context (1D memory).

### Key Designs

1. **ViT-based 1D Tokenization**: The input frame is projected via patch embedding to obtain $E_t \in \mathbb{R}^{D \times (h \cdot w)}$, which is concatenated with learnable 1D latent tokens $L \in \mathbb{R}^{D \times (N \cdot 32)}$ and fed into the encoder. The encoder consists of alternating Local Transformers (parallel processing within windows) and Global Transformers (cross-window global interaction):

    $y_t = \text{Enc}(E_t \oplus L \oplus C)$

   where $C = C_l \oplus C_s$ denotes the combined long- and short-term context. **Key insight**: 1D tokens do not maintain fixed spatial correspondences, allowing them to adaptively attend to semantically salient regions. The token count is decoupled from spatial resolution (only 32 tokens per window vs. $16 \times 16 = 256$ patches in 2D), fundamentally reducing spatial redundancy.

2. **1D Memory Long-term Context Module**: A fixed-size memory state is maintained and operates in two stages:

    - **Update stage**: The memory state is updated using a small number of 1D tokens $\hat{y}_t$.
    - **Readout stage**: Learnable query tokens retrieve long-term context from memory.

   This module is implemented with a simple Transformer architecture. Because 1D tokens are semantically rich and far fewer in number than 2D grids, more information can be stored within the same memory capacity, effectively mitigating information forgetting. Short-term context provides fine-grained structural details, while long-term context supplies global semantics — the two are complementary.

3. **Autoregressive Entropy Model**: An AR Transformer sequentially predicts the probability distribution of quantized 1D tokens $Q(y_t)$. Since the number of 1D tokens per frame is small (only 32), and different windows can be processed in parallel, the computational overhead of the AR model remains manageable. This contrasts sharply with entropy models on 2D grids, which must process $h \times w$ tokens — one to two orders of magnitude more complex for AR modeling.

4. **Decoder Design**: The decoder adopts a symmetric architecture to the encoder. Learnable mask tokens $M \in \mathbb{R}^{D \times (h \cdot w)}$ are concatenated with decoded 1D tokens $\hat{y}_t$ and context $C$, and passed through the decoder to iteratively extract information; a convolutional output head then produces the reconstructed frame:

    $\hat{x}_t = \text{Out}(\text{Dec}(\hat{y}_t \oplus M \oplus C))$

   During decoding, mask tokens progressively "read" information from 1D tokens, recovering complete 2D spatial features.

### Loss & Training

Rate-distortion optimization is employed: $\mathcal{L} = R + \lambda D$, where $R$ is the bitrate and $D$ is the distortion. $\lambda$ is log-uniformly sampled from 8 points in the interval $[0.07, 1.5]$ to train a variable-rate model. Training is conducted on the Vimeo and OpenVid-HD datasets with perceptual loss to enhance visual quality.

## Key Experimental Results

### Main Results

| Dataset | Metric | GVC1D (Ours) | GLC-Video | BD-Rate Savings |
|--------|------|-------------|-----------|-------------|
| HEVC-B | LPIPS | Best | Baseline | **-60.4%** |
| HEVC-B | DISTS | Best | Baseline | **-68.8%** |
| UVG | LPIPS | Best | Baseline | **-66.0%** |
| MCL-JCV | LPIPS | Best | Baseline | **-62.1%** |
| HEVC-B | PSNR | Best | Baseline | **-53.8%** |
| HEVC-B | MS-SSIM | Best | Baseline | **-45.1%** |

### Ablation Study

| Configuration | HEVC-B BD-Rate | UVG BD-Rate | Note |
|------|---------------|-------------|------|
| No AR + No Memory | +67.8% | +67.4% | Baseline configuration |
| AR + No Memory | +20.1% | +40.6% | AR effectively reduces inter-token redundancy |
| AR + 2D Memory | +11.5% | +16.8% | 2D feature memory management offers limited gains |
| AR + 1D Memory (Ours) | 0.0% | 0.0% | 1D memory management achieves best performance |

Token size ablation: 32×16 (count × channels) is the optimal configuration; too few tokens lead to insufficient capacity, while too many increase bitrate.

### Key Findings

- 1D tokens **consistently track the same semantic regions across frames** (e.g., the left foreleg of a horse), even under large motion.
- When new objects appear, attention weights of 1D tokens **dynamically redistribute** to the new content.
- Encoding time is 0.262s and decoding time is 0.207s (1080P on A100), comparable to GLC-Video.

## Highlights & Insights

- **Paradigm innovation**: This is the first work to demonstrate that 1D latent representations outperform conventional 2D grids in video compression, opening a new direction for the field.
- **Elegant redundancy elimination**: Decoupling token count from spatial resolution naturally enables adaptive bitrate allocation.
- The 1D memory design is elegant — leveraging the compactness and semantic richness of 1D tokens, effective long-term context modeling is achieved with a simple Transformer.

## Limitations & Future Work

- With only 32 1D tokens per frame, the information capacity is limited; the current approach is only suitable for **low-bitrate lossy compression**, and the authors explicitly acknowledge it cannot be extended to lossless scenarios.
- The number of tokens is fixed; the possibility of **dynamically adjusting token count** based on frame complexity remains unexplored — simple frames (e.g., static backgrounds) could use fewer tokens, while complex frames (fast motion/scene cuts) may require more.
- The generative decoder may still produce semantically inconsistent **hallucinated details** in certain scenarios; failure cases are not presented in the paper's visual comparisons.
- Validation is limited to 1080p resolution; scalability to 4K and beyond remains uncertain.
- Training data are general-purpose videos (Vimeo + OpenVid-HD); performance on domain-specific videos (e.g., medical imaging, satellite remote sensing) is unknown.

## Related Work & Insights

- **vs. GLC-Video [ECCV24]**: GLC-Video encodes video into a 2D latent grid via VQ-VAE with a generative decoder, constrained by VQ-VAE capacity and 2D structural redundancy. GVC1D uses continuous 1D tokens to entirely bypass the 2D structural limitation, achieving 60–68% BD-Rate reduction.
- **vs. DiffVC**: DiffVC enhances perceptual quality using a pretrained diffusion model but at the cost of high bitrate, failing to fully exploit low-bitrate advantages. GVC1D achieves high perceptual quality at extremely low bitrates through 1D representation combined with long-term context.
- **vs. DCVC-FM/DCVC-RT**: The DCVC series are PSNR-oriented conditional coding frameworks using only short-term context. The 1D Memory concept of GVC1D may be complementary to DCVC-style conditional coding.
- **vs. DLF [Image Compression]**: DLF is the first to apply discrete 1D tokens to image compression, but the discrete format disrupts temporal consistency in video. GVC1D adopts continuous 1D tokens, which are more suitable for video.
- **vs. TiTok/TA-TiTok**: 1D tokenization has already demonstrated the value of compact semantic compression in image generation; GVC1D extends this to video compression and validates its effectiveness.
- Inspiration: Can the flexibility and semantic richness of 1D representations be extended to downstream tasks such as video understanding and action recognition? The semantic tracking property of 1D tokens may be naturally suited for object tracking.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First introduction of 1D latent representation to video compression — a paradigm-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset comparisons, comprehensive ablations, and attention visualizations, though speed–quality Pareto curves are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-designed architecture diagrams, and thorough analysis.
- Value: ⭐⭐⭐⭐⭐ Over 60% bitrate savings, with significant impact on the video compression field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DLF: Extreme Image Compression with Dual-generative Latent Fusion](../../ICCV2025/model_compression/dlf_extreme_image_compression_with_dual-generative_latent_fusion.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)

</div>

<!-- RELATED:END -->
