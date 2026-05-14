---
title: >-
  [Paper Note] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs
description: >-
  [Image Generation] This paper proposes PanoLlama, which extends fixed-size visual autoregressive (VAR) models to endless panorama generation via a token redirection strategy…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 4ea8018e74a2dbaf
---

# PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs

> **Conference**: ICCV 2025
> **arXiv**: [2411.15867](https://arxiv.org/abs/2411.15867)
> **Code**: [GitHub](https://github.com/0606zt/PanoLlama)
> **Area**: Panoramic Image Generation · Autoregressive Models · Image Generation
> **Keywords**: panoramic image generation, next-token prediction, LlamaGen, token redirection, training-free

## TL;DR

This paper proposes PanoLlama, which extends fixed-size visual autoregressive (VAR) models to endless panorama generation via a token redirection strategy, enabling training-free next-crop prediction that surpasses joint diffusion methods in coherence, fidelity, and aesthetics.

## Background & Motivation

Panoramic Image Generation (PIG) aims to produce coherent images of arbitrary length, with broad applications in artistic creation, historical restoration, and related domains. Limitations of existing methods:

**Multi-level consistency challenges in Joint Diffusion methods**: Methods such as MultiDiffusion and SyncDiffusion partition the panoramic latent space into cropped patches, denoise them independently, and then merge the results. These approaches rely on heuristic stitching strategies (weighted averaging, gradient guidance, deep attention) and struggle to simultaneously ensure low-level (color, edge) and high-level (semantic, layout) coherence.

**Limited receptive field in inpainting-based methods**: Such methods infer the next crop solely from the immediately preceding one, lacking global layout and structural planning.

**Fixed-size constraints of VAR models**: Autoregressive models such as LlamaGen are naturally suited to sequential generation but are restricted by their training paradigm to fixed-size outputs (e.g., $512\times512$).

**Core Insight**: The essence of panorama generation—progressively extending an image while maintaining multi-level coherence—is naturally aligned with the next-token prediction paradigm. Low-level coherence depends on the continuity between adjacent crops, while high-level coherence requires awareness of global transitions across the entire sequence, precisely the strength of autoregressive models.

## Method

### Theoretical Formulation

The panorama $x'$ is decomposed into an ordered sequence of crops $\{x_i\}$ and modeled as a joint probability distribution:

$$P(x') = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1})$$

Compared to inpainting (conditioning only on $x_{i-1}$) and joint diffusion (conditioning on $x_{i-1}, x_{i+1}$), the autoregressive paradigm leverages information from **all preceding crops**.

### Overall Architecture (Fig. 2)

PanoLlama consists of three components:
1. **Text Encoding**: The text prompt $y$ is encoded into a conditional embedding $s$ via encoder $f_\mathcal{E}$
2. **Next-Crop Prediction**: The token generator $f_\mathcal{G}$ autoregressively generates image tokens, extended via the redirection strategy
3. **Token Decoding**: The concatenated token sequence $V$ is decoded into a panorama by the image tokenizer decoder $f_{\mathcal{T}d}$

### Key Design: Token Redirection (Training-Free)

**Vertical extension**: When the position index $k$ reaches the token limit $p$, it is redirected to $p - r\sqrt{p}$ to restart, using the last $p - r\sqrt{p}$ tokens of $v_1$ as the initial context for $v_2$:

$$v_2 = f_\mathcal{G}(v_{1, r\sqrt{p}}, \ldots, v_{1, p})$$

**Horizontal extension (interleaving)**: The image is extended row by row; the last $\sqrt{p} - c$ tokens of each row $v_{i-1}^j$ serve as the initial context for $v_i^j$, extending $c$ columns at each step:

$$v_i^j = f_\mathcal{G}(v_{i-1, \epsilon(v_{i-1}^j) - \sqrt{p} + c}, \ldots, v_{i-1, \epsilon(v_{i-1}^j)})$$

The extension stride $u = c / \sqrt{p}$ controls the quality–efficiency trade-off.

### Unified View of Existing Methods

| Method | Modeling | Conditioning Scope |
|------|------|---------|
| Inpainting | $P(x_i | x_{i-1})$ | Left neighbor only |
| Joint Diffusion | $P(x_i | x_{i-1}, x_{i+1})$ | Left and right neighbors |
| **PanoLlama** | $P(x_i | x_1, \ldots, x_{i-1})$ | **All preceding crops** |

## Key Experimental Results

### Main Results: Quantitative Comparison (Tab. 1, $512\times5120$ panoramas)

| Method | LPIPS ↓ | TV ↓ | SSIM ↑ | FID ↓ (relative) | CLIP-aesthetic ↑ | Time ↓ |
|------|---------|------|--------|------------|-----------------|--------|
| MultiDiffusion | 0.694 | 0.061 | 0.184 | +3.16 | 6.84 | 1809s |
| SyncDiffusion | 0.582 | 0.058 | 0.263 | +8.75 | 6.94 | 7233s |
| MAD | 0.520 | 0.040 | 0.268 | +23.09 | 6.90 | 1924s |
| StreamMD | 0.637 | 0.055 | 0.257 | +53.50 | 6.75 | 241s |
| **PanoLlama** | **0.410** | **0.021** | **0.305** | **+2.27** | **6.97** | **726s** |

On **coherence** (the primary metric), PanoLlama substantially outperforms all baselines:
- TV improves by **47.50%** over the best baseline MAD
- LPIPS improves by 21.15%
- SSIM improves by 13.81%
- Relative FID degradation is only +2.27 (the lowest), with inference speed 3–10× faster than most methods

### Ablation Study on Extension Stride (Fig. 4)

| Stride $u$ | PanoLlama COH | MAD COH | MultiDiffusion COH |
|----------|--------------|---------|-------------------|
| 1/8 | 0.18 | 0.35 | 0.52 |
| 3/4 | 0.19 | 0.42 | 0.58 |
| 1 (no overlap) | 0.24 | 0.72 | 0.80 |

Key finding: As the stride increases, competing methods suffer sharp quality degradation, whereas PanoLlama maintains consistently low COH scores—demonstrating strong robustness to extension stride and achieving a superior quality–efficiency balance.

### Ablation Study on Panorama Size (Fig. 5)

From $2\times$ to $10\times$ resolution, other PIG methods exhibit pronounced coherence degradation as size increases, while PanoLlama remains stable—effectively handling the challenges of larger panoramas.

### User Study

A large-scale evaluation spanning 1,000 prompts and 2,000 panoramas across 25 themes with 100+ sub-themes. PanoLlama performs particularly well on expansive scenes (seascapes, grasslands) and faces greater challenges on complex, dense scenes (crowds, patterns).

## Highlights & Insights

1. **Paradigm Innovation**: Reframes PIG from joint diffusion to next-crop prediction, which is theoretically superior as it exploits all preceding context.
2. **Training-Free**: Fixed-size VAR models are extended to unlimited panorama generation purely through token redirection, without any additional training.
3. **Rich Applications**: Supports multi-scale extension, mask-free layout control, and multi-guidance synthesis—capabilities unavailable in other PIG methods.
4. **New Benchmark**: Introduces a standardized evaluation dataset comprising 1,000 prompts across 100+ themes.

## Limitations & Future Work

- Due to the fixed token capacity of pretrained VAR models, global dependencies are approximated using only a subset of preceding tokens.
- The image quality of the LlamaGen-based system is inherently bounded by the generative capability of the underlying model.
- The interleaved generation strategy for horizontal extension increases implementation complexity.

## Related Work & Insights

- Joint diffusion: MultiDiffusion, SyncDiffusion, TwinDiffusion, MAD
- Inpainting-based panorama generation: BLD
- Visual autoregression: LlamaGen, VQGAN, MaskGIT

## Rating

- **Novelty**: ★★★★★ — A fundamental reconceptualization of the panoramic generation paradigm
- **Technical Depth**: ★★★★☆ — The token redirection strategy is concise and effective
- **Experimental Thoroughness**: ★★★★★ — Large-scale evaluation, multi-dimensional ablations, and a new benchmark
- **Writing Quality**: ★★★★★ — The unified theoretical perspective is clearly articulated with precise comparative analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

</div>

<!-- RELATED:END -->
