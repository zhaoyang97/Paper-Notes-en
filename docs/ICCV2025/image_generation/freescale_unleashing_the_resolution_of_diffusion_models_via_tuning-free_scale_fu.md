---
title: >-
  [Paper Note] FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion
description: >-
  [ICCV 2025][Image Generation][High-resolution generation] This paper proposes FreeScale, a tuning-free inference paradigm that extracts and fuses information from different receptive field scales via a Scale Fusion mechanism (global high-frequency + local low-frequency), combined with tailored self-cascade upscaling and restrained dilated convolution, achieving for the first time text-to-image generation at 8K resolution on a single A800 GPU, while also supporting high-resolution video generation.
tags:
  - ICCV 2025
  - Image Generation
  - High-resolution generation
  - diffusion models
  - training-free
  - scale fusion
  - frequency decomposition
date: 2026-05-08
content_hash: 1fdd7988de2353fa
---

# FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion

**Conference**: ICCV 2025
**arXiv**: [2412.09626](https://arxiv.org/abs/2412.09626)
**Code**: [http://haonanqiu.com/projects/FreeScale.html](http://haonanqiu.com/projects/FreeScale.html)
**Area**: Image Generation
**Keywords**: High-resolution generation, diffusion models, training-free, scale fusion, frequency decomposition

## TL;DR

This paper proposes FreeScale, a tuning-free inference paradigm that extracts and fuses information from different receptive field scales via a Scale Fusion mechanism (global high-frequency + local low-frequency), combined with tailored self-cascade upscaling and restrained dilated convolution, achieving for the first time text-to-image generation at 8K resolution on a single A800 GPU, while also supporting high-resolution video generation.

## Background & Motivation

Visual diffusion models are typically trained at limited resolutions (e.g., SDXL at 1024²); direct inference at higher resolutions leads to severe object repetition artifacts. Due to the scarcity of high-resolution training data and prohibitive training costs, tuning-free high-resolution generation has become an active research topic.

**Hierarchical analysis of existing methods**:
- **SDXL Direct Inference (DI)**: Produces large numbers of repeated complete objects, with entirely collapsed visual structure.
- **ScaleCrafter**: Enlarges the receptive field via dilated convolutions to address global repetition, but introduces local repetition (e.g., multiple eyes/noses).
- **DemoFusion**: Nearly eliminates local repetition by fusing local and global patches, but displaces the redundant signal to the background, causing small-object repetition.
- **FouriScale**: Removes high-frequency signals in the frequency domain to eliminate all repetition, but aggressive frequency editing introduces color and texture anomalies.

**Key Challenge**: When a model generates content beyond its training resolution, high-frequency information inevitably increases, causing error accumulation and various forms of repetition. Existing methods each address part of the problem but introduce new side effects.

**Core Idea**: Self-attention features extracted at different receptive field scales are complementary—global attention correctly aggregates the spatial locations of high-frequency signals, while local attention enhances local detail quality. Frequency-aware fusion can capture the advantages of both.

## Method

### Overall Architecture

FreeScale consists of three components:
1. **Tailored Self-Cascade Upscaling**: Starting from the training resolution, progressively generates higher resolutions via noise-and-denoise cycles.
2. **Restrained Dilated Convolution**: Applies dilated convolutions only in downsampling and middle blocks to enlarge the receptive field.
3. **Scale Fusion**: Computes both global and local self-attention within the attention layers and fuses them according to frequency components.

### Key Designs

1. **Tailored Self-Cascade Upscaling**:

    - Function: Provides a well-structured visual layout as the foundation for high-resolution generation.
    - Mechanism: A base image $z_0^r$ is first generated at training resolution, decoded via VAE, upsampled to the target resolution, re-encoded, and then noised to timestep $K$ before denoising: $\tilde{z}_K^{2r} \sim \mathcal{N}(\sqrt{\bar{\alpha}_K} \phi(z_0^r), \sqrt{1-\bar{\alpha}_K} \mathbf{I})$
    - Detail control: A scaled cosine decay factor $c = ((1 + \cos(\frac{T-t}{T}\pi))/2)^\alpha$ is introduced to blend the intermediate latent and the denoised latent: $\hat{z}_t^r = c \times \tilde{z}_t^r + (1-c) \times z_t^r$. Here $\alpha$ can be a spatially varying 2D tensor, allowing different levels of detail to be specified for different semantic regions.
    - Design Motivation: The low-resolution intermediate result determines the overall layout, while the high-resolution stage is responsible only for adding fine details. RGB-space upsampling introduces a degree of blur that suppresses spurious high-frequency content (beneficial for images, but detrimental for video).

2. **Restrained Dilated Convolution**:

    - Function: Enlarges the convolutional receptive field to prevent object repetition.
    - Mechanism: Unlike ScaleCrafter, dilated convolutions are applied **only in the downsampling and middle blocks**, and **not in the upsampling blocks** (where dilation introduces cluttered textures). Standard convolutions are also restored in the final timesteps, when only fine details are rendered and the structure is already fixed.
    - Design Motivation: ScaleCrafter's application of dilation across all layers leads to texture artifacts. By constraining the scope of application, the repetition-suppression benefit of dilated convolutions is preserved while side effects are avoided.

3. **Scale Fusion**:

    - Function: Fuses complementary information from global and local self-attention to eliminate all forms of repetition.
    - Mechanism: Within each self-attention layer, global attention $h_{out}^{global} = \text{SelfAttention}(h_{in})$ and local attention (computed by partitioning $h_{in}$ into overlapping patches, processing each independently, then reassembling) are computed simultaneously. Frequency-decomposed fusion is then performed using Gaussian blurring $G$:

    $h_{out}^{fusion} = \underbrace{h_{out}^{global} - G(h_{out}^{global})}_{\text{global high-freq.}} + \underbrace{G(h_{out}^{local})}_{\text{local low-freq.}}$

    - Design Motivation: The high-quality details in local attention correspond to low-frequency semantics (object layout within local regions), but their high-frequency signals are misplaced, causing small-object repetition. Global attention correctly localizes high-frequency signals. Therefore, taking global high-frequency + local low-frequency yields the optimal result.

### Loss & Training

- Entirely training-free; only inference-time strategies are involved.
- The kernel size of Gaussian blur $G$, which controls the high/low-frequency boundary, is a tunable hyperparameter.
- RGB-space upsampling is used for image generation; latent-space upsampling is used for video generation.
- All experiments are conducted on a single A800 GPU.
- The framework can be further extended to local semantic editing by injecting different text semantics in cross-attention using semantic masks derived from the 1× intermediate result.

## Key Experimental Results

### Main Results (SDXL Image Generation Quality Metrics)

| Method | Resolution | FID↓ | KID↓ | FID_c↓ | KID_c↓ | IS↑ | Time (min) |
|--------|------------|------|------|--------|--------|-----|------------|
| SDXL-DI | 2048² | 64.31 | 0.008 | **31.04** | **0.004** | 10.42 | 0.648 |
| ScaleCrafter | 2048² | 67.55 | 0.013 | 60.15 | 0.020 | 11.40 | 0.653 |
| DemoFusion | 2048² | 65.86 | 0.016 | 63.00 | 0.024 | 13.28 | 1.441 |
| FouriScale | 2048² | 68.97 | 0.016 | 69.66 | 0.026 | 11.06 | 1.224 |
| **FreeScale** | **2048²** | **44.72** | **0.001** | 36.28 | 0.006 | 12.75 | 0.853 |
| SDXL-DI | 4096² | 134.08 | 0.044 | 42.38 | **0.009** | 7.04 | 5.456 |
| **FreeScale** | **4096²** | **49.80** | **0.004** | **71.37** | 0.029 | **12.57** | 6.240 |

### Ablation Study (Image 4096², contribution of each component)

| Configuration | FID↓ | KID↓ | FID_c↓ | KID_c↓ | IS↑ | Notes |
|---------------|------|------|--------|--------|-----|-------|
| w/o Scale Fusion | 68.12 | 0.012 | 100.07 | 0.037 | 12.42 | Evident local repetition |
| Dilation in upsampling blocks | 67.45 | 0.011 | 98.56 | 0.035 | 12.54 | Cluttered textures |
| Latent-space upsampling | 65.08 | 0.009 | 88.63 | 0.029 | 11.31 | Eye-region artifacts |
| **Full FreeScale** | **49.80** | **0.004** | **71.37** | **0.029** | **12.57** | Best overall |

### Key Findings
- FreeScale achieves FID of 44.72 and 49.80 at 2048² and 4096² respectively, significantly outperforming all baselines (second-best >64).
- Scale Fusion is the most critical component: its removal increases FID from 49.80 to 68.12 (+37%).
- RGB-space upsampling is more beneficial than latent-space upsampling for image generation (FID: 49.80 vs. 65.08), but the opposite holds for video.
- The placement of restrained dilated convolution is critical: applying dilation in upsampling blocks introduces cluttered textures.
- FreeScale is equally effective for video generation: FVD (484.71) substantially outperforms DemoFusion (537.61) and ScaleCrafter (723.76).
- Inference time is comparable to or lower than baselines: 4096² images require only 6.24 minutes.

## Highlights & Insights
- The frequency-decomposed fusion strategy of "global high-frequency + local low-frequency" is remarkably elegant, achieving optimal complementarity between two scales using a simple Gaussian blur operation.
- The hierarchical analysis of repetition patterns in high-resolution generation (global repetition → local repetition → small-object repetition) is logically clear and explains why no single prior method can resolve all failure modes.
- Flexible region-level detail control (spatially varying $\alpha$) and local semantic editing capabilities (Figs. 4–5) demonstrate practical utility.
- Achieving 8K text-to-image generation for the first time represents a compelling milestone.

## Limitations & Future Work
- Validation is limited to UNet-based models (SDXL, VideoCrafter2); DiT-based architectures (e.g., FLUX) employ different self-attention mechanisms and may require adaptation.
- The Gaussian blur kernel size is a fixed hyperparameter; adaptive selection could yield further improvements.
- While 8K inference is feasible on a single GPU, the time overhead remains substantial.
- Scale Fusion introduces additional local attention computation (patch partitioning and independent processing), limiting scalability.
- The set of baselines for video generation is limited (FouriScale was excluded due to incompatibility).

## Related Work & Insights
- ScaleCrafter → DemoFusion → FouriScale → FreeScale forms a clear evolutionary trajectory of tuning-free high-resolution generation methods.
- The patch fusion idea from MultiDiffusion underlies FreeScale's local attention; FreeScale's contribution lies in introducing frequency-aware fusion on top of it.
- The observation that "features at different scales exhibit distinct frequency characteristics" generalizes to other multi-scale processing tasks.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency](free4d_tuning-free_4d_scene_generation_with_spatial-temporal_consistency.md)
- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)

<!-- RELATED:END -->
