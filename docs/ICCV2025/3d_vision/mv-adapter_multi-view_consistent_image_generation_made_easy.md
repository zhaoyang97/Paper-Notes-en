---
title: >-
  [Paper Note] MV-Adapter: Multi-view Consistent Image Generation Made Easy
description: >-
  [ICCV 2025][3D Vision][Multi-view Generation] This paper proposes MV-Adapter, the first adapter-based framework for multi-view image generation. By duplicating self-attention layers and adopting a parallel attention architecture, it enables plug-and-play multi-view generation on SDXL at 768 resolution, with compatibility across diverse T2I-derived models.
tags:
  - ICCV 2025
  - 3D Vision
  - Multi-view Generation
  - Adapter
  - diffusion model
  - 3D Generation
  - Texture Generation
date: 2026-05-08
content_hash: 489d36ebca98f85e
---

# MV-Adapter: Multi-view Consistent Image Generation Made Easy

**Conference**: ICCV 2025
**arXiv**: [2412.03632](https://arxiv.org/abs/2412.03632)
**Code**: [Available](https://huanngzh.github.io/MV-Adapter-Page/)
**Area**: 3D Vision / Multi-view Generation
**Keywords**: Multi-view Generation, Adapter, diffusion model, 3D Generation, Texture Generation

## TL;DR

This paper proposes MV-Adapter, the first adapter-based framework for multi-view image generation. By duplicating self-attention layers and adopting a parallel attention architecture, it enables plug-and-play multi-view generation on SDXL at 768 resolution, with compatibility across diverse T2I-derived models.

## Background & Motivation

Multi-view image generation is a fundamental task in 2D/3D content creation. Existing methods (e.g., MVDream, Era3D) suffer from three major limitations:

**High computational cost**: Intrusive modifications to pretrained T2I models followed by full fine-tuning require processing $n$ views simultaneously, making it infeasible to scale to larger base models or higher resolutions.

**Degraded image quality**: High-quality 3D data is scarce, and full-model fine-tuning tends to overfit, leading to generation quality degradation.

**Lack of flexibility**: Modifying the original model architecture precludes compatibility with personalized models, LoRA, ControlNet, and other T2I-derived tools.

**Mechanism**: The adapter paradigm is naturally suited to multi-view generation — it involves fewer parameters, is easier to train, preserves pretrained knowledge, and supports plug-and-play usage. The key challenge lies in effectively encoding 3D geometric knowledge without modifying the original network structure.

## Method

### Overall Architecture

MV-Adapter consists of two core components:
1. **Condition Guider**: Encodes camera parameters or geometric information.
2. **Decoupled Attention Layers**: Comprising multi-view attention and image cross-attention.

At inference time, MV-Adapter can be inserted into any personalized or distilled T2I model to serve as a multi-view generator.

### Key Designs

**1. Condition Guider**

- **Camera conditioning**: Represented via raymaps, encoding the ray origin and direction at each spatial position with the same resolution as the latent representation.
- **Geometry conditioning**: Global representations using position maps and normal maps; position maps provide cross-view point correspondences, while normal maps capture geometric details.
- A lightweight convolutional network extracts multi-scale features that are added to the corresponding levels of the U-Net encoder.

**2. Duplicated Self-Attention Layers**

The core principle is to **preserve the original network structure and feature space unchanged**. Rather than modifying the base model's self-attention, the paper duplicates its structure and weights to create new multi-view attention and image cross-attention layers, with the output projection layers zero-initialized. This ensures that the new layers learn geometric knowledge without interfering with the original model.

**3. Parallel Attention Architecture**

In contrast to a serial organization, MV-Adapter adopts a parallel architecture:

$$f^{self} = \text{SelfAttn}(f^{in}) + \text{MultiViewAttn}(f^{in}) + \text{ImageCrossAttn}(f^{in}, f^{ref}) + f^{in}$$

The key advantage of this design is that the new layers receive the **same input** as the original self-attention layer, making the pretrained weight initialization effective and allowing the new layers to directly inherit image priors. In a serial architecture, the new layers receive inputs in a different domain, rendering such initialization ineffective.

**4. Multi-view Attention Strategies**

- 3D object generation: 0° elevation, row-wise self-attention.
- 3D texture generation: Four 0° views plus two top/bottom views, row-wise and column-wise self-attention.
- Arbitrary-view generation: Full self-attention.

**5. Image Cross-Attention**

A pretrained frozen U-Net serves as the image encoder. The reference image (at timestep $= 0$) is passed through it to extract multi-scale self-attention features, which are then injected into the denoising U-Net.

### Loss & Training

- Standard diffusion training objective; only MV-Adapter parameters are optimized.
- Reference image features are randomly zeroed to support classifier-free guidance.
- Noise schedule is shifted toward higher noise levels: log-SNR shift of $\log(n)$, where $n$ is the number of generated views.
- Training data: a subset of Objaverse.

## Key Experimental Results

### Main Results

**Text-to-multi-view generation:**

| Method | FID↓ | IS↑ | CLIP Score↑ |
|--------|------|-----|-------------|
| MVDream | 32.15 | 14.38 | 31.76 |
| SPAD | 48.79 | 12.04 | 30.87 |
| **Ours (SDXL)** | **29.71** | **16.38** | **33.17** |

**Image-to-multi-view generation:**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| Era3D | 20.890 | 0.8601 | 0.1199 |
| Ouroboros3D | 20.810 | 0.8535 | 0.1193 |
| **Ours (SDXL)** | **22.131** | **0.8816** | **0.1002** |

### Ablation Study

**Training efficiency comparison (batch size = 1):**

| Method | Trainable Params | VRAM | Training Speed |
|--------|-----------------|------|----------------|
| Era3D (SD2.1) | 993M | 36G | 2.2 iter/s |
| Ours (SD2.1) | 127M | 17G | 3.1 iter/s |
| Era3D (SDXL) | 3.1B | >80G | Infeasible |
| Ours (SDXL) | 490M | 60G | 1.05 iter/s |

**Attention architecture ablation:**

| Architecture | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------------|-------|-------|--------|
| Serial (SDXL) | 20.687 | 0.8681 | 0.1149 |
| **Parallel (SDXL)** | **22.131** | **0.8816** | **0.1002** |

### Key Findings

1. **Parallel vs. serial**: The parallel architecture substantially outperforms the serial one (PSNR gain of 1.44); the serial architecture produces artifacts and inconsistent details.
2. **Training efficiency**: Parameter count is only 1/6 that of full fine-tuning (SD2.1), VRAM usage is halved, and Era3D is infeasible on SDXL.
3. **Texture generation**: FID of 27.28 (image-conditioned, SDXL), 24% lower than the best baseline SyncMVD (36.13), with inference in only 33 seconds.
4. **3D reconstruction quality**: Chamfer Distance of 0.0206, significantly outperforming Era3D's 0.0329.

## Highlights & Insights

1. **First introduction of the adapter paradigm**: Applying adapter-based methods to multi-view generation achieves "train once, use everywhere" flexibility.
2. **Elegant parallel attention design**: By sharing the input with the original self-attention, the new layers inherit pretrained weight initialization effectively.
3. **Zero-initialization strategy**: Zero-initializing the output projection of new layers ensures that the original feature space is not disrupted at the start of training.
4. **Decoupled learning paradigm**: Provides a general framework extensible to modeling other types of knowledge, such as physical or temporal priors.

## Limitations & Future Work

1. **Fixed number of views**: Each application currently requires training a separate adapter for a specific number of views.
2. **Room for improvement in 3D consistency**: Post-processing is still required to obtain the final 3D model.
3. **Dependency on training data**: 3D datasets such as Objaverse are still required.
4. **Potential extension to video generation**: The parallel attention architecture may be applicable to temporal consistency modeling.

## Related Work & Insights

- **MVDream**: Replaces self-attention with a 3D variant through intrusive modifications, resulting in incompatibility with T2I-derived models.
- **Era3D**: Achieves efficient multi-view interaction via row-wise self-attention, but requires full fine-tuning.
- **SPAD**: Employs epipolar-constrained cross-attention, with computational cost between dense and row-wise attention.
- **IP-Adapter**: Its decoupled cross-attention design inspired MV-Adapter's image conditioning mechanism.
- **Insight**: In the era of large-scale models, **parameter-efficient fine-tuning is not merely an efficiency concern, but a critical strategy for preserving pretrained priors and enabling flexible compositional use.**

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4.5 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4.5 |
| Practical Value | 5 |
| Overall | 4.5 |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](auto-regressively_generating_multi-view_consistent_images.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)

<!-- RELATED:END -->
