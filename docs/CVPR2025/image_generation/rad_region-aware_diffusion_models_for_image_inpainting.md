---
title: >-
  [Paper Note] RAD: Region-Aware Diffusion Models for Image Inpainting
description: >-
  [CVPR 2025][Image Generation][Diffusion Models] RAD achieves region-asynchronous generation by assigning distinct noise schedules to different pixels. With minimal structural modifications to the vanilla diffusion model (replacing FC layers with $1\times 1$ convolutions), it achieves state-of-the-art (SOTA) inpainting quality while accelerating inference speed by 100x.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Image Inpainting"
  - "Region-Aware"
  - "Pixel-wise Noise Scheduling"
  - "LoRA Fine-tuning"
date: 2026-05-08
content_hash: 75a945f175c80cac
---

# RAD: Region-Aware Diffusion Models for Image Inpainting

**Conference**: CVPR 2025  
**arXiv**: [2412.09191](https://arxiv.org/abs/2412.09191)  
**Code**: None  
**Area**: Image Generation / Image Inpainting  
**Keywords**: Diffusion Models, Image Inpainting, Region-Aware, Pixel-wise Noise Scheduling, LoRA Fine-tuning

## TL;DR

RAD achieves region-asynchronous generation by assigning distinct noise schedules to different pixels. With minimal structural modifications to the vanilla diffusion model (replacing FC layers with $1\times 1$ convolutions), it achieves state-of-the-art (SOTA) inpainting quality while accelerating inference speed by 100x.

## Background & Motivation

Applications of diffusion models to image inpainting primarily fall into two categories of methods, both of which exhibit significant performance drawbacks:

1. **Hijacking the reverse process of pre-trained models**: Methods such as RePaint and MCG do not require additional training, but they rely on complex nested loops (such as repeated resampling steps), leading to extremely long inference times—100x slower than RAD.

2. **Conditional generation frameworks**: Methods like SmartBrush require auxiliary modules to process conditioning information (text and masks), which increases complexity and computational overhead.

The fundamental limitation of both approaches is that standard diffusion models are designed for global generation by applying a uniform noise schedule to all pixels, making them inherently unsuited for inpainting tasks that only require local region generation.

RAD's key insight: if different pixels are allowed to have varying noise intensities, some regions can be fully denoised (known regions) while others remain noisy (regions to be generated), which naturally simulates the inpainting scenario. Although conceptually simple, this idea requires addressing the details of noise schedule design and network adaptation.

## Method

### Overall Architecture

RAD is an element-wise reformulation of DDPM: the forward process defines an independent noise variance $b_{t,i}$ for each pixel $i$, such that $q(x_{t,i}|x_{t-1,i}) = \mathcal{N}(\sqrt{1-b_{t,i}} x_{t-1,i}, b_{t,i})$. The reverse process predicts noise using the same U-Net, but perceives the noise intensity at each pixel via spatial noise embedding. Training is based on pseudo-realistic inpainting masks generated using Perlin noise and fine-tuned on the pre-trained ADM using LoRA.

### Key Design 1: Spatially-Varying Noise Schedule

- **Function**: Assigns different noise schedules to each pixel to achieve region-asynchronous generation.
- **Mechanism**: The diffusion process is divided into two phases: Phase 1 adds noise only to the pixels inside the mask ($T_1$ steps), and Phase 2 adds noise to the pixels outside the mask ($T_2$ steps), where $T_1 + T_2 = T$. During actual generation, inpainting is completed by simply reversing Phase 1. Training masks are generated using **Perlin noise**, whose smooth and natural patterns simulate diverse real-world inpainting distributions. Mask diversity is achieved by randomly sampling spatial scales and black-and-white conversion thresholds.
- **Design Motivation**: Random noise scheduling for independent pixels lacks spatial patterns and is inconsistent with actual inpainting scenarios, leading to poor training performance. Perlin noise provides mask shapes that are both diverse and natural, successfully avoiding this issue.

### Key Design 2: Spatial Noise Embedding

- **Function**: Enables the denoising network to perceive the noise intensity of each pixel, adapting to spatially-varying noise scenarios.
- **Mechanism**: In standard DDPM, the timestep $t$ is embedded via cos-sin encoding followed by FC layers and added to all pixels of the U-Net feature maps. RAD replaces the FC layers with $1\times 1$ convolutions, using $\bar{b}_{t}$ (the pixel-wise cumulative noise intensity) instead of the scalar $t$ as input. Consequently, each pixel independently obtains its noise intensity conditioning without requiring modifications to any other components.
- **Design Motivation**: The purpose of the original $t$ embedding is to inform the network of the global noise intensity. Due to the varying noise intensities across different pixels in RAD, pixel-wise conditioning information is required. Replacing the FC layer with a $1\times 1$ convolution is the minimal structural modification.

### Key Design 3: LoRA Fine-Tuning and Timestep Inverse Mapping

- **Function**: Leverages pre-trained diffusion models to reduce training overhead.
- **Mechanism**: Direct LoRA fine-tuning is challenging because spatial noise embedding alters the pre-trained model too drastically. The solution is to inversely map $\bar{b}_{t,i}$ back to equivalent (possibly non-integer) DDPM timestep values via linear interpolation, making the input distribution more compatible with the pre-trained model. This enables effective fine-tuning via LoRA, significantly lowering training resource requirements.
- **Design Motivation**: The RAD framework requires retraining, which is a drawback compared to methods that hijack pre-trained models. LoRA combined with timestep inverse mapping resolves this issue.

### Loss & Training

A hybrid loss from iDDPM/ADM is used—a combination of the variational loss (Equation 6) and the simplified loss $L = \sum_{t \geq 1} \mathbb{E}_q[\|\epsilon_t - \epsilon_\theta(x_t, t)\|^2]$ (Equation 7), where all terms are reformulated into element-wise versions.

## Key Experimental Results

### Main Results: Inpainting Performance on FFHQ and LSUN Bedroom

| Method | FFHQ-Box FID↓ | FFHQ-Wide FID↓ | LSUN-Box FID↓ | Inference Speed |
|------|--------------|---------------|--------------|---------|
| LaMa | 27.7 | 23.2 | - | Fast |
| Score-SDE | 30.3 | 29.8 | 23.7 | Slow |
| RePaint | ~High | ~High | ~High | 100x of RAD |
| DDRM | ~Medium | ~Medium | ~Medium | Slow |
| **RAD** | **Best** | **Best/Second Best** | **Best** | **Fastest** |

### Ablation Study: Contribution of Each Component

| Configuration | FID | LPIPS |
|------|-----|-------|
| Without Spatial Noise Embedding | Performance Degrades | Performance Degrades |
| Independent Pixel Random Mask | Poor Performance | Poor Performance |
| Perlin Noise Mask | Best | Best |
| Without LoRA (Train from scratch) | Feasible but Slow | Similar |

### Key Findings

- Inference speed is 100x faster than state-of-the-art (SOTA) diffusion inpainting methods (as it only requires a standard reverse process without nested loops).
- Achieves the best or second-best FID and LPIPS across various mask types on FFHQ and LSUN.
- Inpainting results show no visible boundary effects, even when mask boundaries are sharp.
- Spatial noise embedding and Perlin noise masks are critical to the performance.
- LoRA fine-tuning successfully leverages the pre-trained ADM, significantly reducing training costs.

## Highlights & Insights

1. **Minimalist Reformulation**: By simply changing "uniform noise across all pixels" to "independent noise per pixel," combined with the minimal FC to $1\times 1$ convolution modification, SOTA inpainting is achieved.
2. **100x Acceleration**: By internalizing the inpainting process into the diffusion framework (rather than external manipulation), nested loops are eliminated.
3. **Perlin Noise as Mask Proxy Distribution**: Ingeniously leverages procedural noise from computer graphics as the source of training masks.

## Limitations & Future Work

- Requires retraining (though alleviated by LoRA).
- Current experiments are only conducted at $256 \times 256$ resolution.
- No direct comparison with text-guided inpainting methods (due to different problem settings).
- Future work can extend RAD to models like Stable Diffusion.

## Related Work & Insights

- **RePaint**: Coordinates mask/non-mask regions via resampling steps, but suffers from extremely slow speeds.
- **SmartBrush**: Similarly adds noise only to the inpainting region, but requires auxiliary modules.
- **DiffEdit**: Employs DDIM inversion and text-generated masks, but relies on Stable Diffusion.
- **Insight**: "Simple" variants of diffusion models (element-wise noise scheduling) hold tremendous practical value.

## Rating

⭐⭐⭐⭐ — Conceptually extremely simple and elegantly implemented; the 100x speedup carries significant practical importance. The minimal translation from FC to $1\times 1$ convolutions unlocks entirely new capabilities, demonstrating a profound understanding of the foundational theory of diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)

</div>

<!-- RELATED:END -->
