---
title: >-
  [Paper Note] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting
description: >-
  [Image Generation] This paper proposes Trans-Adapter, a plug-and-play adapter module that enables diffusion-based image inpainting models to directly process transparent (RGBA) images. It also introduces the LayerBench b…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: fa65d068193cd6bb
---

# Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting

| Info | Content |
|------|------|
| Conference | ICCV2025 |
| arXiv | [2508.01098](https://arxiv.org/abs/2508.01098) |
| Code | [ykdai/trans-adapter](https://ykdai.github.io/projects/trans-adapter) |
| Area | Image Generation / Image Inpainting |
| Keywords | Transparent image inpainting, RGBA, diffusion models, adapter, alpha channel |

## TL;DR

This paper proposes Trans-Adapter, a plug-and-play adapter module that enables diffusion-based image inpainting models to directly process transparent (RGBA) images. It also introduces the LayerBench benchmark and the Alpha Edge Quality (AEQ) metric.

## Background & Motivation

### Core Problem

RGBA transparent images are widely used in film, animation, and game production, yet existing AI inpainting tools (e.g., generative fill) support only RGB images. This forces designers to manually repair transparent images, which is labor-intensive and difficult to maintain visual consistency.

### Two Limitations of Prior Work

**Inconsistency from separate processing**: RGB is first inpainted on a background, then alpha is extracted via matting — the two channels are generated independently, leading to misalignment.

**Poor edge quality**: Matting and segmentation methods introduce jagged edges at transparency boundaries, degrading compositing quality.

### Gap in the Field

- Methods for transparent image/video **generation** exist (LayerDiffuse, Zippo, etc.), but no dedicated transparent image **inpainting** method has been proposed.
- Existing inpainting methods (SD-Inpainting, BrushNet, etc.) are all designed for RGB.
- No benchmark or metric specifically evaluating transparent image inpainting exists.

## Method

### Overall Architecture

The paper decomposes RGBA images into RGB and alpha channels, treating them as a "two-frame video." A T2I model is inflated to process this "video," with a spatial alignment module and cross-domain self-attention introduced to ensure RGB–alpha consistency.

### Key Design 1: Network Inflation

RGBA is decomposed into a padded RGB image and an alpha channel, forming a 5D input tensor $f \in \mathbb{R}^{b \times c \times 2 \times h \times w}$. During inference through the pretrained diffusion model, this is reshaped to $f_r \in \mathbb{R}^{2b \times c \times h \times w}$, stacking RGB and alpha latent representations along the batch dimension.

Original network parameters are **frozen**; only newly introduced modules are trained. All new modules use zero-initialized output projection layers to avoid disrupting pretrained capabilities at the start of training.

### Key Design 2: Spatial Alignment Module

Introduced in the shallow layers of the U-Net, this module enforces spatial synchronization between alpha and RGB at corresponding positions via convolutional layers:

$$f = f_r + \mathcal{Z}_c(\mathbf{ConvBlock}(f_r))$$

Features are reshaped to $f_r \in \mathbb{R}^{b \times 2c \times h \times w}$ (concatenated along the channel dimension), processed by a convolutional block, and added back via a zero-initialized convolution.

### Key Design 3: Cross-Domain Self-Attention

Introduced in the bottleneck layer of the U-Net, this allows masked regions to reference surrounding context during inpainting — particularly important for high-frequency details such as hair:

$$\textbf{self-attention}(f_r) = \text{softmax}\left(\frac{\mathbf{Q}_i\mathbf{K}_i}{\sqrt{D}}\right)\mathbf{V}_i$$

where $f_r \in \mathbb{R}^{b \times 2hw \times c}$. After adding 2D positional embeddings, self-attention is applied, followed by a zero-initialized MLP:

$$f = f_r + \mathcal{Z}_M(\textbf{AttentionBlock}(f_r))$$

### Key Design 4: Alpha Map LoRA

A LoRA module is introduced to enable the model to learn alpha channel inpainting:

- LoRA weights are zero-initialized, learning only the residual required for alpha generation.
- Both RGB and alpha are used as training targets simultaneously.
- Different text prompts distinguish the two: alpha uses "alpha map of [prompt]," while RGB uses the original prompt.

### Two-Stage Training

- **Stage 1**: Only the Alpha Map LoRA is trained, endowing the model with alpha reconstruction capability.
- **Stage 2**: The spatial alignment module, cross-domain self-attention, and Alpha Map LoRA are jointly fine-tuned to achieve aligned RGBA inpainting.

The training loss follows the standard DDPM objective:

$$\mathcal{L} = \mathbb{E}_{\mathcal{E}(x_0),y,\epsilon\sim\mathcal{N}(0,I),t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y), C)\|_2^2\right]$$

### Two Instantiations

1. **SD-Inpainting style**: Extends UNet input channels to encode the mask and masked image.
2. **BrushNet style**: Introduces a trainable inpainting branch.

### Training Data

- 30K in-house high-quality transparent images (purchased from online PNG stock libraries and manually filtered).
- 90% of the MAGICK dataset merged in (150K SDXL-generated transparent images).
- Each image is paired with a text description generated by LLaVA.

## LayerBench Benchmark

### Dataset Composition

800 transparent images:
- **LayerBench-Natural** (400 images): Online PNGs combined with matting datasets (DIM, etc.).
- **LayerBench-Generated** (400 images): 200 high-aesthetic-score images from MAGICK + 200 generated by LayerDiffusion.

A key characteristic: most inpainting masks overlap with alpha boundaries, specifically designed to test RGB–alpha alignment.

### AEQ Metric

Alpha Edge Quality: a lightweight CNN binary classifier taking 8-channel input (white-background and black-background composites + alpha + alpha edge mask) and outputting the probability of low-quality edges:

$$\text{AEQ} = 1 - \frac{1}{|\mathcal{M}_e|}\sum_{(x,y)\in\mathcal{M}_e}\mathcal{F}(I_w, I_b, \alpha, \mathcal{M}_e)_{x,y}$$

AEQ ranges in $[0, 1]$; higher is better.

## Key Experimental Results

### Main Results (SD1.5-Inpainting + Blended Noise Strategy)

| Method | AS↑ | LPIPS↓ | CLIP Sim↑ | AEQ↑ |
|------|-----|--------|-----------|------|
| ZIM (matting) | 6.044 | 0.0526 | 27.040 | 0.9874 |
| U²-Net (segmentation) | 6.007 | 0.0560 | 26.978 | 0.9537 |
| BiRefNet (segmentation) | 6.055 | 0.0515 | 27.049 | 0.9886 |
| **Ours** | **6.097** | **0.0408** | 27.030 | 0.9878 |

Key findings:
- LPIPS is substantially better than all two-stage methods (0.0408 vs. 0.0515–0.0560), indicating better preservation of inpainted regions.
- AEQ is competitive with the best matting method and far superior to segmentation methods (U²-Net: 0.9537).
- Achieves the highest aesthetic score (6.097).

### SDXL Results

| Method | AS↑ | LPIPS↓ | CLIP Sim↑ | AEQ↑ |
|------|-----|--------|-----------|------|
| LayerDiffusion | 6.016 | 0.0642 | 27.097 | 0.9781 |
| ZIM | 6.115 | 0.0461 | 27.111 | 0.9828 |
| BiRefNet | 6.129 | 0.0453 | 27.104 | 0.9859 |
| **Ours** | **6.140** | **0.0434** | **27.134** | **0.9872** |

The method comprehensively outperforms competing approaches on SDXL as well.

### Ablation Study

| Variant | AS↑ | LPIPS↓ | CLIP Sim↑ | AEQ↑ |
|------|-----|--------|-----------|------|
| **Full model** | **6.097** | **0.0408** | 27.030 | **0.9878** |
| w/o MAGICK data | 6.073 | 0.0435 | 27.037 | 0.9873 |
| w/o in-house data | 6.067 | 0.0457 | 27.071 | 0.9881 |
| AnimateDiff substitute | 6.067 | 0.0459 | 27.032 | 0.9872 |
| w/o spatial alignment | 5.542 | — | — | — |

Key findings:
- The spatial alignment module is the most critical component; removing it causes a severe drop in AS.
- The two data sources are complementary; mixing them yields the best results.
- The Trans-Adapter design outperforms direct use of AnimateDiff.

## Highlights & Insights

1. **Pioneer contribution**: The first dedicated transparent image inpainting method, filling an important gap in the field.
2. **Plug-and-play design**: Compatible with multiple architectures including SD1.5, SDXL, and BrushNet, with original parameters frozen to maintain compatibility.
3. **Elegant "video" formulation**: Treating RGB + alpha as a two-frame video is a natural and effective modeling strategy.
4. **Zero-initialization strategy**: Zero-initializing all new module outputs ensures training stability and progressive learning.
5. **AEQ metric**: Addresses the lack of edge alignment evaluation standards for transparent images.

## Limitations & Future Work

- Currently supports only single-layer transparent images; multi-layer RGBA editing is not supported.
- Performance is bounded by the quality of the pretrained diffusion model.
- The AEQ classifier itself may contain inherent biases.
- Resolution is limited to 512/1024; higher-resolution scenarios have not been validated.

## Related Work & Insights

- **LayerDiffuse**: A latent transparency approach capable of RGBA generation but not focused on inpainting.
- **AnimateDiff**: A video generation adapter that inspired the inflation design of Trans-Adapter.
- **BrushNet**: A plug-and-play inpainting branch that Trans-Adapter can seamlessly integrate.
- **ControlNet**: Source of inspiration for conditional control paradigms and the zero-initialization strategy.
- The proposed framework is extensible to transparent video inpainting and multi-layer RGBA editing.

## Rating

⭐⭐⭐⭐ — Strong pioneering contribution with an elegant and practical design. Delivers a complete package of task definition, benchmark, and method, with direct applicability to real-world production workflows (Photoshop/After Effects).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)

</div>

<!-- RELATED:END -->
