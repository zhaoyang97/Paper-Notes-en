---
title: >-
  [Paper Note] SDMatte: Grafting Diffusion Models for Interactive Matting
description: >-
  [ICCV 2025][Image Generation][Interactive Matting] This paper proposes SDMatte, a Stable Diffusion-based interactive matting model that converts the text interaction capability of diffusion models into visual prompt inte…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Interactive Matting"
  - "Diffusion Model Priors"
  - "Visual Prompts"
  - "Alpha Matte"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: 1636fed3dbe50f1b
---

# SDMatte: Grafting Diffusion Models for Interactive Matting

**Conference**: ICCV 2025
**arXiv**: [2508.00443](https://arxiv.org/abs/2508.00443)
**Code**: [https://github.com/vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)
**Area**: Diffusion Models / Image Matting
**Keywords**: Interactive Matting, Diffusion Model Priors, Visual Prompts, Alpha Matte, Attention Mechanism

## TL;DR

This paper proposes SDMatte, a Stable Diffusion-based interactive matting model that converts the text interaction capability of diffusion models into visual prompt interaction capability via three key designs: visual prompt cross-attention, coordinate/opacity embeddings, and mask self-attention. SDMatte significantly outperforms SAM-based methods across multiple datasets.

## Background & Motivation

**High cost of traditional methods**: Trimap-based matting achieves high accuracy but incurs substantial annotation cost; automatic matting methods perform poorly on non-salient or transparent objects. Interactive matting (point, box, and mask prompts) represents an ideal balance between usability and accuracy.

**Limitations of SAM-based methods**: Methods such as MAM, MatAny, and SEMat rely on frozen SAM to generate coarse masks followed by refinement, but cannot correct SAM's errors, causing error amplification in subsequent modules.

**Potential of diffusion models**: Diffusion models trained on billions of image-text pairs exhibit strong generalization and detail-preservation capabilities (e.g., Marigold achieves excellent depth estimation by fine-tuning solely on synthetic data). However, existing methods typically fine-tune with empty text embeddings, discarding the powerful text interaction capability.

**Core Idea of SDMatte**: Rather than discarding the interaction capability of diffusion models, SDMatte converts text-driven interaction into visual prompt-driven interaction.

## Method

### Overall Architecture

Built upon Stable Diffusion v2 with a single-step deterministic inference paradigm (similar to GenPercept):
1. The VAE encoder maps the input image and visual prompt to the latent space.
2. The concatenated latents are fed into the U-Net (the first convolutional layer weights are doubled to accommodate the additional channels).
3. The VAE decoder maps the output back to pixel space for matting loss computation.

### Key Design 1: Visual Prompt Cross-Attention

The text embeddings in the U-Net's middle block (where semantic information is most concentrated) are replaced with visual prompt embeddings:
- A zero-convolution layer is applied to the latent representation of the visual prompt, projecting it to the same dimension as the text embeddings.
- Zero-convolution initialization ensures the original model is not disrupted at the start of training, progressively converting text interaction capability into visual prompt interaction capability.
- Attention map visualizations confirm that the model accurately focuses on the regions indicated by the visual prompts.

### Key Design 2: Coordinate Embeddings and Opacity Embeddings

Inspired by SDXL's use of image resolution and crop coordinates as U-Net conditioning:

**Coordinate Embeddings**:
- Box prompt: sinusoidal positional encoding is applied to the 4 coordinate values of the top-left and bottom-right corners, yielding $\mathbf{E}_{box} \in \mathbb{R}^{B \times 1280}$.
- Point prompt: $2N$ coordinate values are padded to a fixed length and encoded uniformly, yielding $\mathbf{E}_{point} \in \mathbb{R}^{B \times 1680}$.
- Mask prompt: the minimum bounding box is computed and encoded in the same manner as the box prompt.

**Opacity Embeddings**: sinusoidal encoding of object transparency information (transparent = 0, opaque = 1).

The final conditioning embedding replaces the original time embedding: $\mathbf{E}_{cond} = f_1(\mathbf{E}_{opacity}) + f_2(\mathbf{E}_{coord})$

### Key Design 3: Mask Self-Attention

Inspired by Mask2Former, this design explicitly guides the model to attend to regions indicated by visual prompts:

$$\mathbf{X} = \text{softmax}\left(\mathbf{M} + \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{M}$ is the attention mask:
- Box/Mask prompts: a hard binary mask is generated (indicated region = 1, elsewhere = 0).
- Point prompts: a Gaussian soft mask centered at the point coordinates is generated.
- The $(\mathbf{M}-1) \times \infty$ term effectively suppresses attention in non-indicated regions.

## Key Experimental Results

### Comparison on AIM-500 and AM-2K

| Method | Backbone | Prompt | MSE↓ | MAD↓ | SAD↓ | Grad↓ |
|--------|----------|--------|------|------|------|-------|
| MAM | SAM | point | 0.0752 | 0.1080 | 186.50 | 37.48 |
| MatAny | SAM | point | 0.0425 | 0.0523 | 87.05 | 33.44 |
| SmartMatting | DINOv2 | point | 0.0302 | 0.0388 | 66.27 | 46.63 |
| **SDMatte** | **SD2** | **point** | **0.0109** | **0.0189** | **31.80** | **26.84** |
| MAM | SAM | box | 0.0116 | 0.0222 | 36.66 | 21.04 |
| SmartMatting | DINOv2 | box | 0.0077 | 0.0151 | 25.33 | 27.16 |
| **SDMatte** | **SD2** | **box** | best | best | best | best |

### Ablation Study

| Configuration | MSE | SAD |
|---------------|-----|-----|
| LiteSDMatte (w/o mask self-attention, etc.) | 0.0115 | 34.43 |
| + Visual prompt cross-attention | improved | improved |
| + Coordinate/opacity embeddings | further improved | further improved |
| + Mask self-attention (full SDMatte) | **0.0109** | **31.80** |

### Key Findings

- Under point prompts, SDMatte reduces MSE to 36% of SmartMatting (0.0109 vs. 0.0302), demonstrating the power of diffusion priors.
- Under box prompts, SDMatte surpasses all methods including SEMat (SAM2).
- Visual prompt cross-attention effectively inherits the text interaction capability — attention maps precisely focus on the target region.
- Coordinate and opacity embeddings yield particularly significant improvements for transparent object matting.

## Highlights & Insights

1. **Paradigm innovation**: Converting diffusion model text interaction capability into visual prompt interaction, rather than simply discarding it.
2. **Transparent object handling**: Opacity embeddings represent a unique design tailored to the matting task.
3. **Strong extensibility**: The framework supports three prompt types — point, box, and mask.

## Limitations & Future Work

- The single-step deterministic paradigm is efficient but forgoes the stochastic advantages of diffusion models.
- Information loss introduced by VAE encoding and decoding may affect fine edge reconstruction.
- The approach relies on SD2 pre-trained weights; transferability to newer diffusion architectures (e.g., DiT) remains unverified.

## Related Work & Insights

- **Interactive matting**: MAM, MatAny, SmartMatting, SEMat
- **Diffusion model-based visual perception**: Marigold, GenPercept, DiffDIS
- **Trimap-based methods**: DIM, IndexNet

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of converting text interaction into visual prompt interaction is elegant.
- Technical Depth: ⭐⭐⭐⭐ — The three components are well-motivated and complementary.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across multiple datasets and prompt types.
- Value: ⭐⭐⭐⭐ — Strong fine-edge detail preservation, suitable for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GameFactory: Creating New Games with Generative Interactive Videos](gamefactory_creating_new_games_with_generative_interactive_videos.md)
- [\[ICCV 2025\] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [\[ICCV 2025\] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](../../CVPR2026/image_generation/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)

</div>

<!-- RELATED:END -->
