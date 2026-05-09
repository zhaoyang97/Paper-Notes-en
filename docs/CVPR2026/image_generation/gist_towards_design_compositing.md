---
title: >-
  [Paper Note] GIST: Towards Design Compositing
description: >-
  [CVPR 2026][Image Generation][design compositing] This paper proposes GIST, a training-free identity-preserving image compositing method that achieves style harmonization across multi-source visual elements via cross-attention-guided token injection and Flow Matched latent space initialization, serving as a plug-and-play compositing stage between layout prediction and typography generation.
tags:
  - CVPR 2026
  - Image Generation
  - design compositing
  - identity preservation
  - image harmonization
  - diffusion model
  - graphic design
date: 2026-05-08
content_hash: e46babfec3547497
---

# GIST: Towards Design Compositing

**Conference**: CVPR 2026
**arXiv**: [2604.14605](https://arxiv.org/abs/2604.14605)
**Code**: [abhinav-mahajan10.github.io/GIST/](https://abhinav-mahajan10.github.io/GIST/)
**Area**: Image Generation
**Keywords**: design compositing, identity preservation, image harmonization, diffusion model, graphic design

## TL;DR

This paper proposes GIST, a training-free identity-preserving image compositing method that achieves style harmonization across multi-source visual elements via cross-attention-guided token injection and Flow Matched latent space initialization, serving as a plug-and-play compositing stage between layout prediction and typography generation.

## Background & Motivation

Graphic design requires combining multimodal components — images, text, logos, and more — from disparate sources into visually coherent designs. Existing methods focus either on layout prediction or on complementary element generation, both of which preserve input components as-is and implicitly assume that those components are already stylistically consistent. In practice, elements from different sources often exhibit mismatches in tone, style, and texture, and naive arrangement fails to produce truly harmonious designs. Prior work at most addresses the stylization of text typography, while largely ignoring the compositing of image elements.

## Method

### Overall Architecture

GIST is positioned as a compositing stage between layout prediction and typography generation. Given foreground elements and their predicted positions, it leverages the MLLM architecture of Emu-2: a LLaMA decoder produces stylized tokens and a visual encoder produces identity tokens. Two training-free augmentation techniques are then applied to generate a harmonized background image.

### Key Designs

1. **Cross-Attention-Guided Token Injection**: Identity tokens $T_{auto}$ are obtained via the self-encoding property of the Emu-2 visual encoder, while stylized tokens $T_{gen}$ are produced by LLaMA. The cross-attention (CA) maps of the SDXL UNet are used to compute a foreground/background relevance score for each token: $r_{fg}[i] = \frac{\max(CA[i] \odot \mathbf{m}_{fg})}{\max(CA[i])}$. The Top-N relevant tokens are selected and blended as $T_{final}[\mathcal{S}_{fg}] = (1-\beta_{fg}) \cdot T_{gen} + \beta_{fg} \cdot T_{auto}$, with $\beta_{fg}=0.3$ for foreground and $\beta_{bg}=0.2$ for background. A lightweight single UNet forward pass is performed prior to scoring to obtain the CA maps, which are averaged across all attention layers.

2. **Flow Matched Euler Discrete Sampling Latent Space Initialization**: The VAE-encoded latent of the background canvas is inverted via DDIM inversion to obtain the initial noise latent, providing the diffusion process with a structurally aligned starting point and substantially improving background fidelity.

3. **Sequential Element Compositing**: Multiple visual elements are composited in the order of their predicted layout positions, with the updated canvas after each step serving as the background for the next. The final composited result is passed to the typography prediction module to complete the full design generation pipeline. Both raster image and SVG element types are supported.

### Loss & Training

GIST is a training-free method that relies entirely on the existing capabilities of pretrained Emu-2 and SDXL, performing generative image compositing by manipulating the 64-token bottleneck.

## Key Experimental Results

### Main Results

Compared against naive paste baselines, GIST is integrated into two distinct pipelines — LaDeCo and Design-o-meter:

| Metric | Naive Paste | +GIST | Evaluator |
|--------|-------------|-------|-----------|
| Visual Harmony | Baseline | **Significant Improvement** | LLaVA-OV, GPT-4V |
| Aesthetic Quality | Baseline | **Significant Improvement** | LLaVA-OV, GPT-4V |
| Pairwise Preference | — | **Preferred over Naive Paste** | GPT-4V |

### Key Findings

- A delicate balance between identity preservation and style harmonization is required.
- Cross-attention maps provide spatially precise, token-level control signals.
- Latent space initialization is critical for background fidelity.

## Highlights & Insights

- Positioning "compositing" as the missing link between layout prediction and typography generation is a well-motivated framing.
- Exploiting the architectural bottleneck of an MLLM for training-free manipulation is an elegant design choice.
- The plug-and-play design allows seamless integration with arbitrary existing pipelines.
- Emu-2's 64-token bottleneck is a critical design constraint: the visual encoder and SDXL decoder are jointly trained as an autoencoder, such that direct encoding through the visual encoder yields tokens rich in fine-grained identity information.
- Multiple visual elements are composited sequentially in predicted layout order, with each updated canvas serving as the background for the next step; both raster images and SVGs are supported.

## Limitations & Future Work

- Reliance on Emu-2's 64-token bottleneck limits transferability to more recent models.
- Sequential compositing may introduce sensitivity to element ordering.
- Computational overhead and quality consistency under large numbers of visual elements require further validation.
- Newer models such as FLUX Kontext offer superior generation quality but lack a controllable internal bottleneck, making training-free intervention difficult.
- Integration with both Design-o-meter and LaDeCo pipelines validates the plug-and-play property.
- Flow Matched Euler discrete sampling latent initialization provides a structurally aligned starting point for the diffusion process via DDIM inversion of the background canvas's VAE encoding.
- Both LLaVA-OV and GPT-4V evaluators confirm significant improvements in visual harmony and aesthetic quality.

## Related Work & Insights

- Token-level manipulation for identity-preserving compositing is generalizable to other image editing tasks.
- The background latent space initialization technique offers insights for inpainting and outpainting applications.

## Rating

6/10 — The problem formulation is novel and the method is practical, but reliance on a specific model architecture limits generalizability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)
- [\[CVPR 2026\] PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow](psdesigner_automated_graphic_design_with_a_human-like_creative_workflow.md)
- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](../../ICLR2026/image_generation/rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)

<!-- RELATED:END -->
