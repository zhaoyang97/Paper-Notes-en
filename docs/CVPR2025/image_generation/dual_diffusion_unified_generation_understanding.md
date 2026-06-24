---
title: >-
  [Paper Note] Dual Diffusion for Unified Image Generation and Understanding
description: >-
  [CVPR 2025][Image Generation][multimodal diffusion] Proposes D-DiT (Dual Diffusion Transformer), the first fully end-to-end multimodal diffusion model, which employs continuous flow matching for the image branch and discrete masked diffusion for the text branch to simultaneously train both image generation and text understanding under a unified loss function.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "multimodal diffusion"
  - "unified image-text models"
  - "discrete diffusion"
  - "flow matching"
  - "visual question answering"
date: 2026-05-08
content_hash: 12c0a9cffc25f5c3
---

# Dual Diffusion for Unified Image Generation and Understanding

**Conference**: CVPR 2025  
**arXiv**: [2501.00289](https://arxiv.org/abs/2501.00289)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: multimodal diffusion, unified image-text models, discrete diffusion, flow matching, visual question answering

## TL;DR

Proposes D-DiT (Dual Diffusion Transformer), the first fully end-to-end multimodal diffusion model, which employs continuous flow matching for the image branch and discrete masked diffusion for the text branch to simultaneously train both image generation and text understanding under a unified loss function.

## Background & Motivation

**Background**: Diffusion models dominate the text-to-image field, while autoregressive models excel at vision-language understanding. Can the two be unified into a single bidirectional model?

**Limitations of Prior Work**: Existing multimodal diffusion models either have limited text reasoning capabilities (Unified/UniDiffuser requires an AR decoder to convert diffusion text latents to text), or essentially still rely on autoregressive models for text generation (Show-O, Transfusion).

**Key Challenge**: The lack of an empirically viable discrete text diffusion process previously limited pure diffusion models from language modeling.

**Goal**: Build the first pure-diffusion end-to-end multimodal model that concurrently supports image generation, image captioning, and visual question answering.

**Core Idea**: Jointly train a dual-branch Transformer based on the MM-DiT architecture, utilizing continuous diffusion (flow matching) for the image branch and discrete diffusion (masked diffusion) for the text branch.

## Method

### Overall Architecture

D-DiT is based on the MM-DiT architecture of SD3, featuring a dual-branch Transformer: the image branch predicts the velocity field, and the text branch predicts denoised tokens. During training, image denoising (with clean text) and text denoising (with clean images) are alternatingly performed. During inference, T2I and I2T tasks can be executed separately.

### Key Designs

1. **Joint Image-Text Diffusion Loss**:

    - **Function**: Unify the training of conditional generation for both images and text.
    - **Mechanism**: $L_{dual} = L_{image} + \lambda_{text} L_{text}$, where the image branch uses the flow matching MSE loss and the text branch uses the NELBO loss of masked diffusion. During training, noise is only added to the conditional target side—i.e., during image diffusion, text is clean, and during text diffusion, the image is clean.
    - **Design Motivation**: Simple and elegant, enabling joint optimization of both DiT branches through backpropagation.

2. **Controllable Text Infilling Inference**:

    - **Function**: Enable visual question answering (VQA) tasks.
    - **Mechanism**: In VQA tasks, question tokens remain fixed (no noise added), and masked diffusion sampling is only applied to the token positions of the answer. This leverages the natural text infilling capability of masked diffusion.
    - **Design Motivation**: Previous diffusion models could not perform VQA, whereas masked diffusion allows conditional infilling.

3. **Initialization from Pre-trained SD3**:

    - **Function**: Fast adaptation of text generation capabilities.
    - **Mechanism**: The DiT is initialized with SD3 pre-trained weights, and a linear head is added to the text branch for token prediction. The special `<extra_id0>` token from T5 is used as the mask token, with its embeddings unfrozen during the second stage.
    - **Design Motivation**: Only ~25B text tokens are required to show meaningful text output, demonstrating extremely fast adaptation speed.

### Loss & Training

Three-stage training: (1) Pre-training for 60K steps on Datacomp-1b; (2) Continued training for 200K steps on high-quality understanding data, optionally fine-tuning at 512 resolution; (3) Fine-tuning for 50K steps on LLaVA instruction data. In total, approximately 40M image-text pairs are used.

## Key Experimental Results

### Main Results

- VQA benchmarks: Outperforms Show-O (another unified model) on MME, GQA, and POPE.
- Image generation: Maintains original SD3 performance on GenEval, with improvements in certain color metrics.
- The first pure diffusion model that supports complete VQA.

### Key Findings

- Joint diffusion training does not cause catastrophic forgetting of image generation capabilities.
- Text diffusion does not require text-only training data; image-text pairs are sufficient.
- D-DiT demonstrates fine-grained multimodal understanding capabilities in long-form question answering.

## Highlights & Insights

- First to demonstrate that diffusion models can completely replace autoregressive models for multimodal modeling.
- The loss function design is extremely clean—a weighted sum of two unimodal diffusion losses.
- The ability to quickly adapt from SD3 is impressive.

## Limitations & Future Work

- The model scale and training data are relatively small, leaving a gap compared to state-of-the-art VLMs.
- Text generation requires a large number of diffusion steps (256 steps), making inference speed slower than autoregressive models.
- Unconditional text generation (text-only diffusion) has not yet been explored.

## Rating

- Novelty: 9/10 — The first pure-diffusion multimodal model.
- Technical Depth: 8/10 — Elegant unification of continuous and discrete diffusion.
- Experimental Thoroughness: 7/10 — Limited by model scale.
- Writing Quality: 8/10 — Clear, with comprehensive background introduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
