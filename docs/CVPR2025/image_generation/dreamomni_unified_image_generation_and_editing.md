---
title: >-
  [Paper Note] DreamOmni: Unified Image Generation and Editing
description: >-
  [CVPR 2025][Image Generation][Unified Generation and Editing] This work builds a unified 2.5B DiT model for text-to-image (T2I) generation and various editing tasks (instruction-based editing, inpainting, drag editing, and reference-based generation). It replaces the text encoder with Qwen2-VL to achieve unified vision-language prompt understanding and constructs a synthetic sticker data pipeline to efficiently create editing training data, achieving SOTA results in both gene…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Unified Generation and Editing"
  - "DiT Architecture"
  - "Synthetic Data"
  - "VLM Encoder"
  - "Multi-Task Model"
date: 2026-05-08
content_hash: d677ac0eb001acd7
---

# DreamOmni: Unified Image Generation and Editing

**Conference**: CVPR 2025  
**arXiv**: [2412.17098](https://arxiv.org/abs/2412.17098)  
**Code**: [https://zj-binxia.github.io/DreamOmni-ProjectPage/](https://zj-binxia.github.io/DreamOmni-ProjectPage/)  
**Area**: Image Generation  
**Keywords**: Unified Generation and Editing, DiT Architecture, Synthetic Data, VLM Encoder, Multi-Task Model

## TL;DR
This work builds a unified 2.5B DiT model for text-to-image (T2I) generation and various editing tasks (instruction-based editing, inpainting, drag editing, and reference-based generation). It replaces the text encoder with Qwen2-VL to achieve unified vision-language prompt understanding and constructs a synthetic sticker data pipeline to efficiently create editing training data, achieving SOTA results in both generation and editing.

## Background & Motivation

**Background**: Image generation (T2I) and image editing (instruction-based editing, inpainting, drag-based editing, etc.) are typically handled by standalone models. Unifying them can share visual knowledge but faces challenges related to multi-task conflicts and data imbalance.

**Limitations of Prior Work**: (1) Editing training data is difficult to acquire, as it requires paired pre- and post-editing images. (2) Prompt formats differ significantly across different editing tasks (text instructions, region masks, drag points, reference images). (3) The UNet architecture suffers from slow convergence during multi-task joint training.

**Key Challenge**: A unified model must perform well in both generation and editing, yet editing tasks require precise spatial understanding while generation tasks require creative divergence.

**Goal**: To design a unified framework that simultaneously handles T2I and various editing tasks, resolving three main issues: data, architecture, and prompt unification.

**Key Insight**: Replacing UNet with DiT (more efficient as computation is concentrated in the $2\times$ downsampled latent space) + using a VLM encoder to uniformly understand diverse prompts + leveraging a synthetic sticker data pipeline to efficiently create editing training pairs.

**Core Idea**: A VLM encoder for unified multi-prompt understanding + DiT with UNet-like residual connections to accelerate convergence + a synthetic sticker data pipeline to address the scarcity of editing data.

## Method

### Overall Architecture
The VLM (Qwen2-VL 7B) encodes multimodal prompts $\rightarrow$ DiT self-attention fuses VLM features and the noisy latent space $\rightarrow$ UNet-like residual connections accelerate convergence $\rightarrow$ Rectified Flow training $\rightarrow$ 3-stage progressive resolution training ($256 \rightarrow 512 \rightarrow 1024$).

### Key Designs

1. **Synthetic Sticker Data Pipeline**:

    - **Function**: To efficiently create precise training data for editing tasks.
    - **Mechanism**: Sticker-based synthesis—treating objects as "stickers" to be added, removed, or replaced on images to generate pre- and post-editing pairs. This supports instruction-based editing (addition/deletion/replacement), drag-based editing (translation/scaling/rotation), reference-based generation, segmentation, etc. It provides ~60M synthetic editing pairs and 125M T2I data.
    - **Design Motivation**: It is $1000\times$ more efficient than manual annotation of editing pairs and covers multiple editing types. Key insight: The goal of editing training is to teach the model "editing semantics" rather than "new concepts."

2. **Replacing Text Encoders with a VLM Encoder**:

    - **Function**: To achieve unified understanding of multimodal prompts, including text, images, and regions.
    - **Mechanism**: Qwen2-VL 7B can simultaneously process text instructions, reference images, and region annotations, outputting a unified conditional embedding. The DiT fuses VLM features with the noisy latent space through self-attention.
    - **Design Motivation**: Traditional T5 or CLIP text encoders cannot process image inputs. Incorporating a VLM unifies prompt understanding for both generation and editing.

3. **DiT + UNet-like Residual Connections**:

    - **Function**: To accelerate convergence in multi-task training.
    - **Mechanism**: Introducing UNet-like skip connections (encoder $\rightarrow$ decoder) between DiT blocks, which speeds up convergence by $4\times$. DiT is more suitable than UNet because its computation is concentrated in the $2\times$ downsampled space.
    - **Design Motivation**: Ablation studies show that models with residual connections achieve lower FID under the same training steps and converge faster.

### Loss & Training
Rectified Flow loss. 3-stage progressive resolution training ($256 \rightarrow 512 \rightarrow 1024$). Joint training of T2I and editing tasks prevents conceptual forgetting.

## Key Experimental Results

### Main Results

| Task | DreamOmni | SOTA Comparison |
|------|-----------|----------|
| GenEval Overall Score | 0.70 | SD3-Medium 0.70 |
| Inpainting FID $\downarrow$ | **0.837** | SD-inp 1.352 |
| Instruction Editing | Outperforms InstructPix2Pix | - |
| Drag-based Editing | Outperforms DragGAN | - |

### Ablation Study

| Configuration | Effect |
|------|------|
| UNet Architecture | DiT is better ($2\times$ downsampling is more efficient) |
| Without UNet-like residuals | $4\times$ slower convergence |
| T2I only | Loss of editing capabilities |
| Editing only | Forgetting of T2I concepts |
| **Joint T2I + Editing** | **Optimal synergy between both** |

### Key Findings
- **Joint training creates synergistic effects**: T2I prevents conceptual forgetting, while editing training improves instruction-following capabilities.
- **Synthetic sticker data is sufficient for training high-quality editing**: Real editing pairs are not necessary; the key is teaching "editing semantics."
- **DiT + residual connections = $4\times$ speedup in convergence**.

## Highlights & Insights
- **The insight that "editing training teaches semantics, not concepts"** liberates data construction—synthetic data is entirely sufficient.
- **Unified prompt understanding via the VLM** enables a single model to handle five editing tasks alongside image generation.

## Limitations & Future Work
- The VLM encoder (7B) increases inference overhead.
- The upper limit of editing quality based on synthetic sticker data is restricted by the synthesis method.
- Evaluation was only conducted at 1024 resolution; higher resolutions remain untested.

## Related Work & Insights
- **vs InstructPix2Pix**: Only performs instruction-based editing, whereas DreamOmni unifies five editing tasks and T2I generation.
- **vs SD3 / FLUX**: Only performs T2I generation, whereas DreamOmni supports editing while maintaining high T2I quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a VLM encoder, synthetic data, and a unified framework is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task evaluations and convergence ablations.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear.
- Value: ⭐⭐⭐⭐ Holds significant engineering value for unified generation and editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](../../ICCV2025/image_generation/univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)
- [\[CVPR 2025\] UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation](unic-adapter_unified_image-instruction_adapter_with_multi-modal_transformer_for_.md)

</div>

<!-- RELATED:END -->
