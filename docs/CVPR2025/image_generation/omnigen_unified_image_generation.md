---
title: >-
  [Paper Note] OmniGen: Unified Image Generation
description: >-
  [CVPR 2025][Image Generation][Unified Image Generation] The first general-purpose image generation foundation model, composed solely of a VAE and a Transformer. It achieves end-to-end processing of multiple tasks, including text-to-image, image editing, and controllable generation, on top of any interleaved multimodal input.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Unified Image Generation"
  - "Diffusion Models"
  - "Multi-task"
  - "Knowledge Transfer"
  - "Rectified Flow"
date: 2026-05-08
content_hash: 01c6d3a28e88247f
---

# OmniGen: Unified Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2409.11340](https://arxiv.org/abs/2409.11340)  
**Code**: [https://github.com/VectorSpaceLab/OmniGen](https://github.com/VectorSpaceLab/OmniGen)  
**Area**: Image Generation / Unified Models  
**Keywords**: Unified Image Generation, Diffusion Models, Multi-task, Knowledge Transfer, Rectified Flow

## TL;DR

The first general-purpose image generation foundation model, composed solely of a VAE and a Transformer. It achieves end-to-end processing of multiple tasks, including text-to-image, image editing, and controllable generation, on top of any interleaved multimodal input.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Large Language Models (LLMs) have unified language generation tasks and revolutionized human-computer interaction, but the field of image generation still lacks a unified model capable of handling various tasks within a single framework. Currently, each new task requires designing a task-specific module and fine-tuning it (e.g., ControlNet for controllable generation, InstantID for identity preservation, InstructPix2Pix for editing), resulting in cumbersome multi-step workflows. For instance, ControlNet requires extracting conditions beforehand using a detector, and InstantID requires face detection and encoding. The core problem is: can we complete any image generation task end-to-end with a single model and user instructions, similar to how ChatGPT handles language tasks? This paper proposes OmniGen, striving for a minimalist architecture and flexible instruction following, eliminating the need for additional plugins and intermediate steps.

### Goal

**Goal**: ### Overall Architecture

OmniGen contains only two components: a frozen SDXL VAE for image encoding/decoding, and a large Transformer model initialized with Phi-3 for conditional image generation.

## Method

### Overall Architecture

OmniGen contains only two components: a frozen SDXL VAE for image encoding/decoding, and a large Transformer model initialized with Phi-3 for conditional image generation. It supports arbitrary interleaved text and image multimodal inputs, trained via the rectified flow method. During inference, it starts from Gaussian noise, iteratively predicts the target velocity, and finally generates the image using the VAE decoder. The entire architecture does not require auxiliary components such as CLIP text/image encoders.

### Key Designs

1. **Unified Input Representation and Attention Mechanism**: Text is processed with the Phi-3 tokenizer, and images are encoded by the VAE, then linearly embedded into a sequence of visual tokens, wrapped with `<<<img>>>` and `<<</img>>>` special tokens, and inserted into the text token stream. The attention mechanism blends causal attention and bidirectional attention: text tokens use causal attention (left-to-right dependency), but bidirectional attention is used within each image (mutual attention among patches); different images can only attend to previously appeared images/text. Inference can be accelerated using KV-cache.

2. **X2I Dataset Construction**: The first comprehensive image generation dataset (~100M images), unifying all tasks into an interleaved text-image sequence format. It covers text-to-image (Recap-DataComp 56M, LAION-Aesthetic, etc.), multimodal-to-image (image editing, virtual try-on, style transfer, visual-conditioned control, etc.), and subject-driven generation (GRIT-Entity + web images). All tasks use a unified format without task-specific special tokens.

3. **Edit Region Weighted Loss**: In image editing tasks, the difference between input and output is small, making it easy for the model to learn a shortcut of direct copying. By calculating the difference between the latent representations of the input and target images, a significantly higher loss weight $w_{i,j} = \frac{1}{\|\mathbf{x} - \mathbf{x}'\|^2}$ is assigned to changed regions compared to 1 for unchanged regions, guiding the model to focus on the areas that need modification.

### Loss & Training

- Rectified Flow Objective: $\mathcal{L} = \mathbb{E}[\|(\mathbf{x} - \epsilon) - v_\theta(\mathbf{x}_t, t, c)\|^2]$
- Progressive Resolution Training: 256→512→1024→2240→multi-resolution, total of 5 stages
- AdamW optimizer, 104 A800 GPUs
- Only VAE is frozen, other Transformer parameters are fully fine-tuned

## Key Experimental Results

### Main Results

| Capability | Baselines | OmniGen Performance |
|------|---------|-------------|
| Text-to-Image | SD Series, DALL-E, Imagen | Highly Competitive |
| Controllable Gen | ControlNet | End-to-end, no detector required |
| Subject-Driven | InstantID, IP-Adapter | No face detector required |
| Image Editing | InstructPix2Pix | Completed by a single model |
| Classical CV Tasks | Task-specific models | Also supported |

### Ablation Study

| Design Choice | Effect |
|---------|------|
| Without edit region weighting | The editing task output directly copies the input |
| Causal + bidirectional attention | Outperforms purely causal or purely bidirectional attention |
| Progressive resolution | Low-resolution data is efficient, high-resolution improves aesthetic quality |
| Unified X2I training | Significant cross-task knowledge transfer |

### Key Findings

- OmniGen demonstrates strong cross-task knowledge transfer capability, enabling it to handle unseen tasks and domains.
- Unified training fosters the potential for reasoning capabilities and chain-of-thought-like mechanisms.
- Without a CLIP encoder, a single Transformer autonomously encodes all conditional information, enabling interactions across different modalities.
- A minimalist architecture (VAE + Transformer) can cover complex workflows that previously required multiple specialized models.

## Highlights & Insights

- A true "one model for all" philosophy, mirroring the LLM revolution in the NLP domain.
- Minimalist yet fully functional architecture—demonstrating that a sufficiently large Transformer paired with enough data can lead to the emergence of general-purpose capabilities.
- The unified format design of the X2I dataset is a key contribution, allowing diverse tasks to be trained within a single framework.
- The edit-region weighted loss resolves a practical yet subtle training issue (the copying shortcut).

## Limitations & Future Work

- Large parameter size, resulting in higher inference costs compared to specialized models.
- It may underperform compared to meticulously designed specialized models on specific tasks (such as ControlNet for edge-detection conditional control).
- Immense demands for training data scale and computing resources (104 A800 GPUs).
- More efficient architecture designs and knowledge distillation schemes remain to be explored.

## Related Work & Insights

- **vs ControlNet/IP-Adapter**: Require extra plugins and multi-step workflows, whereas OmniGen achieves this end-to-end.
- **vs SD3/DALL-E**: Focus on text-to-image, whereas OmniGen simultaneously supports diverse tasks with multimodal inputs.
- **vs InstructPix2Pix**: Only performs editing, whereas OmniGen treats editing as one of the sub-tasks in a unified framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first truly general-purpose image generation foundation model.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation across multiple tasks and benchmarks, though comparisons on some individual tasks are not thoroughly detailed.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear vision with a concise architectural description.
- **Value**: ⭐⭐⭐⭐⭐ — Significantly simplifies image generation workflows, with a substantial open-source contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation](unic-adapter_unified_image-instruction_adapter_with_multi-modal_transformer_for_.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)

</div>

<!-- RELATED:END -->
