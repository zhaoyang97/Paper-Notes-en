---
title: >-
  [Paper Note] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning
description: >-
  [ICCV 2025][Multimodal VLM][VPiT] This paper proposes Visual-Predictive Instruction Tuning (VPiT), which extends a pretrained LLM into a unified model—MetaMorph—capable of both visual understanding and generation via lightweight instruction tuning alone. A key finding is that visual generation ability emerges as a natural byproduct of visual understanding, and the two capabilities mutually benefit each other in an asymmetric manner.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VPiT
  - Unified Multimodal Model
  - Visual Generation
  - Instruction Tuning
  - Continuous Visual Tokens
date: 2026-05-08
content_hash: 2c731505a19ca436
---

# MetaMorph: Multimodal Understanding and Generation via Instruction Tuning

**Conference**: ICCV 2025
**arXiv**: [2412.14164](https://arxiv.org/abs/2412.14164)
**Code**: [Project Page](https://tsb0601.github.io/metamorph)
**Area**: Multimodal / Vision-Language Model
**Keywords**: [VPiT, Unified Multimodal Model, Visual Generation, Instruction Tuning, Continuous Visual Tokens]

## TL;DR

This paper proposes Visual-Predictive Instruction Tuning (VPiT), which extends a pretrained LLM into a unified model—MetaMorph—capable of both visual understanding and generation via lightweight instruction tuning alone. A key finding is that visual generation ability emerges as a natural byproduct of visual understanding, and the two capabilities mutually benefit each other in an asymmetric manner.

## Background & Motivation

Multimodal large language models (MLLMs) have made remarkable progress in visual understanding, yet they typically produce only text tokens. Existing "unified model" approaches—supporting both visual understanding and generation—generally require substantial architectural modifications (e.g., discretizing visual inputs, incorporating diffusion objectives, decoupling understanding and generation modes) along with large-scale pretraining or fine-tuning on billions of image-text pairs.

The success of visual instruction tuning suggests that LLMs already possess considerable intrinsic visual knowledge and require only modest data to activate visual understanding capabilities. This naturally raises the question: do LLMs also harbor intrinsic visual generation ability that can be similarly unlocked through lightweight fine-tuning?

## Method

### Overall Architecture

VPiT is a minimal extension of standard visual instruction tuning. It preserves the paradigm of continuous visual tokens as LLM inputs while additionally training the LLM to **output** continuous visual tokens. The architecture comprises: (1) a pretrained visual encoder (SigLIP) encoding images into $m=64$ continuous tokens; (2) a trainable projection layer for dimension alignment; (3) a pretrained LLM (LLaMA-3.1 8B) predicting both text and visual tokens; (4) a text head trained with cross-entropy loss and a vision head trained with cosine similarity loss. Generated visual tokens are mapped back to pixel space via a separately fine-tuned diffusion model.

### Key Designs

1. **Dual-Head Autoregressive Prediction Architecture**: *Function*—enables the LLM to simultaneously predict discrete text tokens and continuous visual tokens within a single autoregressive framework. *Mechanism*—the original LLM text head is retained; an additional vision head (a projection layer) maps LLM hidden states to the visual encoder's embedding space. Special tokens `<image_start>` / `<image_end>` indicate when to switch to the vision head. Text is trained with cross-entropy loss $\mathcal{L}_{\text{text}} = -\log P(w_t | w_{<t})$, and visual tokens with cosine similarity loss $\mathcal{L}_{\text{vis}} = 1 - \cos(\hat{v}_t, v_t)$. *Design Motivation*—avoids information loss from visual token discretization, preserves the simplicity of instruction tuning, and fully reuses the pretrained LLM's capabilities.

2. **Discovery of Asymmetric Mutual Benefit Between Understanding and Generation**: *Function*—systematic data mixture ablations reveal how understanding and generation data contribute to each capability. *Mechanism*—fixing 200k generation data while varying VQA data from 1M to 7M improves both VQA scores and FID; fixing 1M VQA while varying generation data from 200k to 4M improves FID with marginal VQA gains. *Key Finding*—understanding data contributes **substantially more** to both capabilities than generation data (the heatmap shows more pronounced variation along the VQA axis). *Design Motivation*—provides guidance for data mixture strategy: prioritize understanding data; 200k generation samples suffice for strong performance.

### Loss & Training

- Text tokens: cross-entropy loss (standard next-token prediction)
- Visual tokens: cosine similarity loss (aligned with visual encoder outputs)
- Loss computed only on response tokens; prompt tokens serve as context
- Diffusion decoder fine-tuned separately, conditioned on visual encoder embeddings
- Training data comprises three categories: Visual Understanding (Cambrian-7M + VideoQA), Visual Generation (MetaCLIP ≤5M), and Other Visual data (video prediction, visual thinking, image editing)

## Key Experimental Results

### Main Results

| Model | Base LLM | MMBench | SEED | MMMU | FID (COCO) |
|:---|:---|:---|:---|:---|:---|
| GPT-4V (understanding only) | - | 75.8 | 69.1 | 56.8 | - |
| Stable Diffusion 1.5 (generation only) | - | - | - | - | 9.6 |
| EMU-3 | Trained from scratch | 58.5 | 68.2 | 31.6 | 12.8 |
| Janus | DeepSeek 1.3B | 69.4 | 63.7 | 30.5 | 8.5 |
| Chameleon-7B | Trained from scratch | 35.7 | 27.2 | 28.4 | 26.7 |
| **MetaMorph** | **LLaMA-3.1 8B** | **75.2** | **71.8** | **41.8** | **11.8** |

### Ablation Study

| Configuration | FID ↓ | CLIP Score ↑ | Notes |
|:---|:---|:---|:---|
| Generation data only, 5M | ~40 | Low | Poor results with generation data alone |
| Joint training + 5k generation | ~30 | Medium | Generation capability begins to emerge |
| Joint training + 200k generation | ~15 | High | Stable performance |
| Joint training + 5M generation | ~12 | High | Diminishing marginal returns |
| VQA 1M + Gen 200k | ~18 | Med-High | Baseline |
| VQA 7M + Gen 200k | ~13 | High | Understanding data improves generation |
| VQA 1M + Gen 4M | ~14 | High | Limited gain from more generation data |

### Key Findings

- Visual generation ability emerges as a natural byproduct of understanding; only 200k generation samples are needed to unlock it
- Understanding and generation mutually benefit each other asymmetrically: understanding data improves both capabilities far more effectively than generation data
- General, Vision-Centric, and Text & Chart VQA tasks are highly correlated with generation performance ($\rho > 0.85$); Knowledge VQA shows weak correlation
- MetaMorph leverages the LLM's world knowledge to generate visual content (e.g., recognizing "Chhogori" and generating an image of K2)
- MetaMorph demonstrates implicit reasoning: given a riddle-style prompt, the model reasons first and then generates the correct image (e.g., "the larva of a monarch butterfly after metamorphosis" → generates a butterfly)

## Highlights & Insights

- Minimalist design with profound findings: adding only a projection layer and special tokens suffices to activate the LLM's visual generation capability
- The asymmetric "understanding > generation" relationship is a significant discovery, suggesting that visual understanding training implicitly constructs representations that support generation
- The transfer of LLM knowledge to visual generation (e.g., specialized terminology, reasoning puzzles) demonstrates the unique advantage of unified models over CLIP + diffusion pipelines
- This work provides a clear direction for the community: improving visual understanding automatically enhances generation ability

## Limitations & Future Work

- Generated image quality still depends on an external diffusion decoder; the system is not truly end-to-end
- FID of 11.8 is competitive but still lags behind dedicated generative models (Imagen: FID = 7.3)
- Only 64 visual tokens are used, limiting resolution and fine-grained detail
- Generation diversity and controllability are not evaluated (only FID and CLIP Score are reported)
- Video generation is limited to frame prediction; temporal consistency is not evaluated

## Related Work & Insights

- The LLaVA family's visual instruction tuning demonstrates that LLMs possess intrinsic visual understanding; this paper naturally extends that insight to generation
- Unified models trained from scratch (e.g., Chameleon, EMU-3) are comprehensive but impose enormous data and compute requirements; VPiT offers a more efficient alternative
- Transfusion achieves better generation (FID = 6.7) but requires large-scale pretraining, complementing VPiT's lightweight approach
- Diffusion autoencoders are used for mapping visual tokens to pixel space and represent a direction for further improvement

## Rating

⭐⭐⭐⭐ — The method is elegantly simple, the findings are insightful (asymmetric mutual benefit), and the work offers important guidance for unified multimodal modeling. Generation quality and end-to-end integration remain areas for improvement.
nified multimodal modeling. Generation quality and end-to-end integration, however, leave room for improvement.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)

<!-- RELATED:END -->
