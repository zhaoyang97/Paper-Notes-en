---
title: >-
  [Paper Note] UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation
description: >-
  [CVPR 2025][Image Generation][Unified controllable generation] Based on the MM-DiT architecture, UNIC-Adapter designs a unified image-instruction adapter. Through a cross-attention mechanism and RoPE-enhanced spatial-aware injection, a single SD3 model is enabled to handle 14 conditional image generation tasks, including pixel-level control, subject-driven generation, and style transfer.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Unified controllable generation"
  - "adapter"
  - "MM-DiT"
  - "cross-attention"
  - "rotary position embedding"
date: 2026-05-08
content_hash: 8bf1748a79d67355
---

# UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.18928](https://arxiv.org/abs/2412.18928)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Unified controllable generation, adapter, MM-DiT, cross-attention, rotary position embedding

## TL;DR
Based on the MM-DiT architecture, UNIC-Adapter designs a unified image-instruction adapter. Through a cross-attention mechanism and RoPE-enhanced spatial-aware injection, a single SD3 model is enabled to handle 14 conditional image generation tasks, including pixel-level control, subject-driven generation, and style transfer.

## Background & Motivation

**Background**: Text-to-image (T2I) diffusion models can generate high-quality images, but relying solely on text prompts makes it difficult to precisely control pixel-level layout, object appearance, and global style. ControlNet achieves pixel-level control using parallel encoders, and IP-Adapter achieves content control via CLIP features, but each type of control requires training a separate model.

**Limitations of Prior Work**: (1) Each condition requires a specialized adapter, leading to high training costs and complex deployment; (2) Uni-ControlNet trains local and global conditions in two separate groups, which is not fully unified; (3) OmniGen/Instruct-Imagen are unified but require multi-stage training of the entire model, which is extremely expensive.

**Key Challenge**: The contradiction between unity and efficiency—supporting various condition types requires flexible feature interaction, but efficiently adapting on a pre-trained T2I model (without training the entire model) is challenging.

**Goal**: Design a lightweight adapter based on a pre-trained T2I model (SD3) that freezes the base model parameters and only trains the adapter to achieve unified controllable generation across various condition types.

**Key Insight**: The MM-DiT architecture naturally supports all-attention interaction of multi-modal features and can process both task instructions (text) and condition images (image) simultaneously, making it highly suitable as the backbone for a unified adapter.

**Core Idea**: Use MM-DiT blocks to handle the dual-modal interaction between task instructions and condition images, and then inject the extracted information into the main generation branch through cross-attention with RoPE, achieving unified processing for various condition types.

## Method

### Overall Architecture
UNIC-Adapter is built on top of SD3 (MM-DiT architecture). The input consists of three parts: text prompt (main generation branch), task instructions (e.g., "Generate an image from this edge map"), and conditional images (e.g., Canny edge map, reference style image). Task instructions are mapped to features $Z_{\text{ist}}$ via a CLIP/T5 text encoder, and conditional images are mapped to features $Z_{\text{con}}$ via a VAE. In the $N$ MM-DiT blocks of the adapter, $Z_{\text{ist}}$ and $Z_{\text{con}}$ interact with each other via all-attention. Then, the information extracted by the adapter is injected into the image features $Z_{\text{img}}$ of the main generation branch through cross-attention.

### Key Designs

1. **MM-DiT Dual-Modal Feature Extraction**:

    - **Function**: Enables mutual understanding between task instructions and conditional images, extracting task-related conditional features.
    - **Mechanism**: Maps instructions and conditional images to QKV, then performs all-attention interaction: $Z'_{\text{ist}} = \text{Attn}(Q_{\text{ist}}, [K_{\text{ist}} \| K_{\text{con}}], [V_{\text{ist}} \| V_{\text{con}}])$, and similarly for $Z'_{\text{con}}$. Over $N$ stacked layers, features from both modalities are refined layer by layer.
    - **Design Motivation**: Task instructions specify "how to use" the conditional image (whether for edge control or style transfer). The bidirectional attention of MM-DiT allows the instruction to modulate how conditional features are extracted, enabling "one adapter for multiple tasks."

2. **RoPE-Enhanced Cross-Attention Injection**:

    - **Function**: Injects the conditional information extracted by the adapter into the main generation branch.
    - **Mechanism**: The image features $Z_{\text{img}}$ of the main branch generate queries through a newly introduced linear layer $L_{\text{cross}}^q$, while the adapter outputs $K_{\text{ist}}, K_{\text{con}}$ serve as keys/values. After performing cross-attention, it is added back residually: $Z''_{\text{img}} = Z'_{\text{img}} + \text{Attn}(Q'_{\text{img}}, [K_{\text{ist}} \| K_{\text{con}}], [V_{\text{ist}} \| V_{\text{con}}])$. 2D RoPE (encoding height and width dimensions separately) is applied to Q and K, allowing higher attention scores for adjacent pixels.
    - **Design Motivation**: Pixel-level control tasks require precise spatial correspondences. RoPE provides relative position encoding to ensure spatial alignment between the conditional image and the generated image. Ablation studies show that adding RoPE improves Canny control F1 from 29.27 to 31.32.

3. **Unified Training Strategy**:

    - **Function**: Freezes the SD3 base model and only trains the newly introduced parameters in the adapter.
    - **Mechanism**: The adapter is initialized using pre-trained parameters of SD3 (reducing learning difficulty), and the FFN layers following the attention in both the base model and adapter are frozen. GPT-4o is used to generate 20 synonymous instructions for each task to increase diversity. The total training parameters are approximately 1.2B, trained for 100K steps on 16 H100 GPUs.
    - **Design Motivation**: Initializing from pre-trained parameters allows the adapter to inherit the prior knowledge of multi-modal interaction from MM-DiT, significantly reducing the difficulty of training.

### Loss & Training
Using standard diffusion model denoising loss, training is conducted jointly on a mixture of three types of data: pixel-level control (MultiGen-20M 2.8M images), subject-driven (2.1M pairs), and style transfer (90K images).

## Key Experimental Results

### Main Results

| Task/Method | Canny (F1↑) | HED (SSIM↑) | Seg (mIoU↑) | Depth (RMSE↓) |
|----------|-------------|-------------|-------------|---------------|
| ControlNet (Single-task) | 34.65 | 0.7621 | 32.55 | 35.90 |
| ControlNet++ (Single-task) | 37.04 | 0.8097 | 43.64 | 28.32 |
| UniControl (Multi-task) | 30.82 | 0.7969 | 25.44 | 39.18 |
| OmniGen (Multi-task) | 35.54 | 0.8237 | 44.23 | 28.54 |
| **UNIC-Adapter** (Multi-task) | **38.94** | **0.8369** | **42.89** | **31.10** |

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|------|-------|---------|---------|
| DreamBooth | 0.668 | 0.803 | 0.305 |
| OmniGen | 0.801 | 0.847 | 0.301 |
| **UNIC-Adapter** | **0.816** | 0.841 | **0.306** |

### Ablation Study

| Configuration | Canny F1↑ | HED SSIM↑ | DINO↑ |
|------|-----------|-----------|-------|
| Condition Image only as Key | 29.01 | 0.7767 | 0.769 |
| Instruction only as Key | 22.38 | 0.5599 | 0.694 |
| Without RoPE | 29.27 | 0.7707 | 0.771 |
| Without New Query Layer | 29.98 | 0.7840 | 0.778 |
| **Full Model** | **31.32** | **0.7934** | **0.769** |

### Key Findings
- The conditional image feature $K_{\text{con}}$ is crucial for pixel-level control; removing it drops the Canny F1 sharply from 31.32 to 22.38.
- Injecting both instruction and conditional features simultaneously outperforms injecting either alone, demonstrating the effectiveness of the instruction-guided conditional feature mechanism.
- RoPE brings significant improvement to pixel-level control (F1 from 29.27 $\rightarrow$ 31.32) while having a minor impact on subject-driven generation.
- In subject-driven generation, UNIC-Adapter's DINO score (0.816) outperforms test-time tuning methods such as SuTI (0.741).

## Highlights & Insights
- **MM-DiT's Multi-modal Talent**: Cleverly exploits MM-DiT's inherent capability to handle text-image dual-modality, naturally extending it to instruction-conditional image dual-modal interaction, thereby avoiding the overhead of designing a new architecture.
- **Instruction-driven Unity**: Differentiates different control tasks via text instructions, enabling the same adapter to dynamically switch behaviors. This is more elegant and scalable than designing specialized paths for each condition.
- **RoPE-enhanced Spatial Awareness**: Introducing rotary position encoding in cross-attention is a simple yet effective design that can be transferred to other visual generation tasks requiring spatial correspondence.

## Limitations & Future Work
- Only experimented on SD3 Medium; performance on larger models (such as FLUX) remains to be verified.
- The training parameter count of approximately 1.2B is still substantial; further compressing the adapter parameters is a direction worth exploring.
- Quantitative evaluation of style control is insufficient, reliance is mostly on qualitative results.
- The capability for multi-condition combination (e.g., specifying both edge map + style image simultaneously) has not been fully validated.

## Related Work & Insights
- **vs ControlNet**: ControlNet is single-condition, single-model, whereas UNIC-Adapter is multi-condition, single-model. UNIC-Adapter has surpassed ControlNet in Canny and HED control.
- **vs OmniGen**: OmniGen also performs unified generation but requires training the entire model. UNIC-Adapter only trains the adapter portion, which is more efficient, and achieves a higher DINO score in subject-driven generation.
- This unified paradigm of "instruction + condition" can be generalized to video generation control.

## Rating
- Novelty: ⭐⭐⭐⭐ MM-DiT for unified controllable generation is a reasonable and novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three major categories of tasks, 14 conditions, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear equations and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Provides a unified and efficient solution for controllable generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICLR 2026\] Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](../../ICLR2026/image_generation/mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)

</div>

<!-- RELATED:END -->
