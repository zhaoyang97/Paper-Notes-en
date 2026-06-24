---
title: >-
  [Paper Note] MagicQuill: An Intelligent Interactive Image Editing System
description: >-
  [CVPR 2025][Image Generation][image editing] MagicQuill is proposed as an intelligent interactive image editing system that expresses editing intentions using three types of brushstrokes (add/subtract/color). A dual-branch diffusion plugin (inpainting + control) achieves precise control over edges and colors, while an MLLM guesses intentions in real time to automatically generate prompts, enabling a continuous editing workflow without manual text input.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "image editing"
  - "interactive system"
  - "brushstroke control"
  - "diffusion inpainting"
  - "MLLM"
  - "Draw&Guess"
date: 2026-05-08
content_hash: 826ff7a5101f450e
---

# MagicQuill: An Intelligent Interactive Image Editing System

**Conference**: CVPR 2025  
**arXiv**: [2411.09703](https://arxiv.org/abs/2411.09703)  
**Code**: [GitHub](https://magic-quill.github.io)  
**Area**: Image Generation  
**Keywords**: image editing, interactive system, brushstroke control, diffusion inpainting, MLLM, Draw&Guess

## TL;DR

MagicQuill is proposed as an intelligent interactive image editing system that expresses editing intentions using three types of brushstrokes (add/subtract/color). A dual-branch diffusion plugin (inpainting + control) achieves precise control over edges and colors, while an MLLM guesses intentions in real time to automatically generate prompts, enabling a continuous editing workflow without manual text input.

## Background & Motivation

**Background**: Diffusion models have progressed rapidly in image editing, with various approaches emerging such as text-guided, mask-guided, and layout-guided methods. However, they remain inadequate for fine-grained region-level modifications, such as controlling object shape and color.

**Limitations of Prior Work**:
1. Text-guided editing (e.g., InstructPix2Pix, SmartEdit) is overly casual and lacks precise control over shape and color.
2. Repeatedly entering text prompts for each edit disrupts the creative workflow.
3. Existing sketch editing methods (e.g., SketchEdit) are limited by GANs, possessing insufficient open-domain capabilities.
4. Inpainting methods like BrushNet struggle to align both edges and colors simultaneously.

**Key Challenge**: Users require intuitive and efficient interaction (a few brushstrokes should suffice), whereas the model demands precise, multi-dimensional control signals (structure + color + semantics).

**Key Insight**: Using brushstrokes as a unified interaction interface, an MLLM infers semantics in real time, and a dual-branch diffusion architecture handles structural and color control.

## Method

### Overall Architecture

Three core modules work in synergy:
1. **Editing Processor**: A dual-branch controllable inpainting engine based on diffusion models.
2. **Painting Assistor**: An MLLM (LLaVA) that interprets brushstroke intentions in real time to automatically generate prompts.
3. **Idea Collector**: An intuitive user interface supporting Gradio and ComfyUI.

### Key Designs

**1. Dual-Branch Controllable Image Inpainting (Editing Processor)**
- **Function**: Two trainable branches are added to a frozen Stable Diffusion UNet: an inpainting branch provides pixel-level content-aware inpainting, and a control branch (ControlNet architecture) provides structural guidance.
- **Mechanism**:
    - **Brushstroke signals to control conditions**: The *add brush* overlays new edges onto the edge map, and the *subtract brush* erases regional edges, synthesized as $\mathbf{E}_{cond}$. The *color brush* applies color via alpha blending, then downsamples by 16x and upsamples via nearest-neighbor to obtain the color blocks $\mathbf{C}_{cond}$, restricting color influence to large structures.
    - **Editing region**: The union of the three brushstroke regions is dilated by $p$ pixels to obtain the mask $\mathbf{M}$.
    - **Inpainting branch**: A clone of the UNet (without cross-attention) that takes $[z_t, z_{masked}, \mathbf{m}]$ as input and injects features into the main UNet via zero-convolutions: $F_i \mathrel{+}= w_I \cdot \mathcal{Z}(F^I_i)$.
    - **Control branch**: A ControlNet architecture with conditions $\mathcal{C} = \{\mathbf{E}_{cond}, \mathbf{C}_{cond}\}$, which are injected into the latter half of the main UNet: $F_{\lfloor n/2 \rfloor + i} \mathrel{+}= w_C \cdot \mathcal{Z}(F^C_i)$.
- **Design Motivation**: The dual-branch design does not alter pretrained weights, making it plug-and-play for community-finetuned models. The inpainting branch guarantees out-of-region consistency, while the control branch ensures precise alignment of edges and colors.

**2. Draw&Guess Intention Prediction (Painting Assistor)**
- **Function**: A finetuned LLaVA model infers editing intentions in real time from user brushstrokes and image context to automatically generate text prompts.
- **Mechanism**:
    - A new task, "Draw&Guess", is defined: Given an image with brushstrokes and the bounding box of the brushstrokes, the model outputs a word or phrase describing the user's intent.
    - Dataset construction: Select top-5 masks with high edge density from the DCI dataset -> use BrushNet inpainting to erase region content -> overlay original edge maps to simulate brushstrokes -> keep DCI labels as ground truth (24K+ images, 4.4K categories).
    - Finetune LLaVA via LoRA, training only the low-rank adapters.
    - The *subtract brush* requires no prompt (reconstructed directly), whereas the *color brush* combines color values with regional content recognition.
- **Design Motivation**: This eliminates the cognitive burden of repeatedly typing prompts, enabling a continuous editing flow; the finetuning dataset simulates real brushstroke scenarios to ensure accuracy.

**3. User Interface Design (Idea Collector)**
- **Function**: Provides an integrated interface consisting of prompt areas, toolbars, layer management, a canvas, generation previews, and parameter adjustment.
- **Design Motivation**: Lowers the entry barrier and supports continuous iterative editing; achieves a significantly higher SUS score compared to the baseline (ComfyUI + Painter Node).

### Loss & Training

- Control branch training: Standard denoising score matching loss $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon^c(z_t, \mathcal{C}, t)\|^2]$.
- MLLM finetuning: Maximizes the label likelihood $\max_{\Theta^{lora}} \sum_i \log P(u_i | u_{<i}; \{\Theta^{pt}, \Theta^{lora}\})$.
- The weights of Inpainting and Control ($w_I$ and $w_C$) are adjustable to govern control intensity.

## Key Experimental Results

### Main Results

| Method | Text | Edge | Color | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|---|---|
| SmartEdit | ✓ | ✗ | ✗ | 0.339 | 16.695 | 0.561 |
| SketchEdit | ✗ | ✓ | ✗ | 0.138 | 23.288 | 0.835 |
| BrushNet | ✓ | ✗ | ✗ | 0.082 | 25.455 | 0.893 |
| BrushNet+ControlNet | ✓ | ✓ | ✓ | 0.075 | 25.770 | 0.894 |
| **Ours** | ✓ | ✓ | ✓ | **0.067** | **27.282** | **0.902** |

### Painting Assistor Intention Prediction

| Method | GPT-4 Sim↑ | BERT Sim↑ | CLIP Sim↑ |
|---|---|---|---|
| LLaVA-1.5 | 1.894 | 0.721 | 0.795 |
| LLaVA-Next | 1.941 | 0.716 | 0.794 |
| GPT-4o | 1.976 | 0.684 | 0.790 |
| **Ours** | **2.712** | **0.749** | **0.824** |

### User Study

- Among 30 participants, 86.67% rated the prediction accuracy $\ge 4/5$, and 90% rated the efficiency improvement $\ge 4/5$.
- Average accuracy and efficiency scores are 4.07/5 and 4.37/5, respectively.
- Editing time is reduced by an average of 24.92% on iPad and 19.58% on PC.

### Key Findings

1. **Simultaneous control of edges and colors** is a unique advantage of this method; the naive combination of BrushNet+ControlNet is still inferior to the specially designed dual-branch architecture.
2. The **Draw&Guess task** is effective: finetuned LLaVA surpasses the original LLaVA and GPT-4o across all metrics.
3. **Plug-and-play**: The dual-branch model does not modify the base model weights, making it compatible with community-finetuned models.
4. The systematic design of the user interface (layer management, parameter adjustment) significantly outperforms the baseline in SUS scores.

## Highlights & Insights

- "Draw&Guess" is a novel task definition that shifts the MLLM from passive prompt reception to active intention guessing, presenting a paradigm innovation.
- The transformation of brushstroke signals into control conditions is ingeniously designed: edge map overlay/erasure plus color block downsampling-upsampling.
- The dedicated dataset construction workflow (BrushNet erasure + edge overlay simulating brushstrokes) is key to enabling the MLLM to understand hand-drawn inputs.
- High system engineering maturity: a trinity of the editing core, AI assistant, and UI, ready for immediate use.

## Limitations & Future Work

- Only two types of control, *scribble* and *color*, are supported, excluding reference-guided editing.
- The accuracy of Draw&Guess still has room for improvement (GPT-4 Sim is only 2.71/5).
- It does not support layered image generation or complex typographic editing.
- The granularity of color control is restricted by the 16x downsampling.
- Insufficient support for fine typography.

## Related Work & Insights

- BrushNet provides the foundation for mask-guided inpainting, upon which this work adds a control branch to achieve dual-dimensional precise control.
- ControlNet's zero-convolution injection mechanism is elegantly reused across both branches.
- The effectiveness of LLaVA + LoRA finetuning on specific visual reasoning tasks is further verified.
- Insight: Interactive system = strong generative model + intelligent intention understanding + low-cognitive-load interface.

## Rating

⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MagicQuill V2: Precise and Interactive Image Editing with Layered Visual Cues](../../CVPR2026/image_generation/magicquill_v2_precise_and_interactive_image_editing_with_layered_visual_cues.md)
- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)
- [\[CVPR 2025\] Concept Lancet: Image Editing with Compositional Representation Transplant](concept_lancet_image_editing_with_compositional_representation_transplant.md)
- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)

</div>

<!-- RELATED:END -->
