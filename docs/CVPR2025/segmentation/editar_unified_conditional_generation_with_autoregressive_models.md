---
title: >-
  [Paper Note] EditAR: Unified Conditional Generation with Autoregressive Models
description: >-
  [CVPR 2025][Segmentation][Autoregressive Models] EditAR is proposed as the first method to unify image editing (texture modification, object replacement/removal, local editing) and image translation (depth/edge/segmentation map to image) within a single autoregressive framework. By introducing conditional image token prefixing and DINOv2 distillation loss on top of LlamaGen, it achieves performance competitive with specialized models across various conditional generation task…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Autoregressive Models"
  - "Unified Conditional Generation"
  - "Image Editing"
  - "Image-to-Image Translation"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: 57ea41f26f960339
---

# EditAR: Unified Conditional Generation with Autoregressive Models

**Conference**: CVPR 2025  
**arXiv**: [2501.04699](https://arxiv.org/abs/2501.04699)  
**Code**: [https://jitengmu.github.io/EditAR/](https://jitengmu.github.io/EditAR/)  
**Area**: Image Segmentation  
**Keywords**: Autoregressive Models, Unified Conditional Generation, Image Editing, Image-to-Image Translation, Knowledge Distillation

## TL;DR
EditAR is proposed as the first method to unify image editing (texture modification, object replacement/removal, local editing) and image translation (depth/edge/segmentation map to image) within a single autoregressive framework. By introducing conditional image token prefixing and DINOv2 distillation loss on top of LlamaGen, it achieves performance competitive with specialized models across various conditional generation tasks under the standard next-token prediction paradigm.

## Background & Motivation

**Background**: Conditional image generation (image editing + image translation) is currently dominated by diffusion models. Different tasks require different architectural designs and training strategies: image editing requires inversion + content preservation, while image translation (such as ControlNet) requires additional control networks. Integrating these tasks into a single model is highly challenging.

**Limitations of Prior Work**: Diffusion methods perform exceptionally well on single tasks but face three challenges when unifying across tasks: (1) architectural differences across tasks are vast (inversion-based vs. feedforward vs. ControlNet), making it difficult to accommodate them in a single model; (2) combinations of LLM + diffusion incur high computational and memory overheads, making joint optimization difficult; (3) existing autoregressive models (such as MaskGIT, VAR), despite having natural advantages in unified token representation, are primarily used for text-to-image generation and have not yet been proven suitable for conditional generation.

**Key Challenge**: Diffusion models feature specialized designs for individual tasks but are difficult to unify; autoregressive models are naturally unified but have not yet demonstrated conditional generation capabilities.

**Goal**: To verify whether a pure autoregressive model, without relying on diffusion, can simultaneously solve various image editing and translation tasks using a unified architecture.

**Key Insight**: Since autoregressive models are inherently sequence-to-sequence frameworks, conditional images and text instructions can be naturally encoded as token sequences to serve as inputs, outputting the edited image tokens. The key lies in how to effectively introduce image conditions and enhance semantic alignment.

**Core Idea**: Based on the LlamaGen next-token prediction framework, unified conditional image generation is achieved through a conditional image token prefix + DINOv2 distillation.

## Method

### Overall Architecture
EditAR consists of two stages: (1) a VQ-Autoencoder that maps images into discrete token sequences; (2) an autoregressive Transformer based on the Llama2 architecture, which takes conditional image tokens and text embeddings as inputs to predict target image tokens one by one in a next-token prediction manner. The overall loss is a combination of cross-entropy loss and DINOv2 distillation loss. During inference, quality is enhanced via classifier-free guidance (CFG) applied to both image and text conditions.

### Key Designs

1. **Conditional Image Token Prefix**:

    - **Function**: Enables the autoregressive model to accept conditional images as inputs, supporting both editing and translation tasks.
    - **Mechanism**: The conditional image $\mathcal{I}_c$ is encoded into a sequence of token indices $c_{\mathcal{I}_c} = \{c_1, ..., c_{h \cdot w}\}$ using the same VQ-Encoder as the target image, which is then concatenated with the text embeddings as the input prefix for the autoregressive model. The target image tokens $s$ appear on both the input and output sides during training to achieve autoregression. A key detail is using distinct positional embeddings for the conditional image tokens and the output tokens to differentiate the control sequence from the generation sequence. The output token probability is modeled as $p(s_i | s_{<i}, c_\mathcal{T}, c_{\mathcal{I}_c})$.
    - **Design Motivation**: Reusing the same VQ-Encoder avoids introducing additional encoders, and distinguishing positional embeddings ensures the model can differentiate between the input context and the generation target. This "prefix" approach is a natural extension of prefix-tuning in the LLM domain.

2. **Text-Driven Switching of Multimodal Conditions**:

    - **Function**: Supports different types of conditional inputs (natural image editing vs. depth/edge/segmentation map translation) with a unified interface.
    - **Mechanism**: Different condition types are distinguished by modifying text instructions. For example, depth map translation uses "Given the depth, generate the image following the instruction: <INSTRUCTION>", while edge and segmentation maps use similar formats. Natural image editing only uses "<INSTRUCTION>". All condition types share the same model parameters without any task-specific network modules.
    - **Design Motivation**: Leveraging the powerful language understanding capabilities of autoregressive models, text instructions naturally encode task type information, avoiding the need to train separate parameters for each condition as in ControlNet.

3. **DINOv2 Distillation Loss**:

    - **Function**: Enhances the semantic perception capabilities of the autoregressive model and improves text-image alignment.
    - **Mechanism**: Features are extracted from the final layer of the autoregressive Transformer and mapped to the same dimension space as DINOv2 using a single-layer convolutional alignment network $\mathcal{A}$, computing the MSE distillation loss: $\mathcal{L}_{distill} = MSE(\mathcal{A}(\mathcal{F}(\cdot)), \mathcal{E}_{distill}(\cdot))$. The DINOv2 parameters are frozen.
    - **Design Motivation**: Pure token prediction training only learns the distribution of discrete indices, which does not guarantee learning general semantic features. DINOv2 distillation injects rich semantic knowledge from the vision foundation model into the autoregressive model, which is experimentally proven to improve target object localization and editing precision.

### Loss & Training
The total loss is defined as $\mathcal{L} = \mathcal{L}_{CE} + 0.5 \cdot \mathcal{L}_{distill}$. During training, text and image conditions are independently dropped out with a 5% probability (replaced with null tokens), and both conditions are simultaneously dropped out with another 5% probability to support CFG during inference. The inference CFG scale is set to $\eta = 3.0$, applying "image-conditional" guidance to the image condition. The AdamW optimizer is used with a learning rate of $10^{-4}$, a batch size of 64, and training for 40K iterations.

## Key Experimental Results

### Main Results (Image Editing - PIE-Bench)

| Method | Type | Distance↓ | PSNR↑ | LPIPS↓ | Whole CLIP↑ | Edited CLIP↑ |
|---|---|---|---|---|---|---|
| InstructPix2Pix | Feedforward | 107.43 | 16.69 | 271.33 | 23.49 | 22.20 |
| MGIE | Feedforward | 67.41 | 21.20 | 142.25 | 24.28 | 21.79 |
| PnPInversion | Inversion | 11.65 | 27.22 | 54.55 | 25.02 | 22.10 |
| **EditAR** | **Feedforward** | **39.43** | **21.32** | **117.15** | **24.87** | **21.87** |

EditAR achieves the best overall performance among feedforward methods, striking a good balance between background preservation and editing quality.

### Image Translation Results

| Task | Method | FID↓ |
|---|---|---|
| Depth -> Image | ControlNet++ | 16.66 |
| Depth -> Image | **EditAR** | **15.97** |
| Edge -> Image | ControlNet | 14.73 |
| Edge -> Image | **EditAR** | **13.91** |
| Segmentation -> Image | ControlNet++ | 19.29 |
| Segmentation -> Image | **EditAR** | **16.13** |

Best FID is achieved across all three translation tasks, outperforming specialized models.

### Key Findings
- DINOv2 distillation significantly improves text-image alignment, particularly showing noticeable effects on target object localization (outperforming CLIP distillation).
- The CFG scale $\eta = 3.0$ is the optimal balance point between reconstruction quality and editing performance; too low leads to weak text responsiveness, while too high harms reconstruction.
- Operating as a unified model competing with multiple specialized models, it still comprehensively outperforms them in FID, suggesting that the unification of the autoregressive framework does not come at the expense of performance.
- Among feedforward methods, although InstructPix2Pix has a high edited CLIP score, it severely destroys the background; EditAR is superior in both overall CLIP score and background preservation.

## Highlights & Insights
- **First to prove pure autoregressive models can unify conditional generation**: This is a key empirical conclusion. Previously, autoregressive models were completely overshadowed by diffusion models in conditional generation, but this work demonstrates that with simple architectural modifications, they can catch up or even surpass specialized diffusion models.
- **Minimalist unified design**: Task differentiation relies solely on changes in text instructions, requiring no task-specific modules (in contrast to ControlNet's zero convolutions or UniControl's HyperNet). Simpler unification designs possess higher scalability.
- **The brilliance of DINOv2 distillation**: Instead of replacing the loss function, it acts as an auxiliary regularization. Utilizing only a single Conv layer for feature alignment, it incurs virtually no additional training overhead while significantly improving semantic understanding. This trick can be widely applied to other token-prediction models.

## Limitations & Future Work
- The image resolution is fixed at 512×512, restricted by the 16× downsampling of the VQ-Autoencoder (1024 tokens). Higher resolutions would require longer sequences.
- Background preservation is still inferior to inversion-based methods (such as PnPInversion), as the autoregressive model needs to completely regenerate all tokens.
- The VQ quantization itself introduces compression loss and visual artifacts, which may limit detail fidelity.
- Combination with more advanced autoregressive paradigms like VAR (next-scale prediction) has not yet been explored.

## Related Work & Insights
- **vs InstructPix2Pix**: Both are feedforward methods, but IP2P is based on diffusion while EditAR is based on autoregression; EditAR achieves better background preservation.
- **vs ControlNet/ControlNet++**: These require training separate adapters for each condition, whereas EditAR uses a single model for all conditions.
- **vs UniControl/UniControlNet**: These also perform unified conditional generation but are based on diffusion, and their performance is inferior to EditAR.
- **vs LlamaGen**: EditAR builds upon LlamaGen by adding conditional image input and distillation loss, extending the paradigm from pure generation to conditional generation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to unify autoregressive models into conditional generation, validating its feasibility.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both major task categories (editing and translation) are evaluated, but it lacks user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, simple and easy-to-understand method.
- Value: ⭐⭐⭐⭐ Opens up a new direction for autoregressive models in the conditional generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Continuous Locomotive Crowd Behavior Generation](continuous_locomotive_crowd_behavior_generation.md)
- [\[NeurIPS 2025\] Seg-VAR: Image Segmentation with Visual Autoregressive Modeling](../../NeurIPS2025/segmentation/seg-var_image_segmentation_with_visual_autoregressive_modeling.md)
- [\[CVPR 2025\] POSTA: A Go-to Framework for Customized Artistic Poster Generation](posta_a_go-to_framework_for_customized_artistic_poster_generation.md)
- [\[CVPR 2025\] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)

</div>

<!-- RELATED:END -->
