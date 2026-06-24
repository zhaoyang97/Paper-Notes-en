---
title: >-
  [Paper Note] TransPixeler: Advancing Text-to-Video Generation with Transparency
description: >-
  [CVPR 2025][Video Generation][RGBA Video Generation] TransPixeler proposes introducing alpha channel tokens into pre-trained DiT video generation models. Through shared positional encoding, domain embedding, partial LoRA fine-tuning, and attention mask design, it achieves high-quality joint generation of RGB and alpha channels under extremely scarce RGBA training data.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "RGBA Video Generation"
  - "Transparent Channel"
  - "Diffusion Transformer"
  - "LoRA Fine-tuning"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: c8b608771e686e6b
---

# TransPixeler: Advancing Text-to-Video Generation with Transparency

**Conference**: CVPR 2025  
**arXiv**: [2501.03006](https://arxiv.org/abs/2501.03006)  
**Code**: [https://wileewang.github.io/TransPixeler/](https://wileewang.github.io/TransPixeler/)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: RGBA Video Generation, Transparent Channel, Diffusion Transformer, LoRA Fine-tuning, Attention Mechanism

## TL;DR
TransPixeler proposes introducing alpha channel tokens into pre-trained DiT video generation models. Through shared positional encoding, domain embedding, partial LoRA fine-tuning, and attention mask design, it achieves high-quality joint generation of RGB and alpha channels under extremely scarce RGBA training data.

## Background & Motivation

**Background**: Text-to-video (T2V) generation models have made significant progress, with DiT-architecture-based models (such as CogVideoX, Wan) generating high-quality videos. However, these models can only output RGB videos and do not support RGBA videos that contain an alpha transparency channel.

**Limitations of Prior Work**: RGBA videos are crucial in visual effects (VFX)—transparent elements such as smoke, reflections, and glass require an alpha channel to be seamlessly composited into scenes. Current solutions fall into two categories: (1) generating RGB first and then using video matting methods to extract alpha, but matting methods are themselves limited by the scarcity of RGBA data, leading to poor generalization; (2) modifying the VAE as in LayerDiffusion to decode the alpha channel, but VAE lacks semantic understanding capabilities, limiting its effectiveness in complex scenes. A common issue with both approaches is the unidirectional flow of information from RGB to alpha, preventing alpha from retroactively influencing the generation of RGB.

**Key Challenge**: RGBA video training data is extremely scarce (only about 484 videos). Directly training on such sparse data would severely restrict the diversity of the generated content. Therefore, it is necessary to maximize the utilization of pre-trained RGB model capabilities while extending support to the alpha channel.

**Goal**: Extend the capability of pre-trained RGB video models to generate RGB and alpha channels simultaneously, while preserving their original capabilities, allowing the generated content to transcend the limits of the finite RGBA training set.

**Key Insight**: Treat the alpha channel as a token sequence parallel to RGB, and design the interaction between tokens in the DiT attention mechanism carefully to achieve joint generation rather than step-by-step prediction.

**Core Idea**: Double the RGB token sequence into a dual-domain RGB+Alpha sequence. Alpha tokens reuse the positional encoding of RGB and incorporate a learnable domain embedding to distinguish the two domains. LoRA adaptation is applied only to the QKV projection of alpha tokens, and an attention mask is used to block direct text-to-alpha attention, thereby maximizing the preservation of the pre-trained model's RGB generation capability.

## Method

### Overall Architecture
TransPixeler is based on the DiT-architecture video generation models (e.g., CogVideoX). The input is a text prompt, and the outputs are an RGB video and its corresponding alpha video. Internally, the original video token sequence of length $L$ is extended to $2L$, where the first $L$ are decoded into the RGB video, and the remaining $L$ are decoded into the alpha video. Text tokens remain prepended to the video tokens. The entire sequence $[\text{text}; \text{RGB}; \text{alpha}]$ is processed via full self-attention with an attention mask.

### Key Designs

1. **Shared Positional Encoding + Domain Embedding**:

    - **Function**: Align alpha tokens with their corresponding RGB tokens in space and time while enabling the model to distinguish between the two domains.
    - **Mechanism**: Alpha tokens do not use consecutively incremented position indexes (i.e., not from $L+1$ to $2L$), but instead reuse the positional encoding of RGB tokens (from $1$ to $L$), ensuring that RGB and alpha tokens of each frame share the same spatial-temporal position information. A learnable domain embedding $d$, initialized to zero, is added to the alpha tokens to distinguish the two domains. The formula is: $\mathbf{f}^*(\mathbf{x}^m_{\text{video}}) = \mathbf{W}^*(\mathbf{x}^m + \mathbf{p}^{m-L} + d)$, where $m > L$ denotes the alpha token.
    - **Design Motivation**: Experiments show that if consecutive positional encoding is used, the model treats the alpha sequence as "subsequent frames" of the RGB video rather than an independent domain, causing the two to generate similar content. Shared positional encoding eliminates the learning difficulty of spatial-temporal alignment, accelerating convergence (initial convergence is achieved within 1000 iterations).

2. **Partial LoRA Fine-tuning**:

    - **Function**: Adapt to alpha channel generation while maintaining RGB generation quality.
    - **Mechanism**: Apply LoRA adaptation only to the QKV projection layers of alpha tokens ($m > L$): $\mathbf{W}^*(\cdot) = \mathbf{W}(\cdot) + \gamma \cdot \text{LoRA}(\cdot)$, where $\gamma$ controls the residual strength. The QKV projections for RGB tokens and text tokens remain frozen, strictly adhering to pre-trained weights. This implies that in the $3 \times 3$ grouping of the attention matrix, the calculations of text-to-RGB and RGB-to-text remain fully consistent with the original model.
    - **Design Motivation**: Full-parameter fine-tuning on only 484 videos can easily lead to overfitting and destroy the original RGB capabilities. Restricting LoRA only to the alpha domain means RGB output remains unaffected, allowing the model to freely generate RGB content beyond the training set distribution, with the alpha channel adapting accordingly.

3. **Attention Mask Design**:

    - **Function**: Control the attention interaction among the three groups of tokens (text, RGB, and alpha) to eliminate harmful interactions while retaining beneficial ones.
    - **Mechanism**: Construct an attention mask $\mathbf{M}^*_{mn}$, which is set to $-\infty$ (blocking text-to-alpha attention) when $m \leq L_{\text{text}}$ and $n > L_{\text{text}} + L$, and $0$ (allowed) otherwise. This design implies: text↔RGB remains unchanged (preserving original model capability); RGB-attend-to-Alpha is permitted (RGB can adjust itself based on alpha information, enhancing alignment); alpha-attend-to-RGB is permitted (alpha can obtain semantic information from RGB); text-attend-to-alpha is blocked (preventing contamination of text representations by limited training data).
    - **Design Motivation**: The authors systematically analyzed the role of each term in the $3 \times 3$ attention matrix. A key finding is that RGB-attend-to-Alpha is necessary—it permits RGB tokens to adjust themselves based on alpha information, improving alignment. Conversely, text-attend-to-alpha is harmful, because scarce RGBA data is insufficient for text tokens to learn how to interpret alpha information, which instead contaminates text representations.

### Loss & Training
Trained using flow matching or standard diffusion processes. The training data consists of only about 484 RGBA videos. Through the aforementioned design, trainable parameters are extremely minimal (only alpha LoRA + domain embedding), allowing effective training on minimal datasets without overfitting.

## Key Experimental Results

### Main Results
Compared with various baseline methods on the RGBA video generation task (such as Video Matting, Marigold-style prediction):

| Method | Alpha MAE↓ | RGBA SSIM↑ | Gen Diversity | RGB Preservation |
|------|----------|----------|--------------|---------|
| RVM (matting) | Higher | Lower | Limited by matting ability | Unaffected |
| Marigold-style | Medium | Medium | RGB-alpha misaligned | Unaffected |
| LayerDiffusion | Medium | Medium | Limited by VAE | Partially degraded |
| **Ours** | **Lowest** | **Highest** | **Most abundant** | **Fully preserved** |

### Ablation Study

| Configuration | Alpha Quality | RGB-Alpha Alignment | RGB Preservation |
|------|----------|--------------|---------|
| Consecutive Positional Encoding | Poor (slow convergence) | Poor | Poor (alpha affects RGB) |
| Shared Positional Encoding | Good (fast convergence) | Good | Good |
| w/o Domain Embedding | Medium (domain confusion) | Medium | Medium |
| w/ Domain Embedding | Good | Good | Good |
| Allow text-to-alpha | Poor (performance degradation) | Poor | Poor |
| Block text-to-alpha | Good | Good | Good |
| w/o RGB-to-alpha | Medium | Poor (misaligned) | Good |
| w/ RGB-to-alpha | Good | Good | Good |

### Key Findings
- Positional encoding design is key to convergence speed: Shared positional encoding achieves initial convergence in 1000 steps, whereas consecutive encoding requires more steps and performs worse.
- RGB-attend-to-Alpha attention is critical for ensuring RGB-Alpha alignment—this is the core reason TransPixeler competes so well against "generation followed by prediction" schemes.
- Blocking text-to-alpha attention is crucial for maintaining original RGB generation quality; otherwise, limited training data will contaminate text representations.
- Even when trained with only about 484 RGBA videos, TransPixeler can generate diverse RGBA content not present in the training set.

## Highlights & Insights
- **Token Domain Extension Paradigm**: Overcoming prior limitations by extending pre-trained models to new modalities via sequence doubling + domain embedding + partial LoRA fine-tuning. This paradigm is elegant, highly general, and transferable to other joint "RGB+X" generation tasks (such as depth, normal, optical flow, etc.).
- **Systematic Analysis of Attention Interaction**: The item-by-item analysis of the $3 \times 3$ grouped attention matrix provides profound insights into which information flows are beneficial and which are harmful. This analytical methodology is highly applicable to other multi-domain joint generation scenarios.
- **Effective Fine-tuning under Extremely Minimal Data**: Achieving diverse RGBA video generation with only 484 videos, credited to the meticulously designed architecture that maximizes the preservation of pre-trained knowledge.

## Limitations & Future Work
- Relies on RGB VAE to decode the alpha channel (treating alpha as a grayscale image), which may impose limitations on boundary precision.
- Currently supports only text-to-RGBA, without extending to image-to-RGBA or video editing.
- The RGBA training set is minimal (484 videos), leading to potentially suboptimal performance on specific transparent objects (e.g., complex glass refractions).
- Computational overhead introduced by doubling the sequence length must be considered.

## Related Work & Insights
- **vs LayerDiffusion**: LayerDiffusion modifies the VAE to decode alpha, but VAE lacks semantic understanding. TransPixeler performs joint generation at the DiT level, yielding stronger semantic understanding.
- **vs Marigold/Lotus**: Methods like Marigold generate RGB first and then predict depth/alpha, resulting in unidirectional information flow. TransPixeler employs bidirectional attention (enabling both RGB-to-alpha and alpha-to-RGB) to achieve superior alignment.
- **vs Video Matting**: Matting methods are restricted by their own training data coverage, whereas TransPixeler leverages prior knowledge from the pre-trained RGB model for broader generalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first DiT-architecture-based joint RGBA video generation method; its attention analysis provides profound insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies convincingly validate each design choice, though baseline methods for quantitative metrics remain limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly written; the analysis of the attention mechanism goes from simple to profound, with intuitive diagrams.
- Value: ⭐⭐⭐⭐ Successfully addresses practical demands in the VFX domain; the token domain extension paradigm is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)
- [\[CVPR 2026\] Video Generation with Stable Transparency via Shiftable RGB-A Distribution Learner](../../CVPR2026/video_generation/video_generation_with_stable_transparency_via_shiftable_rgb-a_distribution_learn.md)
- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [\[CVPR 2025\] Can Text-to-Video Generation Help Video-Language Alignment?](can_text-to-video_generation_help_video-language_alignment.md)

</div>

<!-- RELATED:END -->
