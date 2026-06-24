---
title: >-
  [Paper Note] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation
description: >-
  [CVPR 2025][Image Generation][Anomaly Image Generation] DualAnoDiff is proposed, which leverages a dual-interrelated diffusion model to simultaneously generate the integrated anomaly image and the corresponding anomaly parts. This solves the issues of insufficient diversity, unnatural blending, and misaligned masks in few-shot anomaly image generation, achieving SOTA performance in downstream anomaly detection tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Anomaly Image Generation"
  - "Few-Shot Generation"
  - "Dual Diffusion Model"
  - "Industrial Defect Detection"
  - "Image-Mask Pairs"
date: 2026-05-08
content_hash: 4a5a539bcb64afc3
---

# DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2408.13509](https://arxiv.org/abs/2408.13509)  
**Code**: [https://github.com/yinyjin/DualAnoDiff](https://github.com/yinyjin/DualAnoDiff)  
**Area**: Image Generation / Anomaly Detection  
**Keywords**: Anomaly Image Generation, Few-Shot Generation, Dual Diffusion Model, Industrial Defect Detection, Image-Mask Pairs

## TL;DR

DualAnoDiff is proposed, which leverages a dual-interrelated diffusion model to simultaneously generate the integrated anomaly image and the corresponding anomaly parts. This solves the issues of insufficient diversity, unnatural blending, and misaligned masks in few-shot anomaly image generation, achieving SOTA performance in downstream anomaly detection tasks.

## Background & Motivation

**Background**: Industrial anomaly detection faces a scarcity of anomaly samples. Existing methods are either unsupervised (using only normal samples) or semi-supervised (using a small amount of anomaly data), showing limited performance in anomaly localization and classification.

**Limitations of Prior Work**: Existing anomaly generation methods fall into two categories: model-agnostic methods (e.g., Cut-Paste) fail to generate realistic images; generative-model-based methods (e.g., AnomalyDiffusion) only focus on anomaly parts, resulting in unnatural blending between the anomaly and the background, and independently generated masks may mistakenly appear in the background.

**Key Challenge**: Anomaly images and their corresponding masks must be highly aligned, but existing methods process overall image generation and anomaly part generation separately, lacking explicit alignment constraints.

**Goal**: To design a method capable of simultaneously generating the overall anomaly image and the corresponding anomaly parts to ensure consistency and precise mask alignment.

**Key Insight**: Inspired by LayerDiffusion, the anomaly image is decomposed into two layers—the overall image and the anomaly-only part—and generated separately using two interrelated diffusion models.

**Core Idea**: Use a dual-interrelated diffusion model to simultaneously generate both the overall anomaly image and the corresponding anomaly regions, maintaining consistency between them through a Self-Attention Interaction Module (SAIM).

## Method

### Overall Architecture

Given a few anomaly image-mask pairs, the model consists of two branches based on pre-trained Stable Diffusion (fine-tuned via LoRA): the global branch SD generates the complete anomaly image, and the anomaly branch SD* generates the image containing only the anomalous regions. The two branches share identical timesteps and exchange information after each attention block via the Self-Attention Interaction Module (SAIM).

### Key Designs

1. **Dual-Interrelated Diffusion Model**:

    - **Function**: Simultaneously generates the overall anomaly image and the corresponding anomaly parts.
    - **Mechanism**: Incorporates two LoRAs on a pre-trained SD to fine-tune it into a global branch and an anomaly branch. It uses nested prompts (e.g., "a vfx with sks" and "sks"), with both branches sharing the same timesteps for synchronous denoising.
    - **Design Motivation**: To achieve natural blending between anomalies and the source image by generating both the global and regional parts simultaneously, ensuring the mask is highly aligned with the anomaly.

2. **Self-Attention Interaction Module (SAIM)**:

    - **Function**: Exchanges position, detail, and semantic details between the two branches.
    - **Mechanism**: Concatenates mid-features $\varphi_i(z)$ and $\varphi_i(z')$ from both branches, reshapes them to "bw 2 c" format, performs shared self-attention computation, and then splits them back into their respective branches with residual connections.
    - **Design Motivation**: To ensure that the generated overall anomaly image and regional anomaly image maintain consistency in spatial position and semantics.

3. **Background Compensation Module (BCM)**:

    - **Function**: Preserves background accuracy and the integrity of object shapes in the generated images.
    - **Mechanism**: Uses U2-Net to segment the background image $I_b$ first, feeding it as an additional condition into the global branch. Key and Value features of the background are extracted and injected into the self-attention layer of the global branch via an adaptive fusion MLP (with a learnable scaling factor $\gamma$).
    - **Design Motivation**: To resolve issues like object deformation and background confusion in few-shot scenarios, encouraging the model to focus more on generating the object itself.

### Loss & Training

The total loss is the sum of the standard diffusion losses of the two branches: the global branch predicts noise for the overall image, while the anomaly branch predicts noise for the anomaly part, both guided by their respective prompts. The text encoder $\tau_\theta$ is trainable. Masks are extracted from the generated anomaly parts using segmentation algorithms such as SAM.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | AnomalyDiffusion | Gain |
|--------|------|------|-------------------|------|
| MVTec AD (pixel) | AUROC | 99.1% | - | SOTA |
| MVTec AD (pixel) | AP | 84.5% | - | SOTA |
| MVTec AD Average | IS | Highest | Second highest | Multi-class leader |
| MVTec AD Average | IC-L | Highest | Second highest | Multi-class leader |

### Ablation Study

The contributions of the dual-branch structure, SAIM, and BCM components are verified: removing any of these components leads to a decline in generation quality and downstream task performance.

### Key Findings

- The dual-branch structure simultaneously ensures generation diversity and mask precision.
- The BCM shows particularly noticeable improvements for categories with simple backgrounds but complex objects (e.g., bottle, pill).
- The nested prompt design enables the model to accurately separate object attributes from anomaly attributes.

## Highlights & Insights

- Formulates the anomaly image generation problem as a dual-stream task that simultaneously generates the whole and the part, offering a novel perspective.
- Parameter-efficient extension of a single diffusion model to a dual-interrelated model using only two LoRAs.
- Direct acquisition of masks from the generated anomaly parts avoids the misalignment issues associated with independent mask generation.

## Limitations & Future Work

- Relies on U2-Net for foreground segmentation, imposing requirements on segmentation quality.
- Due to extremely sparse training data (averaging 8 images per class), there remains room for improvement on certain complex texture categories.
- Generated anomaly types are constrained by the existing anomaly categories present in the training samples.

## Related Work & Insights

- AnomalyDiffusion is the most direct predecessor, but focuses solely on the generation of anomaly parts.
- The layer-decomposition concept of LayerDiffusion inspired the dual-branch design of this work.
- It is promising to extend this method to other few-shot generation tasks that require precise image-label alignment.

## Rating

- Novelty: 8/10 — The design of dual-interrelated diffusion is highly unique.
- Technical Depth: 7/10 — Components are well-designed but relatively straightforward.
- Experimental Thoroughness: 8/10 — Validated extensively across multiple downstream tasks.
- Writing Quality: 7/10 — Clear structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](../../ICCV2025/image_generation/hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)

</div>

<!-- RELATED:END -->
