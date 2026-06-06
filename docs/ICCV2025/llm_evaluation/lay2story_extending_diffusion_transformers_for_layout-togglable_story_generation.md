---
title: >-
  [Paper Note] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation
description: >-
  [ICCV 2025][LLM Evaluation][story generation] Lay2Story introduces the task of layout-togglable story generation, constructs the Lay2Story-1M dataset of over 1 million high-resolution images…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "story generation"
  - "layout control"
  - "diffusion Transformer"
  - "subject consistency"
  - "large-scale dataset"
date: 2026-05-08
content_hash: 8e8937a2f0f1a3de
---

# Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation

**Conference**: ICCV 2025
**arXiv**: [2508.08949](https://arxiv.org/abs/2508.08949)  
**Code**: N/A  
**Area**: LLM Evaluation
**Keywords**: story generation, layout control, diffusion Transformer, subject consistency, large-scale dataset

## TL;DR

Lay2Story introduces the task of layout-togglable story generation, constructs the Lay2Story-1M dataset of over 1 million high-resolution images, and proposes a global–subject dual-branch framework built on the DiT architecture, achieving comprehensive improvements over existing methods in consistency, semantic relevance, and aesthetic quality.

## Background & Motivation

Story generation requires producing image sequences from text prompts while maintaining subject consistency. Existing approaches fall into two categories:

**Training-free methods** (e.g., ConsiStory, StoryDiffusion): maintain consistency by modifying cross-frame self-attention, but lack fine-grained guidance and inter-frame interaction, causing subject appearance drift in complex scenes.

**Training-based methods** (e.g., FLUX.1-dev IP-Adapter): preserve consistency by learning visual concepts across consecutive frames, but lack large-scale, high-quality layout-annotated datasets.

The shared core limitation of both categories is the **inability to exert fine-grained control over subjects**—including position, appearance, clothing, expression, and pose. The root cause lies in the absence of large-scale datasets with detailed subject annotations.

The authors first verify the substantial gain introduced by layout conditioning (subject position + detailed description): it not only enhances inter-frame consistency but also enables precise subject control. Based on this finding, they define a new task—**Layout-Togglable Storytelling**—in which users may optionally provide layout conditions.

## Method

### Overall Architecture

Lay2Story is built upon PixArt-$\alpha$ (a DiT-based T2I model) and comprises two main branches:

- **Global Branch**: takes noisy latents as input, guided by global descriptions, and is responsible for generating overall image content.
- **Subject Branch**: takes noisy latents, reference image latents, and subject masks as input, guided by subject descriptions, and is responsible for maintaining subject consistency while controlling subject position and details.

The output of the subject branch is fed back to the global branch via skip connections to fuse global and local information.

### Key Designs

1. **Lay2Story-1M Dataset**

   Construction pipeline: ~40,000 cartoon videos are collected from PBS Kids, Khan Academy, Internet Archive, and YouTube → ~25,000 videos (~11,300 hours) are retained after aesthetic scoring and NSFW filtering → FFmpeg samples frames at 0.25 FPS → GroundingDINO-B detects subjects → CLIP-L extracts subject features for K-means clustering → frames are grouped into sequences of 4/5/6 frames → GPT-4o mini generates global and subject-level descriptions.

   The resulting dataset contains approximately 1.02 million images at resolutions ≥720p, each accompanied by a global description, subject bounding boxes, and subject detail descriptions. To the best of the authors' knowledge, this is the largest story generation dataset to date.

   A complementary benchmark, **Lay2Story-Bench** (3,000 prompts with high-quality reference images), is also constructed for standardized evaluation.

2. **Core Modules of the Subject Branch**

   **Reference Image Concatenation**: the reference image is encoded by a VAE to yield a 4-channel feature $\mathcal{F}_{rep}$, which is concatenated with the reference mask $\mathcal{M}_{ref}$ and the noisy latent $\mathcal{Z}$ along the channel dimension to form a 9-channel input, then projected to 4 channels via a convolutional layer.

   **Masked Self-Attention**: a mask $\mathcal{M}_s$ is generated from the subject bounding box to restrict self-attention computation to the subject region, focusing on subject-level spatial context. During training, 25% of masks are set to fully valid to accommodate cases where no bounding box is provided.

   **Masked Cross-Attention**: subject descriptions are encoded by T5 into $TM_{subject}$, and attention is restricted via mask $\mathcal{M}_c$. During training, global descriptions replace subject descriptions with 25% probability to improve robustness.

   **Masked 3D Self-Attention**: to maintain cross-frame subject consistency, subject noisy latents are reshaped from $\mathbb{R}^{b \times f \times (hw) \times c}$ to $\mathbb{R}^{b \times (fhw) \times c}$, enabling cross-frame information propagation. An attention mask $\mathcal{M}_t$ constrains the model to attend only within subject-region positions across frames.

   All three masked attention variants share a unified formulation:
   $$\text{MA}(Q, K, V, M) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

3. **Information Propagation from Subject Branch to Global Branch**

   A ControlNet-style injection scheme is adopted: after every two global branch blocks, the output of a subject branch block (processed through a zero-initialized linear layer) is added to the global branch:
   $$\mathcal{Z}^n = \mathcal{Z}^n + F_m(\mathcal{Z}^m_{sub})$$

### Loss & Training

Two-stage training:
- **Stage 1**: the global branch is fine-tuned on Lay2Story-1M for the T2I task using AdamW (lr=2e-5, wd=0.03) for 5 epochs on 16×A100-40GB GPUs.
- **Stage 2**: the global branch is frozen, and the subject branch is trained independently using AdamW (lr=1e-5, wd=0.03) for 10 epochs on 32×A100-80GB GPUs.
- Inference uses 25 denoising steps with a CFG scale of 4.5.

## Key Experimental Results

### Main Results

| Method | Arch | DreamSim↓ | CLIP-I↑ | FID↓ | Recall@1↑ | Human-Pre↑ | Inference(s) |
|--------|------|-----------|---------|------|-----------|------------|--------------|
| 1Prompt1Story | U-Net | 0.2429 | 0.8461 | 66.79 | 0.5583 | 0.6742 | 20.69 |
| FLUX.1-dev IPA | DiT | 0.1533 | 0.9138 | 33.18 | 0.6482 | 0.7059 | 61.38 |
| Lay2Story w/o lc | DiT | 0.1602 | 0.9214 | 35.82 | 0.6376 | 0.7123 | 13.63 |
| **Lay2Story w/ lc** | **DiT** | **0.1324** | **0.9299** | **26.71** | **0.7012** | **0.7561** | **14.02** |

With layout conditioning, Lay2Story achieves leading performance across all metrics: CLIP-I is 1.6% higher than the second-best, DreamSim is 2% lower, FID is 6.4% lower, and Recall@1 is 2% higher. Even without layout conditioning, Lay2Story ranks second in CLIP-I. Inference takes only 14.02s, substantially faster than FLUX.1-dev IP-Adapter at 61.38s.

### Ablation Study

| Configuration | FID↓ | Recall@1↑ | Human-Pre↑ |
|---------------|------|-----------|------------|
| w/o Subject Branch | 110.58 | 0.1733 | 0.1928 |
| w/o Reference Image | 50.27 | 0.3981 | 0.4781 |
| w/o Masked 3D Self-Attn | 66.14 | 0.3274 | 0.3982 |
| **Full Lay2Story** | **26.71** | **0.7012** | **0.7561** |

Removing the subject branch causes a dramatic performance collapse (FID from 26.71 to 110.58), confirming the necessity of the dual-branch architecture. Masked 3D Self-Attention is critical for cross-frame consistency.

### Key Findings

- Layout conditioning is particularly effective in the early stages of denoising, enabling the model to establish correct subject spatial layouts more rapidly.
- Even without layout conditions, Lay2Story performs strongly, demonstrating the effectiveness of its layout-togglable design.
- In terms of computational cost, inference scales approximately linearly: 14s / 29.7GB for 4 frames and 78s / 62.1GB for 32 frames.

## Highlights & Insights

- **Data-driven approach**: constructing a million-scale dataset with layout annotations is the most significant contribution of this work, providing unprecedented training resources for story generation.
- **Layout-togglable design**: by randomly dropping layout conditions during training, the model can flexibly perform inference with or without layout conditions at test time without any architectural modification.
- **Unified masked attention design**: all three masked attention variants (self / cross / 3D) are implemented under a single unified formulation, yielding an elegant and effective design.

## Limitations & Future Work

- The current framework supports only single-subject annotation; multi-subject scenarios require future investigation.
- The dataset is predominantly sourced from cartoon videos; generalization to real-world photographic scenes remains unverified.
- Only T5 is used as the text encoder, which may limit the model's capacity to capture complex semantics.
- The current resolution of 720p could be further increased to 1080p or higher.

## Related Work & Insights

- This work bridges Layout-to-Image generation and Storytelling, representing a valuable research direction.
- The data construction pipeline (video → frame sampling → subject detection → clustering → GPT annotation) is general and transferable to other video understanding tasks.
- Masked 3D Self-Attention draws inspiration from temporal modeling in video generation, achieving cross-frame attention while maintaining computational efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Layout-togglable story generation is a valuable new task formulation; the dataset contribution is particularly notable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative and qualitative comparisons are comprehensive and ablations are clear, though multi-subject experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich illustrations.
- **Value**: ⭐⭐⭐⭐ The dataset and benchmark construction provide lasting contributions to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](../../AAAI2026/llm_evaluation/hybridla_hybrid_generation_for_document_layout_analysis.md)
- [\[ICCV 2025\] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography](degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)
- [\[ICCV 2025\] Imbalance in Balance: Online Concept Balancing in Generation Models](imbalance_in_balance_online_concept_balancing_in_generation_models.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](../../ACL2026/llm_evaluation/dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[NeurIPS 2025\] Rethinking Losses for Diffusion Bridge Samplers](../../NeurIPS2025/llm_evaluation/rethinking_losses_for_diffusion_bridge_samplers.md)

</div>

<!-- RELATED:END -->
