---
title: >-
  [Paper Note] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing
description: >-
  [ICCV 2025][Image Generation][MM-DiT] This paper systematically analyzes the attention mechanism of Multimodal Diffusion Transformers (MM-DiT), decomposing the attention matrix into four functional sub-blocks (I2I/T2I/I2T/T2T). Based on this analysis, it proposes an efficient prompt-based image editing method that operates by replacing image input projections ($\mathbf{q}_i, \mathbf{k}_i$), and is applicable to multiple MM-DiT variants including the SD3 series and Flux.1.
tags:
  - ICCV 2025
  - Image Generation
  - MM-DiT
  - Attention Mechanism Analysis
  - Prompt-based Editing
  - Stable Diffusion 3
  - Flux.1
date: 2026-05-08
content_hash: d6b4a864944f24b8
---

# Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing

**Conference**: ICCV 2025
**arXiv**: [2508.07519](https://arxiv.org/abs/2508.07519)
**Code**: N/A
**Area**: Diffusion Models / Image Editing
**Keywords**: MM-DiT, Attention Mechanism Analysis, Prompt-based Editing, Stable Diffusion 3, Flux.1

## TL;DR

This paper systematically analyzes the attention mechanism of Multimodal Diffusion Transformers (MM-DiT), decomposing the attention matrix into four functional sub-blocks (I2I/T2I/I2T/T2T). Based on this analysis, it proposes an efficient prompt-based image editing method that operates by replacing image input projections ($\mathbf{q}_i, \mathbf{k}_i$), and is applicable to multiple MM-DiT variants including the SD3 series and Flux.1.

## Background & Motivation

Recent diffusion models have progressively shifted from U-Net to Transformer architectures, with the Multimodal Diffusion Transformer (MM-DiT) emerging as the core architecture of state-of-the-art models such as Stable Diffusion 3 and Flux.1. Unlike the traditional U-Net design, which separates image self-attention from text-image cross-attention, MM-DiT concatenates text and image projections and performs a unified full attention operation, enabling bidirectional information flow between text and image modalities.

This architectural shift introduces a fundamental challenge: **existing image editing methods based on U-Net attention operations (e.g., Prompt-to-Prompt) cannot be directly transferred to MM-DiT architectures**. Specific challenges include:

1. The full attention mechanism in MM-DiT merges self-attention and cross-attention into a single unified operation, making it impossible to manipulate them independently.
2. As model scale increases, attention maps become increasingly noisy, leading to artifacts when used directly.
3. Explicitly computing the full attention matrix disables optimized kernels such as SDPA, resulting in significant inference slowdowns (up to 3×).

Understanding the behavioral patterns of MM-DiT attention and designing an editing method that is both efficient and generalizable across multiple MM-DiT variants constitutes the core problem addressed in this paper.

## Method

### Overall Architecture

The proposed method consists of two core components: (1) a systematic analysis of the MM-DiT attention mechanism that reveals the functional roles of four sub-blocks; and (2) an image editing method derived from these insights, achieving high-quality editing by replacing image input projections and selectively applying attention blocks.

### Key Designs

1. **Block Decomposition Analysis of the Attention Matrix**: The concatenated attention matrix $\mathbf{q}\mathbf{k}^T$ in MM-DiT is decomposed into four sub-blocks:

$$\mathbf{q}\mathbf{k}^T = \begin{bmatrix} \mathbf{q}_i\mathbf{k}_i^T & \mathbf{q}_i\mathbf{k}_t^T \\ \mathbf{q}_t\mathbf{k}_i^T & \mathbf{q}_t\mathbf{k}_t^T \end{bmatrix} \sim \begin{bmatrix} \text{I2I} & \text{T2I} \\ \text{I2T} & \text{T2T} \end{bmatrix}$$

   Analysis reveals that: **I2I** resembles U-Net self-attention and captures spatial layout and geometric information; **T2I** encodes token-level image region correspondences and is suitable for generating local editing masks; **T2T** degenerates into an approximate identity matrix; and **I2T** has its attention signal diluted due to row-wise softmax competition. Notably, T2I is more suitable than I2T for obtaining precise localization masks, as its structure allows multiple image regions to simultaneously retain high attention values.

2. **Noisy Attention Map Mitigation Strategies**: A phenomenon consistent with ViT scaling behavior is observed—as model scale increases, attention maps become more accurate in localization but noisier. Two solutions are proposed:

    - **Optimal Transformer Block Selection**: Ground-truth masks are generated for 100 PARTI prompts using Grounded SAM2, and all blocks are ranked based on three metrics—BCE, Soft mIoU, and MSE—with the Top-5 blocks selected. These blocks are not prompt-specific and are fixed across all experiments.
    - **Gaussian Smoothing**: Gaussian blur is applied to the attention masks to smooth boundaries and reduce artifacts.

3. **Image Input Projection Replacement for Editing**: The core editing operation replaces the target branch's image projections $\mathbf{q}_i^{tgt}, \mathbf{k}_i^{tgt}$ with those of the source branch $\mathbf{q}_i^{src}, \mathbf{k}_i^{src}$ during the first 20% of denoising timesteps. The key reasons for replacing input projections rather than the entire I2I attention block are:

    - Replacing the full attention map causes the T2T region to be overwritten by source branch context, resulting in text projection misalignment.
    - Input projection replacement allows continued use of the optimized SDPA kernel, improving computational efficiency by up to 3×.
    - Both approaches yield nearly identical editing results, and projection replacement does not require precise token alignment between source and target prompts.

### Loss & Training

This method is training-free and requires no training or fine-tuning. Key hyperparameters include:
- **Attention replacement step** $\tau = 0.8T$ (replacement applied during the first 20% of timesteps)
- **Local blending stop step** $\eta = 0.5T$ (local blending applied during the first 50% of timesteps)
- **Mask threshold** $\theta$: a higher threshold favors fine-grained local editing (e.g., text modification), while a lower threshold accommodates large-scale transformations.

For few-step models (Flux.1-schnell with 4 steps, SD3.5-L-Turbo with 4 steps), editing all blocks within a single step may cause the output to remain too close to the source image; therefore, only the first 38 or 30 blocks are replaced.

## Key Experimental Results

### Main Results

Evaluated on 60 PARTI prompts (30 simple edits + 30 complex edits), compared against two baselines:

| Model | Method | LPIPS ↓ | CLIPScore ↑ |
|------|------|---------|-------------|
| SD3-M | Fixed Seed | 0.594 | 0.377 |
| SD3-M | Prompt Change | 0.325 | 0.344 |
| SD3-M | **Ours** | **0.380** | **0.359** |
| SD3.5-L | Fixed Seed | 0.557 | 0.388 |
| SD3.5-L | Prompt Change | 0.306 | 0.363 |
| SD3.5-L | **Ours** | **0.418** | **0.377** |
| Flux.1-dev | Fixed Seed | 0.579 | 0.359 |
| Flux.1-dev | Prompt Change | 0.251 | 0.311 |
| Flux.1-dev | **Ours** | **0.369** | **0.339** |

### Ablation Study

| Configuration | Effect | Notes |
|------|------|------|
| Replace all $\mathbf{q}, \mathbf{k}$ vs. only $\mathbf{q}_i, \mathbf{k}_i$ | Similar results; the latter is more stable | Full replacement causes text alignment drift when source/target prompts differ significantly |
| I2I block replacement vs. $\mathbf{q}_i, \mathbf{k}_i$ replacement | Highly similar results | Projection replacement enables SDPA optimization, achieving up to 3× inference speedup |
| Top-5 blocks vs. all blocks (for masking) | Top-5 significantly reduces noise | Combined with Gaussian smoothing, boundary transitions are further improved |
| Block selection for few-step models | Only the first 30–38 blocks are replaced | Prevents overly conservative outputs |
| Threshold $\theta$ ablation | High → fine-grained editing; Low → large-scale transformation | The only hyperparameter requiring manual adjustment |

### Key Findings

- MM-DiT exhibits more precise attention localization than U-Net-based models (e.g., SDXL) across all model scales.
- Larger models produce noisier attention maps, but this can be effectively mitigated through block selection and smoothing.
- Attention maps from the T5 encoder are generally more precise than those from CLIP.
- For few-step distilled models, controlling the range of block replacement effectively regulates editing strength.

## Highlights & Insights

- This is the first work to systematically analyze the functional roles of the four sub-blocks in MM-DiT attention, providing an important foundation for understanding the new architecture.
- A scaling relationship between model size and attention noise is identified, with practical mitigation strategies proposed.
- The editing method is computationally efficient—full attention is computed only for the Top-5 blocks, while SDPA optimization is retained for the remainder.
- The token-mapping-free approach to prompt-based editing substantially broadens its applicability.

## Limitations & Future Work

- Editing performance is highly dependent on attention map quality and may fail for semantics the model itself cannot generate.
- Real image editing quality is constrained by the fidelity of the inversion technique employed.
- Quantitative metrics (LPIPS and CLIPScore) have inherent limitations and cannot fully capture editing quality.
- Validation is currently limited to the SD3 series and Flux.1; other MM-DiT architectures remain untested.

## Related Work & Insights

- The cross-attention manipulation concept from Prompt-to-Prompt (P2P) is transferred to the projection level in MM-DiT.
- Unlike ViT register tokens, which address attention noise through retraining, this work mitigates the issue via block selection and smoothing, requiring no retraining.
- The proposed method can be combined with approaches such as RF Inversion for real image editing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic analysis of MM-DiT attention with a tailored editing method; represents a solid contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 MM-DiT variants with quantitative, qualitative, and user study evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Analysis is clear, figures are abundant, and reasoning is rigorous.
- **Value**: ⭐⭐⭐⭐ Establishes a foundation for editing with new architectures; practical and computationally efficient.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[ICCV 2025\] NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes](nuiscene_exploring_efficient_generation_of_unbounded_outdoor_scenes.md)

<!-- RELATED:END -->
