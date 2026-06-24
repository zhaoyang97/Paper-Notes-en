---
title: >-
  [Paper Note] Improving Editability in Image Generation with Layer-wise Memory
description: >-
  [CVPR 2025][Image Generation][Image Editing] This paper proposes an iterative image editing framework based on layer-wise memory. By storing the latent and prompt embedding of each editing step and combining Background Consistency Guidance (BCG) with Multi-Query Decoupled Attention (MQD), the framework ensures consistent backgrounds and natural integration of new objects during multi-step sequential editing.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing"
  - "Iterative Generation"
  - "Layer-wise Memory"
  - "Attention Decoupling"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 5d55684da0401acc
---

# Improving Editability in Image Generation with Layer-wise Memory

**Conference**: CVPR 2025  
**arXiv**: [2505.01079](https://arxiv.org/abs/2505.01079)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Image Editing, Iterative Generation, Layer-wise Memory, Attention Decoupling, Diffusion Models

## TL;DR

This paper proposes an iterative image editing framework based on layer-wise memory. By storing the latent and prompt embedding of each editing step and combining Background Consistency Guidance (BCG) with Multi-Query Decoupled Attention (MQD), the framework ensures consistent backgrounds and natural integration of new objects during multi-step sequential editing.

## Background & Motivation

**Background**: Text-to-image generation (e.g., Stable Diffusion, PixArt-α, FLUX) is highly mature, but real-world editing scenarios typically require multi-step sequential modifications where users iteratively add or modify multiple objects in a scene. Existing editing methods (e.g., HD-Painter, Blended Latent Diffusion) are primarily designed for single-object, single-step modifications.

**Limitations of Prior Work**: (1) Single-step editing methods perform poorly in multi-step sequential editing, struggling to maintain the consistency of prior edits; (2) they require precise segmentation masks or external modules to maintain background integrity; (3) layout-to-image methods (guided by bounding boxes or depth maps) regenerate the entire image for each modification, failing to preserve the edited context; (4) handling occlusion relationships is difficult, such as placing a new object in front of an existing one.

**Key Challenge**: Iterative editing requires simultaneously satisfying two conflicting objectives: maintaining the stability of prior edits (preserving existing content) and ensuring the natural integration of new objects (requiring context-aware adaptive generation).

**Goal**: (1) How to achieve object placement with coarse masks while preserving the background? (2) How to maintain consistency across multi-step edits? (3) How to handle mask order (occlusion relationships)?

**Key Insight**: The authors introduce the concept of "mask order" to specify the generation sequence of objects (i.e., layer depth relationships) and design a memory mechanism to store the editing history. The key observation is that the latent and prompt information from each editing step can be reused to avoid repeated forward passes and maintain consistency.

**Core Idea**: Utilizing layer-wise memory to store editing history, Background Consistency Guidance to reuse latents to preserve the background, and Multi-Query Decoupled Attention to handle occlusion relationships for natural integration.

## Method

### Overall Architecture

The framework is built upon PixArt-α (a DiT-architecture diffusion model) and is training-free. Users provide a background prompt along with sequentially added object prompts and coarse masks. In each editing step, the Layer-wise Memory stores the current step's latent, prompt embedding, and mask. BCG retrieves the latent of the previous step from memory to perform background region blending. MQD decouples the queries of the current object and historical objects in the cross-attention layers to handle occlusion relationships.

### Key Designs

1. **Layer-wise Memory**:

    - **Function**: Stores the complete information of each editing step to support context preservation for subsequent edits.
    - **Mechanism**: Defines a memory set $L_l = \{l_0, l_1, l_2, ...\}$, where each element $l_i = \{\mathbf{p}_i, \{\mathbf{Z}_i^t\}_{t=1}^T, m_i\}$ contains three items: the prompt embedding, the latent sequence across all denoising steps, and the mask. During background generation, $m_0$ is an all-one mask, and each subsequent object has an independent mask defining its Region of Interest (RoI). The latent of a new object is initialized independently and then blended with the historical latents in memory via BCG.
    - **Design Motivation**: Storing the complete denoising trajectory (rather than just the final output) allows subsequent edits to perform precise latent blending at any denoising step, avoiding the computation overhead of performing forward passes on the original image every time, as required by traditional methods.

2. **Background Consistency Guidance (BCG)**:

    - **Function**: Efficiently maintains the stability of unedited areas.
    - **Mechanism**: At each denoising step $t$, only the area within the mask is updated, while the area outside the mask directly retrieves the latent of the previous step from memory: $\mathbf{Z}_i = \mathbf{Z}_{i-1} \odot (1-m_i) + \mathbf{Z}_i \odot m_i$. Since the latent is retrieved directly from memory, no additional forward pass on the original image is required, saving the computational cost of $C_f$ compared to traditional latent blending.
    - **Design Motivation**: Traditional inpainting methods (e.g., BLD) require a forward pass on the original image for each edit to obtain the background latent, which scales poorly in multi-step editing. BCG empirically saves about 10% of time per step, with even greater advantages in multi-step scenarios.

3. **Multi-Query Decoupled Cross-Attention (MQD)**:

    - **Function**: Ensures the natural integration of new objects under different mask orders, properly handling occlusion relationships.
    - **Mechanism**: In the cross-attention layer, the current prompt is used for attention on the current object's RoI region, while the corresponding historical prompts are used for attention on the non-overlapping regions of prior steps: $\mathbf{z}_i^{attn} = \bigcup_{j=0}^{i-1} \text{CrossAttention}(\mathbf{z}_i^{k,t} \odot (m_j - \Sigma_{l=j+1}^i m_l), p_j)$. Finally, all attention results are combined. The key lies in $m_j - \Sigma_{l=j+1}^i m_l$, which ensures that objects added later occlude those added earlier.
    - **Design Motivation**: Standard cross-attention cannot distinguish between semantic regions corresponding to different mask orders. MQD forces each region to focus only on its corresponding prompt, avoiding semantic confusion while achieving natural occlusion relationships through mask subtraction.

### Loss & Training

This method is a training-free pipeline using the pre-trained PixArt-α model (XL-1024) with DPM-Solver sampling, a guidance scale of 7.5, and 20 total denoising steps. The object deletion function is achieved by blending two historical latents starting from an intermediate step $\tau$ ($\tau = 8$, saving 60% of time).

## Key Experimental Results

### Main Results

| Type | Method | Resolution | BLEU-2/3/4↑ | METEOR↑ | CLIPcrop↑ |
|------|------|--------|-------------|---------|-----------|
| Image Editing | HD-Painter | 1024² | 63.29/47.63/36.28 | 0.1484 | 64.09 |
| Image Editing | BLD | 1024² | 55.30/40.38/29.58 | 0.1480 | 62.40 |
| Layout-to-Image | NoiseCollage | 512² | 55.75/42.43/32.96 | 0.1402 | 64.01 |
| **Ours** | - | **1024²** | **64.99/47.69/36.59** | **0.1513** | **64.29** |

Outperforms both image editing and layout generation baselines across the board on the Multi-Edit Bench.

### Ablation Study

| Configuration | BLEU-2/3/4↑ | METEOR↑ | CLIPcrop↑ | Description |
|------|-------------|---------|-----------|------|
| Baseline (PixArt-α inpaint) | 56.29/42.04/33.06 | 0.1586 | 64.05 | Baseline |
| +BCG | 60.74/46.27/35.20 | 0.1585 | 64.10 | Background consistency improves significantly |
| +QD | 62.68/46.42/35.03 | 0.1530 | 63.99 | Query decoupling improves semantics |
| Ours (Full) | 64.99/47.69/36.59 | 0.1513 | 64.29 | MQD + Memory further improves performance |

### Key Findings

- BCG yields the most significant improvement on BLEU (+4.5), indicating that background consistency is the core challenge of iterative editing.
- Extending QD to the multi-query version (MQD) further improves both BLEU and CLIP, showing the importance of utilizing the complete editing history (rather than just the background and current steps).
- In user study (50 participants, 5-point scale), the proposed method outperforms HD-Painter comprehensively in background consistency (4.59 vs. 3.71), natural integration (4.28 vs. 2.81), and text-scene alignment (4.49 vs. 3.08).
- SD3-ControlNet-Inpaint performs poorly in multi-step editing (BLEU-2 of only 29.90), demonstrating that single-step inpainting methods are fundamentally unsuited for iterative scenarios.

## Highlights & Insights

- **Elegant Mask Order Concept**: Encoding occlusion relations into the editing order naturally supports fore-and-background layer relationships (e.g., "a dog in front of a jeep") without requiring explicit depth estimation.
- **Training-free Design**: High practicality, running directly on the pre-trained PixArt-α without needing any fine-tuning.
- **Clever Implementation of Object Deletion**: Preserving consistency of deleted objects by skipping target latents in memory, removing corresponding prompt influences via MQD, and starting from intermediate steps to save 60% of computation time.

## Limitations & Future Work

- Since memory stores the complete denoising trajectories of all steps, it may incur high GPU memory overhead during long-sequence editing (dozens of steps).
- Validated only on PixArt-α, without testing generalization on newer models like FLUX or SD3.
- Although coarse masks reduce user effort, whether precise masks in combination with this method can yield further improvements remains undiscussed.
- Evaluation on the Multi-Edit Bench relies on LLaVa captioning and BLEU calculations, which might introduce evaluation biases.
- Does not support attribute modifications of existing objects (e.g., changing color/style); it only supports addition and deletion.

## Related Work & Insights

- **vs. HD-Painter**: HD-Painter relaxes mask precision but remains a single-step method, suffering from inconsistent object appearance during multi-step editing (e.g., a bus changing its appearance across iterations).
- **vs. Blended Latent Diffusion (BLD)**: BLD requires forward passes of the original image for latent blending at each step, causing ballooning costs in multi-step editing. The proposed method eliminates redundant computation using memory.
- **vs. NoiseCollage**: NoiseCollage is a layout-to-image method that regenerates the entire image each time, failing to preserve existing content, whereas the proposed method performs incremental updates.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of layer-wise memory and MQD is precisely designed for iterative editing paint points, and the concept of mask order is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Introduces a new benchmark with comprehensive quantitative and human evaluations, though lacking validation on more models.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations (especially the framework diagram in Fig. 2) and straightforward formula derivations.
- Value: ⭐⭐⭐⭐ Fills a gap in iterative image editing, showing high practicality through its training-free nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition](../../CVPR2026/image_generation/qwen-image-layered_towards_inherent_editability_via_layer_decomposition.md)
- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](generative_image_layer_decomposition_with_visual_effects.md)
- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [\[CVPR 2025\] Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)
- [\[CVPR 2025\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)

</div>

<!-- RELATED:END -->
