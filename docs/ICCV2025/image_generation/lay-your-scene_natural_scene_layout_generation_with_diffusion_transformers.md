---
title: >-
  [Paper Note] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers
description: >-
  [Image Generation] This paper proposes LayouSyn, an open-vocabulary text-to-layout generation pipeline that extracts scene elements via a lightweight open-source language model and generates layouts using an aspect-ratio…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 17baf619524cf7ac
---

# Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers

> **Conference**: ICCV 2025
> **arXiv**: [2505.04718](https://arxiv.org/abs/2505.04718)
> **Area**: Layout Generation · Diffusion Models · Controllable Image Generation
> **Keywords**: scene layout generation, diffusion transformer, open-vocabulary, spatial reasoning, text-to-layout

## TL;DR

This paper proposes LayouSyn, an open-vocabulary text-to-layout generation pipeline that extracts scene elements via a lightweight open-source language model and generates layouts using an aspect-ratio-aware diffusion Transformer, achieving state-of-the-art performance on spatial and quantity reasoning benchmarks.

## Background & Motivation

Controllable image generation methods (e.g., GLIGEN, Instance Diffusion) rely on user-provided scene layouts (bounding boxes), yet manual layout annotation is time-consuming and costly. Automatic text-to-layout (T2L) generation faces the following challenges:

**Closed-vocabulary limitations**: Traditional layout generation methods (LayoutGAN, LayoutVAE, BLT, etc.) assume a fixed set of object categories, making it difficult to handle open-vocabulary scenes described in natural language.

**Dependence on closed-source LLMs**: Methods such as LayoutGPT and LLM Blueprint rely on commercial models like GPT for layout generation, which are opaque, high-latency, and costly, and frequently produce unrealistic bounding boxes (e.g., implausible aspect ratios, unnatural positions).

**Document layout vs. scene layout**: Most layout diffusion models are designed and evaluated for document layouts, which differ fundamentally from natural scene layouts.

**LayouSyn's solution**: The T2L task is decomposed into two steps — (1) a lightweight open-source LLM (Llama-3.1-8B) extracts a set of scene element descriptions; (2) a novel aspect-ratio-aware diffusion Transformer generates conditional layouts in an open-vocabulary manner.

## Method

### Overall Architecture (Fig. 2)

**Stage 1: Description Set Generation**
- Llama-3.1-8B is used to extract noun phrases from a text prompt, assign quantities, and filter non-visualizable nouns.
- Output: a description set $\mathcal{D} = \{d_i\}_{i=1}^N$ (e.g., "teapot: 1, food: 1, plate: 1").

**Stage 2: Conditional Layout Generation**
- The diffusion model operates in the continuous coordinate space of bounding boxes.
- Conditioning: global condition (text prompt $p$) + local condition (description set $\mathcal{D}$) + scalar conditions (aspect ratio $r$, timestep $t$).

### Key Designs 1: Layout Diffusion Transformer (LDiT)

The standard DiT block is modified to incorporate cross-attention between description tokens and the global prompt:

- Each bounding box $b_i \in \mathbb{R}^4$ is encoded into a $d_{model}$-dimensional token via an MLP.
- Each description $d_i$ is encoded via a T5-sentence encoder followed by an MLP into a $d_{model}$-dimensional token.
- The two token types are concatenated and fed into the Transformer blocks.
- **Additional cross-attention layers**: global text embeddings (encoded by Google-T5) attend to description/bbox tokens, improving global–local condition alignment.
- The aspect ratio $r$ and timestep $t$ are encoded as scalar conditions via adaLN.

Coordinate normalization: bbox coordinates are normalized by the layout dimensions and scaled to $[-1, 1]$, yielding an aspect-ratio-agnostic representation.

### Key Designs 2: Noise Schedule Scaling

Low-dimensional bbox coordinates lose information too rapidly under standard noise schedules. A scaling factor $s$ is introduced:

$$\bar{\alpha}'_t = \frac{\bar{\alpha}_t \cdot s^2}{1 + (\bar{\alpha}_t \cdot (s^2 - 1))}$$

$s > 1$ slows information destruction; $s = 2.0$ combined with CFG scale 2.0 achieves the lowest L-FID.

### Loss & Training

Standard DDPM noise prediction loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathcal{B}_0 \sim p(\mathcal{B}|C), t \sim \mathcal{U}(1,T)}\left[\|\mathcal{E}_\theta(\mathcal{B}_t, C, t) - \mathcal{E}_t\|^2\right]$$

The model has only ~18M parameters and is trained on 2 A5000 GPUs.

## Key Experimental Results

### Layout Quality Evaluation (Tab. 2, COCO-GR Dataset)

| Method | L-FID ↓ |
|--------|---------|
| LayoutGPT (GPT-3.5) | 3.51 |
| LayoutGPT (GPT-4o-mini) | 6.72 |
| Llama-3.1-8B (fine-tuned) | 13.95 |
| **LayouSyn** | **3.07 (+12.5%)** |
| LayouSyn (GRIT pre-trained) | 3.31 (+5.6%) |

LayouSyn surpasses LayoutGPT (GPT-3.5) by **12.5%** in layout FID without requiring any closed-source LLM.

### Main Results

**Spatial and Quantity Reasoning (Tab. 3, NSR-1K Benchmark)**

| Method | Quantity Acc ↑ | Quantity Recall ↑ | Spatial Acc ↑ | Spatial GLIP ↑ |
|--------|--------------|-----------------|-------------|--------------|
| LayoutGPT (GPT-4o-mini) | 77.51 | 86.84 | 92.01 | 60.49 |
| LLMGroundedDiffusion | 89.94 | 95.94 | 72.46 | 27.09 |
| LLM Blueprint | 38.36 | 67.29 | 73.52 | 50.21 |
| Llama-3.1-8B (fine-tuned) | 70.84 | 93.36 | 86.64 | 52.93 |
| **LayouSyn** | **95.14** | **99.23** | 87.49 | 54.91 |
| **LayouSyn (GRIT)** | **95.14** | **99.23** | **92.58** | **58.94** |

Quantity reasoning: Recall of 99.23% and Accuracy of 95.14%, indicating near-perfect overlap between predicted and ground-truth object sets.
Spatial reasoning: LayouSyn (GRIT) achieves 92.58% accuracy and 58.94% GLIP detection accuracy, surpassing the GT layout baseline of 57.20%.

### Ablation Study

**Description Set Source (Tab. 5)**:

| LLM | L-FID ↓ |
|-----|---------|
| GPT-3.5 | 3.49 |
| GPT-4o-mini | 3.22 |
| Llama-3.1-8B | **2.74** |

The description sets generated by Llama-3.1-8B even outperform those from the GPT series, suggesting that noun phrase extraction is a relatively straightforward language task.

**LDiT Architecture (Tab. 7)**:

| Configuration | L-FID ↓ |
|--------------|---------|
| No cross-attention + no modulation | 2.82 |
| Cross-attention + no modulation | 2.81 |
| **Cross-attention + modulation** | **2.74** |

**Sampling Steps (Tab. 6)**: High-fidelity layouts are obtained with as few as 15 DDIM steps, at approximately 5 ms per sample.

## Highlights & Insights

1. **Moving beyond GPT**: The results demonstrate that lightweight open-source LLMs are fully capable of scene element extraction, obviating the need for expensive closed-source APIs.
2. The **LDiT architecture** for layout generation merits attention — global–local cross-attention improves condition alignment.
3. **Noise schedule scaling** is a critical technical insight for diffusion models operating on low-dimensional coordinates.
4. **LLM initialization** illustrates an interesting compositional paradigm: coarse prediction via LLM initialization followed by diffusion model refinement, completed within 15 steps.
5. The automatic object insertion pipeline (layout completion + GLIGEN inpainting) demonstrates practical potential for real-world image editing applications.

## Limitations & Future Work

- Occlusion relationships (depth ordering) are not addressed, which may lead to implausible object overlaps.
- Description set generation depends on the LLM's noun extraction capability, and implicitly referenced objects may be missed.
- Model performance under extreme aspect ratios outside the training distribution has not been validated.

## Related Work & Insights

- Layout generation: LayoutGAN, BLT, LayoutDM, Dolfin
- LLM-based layout: LayoutGPT, LLM Blueprint, Ranni
- Controllable image generation: GLIGEN, Instance Diffusion, BoxDiff

## Rating

- **Novelty**: ★★★★☆ — The combination of a diffusion Transformer for open-vocabulary layout generation with an open-source LLM is novel.
- **Technical Depth**: ★★★★☆ — The LDiT architecture and noise schedule scaling are well-motivated and soundly designed.
- **Experimental Thoroughness**: ★★★★★ — Three datasets, multiple benchmarks, comprehensive ablations, and application demonstrations.
- **Writing Quality**: ★★★★☆ — The problem formulation is clear and the two-stage pipeline is intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

</div>

<!-- RELATED:END -->
