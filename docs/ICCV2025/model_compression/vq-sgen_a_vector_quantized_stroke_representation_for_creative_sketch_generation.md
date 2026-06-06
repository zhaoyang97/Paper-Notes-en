---
title: >-
  [Paper Note] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation
description: >-
  [ICCV 2025][Model Compression][sketch generation] > VQ-SGen treats each stroke as an independent entity and decouples its shape from positional information. By applying vector quantization (VQ)…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "sketch generation"
  - "vector quantization"
  - "stroke representation"
  - "autoregressive"
  - "creative sketch"
date: 2026-05-08
content_hash: 5350ec563a4e74f3
---

# VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation

**Conference**: ICCV 2025
**arXiv**: [2411.16446](https://arxiv.org/abs/2411.16446)  
**Code**: [Project Page](https://enigma-li.github.io/projects/VQ-SGen/VQ-SGen.html)  
**Area**: Model Compression
**Keywords**: sketch generation, vector quantization, stroke representation, autoregressive, creative sketch

## TL;DR

> VQ-SGen treats each stroke as an independent entity and decouples its shape from positional information. By applying vector quantization (VQ), it constructs a compact discrete stroke codebook, and employs a cascaded autoregressive Transformer to sequentially generate semantic labels, shape codes, and position codes for each stroke. The method significantly outperforms existing approaches on the CreativeSketch dataset.

## Background & Motivation

Creative Sketch Generation aims to produce diverse, complex, and aesthetically rich hand-drawn sketches, distinguishing itself from conventional simple sketch generation. Existing methods suffer from the following limitations:

**Pixel-level methods** (DoodlerGAN, DoodleFormer): Generate sketches as holistic images or part-wise pixels, ignoring the **intrinsic structural relationships** among strokes (shape and relative position), resulting in local blurring, scattered strokes, or stroke fragmentation.

**Stroke point sequence methods** (SketchKnitter): Rearrange stroke points via diffusion models but lack the concept of "stroke entities," performing poorly on complex creative sketches.

**Absence of compact representations**: Continuous stroke embedding spaces are excessively large, making efficient sampling difficult for generators.

Core Idea: **Treating each stroke as an independent entity**, decoupling shape from position, and constructing a discrete, compact codebook space via VQ. The discrete representation not only reduces redundancy but also naturally forms semantically aware clusters — strokes with similar shapes (e.g., bird beaks, eyes, wings) cluster together — providing an ideal sampling foundation for the generator.

## Method

### Overall Architecture (Two-Stage)

**Stage 1: Decoupled Representation Learning**
- Each stroke $\bm{s}_i = (\bm{I}_i, b_i, l_i)$: shape image, position (bounding box center + size), and semantic label
- Decoupling: translate the stroke so that its bounding box is centered → shape image $\bm{I}_i$; position $b_i = (w_i/2, h_i/2, x_i, y_i)$
- Shape codebook $\bm{D}_s$ and position codebook $\bm{D}_l$ are learned separately

**Stage 2: Autoregressive Generation**
- A cascaded Transformer decoder progressively generates: label → shape code + position code

### Key Design 1: Vector-Quantized Stroke Representation

**Stroke Latent Embedding**: A CNN autoencoder encodes the stroke image $\bm{I}_i$ into a latent embedding $\bm{e}_i^s$, trained with reconstruction loss + CoordConv + distance field supervision.

**Shape Codebook Learning**: A 1D CNN encoder $\mathcal{E}^f$ compresses the stroke embedding sequence, which is then mapped to codebook $\bm{D}_s$ (with $V$ codewords) via nearest-neighbor quantization:

$$v_i^s = \arg\min_{j \in [0, V)} \|\bm{z}_i^s - \bm{c}_j\|$$

**VQ Training Loss**:

$$\mathcal{L}_{VQ} = \frac{1}{N}\sum_{i=1}^{N} \alpha(\|\bm{z}_i^s - \text{sg}[\bm{c}_{v_i^s}]\|_2^2 + \|\text{sg}[\bm{z}_i^s] - \bm{c}_{v_i^s}\|_2^2) + \|\bm{z}_i^s - \mathcal{D}^f(\bm{c}_{v_i^s})\|_2^2$$

This incorporates commitment loss, codebook loss, and reconstruction loss. The position codebook $\bm{D}_l$ is learned using the same pipeline.

### Key Design 2: Cascaded Autoregressive Generator

Sketch generation is decomposed as:

$$p(\bm{S}) = \prod_{i=1}^{N} p(v_i^s, v_i^l | l_i) \cdot p(l_i)$$

**Label Transformer $\mathcal{T}^l$**:
- Input: shape code embedding + position code embedding + label embedding (summed for fusion)
- Output: semantic label $l_{i+1}$ for the next stroke

**Codeword Transformer $\mathcal{T}^c$**:
- Takes two inputs: ① predicted label $l_{i+1}$ for semantic guidance; ② fused features from $\mathcal{T}^l$ for historical context
- Output: two branches predicting shape code index $v_{i+1}^s$ and position code index $v_{i+1}^l$ respectively

**Generation Loss**: Negative log-likelihood

$$\mathcal{L}_{gen} = -\log p(\bm{S})$$

### Code Space Analysis

UMAP visualization reveals that the shape codebook automatically forms semantically aware clusters: bird beaks, eyes, tails, wings, and legs each form distinct clusters, while heads and bodies form larger internal clusters. This demonstrates that the VQ process captures meaningful semantic structure in an unsupervised manner.

## Key Experimental Results

### Main Results: Comparison on the CreativeSketch Dataset

| Method | Creative Birds | | | Creative Creatures | | | |
|------|------|------|------|------|------|------|------|
| | FID↓ | GD↑ | CS↑ | FID↓ | GD↑ | CS↑ | SDS↑ |
| Training Data | - | 19.40 | 0.45 | - | 18.06 | 0.60 | 1.91 |
| SketchKnitter | 74.42 | 14.23 | 0.14 | 64.34 | 12.34 | 0.42 | 1.32 |
| DoodlerGAN | 39.95 | 16.33 | 0.69 | 43.94 | 14.57 | 0.55 | 1.45 |
| DoodleFormer | 17.48 | 17.83 | 0.57 | 20.43 | 16.23 | 0.53 | 1.68 |
| **VQ-SGen** | **15.78** | **18.92** | 0.53 | **17.61** | **17.42** | **0.57** | **1.86** |

- Creative Birds: FID reduced by 1.7 (15.78 vs. 17.48), GD improved by 1.09
- Creative Creatures: FID reduced by 2.82, GD improved by 1.19, achieving best results on all four metrics

### Ablation Study: Contribution of Core Components

| Configuration | Creative Birds FID↓ | Creative Birds GD↑ | Creative Creatures FID↓ | Creative Creatures GD↑ |
|------|---------------------|---------------------|-------------------------|------------------------|
| w/o VQ (continuous space) | 48.53 | 13.34 | 54.56 | 14.02 |
| w/o Decouple | 17.14 | 18.12 | 19.42 | 16.42 |
| w/o $\mathcal{T}^l$ (no label) | 16.51 | 18.35 | 19.21 | 16.14 |
| 2048×512 codebook | 26.23 | 15.34 | 43.21 | 14.51 |
| 4096×512 codebook | 16.92 | 18.27 | 18.44 | 16.14 |
| **8192×512 (VQ-SGen)** | **15.78** | **18.92** | **17.61** | **17.42** |

### Key Findings

- **VQ is central**: Removing VQ causes FID to surge from 15.78 to 48.53, confirming that a discrete compact space is critical for generation quality.
- **Decoupling is essential**: Omitting shape-position decoupling raises FID from 15.78 to 17.14 (CB) and from 17.61 to 19.42 (CC).
- **Label Transformer provides marginal but consistent gains**: Removing labels yields a small performance drop (second-best results), indicating the method is extensible to label-free datasets.
- FID consistently improves as codebook size increases from 2048 to 8192, though larger codebooks may suffer from low utilization rates.
- In a user study with 50 participants, VQ-SGen outperforms DoodleFormer in creativity, structural integrity, and overall preference.

## Highlights & Insights

1. **Stroke-level modeling paradigm**: Compared to pixel-level and point-sequence-level methods, treating strokes as independent entities represents a more natural level of abstraction, enabling the simultaneous capture of fine local structure and global semantic relationships.
2. **Elegant decoupling design**: Decoupling shape from position allows the VQ codebook to focus exclusively on shape variation, while positional information is encoded separately, avoiding mutual interference.
3. **Unsupervised semantic clustering**: The VQ codebook's automatic formation of semantically aware clusters is a noteworthy finding that provides natural support for controllable generation.
4. **Flexible conditional generation**: The framework natively supports three conditioning modes — category label, text, and stroke completion — demonstrating strong extensibility.

## Limitations & Future Work

- Validation is limited to the CreativeSketch dataset (two categories: birds and creatures), restricting data diversity.
- Codebook size requires careful trade-off — too small limits representational capacity, while too large may result in low codeword utilization.
- VQ reconstruction of strokes with highly variable shapes (e.g., complex animal ears) still exhibits distortion.
- The creativity of generated results is primarily constrained by the training data distribution, precluding the generation of entirely novel concepts beyond the training domain.
- Autoregressive generation follows the original drawing order; alternative stroke ordering strategies remain unexplored.

## Related Work & Insights

- ContextSeg first introduced the concept of individual stroke entities for semantic segmentation; SketchXAI applied it to interpretability analysis. VQ-SGen extends this paradigm to generative tasks.
- StrokeNUWA also applies VQ-VAE to vector graphics but relies on LLM-based generation; VQ-SGen offers a more lightweight approach specifically tailored to creative sketches.
- The two-stage paradigm of VQ representation combined with autoregressive Transformers is transferable to other structured sequence generation tasks (e.g., vector icon design, line art animation).
- CLIP-based conditioning for text-to-sketch generation warrants further exploration in interactive creative authoring tools.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)

</div>

<!-- RELATED:END -->
