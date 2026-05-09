---
title: >-
  [Paper Note] Differentiable Hierarchical Visual Tokenization
description: >-
  [NeurIPS 2025 (Spotlight)][LLM Pretraining][Visual Tokenizer] This paper proposes an end-to-end differentiable hierarchical visual tokenizer that adaptively partitions images into tokens at pixel-level granularity. It leverages information criteria for hierarchical model selection, serves as a drop-in replacement for the fixed patch tokenization in ViT, and additionally supports raster-to-vector conversion.
tags:
  - NeurIPS 2025 (Spotlight)
  - LLM Pretraining
  - Visual Tokenizer
  - Hierarchical Tokenization
  - Differentiable
  - Superpixel
  - Information Criterion
date: 2026-05-08
content_hash: 3142072aa39a7ff7
---

# Differentiable Hierarchical Visual Tokenization

**Conference**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2511.02652](https://arxiv.org/abs/2511.02652)  
**Code**: Available  
**Area**: Computer Vision / Vision Transformer  
**Keywords**: Visual Tokenizer, Hierarchical Tokenization, Differentiable, Superpixel, Information Criterion

## TL;DR

This paper proposes an end-to-end differentiable hierarchical visual tokenizer that adaptively partitions images into tokens at pixel-level granularity. It leverages information criteria for hierarchical model selection, serves as a drop-in replacement for the fixed patch tokenization in ViT, and additionally supports raster-to-vector conversion.

## Background & Motivation

Vision Transformers (ViT) and their variants have become the dominant architecture in computer vision. However, their tokenization strategy suffers from fundamental limitations:

**Problems with fixed patch tokenization:**

**Spatial structure ignored**: Images are divided into fixed-size (e.g., 16×16) grid patches regardless of content, completely disregarding object boundaries and semantic structure.

**Semantic misalignment**: A single patch may span both foreground and background, resulting in semantically impure tokens.

**Efficiency waste**: Regions with simple textures (e.g., sky) and complex regions (e.g., architectural details) are allocated the same number of tokens, leading to inefficient use of computational resources.

**Information loss**: Fixed grid boundaries may cut through important visual features.

**Limitations of existing adaptive tokenization approaches:**

- Most are non-differentiable and cannot be trained end-to-end.
- Many rely on pretrained segmentation models, introducing additional computational overhead.
- They are difficult to integrate with existing pretrained ViTs.

**Core Idea of DHVT**: Design a fully differentiable visual tokenizer that adaptively determines the number and spatial layout of tokens based on image content, while maintaining backward compatibility with existing architectures.

## Method

### Overall Architecture

DHVT (Differentiable Hierarchical Visual Tokenization) consists of three core components:

1. **Pixel-level feature extraction**: Extracts an embedding feature for each pixel.
2. **Hierarchical segmentation/grouping**: Groups pixels into semantically coherent tokens in a top-down or bottom-up manner using information criteria.
3. **Token aggregation**: Aggregates per-group pixel features into a single token representation.

### Key Designs

**1. Differentiable Superpixel Generation**

DHVT employs a differentiable superpixel method to partition the image into semantically coherent regions:

- Each pixel $p_i$ has a feature vector $\mathbf{f}_i$ and spatial location $(x_i, y_i)$.
- Through differentiable soft assignment, each pixel is probabilistically assigned to a set of token groups.
- Assignment probabilities are based on feature similarity and spatial proximity.

**2. Hierarchical Model Selection via Information Criteria**

DHVT adopts the **Bayesian Information Criterion (BIC)** to automatically determine the optimal number of tokens for each image region:

$$\text{BIC}(k) = -2\ln L + k \cdot \ln n$$

where $L$ is the likelihood given $k$ tokens and $n$ is the number of pixels. BIC balances model fit and complexity:
- Regions with simple textures: a small number of tokens suffices (lower BIC).
- Regions with complex textures: more tokens are required (lower BIC achieved at larger $k$).

Hierarchical procedure:
1. Begin with a coarse granularity (few tokens).
2. Recursively evaluate whether each token region should be further subdivided.
3. Stop when further splitting does not yield a significant BIC reduction.
4. Produce tokens with adaptive count and size.

**3. Backward-Compatible Design**

To enable **direct retrofitting of pretrained ViTs**:
- The variable number of generated tokens is aligned with the pretrained model's token count via padding or truncation.
- Token features are projected to the same dimensionality as patch embeddings via learnable projection layers.
- Positional encodings are generated dynamically based on the spatial location of each token, rather than from a fixed grid.

**4. Raster-to-Vector Conversion**

As a byproduct, the hierarchical segmentation produced by DHVT can be directly applied to raster-to-vector conversion: each token corresponds to a vector primitive (a polygonal region with uniform color/feature), requiring no additional training.

### Loss & Training

The DHVT tokenizer is jointly trained end-to-end with downstream tasks:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{task}}$: downstream task loss (e.g., cross-entropy for classification, dice loss for segmentation).
- $\mathcal{L}_{\text{reg}}$: regularization term that encourages token boundaries to align with semantic boundaries and penalizes excessively many or few tokens.

Differentiability ensures that gradients from the downstream loss propagate back to the tokenizer parameters, enabling true end-to-end learning.

## Key Experimental Results

### Main Results

**Table 1: ImageNet-1K Classification Accuracy**

| Method | Backbone | Top-1 Acc (%) ↑ | #Tokens (avg) |
|--------|----------|-----------------|---------------|
| ViT-B/16 (fixed patch) | ViT-B | 81.8 | 196 (fixed) |
| DynamicViT | ViT-B | 81.3 | ~130 |
| ToMe | ViT-B | 81.5 | ~150 |
| DHVT (Ours) | ViT-B | **82.3** | ~160 (adaptive) |
| ViT-L/16 (fixed patch) | ViT-L | 85.2 | 196 (fixed) |
| DHVT (Ours) | ViT-L | **85.7** | ~170 (adaptive) |

DHVT achieves higher classification accuracy than fixed-patch and token pruning/merging methods while using fewer average tokens.

**Table 2: ADE20K Semantic Segmentation (mIoU)**

| Method | Backbone | mIoU (%) ↑ | #Tokens (avg) |
|--------|----------|------------|---------------|
| ViT-B/16 + UperNet | ViT-B | 47.4 | 196 |
| SegFormer-B2 | MiT-B2 | 46.5 | multi-scale |
| DHVT + UperNet | ViT-B | **48.8** | ~180 (adaptive) |

On dense prediction tasks (semantic segmentation), the semantically aligned tokens of DHVT yield more pronounced gains (+1.4 mIoU), as token boundary alignment with semantic boundaries directly benefits pixel-level prediction.

### Ablation Study

**Effect of Hierarchical Depth**

| Max Hierarchy Depth | Top-1 Acc (%) | Avg #Tokens |
|--------------------|---------------|-------------|
| 1 (no hierarchy) | 81.6 | 196 |
| 2 | 82.0 | ~175 |
| 3 | **82.3** | ~160 |
| 4 | 82.2 | ~145 |

A hierarchy depth of 3 achieves the best accuracy–efficiency trade-off. Excessive depth may result in overly small tokens that lose information.

**Choice of Information Criterion**

| Criterion | Top-1 Acc (%) |
|-----------|--------------|
| Fixed count | 81.9 |
| AIC | 82.0 |
| BIC | **82.3** |
| MDL | 82.1 |

BIC's penalty term best balances token count and representation quality.

### Key Findings

1. **Semantic alignment is critical**: DHVT token boundaries naturally align with object boundaries, yielding greater advantages on dense prediction tasks such as segmentation.
2. **Adaptive token allocation**: Simple regions are represented with fewer tokens; complex regions receive more, leading to more rational allocation of computational resources.
3. **Retrofit feasibility**: The tokenizer can be fine-tuned on top of pretrained ViTs without retraining from scratch.
4. **Vector conversion**: The hierarchical segmentation byproduct can be directly used for SVG generation, demonstrating the generality of the approach.
5. **Effectiveness of information criteria**: BIC provides an automatic, hyperparameter-free mechanism for selecting the number of tokens.

## Highlights & Insights

- **Paradigm shift from fixed to adaptive tokenization**: Patch tokenization in ViT is a widely accepted design choice that has rarely been questioned; DHVT demonstrates a superior alternative.
- **Differentiability + information criteria**: Integrating statistical model selection theory (BIC) into end-to-end deep learning is an elegant combination.
- **Backward compatibility**: No architectural redesign or pretraining from scratch is required; existing models can be directly enhanced.
- **Multiple benefits simultaneously**: Improved classification accuracy, better token efficiency, and free raster-to-vector conversion.
- **NeurIPS Spotlight** recognition reflects the significance of this direction and the maturity of the proposed method.

## Limitations & Future Work

1. **Additional tokenizer computation**: Differentiable superpixel generation introduces computational overhead; a balance must be struck between the speedup from token reduction and the tokenization cost.
2. **Handling variable token counts**: Varying token counts across a batch require special treatment (padding or bucketing), affecting training efficiency.
3. **Integration with large-scale pretraining**: Whether DHVT can be used in CLIP/DINOv2-scale pretraining remains to be verified.
4. **Video extension**: Extending hierarchical tokenization to the temporal dimension is a natural future direction.
5. **Superpixel quality**: The quality of initial superpixels directly affects the subsequent hierarchical construction.

## Related Work & Insights

- **ViT (Dosovitskiy et al. 2021)**: Standard paradigm for fixed patch tokenization.
- **DynamicViT**: Dynamic token pruning, operating after tokenization.
- **ToMe (Token Merging)**: Merges redundant tokens post-training; an orthogonal approach.
- **Superpixel methods (SLIC, etc.)**: Classical superpixel segmentation methods.
- **SegFormer**: Dense prediction Transformer using hierarchical features.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 5 — Differentiable hierarchical visual tokenization is an entirely new direction |
| Technical Quality | 4 — Clear theoretical motivation; clever application of information criteria |
| Experimental Thoroughness | 4 — Validated on classification, segmentation, and vectorization |
| Writing Quality | 4 — Clear presentation at Spotlight standard |
| Impact | 5 — May reshape the tokenization paradigm for ViT |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)
- [\[ICLR 2026\] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](../../ICLR2026/llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](../../ICLR2026/llm_pretraining/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)

<!-- RELATED:END -->
