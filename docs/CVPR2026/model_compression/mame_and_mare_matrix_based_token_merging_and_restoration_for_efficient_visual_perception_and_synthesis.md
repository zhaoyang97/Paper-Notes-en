---
title: >-
  [Paper Note] MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis
description: >-
  [CVPR 2026 (Findings)][Model Compression][token merging] This paper proposes MaMe, a training-free and differentiable token merging method based on fully matrix-based operations…
tags:
  - "CVPR 2026 (Findings)"
  - "Model Compression"
  - "token merging"
  - "token restoration"
  - "efficient transformer"
  - "matrix operation"
  - "image generation"
date: 2026-05-08
content_hash: 3db3091d0e854e2d
---

# MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis

**Conference**: CVPR 2026 (Findings)  
**arXiv**: [2604.13432](https://arxiv.org/abs/2604.13432)  
**Code**: [github.com/cominder/mame](https://github.com/cominder/mame)  
**Area**: Model Compression / Efficient Inference  
**Keywords**: token merging, token restoration, efficient transformer, matrix operation, image generation

## TL;DR

This paper proposes MaMe, a training-free and differentiable token merging method based on fully matrix-based operations, along with its inverse operation MaRe for token restoration, achieving efficient acceleration with minimal performance degradation across image classification, video recognition, and image generation tasks.

## Background & Motivation

The self-attention mechanism of Vision Transformers (ViTs) has complexity $\mathcal{O}(N^2)$, limiting the deployment of large-scale ViTs on resource-constrained devices. Existing token compression methods suffer from several limitations: Top-K operations are non-differentiable, hindering end-to-end training; clustering methods such as k-means are computationally intensive and practically slow; many methods introduce additional learnable parameters, increasing model complexity; some methods rely on specific architectures (e.g., class tokens). Although ToMe is a training-free method, it depends on GPU-unfriendly sorting and scatter-write operations.

## Method

### Overall Architecture

MaMe partitions the input token sequence into two disjoint subsets — a target token set and a source token set. A merging matrix is constructed via normalized cosine similarity, aggregating source tokens into similar target tokens while retaining distinctive source tokens that are not similar to any target token. MaRe serves as the inverse of MaMe to restore the merged tokens.

### Key Designs

1. **Adaptive Weight Refinement and Thresholding**: The method first computes a source-to-target cosine similarity matrix, filters weak connections via ReLU and a shift threshold $\tau$, applies column normalization, and introduces a dynamic column-level threshold $\zeta_j$ (the mean of non-zero weights), followed by a second round of ReLU pruning and re-normalization. The entire process consists solely of matrix operations, making it GPU-friendly.

2. **Distinctive Token Retention**: If the sum of merging weights between a source token and all target tokens is zero, the token is deemed dissimilar to any target and is retained without merging. A union strategy is adopted across batches to ensure consistency, with corresponding columns zeroed out to prevent merging.

3. **MaRe Token Restoration**: As the inverse of MaMe, MaRe restores tokens from the merged state using fully matrix-based operations. The MaMe+MaRe combination is applied in image generation pipelines (e.g., Stable Diffusion), merging tokens in the encoder and restoring them in the decoder.

### Loss & Training

MaMe is a training-free method that can be directly applied to pretrained models or integrated as a plug-and-play module during training from scratch. The entire process is fully differentiable, preserving gradient flow.

## Key Experimental Results

### Main Results

| Task | Model | Metric | MaMe Result | Speedup |
|------|-------|--------|-------------|---------|
| Image Classification | ViT-B | Accuracy | −2% | 2× throughput |
| Image Classification | ViT-B (last-layer fine-tuning) | Accuracy | +1.0% | 1.1× |
| Zero-Shot Classification | SigLIP2-B@512 | Accuracy | Nearly lossless | 1.3× |
| Video Recognition | VideoMAE-L (K400) | Accuracy | −0.84% | +48.5% speed |
| Image Generation | SD v2.1 (MaMe+MaRe) | Quality | Improved | −31% latency |

### Ablation Study

- The threshold $\tau$ controls merging sparsity: too small leads to excessive merging; too large retains too many tokens.
- Distinctive token retention is critical: removing it causes a notable performance drop.
- Adaptive weight refinement yields significant gains over simple normalization.

### Key Findings

- MaMe achieves simultaneous improvements in performance and speed on certain tasks.
- In image generation, MaMe+MaRe not only accelerates inference but also improves generation quality.
- The method naturally preserves causality, making it a candidate for reducing KV cache size in LLMs.

## Highlights & Insights

- The fully matrix-based design is elegant, balancing theoretical efficiency with practical acceleration.
- The triple advantage of differentiability, parameter-free operation, and plug-and-play usability makes the method highly practical.
- Surpassing the original model's accuracy by fine-tuning only the last layer is a compelling result.

## Limitations & Future Work

- The impact of token partitioning strategies (alternating vs. random) on performance warrants deeper analysis.
- Applicability to dense prediction tasks (e.g., object detection, segmentation) has not been sufficiently validated.
- The discussion of LLM KV cache compression remains at the conceptual level.

## Related Work & Insights

- Compared to ToMe's bipartite graph matching, the matrix-based approach is more GPU-friendly.
- The differentiable design enables more natural integration with end-to-end training.
- The MaRe restoration operation introduces a new paradigm for token compression in generative models.

## Rating

7/10 — Elegant design, comprehensive experiments, and strong practical utility, though the core innovation is primarily at the level of engineering implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)
- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)

</div>

<!-- RELATED:END -->
