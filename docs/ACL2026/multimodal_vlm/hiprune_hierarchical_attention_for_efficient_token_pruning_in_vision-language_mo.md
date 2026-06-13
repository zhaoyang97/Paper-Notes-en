---
title: >-
  [Paper Note] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Vision token pruning] This paper identifies a hierarchical attention pattern in vision encoders—middle layers focus on main objects while deep layers focus on global information. Accordingly…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Vision token pruning"
  - "hierarchical attention"
  - "training-free"
  - "model-agnostic"
  - "VLM acceleration"
date: 2026-05-08
content_hash: 01ff952bdd1807ec
---

# HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2508.00553](https://arxiv.org/abs/2508.00553)  
**Code**: [GitHub](https://github.com/Danielement321/HiPrune)  
**Area**: Multimodal Efficiency / Vision Token Compression  
**Keywords**: Vision token pruning, hierarchical attention, training-free, model-agnostic, VLM acceleration

## TL;DR

This paper identifies a hierarchical attention pattern in vision encoders—middle layers focus on main objects while deep layers focus on global information. Accordingly, it proposes HiPrune, a training-free, model-agnostic vision token pruning method. By selecting three types of tokens (Anchor/Buffer/Register) to preserve visual information at different levels, it maintains 99.3% performance using only 1/3 of the tokens, reducing FLOPs by 58.7%.

## Background & Motivation

**Background**: VLMs encode images into large amounts of tokens (576 in LLaVA-1.5, potentially exceeding 10,000 in high-resolution scenarios), leading to significant computational and memory overhead. Vision tokens exhibit high redundancy—the performance degradation from removing 50% of vision tokens is far less than removing 5% of text tokens.

**Limitations of Prior Work**: (1) Methods like FastV prune based on attention scores inside the LLM decoder but do not utilize the intrinsic properties of the vision encoder itself; (2) Methods based on CLS token attention are inapplicable to encoders without a CLS token (e.g., SigLIP); (3) Most methods are sensitive to specific models and require targeted tuning.

**Key Challenge**: Existing pruning methods either rely on feedback from the LLM side (computational waste—encoding all tokens before pruning) or use single-dimensional metrics (e.g., using only the last layer's attention), ignoring the fact that different layers of the vision encoder capture different hierarchical semantic information.

**Goal**: Design a universal token pruning strategy utilizing the inherent hierarchical attention patterns of vision encoders.

**Key Insight**: A systematic analysis of hierarchical attention patterns in various vision encoders (CLIP, SigLIP, DeiT, VJEPA2) reveals a consistent pattern of hierarchical specialization.

**Core Idea**: Middle-layer high-attention tokens correspond to main objects (Anchor), complemented by spatial neighborhoods (Buffer) to retain local semantics; deep-layer high-attention tokens are uniformly distributed across the image (Register) to retain global information. The image is compactly represented by allocating three types of tokens according to a budget.

## Method

### Overall Architecture

HiPrune operates before the vision encoder output: (1) Extract attention scores from an intermediate layer and select high-attention tokens as Anchors; (2) Select spatial neighbors of Anchors as Buffers; (3) Extract attention scores from the output layer and select the remaining budget of high-attention tokens as Registers; (4) HiPrune++ optionally adds tokens based on text similarity. Finally, only output layer tokens corresponding to these indices are retained.

### Key Designs

1. **Anchor Token (Mid-layer object tokens)**:

    - **Function**: Preserve local detail information of the main object.
    - **Mechanism**: Extract attention scores $\mathbf{a}^{[l]}$ from a designated object layer $l$ (an intermediate layer of the vision encoder) and select the top-$N_a$ high-attention tokens. Quantitative validation shows that top-10% mid-layer tokens have the highest IoU with COCO object segmentation masks.
    - **Design Motivation**: Middle-layer attention focuses on the image subject; this pattern is consistent across various encoders like CLIP/SigLIP/DeiT/VJEPA2.

2. **Buffer Token (Spatial neighborhood tokens)**:

    - **Function**: Mitigate attention noise and preserve spatial relationships.
    - **Mechanism**: For each Anchor token, its four spatial neighbors (cross-shape) are selected as Buffers: $\mathcal{I}_B = \cup\{\mathcal{I}_A - 1, \mathcal{I}_A + 1, \mathcal{I}_A - c, \mathcal{I}_A + c\}$.
    - **Design Motivation**: Attention maps are noisy—a few high-attention tokens may be scattered across the image rather than concentrated on the object. Buffers correct this via spatial continuity.

3. **Register Token (Deep-layer global tokens)**:

    - **Function**: Preserve global context and overall understanding of the image.
    - **Mechanism**: Select the remaining budget of top tokens from the output layer (last layer) attention scores, excluding already selected Anchors and Buffers. Deep-layer high-attention tokens are uniformly distributed to encode global information.
    - **Design Motivation**: Retaining only object tokens loses global context (e.g., scene type, spatial layout), which Register tokens supplement.

### Loss & Training

A completely training-free method that does not modify any model parameters. HiPrune++ additionally utilizes the cosine similarity between the text encoder and vision tokens to select a small number of tokens, enhancing instruction following.

## Key Experimental Results

### Main Results

**LLaVA-1.5-7B (576→192 tokens, 33.3%)**

| Method | GQA | MMB | MME | POPE | SQA | VQAv2 | Average |
|------|-----|-----|-----|------|-----|-------|------|
| Original (576 tokens) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 100% |
| ToMe | 54.3 | 60.5 | — | — | — | — | — |
| FastV | 58.2 | 62.1 | — | — | — | — | — |
| **Ours (HiPrune)** | **61.4** | **64.2** | **1852** | **85.6** | **69.1** | **78.1** | **99.3%** |

### Ablation Study

| Configuration | Avg. Performance Retention |
|------|-----------|
| Anchor only (Mid-layer) | 94.2% |
| Register only (Deep-layer) | 92.8% |
| Anchor + Register | 97.5% |
| **Anchor + Buffer + Register** | **99.3%** |

### Key Findings

- Maintaining 99.3% performance with only 1/3 of tokens reduces FLOPs by 58.7%—proving high redundancy in vision tokens.
- The hierarchical attention pattern exists consistently across 6 different architectures (CLIP-L/B, SigLIP, SigLIP2, DeiT, VJEPA2)—indicating it is an inherent property of vision encoders rather than a product of specific training.
- HiPrune++ maintains 96.1% performance at an extremely low budget (1/9 tokens) and significantly reduces hallucinations.
- The contribution of Buffer tokens is small but stable—improving retention from 97.5% to 99.3%—and is insensitive to different shapes (cross, square).

## Highlights & Insights

- The discovery that "middle layers focus on objects, deep layers focus on global" is concise and powerful, verified by both quantitative analysis (IoU) and qualitative visualization (attention maps).
- The training-free and model-agnostic design makes it a true "plug-and-play" tool.
- The design of the three token types corresponds to three levels of image understanding: local details, spatial context, and global semantics.

## Limitations & Future Work

- The choice of the object layer $l$ needs to be determined for each encoder (typically a middle layer).
- For tasks requiring precise pixel-level understanding (e.g., OCR), pruning might lose critical information.
- Expansion to video token pruning has not been fully explored.
- Integration with dynamic resolution encoders requires further validation.

## Related Work & Insights

- **vs FastV**: FastV prunes inside the LLM decoder, requiring all tokens to be encoded first; HiPrune prunes during the vision encoder stage, being earlier and more efficient.
- **vs CLS-based methods**: Reliance on the CLS token is not universal (SigLIP has no CLS); HiPrune uses hierarchical attention scores, making it applicable to any ViT.
- **vs ToMe**: ToMe merges tokens via similarity, requiring training or additional computation; HiPrune uses pure index selection with zero extra overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ The discovery of hierarchical attention analysis is novel, though the pruning method itself is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Consistency verified across 4 VLMs and 6 vision encoders.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation with a complete logical chain from initial observation to final method.
- Value: ⭐⭐⭐⭐⭐ High practical value with plug-and-play capability and 58.7% FLOPs reduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](../../ICLR2026/multimodal_vlm/ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](../../ICML2026/multimodal_vlm/clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)

</div>

<!-- RELATED:END -->
