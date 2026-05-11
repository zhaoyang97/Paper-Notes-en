---
title: >-
  [Paper Note] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][visual token pruning] This paper identifies a hierarchical attention pattern in vision encoders—middle layers attend to foreground objects while deep layers capture global information—and propo…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "visual token pruning"
  - "hierarchical attention"
  - "training-free"
  - "model-agnostic"
  - "VLM acceleration"
date: 2026-05-08
content_hash: 3972cec482f43090
---

# HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2508.00553](https://arxiv.org/abs/2508.00553)
**Code**: [GitHub](https://github.com/Danielement321/HiPrune)
**Area**: Multimodal Efficiency / Visual Token Compression
**Keywords**: visual token pruning, hierarchical attention, training-free, model-agnostic, VLM acceleration

## TL;DR

This paper identifies a hierarchical attention pattern in vision encoders—middle layers attend to foreground objects while deep layers capture global information—and proposes HiPrune, a training-free, model-agnostic visual token pruning method. By selecting three categories of tokens (Anchor/Buffer/Register) to preserve information at different semantic levels, HiPrune retains 99.3% of performance using only 1/3 of the tokens while reducing FLOPs by 58.7%.

## Background & Motivation

**Background**: VLMs encode images into large numbers of tokens (576 in LLaVA-1.5, exceeding 10,000 in high-resolution settings), incurring substantial computational and memory overhead. Visual tokens exhibit high redundancy—randomly removing 50% of visual tokens causes far less performance degradation than removing 5% of text tokens.

**Limitations of Prior Work**: (1) Methods such as FastV prune tokens within the LLM decoder based on attention scores, without exploiting the intrinsic properties of the vision encoder itself. (2) CLS-token-based attention methods cannot be applied to encoders lacking a CLS token (e.g., SigLIP). (3) Most approaches are sensitive to specific model architectures and require targeted tuning.

**Key Challenge**: Existing pruning methods either rely on feedback from the LLM side (wasteful—all tokens are encoded before pruning) or employ single-dimensional metrics (e.g., only the final-layer attention), overlooking the fact that different layers of the vision encoder capture information at different semantic granularities.

**Goal**: Leverage the intrinsic hierarchical attention patterns of vision encoders to design a general token pruning strategy.

**Key Insight**: Systematic analysis of hierarchical attention patterns across multiple vision encoders—CLIP, SigLIP, DeiT, and VJEPA2—reveals a consistent layer-wise specialization.

**Core Idea**: Middle-layer high-attention tokens correspond to foreground objects (Anchor); their spatial neighbors (Buffer) preserve local semantics; deep-layer high-attention tokens are distributed uniformly across the image (Register) and capture global information. Allocating a token budget across these three categories yields a compact yet comprehensive image representation.

## Method

### Overall Architecture

HiPrune operates prior to the vision encoder output: (1) attention scores are extracted from an intermediate layer to select high-attention tokens as Anchors; (2) spatial neighbors of Anchors are selected as Buffers; (3) attention scores from the output layer are used to select remaining budget tokens as Registers, excluding already-selected tokens; (4) HiPrune++ optionally supplements with tokens selected via text-visual cosine similarity. Only the output-layer tokens at the selected indices are retained.

### Key Designs

1. **Anchor Token (middle-layer object tokens)**:

    - Function: Preserve local detail of foreground objects.
    - Mechanism: Attention scores $\mathbf{a}^{[l]}$ are extracted from a designated object layer $l$ (a middle layer of the vision encoder), and the top-$N_a$ high-attention tokens are selected. Quantitative validation shows that the top-10% high-attention tokens at middle layers achieve the highest IoU with COCO object segmentation masks.
    - Design Motivation: Middle-layer attention focuses on image foregrounds, a pattern that holds consistently across CLIP, SigLIP, DeiT, and VJEPA2.

2. **Buffer Token (spatial neighborhood tokens)**:

    - Function: Mitigate attention noise and preserve spatial relationships.
    - Mechanism: For each Anchor token, its four spatial neighbors (top, bottom, left, right, forming a cross pattern) are selected as Buffers: $\mathcal{I}_B = \cup\{\mathcal{I}_A - 1, \mathcal{I}_A + 1, \mathcal{I}_A - c, \mathcal{I}_A + c\}$.
    - Design Motivation: Attention maps are noisy—a small number of high-attention tokens may be scattered across the image rather than concentrated on objects. Buffers correct this via spatial continuity.

3. **Register Token (deep-layer global tokens)**:

    - Function: Preserve global context and holistic image understanding.
    - Mechanism: The remaining budget is allocated to top-attention tokens from the output layer (last layer), excluding already-selected Anchor and Buffer tokens. Deep-layer high-attention tokens are distributed uniformly across the image, encoding global information.
    - Design Motivation: Retaining only object tokens discards global context (e.g., scene type, spatial layout); Register tokens compensate for this deficit.

### Loss & Training

HiPrune is entirely training-free and does not modify any model parameters. HiPrune++ additionally selects a small number of tokens based on cosine similarity between text and visual tokens to enhance instruction following.

## Key Experimental Results

### Main Results

**LLaVA-1.5-7B (576→192 tokens, 33.3% retained)**

| Method | GQA | MMB | MME | POPE | SQA | VQAv2 | Avg. |
|--------|-----|-----|-----|------|-----|-------|------|
| Original (576 tokens) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 100% |
| ToMe | 54.3 | 60.5 | — | — | — | — | — |
| FastV | 58.2 | 62.1 | — | — | — | — | — |
| **HiPrune** | **61.4** | **64.2** | **1852** | **85.6** | **69.1** | **78.1** | **99.3%** |

### Ablation Study

| Configuration | Avg. Performance Retained |
|---------------|--------------------------|
| Anchor only (middle layer) | 94.2% |
| Register only (deep layer) | 92.8% |
| Anchor + Register | 97.5% |
| **Anchor + Buffer + Register** | **99.3%** |

### Key Findings

- Retaining only 1/3 of tokens preserves 99.3% of performance with a 58.7% FLOPs reduction, confirming the high redundancy of visual tokens.
- The hierarchical attention pattern is consistently observed across six diverse architectures (CLIP-L/B, SigLIP, SigLIP2, DeiT, VJEPA2), indicating it is an intrinsic property of vision encoders rather than an artifact of specific training.
- HiPrune++ retains 96.1% performance under an extremely low token budget (1/9 of tokens) and significantly reduces hallucinations.
- The contribution of Buffer tokens, while modest, is stable—improving performance from 97.5% to 99.3%—and is insensitive to the shape of the neighborhood pattern (cross vs. square).

## Highlights & Insights

- The finding that "middle layers attend to objects while deep layers capture global context" is concise and compelling, supported by both quantitative analysis (IoU) and qualitative visualization (attention maps).
- The training-free and model-agnostic design makes HiPrune a truly plug-and-play tool.
- The three token categories correspond to three levels of image understanding: local detail, spatial context, and global semantics.

## Limitations & Future Work

- The object layer $l$ must be determined for each encoder type (typically a middle layer).
- For tasks requiring precise pixel-level understanding (e.g., OCR), pruning may discard critical information.
- Extension to video token pruning remains underexplored.
- Integration with dynamic-resolution encoders requires further validation.

## Related Work & Insights

- **vs. FastV**: FastV prunes within the LLM decoder after encoding all tokens; HiPrune prunes at the vision encoder stage, operating earlier and more efficiently.
- **vs. CLS-based methods**: Reliance on the CLS token limits generality (SigLIP has no CLS token); HiPrune uses hierarchical attention scores and applies to any ViT.
- **vs. ToMe**: ToMe merges tokens by similarity and requires training or additional computation; HiPrune performs pure index selection with zero additional overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ The hierarchical attention analysis is a novel finding, though the pruning mechanism itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Consistent validation across 4 VLMs, multiple encoders, and 6 vision encoder architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear; the logical chain from observation to method is complete.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play deployment with 58.7% FLOPs reduction offers high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](../../CVPR2026/multimodal_vlm/hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
