---
title: >-
  [Paper Note] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Multi-encoder VLM] METEOR proposes the first three-stage progressive token pruning framework for multi-encoder MLLMs: at the encoding stage, feature rank is used to allocate sparsity ratios across encoders; at the fusion stage, collaborative pruning eliminates cross-encoder redundancy; at the decoding stage, pruning ratios are adaptively adjusted based on text prompts. The framework reduces visual tokens by 76% with only a 0.3% performance drop.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Multi-encoder VLM
  - visual token pruning
  - collaborative compression
  - adaptive pruning
  - feature rank allocation
date: 2026-05-08
content_hash: 73d52db284cfdea0
---

# METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models

**Conference**: ICCV 2025
**arXiv**: [2507.20842](https://arxiv.org/abs/2507.20842)
**Code**: [https://github.com/YuchenLiu98/METEOR](https://github.com/YuchenLiu98/METEOR)
**Area**: Multimodal VLM
**Keywords**: Multi-encoder VLM, visual token pruning, collaborative compression, adaptive pruning, feature rank allocation

## TL;DR

METEOR proposes the first three-stage progressive token pruning framework for multi-encoder MLLMs: at the encoding stage, feature rank is used to allocate sparsity ratios across encoders; at the fusion stage, collaborative pruning eliminates cross-encoder redundancy; at the decoding stage, pruning ratios are adaptively adjusted based on text prompts. The framework reduces visual tokens by 76% with only a 0.3% performance drop.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) achieve multimodal understanding by encoding images into visual tokens and concatenating them with text tokens. Single-encoder approaches (e.g., CLIP) suffer from generalization limitations. Recent works such as EAGLE and Cambrian-1 enhance visual perception by fusing multiple visual encoders (CLIP, ConvNeXt, Pix2Struct, EVA-02, etc.).

**Limitations of Prior Work**: Multi-encoder approaches introduce substantial computational overhead. For instance, dual-encoder processing of 672×672 images in Mini-Gemini generates 2,880 visual tokens, and the computational complexity of self-attention scales quadratically with token count. Existing token compression methods (FastV, Pdrop, SparseVLM) are designed for single-encoder settings and cannot address challenges unique to multi-encoder systems: how to allocate token budgets appropriately across encoders with varying information density, and how to eliminate overlapping redundancy across encoders. Furthermore, these methods apply fixed pruning ratios and cannot adapt to different task requirements, leading to severe performance degradation on fine-grained tasks such as OCR.

**Key Challenge**: Multi-encoder architectures improve performance but dramatically increase computational cost. The visual tokens produced by multiple encoders contain substantial redundancy—both within individual encoders and as information overlap across encoders. Existing pruning methods cannot perform collaborative optimization in multi-encoder settings.

**Goal**: Design an end-to-end token pruning framework for multi-encoder systems that progressively eliminates redundancy across the encoding, fusion, and decoding stages.

**Key Insight**: Through in-depth analysis of token redundancy patterns in multi-encoder MLLMs, three key insights are identified: attention-based redundancy identification is unreliable in shallow layers but average token similarity is useful; the rank of feature maps provides a stable measure of information richness; and different tasks require different numbers of visual tokens.

**Core Idea**: Grounded in five key findings, visual tokens are progressively pruned across the encoding → fusion → decoding pipeline, employing rank-guided allocation at the encoding stage, mutual redundancy elimination at the fusion stage, and text-guided adaptive pruning at the decoding stage.

## Method

### Overall Architecture

METEOR introduces token pruning at three processing stages of a multi-encoder MLLM built on the EAGLE architecture. **Stage 1** (Encoding): redundant tokens within each encoder are independently pruned, with feature rank used to allocate sparsity ratios per encoder. **Stage 2** (Fusion): after adapting tokens from each encoder via independent projectors, collaborative pruning eliminates cross-encoder redundancy. **Stage 3** (Decoding): within the LLM layers, the number of retained tokens is adaptively determined based on the text prompt.

### Key Designs

1. **Rank-Guided Collaborative Token Allocation at the Encoding Stage (Stage 1)**:

    - Function: Progressively prune redundant tokens within each encoder and allocate token budgets across encoders according to their information richness.
    - Mechanism: Each encoder is divided into three equal segments. In **shallow layers**, redundancy is measured by cosine similarity to the mean token (Finding 1: shallow-layer attention values are neither sparse nor stable, but the mean token captures low-frequency background information, making tokens similar to it highly redundant). In **deep layers**, redundancy is measured by the attention between the CLS token and visual tokens (Finding 2: deep-layer attention is sparse and reliable). To allocate token budgets across encoders, SVD is applied to feature maps to compute rank $r_b^l$; higher rank indicates greater information richness, and the retention count is allocated proportionally: $k_b^l = k_b \cdot r_b^l / \sum_{l=1}^{C} r_b^l$. The expected rank is robust to input images (extremely low variance) and can be computed offline on a small data batch.
    - Design Motivation: Different encoders (e.g., CLIP favoring semantics, Pix2Struct favoring OCR) produce features with varying information density, making uniform token budget allocation suboptimal. Rank provides a mathematically principled measure of information richness.

2. **Cross-Encoder Collaborative Pruning at the Fusion Stage (Stage 2)**:

    - Function: Eliminate information overlap redundancy across tokens from different encoders.
    - Mechanism: A post-projection fusion strategy is adopted: each encoder is equipped with an independent MLP projector that maps its tokens into a shared semantic space before concatenation. Cross-encoder mutual redundancy is computed in this shared space as: $\mathcal{R}_i^j = \sum_{l=1, l \neq j}^{L} \sum_{m=1}^{n_l} \mathcal{S}(z_i^j, z_m^l)$, and the top-$k$ tokens with the lowest mutual redundancy are retained. Finding 3 shows that eliminating cross-encoder redundancy is more effective than eliminating intra-encoder redundancy.
    - Design Motivation: Existing methods use shared projectors (pre-projection fusion), which lack flexibility and ignore cross-encoder redundancy. Independent projectors combined with collaborative pruning precisely identify and eliminate information overlap in the shared semantic space.

3. **Text-Aware Adaptive Pruning at the Decoding Stage (Stage 3)**:

    - Function: Dynamically adjust the number of retained tokens based on the text prompt to accommodate varying task demands.
    - Mechanism: At a designated LLM layer, attention values from text tokens to visual tokens are used to identify redundant tokens. A key improvement is **attention head selection** (Finding 4): rather than averaging across all attention heads, the top-$k$ heads with the highest Visual Attention Value (VAV) are selected, since most heads are irrelevant to visual localization or even produce hallucinations. **Adaptive token retention** (Finding 5): the total VAV of the top-$k$ heads is strongly correlated with task complexity (coarse-grained tasks such as AI2D yield low VAV, while OCR tasks such as DocVQA yield high VAV), enabling dynamic computation of the retention count: $K = \lambda \cdot \sum_{h=1}^{k} \sum_{i=1}^{N} a_{i, \mathbf{I}(h)}$
    - Design Motivation: Fixed pruning ratios cause severe performance drops on OCR tasks (which require retaining more fine-grained tokens). VAV naturally reflects the contribution of visual information to the current task, making it a principled and efficient basis for adaptive token budget adjustment.

### Loss & Training

The pruning strategies of Stage 1 and Stage 2 are integrated into pretraining and SFT and trained jointly (pretraining on 558K data with all parameters except the projector frozen; SFT on 1M or 1.8M data with full model fine-tuning). Stage 3 is training-free and applied directly at inference time.

## Key Experimental Results

### Main Results

Comparison with EAGLE and other efficient methods on 11 benchmarks (based on Vicuna-7B):

| Method | Visual Tokens | TFLOPS↓ | SQA | AI2D | GQA | POPE | TextVQA | DocVQA | OCRBench | Avg. |
|--------|--------------|---------|-----|------|-----|------|---------|--------|----------|------|
| EAGLE | 1024 | 26.21 | 71.0 | 72.2 | 64.8 | 88.4 | 71.7 | 73.2 | 538 | 69.3 |
| FastV | 256 | 16.83 | 70.5 | 72.5 | 61.8 | 86.4 | 70.5 | 54.2 | 431 | 64.9 |
| SparseVLM | 256 | 17.89 | 70.7 | 72.2 | 59.8 | 86.5 | 70.4 | 51.9 | 383 | 63.9 |
| **METEOR** | **242*** | **13.42** | **71.4** | **73.4** | 63.5 | 87.9 | **71.1** | **71.4** | **533** | **69.0** |

### Ablation Study

Ablation over fusion strategies and pruning methods (SFT with 1M data):

| Fusion | Pruning | #Tokens | Knowledge | General | OCR | Avg. |
|--------|---------|---------|-----------|---------|-----|------|
| Pre-proj | — | 1193 | 64.4 | 73.7 | 64.5 | 67.5 |
| Post-proj | — | 1193 | 65.5 | 74.0 | 65.4 | 68.3 |
| Post-proj | Random | 576 | 63.4 | 72.2 | 48.5 | 61.4 |
| Post-proj | Resampler | 576 | 61.1 | 70.6 | 49.2 | 60.3 |
| Post-proj | **Ours** | 576 | **65.4** | **74.0** | **65.5** | **68.3** |

### Key Findings

- **76% token compression with only 0.3% loss**: METEOR-242 reduces visual tokens by 76% relative to EAGLE-1024, lowers TFLOPS by 49%, increases throughput by 46%, and incurs only a 0.3% average performance drop.
- **Substantial advantage on OCR tasks**: Compared to fixed-ratio methods such as FastV, Pdrop, and SparseVLM, METEOR outperforms them by 8.8–12.3% on OCR tasks due to adaptive pruning retaining more tokens for OCR-intensive inputs.
- **Effectiveness of cross-encoder collaborative pruning**: Collaborative pruning outperforms independent intra-encoder pruning and random pruning by 1.4% and 6.9% (average), respectively, with nuclear norm analysis confirming improved feature diversity.
- **Rank-guided allocation outperforms uniform allocation**: Rank-proportional allocation surpasses uniform allocation by approximately 2% on OCR tasks, as lower-rank encoders produce more redundant tokens.
- **Adaptive vs. fixed ratio**: Under the same average token count (242), adaptive pruning outperforms fixed-ratio pruning by 2.6% on OCR tasks.

## Highlights & Insights

- **Feature rank as an information measure**: SVD-based rank is used to quantify the information richness of each encoder's feature maps. The observation that rank exhibits extremely low variance across input images enables offline computation with a small data batch and reuse across inputs. This finding generalizes from CNNs to Vision Transformers and constitutes a theoretically grounded and practically useful contribution.
- **Five-finding-driven design paradigm**: Every design choice is supported by a corresponding empirical finding (different criteria for shallow vs. deep layers, rank stability, cross-encoder redundancy, non-uniform attention heads, correlation between VAV and task complexity), yielding a methodical and persuasive methodology.
- **Instance-adaptive philosophy**: Dynamically adjusting pruning ratios based on VAV for each input instance embodies a "task-specific" approach that can be transferred to other efficiency–accuracy trade-off scenarios.

## Limitations & Future Work

- Validation is currently limited to two multi-encoder architectures (EAGLE and Cambrian-1); generalization to more diverse architectures remains to be examined.
- Stage 3 adaptive pruning is training-free; incorporating limited training may further improve performance.
- The offline rank computation assumes rank robustness to input variation; its validity under severe distribution shift requires further investigation.
- The framework does not address temporal redundancy elimination for multi-frame visual tokens in video input scenarios.

## Related Work & Insights

- **vs. FastV / Pdrop**: These methods apply fixed-ratio pruning only at the LLM stage, overlooking acceleration opportunities at the encoding and fusion stages as well as the special requirements of tasks such as OCR. METEOR's end-to-end and adaptive design comprehensively outperforms them.
- **vs. EAGLE**: EAGLE is the backbone model of METEOR and uses all 1,024 tokens. METEOR achieves 99.7% of EAGLE's performance using only 24% of its token count, demonstrating the severity of visual token redundancy in multi-encoder MLLMs.
- **vs. Cambrian-1 (SVA)**: When using the same encoder combination, METEOR outperforms Cambrian-1's SVA aggregation by 3.3% on OCR tasks while using 44% fewer tokens.

## Rating

- Novelty: ⭐⭐⭐⭐ First end-to-end token pruning framework for multi-encoder MLLMs; rank-guided allocation and adaptive pruning are novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 benchmarks, extensive ablations, two encoder combinations, and efficiency analysis — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The five-finding organizational structure is clear and compelling, with strong logical flow in method derivation.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical efficiency bottleneck in multi-encoder MLLMs with high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/multimodal_vlm/comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llavaprumerge_adaptive_token_reduction_for_efficient_large_m.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](../../NeurIPS2025/multimodal_vlm/balanced_token_pruning_accelerating_vision_language_models_b.md)

<!-- RELATED:END -->
