---
title: >-
  [Paper Note] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit
description: >-
  [ICLR 2026][Multimodal VLM][Visual token compression] This paper proposes the HiDrop framework, which conducts a systematic layer-wise behavioral analysis of MLLMs (shallow layers = propagators…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Visual token compression"
  - "MLLM acceleration"
  - "progressive pruning"
  - "Late Injection"
  - "diffused attention"
date: 2026-05-08
content_hash: 169190f6fd1cc6ef
---

# HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit

**Conference**: ICLR 2026
**arXiv**: [2602.23699](https://arxiv.org/abs/2602.23699)
**Code**: [https://github.com/EIT-NLP/HiDrop](https://github.com/EIT-NLP/HiDrop)
**Area**: Multimodal VLM
**Keywords**: Visual token compression, MLLM acceleration, progressive pruning, Late Injection, diffused attention

## TL;DR

This paper proposes the HiDrop framework, which conducts a systematic layer-wise behavioral analysis of MLLMs (shallow layers = propagators, middle layers = fusion hubs, deep layers = language reasoners) and designs a three-stage strategy: Late Injection (skipping shallow layers) + Concave Pyramid Pruning (aggressive pruning in middle layers) + Early Exit (discarding tokens in deep layers). The framework compresses approximately 90% of visual tokens with negligible performance degradation and achieves a 1.72× training speedup.

## Background & Motivation

**Background**: The computational cost of processing visual tokens in MLLMs (e.g., LLaVA) scales quadratically with token count. Visual encoders produce far more tokens than text (e.g., 576 patch tokens), creating a major bottleneck for both inference and training.

**Limitations of Prior Work**: Existing visual token pruning methods rest on two fundamental misconceptions: (a) they incorrectly assume that shallow layers are critical multimodal fusion sites requiring dense visual tokens, whereas shallow layers in practice barely process visual tokens and merely propagate them passively; (b) they apply fixed-ratio pyramid or linear pruning schedules (e.g., FastV, PDrop), ignoring the non-uniform information flow across layers.

**Key Challenge**: How can token pruning strategies be made to truly align with the internal layer-wise processing dynamics of the model?

**Goal**: Design a token management strategy aligned with the hierarchical functional roles of MLLM layers — shallow layers do not need to process visual tokens (skip entirely), middle layers are where fusion redundancy is highest (aggressive pruning), and deep layers have already completed fusion (discard remaining tokens).

**Key Insight**: Conduct a systematic layer-wise behavioral analysis (intra-modal similarity + cross-modal influence) and replace heuristic assumptions with data-driven findings.

**Core Idea**: Perform the right operation at the right location based on the hierarchical functional division of MLLM layers (propagation / fusion / reasoning) — inject late, prune aggressively, and exit early.

## Method

### Overall Architecture

HiDrop partitions LLM layers into three stages:
- **Shallow layers (Layers 1–8)**: Late Injection — visual tokens are not injected at all; only text is processed.
- **Middle layers (Layers 9–24)**: Concave Pyramid Pruning — visual tokens are progressively pruned at selected filtering layers using a Differentiable Top-K operator, with faster pruning early and slower pruning later.
- **Deep layers (Layers 25–32)**: Early Exit — all remaining visual tokens are discarded; pure text reasoning proceeds.

Together, the three stages define a "visual processing window" in which visual tokens exist for only approximately half of the total layers.

### Key Designs

1. **Late Injection**:

    - Function: Visual tokens are injected into the sequence only at layer $L_{inj}=9$.
    - Mechanism: Analysis reveals that intra-modal cosine similarity in shallow layers is extremely high (visual tokens barely change), and cross-modal influence is near zero (text representations are unaffected by the image). Since shallow layers do not process visual information, no computation should be wasted there.
    - Design Motivation: Unlike the conventional "inject first, then prune" paradigm, HiDrop is the first to propose "delayed injection" — visual tokens bypass shallow layers entirely, saving computation from the outset.

2. **Concave Pyramid Pruning + ILVAS**:

    - Function: Progressively prune visual tokens in middle layers, with faster pruning early and slower pruning later.
    - Mechanism:
        - **Where to prune (ILVAS)**: The Inter-Layer Visual Attention Similarity metric is proposed to measure the stability of visual token attention distributions between adjacent layers. Layers with high ILVAS indicate that attention allocation has stabilized and are therefore good filtering points. Local maxima of the ILVAS curve are selected (e.g., layers {10, 14, 16, 18}).
        - **What to prune (DTop-K)**: A Differentiable Top-K operator is used for differentiable token selection. Normalized importance scores $c'_i$ are computed, and a soft mask is generated via sigmoid with a learnable threshold $a$: $\text{Mask}(c,a) = \sigma(\lambda(c'_i - a))$. A hard threshold is applied during the forward pass for discrete selection, while gradients flow through during the backward pass.
    - Design Motivation: The concave schedule (fast early, slow late) matches the pattern of increasing fusion sparsity in middle layers — early in the fusion stage, many tokens are redundant and can be removed aggressively, while later tokens are more informative and require more careful pruning.

3. **Early Exit**:

    - Function: All remaining visual tokens are discarded at layer $L_{exit}=25$.
    - Mechanism: Training-free experiments verify that removing all visual tokens after layer 24 causes negligible performance degradation.
    - Design Motivation: Deep layers have completed cross-modal fusion and entered a pure language reasoning phase; visual tokens at this stage consume computation without contributing information.

4. **Engineering Optimizations**:

    - Persistent Position Encoding: Each visual token retains a fixed positional identifier to prevent RoPE positional confusion caused by dynamic pruning.
    - FlashAttention compatibility: Token selection is performed via a lightweight auxiliary attention module, leaving the primary attention computation unchanged.
    - Parallel decoupling: Shallow-layer text forward passes and visual encoding execute in parallel; Late Injection makes this parallelism possible.

### Loss & Training

- Standard two-stage LLaVA training (pre-training + instruction tuning) is followed.
- The temperature coefficient for DTop-K is set to $\lambda = N_v$ (the number of visual tokens).
- Training is conducted on 8× A100 40GB GPUs.

## Key Experimental Results

### Main Results

Comparison across 11 benchmarks on LLaVA-1.5-7B (retaining approximately 64 tokens, 88.9% compression):

| Method | Type | MMEP | GQA | VQAv2 | POPE | MMStar | Avg(%) |
|--------|------|------|-----|-------|------|--------|--------|
| LLaVA-1.5-7B | Upper bound (576 tokens) | 1506.5 | 61.9 | 78.5 | 86.8 | 33.7 | 100.0 |
| FastV | Training-free | 1086.6 | 48.8 | 61.6 | 67.7 | 29.6 | 82.8 |
| PDrop | Training-based | 1350.7 | 56.6 | 71.8 | 82.6 | 32.7 | 94.2 |
| TwigVLM | Training-based | 1404.0 | 58.8 | 75.6 | 82.7 | 33.1 | 95.3 |
| **HiDrop** | **Training-based** | **1473.3** | **60.5** | **76.5** | **86.4** | **32.0** | **98.3** |

Under the most extreme setting of 48 tokens (91.7% compression), HiDrop still achieves 97.1% of the original performance.

### Ablation Study

| Configuration | Avg(%) | Note |
|---------------|--------|------|
| Full HiDrop | 98.3 | Complete framework |
| w/o Late Injection | 96.8 | Shallow layers also process visual tokens |
| w/o Early Exit | 97.5 | Visual tokens retained in deep layers |
| w/o Concave (linear schedule) | 96.9 | Uniform pruning replaces concave pyramid |
| Hard Top-K (replaces DTop-K) | 97.1 | Non-differentiable hard selection |

Training efficiency: HiDrop achieves a 1.72× training speedup compared to the original LLaVA-1.5-7B.

### Key Findings

- Late Injection contributes the most — approximately 1.5% performance retention gain — indicating that avoiding visual token processing in shallow layers not only saves computation but also eliminates meaningless shallow-layer interference.
- Concave > Linear > Convex pyramid: Aggressive pruning in the early fusion stage is optimal, fully consistent with the conclusions of the middle-layer fusion dynamics analysis.
- DTop-K outperforms Hard Top-K by approximately 1.2%: Differentiable selection enables gradient backpropagation to the token importance estimation.
- Even when compressing to only 48 visual tokens per image (576→48, 12× compression), the POPE metric drops only from 86.8 to 86.6, representing near-lossless compression.

## Highlights & Insights

- **Analysis-driven design paradigm**: Rather than designing a method and then seeking experimental justification, this work first conducts systematic layer-wise behavioral analysis (intra-modal similarity, cross-modal influence, early exit experiments) and lets data-driven findings guide algorithmic design. This research paradigm is itself instructive.
- **"Delayed injection" as a conceptual breakthrough**: All prior methods implicitly assume that visual tokens participate in computation from the first layer. HiDrop is the first to assert that "shallow layers do not need visual information at all" — a deep insight into MLLM operating mechanisms that may generalize to other modalities (e.g., audio tokens).
- **One-to-one correspondence between three stages and layer functions**: Late Injection ↔ propagation layers, Concave Pruning ↔ fusion layers, Early Exit ↔ reasoning layers — an elegant design.

## Limitations & Future Work

- **Validated only on LLaVA-1.5**: The conclusions from the layer-wise behavioral analysis may not generalize to all MLLMs (e.g., Qwen-VL or InternVL may have different layer-wise functional assignments), and validation across more architectures is needed.
- **Fixed injection and exit layers**: $L_{inj}=9$ and $L_{exit}=25$ are hard-coded; different inputs (simple vs. complex images) may require different processing windows.
- **Multi-image inputs not considered**: In video understanding or multi-image QA settings, the volume of visual tokens is much larger and layer-wise behavior may differ.
- **DTop-K training overhead**: Differentiable Top-K introduces additional parameters and computation; the cost-benefit ratio for larger models requires further investigation.

## Related Work & Insights

- **vs. FastV**: FastV performs a single one-shot pruning at an early layer — an overly coarse approach with a poorly chosen pruning location (pruning at shallow layers). HiDrop demonstrates that visual tokens should not be present in shallow layers at all.
- **vs. PDrop**: PDrop applies progressive pruning at uniform intervals and uniform ratios, ignoring the non-uniform nature of middle-layer fusion. HiDrop's ILVAS metric and concave pyramid schedule offer a more precise alternative.
- **vs. TwigVLM**: TwigVLM prunes in shallow layers and removes tokens in deep layers, but shallow-layer pruning is superfluous. HiDrop replaces shallow-layer pruning with Late Injection, which is more efficient.
- **Implications for video/multi-image MLLMs**: Analyzing the layer-wise behavior of video tokens may reveal similar "shallow-layer redundancy" phenomena, enabling analogous strategies for substantial compression.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Late Injection represents an entirely new perspective; analysis-driven design is an excellent research paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 benchmarks, 3 model scales, detailed ablations, efficiency analysis, and layer-wise behavioral visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ The analysis→insight→design narrative flows smoothly, with high-quality figures.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value — directly applicable for accelerating any LLaVA-architecture MLLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ICLR 2026\] IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)
- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)
- [\[AAAI 2026\] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](../../AAAI2026/multimodal_vlm/tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)

</div>

<!-- RELATED:END -->
