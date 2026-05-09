---
title: >-
  [Paper Note] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs
description: >-
  [ICLR 2026][Model Compression][visual token pruning] Through systematic empirical analysis using erank (effective rank) and attention entropy, this work reveals the complementary nature of attention-based and diversity-based visual token pruning methods — attention methods suppress hallucinations but suffer from limited coverage, while diversity methods achieve broad coverage but tend to introduce hallucinations. Based on these findings, AgilePruner is proposed to adaptively switch pruning strategies according to image complexity, achieving robust performance across 9 benchmarks.
tags:
  - ICLR 2026
  - Model Compression
  - visual token pruning
  - attention
  - diversity
  - hallucination
  - adaptive pruning
date: 2026-05-08
content_hash: 2a357e1a4d5c201d
---

# AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs

**Conference**: ICLR 2026
**arXiv**: [2603.01236](https://arxiv.org/abs/2603.01236)
**Code**: [https://cvsp-lab.github.io/AgilePruner](https://cvsp-lab.github.io/AgilePruner)
**Area**: Model Compression
**Keywords**: visual token pruning, attention, diversity, hallucination, adaptive pruning

## TL;DR
Through systematic empirical analysis using erank (effective rank) and attention entropy, this work reveals the complementary nature of attention-based and diversity-based visual token pruning methods — attention methods suppress hallucinations but suffer from limited coverage, while diversity methods achieve broad coverage but tend to introduce hallucinations. Based on these findings, AgilePruner is proposed to adaptively switch pruning strategies according to image complexity, achieving robust performance across 9 benchmarks.

## Background & Motivation
**Background**: Visual tokens in LVLMs are highly redundant (often numbering in the hundreds), leading to poor inference efficiency. Existing pruning methods fall into two camps: attention-based methods (retaining high-attention tokens) and diversity-based methods (retaining tokens with the most dispersed features), with some hybrid strategies also proposed.

**Limitations of Prior Work**: The relative merits of these approaches remain poorly understood — (1) How much diversity do diversity-based methods actually preserve? (2) What is the relationship between diversity and hallucination? (3) Which strategy is more suitable for different types of images? These questions lack systematic investigation.

**Key Challenge**: Attention-based methods perform well on simple images but provide insufficient coverage, while diversity-based methods perform better on complex images but are prone to hallucinations. No single method is universally optimal.

**Goal**: To empirically reveal the fundamental behavioral differences between the two paradigms and design an adaptive pruning strategy accordingly.

**Key Insight**: Effective rank (erank) is used to quantify feature diversity, and attention entropy is used to quantify attention concentration, serving as both analytical tools and the basis for adaptive switching.

**Core Idea**: Adaptively switch between attention-based and diversity-based pruning according to the attention entropy of each image.

## Method

### Overall Architecture
AgilePruner consists of two components: (1) a systematic empirical analysis revealing the essential behaviors of pruning methods, and (2) a simple adaptive pruning mechanism derived from analytical insights — tokens are traversed in descending order of attention score, redundant tokens are removed via a similarity threshold, and the threshold is adaptively adjusted according to image complexity (attention entropy).

### Key Designs

1. **erank-based Diversity Analysis**:

    - Function: Quantifies the feature diversity of the token sets retained by different pruning methods using effective rank (erank).
    - Mechanism: SVD is applied to the embedding matrix of retained tokens, and the entropy of the singular value distribution is computed as erank. Results: DivPrune (21.84) >> VisPruner (14.35) ≈ VisionZip (14.02) >> PruMerge+ (10.91).
    - Key Findings: (1) Many methods claiming to preserve diversity exhibit surprisingly low actual diversity; (2) High erank is strongly correlated with high hallucination rates — DivPrune achieves CHAIR $C_S$=57.4 vs. ~45 for attention-based methods.

2. **Image Complexity Dependency Analysis**:

    - Function: Analyzes which pruning strategy is more suitable for images of different complexity levels.
    - Mechanism: Simple images (low attention entropy, low erank) concentrate information in few tokens, favoring attention-based methods; complex images (high attention entropy, high erank) distribute information across many tokens, favoring diversity-based methods.
    - Experimental Validation: Attention-based methods outperform on ScienceQA (simple), while diversity-based methods outperform on POPE (complex).

3. **Adaptive Pruning Mechanism**:

    - Function: Dynamically adjusts the similarity threshold based on the image's attention entropy.
    - Mechanism: Tokens are traversed in descending order of attention score. For each candidate token, the maximum similarity to already-retained tokens is checked; tokens exceeding the threshold are considered redundant and removed. The threshold $\tau$ is set adaptively based on attention entropy — low entropy (simple images) uses a high threshold (more aggressively retaining high-attention tokens), while high entropy (complex images) uses a low threshold (retaining more diverse tokens).
    - Design Motivation: To directly translate empirical findings into an actionable mechanism with minimal design complexity.

### Loss & Training
Training-free. The method operates purely as a token pruning strategy at inference time.

## Key Experimental Results

### Main Results (Average Performance across 9 Benchmarks)

| Method | Type | POPE | ScienceQA | MME | CHAIR $C_S$↓ |
|------|------|------|-----------|-----|-------------|
| FasterVLM | Attention | - | Strong | - | 45.4 |
| DivPrune | Diversity | Strong | - | - | 57.4 |
| PruMerge+ | Hybrid | - | - | - | 45.2 |
| **AgilePruner** | **Adaptive** | **Robust** | **Robust** | **Robust** | **Low** |

### Ablation Study (Attention vs. Diversity Ratio on CHAIR)

| Attention Ratio R | $C_S$↓ | $C_I$↓ | Recall↑ | erank |
|-------------|--------|--------|---------|-------|
| 0 (pure diversity) | 57.4 | 18.0 | 76.4 | 21.14 |
| 0.25 | 50.8 | 16.8 | 74.5 | 14.98 |
| 0.50 | 46.2 | 14.5 | 73.7 | 14.38 |
| 0.75 | 45.2 | 14.1 | 70.5 | 13.58 |

### Key Findings
- **Diversity ↔ Hallucination positive correlation**: Increasing the attention token ratio from 0 to 0.75 reduces $C_S$ from 57.4 to 45.2, but recall drops from 76.4 to 70.5 — a clear trade-off.
- The same trend is observed on LLaVA-1.5-13B, LLaVA-NeXT-7B, and Qwen2.5-VL-7B, indicating model-agnostic findings.
- Applying the image-complexity-adaptive strategy to existing hybrid methods consistently improves performance, validating the generalizability of the empirical findings.

## Highlights & Insights
- **Counter-intuitive finding that "diversity induces hallucination"**: It was previously assumed that retaining more diverse tokens is always beneficial; this work reveals otherwise — retaining more diverse but low-attention tokens tends to introduce spurious information.
- **erank as an analytical tool**: Using effective rank to quantify the feature diversity of token sets is a concise and effective metric that can be reused in other settings requiring evaluation of token selection quality.
- **Simple yet effective adaptive strategy**: Without complex design, merely using attention entropy for threshold adjustment achieves robust cross-scenario performance.

## Limitations & Future Work
- The adaptive threshold setting still relies on hyperparameters that may require tuning across different models.
- The causal relationship between erank and hallucination has not been fully established (whether it arises from the retention of specific token types rather than diversity per se remains unclear).
- Validation is primarily conducted on 7B/13B models; behavior on larger models (70B+) is unknown.
- Analysis of video understanding and high-resolution multi-patch scenarios is insufficient.

## Related Work & Insights
- **vs. VisionZip/FasterVLM (attention-based methods)**: AgilePruner analyzes their limitations on complex images and complements them with diversity signals.
- **vs. DivPrune (diversity-based method)**: Reveals its high hallucination risk and mitigates it through attention-based constraints.
- **vs. PruMerge+/VisPruner (hybrid methods)**: Demonstrates that applying the image-complexity-adaptive strategy to these methods consistently improves performance.

## Rating
- Novelty: ⭐⭐⭐⭐ In-depth empirical analysis; the diversity–hallucination relationship is a novel finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 benchmarks, CHAIR hallucination analysis, multi-model validation, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Analysis-driven narrative is clear and well-illustrated.
- Value: ⭐⭐⭐⭐ Provides an empirical foundation and practical guidance for token pruning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[NeurIPS 2025\] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](../../NeurIPS2025/model_compression/twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)

</div>

<!-- RELATED:END -->
