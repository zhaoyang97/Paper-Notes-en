---
title: >-
  [Paper Note] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][KV Cache Compression] This paper identifies modality-specific and attention-head-specific semantic redundancy in the KV Cache of LVLMs…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "KV Cache Compression"
  - "Semantic Redundancy"
  - "Diversity"
  - "Attention Heads"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 4186b7bf1a0714f6
---

# Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.20707](https://arxiv.org/abs/2510.20707)
**Code**: [GitHub](https://github.com/xuyang-liu16/MixKV)
**Area**: Multimodal VLM / Inference Efficiency
**Keywords**: KV Cache Compression, Semantic Redundancy, Diversity, Attention Heads, Vision-Language Models

## TL;DR
This paper identifies modality-specific and attention-head-specific semantic redundancy in the KV Cache of LVLMs, demonstrating that importance-only selection fails to preserve semantic coverage. The proposed MixKV adaptively mixes importance and diversity scores per attention head for KV Cache compression, achieving an average improvement of 5.1% under extreme compression ratios.

## Background & Motivation

**Background**: LVLMs generate a large number of KV pairs when processing high-resolution images and long videos, making the KV Cache a memory bottleneck. Existing methods (e.g., SnapKV, AdaKV) retain critical KV pairs based on attention importance scores while discarding less important ones.

**Limitations of Prior Work**: (1) Visual information exhibits greater semantic redundancy than text — similar textures and repetitive patterns in images lead to cosine similarities between KV pairs as high as 0.6–0.8 (compared to 0.2–0.4 for text); (2) Redundancy varies substantially across attention heads — some heads exhibit average similarity above 0.9 while others fall below 0.3; (3) Importance-only selection retains highly similar KV pairs, resulting in a loss of global semantic coverage.

**Key Challenge**: t-SNE visualizations clearly show that KV pairs selected by SnapKV (importance-only) cover only a small subset of the full distribution, leading to significant information loss.

**Key Insight**: Incorporating diversity alongside importance — heads with high redundancy (high KV similarity) should emphasize diversity to avoid retaining redundant pairs, while heads with low redundancy should prioritize importance.

**Core Idea**: Use per-head redundancy as an adaptive mixing weight between importance and diversity scores.

## Method

### Overall Architecture
MixKV is a plug-and-play framework: it computes an importance score and a diversity score for each KV pair, adaptively mixes them according to per-head redundancy, and retains the top-$B$ pairs. It is compatible with any existing importance-based KV compression method.

### Key Designs

1. **Head-wise Redundancy Quantification**:

   - Function: Computes the off-diagonal mean cosine similarity $\bar{r}_h^l$ of normalized Key vectors for each attention head.
   - Mechanism: Exploits the algebraic identity $\sum_{i,j} R_{i,j} = T^2 \|\hat{\bar{K}}_h^l\|_2^2$ to compute $\bar{r}_h^l = \frac{T^2\|\hat{\bar{K}}_h^l\|_2^2 - T}{T(T-1)}$ in $O(T)$ time.
   - Design Motivation: $\bar{r} \to 1$ indicates a highly redundant head where diversity should be emphasized; $\bar{r} \to 0$ indicates an already diverse head where importance should be prioritized.

2. **Diversity Score**:

   - Function: Measures each Key's dissimilarity from the global mean Key as a diversity score.
   - Mechanism: $s_i^{\text{div}} = -\hat{K}_{h,i}^l \cdot \hat{\bar{K}}_h^l$ — keys less similar to the mean receive higher scores.
   - Design Motivation: $O(T)$ complexity, requiring no pairwise comparisons.

3. **Head-wise Adaptive Mixing**:

   - Function: Combines importance and diversity scores weighted by per-head redundancy.
   - Mechanism: $s_i^{\text{comp}} = (1-\bar{r}_h^l) \cdot s_{\text{imp},i} + \bar{r}_h^l \cdot s_{\text{scaled},i}^{\text{div}}$
   - Design Motivation: Higher redundancy increases the diversity weight, preventing retention of overly similar KV pairs.

### Importance Score Enhancement
Intrinsic (VNorm) and extrinsic (attention window) importance are jointly integrated: $s_{\text{imp}} = s_{\text{imp}}^{\text{ex}} + s_{\text{imp}}^{\text{in}}$

## Key Experimental Results

### Main Results
Multimodal understanding under extreme compression (budget = 64):

| Method | DocVQA | OCRBench | TextVQA | ChartQA | Avg. Gain |
|--------|--------|----------|---------|---------|-----------|
| SnapKV | 47.3 | 31.9 | 57.1 | 42.7 | — |
| SnapKV+MixKV | **48.8** | **36.1** | **59.0+** | **45+** | +5.1% |
| AdaKV | baseline | baseline | baseline | baseline | — |
| AdaKV+MixKV | **+** | **+** | **+** | **+** | +5.1% |

### GUI Grounding (ScreenSpot-v2)

| Method | Accuracy | Notes |
|--------|----------|-------|
| SnapKV | baseline | budget = 64 |
| SnapKV+MixKV | **+8.0%** | Diversity is critical for UI element localization |
| AdaKV+MixKV | **+9.0%** | Larger gain |

### Key Findings
- t-SNE visualizations confirm that MixKV enables SnapKV's selection to cover a substantially broader region of the KV distribution.
- GUI Grounding tasks yield the largest gains (+8–9%) — UI elements are spatially dispersed, so diversity-aware selection captures more positional information.
- Inference efficiency is on par with baseline methods, as both redundancy and diversity scores are computed in $O(T)$.
- Consistent improvements are also observed on text-only LLMs (Qwen2.5, Llama-3.1).

## Highlights & Insights
- **Quantitative Analysis of Visual KV Redundancy**: This work is among the first to systematically quantify modality-specific and head-specific redundancy in LVLM KV caches. The increase in cosine similarity from 0.2–0.4 in LLMs to 0.6–0.8 in LVLMs is a compelling empirical finding.
- **Intuitive t-SNE Visualization**: A single figure effectively illustrates the failure mode of importance-only selection — SnapKV's selected points cluster in one corner of the distribution, while MixKV achieves broader coverage.
- **$O(T)$ Redundancy Computation**: An algebraic identity is leveraged to avoid the $O(T^2)$ cost of pairwise comparisons, ensuring practical applicability.

## Limitations & Future Work
- The diversity score only considers Keys, not Values; redundancy patterns in Values may differ.
- Whether negative cosine similarity is the optimal diversity proxy remains unexplored; alternative distance metrics are not investigated.
- The global mean Key used as an anchor may be sensitive to outliers.
- Evaluation is limited to 7–8B models; effectiveness on larger models (70B+) is unknown.

## Related Work & Insights
- **vs. SnapKV**: SnapKV relies solely on attention importance; MixKV augments it with diversity in a plug-and-play manner, yielding +5.1%.
- **vs. AdaKV**: AdaKV adaptively allocates eviction budgets across heads; MixKV adaptively balances importance vs. diversity within each head — the two approaches are orthogonal and complementary.
- **vs. SparseMM**: SparseMM assigns asymmetric budgets based on head importance; MixKV focuses on intra-head redundancy characteristics.

## Rating
- Novelty: ⭐⭐⭐⭐ The importance–diversity mixing framework is clear and effective; the redundancy analysis is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across multiple models, tasks, compression budgets, and plug-and-play settings.
- Writing Quality: ⭐⭐⭐⭐ Rich analytical visualizations and clear method description.
- Value: ⭐⭐⭐⭐ Directly applicable to practical LVLM deployment optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/multimodal_vlm/flashcache_frequency_kv_cache_compression.md)
- [\[ICCV 2025\] AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/multimodal_vlm/aircache_activating_inter_modal_relevancy_kv_cache_compression_for_efficient_large_vision_language_model.md)
- [\[NeurIPS 2025\] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](../../NeurIPS2025/multimodal_vlm/prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ICLR 2026\] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)

</div>

<!-- RELATED:END -->
