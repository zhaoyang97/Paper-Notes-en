---
title: >-
  [Paper Note] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models
description: >-
  [ICLR 2026][vlm_efficiency][Vision-Language Model] It is observed that KV Cache in LVLMs exhibits modal-specific and head-specific semantic redundancy. Since selection based solely on importance leads to a loss of semantic coverage, MixKV is proposed to adaptively mix importance and diversity scores per head for KV Cache compression, achieving an average improvement of
tags:
  - ICLR 2026
  - vlm_efficiency
  - Vision-Language Model
date: 2026-05-08
content_hash: 2698e6a13c5e625d
---
# Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models

**Conference**: ICLR 2026  
**arXiv**: [2510.20707](https://arxiv.org/abs/2510.20707)  
**Code**: [GitHub](https://github.com/xuyang-liu16/MixKV)  
**Area**: Multimodal VLM/Inference Efficiency  
**Keywords**: KV Cache Compression, Semantic Redundancy, Diversity, Attention Head, Vision-Language Model

## TL;DR
It is observed that KV Cache in LVLMs exhibits modal-specific and head-specific semantic redundancy. Since selection based solely on importance leads to a loss of semantic coverage, MixKV is proposed to adaptively mix importance and diversity scores per head for KV Cache compression, achieving an average improvement of 5.1% under extreme compression.

## Background & Motivation

**Background**: When LVLMs process high-resolution images and long videos, they generate a vast number of KV pairs, making the KV Cache a memory bottleneck. Existing methods (SnapKV, AdaKV, etc.) retain key KV pairs based on attention importance and discard others.

**Limitations of Prior Work**: (1) Visual information has higher semantic redundancy than text—the cosine similarity between KV pairs for images reaches 0.6-0.8 (compared to only 0.2-0.4 for text) due to similar textures or repetitive patterns; (2) Redundancy varies significantly across different attention heads—some heads have average similarities $>0.9$, while others are $<0.3$; (3) Selection based only on importance results in the retention of highly similar KV pairs, leading to a loss of global semantic coverage.

**Key Challenge**: t-SNE visualizations clearly demonstrate that KV pairs selected by SnapKV (importance-only) cover only a small subset of the full distribution, resulting in substantial information loss.

**Key Insight**: Introduce diversity alongside importance—high-redundancy heads (where KV pairs are highly similar) should emphasize diversity to avoid redundancy, while low-redundancy heads maintain priority for importance.

**Core Idea**: Adaptively use redundancy as a mixing weight for importance and diversity scores on a per-head basis.

## Method

### Overall Architecture
MixKV addresses the limitation of existing KV Cache compression methods that select KV pairs based only on attention importance, which leads to the retention of mutually similar pairs and a loss of global semantic coverage in highly redundant visual contexts. The mechanism calculates two types of scores for each KV pair—an importance score measuring "whether it should be kept" and a diversity score measuring "whether it would be redundant if kept." These are mixed into a composite score based on the specific redundancy level of each attention head. Finally, the top $B$ KV pairs with the highest composite scores are retained. This workflow does not modify the importance calculation of the original methods but inserts diversity during the scoring phase, allowing it to be used as a plug-and-play enhancement for any importance-based compression method like SnapKV or AdaKV.

### Key Designs

**1. Redundancy Quantization: Characterizing per-head redundancy with an $O(T)$ scalar**

Redundancy varies greatly across attention heads—some heads exhibit average KV pair similarities exceeding 0.9, while others are below 0.3. Thus, the mixing weights must be determined per head, requiring an efficient redundancy metric. MixKV takes the normalized Key vectors for each head and calculates the average cosine similarity $\bar{r}_h^l$ between them as the head's redundancy. Since direct pairwise comparison is $O(T^2)$, the paper uses the algebraic identity $\sum_{i,j} R_{i,j} = T^2 \|\hat{\bar{K}}_h^l\|_2^2$ to collapse the summation into the "norm of the average Key," allowing $\bar{r}_h^l = \frac{T^2\|\hat{\bar{K}}_h^l\|_2^2 - T}{T(T-1)}$ to be calculated in $O(T)$. As $\bar{r} \to 1$, the Keys in the head are highly similar, necessitating an emphasis on diversity; as $\bar{r} \to 0$, they are sufficiently dispersed, and importance remains the priority.

**2. Diversity Score: Approximating "uniqueness" via deviation from the average Key**

To avoid the $O(T^2)$ cost of pairwise comparisons, MixKV does not calculate the relationship between each Key and all others. Instead, it calculates the relationship with the global average Key: the negative cosine similarity $s_i^{\text{div}} = -\hat{K}_{h,i}^l \cdot \hat{\bar{K}}_h^l$ serves as the diversity score. Keys that are less similar to the mean are considered more unique and carry distinct information, thus receiving higher scores. Keys clustered near the mean are redundant and receive lower scores. This keeps the complexity of the diversity calculation at $O(T)$, matching the importance score.

**3. Adaptive Mixing: Higher redundancy heads prioritize diversity**

With the per-head redundancy and the two scores for each Key, the final composite score is calculated as $s_i^{\text{comp}} = (1-\bar{r}_h^l) \cdot s_{\text{imp},i} + \bar{r}_h^l \cdot s_{\text{scaled},i}^{\text{div}}$. The weight is directly determined by the redundancy $\bar{r}_h^l$: the more redundant the head, the larger the proportion of diversity, which actively skips KV pairs that are "important but highly similar to already selected ones." If the head is naturally dispersed, the method reverts to pure importance selection. This adaptive, per-head mixing ratio distinguishes MixKV from methods using fixed global hyperparameters.

**4. Importance Score Enhancement: Integrating dual importance signals**

To ensure comprehensive importance signals, $s_{\text{imp}}$ integrates both extrinsic and intrinsic sources: extrinsic importance $s_{\text{imp}}^{\text{ex}}$ comes from the attention window (how much the KV pair is attended to by subsequent queries), and intrinsic importance $s_{\text{imp}}^{\text{in}}$ comes from VNorm (the norm of the Value vector itself). Adding these provides $s_{\text{imp}} = s_{\text{imp}}^{\text{ex}} + s_{\text{imp}}^{\text{in}}$.

## Key Experimental Results

### Main Results
Multimodal understanding under extreme compression (budget=64):

| Method | DocVQA | OCRBench | TextVQA | ChartQA | Avg. Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SnapKV | 47.3 | 31.9 | 57.1 | 42.7 | — |
| SnapKV+MixKV | **48.8** | **36.1** | **59.0+** | **45+** | +5.1% |
| AdaKV | Baseline | Baseline | Baseline | Baseline | — |
| AdaKV+MixKV | **+** | **+** | **+** | **+** | +5.1% |

### GUI Grounding Task (ScreenSpot-v2)

| Method | Accuracy | Description |
| :--- | :--- | :--- |
| SnapKV | Baseline | budget=64 |
| SnapKV+MixKV | **+8.0%** | Diversity is crucial for UI element localization |
| AdaKV+MixKV | **+9.0%** | Greater improvement |

### Key Findings
- t-SNE visualization confirms that MixKV allows SnapKV's selection to cover a broader distribution of KV pairs.
- The largest improvements occur in GUI Grounding tasks (+8-9%) because UI elements are scattered; diversity selection covers more positional information.
- Inference efficiency is comparable to baseline methods—both redundancy and diversity scores are $O(T)$ calculations.
- Consistent improvements are also observed on text-only LLMs (Qwen2.5, Llama-3.1).

## Highlights & Insights
- **Quantitative Analysis of Visual KV Redundancy**: The first systematic quantification of modal-specific and head-specific redundancy in LVLMs. The surge in cosine similarity from 0.2-0.4 in LLMs to 0.6-0.8 in LVLMs provides strong justification.
- **Intuition from t-SNE**: A single visualization explains why importance alone is insufficient—SnapKV's selected points cover only a corner of the distribution, while MixKV covers it more broadly.
- **$O(T)$ Redundancy Calculation**: Utilizing algebraic identities to avoid $O(T^2)$ pairwise comparisons ensures practical feasibility.

## Limitations & Future Work
- The diversity score only considers Keys (not Values); Value redundancy patterns may differ.
- Whether negative cosine similarity is the optimal proxy for diversity remains unexplored; other distance metrics were not tested.
- The global average Key as an anchor point might be sensitive to outliers.
- Validation was mostly on 7-8B models; the effect on larger models (70B+) is unknown.

## Related Work & Insights
- **vs SnapKV**: SnapKV uses only attention importance; MixKV adds diversity on top of it, providing a plug-and-play +5.1% boost.
- **vs AdaKV**: AdaKV adaptively allocates eviction budgets across heads; MixKV adaptively allocates importance vs. diversity weights per head. The two are orthogonal and stackable.
- **vs SparseMM**: SparseMM uses head importance for asymmetric budgets; MixKV focuses on internal redundancy characteristics of the heads.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of mixing importance and diversity is clear and effective; the redundancy analysis is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple models, tasks, budgets, and plug-and-play scenarios.
- Writing Quality: ⭐⭐⭐⭐ Rich visualizations and clear methodology descriptions.
- Value: ⭐⭐⭐⭐ Direct practical value for LVLM deployment optimization.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity](prune_redundancy_preserve_essence_vision_token_compression_in_vlms_via_synergist.md)
- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/vlm_efficiency/flashcache_frequency_kv_cache_compression.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/vlm_efficiency/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)
- [\[ICLR 2026\] LearnPruner: Rethinking Attention-based Token Pruning in Vision Language Models](learnpruner_rethinking_attention-based_token_pruning_in_vision_language_models.md)

</div>

<!-- RELATED:END -->
