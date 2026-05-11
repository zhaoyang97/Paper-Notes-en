---
title: >-
  [Paper Note] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][Visual Token Pruning] This paper proposes SCOPE, a visual token pruning strategy that jointly models saliency and coverage. By iteratively selecting tokens with the highest SCOPE scores…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Visual Token Pruning"
  - "Multimodal LLM Inference Acceleration"
  - "Semantic Coverage"
  - "Submodular Functions"
  - "Training-Free"
date: 2026-05-08
content_hash: c1e405d7ec465d00
---

# SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

**Conference**: NeurIPS 2025  
**arXiv**: [2510.24214](https://arxiv.org/abs/2510.24214)  
**Code**: [https://github.com/kinredon/SCOPE](https://github.com/kinredon/SCOPE)  
**Area**: Multimodal VLM  
**Keywords**: Visual Token Pruning, Multimodal LLM Inference Acceleration, Semantic Coverage, Submodular Functions, Training-Free

## TL;DR
This paper proposes SCOPE, a visual token pruning strategy that jointly models saliency and coverage. By iteratively selecting tokens with the highest SCOPE scores, it preserves semantic completeness and retains 96% of LLaVA-1.5's performance under a 9× token reduction.

## Background & Motivation

**Background**: MLLMs encode images into large numbers of visual tokens (e.g., 576 or 2000+), which are fed into the LLM alongside text tokens. The quadratic complexity of self-attention results in substantial computational overhead.

**Limitations of Prior Work**: Saliency-based pruning methods (e.g., FastV, SparseVLM, VisionZip) retain only the tokens with the highest attention scores, which introduces two problems:
   - **Semantic incompleteness**: High-saliency tokens tend to cluster around a few objects, losing contextual information (e.g., answering "where is the cat" requires both the cat and its surrounding environment).
   - **Skewed attention distribution**: Only a small fraction of tokens receive high attention; the remaining tokens have nearly uniform attention scores, making it difficult to distinguish informative tokens from redundant ones.

**Key Challenge**: Prioritizing saliency leads to selected tokens with high semantic overlap, resulting in low coverage.

**Key Insight**: Drawing on the coverage function concept from submodular optimization, the paper proposes a selection strategy that jointly considers saliency and coverage.

## Method

### Overall Architecture
Token pruning is performed at a specified layer of the MLLM (e.g., layer 2). Cosine similarities between all pairs of visual tokens are computed, and tokens with the highest SCOPE scores are iteratively added to the retained set until the budget $K$ is reached.

### Key Designs

1. **Set-Coverage**:

    - **Function**: Quantifies the degree to which the selected token set semantically covers all tokens.
    - **Mechanism**: For each token $u$, its coverage is defined as the cosine similarity to its most similar token in the selected set: $C(u,\mathcal{S}) = \max_{s \in \mathcal{S}} \text{sim}(u,s)$. Total coverage: $f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max_{s \in \mathcal{S}} \text{sim}(u,s)$
    - **Design Motivation**: Encourages the selection of semantically diverse tokens, ensuring that every unselected token has at least one similar representative.

2. **Token-Coverage Gain**:

    - **Function**: Quantifies the additional coverage gained by adding a new token.
    - **Mechanism**: Marginal gain $\Delta(v;\mathcal{S}) = \sum_{u \in \mathcal{V}}[\max(C(u,\mathcal{S}), \text{sim}(u,v)) - C(u,\mathcal{S})]$
    - **Design Motivation**: Greedily selecting the token with the maximum marginal gain is a classic submodular function maximization strategy with a $(1-1/e)$ approximation guarantee.

3. **SCOPE Score**:

    - **Function**: Combines saliency and coverage gain.
    - **Mechanism**: $\Delta(v, A_v^\alpha; \mathcal{S}) = \Delta(v; \mathcal{S}) \cdot A_v^\alpha$, where $A_v$ is the attention score and $\alpha$ is a scaling factor.
    - **Design Motivation**: Pure coverage gain ignores the intrinsic informativeness of tokens. Multiplying by the attention score achieves a balance between coverage and importance.

### Loss & Training
The method is entirely training-free and plug-and-play. Pruning is performed once at a designated Transformer layer during inference.

## Key Experimental Results

### Main Results — LLaVA-1.5 7B, 64 Tokens Retained (↓88.9%)

| Benchmark | Vanilla (576) | FastV | SparseVLM | VisionZip | SCOPE | Relative Perf. |
|-----------|---------------|-------|-----------|-----------|-------|----------------|
| GQA | 61.9 | 52.7 | 57.6 | 59.3 | **60.3** | 97.4% |
| MME | 1862 | 1612 | 1721 | 1783 | **1805** | 97.0% |
| POPE | 85.9 | 64.8 | 83.6 | 85.3 | **85.6** | 99.7% |
| TextVQA | 58.2 | 52.5 | 56.1 | 56.3 | **57.0** | 97.9% |
| Avg. (relative) | 100% | 89.5% | 96.5% | 97.5% | **98.2%** | - |

### Ablation Study

| Strategy | θ-Coverage (θ=0.95) | GQA | MME |
|----------|---------------------|-----|-----|
| Saliency Only | 18.2% | 57.6 | 1721 |
| Coverage Only | 52.3% | 59.1 | 1778 |
| Random | 23.5% | 50.2 | 1512 |
| SCOPE (ours) | 48.7% | **60.3** | **1805** |

### Key Findings
- The θ-coverage of saliency-only methods is even **lower than random selection**, demonstrating that high-attention tokens are highly concentrated.
- SCOPE retains 96.0% of original performance at 192 tokens and 98.2% at 64 tokens.
- The method generalizes effectively to LLaVA-Next, demonstrating its versatility.
- The parameter $\alpha$ controls the saliency weight; $\alpha=0.5$ is optimal across most tasks.

## Highlights & Insights
- The **θ-coverage metric** is elegantly defined, providing a novel quantitative evaluation dimension for the visual token pruning field that can be applied to analyze any token selection strategy.
- The **greedy strategy for submodular function maximization** is naturally suited to the token selection problem. Although the connection is intuitive, it had not been previously explored—constituting the paper's core contribution.
- The time complexity of the method is $O(NK)$; for $N=576, K=64$, only approximately 36K similarity comparisons are required, with negligible impact on inference speed.

## Limitations & Future Work
- The current SCOPE framework performs a single pruning pass at one layer of the MLLM; progressive multi-layer pruning may yield better results (e.g., following the approach of PyramidDrop).
- Coverage is measured via cosine similarity as a proxy for semantic proximity, but cosine similarity in high-dimensional spaces may not precisely capture semantic relationships.
- The interaction between text tokens and visual tokens is not considered; adaptive pruning conditioned on query content could be explored.
- Extension to video understanding has not yet been validated.

## Related Work & Insights
- **vs FastV**: FastV prunes based on early-layer text-to-visual attention and selects only salient tokens; SCOPE additionally accounts for coverage.
- **vs DivPrune**: DivPrune maximizes diversity without considering saliency; SCOPE unifies both objectives.
- **vs VisionZip**: VisionZip uses CLS token attention combined with token merging; SCOPE's coverage gain provides a superior selection criterion.
- The core idea of this approach is transferable to long-context token compression in NLP.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of submodular coverage and saliency is concise and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple MLLMs, multiple benchmarks, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Complete mathematical derivations and intuitive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Training-free and plug-and-play; extremely practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)

</div>

<!-- RELATED:END -->
