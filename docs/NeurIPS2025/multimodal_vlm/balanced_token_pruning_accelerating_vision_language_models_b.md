---
title: >-
  [Paper Note] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization
description: >-
  [NeurIPS 2025][Multimodal VLM][visual token pruning] This paper proposes Balanced Token Pruning (BTP), which jointly considers the impact of pruning on both the current layer (local) and subsequent layers (global). BTP e…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "visual token pruning"
  - "local-global optimization"
  - "attention pruning"
  - "diversity pruning"
  - "LVLM inference acceleration"
date: 2026-05-08
content_hash: 6f5a56f4b8b5784d
---

# Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.22038](https://arxiv.org/abs/2505.22038)  
**Code**: [https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning](https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning)  
**Area**: Multimodal VLM / Model Acceleration
**Keywords**: visual token pruning, local-global optimization, attention pruning, diversity pruning, LVLM inference acceleration

## TL;DR

This paper proposes Balanced Token Pruning (BTP), which jointly considers the impact of pruning on both the current layer (local) and subsequent layers (global). BTP emphasizes diversity preservation in shallow layers to maintain downstream representation quality, and attention-based selection in deep layers to preserve local output consistency. On multiple LVLMs including LLaVA and Qwen2.5-VL, BTP retains 98% of the original model's performance while keeping only 22% of visual tokens.

## Background & Motivation

**Background**: LVLMs convert images into large numbers of tokens via visual encoders (e.g., 576 tokens for LLaVA-1.5 and up to 2880 for LLaVA-NeXT). The sheer volume of visual tokens constitutes the primary bottleneck for inference efficiency, particularly in edge deployment scenarios. Visual token pruning has emerged as the dominant acceleration strategy.

**Limitations of Prior Work**: Existing pruning methods fall into two categories — attention-based methods (FastV, PyramidDrop) select important tokens based on the attention scores from text to image tokens, while diversity-based methods (DivPrune) maximize the semantic diversity of retained tokens. Both categories, however, suffer from blind spots: attention-based methods optimize only for output consistency at the current layer (local optimum), ignoring the cascading effects of pruning on subsequent layers — different layers attend to different image regions, and tokens that are unimportant at the current layer may be critical at deeper layers. Diversity-based methods better preserve information required by subsequent layers, but fail to maintain output consistency at the local layer level.

**Key Challenge**: Through visualization experiments, the authors identify a key phenomenon: attention pruning achieves high output similarity at the pruning layer, but errors accumulate progressively in subsequent layers; diversity pruning yields lower output similarity at the pruning layer, yet achieves better consistency at deeper layers. This demonstrates that each family of methods optimizes only one aspect of the problem.

**Goal**: To jointly account for the effects of pruning on both the current and subsequent layers, striking a balance between local and global objectives.

**Key Insight**: Since attention-based methods excel at local consistency while diversity-based methods excel at global representation, one can dynamically adjust the weight between the two at different pruning stages — emphasizing diversity retention in shallow layers (where more downstream layers remain) and attention focus in deep layers (where fewer tokens remain and their impact is concentrated).

**Core Idea**: In shallow layers, diversity serves as an information "reserve" for subsequent layers; in deep layers, attention scores serve as a "safety net" for current output quality.

## Method

### Overall Architecture

BTP first uses a small calibration set (64 samples) to determine pruning layer positions and stage boundaries, then selects tokens to retain at each stage according to a local-global objective function. The input is the complete visual token sequence; the output is a progressively compressed token subset. The entire method is plug-and-play, requiring no modification to model architecture or retraining.

### Key Designs

1. **Local-Global Joint Objective Function**:

    - Function: Unifies the attention and diversity optimization objectives.
    - Mechanism: The objective function $\mathcal{L}_{local-global} = -\sum_{i}(\lambda_i \sum_{j \in P_i} Atten^{(i)}(X_I^{(j)}, X_T) + (1-\lambda_i) F_{dis}(P_i))$, where the first term is the attention objective (maintaining current-layer output consistency) and the second term is the diversity objective (preserving information needed by subsequent layers). $\lambda_i$ increases with layer depth — small $\lambda$ in shallow layers emphasizes diversity, large $\lambda$ in deep layers emphasizes attention.
    - Design Motivation: Shallow layers retain more tokens and can accommodate diverse information for downstream use; deep layers have fewer tokens and must prioritize accuracy of the current output.

2. **Position-Based Attention Re-balancing**:

    - Function: Eliminates selection bias introduced by positional encodings.
    - Mechanism: Attention scores are influenced by positional encodings, causing tokens at later positions to receive disproportionately higher scores. BTP first over-selects $k' > k$ candidate tokens, then retains all candidates from the first half of the sequence and fills the remaining $k - |I_{pre}|$ slots from the second half ranked by attention score, ensuring tokens at earlier positions are not eliminated due to positional bias.
    - Design Motivation: Directly ranking by attention scores discards valuable early-position tokens due to positional bias.

3. **Spatial-Position-Based Diversity Initialization**:

    - Function: Reduces the $O(n^2)$ Max-Min Diversity Problem to a practically efficient form.
    - Mechanism: Observing that spatially distant image patches tend to have large semantic differences while nearby patches are similar, BTP first solves a spatial MMDP on the 2D grid using Manhattan distance as initialization, then performs only a small number of additional selections. This avoids solving the full MMDP in high-dimensional semantic space.
    - Design Motivation: DivPrune's MMDP solver is $O(n^2)$ and cannot be GPU-accelerated, resulting in actual inference latency that exceeds the unpruned model.

### Automatic Pruning Layer Selection

By computing cosine similarity of image token hidden states between adjacent layers, layers with abrupt semantic changes are identified — these transition points are ideal pruning locations. Due to the causal mask, image token encoding is independent of the input question, allowing a fixed set of 64 calibration samples to determine pruning layers in a task-agnostic manner.

## Key Experimental Results

### Main Results (LLaVA-1.5-7B, 128 tokens)

| Method | #Tokens | GQA | MME | MMB | POPE | SQA | Avg% |
|--------|---------|-----|-----|-----|------|-----|------|
| Original | 576 | 62.0 | 1510.7 | 64.3 | 85.8 | 69.4 | 100% |
| VTW | 236 | 51.3 | 1475.0 | 63.4 | 82.1 | 68.8 | 89% |
| FastV | 172 | 57.6 | 1465.0 | 61.6 | 81.0 | 68.9 | 96% |
| DivPrune | 128 | 58.8 | 1405.4 | 62.1 | 85.1 | 68.4 | 96% |
| **BTP** | **128** | **59.0** | **1487.0** | **62.7** | **85.6** | **69.1** | **98%** |

### Ablation Study ($\lambda$ Sensitivity)

| Configuration | Description | Performance Trend |
|---------------|-------------|-------------------|
| Shallow fixed, mid-deep adjusted | Shallow layers biased toward attention too early | Performance degrades |
| Mid fixed, shallow-deep adjusted | Mid layers require moderate diversity | Optimal range is narrow |
| Deep fixed, shallow-mid adjusted | Deep layers should emphasize attention | Performance stable |

### Efficiency Comparison (LLaVA-1.5-7B)

| Method | #Tokens | Latency | TFLOPS |
|--------|---------|---------|--------|
| Original | 576 | 0.145s | 3.82 |
| DivPrune | 128 | 0.224s (54%↑) | 0.83 |
| **BTP** | **128** | **0.134s (7%↓)** | **0.85** |

### Key Findings

- BTP maintains 98% of original performance at 128 tokens, outperforming all baselines (DivPrune and FastV both at 96%).
- Although DivPrune achieves the lowest TFLOPS, its actual latency is 54% higher than the unpruned model due to GPU-incompatible MMDP solving; BTP's spatial initialization resolves this issue, reducing actual latency by 7%.
- On Qwen2.5-VL-7B, a dynamic-resolution model, BTP retains 97% performance at 25% token retention, whereas VTW drops sharply to 65%.
- KV cache on LLaVA-1.6-7B is reduced from 1.11 GB to 0.28 GB (a 74.7% reduction).

## Highlights & Insights

- The **in-depth empirical analysis of attention vs. diversity pruning** is the paper's most significant contribution — by visualizing hidden-state similarity layer by layer, the authors clearly reveal the complementary nature of the two approaches, providing a solid empirical foundation for the staged strategy.
- **Spatial-position-based diversity initialization** elegantly reduces the high-dimensional semantic diversity problem to a 2D spatial distance problem, simultaneously ensuring quality and resolving DivPrune's practical deployment issue (where pruning paradoxically increases latency).
- **Transferable design principle**: The idea of emphasizing diversity in shallow layers and attention in deep layers can be applied to other layer-wise pruning and compression tasks.

## Limitations & Future Work

- Although small (64 samples), the calibration set still requires additional data; a fully data-free adaptive pruning strategy warrants exploration.
- Only four models are evaluated; generalization to larger-scale models (70B+) remains unverified.
- The monotonically increasing $\lambda$ schedule across stages is manually specified; learning an optimal $\lambda$ schedule is a natural extension.
- Both diversity and attention objectives assume tokens are processed independently, without modeling inter-token interactions.

## Related Work & Insights

- **vs. FastV**: FastV directly prunes by attention scores after a certain layer, representing purely local optimization. BTP achieves better performance with fewer tokens.
- **vs. DivPrune**: DivPrune's pure diversity strategy incurs large local-level losses, and the slow MMDP solver increases actual inference latency. BTP's spatial initialization combined with the balanced strategy comprehensively surpasses DivPrune in both efficiency and quality.
- **vs. PyramidDrop**: The staged pruning concept is similar, but PyramidDrop applies attention ranking at every stage. This work demonstrates that such an approach is not globally optimal.

## Rating

- Novelty: ⭐⭐⭐⭐ The local-global joint optimization perspective for pruning is novel and well-analyzed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-model, cross-compression-ratio evaluation with comprehensive efficiency analysis and solid ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from empirical analysis to method design is clear and coherent.
- Value: ⭐⭐⭐⭐ Practically significant for LVLM deployment acceleration; the plug-and-play design ensures strong usability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/multimodal_vlm/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)

</div>

<!-- RELATED:END -->
