---
title: >-
  [Paper Note] Video-KTR: Enhancing Video Reasoning via Key Token Attribution
description: >-
  [ICLR 2026][Video Understanding][Video Reasoning] This paper proposes Video-KTR, a modality-aware policy shaping framework that identifies three categories of key tokens—visual-aware, temporal-aware, and entropy-aware—via counterfactual analysis, and applies selective reinforcement learning updates exclusively to these tokens, achieving state-of-the-art performance across multiple video reasoning benchmarks (Video-Holmes 42.7%, surpassing GPT-4o).
tags:
  - ICLR 2026
  - Video Understanding
  - Video Reasoning
  - Reinforcement Learning
  - Token Attribution
  - Multimodal LLM
  - GRPO
date: 2026-05-08
content_hash: 2601e96d4cff5e44
---

# Video-KTR: Enhancing Video Reasoning via Key Token Attribution

**Conference**: ICLR 2026
**arXiv**: [2601.19686](https://arxiv.org/abs/2601.19686)
**Area**: Video Understanding
**Keywords**: Video Reasoning, Reinforcement Learning, Token Attribution, Multimodal LLM, GRPO

## TL;DR

This paper proposes Video-KTR, a modality-aware policy shaping framework that identifies three categories of key tokens—visual-aware, temporal-aware, and entropy-aware—via counterfactual analysis, and applies selective reinforcement learning updates exclusively to these tokens, achieving state-of-the-art performance across multiple video reasoning benchmarks (Video-Holmes 42.7%, surpassing GPT-4o).

## Background & Motivation

Reinforcement learning (RL) has demonstrated strong potential in improving the reasoning capabilities of multimodal LLMs; however, existing video reasoning methods suffer from three critical limitations:

**Coarse-grained rewards**: Reliance on sequence-level rewards prevents precise guidance on which tokens require focused learning.

**Single-factor token selection**: Token selection based solely on information entropy ignores modality-specific dependencies.

**Over-reliance on language priors**: The absence of fine-grained semantic alignment between visual inputs and output tokens increases the risk of hallucination.

Prior methods such as T-GRPO introduce temporal constraints (penalizing predictions after frame shuffling), but this constitutes a coarse global assumption that overlooks cases where certain tasks can be resolved using static visual cues alone.

## Method

### Overall Architecture

Video-KTR introduces a modality-aware, token-level policy shaping mechanism built upon the GRPO framework, consisting of three core steps: (1) multi-perspective token importance analysis; (2) token selection; and (3) selective policy updates.

### Key Designs: Three Attribution Signals

**1. Visual-Aware Tokens**

The dependence of each token on visual input is quantified via **counterfactual masking**, which zeroes out video features and computes the resulting logit change:

$$\Delta^{\text{vis}}_i = |\log \text{softmax}(\mathbf{z}^{\text{full}}_i)_{y_i} - \log \text{softmax}(\mathbf{z}^{\text{masked}}_i)_{y_i}|$$

Tokens with high $\Delta^{\text{vis}}_i$ (e.g., "person," "door," "blue") indicate that their predictions are strongly grounded in visual input.

**2. Temporal-Aware Tokens**

Sensitivity to temporal structure is detected via **frame order shuffling**:

$$\Delta^{\text{temp}}_i = |\log \text{softmax}(\mathbf{z}^{\text{ordered}}_i)_{y_i} - \log \text{softmax}(\mathbf{z}^{\text{shuffled}}_i)_{y_i}|$$

Tokens with high $\Delta^{\text{temp}}_i$ (e.g., "first," "then," "appear") reflect dependence on event order and causal relationships.

**3. Entropy-Aware Tokens**

Prediction uncertainty is captured to identify critical reasoning points:

$$\mathcal{H}(i) = -\sum_w p(z_i = w) \log p(z_i = w)$$

High-entropy tokens (e.g., "however," "wait") typically mark discourse transitions or decision points.

### Token Selection and Policy Update

The top $r\%$ tokens from each attribution strategy are selected, and their union $\mathcal{S} = \mathcal{S}_{\text{vis}} \cup \mathcal{S}_{\text{temp}} \cup \mathcal{S}_{\text{ent}}$ is used to construct a binary mask $m_{i,t}$. The modified GRPO objective is:

$$\mathcal{J}_{\text{Video-KTR}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} m_{i,t} \cdot \min(r_{i,t}\hat{A}_{i,t}, \text{clip}(r_{i,t})\hat{A}_{i,t})\right]$$

Only key tokens where $m_{i,t}=1$ contribute to the loss computation.

## Key Experimental Results

### Main Results: Cross-Benchmark Performance Comparison

| Model | Scale | Video-Holmes | VideoMMMU | MMVU(mc) | TempCompass | VideoMME |
|-------|-------|-------------|-----------|----------|-------------|---------|
| GPT-4o | — | 42.0 | 61.2 | 75.4 | 73.8 | 71.9 |
| GPT-5 | — | 46.7 | 84.6 | 82.6 | 83.3 | 86.7 |
| Video-R1 | 7B | 36.5 | 52.3 | 63.8 | 73.2 | 59.3 |
| TW-GRPO | 7B | 32.9 | 51.3 | 65.8 | 73.3 | 55.1 |
| **Video-KTR** | **7B** | **42.7** | **53.1** | **66.6** | **73.5** | **62.5** |

### Ablation Study: Attribution Signal Combinations

| Strategy | E | V | T | Video-Holmes | VideoMMMU | MMVU | Avg. |
|----------|---|---|---|-------------|-----------|------|------|
| Vanilla GRPO | ✗ | ✗ | ✗ | 38.8 | 49.8 | 64.8 | 51.1 |
| T only | ✗ | ✗ | ✓ | 42.1 | 50.1 | 65.5 | 52.6 |
| V only | ✗ | ✓ | ✗ | 40.5 | 51.9 | 65.1 | 52.5 |
| V+E+T | ✓ | ✓ | ✓ | **41.6** | **52.6** | **65.9** | **53.4** |

### Key Findings

1. **Complementarity of three signals**: Each individual signal outperforms vanilla GRPO, while the full combination yields the best performance.
2. **Hard selection outperforms soft weighting**: A top-20% binary mask consistently outperforms Softmax/Sigmoid/linear/exponential weighting schemes.
3. **Differentiated linguistic distributions**: Visual tokens are predominantly nouns (24.8%), temporal tokens are predominantly verbs (21.2%), and entropy tokens exhibit a higher proportion of adverbs (8.8%).
4. **Optimal update ratio is 20%**: Higher ratios introduce noise, while lower ratios provide insufficient signal.

## Highlights & Insights

1. **Elegant application of counterfactual analysis**: Visual masking and frame shuffling naturally disentangle visual and temporal dependencies through two complementary perturbations.
2. **Plug-and-play design**: Video-KTR can be seamlessly integrated into any GRPO-based RL training pipeline.
3. **7B model surpassing GPT-4o**: 42.7% vs. 42.0% on Video-Holmes, demonstrating that fine-grained token-level optimization can compensate for the gap in model scale.
4. **Analysis of unselected tokens**: Filtered low-information tokens are predominantly function words (auxiliary verbs, pronouns, prepositions, etc.), validating that the attribution mechanism effectively removes redundant content.

## Limitations & Future Work

1. Counterfactual analysis requires additional forward passes (masked visual input + shuffled frame order), increasing training overhead.
2. Validation is limited to 7B-scale models; whether equivalent gains persist at larger scales remains unexplored.
3. The token selection ratio $r$ is a fixed hyperparameter that cannot adaptively adjust based on sample difficulty.
4. Frame counts are limited to 16–64 frames; the method's capability on ultra-long videos has not been validated.

## Rating ⭐⭐⭐⭐⭐

The paper presents an elegant method design, rigorous experimental analysis, and substantial performance gains. The transition from coarse-grained sequence-level rewards to fine-grained modality-aware token-level updates represents a significant advancement in RL training for video reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](../../CVPR2026/video_understanding/videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](../../ICCV2025/video_understanding/dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)

</div>

<!-- RELATED:END -->
