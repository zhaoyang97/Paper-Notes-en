---
title: >-
  [Paper Note] AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference
description: >-
  [ICCV 2025][Multimodal VLM][KV cache compression] This paper proposes AirCache, which achieves model performance retention with only 10% of the visual KV cache—reducing decoding latency by 29%–66%—through an elite observ…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "KV cache compression"
  - "vision-language model"
  - "elite observation window"
  - "adaptive budget allocation"
  - "attention analysis"
date: 2026-05-08
content_hash: df9a3c99cb975d0d
---

# AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference

**Conference**: ICCV 2025
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: KV cache compression, vision-language model, elite observation window, adaptive budget allocation, attention analysis

## TL;DR

This paper proposes AirCache, which achieves model performance retention with only 10% of the visual KV cache—reducing decoding latency by 29%–66%—through an elite observation window (leveraging text self-attention to select critical text tokens for evaluating visual token importance) and adaptive inter-layer budget allocation (based on the intensity and skewness of importance score distributions).

## Background & Motivation

Large Vision-Language Models (LVLMs) generate a substantial number of visual tokens when processing high-resolution images, multi-image inputs, or video, and the linear growth of the KV cache under long-generation demands imposes significant memory and bandwidth pressure. Existing approaches fall into two categories: (1) token pruning—reducing tokens during the prefill stage, but aggressive removal causes severe visual information loss; and (2) KV cache compression—pruning cached data during the decoding stage with less impact on performance. However, existing KV cache compression methods suffer from two issues: inappropriate observation window selection (using all or consecutive local text tokens leads to inconsistent evaluation) and suboptimal uniform inter-layer budget allocation.

## Method

### Overall Architecture

AirCache intervenes in the KV cache storage process after prefill completion and comprises two components: (1) visual token importance scoring—selecting critical text tokens via an elite observation window and using cross-modal attention voting to assess visual token importance; and (2) inter-layer KV cache budget allocation—dynamically assigning compression budgets to different layers based on the intensity and skewness of the importance score distribution.

### Key Designs

1. **Elite Observation Window**: Text self-attention scores are used with the last text token as a reference to filter text tokens whose attention scores exceed a threshold $\alpha \cdot \max$. These selected critical text tokens serve as the observation window, and their cross-modal attention with visual tokens is used to evaluate visual token importance. Compared to using all text tokens or a consecutive local window, tokens within the elite window exhibit more consistent evaluations of the same visual token, reducing noise in voting-based ranking.

2. **Adaptive Inter-layer Budget Allocation**: The compression budget for each layer is quantified along two dimensions: (a) *distribution intensity*—the total attention allocated to all visual tokens in a given layer, reflecting how much that layer attends to visual information; and (b) *distribution skewness*—the head-heavy effect of the importance score distribution, where a small number of visual tokens receive disproportionately high scores while the majority remain marginal. Greater skewness indicates higher compression potential. Combining both dimensions yields a superior inter-layer budget allocation compared to uniform assignment.

3. **Head-heavy Effect in Visual Token Importance**: Experiments reveal that visual token importance follows an extremely heavy-headed distribution—a small fraction of tokens are critical while the vast majority are not. Retaining only 10% of the visual KV cache results in less than 1% performance degradation. Visual information intensity also varies substantially across layers, with early and late layers exhibiting different emphases.

### Loss & Training

AirCache is a training-free method applied directly during KV cache management at inference time. It is compatible with mainstream LVLM architectures such as LLaVA-OV and InternVL.

## Key Experimental Results

### Main Results

| Method | Visual KV Retention | Average Performance | Decoding Latency Reduction |
|--------|--------------------|--------------------|---------------------------|
| Full Cache | 100% | Baseline | — |
| H2O | 10% | Significant drop | — |
| SnapKV | 10% | Moderate drop | — |
| **AirCache** | **10%** | **<1% drop** | **29%–66%** |

Across multiple LVLMs and benchmarks, AirCache achieves the closest performance to the full cache at a 10% retention rate.

### Ablation Study

- Elite window vs. full text window vs. local window: the elite window yields the highest consistency.
- Adaptive allocation vs. uniform allocation: adaptive allocation is significantly superior.
- Retention rates from 10% to 50%: AirCache demonstrates a more pronounced advantage at lower retention rates.
- Visual attention intensity variation across layers: validates the necessity of non-uniform budget allocation.

### Key Findings

- Visual token importance exhibits a strong head-heavy effect; 90% of the KV cache can be safely removed.
- Different text tokens evaluate the same visual tokens very differently, necessitating careful selection of the observation window.
- Inter-layer budgets should not be allocated uniformly—certain layers are more sensitive to visual information.

## Highlights & Insights

- The design intuition behind the elite observation window is clear: not all text tokens are suitable for evaluating visual importance.
- The two-dimensional budget allocation based on distribution intensity and skewness is more theoretically grounded than heuristic approaches.
- Performance degradation of less than 1% at a 10% retention rate demonstrates that visual KV cache redundancy is substantial.
- The method is orthogonal to token pruning and can be combined with it for further acceleration.

## Limitations & Future Work

- The threshold $\alpha$ for the elite window requires hyperparameter tuning.
- Performance may degrade sharply at extremely low retention rates (<5%).
- Optimization is limited to the decoding stage; prefill-stage overhead is not reduced.
- The method has not been thoroughly validated in long-sequence scenarios such as video understanding.

## Related Work & Insights

- H2O and SnapKV are KV cache compression methods from the LLM domain.
- PyramidKV proposes hierarchical budget allocation.
- VL-Cache exploits visual token sparsity and is the most directly comparable LVLM KV cache compression baseline.
- The method is extensible to efficient inference in long-video and multi-image scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The elite window and two-dimensional budget allocation are novel designs.
- Technical Depth: ⭐⭐⭐⭐ — In-depth analysis of attention distributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comparisons across multiple models, benchmarks, and retention rates.
- Writing Quality: ⭐⭐⭐⭐ — Motivation and visualization analyses are well presented.
- Value: ⭐⭐⭐⭐⭐ — Training-free, broadly compatible, and achieves significant speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models](the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)
- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out_of_distribution_detection.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
