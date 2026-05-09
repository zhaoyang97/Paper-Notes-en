---
title: >-
  [Paper Note] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision
description: >-
  [CVPR 2026][Reinforcement Learning][Anomaly Detection and Localization] This paper proposes two modules, ReAL and CGRO, which extract anomaly-relevant tokens from the autoregressive reasoning process of an MLLM and aggregate their visual attention maps to generate pixel-level anomaly maps. A consistency-guided reinforcement learning scheme then aligns reasoning tokens with visual evidence, enabling end-to-end anomaly detection, localization, and interpretable reasoning under image-level supervision only.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Anomaly Detection and Localization
  - Reasoning-Driven
  - Image-Level Supervision
  - MLLM Attention
date: 2026-05-08
content_hash: 5453a137065d95d1
---

# Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision

**Conference**: CVPR 2026
**arXiv**: [2603.27179](https://arxiv.org/abs/2603.27179)
**Code**: [GitHub](https://github.com/YizhouJin313/ReADL)
**Area**: Reinforcement Learning
**Keywords**: Anomaly Detection and Localization, Reasoning-Driven, Image-Level Supervision, MLLM Attention, Reinforcement Learning

## TL;DR

This paper proposes two modules, ReAL and CGRO, which extract anomaly-relevant tokens from the autoregressive reasoning process of an MLLM and aggregate their visual attention maps to generate pixel-level anomaly maps. A consistency-guided reinforcement learning scheme then aligns reasoning tokens with visual evidence, enabling end-to-end anomaly detection, localization, and interpretable reasoning under image-level supervision only.

## Background & Motivation

Industrial anomaly detection faces multiple challenges:
- **Limitations of traditional methods**: Training product-specific models requires large collections of normal samples, incurring high deployment costs and poor generalization across product lines.
- **Existing MLLM-based approaches**: Most methods support only image-level detection and textual reasoning, while pixel-level localization still relies on external visual modules (e.g., AnomalyGPT uses pretrained visual experts; EIAD uses SAM), leading to error propagation, reasoning–localization misalignment, and increased deployment complexity.
- **End-to-end approaches (e.g., OmniAD)**: These depend on dense pixel-level annotations and high-quality reasoning annotations, which are costly to obtain and introduce domain bias.

Core observation (Fig. 1): During MLLM text generation, **only a small subset of tokens attend to genuine anomalous regions**, and these tokens tend to correspond to anomaly-relevant semantics (e.g., "scratch," "mark"). The attention of most reasoning tokens is diffuse or focused on irrelevant regions, diluting localization precision.

## Method

### Overall Architecture

Given an image $\mathbf{X}_v$ and a fixed text prompt ("Are there any defects or anomalies in the image?"), the MLLM generates an output sequence containing a reasoning chain and a final answer. The framework comprises two core modules:
1. **ReAL (Reasoning-Driven Anomaly Localization)**: Selects anomaly-relevant tokens from the reasoning sequence and aggregates their visual attention maps to produce a pixel-level anomaly map.
2. **CGRO (Consistency-Guided Reasoning Optimization)**: Drives reasoning–localization consistency via reinforcement learning, aligning reasoning tokens with visual attention.

### Key Designs

1. **Anomaly-Relevant Reasoning Token Identification (core of ReAL)**: Each reasoning token is evaluated along two complementary dimensions:
    - **Cross-modal semantic relevance $S_T^r$**: The sum of attention weights from the reasoning token to anomaly-related words ("defect"/"anomaly"/"abnormal") in the input text, measuring semantic association with anomaly concepts.
    - **Intra-modal attention concentration $S_I^r$**: The visual attention map is binarized, connected components are extracted, and spatial entropy is computed—low entropy indicates attention focused on a specific region (potentially anomalous), while high entropy indicates diffuse attention.

   After dual-threshold filtering ($\hat{S}_T^r > \tau_t$ and $\hat{S}_I^r > \tau_i$), the visual attention maps $\mathbf{A}_{r,I}$ of retained tokens are aggregated with composite weights $w_r = \alpha\hat{S}_T^r + \beta\hat{S}_I^r$, yielding the reasoning-driven anomaly map $\mathbf{A}_{\text{RDAM}}$.

2. **Consistency-Guided Reasoning Optimization (CGRO)**: Addresses inconsistent reasoning under limited supervision (e.g., the model answers "anomaly present" while the reasoning chain describes the image as normal). A class-conditional consistency reward $R_{\text{cons}}$ is designed:
    - For anomalous images ($y=1$): Encourages high spatial consistency (Jaccard Index $\mathcal{J} > \delta_1$) among the attention regions of top-$t$ reasoning tokens.
    - For normal images ($y=0$): Encourages low spatial consistency ($\mathcal{J} < \delta_2$), suppressing spurious focus on benign regions.

   The total reward $\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{fmt}} + \mathcal{R}_{\text{acc}} + \mathcal{R}_{\text{cons}}$ is optimized via the GRPO framework.

3. **End-to-End without External Modules**: The entire system relies on a single MLLM with no dependency on external segmentation (SAM) or detection modules, achieving true end-to-end anomaly detection, localization, and interpretable reasoning. Training requires only image-level labels (normal/anomalous).

### Loss & Training

- Built on Qwen2.5-VL-7B with LoRA adapters applied to language and cross-modal layers; the visual encoder is frozen.
- Training data: 4K industrial images drawn from VisA, GoodsAD, Vision, and PR-REAL, with image-level annotations only.
- Batch size of 16 samples; 8 candidate outputs sampled per input (GRPO).
- Images uniformly resized to 420×420.
- Zero-shot evaluation (no domain overlap between training and test sets).

## Key Experimental Results

### Main Results

Average across four benchmarks (MVTec-AD, WFDD, SDD, DTD), image-level AUROC/ACC:

| Method | Parameters | Supervision | Image-Level AVG (AUROC, ACC) | Pixel-Level AVG (AUROC, ACC) | Reasoning (ROUGE-L, SBERT) |
|------|--------|----------|----------------------|---------------------|---------------------|
| GPT-4.1 | — | — | 87.2, 88.4 | N/A | 20.8, 69.9 |
| Qwen2.5-VL+CGRO* | 7B | I | **83.9, 86.9** | **80.7, 97.1** | **27.1, 74.7** |
| Qwen2.5-VL+R1* | 7B | I | 80.0, 82.0 | 78.5, 96.7 | 26.3, 73.8 |
| AnomalyGPT | 7B | T+I+P | 71.1, 53.9 | 77.8, 98.4 | 11.9, 36.7 |
| Triad | 7B | T+I | 85.5, 83.8 | N/A | 8.6, 35.9 |

Highlight: Using only image-level supervision, the proposed method achieves localization performance comparable to AnomalyGPT, which relies on dense pixel-level annotations.

### Ablation Study

ReAL + CGRO ablation (Qwen2.5-VL-7B, average over four datasets):

| Configuration | Image-Level AUROC | Pixel-Level AUROC | Pixel-Level ACC |
|------|-------------|-------------|-----------|
| Vanilla | 63.4 | 64.7 | 73.0 |
| Vanilla + ReAL | 63.4 | 61.7 | 85.6 |
| Vanilla + CGRO | 83.9 | 72.7 | 92.6 |
| **Full (ReAL+CGRO)** | **83.9** | **80.7** | **97.1** |

Token selection strategy ablation (pixel-level):
- $S_I$ only: AUROC 74.1
- $S_T$ only: AUROC 76.7
- $S_T + S_I$ (full): AUROC **80.7**

### Key Findings

- ReAL and CGRO are complementary: CGRO improves image-level detection (+20.5 AUROC), while ReAL improves pixel-level localization precision (+8.0 AUROC).
- The consistency reward eliminates reasoning–answer contradictions: without CGRO, the model frequently produces inconsistent outputs ("anomaly detected" but reasoning describes the image as normal) with diffuse attention.
- CGRO provides consistent gains across model scales from 3B to 7B parameters (image-level +15–20 AUROC).
- Improvements in reasoning quality and localization precision are mutually reinforcing.

## Highlights & Insights

- **Deep core insight**: The work reveals that anomaly-aware attention patterns naturally emerge during MLLM reasoning; the key is to correctly identify and leverage them rather than introduce external modules.
- **High annotation efficiency**: Image-level labels—the least costly form of annotation—are sufficient to match methods trained with dense pixel-level supervision.
- **Unified three-dimensional capability**: A single model simultaneously performs detection, localization, and interpretable reasoning without external modules.
- **Elegant consistency reward design**: Class-conditional constraints based on the Jaccard Index align reasoning quality with spatial focus.

## Limitations & Future Work

- Localization precision leaves room for improvement (pixel-level AUPR of 13.3%, well below dedicated segmentation methods).
- Reasoning token selection depends on threshold hyperparameters $\tau_t, \tau_i$, which may require tuning for different product categories.
- Training images are sourced from other public AD datasets, potentially introducing domain bias.
- GRPO training is computationally expensive (8 candidate outputs per input).
- While the attention mechanism provides strong interpretability, performance on complex multi-defect scenarios remains unexplored.

## Related Work & Insights

- **Comparison with LISA**: LISA employs a [SEG] token combined with SAM for reasoning-based segmentation; the proposed method entirely removes the external segmentation module.
- **GRPO/R1 paradigm**: Follows the reinforcement-learning-based reasoning optimization of DeepSeek-R1, with the novel introduction of a consistency reward.
- **Comparison with OmniAD**: OmniAD requires dense annotations for end-to-end training, whereas the proposed method needs only image-level labels.
- **Broader implications**: The attention aggregation strategy is generalizable to other tasks requiring MLLM-based spatial localization, such as referring segmentation and visual grounding.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of activating the intrinsic reasoning potential of MLLMs for pixel-level localization is highly innovative, and the consistency reward is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, comparisons against diverse MLLMs (including the GPT-4 family), and detailed ablations provide strong empirical support.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ — Substantially reduces annotation costs for industrial anomaly detection and opens new avenues for deploying MLLMs in industrial quality inspection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](../../ACL2026/reinforcement_learning/attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)
- [\[AAAI 2026\] TextShield-R1: Reinforced Reasoning for Tampered Text Detection](../../AAAI2026/reinforcement_learning/textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)
- [\[ICLR 2026\] PreferThinker: Reasoning-based Personalized Image Preference Assessment](../../ICLR2026/reinforcement_learning/preferthinker_reasoning-based_personalized_image_preference_assessment.md)
- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)

</div>

<!-- RELATED:END -->
