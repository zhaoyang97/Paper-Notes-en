---
title: >-
  [Paper Note] Identifying and Evaluating Inactive Heads in Pretrained LLMs
description: >-
  [ICLR 2026][LLM Pretraining][Inactive Attention Head] This paper systematically evaluates 12 scoring functions for identifying inactive attention heads in LLMs, finding that the attention head output norm-based scoring function (AHON LN) more consistently identifies inactive heads across model families than traditional attention weight metrics. On average across 14 models, over 12% of heads can be zeroed out while maintaining MMLU accuracy within 1%.
tags:
  - ICLR 2026
  - LLM Pretraining
  - Inactive Attention Head
  - Score Function
  - Attention Sink
  - Model Intervention
  - Head Output Norm
date: 2026-05-08
content_hash: 2ab1494c9e4b93c0
---

# Identifying and Evaluating Inactive Heads in Pretrained LLMs

**Conference**: ICLR 2026
**arXiv**: [2504.03889](https://arxiv.org/abs/2504.03889)
**Code**: [GitHub](https://github.com/psandovalsegura/inactive-heads)
**Area**: LLM Pretraining / Model Analysis
**Keywords**: Inactive Attention Head, Score Function, Attention Sink, Model Intervention, Head Output Norm

## TL;DR

This paper systematically evaluates 12 scoring functions for identifying inactive attention heads in LLMs, finding that the attention head output norm-based scoring function (AHON LN) more consistently identifies inactive heads across model families than traditional attention weight metrics. On average across 14 models, over 12% of heads can be zeroed out while maintaining MMLU accuracy within 1%.

## Background & Motivation

**State of the Field**: The attention mechanism is a core component of Transformer-based LLMs, yet prior work has identified attention heads exhibiting "attention sink" behavior—where the first token receives the majority of attention weight despite limited semantic importance. Guo et al. (2024a) introduced the concept of "dormant heads," identifying inactivity based on first-token attention weight.

**Limitations of Prior Work**: Relying solely on attention weights introduces blind spots: (1) a head may attend to multiple tokens with near-zero value vectors, yielding near-zero output without exhibiting high first-token weight; (2) a head may appear "dormant" by attention weight yet produce non-negligible output; (3) attention patterns vary substantially across model families (Llama, OLMo, Qwen), making first-token weight a non-universal indicator.

**Root Cause**: Inactivity in attention heads admits multiple definitions—attention concentrated on irrelevant tokens, near-zero value vectors, or near-zero head output—yet prior work addresses only the first, leading to a systematic underestimation of inactive head prevalence. AWFT (Average Weight of First Token) identifies only ~4.6% of inactive heads, missing approximately 7.6%.

**Paper Goals**: Provide a systematic answer to "how prevalent are inactive attention heads?" and identify the best model-agnostic detection method.

**Starting Point**: Rather than focusing on attention weights alone, this paper examines all three components of attention—weights, value vectors, and head outputs—and designs 12 simple scoring functions, validating true inactivity via threshold classification and model intervention experiments.

**Core Idea**: Inactive attention heads should be identified via head output norm rather than attention weight patterns, since small output is the true indicator of negligible contribution to the model.

## Method

### Overall Architecture

(1) Define 12 scoring functions spanning attention weights, value vectors, and head outputs (each with raw and layer-normalized variants); (2) Apply multiple thresholds per function to classify heads as "potentially inactive" or "active"; (3) Perform model interventions—dynamically zeroing out classified-inactive head outputs during forward passes—and evaluate accuracy change on MMLU. Evaluation is conducted on 14 pretrained models across three families: Llama-3.1/3.2, OLMo-2, and Qwen2.5.

### Key Designs

1. **12 Scoring Functions**

   - **Function**: Quantify the activity level of attention heads along three dimensions.
   - **Mechanism**: *Attention weight-based*—Avg Weight of First Token (AWFT): first-token average weight $\frac{1}{N}\sum_i \mathbf{A}_{i,0} > \tau$; Avg Entropy of Query Distributions (AEQD): average query distribution entropy $< \tau$ (low entropy = attention concentrated on few tokens). *Value vector-based*—First Token Value Vector Norm (FTVVN): first-token value norm $< \tau$; Avg Value Vector Norm (AVVN): average value norm $< \tau$. *Head output-based*—Last Token Head Output Norm (LTHON): last-token head output norm $< \tau$; Avg Head Output Norm (AHON): average head output norm $< \tau$. Each has a layer-normalized (LN) variant, dividing by the layer-average score across heads: $\frac{\text{AvgNorm}(\text{head}^i)}{\frac{1}{N_{\text{layer}}}\sum_j \text{AvgNorm}(\text{head}^j)}$.
   - **Design Motivation**: Different functions capture different types of inactivity. IoU analysis reveals a maximum IoU of only 0.58 and maximum Precision of 0.73, confirming that different functions identify distinct head sets. Layer normalization addresses the large cross-layer and cross-model variance in raw scores.

2. **Dynamic Model Intervention Validation**

   - **Function**: Verify whether heads identified by each scoring function are truly inactive.
   - **Mechanism**: For each forward pass, a boolean mask $\mathbf{B} \in \{0,1\}^{N_{\text{heads}} \times N_{\text{layers}}}$ is constructed based on per-input scores and thresholds; heads marked True have their outputs zeroed prior to concatenation and output projection. MMLU 5-shot accuracy is then evaluated. Thresholds are dynamically selected via CDF quantiles ($p = 0, 5, 10, \ldots, 30$) computed over MMLU inputs, controlling the maximum fraction of zeroed heads to 30%. Random zeroing serves as a baseline.
   - **Design Motivation**: Unlike permanent pruning, dynamic zeroing determines inactivity per input, providing a more precise measure of per-forward-pass computational waste. If identified heads are truly inactive, zeroing them should yield negligible accuracy degradation.

### Loss & Training

This is an analytical study involving no training. All evaluations are conducted on pretrained or fine-tuned models. Scores are computed via forward passes over 100 FineWeb-Edu training samples (randomly truncated to 10–3000 tokens) or MMLU evaluation samples. Standardized evaluation uses lm-evaluation-harness.

## Key Experimental Results

### Main Results

**Fraction of zeroable heads across 14 models** (Table 2; MMLU accuracy maintained within 1% of baseline):

| Model | AWFT Zeroable (%) | Best Function Zeroable (%) | Gain | Best Scoring Function |
|---|---|---|---|---|
| Llama-3.1-8B | 8.56 | **17.11** | +8.55 | AHON (LN) |
| Llama-3.1-8B-Inst | 1.01 | **10.97** | +9.95 | AHON (LN) |
| OLMo-2-7B | 0.42 | **8.34** | +7.93 | AHON (LN) |
| OLMo-2-7B-DPO | 2.14 | **20.60** | +18.46 | AHON (LN) |
| OLMo-2-7B-Inst | 1.46 | **19.54** | +18.07 | AHON (LN) |
| Qwen2.5-0.5B | 7.43 | **14.42** | +6.99 | LTHON (LN) |
| Qwen2.5-3B | 5.67 | **8.78** | +3.11 | AHON |
| Qwen2.5-7B | 1.25 | **7.54** | +6.29 | AHON (LN) |
| **Average** | **4.61** | **12.18** | **+7.56** | — |

AHON (LN) ranks 1st on 8/14 models and top-3 on 13/14 models.

### Ablation Study

**Cross-dataset threshold stability** (OLMo-2-7B-Inst; 15% of heads identified as inactive):

| Scoring Function | MMLU Threshold | PIQA Threshold | WinoGrande Threshold | Stability |
|---|---|---|---|---|
| AWFT | 0.077 | 0.265 | 0.109 | Unstable (3.4× variation) |
| AHON (LN) | 0.457 | 0.435 | 0.473 | Stable (<9% variation) |

### Key Findings

- **Output over weights**: Head output norm is the true indicator of inactivity—heads appearing "dormant" by attention weight do not necessarily produce near-zero output, and conversely, heads with seemingly active attention patterns may produce negligible output.
- **>12% safely removable**: Substantially higher than the ~4.6% estimated by AWFT; prior methods miss 7.6% of inactive heads.
- **Model-agnostic effectiveness**: AHON (LN) performs consistently across Llama, OLMo, and Qwen families, whereas AWFT nearly completely fails on OLMo (identifying only 0.42–2.14%).
- **Fine-tuning preserves attention behavior**: Score distributions after SFT, DPO, and RLHF are nearly identical to those of base models (minimal Wasserstein distance), indicating that attention head behavior is largely fixed after pretraining.
- **Scale effect with threshold**: Qwen2.5 models from 0.5B to 7B exhibit similar score distributions, but 14B shows a marked divergence, suggesting that large-scale models learn qualitatively different head specialization patterns.

## Highlights & Insights

- Methodological simplicity at its extreme—12 threshold-based functions suffice for effective inactive head identification, requiring no complex optimization or specialized training.
- A key insight: attention weights are a "misleading signal"—appearing dormant does not imply true inactivity; the actual output contribution of the head is what matters.
- A comprehensive experimental matrix of 14 models × 12 functions × multiple thresholds × 3 benchmarks lends high credibility to the conclusions.
- Dynamic zeroing (input-conditioned) measures computational redundancy more precisely than permanent pruning.
- AHON (LN) can be directly deployed in practical systems for KV cache compression and inference acceleration.

## Limitations & Future Work

- The paper focuses on understanding and identification without realizing actual inference speedups (zeroing serves solely as a validation mechanism).
- MLP modules are not analyzed—feed-forward layers may also exhibit per-token inactivity.
- No dedicated treatment of Grouped-Query Attention (GQA)—shared KV heads in modern models may affect the analysis.
- Evaluation relies primarily on MMLU; inactive head patterns may differ on generation tasks.
- Zeroing is not equivalent to parameter removal—realizing actual memory and compute savings requires additional engineering effort.

## Related Work & Insights

- **vs. Dormant Attention (Guo et al., 2024a)**: Attention weight-based identification is insufficient; AHON-based functions identify 2.6× more inactive heads than AWFT.
- **vs. Attention Sinks (Xiao et al., 2024)**: First-token concentration is one manifestation of inactivity but not the whole picture; this paper reveals a richer taxonomy of inactive head patterns.

## Rating

- Novelty: ⭐⭐⭐⭐ — Redefining "inactivity" from the perspective of output rather than weights is the central contribution; the systematic comparison of 12 functions is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 models × 3 families × 12 functions × multiple thresholds, covering pretraining, fine-tuning, and scale variation.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, rich figures, and rigorous experimental design.
- Value: ⭐⭐⭐⭐ — Provides a reliable methodological foundation for understanding attention redundancy in LLMs, with direct implications for inference optimization.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](../../AAAI2026/llm_pretraining/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)
- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](../../NeurIPS2025/llm_pretraining/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](../../CVPR2026/llm_pretraining/evidential_transformation_network_post_hoc_uncertainty_estimation.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)
- [\[NeurIPS 2025\] One Prompt Fits All: Universal Graph Adaptation for Pretrained Models](../../NeurIPS2025/llm_pretraining/one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)

<!-- RELATED:END -->
