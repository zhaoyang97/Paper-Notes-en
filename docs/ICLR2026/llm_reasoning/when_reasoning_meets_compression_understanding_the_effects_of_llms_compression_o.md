---
title: >-
  [Paper Note] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][Model Compression] This paper presents a systematic benchmark and mechanistic analysis of the effects of compression (quantization/distillation/pruning) on large reasoning models (LRMs), yielding three core findings: parameter count affects knowledge memorization more than reasoning ability; the last-layer MLP `up_proj` of distilled models is the most critical weight; protecting only 2% of over-compressed weights improves average accuracy by 6.57%.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Model Compression
  - Large Reasoning Models
  - Quantization
  - Distillation
  - Pruning
  - Mechanistic Interpretability
date: 2026-05-08
content_hash: 19af0bbea21134e7
---

# When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2504.02010](https://arxiv.org/abs/2504.02010)
**Code**: [psunlpgroup/Compression-Effects](https://github.com/psunlpgroup/Compression-Effects)
**Area**: LLM Reasoning
**Keywords**: Model Compression, Large Reasoning Models, Quantization, Distillation, Pruning, Mechanistic Interpretability

## TL;DR

This paper presents a systematic benchmark and mechanistic analysis of the effects of compression (quantization/distillation/pruning) on large reasoning models (LRMs), yielding three core findings: parameter count affects knowledge memorization more than reasoning ability; the last-layer MLP `up_proj` of distilled models is the most critical weight; protecting only 2% of over-compressed weights improves average accuracy by 6.57%.

## Background & Motivation

**Deployment bottleneck of LRMs**: Large reasoning models such as DeepSeek-R1 achieve strong performance on complex reasoning tasks, but their 671B parameter scale makes deployment prohibitively expensive. Compression is thus a key enabler of AI democratization.

**Three gaps in existing research**:
   - Lack of **comprehensive comparison** of quantization, distillation, and pruning on reasoning-intensive datasets
   - Insufficient analysis of how compression affects **knowledge memorization** and **reasoning ability**
   - Absence of **interpretability analysis** of compression effects — which weights matter most for reasoning? This is a fundamental question in compression research

**Core research question**: How is the reasoning ability of large reasoning models degraded during compression?

## Method

### Overall Architecture

The paper adopts a **dual-perspective** approach: performance benchmarking + mechanistic interpretation.

- **Benchmarking**: Comprehensive evaluation of DeepSeek-R1 and its compressed variants, covering dynamic quantization (Unsloth), AWQ, GPTQ, GPTAQ, ANY4/3, SFT distillation, SparseGPT, and AlphaPruning.
- **Mechanistic Interpretation**: The paper adapts *difference of means* and *attribution patching* techniques to quantify the causal contribution of each linear module to four reasoning behaviors (backtracking, uncertainty estimation, example testing, and knowledge addition).

### Key Designs

**Weight importance quantification**: For each linear module (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) in every layer, importance scores for specific reasoning behaviors are computed in two steps:

1. **Difference of Means**: A steering vector $u$ is extracted for each linear module at tokens corresponding to a specific reasoning behavior, representing that behavior's numerical signature in activation space.
2. **Attribution Patching**: The dot product of the steering vector and the module's activation gradient yields an importance score $I_{m\ell}^c$; a higher value indicates a stronger causal relationship between the module and the reasoning behavior.

**Compression effect decoding**: Relative importance (RI), a normalized form of $I$, is computed to track importance shifts before and after compression. Ideally, shifts should be minimal — larger shifts indicate more severe damage to that module from compression.

### Evaluation Design

- **4 reasoning datasets**: AIME 2024 (math), FOLIO (logic), Temporal Sequences (temporal), MuSiQue (multi-hop + knowledge, closed-book)
- **120 instances** for interpretability analysis (30 per dataset)
- Each model is run 3 times and averaged (except R1 and dynamic quantization variants)

## Key Experimental Results

### Main Results

Performance of DeepSeek-R1 and its compressed variants across 4 reasoning benchmarks:

| Model / Compression | AIME 2024 | FOLIO | Temporal | Avg. | MuSiQue EM |
|---|---|---|---|---|---|
| R1 (671B, original) | 73.3 | 76.4 | 99.6 | 83.1 | 17.0 |
| R1 (2.51-bit) | 76.7 | 77.8 | 100.0 | **84.8** | 17.0 |
| R1 (1.58-bit) | 66.7 | 75.4 | 94.0 | 78.7 | 14.0 |
| R1-Distill-Llama-70B | 65.6 | 79.8 | 99.9 | 81.8 | 13.3 |
| R1-Distill-Llama-70B + 50% SparseGPT | 23.3 | 71.6 | 97.6 | 64.2 | 6.7 |
| R1-Distill-Llama-8B | 42.2 | 71.9 | 81.5 | 65.2 | 0.0 |
| R1-Distill-Llama-8B + 4-bit AWQ | 47.8 | 68.0 | 84.0 | 66.6 | 0.3 |
| R1-Distill-Llama-8B + 3-bit GPTQ | 11.1 | 65.0 | 67.3 | 47.8 | 0.0 |

### Ablation Study

Selective quantization to validate weight importance (R1-Distill-Llama-8B; only one module quantized to 3-bit):

| Quantized Module | Importance Rank | AIME 2024 | FOLIO | Temporal | Avg. |
|---|---|---|---|---|---|
| No quantization (reference) | — | 42.2 | 71.9 | 81.5 | 65.2 |
| **32_up (last-layer up_proj)** | **1st** | **20.0** | **63.1** | **63.6** | **48.9** |
| 32_gate | 2nd (same layer) | 33.3 | 62.1 | 67.2 | 54.2 |
| 32_v | Last (same layer) | 43.3 | 68.0 | 79.6 | 63.6 |
| 31_up | 2nd (same row) | 33.3 | 70.0 | 64.4 | 55.9 |

**Selective protection experiment**: 3-bit AWQ + protecting the last-layer MLP (only 2% of weights retained at 16-bit):

| Configuration | AIME 2024 | FOLIO | Temporal | Avg. | Gain |
|---|---|---|---|---|---|
| 3-bit AWQ | 10.0 | 59.6 | 68.4 | 46.0 | — |
| **3-bit AWQ + protect last-layer MLP** | **16.7** | **67.0** | **74.0** | **52.57** | **+6.57** |

### Key Findings

1. **2.51-bit dynamic quantization is the optimal compression strategy**: Among all compression methods, it most closely matches the original R1, even surpassing it on some metrics.
2. **4-bit quantization is generally safe; 3-bit triggers collapse**: All 4-bit methods (AWQ/GPTQ/GPTAQ/ANY4) perform comparably to the uncompressed model, whereas 3-bit leads to significant degradation.
3. **Pruning inflicts the greatest damage on knowledge memorization**: 50% SparseGPT causes AIME accuracy to plummet from 65.6 to 23.3; the collapse threshold on MuSiQue (30–40% sparsity) is reached earlier than on reasoning tasks.
4. **The last-layer MLP `up_proj` is the most critical module**: Quantizing this single matrix (0.7% of weights) alone reduces average accuracy by 16.3%.
5. **Existing quantization methods over-compress the last layer and `gate_proj`**: Protecting 2% of weights yields a 6.57% improvement, surpassing SOTA by up to 23.17% in some settings.

## Highlights & Insights

- **Mechanistic interpretability applied to compression effects for the first time**: The paper goes beyond measuring performance drops to reveal *why* accuracy degrades — identifying which specific weights are damaged.
- **"Parameter count affects knowledge more than reasoning"** carries important practical implications: knowledge-intensive tasks should favor quantization over pruning or distillation.
- **Protecting only 2% of weights yields a 6.57% improvement**: This simple mixed-precision remedy validates the analysis and points toward future adaptive quantization strategies.

## Limitations & Future Work

1. **Limited scale of interpretability analysis**: Only 120 instances are used for attribution patching, raising concerns about statistical significance.
2. **Only linear layers are analyzed**: Other components such as LayerNorm and Embedding layers are not examined.
3. **Protection strategy is simplistic**: Critical weights are naively retained at 16-bit; more sophisticated mixed-precision or adaptive quantization schemes are not explored.
4. **Distillation considers only SFT**: RL-based or on-policy distillation methods are not addressed.

## Related Work & Insights

- The comparison with **AlphaPruning** is noteworthy: its pruning importance heatmap resembles that of quantization, suggesting the two share underlying bottlenecks.
- The paper finds that **Qwen exhibits stronger reasoning ability than Llama, while Llama retains knowledge better** — model architecture selection should be informed by task type.
- Implication: Future model compression should adopt **importance-aware adaptive precision allocation** rather than uniform quantization.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic combination of benchmarking and mechanistic interpretation for compression effects on LRMs
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 compression families × 4 model families × 4 datasets, with validation experiments
- Writing Quality: ⭐⭐⭐⭐ Clear structure, findings are explicit and actionable
- Value: ⭐⭐⭐⭐⭐ Three key findings provide direct guidance for LRM compression research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)

</div>

<!-- RELATED:END -->
