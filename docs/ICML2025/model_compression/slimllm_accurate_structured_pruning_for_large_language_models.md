---
title: >-
  [Paper Note] SlimLLM: Accurate Structured Pruning for Large Language Models
description: >-
  [ICML2025][Model Compression][Structured Pruning] This paper proposes SlimLLM, a structured pruning method for LLMs. It evaluates channels via feature space importance (considering both weight direction and magnitude), assesses attention heads holistically using Pearson similarity, and couples this with a simple linear regression recovery strategy and layer-wise pruning ratio allocation. SlimLLM retains 98.7% of performance on LLaMA under 20% pruning.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Structured Pruning"
  - "LLM Compression"
  - "Channel Importance"
  - "Attention Head Pruning"
  - "Linear Regression Recovery"
date: 2026-05-08
content_hash: 42e9d6edbc93e3dc
---

# SlimLLM: Accurate Structured Pruning for Large Language Models

**Conference**: ICML2025  
**arXiv**: [2505.22689](https://arxiv.org/abs/2505.22689)  
**Area**: Model Compression  
**Keywords**: Structured Pruning, LLM Compression, Channel Importance, Attention Head Pruning, Linear Regression Recovery

## TL;DR
This paper proposes SlimLLM, a structured pruning method for LLMs. It evaluates channels via feature space importance (considering both weight direction and magnitude), assesses attention heads holistically using Pearson similarity, and couples this with a simple linear regression recovery strategy and layer-wise pruning ratio allocation. SlimLLM retains 98.7% of performance on LLaMA under 20% pruning. 

## Background & Motivation

### Advantages of Structured Pruning
Compared to unstructured pruning (such as SparseGPT/Wanda), structured pruning removes entire channels, heads, or layers, directly reducing computational overhead and maintaining compatibility with hardware accelerators.

### Limitations of Prior Work
- LLM-Pruner requires gradients, leading to high memory and computational costs.
- LoRAP evaluates importance element-wise, thereby ignoring weight vector directions.
- A lack of efficient performance recovery strategies.

## Method

### Channel Pruning: Feature Space Importance
Constructs the feature space of outputs, simultaneously considering both the direction and magnitude of weight vectors within the feature space.

### Attention Head Pruning: Pearson Similarity
Treats each head as an individual entity, evaluating its importance through the Pearson similarity between the original output and the output after removing that head. Additionally, a greedy search is utilized to find better combinations of heads.

### Linear Regression Recovery
After pruning, a simple linear regression is applied to the output matrices to rapidly recover performance without the need for complex fine-tuning.

### Layer-wise Pruning Ratio
Automatically determines the optimal pruning ratio for each layer.

## Key Experimental Results

### LLaMA-7B Common Sense Reasoning

| Method | Pruning Ratio | Performance Retention |
|------|--------|---------|
| LLM-Pruner | 20% | 96.8% |
| LoRAP | 20% | 97.2% |
| **SlimLLM** | **20%** | **98.7%** |

### Different Model Scales

| Model | SlimLLM 20% | SlimLLM 30% |
|------|-----------|-----------|
| LLaMA-7B | 98.7% | 95.2% |
| LLaMA-13B | 99.1% | 96.5% |

### Key Findings
1. Feature space evaluation captures channel importance more accurately than element-wise evaluation.
2. Pearson similarity assesses head contributions better than attention scores.
3. Linear regression recovery is extremely fast (on the order of seconds) and highly effective.
4. Layer-wise ratio allocation outperforms uniform pruning.
5. Achieves SOTA results across various LLaMA models.

## Highlights & Insights

1. "Considering direction and magnitude in the feature space" — a simple yet crucial improvement.
2. Evaluating heads holistically rather than element-wise aligns better with intuition.
3. Linear regression recovery is minimalist and efficient, eliminating the need for LoRA fine-tuning.
4. Retaining 98.7% of performance under 20% pruning is practically lossless.
5. The components of the method can be used independently (channels, heads, recovery, and ratios).

## Limitations & Future Work

1. Evaluation is restricted to the LLaMA series, with Mistral/Qwen yet to be tested.
2. The efficacy of linear regression recovery may be limited at higher pruning ratios (>40%).
3. Greedy search for attention heads can be time-consuming when there are many heads.
4. Joint application with quantization methods remains unexplored.
5. The impact on long-context scenarios has not been evaluated.

## Related Work & Insights

- Difference from LLM-Pruner: Gradient-free.
- Difference from Wanda: Structured rather than unstructured.
- Insight: Feature space evaluation can be extended to other scenarios requiring channel importance estimation.

## Rating
- Novelty: 4.0/5 — Incremental work, but each component makes a clear contribution
- Experimental Thoroughness: 4.5/5 — Multi-model and multi-benchmark evaluation
- Writing Quality: 4.0/5
- Value: 4.5/5 — Possesses direct practical value for LLM compression

## Supplementary

### Intuition of Feature Space Importance
Element-wise importance focuses solely on weight magnitude, yet two channels sharing the same direction are redundant. Feature space importance considers both direction and magnitude, allowing the identification of truly unique channels.

### Linear Regression Recovery vs. LoRA Fine-Tuning
LoRA requires multi-step training, whereas linear regression requires only a single-step closed-form solution—greatly accelerating the compression pipeline while maintaining accuracy.

### Automation of Layer-wise Pruning Ratios
Different layers exhibit varying sensitivity to pruning. SlimLLM automatically computes the optimal pruning ratio for each layer, avoiding the sub-optimality associated with uniform pruning.

### Effect of Greedy Head Search
Evaluating the importance of each head individually may be insufficient, as two individually unimportant heads might be crucial when combined. Greedy search identifies better combinations of heads within a reasonable computational budget.

### Complementarity with Unstructured Methods
SlimLLM's structured pruning can be stacked with unstructured methods like SparseGPT—first removing channels/heads, then sparsifying the remaining weights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](../../ACL2025/model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](../../ACL2026/model_compression/grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ICLR 2026\] RCPU: Rotation-Constrained Error Compensation for Structured Pruning of Large Language Models](../../ICLR2026/model_compression/rcpu_rotation-constrained_error_compensation_for_structured_pruning_of_large_lan.md)

</div>

<!-- RELATED:END -->
