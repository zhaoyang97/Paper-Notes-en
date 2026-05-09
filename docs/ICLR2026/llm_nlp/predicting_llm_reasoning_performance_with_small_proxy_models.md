---
title: >-
  [Paper Note] Predicting LLM Reasoning Performance with Small Proxy Models
description: >-
  [ICLR 2026][LLM/NLP][small proxy models] This paper proposes rBridge, a method that combines NLL evaluation on frontier-model reasoning traces with token-level task alignment weights, enabling models with ≤1B parameters to effectively predict the reasoning performance of 13B–32B models, reducing data ranking computation cost by over 100×.
tags:
  - ICLR 2026
  - LLM/NLP
  - small proxy models
  - reasoning performance prediction
  - pretraining data
  - scaling law
  - NLL
date: 2026-05-08
content_hash: 34ba543750e8b37e
---

# Predicting LLM Reasoning Performance with Small Proxy Models

**Conference**: ICLR 2026
**arXiv**: [2509.21013](https://arxiv.org/abs/2509.21013)
**Code**: Not released
**Area**: LLM/NLP
**Keywords**: small proxy models, reasoning performance prediction, pretraining data, scaling law, NLL

## TL;DR

This paper proposes rBridge, a method that combines NLL evaluation on frontier-model reasoning traces with token-level task alignment weights, enabling models with ≤1B parameters to effectively predict the reasoning performance of 13B–32B models, reducing data ranking computation cost by over 100×.

## Background & Motivation

### The Need for Proxy Models in Pretraining

Pretraining large language models is extremely costly (training a 7B model on 500B tokens exceeds \$50,000), making small proxy models a key research direction for dataset optimization and large-model performance prediction. Existing approaches (scaling laws, data ranking invariance) work well for non-reasoning tasks, but the **emergent nature of reasoning ability** poses a fundamental challenge for small proxy models.

### Core Challenge: Emergence of Reasoning

Reasoning ability exhibits clear emergent behavior, appearing reliably only in models with 7B+ parameters:

- 1B models produce noisy results on MATH500 with incorrect slope directions (very low $R^2$)
- In contrast, non-reasoning tasks such as TriviaQA and HellaSwag show smooth improvement at small scales
- This forces practitioners to use 7–15B "proxy models," which is prohibitively expensive

### Problem Formulation

Let $\pi^{\text{p}}$ and $\pi^{\text{t}}$ denote the small proxy and large target models, respectively. The goal is to design a proxy evaluation metric $\text{metric}^{\text{p}}$ such that:

$$\text{metric}^{\text{p}} := \max_{\text{metric}} \text{corr}(\text{metric}(\pi^{\text{p}}), \text{metric}^{\text{t}}(\pi^{\text{t}}))$$

That is, improvements in the small-model metric should reliably predict improvements in the large-model metric.

## Method

### Overall Architecture

The core insight of rBridge is that existing methods fail along two axes: **evaluation objective alignment** and **target task alignment**.

### Preliminary Analysis: Limitations of Existing Methods

**(1) Misaligned Evaluation Objectives**

- **Acc./Pass@K mismatch with pretraining objectives**: Small pretrained models lack strong generalization; Accuracy is extremely noisy for 1B models.
- **Distribution alignment issue with NLL**: Not all NLL signals are equivalent. When the gold label $Y^*$ is out-of-distribution (OOD), the NLL signal becomes noisy. In ScalingBench, $Y^*$ includes formatting text such as "\\n" and "Final Answer:", which is OOD for small models.

**(2) Misaligned Target Task**

Standard NLL treats all tokens equally, failing to distinguish task-critical tokens (e.g., "sum modulo 9") from formatting tokens (e.g., newlines, numbering).

### Key Designs: rBridge NLL

**Step 1: Use frontier-model reasoning traces $R^\phi$ as gold labels**

Only the reasoning trace from the frontier model $\pi^\phi$ is used (excluding final answer formatting), because:
- $R^\phi$ is closer to the pretraining distribution (in-distribution), reducing average NLL by 74.7%
- $R^\phi$ represents the reasoning process toward the correct answer, making it naturally task-aligned

**Step 2: Token-level task alignment weights**

$$\text{rBridge NLL}(\text{token}_i) := \underbrace{-\text{log}(p^{\text{p}}(\text{token}_i))}_{\text{standard NLL}} \cdot \underbrace{\frac{1}{|\text{token}_i|} \sum_{\text{letter} \in \text{token}_i} p^\phi(\text{letter})}_{\text{automatic tokenizer-agnostic task alignment weight}}$$

Each token's NLL is weighted by the frontier model's confidence for that token. To handle different tokenizers, weights are computed at the character level and averaged within each token. MinMax normalization is applied to the weight factors to amplify the effect.

### Loss & Training

rBridge does not train a new model; it is purely an evaluation metric. The computation pipeline is:

1. Generate reasoning traces $R^\phi$ using the frontier model
2. Compute per-token NLL using the small proxy model
3. Compute task alignment weights using frontier-model token probabilities
4. Produce the rBridge score via weighted summation

## Key Experimental Results

### Main Results 1: Dataset Ranking (<100M → 1.2B)

Under the DataDecide protocol, proxy models are used to rank 25 pretraining datasets:

| Method | Best DAcc. | Compute Savings |
|--------|-----------|-----------------|
| CF Accuracy (largest proxy) | ~baseline | 1× |
| Norm Correct Prob | ~50% (random level) | — |
| **rBridge (3.7M model)** | 27% above baseline | **100.2×–733.4×** |

rBridge achieves ranking accuracy far exceeding the baseline even at the smallest proxy scale (3.7M parameters, trained on 87.3M tokens).

### Main Results 2: Proxy–Target Relationship (1B → 13B/32B)

5-fold cross-validation results across 6 reasoning benchmarks:

| Method | 1B→13B Avg Train $R^2$ ↑ | 1B→13B Avg Test MAE ↓ | 1B→32B Avg $R^2$ ↑ |
|--------|--------------------------|----------------------|---------------------|
| Acc./p@1 | 0.304 | 3.709 | 0.312 |
| iSFT | 0.290 | 5.123 | 0.349 |
| TED | 0.375 | 3.377 | 0.352 |
| MPCA | 0.194 | 302.642 | 0.205 |
| NLL | 0.485 | 5.173 | 0.488 |
| $R^\phi$ | 0.867 | 1.455 | 0.820 |
| **rBridge** | **0.874** | **1.384** | **0.826** |

rBridge achieves best performance on 10 out of 12 metrics, improving average $R^2$ from the baseline of 0.304 to 0.874.

### Ablation Study

| Setting | 1B→13B $R^2$ | 1B→13B+SFT $R^2$ | 1B→32B $R^2$ |
|---------|-------------|-------------------|-------------|
| Standard NLL | 0.485 | 0.413 | 0.488 |
| NLL on $R^\phi$ | 0.867 | 0.820 | 0.820 |
| **rBridge (+ weights)** | **0.874** | **0.846** | **0.826** |

Each component (reasoning traces, alignment weights, normalization) contributes consistent improvements.

### Zero-Shot Transfer (1B → 7B, Cross-Dataset)

A proxy-target function fitted on one dataset can be zero-shot transferred to another dataset:

| Method | Dataset Ranking Accuracy | Avg. MAE |
|--------|--------------------------|----------|
| $R^\phi$ | 4/5 | 3.425 |
| **rBridge** | **5/5** | **2.490** |

### Key Findings

1. **rBridge outperforms proxy models 7–13× larger**: A 1B rBridge surpasses direct Acc. evaluation with 7B/13B models.
2. **Substantial compute savings**: 100–733× FLOP reduction in data ranking tasks.
3. **Effective zero-shot cross-dataset transfer**: A single fitted function can be reused across datasets.
4. **Discrete metrics (Acc./p@k, iSFT) perform worst**: Validating the importance of evaluation objective alignment.

## Highlights & Insights

1. **Deep methodological insight**: The problem is not that small models are "too small," but that the evaluation approach is misaligned with both the pretraining objective and the target task.
2. **Elegant solution**: No new model training is required; changing the evaluation metric alone yields order-of-magnitude improvements.
3. **Clever use of frontier-model probabilities**: Frontier-model token probabilities serve as an automatic measure of task-token importance.
4. **Handling tokenizer mismatch**: Character-level probability averaging is a concise and practical design choice.
5. **Clear theoretical motivation**: Each design decision is grounded in pretraining objective alignment and in-distribution/out-of-distribution analysis.

## Limitations & Future Work

1. Frontier-model reasoning trace generation is required; though low-cost (<\$10 per benchmark), it introduces a dependency on frontier models.
2. The largest target model in experiments is 32B; applicability to larger models (70B+) remains unverified.
3. Zero-shot transfer is validated across only two datasets; generalizability requires further confirmation.
4. The method focuses on reasoning tasks; its applicability to non-reasoning tasks (knowledge, comprehension) is unexplored.

## Related Work & Insights

- **Scaling Laws (Kaplan et al. 2020)**: rBridge can be viewed as an improved scaling law for the reasoning domain.
- **ScalingBench**: rBridge improves upon ScalingBench by excluding answer formatting and retaining only reasoning traces as gold labels.
- **DataDecide (Magnusson et al. 2025)**: rBridge validates its data ranking capability under the DataDecide protocol.
- **Insight**: Evaluation metric design matters more than model scale—aligning evaluation objectives and tasks is the key to leveraging small models effectively.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic treatment of predicting large-model reasoning performance with small models
- **Theoretical Depth**: ⭐⭐⭐⭐ — Complete in-distribution/out-of-distribution analysis and evaluation alignment framework
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three categories of experiments (ranking/relationship/transfer) covering 6 benchmarks with 6+ baselines
- **Value**: ⭐⭐⭐⭐⭐ — 100× compute savings carry substantial practical significance
- **Overall**: ⭐⭐⭐⭐☆ — Important problem, elegant method, comprehensive experiments; directly applicable to pretraining data optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](../../ACL2026/llm_nlp/don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)

</div>

<!-- RELATED:END -->
