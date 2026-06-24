---
title: >-
  [Paper Note] Instruction-Following Pruning for Large Language Models
description: >-
  [ICML2025][Model Compression][Structured Pruning] IFPruning is proposed: a small sparsity predictor dynamically generates pruning masks based on user instructions, trimming the intermediate dimensions of FFNs on demand. This enables a 9B model activating only 3B parameters to outperform dense models of the same scale by 5-8 percentage points in coding/math, while maintaining inference latency on par with a 3B dense model.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Structured Pruning"
  - "Dynamic Pruning"
  - "Instruction-Aware"
  - "Sparsity Predictor"
  - "SoftTopK"
  - "On-Device Inference"
date: 2026-05-08
content_hash: 1ad9ac0e27c9dfdd
---

# Instruction-Following Pruning for Large Language Models

**Conference**: ICML2025  
**arXiv**: [2501.02086](https://arxiv.org/abs/2501.02086)  
**Code**: Not open-sourced  
**Area**: Model Compression  
**Keywords**: Structured Pruning, Dynamic Pruning, Instruction-Aware, Sparsity Predictor, SoftTopK, On-Device Inference

## TL;DR

IFPruning is proposed: a small sparsity predictor dynamically generates pruning masks based on user instructions, trimming the intermediate dimensions of FFNs on demand. This enables a 9B model activating only 3B parameters to outperform dense models of the same scale by 5-8 percentage points in coding/math, while maintaining inference latency on par with a 3B dense model.

## Background & Motivation

Traditional structured pruning generates a **fixed pruning mask** for the model, meaning the compressed subnetwork shares the same parameters across all tasks. This creates an inherent bottleneck when facing different skill requirements such as programming, mathematics, and domain knowledge—a fixed subnetwork cannot be optimal across all tasks simultaneously.

Meanwhile, though existing dynamic methods (Contextual Sparsity, MoE) can select different parameters, they require loading different weights at **each decoding step**, resulting in significant weight transfer overhead and making them unsuitable for on-device deployment.

The core problem proposed in this paper is: **Can LLMs autonomously select the most appropriate parameter subset based on task descriptions?** Namely, determining the subnetwork once prior to decoding, keeping dynamic flexibility while avoiding the overhead of step-by-step reloading.

## Method

### Overall Architecture

IFPruning consists of two components:

1. **Sparsity Predictor**: A small LLM with 302M parameters + a two-layer MLP prediction head.
2. **Masked LLM**: Scale of 6B / 9B / 12B, where FFN layers are dynamically pruned according to the mask.

### Structured Pruning Formulation

For a standard FFN layer:

$$F_{\text{ffn}}(X) = \sigma(X W_1) W_2$$

A mask vector $\mathbf{m} \in \{0,1\}^{d_{\text{ffn}}}$ is introduced, and the pruned FFN output is:

$$F_{\text{ffn}}(X, \mathbf{m}) = \text{FF}_2(\text{FF}_1(X) \odot \mathbf{m})$$

where $m_i = 0$ indicates that the $i$-th column of $W_1$ and the $i$-th row of $W_2$ are pruned. The mask satisfies the sparsity constraint $\sum_i m_i = t_{\text{ffn}}$.

### Differentiable Mask Generation (SoftTopK)

The sparsity predictor outputs scores $\mathbf{z} \in \mathbb{R}^{L \times d_{\text{ffn}}}$, which are converted into a differentiable mask via SoftTopK:

$$\boldsymbol{\lambda}^{(i)} = g(\mathbf{z}^{(i)}), \quad \mathbf{m}^{(i)} = \boldsymbol{\lambda}^{(i)} \odot \text{Top}(\boldsymbol{\lambda}^{(i)}, t_{\text{ffn}})$$

where $g(\cdot)$ is a normalization function (ensuring $\sum_k \lambda_k^{(i)} = t_{\text{ffn}}$), and $\text{Top}(\cdot, t_{\text{ffn}})$ returns the top-k indicator mask. This mechanism enables gradients to flow back to the predictor.

### Loss & Training

**Stage 1: Continual Pre-training.** Text is segmented into chunks of fixed length. The $k$-th chunk is used to predict the mask, and the next-token prediction loss is computed on the $(k+1)$-th chunk:

$$\mathcal{L} = \sum_{k=1}^{K-1} \sum_{x_i \in \mathbf{x}^{(k+1)}} \ell\big[f(\mathbf{x}_{<i}; \boldsymbol{\theta}, \mathbf{m}^{(k)}), x_i\big]$$

**Stage 2: SFT Fine-tuning.** Jointly optimize the predictor and the LLM on millions of instruction data, where the user prompt is directly fed into the predictor to generate the mask. For multi-turn dialogues, only the **first user message** is used to select the subnetwork.

### Inference Modes

- **Per-input mode**: Each input independently predicts the mask, offering maximum flexibility.
- **Per-task mode**: A single task description is used to generate a shared mask for the same task (e.g., "mathematics"), reducing overhead.

## Key Experimental Results

Base models: 6B/9B/12B parameter LLMs, with 3B active parameters in all cases. Baselines: Dense-3B (trained on 9T tokens), Pruning+Distill 3B (static pruning + distillation, 12B teacher), and Dense-9B (upper bound without pruning).

| Task Category | Dataset | Dense-3B | Pruning+Distill | IFP 9B→3B | Dense-9B |
|---------|--------|----------|-----------------|-----------|----------|
| Coding | HumanEval | 35.2 | 37.1 | **42.4** | 46.5 |
| Coding | MBPP | 28.8 | 38.0 | **41.8** | 42.2 |
| Coding | MultiPL-E | 39.0 | 37.9 | **41.8** | 44.0 |
| Coding | Average | 34.3 | 37.7 | **42.0** | 44.2 |
| Math | GSM8K | 69.3 | 70.0 | **72.0** | 75.4 |
| Math | MATH | 31.8 | 32.7 | **36.7** | 37.3 |
| Instruction Following | AlpacaEval 2.0 | 27.3 | 30.0 | **31.3** | 38.6 |
| Knowledge | MMLU | 61.8 | 62.8 | **65.5** | 67.8 |
| Core Language | Average | 69.9 | 70.0 | **71.1** | 73.8 |

**Key Conclusions:**

- IFPruning 9B→3B outperforms Dense-3B by **+7.7pp** in coding and **+4.9pp** in mathematics.
- It outperforms Pruning+Distill (enhanced by knowledge distillation) by more than 4 percentage points.
- It closely approaches the upper bound performance of Dense-9B, with extremely small gaps particularly on MATH (36.7 vs 37.3) and MBPP (41.8 vs 42.2).

**Inference Efficiency:** Compared to the full model, TTFT is reduced by up to 57%, and generation time is reduced by up to 41%. The overhead of dynamic pruning and caching is <0.1s per sample, accounting for only 1-2% of the total generation time.

**Per-task vs Per-input:** Using a single task description to share masks for each task achieves performance almost on par with per-input mask prediction (HumanEval 40.9 vs 42.4), proving that pruning patterns for similar tasks are highly consistent.

## Highlights & Insights

1. **Paradigm Innovation**: For the first time, structured pruning is shifted from "static masks" to "instruction-driven dynamic masks", transforming pruning from a one-time compression into an adaptive capability during inference.
2. **Selecting Parameters Before Decoding, Fixing Them During Decoding**: Cleverly circumvents the I/O bottleneck of changing parameters step-by-step in MoE/Contextual Sparsity, which is particularly suitable for on-device scenarios.
3. **Strong Interpretability**: Subnetwork overlap rates for similar tasks (e.g., Math-GSM8K and MMLU-Math) reach up to ~80%, while cross-domain overlap rates (e.g., Math vs. History) are low, showing that the model learns meaningful domain-specialized parameter groups.
4. **Miniature Sparsity Predictor** (302M): The overhead is negligible compared to the pruned LLM (6-12B).
5. **Good Scalability**: Under IFPruning, the performance of 6B$\rightarrow$9B$\rightarrow$12B models steadily improves, with the most significant gains observed in coding and mathematics.

## Limitations & Future Work

1. **Only Pruning FFN Layers**: Attention heads and embedding layers are not pruned, which theoretically leaves room for further compression.
2. **High Training Cost**: Joint optimization of the predictor and LLM is required, and the two-stage training demands millions of SFT samples + pre-training data.
3. **Predictor Inference Overhead in Per-input Mode**: Although the paper claims <0.1s, it is still non-zero for ultra-low latency scenarios.
4. **No Direct Comparison with MoE**: There is no fair comparison with MoE models under the same parameter budget.
5. **Limited Gain on AlpacaEval**: The authors acknowledge that the improvement from increasing model scale on AlpacaEval is limited, and open-ended instruction-following scenarios may not fully benefit.
6. **Data Dependency**: The use of proprietary internal SFT datasets (millions of samples) limits reproducibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Dynamic instruction-driven pruning is a significant paradigm breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple scales, multiple baselines, and multiple tasks, including interpretability analysis and efficiency evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is well-structured, with motivation, method, and experiments progressing logically.
- **Value**: ⭐⭐⭐⭐ — Provides a highly promising new direction for on-device LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ICML 2025\] Olica: Efficient Structured Pruning of Large Language Models without Retraining](olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [\[ICML 2025\] DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](../../ACL2025/model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](../../ACL2025/model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)

</div>

<!-- RELATED:END -->
