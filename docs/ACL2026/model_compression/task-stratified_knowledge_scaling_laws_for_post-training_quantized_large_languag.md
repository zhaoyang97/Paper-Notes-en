---
title: >-
  [Paper Note] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs
description: >-
  [ACL 2026][Model Compression][Post-Training Quantization] This paper establishes the first task-stratified knowledge scaling laws for post-training quantization (PTQ), categorizing LLM capabilities into memory…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Scaling Laws"
  - "Knowledge Stratification"
  - "Memory-Application-Reasoning"
  - "Fine-grained Quantization Factors"
date: 2026-05-08
content_hash: 77bf51792aa957b5
---

# Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**Conference**: ACL 2026  
**arXiv**: [2508.18609](https://arxiv.org/abs/2508.18609)  
**Code**: None  
**Area**: Model Compression / Quantization  
**Keywords**: Post-Training Quantization, Scaling Laws, Knowledge Stratification, Memory-Application-Reasoning, Fine-grained Quantization Factors

## TL;DR

This paper establishes the first task-stratified knowledge scaling laws for post-training quantization (PTQ), categorizing LLM capabilities into memory, application, and reasoning layers. It provides a unified model for four factors: model size, bit-width, group size, and calibration set size. Validated across 293 PTQ configurations, the study reveals that reasoning is precision-sensitive, application improves with scale, and memory is particularly sensitive to calibration.

## Background & Motivation

**Background**: PTQ has become a mainstream strategy for LLM compression (approximately 70% of quantization research focuses on PTQ). Existing scaling laws, such as Chinchilla, primarily describe the behavior of full-precision models, while fewer quantization-related scaling laws consider only model size and bit-width.

**Limitations of Prior Work**: (1) Prior studies ignore the systematic impact of fine-grained PTQ parameters, such as group size and calibration set size; (2) Performance across all tasks is typically aggregated, failing to capture the differentiated impact of quantization on memory, application, and reasoning capabilities.

**Key Challenge**: Existing scaling laws cannot guide practical questions such as "how to configure group size and calibration set size under low-bit quantization to maintain specific capabilities."

**Goal**: Establish a unified four-factor power-law framework to fit scaling laws for three distinct layers of knowledge capabilities.

**Key Insight**: Based on Bloom's Taxonomy, LLM capabilities are divided into Memory (exact fact recall), Application (flexible knowledge utilization), and Reasoning (multi-step logic), covering these three layers with 14 benchmarks.

**Core Idea**: $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$, where the exponents $\alpha, \beta, \gamma, \delta$ are task-hierarchy specific, quantifying the sensitivity of different capabilities to each factor.

## Method

### Overall Architecture

A systematic PTQ configuration scan was performed on the Qwen3 series (0.6B-14B) and Llama-3 series across 293 configurations (bit-widths 3/4/8, group sizes 32/64/128/1024, and calibration set sizes 8/32/128/1024). GPTQ served as the unified quantization method. Scaling laws were fitted using OLS regression on log-transformed power laws after evaluation on 14 benchmarks.

### Key Designs

1.  **Unified Four-Factor Power-Law Framework**:
    - **Function**: Models the joint influence of model size $N$, bit-width $B$, calibration set size $C_b$, and group size $G$ on quantization performance.
    - **Mechanism**: Logarithms of $B$ and $C_b$ are used to model diminishing marginal returns. Performance is transformed into an unbounded "loss" space via $-\ln(\text{Acc}_{\text{adj}})$, and the log-transformed equation is fitted via OLS. Baseline adjustment $\text{Acc}_{\text{adj}} = \frac{\text{Acc} - \text{Acc}_{\text{random}}}{1 - \text{Acc}_{\text{random}}}$ eliminates differences in random baselines across tasks.
    - **Design Motivation**: Exponents are interpretable as elasticity coefficients, measuring performance sensitivity to relative changes in factors. Log-transformation ensures the monotonic convexity required for bounded accuracy.

2.  **Task-stratified Knowledge System**:
    - **Function**: To isolate the differentiated impacts of quantization across cognitive levels.
    - **Mechanism**: Capabilities are categorized into L1 Memory (e.g., TriviaQA, NQ, LAMA), L2 Application (e.g., MMLU, Hellaswag), and L3 Reasoning (e.g., GSM8K, ARC-C), with independent scaling laws fitted for each.
    - **Design Motivation**: Aggregated performance fits can mask critical differences; for example, reasoning might collapse while application capabilities remain functional.

3.  **Critical Role of Fine-grained Factors in Low-bit Scenarios**:
    - **Function**: Demonstrates that in 2-3 bit quantization, group size and calibration sets are necessary requirements rather than optional parameters for preventing collapse.
    - **Mechanism**: Ablation studies show that $f(N,B)$ has an $R^2 = 0.91$, which jumps to $0.95$ when $G$ is included, indicating that group size explains approximately 4% of additional variance, concentrated in low-bit regions.
    - **Design Motivation**: Practitioners often use default group sizes and calibration sets during low-bit quantization, which can lead to unnecessary performance collapse.

### Loss & Training

This work does not involve training. GPTQ is utilized to minimize layer-wise quantization reconstruction error using the Hessian matrix.

## Key Experimental Results

### Main Results

**Comparison of Scaling Exponents across Knowledge Layers**

| Capability Layer | α(N) | β(B) | γ(Cb) | δ(G) | Adj R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| General | -0.359 | -1.067 | -0.032 | 0.073 | 0.9475 |
| L1 Memory | -0.315 | -0.964 | **-0.040** | 0.064 | 0.9350 |
| L2 Application | **-0.400** | -1.100 | -0.030 | 0.075 | 0.9500 |
| L3 Reasoning | -0.320 | **-1.200** | -0.025 | **0.085** | 0.9300 |

### Key Findings

- **Reasoning Precision Bottleneck**: $\beta_{\text{KR}} = -1.200$ (highest absolute value), indicating that reasoning is most sensitive to bit-width.
- **Application Scaling Response**: $\alpha_{\text{KA}} = -0.400$ (highest absolute value), suggesting application capability improves most significantly with model scale.
- **Memory Calibration Sensitivity**: $\gamma_{\text{KM}} = -0.040$ (highest absolute value), showing that exact fact recall is most sensitive to the volume of calibration data.
- The four-factor model improves Adj R² by 3.5% compared to the two-factor baseline ($N, B$), and successfully passes extrapolation verification on Qwen3-32B.

## Highlights & Insights

- The approach of task-stratified scaling laws is highly practical, guiding practitioners toward informed quantization configurations under resource constraints.
- The criticality of fine-grained factors in low-bit scenarios is a significant discovery—default configurations at 3-bit may cause total capability collapse.
- Consistency across architectures (Qwen3 → Llama-3) demonstrates the universality of the established laws.

## Limitations & Future Work

- Only GPTQ was employed as the quantization method.
- 2-bit data was excluded from fitting due to performance collapse.
- Quantization-Aware Training (QAT) and mixed-precision scenarios were not considered.
- Generative task evaluations were not covered.

## Related Work & Insights

- **vs Chinchilla Laws**: While Chinchilla focuses on full-precision models, this work extends to quantization and includes group size/calibration set factors.
- **vs QiD Laws**: Unlike QiD, which models aggregate degradation, this work provides stratified modeling for three knowledge capabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First task-stratified PTQ scaling law; unified four-factor framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 293 configurations, 14 benchmarks, cross-architecture validation, and extrapolation tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations and persuasive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Direct practical implications for LLM quantization deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](../../ICML2026/model_compression/llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)

</div>

<!-- RELATED:END -->
