---
title: >-
  [Paper Note] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs
description: >-
  [ACL 2026 Findings][Model Compression][Post-training quantization] This paper establishes the first task-stratified knowledge scaling laws for post-training quantization (PTQ), categorizing LLM capabilities into memory, application, and reasoning layers. It provides a unified model for four factors: model size, bit-width, group size, and calibration set size. Validated across 293 PTQ configurations, the study reveals distinct patterns: reasoning is highly sensitive to precisi…
tags:
  - "ACL 2026 Findings"
  - "Model Compression"
  - "Post-training quantization"
  - "Scaling laws"
  - "Knowledge stratification"
  - "Memory application reasoning"
  - "Fine-grained quantization factors"
date: 2026-05-08
content_hash: 9ce1b668a6cb9820
---

# Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2508.18609](https://arxiv.org/abs/2508.18609)  
**Code**: None  
**Area**: Model Compression / Quantization  
**Keywords**: Post-training quantization, Scaling laws, Knowledge stratification, Memory application reasoning, Fine-grained quantization factors

## TL;DR

This paper establishes the first task-stratified knowledge scaling laws for post-training quantization (PTQ), categorizing LLM capabilities into memory, application, and reasoning layers. It provides a unified model for four factors: model size, bit-width, group size, and calibration set size. Validated across 293 PTQ configurations, the study reveals distinct patterns: reasoning is highly sensitive to precision, application scales with model size, and memory is sensitive to calibration.

## Background & Motivation

**Background**: PTQ has become the mainstream strategy for LLM compression (with ~70% of quantization research focusing on PTQ). Existing scaling laws (e.g., Chinchilla) primarily describe the behavior of full-precision models, while the few existing quantization scaling laws only consider model size and bit-width.

**Limitations of Prior Work**: (1) Systemic impacts of fine-grained PTQ parameters, such as group size and calibration set size, are ignored. (2) Performance across all tasks is conflated, failing to capture the differentiated impact of quantization on memory, application, and reasoning capabilities.

**Key Challenge**: Existing scaling laws cannot guide practical questions such as "how to configure group size and calibration set size at low bit-widths to maintain specific capabilities."

**Goal**: Establish a unified four-factor power-law framework to fit scaling laws for three layers of knowledge capabilities.

**Key Insight**: LLM capabilities are categorized based on Bloom's Taxonomy into memory (precise factual recall), application (flexible knowledge usage), and reasoning (multi-step logic), with 14 benchmarks covering these three layers.

**Core Idea**: $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$, where the exponents $\alpha, \beta, \gamma, \delta$ are task-specific, quantifying the sensitivity of different capabilities to each factor.

## Method

### Overall Architecture

A systematic PTQ configuration scan was performed on the Qwen3 series (0.6B-14B) and Llama-3 series. Configurations included bit-widths (3/4/8), group sizes (32/64/128/1024), and calibration set sizes (8/32/128/1024), totaling 293 configurations. GPTQ was used as the unified quantization method. After evaluation on 14 benchmarks, OLS regression was used to fit the log-transformed power laws.

### Key Designs

**1. Unified Four-Factor Power-Law Framework: Integrating model size, bit-width, calibration set, and group size into a single interpretable formula**

Existing quantization scaling laws typically only consider model size $N$ and bit-width $B$, treating "fine-grained knobs" like group size and calibration sets as noise. Consequently, they cannot address practical deployment questions. This work incorporates all four factors: $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$. Logarithms are applied to bit-width $B$ and calibration set $C_b$ because the marginal gains from an additional bit or doubling calibration data diminish (saturation). The $-\ln(\cdot)$ transformation maps bounded accuracy to an unbounded "loss" space to satisfy the monotonic convexity required for power-law fitting.

Before fitting, a baseline adjustment is performed: $\text{Acc}_{\text{adj}} = \frac{\text{Acc} - \text{Acc}_{\text{random}}}{1 - \text{Acc}_{\text{random}}}$. This eliminates baseline differences from random guessing across tasks (e.g., 50% for binary tasks vs. 25% for multiple-choice). Finally, OLS regression is applied to the log-transformed equation. The fitted exponents $\alpha, \beta, \gamma, \delta$ represent elasticity coefficients: the relative change in performance loss for a 1% relative change in each factor.

**2. Task-Stratified Knowledge Hierarchy: Fitting memory, application, and reasoning separately rather than using an aggregate score**

Fitting aggregate performance masks critical differences: at a specific bit-width, reasoning might collapse while application tasks remain stable, yet the average score might appear acceptable. This paper utilizes Bloom’s Taxonomy to divide 14 benchmarks into three layers: L1 Memory (e.g., TriviaQA, NQ), L2 Application (e.g., MMLU, Hellaswag), and L3 Reasoning (e.g., GSM8K, ARC-C). Separate $\alpha, \beta, \gamma, \delta$ parameters are fitted for each layer, providing sensitivity profiles rather than a single curve to determine which capabilities fail first under low-bit quantization.

**3. Critical Role of Fine-Grained Factors in Low-Bit Scenarios: Proving group size and calibration sets are necessities, not options, at 2-3 bits**

Practitioners often use default group sizes and calibration sets for low-bit quantization, assuming these parameters are negligible. Ablation studies show that fitting with only two factors ($f(N, B)$) yields $R^2 = 0.91$, which jumps to $0.95$ when group size $G$ is included. Group size independently explains approximately 4% of additional variance, concentrated primarily in low-bit regions. While fine-grained factors may be redundant at higher bit-widths, they serve as a "safety net" against capability collapse at 3-bit or 2-bit levels.

### Loss & Training

No training is involved. GPTQ is used to minimize quantization reconstruction error layer-by-layer using the Hessian matrix.

## Key Experimental Results

### Main Results

**Comparison of Scaling Exponents Across Knowledge Capability Layers**

| Capability Layer | α(N) | β(B) | γ(Cb) | δ(G) | Adj R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| General | -0.359 | -1.067 | -0.032 | 0.073 | 0.9475 |
| L1 Memory | -0.315 | -0.964 | **-0.040** | 0.064 | 0.9350 |
| L2 Application | **-0.400** | -1.100 | -0.030 | 0.075 | 0.9500 |
| L3 Reasoning | -0.320 | **-1.200** | -0.025 | **0.085** | 0.9300 |

### Key Findings

- **Reasoning Precision Bottleneck**: $\beta_{\text{KR}} = -1.200$ (highest absolute value), indicating reasoning is most sensitive to bit-width.
- **Application Scaling Response**: $\alpha_{\text{KA}} = -0.400$ (highest absolute value), indicating application capability improves significantly with model scale.
- **Memory Calibration Sensitivity**: $\gamma_{\text{KM}} = -0.040$ (highest absolute value), indicating precise factual recall is most sensitive to the amount of calibration data.
- The four-factor model improves Adj R² by 3.5% over the two-factor baseline ($N, B$), and extrapolation on Qwen3-32B was successful.

## Highlights & Insights

- The task-stratified scaling law approach provides high practical value, helping practitioners make informed quantization configuration decisions under resource constraints.
- A significant finding is the criticality of fine-grained factors in low-bit scenarios; default configurations may lead to capability collapse at 3-bit.
- Consistency across architectures (Qwen3 to Llama-3) demonstrates the universality of these laws.

## Limitations & Future Work

- Only GPTQ was utilized as a quantization method.
- 2-bit data was excluded from fitting due to performance collapse.
- Quantization-aware training (QAT) and mixed-precision scenarios were not considered.
- Generative tasks were not covered in the evaluation.

## Related Work & Insights

- **vs. Chinchilla Laws**: While original laws target full-precision models, this work extends to quantization and adds group size/calibration set factors.
- **vs. QiD Laws**: Unlike models that only look at aggregate degradation, this work models three knowledge capabilities hierarchically.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First task-stratified PTQ scaling law with a unified four-factor framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 293 configurations, 14 benchmarks, cross-architecture validation, and extrapolation tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formula derivations and persuasive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Direct practical implications for LLM quantization deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](../../ICML2026/model_compression/llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ICLR 2026\] SliderQuant: Accurate Post-Training Quantization for LLMs](../../ICLR2026/model_compression/sliderquant_accurate_post-training_quantization_for_llms.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)

</div>

<!-- RELATED:END -->
