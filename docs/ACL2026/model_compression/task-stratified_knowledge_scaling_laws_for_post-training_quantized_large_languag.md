---
title: >-
  [Paper Note] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs
description: >-
  [ACL 2026][Model Compression][Paper Note] This paper establishes the first task-stratified knowledge scaling law for Post-Training Quantization (PTQ), categorizing LLM capabilities into three layers: memory, application, and reasoning. It uniformly models four factors: model size, bit-width, group size, and calibration set size. Validated on 293 PTQ configurat
tags:
  - ACL 2026
  - Model Compression
date: 2026-05-08
content_hash: 1192f04d2cf6fe1a
---
# Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2508.18609](https://arxiv.org/abs/2508.18609)  
**Code**: None  
**Area**: Model Compression / Quantization  
**Keywords**: Post-Training Quantization, Scaling Laws, Knowledge Stratification, Memory Application Reasoning, Fine-grained Quantization Factors

## TL;DR

This paper establishes the first task-stratified knowledge scaling law for Post-Training Quantization (PTQ), categorizing LLM capabilities into three layers: memory, application, and reasoning. It uniformly models four factors: model size, bit-width, group size, and calibration set size. Validated on 293 PTQ configurations, it reveals differentiated patterns where reasoning is sensitive to precision, application improves with scale, and memory is sensitive to calibration.

## Background & Motivation

**Background**: PTQ has become the mainstream strategy for LLM compression (~70% of quantization-related research focuses on PTQ). Existing scaling laws (such as Chinchilla) primarily describe the behavior of full-precision models, while a few quantization scaling laws only consider model size and bit-width.

**Limitations of Prior Work**: (1) The systematic impact of fine-grained PTQ parameters such as group size and calibration set size is ignored; (2) Performance across all tasks is aggregated, failing to capture the differentiated impact of quantization on memory, application, and reasoning capabilities.

**Key Challenge**: Existing scaling laws cannot guide practical questions such as "how to configure group size and calibration set size under low-bit quantization to maintain specific capabilities."

**Goal**: To establish a unified four-factor power-law framework and fit scaling laws for each of the three layers of knowledge capabilities.

**Key Insight**: Based on Bloom's Taxonomy, LLM capabilities are categorized into memory (precise factual recall), application (flexible knowledge usage), and reasoning (multi-step logic), covering these three layers with 14 benchmarks.

**Core Idea**: $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$, where the exponents $\alpha, \beta, \gamma, \delta$ are task-hierarchy-specific, quantifying the sensitivity of different capabilities to each factor.

## Method

### Overall Architecture

A systematic PTQ configuration scan was performed on the Qwen3 series (0.6B-14B) and Llama-3 series (bit-widths 3/4/8, group sizes 32/64/128/1024, calibration set sizes 8/32/128/1024), totaling 293 configurations. GPTQ was employed as the unified quantization method. Following evaluation on 14 benchmarks, log-transformed power laws were fitted using OLS regression.

### Key Designs

**1. Four-factor Unified Power Law Framework: Integrating model size, bit-width, calibration set, and group size into a single interpretable formula**

Existing quantization scaling laws only consider model size $N$ and bit-width $B$, treating "fine-grained knobs" like group size and calibration sets as noise. Consequently, they cannot address deployment issues such as "how to configure group size and calibration sets at low bits." This work incorporates four factors into $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$. Logarithms are applied to bit-width $B$ and calibration set size $C_b$ because the marginal gains from an additional bit or doubling calibration data are diminishing, which the logarithm accurately characterizes. The $-\ln(\cdot)$ transformation maps bounded accuracy into an unbounded "loss" space, ensuring the monotonic convexity required for power-law fitting.

Before fitting, a baseline adjustment is performed: $\text{Acc}_{\text{adj}} = \frac{\text{Acc} - \text{Acc}_{\text{random}}}{1 - \text{Acc}_{\text{random}}}$. This eliminates baseline differences in random guessing across tasks—otherwise, tasks with 50% random accuracy (binary choice) versus 25% (multiple choice) would contaminate cross-task exponent comparisons. Finally, OLS regression is applied to the log-transformed equation; the fitted exponents $\alpha, \beta, \gamma, \delta$ can be interpreted directly as elasticity coefficients, representing the relative change in performance loss for a 1% relative change in each factor.

**2. Task-Stratified Knowledge System: Fitting memory, application, and reasoning layers separately rather than using an average score**

Fitting only aggregate performance masks critical differences: at a specific bit-width, reasoning might have collapsed while application tasks still appear functional, yet the average score might suggest "no major issues." This study utilizes Bloom's Taxonomy to divide 14 benchmarks into three layers—L1 Memory (precise factual recall like TriviaQA/NQ/LAMA), L2 Application (flexible knowledge usage like MMLU/Hellaswag), and L3 Reasoning (multi-step logic like GSM8K/ARC-C). A separate set of $\alpha, \beta, \gamma, \delta$ is fitted for each layer. This produces sensitivity profiles for each capability layer rather than a single curve, answering which capability fails first under low-bit quantization.

**3. Critical Role of Fine-grained Factors in Low-bit Scenarios: Demonstrating that group size and calibration sets are necessities for preventing collapse at 2-3 bits**

Practitioners often use default group sizes and calibration sets for low-bit quantization, assuming these parameters are negligible. The ablation study refutes this. Fitting with only two factors $f(N,B)$ yields an $R^2 = 0.91$, which jumps to $0.95$ when group size $G$ is included; group size independently explains approximately 4% of additional variance. Crucially, this 4% is not uniformly distributed but concentrated in low-bit regions. In other words, while group size is nearly redundant at high bits, it serves as a "safety buffer" at 3-bit or 2-bit levels, where default configurations might otherwise drive certain capabilities to collapse.

### Loss & Training

No training is involved. GPTQ uses the Hessian matrix to minimize layer-wise quantization reconstruction error.

## Key Experimental Results

### Main Results

**Comparison of Scaling Exponents Across Knowledge Capability Layers**

| Capability Layer | α(N) | β(B) | γ(Cb) | δ(G) | Adj R² |
|------------------|------|------|-------|------|--------|
| General          | -0.359 | -1.067 | -0.032 | 0.073 | 0.9475 |
| L1 Memory        | -0.315 | -0.964 | **-0.040** | 0.064 | 0.9350 |
| L2 Application   | **-0.400** | -1.100 | -0.030 | 0.075 | 0.9500 |
| L3 Reasoning     | -0.320 | **-1.200** | -0.025 | **0.085** | 0.9300 |

### Key Findings

- **Reasoning Precision Bottleneck**: $\beta_{\text{KR}} = -1.200$ (largest absolute value), indicating reasoning is most sensitive to bit-width.
- **Application Scale Response**: $\alpha_{\text{KA}} = -0.400$ (largest absolute value), indicating application capability improves significantly with model scale.
- **Memory Calibration Sensitivity**: $\gamma_{\text{KM}} = -0.040$ (largest absolute value), indicating precise factual recall is most sensitive to the amount of calibration data.
- The four-factor model improves Adj R² by 3.5% over the two-factor baseline (N, B), and extrapolation validation on Qwen3-32B was successful.

## Highlights & Insights

- The approach of task-stratified scaling laws is highly practical, guiding practitioners to make informed quantization configuration decisions under resource constraints.
- The criticality of fine-grained factors in low-bit scenarios is a significant finding—default configurations at 3-bit can lead to capability collapse.
- Consistency across architectures (Qwen3 → Llama-3) demonstrates the universality of the law.

## Limitations & Future Work

- Only one quantization method (GPTQ) was used.
- 2-bit data was excluded from fitting due to performance collapse.
- Quantization-Aware Training (QAT) or mixed-precision scenarios were not considered.
- Evaluation of generative tasks was not covered.

## Related Work & Insights

- **vs Chinchilla Laws**: While Chinchilla targets full-precision models, this work extends to quantization and adds group size/calibration set factors.
- **vs QiD Laws**: Unlike QiD which only models aggregate degradation, this work provides stratified modeling for three distinct knowledge capabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First task-stratified PTQ scaling law with a unified four-factor framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 293 configurations + 14 benchmarks + cross-architecture validation + extrapolation testing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formula derivation and persuasive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Provides direct guidance for LLM quantization practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer](../../CVPR2026/model_compression/qvggt_post-training_quantized_visual_geometry_grounded_transformer.md)
- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](../../ICML2026/model_compression/llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)

</div>

<!-- RELATED:END -->
