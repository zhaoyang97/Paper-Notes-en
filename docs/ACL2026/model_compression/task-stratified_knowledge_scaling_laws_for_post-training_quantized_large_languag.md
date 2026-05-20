---
title: >-
  [Paper Note] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs
description: >-
  [ACL 2026][Model Compression][Post-training quantization] This paper establishes the first task-stratified knowledge scaling laws for post-training quantization (PTQ)…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Post-training quantization"
  - "scaling laws"
  - "knowledge stratification"
  - "memorization-application-reasoning"
  - "fine-grained quantization factors"
date: 2026-05-08
content_hash: 7bf549d7bc2dea26
---

# Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**Conference**: ACL 2026
**arXiv**: [2508.18609](https://arxiv.org/abs/2508.18609)  
**Code**: None  
**Area**: Model Compression / Quantization
**Keywords**: Post-training quantization, scaling laws, knowledge stratification, memorization-application-reasoning, fine-grained quantization factors

## TL;DR

This paper establishes the first task-stratified knowledge scaling laws for post-training quantization (PTQ), decomposing LLM capabilities into three levels—memorization, application, and reasoning—and jointly modeling four factors: model size, bit-width, group size, and calibration set size. The laws are validated across 293 PTQ configurations, revealing differentiated patterns: reasoning is sensitive to precision, application improves with scale, and memorization is sensitive to calibration data.

## Background & Motivation

**Background**: PTQ has become the dominant strategy for LLM compression (~70% of quantization research focuses on PTQ). Existing scaling laws (e.g., Chinchilla) primarily characterize full-precision models, and the few quantization-aware scaling laws consider only model size and bit-width.

**Limitations of Prior Work**: (1) The systematic effects of fine-grained PTQ parameters such as group size and calibration set size are ignored; (2) aggregating performance across all tasks obscures the differentiated impact of quantization on memorization, application, and reasoning capabilities.

**Key Challenge**: Existing scaling laws cannot guide practical decisions such as "how to configure group size and calibration set size under low-bit quantization to preserve specific capabilities."

**Goal**: Establish a unified four-factor power-law framework and fit separate scaling laws for each of the three knowledge capability levels.

**Key Insight**: LLM capabilities are stratified following Bloom's Taxonomy into memorization (exact factual recall), application (flexible knowledge use), and reasoning (multi-step logic), covered by 14 benchmarks across three levels.

**Core Idea**: $-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$, where the exponents $\alpha, \beta, \gamma, \delta$ are task-level-specific, quantifying the sensitivity of each capability to each factor.

## Method

### Overall Architecture

A systematic PTQ configuration sweep is conducted on the Qwen3 series (0.6B–14B) and Llama-3 series, covering bit-widths of 3/4/8, group sizes of 32/64/128/1024, and calibration set sizes of 8/32/128/1024, yielding 293 configurations in total. GPTQ is adopted as the unified quantization method. After evaluation on 14 benchmarks, OLS regression is applied to fit the log-transformed power law.

### Key Designs

1. **Four-Factor Unified Power-Law Framework**:

    - Function: Jointly models the combined effect of model size $N$, bit-width $B$, calibration set $C_b$, and group size $G$ on quantized performance.
    - Mechanism: Logarithms are applied to $B$ and $C_b$ to model diminishing marginal returns; $-\ln(\text{Acc}_{\text{adj}})$ maps performance to an unbounded "loss" space; OLS is applied to the log-transformed equation. The baseline-adjusted accuracy $\text{Acc}_{\text{adj}} = \frac{\text{Acc} - \text{Acc}_{\text{random}}}{1 - \text{Acc}_{\text{random}}}$ removes the random-chance baseline differences across tasks.
    - Design Motivation: The exponents are interpretable as elasticity coefficients measuring the sensitivity of performance to relative changes in each factor. The log transformation restores the monotone convexity required for bounded accuracy metrics.

2. **Task-Stratified Knowledge Taxonomy**:

    - Function: Disentangles the differentiated impact of quantization on cognitive capabilities at distinct levels.
    - Mechanism: L1 Memorization (exact factual recall: TriviaQA/NQ/LAMA etc.), L2 Application (flexible knowledge use: MMLU/HellaSwag etc.), and L3 Reasoning (multi-step logic: GSM8K/ARC-C etc.) each receive independently fitted scaling laws.
    - Design Motivation: Fitting only aggregate performance conceals critical differences—reasoning may have already collapsed while application remains intact.

3. **Critical Role of Fine-Grained Factors in Low-Bit Regimes**:

    - Function: Demonstrates that at 2–3-bit quantization, group size and calibration set size are no longer optional parameters but necessary conditions for preventing capability collapse.
    - Mechanism: Ablations show $f(N,B)$ achieves $R^2 = 0.91$; adding $G$ raises it to 0.95, indicating that group size accounts for approximately 4% of additional variance—concentrated precisely in the low-bit regime.
    - Design Motivation: Practitioners often use default group sizes and calibration sets in low-bit quantization, which may lead to unnecessary performance collapse.

### Loss & Training

No training is involved. GPTQ minimizes layer-wise quantization reconstruction error using the Hessian matrix.

## Key Experimental Results

### Main Results

**Scaling Exponents Across Knowledge Capability Levels**

| Capability Level | α(N) | β(B) | γ(Cb) | δ(G) | Adj R² |
|-----------------|------|------|-------|------|--------|
| General | -0.359 | -1.067 | -0.032 | 0.073 | 0.9475 |
| L1 Memorization | -0.315 | -0.964 | **-0.040** | 0.064 | 0.9350 |
| L2 Application | **-0.400** | -1.100 | -0.030 | 0.075 | 0.9500 |
| L3 Reasoning | -0.320 | **-1.200** | -0.025 | **0.085** | 0.9300 |

### Key Findings

- **Reasoning precision bottleneck**: $\beta_{\text{KR}} = -1.200$ (largest absolute value), indicating that reasoning is the most sensitive capability to bit-width.
- **Application scale responsiveness**: $\alpha_{\text{KA}} = -0.400$ (largest absolute value), indicating that application capability benefits most from increased model scale.
- **Memorization calibration sensitivity**: $\gamma_{\text{KM}} = -0.040$ (largest absolute value), indicating that exact factual recall is the most sensitive capability to calibration data volume.
- The four-factor model improves Adj R² by 3.5% over the two-factor baseline $(N, B)$, and extrapolation to Qwen3-32B is validated successfully.

## Highlights & Insights

- The task-stratified scaling law framework is highly practical—it guides practitioners toward informed quantization configuration decisions under resource constraints.
- The critical role of fine-grained factors in low-bit regimes is an important finding: default configurations at 3-bit may cause capability collapse.
- Consistent behavior across architectures (Qwen3 → Llama-3) confirms the generalizability of the derived laws.

## Limitations & Future Work

- Only GPTQ is employed as the quantization method.
- 2-bit data are excluded from fitting due to performance collapse.
- QAT and mixed-precision scenarios are not considered.
- Evaluation of generative tasks is not covered.

## Related Work & Insights

- **vs. Chinchilla Laws**: Chinchilla targets full-precision models; this work extends the framework to quantized settings and incorporates group size and calibration set as additional factors.
- **vs. QiD Laws**: QiD models only aggregate performance degradation; this work stratifies the analysis across three distinct knowledge capability levels.

## Rating

- Novelty: ⭐⭐⭐⭐ First task-stratified PTQ scaling laws; unified four-factor framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 293 configurations + 14 benchmarks + cross-architecture validation + extrapolation testing.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations; convincing figures and tables.
- Value: ⭐⭐⭐⭐⭐ Directly actionable for LLM quantization practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ICLR 2026\] A universal compression theory for lottery ticket hypothesis and neural scaling laws](../../ICLR2026/model_compression/a_universal_compression_theory_for_lottery_ticket_hypothesis_and_neural_scaling_.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)

</div>

<!-- RELATED:END -->
