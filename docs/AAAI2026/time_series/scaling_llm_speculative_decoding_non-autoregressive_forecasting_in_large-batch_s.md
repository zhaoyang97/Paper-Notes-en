---
title: >-
  [Paper Note] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios
description: >-
  [AAAI 2026][Time Series][Speculative Decoding] This paper proposes SpecFormer, a non-autoregressive draft model architecture that integrates unidirectional and bidirectional attention. By reducing reliance on large prefi…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Speculative Decoding"
  - "Non-Autoregressive Generation"
  - "Large-Batch Inference"
  - "LLM Acceleration"
  - "SpecFormer"
date: 2026-05-08
content_hash: 9856b9e4778593ce
---

# Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios

**Conference**: AAAI 2026
**arXiv**: [2511.20340](https://arxiv.org/abs/2511.20340)
**Code**: [https://github.com/ShiLuohe/SpecFormer](https://github.com/ShiLuohe/SpecFormer)
**Area**: Time Series / LLM Inference Acceleration
**Keywords**: Speculative Decoding, Non-Autoregressive Generation, Large-Batch Inference, LLM Acceleration, SpecFormer

## TL;DR

This paper proposes SpecFormer, a non-autoregressive draft model architecture that integrates unidirectional and bidirectional attention. By reducing reliance on large prefix trees and minimizing position-dependent parameters, SpecFormer achieves consistent LLM inference acceleration in large-batch scenarios.

## Background & Motivation

### Problem Origin

Large language models (LLMs) employ autoregressive decoding, generating one token per step, which results in low Arithmetic Intensity (AI) and significant underutilization of chip compute capacity. **Speculative Decoding (SD)** is a key method for improving AI: a small draft model rapidly generates multiple candidate tokens, which are then verified in parallel by the large model, allowing multiple tokens to be accepted in a single forward pass.

### Limitations of Prior Work

**Failure in large-batch scenarios**: Continuous batching has been widely adopted by mainstream inference frameworks and inherently compresses available idle compute. As batch size increases, per-parameter compute intensity grows, drastically reducing the computational budget available for draft tokens. Existing SD methods rely on large-scale prefix trees and cannot function properly under large batches.

**Excessive position-dependent parameters**: Both autoregressive methods (e.g., EAGLE, HASS) and non-autoregressive methods (e.g., Medusa, MTP) have parameter counts that scale linearly with draft sequence length. Autoregressive methods must repeatedly access parameters for each position, while non-autoregressive methods assign independent parameters per position. This makes scaling these methods difficult under resource constraints.

**Trade-off between draft quality and draft quantity**: Under low computational budgets, large prefix trees can no longer be relied upon; the accuracy of each individual draft token must be improved instead.

### Core Insight

Draft generation in SD only requires a fixed-length sequence of future tokens rather than open-ended generation, making **bidirectional attention** suitable for parallel generation. Furthermore, by extracting rich contextual information from multiple hidden layers of the LLM, draft quality can be significantly improved without fine-tuning the original LLM.

## Method

### Overall Architecture

SpecFormer consists of two core modules:
- **Context Causal Attention**: Extracts information from multiple hidden layers of the LLM to generate position-specific initial representations.
- **Draft Bi-directional Attention**: Applies standard bidirectional self-attention across the draft token dimension for parallel refinement.

### Key Designs

#### 1. Context Causal Attention

This module aims to fully extract contextual information from the LLM's hidden states, providing high-quality inputs for subsequent draft generation.

**Hook & Downsampler**: Hidden states are extracted from 4 key layers of the LLM:
- $\mathrm{HS}[0]$: Embedding layer, containing raw token information without contextual processing.
- $\mathrm{HS}[L/2]$: Middle layer, serving as supplementary information.
- $\mathrm{HS}[L-1]$: Second-to-last layer, encoding the most abstract current-token information.
- $\mathrm{HS}[L]$: Final layer, used directly for next-token prediction.

The hidden states from these 4 layers are concatenated, normalized via Grouped RMS Norm (with independent scaling parameters per layer slice), and then linearly downsampled from $4d_h$ to $d_h$ dimensions:

$$I_D = (\mathrm{MSA} \cdot \mathrm{RMS} + \mathbb{I})(W_D \cdot I_{\mathrm{Cat}})$$

The MSA here can be viewed as the $(L+1)$-th layer of the LLM, enabling seamless integration with existing KV cache management frameworks.

**Positional FFN**: A linear projection maps $d_h$ dimensions to $l_d \cdot d_h$ dimensions, injecting position-specific information for each draft position:

$$D = W_P \cdot \mathrm{RMS}(I_D) + b_P$$

The number of position-dependent parameters is $l_d \cdot d_h^2$, which is more efficient than Medusa's $8 \cdot l_d \cdot d_h^2$.

#### 2. Draft Bi-directional Attention

The core innovation: **standard bidirectional self-attention** (without causal masking) is applied across the draft token dimension, allowing all draft positions to exchange information with one another.

$$E = (\mathrm{SwiGLU} \cdot \mathrm{RMS} + \mathbb{I})((\mathrm{SA} \cdot \mathrm{RMS} + \mathbb{I})(D))$$

Key points:
- Attention operates on the draft token dimension, with effective batch size becoming $bs \cdot |c|$.
- Due to FlashAttention 2's batch size limit (4095), a grouped computation strategy is adopted (group size 3072).
- The majority of parameters are **position-independent** (SA and SwiGLU parameters are shared across all positions); only the Positional FFN contains position-dependent parameters.

#### 3. Efficiency Analysis Framework

The paper proposes a systematic SD efficiency evaluation framework:

**Redundancy ratio** $\rho = AI_c / AI_m$: Measures the theoretical upper bound on hardware speedup. For A100, $\rho \approx 152.86$.

**Optimization coefficient** $\kappa$: Evaluates method efficiency under a fixed draft token budget:

$$\kappa = \frac{a \cdot l_d}{k}$$

where $a$ is the average acceptance length, $l_d$ is the draft sequence length, and $k$ is the total number of draft tokens.

**Additional computational overhead** $p$:

$$p = 1 + \frac{m_s + l_d \cdot m_p}{M}$$

SpecFormer's $m_p$ (position-dependent parameters) is far smaller than that of Medusa and similar methods, yielding higher $\kappa$ under the same budget.

### Loss & Training

- **Self-Distillation**: Rather than using the original responses from UltraChat-200K directly, only the question portions are retained and responses are regenerated by the base LLM, ensuring strict distribution alignment between the draft model and the base model.
- **Training Objective**: Standard multi-position next-token prediction loss (Equation 6b).
- **No LLM Fine-tuning**: All training is restricted to SpecFormer's own parameters.

## Key Experimental Results

### Main Results

Comparison on Qwen2.5-7B (averaged across multiple benchmarks):

| Batch Size | Draft Tokens $k$ | Method | $\kappa$ | TPS | Speedup |
|:---:|:---:|:---|:---:|:---:|:---:|
| 1 | 4 | W/o SD | 1 | 41 | 1.00× |
| 1 | 4 | HASS | 2.14 | 69 | 1.70× |
| 1 | 4 | EAGLE-3 | 2.16 | 70 | 1.73× |
| 1 | 4 | **SpecFormer** | **2.20** | **73** | **1.78×** |
| 64 | 4 | W/o SD | 1 | 2590 | 1.00× |
| 64 | 4 | HASS | 2.13 | 4454 | 1.72× |
| 64 | 4 | EAGLE-3 | 2.15 | 4429 | 1.71× |
| 64 | 4 | **SpecFormer** | **2.19** | **4610** | **1.78×** |
| 128 | 4 | W/o SD | 1 | 5143 | 1.00× |
| 128 | 4 | HASS | 2.14 | 8800 | 1.71× |
| 128 | 4 | EAGLE-3 | 2.16 | 8846 | 1.72× |
| 128 | 4 | **SpecFormer** | **2.18** | **9154** | **1.78×** |

Cross-model scaling results (Qwen3 series):

| Batch Size | Model | No SD TPS | SpecFormer TPS | Speedup |
|:---:|:---|:---:|:---:|:---:|
| 1 | Qwen3-4B | 30 | 46 | 1.54× |
| 1 | Qwen3-8B | 31 | 46 | 1.49× |
| 1 | Qwen3-14B | 26 | 39 | 1.46× |
| 64 | Qwen3-4B | 2346 | 3621 | 1.53× |
| 64 | Qwen3-8B | 1904 | 2834 | 1.48× |
| 64 | Qwen3-14B | 1713 | 2524 | 1.47× |

### Ablation Study

| Configuration | $\kappa$ | TPS | Note |
|:---|:---:|:---:|:---|
| SpecFormer (w/ self-distillation) | 1.90 | 56 (1.76×) | Full method |
| SpecFormer (w/o self-distillation) | 1.19 | 30 (0.94×) | Severe performance degradation |

The effect of self-distillation is highly significant: without it, $\kappa$ drops to only 1.19, yielding virtually no speedup or even a slight slowdown.

### Key Findings

1. **Consistent acceleration**: SpecFormer maintains approximately 1.78× speedup across all settings from bs=1 to bs=128, whereas baseline methods show noticeable degradation at larger batch sizes.
2. **Self-distillation is critical**: Aligning training data distribution with the base model is key to obtaining high-quality drafts.
3. **Cross-model generalization**: The method remains effective across model scales from 4B to 14B parameters.
4. **Low training cost**: Compared to methods that modify the LLM itself, SpecFormer only trains a small draft head, requiring significantly fewer training resources.

## Highlights & Insights

1. **Precise problem formulation**: The paper refines SD efficiency evaluation from the coarse-grained "average acceptance length" to a per-budget optimization coefficient $\kappa$, providing a unified evaluation standard across different deployment scenarios.
2. **Principled use of bidirectional attention**: The paper cleverly leverages the property that SD only requires finite-length drafts, breaking free from the autoregressive constraint imposed by open-ended generation.
3. **Engineering rigor**: Implementation details including Triton-based Grouped RMS Norm and gradient accumulation strategies for the LM Head ensure that the proposed method achieves genuine speedups in practical deployment.

## Limitations & Future Work

1. **Only lossless SD is evaluated**: The paper focuses on lossless speculative decoding and does not explore the potential of lossy (approximate) settings.
2. **Training data dependency**: Self-distillation requires regenerating data with the base model, which may increase preparation costs in certain scenarios.
3. **FlashAttention batch size constraint**: When $bs \cdot |c|$ is large, grouped processing is required, introducing additional scheduling overhead.
4. **No discussion of synergy with system-level optimizations**: In practice, SD must be co-designed with techniques such as prefill-decode disaggregation and continuous batching.

## Related Work & Insights

- **EAGLE series** (EAGLE, EAGLE-3): Autoregressive methods at the hidden-state level; SpecFormer surpasses EAGLE-3.
- **Medusa**: A representative non-autoregressive method; its position-dependent parameters are 8× those of SpecFormer.
- **MTP (DeepSeek-V3)**: Multi-token prediction with few position-shared parameters and many position-specific parameters.
- Inspiration: The bidirectional attention paradigm could be extended to other "finite-length generation" tasks in future work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing bidirectional attention into SD draft generation is a novel and well-motivated design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers diverse model scales and batch sizes; the self-distillation ablation is convincing.
- **Writing Quality**: ⭐⭐⭐⭐ — The efficiency analysis framework is clear, though some equations involve heavy notation.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the core bottleneck of SD in real-world large-batch deployment; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](../../ICLR2026/time_series/from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[AAAI 2026\] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)
- [\[AAAI 2026\] A Theoretical Analysis of Detecting Large Model-Generated Time Series](a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)
- [\[NeurIPS 2025\] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](../../NeurIPS2025/time_series/parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)

</div>

<!-- RELATED:END -->
