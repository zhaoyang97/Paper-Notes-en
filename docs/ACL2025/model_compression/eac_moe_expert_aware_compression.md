---
title: >-
  [Paper Note] EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
description: >-
  [ACL 2025][Model Compression][MoE Quantization] EAC-MoE provides an in-depth analysis of the expert selection characteristics of MoE models and proposes two complementary modules: Quantization with Expert Selection Calibration (QESC) to alleviate the expert-shift issue by calibrating routers layer-by-layer during quantization, and Pruning based on Expert Selection Frequency (PESF) to dynamically prune unimportant experts during inference based on selection frequency. It achie…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "MoE Quantization"
  - "Router Calibration"
  - "Dynamic Expert Pruning"
  - "Expert-Shift"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 744335da591890d3
---

# EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2508.01625](https://arxiv.org/abs/2508.01625)  
**Code**: —  
**Area**: Model Compression, MoE  
**Keywords**: MoE Quantization, Router Calibration, Dynamic Expert Pruning, Expert-Shift, Inference Acceleration  

## TL;DR

EAC-MoE provides an in-depth analysis of the expert selection characteristics of MoE models and proposes two complementary modules: Quantization with Expert Selection Calibration (QESC) to alleviate the expert-shift issue by calibrating routers layer-by-layer during quantization, and Pruning based on Expert Selection Frequency (PESF) to dynamically prune unimportant experts during inference based on selection frequency. It achieves significant memory compression and inference acceleration with minimal accuracy loss across 4 MoE models.

## Background & Motivation

- **Dual Bottlenecks of MoE Deployment**: (1) Although MoE only activates a subset of experts, the weights of all experts must be loaded into GPU memory (Mixtral-8x7B has active parameters similar to LLaMA2-13B, but its total parameters are about 4 times larger, occupying 94GB of VRAM); (2) Low active parameter count does not equate to inference acceleration—in long-sequence or batched inference, different tokens select different experts, meaning all experts still participate in computation.
- **Limitations of Prior Work in Applying Dense LLM Compression Methods Directly**: (1) Regarding quantization, low-bit quantization causes the router to select the wrong experts (expert-shift), which is the primary factor in MoE performance degradation; (2) Regarding pruning, static pruning fails to adapt to the varying expert preferences across different tasks.
- **Key Insight**: MoE models exhibit **completely different expert preferences** across different types of tasks (Math, Code, QA/CR)—the expert selection frequencies are highly similar within the same type of task (cosine similarity >0.8) but differ significantly across task categories. This implies that: quantization should ensure correct expert selection rather than permanently evaluating expert importance; pruning should dynamically evaluate expert importance for the current task.

## Method

### Overall Architecture

EAC-MoE compresses MoE models from both pre-inference (offline pre-computation) and in-inference perspectives: (1) QESC calibrates the router layer-by-layer during the offline quantization phase to alleviate expert-shift; (2) PESF dynamically prunes low-frequency experts during inference to reduce computation. These two components can be used in combination.

### Key Designs

1. **Quantization with Expert Selection Calibration (QESC)**: Utilizing WikiText2 calibration data, execution is performed layer-by-layer: first quantizing the MHSA component $\rightarrow$ calibrating the MoE layer router $\rightarrow$ quantizing all experts. Router calibration employs a TopK-MSE loss (computing MSE only for the top-K categories in the probability distribution), avoiding noise introduced by a large number of low-probability experts dominating the loss.
2. **Pruning based on Expert Selection Frequency (PESF)**: During inference, for an input sequence length $l$, if an expert is selected $c < \frac{l \times K}{N} \times \alpha$ times (i.e., the selection count is lower than $\alpha$ times the average), this expert is pruned for the current sequence. This approach prunes from the expert dimension rather than the token dimension, allowing the computation of unimportant experts to be bypassed entirely.
3. **TopK-MSE Loss**: $\mathcal{L} = \frac{1}{K}\sum_{i \in \text{top-K}(\mathbf{W}\mathbf{x})}((\mathbf{W}\mathbf{x})_i - (\mathbf{W}\hat{\mathbf{x}})_i)^2$, which focuses solely on the $K$ experts most likely to be selected, effectively avoiding ineffective optimization on a large number of low-probability experts among the 64 experts.

### Loss & Training

QESC router calibration uses the TopK-MSE loss. PESF does not involve training loss and is a purely dynamic decision process during inference.

## Experimental Results

### Main Results: Quantization Performance Comparison

| Bits | Method | Mixtral PPL↓ | Mixtral Acc↑ | DeepSeek PPL↓ | DeepSeek Acc↑ |
|------|------|-------------|-------------|-------------|-------------|
| 16 | Baseline | 3.84 | 72.64 | 6.51 | 61.38 |
| 2.06 | GPTQ | 5.51 | 62.56 | 8.27 | 54.88 |
| 2.06 | PMQ | 5.41 | 63.25 | 8.42 | 54.79 |
| 2.06 | **QESC** | **5.09** | **66.31** | **7.99** | **57.05** |
| 3.03 | GPTQ | 4.16 | 68.92 | 6.82 | 59.33 |
| 3.03 | **QESC** | **4.14** | **72.21** | **6.71** | **61.22** |

The 3.03-bit QESC experiences an accuracy loss of only 0.43% (72.21 vs 72.64) on Mixtral-8x7B, and only 0.16% on DeepSeek.

### Pruning Performance Comparison

| Method | Mixtral Acc↑ | Speedup↑ | Phi3.5 Acc↑ | Speedup↑ |
|------|-------------|---------|-----------|---------|
| Baseline | 72.64 | 1.00× | 69.62 | 1.00× |
| EES | 71.40 | 1.06× | 67.96 | 1.05× |
| ODP | 71.98 | 1.05× | 68.92 | 1.04× |
| **PESF (α=0.3)** | **72.19** | **1.08×** | **69.27** | **1.12×** |
| **PESF (α=0.7)** | 58.22 | 1.13× | 67.95 | 1.30× |

### Ablation Study: Expert-Shift Validation

| Configuration | Quantization | Expert-Shift | Mixtral PPL |
|------|-----|-------------|------------|
| Original Model | ✘ | ✘ | 3.84 |
| Shift Only | ✘ | ✔ | 4.17 |
| Quantization Only | ✔ | ✘ | 4.21 |
| Quantization + Shift | ✔ | ✔ | 4.65 |

The increase in PPL caused solely by expert-shift (0.33) is almost comparable to the quantization error itself (0.37), validating the importance of calibrating expert selection.

### Key Findings

- Expert-shift is one of the core factors causing performance degradation in MoE quantization, with an impact comparable to that of weight quantization error itself.
- QESC consistently outperforms GPTQ, PMQ, and BSP across all bit configurations on all 4 models.
- The pruning threshold $\alpha=0.3$ of PESF represents a conservative sweet spot (accuracy loss <0.5%), while $\alpha=0.7$ is an aggressive sweet spot (>1.3× speedup).
- The finding that expert selection frequencies within the same task category are highly similar provides a theoretical foundation for task-adaptive dynamic pruning.
- Combining QESC (3.03-bit) + PESF (α=0.3) compresses Mixtral-8x7B from 93GB to approximately 19GB, enabling single-card RTX 3090 deployment.

## Highlights & Insights

- The concept and validation of expert-shift are clear and logical, offering a fresh perspective on MoE quantization.
- The TopK-MSE loss is elegantly designed, resolving the issue where a large number of low-probability experts dominate the traditional MSE loss.
- QESC and PESF are orthogonal and complementary, allowing them to be used independently or combined.
- The cross-task expert preference analysis provides solid empirical support for dynamic pruning strategies.

## Limitations & Future Work

- Aggressive pruning ($\alpha=0.7$) leads to a significant drop in accuracy on Mixtral-8x7B (from 72.64 to 58.22), showing that redundancy is limited when there are only 8 experts per layer.
- PESF requires collecting expert selection statistics early in inference, which may not be applicable to extremely short sequences (e.g., single-token generation).
- Router calibration relies on the specific calibration dataset WikiText2, and cross-domain generalization capability has not been fully verified.
- Comparisons with more advanced quantization methods such as QuIP and AQLM are missing.

## Related Work & Insights

- **MoE Quantization**: PMQ and BSP (Li et al., 2024a) implement mixed-precision quantization based on expert frequency, but suffer from limited generalization.
- **MoE Pruning**: EES (Lu et al., 2024) skips low-scoring experts from the token dimension, which yields limited speedup.
- **Dense LLM Quantization**: Methods like GPTQ (Frantar et al., 2022) and SmoothQuant (Xiao et al., 2023) do not take MoE characteristics into account.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)

</div>

<!-- RELATED:END -->
