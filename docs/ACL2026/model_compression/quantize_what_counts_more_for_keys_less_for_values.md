---
title: >-
  [Paper Note] Quantize What Counts: More for Keys, Less for Values
description: >-
  [ACL 2026][Model Compression][Paper Note] From a linear algebra perspective, this paper proves that the spectral and Frobenius norms of Key weights in Transformers are systematically larger than those of Value weights. Based on this, it proposes a Key-prioritized mixed-precision KV cache quantization strategy (e.g., K4V2), reducing memory by 25% while maintain
tags:
  - ACL 2026
  - Model Compression
date: 2026-05-08
content_hash: b1010d3a82d80307
---
<!-- Automatically generated from src/gen_stubs.py -->
# Quantize What Counts: More for Keys, Less for Values

**Conference**: ACL 2026  
**arXiv**: [2502.15075](https://arxiv.org/abs/2502.15075)  
**Code**: [https://github.com/mohsenhariri/spectral-kv](https://github.com/mohsenhariri/spectral-kv)  
**Area**: Model Compression  
**Keywords**: KV Cache Quantization, Mixed Precision, Spectral Norm, Key-Value Asymmetry, LLM Inference Optimization

## TL;DR
From a linear algebra perspective, this paper proves that the spectral and Frobenius norms of Key weights in Transformers are systematically larger than those of Value weights. Based on this, it proposes a Key-prioritized mixed-precision KV cache quantization strategy (e.g., K4V2), reducing memory by 25% while maintaining 98.3% of full-precision accuracy.

## Background & Motivation
**Background**: KV cache is the primary memory bottleneck during LLM inference. As context lengths (reaching 10 million tokens) and model scales grow, KV cache quantization has become a critical necessity.

**Limitations of Prior Work**: Existing KV quantization methods either assign fixed precision to Key and Value (ignoring their differences) or tune parameters through heuristic grid searches (lacking theoretical foundation and generalizability).

**Key Challenge**: Key and Value play fundamentally different roles in the attention mechanism, yet existing methods lack theoretical guidance to determine how to allocate quantization bits asymmetrically.

**Goal**: Establish a theoretically grounded KV mixed-precision quantization strategy based on the intrinsic geometric properties of model weights.

**Key Insight**: Analyze the differences in spectral and Frobenius norms of Key/Value projection weight matrices and derive the relationship between quantization error and these norms.

**Core Idea**: Key weights accumulate larger norms during training because they participate in both attention map computation and cache storage, making them more sensitive to quantization errors—thus, higher precision should be prioritized for Keys.

## Method

### Overall Architecture
This paper follows a purely theoretical path: "prove norm disparity, derive bit allocation, and demonstrate stackability," introducing no overhead during training or inference. The first step proves the **Key-Value Norm Inequality Theorem**: the Frobenius norm of Key projection weights after training is systematically larger than that of Value weights ($\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$). The second step substitutes this disparity into the error bounds of uniform quantization to derive the **Key-Priority Quantization Theorem**: under a fixed bit budget, assigning higher precision to Keys and lower precision to Values (e.g., K4V2) strictly minimizes total quantization error. The third step demonstrates that this mixed-precision allocation is **orthogonal** to rotation-based outlier methods like QuaRot and can be integrated into existing quantization frameworks as a plug-and-play component. This strategy requires only a one-time analysis of weight norms and is validated across 11 models, 6 datasets, and 2 quantization backends.

### Key Designs

**1. Key-Value Norm Inequality Theorem (Theorem 3.1): Key weights are systematically "heavier" than Value weights after training.**

Previous methods relied on grid searches for bit allocation because the difference between Key and Value remained unclear. This paper establishes it as a provable proposition: $\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$ holds generally after training. The proof starts from Xavier initialization and tracks SGD gradient updates—the key lies in $W^K$ serving a dual role: shaping attention maps and determining cache content. Consequently, the growth of $W^Q$ during training amplifies the backpropagated gradient signal to $W^K$ through the chain rule, causing its norm to accumulate; whereas $W^V$ only affects post-attention representations and lacks this multiplicative amplification. This asymmetric gradient path provides a geometric basis—rather than just empirical observation—for "Keys being more information-dense and deserving of precision preservation."

**2. Key-Priority Quantization Theorem (Theorem 3.2): Prioritizing Key precision strictly minimizes total quantization error under a fixed bit budget.**

With the norm disparity established, bit allocation can be calculated directly. The expected MSE for uniform scalar quantization at $b$ bits is $\Theta(\|M\|_F^2 \cdot 2^{-2b})$, meaning larger matrix norms lead to higher sensitivity to quantization. Applying the fact that $\|K\|_F \gg \|V\|_F$, Key quantization error dominates total error under equal-precision allocation (e.g., K2V2). Conversely, reallocating bits saved from Value to Key (e.g., K4V2) strictly reduces the error upper bound. This step elevates "allocating more bits to Key" from empirical intuition to a provably optimal, geometry-driven design principle under fixed memory budgets.

**3. Orthogonal Combination with Rotation Methods: Mixed precision can be stacked directly on outlier redistribution methods like QuaRot.**

In practice, a concern is whether this bit allocation conflicts with existing KV quantization techniques. This paper proves it is orthogonal to rotation methods—QuaRot applies Hadamard rotations to activations to disperse outliers (addressing "value distribution"), while bit allocation addresses "how many bits for whom." Since they operate on different dimensions, experiments sweep bit width × group size × rotation strategy as a three-dimensional design space, verifying that mixed precision works as a plug-and-play component.

### Loss & Training
This method is purely Post-Training Quantization (PTQ) and requires no additional training. Quantization error analysis is based on theoretical bounds of uniform scalar quantization. Experiments utilize two quantization backends: Optimum Quanto (token-wise) and HQQ (channel-wise).

## Key Experimental Results

### Main Results (Downstream Accuracy on GSM8K, Optimum Quanto)

| Model | K2V2 | K2V4 | **K4V2** | K4V4 |
|------|------|------|----------|------|
| Llama-3.2-1B (1-shot) | 0.033 | 0.035 | **0.338** | 0.357 |
| Llama-3.1-8B (1-shot) | 0.511 | 0.547 | **0.752** | 0.754 |
| Phi-4-14B (1-shot) | 0.759 | 0.783 | **0.913** | 0.923 |
| DeepSeek-R1Q-14B (1-shot) | 0.772 | 0.775 | **0.865** | 0.867 |

### Key Experimental Results (MMLU, 2-bit, MSE ↓)

| Model | K₂ Error | V₂ Error | K/V Error Ratio |
|------|---------|---------|-----------|
| Llama-3.2-1B | 4.851 | 0.127 | 38.2× |
| Llama-3.1-8B | 6.003 | 0.187 | 32.1× |
| Llama-3.3-70B | 4.883 | 0.112 | 43.6× |
| Phi-4-14B | 5.929 | 0.657 | 9.0× |
| Mistral-0.3-7B | 4.718 | 0.398 | 11.9× |

### Key Findings
- K4V2 recovers approximately 98.3% of the full-precision K4V4 baseline accuracy on 1-shot GSM8K while reducing KV cache memory by 25%.
- K4V2 outperforms K2V4 by 30 percentage points on Llama-3.2-1B and by 16 percentage points on Phi-4-14B.
- Quantization reconstruction error for Key cache at the same bit width is 9-44x higher than that of Value, validating the Norm Inequality Theorem.
- Combining K4V2 with QuaRot's Key-only rotation surpasses the K4V4 baseline by 4.4-18% on CoQA/GSM8K/EQ-Bench.

## Highlights & Insights
- Provides the first rigorous theoretical foundation (two theorems) for KV asymmetric quantization, transforming ad hoc tuning into geometry-driven principles.
- Extremely lightweight: requires only one-time analysis of weight norms without inference-time introspection or additional training.
- Key-priority strategy is consistently effective across five model families (Llama/Phi/Mistral/Qwen/DeepSeek), showing strong generalizability.
- Proved orthogonality with rotation methods, serving as a plug-and-play module for existing quantization frameworks.

## Limitations & Future Work
- Theoretical analysis is based on SGD + Xavier initialization assumptions; rigorous derivation for optimizers like AdamW is not yet complete.
- Experiments do not cover extreme long-context scenarios (e.g., 100K+ tokens).
- Currently only validates 2/4-bit combinations; finer-grained mixed precision (e.g., 3-bit Key + 1.5-bit Value) remains to be explored.
- Applicability to MoE architectures (e.g., Mixtral) has not been discussed.

## Related Work & Insights
- **KIVI / FlashDecoding**: Fixed-precision KV quantization methods; the mixed-precision strategy can be stacked on them.
- **KVTuner / SKVQ / QAQ**: Observed that Key should be allocated more bits, but all lack theoretical explanation.
- **QuaRot**: Rotation-based outlier redistribution method, which is orthogonal and complementary to the bit allocation strategy.
- Insight: In LLM inference optimization, understanding the intrinsic geometric structure of the model is more valuable than pure empirical tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for KV quantization from a spectral analysis perspective; the viewpoint is novel and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 11 models, 6 datasets, and 2 quantization backends with comprehensive 3D ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ Provides a directly implementable quantization strategy with theoretical universality.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](../../NeurIPS2025/model_compression/homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)

</div>

<!-- RELATED:END -->
