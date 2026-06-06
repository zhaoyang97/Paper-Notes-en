---
title: >-
  [Paper Note] Quantize What Counts: More for Keys, Less for Values
description: >-
  [ACL 2026][Model Compression][KV cache quantization] This paper proves from a linear algebra perspective that the spectral and Frobenius norms of Key weights in Transformers are systematically larger than those of Value…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "KV cache quantization"
  - "mixed precision"
  - "spectral norm"
  - "Key-Value asymmetry"
  - "LLM inference optimization"
date: 2026-05-08
content_hash: 4c48b653b1a2ba43
---

# Quantize What Counts: More for Keys, Less for Values

**Conference**: ACL 2026  
**arXiv**: [2502.15075](https://arxiv.org/abs/2502.15075)  
**Code**: [https://github.com/mohsenhariri/spectral-kv](https://github.com/mohsenhariri/spectral-kv)  
**Area**: model_compression  
**Keywords**: KV cache quantization, mixed precision, spectral norm, Key-Value asymmetry, LLM inference optimization

## TL;DR
This paper proves from a linear algebra perspective that the spectral and Frobenius norms of Key weights in Transformers are systematically larger than those of Value weights. Based on this, it proposes a Key-first mixed-precision KV cache quantization strategy (e.g., K4V2), which reduces memory by 25% while maintaining 98.3% of full-precision accuracy.

## Background & Motivation
**Background**: In LLM inference, the KV cache is the primary memory bottleneck. As context lengths (reaching up to 10 million tokens) and model scales grow, KV cache quantization has become a critical necessity.

**Limitations of Prior Work**: Existing KV quantization methods either allocate identical precision for both Keys and Values (ignoring their differences) or rely on heuristic grid searches for parameter tuning (lacking theoretical foundation and generalizability).

**Key Challenge**: Keys and Values play fundamentally different roles in the attention mechanism, yet existing methods lack theoretical guidance to determine how to allocate quantization bits asymmetrically.

**Goal**: Establish a theoretically grounded KV mixed-precision quantization strategy based on the intrinsic geometric properties of model weights.

**Key Insight**: Analyze the differences in spectral and Frobenius norms of Key/Value projection weight matrices and derive the relationship between quantization error and these norms.

**Core Idea**: Key weights accumulate larger norms during training because they participate in both the computation of attention maps and the storage of cache content. This makes them more sensitive to quantization errors; thus, higher precision should be prioritized for Keys.

## Method

### Overall Architecture
The approach consists of theoretical derivation and practical validation: (1) Proving the Key-Value Norm Inequality Theorem (Key weight norms are systematically larger than Value norms); (2) Proving the Key-Priority Quantization Theorem (under a fixed memory budget, a high-precision Key + low-precision Value allocation strictly reduces quantization error); (3) Validating through experiments across multiple models, tasks, and quantization backends.

### Key Designs
1.  **Key-Value Norm Inequality Theorem (Theorem 3.1)**:

    - **Function**: Proves that $\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$ generally holds after training.
    - **Mechanism**: Starting from Xavier initialization, the theorem tracks gradient updates during SGD training. $W^K$ simultaneously shapes the attention map and determines cache content; the growth of $W^Q$ amplifies the gradient signal backpropagated to $W^K$. In contrast, $W^V$ only affects the post-attention representation and lacks this multiplicative amplification.
    - **Design Motivation**: Establish a theoretical foundation for Keys being "information-dense" to provide a basis for asymmetric quantization.

2.  **Key-Priority Quantization Theorem (Theorem 3.2)**:

    - **Function**: Proves that for a fixed total bit budget, allocating high precision to Keys and low precision to Values strictly minimizes the quantization error.
    - **Mechanism**: The expected MSE of uniform scalar quantization is $\Theta(\|M\|_F^2 \cdot 2^{-2b})$, meaning larger norms are more sensitive to bit width. Since $\|K\|_F \gg \|V\|_F$, the quantization error of the Key dominates the total error under equal-precision allocation.
    - **Design Motivation**: Elevate bit allocation from empirical tuning to a theoretically guaranteed, geometry-driven design principle.

3.  **Orthogonal Combination with Rotation Methods**:

    - **Function**: Proves that the mixed-precision strategy can be stacked with outlier redistribution methods like QuaRot.
    - **Mechanism**: QuaRot applies Hadamard rotations to activations to disperse outliers, which is orthogonal to bit allocation strategies. Experiments explore a three-dimensional design space: bit width × group size × rotation strategy.
    - **Design Motivation**: Ensure the method acts as a plug-and-play component for existing KV quantization frameworks.

### Loss & Training
This method is a pure Post-Training Quantization (PTQ) approach and requires no additional training. The quantization error analysis is based on the theoretical bounds of uniform scalar quantization. Experiments utilize two quantization backends: Optimum Quanto (token-wise) and HQQ (channel-wise).

## Key Experimental Results

### Main Results (Downstream Accuracy on GSM8K, Optimum Quanto)
| Model | K2V2 | K2V4 | **K4V2** | K4V4 |
|------|------|------|----------|------|
| Llama-3.2-1B (1-shot) | 0.033 | 0.035 | **0.338** | 0.357 |
| Llama-3.1-8B (1-shot) | 0.511 | 0.547 | **0.752** | 0.754 |
| Phi-4-14B (1-shot) | 0.759 | 0.783 | **0.913** | 0.923 |
| DeepSeek-R1Q-14B (1-shot) | 0.772 | 0.775 | **0.865** | 0.867 |

### Quantization Error Comparison (MMLU, 2-bit, MSE ↓)
| Model | K₂ Error | V₂ Error | K/V Error Ratio |
|------|---------|---------|-----------|
| Llama-3.2-1B | 4.851 | 0.127 | 38.2× |
| Llama-3.1-8B | 6.003 | 0.187 | 32.1× |
| Llama-3.3-70B | 4.883 | 0.112 | 43.6× |
| Phi-4-14B | 5.929 | 0.657 | 9.0× |
| Mistral-0.3-7B | 4.718 | 0.398 | 11.9× |

### Key Findings
- K4V2 recovers approximately 98.3% of the full-precision K4V4 baseline accuracy on 1-shot GSM8K while reducing KV cache memory by 25%.
- K4V2 outperforms K2V4 by 30 percentage points on Llama-3.2-1B and by 16 points on Phi-4-14B.
- Key cache quantization reconstruction error is 9-44x higher than that of the Value at the same bit width, validating the norm inequality theorem.
- Combining K4V2 with QuaRot’s Key-only rotation exceeds the K4V4 baseline by 4.4-18% on CoQA/GSM8K/EQ-Bench.

## Highlights & Insights
- Provides a rigorous theoretical basis (two theorems) for KV asymmetric quantization for the first time, evolving ad hoc tuning into geometry-driven principles.
- Extremely lightweight: Requires only a one-time analysis of model weight norms without inference-time introspection or retraining.
- The Key-priority strategy is consistently effective across five model families (Llama/Phi/Mistral/Qwen/DeepSeek), demonstrating strong generalizability.
- Proven orthogonality with rotation methods allows it to be used as a plug-and-play module for existing quantization frameworks.

## Limitations & Future Work
- Theoretical analysis assumes SGD + Xavier initialization; rigorous derivation for optimizers like AdamW is not yet complete.
- Experiments do not cover more extreme long-context scenarios (e.g., 100K+ tokens).
- Currently, only 2/4 bit combinations have been validated; finer-grained mixed precision (e.g., 3-bit Key + 1.5-bit Value) remains to be explored.
- Applicability to MoE architectures (e.g., Mixtral) was not discussed.

## Related Work & Insights
- **KIVI / FlashDecoding**: Fixed-precision KV quantization methods; the mixed-precision strategy proposed here can be stacked with them.
- **KVTuner / SKVQ / QAQ**: Observed that Keys should be allocated more bits, but lacked theoretical explanation.
- **QuaRot**: An outlier redistribution method using rotation, which is complementary to the bit allocation strategy of this work.
- **Insight**: In LLM inference optimization, understanding the model's intrinsic geometric structure is more valuable than pure empirical parameter tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for KV quantization from a spectral analysis perspective; the viewpoint is novel and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 11 models, 6 datasets, and 2 quantization backends with comprehensive three-dimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ Offers a quantization strategy that can be directly implemented and possesses theoretical universality.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](../../NeurIPS2025/model_compression/homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICML 2026\] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](../../ICML2026/model_compression/projq_project-and-quantize_for_adapter-aware_llm_compression.md)

</div>

<!-- RELATED:END -->
