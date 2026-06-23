---
title: >-
  [Paper Note] Quantize What Counts: More for Keys, Less for Values
description: >-
  [ACL 2026][Model Compression][Paper Note] This paper theoretically demonstrates from a linear algebra perspective that the spectral and Frobenius norms of Key weights in Transformers are systematically larger than those of Value weights. Based on this, it proposes a Key-priority mixed-precision KV cache quantization strategy (e.g., K4V2), which reduces memory
tags:
  - ACL 2026
  - Model Compression
date: 2026-05-08
content_hash: 53ace721d88dac5e
---
# Quantize What Counts: More for Keys, Less for Values

**Conference**: ACL 2026  
**arXiv**: [2502.15075](https://arxiv.org/abs/2502.15075)  
**Code**: [https://github.com/mohsenhariri/spectral-kv](https://github.com/mohsenhariri/spectral-kv)  
**Area**: Model Compression  
**Keywords**: KV Cache Quantization, Mixed Precision, Spectral Norm, Key-Value Asymmetry, LLM Inference Optimization

## TL;DR
This paper theoretically demonstrates from a linear algebra perspective that the spectral and Frobenius norms of Key weights in Transformers are systematically larger than those of Value weights. Based on this, it proposes a Key-priority mixed-precision KV cache quantization strategy (e.g., K4V2), which reduces memory by 25% while maintaining 98.3% of full-precision accuracy.

## Background & Motivation
**Background**: The KV cache is a primary memory bottleneck during LLM inference. As context lengths (reaching up to 10 million tokens) and model scales grow, KV cache quantization has become a critical requirement.

**Limitations of Prior Work**: Existing KV quantization methods either assign fixed precision to both Key and Value (ignoring their differences) or rely on heuristic grid searches for parameter tuning (lacking a theoretical foundation and failing to generalize).

**Key Challenge**: Keys and Values play fundamentally different roles in the attention mechanism; however, existing methods lack theoretical guidance on how to allocate quantization bits asymmetrically.

**Goal**: To establish a theoretically grounded KV mixed-precision quantization strategy based on the intrinsic geometric properties of model weights.

**Key Insight**: By analyzing the differences in spectral and Frobenius norms of Key/Value projection weight matrices, the relationship between quantization error and norms is derived.

**Core Idea**: Since Key weights are involved in both attention map computation and cache storage, they accumulate larger norms during training, making them more sensitive to quantization errors. Therefore, higher precision should be prioritized for Keys.

## Method

### Overall Architecture
The paper follows a purely theoretical trajectory: "proving the norm gap $\rightarrow$ deriving bit allocation $\rightarrow$ demonstrating orthogonality," without introducing any training or inference overhead. The first step proves the **Key-Value Norm Inequality Theorem**: the Frobenius norm of Key projection weights after training is systematically larger than that of the Value weights ($\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$). The second step substitutes this gap into the error bounds of uniform quantization to derive the **Key-priority Quantization Theorem**: under a fixed bit budget, assigning high precision to Keys and low precision to Values (e.g., K4V2) strictly minimizes the total quantization error. The third step demonstrates that this mixed-precision allocation is **orthogonal** to rotation-based outlier methods like QuaRot and can be integrated into existing quantization frameworks as a plug-and-play component. The entire strategy requires only a one-time analysis of model weight norms and has been validated across 11 models, 6 datasets, and 2 quantization backends.

### Key Designs

**1. Key-Value Norm Inequality Theorem (Theorem 3.1): Key weights are systematically "heavier" than Values after training**

Previous methods relied on grid searches for bit allocation because the fundamental difference between Keys and Values was unclear. This paper establishes it as a provable proposition: $\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$ holds generally after training. The proof tracks SGD gradient updates starting from Xavier initialization—the key lies in $W^K$ having a dual role: shaping the attention map and determining cache content. Consequently, the growth of $W^Q$ during training amplifies the backpropagated gradient signal to $W^K$ via the chain rule, causing its norm to accumulate; whereas $W^V$ only affects the post-attention representation and lacks this multiplicative amplification. This asymmetric gradient path provides a geometric basis for the idea that "Keys are more information-dense and deserve higher precision."

**2. Key-priority Quantization Theorem (Theorem 3.2): Prioritizing high precision for Keys strictly minimizes total error under a fixed budget**

With the norm gap established, bit allocation can be directly calculated. The expected MSE for uniform scalar quantization at $b$ bits is $\Theta(\|M\|_F^2 \cdot 2^{-2b})$, meaning the larger the norm of a matrix, the more sensitive it is to quantization precision. Applying this to the fact that $\|K\|_F \gg \|V\|_F$, the quantization error of the Key dominates the total error under equal precision allocation (e.g., K2V2). Conversely, reallocating bits saved from the Value to the Key (e.g., K4V2) strictly lowers the error upper bound. This step upgrades "giving Keys more bits" from an empirical intuition to a provably optimal geometric-driven design principle under fixed memory budgets.

**3. Orthogonal Combination with Rotation Methods: Mixed precision can be stacked directly on outlier redistribution methods like QuaRot**

A practical concern is whether this bit allocation conflicts with existing KV quantization techniques. The paper proves it is orthogonal to rotation-based methods—QuaRot applies Hadamard rotations to activations to disperse outliers (addressing "value distribution"), while bit allocation addresses "how many bits for whom." These operate on different dimensions. Experiments explore the design space across bit-width, group size, and rotation strategy, verifying that mixed precision serves as a plug-and-play component for existing frameworks rather than a replacement.

### Loss & Training
The method is pure Post-Training Quantization (PTQ) and requires no additional training. Quantization error analysis is based on theoretical bounds for uniform scalar quantization. Experiments utilize two quantization backends: Optimum Quanto (token-wise) and HQQ (channel-wise).

## Key Experimental Results

### Main Results (Downstream Accuracy on GSM8K, Optimum Quanto)

| Model | K2V2 | K2V4 | **K4V2** | K4V4 |
|------|------|------|----------|------|
| Llama-3.2-1B (1-shot) | 0.033 | 0.035 | **0.338** | 0.357 |
| Llama-3.1-8B (1-shot) | 0.511 | 0.547 | **0.752** | 0.754 |
| Phi-4-14B (1-shot) | 0.759 | 0.783 | **0.913** | 0.923 |
| DeepSeek-R1Q-14B (1-shot) | 0.772 | 0.775 | **0.865** | 0.867 |

### Key Reconstruction Error Comparison (MMLU, 2-bit, MSE ↓)

| Model | $K_2$ Error | $V_2$ Error | K/V Error Ratio |
|------|---------|---------|-----------|
| Llama-3.2-1B | 4.851 | 0.127 | 38.2× |
| Llama-3.1-8B | 6.003 | 0.187 | 32.1× |
| Llama-3.3-70B | 4.883 | 0.112 | 43.6× |
| Phi-4-14B | 5.929 | 0.657 | 9.0× |
| Mistral-0.3-7B | 4.718 | 0.398 | 11.9× |

### Key Findings
- K4V2 recovers ~98.3% of the full-precision K4V4 accuracy baseline on 1-shot GSM8K while reducing KV cache memory by 25%.
- K4V2 outperforms K2V4 by 30 percentage points on Llama-3.2-1B and 16 percentage points on Phi-4-14B.
- The quantization reconstruction error of the Key cache at the same bit-width is 9-44 times that of the Value, validating the Norm Inequality Theorem.
- Combining K4V2 with QuaRot's Key-only rotation exceeds the K4V4 baseline by 4.4-18% on CoQA/GSM8K/EQ-Bench.

## Highlights & Insights
- Provides the first rigorous theoretical foundation (two theorems) for asymmetric KV quantization, elevating ad hoc tuning to a geometric-driven principle.
- Highly lightweight: Requires only a one-time analysis of model weight norms, with no inference-time introspection or extra training.
- The Key-priority strategy is consistently effective across five model families (Llama, Phi, Mistral, Qwen, DeepSeek), demonstrating strong generalization.
- Proves orthogonality with rotation methods, serving as a plug-and-play module for existing quantization frameworks.

## Limitations & Future Work
- Theoretical analysis is based on SGD + Xavier initialization assumptions; rigorous derivation for optimizers like AdamW is not yet complete.
- Experiments do not cover extreme long-context scenarios (e.g., 100K+ tokens).
- Currently, only 2/4-bit combinations are validated; finer-grained mixed precision (e.g., 3-bit Key + 1.5-bit Value) remains to be explored.
- Applicability to MoE architectures (e.g., Mixtral) has not been discussed.

## Related Work & Insights
- **KIVI / FlashDecoding**: Fixed-precision KV quantization methods; the mixed-precision strategy of this paper can be stacked with them.
- **KVTuner / SKVQ / QAQ**: Observed that Keys should be assigned more bits but lacked theoretical explanation.
- **QuaRot**: Rotation-based outlier redistribution method, which is orthogonally complementary to the bit allocation strategy here.
- Insight: In LLM inference optimization, understanding the intrinsic geometric structure of the model is more valuable than pure empirical tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for KV quantization from a spectral analysis perspective; the viewpoint is novel and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 11 models, 6 datasets, and 2 quantization backends with comprehensive three-dimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ Provides a directly applicable quantization strategy with theoretical universality.

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
