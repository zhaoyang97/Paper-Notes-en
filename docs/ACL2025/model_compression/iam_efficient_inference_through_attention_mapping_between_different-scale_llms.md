---
title: >-
  [Paper Note] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs
description: >-
  [ACL 2025][Model Compression] Having discovered high similarity between the attention matrices of LLMs of different scales, this work proposes the IAM framework. During the prefill stage, IAM establishes a cosine-similarity mapping between the attention heads of a small language model (SLM) and those of a large language model (LLM). During the decode stage, it replaces the attention computation of the LLM's mapped layers with the attention matrices of the SLM. This achieves a…
tags:
  - "ACL 2025"
  - "Model Compression"
date: 2026-05-08
content_hash: 8c2692e4d2c44e05
---

# IAM: Efficient Inference through Attention Mapping between Different-scale LLMs

**Conference**: ACL 2025  
**arXiv**: [2507.11953](https://arxiv.org/abs/2507.11953)  
**Code**: None  
**Area**: Model Compression  

## TL;DR

Having discovered high similarity between the attention matrices of LLMs of different scales, this work proposes the IAM framework. During the prefill stage, IAM establishes a cosine-similarity mapping between the attention heads of a small language model (SLM) and those of a large language model (LLM). During the decode stage, it replaces the attention computation of the LLM's mapped layers with the attention matrices of the SLM. This achieves a 22% reduction in KV cache and an 11% inference speedup, while remaining orthogonal to existing KV cache compression methods.

## Background & Motivation

1. **Surging resource consumption in long-context inference**: Long chains of thought in reasoning models (such as DeepSeek-R1) and RAG scenarios make KV cache the primary bottleneck in long-context inference.
2. **Existing methods only exploit internal sparsity**: KV cache eviction methods (such as H₂O and StreamingLLM) only leverage the sparsity of the model's own attention for token-level pruning, failing to exploit external information for optimization.
3. **Similar attention patterns across different model scales**: Studies from the BERT era have found that large and small models share similar attention patterns. The authors further verify that this characteristic also holds true for decoder-only LLMs.
4. **A new optimization perspective leveraging similarity**: If the attention matrices of a small model can replace certain layers of a large model, both computation and KV cache storage can be saved simultaneously.

## Method

### Overall Architecture

IAM establishes mapping relationships during the **prefill stage** and utilizes these mappings to perform efficient inference during the **decode stage**.

### 1. Similarity Measurement Selection

The authors compared measurement methods including cosine similarity, Minkowski distance, and Pearson correlation coefficient. The attention matrices of the SLM and LLM are flattened into vectors to compute similarity.

Experimental results: **Cosine similarity** performs the best (Qwen2-7B $\rightarrow$ 72B: log PPL 2.414 vs. original 2.136), while Pearson is unsuitable (PPL 5.376).

### 2. Mapping Layer Selection Strategy

The 80 layers of Qwen2-72B are divided into 10 sub-blocks (8 layers per block) to test the performance on MMLU after mapping block by block:

- **Optimal mapping regions**: The last 2 sub-blocks (layers 64-80) and the 2nd to 4th sub-blocks (layers 16-40).
- **Forbidden mapping regions**: The first 2 sub-blocks (layers 0-16), where mapping leads to collapsed performance.
- Mapping performance across different tasks is highly consistent (Pearson correlation of 0.74-0.95).

**Mapping strategy**: Based on the user-specified mapping ratio, the algorithm prioritizes mapping backward from the last layer down to layer 64. If more layers are needed, it translates forward starting from layer 16.

### 3. Mapping Consistency Verification

The mapping relationship established during the prefill stage remains highly consistent during the decode stage (consistency rate $>90\%$), proving that the mapping can be constructed once and reused throughout the entire process.

### 4. Mapping and Inference

- **Prefill**: The SLM and LLM process the prompt simultaneously. Paired cosine similarities between all attention matrices are calculated to find the most similar attention head in the SLM for each head in each mapping layer of the LLM.
- **Decode**: The mapped layers directly utilize the SLM's attention matrices, bypassing the Q/K projection and QK matrix multiplication, and do not store the K cache.
- **Fine-tuning**: Instruction tuning is performed on the SLM using the Alpaca dataset to adapt to the mapping scenario.
- **Threshold Mechanism**: When the prompt is too short ($<\tau_e$), the system generates normally until the threshold is reached before establishing the mapping; when it is too long ($>\tau_t$), it truncates the input to calculate similarity.

## Key Experimental Results

### Performance Retention (Qwen2-72B, mapping ratio 0-50%)

Under a 30% mapping ratio, performance on each task is nearly lossless:
- MMLU: ~81% (original ~82%)
- WikiText PPL: ~2.4 (original 2.14)
- HotpotQA F1: ~38 (original ~40)

It still maintains a high level of performance even under a 50% mapping ratio.

### Efficiency Evaluation (Qwen2-72B + Qwen2-0.5B, 50% mapping)

| Scenario | KV Cache Reduction | TPOT Speedup | TTFT Speedup | Throughput Gain |
|:---|:---:|:---:|:---:|:---:|
| Multi-user concurrency (bs=64, 512+512) | 21.7% | 1.11× | 1.12× | 1.11× |
| Long context (bs=4, 8192+512) | 22.5% | 1.10× | 1.17× | 1.10× |

The TTFT improvement is more significant in the long-context scenario (17%), as the complexity of QK matrix multiplication grows quadratically with sequence length.

### Cross-Series Generalization

LLaMA 3.2-1B $\rightarrow$ LLaMA 3.1-70B is also effective, though the optimal mapping region is located only at the end of the model (differing from Qwen's two-stage pattern).

### Compatibility with KV Cache Compression Methods

When combining IAM + H₂O (80% budget), there is almost no additional degradation in PPL, showing that the two are orthogonal and complementary.

## Highlights & Insights

- **A new perspective leveraging external information**: Systematically utilizes the attention similarity between large and small models for inference acceleration for the first time, distinguishing itself from all existing internal sparsity methods.
- **Dual benefits**: Simultaneously reduces computational overhead (bypassing QK computation) and KV cache (avoiding K cache storage), benefiting both the prefill and decode stages.
- **Orthogonal to existing methods**: Can be stacked with token-level KV cache methods such as H₂O and StreamingLLM.
- **High mapping consistency**: The mapping established in the prefill stage remains stable throughout the entire decode phase, requiring no dynamic updates.

## Limitations & Future Work

- **Requirement of small models from the same series**: The SLM and LLM must belong to the same series (e.g., both Qwen2 or both LLaMA3); cross-series effectiveness remains unknown.
- **Incompatibility with FlashAttention**: IAM requires explicit computation of attention scores, making it incompatible with FlashAttention's fused kernels, which is a major constraint in actual deployment.
- **Coarse grain size at the layer level**: The current mapping is conducted on a per-layer basis; refining it to the attention-head level could yield further improvements.
- **Not entirely lossless**: Notable performance degradation still exists across tasks under a 50% mapping ratio.

## Related Work & Insights

| Dimension | IAM | H₂O | StreamingLLM | KVQuant |
|:---|:---|:---|:---|:---|
| Optimization Granularity | Layer-level | Token-level | Token-level | Numerical Precision |
| Information Source | External (small model) | Internal (attention scores) | Internal (positions) | Internal (numerical distribution) |
| KV Cache Saving | ~22% (layer-level skip) | ~20-40% (token pruning) | Fixed window | 2-4× (quantization) |
| Computation Saving | Yes (bypassing QK) | No | No | Yes (low precision) |
| Orthogonality | Orthogonal to all above | Orthogonal to IAM | Orthogonal to IAM | Orthogonal to IAM |

## Rating

- ⭐⭐⭐⭐ **Novelty**: Utilizing the attention similarity between large and small models for inference optimization builds a brand-new perspective, distinguishing itself from all existing KV cache methods.
- ⭐⭐⭐⭐ **Practicality**: It is orthogonal to existing methods and can be used in combination, though incompatibility with FlashAttention is a critical flaw for deployment.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Evaluation across 4 scenarios + cross-series validation + compatibility validation + efficiency analysis, providing a comprehensive assessment.
- ⭐⭐⭐⭐ **Writing Quality**: The systematic three-step analysis (metrics $\rightarrow$ layer selection $\rightarrow$ consistency) is logically clear and supported by rich charts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[ACL 2025\] One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

</div>

<!-- RELATED:END -->
