---
title: >-
  [Paper Note] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference
description: >-
  [NeurIPS 2025][Model Compression][KV cache compression] This paper proposes MUSTAFAR, a framework that systematically demonstrates the superiority of unstructured sparsity for KV cache pruning—achieving 70% sparsity on both Key and Value caches without accuracy degradation—and introduces a bitmap-based sparse format with a custom attention kernel, yielding up to 2.23× end-to-end inference throughput improvement.
tags:
  - NeurIPS 2025
  - Model Compression
  - KV cache compression
  - unstructured sparsity
  - LLM inference
  - attention acceleration
  - sparse format
date: 2026-05-08
content_hash: 551ceb7c6e4888aa
---

# MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

**Conference**: NeurIPS 2025
**arXiv**: [2505.22913](https://arxiv.org/abs/2505.22913)
**Code**: [GitHub](https://github.com/dhjoo98/mustafar)
**Area**: Model Compression
**Keywords**: KV cache compression, unstructured sparsity, LLM inference, attention acceleration, sparse format

## TL;DR

This paper proposes MUSTAFAR, a framework that systematically demonstrates the superiority of unstructured sparsity for KV cache pruning—achieving 70% sparsity on both Key and Value caches without accuracy degradation—and introduces a bitmap-based sparse format with a custom attention kernel, yielding up to 2.23× end-to-end inference throughput improvement.

## Background & Motivation

As large language models process increasingly long sequences, the memory overhead of the KV cache has become a critical bottleneck for scaling context length. Existing compression techniques include quantization, low-rank approximation, token eviction, and structured pruning. The following core issues arise in the pruning dimension:

**Limitations of Structured Pruning**: Prior KV cache pruning work (e.g., ThinK) is confined to structured patterns (channel-wise removal), severely limiting achievable sparsity. ThinK reaches only ~50% structured sparsity on the Key cache and can tolerate only 30% on the Value cache.

**Difficulty of Compressing the Value Cache**: The activation distribution of the Value cache is relatively uniform, lacking prominent channel-level outliers, making structured pruning prone to catastrophic accuracy degradation.

**Computational Challenges of Unstructured Sparsity**: Although unstructured sparsity theoretically preserves more important elements, its irregular patterns are difficult to exploit efficiently on GPUs, and no suitable sparse computation solution existed previously.

The core insight of MUSTAFAR is that abandoning all constraints on sparsity patterns (i.e., adopting unstructured sparsity) enables higher sparsity while maintaining model accuracy. The key lies in addressing two problems: identifying an appropriate pruning strategy and designing computation kernels that efficiently exploit unstructured sparsity.

## Method

### Overall Architecture

MUSTAFAR consists of two components (Figure 1): (1) the green component—a pruning algorithm that explores optimal pruning strategies for Key/Value caches; and (2) the pink component—a custom sparse attention kernel that enables efficient computation over compressed KV caches using a bitmap sparse format.

### Key Designs

1. **Key Cache Pruning Strategy**: The Key cache exhibits pronounced channel-level outliers, where certain channels have values far larger than others. Based on this observation, **per-token magnitude-based pruning** is adopted, which effectively preserves outlier channel elements within each token. The pruning score is formulated as:

$$S = |K| \odot \text{broadcast}\left(\sum_{t=T}^{T+31} |Q_t|\right)$$

The output-aware variant multiplies key elements by the L1-accumulated scores from the subsequent 32 queries. Empirically, however, simple per-token magnitude pruning already slightly outperforms structured pruning and approaches the output-aware method.

2. **Value Cache Pruning Strategy**: The Value cache has a more uniform distribution without channel-level outliers. The authors systematically compare four combinations of (per-channel / per-token) × (magnitude / output-aware). A key finding is that for the Value cache, **per-token magnitude-based pruning is equivalent to per-token output-aware pruning**: since the attention computation $\text{AttentionScore} \times \text{Value}$ multiplies all Value elements of the same token by the same attention score, magnitude inherently reflects contribution to the output. Per-token magnitude-based pruning is thus uniformly adopted for both Key and Value caches.

3. **Bitmap Sparse Format and Custom Attention Kernel**:

    - **Sparse Format**: Extending the bitmap format from Coruscant, the pruned KV cache is partitioned into $1 \times 64$ column tiles, each represented by a 64-bit bitmap indicating non-zero positions, along with a tile offset to locate the starting non-zero element, achieving maximum compression ratio.
    - **SpMV Kernel**: The attention operations in the decode phase (Query × Key$^\top$ and AttentionScore × Value) are inherently memory-bound matrix-vector multiplications. The custom CUDA kernel adopts a "load-as-compressed, compute-as-dense" paradigm—loading compressed data from global memory into registers, decompressing into shared memory, and then performing tiled dense computation.
    - **Hybrid Computation**: Decode-phase attention is restructured into two parts: SpMV over the compressed KV cache plus dense MV over a local window (the most recent 32 tokens), with the two results combined via online softmax.

### Loss & Training

MUSTAFAR is a training-free method. Pruning and compression are performed at runtime: the KV cache generated during the prefill phase is pruned and compressed before decoding begins; KV cache generated during decoding is kept dense within the local window and pruned/compressed upon exiting the window. The method is fully compatible with FlashAttention prefill.

## Key Experimental Results

### Main Results (LongBench)

| Model / Config | K Sparsity | V Sparsity | LongBench Avg. | Baseline |
|----------------|-----------|-----------|----------------|---------|
| Llama-3-8B Dense | 0% | 0% | 43.19 | — |
| ThinK (structured) | 50% | 0% | 38.53 | Structured SOTA |
| **MUSTAFAR** | **50%** | **0%** | **42.84** | +4.31 vs ThinK |
| ThinK (structured) | 70% | 0% | 26.55 | Accuracy collapse |
| **MUSTAFAR** | **70%** | **0%** | **41.55** | +14.98 vs ThinK |
| **MUSTAFAR** | **70%** | **70%** | **40.96** | Still outperforms ThinK@50% |
| Mistral-7B Dense | 0% | 0% | 42.65 | — |
| **MUSTAFAR** | **70%** | **70%** | **40.95** | Only −1.70 drop |

### Ablation Study (Value Cache Pruning Strategy Comparison)

| Pruning Method | V Sparsity=50% | V Sparsity=70% | Note |
|----------------|---------------|---------------|------|
| ThinK structured | 38.45 | 30.60 | Structured performs poorly on V cache |
| Per-channel magnitude | 42.50 | 41.69 | Channel-wise magnitude |
| Per-channel output-aware | 42.84 | 42.67 | Weighted by attention scores |
| **Per-token magnitude** | **43.04** | **42.78** | **Simplest and best; no extra computation** |

### Key Findings

- **Unstructured vs. Structured**: At 70% Key sparsity, unstructured pruning (41.55) outperforms structured ThinK (26.55) by 15 points—the latter having completely collapsed.
- **Simultaneous KV Cache Pruning**: Applying 70% sparsity to both Key and Value caches simultaneously (40.96) still outperforms ThinK with only 50% structured Key sparsity (38.53), a result previously considered unattainable.
- **Compatibility with Orthogonal Methods**: MUSTAFAR can be combined with KIVI quantization (e.g., K70%+V70%+4-bit quantization) as well as H2O token eviction for further compression.
- **End-to-End Speedup**: KV cache is compressed to 45% of the dense size, yielding up to 2.23× throughput improvement. The SpMV kernel's speedup fully offsets the runtime overhead of pruning and compression.

## Highlights & Insights

1. **Breaking the Structured Pruning Mindset**: The KV cache pruning community had implicitly assumed that structured patterns were necessary. This paper convincingly demonstrates the counterintuitive conclusion that "removing constraints leads to better outcomes."
2. **Differentiated Analysis of Key vs. Value Caches**: The Key cache has channel-level outliers (favoring per-token retention of outliers), while the Value cache has a uniform distribution (making per-token magnitude equivalent to output-aware pruning). This nuanced analysis provides important guidance for future work.
3. **System-Level Solution**: Beyond proposing a pruning algorithm, the paper delivers a complete package of format design and kernel implementation, making unstructured sparsity genuinely deployable in practice.

## Limitations & Future Work

- Validation is currently limited to 7B–8B models; experiments on larger-scale models (70B+) are absent.
- The overhead of the bitmap sparse format may not be worthwhile for very short sequences, limiting benefits in short-context scenarios.
- The custom CUDA kernel currently supports only specific GPU architectures (RTX 6000 ADA); porting to other hardware requires additional engineering effort.
- The local window size (32 tokens) is a fixed hyperparameter; adaptive window sizing has not been explored.

## Related Work & Insights

- Compatibility experiments with KIVI (quantization) and H2O (token eviction) suggest that multiple compression techniques can be stacked, providing a foundation for building a "compression stack."
- The bitmap sparse format is adapted from Coruscant, which targets SpMM in LLM weight projection layers; this work extends it to the SpMV setting in attention computation.
- The simplicity of per-token magnitude pruning is noteworthy—the simplest method often proves the most robust, a principle worth validating in other compression contexts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The systematic exploration of unstructured sparsity for KV caches is a first; the combination of bitmap format and custom kernel represents meaningful engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three models (Llama-2/3, Mistral), all LongBench tasks, and compatibility with orthogonal methods, though larger models are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear; Figure 1 provides an effective global overview.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a core bottleneck in LLM inference with a simple, effective, and deployable approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] Inference-Time Hyper-Scaling with KV Cache Compression](inference-time_hyper-scaling_with_kv_cache_compression.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [\[NeurIPS 2025\] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)

</div>

<!-- RELATED:END -->
