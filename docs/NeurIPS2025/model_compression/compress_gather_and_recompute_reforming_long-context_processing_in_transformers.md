---
title: >-
  [Paper Note] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers
description: >-
  [NeurIPS 2025][Model Compression][KV cache compression] This paper proposes REFORM, an inference framework that efficiently processes ultra-long contexts (up to millions of tokens) via a compress–gather–recompute three-stage pipeline. REFORM achieves improvements of 52% and 34% over the strongest baselines on RULER and BABILong respectively, while reducing inference time by 30% and peak memory usage by 5%.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "KV cache compression"
  - "long-context inference"
  - "token retrieval"
  - "early exit"
  - "on-demand recomputation"
date: 2026-05-08
content_hash: 9376b337b692ed20
---

# Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2506.01215](https://arxiv.org/abs/2506.01215)  
**Code**: To be confirmed  
**Area**: Model Compression
**Keywords**: KV cache compression, long-context inference, token retrieval, early exit, on-demand recomputation

## TL;DR

This paper proposes REFORM, an inference framework that efficiently processes ultra-long contexts (up to millions of tokens) via a compress–gather–recompute three-stage pipeline. REFORM achieves improvements of 52% and 34% over the strongest baselines on RULER and BABILong respectively, while reducing inference time by 30% and peak memory usage by 5%.

## Background & Motivation

Large language models are increasingly required to handle ultra-long contexts in real-world scenarios (e.g., lifelong user conversations, repository-level code comprehension, and multi-modal interleaved sequences). However, the quadratic complexity of Transformers and limited pretraining context windows make direct processing of million-token sequences impractical. Existing approaches fall into two categories:

1. **Recurrent compression methods** (StreamingLLM, H2O, TOVA, InfiniPot): process input in chunks iteratively, controlling memory via KV cache eviction or compression. These methods achieve low memory footprints but suffer from information loss during compression, leading to "forgetting" and significantly degraded retrieval accuracy.
2. **Random-access methods** (InfLLM, ReAttention): retain the full KV cache and retrieve relevant portions on demand. While these methods allow flexible access to historical context, they require enormous memory (typically relying on CPU offloading), incur substantial latency, and their flexibility does not necessarily translate into high retrieval performance.

Both categories have fundamental drawbacks, motivating the need for a new approach that balances efficiency with accurate retrieval.

## Core Problem

How can high-accuracy information retrieval and generation be achieved over sequences far exceeding the pretraining window length, under constrained compute and memory budgets? The central tension is that compression saves memory at the cost of information loss, while full caching preserves information at prohibitive resource cost.

## Method

REFORM (**RE**current chunked **F**orwarding with **O**n-demand cache **R**eco**M**putation) adopts a two-phase inference pipeline:

### Phase 1: Recurrent Chunked Forward Pass + Embedding Extraction

1. **Chunked processing**: The long input is split into fixed-size chunks (32k tokens in experiments) and fed into the model sequentially.
2. **Progressive KV cache compression**: After processing each chunk, the KV cache is compressed using H2O's attention-based token eviction strategy, retaining only "heavy hitter" tokens (those with high attention scores). Position IDs are reassigned to be contiguous after eviction, enabling the model to surpass its pretraining window limit.
3. **Cross-layer context embedding construction**: During compression, QKV states are extracted from selected intermediate layers and attention heads, then concatenated to construct lightweight per-token retrieval embeddings. A key finding is that cosine similarity over QKV states outperforms the commonly used attention scores for retrieval, while requiring smaller dimensionality (160 vs. 5120).
4. **Early exit strategy**: Since the most effective retrieval heads are concentrated in the lower-to-middle layers (depth < 70%), the forward pass need not propagate to upper layers. Early exit reduces both computation and memory overhead.

**Embedding head selection**: All attention heads are evaluated on synthetic tasks (pattern matching and multi-hop QA) using the MNR metric. The top 4 heads (2 for pattern matching, 2 for multi-hop QA) are selected. Their embeddings are individually L2-normalized and concatenated, which is equivalent to independently computing cosine similarities per head and averaging.

### Phase 2: On-Demand Cache Recomputation

1. **Relevant token identification**: The stored cross-layer context embeddings are used to compute cosine similarities between query tokens (the final portion of the input) and all historical tokens. Max-pooling over the query dimension yields per-token scores, which are further smoothed via a 129-token window max-pooling to preserve contextual continuity.
2. **Gather**: The highest-scoring tokens are selected (8k for Mistral-Nemo, 16k for other models), with the first and last 256 tokens always retained.
3. **Recompute**: The original embeddings of the selected tokens are passed through the full model in a complete forward pass to reconstruct a high-fidelity KV cache for generation.

The decoupled design—using compression solely for retrieval and recomputation solely for generation—is the core innovation of REFORM. The compression phase does not directly serve generation; instead, it supports lightweight retrieval only. The KV cache required for generation is obtained via full recomputation over carefully selected tokens, thereby avoiding the information loss inherent to compression.

## Key Experimental Results

### Needle-in-a-Haystack

Qwen2.5-7B-Instruct achieves 100% retrieval accuracy across all depths and lengths up to 1M tokens.

### RULER & BABILong (1M tokens, Mistral-Nemo)

| Method | RULER 1M | BABILong 1M |
|--------|----------|-------------|
| InfLLM (strongest baseline) | 23.3 | 9.6 |
| InfiniPot | 12.0 | 8.8 |
| **REFORM** | **75.5** | **48.8** |

REFORM outperforms InfLLM by more than 52 and 34 percentage points, respectively. On Qwen2.5-7B, REFORM achieves 75.1 / 58.8 at 1M tokens, also leading by a large margin.

### ∞-Bench (Mistral-Nemo)

REFORM achieves an average of 50.2%, compared to InfLLM at 37.6% and InfiniPot at 24.0%. On the R.KV subtask, REFORM reaches 88.2% (vs. InfLLM at 1.0%).

### RepoEval Code Completion (Qwen2.5-Coder-1.5B, API-level)

REFORM achieves 65.3% ES, vs. InfLLM at 61.8% and InfiniPot at 59.4%.

### Multi-modal MM-NIAH (Pixtral-12B)

REFORM achieves an average of 57.5%, vs. TOVA at 52.0% and InfiniPot at 53.0%.

### Efficiency Analysis (256k tokens, single H100 GPU)

| Method | Inference Time (s) | Peak Memory (GB) |
|--------|--------------------|------------------|
| InfLLM | 129.14 | 51.62 |
| H2O | 41.33 | 37.85 |
| **REFORM** | **27.24** | **35.00** |

Compared to InfLLM: 80% reduction in time, 32% reduction in memory. Compared to InfiniPot: 33% reduction in time, 5% reduction in memory.

### Comparison with RAG (RULER 300k NIAH)

REFORM outperforms both BM25 and Dense RAG on all four needle task variants; combining REFORM with RAG yields further marginal gains.

## Highlights & Insights

- The **three-stage compress–retrieve–recompute decoupled design** cleverly combines the efficiency of recurrent compression with the precision of random-access methods, avoiding the fundamental deficiencies of each.
- The finding that **cosine similarity over QKV states outperforms attention scores** for retrieval provides more compact and effective embeddings; cross-layer combination further improves performance.
- **Early exit** naturally complements embedding extraction by leveraging the observation that optimal retrieval heads are concentrated in lower-to-middle layers, reducing unnecessary computation.
- **Modality-agnostic**: Operating at the architecture level, REFORM applies directly to multi-modal models (e.g., Pixtral) without modification.
- Experiments are exceptionally comprehensive, covering synthetic retrieval, realistic NLU, code completion, multi-modal tasks, RAG comparison, ablation studies, and efficiency analysis.

## Limitations & Future Work

- Token selection in the Gather phase depends on embedding quality and may miss critical tokens in extreme cases.
- The current implementation requires computing attention scores separately for token eviction (since Flash Attention does not output attention weights), introducing redundant computation; integrating into the Flash Attention kernel could further accelerate the method.
- The compression component directly reuses H2O without specialized optimization for embedding construction, leaving room for improvement.
- Validation on additional modalities such as audio and video remains unexplored.
- Head selection is based on short synthetic data (8k tokens); while experiments demonstrate generalization to million-scale sequences, theoretical guarantees are lacking.

## Related Work & Insights

| Dimension | StreamingLLM / H2O / TOVA / InfiniPot | InfLLM | REFORM |
|-----------|---------------------------------------|--------|--------|
| Mechanism | Recurrent KV cache compression | Full KV cache + CPU offloading random access | Compression for retrieval only + recomputation over selected tokens |
| Information retention | Severe loss due to compression | Fully preserved | Selective high-fidelity recomputation |
| Memory | Low | Very high | Low (early exit + compact embeddings) |
| Latency | Low | High (CPU↔GPU transfer) | Lowest |
| Retrieval accuracy | Low | Moderate | High |
| Requires model modification | No | No | No |

Compared to RAG: REFORM constructs retrieval embeddings conditioned on the full context (avoiding context fragmentation), requires no external retrieval model, and natively supports multi-modal inputs.

**Broader implications:**
- The core insight—"use compression for retrieval, use recomputation for generation"—generalizes to other settings, such as video understanding where lightweight frame embeddings are stored and full decoding is performed on demand.
- The finding that QKV states serve as effective retrieval embeddings can inform future work on retrieval head identification.
- The combined strategy of early exit and retrieval head selection may inspire hierarchical scheduling in speculative decoding.
- REFORM is complementary to post-hoc KV compression methods such as SnapKV and HOMER; its embedding construction can be integrated with more advanced compression strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-stage decoupled design and QKV-based retrieval embeddings represent meaningful novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers synthetic, realistic, code, multi-modal, RAG comparison, ablation, and efficiency analyses comprehensively.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Highly practical, decisively outperforms baselines, and requires no training (plug-and-play).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](../../ICML2025/model_compression/core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[NeurIPS 2025\] FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing](fastlongspeech_enhancing_large_speech-language_models_for_efficient_long-speech_.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)

</div>

<!-- RELATED:END -->
