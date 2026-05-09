---
title: >-
  [Paper Note] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
description: >-
  [NeurIPS 2025 (Oral)][Model Compression][KV Cache Compression] This paper proposes KVzip, a query-agnostic KV cache eviction method that quantifies the importance of each KV pair by leveraging the LLM itself to reconstruct the original context from the cached KV pairs. KVzip achieves 3–4× KV cache compression and approximately 2× reduction in FlashAttention decoding latency, while significantly outperforming existing query-aware methods in multi-query scenarios.
tags:
  - NeurIPS 2025 (Oral)
  - Model Compression
  - KV Cache Compression
  - Query-Agnostic
  - Context Reconstruction
  - Long-Context Inference
  - Cache Reuse
date: 2026-05-08
content_hash: 66f211aeb10d38d4
---

# KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

**Conference**: NeurIPS 2025 (Oral)

**arXiv**: [2505.23416](https://arxiv.org/abs/2505.23416)

**Code**: [Available](https://github.com/snu-mllab/KVzip)

**Area**: Model Compression / LLM Inference Optimization

**Keywords**: KV Cache Compression, Query-Agnostic, Context Reconstruction, Long-Context Inference, Cache Reuse

## TL;DR

This paper proposes KVzip, a query-agnostic KV cache eviction method that quantifies the importance of each KV pair by leveraging the LLM itself to reconstruct the original context from the cached KV pairs. KVzip achieves 3–4× KV cache compression and approximately 2× reduction in FlashAttention decoding latency, while significantly outperforming existing query-aware methods in multi-query scenarios.

## Background & Motivation

As the context lengths processed by LLMs continue to grow, the memory overhead of KV caches and attention computation latency have become deployment bottlenecks:

**Memory Challenge**: In long-context scenarios (e.g., 170K tokens), KV caches consume substantial GPU memory.

**Latency Challenge**: Attention computation latency scales linearly with KV cache length.

Existing KV cache compression methods are predominantly **query-aware**, determining which KV pairs to retain based on attention scores of the current query. However, this approach has fundamental limitations:

- **Cache non-reusability**: Each query requires recompression; the same compressed cache cannot be reused across multiple queries.
- **Performance degradation**: Even at 90% cache budget ratio, query-aware methods suffer performance degradation in multi-query scenarios.
- **Query dependency**: Decisions about which KV pairs are important can only be made at query time.

The core insight of KVzip is that the importance of a KV pair should be measured by its contribution to **context reconstruction**, rather than by the attention scores of a specific query.

## Method

### Overall Architecture

The KVzip pipeline proceeds as follows:

1. **Prefill Stage**: The LLM processes the context normally, generating a complete KV cache.
2. **Importance Estimation**: The LLM itself is used to evaluate the importance of each KV pair with respect to reconstructing the original context.
3. **KV Eviction**: Low-importance KV pairs are removed, yielding a compressed KV cache.
4. **Multi-Query Inference**: The compressed KV cache is reused across multiple distinct queries.

### Key Designs

**1. Importance Estimation via Context Reconstruction**

KVzip uses the LLM itself as an evaluator: given a candidate set of retained KV pairs, it measures whether the LLM can reconstruct the original input context from those pairs. Stronger reconstruction ability indicates that the retained KV pairs preserve more critical information.

Concretely, importance scores are computed as follows:
- The marginal increase in context reconstruction loss incurred by removing each KV pair is measured.
- A larger increase implies greater importance, and the corresponding KV pair is retained.

**2. Query-Agnosticism**

Importance estimation is entirely based on the context itself, independent of any future queries. This means:
- Compression is performed only once.
- The compressed KV cache can serve arbitrary subsequent queries.
- The approach is particularly well-suited for prefill-once, decode-many deployment scenarios (e.g., document QA, multi-turn dialogue).

**3. Layer-Wise Eviction Strategy**

Different attention layers and attention heads may assign different importance to KV pairs. KVzip independently estimates importance and performs eviction at each layer, enabling finer-grained compression.

### Loss & Training

KVzip is a **training-free** method that directly leverages the capabilities of existing LLMs for importance estimation. The context reconstruction loss is the standard cross-entropy loss, measuring the ability to reconstruct the original token sequence from the compressed KV cache.

## Key Experimental Results

### Main Results

**Table 1: Performance in Multi-Query Scenarios (25% KV Cache Budget)**

| Method | Type | QA Accuracy | Retrieval Accuracy | Reasoning Accuracy | Code Understanding Accuracy | Average |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Full Cache (100%) | — | Baseline | Baseline | Baseline | Baseline | 100% |
| H2O | Query-Aware | Significant Drop | Significant Drop | Significant Drop | Significant Drop | <80% |
| SnapKV | Query-Aware | Drop | Drop | Drop | Drop | <85% |
| **KVzip** | **Query-Agnostic** | **Near Baseline** | **Near Baseline** | **Near Baseline** | **Near Baseline** | **~97%** |

**Table 2: Compression Results Across Models**

| Model | KV Cache Compression Ratio | FlashAttention Latency Reduction | Performance Loss |
|:---|:---:|:---:|:---|
| LLaMA3.1-8B | 3–4× | ~2× | Negligible |
| Qwen2.5-14B | 3–4× | ~2× | Negligible |
| Gemma3 | 3–4× | ~2× | Negligible |

**Table 3: Comparison at Different Cache Budget Ratios (Multi-Query Scenario)**

| Cache Budget Ratio | H2O (Query-Aware) | SnapKV (Query-Aware) | KVzip (Query-Agnostic) |
|:---|:---:|:---:|:---:|
| 90% | Degradation | Degradation | Near Full Cache |
| 50% | Severe Degradation | Noticeable Degradation | Minor Impact |
| 25% | Unusable | Large Drop | Still Acceptable |

### Ablation Study

**Context Reconstruction vs. Other Importance Metrics**

| Importance Metric | Average Performance at 25% Budget |
|:---|:---:|
| Random Eviction | Very Poor |
| Attention Score (first query) | Moderate but Unstable |
| Accumulated Attention Score | Good but Degrades |
| **Context Reconstruction (KVzip)** | **Best and Stable** |

**Long-Context Scaling Test**

| Context Length | KVzip Compression Ratio | Performance Retention |
|:---|:---:|:---:|
| 8K tokens | 3× | >98% |
| 32K tokens | 3× | >97% |
| 128K tokens | 4× | >95% |
| 170K tokens | 4× | >94% |

### Key Findings

1. **Query-aware methods fail in multi-query scenarios**: Even at 90% budget ratio, methods such as H2O degrade on non-targeted queries due to query-specific optimization.
2. **Context reconstruction is a superior importance signal**: It better captures the global importance of KV pairs compared to attention scores.
3. **Cross-model consistency**: The method is effective across LLaMA3.1, Qwen2.5, and Gemma3.
4. **Feasibility for extremely long contexts**: Effective compression is maintained at 170K token context lengths.
5. **Practical speedup**: FlashAttention decoding latency is reduced by approximately 2×, with 3–4× memory reduction.

## Highlights & Insights

1. **Query-agnostic paradigm**: KVzip reframes the KV cache compression problem — importance should be determined by the information density of the context, not by the query.
2. **Self-supervised signal**: The method cleverly repurposes the LLM itself as a "judge" to assess KV pair importance, requiring no additional annotation or training.
3. **Prefill-once deployment**: The approach naturally fits real-world scenarios requiring multiple queries over the same context, such as long-document QA and agentic reasoning.
4. **NeurIPS 2025 Oral**: Reflects strong recognition from reviewers for this concise and effective approach.
5. **Plug-and-play**: Requires no modification to model architecture and can be directly applied to any Transformer-based LLM.

## Limitations & Future Work

1. **Overhead of importance estimation**: Additional forward computation is required to assess KV pair importance (a one-time cost).
2. **Static compression**: Once compressed, the retained KV pairs cannot be dynamically adjusted for new queries.
3. **Combination with quantization**: KV cache eviction and quantization are orthogonal techniques; combining them could yield greater compression ratios.
4. **Attention head heterogeneity**: Different attention heads may have different optimal compression ratios; adaptive strategies warrant further exploration.
5. **Streaming scenarios**: In settings with continuously growing context (e.g., dialogue), incremental importance updates are needed.

## Related Work & Insights

- **H2O** (Zhang et al., 2024): KV cache eviction based on attention scores.
- **SnapKV** (Li et al., 2024): Compression via observation-window attention patterns.
- **StreamingLLM** (Xiao et al., 2023): Streaming inference that retains attention sink tokens.
- **GQA / MQA**: Reducing KV cache size through shared KV heads.
- Key insight: The critical question in KV cache compression is not "which tokens are important to the current query," but rather "which tokens carry contextual information."

## Rating

| Dimension | Score (1–5) | Note |
|:---|:---:|:---|
| Novelty | 5 | Query-agnostic compression paradigm; novel perspective via context reconstruction |
| Technical Quality | 4 | Simple yet effective method; NeurIPS Oral |
| Experimental Thoroughness | 5 | Multi-model, multi-task, multi-length, multi-baseline evaluation |
| Practicality | 5 | Plug-and-play; addresses real deployment pain points |
| Writing Quality | 4 | Clear problem formulation; thorough presentation of results |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] Inference-Time Hyper-Scaling with KV Cache Compression](inference-time_hyper-scaling_with_kv_cache_compression.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[ACL 2026\] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](../../ACL2026/model_compression/fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)

</div>

<!-- RELATED:END -->
