---
title: >-
  [Paper Note] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments
description: >-
  [NeurIPS 2025][Model Compression][KV cache eviction] This paper proposes KeyDiff — an attention-score-free KV cache eviction strategy that maintains the cache by retaining keys with the lowest average cosine similarity t…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "KV cache eviction"
  - "key similarity"
  - "long-context inference"
  - "attention-free"
  - "block prompt processing"
date: 2026-05-08
content_hash: fa3ca1a0af5e26e9
---

# KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments

**Conference**: NeurIPS 2025
**arXiv**: [2504.15364](https://arxiv.org/abs/2504.15364)
**Code**: None
**Area**: Model Compression
**Keywords**: KV cache eviction, key similarity, long-context inference, attention-free, block prompt processing

## TL;DR
This paper proposes KeyDiff — an attention-score-free KV cache eviction strategy that maintains the cache by retaining keys with the lowest average cosine similarity to other keys (i.e., geometrically most unique). Under strict memory constraints in block-wise inference settings, KeyDiff achieves ≤0.04% accuracy loss on LongBench with an 8K cache budget, while reducing end-to-end inference latency by up to 30%.

## Background & Motivation

**Background**: KV cache is a standard technique for accelerating LLM inference, but its memory footprint grows linearly with context length. Existing cache eviction methods (H2O, TOVA, SnapKV, StreamingLLM) assess token importance via attention scores and evict less important KV pairs.

**Limitations of Prior Work**: In resource-constrained environments (e.g., edge devices), block-wise inference is required — the prompt is split into small chunks processed sequentially. Existing attention-score-based eviction methods degrade severely in this setting, as each block can only observe attention weights over local tokens and cannot anticipate which tokens will be important in future blocks; eviction errors accumulate and propagate across blocks.

**Key Challenge**: Attention scores are query-dependent. In block-wise inference, they can only be computed based on local queries within the current block, making eviction decisions myopic.

**Key Observation**: Keys with low cosine similarity to other keys (i.e., geometrically unique keys) tend to receive high attention scores. This property is entirely intrinsic to the keys themselves, independent of the query, and thus remains valid in block-wise settings.

**Core Idea**: Replace attention scores with inter-key cosine similarity as the eviction criterion — retaining the most "unique" keys and evicting redundant ones.

## Method

### Overall Architecture
In block-wise inference, new KV pairs are appended to the cache after each block is processed. If the cache exceeds the budget $N$, the eviction policy is invoked. KeyDiff computes the average cosine similarity of each key to all other keys in the cache and retains the $N$ KV pairs with the lowest similarity scores (i.e., the most unique keys), evicting the rest.

### Key Designs

1. **KeyDiff Base Formulation**:

    - **Function**: Scores each cached key to retain the most unique ones.
    - **Mechanism**: $S = \text{topk}(-\text{CosSim}(K)\mathbf{1}, N)$, where $\text{CosSim}(K) \in \mathbb{R}^{n \times n}$ is the pairwise cosine similarity matrix among keys, and $\mathbf{1}$ is the all-ones vector. The score is the negated row sum — lower similarity yields a higher score.
    - **Design Motivation**: Unlike attention scores, inter-key similarity does not depend on the query, enabling accurate global importance estimation in block-wise inference.

2. **Efficient Variant (Linear Complexity)**:

    - **Function**: Reduces the $O(n^2)$ pairwise similarity computation to $O(n)$.
    - **Mechanism**: $S = \text{topk}(-\text{CosSim}(\mu(\hat{K}), \hat{k}_i), N)$, where the anchor vector $\mu(\hat{K}) = \frac{1}{n}\sum_i \hat{k}_i$ is the mean of normalized keys. Each key requires only a single cosine similarity computation against the anchor.
    - **Design Motivation**: Under mild conditions, this variant retains the same KV pairs as the full formulation (proven in the Appendix). Experiments further show that using the mean of unnormalized keys $\mu(K)$ as the anchor incurs no accuracy loss.

3. **Theoretical Justification**:

    - **Lemma 3.1**: Establishes a lower-bound relationship between the attention weight $w$ and the key-query cosine similarity: $\frac{-\log(1-w)}{2M} - 1 \leq \text{CosSim}(k^*, q)$.
    - **Theorem 3.2**: Establishes a triangular relationship — if key $k^*$ is highly aligned with query $q$ ($\text{CosSim}(k^*, q) = \beta_q > 0$) while the mean key is not ($\text{CosSim}(\bar{k}, q) = \alpha_q < 0$), then $\text{CosSim}(\bar{k}, k^*) \leq 1 + \alpha_q\beta_q - 0.5\alpha_q^2 - 0.5\beta_q^2$ (approaching $-1$).
    - **Implication**: High-attention keys tend to have cosine similarity close to $-1$ with the anchor vector; thus, KeyDiff naturally retains keys most aligned with the query.

4. **Sliding Window Extension**:

    - **Function**: Allocates a portion of the cache budget to the most recent tokens.
    - **Mechanism**: For tasks where recent tokens are more important (e.g., reasoning and coding), a fraction of the cache budget is reserved for the most recent tokens.
    - **Design Motivation**: This extension yields significant gains on reasoning models such as DeepSeek-R1, with no additional computational overhead.

### Compatibility with FlashAttention
KeyDiff does not require explicit computation of the attention matrix $A$, making it directly compatible with optimized kernels such as FlashAttention — a significant practical advantage over methods like H2O and SnapKV.

## Key Experimental Results

### Main Results — Llama 3.1-8B on LongBench (B=128)

| Method | Cache | NarrQA | HotpotQA | GovReport | TriviaQA | PR-en | Avg |
|--------|-------|--------|----------|-----------|----------|-------|-----|
| No Eviction | Full | 30.05 | 57.33 | 34.86 | 91.61 | 99.50 | 49.20 |
| H2O | 8K | 13.85 | 43.64 | 18.78 | 69.05 | 62.50 | 38.20 |
| TOVA | 8K | 24.86 | 54.52 | 33.44 | 91.11 | 87.00 | 47.09 |
| Sink | 8K | — | — | — | — | — | ~40 |
| **KeyDiff** | **8K** | **26.59** | **55.98** | **34.25** | **91.38** | **95.50** | **49.16** |

With an 8K cache (~23% compression), KeyDiff achieves only 0.04% accuracy degradation relative to the no-eviction baseline.

### Ablation Study: Cache Budget

| Method | 2K | 4K | 6K | 8K |
|--------|-----|-----|-----|-----|
| H2O | 16.89 | 26.37 | 33.19 | 38.20 |
| TOVA | 37.52 | 41.49 | 45.24 | 47.09 |
| KeyDiff | **38.75** | **44.02** | **47.70** | **49.16** |
| No Eviction | — | — | — | 49.20 |

With a 6K cache (~33% compression), KeyDiff incurs ≤1.5% accuracy loss.

### Inference Efficiency

| Method | End-to-End Latency |
|--------|--------------------|
| TOVA | Baseline |
| SnapKV | ~Baseline |
| **KeyDiff** | **30% reduction** |

By eliminating the need to compute the attention matrix and supporting FlashAttention, KeyDiff achieves substantially lower latency.

### Key Findings
- H2O degrades severely in block-wise inference (16.89 at 2K cache vs. KeyDiff's 38.75), confirming that attention scores are unreliable when propagated across blocks.
- PCA visualizations show that KeyDiff retains more diverse and spatially spread keys, whereas TOVA and Sink retain keys that cluster together.
- On DeepSeek-R1-Distill-Llama-8B for Math-500 reasoning tasks, KeyDiff with the sliding window extension approaches the no-eviction baseline.
- Negative cosine similarity consistently outperforms alternative eviction criteria such as positive similarity or Euclidean distance.

## Highlights & Insights
- **"Key uniqueness ≈ attention importance" insight**: This is an elegant observation — high-attention keys tend to differ from other keys, allowing key importance to be assessed without inspecting the query.
- **Closed theoretical loop**: Lemma 3.1 (attention → key-query similarity) combined with Theorem 3.2 (key-query → key-anchor similarity) establishes a complete theoretical bridge from attention scores to KeyDiff scores.
- **FlashAttention compatibility**: By avoiding materialization of the attention matrix and relying solely on key geometry, KeyDiff is highly practical for real-world deployment.
- **Linear-complexity efficient variant**: The anchor-vector approximation reduces complexity from $O(n^2)$ to $O(n)$ with negligible accuracy loss.

## Limitations & Future Work
- Theoretical assumptions (e.g., $\text{CosSim}(\bar{k}, q) < 0$) may not hold across all attention heads and layers.
- Evaluation is limited to the Llama and Qwen model families; generalizability to other architectures (e.g., Mistral, Phi) remains unverified.
- The sliding window ratio requires manual tuning, with no adaptive selection strategy proposed.
- Compatibility with Grouped Query Attention (GQA) is not discussed.
- Under extreme compression ratios (e.g., 2K cache), accuracy degradation remains notable and may necessitate combination with complementary compression techniques such as quantization.

## Related Work & Insights
- **vs. H2O**: H2O accumulates attention scores for eviction decisions; in block-wise inference, reliance on incomplete attention information leads to severe degradation. KeyDiff, relying purely on key geometry, is unaffected by this limitation.
- **vs. TOVA**: TOVA retains KV pairs with the highest attention to the current token and remains query-dependent; KeyDiff is query-independent.
- **vs. SnapKV**: SnapKV performs attention-based clustering over a window and requires materializing the attention matrix, making it incompatible with FlashAttention.
- **vs. StreamingLLM**: StreamingLLM fixedly retains initial sink tokens and recent tokens — a simple but non-adaptive strategy; KeyDiff dynamically selects the most unique keys.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The insight is concise and elegant with substantial theoretical depth, though the core operation (cosine similarity ranking) is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple models and benchmarks with comprehensive ablations and efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with tight integration of theory and experiments, supported by intuitive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to edge deployment and long-context inference; FlashAttention compatibility is a decisive practical advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](../../ACL2026/model_compression/dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)

</div>

<!-- RELATED:END -->
