---
title: >-
  [Paper Note] EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments
description: >-
  [ICML 2026][Model Compression][KV cache compression] EpiCache is proposed as a training-free KV cache management framework. It controls memory ceilings through block-wise prefilling…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "KV cache compression"
  - "long-term conversation"
  - "episodic management"
  - "block-wise prefill"
  - "memory-constrained inference"
date: 2026-05-08
content_hash: c8f2768247164996
---

# EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments

**Conference**: ICML 2026  
**arXiv**: [2509.17396](https://arxiv.org/abs/2509.17396)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: KV cache compression, long-term conversation, episodic management, block-wise prefill, memory-constrained inference

## TL;DR
EpiCache is proposed as a training-free KV cache management framework. It controls memory ceilings through block-wise prefilling, preserves topic-relevant contexts via episodic clustering, and optimizes inter-layer cache distribution using sensitivity-aware budget allocation. It achieves near full-cache accuracy at 4-6x compression ratios and reduces peak memory by 3.7x across three long-conversation QA benchmarks.

## Background & Motivation

**Background**: Modern LLMs have extended context lengths to the million-token scale, enabling conversational AI to generate coherent and personalized responses using long-term histories. Mainstream KV cache compression methods (e.g., H2O, SnapKV, KVzip) typically perform post-prefill eviction, retaining important KV pairs based on attention scores after processing the full context.

**Limitations of Prior Work**: First, post-prefill methods require caching the entire context during the prefill phase, causing peak memory to grow linearly with input length, which prevents deployment on memory-constrained devices like smartphones. For instance, the LLaMA3.2-3B KV cache exceeds 7GB after only 30 dialogue sessions—larger than the model parameters. Second, query-dependent eviction (e.g., SnapKV) narrows cache semantics to a single query, potentially evicting evidence needed for subsequent questions in multi-turn dialogues.

**Key Challenge**: A severe trade-off exists between bounded memory and answer accuracy. Directly applying post-prefill methods within a block-prefill framework leads to sharp accuracy degradation due to the lack of global context needed to judge token importance during chunked processing.

**Goal**: To achieve high-quality long-conversation query answering (LongConvQA) under a strictly fixed memory budget while ensuring controllable peak memory.

**Key Insight**: Conversation histories naturally possess an episodic structure where consecutive utterances revolve around specific topics. By clustering history into multiple topic episodes and constructing dedicated KV caches for each, the system can load only the most relevant episodic cache during query time, saving memory while preserving topic-related context. Furthermore, different Transformer layers exhibit varying sensitivities to block-wise prefilling, allowing for adaptive budget allocation across layers.

**Core Idea**: Long conversations are clustered into semantically coherent episodes. A compressed KV cache is constructed for each episode, and the most relevant episodic cache is retrieved via embedding matching for decoding during inference.

## Method

### Overall Architecture
EpiCache consists of an offline construction phase and an online decoding phase. Offline Phase (Phase A): (1) Dialogue history is segmented, embedded, and clustered into $E$ topic episodes; (2) Layer-wise sensitivity is calculated via calibration to allocate KV budgets; (3) Block-wise prefill is executed for each episode, using a representative segment as a patched prompt to guide eviction and build episode-specific caches. Online Phase (Phase B): The user query is embedded to match the nearest episode centroid, and the corresponding KV cache is retrieved for decoding.

### Key Designs

1.  **Episodic KV Cache**:
    - **Function**: Organizes long dialogue history into multiple topic episodes and builds independent compressed KV caches for each.
    - **Mechanism**: The dialogue history $\mathcal{H}$ is segmented every $w_{\text{embed}}$ utterances. A lightweight encoder $f_{\text{embed}}$ encodes each segment into a vector, followed by K-Means clustering into $E$ episodes $\{\mathcal{E}_1, \ldots, \mathcal{E}_E\}$. For each episode, the segment $S_{\text{centroid-closest}}$ nearest to the centroid is identified as a patched prompt to guide the block-wise prefill eviction—tokens with high attention scores are retained to form the episodic cache $C_{\text{KV}}^{(e)}$. During decoding, the query $q_i$ is embedded into the same space to match the nearest centroid $e^\dagger = \arg\max_e \cos(\mathbf{q}_i, \mathbf{c}_e)$.
    - **Design Motivation**: Leveraging the natural topic structure of conversations ensures the patched prompt is semantically close to future queries, resolving the issue where query-dependent eviction requires knowing the query in advance.

2.  **Block-wise Prefill**:
    - **Function**: Strictly limits peak GPU memory to $M + M_{\text{block}}$, independent of input length.
    - **Mechanism**: The input is divided into blocks of size $M_{\text{block}}$ and processed sequentially. After each block, low-scoring tokens are evicted based on attention scores (guided by the patched prompt) to compress the KV cache back to budget $M$. The token importance score is defined as $s_i^{\max} = \max_{t \in [n+1, n+p]} \text{Attn}(x_t \to x_i)$, representing the maximum attention weight from the patched prompt tokens to the context tokens.
    - **Design Motivation**: Prevents the linear memory growth of post-prefill methods, ensuring constant memory usage for on-device deployment.

3.  **Sensitivity-aware Layer-wise Budget Allocation**:
    - **Function**: Adaptively allocates KV cache budgets based on each layer's sensitivity to block-wise prefilling.
    - **Mechanism**: Forward passes are performed using a full causal mask $\mathcal{M}$ and a block-wise mask $\mathcal{M}'$. The cosine similarity of Key states $\sigma_\ell = \frac{1}{HN}\sum_{h,i} \cos(k_{\text{full},i}^{(\ell,h)}, k_{\text{block},i}^{(\ell,h)})$ is compared. Sensitivity is defined as $s_\ell = 1 - \sigma_\ell$, and budgets are allocated as $M_\ell^{\text{alloc}} = \frac{s_\ell^\alpha}{\sum_j s_j^\alpha} \cdot (L \cdot M)$. $\alpha$ controls the sharpness of the distribution, with $\alpha = 2\text{-}4$ yielding optimal results.
    - **Design Motivation**: Observations show that different layers have distinct and consistent sensitivities (model-dependent rather than input-dependent). Sensitivity-aware allocation prevents resource waste in insensitive layers.

## Key Experimental Results

### Main Results (Qwen3-4B, RealTalk)

| Method | Budget | Multi-hop | Temporal | Common | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Full KV | — | 53.6 | 61.7 | 52.2 | 56.9 |
| RAG-Episodic | 8K | 42.3 | 22.4 | 41.0 | 33.4 |
| KVzip | 8K | 34.4 | 35.0 | 43.3 | 36.0 |
| EpiCache (Ours) | 8K | **51.7** | **55.7** | **54.7** | **53.9** |

### Ablation Study (Qwen3-4B, RealTalk, 8K Budget)

| Configuration | Avg | Description |
| :--- | :--- | :--- |
| Utterance segment + Qwen3-Emb-0.6B | 49.8 | Base configuration (no budget allocation) |
| Word segment | 47.5 | Breaking natural boundaries degrades accuracy |
| LLM-embedding replacement | 43.0 | Poor performance using LLM embedding layers |
| E=2 (Few episodes) | 47.9 | Lack of granularity |
| E=8 (Many episodes) | 51.3 | Slight gain with more episodes |
| + Layer budget alpha=2 | **53.9** | Sensitivity allocation contribution +4.1 |
| + Layer budget alpha=8 | 49.8 | Overly sharp allocation is harmful |

### Efficiency Analysis (LLaMA3.2-3B, 90K token history, 300 round follow-up)

| Method | Peak Memory (GB) | Total Latency (s) | Per-round Latency (s) | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Full KV (No cache) | 36.3 | 9339.0 | 31.1 | 46.2 |
| Full KV (Prefix caching) | 36.3 | 1062.8 | 3.5 | 46.2 |
| EpiCache (8K) | **9.6** | **545.4** | **1.8** | 45.6 |

### Key Findings
- EpiCache consistently outperforms all baselines (KVzip, OracleKV, SnapKV, StreamingLLM, InfiniPot, KeyDiff) across all budget levels, with gains of up to 30 absolute points on the Qwen3 series at low budgets (2-4K).
- It reaches near Full KV accuracy at 4-6x compression while reducing peak memory by 3.5x and accelerating decoding latency by 2.4x.
- Layer sensitivity is model-specific rather than input-specific; a single-sample calibration provides stable layer weights.
- Episodic caching is robust to cross-episode queries because each episodic cache is built during block-wise prefill within the full context, retaining globalized representations.

## Highlights & Insights
- Integrating episodic structures into KV cache management is an elegant abstraction—approximating future query semantics via clustering and representative segments without needing the query beforehand. This insight is transferable to long-document QA.
- Sensitivity-aware allocation is a low-cost, high-gain design: requiring only two calibration forward passes to achieve a +4.1 accuracy Gain.
- The framework is entirely training-free, deployment-friendly, and directly applicable to off-the-shelf LLMs.

## Limitations & Future Work
- The number of episodes $E$ currently requires manual setting; adaptive determination is a target for improvement.
- Incremental updates for compressed caches when episodic budgets are exceeded have not yet been implemented.
- Validation is limited to dialogue and document QA; complex long-term memory scenarios (e.g., preference tracking, knowledge forgetting) remain untested.
- Dependence on an external lightweight encoder (Qwen3-Emb-0.6B) requires further cross-domain generalization testing.

## Related Work & Insights
This work intersects with KV cache compression (H2O, SnapKV, KVzip), retrieval-based dialogue memory (MemoryBank, SeCom), and cache retrieval methods (Quest, ClusterKV, IceCache). The core insight is that KV cache compression should be topic-aware rather than strictly query-agnostic or query-dependent—organizing by topic and retrieving relevant caches balances universality and relevance. This approach can be extended to multimodal long-context scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](../../NeurIPS2025/model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)
- [\[ICML 2026\] Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds](memory-efficient_partitioned_dnn_inference_on_resource-constrained_android_crowd.md)
- [\[ICML 2026\] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](xkv_cross-layer_kv-cache_compression_via_aligned_singular_vector_extraction.md)
- [\[ACL 2026\] The Pitfalls of KV Cache Compression](../../ACL2026/model_compression/the_pitfalls_of_kv_cache_compression.md)

</div>

<!-- RELATED:END -->
