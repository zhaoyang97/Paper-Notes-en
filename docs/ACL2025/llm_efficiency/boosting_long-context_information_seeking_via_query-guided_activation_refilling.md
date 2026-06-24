---
title: >-
  [Paper Note] Boosting Long-Context Information Seeking via Query-Guided Activation Refilling
description: >-
  [ACL 2025 Main][LLM Efficiency][Long Context] This paper proposes ACRE (Activation Refilling), which constructs a bi-layer KV Cache architecture—consisting of an L1 layer to compactly capture global information and an L2 layer to provide detailed local information. By using the input query to dynamically replenish relevant items from L2 to L1, ACRE achieves highly efficient processing of long-context information retrieval tasks, with significant improvements in both performan…
tags:
  - "ACL 2025 Main"
  - "LLM Efficiency"
  - "Long Context"
  - "KV Cache"
  - "Information Retrieval"
  - "Dynamic Attention"
  - "Activation Refilling"
date: 2026-05-08
content_hash: 7d6349a64323ab09
---

# Boosting Long-Context Information Seeking via Query-Guided Activation Refilling

**Conference**: ACL 2025 Main  
**arXiv**: [2412.12486](https://arxiv.org/abs/2412.12486)  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: Long Context, KV Cache, Information Retrieval, Dynamic Attention, Activation Refilling

## TL;DR
This paper proposes ACRE (Activation Refilling), which constructs a bi-layer KV Cache architecture—consisting of an L1 layer to compactly capture global information and an L2 layer to provide detailed local information. By using the input query to dynamically replenish relevant items from L2 to L1, ACRE achieves highly efficient processing of long-context information retrieval tasks, with significant improvements in both performance and efficiency.

## Background & Motivation

**Background**: Processing long context is one of the core challenges faced by LLMs. Context window limitations and the computational overhead of massive KV activations seriously affect efficiency. Existing methods mainly include: context compression (e.g., StreamingLLM retaining sink tokens + recent tokens), sparse attention (e.g., LongLoRA), and KV cache eviction (e.g., H2O and SnapKV selecting important KV pairs to retain based on attention scores).

**Limitations of Prior Work**: (1) Static compression methods (discarding parts of KV in a one-off manner) cannot adapt to the dynamic information demands of different queries—one query might require a global overview, while another may need precise details from a specific paragraph; (2) existing KV cache eviction methods are typically query-independent (deciding which KVs to keep before generation begins), resulting in cases where "the information required by the query might happen to be discarded"; (3) fully keeping all KVs guarantees accuracy but incurs linearly growing computational and memory overheads, which is unsustainable.

**Key Challenge**: For information retrieval tasks (such as QA and summarization), different queries have vastly different scopes of information requirements over the context—simple factual queries only need to locate a single segment, whereas comprehensive analytical queries need to integrate the entire text. A fixed KV cache size cannot satisfy both needs simultaneously.

**Goal**: Design a query-adaptive long-context processing method that significantly reduces computational overhead while maintaining high accuracy.

**Key Insight**: The authors observe that in information retrieval tasks, coarse global perception at the L1 level is always needed (to quickly locate relevant areas), while fine local information at the L2 level is only needed in specific areas (for a deep understanding of the answer). This naturally maps to a bi-layer cache architecture.

**Core Idea**: Build a bi-layer KV Cache and a query-dynamically-driven L2 $\to$ L1 activation "refilling" mechanism, supplementing local details on demand.

## Method

### Overall Architecture
The workflow of ACRE is as follows: (1) Precompute KV activations of the long context and store them in the L1 and L2 caches, respectively—L1 retains global summary-level information via a compression method, while L2 retains complete detailed information but does not directly participate in attention calculation; (2) when receiving an input query, first let the query attend to the L1 cache to obtain a global understanding; (3) based on the L1 attention distribution, identify the regions most relevant to the query, extract detailed KV pairs of these regions from the L2 cache, and "refill" them into the active cache; (4) the query performs the final attention calculation and answer generation on the enhanced cache.

### Key Designs

1. **Bi-layer KV Cache**:

    - **Function**: Provide two granularities of information snapshots for long context.
    - **Mechanism**: The L1 cache retains a small number of representative entries from the complete KV through uniform sampling or attention score filtering (e.g., retaining 1 token for every N tokens) to form a "summary" view of the context, which has a fixed size and is much smaller than the original length. The L2 cache stores complete KV activations but does not directly participate in the forward calculation—it is stored in chunks in memory and read chunk by chunk only when needed. A "proxy relationship" is established between the two cache layers: each entry in L1 knows which detailed entries in L2 it represents.
    - **Design Motivation**: The bi-layer design balances global perception (L1 is always available but coarse-grained) and local accuracy (L2 is called on demand but high-precision). The hierarchical cache design is inspired by cache hierarchies in computer architecture (L1 cache is fast but small, L2 cache is large but slow).

2. **Query-Guided Activation Refilling**:

    - **Function**: Dynamically supplement relevant information from L2 to the active cache according to the information needs of the current query.
    - **Mechanism**: (a) The query first attends to the L1 cache to compute the attention weight distribution; (b) Top-k most relevant L1 entries are identified based on the attention weights (representing the regions the query focuses on most); (c) The corresponding detailed L2 blocks for these L1 entries are located through proxy mapping; (d) The KV pairs of these L2 blocks are loaded into the active cache and merged with the L1 cache; (e) The query performs a complete attention calculation on the merged cache. The amount of refilled data can be dynamically adjusted—simple queries refill only a few blocks, while complex queries refill more.
    - **Design Motivation**: Traditional methods either discard information at the very beginning (irrecoverable) or retain everything (too expensive). ACRE achieves on-demand refinement through a "coarse-to-fine" strategy, where discarded information can be recalled when needed.

3. **Dynamic Refilling Volume Control**:

    - **Function**: Adaptively decide how much L2 information to refill.
    - **Mechanism**: Control the refilling volume based on the entropy of the L1 attention distribution. If the attention is concentrated on a few locations (low entropy, indicating the query has a clear local demand), a small but concentrated amount of L2 blocks is refilled; if the attention is scattered (high entropy, indicating the query requires synthesised information), more scattered L2 blocks are refilled. The refilling volume is proportional to the attention entropy: $k = k_{min} + (k_{max} - k_{min}) \cdot \frac{H(a)}{H_{max}}$, where $H(a)$ is the entropy of the attention distribution.
    - **Design Motivation**: A fixed refilling volume cannot adapt to variations in query complexity. Dynamic control ensures that simple queries do not waste computation, and complex queries do not miss key information.

### Loss & Training
ACRE is a training-free, plug-and-play method that is directly applied to the inference stage of existing LLMs. It only requires precomputing the bi-layer cache and proxy mapping relationships.

## Key Experimental Results

### Main Results

| Dataset | Metric | ACRE | Full KV | StreamingLLM | H2O | SnapKV |
|--------|------|------|---------|-------------|-----|--------|
| LongBench-QA | F1 | 42.8 | 43.5 | 35.2 | 38.6 | 39.4 |
| LongBench-Summary | ROUGE-L | 26.3 | 27.1 | 21.5 | 23.8 | 24.5 |
| NIAH (128K) | Acc | 95.2 | 97.8 | 68.3 | 82.5 | 88.1 |
| InfiniteBench | Score | 38.7 | 40.2 | 29.8 | 33.4 | 35.1 |
| **KV Size** | **Ratio** | **~15%** | **100%** | **~5%** | **~10%** | **~10%** |

### Ablation Study

| Configuration | LongBench F1 | KV Size | Description |
|------|-------------|--------|------|
| Full ACRE | 42.8 | 15% | Full method |
| L1 only (No refilling) | 37.5 | 8% | Coarse-grained representation is insufficient for precise QA |
| Fixed refilling volume (k=50) | 41.3 | 15% | Inferior to dynamic control |
| Random refilling (Non-attention guided) | 39.1 | 15% | Query-guided is 3.7 points better than random refilling |
| Full L2 participation | 43.5 | 100% | Upper bound (equivalent to Full KV) |

### Key Findings
- ACRE achieves performance close to Full KV (100%) using only 15% of the KV cache (an F1 gap of only 0.7), while significantly outperforming baselines with equivalent or larger cache ratios.
- On the Needle In A Haystack (NIAH) task, ACRE performs exceptionally well (95.2% vs. 82.5% of H2O), demonstrating that the bi-layer cache and refilling mechanism are particularly effective for precise information localization.
- Query-guided refilling outperforms random refilling by 3.7 F1 points, validating the core assumption of "on-demand retrieval".
- Dynamic refilling volume control brings a 1.5 F1 point improvement, indicating that different queries indeed require varying amounts of detailed information.
- Using only the L1 cache (no refilling) causes a sharp performance drop of 5.3 points, proving that L2 refilling is critical—coarse global information alone is insufficient to support accurate answer generation.

## Highlights & Insights
- The bi-layer cache architecture design is inspired by computer cache hierarchies, making this cross-disciplinary borrowing very elegant. The design principle of "L1 is fast but coarse, L2 is detailed but on-demand" can be transferred to other scenarios requiring an efficiency-accuracy balance.
- The principle that "discarded information can be recalled" constitutes the fundamental difference from traditional KV eviction methods. In traditional methods, discarding is irreversible, whereas ACRE's L2 retention allows any info to be restored on demand.
- The training-free, plug-and-play nature gives ACRE high practical value, allowing it to be directly applied to any pretrained LLM.

## Limitations & Future Work
- The L2 cache still needs to be stored in memory; for ultra-long context (million-token scale), the memory overhead may become non-negligible.
- The refilling operation introduces additional memory retrieval latency, which may require optimization in latency-sensitive real-time applications.
- The heuristic of dynamic refilling volume based on attention entropy might not be optimal; a supervised learning strategy for refilling could be better.
- The experiments primarily validate on information retrieval tasks; the effectiveness on reasoning-heavy long-context tasks (such as multi-hop reasoning) remains to be verified.

## Related Work & Insights
- **vs StreamingLLM**: StreamingLLM only retains sink + recent tokens, leading to weak global perception; ACRE's L1 cache preserves a global summary.
- **vs SnapKV/H2O**: These methods prune KV in a one-off manner with no way to recover; ACRE keeps L2 as a backup to recover on demand.
- **vs RAG methods**: RAG retrieves from external sources, whereas ACRE retrieves from within the context; the two can complement each other.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of bi-layer caching and dynamic refilling is novel, and borrowing from cache hierarchies is very elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough comparisons across multiple long-context benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology description is clear.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical, training-free, efficient method published in the ACL Main conference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs](distance_between_relevant_information_pieces_causes_bias_in_long-context_llms.md)
- [\[CVPR 2026\] QuietPrune: Query-Guided Early Token Pruning for Vision-Language Models](../../CVPR2026/llm_efficiency/quietprune_query-guided_early_token_pruning_for_vision-language_models.md)
- [\[ICLR 2026\] ProtoKV: Knowledge in Long Context is Already Organized Before You Query](../../ICLR2026/llm_efficiency/protokv_long-context_knowledges_are_already_well-organized_before_your_query.md)
- [\[ICML 2026\] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](../../ICML2026/llm_efficiency/team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)

</div>

<!-- RELATED:END -->
