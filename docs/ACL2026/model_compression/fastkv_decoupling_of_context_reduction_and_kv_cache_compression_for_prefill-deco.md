---
title: >-
  [Paper Note] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration
description: >-
  [ACL 2026][Model Compression][KV Cache Compression] FastKV is proposed to decouple context reduction (Token-Selective Propagation in the prefill stage) from KV cache compression (layer-wise KV retention in the decoding s…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "KV Cache Compression"
  - "Prefill Acceleration"
  - "Token-Selective Propagation"
  - "Inter-layer Context Dynamics"
  - "Decoding Acceleration"
date: 2026-05-08
content_hash: 9dc3b101ce5b90d5
---

# FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration

**Conference**: ACL 2026  
**arXiv**: [2502.01068](https://arxiv.org/abs/2502.01068)  
**Code**: [GitHub](https://github.com/dongwonjo/FastKV)  
**Area**: Model Compression / Inference Acceleration  
**Keywords**: KV Cache Compression, Prefill Acceleration, Token-Selective Propagation, Inter-layer Context Dynamics, Decoding Acceleration

## TL;DR

FastKV is proposed to decouple context reduction (Token-Selective Propagation in the prefill stage) from KV cache compression (layer-wise KV retention in the decoding stage). It achieves 1.82× prefill and 2.87× decoding acceleration on LLaMA-3.1-8B-Instruct while maintaining accuracy within a 1% drop on LongBench.

## Background & Motivation

**Background**: LLMs support context windows of 128K or even millions of tokens. however, long-context inference faces a two-stage bottleneck: attention computation in the prefill stage grows quadratically with input length, and the linearly growing KV cache in the decoding stage becomes a memory and bandwidth bottleneck.

**Limitations of Prior Work**: (1) Decoding-side methods (e.g., SnapKV, H2O) only compress the generated KV cache and do not accelerate prefill; (2) Prefill-side methods (e.g., GemFilter) prune tokens starting from early layers, but critical tokens in early layers are highly unstable, leading to irrecoverable information loss; (3) Existing prefill-aware methods tightly couple context reduction with the KV budget—to accelerate decoding sufficiently, prefill must be aggressively pruned, causing accuracy degradation.

**Key Challenge**: Prefill requires full-context processing to maintain accuracy, but decoding relies on only a few tokens. Coupling the two implies an inability to optimize both stages simultaneously.

**Goal**: Decouple prefill context reduction and decoding KV compression to independently control the efficiency-accuracy trade-off for both stages.

**Key Insight**: Two critical observations are utilized: (1) Critical tokens in early layers are highly unstable (low overlap rate) but stabilize in later layers (high overlap rate); (2) All layers rely on only a few tokens during decoding (Top-512, or 0.38%, captures most attention quality).

**Core Idea**: A TSP boundary point is set in stable layers—the first half performs full-context computation to maintain flexibility, while the second half propagates only critical tokens to accelerate prefill. Simultaneously, each layer independently retains a small portion of the KV cache for decoding, with the two ratios being independently adjustable.

## Method

### Overall Architecture

FastKV operates in two steps: (1) Two-stage prefill—The first half (Layer 0 to the TSP layer) processes the full context and builds a complete KV cache; the TSP layer selects top-K tokens based on the attention weights of window tokens and propagates only their hidden states to subsequent layers. (2) Layer-wise KV retention—Each layer independently discards non-critical KV entries, retaining only a specified proportion of the cache for decoding.

### Key Designs

1. **Token-Selective Propagation (TSP)**:

    - **Function**: Truncate context propagation after critical tokens stabilize to accelerate the second half of prefill.
    - **Mechanism**: At the TSP layer, the average attention weight of each token when queried by the most recent window tokens is calculated as the saliency score $S_i^{TSPlayer} = \frac{1}{H}\sum_{h=0}^{H-1} S_i^{TSPlayer,h}$. Top-ranked tokens (based on a predefined TSP rate) are selected for propagation. The TSP layer is typically set in the middle of the network (e.g., Layer 15), where critical tokens have stabilized.
    - **Design Motivation**: The overlap rate of critical tokens in early layers is low (dropping sharply as layer distance increases), but the attenuation slows down after Layer 15, indicating that the token subsets required by subsequent layers are highly consistent and can be safely pruned.

2. **Decoupled KV Retention**:

    - **Function**: Independently control the KV cache size during decoding.
    - **Mechanism**: Each layer independently selects KV entries to retain based on attention scores. Although the first half of layers processes the full context, their KV caches are still compressed according to the specified retention rate since each layer only depends on minimal tokens during decoding. The second half of layers naturally possesses smaller KV caches (containing only TSP-propagated tokens). Retention rates and TSP rates are set independently.
    - **Design Motivation**: GemFilter and PyramidInfer bind prefill compression with the KV budget—aggressive prefill compression inevitably leads to insufficient KV cache. With decoupling, the KV retention rate can remain low to accelerate decoding even if the TSP rate is high (more tokens propagated).

3. **Inter-layer Context Dynamics Analysis**:

    - **Function**: Provide a theoretical basis for TSP layer selection.
    - **Mechanism**: By inputting 128K tokens into LLaMA-3.1-8B-Instruct, top-512 critical token indices are collected for each layer to calculate inter-layer overlap rates. Overlap rates drop sharply for layers $\le 15$ and decay slowly for layers $> 15$. Top-K attention recall analysis shows that only 0.38% of tokens dominate the attention distribution across all layers.
    - **Design Motivation**: These two observations provide an empirical foundation for the core design—when to compress (after stabilization) and how much to compress (0.38% is sufficient).

### Loss & Training

Ours is a training-free method applied only during inference. It is primarily validated on LLaMA-3.1-8B-Instruct, with LongBench serving as the main evaluation benchmark.

## Key Experimental Results

### Main Results

| Method | Prefill | Decoding | Accuracy |
|------|---------|---------|------|
| Full-context | Slow | Slow | High |
| StreamingLLM | Slow | Fast | Low |
| SnapKV | Slow | Fast | High |
| GemFilter | Fast | Fast | Low |
| **FastKV** | **Fast (1.82×)** | **Fast (2.87×)** | **High (<1% drop)** |

### Ablation Study

| Analysis Dimension | Results |
|----------|------|
| TSP Layer Position | Layer 15 is optimal—earlier leads to accuracy drop, later reduces acceleration gains. |
| TSP Rate vs KV Retention Rate Decoupling | Decoupling allows independent optimization, outperforming coupled schemes. |
| Inter-layer Critical Token Overlap | Sharp decline for layers $\le 15$, stable for layers $> 15$. |
| Top-K Attention Recall | K=512 (0.38%) captures most attention quality. |

### Key Findings

- FastKV is the only method achieving simultaneous prefill acceleration, decoding acceleration, and high accuracy.
- The decoupled design allows for a flexible efficiency-accuracy trade-off by adjusting the two ratios independently.
- The two-stage nature of inter-layer context dynamics (unstable → stable) provides clear guidance for the timing of token pruning.
- An accuracy drop of <1% on LongBench indicates that the reliance of the second half of layers on the full context is indeed very low.

## Highlights & Insights

- The concept of "decoupling" prefill and decoding compression is simple yet profound—prior methods implicitly assumed the two must be synchronized, which is an unnecessary constraint.
- Analysis of inter-layer critical token overlap provides a data-driven basis for TSP layer selection, avoiding blind hyperparameter tuning.
- The discovery of sparse context utilization (0.38% tokens dominating attention) provides theoretical support for extreme KV compression.

## Limitations & Future Work

- Validation is mainly on LLaMA-3.1-8B-Instruct; generalization to larger models or different architectures needs verification.
- Analysis at 128K tokens may not fully apply to even longer contexts.
- The TSP layer is fixed—adaptive selection might further improve performance.
- Combinations with orthogonal compression techniques like quantization have not been explored.

## Related Work & Insights

- **vs SnapKV/H2O**: These methods only compress decoding-side KV cache, while prefill still requires full computation; FastKV accelerates both.
- **vs GemFilter**: GemFilter selects tokens from a single layer's attention and forces all subsequent layers to use them, harming early layers; FastKV retains the full context for the first half.
- **vs PyramidInfer**: Pruning starts progressively from the first layer, which is too aggressive; FastKV waits until stabilization.

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupling idea and TSP design are simple and effective, though significant work already exists in KV compression.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on LongBench is thorough, though model coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation analysis is clear, experimental logic is rigorous, and comparison tables are intuitive.
- Value: ⭐⭐⭐⭐⭐ High practical value for simultaneously accelerating prefill and decoding while maintaining accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ACL 2026\] The Pitfalls of KV Cache Compression](the_pitfalls_of_kv_cache_compression.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)

</div>

<!-- RELATED:END -->
