---
title: >-
  [Paper Note] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration
description: >-
  [ACL 2026][Model Compression][KV cache compression] This paper proposes FastKV, which decouples context reduction (Token-Selective Propagation during the prefill phase) from KV cache compression (layer-wise KV retention during the decoding phase), achieving 1.82× prefill speedup and 2.87× decoding speedup on LLaMA-3.1-8B-Instruct while limiting accuracy degradation to within 1% on LongBench.
tags:
  - ACL 2026
  - Model Compression
  - KV cache compression
  - prefill acceleration
  - token-selective propagation
  - inter-layer context dynamics
  - decoding acceleration
date: 2026-05-08
content_hash: 5d723aababcb70b7
---

# FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration

**Conference**: ACL 2026
**arXiv**: [2502.01068](https://arxiv.org/abs/2502.01068)
**Code**: [GitHub](https://github.com/dongwonjo/FastKV)
**Area**: Model Compression / Inference Acceleration
**Keywords**: KV cache compression, prefill acceleration, token-selective propagation, inter-layer context dynamics, decoding acceleration

## TL;DR

This paper proposes FastKV, which decouples context reduction (Token-Selective Propagation during the prefill phase) from KV cache compression (layer-wise KV retention during the decoding phase), achieving 1.82× prefill speedup and 2.87× decoding speedup on LLaMA-3.1-8B-Instruct while limiting accuracy degradation to within 1% on LongBench.

## Background & Motivation

**Background**: LLMs support context windows of 128K tokens or even millions, yet long-context inference faces bottlenecks in both phases—attention computation during prefill grows quadratically with input length, while linearly growing KV caches during decoding become memory and bandwidth bottlenecks.

**Limitations of Prior Work**: (1) Decoding-side methods (e.g., SnapKV, H2O) only compress the already-generated KV cache without accelerating prefill; (2) Prefill-side methods (e.g., GemFilter) prune tokens starting from early layers, but critical tokens in early layers are highly unstable, causing irrecoverable information loss from premature pruning; (3) Existing prefill-aware methods tightly couple context reduction with KV budget—achieving sufficient decoding speedup requires aggressive prefill pruning, leading to accuracy degradation.

**Key Challenge**: Prefill requires full-context processing to maintain accuracy, yet decoding relies on only a tiny fraction of tokens. Coupling the two phases makes it impossible to optimize both simultaneously.

**Goal**: Decouple context reduction during prefill from KV compression during decoding, enabling independent control over the efficiency–accuracy trade-off in each phase.

**Key Insight**: Two key observations motivate the design: (1) Critical tokens in early layers are highly unstable (low overlap rate), whereas later layers exhibit stable overlap; (2) During decoding, all layers rely on only a very small number of tokens (Top-512, i.e., 0.38%, captures the majority of attention quality).

**Core Idea**: Place a TSP boundary at a stable layer—the first half performs full-context computation to preserve flexibility, while the second half propagates only critical tokens to accelerate prefill. Simultaneously, each layer independently retains a small fraction of KV cache for decoding, with the two ratios tunable independently.

## Method

### Overall Architecture

FastKV operates in two steps: (1) Two-phase prefill—the first half (layers 0 to the TSP layer) processes the full context and constructs a complete KV cache; at the TSP layer, top-K tokens are selected based on attention weights from window tokens, and only the hidden states of these tokens are passed to subsequent layers. (2) Layer-wise KV retention—each layer independently discards non-critical KV entries, retaining only a specified fraction of the cache for decoding.

### Key Designs

1. **Token-Selective Propagation (TSP)**:

    - **Function**: Truncates context propagation after critical tokens have stabilized, accelerating the second-half prefill.
    - **Mechanism**: At the TSP layer, a saliency score is computed for each token as the average attention weight it receives when queried by recent window tokens: $S_i^{TSPlayer} = \frac{1}{H}\sum_{h=0}^{H-1} S_i^{TSPlayer,h}$. Top-ranked tokens (according to a predefined TSP rate) are selected and propagated to subsequent layers. The TSP layer is typically set at the network midpoint (e.g., layer 15), where critical tokens have already stabilized.
    - **Design Motivation**: Critical token overlap rates in early layers are low (declining sharply as layer distance increases), but decay slowly after layer 15—indicating that the token subset required by subsequent layers is highly consistent and can be safely pruned.

2. **Decoupled KV Retention**:

    - **Function**: Independently controls the KV cache size during decoding.
    - **Mechanism**: Each layer independently selects KV entries to retain based on attention scores. In the first-half layers, although the full context is processed, the KV cache is still compressed according to a specified retention rate—since each layer relies on only a tiny number of tokens during decoding. The second-half layers naturally have smaller KV caches (containing only TSP-propagated tokens). The retention rate is set independently of the TSP rate.
    - **Design Motivation**: GemFilter and PyramidInfer couple prefill compression with KV budget—aggressive prefill compression inevitably yields an insufficient KV cache. After decoupling, even when the TSP rate is high (more tokens propagated), the KV retention rate can still be set very low to accelerate decoding.

3. **Inter-Layer Context Dynamics Analysis**:

    - **Function**: Provides a principled basis for selecting the TSP layer.
    - **Mechanism**: With 128K tokens fed into LLaMA-3.1-8B-Instruct, top-512 critical token indices are collected at each layer and inter-layer overlap rates are computed. Layers ≤15 show rapidly declining overlap (unstable), while layers >15 exhibit slow decay (stable). Additionally, Top-K attention recall analysis reveals that only 0.38% of tokens dominate the attention distribution across all layers.
    - **Design Motivation**: These two observations provide the empirical foundation for FastKV's core design—when to compress (after stabilization) and how aggressively (0.38% suffices).

### Loss & Training

FastKV is a training-free method applied solely at inference time. Validation is primarily conducted on LLaMA-3.1-8B-Instruct, with LongBench as the main evaluation benchmark.

## Key Experimental Results

### Main Results

| Method | Prefill | Decoding | Accuracy |
|--------|---------|----------|----------|
| Full-context | Slow | Slow | High |
| StreamingLLM | Slow | Fast | Low |
| SnapKV | Slow | Fast | High |
| GemFilter | Fast | Fast | Low |
| **FastKV** | **Fast (1.82×)** | **Fast (2.87×)** | **High (<1% drop)** |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| TSP layer position | Layer 15 is optimal—earlier placement degrades accuracy; later placement reduces speedup gains |
| TSP rate vs. KV retention rate decoupling | Decoupled tuning enables independent optimization, outperforming coupled schemes |
| Inter-layer critical token overlap | Rapidly declining for ≤15 layers; stable for >15 layers |
| Top-K attention recall | K=512 (0.38%) captures the majority of attention quality |

### Key Findings

- FastKV is the only method that simultaneously achieves prefill acceleration, decoding acceleration, and high accuracy.
- The decoupled design enables flexible efficiency–accuracy trade-offs, allowing the two ratios to be adjusted independently.
- The two-phase nature of inter-layer context dynamics (unstable → stable) provides clear guidance on when to prune tokens.
- Accuracy degradation of <1% on LongBench confirms that second-half layers have a genuinely low dependency on the full context.

## Highlights & Insights

- The concept of "decoupling" prefill and decoding compression is both elegant and profound—prior methods implicitly assumed synchronization between the two, an unnecessary constraint.
- The inter-layer critical token overlap analysis provides data-driven justification for TSP layer selection, avoiding blind hyperparameter tuning.
- The sparse context utilization finding (0.38% of tokens dominating attention) provides theoretical support for extreme KV compression.

## Limitations & Future Work

- Validation is primarily limited to LLaMA-3.1-8B-Instruct; generalization to larger models and different architectures requires further study.
- The 128K token analysis may not fully extend to even longer contexts.
- The TSP layer is fixed; adaptive selection could potentially yield further performance gains.
- Combinations with orthogonal compression techniques such as quantization have not been explored.

## Related Work & Insights

- **vs. SnapKV/H2O**: These methods compress only the decoding-side KV cache, leaving prefill as full computation; FastKV accelerates both phases.
- **vs. GemFilter**: GemFilter selects tokens from a single layer's attention and forces all subsequent layers to use that subset, harming early layers; FastKV preserves full context in the first half.
- **vs. PyramidInfer**: Progressive pruning from the first layer onward is overly aggressive; FastKV waits until critical tokens stabilize before pruning.

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupling approach and TSP design are concise and effective, though KV compression is already a crowded area.
- Experimental Thoroughness: ⭐⭐⭐⭐ LongBench evaluation is comprehensive, but model coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, experimental design is logically rigorous, and comparison tables are intuitive.
- Value: ⭐⭐⭐⭐⭐ Simultaneously accelerating prefill and decoding while maintaining accuracy offers exceptionally high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)

</div>

<!-- RELATED:END -->
