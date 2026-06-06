---
title: >-
  [Paper Note] Latent-Condensed Transformer for Efficient Long Context Modeling
description: >-
  [ACL2026][Model Compression][Long Context Modeling] LCA proposes performing context compression directly within the latent space of MLA—aggregating semantic latent vectors via query-aware weighted pooling and maintaining…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Long Context Modeling"
  - "KV Cache Compression"
  - "MLA"
  - "Latent Space Compression"
  - "Efficient Attention"
date: 2026-05-08
content_hash: 04fc4d89196601ea
---

# Latent-Condensed Transformer for Efficient Long Context Modeling

**Conference**: ACL 2026  
**arXiv**: [2604.12452](https://arxiv.org/abs/2604.12452)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Long Context Modeling, KV Cache Compression, MLA, Latent Space Compression, Efficient Attention

## TL;DR

LCA proposes performing context compression directly within the latent space of MLA—aggregating semantic latent vectors via query-aware weighted pooling and maintaining positional accuracy of position keys through anchor selection. It achieves a 2.5x prefilling speedup and 90% KV cache compression on 128K contexts while maintaining competitive performance.

## Background & Motivation

**Background**: LLM long-context processing faces two major bottlenecks: the linear growth of the KV cache and the quadratic computational complexity of self-attention. Multi-head Latent Attention (MLA) effectively reduces per-token KV cache size by projecting tokens into a low-dimensional latent space, a method widely adopted by models like DeepSeek-V2/V3. Sparse attention methods reduce computation by skipping or evicting unimportant tokens.

**Limitations of Prior Work**: These two technical routes cannot be directly combined. Sparse attention methods typically require reconstructing the full KV matrix from the MLA latent representation before sparsification, which completely negates the latent space compression advantages of MLA.

**Key Challenge**: While MLA prioritizes compressing the per-token cache, it still retains all $L$ tokens for attention calculation. To reduce the number of tokens within the latent space, semantic latent vectors $\mathbf{C}^{KV}$ can be aggregated, but position keys $\mathbf{K}^R$ (RoPE) cannot be simply mixed without distortion.

**Goal**: Design an efficient attention mechanism that operates natively within the MLA latent space to simultaneously reduce KV cache and computational overhead.

**Key Insight**: Semantic information is continuous and smooth, making it aggregatable, whereas positional encoding is non-linear and requires hard selection. Different compression strategies are applied to these two components.

**Core Idea**: Group the context into segments, aggregate semantic latent vectors $\mathbf{C}^{KV}$ for each group using query-aware weighted pooling, and maintain position key $\mathbf{K}^R$ accuracy through maximum correlation selection, compressing $L$ tokens into $L/g$ representatives.

## Method

### Overall Architecture

LCA divides the historical context into $m = \lfloor(L-w)/g\rfloor$ groups (group size $g=16$), where each group is compressed into a single representative. The most recent $w=1024$ tokens are kept intact. The compressed representatives and the local window are concatenated for standard attention calculation.

### Key Designs

1.  **Semantic Compression (Weighted Pooling)**:
    -   **Function**: Aggregates semantic latent vectors of each group into a single representative.
    -   **Mechanism**: Uses the average of the most recent $g$ queries $\bar{\mathbf{q}}$ as a summary query to calculate intra-group importance scores. After softmax normalization, weighted pooling is performed: $\mathbf{c}_j^{rep} = \sum \alpha_i^{(j)} \mathbf{c}_i^{KV}$. This is theoretically proven to be the optimal solution for minimizing expected reconstruction error.
    -   **Design Motivation**: Retains information from all tokens within a group (non-eviction) and biases the query-awareness toward tokens relevant to current decoding.

2.  **Positional Anchoring (Max Selection)**:
    -   **Function**: Maintains precise positional information for each group.
    -   **Mechanism**: Selects the token with the highest importance score within the group as the positional anchor $\mathbf{k}_j^{R_{rep}} = \mathbf{k}_{I_j}^R$.
    -   **Design Motivation**: RoPE is a non-linear function; pooling would mix different positional signals, leading to distortion.

3.  **Local Window Retention**:
    -   **Function**: Preserves fine-grained information for the immediate context.
    -   **Mechanism**: The most recent $w$ tokens are not compressed.
    -   **Design Motivation**: Next-token prediction is highly dependent on the immediate preceding context.

### Loss & Training

Lightweight fine-tuning was conducted on SlimPajama for 1000 steps. No additional parameters are introduced. Triton-optimized kernels were used on 8×H200 GPUs.

## Key Experimental Results

### Main Results (RULER 4-128K)

| Method | Average | 128K Latency |
| :--- | :--- | :--- |
| MLA Original | 58.91 | 10.78s |
| MInference | 37.60 | 5.66s (1.9×) |
| FlexPrefill | 39.11 | 5.38s (2.0×) |
| KDA | 54.63 | 4.96s (2.2×) |
| **Ours (LCA)** | **58.80** | **4.40s (2.5×)** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Semantic Pooling + Positional Anchoring | 58.80 | Full LCA |
| Pure Pooling (including Position) | Decrease | RoPE mixing leads to positional distortion |
| Pure Sparsity | Severe Decrease | Irreversible information loss |

### Key Findings

-   Achieves 2.5× prefilling acceleration and 90% KV cache compression.
-   Performance remains nearly lossless (58.80 vs 58.91), significantly outperforming sparse methods.
-   MInference and FlexPrefill collapse at 32K+ context, while LCA remains stable.
-   The design is architecture-agnostic and can be extended to GQA.
-   The approximation error bound is independent of context length.

## Highlights & Insights

-   The decoupled compression principle (aggregatable semantics, preserved positions) aligns with the decoupled design philosophy of MLA.
-   The optimality of weighted pooling is supported by theoretical proof (Proposition 1).
-   High practical utility due to zero additional parameters and lightweight fine-tuning requirement.

## Limitations & Future Work

-   Validation was only performed on DeepSeek-V2-Lite (16B).
-   Uses a fixed group size $g=16$; adaptive sizing might be more effective.
-   Positional anchoring uses hard selection, losing positional details of other tokens within the group.

## Related Work & Insights

-   **vs FlexPrefill/MInference**: These methods reconstruct full KV before sparsification, failing to exploit MLA's latent space advantages and suffering performance collapse in long contexts.
-   **vs KDA**: KDA requires integration during training from scratch, whereas LCA can be applied to existing models via lightweight fine-tuning.

## Rating

-   **Novelty**: ⭐⭐⭐⭐ First to perform context compression within the MLA latent space.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐ Multidimensional evaluation but limited to one model.
-   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear organization of theory, algorithm, and experiments.
-   **Value**: ⭐⭐⭐⭐⭐ Addresses the practical integration of MLA and efficient attention.

**Conference**: ACL2026  
**arXiv**: [2604.12452](https://arxiv.org/abs/2604.12452)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](../../ICCV2025/model_compression/context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)

</div>

<!-- RELATED:END -->
