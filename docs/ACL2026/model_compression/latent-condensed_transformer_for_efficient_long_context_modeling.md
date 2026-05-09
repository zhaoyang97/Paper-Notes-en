---
title: >-
  [Paper Note] Latent-Condensed Transformer for Efficient Long Context Modeling
description: >-
  [ACL 2026][Model Compression][To be supplemented] To be supplemented after thorough reading.
tags:
  - ACL 2026
  - Model Compression
  - To be supplemented
date: 2026-05-08
content_hash: 76a941dc8e9bd0d9
---

# Latent-Condensed Transformer for Efficient Long Context Modeling

**Conference**: ACL 2026
**arXiv**: [2604.12452](https://arxiv.org/abs/2604.12452)
**Code**: None
**Area**: Model Compression
**Keywords**: long context modeling, KV cache compression, MLA, latent space compression, efficient attention

## TL;DR

LCA proposes performing context compression directly in the latent space of MLA — aggregating semantic latent vectors via query-aware weighted pooling and preserving positional accuracy through anchor selection for positional keys — achieving 2.5× prefill speedup and 90% KV cache compression on 128K contexts while maintaining competitive performance.

## Background & Motivation

**Background**: Long-context processing in LLMs faces two major bottlenecks: linear growth of KV cache and quadratic computational complexity of self-attention. MLA (Multi-head Latent Attention) effectively reduces per-token KV cache size by projecting tokens into a low-dimensional latent space, and has been widely adopted in models such as DeepSeek-V2/V3. Sparse attention methods reduce computation by skipping or evicting unimportant tokens.

**Limitations of Prior Work**: These two technical directions cannot be directly combined — sparse attention methods must first reconstruct the full KV matrices from MLA's latent representations before sparsification, completely negating the compression advantage of MLA's latent space.

**Key Challenge**: Although MLA compresses the per-token cache, all $L$ tokens still participate in attention computation. To reduce the number of tokens in the latent space, semantic latent vectors $\mathbf{C}^{KV}$ can be aggregated, but positional keys $\mathbf{K}^R$ (RoPE) cannot be naively mixed.

**Goal**: Design an efficient attention mechanism that operates natively in the MLA latent space, simultaneously reducing KV cache size and computation.

**Key Insight**: Semantic information is continuous and smooth and thus amenable to aggregation, whereas positional encodings are nonlinear and must be preserved via hard selection. Different compression strategies are applied to the two components separately.

**Core Idea**: Partition the context into groups; for each group, aggregate semantic latent vectors $\mathbf{C}^{KV}$ via query-aware weighted pooling and preserve positional key accuracy via maximum-relevance selection, compressing $L$ tokens into $L/g$ representatives.

## Method

### Overall Architecture

LCA divides the historical context into $m = \lfloor(L-w)/g\rfloor$ groups (group size $g=16$), each compressed into one representative; the most recent $w=1024$ tokens are kept intact. The compressed representatives and the local window are concatenated for standard attention computation.

### Key Designs

1. **Semantic Compression (Weighted Pooling)**:

    - **Function**: Aggregate the semantic latent vectors within each group into a single representative.
    - **Mechanism**: The average of the most recent $g$ queries $\bar{\mathbf{q}}$ is used as a summary query; importance scores within the group are computed and normalized via softmax, then used for weighted pooling: $\mathbf{c}_j^{rep} = \sum \alpha_i^{(j)} \mathbf{c}_i^{KV}$. This is theoretically proven to be the optimal solution minimizing expected reconstruction error.
    - **Design Motivation**: Retains all token information within the group (no discarding); query-awareness biases the representative toward tokens relevant to the current decoding step.

2. **Positional Anchoring (Maximum Selection)**:

    - **Function**: Preserve precise positional information for each group.
    - **Mechanism**: The token with the highest importance score within the group is selected as the positional anchor: $\mathbf{k}_j^{R_{rep}} = \mathbf{k}_{I_j}^R$.
    - **Design Motivation**: RoPE is a nonlinear function; pooling mixes position signals from different positions, leading to distortion.

3. **Local Window Retention**:

    - **Function**: Maintain fine-grained information for nearby context.
    - **Mechanism**: The most recent $w$ tokens are not compressed.
    - **Design Motivation**: Next-token prediction is highly dependent on nearby context.

### Loss & Training

Lightweight fine-tuning on SlimPajama for 1,000 steps. No additional parameters introduced. Triton-optimized kernels. 8×H200 GPUs.

## Key Experimental Results

### Main Results (RULER 4–128K)

| Method | Average | 128K Latency |
|--------|---------|--------------|
| MLA (original) | 58.91 | 10.78s |
| MInference | 37.60 | 5.66s (1.9×) |
| FlexPrefill | 39.11 | 5.38s (2.0×) |
| KDA | 54.63 | 4.96s (2.2×) |
| **LCA** | **58.80** | **4.40s (2.5×)** |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| Semantic pooling + positional anchoring | 58.80 | Full LCA |
| Pooling only (with position) | Degraded | RoPE mixing causes positional distortion |
| Sparse only | Severe degradation | Irreversible information loss |

### Key Findings

- 2.5× prefill speedup + 90% KV cache compression.
- Negligible performance degradation (58.80 vs. 58.91), far outperforming sparse methods.
- MInference/FlexPrefill collapse at 32K+, while LCA remains stable.
- Architecture-agnostic design; extensible to GQA.
- Approximation error bound is independent of context length.

## Highlights & Insights

- The decoupled compression principle — semantics are aggregatable, positions must be preserved — is consistent with the decoupled design philosophy of MLA.
- The optimality of weighted pooling is theoretically proven (Proposition 1).
- No additional parameters and lightweight fine-tuning make the method highly practical.

## Limitations & Future Work

- Validated only on DeepSeek-V2-Lite (16B).
- Fixed group size $g=16$; adaptive grouping may yield further improvements.
- Positional anchoring is a hard selection, discarding positional details of other tokens within each group.

## Related Work & Insights

- **vs. FlexPrefill/MInference**: These methods reconstruct the full KV cache before sparsification and thus cannot exploit the MLA latent space advantage; performance collapses at long contexts.
- **vs. KDA**: Requires training from scratch with integration; LCA can be applied to existing models via lightweight fine-tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to perform context compression in the MLA latent space.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation, but limited to a single model.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory, algorithm, and experiments are clearly organized.
- Value: ⭐⭐⭐⭐⭐ Addresses a practical problem at the intersection of MLA and efficient attention.

**Conference**: ACL 2026
**arXiv**: [2604.12452](https://arxiv.org/abs/2604.12452)
**Code**: To be confirmed
**Area**: object_detection
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](../../ICCV2025/model_compression/context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)

</div>

<!-- RELATED:END -->
