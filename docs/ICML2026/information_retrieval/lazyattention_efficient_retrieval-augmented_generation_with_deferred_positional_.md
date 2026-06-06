---
title: >-
  [Paper Note] LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding
description: >-
  [ICML 2026][Information Retrieval & RAG][KV Cache Reuse] LazyAttention defers Rotary Positional Encoding (RoPE) from the KV cache write stage to being computed on-the-fly within the attention kernel. This allows a single…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "KV Cache Reuse"
  - "RoPE Decoupling"
  - "Fused Attention Kernel"
  - "Position-agnostic Cache"
  - "vLLM"
date: 2026-05-08
content_hash: 65aa3b63681e23c1
---

# LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding

**Conference**: ICML 2026  
**arXiv**: [2606.04302](https://arxiv.org/abs/2606.04302)  
**Code**: https://github.com/illinoisdata/lazy-attention  
**Area**: LLM Inference Efficiency / RAG / KV Cache  
**Keywords**: KV Cache Reuse, RoPE Decoupling, Fused Attention Kernel, Position-agnostic Cache, vLLM

## TL;DR
LazyAttention defers Rotary Positional Encoding (RoPE) from the KV cache write stage to being computed on-the-fly within the attention kernel. This allows a single physical KV copy to be reused by any logical position. On skewed RAG workloads, it reduces TTFT by 1.37× and increases throughput by 1.40× compared to SOTA Block-Attention, with negligible loss in generation quality.

## Background & Motivation
**Background**: In long-context scenarios such as RAG and ICL, prefilling is the latency bottleneck, and KV cache reuse is the primary path for cost reduction. Existing methods like Prompt Cache, CacheBlend, TurboRAG, and Block-Attention attempt to reuse KV caches across requests to avoid redundant document computation.

**Limitations of Prior Work**: Current KV caches are **position-aware**, meaning positional information is eagerly encoded into the Keys (typically via RoPE) before being written to the cache. This forces the storage of multiple KV replicas for the same document if it appears at different positions in different prompts. Block-Attention and TurboRAG choose to re-encode positions, but this requires either copying KV or restricting reuse to prefixes; in-place updates introduce race conditions within the same batch.

**Key Challenge**: KV cache reuse is limited by GPU HBM capacity, yet position-aware designs waste this capacity on "variant document positions." The authors provide a quantitative analysis: given a Zipf popularity, $D$ possible positions, and a budget of $C$ KV entries, a position-agnostic cache can store the top-$C$ documents, while a position-aware cache can only store $\lfloor C/D \rfloor$. The hit ratio ratio is $\sum_{i=1}^{C} i^{-\alpha} / \sum_{i=1}^{\lfloor C/D \rfloor} i^{-\alpha}$, which reaches 2.86× at $D=20, C=100$, and moderate Zipf skewness.

**Goal**: To make the KV cache truly position-agnostic without increasing HBM copy overhead or significantly adding to the computation/bandwidth costs of the attention kernel.

**Key Insight**: A core fact of RoPE is that the attention score depends only on the **relative** position $n-m$ of the query and key: $(R_m q)^\top (R_n k) = q^\top R_{n-m} k$. This implies that positional encoding can theoretically be applied "on-the-fly" during attention computation rather than being pre-baked into $k$.

**Core Idea**: Kernelize the RoPE-decoupling process so that deferred positional encoding is performed transiently within the inner loop of a fused Triton attention kernel. This achieves zero-copy, zero additional HBM writes, and allows a single KV copy to serve any logical position.

## Method

### Overall Architecture
LazyAttention treats the Transformer block in two parts (see Figure 1): **When writing the cache**, $Q/K/V$ are stored in the KV cache in "pure content" form without positional information (each document starts at local position 0). **When using the cache**, inside the fused attention kernel, a relative rotation $R_\Delta$ is applied to $Q$ or $K$ on-the-fly based on the global offset of the document in the current request before calculating the softmax. Mathematically, this is equivalent to standard RoPE attention where $K$ is pre-rotated, but physically only one KV cache copy is maintained.

Intuition from Example 3.1: Two documents $d_1, d_2$ are cached as $C_1, C_2$ (both starting from position 0). For a request $d_1 \mathbin\Vert d_2 \mathbin\Vert Q$, using $C_2$ requires rotating $Q$ backward by $|d_1|$ steps to align with the state where $C_2$ "pretends" it is still at position $0..|d_2|$.

### Key Designs

1. **Deferred Positional Encoding**:
    - **Function**: Moves RoPE from "before writing KV" to "during attention calculation," making the KV cache a set of position-agnostic "content key-value pairs."
    - **Mechanism**: Utilizes the relativity $q^\top R_{n-m} k$. $Q/K/V$ are cached without position. During attention, only a single offset $\Delta$ is needed to rotate the RoPE half-dimensions of $K$ (or $Q$) once: $k'_1 = k_1\cos\Delta - k_2\sin\Delta,\; k'_2 = k_1\sin\Delta + k_2\cos\Delta$. This is a **single relative rotation**, avoiding a naive "rotate back to 0 then to target" approach which would double FLOPs/IO.
    - **Design Motivation**: Directly addresses the capacity waste of position-aware caches. A single document no longer occupies physical entries for multiple positional variants, increasing document coverage and hit rates within a fixed HBM budget.

2. **Tiling-aware Q/K Rotation Dispatch (Rotate K for Prefill, Q for Decode)**:
    - **Function**: Minimizes deferred RoPE overhead by selecting which side to rotate based on the different bottlenecks of the prefill and decode stages.
    - **Mechanism**: Prefilling is compute-bound. With PagedAttention settings like $M=128, N=16$, the $Q$ tile is much larger than the $K$ tile, making rotating $K$ cheaper. A single rotation adds 3 FLOPs per $K$ scalar, making relative overhead $\epsilon_{\text{prefill}} = \tfrac{3}{4M} \approx 0.59\%$. Decoding is bandwidth-bound ($M=1$). Rotating $K$ would require scanning a whole tile, whereas rotating $Q$ involves minimal fixed FLOPs. Rotation is only triggered at document boundaries with a rate $r = 1/B$ ($B$ is document block count), resulting in average $\epsilon_{\text{decode}} = r\cdot \tfrac{3}{4N} = \tfrac{3}{4BN}$, or $\epsilon_{\text{decode}} \le 0.01\%$ for documents $>1600$ tokens.
    - **Design Motivation**: Modern attention kernels operate near hardware limits; any extra uncoalesced memory access or register pressure in the inner loop degrades performance. This dispatch places "extra work" in cheap slots of the compute-bound phase and sparse trigger points of the bandwidth-bound phase.

3. **Fused Triton Kernel + Bit-packed Metadata**:
    - **Function**: Enables production-ready implementation within vLLM/FlashAttention-style fused kernels while eliminating extra HBM access in the inner loop.
    - **Mechanism**: Implemented using vLLM v0.8.5 + Triton with two separate kernels. The prefill kernel splits each $K$ tile for half-dimension rotation before GEMM (Figure 3b). The decode kernel **bit-packs (block id, offset, mask) into a single 64-bit register**, using register shifts to extract metadata in the inner loop, bypassing global loads. In extremely IO-bound cases, cos/sin can be computed on-the-fly. The implementation spans ~5K lines of Python/Triton with ~0.2% measured runtime overhead.
    - **Design Motivation**: While deferred RoPE is theoretically elegant, any increased HBM access or register spill would negate the cache benefits. Fused kernels and in-register metadata provide the final step in translating algorithmic savings into end-to-end efficiency.

### Generality
The method is not tied to standard RoPE: interleaved RoPE and NTK/YaRN scaled RoPE only require metadata changes. GQA/MQA require zero modifications to score calculation. Even score-space relative position methods like ALiBi are compatible (verified on Falcon-7B with decode overhead <0.06%). Linear attention is out of scope as it alters attention state representation rather than just scores.

## Key Experimental Results

### Main Results
Model: Tulu3-Block-FT (Llama-3.1-8B derivative), H100 96GB. QA benchmarks: 2WikiMQA / HotpotQA / TriviaQA / NarrativeQA.

**TTFT and Throughput**: Under skewed traffic (Zipf $\alpha=2.1$), TTFT is reduced by 1.37× and throughput increased by 1.40× compared to Block-Attention (vLLM). Under uniform traffic, it matches Block-Attention and significantly outperforms Prefix Caching, Prompt Cache, and CacheBlend.

**KV Cache Hit Ratio** (VRAM hit ratio %, trace-driven, Zipf $\alpha = 1.1/1.5/2.1$):

| KV Budget | Skew | Prefix | CacheBlend | Block-Attn (vLLM) | LazyAttn (Ours) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 GB | High (α=2.1) | 0.00 | 5.96 | 7.27 | **13.57** |
| 1 GB | Low (α=1.1) | 0.00 | 1.51 | 1.84 | **3.47** |
| 10 GB | Mid | 0.55 | 17.33 | 21.13 | **23.89** |
| 50 GB | Mid | 1.95 | 21.87 | 26.67 | **28.44** |
| No-limit (~66 GB) | Mid | 2.16 | 22.45 | 27.38 | **29.09** |

Hit ratios nearly double under tight constraints and maintain a lead with generous budgets, demonstrating the utility of single-replica design across the memory spectrum.

**Generation Quality** (Exact Match):

| Dataset | Full-Attn | Block-Attn (vLLM) | LazyAttn |
| :--- | :--- | :--- | :--- |
| 2WikiMQA | 73.6 | 71.4 | 70.7 |
| TriviaQA | 75.2 | 72.1 | 73.0 |
| NarrativeQA | 62.2 | 61.0 | 59.7 |
| HotpotQA | 76.2 | 72.5 | 73.3 |
| Average | 71.8 | 69.3 | 69.2 |

LazyAttention is mathematically equivalent to Block-Attention; score differences stem only from tokenization and floating-point precision variability.

### Ablation Study
Comparison of "without deferred rotation" vs. "with deferred rotation" for a single RAG request (five 4096-token docs + 64-token query, 3 docs pre-cached):

| Stage | Key Findings |
| :--- | :--- |
| Document processing | Latency for "hot" docs drops to near zero (reuse avoids recomputation); baseline is dominated by recomputation costs. |
| Query prefilling | Comparable to baseline; extra FLOPs for K-tile rotation is only $3/(4M)$. |
| Decoding | Extra overhead per token is 0.13%, consistent with theoretical $r \cdot 3/(4N)$. |
| Long Context | Conclusions hold for docs up to 16K and contexts up to 128K (Appendix C.4). |

### Key Findings
- Gains originate from a **capacity multiplier** rather than faster raw computation. Position-agnosticism allows more documents in HBM, and the hit rate increase directly improves TTFT and throughput, especially under small budgets and skewed traffic.
- Decoding overhead $\le 0.2\%$ is a critical engineering result. While deferred rotation implies computing position for every token, using (a) Q-rotation, (b) boundary triggering, and (c) bit-packed registers reduces theoretical 100% IO increases to almost unmeasurable levels.
- Trends are consistent across different GPUs (A100/A40) and larger models (Llama-3.1-70B, Qwen3-8B).

## Highlights & Insights
- **Reducing reuse to position dependency**: The authors use a clear Zipf hit-rate formula $H_{\text{agnostic}}/H_{\text{aware}} = \sum_{i=1}^{C} i^{-\alpha} / \sum_{i=1}^{\lfloor C/D \rfloor} i^{-\alpha}$ to identify the bottleneck: the representation (position-aware) is wrong, not the caching strategy.
- **Kernel-aware algorithm design**: Prefill-K and Decode-Q rotation are not post-hoc optimizations but optimal choices derived from a roofline model. This exemplifies the "algorithm-system co-design" paradigm.
- **Generalizable idea**: The deferred encoding concept ("as long as position is correct during attention score calculation, it need not be baked into the state") can be extended to other score-space encoding and inspire "deferred metadata materialization" in MoE routing or speculative decoding caches.

## Limitations & Future Work
- Incompatibility with linear attention, where states aggregate sequences and position is not injected solely at the score level.
- Cannot handle reuse where content is strictly modified (e.g., cross-document semantic drift), though block-FT models help mitigate this.
- Evaluation is primarily on vLLM. Migration to other frameworks (SGLang, TensorRT-LLM) requires rewriting bit-packed and fused paths.
- Systematizing gains with orthogonal methods like KV compression or grouped-query attention remains an open direction.

## Related Work & Insights
- **vs. Block-Attention / TurboRAG (Ma 2025; Lu 2025)**: These also recognize RoPE/KV decoupling value but materialized position-adjusted KV replicas, thus hitting HBM limits or prefix constraints. LazyAttention eliminates this trade-off via fused kernels.
- **vs. CacheBlend (Yao 2025)**: CacheBlend uses mask reconstruction for accuracy within a position-aware framework but introduces reconstruction overhead. LazyAttention offers better TTFT and hit rates under skewed traffic.
- **vs. Prompt Cache / Prefix Caching**: Limited to strict prefix reuse. LazyAttention enables document-level reuse at any offset.
- **Insight**: For any scenario where "computation depends on metadata but metadata can be injected late" (e.g., shared embeddings, shared LoRA deltas), the "physical single copy + transient kernel injection" pattern is applicable.

## Rating
- Novelty: ⭐⭐⭐⭐ (Decoupling RoPE is known, but achieving zero-copy, zero-HBM fused Triton implementation is novel).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers multiple models, GPUs, traffic distributions, and micro-breakdowns).
- Writing Quality: ⭐⭐⭐⭐ (Quantifies motivation via Zipf and overhead via roofline; clear logical chain).
- Value: ⭐⭐⭐⭐⭐ (Directly improves cost/latency for long-context RAG services; generalizable idea).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](../../ICLR2026/information_retrieval/bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[ICML 2026\] Predictive Prefetching for Retrieval-Augmented Generation](predictive_prefetching_for_retrieval-augmented_generation.md)
- [\[ICML 2026\] ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)
- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](very_efficient_listwise_multimodal_reranking_for_long_documents.md)

</div>

<!-- RELATED:END -->
