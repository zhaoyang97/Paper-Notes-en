---
title: >-
  [Paper Note] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps
description: >-
  [ACL 2025][Information Retrieval & RAG][Adaptive-RAG] IDR² is proposed, a model-agnostic adaptive RAG acceleration framework. By eliminating redundant representations of overlapping documents across multi-round retrieval and utilizing retrieved content to guide parallel decoding, it achieves approximately 2× end-to-end acceleration without compromising generation quality.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Adaptive-RAG"
  - "KV Cache Sharing"
  - "Inference Acceleration"
  - "Speculative Decoding"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 159bb4a3c5a17772
---

# Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps

**Conference**: ACL 2025  
**arXiv**: [2505.12731](https://arxiv.org/abs/2505.12731)  
**Code**: None  
**Area**: Information Retrieval / RAG Acceleration  
**Keywords**: Adaptive-RAG, KV Cache Sharing, Inference Acceleration, Speculative Decoding, Retrieval-Augmented Generation

## TL;DR

IDR² is proposed, a model-agnostic adaptive RAG acceleration framework. By eliminating redundant representations of overlapping documents across multi-round retrieval and utilizing retrieved content to guide parallel decoding, it achieves approximately 2× end-to-end acceleration without compromising generation quality.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) mitigates the knowledge limitations of LLMs by introducing external knowledge. Adaptive RAG (A-RAG) further handles complex queries through multi-round retrieval-generation interactions, significantly improving response quality.

**Limitations of Prior Work**: The multi-round interaction mechanism of A-RAG exacerbates the inherent efficiency issues of RAG. **Key Challenge**: There is a significant overlap in retrieved results between adjacent rounds (experiments show ~60-80% of documents recur across consecutive rounds). However, existing methods reprocess all retrieved content from scratch in each round, leading to substantial redundant computation.

**Goal**: Eliminate redundant representation computations caused by overlapping retrieval results in A-RAG, while simultaneously accelerating the autoregressive decoding process.

**Key Insight**: Introduce acceleration mechanisms in both the prefilling and decoding phases: the prefilling phase eliminates redundant representations of overlapping documents through cache reuse and instruction guidance, while the decoding phase utilizes retrieved documents to construct an approximate language model for parallel generation.

**Core Idea**: Since multi-round retrieval results in A-RAG are highly overlapping, the KV representations of processed documents can be cached and directly reused. Meanwhile, since the generated content of RAG is highly correlated with the retrieved documents, these documents can be used to construct draft tokens for speculative-decoding-like parallel generation.

## Method

### Overall Architecture

IDR² divides the generation process of each A-RAG round into three phases: retrieval, prefilling, and decoding. Acceleration modules are introduced into the latter two phases: CICS+IDGR accelerates prefilling, while IGPG accelerates decoding.

### Key Designs

1. **Cross-Iteration Cache Sharing (CICS)**:
    - **Function**: Establish a shared cache space $\mathbb{C}$ to store the KV representations of processed documents in each round.
    - **Mechanism**: In the $t$-th round, first check which documents in the retrieved set $D_t$ are already cached. For cached documents $D_t^o$, directly load their $K_t^o, V_t^o$; execute prefilling only for new documents $D_t^n = D_t \setminus D_t^o$. Formalized as $a_t^1, K_t, V_t = \text{LLM}_P(q_t, D_t^n, A_{<t}, K_t^o, V_t^o)$.
    - **Design Motivation**: The document overlap rate between adjacent rounds in A-RAG is extremely high (reaching 60-80% under the 3-retrieved-document setting in experiments). Cache reuse can substantially reduce prefilling computation.

2. **Instruction-Driven De-duplication Guidance Reinforcement (IDGR)**:
    - **Function**: Guide the LLM to correctly handle redundant information in the cache through natural language instructions.
    - **Mechanism**: Due to the self-attention mechanism, the KV representations of cached documents contain information from other documents of the previous round (e.g., the representation of Document A is fused with information from Document B). IDGR automatically generates an instruction $I_t$ to inform the LLM of: (a) which documents are relevant/irrelevant to the current round; (b) the relevance ranking of the documents. For example: "#5881721 ...is a relevant document. #10028469 is an irrelevant document. Relevance score is...".
    - **Design Motivation**: Directly using the cached KV representations introduces irrelevant noise from previous rounds, leading to degraded generation quality (experiments show a 2-3% drop in EM). IDGR utilizes the instruction-following capability of LLMs to filter noise, which not only recovers but also enhances performance.

3. **Information-Guided Parallel Generation (IGPG)**:
    - **Function**: Build draft token sequences using the content of retrieved documents to achieve parallel verification and generation.
    - **Mechanism**: The content generated by RAG is highly correlated with the retrieved documents (about 70% of 2-token combinations appear in the retrieved documents). IGPG constructs an approximate N-gram language model $P(x_t|x_{t-N+1},...,x_{t-1})$ using the retrieved documents $D_t$ and queries matching subsequent phrase snippets as drafts before each autoregressive step. The LLM validates $M$ draft tokens in a single forward pass, generating multiple tokens at once if verification succeeds.
    - **Design Motivation**: Unlike traditional speculative decoding that requires training a smaller draft model, IGPG directly leverages the readily available retrieved documents in the RAG scenario as the draft source, requiring zero training cost.

### Loss & Training

IDR² is an inference-time acceleration method and does not involve model training. It uses BM25 or SGPT as the retriever and is compatible with various A-RAG methods (FL-RAG, FS-RAG, FLARE, DRAGIN).

## Key Experimental Results

### Main Results

Average speedup across 4 A-RAG methods × 3 models (LLaMA2-7B/13B, Vicuna-13B) × 4 datasets:

| Acceleration Phase | Min Speedup | Max Speedup | Average Speedup |
|----------|---------|---------|---------|
| Prefilling | 1.75× | 4.72× | 2.79× |
| Decoding | 1.49× | 4.00× | 2.33× |
| End-to-End | 1.31× | 3.53× | 2.0× |

Specific latency of DRAGIN+IDR² on LLaMA2-7B (2WikiMultihopQA, n=3):

| Phase | Original (s) | IDR² (s) | Speedup |
|------|---------|---------|--------|
| Prefilling | 3.71 | 1.18 | 3.14× |
| Decoding | 12.55 | 6.07 | 2.07× |
| End-to-End | 19.31 | 9.56 | 2.02× |

### Ablation Study

Effects of each module on DRAGIN (2WikiMultihopQA):

| CICS | IGPG | IDGR | LLaMA2-7B EM↑ | LLaMA2-13B EM↑ |
|------|------|------|---------------|----------------|
| ✗ | ✗ | ✗ | 22.5 | 30.4 |
| ✓ | ✗ | ✗ | 20.3 | 28.0 |
| ✗ | ✓ | ✗ | 22.4 | 30.4 |
| ✓ | ✓ | ✓ | **25.4** | **34.4** |

### Key Findings

1. Using CICS alone introduces redundant informational noise, leading to performance degradation; however, incorporating IDGR not only recovers but also surpasses the baseline.
2. IGPG has almost no impact on generation quality (EM variation < 0.1), as it is essentially parallel generation with verification.
3. DRAGIN benefits the most from IDR² (with prefilling speedup up to 4.72×) because its query refinement mechanism makes queries from adjacent rounds more similar, leading to higher document overlap.
4. IDR² is effective across different retrievers (BM25 vs SGPT), achieving speedup ratios > 2.6×.

## Highlights & Insights

- **Key Insight**: The overlap of multi-round retrieval results in A-RAG is completely ignored by existing methods, representing a "hidden" efficiency bottleneck.
- **Method Elegance**: IDR² is model-agnostic and can be applied out-of-the-box to any A-RAG method.
- **IDGR Novelty**: Resolves information conflicts in KV cache reuse using natural language instructions, cleverly exploiting the instruction-following capabilities of LLMs.
- **IGPG Efficiency**: Leverages the high correlation between generated and retrieved documents in RAG scenarios, constructing a draft model at zero cost.

## Limitations & Future Work

- It is only applicable to open-source LLMs and cannot be used with text-only LLM APIs.
- KV cache storage introduces additional VRAM overhead, which may become a bottleneck for extremely long documents.
- The acceleration performance of IGPG depends on the overlap between generated content and retrieved documents, which may be diminished in creative tasks.
- Further compression of KV cache to reduce storage overhead has not been explored.

## Related Work & Insights

- **Difference from TurboRAG**: TurboRAG precomputes the KV representations of the entire knowledge base, which is suitable for single-round RAG; whereas IDR² dynamically caches to address the overlap issue across multiple rounds in A-RAG.
- **Connection with Speculative Decoding**: IGPG borrows the validation idea of speculative decoding, but does not require an additional draft model, directly utilizing the retrieved documents instead.
- **Insight**: In scenarios involving frequent LLM invocations, such as multi-round conversations or multi-agent workflows, similar cache-sharing strategies may also yield effective results.

## Rating

- Novelty: ⭐⭐⭐⭐ Mentions and resolves the retrieval overlap redundancy issue in A-RAG systematically for the first time, with novel designs for IDGR and IGPG.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments covering 4 A-RAG methods × 3 models × 4 datasets, along with detailed ablation studies and case analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rich tables/figures, and rigorous formal definitions.
- Value: ⭐⭐⭐⭐ High practical value, offering out-of-the-box acceleration for A-RAG inference with a substantial ~2× speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] VISA: Retrieval Augmented Generation with Visual Source Attribution](visa_retrieval_augmented_generation_with_visual_source_attribution.md)

</div>

<!-- RELATED:END -->
