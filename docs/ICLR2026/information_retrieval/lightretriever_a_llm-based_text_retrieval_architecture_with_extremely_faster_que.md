---
title: >-
  [Paper Note] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference
description: >-
  [ICLR 2026][LLM retrieval] This paper proposes LightRetriever, an extremely asymmetric LLM-based retrieval architecture: the document side retains a full LLM encoder, while the query side eliminates deep modeling entirely — dense retrieval reduces to embedding lookup and averaging, and sparse retrieval reduces to token counting — achieving 1000× query encoding speedup, 10× end-to-end throughput improvement, while retaining 95% of retrieval performance.
tags:
  - ICLR 2026
  - LLM retrieval
  - asymmetric encoder
  - ultra-fast query inference
  - hybrid retrieval
  - embedding cache
date: 2026-05-08
content_hash: 80e19703ac9a586c
---

# LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference

**Conference**: ICLR 2026
**arXiv**: [2505.12260](https://arxiv.org/abs/2505.12260)
**Code**: [GitHub](https://github.com/caskcsg/lightretriever)
**Area**: Information Retrieval
**Keywords**: LLM retrieval, asymmetric encoder, ultra-fast query inference, hybrid retrieval, embedding cache

## TL;DR
This paper proposes LightRetriever, an extremely asymmetric LLM-based retrieval architecture: the document side retains a full LLM encoder, while the query side eliminates deep modeling entirely — dense retrieval reduces to embedding lookup and averaging, and sparse retrieval reduces to token counting — achieving 1000× query encoding speedup, 10× end-to-end throughput improvement, while retaining 95% of retrieval performance.

## Background & Motivation
LLM-based retrievers (e.g., E5-Mistral, LLM2Vec) adopt symmetric dual-encoder architectures in which documents and queries share the same LLM encoder. Documents can be pre-computed offline, but queries must be encoded online, making deployment of a full-scale LLM as a query encoder challenging due to:

**Throughput bottleneck**: Encoding 65K queries with a full-size LLM requires 100+ seconds.

**Resource consumption**: Online serving requires GPU accelerators.

**Latency sensitivity**: Real-time search imposes strict latency constraints.

A key insight is that while documents benefit from the LLM's full modeling capacity (capturing rich contextual semantics), queries may not require equally deep modeling. BM25 achieves competitive performance with nearly zero inference cost via lexical matching, suggesting that query-side computation can be substantially simplified.

**Core Idea**: Break the query–document encoder symmetry — remove deep modeling from the query side entirely. During training, each token is independently encoded through the LLM; the resulting per-token embeddings are then cached, so that at inference time the entire forward pass is replaced by a table lookup and averaging.

## Method

### Overall Architecture
LightRetriever = Dense Retrieval (cached token embeddings + mean pooling) + Sparse Retrieval (encoder-free term-frequency vectors), with final scores obtained via linear interpolation of the two components.

### Key Designs
1. **Dense Retrieval: Cacheable Token Embeddings**

    - Training: A task instruction and a single query token are independently fed into the LLM encoder; the token representation is obtained via last-token pooling as $v_{t_i}^{\text{den}} = Enc_q(Inst; t_i)$, and the query vector is the mean of all token vectors $v_q^{\text{den}} = \frac{1}{n}\sum v_{t_i}^{\text{den}}$.
    - Caching: All token embeddings across the full vocabulary are pre-computed and stored in a lookup table $E \in \mathbb{R}^{V \times H}$. Caching with Llama-8B on 8×H800 GPUs takes under 20 seconds.
    - Online serving: $v_q^{\text{den}} = \frac{1}{n}\sum E[t_i]$, requiring only embedding lookup and averaging — no GPU needed.
    - Design Motivation: Token-independent encoding enables caching; eliminating inter-token interactions is the critical trade-off.

2. **Sparse Retrieval: Encoder-Free Query Representation**

    - Query vector: Directly uses token counts, $v_q^{\text{spr}}[t] = \text{count}(t)$, requiring no encoder at all.
    - Document vector: The last-layer hidden states are projected to the vocabulary space via the language model head, then processed with ReLU, log-saturation, and max pooling to yield a sparse vector $v_d^{\text{spr}} = \max(\ln(\max(h_{\text{last}} \cdot P, 0) + 1))$.
    - A FLOPs regulator controls the sparsity of document vectors.
    - Design Motivation: Sparse retrieval does not inherently require deep query understanding; raw term frequencies suffice.

3. **Contrastive Learning Training**

    - Standard listwise contrastive loss: $\ell^{CL} = -\log \frac{e^{v_q \cdot v_{d^+}/\tau}}{\sum e^{v_q \cdot v_d/\tau}}$.
    - Dense and sparse components are trained separately; scores are linearly interpolated at inference.

### Loss & Training
- Contrastive loss + FLOPs regularization (sparse component)
- 20 English + 3 Chinese datasets, 8.38M training samples
- LoRA fine-tuning, batch size = 128, 7 hard negatives, 12K steps

## Key Experimental Results

### Main Results

| Model | BeIR (nDCG@10) | CMTEB-R | Encoding Time (s) | Total Time (s) | QPS |
|---|---|---|---|---|---|
| Full-Llama8b | 56.8 | 67.6 | 109.49 | 119.37 | 549 |
| Full-Llama3b | 55.6 | 66.1 | 52.59 | 62.42 | 1050 |
| Llama8b (1st layer only) | 52.5 | 59.0 | 2.34 | — | — |
| **LightRetriever-Llama8b** | **54.0** | **63.8** | **0.04** | **10.08** | **6500** |
| Static Embedding | 44.9 | 49.1 | 0.04 | — | — |
| BM25 | 42.0 | 53.4 | 0 | — | — |

### Ablation Study

| Configuration | BeIR | CMTEB-R | Notes |
|---|---|---|---|
| Dense only | ~50 | ~60 | No sparse complement |
| Sparse only | ~42 | ~53 | Comparable to BM25 |
| Hybrid (default) | 54.0 | 63.8 | Best performance–efficiency trade-off |
| Full LLM encoder | 56.8 | 67.6 | Performance upper bound |
| Dimension truncation | ~53 | ~62 | Enables further embedding compression |

### Key Findings
- Query encoding time decreases from 109.5s to 0.04s, achieving a **2500× speedup**, with 12× end-to-end QPS improvement.
- The method retains 95% of full-size LLM retrieval performance, substantially outperforming first-layer-only Llama encoding.
- Sparse–dense hybrid retrieval significantly outperforms either modality alone.
- The approach generalizes effectively across diverse LLM backbones (Llama-1B/3B/8B, Qwen-1.5B/3B/7B).

## Highlights & Insights
- The insight that "queries do not require deep modeling" is highly thought-provoking, challenging the symmetric dual-encoder assumption.
- Caching the full vocabulary's token embeddings is an elegantly simple and effective one-time operation (< 20 seconds).
- The zero-encoder query design on the sparse retrieval side pushes lightweight inference to its limit.
- The strategy of shifting deep semantic modeling cost from the query side to the document side has broad applicability.

## Limitations & Future Work
- Token-independent encoding sacrifices intra-query contextual interactions, potentially degrading performance on complex queries.
- The embedding table must be re-cached for each instruction–model combination.
- The degree of performance degradation on long queries is not thoroughly analyzed.
- Dense vector dimensionality remains large (matching the LLM hidden size), resulting in non-trivial storage overhead.

## Related Work & Insights
- **vs. E5-Mistral**: Performance is maintained at 95% while achieving 2500× faster query encoding.
- **vs. BM25**: Performance is 12 nDCG points higher, with similarly near-zero query inference cost.
- **vs. Static Embedding**: Performance is 9 points higher, validating the gains from LLM-based training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic exploration of an extremely asymmetric encoder architecture, pushing query-side simplification to its limit.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated across 6 LLM backbones, 23 datasets, and both speed and quality dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear and intuitive presentation with rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Offers substantial practical value for real-world retrieval system deployment; the thousand-fold speedup is highly compelling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)
- [\[AAAI 2026\] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](../../AAAI2026/information_retrieval/opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](../../AAAI2026/information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)

</div>

<!-- RELATED:END -->
