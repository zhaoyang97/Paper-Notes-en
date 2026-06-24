---
title: >-
  [Paper Note] Star Attention: Efficient LLM Inference over Long Sequences
description: >-
  [ICML2025][LLM (Other)][Long-sequence inference] Proposes Star Attention, a two-phase block-sparse attention mechanism: Phase 1 encodes partitioned context blocks via local attention across multiple hosts, and Phase 2 generates queries by aggregating global attention. It is compatible with existing LLMs without fine-tuning, achieving up to an 11x speedup in inference while retaining 97-100% accuracy.
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "Long-sequence inference"
  - "sparse attention"
  - "distributed inference"
  - "Anchor Block"
  - "KV cache"
date: 2026-05-08
content_hash: 364e55b1e292a858
---

# Star Attention: Efficient LLM Inference over Long Sequences

**Conference**: ICML2025  
**arXiv**: [2411.17116](https://arxiv.org/abs/2411.17116)  
**Code**: [GitHub - Star-Attention](https://github.com/NVIDIA/Star-Attention)  
**Area**: LLM/NLP  
**Keywords**: Long-sequence inference, sparse attention, distributed inference, Anchor Block, KV cache

## TL;DR
Proposes Star Attention, a two-phase block-sparse attention mechanism: Phase 1 encodes partitioned context blocks via local attention across multiple hosts, and Phase 2 generates queries by aggregating global attention. It is compatible with existing LLMs without fine-tuning, achieving up to an 11x speedup in inference while retaining 97-100% accuracy.

## Background & Motivation

### Key Challenge

**Key Challenge**: Paragraphs or contexts spanning millions of tokens are increasingly common (e.g., code analysis, multi-document summarization), but the quadratic complexity of self-attention makes inference extremely expensive.

### Goal

**Goal**: 
- FlashAttention: Accelerates computation but does not reduce complexity
- Ring Attention: Distributed but requires ring communication
- Chunk-based encoding methods: Require fine-tuning or extra components

### Core Insights of Star Attention

Inference typically consists of two phases: (1) long-context encoding, and (2) short-query generation. Context tokens only require local attention, while query tokens require global attention.

## Method

### Phase 1: Local Context Encoding + Anchor Block
- The context is split into contiguous blocks and distributed across different hosts.
- Each block is prefixed with an "anchor block" (a copy of the first block).
- Each host performs self-attention independently without communication.
- Only the KV values of the non-anchor parts are cached.

### Phase 2: Global Query Encoding
- Queries are replicated to all hosts.
- Each host computes local attention using its local KV cache.
- The "query host" aggregates softmax statistics to obtain global attention.
- Minimal communication overhead: each host transmits only 1 vector + 1 scalar per token.

### Characteristics
- The context length scales horizontally (linearly) with the number of hosts.
- Requires no model fine-tuning.
- Fully compatible and stackable with FlashAttention.

## Key Experimental Results

### Inference Acceleration


### Main Results

| Context Length | Number of Hosts | Speedup | Accuracy Retention |
|-----------|--------|---------|---------|
| 128K | 4 | 4x | 99% |
| 256K | 8 | 7x | 98% |
| 512K | 16 | **11x** | **97%** |

### Accuracy Comparison (Llama3.1-8B/70B)


### Ablation Study

| Benchmark | Standard Attention | Star Attention |
|------|----------|---------------|
| RULER | 100% | 97-100% |
| LongBench | Baseline | Close to Baseline |
| Needle-in-Haystack | 100% | 98%+ |

### Key Findings
1. The anchor block is critical; accuracy drops significantly without it.
2. Local attention in Phase 1 is sufficient for most context understanding tasks.
3. Global aggregation in Phase 2 requires minimal communication.
4. Demonstrates effectiveness on both Llama3.1-8B and 70B.

## Highlights & Insights

1. The two-phase design of "local context + global query" is highly intuitive and natural.
2. The introduction of the anchor block elegantly maintains global consistency.
3. Zero fine-tuning required—plug-and-play with any global-attention LLM.
4. Extremely low communication overhead (only 1 vector + 1 scalar per token from each host).
5. Can be combined and stacked with FlashAttention.

## Limitations & Future Work

1. Accuracy may suffer a 2–3% drop in tasks with extremely strong position dependence.
2. The choice of anchor block size affects performance and requires tuning.
3. May not be optimal for tasks requiring global interaction across the entire context (e.g., full-text summarization).
4. The single query host in Phase 2 could potentially become a bottleneck.
5. The integration with KV cache compression methods remains unexplored.

## Related Work & Insights

- Difference from Ring Attention: Ring Attention requires circular ring-communication, whereas Star Attention does not.
- Difference from Longformer: Longformer requires fine-tuning, whereas Star Attention is applicable zero-shot.
- Insight: The two-phase design can be generalized to other long-sequence tasks.

## Rating
- Novelty: 4.5/5 — Two-phase + anchor block design
- Experimental Thoroughness: 4.5/5 — Multi-model and multi-benchmark
- Writing Quality: 4.5/5
- Value: 5.0/5 — 11x speedup + plug-and-play

## Supplementary Technical Details

### Role of the Anchor Block
A copy of the first block is pre-pended to the block of each host, ensuring that every host can "see" the beginning of the global context. This design is inspired by the "attention sink" phenomenon.

### Communication Overhead Analysis in Phase 2
Each context host only needs to transmit softmax normalization statistics (one vector + one scalar per token), making the communication volume independent of context length.

### Typical Application Scenarios
Best suited for the "long context + short query + short answer" paradigm, such as RAG, document QA, and code analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](../../ACL2025/llm_nlp/bitnetcpp_efficient_edge_inference_for_ternary_llms.md)
- [\[ACL 2025\] X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](../../ACL2025/llm_nlp/xturing_enhanced_turing_test.md)
- [\[ECCV 2024\] Stripe Observation Guided Inference Cost-Free Attention Mechanism](../../ECCV2024/llm_nlp/stripe_observation_guided_inference_cost-free_attention_mechanism.md)
- [\[ACL 2025\] LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](../../ACL2025/llm_nlp/llm_mapreduce_simplified_long_sequence_processing.md)
- [\[ACL 2025\] MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](../../ACL2025/llm_nlp/mha2mla_deepseek_latent_attention.md)

</div>

<!-- RELATED:END -->
