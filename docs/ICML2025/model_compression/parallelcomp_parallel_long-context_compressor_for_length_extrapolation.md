---
title: >-
  [Paper Note] ParallelComp: Parallel Long-Context Compressor for Length Extrapolation
description: >-
  [ICML 2025][Model Compression][length extrapolation] This paper proposes ParallelComp, a training-free parallel long-context compression method. By leveraging parallel KV cache eviction and attention calibration strategies, it enables an 8B LLM to extrapolate from an 8K context window up to 128K tokens on a single A100 GPU.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "length extrapolation"
  - "KV cache eviction"
  - "parallel attention"
  - "attention sink"
  - "training-free"
date: 2026-05-08
content_hash: 2fba09059e9906f9
---

# ParallelComp: Parallel Long-Context Compressor for Length Extrapolation

**Conference**: ICML 2025  
**arXiv**: [2502.14317](https://arxiv.org/abs/2502.14317)  
**Code**: [GitHub](https://github.com/menik1126/ParallelComp)  
**Area**: Model Compression  
**Keywords**: length extrapolation, KV cache eviction, parallel attention, attention sink, training-free  

## TL;DR

This paper proposes ParallelComp, a training-free parallel long-context compression method. By leveraging parallel KV cache eviction and attention calibration strategies, it enables an 8B LLM to extrapolate from an 8K context window up to 128K tokens on a single A100 GPU.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Ultra-long context extrapolation (>128K) remains a major challenge for LLMs:

### Background

**Background**: NTK-based methods and text chunking techniques are limited by the **attention sink** phenomenon.

### Key Challenge

**Key Challenge**: The attention bias in chunked parallel attention is fundamentally different from that in classical attention mechanisms, yet it has not been fully investigated before.

### Core Idea

**Core Idea**: Memory bottlenecks limit the feasibility of long-sequence inference.

## Method

### 1. Parallel Attention

The input sequence $X \in \mathbb{R}^{N \times d}$ is partitioned into $C = \lceil N/w \rceil$ chunks, with each chunk containing at most $w$ tokens:

$$A^c_\mathfrak{l} = \text{Softmax}\left(\frac{f_Q(X^c) \cdot f_K(X^c)^T}{\sqrt{d}}\right)$$

Local attention is computed independently within each chunk, reusing positional encodings to achieve training-free extrapolation.

### 2. Chunk Eviction

The most relevant chunks are selected based on the self-information score of the query token:

$$I_c = -\log P(X^q \mid X^c)$$

A fixed-size priority queue is utilized to retain the chunks with the lowest scores (i.e., most relevant), thereby controlling memory usage during the pre-filling phase.

### 3. Parallel KV Cache Eviction

Prior to local attention computation, cumulative attention scores are leveraged to rapidly identify and evict tokens of low importance:

$$S_{c,j} = \sum_{i=1}^{w_q} A^c_{\mathfrak{l}(i,j)}, \quad j=1,2,...,w$$

By retaining the KV cache of high-scoring tokens, the computational overhead of subsequent global attention is significantly reduced.

### 4. Attention Calibration

**Key Novelty**: Evicting tokens with **abnormally high** attention scores (rather than merely retaining high-scoring tokens) to mitigate the attention bias exacerbated by parallel KV cache eviction:

$$K^h_{r'} = K^h_x[R^h_H], \quad V^h_{r'} = V^h_x[R^h_H]$$

where $R^h_H$ represents the set of tokens whose attention scores exceed the threshold $\lambda$.

### 5. Theoretical Analysis

**Theorem 3.1**: This theorem formalizes the inevitability of attention collapse in parallel attention. As the input length increases, the number of effective entries in the local attention matrix decreases:

$$k \leq w - \exp\left(O\left(\frac{\log^2(\epsilon \cdot w)}{R^2}\right)\right) \cdot \frac{\delta}{wd}$$

### 6. Three Attention Patterns

Three attention distributions are empirically identified:
- **U-shape**: Attention is concentrated on the first and last tokens (attention sink + recency bias).
- **Mountain-shape**: Attention is concentrated on a small number of tokens in the middle (middle bias).
- **Uniform-shape**: Attention is uniformly distributed.

## Experimental Results

### Main Results: Long-Context Benchmarks

- **Performance**: The 8B model (trained on 8K context length) achieves **91.17%** of GPT-4's performance, surpassing Claude-2 and Kimi-Chat.
- **Efficiency**: Chunk throughput is increased by **1.76×**, and the pre-filling phase is accelerated by **23.50×**.
- Performs better on average than baselines like InfLLM across 16 sub-tasks in LongBench.
- Supports processing 128K+ context length on a single A100 80GB GPU.

### Ablation Study

| Component | Performance Change Upon Removal |
|------|--------------|
| Attention Calibration | Performance drops by 2-5% across multiple tasks |
| Chunk Eviction | Out of Memory (OOM) / Execution Failure |
| Parallel KV Eviction | Throughput drops significantly |

## Highlights & Insights

- Provides the first systematic analysis of the unique attention bias patterns in parallel attention.
- The counter-intuitive attention calibration strategy (evicting high-scoring tokens) yields significant improvements.
- Combines theoretical and empirical analysis to establish a formalized bound on attention collapse.
- Highly practical, achieving 128K context extrapolation on a single GPU without requiring training.

## Limitations & Future Work

- The attention calibration threshold $\lambda$ requires manual setting and lacks an adaptive mechanism.
- Chunk eviction may lose critical information, particularly in scenarios where information is highly dispersed.
- The theoretical analysis assumes a fixed number of chunks $C$, which may not hold in practical applications.
- Compared to training-based methods (such as Yarn and LongRoPE), there remains a gap in the extrapolation range.

## Rating

⭐⭐⭐⭐ — The addressed problem is significant and the proposed solution is highly practical. The analysis of parallel attention bias is novel and deep. It successfully achieves significant length extrapolation without requiring training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [\[ICML 2025\] Context Tuning for In-Context Optimization](context_tuning_for_in-context_optimization.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](../../ACL2025/model_compression/efficient_long_context_language_model_retrieval_with_compression.md)

</div>

<!-- RELATED:END -->
