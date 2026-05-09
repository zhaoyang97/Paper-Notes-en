---
title: >-
  [Paper Note] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training
description: >-
  [ICLR 2026][LLM Efficiency][Linear Attention] This paper proposes RACE Attention — replacing softmax with a power angular kernel and approximating attention outputs via differentiable LSH sketches — achieving strictly linear time complexity, supporting up to 12M tokens on a single GPU and 75M tokens on a single CPU, while matching or surpassing softmax accuracy across diverse tasks.
tags:
  - ICLR 2026
  - LLM Efficiency
  - Linear Attention
  - LSH
  - Angular Kernel
  - Long-Sequence Training
  - Attention Approximation
date: 2026-05-08
content_hash: 31fc9d534fae7f27
---

# RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training

**Conference**: ICLR 2026
**arXiv**: [2510.04008](https://arxiv.org/abs/2510.04008)
**Code**: [https://github.com/sahiljoshi515/RACE_Attention](https://github.com/sahiljoshi515/RACE_Attention)
**Area**: LLM Efficiency / Attention Mechanisms
**Keywords**: Linear Attention, LSH, Angular Kernel, Long-Sequence Training, Attention Approximation

## TL;DR
This paper proposes RACE Attention — replacing softmax with a power angular kernel and approximating attention outputs via differentiable LSH sketches — achieving strictly linear time complexity, supporting up to 12M tokens on a single GPU and 75M tokens on a single CPU, while matching or surpassing softmax accuracy across diverse tasks.

## Background & Motivation

**Background**: The $O(N^2 d)$ complexity of softmax attention is a fundamental bottleneck for long-context training. Even with FlashAttention-2/3 optimizations, a single layer on GH200 cannot process more than ~4M tokens.

**Limitations of Prior Work**: Linear attention (Linear Attention, Performer) suffers from accuracy degradation; low-rank approximations (Linformer) do not support autoregressive inference; YOSO employs hard LSH but lacks theoretical guarantees and does not support causal language modeling.

**Key Challenge**: Existing approximation methods lack a rigorous mathematical framework for characterizing the efficiency–accuracy trade-off, leading to ad hoc design choices that are unstable across tasks.

**Goal**: Design a strictly linear-time attention mechanism with theoretical guarantees, supporting both causal and non-causal settings and scaling to tens of millions of tokens.

**Key Insight**: The LSH collision probability of angular kernels equals the angular similarity exactly, and RACE sketches enable unbiased estimation of kernel density sums in linear time.

**Core Idea**: Replace softmax with a power angular kernel and approximate attention in $O(N)$ via differentiable RACE sketches.

## Method

### Overall Architecture
Softmax is replaced by the angular kernel $(1 - \frac{\arccos(\hat{q}_i \cdot \hat{k}_j)}{\pi})^\gamma$. Rather than constructing the attention matrix, keys and values are hashed into $S = L \times R$ buckets, and queries aggregate per-bucket statistics at inference time.

### Key Designs

1. **Power Angular Kernel**: Angular similarity raised to the $\gamma$-th power replaces softmax; larger $\gamma$ yields sharper attention. Since the LSH collision probability equals angular similarity, RACE sketch theory applies directly.

2. **Differentiable RACE Sketch**: Hard SimHash is replaced by sigmoid-based soft assignment, preserving approximation quality while enabling gradient-based training. Averaging over $L$ independent hash tables reduces variance.

3. **Causal RACE**: Causal bucket counters are maintained via prefix-sum streaming, supporting autoregressive language modeling.

### Loss & Training
RACE Attention serves as a drop-in replacement for softmax attention, trained with standard cross-entropy loss. $L$ (number of hash tables) and $R$ (number of buckets) govern the variance–accuracy trade-off.

## Key Experimental Results

### Main Results

| Method | Complexity | 64K Support | Accuracy |
|--------|-----------|-------------|----------|
| Softmax (FA2) | $O(N^2)$ | OOM | Baseline |
| Linear Attn | $O(N)$ | ✓ | Poor |
| Performer | $O(Nd^2)$ | Partial | Poor |
| **RACE** | **$O(N)$** | **✓** | **≈ Baseline** |

### Scalability

| Hardware | Softmax Max | RACE Max |
|----------|------------|----------|
| GH200 (96GB) | ~4M | **12M** |
| CPU | N/A | **75M** |

### Key Findings
- RACE achieves lower wall-clock time than FlashAttention-2 at 64K with comparable accuracy.
- RACE outperforms Linformer in accuracy while using 13% fewer parameters.
- The $\gamma$ parameter controls sharpness; excessively large values increase variance, requiring more hash tables to compensate.
- CPU training support opens new possibilities for long-context training without GPUs.

## Highlights & Insights
- **Elegant theoretical chain**: angular kernel → LSH collision probability → RACE sketch → linear-time attention, with theoretical guarantees at every step.
- **Truly linear time** with small constants — $S = L \times R$ buckets rather than $N$ keys, yielding substantial practical speedups.
- Training on 75M tokens on CPU is a distinctive contribution, decoupling long-context research from GPU availability.

## Limitations & Future Work
- Validated only on ~120M-parameter models; effectiveness at larger scales remains unknown.
- $\gamma$, $L$, and $R$ require manual tuning; no automatic selection strategy is currently available.
- Integration with sparse attention is a promising future direction.

## Related Work & Insights
- **vs. FlashAttention**: FA optimizes constant factors but retains $O(N^2)$ complexity; RACE achieves genuine $O(N)$.
- **vs. YOSO**: Both employ angular kernels with LSH, but RACE uses soft LSH with theoretical guarantees and supports causal modeling.
- **vs. Performer**: Performer is quadratic in embedding dimension and exhibits inferior accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The combination of angular kernel and RACE sketch constitutes a wholly novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task evaluation, scalability analysis, and theoretical support; large-model validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and well-organized.
- Value: ⭐⭐⭐⭐⭐ — Significant practical value for ultra-long-context training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/llm_efficiency/linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](../../ACL2026/llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](../../NeurIPS2025/llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)

<!-- RELATED:END -->
