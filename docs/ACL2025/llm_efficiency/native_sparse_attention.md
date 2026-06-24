---
title: >-
  [Paper Note] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
description: >-
  [ACL 2025][LLM Efficiency][Sparse Attention] DeepSeek proposes NSA—a natively trainable hierarchical sparse attention mechanism that achieves efficient long-context modeling through three parallel attention paths: token compression, token selection, and sliding window. After pre-training on a 27B parameter model, its performance matched or even surpassed Full Attention across all metrics, while delivering significant acceleration on 64k sequences.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Sparse Attention"
  - "Long-Context Modeling"
  - "Hardware Alignment"
  - "End-to-End Training"
  - "KV Cache Compression"
date: 2026-05-08
content_hash: f5a897ad2fa199ad
---

# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**Conference**: ACL 2025  
**arXiv**: [2502.11089](https://arxiv.org/abs/2502.11089)  
**Code**: None (DeepSeek internal implementation)  
**Area**: LLM Efficiency / Attention Mechanism  
**Keywords**: Sparse Attention, Long-Context Modeling, Hardware Alignment, End-to-End Training, KV Cache Compression

## TL;DR

DeepSeek proposes NSA—a natively trainable hierarchical sparse attention mechanism that achieves efficient long-context modeling through three parallel attention paths: token compression, token selection, and sliding window. After pre-training on a 27B parameter model, its performance matched or even surpassed Full Attention across all metrics, while delivering significant acceleration on 64k sequences.

## Background & Motivation

**Background**: Long-context modeling is a critical capability for next-generation LLMs. However, the $O(n^2)$ computational complexity of standard attention mechanisms becomes a latency bottleneck in long sequences—during the decoding phase of a 64k context, attention computation accounts for 70-80% of the total latency. Existing sparse attention methods (such as H2O, Quest, MInference) are primarily applied during the inference phase.

**Limitations of Prior Work**: (1) **Illusion of Speedup**: Many methods reduce theoretical FLOPs but offer limited practical latency reduction because the algorithm designs do not align with hardware characteristics (e.g., scattered memory accesses clash with the GQA architecture); (2) **Lack of Trainability**: Most methods apply sparsity only during inference. Discrete operations (such as clustering or hashing) block the gradient flow, and token-level selection leads to non-contiguous memory accesses, preventing the use of FlashAttention.

**Key Challenge**: A fundamental gap exists in current sparse attention methods regarding both hardware acceleration and end-to-end training—they either function only during specific inference stages or cannot be integrated into training.

**Goal**: To design a sparse attention mechanism that is both hardware-aligned for efficient inference and end-to-end trainable, achieving acceleration throughout the pre-training, fine-tuning, and inference lifecycle.

**Key Insight**: Leveraging the spatial continuity of attention scores (blockwise clustering), block-level token compression and selection strategies are designed, and the kernel implementation is optimized in conjunction with hardware characteristics (Tensor Cores, GQA-shared KV cache).

**Core Idea**: Deconstructing attention into three paths: compression (global coarse-grained), selection (local fine-grained), and sliding window (short-distance), achieving a sparse attention that is trainable throughout its lifecycle via a hardware-aligned blockwise design.

## Method

### Overall Architecture

NSA replaces standard full attention with three parallel attention branches: (1) **Compression Attention**—aggregating the KV sequence into coarse-grained blocks to capture global information; (2) **Selection Attention**—selecting the most crucial fine-grained token blocks based on compression scores to retain local precision; (3) **Sliding Window**—maintaining short-distance local context. The three branches are weighted and fused through a learnable gating mechanism (MLP + sigmoid) and utilize independent KVs to prevent shortcut learning.

### Key Designs

1. **Token Compression**
    - Function: Compresses contiguous token blocks into a single coarse-grained KV representation.
    - Mechanism: Uses a learnable MLP (with intra-block positional encodings) to map critical KV blocks of length $l$ to a single compressed token, adopting a sliding stride $d < l$ to avoid information fragmentation.
    - Design Motivation: Coarse-grained representations cover global context at an extremely low computational cost.

2. **Block-based Token Selection**
    - Function: Selects the most relevant fine-grained token blocks from the entire sequence for the current query.
    - Mechanism: Reuses intermediate attention scores from compression attention as block importance scores (zero overhead), aggregates scores across all query heads within a GQA group to ensure shared selection, and selects the top-n most important blocks.
    - Design Motivation: Using only compressed tokens would lose fine-grained details; blockwise selection (instead of token-level) balances hardware efficiency with the spatial continuity of attention scores.

3. **Independent Sliding Window Branch**
    - Function: Explicitly processes local context to prevent local patterns from short-circuiting the learning of other branches.
    - Mechanism: Maintains a window of the nearest $w$ tokens, isolated from the calculation of the compression and selection branches, and fuses them via gating.
    - Design Motivation: Without isolation, local patterns would dominate the learning process, hindering the model from acquiring long-range compression and selection capabilities.

### Loss & Training

- Pre-training: A 27B parameter (3B active) MoE model, pre-trained on 270B tokens with an 8k length, followed by a 32k length YaRN adaptation for long context.
- The three branches employ independent KV projections, weighted and fused with a gate score $g_t^c \in [0,1]$ (sigmoid activation).
- During the training phase, NSA directly replaces Full Attention for end-to-end training, yielding a stable loss curve that remains consistently lower than Full Attention.
- Inference Optimization: Tailored Triton kernel with a group-centric data loading strategy—each time loading sparse KV blocks shared by all query heads in the GQA group to maximize Tensor Core utilization.

## Key Experimental Results

### Main Results

| Benchmark | Full Attention | NSA |
|------|---------------|-----|
| MMLU (5-shot) | 0.567 | 0.565 |
| BBH (3-shot) | 0.497 | 0.521 |
| GSM8K (8-shot) | 0.486 | 0.520 |
| DROP (1-shot) | 0.503 | 0.545 |
| HumanEval (0-shot) | 0.335 | 0.348 |
| **Overall Average** | **0.443** | **0.456** |
| LongBench Average | 0.437 | **0.469** |
| AIME 8k | 0.046 | **0.121** |
| AIME 16k | 0.092 | **0.146** |

### Ablation Study

| Method | LongBench Average |
|------|-------------|
| H2O | 0.303 |
| InfLLM | 0.383 |
| Quest | 0.392 |
| Exact-Top | 0.423 |
| Full Attention | 0.437 |
| **NSA** | **0.469** |

### Key Findings

- NSA outperforms Full Attention in 7 out of 9 general benchmarks, with significant gains in reasoning-related tasks (DROP: +0.042, GSM8K: +0.034).
- On LongBench, NSA surpasses Full Attention (+0.032) and all inference-only sparse methods.
- On 64k sequences, training acceleration reaches up to 9.0× (forward) / 6.0× (backward), and decoding acceleration reaches up to 11.6×.
- On the AIME mathematical reasoning task, NSA-R is significantly superior to Full Attention-R, validating the compatibility of sparse attention with advanced reasoning.
- NSA achieves perfect accuracy in the 64k Needle-in-a-Haystack test.

## Highlights & Insights

- **Counter-intuitive Advantage of Sparse Attention**: Rather than degrading performance, NSA outperforms Full Attention on reasoning tasks—suggesting that sparsity may act as "attention regularization," filtering out irrelevant noise.
- The design of **reusing compression scores for selection** is highly elegant, enabling importance evaluation with zero additional overhead.
- The systematic analysis of prior methods (Phase-Restricted Sparsity, GQA compatibility, non-differentiable gradient issues) is highly insightful, shedding light on future research directions.
- Pioneered the introduction of sparse attention directly into the pre-training phase and proved its effectiveness, rather than treating it merely as an inference post-processing step.

## Limitations & Future Work

- Evaluated only on a 27B MoE model; performance on larger-scale models (e.g., 100B+) remains to be validated.
- Sensitivity analysis for key hyperparameters such as compression block size and the number of selected blocks is insufficient.
- Currently, the kernel is optimized only for A100 GPUs; adaptation to other GPU architectures (e.g., H100, AMD MI300X) is unknown.
- The source code is not open-sourced, casting doubt on its reproducibility.
- The sliding window size is fixed (512 tokens); dynamic adjustment might yield further improvements.

## Related Work & Insights

- Orthogonal to FlashAttention: While FlashAttention optimizes the I/O efficiency of standard attention, NSA programmatically reduces the number of KV pairs to be computed from an algorithmic level.
- Relationship with StreamingLLM (attention sink + sliding window): NSA generalizes the concept of "sinks" into learnable compressed tokens.
- Insights: The future direction of sparse attention should pivot toward "native training" instead of "inference-time post-processing," allowing the model to learn the optimal sparsity patterns during pre-training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First hardware-aligned sparse attention trainable throughout the entire lifecycle)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive evaluation across three dimensions: general, long-context, and reasoning, but hyperparameter analysis is slightly lacking)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (In-depth motivation analysis, clear methodological exposition, and precise tables/figures)
- **Value**: ⭐⭐⭐⭐⭐ (Potentially alters the paradigm of attention design for long-context LLMs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ACL 2025\] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)
- [\[ICLR 2026\] Retrospective Sparse Attention for Efficient Long-Context Generation](../../ICLR2026/llm_efficiency/retrospective_sparse_attention_for_efficient_long-context_generation.md)
- [\[NeurIPS 2025\] The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition](../../NeurIPS2025/llm_efficiency/the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)

</div>

<!-- RELATED:END -->
