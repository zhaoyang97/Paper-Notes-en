---
title: >-
  [Paper Note] Native Hybrid Attention for Efficient Sequence Modeling
description: >-
  [ACL 2026][LLM Efficiency][Hybrid Attention] NHA concatenates linear RNN long-term memory slots with sliding window short-term tokens for unified softmax processing, achieving native intra-layer and inter-layer hybridization without extra fusion parameters.
tags:
  - ACL 2026
  - LLM Efficiency
  - Hybrid Attention
  - Linear Attention
  - Sliding Window
  - Long-Short Memory Fusion
content_hash: 4c45e7d899071930
---

# Native Hybrid Attention for Efficient Sequence Modeling

**Conference**: ACL 2026
**arXiv**: [2510.07019](https://arxiv.org/abs/2510.07019)
**Code**: [GitHub](https://github.com/JusenD/NHA)
**Area**: LLM Efficiency / Attention Mechanism
**Keywords**: Hybrid Attention, Linear Attention, Sliding Window, Long-Short Memory Fusion, Efficient Sequence Modeling

## TL;DR
Native Hybrid Attention (NHA) concatenates linear RNN long-term memory slots with sliding window short-term precise tokens and processes them through a single softmax attention, achieving native intra-layer and inter-layer hybridization — dynamically allocating long-short attention weights without extra fusion parameters, outperforming Transformer and other hybrid baselines on recall-intensive and commonsense reasoning tasks.

## Method

### Key Designs

1. **Intra-Layer Hybrid — Unified Softmax Fusion**: Long-term memory via gated linear RNN concatenated with sliding window KV cache, processed by single softmax. Weights are query-key similarity dependent — achieving per-token, per-head context-aware weighting with zero extra parameters.

2. **Inter-Layer Hybrid — Window Size Tuning**: All NHA layers share the same architecture; only window size $w$ controls behavior ($w=0$ = pure linear RNN, $w=N$ = full attention). Supports zero-cost inference-time architecture search.

3. **Chunkwise Parallel Computation**: Efficient GPU implementation via Triton kernels maintaining near-linear complexity.

## Key Experimental Results

| Model | Commonsense Avg↑ | Recall-Dense Avg↑ |
|-------|-----------------|-------------------|
| Trans++ | 50.71 | 37.31 |
| GSA-H | 50.76 | 44.99 |
| **NHA** | **52.89** | **46.43** |

## Highlights & Insights
- Unified softmax fusion is the core innovation — demoting fusion from explicit parameter learning to implicit softmax allocation
- "Architecture duality" is highly practical — same model can zero-cost switch between different efficiency-accuracy configurations at inference time

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/llm_efficiency/linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[ACL 2026\] BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](../../NeurIPS2025/llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[NeurIPS 2025\] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](../../NeurIPS2025/llm_efficiency/jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)

<!-- RELATED:END -->
