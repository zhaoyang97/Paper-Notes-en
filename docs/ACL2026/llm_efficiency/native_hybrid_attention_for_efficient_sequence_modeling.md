---
title: >-
  [Paper Note] Native Hybrid Attention for Efficient Sequence Modeling
description: >-
  [ACL 2026][LLM Efficiency][Hybrid Attention] Native Hybrid Attention (NHA) concatenates linear RNN long-term memory slots with sliding window short-term precise tokens and processes them through a single softmax attentio…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Hybrid Attention"
  - "Linear Attention"
  - "Sliding Window"
  - "Long-Short Memory Fusion"
  - "Efficient Sequence Modeling"
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

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](../../NeurIPS2025/llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ACL 2026\] Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs](saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)

</div>

<!-- RELATED:END -->
