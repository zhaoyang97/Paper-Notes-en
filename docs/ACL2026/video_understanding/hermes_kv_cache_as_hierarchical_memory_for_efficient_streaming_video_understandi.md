---
title: >-
  [Paper Note] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
description: >-
  [ACL 2026][Video Understanding][Streaming Video] This paper proposes HERMES, based on a mechanistic analysis of hierarchical attention preferences in MLLM decoders. It conceptualizes the KV cache as a hierarchical memory…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Streaming Video"
  - "KV Cache Management"
  - "Hierarchical Memory"
  - "Real-time Response"
  - "Training-free"
date: 2026-05-08
content_hash: 53f60520b008394c
---

# HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

**Conference**: ACL 2026  
**arXiv**: [2601.14724](https://arxiv.org/abs/2601.14724)  
**Code**: [GitHub](https://github.com/haowei-freesky/HERMES)  
**Area**: Video Understanding / Streaming Inference  
**Keywords**: Streaming Video, KV Cache Management, Hierarchical Memory, Real-time Response, Training-free

## TL;DR

This paper proposes HERMES, based on a mechanistic analysis of hierarchical attention preferences in MLLM decoders. It conceptualizes the KV cache as a hierarchical memory framework (shallow layers = sensory memory, middle layers = working memory, deep layers = long-term memory). It achieves training-free efficient streaming video understanding, maintaining or improving accuracy while reducing video tokens by 68%. The TTFT latency is <30ms, which is 10 times faster than the previous SOTA.

## Background & Motivation

**Background**: MLLMs have made significant progress in offline video understanding, but extending them to streaming video inputs faces challenges in simultaneously maintaining understanding performance, real-time response, and low GPU memory overhead. Existing streaming methods are divided into external memory (storing video content as descriptions or patches, retrieving them during query) and internal memory (managing KV cache directly).

**Limitations of Prior Work**: (1) External memory methods require retrieval and multimodal pre-filling when a query arrives, leading to high latency and a lack of end-to-end coherence; (2) Caching methods like ReKV and LiveVLM offload video segments to CPU/disk, requiring additional retrieval operations during query, which still results in significant latency; (3) Existing methods use coarse-grained eviction strategies (such as applying FIFO uniformly to all layers), ignoring the differences in attention preferences across different layers.

**Key Challenge**: KV cache is naturally an internal latent memory of the model, suitable for training-free management in streaming scenarios. However, existing methods fail to utilize the differences in inter-layer attention patterns—different layers "remember" video information in different ways.

**Goal**: Design a KV cache management method based on hierarchical attention analysis that can be plugged into existing MLLMs without training to achieve true real-time streaming video QA.

**Key Insight**: Attention visualization analysis of the 28-layer decoder of LLaVA-OV-7B reveals three distinct hierarchical memory patterns.

**Core Idea**: Shallow layers exhibit a strong recency bias (sensory memory), managed by exponential decay; deep layers focus on frame-level "anchor tokens" (long-term memory), managed by attention weights; middle layers transition between the two (working memory), managed by interpolation. Cross-layer smoothing and position re-indexing are added to ensure consistency.

## Method

### Overall Architecture

HERMES consists of three components: (1) **Hierarchical KV Cache Management**—uses different token importance scoring and eviction strategies based on the layer type (shallow/middle/deep); (2) **Cross-Layer Memory Smoothing**—prevents cross-layer inconsistency caused by independent layer eviction; (3) **Position Re-Indexing**—remaps position encodings after eviction to maintain continuity. During inference, the compressed KV cache is directly reused, requiring no extra computation when the user asks a question.

### Key Designs

1.  **Hierarchical KV Cache Management**:

    - **Function**: Implements differentiated token eviction strategies based on the attention characteristics of different layers.
    - **Mechanism**: Shallow layers use an exponential forgetting curve $S_i^l = \alpha_i^l \cdot e^{-k\Delta t_i}$ to evaluate token importance (newer is more important); deep layers use attention weights $S_i^l = \alpha_i^l \cdot W_i^l$ (based on attention from pseudo-queries); middle layers use layer-dependent weights $\omega_l$ to interpolate between recency and attention scores $S_i^l = (1-\omega_l) A_i^l + \omega_l R_i^l$.
    - **Design Motivation**: Attention visualization clearly shows that different layers have different memory functions—uniform FIFO or attention eviction strategies cannot satisfy the requirements of all layers simultaneously.

2.  **Cross-Layer Memory Smoothing**:

    - **Function**: Prevents inconsistent retention of the same token across different layers due to independent eviction.
    - **Mechanism**: Shares partial eviction decisions between adjacent layers to ensure that the retention/eviction of the same video token across multiple layers has a certain degree of consistency.
    - **Design Motivation**: Independent hierarchical management would lead to fragmented visual memory—information from the same frame being retained in some layers but evicted in others, disrupting the coherence of end-to-end inference.

3.  **Position Re-Indexing**:

    - **Function**: Remaps position encodings after token eviction to maintain the correctness of RoPE.
    - **Mechanism**: After each eviction, the positions of retained tokens are re-indexed to a continuous range $[0, |M|)$, avoiding attention calculation anomalies caused by discontinuous position encodings.
    - **Design Motivation**: Directly deleting intermediate tokens leads to jumps in position encodings, affecting the position-based attention mechanism.

### Loss & Training

A completely training-free method. Designed based on the Ebbinghaus forgetting curve theory and hierarchical memory models from cognitive psychology. Uses a generic guidance prompt as a pseudo-query to calculate deep-layer attention weights.

## Key Experimental Results

### Main Results

**Streaming Video Benchmarks (LLaVA-OV-7B)**

| Method | StreamingBench | EgoSchema | MVBench | Video-MME | Average |
|------|---------------|-----------|---------|-----------|------|
| Full (No Comp.) | 53.2 | 58.1 | 69.3 | 61.8 | 60.6 |
| Prev. SOTA (ReKV) | 51.8 | 55.2 | 67.1 | 59.4 | 58.4 |
| StreamMem | 52.1 | 56.8 | 68.5 | 60.1 | 59.4 |
| **Ours (HERMES)** | **59.3** | **58.9** | **69.8** | **62.4** | **62.6** |

### Ablation Study

**Efficiency Comparison (Single A800 GPU)**

| Method | TTFT (ms) | GPU Memory | Token Reduction |
|------|----------|---------|-----------|
| Full | ~3000+ | Linear growth | 0% |
| ReKV | ~1500 | Requires CPU mem | ~50% |
| **Ours (HERMES)** | **<30** | **Constant** | **68%** |

### Key Findings

- While reducing video tokens by 68%, HERMES actually improves performance on streaming benchmarks by 11.4% (Gain)—proving that removing redundant tokens improves inference quality.
- TTFT < 30ms and constant GPU memory, with no risk of OOM as the number of input frames increases—zero additional computation when a query arrives.
- The hierarchical memory model generalizes across multiple MLLMs—it is not limited to LLaVA-OV.
- The recency bias of shallow attention aligns with the Ebbinghaus forgetting curve, and the interval of anchor patterns in deep attention exactly equals the number of tokens per frame (196).

## Highlights & Insights

- The concept of hierarchical memory borrowed from cognitive psychology corresponds precisely to the attention patterns of Transformer layers—this is not just an analogy but a finding supported by quantitative attention analysis.
- The design of zero additional latency is critical for real-time applications—methods like ReKV reduce storage but still require retrieval during queries.
- The training-free and plug-and-play characteristics allow it to be directly applied to existing MLLMs, lowering the barrier for practical use.

## Limitations & Future Work

- The division of hierarchical boundaries (shallow/middle/deep) depends on the analysis of specific models; different architectures might require re-determination.
- Using pseudo-queries instead of real user queries might introduce bias in specific scenarios.
- Only verified in video streaming scenarios; applicability to text or multimodal streaming has not been explored.
- The exponential forgetting rate $k$ and interpolation parameters need to be set manually.

## Related Work & Insights

- **vs ReKV/LiveVLM**: These require CPU offloading and retrieval operations, resulting in high latency; HERMES directly reuses the KV cache on the GPU.
- **vs StreamMem**: Uses chat template tokens to guide compression but lacks fine-grained management; HERMES achieves precise management based on hierarchical attention analysis.
- **vs StreamingLLM**: The attention sink mechanism retains initial tokens but ignores inter-layer differences; HERMES uses hierarchical specialization for smarter eviction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Hierarchical memory conceptualization and differentiated management strategies based on attention analysis are very novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes multiple streaming benchmarks, efficiency analysis, attention visualization, and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from mechanistic analysis to method design is very clear.
- Value: ⭐⭐⭐⭐⭐ A practical solution for real-time streaming video understanding, with 10x TTFT acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](../../NeurIPS2025/video_understanding/infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](../../ICCV2025/video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)

</div>

<!-- RELATED:END -->
