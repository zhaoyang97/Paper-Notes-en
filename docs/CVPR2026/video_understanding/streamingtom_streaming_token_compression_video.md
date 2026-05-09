---
title: >-
  [Paper Note] StreamingTOM: Streaming Token Compression for Efficient Video Understanding
description: >-
  [CVPR 2026][Video Understanding][streaming video understanding] The first training-free framework to simultaneously address both pre-LLM prefill and post-LLM KV-cache efficiency bottlenecks in streaming video VLMs, achieving 15.7× compression with bounded active memory.
tags:
  - CVPR 2026
  - Video Understanding
  - streaming video understanding
  - token compression
  - KV-cache optimization
  - causal temporal reduction
  - 4-bit quantized memory
date: 2026-05-08
content_hash: 5dbf2cf27a85970c
---

# StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**Conference**: CVPR 2026  
**arXiv**: [2510.18269](https://arxiv.org/abs/2510.18269)  
**Code**: [Project Page](https://yige24.github.io/StreamingTOM)  
**Area**: Multimodal VLM / Video Understanding  
**Keywords**: streaming video understanding, token compression, KV-cache optimization, causal temporal reduction, 4-bit quantized memory  

## TL;DR

The first training-free framework to simultaneously address both pre-LLM prefill and post-LLM KV-cache efficiency bottlenecks in streaming video VLMs, achieving 15.7× compression with bounded active memory.

## Background & Motivation

**Streaming video understanding is fundamentally different from offline processing**, facing two unique constraints: (1) *Causality*: only past frames are accessible; future frames cannot be utilized. (2) *Accumulation*: token counts grow unboundedly over time, causing continuous degradation in memory and latency. For instance, processing a 1-hour video with LLaVA-OV-7B yields an 18.8 GB KV-cache, far exceeding GPU capacity.

**Existing training-free methods address only the post-LLM KV-cache** (e.g., eviction strategies) while completely ignoring the pre-LLM prefill overhead — every visual token $N$ per frame must pass through a full transformer forward pass, which is the dominant source of latency. More critically, existing offline token compression methods rely on global or future frame information, **violating the causal constraint inherent to streaming scenarios**.

**Therefore, the joint optimization of causal pre-LLM token reduction and post-LLM memory management remains an unexplored key problem.** The core insight is that effective streaming compression must occur before the LLM under strict causal constraints — post-LLM methods cannot reduce the prefill computation already incurred.

## Method

### Overall Architecture

A two-stage training-free framework: Stage 1, Causal Temporal Reduction (CTR), addresses the pre-LLM bottleneck by reducing each frame's $N$ tokens to a fixed budget of $G$ tokens; Stage 2, Online Quantized Memory (OQM), addresses the post-LLM bottleneck by storing the KV-cache in 4-bit format and retrieving it on demand. The two stages are coordinated through a frame-aligned group abstraction.

### Key Designs

1. **Causal Temporal Reduction (CTR)**:

    - **Function**: Compresses each frame's visual tokens from $N$ to a fixed budget $G$ under strict causal constraints.
    - **Mechanism**: Uses only a two-frame sliding window; tokens are divided into static and dynamic sets via cosine similarity, with the budget allocated proportionally. Static tokens are merged via DPC clustering; dynamic tokens are selected by attention saliency.
    - **Design Motivation**: A fixed budget $G$ guarantees predictable latency; adaptive allocation preserves more tokens in high-motion scenes and compresses more in static scenes.

2. **Online Quantized Memory (OQM)**:

    - **Function**: Stores the post-LLM KV-cache in 4-bit format and dequantizes retrieved entries on demand.
    - **Mechanism**: Preserves the frame-aligned group structure (each group of $G$ tokens corresponds to one frame). At query time, the $k$ most relevant groups are retrieved and dequantized to FP16 for attention computation. The active KV-cache is bounded and does not grow with video length.
    - **Design Motivation**: 4-bit quantization reduces storage by 4×; group-level retrieval maintains temporal integrity and avoids token fragmentation.

3. **Unified Compression Ratio Analysis**:

    - **Function**: Quantifies end-to-end compression effectiveness.
    - **Mechanism**: Prefill complexity is reduced from $O(TNLd^2)$ to $O(TGLd^2)$; storage is reduced from $O(TN \cdot d \cdot 16)$ bits to $O(TG \cdot d \cdot 4)$ bits, yielding a combined compression ratio of $4N/G \approx 15.7\times$ (with $N=196, G=50$).
    - **Design Motivation**: The budget $G$ simultaneously controls computation and storage, achieving dual compression.

### Loss & Training

A fully training-free method that can be directly applied to existing VLMs (e.g., LLaVA-OV-7B).

## Key Experimental Results

### Main Results

| Metric | StreamingTOM | Prev. SOTA (LiveVLM) | Gain |
|------|-------------|-------------------|------|
| KV-cache compression ratio | 15.7× | — | — |
| Peak memory | — | — | 1.2× lower |
| TTFT | — | — | 2× faster |
| Offline average accuracy | 63.8% | ~61% | +2.8% |
| RVS accuracy | 55.8% | ~54% | +1.8% |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| CTR only | Memory uncontrolled | Prefill accelerated but KV-cache remains unbounded |
| OQM only | Latency unchanged | Storage compressed but prefill unaffected |
| CTR + OQM | Both optimized | Both stages are indispensable |
| Varying budget $G$ | $G=50$ optimal | Lower $G$ degrades accuracy; higher $G$ insufficient compression |

### Key Findings

- KV-cache for a 1-hour video is reduced from 18.8 GB to 1.2 GB; bounded growth makes infinite streaming video theoretically feasible.
- The dual-path design in CTR is critical: pure clustering or pure selection each underperforms the hybrid strategy.
- Achieves training-free SOTA simultaneously on both offline and streaming benchmarks.

## Highlights & Insights

- The core contribution lies in identifying the insight that "the pre-LLM and post-LLM stages constitute two independent bottlenecks that must be addressed separately." The frame-aligned group abstraction bridges the two stages, enabling token reduction and storage optimization to be decoupled yet coordinated — an elegant design with practical significance.

## Limitations & Future Work

- 4-bit quantization may introduce quality loss in scenarios demanding extreme precision.
- The cosine similarity-based classification over adjacent frames may miss critical changes in fast-motion scenes.
- The fixed budget $G$ does not adapt to content complexity.
- The effectiveness of CTR/OQM as plug-and-play modules has not been validated on training-based methods.

## Related Work & Insights

- **vs. LiveVLM**: Addresses only KV-cache management (post-LLM); StreamingTOM is the first to also optimize at the pre-LLM level.
- **vs. FastV/TokenPacker**: Designed for single images or offline video; require global information and thus do not satisfy the causal constraint of streaming scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ First to simultaneously resolve dual-level efficiency bottlenecks; the combination of causal token reduction and 4-bit quantized memory is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Achieves SOTA on both offline and streaming benchmarks with comprehensive efficiency metrics.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and mathematical derivations are complete.
- Value: ⭐⭐⭐⭐⭐ Addresses core deployment pain points for streaming video VLMs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)

<!-- RELATED:END -->
