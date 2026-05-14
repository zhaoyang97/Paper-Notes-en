---
title: >-
  [Paper Note] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
description: >-
  [ACL 2026][Video Understanding][Streaming Video] This paper proposes HERMES, a training-free framework for efficient streaming video understanding grounded in a mechanistic analysis of layer-wise attention preferences in…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Streaming Video"
  - "KV Cache Management"
  - "Hierarchical Memory"
  - "Real-Time Response"
  - "Training-Free"
date: 2026-05-08
content_hash: 329417ed57b45a97
---

# HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

**Conference**: ACL 2026
**arXiv**: [2601.14724](https://arxiv.org/abs/2601.14724)
**Code**: [GitHub](https://github.com/haowei-freesky/HERMES)
**Area**: Video Understanding / Streaming Inference
**Keywords**: Streaming Video, KV Cache Management, Hierarchical Memory, Real-Time Response, Training-Free

## TL;DR

This paper proposes HERMES, a training-free framework for efficient streaming video understanding grounded in a mechanistic analysis of layer-wise attention preferences in MLLMs. KV caches are conceptualized as a hierarchical memory system — shallow layers as sensory memory, middle layers as working memory, and deep layers as long-term memory — enabling real-time streaming video QA with a 68% reduction in video tokens while maintaining or improving accuracy, achieving TTFT latency below 30ms, which is 10× faster than the previous SOTA.

## Background & Motivation

**State of the Field**: MLLMs have achieved notable progress in offline video understanding, yet extending them to streaming video inputs remains challenging, as the system must simultaneously maintain comprehension performance, real-time responsiveness, and low GPU memory overhead. Existing streaming approaches fall into two categories: external memory (storing video content as captions or patches and retrieving at query time) and internal memory (managing information directly within the KV cache).

**Limitations of Prior Work**: (1) External memory methods incur high latency due to retrieval and multimodal prefilling upon query arrival, lacking end-to-end coherence; (2) Cache-based methods such as ReKV and LiveVLM offload video segments to CPU or disk and require additional retrieval at query time, resulting in still-significant latency; (3) Existing methods apply coarse-grained eviction strategies (e.g., FIFO uniformly across all layers), ignoring the distinct attention preferences exhibited at different layers.

**Root Cause**: The KV cache is inherently an intrinsic latent memory of the model, making it well-suited for training-free management in streaming scenarios; however, existing methods fail to exploit inter-layer differences in attention patterns — different layers encode video information in fundamentally different ways.

**Paper Goals**: To design a KV cache management strategy based on layer-wise attention analysis that can be plugged into existing MLLMs without any training, enabling truly real-time streaming video QA.

**Starting Point**: Attention visualization analysis across the 28 decoder layers of LLaVA-OV-7B reveals three distinct layer-wise memory patterns.

**Core Idea**: Shallow layers exhibit strong recency bias (sensory memory) and are managed via exponential decay; deep layers attend to frame-level "anchor tokens" (long-term memory) and are managed via attention weights; middle layers transition between the two (working memory) and are managed via interpolation. Cross-layer smoothing and position re-indexing are additionally applied to ensure consistency.

## Method

### Overall Architecture

HERMES comprises three components: (1) **Hierarchical KV Cache Management** — applying distinct token importance scoring and eviction strategies according to layer type (shallow/middle/deep); (2) **Cross-Layer Memory Smoothing** — preventing cross-layer inconsistencies caused by independent per-layer eviction; (3) **Position Re-Indexing** — remapping positional encodings after eviction to maintain continuity. At inference time, the compressed KV cache is directly reused, requiring zero additional computation upon query arrival.

### Key Designs

1. **Hierarchical KV Cache Management**:

    - **Function**: Applies differentiated token eviction strategies according to the attention characteristics of each layer.
    - **Mechanism**: Shallow layers evaluate token importance via an exponential forgetting curve $S_i^l = \alpha_i^l \cdot e^{-k\Delta t_i}$ (more recent tokens are more important); deep layers use attention weights $S_i^l = \alpha_i^l \cdot W_i^l$ computed via a pseudo-query; middle layers interpolate recency scores and attention scores using a layer-dependent weight $\omega_l$: $S_i^l = (1-\omega_l) A_i^l + \omega_l R_i^l$.
    - **Design Motivation**: Attention visualization clearly demonstrates that different layers serve distinct memory functions — a uniform FIFO or attention-based eviction policy cannot simultaneously satisfy the requirements of all layers.

2. **Cross-Layer Memory Smoothing**:

    - **Function**: Prevents inconsistencies in token retention across layers caused by independent per-layer eviction decisions.
    - **Mechanism**: Eviction decisions are partially shared between adjacent layers, ensuring a degree of consistency in whether a given video token is retained or evicted across multiple layers.
    - **Design Motivation**: Independent layer-wise management leads to fragmentation of visual memory — information from the same frame may be retained in some layers while being evicted in others, undermining the coherence of end-to-end reasoning.

3. **Position Re-Indexing**:

    - **Function**: Remaps positional encodings after token eviction to preserve the correctness of RoPE.
    - **Mechanism**: After each eviction, the positions of retained tokens are re-indexed to a contiguous range $[0, |M|)$, avoiding attention computation anomalies caused by discontinuous positional encodings.
    - **Design Motivation**: Directly removing intermediate tokens introduces positional gaps that disrupt position-dependent attention mechanisms.

### Loss & Training

HERMES is entirely training-free. The hierarchical memory model is motivated by Ebbinghaus forgetting curve theory and cognitive psychology. A generic guidance prompt is used as a pseudo-query to compute attention weights in deep layers.

## Key Experimental Results

### Main Results

**Streaming Video Benchmarks (LLaVA-OV-7B)**

| Method | StreamingBench | EgoSchema | MVBench | Video-MME | Avg. |
|--------|---------------|-----------|---------|-----------|------|
| Full (no compression) | 53.2 | 58.1 | 69.3 | 61.8 | 60.6 |
| ReKV | 51.8 | 55.2 | 67.1 | 59.4 | 58.4 |
| StreamMem | 52.1 | 56.8 | 68.5 | 60.1 | 59.4 |
| **HERMES** | **59.3** | **58.9** | **69.8** | **62.4** | **62.6** |

### Ablation Study

**Efficiency Comparison (Single A800 GPU)**

| Method | TTFT (ms) | GPU Memory | Token Reduction |
|--------|-----------|------------|-----------------|
| Full | ~3000+ | Linear growth | 0% |
| ReKV | ~1500 | Requires CPU memory | ~50% |
| **HERMES** | **<30** | **Constant** | **68%** |

### Key Findings

- HERMES achieves an 11.4% improvement on streaming benchmarks despite reducing video tokens by 68%, demonstrating that removing redundant tokens actually improves reasoning quality.
- With TTFT below 30ms and constant GPU memory consumption, there is no risk of OOM as the number of input frames increases — zero additional computation is required upon query arrival.
- The hierarchical memory model generalizes across multiple MLLMs beyond LLaVA-OV.
- The recency bias in shallow-layer attention conforms to the Ebbinghaus forgetting curve; the anchor token intervals in deep-layer attention correspond precisely to the number of tokens per frame (196).

## Highlights & Insights

- The hierarchical memory concept borrowed from cognitive psychology maps precisely onto the layer-wise attention patterns of Transformers — this is not merely an analogy but a finding supported by quantitative attention analysis.
- The zero-additional-latency design is critical for real-time applications; methods such as ReKV reduce storage but still incur retrieval overhead at query time.
- The training-free, plug-and-play nature of HERMES allows direct deployment on existing MLLMs, lowering the barrier to practical adoption.

## Limitations & Future Work

- The delineation of layer boundaries (shallow/middle/deep) is derived from model-specific analysis and may need to be redetermined for different architectures.
- The pseudo-query used as a substitute for real user queries may introduce bias in specific scenarios.
- Validation is limited to streaming video settings; applicability to text streaming or other multimodal streaming scenarios remains unexplored.
- The exponential forgetting rate $k$ and interpolation parameters require manual configuration.

## Related Work & Insights

- **vs. ReKV/LiveVLM**: These methods rely on CPU offloading and retrieval operations, resulting in high latency; HERMES directly reuses the KV cache residing on the GPU.
- **vs. StreamMem**: StreamMem leverages chat template tokens to guide compression but lacks fine-grained management; HERMES achieves precise control through layer-wise attention analysis.
- **vs. StreamingLLM**: The attention sink mechanism retains initial tokens but ignores inter-layer differences; HERMES exploits layer specialization for more intelligent eviction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The hierarchical memory conceptualization and differentiated management strategy grounded in attention analysis are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple streaming benchmarks, efficiency analysis, attention visualization, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from mechanistic analysis to method design is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ A practical solution for real-time streaming video understanding with a 10× speedup in TTFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](../../NeurIPS2025/video_understanding/infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](../../ICCV2025/video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)

</div>

<!-- RELATED:END -->
