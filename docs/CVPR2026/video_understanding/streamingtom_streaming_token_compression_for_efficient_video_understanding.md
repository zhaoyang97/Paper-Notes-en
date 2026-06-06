---
title: >-
  [Paper Note] StreamingTOM: Streaming Token Compression for Efficient Video Understanding
description: >-
  [CVPR2026][Video Understanding][streaming video understanding] This paper proposes StreamingTOM, a training-free two-stage framework for streaming video understanding. Causal Temporal Reduction (CTR) compresses per-frame…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "streaming video understanding"
  - "token compression"
  - "kv-cache quantization"
  - "training-free"
  - "causal inference"
date: 2026-05-08
content_hash: 9a21d82c42c46ef1
---

# StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**Conference**: CVPR2026  
**arXiv**: [2510.18269](https://arxiv.org/abs/2510.18269)  
**Code**: [yige24/StreamingTOM](https://yige24.github.io/StreamingTOM)  
**Area**: Video Understanding / Streaming Video QA / Token Compression  
**Keywords**: streaming video understanding, token compression, kv-cache quantization, training-free, causal inference

## TL;DR

This paper proposes StreamingTOM, a training-free two-stage framework for streaming video understanding. Causal Temporal Reduction (CTR) compresses per-frame tokens from 196 to 50 via causal temporal selection before the LLM, while Online Quantized Memory (OQM) constrains kv-cache growth after the LLM through 4-bit quantization and on-demand retrieval. The framework achieves a 15.7× compression ratio, 1.2× lower peak memory, and 2× faster TTFT.

## Background & Motivation

1. **Dual constraints of streaming video**: Unlike offline processing, streaming video VLMs face two fundamental constraints — causality (no access to future frames) and accumulation (unbounded token growth over time) — making token compression not merely an optimization but a necessity.
2. **Unbounded kv-cache growth**: Using LLaVA-OV-7B as an example, a 1-hour video at 0.5 fps generates an 18.8 GB kv-cache, far exceeding typical GPU memory capacity and rendering real-time inference infeasible.
3. **Existing methods only manage post-LLM state**: Current training-free streaming approaches (ReKV, LiveVLM, StreamMem) apply eviction or compression only to the kv-cache after the LLM, without reducing the $O(tNLd^2)$ computational cost of pre-LLM prefill.
4. **Offline compression violates causality**: Well-established offline token merging and pruning methods (ToMe, DyCoke, HoliTom) rely on global or bidirectional attention and future frame information, making them inapplicable to streaming scenarios.
5. **Training-based methods are costly**: Training-based streaming approaches (Flash-VStream, Dispider) require expensive backbone-specific retraining and do not transfer easily across architectures.
6. **Gap in causal pre-LLM compression**: To the authors' knowledge, no prior training-free streaming method performs strictly causal token reduction before the LLM, leaving a significant efficiency opportunity unexplored.

## Method

### Overall Architecture: Two-Stage Pipeline

StreamingTOM = OQM₁₆→₄ ∘ CTR_{N→G}, using a **group abstraction** (frame-aligned groups of fixed $G=50$ tokens per frame) as the interface between the two stages:

- **Visual pipeline**: Visual encoder extracts features → CTR compression → written to online memory
- **Query pipeline**: User question drives the decoder → OQM retrieves relevant groups → 4-bit dequantization → efficient generation

### Stage 1: Causal Temporal Reduction (CTR)

CTR adheres to three design principles: strict causality (2-frame window), single-pass processing, and a fixed per-frame budget $G$.

1. **Temporal similarity computation**: Cosine similarity $s_t^{(i)}$ is computed between tokens at the same spatial position across adjacent frames $t$ and $t{-}1$, measuring cross-frame redundancy.
2. **Spatial saliency**: Attention scores $\alpha_t^{(i)}$ from the visual encoder are reused as a zero-cost byproduct; chunked attention is applied to avoid memory spikes.
3. **Static/dynamic classification**: Tokens are partitioned into a static set $\mathcal{S}_t$ (high similarity, redundant) and a dynamic set $\mathcal{D}_t$ (low similarity, novel information) using threshold $\tau_c = 0.9$.
4. **Adaptive budget allocation**: The $G$ slots are divided into $k_s$ and $k_d$ proportionally to the static/dynamic ratio, allocating more capacity to dynamic tokens when scene content changes rapidly.
5. **Dual-path processing**:
    - Dynamic path: Top-$k_d$ tokens selected by saliency (preserving key novel information)
    - Static path: Density-based clustering merges tokens into $k_s$ representative tokens (removing redundancy)
6. **Complexity**: $O(N + G^2)$ per frame; state requires only the previous frame's features $O(Nd)$, independent of stream length.

### Stage 2: Online Quantized Memory (OQM)

OQM addresses the residual linear growth of the kv-cache after CTR:

1. **Incremental group quantization**: Each group is independently quantized to 4-bit (per-head, per-channel scale/offset), with a representative key $\bar{\mathbf{k}}_t$ stored alongside.
2. **Retrieve-then-dequantize paradigm**: At query time, cosine similarity is computed between the decoder state and all group representative keys; the top-$k$ most relevant groups are selected and dequantized from 4-bit to FP16.
3. **Bounded active memory**: Total storage is $O(T \cdot G \cdot d / 4)$ retaining the full history, while active kv is $O(k \cdot G \cdot d)$ ($k \ll T$), keeping decoding latency independent of stream length.

### Compression Ratio

Combining CTR and OQM: compression ratio $= 4N/G = 4 \times 196/50 \approx 15.7\times$.

## Key Experimental Results

### Offline Long-Video Benchmarks (LLaVA-OV-7B backbone)

| Method | VideoMME Overall | MLVU | EgoSchema | Avg |
|---|---|---|---|---|
| LLaVA-OV-7B (offline baseline) | 58.4 | 64.7 | 60.1 | 61.0 |
| +LiveVLM (training-free SOTA) | 57.3 | 66.3 | 59.0 | 60.9 |
| +StreamMem | 59.4 | 66.9 | 63.0 | 63.1 |
| **+StreamingTOM (ours)** | **59.9** | **67.9** | **63.7** | **63.8** |

### Online Streaming Evaluation (RVS benchmark, 28 GB memory limit)

| Method | RVS-Ego Acc/Score | RVS-Movie Acc/Score | Avg Acc/Score |
|---|---|---|---|
| Flash-VStream (training-based) | 57.0 / 4.0 | 53.1 / 3.3 | 55.0 / 3.6 |
| StreamMem | 57.6 / 3.8 | 52.7 / 3.4 | 55.2 / 3.6 |
| **StreamingTOM** | **58.3 / 3.9** | **53.2 / 3.5** | **55.8 / 3.7** |

### Efficiency Metrics

- **kv-cache compression ratio**: 15.7×
- **Peak memory**: 1.2× reduction compared to LiveVLM
- **TTFT**: 2× speedup compared to LiveVLM
- **1-hour video kv-cache**: 18.8 GB → 1.2 GB
- **Memory growth**: only 16.0 GB → 16.7 GB from 16 to 512 frames (sub-linear)
- **Throughput**: stable at ~20 tokens/s on long sequences

### Ablation Study

| Token Count | Quantization | Compression Ratio | VideoMME Overall |
|---|---|---|---|
| 40 | 4-bit | 5.1% | 58.9 |
| **50** | **4-bit** | **6.4%** | **59.9** |
| 60 | 4-bit | 7.7% | 59.3 |
| 50 | 2-bit | 3.2% | 58.5 |

- 50 tokens is the optimal trade-off: fewer (40) loses critical details, more (60) reduces temporal coverage under fixed memory budget
- 4-bit quantization outperforms 2-bit, achieving the best accuracy–compression balance

## Highlights & Insights

1. **First causal pre-LLM token compression**: Fills the gap in training-free streaming methods by reducing prefill complexity from $O(tNLd^2)$ to $O(tGLd^2)$.
2. **Elegant group abstraction**: Fixed-size, frame-aligned groups serve both as CTR output units and OQM storage/retrieval units, ensuring temporal consistency and predictable latency.
3. **Fully plug-and-play**: Requires no training and can be directly applied to different backbones such as LLaVA-OV.
4. **Deployment-friendly**: Runs on a single A6000 GPU, is batch-agnostic, and exhibits sub-linear memory growth.
5. **Complementary two-stage design**: CTR reduces computation while OQM reduces memory; both stages are necessary, and their combination far surpasses either alone.

## Limitations & Future Work

1. **Fixed $G$ may be suboptimal**: A uniform budget of 50 tokens per frame lacks flexibility for frames with heterogeneous information density (e.g., keyframes vs. static frames).
2. **Single backbone evaluated**: Experiments are primarily conducted on LLaVA-OV-7B; generalization to larger models (e.g., 72B) or other architectures remains unverified.
3. **2-frame window limitation**: CTR's causal window considers only adjacent frame pairs, potentially accumulating errors in slowly evolving scenes.
4. **Retrieval quality of representative keys**: OQM uses mean keys for retrieval, which may be insufficiently precise for fine-grained temporal reasoning.
5. **Multimodal audio streams not evaluated**: Only the visual stream is considered; practical streaming applications typically involve concurrent audio streams.

## Related Work & Insights

| Dimension | StreamingTOM | LiveVLM/StreamMem | DyCoke/HoliTom | Flash-VStream |
|---|---|---|---|---|
| Pre-LLM compression | ✅ CTR | ❌ | ✅ (non-causal) | ✅ (requires training) |
| Post-LLM management | ✅ OQM 4-bit | ✅ kv-cache eviction | ❌ | ✅ (requires training) |
| Causal constraint | ✅ strict | ✅ | ❌ requires future frames | ✅ |
| Training required | No | No | No | Retraining needed |
| Compression ratio | 15.7× | ~4× | ~4× | N/A |

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce causal pre-LLM token compression in a training-free streaming framework; group abstraction elegantly unifies the two stages
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both offline and online benchmarks with detailed efficiency analysis and complete ablations
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, mathematical derivations are rigorous, and pipeline diagrams are intuitive
- Value: ⭐⭐⭐⭐ — Addresses a real deployment bottleneck in streaming video VLMs; plug-and-play design ensures strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](../../ICML2026/video_understanding/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)

</div>

<!-- RELATED:END -->
