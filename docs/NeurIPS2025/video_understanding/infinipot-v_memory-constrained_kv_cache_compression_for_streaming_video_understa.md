---
title: >-
  [Paper Note] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding
description: >-
  [NeurIPS 2025][Video Understanding][KV Cache Compression] This paper proposes InfiniPot-V, the first training-free and query-agnostic streaming video understanding framework. It achieves online KV cache compression via two complementary metrics — Temporal-axis Redundancy (TaR) and Value-Norm (VaN) — enabling streaming video understanding of arbitrary length under a fixed memory budget.
tags:
  - NeurIPS 2025
  - Video Understanding
  - KV Cache Compression
  - Streaming Video Understanding
  - Multimodal Large Language Models
  - Temporal Redundancy
  - Edge Deployment
date: 2026-05-08
content_hash: 43f1940ff76075fb
---

# InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2506.15745](https://arxiv.org/abs/2506.15745)
**Code**: [GitHub](https://github.com/aiha-lab/InfiniPot-V)
**Area**: Video Understanding / Model Efficiency
**Keywords**: KV Cache Compression, Streaming Video Understanding, Multimodal Large Language Models, Temporal Redundancy, Edge Deployment

## TL;DR

This paper proposes InfiniPot-V, the first training-free and query-agnostic streaming video understanding framework. It achieves online KV cache compression via two complementary metrics — Temporal-axis Redundancy (TaR) and Value-Norm (VaN) — enabling streaming video understanding of arbitrary length under a fixed memory budget.

## Background & Motivation

Modern multimodal large language models (MLLMs) have demonstrated the capability to process hour-long videos, yet this brings a critical practical challenge: **KV cache size grows linearly with the number of video frames, rapidly exceeding device memory limits**.

This problem is particularly acute in the following scenarios:

**Mobile/Edge Devices**: Smartphones, AR glasses, and robots have fixed and limited GPU memory.

**Streaming Video**: Video length is unknown and continuously growing, making memory pre-allocation infeasible.

**Multi-turn Dialogue**: Users may query the model multiple times during video playback, requiring long-term cache maintenance.

Existing KV cache compression methods suffer from two fundamental limitations:
- **Offline Assumption**: Most methods assume the entire video and user query are available before processing, making them unsuitable for real-time streaming.
- **Memory Still Scales with Length**: Even "compression" methods require constructing the full KV cache before compressing, so peak memory remains proportional to video length.

InfiniPot-V aims to: **perform online compression during video encoding, enforcing a fixed memory upper bound independent of video length**.

## Method

### Overall Architecture

InfiniPot-V adopts a block-wise processing strategy:

1. Input video frames are grouped into fixed-size blocks and encoded sequentially.
2. After encoding each block, the system checks whether the current KV cache size exceeds a user-defined threshold.
3. If the threshold is exceeded, a lightweight compression procedure reduces the cache below the budget.
4. The compression combines two complementary metrics: Temporal-axis Redundancy (TaR) and Value-Norm (VaN).

The entire process is **training-free** and **query-agnostic**, making it fully plug-and-play.

### Key Designs

**1. Temporal-axis Redundancy (TaR)**

TaR identifies and removes tokens that are redundant along the temporal dimension:
- Computes the cosine similarity between KV vectors at corresponding positions across adjacent frames.
- High similarity indicates that the position changes little over time (e.g., static background) and can be safely discarded.
- Formula: $\text{TaR}(i) = \text{CosSim}(\mathbf{k}_i^{(t)}, \mathbf{k}_i^{(t-1)})$
- Tokens with the highest TaR scores (most redundant) are removed.

Intuition: A large proportion of video tokens correspond to static backgrounds or slowly changing regions, which are highly redundant along the temporal axis.

**2. Value-Norm (VaN)**

VaN retains the semantically most important tokens:
- Computes the L2 norm of each token's Value vector.
- Larger value norms typically correspond to semantically salient content (moving objects, key actions, etc.).
- Tokens with the highest VaN scores are preserved.

Intuition: Tokens with larger Value norms contribute more to the attention aggregation output.

**3. Two-Stage Compression Pipeline**

- Stage 1 (TaR Filtering): Remove the top-$r_1\%$ tokens by temporal redundancy.
- Stage 2 (VaN Ranking): Among the remaining tokens, retain the top-$r_2\%$ by VaN score.
- The two stages are complementary: TaR eliminates redundancy while VaN preserves importance, enabling precise compression.

### Loss & Training

- **Training-free**: InfiniPot-V is entirely based on statistical metrics and introduces no learnable parameters.
- No fine-tuning of the underlying MLLM is required; the method is applied directly at inference time.
- Compatible with multiple open-source MLLMs (Qwen2-VL, Qwen2.5-VL, etc.).

## Key Experimental Results

### Main Results: Long Video Understanding Benchmarks

Performance of InfiniPot-V on Qwen2.5-VL-7B, compressing the KV cache from ~32K tokens to ~4K tokens:

| Benchmark | Full Cache | Uniform | SWA | InfiniPot-V | Compression Ratio |
|-----------|-----------|---------|-----|-------------|-------------------|
| MLVU | 70.2 | 64.8 | 66.1 | 69.5 | 8× |
| Video-MME | 63.4 | 58.2 | 59.7 | 62.8 | 8× |
| LongVideoBench | 55.1 | 49.6 | 51.3 | 54.7 | 8× |
| EgoSchema | 67.8 | 61.5 | 63.2 | 67.1 | 8× |

At 8× compression, InfiniPot-V incurs only a 0.5–1.0 point drop, significantly outperforming uniform sampling and sliding window attention.

### Cross-Model Generalization

Performance across different MLLMs (MLVU benchmark, 8× compression):

| Model | Full Cache | InfiniPot-V | Accuracy Retention |
|-------|-----------|-------------|-------------------|
| Qwen2-VL-7B | 65.3 | 64.1 | 98.2% |
| Qwen2.5-VL-7B | 70.2 | 69.5 | 99.0% |
| Qwen2-VL-72B | 78.1 | 77.3 | 99.0% |
| Qwen2.5-VL-72B | 80.5 | 79.8 | 99.1% |

Accuracy retention improves with model scale.

### Ablation Study

**Effectiveness of Compression Components**:

| Configuration | MLVU | Video-MME |
|---------------|------|-----------|
| Full Cache (no compression) | 70.2 | 63.4 |
| TaR only | 67.2 | 60.8 |
| VaN only | 66.8 | 60.1 |
| TaR + VaN (InfiniPot-V) | 69.5 | 62.8 |

The two metrics are complementary: TaR excels at removing background redundancy, while VaN excels at preserving foreground-salient information.

**Memory Reduction Efficiency**:

| Configuration | Peak GPU Memory | Relative to Full Cache |
|---------------|----------------|------------------------|
| Full Cache (768 frames) | ~48 GB | 100% |
| InfiniPot-V (4K budget) | ~3 GB | 6% (94% reduction) |
| InfiniPot-V (8K budget) | ~5 GB | 10% (90% reduction) |

### Key Findings

1. **Up to 94% GPU memory reduction**: From 48 GB to 3 GB, enabling hour-long video processing on consumer-grade GPUs.
2. **Near-lossless accuracy**: Accuracy retention exceeds 98% across multiple benchmarks; in some cases, InfiniPot-V even surpasses full cache performance.
3. **Real-time generation speed**: Inference throughput is no lower than that of full cache, as fewer tokens reduce per-step computation.
4. **Multi-turn dialogue support**: Under a fixed memory budget, the system continuously accepts new frames and queries, naturally supporting streaming multi-turn interaction.

## Highlights & Insights

1. **First genuinely streaming solution**: All prior "long video understanding" methods follow a "read-all-then-process" paradigm; InfiniPot-V truly achieves block-wise processing under a fixed memory constraint.
2. **Elegant training-free design**: TaR and VaN are intuitively motivated, computationally lightweight, and require no additional training.
3. **Clever exploitation of video properties**: TaR leverages the inherent temporal redundancy of video, a unique source of compressibility not present in text or images.
4. **Industry-deployment friendly**: Plug-and-play, cross-model generalizable, and fixed memory budget — highly suitable for edge device deployment.

## Limitations & Future Work

1. TaR currently relies on adjacent-frame comparisons, which may inadvertently discard key tokens during rapid scene cuts.
2. The semantic saliency assumption underlying VaN does not always hold; a large Value norm does not necessarily imply semantic importance.
3. Fixed compression ratios ($r_1$, $r_2$) are not adaptively adjusted based on video content complexity.
4. Validation is limited to the Qwen model family; generalization to other architectures (e.g., LLaVA variants) remains to be confirmed.
5. Robustness in extreme scenarios (e.g., hours of static surveillance footage followed by a sudden anomaly) has not been evaluated.

## Related Work & Insights

- **StreamingLLM**: Achieves streaming inference by retaining attention sinks and recent tokens, but does not account for semantic importance.
- **FastV / FreeVideoLLM**: Token pruning methods, but not designed for streaming settings.
- **KVzip**: Query-aware KV cache compression, but requires knowledge of the query content.
- **LiveVLM**: A concurrent work on streaming video understanding using a retrieval-based approach.
- Insight: The TaR metric can be generalized to token compression in other temporal data modalities, such as audio and sensor streams.

## Rating

- Novelty: ⭐⭐⭐⭐ (first streaming + fixed-memory solution)
- Technical Depth: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐ (directly deployable)
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](../../ACL2026/video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

</div>

<!-- RELATED:END -->
