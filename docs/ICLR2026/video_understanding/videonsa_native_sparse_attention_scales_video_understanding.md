---
title: >-
  [Paper Note] VideoNSA: Native Sparse Attention Scales Video Understanding
description: >-
  [ICLR 2026][Video Understanding][sparse attention] This paper proposes VideoNSA, which introduces Native Sparse Attention (NSA) into video-language models. Through a mixed sparse attention mechanism combining compression…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "sparse attention"
  - "long context"
  - "multimodal LLM"
date: 2026-05-08
content_hash: 2d1300890a538464
---

# VideoNSA: Native Sparse Attention Scales Video Understanding

**Conference**: ICLR 2026
**arXiv**: [2510.02295](https://arxiv.org/abs/2510.02295)  
**Code**: N/A  
**Area**: Video Understanding
**Keywords**: sparse attention, video understanding, long context, multimodal LLM

## TL;DR

This paper proposes VideoNSA, which introduces Native Sparse Attention (NSA) into video-language models. Through a mixed sparse attention mechanism combining compression, selection, and sliding window branches with dynamic gating, VideoNSA achieves 128K-token video understanding using only 3.6% of the attention budget, surpassing token compression and training-free sparse attention baselines on long video understanding, temporal reasoning, and spatial understanding tasks.

## Background & Motivation

1. **Video understanding constrained by context length**: Existing multimodal large language models (MLLMs) are limited by context windows when processing long videos, often missing critical transition frames and struggling to maintain coherence over long time scales. For instance, a decisive moment in a soccer match lasts only seconds, yet an entire game spans 90 minutes.

2. **Token compression incurs irreversible information loss**: Existing token compression methods (FastV, VScan, VisionZip, etc.) reduce redundancy but suffer significant performance degradation on complex reasoning tasks, as compression strategies limit the generalizability of perceptual and reasoning capabilities.

3. **Training-free sparse attention lacks hardware alignment**: Existing training-free sparse attention methods (Tri-Shape, MInference, etc.) are typically not hardware-aligned, impose static adjacency matrices, restrict the flexibility of information flow, and cannot improve training efficiency.

4. **Video tokens exhibit high temporal redundancy**: Consecutive video frames contain substantial redundancy, making sparse attention suitable; however, the complexity of video (spatiotemporal dependencies) prevents direct application of sparse attention methods developed for LLMs.

5. **NSA has been validated in LLMs**: Native Sparse Attention has demonstrated learnable, hardware-aware sparse attention advantages in pure-text long-context modeling, yet has not been applied to video multimodal scenarios.

6. **Increasing sampled frames improves accuracy at high cost**: Intuitively, sampling more frames improves accuracy, but the quadratic growth in computational complexity demands efficient attention mechanisms to break through this bottleneck.

## Method

### Overall Architecture

VideoNSA is built on Qwen2.5-VL-7B and employs a **hybrid attention mechanism** at every LLM decoder layer: NSA sparse attention is applied to video tokens, while standard GQA (Grouped Query Attention) is retained for text tokens, balancing efficiency with instruction-following capability.

### Three-Branch Sparse Attention

The core of NSA distributes each query's attention across three complementary branches, dynamically weighted via learnable gates $g_t^c$:

$$\mathbf{o}_t = \sum_{c \in \{\text{cmp}, \text{slc}, \text{win}\}} g_t^c \cdot \text{Attn}(q_t, \tilde{\mathbf{K}}_t^c, \tilde{\mathbf{V}}_t^c)$$

1. **Compression Branch (CMP)**: Aggregates consecutive token blocks into coarse-grained block-level representations via a learnable MLP, capturing global semantics. Block size is set to the number of tokens per frame (64), with intra-frame mean pooling used to obtain block representations.

2. **Selection Branch (SLC)**: Computes importance scores for each KV block and retains the top-$n$ most salient blocks, preserving fine-grained critical information.

3. **Sliding Window Branch (SWA)**: Retains the most recent $w$ KV pairs ($w=256$), ensuring local temporal coverage.

The gates $g_t^c$ are implemented via a two-layer MLP with Sigmoid activation, enabling data-dependent dynamic routing.

### Hybrid Attention Design

At each layer $l$, input tokens are split by position ID into video tokens $\mathbf{X}_\mathcal{V}$ and text tokens $\mathbf{X}_\mathcal{T}$:
- Video tokens → NSA three-branch sparse attention
- Text tokens → Standard GQA (28 query heads, 4 shared KV heads)
- Final output is the concatenation of both parts: $\mathbf{o}^{(l)} = [\mathbf{o}_\mathcal{V}^{(l)}; \mathbf{o}_\mathcal{T}^{(l)}]$

### Loss & Training

- Data: 216K QA pairs filtered from LLaVA-Video-178K, selecting videos with 350–550 frames
- Constraints: Maximum 50,176 pixels per frame; maximum context of 36K tokens per instance
- Hyperparameters: Block size $s=64$, number of blocks $b=32$, sliding window $w=256$
- End-to-end training totaling 4,600 H100 GPU hours
- SWIFT framework with FLA-adapted NSA implementation

## Key Experimental Results

### Main Results: Comprehensive Multi-Task Evaluation

| Model | LongVideoBench | MLVU_test | TimeScope | LongTimeScope | Tomato | VSIBench |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| Qwen2.5-VL-7B (baseline) | 58.7 | 51.2 | 81.0 | 40.7 | 22.6 | 29.7 |
| + FastV (token compression) | 57.3 | 41.8 | 46.5 | 35.6 | 21.6 | 32.0 |
| + VisionZip (token compression) | 52.4 | 33.1 | 43.5 | 40.4 | 23.6 | 32.1 |
| + MInference (sparse attention) | 59.2 | 49.2 | 82.7 | 44.4 | 23.0 | 36.5 |
| + XAttention (sparse attention) | 59.1 | 50.2 | 83.1 | 41.1 | 21.4 | 36.6 |
| **VideoNSA** | **60.0** | **51.8** | **83.7** | **44.4** | **26.5** | **36.1** |

**Key Findings**:
- Sparse attention methods consistently outperform token compression methods
- VideoNSA shows a clear advantage in temporal reasoning (Tomato +3.9) and long video understanding
- VideoNSA matches the strongest sparse attention baseline on spatial understanding (VSIBench) while significantly outperforming compression methods

### Ablation Study: Branch Combination Analysis

| CMP | SLC | SWD | LongVideoBench | MLVU | TimeScope | LongTimeScope | Tomato | VSIBench |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | | | 48.1 | 43.9 | 41.5 | 25.1 | 23.3 | 29.2 |
| | ✓ | | 48.4 | 47.7 | 63.7 | 37.1 | 24.0 | 27.6 |
| | | ✓ | 49.1 | 40.2 | 59.3 | 29.8 | 24.0 | 29.8 |
| ✓ | ✓ | ✓ | **60.0** | **51.8** | **83.7** | **44.4** | **26.5** | **36.1** |

The three-branch combination significantly outperforms any single- or two-branch configuration, demonstrating the necessity of dynamically gating all three branches.

### Six Key Findings from Scaling Analysis

1. **Sparse weights transfer to dense attention**: Dense-NSA (using VideoNSA weights but with dense attention at inference) outperforms the baseline on most tasks, indicating that sparse training provides effective attentional inductive biases.
2. **Reliable scaling to 128K tokens**: Performance continues to improve beyond the training length (36K).
3. **Optimal attention allocation is highly task-dependent**: LongVideoBench favors more tokens per frame, while Tomato favors higher frame rates.
4. **Gate distributions evolve across layers**: The compression branch remains dominant throughout all layers, while the selection and sliding window branches gradually diminish in deeper layers.
5. **Compression branch is the efficiency bottleneck**: Inference latency of the compression branch becomes dominant as context length grows.
6. **Learnable sparse attention induces dynamic attention sinks**: The selection branch exhibits almost no sinks; the compression branch has the most sinks but they are effectively offset by the gating mechanism, resulting in an overall sink ratio of only 0.3%.

## Highlights & Insights

- **First learnable and hardware-aware sparse attention for video**: Unlike static sparse patterns, VideoNSA achieves data-dependent sparse connectivity through end-to-end training.
- **Elegant hybrid attention design**: Sparse attention for video tokens and dense attention for text tokens, balancing efficiency and instruction-following.
- **Optimal performance at only 3.6% attention budget**: Extreme computational efficiency.
- **Systematic scaling analysis**: Six key findings provide deep insight into the behavioral properties of sparse attention in video understanding.

## Limitations & Future Work

- Limited training data quality (a subset of LLaVA-Video-178K); performance on some benchmarks slightly decreases after SFT.
- The compression branch remains an inference bottleneck; kernel and memory efficiency require further optimization.
- Validation is limited to 7B-scale models; experiments on larger-scale models are absent.
- Block size is fixed to the number of tokens per frame; adaptive block partitioning strategies remain unexplored.

## Related Work & Insights

### vs. MInference (Jiang et al., 2024)
MInference is a training-free sparse attention method using predefined sparse patterns (A-shape, Vertical-Slash, etc.) without additional training. VideoNSA learns data-dependent sparse patterns through end-to-end training, achieving superior performance on Tomato (26.5 vs. 23.0) and matching performance on VSIBench (36.1 vs. 36.5), at the cost of 4,600 H100 GPU hours of training.

### vs. FastV / VisionZip (Token Compression Methods)
Token compression methods directly discard or merge tokens, causing irreversible information loss. FastV achieves only 46.5 on TimeScope (vs. VideoNSA's 83.7), and VisionZip only 33.1 on MLVU (vs. 51.8). VideoNSA retains all tokens but focuses on critical dependencies through sparse attention, yielding substantial advantages on complex reasoning tasks.

### vs. XAttention (Xu et al., 2025)
XAttention is also a training-free sparse attention method using the same configuration as VideoNSA but without training. VideoNSA significantly outperforms it on LongTimeScope (44.4 vs. 41.1) and Tomato (26.5 vs. 21.4), demonstrating that end-to-end training is critical for learning effective sparse patterns.

## Rating

- ⭐⭐⭐⭐ **Novelty**: First systematic introduction of learnable sparse attention to video understanding, with a distinctive hybrid attention design.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Comprehensive experiments, thorough six-finding scaling analysis, and sufficient ablation studies.
- ⭐⭐⭐⭐ **Value**: Directly applicable to existing VLM architectures; code and models are open-sourced.
- ⭐⭐⭐ **Writing Quality**: Structure is clear, but some notation definitions are scattered and figure captions could be more concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](../../CVPR2026/video_understanding/vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](../../CVPR2026/video_understanding/autogaze_attend_before_attention_efficient_video.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[ICLR 2026\] Video-KTR: Enhancing Video Reasoning via Key Token Attribution](video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)

</div>

<!-- RELATED:END -->
