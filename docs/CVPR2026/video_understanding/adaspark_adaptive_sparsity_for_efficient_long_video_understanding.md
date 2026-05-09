---
title: >-
  [Paper Note] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding
description: >-
  [CVPR 2026][Video Understanding][long video] AdaSpark is proposed to reduce FLOPs for long-video processing by up to 57% while maintaining performance, via 3D spatiotemporal cube partitioning and two synergistic adaptive sparsity mechanisms: cube-level attention selection and token-level FFN selection.
tags:
  - CVPR 2026
  - Video Understanding
  - long video
  - adaptive sparsity
  - Video-LLM
  - efficient inference
  - 3D cube
date: 2026-05-08
content_hash: af440b18ae760904
---

# AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2604.08077](https://arxiv.org/abs/2604.08077)
**Code**: None
**Area**: Video Understanding / Efficient Inference
**Keywords**: long video, adaptive sparsity, Video-LLM, efficient inference, 3D cube

## TL;DR

AdaSpark is proposed to reduce FLOPs for long-video processing by up to 57% while maintaining performance, via 3D spatiotemporal cube partitioning and two synergistic adaptive sparsity mechanisms: cube-level attention selection and token-level FFN selection.

## Background & Motivation

Long videos can produce token sequences on the order of hundreds of thousands to millions, rendering standard Video-LLMs infeasible due to quadratic attention complexity and FFN activation costs. Existing efficiency methods suffer from two key limitations: (1) irreversible information loss from frame sampling or token pruning harms fine-grained perception; (2) rigid predefined patterns such as local attention constrain long-range temporal modeling.

Preliminary analysis reveals two key phenomena: (1) video attention exhibits high intrinsic sparsity—a small number of tokens concentrate the majority of attention probability, with significantly varying token requirements across layers; (2) FFN layers exhibit "computational laziness" toward visual tokens—text tokens undergo substantial transformation after FFN (high variance), whereas visual tokens change stably.

## Method

### Overall Architecture

Video tokens are partitioned into 3D spatiotemporal cubes ($h \times w \times t$), and entropy-based adaptive sparse computation is applied separately at the attention layer and the FFN layer.

### Key Designs

1. **Adaptive Cube Selection Attention (AdaS-Attn)**: Each query token computes a relevance score against all preceding cubes (via similarity with the cube mean key $\bar{k}_j$), and Top-p (nucleus) selection determines the set of attended cubes: $P_i = \text{Softmax}([q \cdot \bar{k}_1/\sqrt{d_k}, ..., q \cdot \bar{k}_{i-1}/\sqrt{d_k}]^T)$, $\mathcal{S}_i = \{j | p_j \in \text{Top-p}(P_i, p)\}$. High-entropy distributions (dispersed attention) lead to selecting more cubes; low-entropy distributions (concentrated attention) select only a few. Full attention within the token's own cube is always retained.

2. **Adaptive Token Selection FFN (AdaS-FFN)**: Token importance is estimated by L2 norm, and Top-p selection similarly determines which tokens pass through the FFN. Skipped tokens are updated via mean compensation—the mean of FFN transformations from active tokens—to prevent complete information stagnation. Text tokens always pass through the FFN in a dense manner.

3. **Entropy-Based Top-p Selection**: Unified across both AdaS-Attn and AdaS-FFN. Adaptive sparsity allocates computation according to input complexity—more computation is allocated when information density is high, and computation is largely bypassed when information is sparse.

### Loss & Training

The sparsity strategy is applied on top of Qwen2.5-VL with lightweight fine-tuning. The sparsity threshold $p$ uniformly controls the computational budget of both modules.

## Key Experimental Results

### Main Results

| Benchmark | AdaSpark | Dense Baseline | FLOPs Reduction |
|-----------|----------|----------------|-----------------|
| MLVU Dev | Comparable | baseline | Up to 57% |
| VideoMME | Comparable | baseline | Up to 57% |
| VideoNIAH (ultra-long video) | Comparable | baseline | Significant |

### Key Findings

- Up to 57% FLOPs reduction while maintaining comparable performance across multiple benchmarks
- Top-p selection outperforms fixed sparsity ratios—different layers and inputs require different sparsity levels
- Mean compensation is critical for preserving information flow in skipped tokens
- The semantic homogeneity of cube partitioning underpins the accuracy of sparse selection
- Skipped tokens in AdaS-FFN are updated as $y_k = x_k + \bar{m}_i$, where $\bar{m}_i = \frac{1}{|\mathcal{M}_i|}\sum_{j \in \mathcal{M}_i} FFN(x_j)$
- Preliminary analysis shows that FFN exhibits "computational laziness" toward visual tokens: the variance of the L2-norm ratio is far lower than that of text tokens
- The sparsity strategy is applied on top of Qwen2.5-VL via lightweight fine-tuning

## Highlights & Insights

- The cube-token two-level hierarchical sparsity design is systematic and comprehensive
- The entropy-based adaptive mechanism elegantly avoids the suboptimality of fixed sparsity ratios
- The discovery of FFN "laziness" toward visual tokens in the preliminary analysis provides a solid motivation for token-selective FFN computation
- The mean compensation strategy is simple yet effective

## Limitations & Future Work

- The Top-p threshold still requires manual specification
- Hardware efficiency of sparse patterns depends on underlying framework support
- The impact on fine-grained temporal reasoning (e.g., precise temporal localization) warrants further evaluation
- Video cube partitioning uses a fixed window $h \times w \times t$; adaptive partitioning could yield further improvements
- Text tokens always pass through the FFN densely, preserving rich instruction and semantic content
- Comparable performance is maintained across benchmarks including MLVU Dev, VideoMME, and VideoNIAH, including hour-long ultra-long videos

## Rating

- Novelty: ⭐⭐⭐⭐ — Unified cube-token two-level sparsity framework
- Technical Depth: ⭐⭐⭐⭐ — Rigorous logic from preliminary analysis to method design
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark validation including ultra-long videos
- Practical Value: ⭐⭐⭐⭐⭐ — 57% FLOPs reduction offers strong practical utility

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_video.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[CVPR 2026\] A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a4vl_multiagent_long_video_reasoning.md)
- [\[CVPR 2026\] VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)

<!-- RELATED:END -->
