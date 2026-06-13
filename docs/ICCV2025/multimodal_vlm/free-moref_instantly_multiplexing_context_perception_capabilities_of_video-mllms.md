---
title: >-
  [Paper Note] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference
description: >-
  [ICCV 2025][Multimodal VLM][Long Video Understanding] This paper proposes Free-MoRef, a training-free method inspired by Mixture-of-Experts (MoE) that partitions long video tokens into multiple short sequences as multi-r…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Long Video Understanding"
  - "Video-MLLM"
  - "Training-Free Inference"
  - "MoE-Inspired"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: fb8640d23c0748f3
---

# Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference

**Conference**: ICCV 2025
**arXiv**: [2508.02134](https://arxiv.org/abs/2508.02134)  
**Code**: [github.com/wkfdb/Free-MoRef](https://github.com/wkfdb/Free-MoRef)  
**Area**: Video Understanding / Multimodal Large Language Models
**Keywords**: Long Video Understanding, Video-MLLM, Training-Free Inference, MoE-Inspired, Attention Mechanism

## TL;DR

This paper proposes Free-MoRef, a training-free method inspired by Mixture-of-Experts (MoE) that partitions long video tokens into multiple short sequences as multi-references, queries them in parallel via the MoRef attention mechanism, and fuses unified activation values. The approach enables efficient and comprehensive understanding of 2× to 8× longer frame inputs on a single A100 GPU, surpassing dedicated long-video models on VideoMME, MLVU, and LongVideoBench.

## Background & Motivation

Video-MLLMs have achieved notable progress in video understanding, yet they perform poorly on long-video scenarios due to the context length constraints of the underlying LLM. Existing solutions each exhibit critical drawbacks:

**Token Compression**: Reduces visual token count to accommodate more frames, but information loss scales with compression ratio.

**Streaming Inference**: Retains historical KV-Cache to support ultra-long context dependencies, but latency is proportional to context length (2× context = 2× latency).

**Context Extension**: Post-training to extend the context window incurs heavy computational overhead.

Core Problem: Can a single inference pass achieve longer context perception while guaranteeing comprehensive understanding and efficient computation?

## Method

### Overall Architecture

The Free-MoRef pipeline proceeds as follows:
1. **Multi-Reference Partition**: The long video token sequence is split temporally into $N$ short chunks, each representing an abstraction of the original video.
2. **MoRef Attention**: In the shallow decoder layers, the same query is used to attend to each chunk in parallel, and a unified response is fused.
3. **Reference Fusion**: In an intermediate deep decoder layer, key visual tokens are selected based on attention weights and merged into a global reference.

### Key Designs

1. **Multi-Reference Partition**: Video tokens are first divided into $M$ temporal units, each further split into $N$ fragments. Fragments from different units are aggregated to form $N$ reference chunks. $M$ controls the degree of temporal interleaving across references: when $M=1$, the $N$ chunks are temporally non-overlapping; larger $M$ yields greater interleaving. Each chunk is prepended with the same system prompt and question to form parallel inference sequences.

2. **Mixture of Reference Attention (MoRef Attention)**: The central component. Flash Attention is applied to parallel chunks to obtain initial outputs $O = [O^{sys}, O^{vis}, O^{ques}]$. Due to causal attention, $O^{sys}$ is identical across chunks, while $O^{vis}$ and $O^{ques}$ differ owing to distinct visual references. The diversity of $O^{vis}$ is preserved, while $O^{ques}$ is fused via gated weighting:

$$O^{fusion} = (\sum_{i=1}^N \omega_i \cdot O_i^{ques}).repeat(N)$$

The gating weights $\omega_i$ are computed from the query-vision cross-modal attention map: $\omega_i = \frac{max(A[i])}{\sum max(A[i])}$, where $A = softmax(Q^{ques} \times (K^{vis})^T)$, capturing the relevance between the query and each reference. This ensures that all visual tokens effectively participate in updating the query at every decoder layer.

3. **Reference Fusion**: Motivated by FastV's observation that visual tokens contribute uniformly in shallow decoder layers but attract attention concentrated on query tokens in deeper layers, this step performs merging at layer $L$. An importance matrix $E$ is evaluated for each visual token based on the attention map $A$; within each chunk, $1-1/N$ of unimportant tokens are pruned, and the remaining tokens are aggregated temporally into a global reference. This step compensates for the absence of cross-chunk visual interactions in MoRef attention.

### Loss & Training

**Completely training-free.** All designs are applied directly at inference time, requiring no additional training or fine-tuning.

- Base model: LLaVA-Video-7B (default maximum 64 frames)
- Frame input multipliers: 128 (2×), 256 (4×), 512 (8×)
- $M = 64$ temporal units; $N = \text{frames} / 64$
- Reference fusion layer: $L=3$ for $N=2$; $L=6$ for $N=4$; $L=12$ for $N=8$
- Compatible with Flash-Attention; combinable with streaming inference and token compression strategies

## Key Experimental Results

### Main Results

**Performance comparison across frame counts**:

| Context | FLOPs | MLVU | VideoMME (Medium/Long/Overall) | LongVideoBench |
|--------|-------|------|-------------------------------|----------------|
| 64 frames (baseline) | 100% | 70.3 | 62.1/53.4/64.3 | 58.8 |
| 128 frames (native) | 400% | 70.2 | 63.2/54.1/64.9 | 58.7 |
| 128 frames@MoRef | **110.4%** | **70.8** | **65.8/55.8/66.3** | **59.3** |
| 256 frames (native) | 1600% | 67.2 | 61.4/54.1/63.1 | 56.7 |
| 256 frames@MoRef | **163.2%** | **72.5** | **66.4/55.3/66.3** | **59.3** |
| 512 frames@MoRef | 400% | **72.8** | **67.3/56.0/66.9** | **59.9** |

Native inference at 512 frames either runs OOM or suffers severe performance degradation; Free-MoRef achieves state-of-the-art performance at only 400% FLOPs (versus 6400%).

**Comparison with other 7B–8B models**:

| Method | MLVU | LVideoBench | VideoMME Long | VideoMME Overall |
|------|------|-------------|---------------|-----------------|
| LLaVA-Video | 70.2 | 58.2 | 53.4 | 64.3 |
| Qwen2-VL | 64.8 | 55.6 | 55.7 | 63.3 |
| InternVL2.5 | 68.4 | 57.5 | 53.0 | 64.5 |
| Video-XL | 64.9 | 50.7 | - | 55.5 |
| RETAKE | 69.8 | - | 56.2 | 63.9 |
| **LLaVA-Video@MoRef** | **72.8** | **59.9** | **56.0** | **66.9** |

Free-MoRef surpasses all models at the same scale, including purpose-trained long-video models.

### Ablation Study

**Component-wise contribution** (128 frames, VideoMME Overall):

| Multi-Ref | MoRef Attn | Ref Fusion | Overall |
|-----------|------------|------------|---------|
| ✗ | ✗ | ✗ | 64.9 |
| ✗ | ✗ | ✓ | 63.9 |
| ✓ | ✗ | ✓ | 62.0 |
| ✓ | ✓ | ✗ | 65.8 |
| ✓ | ✓ | ✓ | **66.3** |

Partition without fusion leads to performance degradation; MoRef attention is the primary source of gain (+3.8); Reference Fusion provides further improvement (+0.5).

**Effect of the number of parallel chunks $N$**:

| N | FLOPs | Overall |
|---|-------|---------|
| 1 (default) | 100% | 64.9 |
| 2 | 27.6% | **66.3** |
| 4 | 25% | 66.1 |
| 8 | 23.6% | 65.9 |

$N=2$ yields the best performance while reducing FLOPs to only 27.6% of the baseline.

### Key Findings

- The core advantage of Free-MoRef lies in MoRef attention enabling effective participation of all visual tokens in query updates, approximating full attention at substantially reduced cost.
- The number of temporal units $M$ governs the trade-off between temporal perception (TP) and spatial perception (SP): small $M$ favors SP while large $M$ favors TP.
- The choice of fusion layer $L$ is critical: too early risks information loss; too late reduces the compensatory effect of cross-chunk visual interaction.
- Performance improves across nearly all question types in VideoMME, with the sole exception of attribute perception tasks, where the queried content spans only short video segments and extended context introduces redundancy.

## Highlights & Insights

- **Plug-and-play without training**: No training or additional parameters are required; any existing Video-MLLM can immediately benefit from the method.
- **Creative transfer of the MoE paradigm**: The notion of "multiple experts processing different data" is recast as "a single model querying different reference video segments."
- **Remarkable computational efficiency**: 8× frame input requires only 27.6%–400% of the original FLOPs; up to 1024 frames can be processed on a single A100 GPU.
- **High compatibility**: Supports Flash-Attention and is orthogonally composable with streaming inference or token compression schemes.

## Limitations & Future Work

- Multi-reference partitioning disrupts the continuity of visual features across chunks; Reference Fusion only partially compensates for this.
- Hyperparameters ($M$, $N$, $L$) require manual configuration and lack an adaptive mechanism.
- Validation is limited to LLaVA-Video-7B; larger models and alternative architectures remain untested.
- Attribute perception (AP) tasks exhibit a slight performance decline due to context extension.
- The MoRef attention design may inspire training-time long-context learning schemes, but this direction is not explored in the paper.

## Related Work & Insights

- FastV reveals the difference in how LLMs process visual tokens in shallow versus deep layers, informing the timing of Reference Fusion.
- The MoE paradigm is elegantly analogized to mixture of references.
- Streaming inference (e.g., INF-MLLM) and token compression (e.g., PruneVid) are orthogonal to and composable with Free-MoRef.
- LLaVA-Video's default 64-frame limit highlights the inherent tension between frame count and context length.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The training-free approach is highly innovative; the MoRef attention design is elegant; the analogy from MoE to multi-reference is inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across three long-video benchmarks with detailed ablations, though limited to a single base model.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear; attention pattern visualizations are persuasive.
- **Value**: ⭐⭐⭐⭐⭐ Extremely high practical value; any Video-MLLM can benefit immediately; code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[AAAI 2026\] AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](../../AAAI2026/multimodal_vlm/abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)

</div>

<!-- RELATED:END -->
