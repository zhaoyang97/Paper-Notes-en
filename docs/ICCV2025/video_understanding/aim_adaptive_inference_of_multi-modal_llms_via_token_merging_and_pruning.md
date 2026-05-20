---
title: >-
  [Paper Note] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
description: >-
  [ICCV 2025][Video Understanding][Multimodal Large Language Models] This paper proposes a training-free adaptive inference framework that achieves flexible accuracy–efficiency trade-offs across a 40× FLOPs range for multi…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Multimodal Large Language Models"
  - "Adaptive Inference"
  - "Token Merging"
  - "Token Pruning"
  - "Visual Token Redundancy"
date: 2026-05-08
content_hash: 5f312fc067cf2095
---

# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: [https://github.com/LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM)  
**Area**: Video Understanding
**Keywords**: Multimodal Large Language Models, Adaptive Inference, Token Merging, Token Pruning, Visual Token Redundancy

## TL;DR

This paper proposes a training-free adaptive inference framework that achieves flexible accuracy–efficiency trade-offs across a 40× FLOPs range for multimodal LLMs. The method combines iterative token merging based on embedding cosine similarity before the LLM, and progressive token pruning based on PageRank-derived multimodal importance scores within LLM layers. Strong performance is demonstrated on both video and image understanding benchmarks.

## Background & Motivation

### Root Cause
Multimodal LLMs rely on large numbers of visual tokens (hundreds for images, thousands for videos) to enable visual understanding, resulting in substantial computational overhead:
- **Resource-constrained scenarios**: Mobile devices, AR glasses, and similar platforms cannot afford high computational costs.
- **Long-video understanding**: The total number of tokens grows explosively with the number of frames, limiting the model's ability to process dense video frames and causing loss of critical temporal information.

### Why Can Visual Tokens Be Reduced?
The authors' core observation is that **visual data contains substantial intrinsic redundancy**. Experiments show that retaining only 25% of visual tokens is sufficient to maintain performance close to the full model. This redundancy provides the optimization space for adaptive inference.

### Limitations of Prior Work
- **FastV / VTW**: Prune or discard all visual tokens at a specific LLM layer, lacking flexibility to adapt to varying computational constraints.
- **PDrop**: Divides the LLM into 4 stages and prunes only at stage boundaries.
- **LLaVA-Prumerge**: Applies pruning only before the LLM using key-value pairs from the visual encoder.
- These methods operate either exclusively before or exclusively within the LLM, and none supports **adaptive inference** (i.e., dynamically adjusting to different computational budgets).

### Key Insights
1. Approximately 75% of visual tokens are redundant.
2. Fewer tokens per frame → more frames can be sampled → beneficial for long-video understanding.
3. Early LLM layers focus on cross-modal fusion while later layers focus on textual reasoning → large-scale visual token pruning is feasible in later layers.

## Method

### Overall Architecture

AIM comprises two core operations forming a two-stage token reduction strategy:

1. **Token Merging (before LLM)**: Iteratively merges highly similar visual tokens based on embedding cosine similarity.
2. **Token Pruning (within LLM layers)**: Progressively prunes unimportant visual tokens at each LLM layer using the PageRank algorithm.

Key design principle: **training-free** — directly applied to the inference process of pretrained models without any additional training.

### Key Design 1: Token Merging before LLM

Given visual tokens $v_0 \in \mathbb{R}^{N_0 \times D}$ at the LLM input, the following iterative merging procedure is applied:

1. Partition neighboring tokens into sets A and B.
2. Compute cosine similarity between each token in A and all tokens in B.
3. Find the most similar match in B for each token in A.
4. Merge the most similar token pairs by averaging their embeddings.
5. Each iteration reduces the token count by at most half; repeating for multiple iterations (e.g., 2) achieves the target retention ratio.

**Special handling for video**: Tokens are merged only within individual frames, not across frames.

**Why not merge across frames?** Ablation experiments show that cross-frame merging disrupts the temporal ordering of tokens, leading to loss of critical temporal information and degraded video understanding performance. Intra-frame merging has negligible impact on final inference quality.

**Design advantage**: Unlike merging at every layer of the visual encoder (e.g., ToMe), this approach applies merging after the visual encoder, making it agnostic to encoder architecture and plug-and-play compatible.

### Key Design 2: Token Pruning within LLM

The merged visual tokens $v_1$ are concatenated with text tokens $t_1$ to form $x_1 = [v_1; t_1]$, which is fed into the LLM. At each LLM layer:

**Importance Scoring — PageRank Algorithm**:

Attention weights are used as an adjacency matrix, and the PageRank algorithm computes an importance score for each token:

$$s_i^l = \frac{1}{N_l + M_l} \sum_{j=1}^{N_l+M_l} A_{i,j}^l \cdot s_j^l$$

where $s_j^l$ is initialized to a uniform distribution and $A^l$ is the softmax-normalized attention weight matrix.

**Only visual tokens are pruned; all text tokens are retained.**

**Why not prune text tokens?** Experiments show that pruning text tokens causes severe performance degradation (VideoMME drops from 58.2 to 45.7), as LLMs rely on text tokens for text-centric reasoning.

### Key Design 3: Piecewise Retention Rate Scheduler

A piecewise function controls the visual token retention rate $r_l$ at each layer:

$$r_l = \begin{cases} 1, & \text{if } l < l_1 \\ 1 - k(l - l_1), & \text{if } l_1 \leq l \leq l_2 \\ 0, & \text{if } l > l_2 \end{cases}$$

where $k = \frac{1}{l_2 - l_1}$ is the pruning slope.

| Parameter | Description | Video LLM Default | Image LLM Default |
|-----------|-------------|-------------------|-------------------|
| Merging retention rate | Fraction of tokens retained before LLM | 25% | 12.5% |
| $l_1$ | Layer at which pruning begins | 14 | 13 |
| $l_2$ | Layer at which visual tokens are fully removed | 22 | 21 |

**Rationale**: Based on the key finding that:
- Early layers ($< l_1$) handle cross-modal fusion; pruning visual tokens here severely degrades performance.
- Middle layers ($l_1 \leq l \leq l_2$) apply gradual pruning, balancing information retention and efficiency.
- Late layers ($> l_2$) primarily perform textual reasoning and do not require visual tokens.

### Adaptive Inference

By adjusting the merging retention rate and scheduler parameters $(l_1, l_2)$, a continuous control spectrum from lossless performance to maximum efficiency is achieved:
- Conservative configuration: 50% merging retention → 46.48 TFLOPs, performance slightly improved.
- Default configuration: 25% merging + (14, 22) pruning → 14.76 TFLOPs, performance maintained.
- Aggressive configuration: 1.6% merging + (14, 22) pruning → 2.51 TFLOPs, ~13% performance drop.

### Loss & Training

**No training is required.** The method is applied directly to the inference process of pretrained models. The additional computational overhead introduced is negligible:
- Video LLM (Qwen2-7B): token merging 88.25 GFLOPs + pruning 4.18 GFLOPs → only 0.6% of LLM inference FLOPs.
- Image LLM (Vicuna-v1.5-7B): total 0.26 GFLOPs → only 0.03%.

## Key Experimental Results

### Main Results

**Video benchmarks** (base model: LLaVA-OV-7B, 32 frames):

| Method | FLOPs (TB) | Prefill (ms) | VideoMME | MVBench | MLVU | EgoSchema |
|--------|------------|--------------|----------|---------|------|-----------|
| LLaVA-OV-7B | 99.63 | 439.58 | 58.2 | 56.7 | 64.7 | 60.1 |
| FastV | 21.24 | 79.56 | 55.9 | 55.9 | 61.1 | 57.5 |
| LLaVA-Prumerge | 23.65 | 86.89 | 57.0 | 56.5 | 60.6 | 61.0 |
| **AIM** | **14.76** | **55.03** | **58.2** | **57.1** | **63.7** | **59.6** |

AIM achieves near-zero performance loss (VideoMME 58.2 unchanged, MVBench slightly improved) with the lowest computational cost (14.76 TB FLOPs, 1/6.8× of the base model).

**Long-video enhancement**: 192 frames vs. base model's 32 frames under the same computational budget:

| Configuration | Frames | FLOPs (TB) | VideoMME | MLVU |
|---------------|--------|------------|----------|------|
| LLaVA-OV-7B | 32 | 99.63 | 58.2 | 64.7 |
| AIM | 32 | 14.76 | 58.2 | 63.7 |
| **AIM** | **192** | **99.27** | **59.2** | **69.3** |

MLVU (long-video understanding) improves by **+4.6**, validating the hypothesis that "fewer tokens per frame → more frames → better long-video understanding."

**Image benchmarks** (base model: LLaVA-1.5-7B):

| Method | FLOPs (TB) | VQA-v2 | GQA | MME | POPE |
|--------|------------|--------|-----|-----|------|
| LLaVA-1.5-7B | 8.18 | 78.5 | 62.0 | 1510.7 | 85.9 |
| FastV | 2.58 | 74.1 | 56.6 | 1438.5 | 73.6 |
| LLaVA-Prumerge+ | 2.41 | 74.6 | 57.4 | 1391.9 | 82.2 |
| **AIM** | **2.22** | **75.4** | **58.6** | **1443.5** | **85.7** |

### Ablation Study

**Token merging retention rate ablation** (pruning disabled):

| Retention Rate | FLOPs (TB) | Prefill (ms) | VideoMME |
|----------------|------------|--------------|----------|
| 100% | 99.63 | 439.58 | 58.2 |
| 50% | 46.48 | 182.65 | **58.5** |
| 25% | 22.90 | 83.94 | 58.0 |
| 12.5% | 11.64 | 41.22 | 56.6 |
| 3.1% | 3.85 | 13.68 | 52.3 |

Performance remains virtually unchanged when retaining 25% or more tokens, confirming that approximately 75% of visual tokens are redundant.

**Token pruning scheduler ablation** (25% merging retention rate):

| $l_1$ | $l_2$ | FLOPs (TB) | VideoMME |
|-------|-------|------------|----------|
| 28 | 29 | 22.90 | 58.0 |
| 14 | 22 | 14.76 | **58.2** |
| 14 | 15 | 12.10 | 54.3 |
| 7 | 8 | 6.71 | 41.9 |

Removing visual tokens starting from layer 8 causes catastrophic performance degradation (58.0 → 41.9), while removing them from layer 22 onward incurs no loss.

**Text token pruning ablation**:

| Setting | VideoMME |
|---------|----------|
| Prune visual tokens only | 58.2 |
| Prune both visual and text tokens | 45.7 |

Pruning text tokens causes a severe **−12.5** performance drop.

### Key Findings

1. **75% of visual tokens are redundant**: Retaining only 25% is sufficient to maintain performance.
2. **Layer-wise behavioral differences in LLMs**: Early layers perform cross-modal fusion (must not be pruned); late layers perform textual reasoning (can be heavily pruned).
3. **Text tokens are indispensable**: Text tokens are central to LLM reasoning; any pruning causes severe performance degradation.
4. **Acceleration advantage for long-video understanding**: Token compression enables sampling more frames within the same computational budget, yielding MLVU +4.6.
5. **General applicability**: The method is effective for both video and image LLMs, and across different LLM architectures (Qwen2, Vicuna).

## Highlights & Insights

1. **Practical value of adaptive inference**: A single method covers a 40× FLOPs range, accommodating diverse devices from AR glasses to workstations, addressing real deployment challenges.
2. **Training-free design**: The method is plug-and-play, directly applicable to existing pretrained models without retraining or fine-tuning, minimizing migration cost.
3. **Complementarity of the two-stage token reduction**: The combination of merging (global redundancy removal) and pruning (layer-wise adaptive refinement) outperforms either strategy alone.
4. **Creative application of PageRank**: Adapting the web page ranking algorithm to attention weight analysis provides a more holistic assessment of token importance than simple attention scores.
5. **Deep insight into LLM layer-wise behavior**: The finding that early layers handle cross-modal fusion while late layers handle textual reasoning provides guidance for future multimodal LLM design.

## Limitations & Future Work

1. **Degraded performance on TextVQA**: The method underperforms on text-rich images, as merging may discard fine-grained textual details.
2. **Manual scheduler parameter tuning**: The optimal values of $l_1$ and $l_2$ depend on the specific model and task, currently requiring heuristic selection.
3. **Visual tokens only**: Text token redundancy is not addressed; although experiments show text tokens cannot be pruned, optimizing their representations may be worth exploring.
4. **PageRank computational overhead**: Although minimal, further reduction may be desirable in extreme low-latency scenarios.
5. **Lack of validation on generative tasks**: The method is currently validated only on understanding tasks (VQA, multiple-choice); visual generation scenarios remain unexplored.

## Related Work & Insights

- **ToMe**: The seminal work on token merging at every layer of vision Transformers; this paper transfers the core idea to the LLM input stage.
- **FastV**: The first method to prune visual tokens within LLM layers, but applied only at a single layer in a non-progressive manner.
- **Adaptive computation**: This paper naturally introduces the classical concept of adaptive inference into multimodal LLMs, filling an important gap in the literature.
- **Implications for the field**: Future multimodal LLM design should account for layer-wise role differentiation (cross-modal fusion vs. textual reasoning), potentially enabling architectural optimizations at the design stage.

## Rating

- Novelty: ⭐⭐⭐⭐ (Combines existing techniques with clever design; the adaptive inference perspective is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Video + image, multiple base models, comprehensive ablations, computational cost analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, well-defined conclusions, rich insights)
- Value: ⭐⭐⭐⭐⭐ (Training-free + adaptive inference; highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] Breaking the Encoder Barrier for Seamless Video-Language Understanding](breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)

</div>

<!-- RELATED:END -->
