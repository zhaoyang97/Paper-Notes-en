---
title: >-
  [Paper Note] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
description: >-
  [ICCV 2025][Video Understanding][token merging] This paper proposes AIM, a training-free adaptive inference method that combines iterative token merging before the LLM (based on embedding cosine similarity) with progressive token pruning within LLM layers (based on PageRank importance scores), achieving a 6.8× FLOPs reduction with negligible performance loss, and even surpassing SOTA on long video understanding benchmarks.
tags:
  - ICCV 2025
  - Video Understanding
  - token merging
  - token pruning
  - adaptive inference
  - multi-modal LLM
  - training-free
date: 2026-05-08
content_hash: 0778c46c636e6dea
---

# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

**Conference**: ICCV 2025
**arXiv**: N/A
**Code**: [GitHub](https://github.com/LaVi-Lab/AIM)
**Area**: Video Understanding
**Keywords**: token merging, token pruning, adaptive inference, multi-modal LLM, training-free

## TL;DR

This paper proposes AIM, a training-free adaptive inference method that combines iterative token merging before the LLM (based on embedding cosine similarity) with progressive token pruning within LLM layers (based on PageRank importance scores), achieving a 6.8× FLOPs reduction with negligible performance loss, and even surpassing SOTA on long video understanding benchmarks.

## Background & Motivation

Multi-modal LLMs (e.g., LLaVA-OneVision) rely on large numbers of visual tokens (potentially thousands for video), incurring prohibitive computational costs that limit deployment in resource-constrained settings and long-video tasks. A core insight is that visual data contains substantial inherent redundancy — retaining only 25% of visual tokens is sufficient to preserve near-complete performance. Furthermore, reducing per-frame token counts allows the LLM to process more frames, compensating for information loss in long videos. Existing methods (FastV, PDrop, LLaVA-Prumerge) prune at only a single location or require fine-tuning, lacking flexible adaptive inference capability.

## Method

### Overall Architecture

AIM consists of two stages: (1) token merging before the LLM — iteratively merging high-similarity token pairs based on cosine similarity among visual tokens; and (2) token pruning within LLM layers — applying the PageRank algorithm at each layer to assess token importance from attention weights, and progressively removing unimportant visual tokens according to a scheduler-controlled retention ratio. Parameters of both stages are tunable, enabling adaptive inference spanning 2.5% to 100% of baseline FLOPs.

### Key Designs

1. **Iterative Token Merging (Pre-LLM)**: Adjacent visual tokens are partitioned into two groups A and B. Pairwise cosine similarities between groups are computed, and each token in A is matched to its most similar counterpart in B. Token pairs with the highest similarity are merged by averaging their embeddings. Each iteration reduces token count by at most half; multiple iterations can achieve a target retention rate. For video inputs, merging is performed only within frames — cross-frame merging disrupts temporal information.

2. **PageRank-based Progressive Token Pruning (Within LLM Layers)**: At each Transformer layer, the PageRank algorithm is executed using the attention weight matrix as an adjacency matrix to compute per-token importance scores. Only visual tokens are pruned; all text tokens are preserved (pruning text tokens causes severe performance degradation). A piecewise linear scheduler controls the retention ratio: all tokens are retained for layers $l < l_1$, the ratio decreases linearly for $l_1 \leq l \leq l_2$, and 0% of visual tokens are retained for $l > l_2$.

3. **Design Principles from Key Findings**: (1) Pruning visual tokens in early layers severely degrades performance, whereas heavy pruning in later layers preserves performance — indicating that early LLM layers perform cross-modal fusion while later layers focus on textual reasoning. (2) Text tokens must not be pruned at any layer. (3) Reducing per-frame token counts enables more frames as input, benefiting long-video understanding.

### Loss & Training

AIM is entirely training-free. It operates via direct inference on pretrained models LLaVA-OV-7B (Qwen2, 28 layers) and LLaVA-1.5-7B (Vicuna, 32 layers). Video settings: merging retention rate 25%, $l_1=14$, $l_2=22$. Image settings: merging retention rate 12.5%, $l_1=13$, $l_2=21$.

## Key Experimental Results

### Main Results

| Method | FLOPs (TB) | Prefill Time (ms) | VideoMME | MLVU | EgoSchema |
|---|---|---|---|---|---|
| LLaVA-OV-7B Baseline | 99.63 | 439.58 | 58.2 | 64.7 | 60.1 |
| FastV | 21.24 | 79.56 | 55.9 | 61.1 | 57.5 |
| LLaVA-Prumerge | 23.65 | 86.89 | 57.0 | 60.6 | 61.0 |
| **AIM** | **14.76** | **55.03** | **58.2** | **63.7** | **59.6** |
| AIM (192 frames) | 99.27 | 471.20 | 59.2 | **69.3** | 60.8 |

AIM achieves a 6.8× FLOPs reduction and an 8.0× prefill time reduction with negligible performance loss. At 192 frames, MLVU reaches 69.3 (+4.6 over the baseline).

### Ablation Study

- A 25% visual token retention rate is sufficient to maintain near-full performance.
- Intra-frame merging outperforms cross-frame merging.
- All visual tokens must be retained in early layers (before $l_1=14$).
- Text tokens must not be pruned.

### Key Findings

- Visual tokens in multi-modal LLMs are highly redundant; only 25% are necessary.
- Early LLM layers handle cross-modal fusion, while later layers shift toward pure textual reasoning.
- Token savings can be converted into additional frames, particularly benefiting long-video understanding.

## Highlights & Insights

- A minimalist design (merging + PageRank pruning) achieves state-of-the-art efficiency.
- Training-free and plug-and-play, compatible with various multi-modal LLMs.
- Adaptive inference capability — a single method covers 2.5% to 100% of baseline FLOPs.
- The analysis of layer-wise LLM behavior (early fusion, late reasoning) offers guidance for future MLLM architecture design.

## Limitations & Future Work

- PageRank computation itself introduces a small amount of additional overhead.
- The merging strategy is relatively simple (cosine similarity only) and may discard fine-grained visual details.
- Validation is limited to the LLaVA family; applicability to other architectures (e.g., InternVL) remains to be confirmed.
- Scheduler parameters ($l_1$, $l_2$) require manual tuning for different models.

## Related Work & Insights

- The success of ToMe (Token Merging) in ViTs is extended to the multi-modal LLM setting.
- FastV and PDrop serve as direct comparison baselines for visual token pruning.
- The adaptive inference paradigm is extensible to additional modalities such as 3D and audio.

## Rating

- **Novelty**: ⭐⭐⭐ — The merging + pruning combination is conceptually straightforward.
- **Technical Depth**: ⭐⭐⭐⭐ — The PageRank application and scheduler design are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers video and image, 6 benchmarks, adaptive efficiency curves, and in-depth analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with clear summarization of findings.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free, plug-and-play, and highly effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] Breaking the Encoder Barrier for Seamless Video-Language Understanding](breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)

</div>

<!-- RELATED:END -->
