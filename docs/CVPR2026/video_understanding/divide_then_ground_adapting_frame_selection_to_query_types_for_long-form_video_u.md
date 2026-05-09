---
title: >-
  [Paper Note] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding
description: >-
  [CVPR 2026][Video Understanding][long-form video understanding] This paper proposes DIG, a training-free frame selection framework that classifies queries into global and localization types. For global queries, uniform sampling is applied directly; for localization queries, a dedicated pipeline consisting of content-adaptive frame selection (CAFS), LMM-based reward scoring, and video refinement is employed. DIG consistently outperforms existing methods on three long-form video understanding benchmarks.
tags:
  - CVPR 2026
  - Video Understanding
  - long-form video understanding
  - frame selection
  - query classification
  - content-adaptive sampling
  - large multimodal models
date: 2026-05-08
content_hash: 9d068064cda1600c
---

# DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2512.04000](https://arxiv.org/abs/2512.04000)
**Code**: [GitHub](https://github.com/Jialuo-Li/DIG)
**Area**: Video Understanding
**Keywords**: long-form video understanding, frame selection, query classification, content-adaptive sampling, large multimodal models

## TL;DR

This paper proposes DIG, a training-free frame selection framework that classifies queries into global and localization types. For global queries, uniform sampling is applied directly; for localization queries, a dedicated pipeline consisting of content-adaptive frame selection (CAFS), LMM-based reward scoring, and video refinement is employed. DIG consistently outperforms existing methods on three long-form video understanding benchmarks.

## Background & Motivation

Large multimodal models (LMMs) face a fundamental tension in video understanding: the number of video tokens is enormous, yet LLM context lengths are limited, forcing the model to process only a sampled subset of frames. Uniform sampling maximizes temporal coverage but is entirely query-agnostic.

Prior work has proposed query-aware adaptive frame selection mechanisms (e.g., AKS, Q-Frame), but these incur substantial computational overhead. The authors identify a **critical question that has been widely overlooked**:

> Is a complex search mechanism necessary for all query types? The answer is **no**.

This core finding drives the entire method design:

**Global queries** (e.g., "What is the theme of this video?"): require comprehensive video information; uniform sampling is already sufficient and efficient.

**Localization queries** (e.g., "What vehicle is that person riding?"): target specific temporal segments; uniform sampling introduces large amounts of irrelevant frames, degrading performance.

**More frames ≠ better performance**: Experiments show that accuracy across all LMMs degrades beyond an optimal frame count, and this degradation is primarily contributed by localization queries.

## Method

### Overall Architecture

DIG consists of two pathways:
- **Global query pathway**: LLM classification → uniform sampling → LMM inference
- **Localization query pathway**: LLM classification → CAFS content-adaptive frame selection → LMM reward scoring → video refinement → uniform sampling → LMM inference

### Key Designs

1. **Query Type Identification (§4.1)**:

    - An LLM (Qwen3-Next-80B-A3B) automatically classifies queries as global or localization.
    - Global queries are routed directly to uniform sampling, avoiding unnecessary computation.
    - **Design Motivation**: Experiments demonstrate that complex frame selection yields negligible or even negative gains for global queries.

2. **Content-Adaptive Frame Selection (CAFS, §4.2)**:

    - The video is first sampled at 2 fps to obtain $M$ frames; DINOv2 is used to extract features.
    - The cosine distance sequence between adjacent frames is computed: $d_i = 1 - \text{sim}(V_{I_i}, V_{I_{i+1}})$
    - Peaks in the distance sequence (prominence > 0.1) are detected as scene boundaries.
    - The midpoint frame within each segment is selected as a **representative frame (r-frame)**.
    - **Design Motivation**: Addresses the dilemma of fixed-rate sampling—low frame rates miss key events while high frame rates introduce redundancy. CAFS adapts to the semantic change rate of video content.

3. **Reward Scoring (§4.3)**:

    - An LMM (rather than surface-level matching methods such as CLIPScore) is used directly to assess the relevance of each r-frame to the query.
    - **Two-dimensional scoring design**: (1) direct relevance of the current frame to the query; (2) whether the current frame's content implies that adjacent frames may contain complementary information.
    - **Design Motivation**: CLIPScore is unreliable for complex reasoning, whereas LMMs possess deeper contextual reasoning capabilities.

4. **Video Refinement (§4.4)**:

    - **Iterative reward-guided selection**: A parameter-free method that iteratively subtracts the mean until the set stabilizes, retaining r-frames whose reward consistently exceeds the average.
    - **Segment merging**: For each selected r-frame, the method uses not only the single frame but also the surrounding segment and neighboring segments within a window of size $w_{len}$, merging them into a continuous video clip.
    - Uniform sampling is then applied over the refined video to obtain the final input.
    - **Design Motivation**: Top-K requires a fixed hyperparameter, whereas the iterative approach adapts to different videos; segment merging preserves continuous fine-grained information rather than sparse frames.

### Loss & Training

DIG is a completely **training-free** framework with no training involved. All components (DINOv2, LLM classifier, LMM scorer) use off-the-shelf pretrained models. Hyperparameter settings: 56 tokens per frame, $w_{len}=2$.

## Key Experimental Results

### Main Results

**Qwen2.5-VL-32B as the base LMM:**

| Method | #Frames | MLVU | LVB | VideoMME-Med | VideoMME-Long |
|--------|---------|------|-----|-------------|--------------|
| UNI | 32 | 61.91 | 57.89 | 57.89 | 53.33 |
| AKS | 32 | 66.42 | 59.31 | 59.89 | 56.00 |
| Q-Frame | 32 | 60.95 | 57.37 | 60.43 | 55.90 |
| **DIG** | 32 | **70.69** | **61.86** | **60.87** | **57.76** |

**Qwen2.5-VL-7B as the base LMM:**

| Method | #Frames | MLVU | LVB | VideoMME-Med | VideoMME-Long |
|--------|---------|------|-----|-------------|--------------|
| UNI | 32 | 59.52 | 56.92 | 59.08 | 52.02 |
| **DIG** | 32 | **67.20** | **60.43** | **61.62** | **53.24** |

**Scalability to high frame counts (7B model):**

| Method | #Frames | MLVU | LVB |
|--------|---------|------|-----|
| UNI | 256 | 69.15 | 61.48 |
| AKS | 256 | 71.50 | 61.03 |
| **DIG** | 256 | **72.46** | **64.62** |

### Ablation Study

| Configuration | Key Metric | Observation |
|--------------|-----------|-------------|
| UNI on GQ vs. DIG pipeline on GQ | Comparable performance | Complex frame selection is unnecessary for global queries |
| UNI on LQ vs. DIG pipeline on LQ | DIG significantly outperforms | Localization queries are the primary source of DIG's gains |
| CLIPScore reward vs. LMM reward | LMM reward +1–3% | LMM scoring is more reliable for complex reasoning |
| Frame count scaling from 8 to 256 | DIG consistently leads | AKS and Q-Frame may fall below UNI at high frame counts |

### Key Findings

1. **At 32 frames, DIG improves over uniform sampling by 7.68% on MLVU and 4.51% on LVB.**
2. **Query type classification is critical**: uniform sampling is optimal for global queries; complex search is only warranted for localization queries.
3. **Performance degradation from increased frame counts is primarily driven by localization queries**: global queries remain stable as frame count increases.
4. **DIG maintains its advantage at high frame counts**: AKS and Q-Frame may degrade below uniform sampling beyond 128 frames.
5. **CAFS is more efficient than fixed-rate sampling**: it achieves higher coverage with fewer frames.

## Highlights & Insights

- **Concise yet powerful core insight**: not all queries require complex frame selection—this observation is simple but widely overlooked.
- **Training-free design**: fully leverages off-the-shelf models with no additional training, resulting in extremely low deployment cost.
- **Content-adaptive sampling**: peak detection over DINOv2 feature distances is an elegant solution for video segmentation.
- **Iterative reward-guided selection**: a parameter-free selection algorithm that eliminates the need to tune the Top-K hyperparameter.
- **Two-dimensional LMM scoring**: evaluates not only the direct relevance of the current frame but also the potential supplementary value of adjacent frames.
- **Strong scalability**: consistent gains are maintained from 8 to 256 frames.

## Limitations & Future Work

1. **Query classification depends on an LLM**: classification accuracy directly affects the downstream pipeline; misclassification can result in worse performance than uniform sampling.
2. **Computational overhead**: although the global query pathway is efficient, the localization query pathway requires DINOv2 feature extraction, LMM scoring, and video refinement, incurring non-trivial total cost.
3. **The binary taxonomy may be overly simplistic**: some queries fall between global and localization (e.g., "How does the mood shift between the first and second half of the video?"); finer-grained type distinctions warrant further exploration.
4. **The fixed peak prominence threshold (0.1) in r-frame selection** may not generalize well to all video types (e.g., slow-motion vs. fast-cut footage).
5. **Validation is limited to VQA tasks**: applicability to other video understanding tasks such as video summarization and video captioning remains to be verified.

## Related Work & Insights

- DIG is directly compared against current state-of-the-art frame selection methods including AKS and Q-Frame, demonstrating the advantage of strategy-adaptive routing.
- The general utility of DINOv2 as a visual feature extractor is once again validated.
- Using an LMM itself as the scorer (rather than relying on CLIP) represents a promising emerging trend.
- The query classification + strategy routing design pattern is generalizable to other multimodal understanding tasks.
- The segment merging concept also has application potential in video retrieval and video summarization.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core insight (query type determines the optimal strategy) is concise and compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks, two LMMs, full coverage from 8 to 256 frames, with extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; figures and tables are informative.
- Value: ⭐⭐⭐⭐ — Highly practical and directly deployable into existing LMM-based video understanding pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](../../ICCV2025/video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)

</div>

<!-- RELATED:END -->
