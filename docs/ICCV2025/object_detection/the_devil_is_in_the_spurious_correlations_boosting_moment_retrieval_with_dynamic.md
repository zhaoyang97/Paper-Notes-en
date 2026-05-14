---
title: >-
  [Paper Note] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning
description: >-
  [ICCV 2025][Object Detection][Moment Retrieval] This paper is the first to identify spurious correlations between text queries and background frames as the fundamental bottleneck in moment retrieval performance. It propo…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Moment Retrieval"
  - "Spurious Correlation"
  - "Video Synthesis"
  - "Temporal Dynamics"
  - "DETR"
date: 2026-05-08
content_hash: 756801b877c985c0
---

# The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning

**Conference**: ICCV 2025
**arXiv**: [2501.07305](https://arxiv.org/abs/2501.07305)
**Code**: Coming soon
**Area**: Object Detection
**Keywords**: Moment Retrieval, Spurious Correlation, Video Synthesis, Temporal Dynamics, DETR

## TL;DR

This paper is the first to identify spurious correlations between text queries and background frames as the fundamental bottleneck in moment retrieval performance. It proposes TD-DETR, a framework that mitigates this issue via two strategies: dynamic context video synthesis and text-dynamics interaction enhancement, achieving state-of-the-art results on QVHighlights and Charades-STA.

## Background & Motivation

Moment Retrieval aims to localize video segments corresponding to a given text query. Existing DETR-based methods perform well on text-video alignment but struggle to predict accurate temporal spans.

This paper identifies a previously overlooked root cause: **spurious correlations** — models tend to associate text queries with background frames rather than distinguishing the target moment. For instance, the SOTA method BAM-DETR predicts nearly identical temporal spans on both the original video and a version where the target moment is masked, revealing that the model relies on background frames rather than the target segment itself.

This is the first systematic study of spurious correlations in moment retrieval. Unlike spatial biases in image-domain tasks, such correlations in video tasks are considerably more complex.

## Method

### Overall Architecture

TD-DETR (Temporal Dynamics DETR) consists of three core modules:
1. **Video Synthesizer**: Constructs dynamic context via video synthesis
2. **Dynamics Enhancement**: Enhances representations through temporal dynamics and text interaction
3. **Transformer Encoder-Decoder + Prediction Heads**: Standard DETR detection heads

### Key Designs

1. **Video Synthesizer for Dynamic Context**:

    - **Spurious Pair Selection**: For each video $V_i$ in the training batch, the video $V_k$ with the highest cosine similarity is selected to form a spurious pair $p_i = \{V_i, V_k\}$, ensuring the synthesized video is sufficiently challenging.
    - **Dynamic Context Synthesis**: The target moment is preserved in its entirety (selection probability set to 1). Non-target frames are sampled from $V_i$ at ratio $\alpha$ and from $V_k$ at ratio $1-\alpha$, concatenated into a new video $\tilde{V}_i$, with ground truth annotations updated accordingly.
    - **Design Motivation**: Enables the model to learn to recognize target moments against varying background contexts, reducing reliance on specific backgrounds.

2. **Dynamics Enhancement**:

    - **Dynamic Tokenizer**: A learnable start token $st$ is prepended to the video sequence; temporal dynamics are extracted via frame-wise differences $T = \{\tilde{v}_1 - st, \tilde{v}_2 - \tilde{v}_1, \ldots\}$, introducing negligible additional computation.
    - **Text-Dynamics Interaction**: Cross-attention is applied separately to compute video-text and dynamics-text interaction representations, which are fused via weighted combination $\tilde{V}_i' = \beta \cdot \tilde{V}_i + (1-\beta) \cdot T'$ to inject dynamic information into video representations.
    - **Design Motivation**: Encourages the model to attend not only to static frame semantics but also to background-agnostic temporal changes, establishing robust associations between text queries and target moments.

3. **Hungarian Matching and Prediction**:

    - Predictions are made separately for each video in a synthesized pair and matched against the corresponding ground truth.
    - Saliency scores are predicted simultaneously for Highlight Detection.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{hl} + \mathcal{L}_{moment}$$

where:
- $\mathcal{L}_{moment} = \lambda_{L_1}\mathcal{L}_{L_1} + \lambda_{iou}\mathcal{L}_{gIoU} + \lambda_{cls}\mathcal{L}_{cls}$ (L1 loss + gIoU loss + classification cross-entropy)
- $\mathcal{L}_{hl} = \lambda_{margin}\mathcal{L}_{margin} + \lambda_{cont}\mathcal{L}_{cont} + \lambda_{neg}\mathcal{L}_{neg}$ (margin ranking + rank-contrastive + negative sample loss)

## Key Experimental Results

### Main Results (QVHighlights test split)

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|--------|---------|----------|---------|
| Moment-DETR | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| QD-DETR | 62.40 | 44.98 | 62.52 | 39.88 | 39.86 |
| CG-DETR | 65.40 | 48.40 | 64.50 | 42.80 | 42.90 |
| BAM-DETR | 64.53 | 48.64 | 64.57 | 46.33 | 45.36 |
| **TD-DETR (Ours)** | **64.53** | **50.37** | **66.21** | **47.32** | **46.69** |
| SnAG w/ TD-DETR | 66.48 | 52.93 | 63.71 | 49.11 | 46.75 |

**Charades-STA test split**:

| Method | R1@0.5 | R1@0.7 |
|--------|--------|--------|
| QD-DETR | 57.31 | 32.55 |
| BAM-DETR | 59.95 | 39.38 |
| **TD-DETR (Ours)** | **60.89** | **40.35** |

### Ablation Study

**Ablation on sampling ratio $\alpha$ (QVHighlights val)**:

| $\alpha$ | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|----------|--------|--------|---------|----------|---------|
| 0.0 | 11.61 | 3.35 | 23.93 | 7.50 | 10.09 |
| 0.3 | 65.10 | 51.94 | 65.77 | 48.13 | 47.55 |
| 0.7 | **65.88** | **53.67** | 66.43 | **49.86** | **49.05** |
| 0.9 | 64.19 | 51.23 | 66.29 | 48.88 | 47.94 |

**Ablation on dynamics fusion weight $\beta$**: $\beta=0.7$ yields optimal performance; using dynamics only ($\beta=0$) or excluding dynamics ($\beta=1$) both degrade results.

**Sampling strategy comparison**:

| Strategy | R1@0.7 | mAP@0.75 | mAP |
|----------|--------|----------|-----|
| baseline (QD-DETR) | 46.66 | 41.82 | 41.22 |
| w/ random | 51.29 | 47.82 | 47.56 |
| w/ similarity | **53.67** | **49.86** | **49.05** |

### Key Findings

- Similarity-guided pair selection outperforms random selection, though random selection already yields substantial gains over the baseline.
- With InternVideo2 features, TD-DETR achieves R1@0.5 of 73.49 and R1@0.7 of 53.01 on Charades-STA, surpassing prior methods by a large margin.
- TD-DETR maintains SOTA performance on the dynamic context validation set, confirming that spurious correlations are effectively mitigated.
- Visualizations show that the baseline predicts identical spans even after target masking, whereas TD-DETR correctly adjusts its predictions.

## Highlights & Insights

- **Precise problem formulation**: The first work to systematically identify and analyze spurious correlations in moment retrieval, supported by intuitive and compelling masking experiments.
- **Simple yet effective method**: Video synthesis requires no external data or generative models, relying solely on intra-batch video recombination; the temporal dynamic tokenizer is implemented via simple frame differencing with near-zero overhead.
- **Strong generalizability**: Compatible with diverse architectures including SnAG and QD-DETR, and remains effective with InternVideo2 features.
- The spurious correlation analysis paradigm is transferable to other video understanding tasks.

## Limitations & Future Work

- The synthesis strategy relies on intra-batch video pairs; high-quality pairs may be difficult to obtain with small batch sizes.
- Frame-wise differencing is a linear operation and may fail to capture complex nonlinear temporal patterns.
- Evaluation is limited to QVHighlights and Charades-STA.
- The behavior of spurious correlations in longer-video scenarios (e.g., movie-length content) remains unexplored.

## Related Work & Insights

- **BAM-DETR / CG-DETR**: Current SOTA for DETR-based moment retrieval, focusing on architectural design while overlooking spurious correlations.
- **SBI (Self-Blended Images)**: One source of inspiration for the video synthesis strategy.
- **Spurious Correlation in Vision**: Extensively studied in the image domain, but systematically addressed for the first time in video-domain moment retrieval by this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to reveal spurious correlations in moment retrieval; a novel and well-motivated perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations including cross-architecture generalization and targeted spurious correlation evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Problem statement is clear and well-illustrated.
- **Value**: ⭐⭐⭐⭐ The proposed framework is concise and generalizable; the spurious correlation perspective offers meaningful inspiration for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)
- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)
- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](../../CVPR2026/object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)

</div>

<!-- RELATED:END -->
