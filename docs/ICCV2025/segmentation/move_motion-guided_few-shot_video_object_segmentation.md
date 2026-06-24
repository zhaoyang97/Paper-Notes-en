---
title: >-
  [Paper Note] MOVE: Motion-Guided Few-Shot Video Object Segmentation
description: >-
  [ICCV 2025][Segmentation][few-shot video segmentation] This paper introduces a novel task of motion-guided few-shot video object segmentation along with a large-scale dataset MOVE (224 motion categories, 4,300 videos, 314K masks), and proposes a Decoupled Motion-Appearance (DMA) network. Through a dual-branch architecture combining frame-differencing-based motion prototypes and appearance prototypes, the proposed method significantly outperforms existing FSVOS methods on the…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "few-shot video segmentation"
  - "motion understanding"
  - "video object segmentation"
  - "temporal modeling"
  - "benchmark"
date: 2026-05-08
content_hash: 0d4cd078aff1f8c1
---

# MOVE: Motion-Guided Few-Shot Video Object Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.22061](https://arxiv.org/abs/2507.22061)  
**Code**: [https://henghuiding.com/MOVE/](https://henghuiding.com/MOVE/)  
**Area**: Segmentation / Video Understanding / Few-Shot
**Keywords**: few-shot video segmentation, motion understanding, video object segmentation, temporal modeling, benchmark

## TL;DR

This paper introduces a novel task of motion-guided few-shot video object segmentation along with a large-scale dataset MOVE (224 motion categories, 4,300 videos, 314K masks), and proposes a Decoupled Motion-Appearance (DMA) network. Through a dual-branch architecture combining frame-differencing-based motion prototypes and appearance prototypes, the proposed method significantly outperforms existing FSVOS methods on the new benchmark.

## Background & Motivation

**Background**: Few-shot video object segmentation (FSVOS) aims to segment novel-category objects in query videos given a small number of annotated examples. Existing methods (DANet, HPAN, TTI) are **semantics-centric** — they associate support and query sets based on object category, e.g., "given a panda image, segment all pandas."

**Limitations of Prior Work**: (a) Existing FSVOS methods overlook the most essential information in videos — **motion patterns** — reducing the task to static image matching; (b) On datasets such as YouTube-VIS, image-level FSS methods and video-level FSVOS methods achieve comparable performance (62.3 vs. 63.0 $\mathcal{J}\&\mathcal{F}$), indicating that current evaluations do not assess temporal understanding; (c) Referring video object segmentation (RVOS) uses text to describe motion, but novel or complex motions are difficult to describe precisely in language (e.g., Cristiano Ronaldo's signature celebration, the Joker's dance).

**Key Challenge**: Motion patterns are a defining characteristic of video, yet existing few-shot segmentation methods primarily extract appearance/semantic features and lack effective mechanisms for motion feature extraction and matching — making cross-category recognition of identical motions infeasible.

**Goal**: (a) Establish a FSVOS benchmark organized by motion categories; (b) Design a method capable of effectively extracting motion prototypes from video to enable "observe a motion pattern → locate objects performing the same motion in a new video."

**Key Insight**: The support set is extended from images to video clips (since static images cannot represent motion), a motion category vocabulary is constructed, and frame differencing is employed to explicitly extract motion features decoupled from appearance features.

**Core Idea**: Transform FSVOS from "object category matching" to "motion pattern matching" by decoupling motion and appearance prototypes to enable cross-category motion segmentation.

## Method

### Overall Architecture

Given a support video (with masks) and a query video: a shared encoder extracts multi-scale features → a Proposal Generator produces coarse masks for the query → the DMA module extracts decoupled motion-appearance prototypes from both support and query videos → a Prototype Attention module fuses the support and query prototypes → a Mask Decoder generates the final segmentation masks. A [CLS] token is also used to compute a matching score to determine whether the query target exhibits the same motion as the support.

### Key Designs

1. **Decoupled Motion-Appearance Module (DMA)**:

    - **Function**: Extract decoupled motion prototypes and appearance prototypes from video.
    - **Mechanism**:
        - **Appearance prototype** $P_a$: Mask pooling applied to $\frac{1}{4}$-resolution features $F_{l1}$: $P_a = \frac{\sum_{h,w} F_{l1} \odot M}{\sum_{h,w} M} \in \mathbb{R}^{T \times d}$
        - **Motion prototype** $P_m$: Temporal feature differences between adjacent frames $D_{l1,t} = F_{l1,t+1} - F_{l1,t}$, followed by 3D convolutions for temporal enhancement and spatial pooling to yield $P_m \in \mathbb{R}^{T \times d}$
        - **Auxiliary classification heads**: The appearance prototype is passed through an MLP to predict object categories ($C_o$ classes); the motion prototype is passed through an MLP to predict motion categories ($C_m$ classes), explicitly supervising disentanglement.
        - **Transformer refinement**: Motion prototypes are refined via cross-attention (attending to both $P_m$ and $P_a$) + self-attention + FFN to produce the final $P_{\text{dma}}$.
    - **Design Motivation**: Frame differencing is the most direct approach to extract motion signals — static appearance information is eliminated while object displacement and deformation are retained. Auxiliary classification ensures that each branch learns its respective representation without conflation.

2. **Prototype Attention Module**:

    - **Function**: Fuse the motion-appearance prototypes from support and query.
    - **Mechanism**: $P^q_{\text{dma}}$ serves as the query and $P^s_{\text{dma}}$ as key/value in cross-attention, followed by self-attention and multiple iterative refinement layers.
    - **Design Motivation**: Guide the query prototype to focus on features consistent with the support motion pattern.

3. **Matching Score**:

    - **Function**: Determine whether the query instance exhibits the same motion as the support.
    - **Mechanism**: $S_{\text{match}} = \cos([\text{CLS}]_s, [\text{CLS}]_q)$, computed as the cosine similarity between the support and query [CLS] tokens.
    - **Design Motivation**: In practical settings, it is necessary to first determine "whether the target motion is present" before segmentation, to avoid erroneous predictions on frames not containing the target motion.

4. **Proposal Generator + Mask Decoder**:

    - **Function**: Generate coarse masks from multi-scale features, followed by fine segmentation.
    - **Mechanism**: The Proposal Generator applies three convolutional blocks at different scales to query features to produce coarse mask proposals; the Mask Decoder generates the final masks via cross-attention with prototypes and top-down feature fusion.

### Loss & Training

- Mask prediction: Cross-Entropy Loss + IoU Loss
- Auxiliary classification heads: Cross-Entropy Loss (object category + motion category)
- Matching score: Cross-Entropy Loss
- Backbone: ResNet50 (ImageNet pre-trained) or VideoSwin-Tiny (Kinetics-400 pre-trained)
- Learning rate 1e-5 with cosine annealing; trained for 240K episodes

## Key Experimental Results

### Main Results

MOVE benchmark, Overlapping Split (OS), 2-way-1-shot:

| Method | Backbone | $\mathcal{J}\&\mathcal{F}$ | T-Acc | N-Acc |
|--------|----------|---------------------------|-------|-------|
| DMA (Ours) | ResNet50 | **50.1** | 98.6 | **11.5** |
| DANet | ResNet50 | 45.4 | 97.1 | 8.2 |
| TTI | ResNet50 | 45.2 | 97.6 | 9.4 |
| HPAN | ResNet50 | 44.4 | 97.3 | 7.2 |
| SCCAN (FSS) | ResNet50 | 40.6 | 93.9 | 5.8 |
| DMA (Ours) | VSwin-T | **51.5** | 98.9 | **21.2** |
| DANet | VSwin-T | 49.8 | 93.4 | 16.5 |

5-way-1-shot setting:

| Method | $\mathcal{J}\&\mathcal{F}$ | T-Acc | N-Acc |
|--------|---------------------------|-------|-------|
| DMA (Ours, R50) | **40.2** | 99.5 | **28.7** |
| TTI | 35.6 | 70.6 | 26.2 |
| HPAN | 34.0 | 99.1 | 3.1 |
| DMA (Ours, VSwin-T) | **41.4** | 99.8 | **31.0** |

### Ablation Study

Motion extractor comparison:

| Configuration | $\mathcal{J}\&\mathcal{F}$ | Description |
|---------------|---------------------------|-------------|
| Differencing (Ours) | **46.8** | Frame-differencing for motion extraction |
| Mask Adapter | 43.4 | Adapter-based motion learning |
| Mask Pooling | 41.3 | Direct pooling |

DMA prototype decomposition:

| Appearance | Motion | $\mathcal{J}\&\mathcal{F}$ | Description |
|-----------|--------|---------------------------|-------------|
| ✓ | ✗ | 36.5 | Appearance only — cannot distinguish motion |
| ✗ | ✓ | 43.8 | Motion only — lacks appearance support |
| ✓ | ✓ | **46.8** | Complementary combination, optimal |

Auxiliary classification ablation:

| Object Class. | Motion Class. | $\mathcal{J}\&\mathcal{F}$ |
|--------------|--------------|---------------------------|
| ✗ | ✗ | 43.8 |
| ✓ | ✓ | **46.8** |

### Key Findings

- **Motion prototypes outperform appearance prototypes** (43.8 vs. 36.5), confirming that motion is the core cue for the MOVE task.
- Frame differencing is the most effective motion extraction strategy, outperforming mask adapter (+3.4) and mask pooling (+5.5).
- **Auxiliary classification heads yield substantial gains** (43.8 → 46.8), demonstrating that explicit disentanglement supervision is essential.
- Necessity of MOVE: On YTVIS, FSS and FSVOS achieve comparable performance (62.3 vs. 63.0), whereas the gap widens substantially on MOVE (40.6 vs. 44.4), confirming that the MOVE benchmark genuinely requires motion understanding.
- Oracle experiments: Perfect motion labels → 63.6%; perfect masks → 74.3%, indicating significant room for improvement in both motion understanding and mask quality.
- t-SNE visualization: Without DMA, prototypes cluster by object category (color); with DMA, they cluster by motion category (shape).

## Highlights & Insights

- **A new motion-centric paradigm**: Reframes FSVOS from "what object is it?" to "what motion is being performed?", opening an entirely new research direction — "a video clip speaks louder than a thousand words."
- **Frame-differencing + auxiliary classification for decoupling**: Elegant and effective — frame differencing naturally suppresses static appearance while preserving motion, and auxiliary classification heads ensure each branch learns its designated representation. This strategy is transferable to any task requiring motion-appearance disentanglement.
- **MOVE dataset construction**: 224 motion categories, 4,300 videos, 314K masks spanning 88 object categories — the first FSVOS dataset organized by motion categories.
- **Matching score mechanism**: Motion matching is determined via [CLS] token cosine similarity, providing a concise and effective solution to the false-positive segmentation problem in empty-foreground frames.

## Limitations & Future Work

- N-Acc (non-target accuracy) remains low across all methods (best at only 31%), indicating that background modeling and false positive suppression are major bottlenecks.
- Frame differencing may be insensitive to subtle or slow-paced motions.
- The current study addresses only the 1-shot setting; aggregating motion prototypes from multiple support videos in the few-shot regime warrants further exploration.
- Compositional motion decomposition (e.g., jump followed by rotation) and relational motion modeling (multi-object interactions) remain unaddressed.
- The dataset is dominated by everyday actions and sports; industrial and medical motion scenarios are not yet covered.

## Related Work & Insights

- **vs. DANet/HPAN/TTI**: These semantics-centric FSVOS methods use images as support. This paper demonstrates their poor performance on MOVE due to the absence of motion modeling.
- **vs. LMPM (RVOS)**: Text-based motion description achieves only 41.8% on MOVE, far below DMA's 50.1%, confirming that many fine-grained motions cannot be accurately expressed in language.
- **vs. SAM2**: Despite being a powerful model, SAM2 requires a first-frame mask and is not applicable to the few-shot setting — MOVE represents an orthogonal task formulation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Motion-guided FSVOS constitutes an entirely new task definition; the frame-differencing decoupling strategy is both principled and efficient.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six baseline methods × 2 settings × 2 backbones × 2 splits, comprehensive ablations, and in-depth oracle analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated (the YouTube-VIS comparative experiment is highly convincing); visualizations are rich and informative.
- Value: ⭐⭐⭐⭐⭐ New dataset + new task + strong baseline; poised to advance motion-centric video understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](object-level_correlation_for_few-shot_segmentation.md)
- [\[ICCV 2025\] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup](dictas_a_framework_for_class-generalizable_few-shot_anomaly_segmentation_via_dic.md)
- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)

</div>

<!-- RELATED:END -->
