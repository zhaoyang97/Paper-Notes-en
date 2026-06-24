---
title: >-
  [Paper Note] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement
description: >-
  [ECCV 2024][Segmentation][Video Instance Segmentation] To address the association errors caused by over-reliance on spatial-temporal position information in existing online video instance segmentation (VIS) methods, VISAGE is proposed to enhance instance association accuracy. By explicitly extracting appearance embeddings from backbone features, combined with contrastive learning and a simplified tracker, the method achieves state-of-the-art results on YTVIS and OVIS benchmar…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Video Instance Segmentation"
  - "Appearance Guidance"
  - "Query Matching"
  - "Contrastive Learning"
  - "Memory Bank"
date: 2026-05-08
content_hash: e45db0d417cad05d
---

# VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement

**Conference**: ECCV 2024  
**arXiv**: [2312.04885](https://arxiv.org/abs/2312.04885)  
**Code**: [https://github.com/KimHanjung/VISAGE](https://github.com/KimHanjung/VISAGE)  
**Area**: Segmentation  
**Keywords**: Video Instance Segmentation, Appearance Guidance, Query Matching, Contrastive Learning, Memory Bank

## TL;DR

To address the association errors caused by over-reliance on spatial-temporal position information in existing online video instance segmentation (VIS) methods, VISAGE is proposed to enhance instance association accuracy. By explicitly extracting appearance embeddings from backbone features, combined with contrastive learning and a simplified tracker, the method achieves state-of-the-art results on YTVIS and OVIS benchmarks.

## Background & Motivation

Video instance segmentation requires classifying, segmenting, and tracking different instances across video sequences. Recently, query-based detectors (e.g., Mask2Former) have significantly advanced online VIS methods, which primarily follow query-propagation or query-matching strategies.

However, the authors identified a critical limitation through in-depth analysis: **existing methods heavily rely on positional information while neglecting appearance details.** This is manifested as:
1. In scene-cut scenarios, abrupt changes in object positions lead to tracking failures.
2. In trajectory-crossing scenarios, position-based matching causes ID switches.
3. Experimental validation using pseudo-videos generated from horizontally flipped images shows that existing models still suffer from association errors even when object appearances are distinct (because they tend to maintain the spatial order predicted in previous frames).

These findings suggest that appearance information is a crucial dimension for object matching, particularly when spatial cues are insufficient for identity disambiguation.

## Method

### Overall Architecture

VISAGE is built upon the Mask2Former detector and comprises three core components:
1. Detector: A standard backbone + transformer encoder + transformer decoder architecture.
2. Appearance-Guided Enhancement Module: Extracts appearance queries from backbone features via mask pooling.
3. Simplified Tracker: Jointly utilizes object embeddings and appearance embeddings for Hungarian matching.

### Key Designs

1. **Appearance Query Extraction**:

    - Core operation: Average pool the backbone feature map using masks predicted by object queries (mask pooling) to obtain the appearance query.
    - Design Motivation: Traditional tracking methods (e.g., RoIPool/RoIAlign) have long used feature maps to extract instance features, but modern query-based methods have lost this capability.
    - Project the two types of queries into appearance embeddings $\mathbf{e}_a \in \mathbb{R}^{N \times C}$ and object embeddings $\mathbf{e}_i \in \mathbb{R}^{N \times C}$ respectively.
    - Key Finding: Utilizing backbone features yields significantly better results than using transformer encoder features (AP: 55.1 vs 51.4), because the backbone preserves richer visual information.

2. **Contrastive Learning for Enhancing Embedding Discriminativeness**:

    - Apply contrastive loss separately to object embeddings and appearance embeddings.
    - Pull embeddings of the same instance in different frames closer, while pushing embeddings of different instances further apart.
    - Key Challenge: Unlike prior methods (IDOL, CTVIS), the two types of embeddings are processed separately, allowing each embedding to maintain its distinct properties and complement each other during matching.
    - The contrastive loss weight is set to 2.0, which is weighted and summed with the detector's original loss.

3. **Simplified Tracker and Memory Bank**:

    - Matching score: $\mathbf{s} = (1-\alpha) \cdot \cos(\mathbf{e}_i^t, \mathbf{m}_i^t) + \alpha \cdot \cos(\mathbf{e}_a^t, \mathbf{m}_a^t)$
    - Employs the Hungarian algorithm for optimal assignment, with $\alpha=0.75$ used during inference.
    - Memory bank size $W=5$. When retrieving memory embeddings, temporal weighting and confidence weighting are used: $\mathbf{m}^t = \sum_{w=1}^{W} \mathbf{e}^{t-w} s^{t-w} \times \frac{W}{w}$
    - Avoids heuristic operations like NMS or tracklet initialization/deletion thresholds, significantly simplifying the pipeline.

### Loss & Training

- Base Loss: Inherits the loss functions and weights from Mask2Former.
- Additional Loss: Contrastive losses are applied to both appearance embeddings and object embeddings with a weight of 2.0.
- Backbone: ResNet-50, initialized with COCO pre-training, employing COCO joint training strategy.
- Batch Size: 16 videos.
- Training Hardware: 4 NVIDIA A6000 GPUs.

## Key Experimental Results

### Main Results

| Method | Setting | YTVIS19 AP | YTVIS21 AP | OVIS AP |
|------|------|-----------|-----------|---------|
| MinVIS | online | 47.4 | 44.2 | 25.0 |
| IDOL | online | 49.5 | 43.9 | 30.2 |
| GenVIS | online | 50.0 | 47.1 | 35.8 |
| DVIS | online | 51.2 | 46.4 | 31.0 |
| TCOVIS | online | 52.3 | 49.5 | 35.3 |
| CTVIS | online | 55.1 | 50.1 | 35.5 |
| **VISAGE** | **online** | **55.1** | **51.6** | **36.2** |

### Ablation Study

| Appearance Info | Memory Bank | YTVIS19 AP | AP50 | AP75 | AR1 | AR10 |
|---------|-------|-----------|------|------|-----|------|
| ✗ | ✗ | 49.9 | 71.4 | 54.7 | 47.0 | 58.7 |
| ✗ | ✓ | 50.2 | 72.1 | 54.7 | 47.3 | 60.7 |
| ✓ | ✗ | 53.4 | 76.8 | 58.7 | 49.8 | 61.2 |
| ✓ | ✓ | **55.1** | **78.1** | **60.6** | **51.0** | **62.3** |

Appearance weight $\alpha$ analysis ($\alpha=0$ represents position only, $\alpha=1$ represents appearance only):

| α | YTVIS19 AP | OVIS AP |
|---|-----------|---------|
| 0.00 | 53.4 | 32.2 |
| 0.25 | 53.6 | 34.5 |
| 0.50 | 54.5 | 34.8 |
| **0.75** | **55.1** | **36.2** |
| 1.00 | 24.9 | - |

### Key Findings

1. **Significant Appearance-Guided Boost**: Introducing appearance information leads to a gain of 3.5 AP (49.9 $\to$ 53.4), which is more than 10 times the gain from the memory bank (0.3 AP).
2. **More Pronounced Effect in Complex Scenarios (OVIS)**: OVIS AP increases from 32.2 to 36.2 (+4.0), demonstrating that position cues are less reliable under frequent occlusions.
3. **Both Appearance and Position are Indispensable**: Performance drops sharply to 24.9 AP when $\alpha=1.0$ (appearance only), indicating that they work synergistically.
4. **Simplified Tracker is Better**: The simplified tracker without heuristic NMS (55.1 AP) outperforms the traditional tracker with NMS (54.4 AP).
5. **Backbone Features Outperform Encoder Features**: Generating appearance queries from backbone features outperforms using transformer encoder features by 3.7 AP.

## Highlights & Insights

1. **Precise Problem Identification**: The "position bias" phenomenon is elegantly demonstrated through a simple horizontal image-flipping experiment.
2. **Extremely Simple Method**: The core innovation is merely a mask pooling operation paired with a weighted matching formula, yet it yields remarkable results.
3. **Breaking Convention**: Query-based methods typically assume that queries already contain sufficient information, but VISAGE demonstrates that explicit appearance feature extraction is both necessary and effective.
4. **Simplicity > Complexity**: Removing numerous heuristic hyperparameters (like NMS thresholds and tracklet management thresholds) actually increases performance.

## Limitations & Future Work

1. **Limited to ResNet-50**: The method has not been verified on stronger backbones (like Swin-L), which could potentially yield greater improvements.
2. **Insufficient Globality of Appearance Features**: The appearance feature from the current mask pooling represents a "local average", which might lose fine-grained texture information.
3. **Fixed Memory Bank Size**: A window size of $W=5$ may not apply to all video lengths; longer videos might require a larger memory bank.
4. **Unexplored Online Learning**: Contrastive learning for appearance embeddings is performed only during training and is not updated during inference.

## Related Work & Insights

- MinVIS pioneered the paradigm of training only the detector and performing frame-to-frame matching during inference.
- CTVIS uses a memory bank for contrastive learning during training to enhance discriminability.
- The way traditional trackers (e.g., RoIPool/RoIAlign) extract instance features inspired VISAGE.
- The concept of appearance guidance can be extended to other query-based vision tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Insightful problem identification, though the method itself (mask pooling + contrastive learning) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across four benchmarks and detailed ablation studies (feature sources, $\alpha$ values, tracker designs).
- Writing Quality: ⭐⭐⭐⭐⭐ — Excellent proof-of-concept design with the image-flipping experiment, and a clear algorithmic pipeline.
- Value: ⭐⭐⭐⭐ — Provides a simple yet effective solution for improving VIS, offering strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Dataset Enhancement with Instance-Level Augmentations](dataset_enhancement_with_instance-level_augmentations.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[ECCV 2024\] General and Task-Oriented Video Segmentation](general_and_task-oriented_video_segmentation.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[ECCV 2024\] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)

</div>

<!-- RELATED:END -->
