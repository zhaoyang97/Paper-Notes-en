---
title: >-
  [Paper Note] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking
description: >-
  [ECCV 2024][Video Understanding][Open-vocabulary tracking] SLAck proposes to uniformly fuse three cues—semantics, location, and appearance—during the early association stage of multi-object tracking. By learning implicit motion priors and cross-cue synergy through a lightweight Spatial-Temporal Object Graph (STOG), it avoids heuristic post-processing rules and significantly improves the tracking performance of novel categories on open-vocabulary MOT and TAO TETA benchmarks.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Open-vocabulary tracking"
  - "multi-object tracking"
  - "semantic-aware association"
  - "spatial-temporal object graph"
  - "feature fusion"
date: 2026-05-08
content_hash: 97845a2f820a951a
---

# SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking

**Conference**: ECCV 2024  
**arXiv**: [2409.11235](https://arxiv.org/abs/2409.11235)  
**Code**: [https://github.com/siyuanliii/SLAck](https://github.com/siyuanliii/SLAck)  
**Area**: Video Understanding  
**Keywords**: Open-vocabulary tracking, multi-object tracking, semantic-aware association, spatial-temporal object graph, feature fusion

## TL;DR

SLAck proposes to uniformly fuse three cues—semantics, location, and appearance—during the early association stage of multi-object tracking. By learning implicit motion priors and cross-cue synergy through a lightweight Spatial-Temporal Object Graph (STOG), it avoids heuristic post-processing rules and significantly improves the tracking performance of novel categories on open-vocabulary MOT and TAO TETA benchmarks.

## Background & Motivation

Multi-Object Tracking (MOT) is traditionally limited to a few categories such as pedestrians and vehicles. Open-vocabulary tracking extends tracking to hundreds of categories, but also introduces major challenges—appearances, behaviors, and motion patterns differ wildly across different categories.

**Three Limitations of Prior Work**:

**Motion Cues (Kalman Filter)**: Relies on linear motion assumptions, which are effective in pedestrian/vehicle scenarios, but severe failures occur in open-vocabulary scenarios where object motion is highly non-linear (e.g., running animals, tumbling objects).

**Appearance Cues (Pure Appearance Matching)**: State-of-the-art methods (e.g., OVTrack, MASA) primarily rely on appearance similarity, which suffers from issues like sensitivity to occlusion, confusion of similar-looking targets, and overfitting to base classes.

**Semantic Cues**: Existing methods either entirely ignore semantics or only use them heuristically at the final stage via hard grouping (same-class association) or soft grouping, which performs poorly under unstable classification in open-vocabulary scenarios.

**Key Challenge**: Different cues have their own strengths and weaknesses, but existing hybrid methods fuse them at the finalized association stage via heuristic rules (e.g., weighted average of IoU matrix and appearance matrix). This late fusion fails to learn the synergistic relationships among cues.

**Key Insight**: Motion patterns are highly correlated with semantic categories—if a model learns the motion patterns of horses during training, it can transfer this knowledge to unseen zebras through semantic similarity. This implies that the joint modeling of semantics and motion is crucial for generalization to novel categories.

**Core Idea**: Uniformly fuse semantic, location, and appearance cues in the early stage of association, replacing heuristic post-processing with a learnable spatial-temporal object graph to end-to-end optimize and yield a single association matrix.

## Method

### Overall Architecture

SLAck is built on top of a pre-trained open-vocabulary detector. The pipeline consists of three steps: (1) extracting three embeddings—semantic, location, and appearance—from the frozen detector; (2) fusing them via feature summation into a unified representation, which is then fed into the Spatial-Temporal Object Graph (STOG); (3) STOG modeling object dynamics via intra-frame self-attention and inter-frame cross-attention, ultimately outputting an association matrix trained end-to-end using a differentiable Sinkhorn algorithm.

### Key Designs

1. **Three-Cue Extraction Head (Semantic / Location / Appearance Head)**:

    - **Function**: Extract three complementary target descriptors from the frozen detector.
    - **Mechanism**:
        - **Semantic Head**: Uses the class embeddings from the CLIP-aligned RCNN classification head, projected through a 5-layer MLP to obtain the semantic embedding $E_{\text{sem}}$. This allows configuration for novel categories without retraining.
        - **Location Head**: Normalizes bounding box coordinates—using the image center as the origin and 70% of the maximum dimension as the scaling factor: $\left(\frac{x_{\min} - W/2}{s}, \frac{y_{\min} - H/2}{s}, \frac{w}{s}, \frac{h}{s}\right)$, then projected via MLP to the location embedding $E_{\text{loc}}$.
        - **Appearance Head**: Processes RoI features through 4 convolutional layers + MLP to output the appearance embedding $E_{\text{app}}$.
    - **Design Motivation**: Freezing the detector ensures original detection performance does not degrade; normalized coordinates guarantee scale invariance; the three embeddings capture different aspects of the target.

2. **Spatial-Temporal Object Graph (STOG)**:

    - **Function**: Models spatial relations among intra-frame targets and temporal correspondence of inter-frame targets.
    - **Mechanism**: First fuses the three embeddings via addition $E_{\text{fused}}^i = E_{\text{app}}^i + E_{\text{loc}}^i + E_{\text{sem}}^i$, and then alternates between:
        - **Intra-frame self-attention** (Spatial Object Graph): $\text{SA}_K(Q_K, K_K, V_K) = \sigma\left(\frac{Q_K K_K^T}{\sqrt{d}}\right)V_K$, processing object relations within the key frame and reference frame respectively, enabling the model to perceive the relative positions and mutual relationships of targets within a frame.
        - **Inter-frame cross-attention** (Temporal Object Graph): $\text{CA}_{K \to R}(Q_K, K_R, V_R)$, aligning and updating target features across different frames to capture temporal motion patterns.
    - **Design Motivation**: Replaces the linear motion assumption of explicit Kalman Filters. It learns implicit motion priors from data through attention mechanisms, capturing both linear and non-linear motions. Intra-frame self-attention allows the model to understand scene-level object layout, and inter-frame cross-attention achieves feature alignment across frames.

3. **Detection Aware Training (DAT)**:

    - **Function**: Addresses the issue of incomplete annotations in the TAO dataset.
    - **Mechanism**: Freezes the detector weights and uses the predicted bounding boxes from the detector (rather than only sparse GT) as training inputs, calculating the association loss only when predicted boxes match GT.
    - **Design Motivation**: Directly training on sparse GT leads to domain discrepancy between training and testing. DAT simulates testing conditions, ensuring the model sees a detection box distribution during training that is consistent with inference, boosting AssocA by $+13.7$.

### Loss & Training

- Uses the differentiable Sinkhorn algorithm to solve the optimal transport problem: $\mathcal{L}_{\text{Sinkhorn}} = -\sum_{i,j} T_{ij}' \log(S_{ij}')$
- The target matching matrix $\mathbf{T}$ is constructed from GT correspondences, adding a "dustbin" class to handle appearing/disappearing targets.
- End-to-end training without additional heuristic rules.
- Training frame pairs are sampled from adjacent frames within 3 seconds.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|---|---|---|---|---|
| OV-MOT val (Novel) | TETA | 31.1 | 30.0 (MASA-R50) | +1.1 |
| OV-MOT val (Novel) | AssocA | 37.8 | 34.6 (MASA-R50) | +3.2 |
| OV-MOT test (Novel) | TETA | 27.1 | 24.1 (OVTrack) | +3.0 |
| TAO TETA (Swin-L) | AssocA | 41.8 | 40.9 (GLEE-Plus) | +0.9 |
| TAO TETA (Swin-T) | AssocA | 38.9 | 36.7 (TETer-T) | +2.2 |

### Ablation Study

| Configuration | AssocA | Explanation |
|---|---|---|
| Lck (Location Only) | 28.3 | Implicit motion, already outperforms KF-based OC-SORT (20.4) |
| SLck (Semantic + Location) | 35.4 (+7.1) | Semantics significantly improves motion tracking |
| Ack (Appearance Only) | 32.7 | Pure appearance baseline |
| SAck (Semantic + Appearance) | 35.1 (+2.4) | Semantics also improves appearance tracking |
| LAck (Location + Appearance) | 36.4 | Hybrid but without semantics |
| SLAck (Full Model) | 37.8 (+1.4) | Optimal performance with three-cue synergy |
| w/o DAT | 24.1 | DAT yields +13.7 |
| Hard Grouping vs SLAck-SAck | 30.6 vs 38.0 | Early fusion far outperforms hard grouping |

### Key Findings

- **Semantic cues offer the largest improvement for novel category tracking**: Merely adding semantics boosts location tracking AssocA from 28.3 to 35.4 ($+7.1$), even surpassing pure appearance SOTA (OVTrack 33.6).
- **Temporal graphs (TOG) are more important for semantic and location cues** ($+2.4$ and $+4.1$), while spatial graphs (SOG) are more crucial for appearance ($+0.9$).
- **The DAT strategy has a huge impact**: $+13.7$ AssocA, highlighting that resolving the training-testing distribution domain gap is key.
- Semantic cues alone are insufficient to replace appearance ($-4.4$), but are highly effective as a complement.

## Highlights & Insights

- Comparison experiments of **early fusion vs. late heuristic fusion** are highly convincing—hard grouping degrades performance by $-4.6$, whereas SLAck's early semantic fusion improves it by $+2.8$.
- The insight regarding **semantic-motion synergy** is clever: motion patterns learned on base categories can be transferred to novel categories via semantic similarity (e.g., horse $\rightarrow$ zebra).
- **Implicit motion modeling** replaces explicit Kalman Filters, rendering it more robust to non-linear motions in open-vocabulary scenarios.
- The DAT training strategy is simple and effective, easily transferable to any MOT method utilizing incomplete annotations.

## Limitations & Future Work

- Training and evaluation are limited to only one large-vocabulary dataset (TAO); generalization needs verification on more datasets.
- Currently utilizing ResNet-50 as the backbone, leading to lower localization accuracy compared to methods leveraging stronger backbones (e.g., GroundingDINO).
- The computational complexity of the STOG attention mechanism scales with the number of objects, posing potential efficiency issues in crowded scenarios.
- The CLIP-aligned categorization capability of the semantic head might still be unstable for long-tail categories.

## Related Work & Insights

- **vs. OVTrack**: Relies on pure appearance matching + Stable Diffusion augmentation, ignoring semantics and location; SLAck outperforms it on Novel AssocA by $+4.2$.
- **vs. MASA**: Learns a universal appearance model without using semantics; SLAck outperforms it on TETA by $+1.1$.
- **vs. TETer**: Employs CEM encoding for late soft grouping; SLAck's early fusion surpasses it by $+2.2$ AssocA (using the same backbone).
- **vs. GLEE**: A foundation model trained on tens of millions of images; SLAck outperforms it on AssocA by $+0.9$ while only using the TAO training set.

## Rating

- Novelty: ⭐⭐⭐⭐ Clear proposal to replace late heuristic fusion with early fusion, valuable insight on semantic-motion synergy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely exhaustive ablations, analyzing the contribution of each cue and module individually.
- Writing Quality: ⭐⭐⭐⭐ Fully justified motivations with clear figures and tables, although the methodology section is notation-heavy and requires careful reading.
- Value: ⭐⭐⭐⭐ Provides a clean, unified framework for open-vocabulary tracking; findings regarding the importance of semantics offer great guidance to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[ECCV 2024\] SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild](semtrack_a_large-scale_dataset_for_semantic_tracking_in_the_wild.md)
- [\[CVPR 2025\] Anomize: Better Open Vocabulary Video Anomaly Detection](../../CVPR2025/video_understanding/anomize_better_open_vocabulary_video_anomaly_detection.md)
- [\[ECCV 2024\] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [\[ICLR 2026\] Video-STAR: Reinforcing Open-Vocabulary Action Recognition with Tools](../../ICLR2026/video_understanding/video-star_reinforcing_open-vocabulary_action_recognition_with_tools.md)

</div>

<!-- RELATED:END -->
