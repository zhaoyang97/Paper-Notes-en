---
title: >-
  [Paper Note] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers
description: >-
  [ECCV 2024][Video Understanding][3D Multi-Object Tracking] This paper provides an in-depth analysis of the fundamental cause of the performance conflict between detection and tracking tasks in end-to-end 3D trackers—subtle differences in positive sample assignment lead to contradictory classification gradients. It proposes OneTrack, which leverages gradient coordination, query grouping, and attention masking to achieve conflict-free joint optimization of detection and trackin…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "3D Multi-Object Tracking"
  - "Detection-Tracking Conflict"
  - "Gradient Coordination"
  - "End-to-End Tracking"
  - "nuScenes"
date: 2026-05-08
content_hash: 8bd8d4b329739b54
---

# OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Video Understanding / 3D Object Tracking  
**Keywords**: 3D Multi-Object Tracking, Detection-Tracking Conflict, Gradient Coordination, End-to-End Tracking, nuScenes

## TL;DR
This paper provides an in-depth analysis of the fundamental cause of the performance conflict between detection and tracking tasks in end-to-end 3D trackers—subtle differences in positive sample assignment lead to contradictory classification gradients. It proposes OneTrack, which leverages gradient coordination, query grouping, and attention masking to achieve conflict-free joint optimization of detection and tracking under a unified feature representation for the first time, achieving SOTA performance on nuScenes.

## Background & Motivation

**Background**: Vision-based 3D Multi-Object Tracking (MOT) is a critical task in autonomous driving perception. In recent years, end-to-end paradigms have emerged, where query-based detection frameworks represented by DETR are extended to tracking tasks by propagating object queries from the previous frame to the current frame. Representative methods like MUTR3D, PF-Track, and StreamPETR adopt similar architectures to perform 3D detection and tracking simultaneously using multi-view camera inputs.

**Limitations of Prior Work**: Existing end-to-end trackers generally face a key issue—**performance conflict between detection and tracking tasks**. Specifically, when detection and tracking are jointly optimized, the performance of neither task reaches the level of individual optimization, with tracking performance in particular dropping significantly. Previous research attributed this to "different tasks requiring different object features," but this explanation is too general and fails to point out the specific technical causes and solutions.

**Key Challenge**: Although both detection and tracking require target localization and classification, they differ subtly but critically in the definition of "which queries should be considered positive samples." The detection task treats new queries that match the ground truth best as positive samples, while the tracking task treats propagated queries associated with previous target IDs as positive samples. **The same query may be positive in the detection task (due to a spatial match with some GT) but negative in the tracking task (due to a mismatched ID)**, and vice versa. This results in the classification head receiving contradictory gradient signals, failing to satisfy both tasks simultaneously.

**Goal**: (1) Pinpoint the root cause of the detection-tracking conflict; (2) Propose concrete gradient-level solutions; (3) Develop the first conflict-free, single-stage end-to-end joint detection and tracking model.

**Key Insight**: Through a careful analysis of positive sample assignment strategies and gradient flows, the authors discovered that the essence of the conflict is a classification gradient polarity conflict—certain queries receive classification gradients in opposite directions in the two tasks (one pushing the classification score up, the other pulling it down). Once identified, the solution becomes clear: queries must be grouped based on their polarity in both tasks, and information interaction between conflicting queries must be blocked.

**Core Idea**: Identify and coordinate classification gradient contradictions caused by positive sample assignment discrepancies between detection and tracking tasks, enabling conflict-free joint optimization through query grouping and attention masking.

## Method

### Overall Architecture
OneTrack is based on a standard query-based 3D detection and tracking framework. The input consists of multi-view camera image sequences. After extracting features through an image backbone, two groups of queries are used for object representation: Detection Queries for detecting new objects, and Track Queries propagated from the previous frame to track existing objects. Both query groups share a decoder for feature updates, outputting 3D bounding boxes and classification scores. The key innovations reside in gradient coordination and selective attention masking among the query groups.

### Key Designs

1. **Gradient Polarity Coordination**:

    - **Function**: Eliminate contradictory classification gradients on the same query from detection and tracking tasks.
    - **Mechanism**: First, analyze the positive and negative sample attribution of each query in both detection and tracking tasks, dynamically partitioning all queries into four groups: (a) positive in both tasks (Pos-Pos)—no conflict; (b) detection-positive/tracking-negative (Pos-Neg)—conflict group; (c) detection-negative/tracking-positive (Neg-Pos)—conflict group; (d) negative in both tasks (Neg-Neg)—no conflict. For queries in conflict groups (b) and (c), their classification loss gradients require special handling. Specifically, the classification targets for conflicting queries are modified to prevent the two tasks from yielding completely opposite optimization signals. For example, for Pos-Neg queries, the positive gradient of the detection task is preserved but the target in the tracking task is modified so that it does not generate a conflicting negative gradient.
    - **Design Motivation**: This is a precise surgical intervention on the source of the conflict—rather than simply decoupling the two tasks (which would sacrifice the advantages of end-to-end training), it resolves the issue specifically at the gradient conflict level.

2. **Polarity-based Query Grouping & Attention Masking**:

    - **Function**: Prevent feature contamination between conflicting queries.
    - **Mechanism**: In the self-attention layers of the Transformer decoder, selective masking is applied to attention between queries based on the four classification groups mentioned above. The specific rule is: **attention between query groups with conflicting positive sample assignments is masked**. That is, queries in the Pos-Neg group cannot attend to queries in the Neg-Pos group, and vice versa. This prevents conflict propagation at the feature level—if a detection-positive query attends to a tracking-positive query (yet they hold opposite detection/tracking polarities), the feature update may drift toward contradictory directions. The specific pattern of the attention mask is dynamically generated based on the polarity of each query in the current frame.
    - **Design Motivation**: Gradient coordination resolves conflicts at the loss level, but if conflicting queries remain coupled at the feature level, conflict signals will still propagate through the attention mechanism. Attention masking further isolates conflicts from the perspective of feature interaction.

3. **Tracking Classification Loss Modification**:

    - **Function**: Suppress inaccurate predictions in the tracking task to improve tracking quality.
    - **Mechanism**: Standard classification losses can be overconfident for track queries—a query propagated from the previous frame might not actually match the object due to inaccurate localization predictions, yet the classification loss still treats it as a positive sample. The authors introduce a localization-quality-based weighting to the tracking classification loss: $\mathcal{L}_{cls}^{track} = w_i \cdot \text{FocalLoss}(p_i, y_i)$, where the weight $w_i$ is determined by the IoU between the predicted box of the tracking query and the ground truth. Tracking queries with low IoU (inaccurate localization) have their classification losses downweighted, preventing inaccurate high-confidence predictions from interfering with tracking association.
    - **Design Motivation**: Tracking quality depends not only on correct ID association but also on localization accuracy. Integrating localization quality into the classification objective allows classification scores to truly reflect tracking reliability.

### Loss & Training
The total loss consists of: (1) detection classification loss $\mathcal{L}_{cls}^{det}$ (Focal Loss); (2) detection regression loss $\mathcal{L}_{reg}^{det}$ (L1 + GIoU); (3) modified tracking classification loss $\mathcal{L}_{cls}^{track}$; (4) tracking regression loss $\mathcal{L}_{reg}^{track}$. Hungarian matching is used for positive sample assignment. A two-stage training strategy is adopted: first pre-training the detection capability on short sequences (2 frames), then fine-tuning the tracking capability on long sequences (multiple frames).

## Key Experimental Results

### Main Results

| Method | nuScenes Val AMOTA↑ | nuScenes Val AMOTP↓ | nuScenes Test AMOTA↑ | Remarks |
|------|---------------------|---------------------|----------------------|------|
| OneTrack (Ours) | 55.8 | 1.21 | 51.2 | Single-stage joint model |
| MUTR3D | 45.1 | 1.45 | 42.8 | Two-stage |
| PF-Track | 48.9 | 1.32 | 48.1 | Two-stage |
| StreamPETR | 50.4 | 1.28 | 49.5 | Suffers from detection-tracking conflict |
| DQTrack | 52.3 | 1.25 | 49.8 | Decoupled features |

### Ablation Study

| Configuration | AMOTA↑ | AMOTP↓ | mAP (Det)↑ | Explanation |
|------|--------|--------|-----------|------|
| OneTrack Full | 55.8 | 1.21 | 44.2 | Full model |
| w/o Gradient Coordination | 51.2 | 1.30 | 42.5 | Remove gradient polarity coordination |
| w/o Attention Masking | 53.1 | 1.26 | 43.3 | Remove query grouping masking |
| w/o Classification Loss Modification | 54.3 | 1.23 | 44.0 | Remove tracking classification weighting |
| w/o All Improvements (baseline) | 48.9 | 1.35 | 41.8 | Baseline without conflict mitigation |
| Fully Decoupled (two independent heads) | 52.0 | 1.27 | 43.8 | Separate detection and tracking features |

### Key Findings
- Gradient coordination is the most critical component; removing it drops AMOTA by 4.6%, indicating that gradient polarity conflict is indeed the core performance bottleneck.
- Attention masking and classification loss modification contribute 2.7% and 1.5% AMOTA improvements, respectively, and the three components are complementary.
- Compared to the "fully decoupled" strategy, OneTrack's unified features + conflict mitigation scheme is superior (55.8 vs 52.0 AMOTA), suggesting that sharing features between detection and tracking is beneficial if conflicts are correctly handled.
- On the nuScenes test set, OneTrack outperforms all previous methods by +3.1% AMOTA, validating the practical effectiveness and generalizability of the method.
- Detection performance (mAP) also improves due to conflict mitigation, indicating that the conflict is mutual—tracking also drags down detection.

## Highlights & Insights
- **Depth of Problem Diagnosis**: Rather than settling for a vague understanding of "conflicts between detection and tracking," this work traces it all the way to its root cause: positive sample assignment discrepancy leading to classification gradient polarity contradictions. This in-depth analysis from phenomenon to mechanism is a hallmark of top-tier conference papers.
- **Precise Solutions**: Three complementary components (gradient coordination, attention masking, and loss modification) are designed specifically for the root cause. Each has a clear motivation and effect, leaving no redundant designs.
- **New Take on Unified vs. Decoupled**: For a long time, the default solution to handle multi-task conflicts was "decoupling features." This paper demonstrates that "unified features + conflict-aware optimization" is a superior choice. This insight can be extended to other multi-task learning scenarios.
- **Dynamism of Query Grouping**: The polarity grouping of queries is dynamically determined at the frame level (the same query may belong to different polarity groups in different frames). The design fully accounts for this dynamism.

## Limitations & Future Work
- **Limited to Camera Input**: Current evaluations only test camera-input 3D tracking. Whether detection-tracking conflicts exist and whether the solutions generalize to LiDAR-based or fusion-based scenarios remains unverified.
- **Computational Overhead**: Dynamic generation of query grouping and attention masks introduces certain computational overhead, which may require optimization for real-time deployment scenarios.
- **Occlusion and Re-appearance**: The paper does not thoroughly discuss re-identification capabilities when targets reappear after long-term occlusion, which is important in real-world scenarios.
- **Sensitivity to Positive Sample Assignment Strategies**: The severity of conflicts depends on the positive sample assignment strategies of detection and tracking (like different IoU thresholds). Conflict patterns may vary under different configurations.
- **Future Directions**: Extend the gradient coordination concept to LiDAR 3D tracking; explore better dynamic positive sample assignment strategies to reduce conflicts from the root.

## Related Work & Insights
- **vs MUTR3D**: MUTR3D is an early end-to-end 3D tracker that directly extends detection queries to track queries without handling task conflicts, severely limiting its performance. OneTrack addresses this issue from its root cause.
- **vs StreamPETR**: StreamPETR performs online tracking via streaming query propagation, which achieves good performance but still suffers from conflicts. OneTrack proposes a conflict mitigation scheme on top of StreamPETR.
- **vs Multi-task Gradient Coordination Methods (e.g., GradNorm, PCGrad)**: These general multi-task methods coordinate gradients at the task level but do not consider sample-level polarity conflicts. OneTrack's contribution is identifying this finer-grained, sample-level gradient conflict.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The root-cause analysis of detection-tracking conflicts is profound and original, and the solution is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale nuScenes verification + detailed ablation studies + clear contribution from each component.
- Writing Quality: ⭐⭐⭐⭐⭐ Accurate problem definition, rigorous logical analysis, and complete experimental design.
- Value: ⭐⭐⭐⭐ Significantly drives the field of end-to-end tracking forward; the conflict coordination idea is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](../../CVPR2026/video_understanding/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)
- [\[ECCV 2024\] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)
- [\[CVPR 2026\] TimeBridge: Self-Supervised Video Representation Learning via Start-End Joint Embedding and In-Between Frame Prediction](../../CVPR2026/video_understanding/timebridge_self-supervised_video_representation_learning_via_start-end_joint_emb.md)
- [\[ECCV 2024\] SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow](spamming_labels_efficient_annotations_for_the_trackers_of_tomorrow.md)
- [\[ECCV 2024\] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)

</div>

<!-- RELATED:END -->
