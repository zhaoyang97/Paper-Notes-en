---
title: >-
  [Paper Note] TAPTR: Tracking Any Point with Transformers as Detection
description: >-
  [ECCV 2024][Object Detection][Point tracking] TAPTR reformulates the Tracking Any Point (TAP) task as a DETR-like detection problem. It represents each tracking point as a point query containing both position and content, which is layer-wise optimized through a multi-layer Transformer decoder. Combined with a cost volume and sliding-window feature update strategy, it achieves SOTA performance on the TAP-Vid benchmark with faster inference speed.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Point tracking"
  - "Transformer"
  - "DETR"
  - "optical flow"
  - "TAP"
date: 2026-05-08
content_hash: 9158a25b25882363
---

# TAPTR: Tracking Any Point with Transformers as Detection

**Conference**: ECCV 2024  
**arXiv**: [2403.13042](https://arxiv.org/abs/2403.13042)  
**Code**: [https://taptr.github.io](https://taptr.github.io)  
**Area**: Video Understanding  
**Keywords**: Point tracking, Transformer, DETR, optical flow, TAP

## TL;DR

TAPTR reformulates the Tracking Any Point (TAP) task as a DETR-like detection problem. It represents each tracking point as a point query containing both position and content, which is layer-wise optimized through a multi-layer Transformer decoder. Combined with a cost volume and sliding-window feature update strategy, it achieves SOTA performance on the TAP-Vid benchmark with faster inference speed.

## Background & Motivation

**Background**: Understanding pixel-wise motion in videos is a fundamental computer vision task. Optical flow estimation is the mainstream approach but only processes correspondences between consecutive frames, struggling to handle occlusion. Semantic keypoint tracking can address occlusion but suffers from restricted semantic categories (e.g., human joints). Recently, the Tracking Any Point (TAP) task has been introduced to track arbitrary user-specified points across an entire video. Representative methods include PIPs, TAP-Net, TAPIR, and CoTracker.

**Limitations of Prior Work**: Existing TAP methods (PIPs, TAPIR, CoTracker) lack clear modeling of tracking points, simply concatenating multiple features—such as flow vectors, flow embeddings, visibility, content features, and cost volumes—into a "black-box" vector. This vector is then fed into an MLP or Transformer with the expectation that the model will interpret and utilize it, which lacks structured design and hinders understanding and optimization.

**Key Challenge**: The TAP task requires simultaneously handling long-range temporal modeling (recovery from occlusion) and detailed low-level feature matching (precise localization), demanding a unified framework that is both simple and powerful. Most prior methods independently process each tracking point, ignoring the contextual information shared among points belonging to the same object.

**Goal**: To design a point tracking framework with simple concepts and clear definitions, ensuring that each component has a distinct physical meaning while outperforming existing methods in both performance and speed.

**Key Insight**: It is observed that point tracking is highly analogous to object detection/tracking—in each frame, the tracking points are essentially targets to be detected. Therefore, the mature design of the DETR series can be directly borrowed.

**Core Idea**: Modeling each tracking point as a DETR-like point query (position part + content part), which is optimized layer-by-layer using a Transformer decoder. This naturally reuses the self-attention, cross-attention, and iterative refinement mechanisms well-proven in detection tasks.

## Method

### Overall Architecture

TAPTR consists of four main parts: (1) **Video Preparation** — extracts multi-scale feature maps using a CNN backbone + Transformer Encoder; (2) **Query Preparation** — prepares the initial position, content features, and cost volume for each tracking point in each frame; (3) **Point Decoder** — processes all frames within a sliding window in parallel using a multi-layer Transformer decoder; (4) **Window Post-processing** — updates and propagates the explicit status and content features of queries across windows. A sliding-window strategy is used to process $W$ frames at a time (default $W=8$, stride 4).

### Key Designs

1. **Point Query Modeling**:

    - **Function**: In each frame, each tracking point is represented as a point query $q_t^i = [f_t^i, l_t^i]$, consisting of two components: content feature $f_t^i$ and position $l_t^i$, offering clear representation.
    - **Mechanism**: The content feature is obtained via bilinear interpolation on the multi-scale feature maps of the frame where the tracking point first appears: $f_e^i = \text{MLP}(\text{Cat}(\text{Bili}(F_{e^i,1}, l_e^i), \ldots, \text{Bili}(F_{e^i,S}, l_e^i)))$. All queries belonging to the same tracking point share the initial content features and initial position.
    - **Design Motivation**: In contrast to prior methods that concatenate all information into a black-box vector, the DETR-like query (position + content) has explicit physical meanings—position describes "where" and content describes "what"—consistent with the query design in detection.

2. **Cost Volume Aggregation**:

    - **Function**: Computes the inner product between the point query and image features to obtain visual similarity maps, and locally samples the cost volume within the decoder to enhance the query content features.
    - **Mechanism**: The cost volume is computed once before the decoder starts as $C_{t,s}^i = \text{InnerProd}(F_{t,s}, f_t^i)$, and reused as a static feature map across decoder layers. A RAFT-style local grid sampling is adopted: $c_{t,s}^i = \text{GridSample}(C_{t,s}^i, \text{Grid}(l_t^i, G))$, and the sampled cost vector is concatenated with the content feature and fused via an MLP.
    - **Design Motivation**: (a) Point tracking requires more local low-level features for precise localization than object detection; (b) unlike prior methods that recompute the cost volume after each content feature update, one-time computation maintains decoder simplicity and multi-layer optimization stability; (c) cost volumes are well-proven in optical flow and stereo matching. Experiments demonstrate that one-time computation + window-wise updates surpass iteration-wise updates (+1.1 AJ on DAVIS, +2.8 AJ on RGB-Stacking).

3. **Point Decoder Multi-Module Design**:

    - **Visual Feature Enhancer (Cross-Attention)**: Uses 2D deformable attention to sample multi-scale local image features around tracking points, compensating for detailed geometric information missing from the cost volume.
    - **Point Query Self-Attention**: All point queries within the same frame interact via self-attention, augmented with sinusoidal positional encodings (reducing temperature $\tau$ for sharper positional embeddings to meet point tracking’s high-precision demands). A 100x temperature reduction yields a 1.1 AJ gain.
    - **Temporal Attention**: Queries belonging to the same tracking point undergo dense attention interaction along the temporal dimension $f^i \Leftarrow \text{Attention}(f^i, f^i)$, $f^i \in \mathbb{R}^{W \times C}$, modeling short-term temporal information.
    - **Residual Content Update**: Position refinement uses DETR-style iterative Sigmoid optimization. Content features use residual updates $f_t^i \Leftarrow f_t^i + \text{MLP}(\text{Cat}(f_t^i, f_{e^i}^i))$, always referencing the initial features to prevent drift. This outperforms the standard direct update of DETR by 1.7 AJ.

4. **Window Post-Processing and Feature Drift Mitigation**:

    - **Function**: Updates trajectory states across sliding windows while propagating content features to retain long-range temporal information, addressing the feature drift problem.
    - **Mechanism**: During training, feature updates are randomly dropped with a probability of 0.6 (Random Drop), forcing the network to adapt to both update and non-update scenarios. During inference, feature updates remain enabled, but feature padding is discarded at a dynamic frequency ($T/24$ window intervals), striking the optimal balance between temporal preservation and drift control.
    - **Design Motivation**: Direct, unrestricted content feature updates cause severe drift in long videos (AJ on RGB-Stacking drops drastically from 60.8 to 23.1), because videos during training are only 24 frames long while inference videos can be much longer, resulting in a length inconsistency issue.

### Loss & Training

- **Loss Function**: Multi-layer loss of the entire sequence, computed after obtaining the complete trajectory: $\text{Loss} = (\omega_V \cdot \text{CE}(V, \tilde{V}) + \sum_{d=1}^{D} \omega_L \cdot \text{L1}(L_d, \tilde{L})) / N$
- Positions use L1 loss, supervised at every decoder layer output (auxiliary loss); visibility uses cross-entropy loss, predicted only at the final layer.
- Trained on the TAP-Vid-Kubric synthetic dataset (11,000 videos of 24 frames), randomly sampling 700-800 points.
- Training resolution is $512 \times 512$, evaluation resolution is $256 \times 256$.
- Employs a ResNet50 backbone, a 2-layer Transformer Encoder, and a 6-layer decoder.
- Uses AdamW optimizer + EMA, trained for around 36,000 steps on 8 A100 GPUs with a learning rate of 2e-4 and 4 gradient accumulations (equivalent to batch size 32).

## Key Experimental Results

### Main Results

Comparison with SOTA on the TAP-Vid benchmark (First mode):

| Method | DAVIS AJ | DAVIS $<\delta_{avg}^x$ | DAVIS OA | RGB-Stacking AJ | Kinetics AJ | Speed (PPS) |
|---|---|---|---|---|---|---|
| TAPIR | 56.2 | 70.0 | 86.5 | 55.5 | 49.6 | - |
| CoTracker-All | 60.7 | 75.7 | 88.1 | - | - | 15.7 |
| CoTracker-Single | 62.2 | 75.7 | 89.3 | - | - | 0.8 |
| BootsTAP† (extra 15M data) | 61.4 | 74.0 | 88.4 | - | 54.7 | - |
| **TAPTR (Ours)** | **63.0** | **76.1** | **91.1** | **60.8** | 49.0 | **20.4** |

Key comparison: TAPTR outperforms CoTracker-All by 2.3 AJ (63.0 vs 60.7) and CoTracker-Single by 0.8 AJ (63.0 vs 62.2) with a speedup of **25x** (20.4 vs 0.8 PPS).

### Ablation Study

**Ablation of key components (DAVIS, Table 2)**:

| Configuration | AJ | $<\delta_{avg}^x$ | OA | Description |
|---|---|---|---|---|
| Full TAPTR | 63.0 | 76.1 | 91.1 | Full model |
| - Small Temperature | 61.9 | 75.4 | 90.3 | Temperature reduction is important for self-attention (-1.1) |
| - Transformer Encoder | 60.9 | 75.2 | 88.9 | Affects visibility estimation (-1.0) |
| - Self Attention | 58.4 | 72.1 | 88.3 | Inter-point interaction is crucial (-2.5) |
| - Temporal Attention | 51.6 | 66.7 | 84.5 | Temporal information is most critical (-6.8) |
| - Cost Volume | 46.8 | 61.3 | 82.4 | Foundation of visual similarity (-4.8) |
| - Cross Attention | 50.0 | 65.0 | 83.4 | Geometric information supplement (-1.6) |
| Original DETR Update (vs. Residual) | 45.1 | 60.0 | 82.4 | Residual update is crucial (-1.7) |

**Ablation of Cost Volume update frequency (Table 6)**:

| Update Strategy | DAVIS AJ | RGB-Stacking AJ | Description |
|---|---|---|---|
| Update at each iteration | 61.9 | 58.0 | Too frequent, unstable optimization objective |
| Never update | 63.0 | 58.3 | Acceptable for short videos, but insufficient information for long videos |
| **Update per window** | **63.0** | **60.8** | Best: Stability + long-range information |

### Key Findings

1. **Temporal information is the most critical factor**: Removing Temporal Attention + feature updates across windows leads to a 10.1 AJ degradation (primary contribution).
2. **Cost volume as indispensable basic perception**: Removing it drops performance by 4.8 AJ, consistent with empirical findings in optical flow.
3. **Low-temperature positional encodings are vital for point-level precision**: Scaling down the temperature by 100x allows neighboring queries to gather more attention, boosting performance by 1.1 AJ.
4. **Feature drift is the core challenge in long videos**: Unrestricted feature updates cause the AJ on RGB-Stacking (long videos) to plunge from 60.8 to 23.1.
5. **Multi-layer supervision is essential**: Supervising only the last layer vs. all layers: 55.4 vs 63.0 AJ, a gap of 7.6.
6. **Residual content updates outperform direct updates**: Referencing the initial features continuously avoids noise accumulation, bringing a 1.7 AJ gain.
7. **Positive returns with more decoder layers**: 2 layers -> 4 layers -> 6 layers correspond to 58.2 -> 61.6 -> 63.0 AJ.

## Highlights & Insights

- **Unified Perspective of Detection & Tracking**: Reformulating point tracking as frame-by-frame point detection, fully reusing the mature query mechanisms of DETR inside a conceptually simple yet powerful framework.
- **Clear Point Modeling**: Unlike prior methods using black-box feature concatenation, the design of positional + content queries is clearly defined, where each operation (self-attention, cross-attention, temporal attention) has a distinct physical interpretation.
- **One-time Cost Volume Computation Strategy**: Counter-intuitively avoiding frequent cost volume updates within the decoder maintains the stability of the optimization target, presenting a valuable engineering insight.
- **Alleviating Feature Drift with Random Drop**: A simple yet effective solution to the training-inference discrepancy, forcing the network to learn robust representations via random drops.
- **Highly Comprehensive Ablation Study**: Validating the contribution of each component step-by-step, providing a valuable reference baseline for future work.

## Limitations & Future Work

- **Training Data Limited to Synthetic Data**: Only trained on Kubric synthetic data. How to leverage real-world detection or segmentation annotations to assist TAP training remains an open question.
- **Mediocre Performance on Kinetics Dataset**: TAPTR's performance on Kinetics (49.0 AJ) is lower than BootsTAP (54.7), likely because BootsTAP leverages an additional 15M real video data.
- **Fixed Sliding Window Design**: Window size ($W=8$) and stride ($4$) are fixed; adaptive windowing strategies might perform better under varying video complexities.
- **Backbone Restricted to ResNet50**: Exploring stronger backbones (e.g., Swin Transformer, ConvNeXt) or self-supervised pre-trained features remains a potential direction.
- **Efficiency of Inter-point Interaction**: Global self-attention yields an $O(N^2)$ complexity when tracking a large number of points. Scenarios with large-scale point tracking may require more efficient interaction schemes.

## Related Work & Insights

- **DETR Series**: Directly borrowing query design, multi-layer iterative optimization, and auxiliary loss from DETR / Deformable DETR / DINO, validating the powerful transferability of detection frameworks to tracking tasks.
- **RAFT**: The local grid sampling strategy for cost volume is directly inherited from RAFT, acting as a crucial bridge from optical flow estimation to point tracking.
- **CoTracker**: The first method using Transformers for joint multi-point tracking. TAPTR surpasses it through clearer query modeling and positional encoding designs.
- **PIPs / TAPIR**: Pioneers of the iterative optimization paradigm in point tracking, which, however, process each point independently and lack point-to-point interaction.
- **DAB-DETR**: The query design with dynamic anchor boxes inspired the position update scheme of TAPTR.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of unifying point tracking into the DETR framework is novel and natural, though the core components are combined from existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablation studies cover all key components, with quantitative validation for each design decision, and supplementary evaluations on additional datasets like BADJA.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, intuitive analogy with target detection, and consistent mathematical notations.
- **Value**: ⭐⭐⭐⭐ Provides a simple yet strong baseline for the TAP task, with ablation studies holding high reference value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)
- [\[ECCV 2024\] Shifted Autoencoders for Point Annotation Restoration in Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)
- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[CVPR 2025\] Multiple Object Tracking as ID Prediction](../../CVPR2025/object_detection/multiple_object_tracking_as_id_prediction.md)

</div>

<!-- RELATED:END -->
