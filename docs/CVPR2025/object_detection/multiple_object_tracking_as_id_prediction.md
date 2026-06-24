---
title: >-
  [Paper Note] Multiple Object Tracking as ID Prediction
description: >-
  [CVPR 2025][Object Detection][Multiple Object Tracking] This paper proposes MOTIP, which reformulates the object association problem in multiple object tracking (MOT) as an in-context ID prediction task. Given historical trajectories with ID embeddings, a standard Transformer decoder directly predicts the ID labels of the current detections without relying on heuristic matching algorithms, achieving a HOTA of 69.6 on DanceTrack and significantly outperforming the previous SOT…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Multiple Object Tracking"
  - "ID Prediction"
  - "In-Context Learning"
  - "DETR"
  - "End-to-End"
date: 2026-05-08
content_hash: 2109f38883154285
---

# Multiple Object Tracking as ID Prediction

**Conference**: CVPR 2025  
**arXiv**: [2403.16848](https://arxiv.org/abs/2403.16848)  
**Code**: [https://github.com/MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP)  
**Area**: Object Detection / Multiple Object Tracking  
**Keywords**: Multiple Object Tracking, ID Prediction, In-Context Learning, DETR, End-to-End

## TL;DR

This paper proposes MOTIP, which reformulates the object association problem in multiple object tracking (MOT) as an in-context ID prediction task. Given historical trajectories with ID embeddings, a standard Transformer decoder directly predicts the ID labels of the current detections without relying on heuristic matching algorithms, achieving a HOTA of 69.6 on DanceTrack and significantly outperforming the previous SOTA CO-MOT (65.3).

## Background & Motivation

**Background**: The mainstream paradigms of Multiple Object Tracking (MOT) are divided into two categories: (1) Tracking-by-Detection (e.g., ByteTrack, OC-SORT) which relies on Kalman filters and manual matching rules; (2) Tracking-by-Propagation (e.g., MOTR, CO-MOT) which propagates track queries across frames, though conflicts exist in the joint decoding of detection and tracking queries.

**Limitations of Prior Work**: (1) Heuristic matching algorithms (Hungarian matching, IoU matching) rely on hand-designed priors, which fail under non-linear motion and highly similar appearances, requiring heavy manual analysis and hyperparameter tuning for any improvement; (2) The detection-tracking conflict of track query methods limits joint decoding performance; (3) Although ReID-based methods are trained with classification supervision, they still require cosine similarity and matching rules during inference, causing training-inference inconsistency.

**Key Challenge**: Traditional classification cannot handle "unseen trajectory IDs" encountered during inference (out-of-distribution problem), making it impossible to directly model association as a classification task.

**Goal**: Design an end-to-end association method that: (1) has consistent training and inference pipelines; (2) maintains decoupling between detection and association to avoid conflicts; (3) does not rely on any hand-designed matching rules.

**Key Insight**: The ID labels in MOT essentially represent "consistency" rather than "semantic categories"—a trajectory is considered correct as long as the same label is predicted across all frames, with no need to predict a specific, fixed number. Therefore, one can randomly assign ID embeddings as in-context prompts, transforming association into an in-distribution classification task.

**Core Idea**: Use a learnable ID dictionary to assign random ID embeddings to each trajectory as in-context prompts, and then transform the association of current detections into a K+1 class classification problem using a standard Transformer Decoder.

## Method

### Overall Architecture

Input video frames are processed by Deformable DETR to obtain object-level features and bounding boxes. An ID dictionary (containing K+1 learnable embeddings) is maintained. The object features of historical trajectories are concatenated with their corresponding ID embeddings to form tracklet tokens. The detection features of the current frame are concatenated with a special token $i^{spec}$ to serve as Queries, while the historical tracklets act as Keys/Values. These are fed into a standard Transformer Decoder, which outputs predictions for the K+1 ID labels through a linear classification head.

### Key Designs

1. **In-context ID Prediction Paradigm**:
    - **Function**: Solves the out-of-distribution generalization problem of traditional ID classification.
    - **Mechanism**: Randomly assigns an ID label $k_m \in \{1, ..., K\}$ and the corresponding ID embedding $i^{k_m}$ to each active trajectory $\mathcal{T}^m$. During prediction, the model only needs to predict the same label based on the historical ID information carried, rather than a globally fixed label.
    - **Design Motivation**: The exact value of an ID in MOT is unimportant; what matters is "consistency". Allowing random assignment ensures that any new trajectory encountered during inference is still within the $\{1, ..., K+1\}$ distribution.
    - In practice, K is set to 50, and IDs of terminated trajectories are recycled to handle long videos.

2. **Tracklet Token Construction (Concatenation of Feature and ID Embedding)**:
    - **Function**: Fuses tracking cues and identity prompts into a unified representation.
    - **Mechanism**: $\tau_t^{m, k_m} = \text{concat}(f_t^m, i^{k_m})$, where $f_t^m$ is the C-dimensional DETR output embedding, and $i^{k_m}$ is the C-dimensional learnable ID embedding, resulting in a 2C-dimensional concatenated vector.
    - **Design Motivation**: Concatenation instead of addition preserves ID and appearance information in independent channels, allowing the ID Decoder to utilize both cues separately.
    - Current frame detections use a special token $i^{spec}$ instead of an ID embedding to indicate "unknown identity".

3. **Trajectory Augmentation**:
    - **Function**: Alleviates overfitting caused by using ground-truth trajectories during training.
    - Two types of augmentation: (a) Random Occlusion: randomly drops tokens in a trajectory with probability $\lambda_{occ}$ to simulate occlusion; (b) Random Switch: randomly swaps ID tokens of two trajectories within the same frame with probability $\lambda_{sw}$ to simulate incorrect ID assignments during inference.
    - **Design Motivation**: During inference, trajectories can be corrupted by occlusions or mismatching of similar targets. Introducing similar noise during training enhances robustness.
    - Both probabilities are set to 0.5.

### Loss & Training

$$\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{L1}\mathcal{L}_{L1} + \lambda_{giou}\mathcal{L}_{giou} + \lambda_{id}\mathcal{L}_{id}$$

The first three terms are standard DETR detection losses (Focal + L1 + GIoU), and the fourth term $\mathcal{L}_{id}$ is the standard cross-entropy ID classification loss. The weights are set to 2.0, 5.0, 2.0, and 1.0, respectively. The entire model is trained end-to-end, taking less than a day on 8 RTX 4090 GPUs for DanceTrack.

## Key Experimental Results

### Main Results: DanceTrack Test Set (Table 1)

| Method | Extra Data | HOTA↑ | DetA | AssA↑ | IDF1 |
|------|---------|-------|------|-------|------|
| ByteTrack | ✗ | 47.7 | 71.0 | 32.1 | 53.9 |
| OC-SORT | ✗ | 55.1 | 80.3 | 38.3 | 54.6 |
| CO-MOT | ✗ | 65.3 | 80.1 | 53.5 | 66.5 |
| **MOTIP** | ✗ | **69.6** | 80.4 | **60.4** | **74.7** |
| MOTRv2 | ✓ | 69.9 | 83.0 | 59.0 | 71.7 |
| **MOTIP** | ✓ | **72.0** | 81.8 | **63.5** | **76.8** |

### SportsMOT Test Set (Table 2)

| Method | HOTA↑ | AssA↑ |
|------|-------|-------|
| OC-SORT | 68.1 | 54.8 |
| MeMOTR | 68.8 | 57.8 |
| **MOTIP** | **72.6** | **63.2** |

### Ablation Study (Table 4, DanceTrack Validation Set)

| self-attn | Trajectory Aug. | HOTA | AssA |
|-----------|---------|------|------|
| ✗ | ✗ | 57.7 | 43.9 |
| ✓ | ✗ | 60.2 | 48.2 |
| ✓ | ✓ | **69.6** | **60.4** |

### Key Findings

- On the highly challenging DanceTrack dataset, it outperforms CO-MOT by **4.3 points** in HOTA and **6.9 points** in AssA (without extra data).
- MOTIP without extra data (69.6) is already close to MOTRv2 with extra data (69.9).
- Simple argmax inference without using the Hungarian algorithm achieves SOTA, showing that the model itself has learned globally optimal association.
- It also achieves SOTA on the BFT bird tracking dataset (70.5 HOTA), validating its cross-scene generalization capability.

## Highlights & Insights

1. **Redefining the Problem Space**: Reformulates MOT association from "distance metric + matching" to "in-context ID prediction + classification", elegantly solving the out-of-distribution problem during inference.
2. **Minimalist Architecture, Powerful Performance**: The entire ID prediction module is just a standard Transformer Decoder + a linear classification head, without any customized modules.
3. **Importance of Trajectory Augmentation**: The HOTA improvement from random occlusion and random switch (60.2 -> 69.6) shows that the training-inference gap is a key bottleneck in end-to-end MOT.
4. **Implicit Matching from Self-Attention**: Self-attention among detections allows the model to exchange identity information and avoid duplicate assignments, making the Hungarian algorithm unnecessary.

## Limitations & Future Work

1. The inference strategy is relatively simple (argmax + threshold); more sophisticated ID assignment strategies might yield further improvements.
2. Only object-level features are used as tracking cues, without exploiting position/motion information (although results are already outstanding).
3. K=50 limits the maximum number of targets in a single frame; ultra-dense scenarios may require adjustment.
4. Performance heavily relies on detection quality, as the DETR detector's performance directly bounds tracking quality.

## Related Work & Insights

- **MOTR / MeMOTR**: Track query propagation paradigm; MOTIP's decoupled design avoids detection-tracking conflicts.
- **CO-MOT**: Divides detection queries into track and detect groups to reduce conflict; MOTIP decouples them more thoroughly.
- **In-Context Learning**: MOTIP borrows the concept of in-context learning from NLP—predicting the current label given historical "prompts".
- **ReID Methods (FairMOT, etc.)**: Addresses the inconsistency where training uses classification and inference uses cosine similarity; MOTIP resolves this completely.

## Rating

- ⭐ **Novelty**: 9/10 — Reformulates MOT association as in-context ID prediction, which is fresh and targets the core issue.
- ⭐ **Experimental Thoroughness**: 9/10 — Comprehensively achieves SOTA across DanceTrack, SportsMOT, and BFT, with thorough ablation studies.
- ⭐ **Value**: 8/10 — Simple and reproducible architecture, trained within a day on 8x 4090 GPUs.
- ⭐ **Overall**: 9/10 — A pioneering paradigm for MOT association; a "surprisingly simple" design yielding dominant performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](../../ECCV2024/object_detection/walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[AAAI 2026\] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](../../AAAI2026/object_detection/when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](../../ICCV2025/object_detection/automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)

</div>

<!-- RELATED:END -->
