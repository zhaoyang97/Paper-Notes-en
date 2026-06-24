---
title: >-
  [Paper Note] PAP: A Prediction-as-Perception Framework for 3D Object Detection
description: >-
  [CVPR 2025][Autonomous Driving][3D Object Detection] Inspired by the brain's "predictive perception," PAP uses the trajectory prediction results of the previous frame as query inputs for the perception module of the current frame to replace some random queries. This achieves a 10% improvement in AMOTA (0.359 to 0.395), a 15% increase in inference speed (14 to 16 FPS), and a 14% reduction in training time on UniAD.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Object Detection"
  - "Prediction-as-Perception Framework"
  - "Multi-Object Tracking"
  - "Bionic Design"
  - "Query Cycle"
  - "End-to-End Perception"
date: 2026-05-08
content_hash: 01b0c9dafb3be5ad
---

# PAP: A Prediction-as-Perception Framework for 3D Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2603.12599](https://arxiv.org/abs/2603.12599)  
**Code**: None  
**Area**: Autonomous Driving / 3D Perception  
**Keywords**: 3D Object Detection, Prediction-as-Perception Framework, Multi-Object Tracking, Bionic Design, Query Cycle, End-to-End Perception

## TL;DR
Inspired by the brain's "predictive perception," PAP uses the trajectory prediction results of the previous frame as query inputs for the perception module of the current frame to replace some random queries. This achieves a 10% improvement in AMOTA (0.359 to 0.395), a 15% increase in inference speed (14 to 16 FPS), and a 14% reduction in training time on UniAD.
The key insight of this framework is that predicted future locations are closer to the ground truth targets than random initializations, thereby reducing inefficient search.

## Background & Motivation
**Background**: Attention-based 3D perception models (DETR3D, StreamPETR, Sparse4D) use queries for object detection and tracking, while end-to-end models like UniAD unify perception, prediction, and planning.

**Limitations of Prior Work**: Queries for each frame are mostly randomly generated, which requires extensive computation to locate objects from scratch and loses temporal continuity clues between frames. Perception and prediction typically form a unidirectional flow (perception -> prediction) without a feedback loop.

**Key Challenge**: When tracking fast-moving targets, the human brain utilizes a bidirectional loop (predict next position -> focus gaze -> verify -> update prediction -> ...), whereas current models lack such a feedback loop from prediction to perception.

**Goal**: How to feedback the trajectory prediction results from historical frames into the perception module of the current frame, forming a bidirectional perception-prediction loop while simultaneously improving efficiency and accuracy?

**Key Insight**: Bionics—simulating the mechanism of "predicting the next position -> focusing -> verifying" when the human brain tracks flying insects or birds.

**Core Idea**: Generating queries from the prediction output of the previous frame and injecting them into the current frame's perception module, forming a prediction-perception loop.

## Method

### Overall Architecture
PAP = Perception Module + Prediction Module + Query Cycle Lane. Input current frame image + predicted queries from the previous frame -> perception module outputs detection results and queries -> prediction module outputs future position queries -> stored in query bank -> retrieved and injected into the perception module in the next frame, forming an iterative loop. When there is no historical prediction for the first frame, random queries are used.

### Key Designs

1. **Prediction Query Injection into Perception Module**

    - **Function**: Replacing some random queries in the current frame with the query embeddings of the predicted future positions from the previous frame.
    - **Mechanism**: $q_i^T \in (q_{random}^T \cup q_{predict}^{T-1})$; $q_{predict}^{T-1} = \phi^{embd}(c_{predict}^{T-1})$ represents the predicted position of the previous frame transformed by the embedding layer.
    - **Design Motivation**: The spatial positions of the predicted queries are already close to the ground truth targets, making them easier to match targets than random queries, thereby reducing computational waste and retaining temporal continuity clues.

2. **Prediction Module Reusability**

    - **Function**: Directly utilizing the prediction components of existing models to output coordinates of target positions in future frames.
    - **Mechanism**: $c_{predict}^T = \text{PRED}(\text{PECP}(c_i^T))$, where results are embedded and stored in the query bank with temporal indices.
    - **Design Motivation**: No need to design a new prediction module—MotionFormer can be directly reused in UniAD.

3. **Integration with UniAD**

    - **Function**: Embedding the output queries of MotionFormer into the same dimension as Track Queries, and injecting them into TrackFormer for the next frame.
    - **Mechanism**: The interactions between UniAD modules are inherently query-based; PAP only adds a feedback channel from MotionFormer to TrackFormer.
    - **Design Motivation**: Minimizing modifications—the planning module and loss remain untouched.

### Loss & Training
Consistent with the original UniAD, the learning of predicted queries is naturally achieved through the joint loss of the perception and prediction modules. There are no additional hyperparameters or loss terms.
Training configuration: 4×A100 GPUs, total training time of 78h (saving 14% compared to the original UniAD's 91h).
The storage overhead of the query bank during inference is minimal, requiring only the retention of the prediction embeddings from the previous frame.
When there is no historical prediction in the first frame, all-random queries are used. Starting from the second frame, a mixture of predicted and random queries is utilized.
The proportion of predicted queries is not discussed in detail, and future work is suggested to explore the optimal ratio.

## Key Experimental Results

### Main Results (nuScenes validation)

| Model | AMOTA↑ | AMOTP↓ | Recall↑ | IDS↓ | Training Time | FPS |
|------|--------|--------|---------|------|---------|-----|
| UniAD | 0.359 | 1.32 | 0.467 | 906 | 91h | 14 |
| UniAD+PAP | **0.395** | **1.22** | **0.493** | **826** | **78h** | **16** |

### Ablation Study (By Category)

| Category | AMOTA | AMOTP | Recall | IDS |
|------|-------|-------|--------|-----|
| Car | 0.613 | 0.744 | 0.667 | 405 |
| Pedestrian | 0.411 | 1.192 | 0.487 | 342 |
| Bus | 0.465 | 1.225 | 0.535 | 8 |
| Motorcycle | 0.438 | 1.253 | 0.500 | 24 |
| Truck | 0.411 | 1.267 | 0.611 | 28 |

### Key Findings
- AMOTA is improved by 10% (0.359 -> 0.395), AMOTP is improved by 0.1m (1.32 -> 1.22), and ID switches decrease from 906 to 826.
- Training time is reduced from 91h to 78h (-14%), and inference speed is increased from 14 FPS to 16 FPS (+15%).
- The efficiency improvement stems from predicted queries being closer to target positions, which reduces ineffective search and updates associated with random queries.
- The Car category achieves the highest AMOTA (0.613), because vehicle motion is the most regular and its prediction is the most accurate, leading to the best feedback effect.

## Highlights & Insights
- **Minimalistic design philosophy**: It only adds a query feedback channel from prediction to perception, without requiring new modules, new loss functions, or new hyperparameters. The modification is minimal yet yields significant results.
- **Bionically-inspired system design**: Predictive perception serves as a classic theory in cognitive science. This work is the first to formalize it into a 3D object detection framework.
- **Dual improvement in efficiency and accuracy**: Predicted queries not only improve accuracy (via better initialization) but also reduce computational load (by decreasing ineffective query updates), achieving a rare win-win scenario.
- **Framework universality**: PAP can be applied to any query-based perception model, such as DETR3D, StreamPETR, and Sparse4D.

## Limitations & Future Work
- Only validated on UniAD without being tested on stronger perception/prediction models (such as Sparse4D v3), whereas UniAD itself is not SOTA in perception and prediction.
- Lack of ablation studies—failing to analyze key design choices such as the proportion of predicted queries and first-frame initialization strategies.
- Errors in the prediction module itself propagate to perception—inaccurate predictions may introduce misleading queries (though mitigated by mixing with random queries).
- Lack of explicit analysis of scenarios where predicted queries benefit the most (e.g., reappearing after occlusion, high-speed movement, etc.).

## Related Work & Insights
- **vs StreamPETR**: StreamPETR maintains temporal continuity through inter-frame query propagation, but it propagates perception queries rather than prediction queries, lacking "future prediction" information.
- **vs HOP**: HOP enhances multi-view 3D detectors with historical target predictions, but it only performs temporal enhancement during training and does not change the inference flow; PAP modifies the composition of queries during inference.
- **vs Sparse4D**: Sparse4D uses spatiotemporal fused queries, but they are still propagated from the perception side, whereas PAP adds information injection from the prediction side.
- **vs Original UniAD**: The interaction between modules in UniAD is unidirectional (Track -> Map -> Motion -> Plan). PAP adds a feedback loop from Motion to Track.
- **Cognitive science perspective**: Predictive coding is one of the mainstream theories in neuroscience. This work is the first to structure it into a 3D perception framework. Future work can explore deeper prediction-perception interactions (such as multi-scale prediction).
- **Extensible directions**: Theoretically, the PAP framework can also be applied to other query-based tasks such as BEV segmentation and lane detection.

## Rating
- Novelty: ⭐⭐⭐⭐ The bionic prediction-perception loop concept is novel, but the technical implementation is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐ Only one model on one dataset, lacking ablation studies and validation on multiple models.
- Writing Quality: ⭐⭐⭐ The paper is structured reasonably but contain some redundant expressions, with minor formatting issues.
- Value: ⭐⭐⭐⭐ Proposes a valuable framework idea; the cyclic design of "prediction-as-perception" is inspiring to the field.
- Overall: ⭐⭐⭐☆ The idea is simple and elegant, but the experimental validation is insufficient. Looking forward to validation on more robust models in the future.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection](sparsealign_a_fully_sparse_framework_for_cooperative_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[CVPR 2025\] EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)

</div>

<!-- RELATED:END -->
