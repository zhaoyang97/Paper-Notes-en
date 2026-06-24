---
title: >-
  [Paper Note] TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception
description: >-
  [CVPR 2025][Cooperative Perception] Proposes the TraF-Align framework, which learns the spatiotemporal flow path of features by predicting object motion trajectories at the feature level. It generates temporally ordered sampling points along the trajectory to guide the current-timestamp query to relevant historical features, achieving precise feature alignment in asynchronous multi-agent perception. It achieves state-of-the-art (SOTA) performance on two real-world datasets: V…
tags:
  - "CVPR 2025"
  - "Cooperative Perception"
  - "Asynchronous Fusion"
  - "Feature Alignment"
  - "Trajectory Prediction"
  - "V2V Communication"
date: 2026-05-08
content_hash: 5c48827950246ea2
---

# TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception

**Conference**: CVPR 2025  
**arXiv**: [2503.19391](https://arxiv.org/abs/2503.19391)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Cooperative Perception, Asynchronous Fusion, Feature Alignment, Trajectory Prediction, V2V Communication

## TL;DR

Proposes the TraF-Align framework, which learns the spatiotemporal flow path of features by predicting object motion trajectories at the feature level. It generates temporally ordered sampling points along the trajectory to guide the current-timestamp query to relevant historical features, achieving precise feature alignment in asynchronous multi-agent perception. It achieves state-of-the-art (SOTA) performance on two real-world datasets: V2V4Real and DAIR-V2X-Seq.

## Background & Motivation

**Background**: Cooperative perception enhances the perception range of ego vehicles by sharing sensor data among multiple vehicles. However, latency is inevitable in real-world communication, causing received features to be out of sync with the ego vehicle's current observation in the time domain.

**Limitations of Prior Work**: Latency causes two types of mismatch: (1) **Spatial mismatch**: objects have moved during the latency period, making the object locations in the received features inconsistent with reality; (2) **Semantic mismatch**: object appearance, occlusion status, etc., change over time, and direct fusion introduces inconsistent information. Existing methods mostly use simple motion compensation (e.g., warping), which cannot handle semantic-level inconsistency.

**Key Challenge**: Correcting spatial displacement while ensuring semantic consistency simultaneously is difficult. Spatial compensation is a geometric problem while semantic alignment is a high-level feature matching problem, making their joint optimization challenging.

**Goal**: Design a unified framework to address both spatial and semantic mismatches in asynchronous fusion simultaneously.

**Key Insight**: If object trajectories at the feature level (i.e., the motion paths of features over time) can be predicted, information can be precisely aggregated from historical features along the trajectory to the current timestamp.

**Core Idea**: Predict object trajectories in the feature space from historical observations, and use sampling points along the trajectory to guide attention to "trace back" from the current query to the corresponding feature locations at each timestamp.

## Method

### Overall Architecture

The three-stage pipeline of TraF-Align: (1) Each agent extracts BEV features and shares them along with timestamps; (2) Future/historical frames are utilized to predict the target motion trajectory for each query position; (3) Temporally ordered sampling points are generated along the predicted trajectory to aggregate information from multi-frame historical features into the current-timestamp features using deformable attention.

### Key Designs

1. **Feature-level Trajectory Prediction**:

    - **Function**: Predict the motion trajectory of the target at each spatial location.
    - **Mechanism**: Given the current-timestamp query position and multi-frame historical BEV features, a lightweight MLP is used to predict the target's motion trajectory from the past to the present (a sequence of 2D offsets). The trajectory starts at the earliest historical frame and terminates at the current timestamp. These trajectories provide precise spatial reference lines for subsequent attention sampling.
    - **Design Motivation**: Simple warping assumes uniform linear motion and fails to handle complex motions such as acceleration and turning. Trajectory prediction provides more precise spatial correspondences.

2. **Trajectory-guided Deformable Attention**:

    - **Function**: Aggregate information from historical features along the predicted trajectories.
    - **Mechanism**: Temporal points are uniformly sampled along each predicted trajectory, indicating the target's feature positions at historical timestamps. Centered on the current-timestamp query, historical features are extracted at these sampling points using a deformable attention mechanism, and multi-frame information is adaptively fused through attention weights. The temporal order of the sampling points guarantees spatiotemporal consistency.
    - **Design Motivation**: Deformable attention allows sampling at precise locations rather than global scanning, which is computationally efficient and highly accurate. The temporal sampling points provided by the trajectory ensure that attention focuses on the "right places."

3. **Cross-frame Semantic Interaction**:

    - **Function**: Promote semantic consistency across multi-frame features.
    - **Mechanism**: In trajectory-guided attention, features from different timestamps implicitly interact through the shared query. This helps the model learn semantic evolution over time (e.g., an object changing from fully visible to partially occluded). This cross-frame interaction naturally handles the semantic mismatch problem.
    - **Design Motivation**: Pure spatial alignment cannot resolve semantic changes. Cross-frame feature interaction allows the model to understand "how the same object looks different at different times."

### Loss & Training

End-to-end training, including 3D detection loss and trajectory prediction auxiliary loss.

## Key Experimental Results

### Main Results

| Method | V2V4Real AP@0.5↑ | DAIR-V2X AP@0.5↑ |
|------|-------------------|-------------------|
| SyncNet | Baseline | Baseline |
| CoBEVFlow | +Improvement | +Improvement |
| **TraF-Align** | **SOTA** | **SOTA** |

*Achieves state-of-the-art performance on both real-world asynchronous cooperative perception datasets.*

### Key Findings
- The trajectory prediction module significantly outperforms the simple linear motion assumption.
- The advantage is more pronounced in scenarios with large latency (>300ms).
- Temporally ordered sampling yields better semantic consistency than random sampling.

## Highlights & Insights
- **Trajectory as Spatial Navigation for Attention**: Encodes prior knowledge of target motion as reference lines for attention sampling, balancing physical plausibility and learning flexibility.
- **Unified Handing of Spatial and Semantic Mismatches**: Addresses two mismatches of different natures simultaneously using a single attention mechanism.
- **Validated on Real-world Datasets**: Unlike most works that are tested only in simulators, this approach is validated on real-world data.

## Limitations & Future Work
- Trajectory prediction can be inaccurate for occluded or newly appearing objects.
- Currently assumes that the delay of all agents is known, without handling the uncertainty of delay estimation.
- Validated only on 3D detection tasks; extensions to tasks like cooperative segmentation are needed.

## Related Work & Insights
- **vs CoBEVFlow**: CoBEVFlow utilizes optical flow to compensate for spatial mismatch, whereas TraF-Align uses trajectory prediction; the latter is more robust to non-rigid motions.
- **vs SyncNet**: SyncNet uses simple warping, whereas TraF-Align utilizes an attention mechanism, which is more flexible.
- The concept of trajectory-guided attention can be transferred to cross-frame feature alignment in video understanding.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The trajectory-guided attention design is highly targeted.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two real-world datasets with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition.
- **Value**: ⭐⭐⭐⭐ Provides significant advancement to asynchronous cooperative perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception](../../ECCV2024/others/align_before_collaborate_mitigating_feature_misalignment_for_robust_multi-agent_.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[CVPR 2025\] EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching](edm_equirectangular_projection-oriented_dense_kernelized_feature_matching.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)

</div>

<!-- RELATED:END -->
