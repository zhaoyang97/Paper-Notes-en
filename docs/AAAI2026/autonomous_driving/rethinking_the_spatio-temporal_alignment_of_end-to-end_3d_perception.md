---
title: >-
  [Paper Note] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception
description: >-
  [AAAI2026][Autonomous Driving][Spatio-temporal alignment] This paper proposes HAT (multiple Hypotheses spAtio-Temporal alignment), a plug-and-play spatio-temporal alignment module that generates alignment hypotheses via multiple explicit motion models and adaptively decodes the optimal alignment using motion cues latent in queries. HAT consistently improves multiple 3D temporal detectors and trackers on nuScenes, and reduces collision rates by 32–48% in end-to-end autonomous driving.
tags:
  - AAAI2026
  - Autonomous Driving
  - Spatio-temporal alignment
  - end-to-end 3D perception
  - multiple hypothesis motion models
  - multi-object tracking
date: 2026-05-08
content_hash: 907a0ce383a219fe
---

# Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception

**Conference**: AAAI2026  
**arXiv**: [2512.23635](https://arxiv.org/abs/2512.23635)  
**Code**: [lixiaoyu2000/HAT](https://github.com/lixiaoyu2000/HAT)  
**Authors**: Xiaoyu Li, Peidong Li, Xian Wu et al.  
**Area**: Autonomous Driving  
**Keywords**: Spatio-temporal alignment, end-to-end 3D perception, multiple hypothesis motion models, autonomous driving, multi-object tracking  

## TL;DR

This paper proposes HAT (multiple Hypotheses spAtio-Temporal alignment), a plug-and-play spatio-temporal alignment module that generates alignment hypotheses via multiple explicit motion models and adaptively decodes the optimal alignment using motion cues latent in queries. HAT consistently improves multiple 3D temporal detectors and trackers on nuScenes, and reduces collision rates by 32–48% in end-to-end autonomous driving.

## Background & Motivation

Spatio-temporal alignment (STA) is a core component of temporal modeling in end-to-end (E2E) autonomous driving perception systems. The STA module propagates instance features and anchors from historical frames to the current frame, providing structured and semantic prior information for detection and tracking. Existing query-based methods (e.g., StreamPETR, Sparse4D) typically employ a single explicit physical model (e.g., the constant velocity model) for motion compensation, relying on query propagation for feature alignment in latent space.

However, this simplified motion modeling has fundamental shortcomings: the motion patterns of different object categories vary substantially (pedestrians vs. vehicles, straight-line motion vs. turning), and the motion state of a single object also changes over time. A single hypothesis cannot capture this diversity. Although modular approaches (e.g., Kalman filter-based trackers) consider multiple motion models, they require manual parameter tuning and tend to overfit specific motion patterns.

At a deeper level, queries propagated in current E2E methods contain rich but underutilized motion cues, which could be leveraged to discriminate among and construct structured priors best suited to each object. How to integrate the advantages of multiple motion models within an E2E framework while avoiding the fragility of traditional methods constitutes the central research problem of this paper.

## Core Problem

How can the STA module in E2E perception transcend the limitations of a single motion hypothesis and adaptively decode the optimal alignment scheme for each object from multiple motion models, without requiring additional direct supervision signals?

## Method

### Overall Architecture

HAT consists of two stages: a **Temporal Alignment Module** that generates multiple motion-aware hypotheses, and a **Spatial Alignment Module** that decodes the optimal alignment using motion cues embedded in queries.

Given the set of 3D anchors $B_{t-1} = \{b_{t-1}^i\}$ and queries $Q_{t-1} = \{q_{t-1}^i\}$ from the historical frame $t-1$, STA propagates them to the current frame $t$:

$$B_{t,t-1}, Q_{t,t-1} \leftarrow \text{STA}(B_{t-1}, Q_{t-1}, \Delta t, E_{t-1}^t)$$

where $E_{t-1}^t = [R_{t-1}^t | T_{t-1}^t]$ is the ego pose transformation matrix.

### Multi-Hypothesis Anchor Generator

A Motion Model Library (MML) is defined comprising five classical motion models:
- **STATIC**: stationary model
- **CV (Constant Velocity)**: constant velocity model
- **CA (Constant Acceleration)**: constant acceleration model
- **CTRV (Constant Turn Rate and Velocity)**: constant turn rate and velocity model
- **CTRA (Constant Turn Rate and Acceleration)**: constant turn rate and acceleration model

Each model extrapolates an anchor hypothesis based on $\Delta t$ and the historical anchor $B_{t-1}$:

$$\hat{s}_{t,t-1} = s_{t-1} + \int_{(t-1)\Delta t}^{t\Delta t} \dot{s}(\tau) d\tau = s_{t-1} + \Delta s$$

Unobservable states such as acceleration and yaw rate are decoded from the instance feature $q_{t-1}$ via an MLP. After ego pose transformation, the multi-hypothesis anchors $\tilde{B}_{t,t-1} \in \mathbb{R}^{K \times M \times 10}$ are obtained.

### Multi-Hypothesis Feature Generator

A state-decoupled encoder maps the anchor hypotheses into motion embeddings, which are concatenated with the propagated queries to produce motion-aware feature hypotheses:

$$\tilde{Q}_{t,t-1} = \text{Cat}(\tilde{Q}'_{t,t-1}, Q_{t-1}) \in \mathbb{R}^{K \times M \times 2C}$$

### Adaptive Multi-Hypothesis Decoder

**Feature decoding**: Dynamic weights $W_c$ and $W_f$ are generated from the propagated queries, and multi-hypothesis features are fused via an MLP-like architecture:

$$\bar{Q}_{t,t-1} = \sigma(\text{LN}(W_f \otimes \sigma(\text{LN}(\tilde{Q}_{t,t-1} \otimes W_c))))$$

**Anchor decoding**: Inspired by the posterior estimation of IMM filters, the optimal anchor is decoded via softmax-weighted summation:

$$\bar{B}_{t,t-1} = \text{Softmax}(L_a(W_f)) \otimes \tilde{B}_{t,t-1}$$

**Feature–anchor hybridization**: The anchor is further refined by a motion refinement MLP $\Phi_r$:

$$B_{t,t-1} = \bar{B}_{t,t-1} + \Phi_r(Q_{t,t-1})$$

### Stability Guarantee

The aligned position $\bar{X}_{t,t-1}$ is constrained within the range compensated by all motion models. Since the models are physics-based, this constraint is inherently stable and requires no additional supervision.

## Key Experimental Results

### E2E Autonomous Driving (nuScenes val, SparseDrive baseline)

| Method | mAP↑ | AMOTA↑ | L2(m)↓ | CR(%)↓ |
|------|------|--------|--------|--------|
| SparseDrive | 41.2 | 36.9 | 0.63 | 0.123 |
| SparseDrive-HAT | **42.5**(+1.3) | **40.0**(+3.1) | **0.60** | **0.084**(-32%) |
| DiffusionDrive | 41.2 | 37.5 | 0.57 | 0.080 |
| DiffusionDrive-HAT | **42.7**(+1.5) | **40.2**(+2.7) | 0.58 | **0.042**(-48%) |

### 3D Detection (nuScenes val)

| Detector | NDS↑ | mAP↑ | mAVE↓ |
|--------|------|------|-------|
| StreamPETR | 57.1 | 48.2 | 0.26 |
| +HAT | **57.8**(+0.7) | **48.7**(+0.5) | **0.24** |
| Sparse4D | 56.4 | 46.5 | 0.22 |
| +HAT | **57.3**(+0.9) | **47.0**(+0.5) | **0.21** |
| SimPB | 58.6 | 47.9 | 0.22 |
| +HAT | **59.0**(+0.4) | **48.8**(+0.9) | **0.21** |

### 3D MOT (nuScenes test)

| Tracker | AMOTA↑ | MOTA↑ | IDS↓ |
|--------|--------|-------|------|
| ADA-Track | 45.6 | 40.6 | 834 |
| ADA-Track-HAT | **46.0**(+0.4) | **41.6**(+1.0) | 850 |

### Robustness Evaluation (nuScenes-C Snow)

| Method | NDS↑ | AMOTA↑ | CR(%)↓ |
|------|------|--------|--------|
| SparseDrive | 34.1 | 13.1 | 0.156 |
| SparseDrive-HAT | **39.1**(+5.0) | **18.0**(+4.9) | **0.122**(-22%) |

### Ablation Study on MML (Sparse4D baseline)

| CV | STATIC | CA | CTRA | CTRV | NDS | mAP |
|:--:|:------:|:--:|:----:|:----:|-----|-----|
| ✓ | | | | | 56.5 | 45.7 |
| ✓ | ✓ | ✓ | | | 56.6 | 46.3 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **57.3** | **47.0** |
| | | | | | 55.5 | 45.7 |

## Highlights & Insights

- **Plug-and-play universal module**: HAT integrates seamlessly into diverse query-based detectors (StreamPETR/Sparse4D/SimPB), trackers (ADA-Track), and E2E methods (SparseDrive/DiffusionDrive), consistently improving performance.
- **Explicit–implicit hybrid alignment**: The approach elegantly combines the interpretability of physics-based motion models with the adaptability of neural networks, learning optimal alignment without direct supervision.
- **Significant collision rate reduction**: Collision rates are reduced by 32% on SparseDrive and 48% on DiffusionDrive, directly improving autonomous driving safety.
- **Robustness under adverse weather**: Under nuScenes-C Snow conditions, the enhanced motion modeling of HAT yields a 5.0% NDS gain, compensating for perceptual degradation caused by corrupted semantic features.
- **Low additional overhead**: Only 7 ms of additional latency is introduced (baseline: 111 ms), making the approach practically deployable.

## Limitations & Future Work

- **Fixed motion model library**: The five models in the MML are predefined; data-driven motion model learning and dynamic library expansion are not explored.
- **Camera-only validation**: The effectiveness of HAT under LiDAR or multi-modal fusion settings has not been verified.
- **Unsupervised regression of acceleration and yaw rate**: Unobservable states are decoded from queries via MLPs, limiting accuracy; the authors constrain outputs to a narrow range of $\pm 0.1$.
- **Limited gains with purely structural anchor propagation**: Improvements on 3DMOTFormer are marginal, indicating that HAT relies on rich semantic and motion cues in queries.

## Related Work & Insights

- **MLN (StreamPETR)**: Performs implicit alignment using only semantic cues; HAT improves NDS by 0.7% and mAP by 0.5% on StreamPETR, reducing mAVE from 0.26 to 0.24.
- **LMM (STAR-Track)**: Uses a pretrained trajectory prediction network for supervised feature projection; HAT surpasses it by 0.3% NDS and 0.2% mAP without pretraining.
- **IMM filter**: The classical multi-model filter requires manually specified switching probabilities; HAT adaptively regresses weights via a network, eliminating manual tuning.
- **BEVFormer**: Employs BEV features for temporal modeling at high computational cost; HAT is more efficient due to its object-centric propagation.

The central insight of this paper is that motion modeling deserves equal importance to semantic modeling in E2E perception. Existing methods overly rely on semantic features for implicit alignment, overlooking the value of classical motion models. HAT's multi-hypothesis decoding mechanism is analogous to particle filtering — generating multiple candidates and selecting the best via weighted fusion. This paradigm is broadly applicable to other tasks requiring temporal reasoning, such as video understanding and trajectory prediction. The importance of motion priors becomes even more pronounced when semantic features degrade under adverse weather conditions.

## Rating

- Novelty: ⭐⭐⭐⭐ — The multi-hypothesis explicit–implicit hybrid alignment approach is original, though individual components (motion models, adaptive decoding) have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers detection, tracking, and E2E tasks with multiple baselines, comprehensive ablations, and robustness evaluations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; method description is rigorous with complete mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play module with open-source code; substantial improvements on safety-critical metrics (collision rate) demonstrate high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)

</div>

<!-- RELATED:END -->
