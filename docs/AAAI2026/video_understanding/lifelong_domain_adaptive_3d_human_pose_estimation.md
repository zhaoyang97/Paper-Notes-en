---
title: >-
  [Paper Note] Lifelong Domain Adaptive 3D Human Pose Estimation
description: >-
  [AAAI2026][Video Understanding][3D human pose estimation] This paper introduces a new task of lifelong domain adaptive 3D HPE, and proposes a GAN framework incorporating pose-aware, temporal-aware, and domain-aware encodings. A diffusion sampler is employed to generate domain-aware priors to mitigate catastrophic forgetting, achieving significant improvements over existing methods across multiple cross-scene/cross-dataset adaptation tasks.
tags:
  - "AAAI2026"
  - "Video Understanding"
  - "3D human pose estimation"
  - "lifelong domain adaptation"
  - "catastrophic forgetting"
  - "GAN"
  - "diffusion model"
date: 2026-05-08
content_hash: 466d2a7c3aa1fd3a
---

# Lifelong Domain Adaptive 3D Human Pose Estimation

**Conference**: AAAI2026
**arXiv**: [2512.23860](https://arxiv.org/abs/2512.23860)  
**Code**: [davidpengucf/lifelongpose](https://github.com/davidpengucf/lifelongpose)  
**Area**: Video Understanding
**Keywords**: 3D human pose estimation, lifelong domain adaptation, catastrophic forgetting, GAN, diffusion model

## TL;DR
This paper introduces a new task of lifelong domain adaptive 3D HPE, and proposes a GAN framework incorporating pose-aware, temporal-aware, and domain-aware encodings. A diffusion sampler is employed to generate domain-aware priors to mitigate catastrophic forgetting, achieving significant improvements over existing methods across multiple cross-scene/cross-dataset adaptation tasks.

## Background & Motivation
The 2D-to-3D lifting paradigm for 3D Human Pose Estimation (3D HPE) relies on 3D annotations collected in controlled environments, and suffers from domain shift when generalizing to in-the-wild scenarios. Limitations of existing DA methods:
- **General DA**: requires simultaneous access to source and target domain data
- **Source-free DA**: assumes a static target domain distribution and permits joint training on all target data
- Both paradigms ignore the **non-stationary nature of target pose distributions** in practice (e.g., shifts from pedestrian intent prediction in autonomous driving to in-vehicle safety monitoring)

Core motivation: This paper proposes **lifelong domain adaptive 3D HPE**—after source-domain pretraining, the model is sequentially adapted to multiple target domains, with access only to the current target domain data at each step, without revisiting the source domain or any previous target domains. The framework must simultaneously address current-domain adaptation and historical-domain knowledge retention.

## Method

### Overall Architecture
The framework comprises three core components: 3D pose generators, a 2D pose discriminator, and a 2D-to-3D lifting pose estimator, organized in a GAN structure to reduce domain shift.

### 3D Pose Generator
Given the 3D pose estimated in the current domain, three cascaded generators $G = G_{BA} \circ G_{BL} \circ G_{RT}$ (bone angle / bone length / rotation-translation) produce augmented 3D poses with three types of encoding:
1. **Pose-aware encoding**: In addition to joint coordinates and bone vectors, **6 body part segments** (left/right arms, left/right legs, torso, and extended torso) are introduced to capture relationships between non-adjacent joints.
2. **Temporal-aware encoding**: Multi-frame consecutive 3D poses are passed through a temporal weighted convolutional network to produce a weighted single-frame pose.
3. **Domain-aware encoding**: A **2D pose diffusion sampler** trained with DDIM samples from prior-domain 2D poses (using only $T/10$ steps), generating domain-aware priors to replace random noise.

### Optimization
- $\mathcal{L}_{3D}$: MSE + feedback loss, constraining the similarity between augmented and predicted 3D poses
- $\mathcal{L}_{2D}$: MSE + normalized L1, preserving both scale and alignment direction
- $\mathcal{L}_{dis}$: Wasserstein GAN with gradient penalty, discriminating between original and augmented 2D poses
- **EMA**: $\mathcal{P}_{j+1} = \eta \mathcal{P}_j + (1-\eta)\hat{\mathcal{P}}_j$ ($\eta=0.99$), smoothly updating the pose estimator to alleviate forgetting

## Key Experimental Results

### Cross-Scene Adaptation H3.6M: S1→S5→S6→S7→S8 (MPJPE/PA-MPJPE)

| Method | S5 | S6 | S7 | S8 | Avg |
|---|---|---|---|---|---|
| PoseDA-LL | 51.5/44.9 | 51.9/44.5 | 46.2/39.5 | 40.9/28.6 | 47.6/39.4 |
| **Ours** | **48.7/42.5** | **48.6/40.8** | **42.3/36.9** | **40.0/27.4** | **44.9/36.9** |

### Cross-Dataset Adaptation H3.6M→3DHP (Average over 6 test sets)

| Method | Avg MPJPE/PA-MPJPE |
|---|---|
| PoseDA-LL | 80.7/54.5 |
| **Ours** | **75.3/50.7** |

### Multi-Dataset Adaptation (H3.6M→3DHP→3DPW)

| Method | 3DHP | 3DPW | Avg |
|---|---|---|---|
| PoseDA-LL | 88.9/62.1 | 87.6/49.4 | 88.3/55.8 |
| **Ours** | **75.3/51.1** | **81.7/45.6** | **78.5/48.4** |

Ablation studies confirm that domain-aware embedding (DE) is the most critical component (its removal causes an 8.2 mm MPJPE degradation on 3DHP), and EMA also plays a substantial role in mitigating forgetting (5.9 mm degradation upon removal).

## Highlights & Insights
- **First to introduce lifelong DA into 3D HPE**, formalizing the sequential adaptation problem under non-stationary target domains
- **Diffusion sampler as domain memory**: DDIM is used to retain prior-domain pose distributions, avoiding GAN mode collapse while enabling efficient prior generation with only $T/10$ sampling steps
- **Part-aware encoding**: 6 body part segments enrich the comprehensiveness of pose representations
- Consistently outperforms all 5 baselines across all 3 experimental settings

## Limitations & Future Work
- The diffusion sampler requires retraining/updating for each new domain, with potentially growing overhead as the number of domains increases
- Experiments are limited to a 16-keypoint body model and an FC-based estimator (VideoPose3D); generalization to Transformer-based architectures remains unverified
- The adaptation order across target domains is fixed; the effect of ordering on final performance is not discussed
- Online/streaming settings are not explored; the current framework operates in an offline batch adaptation mode

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to define lifelong DA for 3D HPE; the use of a diffusion sampler as domain memory is a novel idea
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three adaptation settings, five baselines, and detailed ablations, though validation is limited to pose datasets
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear, method description is thorough, and figures are of high quality
- **Value**: ⭐⭐⭐⭐ — Practically meaningful for continual adaptation in non-stationary environments; the framework demonstrates good extensibility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation](../../CVPR2026/video_understanding/beyond_static_frames_temporal_aggregate-and-restore_vision_transformer_for_human.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](../../CVPR2026/video_understanding/egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)
- [\[AAAI 2026\] StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis](stegavar_privacy-preserving_video_action_recognition_via_steganographic_domain_a.md)
- [\[ECCV 2024\] Benchmarks and Challenges in Pose Estimation for Egocentric Hand Interactions with Objects](../../ECCV2024/video_understanding/benchmarks_and_challenges_in_pose_estimation_for_egocentric_hand_interactions_wi.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](../../ICCV2025/video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)

</div>

<!-- RELATED:END -->
