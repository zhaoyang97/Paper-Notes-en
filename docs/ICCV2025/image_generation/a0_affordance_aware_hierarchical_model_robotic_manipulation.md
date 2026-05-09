---
title: >-
  [Paper Note] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation
description: >-
  [ICCV 2025][Image Generation][Robotic Manipulation] This paper proposes A₀, an affordance-aware hierarchical diffusion model that decomposes manipulation tasks into high-level spatial affordance understanding (predicting contact points and trajectories) and low-level action execution. Pretrained on 1M contact point data and fine-tuned with minimal task-specific data, A₀ achieves cross-platform deployment across Franka/Kinova/Realman/Dobot, reaching a 45% success rate on complex trajectory tasks such as whiteboard wiping.
tags:
  - ICCV 2025
  - Image Generation
  - Robotic Manipulation
  - Spatial Affordance
  - Hierarchical Model
  - Diffusion Model
  - Cross-Platform Generalization
date: 2026-05-08
content_hash: deac16166caf5384
---

# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation

**Conference**: ICCV 2025
**arXiv**: [2504.12636](https://arxiv.org/abs/2504.12636)
**Code**: [https://a-embodied.github.io/A0/](https://a-embodied.github.io/A0/)
**Area**: Image Generation
**Keywords**: Robotic Manipulation, Spatial Affordance, Hierarchical Model, Diffusion Model, Cross-Platform Generalization

## TL;DR
This paper proposes A₀, an affordance-aware hierarchical diffusion model that decomposes manipulation tasks into high-level spatial affordance understanding (predicting contact points and trajectories) and low-level action execution. Pretrained on 1M contact point data and fine-tuned with minimal task-specific data, A₀ achieves cross-platform deployment across Franka/Kinova/Realman/Dobot, reaching a 45% success rate on complex trajectory tasks such as whiteboard wiping.

## Background & Motivation

**Background**: Robotic manipulation approaches fall into two categories: modular methods (leveraging visual foundation models) and end-to-end VLA methods (directly generating actions).

**Limitations of Prior Work**: Modular methods lack deep understanding of object spatial affordances; end-to-end methods generate actions without understanding spatial positions, leading to poor performance on complex manipulations (e.g., wiping whiteboards, stacking objects).

**Core Idea**: The paper proposes an Embodiment-Agnostic Affordance Representation — object-centric prediction of contact points and post-contact 2D waypoint trajectories — decoupling the method from specific robot platforms.

## Method

### Key Designs

1. **Embodiment-Agnostic Affordance Representation**: Unifies affordance information from robot data, hand-object interaction (HOI) data, and custom data into the $(I, L, C, T)$ format — image, language instruction, contact point, and trajectory waypoints.

2. **A₀ Diffusion Model**: Based on the DiT architecture, takes noisy waypoints and diffusion timesteps as input, and injects SigLiP visual features and Qwen2.5-7B text features via cross-attention. A Position Offset Attention module is introduced to extract inter-frame motion information.

3. **Two-Stage Training**:
   - Pretraining: Learns general object localization ability on 1M PixMo-One-Point data.
   - Fine-tuning: Learns dynamic manipulation on annotated trajectory data.

### Loss & Training
Pretraining: $\mathcal{L}_p = \text{MSE}(x_t^0, f_\theta(k, x_t^k, I_t, \ell))$; Fine-tuning: $\mathcal{L}_s = \text{MSE}(x_{t:t+T}^0, f_\theta(k, x_{t:t+T}^k, I_{t-1:t}, \ell))$

## Key Experimental Results

| Platform | A₀ Success Rate | Best Baseline | Notes |
|----------|----------------|---------------|-------|
| Franka | **62.50%** | 55.0% (OpenVLA) | Average over 8 tasks |
| Kinova | **53.75%** | 42.5% | Cross-platform generalization |
| Wipe Board | **45%** | ~20% | Trajectory-following task |

### Key Findings
- Pretraining contact point localization capability significantly improves post-fine-tuning manipulation performance.
- The 2D waypoint representation is inherently cross-platform; deployment to different robots requires only 2D→3D back-projection and grasp sampling.
- The advantage is most pronounced on trajectory-following tasks, as conventional methods lack modeling of post-contact trajectories.

### Pretraining Data Scale

| Data Source | Contact Points | Usage |
|-------------|---------------|-------|
| PixMo-One-Point | 1M | Object localization pretraining |
| HOI Data | 50K | Hand-object interaction |
| Robot Data | 20K | Manipulation tasks |

### Cross-Platform Deployment Results

| Platform | Tasks | Avg. Success Rate | Deployment |
|----------|-------|------------------|------------|
| Franka | 8 | 62.5% | Direct |
| Kinova | 6 | 53.8% | Direct |
| Realman | 4 | 48.5% | Adapted |
| Dobot | 3 | 45.0% | Adapted |

## Highlights & Insights
- The "embodiment-agnostic" design is highly practical: by predicting 2D points and trajectories on objects rather than robot-specific configurations, the method adapts to arbitrary platforms via depth back-projection and a grasp sampler.
- The large-scale contact point pretraining paradigm is noteworthy: cheap point-annotation data establishes a strong spatial localization prior that transfers effectively to downstream tasks.

## Limitations & Future Work
- The method relies on an external grasp sampler for precise grasp pose estimation; task execution fails when the sampler fails.
- The 2D-to-3D depth back-projection is limited by depth estimation accuracy and may fail on transparent or reflective objects.
- The 45% success rate on complex trajectory tasks such as whiteboard wiping leaves substantial room for improvement.
- The PixMo-One-Point pretraining data primarily covers static localization; dynamic trajectory data remains scarce.
- The 2D waypoint representation cannot handle tasks requiring precise force control (e.g., assembly).
- Multi-step manipulation planning and long-horizon task execution are not explored.
- The computational overhead of Position Offset Attention is not analyzed in detail.

## Rating
- Novelty: ⭐⭐⭐⭐ Affordance-based hierarchy combined with embodiment-agnostic representation is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four robot platforms with diverse tasks.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and well-organized.
- Value: ⭐⭐⭐⭐⭐ Directly relevant to practical robotic deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](aether_geometric-aware_unified_world_modeling.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](diffusion_image_prior.md)

<!-- RELATED:END -->
