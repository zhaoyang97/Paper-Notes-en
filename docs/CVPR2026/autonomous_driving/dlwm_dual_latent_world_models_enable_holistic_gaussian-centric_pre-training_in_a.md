---
title: >-
  [Paper Note] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][World Models] This paper proposes DLWM, a holistic Gaussian-centric pre-training paradigm based on dual latent world models for autonomous driving. Stage 1 learns 3D Gaussian scene representations via self-supervised rendering of multi-view semantic and depth maps. Stage 2 trains two latent world models: a Gaussian-flow-guided model for downstream occupancy perception/prediction (+1.02/+2.68 mIoU), and an ego-trajectory-guided model for motion planning (−16% L2 error). The framework resolves the permutation equivariance challenge that prevents direct cross-frame supervision of Gaussian queries.
tags:
  - CVPR 2026
  - Autonomous Driving
  - World Models
  - 3D Gaussians
  - Self-supervised Pre-training
  - Occupancy Prediction
  - Motion Planning
date: 2026-05-08
content_hash: 16ee20fdf93b27b8
---

# DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2604.00969](https://arxiv.org/abs/2604.00969)
**Code**: None
**Area**: Autonomous Driving / Self-supervised Pre-training for Autonomous Driving
**Keywords**: World Models, 3D Gaussian Splatting, Self-supervised Pre-training, Occupancy Prediction, Motion Planning

## TL;DR
This paper proposes DLWM, a two-stage Gaussian-centric self-supervised pre-training paradigm. Stage 1 learns 3D Gaussian representations by reconstructing depth and semantic maps. Stage 2 trains dual latent world models — a Gaussian-flow-guided temporal prediction model (for occupancy perception/prediction) and an ego-planning-guided temporal prediction model (for motion planning) — achieving significant performance gains across all three core tasks.

## Background & Motivation

**State of the Field**: In vision-based autonomous driving, Gaussian-centric representations (e.g., GaussianFormer) provide comprehensive yet sparse scene descriptions via 3D semantic Gaussians, offering a superior expressiveness-efficiency trade-off over dense BEV and sparse query methods for perception/prediction/planning. However, reliance on large-scale manual annotations limits scalable deployment.

**Limitations of Prior Work**: (a) MAE-style pre-training does not explicitly learn 3D geometry; (b) rendering-based pre-training (UniPAD/ViDAR) depends on LiDAR depth supervision; (c) latent world models have been applied to motion planning but remain unexplored for perception and prediction tasks; (d) Gaussian queries lack one-to-one cross-frame correspondence (permutation equivariance), precluding direct temporal feature supervision.

**Root Cause**: How to design a unified self-supervised pre-training framework for the full lifecycle (perception + prediction + planning) of Gaussian-centric models? And in which representation space should temporal prediction be performed?

**Paper Goals**: To design a comprehensive self-supervised pre-training scheme for Gaussian-centric autonomous driving, covering 3D occupancy perception, 4D occupancy prediction, and motion planning.

**Starting Point**: BEV rasterization is leveraged as a bridge from Gaussian queries to dense representations — BEV features preserve height information and maintain explicit inter-frame spatial correspondence, elegantly resolving the permutation equivariance challenge.

**Core Idea**: Stage 1 learns Gaussian representations (via self-supervised reconstruction) → Stage 2 learns temporal evolution (dual latent world models serving perception/prediction and planning respectively).

## Method

### Overall Architecture
**Stage 1**: Image encoder (ResNet+FPN) → Gaussian encoder (self-attention + image cross-attention + refinement) → differentiable rendering to reconstruct depth and semantic maps.
**Stage 2a**: Frozen perception module + Gaussian flow prediction → ego-motion alignment → BEV rasterization → L2-supervised prediction of future BEV features (for downstream perception/prediction tasks).
**Stage 2b**: Current Gaussian latent + predicted ego trajectory → conditional prediction of future BEV features → L2 supervision (for downstream planning tasks).

### Key Designs

1. **Gaussian Representation Learning (Stage 1)**:

    - **Function**: Self-supervised learning of 3D semantic Gaussians to reconstruct multi-view depth and semantic maps.
    - **Mechanism**: The Gaussian encoder iteratively refines queries in three steps — (1) self-attention module: inter-Gaussian interaction + 4D sparse convolution for cross-frame temporal propagation; (2) image cross-attention: deformable sampling to aggregate visual features using camera intrinsics and extrinsics; (3) refinement module: adjusts Gaussian attributes (position, covariance, opacity, semantic logits).
    - **Supervision Signal**: Depth GT from LiDAR (sparse but precise) + Metric3D pseudo-depth (dense but noisy); semantic GT from Grounded SAM — **no manual annotation required**.
    - **Design Motivation**: Rendering constraints force Gaussians to learn physical priors. LiDAR and Metric3D are complementary — the former provides accuracy while the latter provides dense coverage.

2. **Gaussian-Flow-Guided Latent World Model (Stage 2a)**:

    - **Function**: Predicts motion flow of 3D Gaussians and performs temporal feature prediction via BEV rasterization.
    - **Mechanism**:
        - A flow prediction head estimates the dynamic displacement $\Delta\mu_k^t$ for each Gaussian.
        - Ego-motion alignment: $\mu_k^{t+1} = \mathbf{T}_{ego}^{t \to t+1}(\mu_k^t + \Delta\mu_k^t)$
        - BEV rasterization: transformed 3D Gaussians are projected onto a 2D BEV plane to obtain the predicted latent $\hat{B}_{t+1}$.
        - Supervision: $\mathcal{L}_{bev} = \|\hat{B}_{t+1} - B_{t+1}\|_2$, where $B_{t+1}$ is produced by the frozen perception module from ground-truth $t+1$ frame images.
    - **Design Motivation**: Permutation equivariance of Gaussian queries makes direct feature supervision infeasible. BEV rasterization projects sparse queries onto a dense grid that preserves height information (via vertical stacking) and has well-defined spatial correspondence — serving as the perfect bridge between query space and supervision space.

3. **Ego-Planning-Guided Latent World Model (Stage 2b)**:

    - **Function**: Predicts future scenes conditioned on the predicted ego trajectory, improving motion planning.
    - **Mechanism**: Current Gaussian latent + predicted ego trajectory → conditional prediction of future BEV features → L2 supervision.
    - **Design Motivation**: Planning requires the ability to imagine future scenes conditioned on the ego's intended actions ("what if I drive this way"), which differs from the precise scene flow needed for perception/prediction. A dedicated world model branch is therefore necessary.

4. **4D Occupancy Prediction Design**:

    - **Function**: Streaming autoregressive prediction of future occupancy.
    - **Mechanism**: (1) Align current Gaussians to the next frame using the ego trajectory; (2) supplement newly entered regions with random Gaussians; (3) process both existing and supplemented Gaussians jointly with 3D sparse convolution and refinement layers; (4) apply Gaussian-to-Occupancy Splatting to predict next-frame occupancy.
    - **Design Motivation**: The autoregressive formulation enables long-horizon prediction, while random Gaussian initialization supports scene extrapolation into previously unobserved regions.

### Loss & Training
- **Stage 1**: $\mathcal{L}_{rec} = 1.0 \cdot \mathcal{L}_d + 0.05 \cdot \mathcal{L}_{pd} + 1.0 \cdot \mathcal{L}_{sem}$
- **Stage 2a**: $\mathcal{L}_{rec} + \mathcal{L}_{bev}$ (reconstruction + BEV prediction)
- **Stage 2b**: ego planning loss + $\mathcal{L}_{bev}$
- The two stages are pre-trained sequentially; pre-trained weights are loaded into corresponding modules during fine-tuning.

## Key Experimental Results

### Main Results (3D Occupancy Perception, SurroundOcc-nuScenes)

| Method | IoU | mIoU | Note |
|--------|-----|------|------|
| GaussianFormer | 29.83 | 19.10 | Base Gaussian method |
| GaussianFormer-2 | 31.74 | 20.82 | Improved version |
| GaussianWorld* | 32.77 | 21.79 | Temporal modeling |
| Baseline (no pre-training) | 31.77 | 20.83 | Paper's baseline |
| **DLWM (Ours)** | **34.61** | **21.85** | **+2.84 IoU, +1.02 mIoU** |

### Pre-training Gain Summary

| Task | Pre-training Gain | Specific Values |
|------|-------------------|-----------------|
| 3D Occupancy Perception | +1.02 mIoU | IoU: 31.77→34.61 |
| 4D Occupancy Prediction | +2.68 mIoU (avg) | Significant gains at 1s/2s/3s |
| Motion Planning | −16% L2 error | Substantial planning improvement |

### Ablation Study

| Pre-training Config | Perception | Prediction | Planning | Note |
|--------------------|-----------|-----------|---------|------|
| No pre-training | baseline | baseline | baseline | — |
| Stage 1 only | ++ | + | + | Gaussian representation learning is effective |
| Stage 1 + 2a | +++ | +++ | + | Temporal feature learning further improves perception/prediction |
| **Stage 1 + 2a + 2b** | +++ | +++ | +++ | **Dual world models achieve best overall performance** |

### Key Findings
- **Pre-training consistently improves all three tasks**: perception +1.02, prediction +2.68, planning −16% L2, validating the efficacy of full-lifecycle pre-training.
- **BEV as the latent representation space**: resolves the critical technical challenge of Gaussian query permutation equivariance.
- **Dual world model design is well-motivated**: perception/prediction requires precise scene flow while planning requires conditional imagination — functional decoupling yields optimal performance.
- **Multi-source self-supervised labels**: LiDAR + Metric3D + Grounded SAM, entirely annotation-free.

## Highlights & Insights
- **Full-lifecycle pre-training**: The first work to design a unified pre-training framework covering all core tasks for Gaussian-centric autonomous driving.
- **BEV bridge idea**: Using BEV rasterization to resolve Gaussian query permutation equivariance is a generalizable technical approach.
- **−16% planning L2 error**: The substantial planning improvement from pre-training demonstrates that temporal perceptual capability directly benefits decision-making modules.
- **No manual annotation**: All supervision signals are automatically generated (LiDAR depth, pseudo-depth, automated segmentation).

## Limitations & Future Work
- The two-stage pre-training involves sequential dependencies; end-to-end joint training could be more efficient.
- The two world models are trained independently, leaving information sharing between them unexplored.
- The current design predicts only one future step; multi-step long-horizon prediction capability remains to be validated.
- Robustness under out-of-distribution scenarios such as extreme weather and nighttime conditions is not discussed.

## Related Work & Insights
- **vs. UniPAD/ViDAR**: These methods use volumetric rendering and require LiDAR depth; this paper uses Gaussian splatting with multi-source automatic labels.
- **vs. GaussianWorld**: GaussianWorld performs scene evolution in 3DGS space but lacks a systematic pre-training strategy.
- **vs. SQS**: SQS targets only sparse perception pre-training without covering planning; DLWM covers the full task lifecycle.
- **Insight**: The choice of representation space for latent world models is critical — BEV is a natural bridge for Gaussian-centric models.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual latent world models and BEV bridge design are valuable contributions, though the overall framework is a combination of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ All three tasks are covered with per-stage ablations, though planning experiment details are somewhat sparse.
- Writing Quality: ⭐⭐⭐⭐ The architecture diagram is clear and the technical motivation is well articulated.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic pre-training solution for Gaussian-centric autonomous driving with consistent gains across all three tasks.

---

# DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2604.00969](https://arxiv.org/abs/2604.00969)
**Code**: Available
**Area**: Autonomous Driving / 3D Scene Understanding
**Keywords**: World Models, 3D Gaussians, Self-supervised Pre-training, Occupancy Prediction, Motion Planning

## TL;DR
This paper proposes DLWM, a holistic Gaussian-centric pre-training paradigm based on dual latent world models for autonomous driving. Stage 1 learns 3D Gaussian scene representations via self-supervised rendering of multi-view semantic and depth maps. Stage 2 trains two latent world models: a Gaussian-flow-guided model for downstream occupancy perception/prediction (+1.02/+2.68 mIoU), and an ego-trajectory-guided model for motion planning (−16% L2 error). The framework resolves the permutation equivariance challenge that prevents direct cross-frame supervision of Gaussian queries.

## Background & Motivation

1. **State of the Field**: In vision-based autonomous driving, 3D Gaussian representations achieve a better expressiveness-efficiency trade-off than BEV or sparse query methods across perception/prediction/planning multi-task settings. However, reliance on large-scale annotations limits scalable deployment.
2. **Limitations of Prior Work**: (1) MAE-style pre-training does not explicitly learn 3D geometry; (2) rendering-based pre-training (UniPAD/ViDAR) requires LiDAR depth; (3) latent world models have proven effective for planning but **remain unexplored for perception/prediction**; (4) the permutation equivariance of Gaussian queries — independently initialized across frames with no one-to-one correspondence — makes direct feature supervision infeasible.
3. **Key Breakthrough**: Transforming Gaussian queries into dense grid representations via BEV rasterization → height information is preserved and inter-frame spatial correspondence is well-defined → this serves as a viable latent space for temporal supervision in the world model.
4. **Core Idea**: Two stages — Stage 1 learns Gaussian spatial representations (via rendering reconstruction) → Stage 2 trains dual world models: (a) Gaussian-flow-guided → perception/prediction; (b) ego-trajectory-guided → planning.

## Method

### Overall Architecture
Multi-view images → 2D backbone → 3D Gaussian query prediction → **Stage 1**: differentiable rendering of multi-view semantic and depth maps → self-supervised reconstruction loss. Pre-trained weights initialize → **Stage 2a**: Gaussian-flow-guided world model — predicts 3D Gaussian flow (displacements) → propagates current Gaussians to future frames → BEV rasterization into latent representation → supervised against frozen future-frame perception features. **Stage 2b**: ego-trajectory-guided world model — predicts ego trajectory → conditions current Gaussian latent → predicts future Gaussian BEV features → aligned against the same frozen features.

### Key Designs

1. **Stage 1: Gaussian Spatial Representation Learning**:
    - Predicts 3D Gaussians (position/scale/rotation/color/semantic attributes) → differentiable rendering of multi-view semantic and depth maps → self-supervised reconstruction.
    - No LiDAR depth required — depth is also obtained via rendering.
    - Learned Gaussian queries carry rich contextual features → provide strong initialization for Stage 2.

2. **Stage 2a: Gaussian-Flow-Guided World Model (Perception/Prediction)**:
    - Predicts 3D flow (displacement vectors) for each Gaussian → propagates current-frame Gaussians to their future positions.
    - BEV rasterization: propagated Gaussians are splatted onto a BEV grid → dense latent representation is obtained.
    - The frozen Stage 1 perception module processes future-frame multi-view images → produces "target" features → serves as the supervision signal for the latent representation.
    - **Resolving permutation equivariance**: Gaussian queries are permutation-equivariant → but BEV rasterization produces a fixed grid → inter-frame spatial correspondence is well-defined → temporal supervision becomes feasible.

3. **Stage 2b: Ego-Trajectory-Guided World Model (Planning)**:
    - Predicts ego trajectory → encodes it as a motion condition.
    - Combines motion condition with current Gaussian latent → predicts future scene Gaussian BEV features.
    - Aligned with the same frozen perception features.
    - **Design Motivation**: Planning and perception require different temporal prediction signals — planning needs trajectory conditioning (ego motion affects observations), while perception requires scene-level flow.

4. **Why Dual World Models Instead of a Single Model**:
    - Perception/prediction focus on scene-level motion (other vehicles/pedestrians) → Gaussian flow is the natural supervision.
    - Planning focuses on the consequences of ego decisions → trajectory conditioning is the natural supervision.
    - The two tasks have different optimization objectives → separate training avoids mutual interference.

## Key Experimental Results

### Main Results (SurroundOcc + nuScenes)

| Downstream Task | No Pre-training | +DLWM | Gain |
|----------------|:-:|:-:|:-:|
| 3D Occupancy Perception (mIoU) | Baseline | +1.02 | Significant |
| 4D Occupancy Prediction (mIoU) | Baseline | +2.68 | More Significant |
| Motion Planning (L2 error) | Baseline | −16% | Substantial |

### Ablation Study

| Configuration | Perception | Prediction | Planning |
|--------------|:-:|:-:|:-:|
| w/o Stage 1 | Degraded | Degraded | Degraded |
| w/o Stage 2a (Gaussian Flow WM) | Degraded | Largely Degraded | Unchanged |
| w/o Stage 2b (Trajectory WM) | Unchanged | Unchanged | Degraded |
| **Full DLWM** | **Best** | **Best** | **Best** |

→ Confirms that the two world models each serve distinct downstream tasks.

### Key Findings
- 4D occupancy prediction benefits most (+2.68 mIoU) → temporal prediction tasks gain the most from world model pre-training.
- The Gaussian-flow-guided WM also improves perception (+1.02) → the model learns better scene dynamics representations.
- The trajectory-guided WM exclusively benefits planning → validates the rationale for separate training.
- BEV rasterization successfully resolves Gaussian permutation equivariance → temporal supervision in latent space is viable.

## Highlights & Insights
- **Permutation Equivariance → Elegant Resolution via BEV Rasterization**: Gaussian queries have no cross-frame correspondence → but after splatting to BEV they form a fixed grid → this transformation is the key enabler of the entire framework.
- **Clear Task Division Between Dual World Models**: Perception/prediction → driven by scene flow; planning → driven by trajectory conditioning. Different tasks require different temporal prediction signals → division of labor outperforms a unified model.
- **Full-Lifecycle Pre-training**: From spatial (Stage 1 rendering reconstruction) to temporal (Stage 2 world models) → covers the complete chain of perception/prediction/planning → a single pre-training benefits multiple tasks.

## Limitations & Future Work
- The two world models in Stage 2 are trained separately → joint training may enable discovery of shared temporal features.
- BEV rasterization discards fine-grained height-dimension information → 3D voxel rasterization may be superior but at greater computational cost.
- The two-stage training pipeline is relatively complex → whether it can be simplified into a single end-to-end stage remains open.
- Current world models predict only one future step → multi-step prediction would be more beneficial for long-horizon planning.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual world models + Gaussian-flow-guided temporal pre-training.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-task validation on SurroundOcc + nuScenes with per-component ablations.
- Writing Quality: ⭐⭐⭐⭐ Pipeline diagram is clear; the permutation equivariance problem and its solution are well explained.
- Value: ⭐⭐⭐⭐⭐ Provides an important advancement in pre-training for Gaussian-centric autonomous driving methods.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)
- [\[CVPR 2026\] ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](colavla_leveraging_cognitive_latent_reasoning_for_hierarchical_parallel_trajecto.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](../../AAAI2026/autonomous_driving/worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)

<!-- RELATED:END -->
