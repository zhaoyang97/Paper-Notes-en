---
title: >-
  [Paper Note] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][World Models] This paper proposes DLWM, a two-stage Gaussian-centric self-supervised pre-training paradigm. Stage 1 learns 3D Gaussian representations by reconstructing depth and semantic…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "World Models"
  - "3D Gaussian Splatting"
  - "Self-supervised Pre-training"
  - "Occupancy Prediction"
  - "Motion Planning"
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

**Background**: In vision-based autonomous driving, Gaussian-centric representations (e.g., GaussianFormer) provide comprehensive yet sparse scene descriptions via 3D semantic Gaussians, offering a superior expressiveness-efficiency trade-off over dense BEV and sparse query methods for perception/prediction/planning. However, reliance on large-scale manual annotations limits scalable deployment.

**Limitations of Prior Work**: (a) MAE-style pre-training does not explicitly learn 3D geometry; (b) rendering-based pre-training (UniPAD/ViDAR) depends on LiDAR depth supervision; (c) latent world models have been applied to motion planning but remain unexplored for perception and prediction tasks; (d) Gaussian queries lack one-to-one cross-frame correspondence (permutation equivariance), precluding direct temporal feature supervision.

**Key Challenge**: How to design a unified self-supervised pre-training framework for the full lifecycle (perception + prediction + planning) of Gaussian-centric models? And in which representation space should temporal prediction be performed?

**Goal**: To design a comprehensive self-supervised pre-training scheme for Gaussian-centric autonomous driving, covering 3D occupancy perception, 4D occupancy prediction, and motion planning.

**Key Insight**: BEV rasterization is leveraged as a bridge from Gaussian queries to dense representations — BEV features preserve height information and maintain explicit inter-frame spatial correspondence, elegantly resolving the permutation equivariance challenge.

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

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)
- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](colavla_leveraging_cognitive_latent_reasoning_for_hierarchical_parallel_trajecto.md)

</div>

<!-- RELATED:END -->
