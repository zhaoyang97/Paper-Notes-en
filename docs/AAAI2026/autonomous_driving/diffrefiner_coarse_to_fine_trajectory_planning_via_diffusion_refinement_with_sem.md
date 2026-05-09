---
title: >-
  [Paper Note] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving
description: >-
  [AAAI2026][Autonomous Driving][end-to-end autonomous driving] This paper proposes DiffRefiner, a coarse-to-fine two-stage framework that first employs a discriminative Proposal Decoder to generate coarse trajectory proposals, then iteratively refines them via a diffusion model, combined with a fine-grained semantic interaction module. The method achieves state-of-the-art performance on both NAVSIM v2 and Bench2Drive benchmarks.
tags:
  - AAAI2026
  - Autonomous Driving
  - end-to-end autonomous driving
  - diffusion model
  - trajectory planning
  - coarse-to-fine
  - semantic interaction
date: 2026-05-08
content_hash: ff4d904b58872765
---

# DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving

**Conference**: AAAI2026
**arXiv**: [2511.17150](https://arxiv.org/abs/2511.17150)
**Code**: [nullmax-vision/DiffRefiner](https://github.com/nullmax-vision/DiffRefiner)
**Area**: Autonomous Driving
**Keywords**: end-to-end autonomous driving, diffusion model, trajectory planning, coarse-to-fine, semantic interaction

## TL;DR

This paper proposes DiffRefiner, a coarse-to-fine two-stage framework that first employs a discriminative Proposal Decoder to generate coarse trajectory proposals, then iteratively refines them via a diffusion model, combined with a fine-grained semantic interaction module. The method achieves state-of-the-art performance on both NAVSIM v2 and Bench2Drive benchmarks.

## Background & Motivation

End-to-end autonomous driving (E2E-AD) directly maps raw sensor inputs to trajectory planning, offering greater efficiency than traditional modular pipelines. Current trajectory prediction methods fall into three categories:

1. **Single-step regression methods**: Computationally efficient but unable to handle the multimodal nature of driving behaviors, prone to producing averaged suboptimal predictions at complex intersections.
2. **Anchor-based scoring methods** (e.g., VADv2, HydraMDP++): Cast prediction as a classification problem, but scaling up the anchor set substantially increases computational complexity, making real-time deployment difficult.
3. **Diffusion-based methods** (e.g., DiffusionDrive): Generate diverse trajectories via iterative denoising, naturally capturing multimodality, but rely on unstructured Gaussian noise or fixed anchors for initialization, lacking scene adaptability and requiring many denoising steps.

Core motivation: **Can the advantages of discriminative models—fast inference and structured priors—be combined with the flexible generative capability of diffusion models?** Specifically, a discriminative method first produces coarse trajectory proposals, which are then refined by a diffusion model.

## Core Problem

- Existing diffusion methods start denoising from random noise or fixed anchors, whose initial distributions deviate from the feasible motion space, necessitating many denoising steps and increasing latency.
- Trajectory prediction lacks fine-grained interaction with environmental semantics (drivable areas, obstacles), leading to collisions or lane departures.

## Method

### Overall Architecture

DiffRefiner consists of three core modules:

1. **BEV Perception Module**: Encodes multi-view camera inputs into BEV features, with a sparse detection head and a dense segmentation head for object detection and semantic segmentation, respectively.
2. **Proposal Decoder (coarse stage)**: A lightweight Transformer that predicts offsets from clustered trajectory anchors to generate coarse trajectory proposals.
3. **Diffusion Refiner (fine stage)**: A conditional diffusion model that takes coarse trajectories as initialization and generates high-quality final trajectories through iterative denoising.

### Proposal Decoder

- Uses 20 offline-clustered trajectory anchors as discrete motion candidates.
- Each anchor is projected into a query vector via positional encoding and MLP.
- Cross-attention with the Planning Token produces context-enhanced trajectory proposals.
- This stage is essentially a discriminative method, providing strong initialization for subsequent diffusion refinement.

### Diffusion Refiner

**Training**: A forward diffusion process adds Gaussian noise to the coarse trajectory; at a random step $t$, the noisy trajectory $\tilde{Y}$ is obtained, and the model learns the reverse denoising process.

**Inference**: Starting from the coarse trajectory rather than pure noise, high-quality results are obtained with very few denoising steps (experiments show that a single step approaches optimal performance).

### Fine-Grained Semantic Interaction Module (FGSIM)

This is the core innovation of the method, embedded in the denoising decoder and operating in two stages:

1. **Road-aware refinement**: Trajectory features interact with BEV features and drivable-area segmentation, constraining predictions to physically traversable regions.
2. **Interaction-aware refinement**: Dynamic agent features are introduced to model interactions among traffic participants and facilitate collision avoidance.

Interaction for each semantic category is realized through a three-level mechanism:

- **Global cross-attention**: Establishes dense correspondences between trajectory features and BEV semantic regions, encoding global scene context.
- **Local deformable attention**: Adaptively attends to critical regional semantics near trajectory endpoints, extracting local geometric structure.
- **Gated fusion**: A learnable gating network dynamically balances global and local representations: $Q_r = Q_r^{(c)} \cdot \text{Gate} + Q_r^{(d)} \cdot (1 - \text{Gate})$

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proposal}} + \mathcal{L}_{\text{refinement}} + \mathcal{L}_{\text{perception}}$$

A two-stage training strategy is adopted: the perception network is trained independently in the first stage, followed by end-to-end joint optimization of perception and planning in the second stage. A winner-takes-all strategy selects the trajectory closest to the ground truth for loss computation.

## Key Experimental Results

### NAVSIM v2 (Open-loop Evaluation)

| Method | Backbone | EPDMS |
|--------|----------|-------|
| HydraMDP++ | V2-99 | 85.1 |
| DriveSuprim | V2-99 | 86.0 |
| **DiffRefiner** | **V2-99** | **87.4** |
| DiffusionDrive | ResNet34 | 84.0 |
| **DiffRefiner** | **ResNet34** | **86.2** |

DiffRefiner surpasses DriveSuprim by 3.7% with ResNet34 backbone and by 1.6% with V2-99.

### Bench2Drive (Closed-loop Evaluation)

| Method | DS | SR(%) |
|--------|----|-------|
| TF++ | 84.2 | 67.3 |
| HiPAD | 86.8 | 69.1 |
| **DiffRefiner** | **87.1** | **71.4** |

Without model ensembling, DiffRefiner improves DS by 0.3 and SR by 2.3, leading on most sub-metrics across capability categories.

### Ablation Study

- **Two-stage vs. single-stage**: Adding the Refiner improves EPDMS from 85.0 to 86.2 (+1.2).
- **Generative vs. discriminative Refiner**: The generative Refiner (86.2) outperforms the discriminative variant (78.3), validating that diffusion refinement is better suited for fine-grained adjustment.
- **FGSIM components**: Incrementally adding Planning Token → Agent Token → BEV Modulation → Drivable Area → Traffic Participants progressively improves EPDMS from 82.4 to 85.0.
- **Gated fusion vs. additive fusion**: Gated fusion (86.2) outperforms additive fusion (85.9), with adaptive balancing preventing information conflicts.
- **Denoising steps**: A single denoising step achieves near-optimal performance, with an end-to-end latency of only 27ms.

## Highlights & Insights

- **Hybrid coarse-to-fine paradigm**: The discriminative stage provides strong priors while the generative stage performs fine-grained optimization—an elegant design that leverages the complementary strengths of both paradigms.
- **Sophisticated FGSIM design**: The three-level global–local–gated semantic interaction explicitly models alignment between trajectories and the environment.
- **Minimal denoising steps**: High-quality coarse proposals reduce diffusion refinement to a single step, meeting real-time requirements (27ms).
- **Dual-benchmark SOTA**: Achieving state-of-the-art results simultaneously on both open-loop (NAVSIM v2) and closed-loop (Bench2Drive) benchmarks validates the robustness of the approach.

## Limitations & Future Work

- Only camera inputs are used; LiDAR is not incorporated. GaussianFusion (Camera+LiDAR) still holds advantages on certain sub-metrics in NAVSIM.
- A notable gap remains between DiffRefiner and the rule-based PDM-Lite (DS 97.0) on Bench2Drive, indicating that learning-based methods still have room for improvement in stability.
- The impact of the number and selection strategy of the 20 clustered anchors on performance is not thoroughly investigated.
- Overtake (60.0) and GiveWay (50.0) capabilities remain relatively weak in closed-loop evaluation, leaving room for improvement in complex interaction scenarios.
- Extending the framework to joint multi-agent prediction and planning has not been explored.

## Related Work & Insights

| Dimension | DiffusionDrive | DriveSuprim | DiffRefiner |
|-----------|---------------|-------------|-------------|
| Paradigm | Purely generative | Purely discriminative | Hybrid (discriminative + generative) |
| Initialization | Anchor Gaussian mixture | Anchor classification + offset | Discriminative coarse proposals |
| Semantic interaction | BEV spatial modulation | Implicit feature interaction | Explicit fine-grained FGSIM |
| NAVSIM v2 (V2-99) | — | 86.0 | **87.4** |
| Bench2Drive DS | — | — | **87.1** |

The key distinction of DiffRefiner lies in replacing unstructured initialization with a discriminative module and explicitly incorporating semantic constraints via FGSIM.

The coarse-to-fine two-stage paradigm is generalizable to other generative planning tasks (e.g., robot motion planning, UAV path planning). The global–local gated fusion design in FGSIM can be adopted in other tasks requiring multi-scale semantic alignment. The finding that high-quality priors substantially reduce the required denoising steps has broader implications for accelerating diffusion model inference. Compared to other coarse-to-fine works such as GoalFlow and DriveTransformer, the distinctive feature of this paper is the use of diffusion rather than a deterministic Transformer in the refinement stage.

## Rating

- **Novelty**: 7/10 — The combination of coarse-to-fine and diffusion is not entirely new, but the FGSIM design is distinctive.
- **Experimental Thoroughness**: 9/10 — Dual-benchmark SOTA with detailed ablations clearly attributing contributions of individual components.
- **Writing Quality**: 8/10 — Well-structured, with clear figures and complete mathematical formulations.
- **Value**: 8/10 — Practical methodology, strong performance, and controllable latency make it valuable both for engineering and academic research.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)

<!-- RELATED:END -->
