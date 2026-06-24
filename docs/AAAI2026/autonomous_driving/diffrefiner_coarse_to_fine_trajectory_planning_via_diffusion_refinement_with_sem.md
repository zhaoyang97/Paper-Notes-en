---
title: >-
  [Paper Note] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving
description: >-
  [AAAI2026][Autonomous Driving][end-to-end autonomous driving] This paper proposes DiffRefiner, a coarse-to-fine two-stage framework that first generates coarse trajectories using a discriminative Proposal Decoder and then refines them iteratively using a diffusion model. Combined with a fine-grained semantic interaction module, it achieves SOTA performance on both NAVSIM v2 and Bench2Drive benchmarks.
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "end-to-end autonomous driving"
  - "diffusion model"
  - "trajectory planning"
  - "coarse-to-fine"
  - "semantic interaction"
date: 2026-05-08
content_hash: a3841f0509118fd1
---

# DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving

**Conference**: AAAI2026  
**arXiv**: [2511.17150](https://arxiv.org/abs/2511.17150)  
**Code**: [nullmax-vision/DiffRefiner](https://github.com/nullmax-vision/DiffRefiner)  
**Area**: Autonomous Driving  
**Keywords**: end-to-end autonomous driving, diffusion model, trajectory planning, coarse-to-fine, semantic interaction

## TL;DR

This paper proposes DiffRefiner, a coarse-to-fine two-stage framework that first generates coarse trajectories using a discriminative Proposal Decoder and then refines them iteratively using a diffusion model. Combined with a fine-grained semantic interaction module, it achieves SOTA performance on both NAVSIM v2 and Bench2Drive benchmarks.

## Background & Motivation

End-to-end autonomous driving (E2E-AD) directly maps raw sensor inputs to trajectory planning, offering higher efficiency compared to traditional modular pipelines. Current trajectory forecasting methods can be categorized into three main paradigms:

1. **Single-shot regression methods**: These are computationally efficient but fail to handle the multi-modality of driving behaviors, often resulting in averaged, sub-optimal predictions at complex intersections.
2. **Anchor-based classification/scoring methods** (e.g., VADv2, HydraMDP++): These formulate prediction as a classification task over predefined anchors. However, expanding the anchor set significantly increases computational complexity, making it difficult to meet real-time requirements.
3. **Diffusion-based methods** (e.g., DiffusionDrive): These generate diverse trajectories through iterative denoising, naturally capturing multi-modality. However, their initialization relies on unstructured Gaussian noise or fixed anchors, lacking adaptivity to the scene and requiring a high number of denoising steps.

Core Motivation: **Can the advantages of discriminative models (fast, structured priors) be combined with the flexible generation capabilities of diffusion models?** That is, first utilizing a discriminative approach to generate "coarse" trajectory proposals, and then employing a diffusion model for "fine" refinement.

## Core Problem

- Existing diffusion methods start denoising from random noise or fixed anchors. The initial distribution deviates from the feasible motion space, leading to a high number of denoising steps and increased latency.
- There is a lack of fine-grained interaction between trajectory prediction and environmental semantics (drivable areas, obstacles), which easily leads to collisions or lane deviations.

## Method

### Overall Architecture

DiffRefiner consists of three core components:

1. **BEV Perception Module**: Encodes multi-view camera inputs into BEV (Bird's-Eye-View) features, performing object detection and semantic segmentation using a sparse detection head and a dense segmentation head respectively.
2. **Proposal Decoder (Coarse Stage)**: Based on a lightweight Transformer, it predicts offsets from clustered trajectory anchors to generate coarse trajectory proposals.
3. **Diffusion Refiner (Fine Stage)**: A conditional diffusion model that utilizes the coarse trajectory as initialization, generating high-quality final trajectories through iterative denoising.

### Proposal Decoder

- Uses 20 trajectory anchors obtained via offline clustering as discrete motion candidates.
- Each anchor is encoded through positional encoding and projected into a query vector by an MLP.
- Intermediary interaction with the Planning Token via cross-attention is performed to output context-enhanced trajectory proposals.
- This stage is essentially a discriminative approach, providing a strong initialization for the subsequent diffusion refinement.

### Diffusion Refiner

**Training Phase**: Performs a forward diffusion process (adding Gaussian noise) on the coarse trajectory. At a random step $t$, a noisy trajectory $\tilde{Y}$ is obtained, and the model learns the reverse denoising process.

**Inference Phase**: Starts with the coarse trajectory (instead of pure noise), requiring very few denoising steps to achieve high-quality results (experiments show that 1 step is close to optimal).

### Fine-Grained Semantic Interaction Module (FGSIM)

This is the core innovation of the method, embedded within the denoising decoder and operating in two stages:

1. **Road-Aware Refinement**: Trajectory features interact with BEV features and drivable area segmentation to constrain predictions within physically passable regions.
2. **Interaction-Aware Refinement**: Integrates dynamic agent features to model interaction and collision avoidance among traffic participants.

The interaction for each semantic category is achieved via a three-level mechanism:

- **Global cross-attention**: Establishes dense correspondences between trajectory features and BEV semantic regions to encode global scene context.
- **Local deformable attention**: Adaptively focuses on critical region semantics near the trajectory endpoints to extract local geometric structures.
- **Gated Fusion**: Dynamically balances global and local representations through a learnable gating network, formulated as $Q_r = Q_r^{(c)} \cdot \text{Gate} + Q_r^{(d)} \cdot (1 - \text{Gate})$.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proposal}} + \mathcal{L}_{\text{refinement}} + \mathcal{L}_{\text{perception}}$$

A two-stage training strategy is adopted: the first stage trains the perception network individually, while the second stage jointly optimizes perception and planning end-to-end. A winner-takes-all strategy is used to select the trajectory closest to the ground truth to compute the loss.

## Key Experimental Results

### NAVSIM v2 (Open-loop Evaluation)

| Method | Backbone | EPDMS |
|------|----------|-------|
| HydraMDP++ | V2-99 | 85.1 |
| DriveSuprim | V2-99 | 86.0 |
| **DiffRefiner** | **V2-99** | **87.4** |
| DiffusionDrive | ResNet34 | 84.0 |
| **DiffRefiner** | **ResNet34** | **86.2** |

Under the ResNet34 backbone, it outperforms the prior work DriveSuprim by 3.7%, and by 1.6% under V2-99.

### Bench2Drive (Closed-loop Evaluation)

| Method | DS | SR(%) |
|------|-----|-------|
| TF++ | 84.2 | 67.3 |
| HiPAD | 86.8 | 69.1 |
| **DiffRefiner** | **87.1** | **71.4** |

Without model ensemble, DS increases by 0.3 and SR increases by 2.3, leading in most sub-metrics of multi-capability evaluations.

### Ablation Study

- **Two-stage vs. Single-stage**: Adding the Refiner improves EPDMS from 85.0 to 86.2 (+1.2).
- **Generative vs. Discriminative Refiner**: The generative Refiner (86.2) outperforms the discriminative one (78.3), verifying that diffusion refinement is better suited for fine-grained adjustments.
- **FGSIM Components**: Planning Token -> +Agent Token -> +BEV Modulation -> +Drivable Area -> +Traffic Participants consecutively increases EPDMS from 82.4 to 85.0.
- **Gated Fusion vs. Additive Fusion**: Gated fusion (86.2) outperforms additive fusion (85.9), demonstrating that adaptive balancing avoids information conflicts.
- **Denoising Steps**: Only 1 step is required to achieve near-optimal performance, with an end-to-end latency of only 27 ms.

## Highlights & Insights

- **Coarse-to-fine hybrid paradigm**: Integrates a discriminative part providing a strong prior with a generative part performing fine-grained optimization. Their complementary advantages yield an elegant design.
- **Exquisite FGSIM module design**: Uses global-local-gated three-level semantic interactions to explicitly model the alignment between trajectories and the environment.
- **Minimal denoising steps**: The high-quality coarse proposals enable the diffusion model to refine in just 1 step, meeting real-time requirements (27 ms).
- **Dual benchmark SOTA**: Sets new performance records on both open-loop (NAVSIM v2) and closed-loop (Bench2Drive) benchmarks, verifying the robustness of the proposed method.

## Limitations & Future Work

- It relies solely on camera inputs without integrating LiDAR; on NAVSIM, the Camera+LiDAR-based GaussianFusion still holds advantages in some sub-metrics.
- The gap between this method and rule-based PDM-Lite (DS 97.0) on Bench2Drive remains significant, indicating room for improvement in the stability of learning-based approaches.
- The impact of the selection and quantity of the 20 clustered anchors on performance has not been deeply explored.
- Performance in closed-loop evaluations for Overtake (60.0) and GiveWay (50.0) remains relatively weak, requiring improvement in complex interactive scenarios.
- Extending the framework to multi-agent joint prediction and planning has not yet been explored.

## Related Work & Insights

| Dimension | DiffusionDrive | DriveSuprim | DiffRefiner |
|------|---------------|-------------|-------------|
| Paradigm | Pure Generative | Pure Discriminative | Hybrid (Discriminative + Generative) |
| Initialization | Anchor Gaussian Mixture | Anchor Classification + Offset | Discriminative Coarse Proposal |
| Semantic Interaction | BEV Space Modulation | Implicit Feature Interaction | Explicit Fine-Grained FGSIM |
| NAVSIM v2 (V2-99) | - | 86.0 | **87.4** |
| Bench2Drive DS | - | - | **87.1** |

The core difference is that DiffRefiner replaces unstructured initialization with a discriminative module and explicitly introduces semantic constraints through FGSIM.

### Insights & Connections

- The "coarse-to-fine" two-stage mindset is highly generalizable and can be extended to other generative planning tasks (e.g., robotic motion planning, UAV path planning).
- The design of global-local gated fusion in FGSIM can be applied to other tasks requiring multi-scale semantic alignment.
- The finding that a high-quality prior significantly reduces diffusion steps provides valuable insights for accelerating diffusion model inference.
- In contrast to other "coarse-to-fine" works like GoalFlow and DriveTransformer, the unique feature of this work lies in employing a diffusion model rather than a deterministic Transformer during the refinement stage.

## Rating

- Novelty: 7/10 — The combination of coarse-to-fine and diffusion is not entirely new, but the design of FGSIM is unique.
- Experimental Thoroughness: 9/10 — SOTA on dual benchmarks + detailed ablation studies, with clear contributions from each component.
- Writing Quality: 8/10 — Clear structure, standardized illustrations, and complete mathematical formulations.
- Value: 8/10 — Practical methodology, outstanding performance, and controllable latency, offering both engineering and academic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[CVPR 2026\] ActiveAD: Planning-Oriented Active Learning for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/activead_planning-oriented_active_learning_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/resad_normalized_residual_trajectory_modeling_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving](../../CVPR2026/autonomous_driving/wam-flow_parallel_coarse-to-fine_motion_planning_via_discrete_flow_matching_for_.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)

</div>

<!-- RELATED:END -->
