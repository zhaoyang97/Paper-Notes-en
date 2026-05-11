---
title: >-
  [Paper Note] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching
description: >-
  [ICCV 2025][Image Generation][robotic manipulation] EC-Flow introduces an "embodiment-centric flow" paradigm that predicts pixel-level motion trajectories of the robot body from action-unlabeled RGB videos…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "robotic manipulation"
  - "action-unlabeled video learning"
  - "optical flow prediction"
  - "URDF kinematics"
  - "diffusion models"
date: 2026-05-08
content_hash: 7e7f6e470abfd51a
---

# EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching

**Conference**: ICCV 2025
**arXiv**: [2507.06224](https://arxiv.org/abs/2507.06224)
**Code**: [ec-flow1.github.io](https://ec-flow1.github.io)
**Area**: Image Generation
**Keywords**: robotic manipulation, action-unlabeled video learning, optical flow prediction, URDF kinematics, diffusion models

## TL;DR

EC-Flow introduces an "embodiment-centric flow" paradigm that predicts pixel-level motion trajectories of the robot body from action-unlabeled RGB videos, and converts visual predictions into executable actions via URDF kinematic constraints. It substantially outperforms object-centric methods in scenarios involving deformable objects, occlusion, and non-displacement manipulation.

## Background & Motivation

Current language-guided robotic manipulation systems primarily rely on imitation learning (VLA models), which require large-scale datasets with low-level action annotations. Although vast quantities of action-unlabeled manipulation videos exist on the internet, learning manipulation policies from such data remains an open problem.

Existing video-based manipulation learning methods fall into two categories:

**Video prediction + imitation learning**: Future frames are predicted first, and actions are inferred via an inverse dynamics model. However, this still requires action-annotated data to train the inverse dynamics model.

**Object-centric optical flow methods** (e.g., AVDC, Track2Act): Object flow trajectories are predicted and actions are inferred from object transformations. While no action annotations are required, three fundamental limitations remain:
   - **Rigid body assumption**: All parts of the object are assumed to undergo uniform rigid transformation, making it inapplicable to deformable objects (e.g., folding clothes).
   - **Occlusion fragility**: Actions are inferred entirely from object state changes and fail when the object is occluded.
   - **Failure on non-displacement tasks**: Rotational motions (e.g., turning a switch) or subtle movements (e.g., pressing a mouse button) cannot be captured.

Core insight: **Shifting the prediction target from object flow to embodiment flow fundamentally avoids all three limitations above.** Robot body motion: (a) does not depend on object properties → applicable to any object; (b) remains visible in most scenarios → robust to occlusion; (c) directly reflects the executed action → handles non-displacement tasks.

## Method

### Overall Architecture

EC-Flow consists of two core modules:
1. **Embodiment-centric flow prediction** (Sec. 3.2): Given an initial frame and a language instruction, the model predicts 2D trajectories over $T$ future steps for randomly sampled points on the robot body.
2. **Kinematics-aware action computation** (Sec. 3.3): Using kinematic constraints from the robot's URDF file, 2D optical flow is converted into 6-DoF end-effector pose transformations.

### Key Designs

1. **Flow Prediction with Diffusion Models**:

    - Data construction: Grounded SAM is used to segment the robot body mask → $N_p = 400$ points are randomly sampled within the mask → CoTracker tracks their 2D trajectories throughout the video.
    - A conditional diffusion model is adopted for flow prediction, with conditioning signal $\mathbf{c} = [\tilde{\mathbf{v}}, \tilde{\mathbf{l}}, \tilde{\mathbf{s}}]$ comprising: visual context (initial frame encoded by ResNet-50), language guidance (CLIP text encoder), and initial state (starting point coordinates).
    - Training objective: $\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \mathbf{z}_0, \epsilon}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{c})\|_2^2]$
    - At inference, DDIM sampling generates complete $T=8$ step trajectories.

2. **Target Image Prediction — Aligning Object Interaction with Language Instructions**:

    - Challenge: Pure embodiment flow prediction may generate actions that do not effectively interact with the target object.
    - Solution: An auxiliary target image prediction branch is introduced, sharing the diffusion timestep with the flow prediction branch.
    - The target image generator receives augmented conditioning $\mathbf{c_t}^{\text{img}} = [\tilde{\mathbf{v}}, \tilde{\mathbf{l}}, \tilde{\mathbf{s}}, f_t^{\text{flow}}]$, where $f_t^{\text{flow}}$ denotes the flow prediction output.
    - Total loss: $L = L_{\text{flow}} + \lambda L_{\text{image}}$ ($\lambda = 0.4$)
    - Design motivation: Joint optimization establishes an implicit constraint — generated actions must lead to physically plausible object state changes aligned with the language instruction.

3. **URDF-Aware Action Computation**:

    - Step 1 (point assignment): The geometric properties of the URDF file are used to compute 2D bounding boxes for each joint; sampled points are assigned to their corresponding joints (retaining only points uniquely belonging to a single joint).
    - Step 2 (action optimization): End-effector pose is optimized by minimizing reprojection error:

    $\mathbf{T}_{ee}^* = \arg\min_{\mathbf{T}_{ee}} \sum_{j=1}^M \sum_{i=1}^{N_j} \|\pi(\mathbf{T}_{ee} \cdot {}_j^{ee}\mathbf{T} \cdot \mathbf{P}_{ji}^{3D}) - \mathbf{P}_{ji}^{(t+1)_{2D}}\|_2$

    - Design motivation: Different joints are subject to different kinematic constraints and cannot be handled by a single unified transformation; the URDF is a standard configuration file for robotic systems, requiring no additional annotation.

### Loss & Training

- Flow prediction: diffusion noise prediction MSE loss.
- Target image: diffusion noise prediction MSE loss (weight 0.4).
- Training configuration: 8 × RTX 4090 GPUs, batch size 56, flow lr = 5e-5, image lr = 1e-4, DDIM 250-step sampling.

## Key Experimental Results

### Main Results — Meta-World Benchmark

| Method | Data Requirement | Avg. Success Rate ↑ |
|--------|-----------------|---------------------|
| BC-Scratch | Action annotations | 0.204 |
| BC-R3M | Action annotations | 0.360 |
| Diffusion Policy | Action annotations | 0.298 |
| UniPi | Video + action annotations | 0.093 |
| AVDC (object-centric) | Video only | 0.489 |
| Track2Act (object-centric) | Video only | 0.556 |
| **EC-Flow (embodiment-centric)** | **Video only** | **0.720** |

EC-Flow surpasses the strongest baseline by 16.4%. Particularly notable improvements are observed on occlusion/non-displacement tasks such as btn-top-press (1.00 vs. 0.40) and hammer-strike (0.88 vs. 0.24).

### Ablation Study — Contribution of Each Component

| # | Flow Prediction | Target Image | Point Filtering | EE Only | Avg. Success Rate |
|---|----------------|-------------|-----------------|---------|-------------------|
| 1 | End-to-end EC-Flow | ✓ | ✓ | ✗ | **0.720** |
| 2 | Video + GT flow | ✗ | ✓ | ✗ | 0.636 |
| 3 | End-to-end EC-Flow | ✗ | ✗ | ✗ | 0.582 |
| 4 | End-to-end EC-Flow | ✓ | ✓ | ✓ | 0.604 |
| 5 | Video + GT flow | ✓ | ✓ | ✗ | 0.667 |

Target image prediction (#3 vs. #1: +13.8%) and full-body point modeling (#4 vs. #1: +11.6%) contribute the most.

### Key Findings

1. **Occlusion robustness**: Object-centric methods fail completely when the object is occluded, whereas EC-Flow infers actions from the motion of other visible robot joints.
2. **Hallucination in video prediction**: Video prediction models may generate hallucinated multiple robot arms, causing optical flow tracking errors; end-to-end flow prediction avoids this issue.
3. **Real-world validation** (7 tasks): EC-Flow achieves a 45% improvement on deformable object manipulation and an 80% improvement on non-displacement tasks relative to Track2Act.
4. **Cross-embodiment data availability**: Preliminary experiments show that incorporating 50 human videos improves success rate from 46% to 70% in 2-demo scenarios.

## Highlights & Insights

- The paradigm shift from "object-centric" to "embodiment-centric" is the paper's most fundamental contribution — remarkably concise yet highly effective.
- Using target image prediction as an auxiliary task to align language, objects, and actions is an elegant design choice that avoids the difficulty of directly predicting object optical flow.
- The entire system requires only RGB video and a URDF file, with no action annotations, 3D point clouds, or specialized sensors, resulting in extremely low deployment cost.

## Limitations & Future Work

- The current system requires manual specification of the manipulation starting pose; automatic grasp pose generation has not been integrated.
- Gripper state (open/close) is borrowed from object-centric methods and still depends on object segmentation.
- DDIM 250-step sampling leads to slow inference (4.37s for flow prediction); flow matching could be used to accelerate this.
- Depth perception relies on a D435i camera, limiting accuracy on tasks involving complex deformations (e.g., folding clothes).

## Related Work & Insights

- Distinction from General Flow: General Flow requires RGBD input to predict 3D flow fields, whereas EC-Flow uses RGB only.
- Distinction from MT-π: MT-π relies on 5 predefined gripper keypoints that must always be visible; EC-Flow is robust to partial occlusion.
- Insight: The embodiment-centric approach may generalize to more complex embodied scenarios such as dual-arm collaboration and dexterous hand manipulation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The paradigm shift to embodiment-centric optical flow is highly elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Simulation + real-world, thorough ablations, though real-world experiments are limited in scale)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, detailed method description)
- Value: ⭐⭐⭐⭐⭐ (Significantly lowers the barrier for learning manipulation from video)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[NeurIPS 2025\] Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems](../../NeurIPS2025/image_generation/equivariant_flow_matching_for_symmetry-breaking_bifurcation_problems.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](../../CVPR2026/image_generation/image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](../../NeurIPS2025/image_generation/high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)

</div>

<!-- RELATED:END -->
