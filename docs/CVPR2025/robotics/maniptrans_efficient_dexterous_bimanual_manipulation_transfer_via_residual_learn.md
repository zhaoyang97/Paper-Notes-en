---
title: >-
  [Paper Note] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning
description: >-
  [CVPR 2025][Robotics][Bimanual Dexterous Manipulation] This work proposes ManipTrans, a two-stage residual learning framework that transfers human motion capture data to bimanual dexterous hand manipulation: Stage-1 pre-trains an imitation model on pure hand trajectories (wrist + finger tracking + smoothness rewards), and Stage-2 incorporates object interaction constraints (object tracking + contact forces) via a residual module and curriculum learning…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Bimanual Dexterous Manipulation"
  - "Residual Learning"
  - "Motion Capture Transfer"
  - "Curriculum Learning"
  - "Contact Force"
date: 2026-05-08
content_hash: 4ee40fbb2f5a8437
---

# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.21860](https://arxiv.org/abs/2503.21860)  
**Code**: To be released  
**Area**: Reinforcement Learning  
**Keywords**: Bimanual Dexterous Manipulation, Residual Learning, Motion Capture Transfer, Curriculum Learning, Contact Force

## TL;DR

This work proposes ManipTrans, a two-stage residual learning framework that transfers human motion capture data to bimanual dexterous hand manipulation: Stage-1 pre-trains an imitation model on pure hand trajectories (wrist + finger tracking + smoothness rewards), and Stage-2 incorporates object interaction constraints (object tracking + contact forces) via a residual module and curriculum learning, achieving an object rotation error of only 8.60° and a bimanual success rate of 39.5% on OakInk-V2.

## Background & Motivation

### Background

**Background**: Dexterous manipulation (such as grasping/rotating objects) is a core challenge in robotics. Motion capture data provides rich demonstrations of human hand manipulation, but the human hand possesses 27 degrees of freedom (DoFs) while dexterous hands have different kinematic structures—making direct retargeting unable to guarantee physically valid interactions.

**Limitations of Prior Work**: (1) Physical simulation methods, such as QuasiSim, require 40+ hours to optimize a single trajectory; (2) directly training reinforcement learning (RL) for dexterous hand manipulation requires task-specific reward designs, and the bimanual version suffers from a dimensional explosion; (3) simple combinations of retargeting and RL residuals struggle to converge due to the complex action space.

**Key Challenge**: Trajectory imitation (focusing solely on hand movement without considering objects) is straightforward but cannot guarantee successful object interaction; interaction learning (object tracking + contact) is challenging but remains the ultimate goal. Learning both simultaneously leads to an excessively high dimensionality.

**Key Insight**: Decoupling—first learn the hand's motion patterns (Stage-1, without objects), then utilize a residual network to learn only the "corrections required for object interaction" (Stage-2). The action space of the residual module is significantly smaller.

**Core Idea**: Hand motion pre-training + object interaction residual + curriculum learning = efficient transfer of bimanual dexterous manipulation.

### Proposed Approach

**Goal**: ### Key Designs

1. **Stage-1: Pure-Hand Trajectory Imitation**: Use RL to train a policy to imitate the wrist pose and finger joint angles from motion capture, without involving the object.


## Method

### Key Designs

1. **Stage-1: Pure-Hand Trajectory Imitation**: Uses RL to train policies to imitate wrist poses and finger joint angles from motion capture, without involving objects. The reward includes wrist tracking, finger tracking, and smoothness terms.

2. **Stage-2: Residual Interaction Learning**: Freezes the Stage-1 policy and trains a residual module to add corrective actions. Added rewards: object pose tracking + contact force reward $r_{contact} = w_c \exp(-\lambda_c / \sum_f C_f \cdot \mathbb{1}(D < \xi_c))$ + contact termination conditions. Curriculum learning is applied to progressively tighten the tolerance for finger and object tracking.

3. **DexManipNet Dataset**: 3.3K episodes, 1.34M frames, 1.2K objects, 61 tasks (including new bimanual tasks).

### Loss & Training

Hand imitation reward: $r_\mathcal{I} = w_{wrist}r_{wrist} + w_{finger}r_{finger} + w_{smooth}r_{smooth}$. The finger reward utilizes Gaussian decay. Training takes approximately 15 minutes per new trajectory (vs. 40+ hours for QuasiSim).

## Key Experimental Results

| Method | Object Rotation Error ↓ | Object Translation Error ↓ | Bimanual Success Rate ↑ |
|------|------------|------------|-----------|
| Retarget+Residual | 11.58° | 0.79cm | 13.9% |
| RL-only | 9.72° | 1.23cm | — |
| **ManipTrans** | **8.60°** | **0.49cm** | **39.5%** |

### Ablation Study
- Contact force as observation input: Accelerates convergence.
- Contact force reward: Crucial for the success rate of contact-intensive tasks.
- Curriculum learning (gradually tightening tolerances): Prevents network collapse.
- Gravity relaxation + high-friction initialization: Necessary conditions for early training.

### Key Findings
- Residual learning is ~160× more efficient than end-to-end RL (15 minutes vs. 40 hours).
- Bimanual manipulation achieves a success rate of 39.5% (vs. 13.9%)—residual decoupling significantly reduces the difficulty of learning bimanual coordination.
- Contact termination conditions ensure stable grasping.

## Highlights & Insights
- **Core insight of two-stage decoupling**—Hand motion patterns and object interaction corrections are learning objectives at two distinct levels.
- **15 minutes vs 40 hours**—An efficiency improvement of two orders of magnitude.

## Limitations & Future Work
- Some motion capture sequences cannot be transferred due to excessive noise.
- Direct transfer from a simulated 12-DoF hand to a real 6-DoF hand requires additional fingertip adaptation.
- Limited to manipulation tasks; not applicable to locomotion.

## Rating
- Novelty: ⭐⭐⭐⭐ Efficient application of residual learning in dexterous manipulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ New dataset + quantitative evaluation + real robot + bimanual.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Provides an efficient solution for transferring motion capture to dexterous manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RFS: Reinforcement Learning with Residual Flow Steering for Dexterous Manipulation](../../ICLR2026/robotics/rfs_reinforcement_learning_with_residual_flow_steering_for_dexterous_manipulatio.md)
- [\[AAAI 2026\] Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment](../../AAAI2026/robotics/dexterous_manipulation_transfer_via_progressive_kinematic-dynamic_alignment.md)
- [\[CVPR 2025\] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)

</div>

<!-- RELATED:END -->
