---
title: >-
  [Paper Note] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation
description: >-
  [CVPR 2025][Robotics][Vision-Language-Action Models] Proposes MoManipVLA to transfer pre-trained fixed-base VLA models to mobile manipulation scenarios. By jointly planning base movement and manipulator trajectories using a bi-level trajectory optimization (optimizing reachability, smoothness, and collision avoidance), it achieves a 66.1% success rate (+4.2%) on the OVMM benchmark and can be deployed in the real world with only 50 demonstrations.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Vision-Language-Action Models"
  - "Mobile Manipulation"
  - "Bi-level Trajectory Optimization"
  - "VLA Transfer"
  - "Reachability"
date: 2026-05-08
content_hash: 1275de722fad141c
---

# MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2503.13446](https://arxiv.org/abs/2503.13446)  
**Code**: Yes (Project Page)  
**Area**: Robotics / Mobile Manipulation  
**Keywords**: Vision-Language-Action Models, Mobile Manipulation, Bi-level Trajectory Optimization, VLA Transfer, Reachability

## TL;DR

Proposes MoManipVLA to transfer pre-trained fixed-base VLA models to mobile manipulation scenarios. By jointly planning base movement and manipulator trajectories using a bi-level trajectory optimization (optimizing reachability, smoothness, and collision avoidance), it achieves a 66.1% success rate (+4.2%) on the OVMM benchmark and can be deployed in the real world with only 50 demonstrations.

## Background & Motivation

### Background

**Background**: VLA models (such as RT-2, Octo) perform excellently in fixed-base manipulation, but mobile manipulation requires coordinated planning of both the base and the manipulator—the base must move to an appropriate position before the arm can reach the target.

**Limitations of Prior Work**: Directly applying end-effector trajectories from fixed-base VLAs to mobile robots fails because the target position may exceed the reach of the arm length from the current base position. Additional base motion planning is required.

**Key Challenge**: VLAs output end-effector waypoints without including base motion information. Training a mobile manipulation VLA from scratch is extremely data-costly.

**Key Insight**: Keep the VLA unchanged and add a bi-level optimizer on top of its output waypoints: the inner loop solves for joint angles using IK, while the outer loop optimizes the base position to maximize reachability, maximize smoothness, and minimize collisions.

**Core Idea**: VLA-generated waypoints + bi-level trajectory optimization (reachability/smoothness/collision) = low-cost mobile manipulation transfer.

## Method

### Key Designs

1. **Bi-level Trajectory Optimization**:

    - **Function**: Jointly plans base movement and manipulator trajectories.
    - **Mechanism**: The outer loop optimizes the base position sequence $\mathbf{x}_b$ and arm joint angles $\boldsymbol{\theta}$, with the objective function $\mathcal{O} = 10\mathcal{F}_r + 1\mathcal{F}_s + 0.6\mathcal{F}_c$, where $\mathcal{F}_r$ is the IK reachability (imposes a large constant penalty if unreachable), $\mathcal{F}_s$ is smoothness (first-order difference of base and joint angles), and $\mathcal{F}_c$ is collision (signed distance function to obstacles).
    - **Design Motivation**: Reachability has the highest weight (10) because the primary challenge of mobile manipulation is "whether it can be reached".

### Loss & Training

The VLA is fine-tuned using a small amount of embodiment-specific data. Base optimization requires no learning—it is solved directly during inference. Only 50 expert demonstrations are required for real-world deployment.

## Key Experimental Results

### Main Results

| Method | OVMM Overall Success Rate | Pick Success Rate |
|------|-------------|-----------|
| SOTA Baseline | 61.9% | 50.2% |
| **MoManipVLA** | **66.1%** | **62.6%** |

### Key Findings
- Reachability is the most critical constraint: removing it drops the success rate from 66.1% to 48.2%.
- Only 50 demonstrations are needed to achieve a 40% success rate in the real world.
- Inference latency is 693ms (vs. 742ms for direct optimization).
- Smoothness constraint contributes significantly: removing it increases trajectory jitter and reduces the success rate to 56.3%.
- Collision avoidance constraint plays an obvious role in cluttered scenarios; without this constraint, the success rate drops to 60.1%.

## Highlights & Insights
- **Zero-data base planning**—no mobile manipulation data is required, as the base trajectory is solved by the optimizer during inference.
- **VLA reuse strategy**—the VLA is not retrained, and only a planning optimization layer is added.
- The decoupled design fully reuses the generalization capability of the VLA, avoiding the expensive data collection costs of mobile manipulation.
- The bi-level optimization framework is elegantly designed, with the outer loop optimizing the base position and the inner loop solving joint angles, offering clear logic and scalability.

## Limitations & Future Work
- Relies on the quality of visual segmentation masks (success rate drops to 23.7% without masks).
- Trajectory length is limited to < 150 steps.
- Requires embodiment-specific fine-tuning.
- Computational overhead of the optimizer solving base trajectories during inference may become a bottleneck in complex environments.
- Only validated in tabletop manipulation scenarios; larger-scale mobile manipulation (such as household navigation + retrieval) is not covered.
- The perception range of the VLA model is limited by a single camera view; multi-camera fusion might further improve performance.
- For tasks requiring fine manipulation (e.g., insertion, tightening), waypoint-level planning may lack sufficient precision.
- Currently, the objective function weights (10:1:0.6) of the optimizer need to be adjusted for different robot platforms.
- Multi-arm or bimanual mobile manipulation scenarios have not been explored.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Practical decoupled design of VLA + trajectory optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Simulation + Real-world.
- **Writing Quality**: ⭐⭐⭐⭐ Clear.
- **Value**: ⭐⭐⭐⭐ Low-cost transfer of VLA to mobile manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation](../../ICCV2025/robotics/anybimanual_transferring_unimanual_policy_for_general_bimanual_manipulation.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)
- [\[CVPR 2025\] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)

</div>

<!-- RELATED:END -->
