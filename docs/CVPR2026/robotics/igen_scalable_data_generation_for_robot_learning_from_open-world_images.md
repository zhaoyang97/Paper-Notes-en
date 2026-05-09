---
title: >-
  [Paper Note] IGen: Scalable Data Generation for Robot Learning from Open-World Images
description: >-
  [CVPR 2026][Robotics][Robot learning] IGen starts from a single open-world image and automatically generates large-scale vision-action training data through a pipeline of 3D scene reconstruction → VLM task planning → SE(3) action generation → point cloud synthesis → frame rendering. Policies trained exclusively on the generated data can successfully perform real-world manipulation tasks.
tags:
  - CVPR 2026
  - Robotics
  - Robot learning
  - data generation
  - open-world images
  - visuomotor policy
  - 3D reconstruction
date: 2026-05-08
content_hash: d21acb2b4859fd5a
---

# IGen: Scalable Data Generation for Robot Learning from Open-World Images

**Conference**: CVPR 2026
**arXiv**: [2512.01773](https://arxiv.org/abs/2512.01773)
**Code**: [https://chenghaogu.github.io/IGen/](https://chenghaogu.github.io/IGen/)
**Area**: Robotics
**Keywords**: Robot learning, data generation, open-world images, visuomotor policy, 3D reconstruction

## TL;DR
IGen starts from a single open-world image and automatically generates large-scale vision-action training data through a pipeline of 3D scene reconstruction → VLM task planning → SE(3) action generation → point cloud synthesis → frame rendering. Policies trained exclusively on the generated data can successfully perform real-world manipulation tasks.

## Background & Motivation
1. **State of the Field**: Generalizable robot policies require large-scale vision-action paired data, yet real-world data collection is costly and constrained to specific environments.
2. **Limitations of Prior Work**: Real-to-Sim approaches require explicit reconstruction of physical workspaces; video generation methods cannot provide explicit actions and perform poorly on complex long-horizon tasks.
3. **Root Cause**: Open-world images are abundantly diverse but lack robot-relevant action information, making them unsuitable for direct policy learning.
4. **Paper Goals**: Automatically generate grounded vision-action data from unstructured open-world images.
5. **Starting Point**: Convert 2D pixels into structured 3D representations, then leverage VLMs for task planning and action generation.
6. **Core Idea**: Image → 3D point cloud + keypoints → VLM high-level planning + low-level control → SE(3) trajectory → point cloud sequence synthesis → frame rendering.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Scene Reconstruction** — transforms the input image into a manipulable robot workspace (3D point cloud + spatial keypoints); (2) **Action Planning** — a VLM reasons over task instructions to produce high-level plans and low-level control; (3) **Observation Synthesis** — synthesizes dynamic point cloud sequences along SE(3) trajectories and renders them frame by frame.

### Key Designs

1. **From Pixels to Structured 3D Representations**
    - **Function**: Transform unstructured 2D images into 3D representations interpretable by a robot.
    - **Mechanism**: A monocular geometric foundation model estimates depth; a VLM identifies task-relevant objects; SAM segments object masks; DINOv2 extracts features combined with K-means clustering to obtain spatial keypoints. Manipulation target objects are reconstructed with complete shapes via 3D generative models, while backgrounds are back-projected into point clouds using inpainting and depth estimation.
    - **Design Motivation**: Robot manipulation requires 3D spatial understanding; planning directly in 2D image space cannot yield physically grounded actions.

2. **VLM-Based Spatial Planning**
    - **Function**: Generate executable robot action sequences from task instructions.
    - **Mechanism**: The VLM translates scene understanding and task descriptions into high-level plans (e.g., "grasp → move → place"), which are then mapped to low-level SE(3) end-effector pose sequences. Spatial keypoints serve as spatial anchors for the actions.
    - **Design Motivation**: VLMs possess strong scene understanding and reasoning capabilities, enabling grounding of natural language instructions to concrete operations in 3D space.

3. **Simulation-Free Point Cloud Synthesis**
    - **Function**: Generate action-consistent visual observation sequences.
    - **Mechanism**: Rigid-body transformations are applied to the scene point cloud along SE(3) trajectories to produce dynamic point cloud sequences throughout the manipulation process, which are then rendered into RGB observations frame by frame. This avoids the overhead of constructing a full physics simulation environment.
    - **Design Motivation**: Rigid-body synthesis on point clouds is substantially more lightweight than physics simulators and imposes more relaxed requirements on rendering quality.

### Loss & Training
Standard imitation learning policies (e.g., ACT, DP3) are trained on the generated vision-action data using a standard behavior cloning loss.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | IGen | TesserAct | Cosmos | Notes |
|---|---|---|---|---|---|
| Visual Fidelity | Consistency Score | High | Medium | Low | Closer to real-world appearance |
| Action Quality | Instruction Following + Physical Alignment | Best | Second | Poor | More physically plausible actions |
| Policy Transfer | Real-world Task Success Rate | Comparable / superior to real data | — | — | Purely generated data is effective |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Full IGen | Best | Complete pipeline |
| w/o 3D Reconstruction | Significant drop | 3D understanding is foundational |
| w/o Spatial Keypoints | Drop | Keypoints provide spatial grounding |
| 2D Generation Substitute | Drop | 2D methods lack physical grounding |

### Key Findings
- Policies trained solely on IGen-generated data can successfully execute manipulation tasks in the real world without any real collected data.
- In certain scenarios, policies trained on IGen-generated data even surpass those trained on real data, possibly due to greater scene diversity.
- Compared with video generation methods, IGen produces actions that are more physically consistent and exhibit higher instruction-following accuracy.

## Highlights & Insights
- The paradigm of **"images as data sources"** is highly compelling: internet images constitute the richest available visual resource.
- **Simulation-free point cloud synthesis** eliminates the most time-consuming step in traditional Real-to-Sim pipelines — constructing a physics simulation environment.
- The combination of 3D representation and VLM planning provides physical grounding while remaining lightweight.

## Limitations & Future Work
- Relies on monocular depth estimation, whose accuracy is bounded by the estimation model.
- The rigid-body motion assumption limits modeling of deformable or fluid object manipulation.
- Modeling of complex physical interactions (e.g., contact force feedback) remains insufficient.

## Related Work & Insights
- **vs. RoLA**: RoLA also generates data from open-world images but relies on physical property estimation and is limited to simple interactions. IGen supports more complex tasks through VLM-based planning.
- **vs. TesserAct / Cosmos**: Video generation-based methods lack explicit actions; IGen provides complete vision-action pairs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — A complete pipeline from open-world images to robot training data is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three-dimensional evaluation combined with real-world validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Pipeline description is clear with thorough comparisons.
- **Value**: ⭐⭐⭐⭐⭐ — Has the potential to fundamentally transform robot data acquisition.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](../../NeurIPS2025/robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[AAAI 2026\] Realistic Synthetic Household Data Generation at Scale](../../AAAI2026/robotics/realistic_synthetic_household_data_generation_at_scale.md)
- [\[CVPR 2026\] Chain of World: World Model Thinking in Latent Motion (CoWVLA)](chain_of_world_world_model_thinking_in_latent_motion.md)

<!-- RELATED:END -->
