---
title: >-
  [Paper Note] RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins
description: >-
  [CVPR 2025 (Highlight)][Robotics][Dual-arm robots] RoboTwin proposes a dual-arm robot benchmark framework based on generative digital twins. It leverages 3D generative foundation models to reconstruct 3D digital twins of objects from single 2D images and combines them with Large Language Models to automatically generate robot manipulation code. Under the paradigm of simulation pre-training followed by real-world fine-tuning with sparse data, it achieves significant success ra…
tags:
  - "CVPR 2025 (Highlight)"
  - "Robotics"
  - "Dual-arm robots"
  - "digital twins"
  - "3D generation"
  - "Large Language Models"
  - "simulation benchmark"
date: 2026-05-08
content_hash: 4bc6056423993dcc
---

# RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2504.13059](https://arxiv.org/abs/2504.13059)  
**Code**: [https://github.com/robotwin-Platform/RoboTwin](https://github.com/robotwin-Platform/RoboTwin)  
**Area**: Robotics  
**Keywords**: Dual-arm robots, digital twins, 3D generation, Large Language Models, simulation benchmark

## TL;DR

RoboTwin proposes a dual-arm robot benchmark framework based on generative digital twins. It leverages 3D generative foundation models to reconstruct 3D digital twins of objects from single 2D images and combines them with Large Language Models to automatically generate robot manipulation code. Under the paradigm of simulation pre-training followed by real-world fine-tuning with sparse data, it achieves significant success rate gains of over 70% in single-arm tasks and over 40% in dual-arm tasks.

## Background & Motivation

**Background**: Dual-arm coordination and complex object manipulation are core capabilities for building advanced autonomous robotic systems. Current robot learning policies (such as imitation learning and reinforcement learning) rely heavily on large-scale, high-quality demonstration data and real-world-aligned evaluation benchmarks for training and validation.

**Limitations of Prior Work**: First, acquiring diverse, high-quality dual-arm manipulation demonstration data is extremely costly, requiring physical robots to execute tasks repeatedly in real environments and record them. Second, object models and scenes in existing simulation environments lack diversity, making it difficult to cover the long-tail distribution of the real world. Finally, there is a prominent sim-to-real gap between simulation training and real deployment, as the visual appearance and physical interactions in simulation are often inconsistent with the real world.

**Key Challenge**: Constructing diverse simulation datasets requires a massive number of 3D object models and manipulation programs, but manual modeling and programming incur enormous overhead. Conversely, if the simulation environment lacks diversity and realism, the trained policies cannot transfer effectively to the real world.

**Goal**: (1) Automatically generate diverse 3D object digital twins; (2) Automatically generate robot manipulation code; (3) Provide a comprehensive evaluation benchmark with simulation-to-real alignment.

**Key Insight**: The authors observe the rapid development of 3D generative foundation models (e.g., generating 3D objects from a single image) and Large Language Models (which possess code generation and spatial reasoning capabilities). These two technologies can respectively automate object modeling and manipulation programming.

**Core Idea**: By combining 3D generative models and LLMs, interactive digital twin scenes are automatically created from single 2D images, and robot manipulation trajectories are automatically generated via a spatial-relation-aware code generation framework. This enables the low-cost construction of large-scale, diverse benchmark data for dual-arm manipulation.

## Method

### Overall Architecture

The pipeline of RoboTwin consists of three core phases: (1) Digital Twin Generation—reconstructing 3D digital twins of objects from single 2D images using 3D generative foundation models and deploying them into the simulator; (2) Manipulation Code Generation—utilizing LLMs combined with object spatial annotations to automatically decompose tasks, determine spatial constraints, and generate precise robot motion code; (3) Policy Training & Evaluation—pre-training manipulation policies on the generated simulation data, followed by fine-tuning and evaluation on the physical COBOT Magic Robot platform.

### Key Designs

1. **Automatic Digital Twin Generation**:

    - Function: Automatically creates interactive 3D models from a single 2D object image.
    - Mechanism: Utilizes 3D generative foundation models (such as diffusion-based single-image-to-3D reconstruction methods) to convert object photos into 3D mesh models, and then automatically appends physical attributes (mass, friction, colliders, etc.) to enable physical interaction in the simulator. By swapping out different object images, it can rapidly generate large sets of digital twin objects with significant visual variations, covering real-world object diversity.
    - Design Motivation: Traditional approaches require manual 3D modeling for each object, which is expensive and difficult to scale. Utilizing off-the-shelf 3D generative models allows automatic digital twin creation from just a single photo, dramatically reducing dataset construction costs.

2. **Spatial-Relation-Aware Code Generation**:

    - Function: Automatically translates task descriptions into executable robot manipulation code.
    - Mechanism: It first performs spatial annotation of objects in the scene (positions, orientations, keypoints, etc.) and feeds these annotations as context to the LLM. The LLM performs multi-step reasoning: (a) Task decomposition—breaking down high-level tasks into sequences of sub-tasks (e.g., "grasp the cup with the left hand first, then turn on the faucet with the right hand"); (b) Spatial constraint determination—inferring spatial constraints for each sub-task based on object annotations (grasp poses, target placement positions, collision-free paths, etc.); (c) Motion code generation—generating exact robot end-effector waypoints and gripper action code.
    - Design Motivation: Manually writing dual-arm coordinated manipulation code is highly complex, especially when spatial relation reasoning (such as relative object positions and collision avoidance) is involved. LLMs naturally possess semantic understanding and code generation capabilities, which, when coupled with spatial annotations, can automate this process.

3. **Simulation-to-Real Aligned Evaluation System**:

    - Function: Provides standardized evaluation supporting both simulation and real-world environments.
    - Mechanism: Built upon the open-source COBOT Magic Robot dual-arm platform, it constructs various dual-arm collaborative tasks (such as collaborative transport, tool use, and fine manipulation), with corresponding versions in both simulation and real environments. The evaluation protocol unifies task success criteria, allowing the policies' simulation performance to be compared fairly with real-world deployments.
    - Design Motivation: Previous robotic simulation benchmarks often focused solely on simulation performance and lacked direct real-world comparisons, making it difficult to evaluate sim-to-real transfer.

### Loss & Training

A two-stage paradigm of "simulation pre-training + real-world fine-tuning" is adopted: manipulation policies (such as ACT, Diffusion Policy, etc.) are first trained on the massive simulation data generated by RoboTwin, and then fine-tuned using a small amount of real-world data. This approach leverages the scale advantage of simulation data while mitigating the sim-to-real gap through fine-tuning.

## Key Experimental Results

### Main Results

| Task Type | Metric | Sim Pre-training + FT | Real-only Training | Gain |
|----------|------|----------------|---------------|------|
| Single-arm Task | Success Rate | Significant Improvement | Baseline | >70% |
| Dual-arm Task | Success Rate | Significant Improvement | Baseline | >40% |

The framework demonstrates significant performance improvements across multiple dual-arm tasks, indicating that the data generated via digital twins is of sufficient quality to support effective policy learning. Particularly in dual-arm collaborative tasks, the generalization improvements brought by data diversity are highly pronounced.

### Ablation Study

| Configuration | Performance Change | Description |
|------|---------|------|
| Real-only (No Pre-training) | Baseline | Poor performance due to limited data volume |
| Sim Pre-training + Real FT | Substantial Improvement | Digital twin data provides key priors |
| Different 3D Generative Models | Minimal Impact | Indicates the framework is robust to the choice of generative models |
| Varying Real FT Data Volume | Gradual Improvement | A small amount of real data is sufficient to narrow the gap significantly |

### Key Findings

- Diversity in simulation data is more important than raw volume—replacing object appearances via digital twins effectively enhances policy generalization.
- The quality of LLM-generated manipulation code is high, and when augmented with spatial-relation annotations, it can correctly handle most dual-arm collaboration scenarios.
- In version 2.0, RoboTwin has been further expanded to 50 dual-arm tasks, 731 objects, and 5 robot morphologies, validating the scalability of the framework.

## Highlights & Insights

- **The combination of 3D Generation and LLMs** is the prominent highlight of this work: it elegantly maps the capabilities of two rapidly developing foundation models to the bottleneck of robot data generation, establishing a fully automated pipeline from 2D photos to executable simulation environments.
- **Spatial-relation-aware** design is highly practical: instead of expecting the LLM to generate robot actions "out of thin air," it provides precise spatial annotations as grounding, greatly improving the reliability of code generation.
- The methodology behind this framework can be generalized to other embodied AI tasks: tasks such as navigation, grasping, and assembly can also adopt similar "generative digital twin + LLM programming" paradigms to build training data rapidly.

## Limitations & Future Work

- 3D generative models currently have limited support for complex deformable objects (such as fabric or ropes), meaning digital twins are mostly restricted to rigid objects.
- LLM-generated manipulation code might lack precision for high-accuracy tasks (e.g., screw assembly) and may still require manual verification.
- Although the sim-to-real gap is mitigated by fine-tuning, it might still fail in scenarios with huge visual domain shifts (e.g., dramatic lighting changes or occlusions).
- Future directions include: introducing stronger domain randomization (already explored in RoboTwin 2.0), supporting deformable object manipulation, and integrating Vision-Language-Action (VLA) models.

## Related Work & Insights

- **vs RLBench / VIMA**: These benchmarks predominantly focus on single-arm tasks, whereas RoboTwin targets more complex dual-arm collaboration and provides superior object diversity via digital twins.
- **vs ManiSkill / IsaacGym**: These simulation platforms provide general-purpose manipulation environments but lack automated scene generation capabilities. RoboTwin's contribution lies in its end-to-end data generation pipeline.
- **vs SayCan / Code as Policies**: These works also use LLMs for robot task planning and code generation, but RoboTwin places a greater focus on precise spatial relation modeling and the large-scale generation of digital twin data.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying the combination of 3D generation and LLMs to robot data generation is novel, though the individual technologies themselves are not first-of-their-kind.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation on both simulation and real-world platforms is compelling, though the ablation studies could be more elaborate.
- Writing Quality: ⭐⭐⭐⭐ Structured clearly with intuitive pipeline descriptions, reflecting the high standard of a Highlight paper.
- Value: ⭐⭐⭐⭐⭐ The open-source framework and benchmark hold high practical value for the dual-arm robotics community, having already been widely cited and expanded to version 2.0.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](../../ICLR2026/robotics/robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)
- [\[ICLR 2026\] AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory](../../ICLR2026/robotics/autobio_a_simulation_and_benchmark_for_robotic_automation_in_digital_biology_lab.md)
- [\[CVPR 2025\] DRAWER: Digital Reconstruction and Articulation with Environment Realism](drawer_digital_reconstruction_and_articulation_with_environment_realism.md)
- [\[CVPR 2025\] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)
- [\[ICLR 2026\] Ctrl-World: A Controllable Generative World Model for Robot Manipulation](../../ICLR2026/robotics/ctrl-world_a_controllable_generative_world_model_for_robot_manipulation.md)

</div>

<!-- RELATED:END -->
