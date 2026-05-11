---
title: >-
  [Paper Note] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation
description: >-
  [CVPR 2026][Robotics][Robot manipulation evaluation] ManipArena proposes a standardized real-world robot manipulation evaluation framework comprising 20 reasoning-oriented tasks and 10…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Robot manipulation evaluation"
  - "VLA models"
  - "real-world benchmark"
  - "reasoning-oriented manipulation"
  - "Sim-to-Real"
date: 2026-05-08
content_hash: 27ba9f249c3d353d
---

# ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation

**Conference**: CVPR 2026  
**arXiv**: [2603.28545](https://arxiv.org/abs/2603.28545)  
**Code**: [https://github.com/maniparena/maniparena-repo](https://github.com/maniparena/maniparena-repo)  
**Area**: Robot Manipulation / Benchmark  
**Keywords**: Robot manipulation evaluation, VLA models, real-world benchmark, reasoning-oriented manipulation, Sim-to-Real

## TL;DR

ManipArena proposes a standardized real-world robot manipulation evaluation framework comprising 20 reasoning-oriented tasks and 10,812 expert trajectories. Through a green-screen controlled environment, systematic diversity design, and hierarchical OOD evaluation, it provides a fair and reproducible benchmark for VLA models and world models.

## Background & Motivation

1. **Background**: VLA (Vision-Language-Action) models and world models represent two dominant paradigms in generalist robot intelligence, demonstrating promise in manipulation, mobile manipulation, and long-horizon tasks.

2. **Limitations of Prior Work**: Existing evaluations are heavily concentrated in simulation environments (RLBench, LIBERO, CALVIN, etc.), which offer controllability and reproducibility but fail to capture the "reality gap" introduced by perceptual noise, complex contact dynamics, system latency, and hardware constraints in real deployment. Meanwhile, real-world evaluations remain highly fragmented—different researchers use different robot platforms and environments, making cross-study comparisons unfair and difficult to reproduce.

3. **Key Challenge**: Simulation success rates are unreliable predictors of real-world performance, while existing real-world evaluations lack standardized protocols.

4. **Goal**: To construct a standardized evaluation framework bridging simulation and real execution, supporting fair and reproducible assessment of reasoning-intensive manipulation tasks.

5. **Key Insight**: Five core design principles—reasoning-oriented tasks, multi-level generalization, mobile manipulation, rich sensory diagnostics, and Real2Sim synchronization.

6. **Core Idea**: Combine a green-screen controlled environment, systematic diversity design, and hierarchical OOD evaluation to build the first standardized real-world benchmark for reasoning-oriented robot manipulation.

## Method

### Overall Architecture

ManipArena consists of the following components:
- **Input**: Multi-view camera images (front + two wrist cameras) + proprioceptive state (56D/62D)
- **Evaluation Architecture**: Server-side inference—participants only need to expose an HTTP endpoint that receives observation data and returns action commands
- **Key Constraint**: One model for all tasks; training separate expert models per task is prohibited
- **Task Taxonomy**: 20 tasks divided into three categories—Execution Reasoning (10), Semantic Reasoning (5), and Mobile Manipulation (5)
- **Dataset**: 10,812 teleoperated trajectories, approximately 188 hours

### Key Designs

1. **Green-Screen Controlled Evaluation Environment**:

    - Function: Eliminates uncontrolled visual variation, enabling performance differences to be attributed to specific generalization axes
    - Mechanism: All evaluations are conducted within self-contained green-screen enclosures with uniform chroma backgrounds and fixed artificial lighting (constant color temperature and intensity). Object and spatial variation thus become the sole factors driving performance differences, transforming the benchmark from a black-box leaderboard into a controlled experiment.
    - Design Motivation: In open-environment evaluations, background lighting, furniture positions, and other factors vary simultaneously, making it impossible to attribute performance differences. The green screen also enables future visual robustness research by compositing synthetic natural scene backgrounds.

2. **Systematic Diversity Design (Three Levels)**:

    - Function: Ensures high scores reflect genuine generalization ability rather than memorization of training data
    - Mechanism: Each task is accompanied by a diversity guide defining required object variants, color sets, and spatial configurations. Three levels are defined: Level 1—physical attribute diversity (material/color/size); Level 2—spatial configuration diversity (position/orientation randomization); Level 3—semantic compositional diversity (different object combinations and arrangements). Training data is distributed uniformly across all dimensions (±10–15%). Training and test objects are strictly separated—OOD test objects never appear in training data.
    - Design Motivation: Without diversified training data, OOD evaluation measures interpolation rather than true extrapolation.

3. **Hierarchical OOD Evaluation Design**:

    - Function: Yields a complete generalization profile within a single evaluation session
    - Mechanism: Each task's 10 trials are stratified by increasing difficulty: T1–T4 assess in-distribution capability (objects within the training distribution); T5–T8 introduce visual shift (appearance variation); T9–T10 use semantically OOD objects never seen during training. For example, in `put_spoon_to_bowl`, T1–T4 use stainless steel spoons, T5–T8 use children's spoons (different shape), and T9–T10 use black plastic spoons (novel material and color).
    - Design Motivation: A single evaluation session yields comparisons across three generalization levels, and degradation curves can be computed directly without separate experiments.

### Loss & Training

ManipArena is an evaluation framework rather than a training method. Scoring uses a partial-credit scheme: each task is decomposed into ordered sub-tasks, so completing 7 out of 10 sub-tasks yields 7 points rather than 0. Each task has a maximum score of 100 points (10 trials × 10 points), with a total of 1,500 points across 15 tabletop tasks.

## Key Experimental Results

### Main Results

| Feature | ManipArena | RLBench | LIBERO | CALVIN | VLABench | RoboArena |
|---------|-----------|---------|--------|--------|----------|-----------|
| Environment | Real | Sim | Sim | Sim | Sim | Real |
| Reasoning Demand | High | Low | Low | Medium | High | Medium |
| Generalization | Systematic | Limited | Moderate | Moderate | Strong | Weak |
| Mobile Manipulation | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Sensory Diagnostics | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Real2Sim | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

### Dataset Statistics

| Task Category | # Tasks | # Trajectories | Avg. Frames | Avg. Duration |
|--------------|---------|----------------|-------------|---------------|
| Execution Reasoning | 10 | 5,157 | 784 | 39.2s |
| Semantic Reasoning | 5 | 2,783 | 499 | 25.0s |
| Mobile Manipulation | 5 | 2,872 | 2,878 | 143.9s |
| Total | 20 | 10,812 | — | — |

### Key Findings

- Mobile manipulation tasks are on average 4.3× longer than tabletop tasks (143.9s vs. 39.2s/25.0s), accounting for 60.6% of total frames but only 26.7% of trajectories
- Semantic reasoning tasks, despite requiring higher cognitive complexity, have the shortest episodes (25.0s)—once semantic ambiguity is resolved, the physical manipulation itself is relatively simple
- The long-horizon structure of mobile manipulation poses particular challenges for VLA architectures with fixed context windows
- Sensory data provides 56D state (tabletop) / 62D (mobile), including motor current and joint velocity, far exceeding the standard LeRobot format
- The three pillars—green-screen environment, systematic diversity, and hierarchical OOD evaluation—together constitute a complete, controlled generalization measurement framework

## Highlights & Insights

- **Server-Side Inference Architecture**: Participants only need to expose an HTTP endpoint without requiring specific hardware, lowering the barrier to participation while ensuring fairness and IP protection. This design pattern is transferable to other hardware-intensive benchmark evaluations.
- **One-Model-for-All-Tasks Rule**: Requiring a single model to solve all tasks prevents task-specific overfitting and genuinely tests generalization. This design philosophy carries important implications for benchmark design.
- **Green Screen and Future Extensibility**: The green screen is not merely a practical convenience—it opens the door to systematic visual robustness research by enabling independent testing of visual transfer capabilities through composited or projected backgrounds.
- **Motor Current as Torque Proxy**: Low-level sensory signals (motor current, joint velocity) are provided to encourage research into force-aware policies.

## Limitations & Future Work

- **Single Robot Platform**: All tasks use the X2Robot dual-arm system, which eliminates embodiment variability but limits evaluation of cross-platform generalization
- **Tabletop Task Dominance**: 15 of the scored tasks are tabletop tasks; mobile manipulation tasks are included but underrepresented
- **Absence of Baseline Model Results**: The paper primarily describes the framework design without presenting detailed performance results of existing VLA models on the benchmark
- **High Data Collection Cost**: Collecting approximately 500 trajectories per task by expert operators following diversity guides is difficult to scale
- **Limited Dynamic Interaction**: All evaluations are non-reactive, and the benchmark does not assess model adaptation to dynamically changing environments

## Related Work & Insights

- **vs. RLBench/LIBERO/CALVIN**: These simulation benchmarks offer controllability but lack realism; ManipArena achieves controlled generalization measurement in the real world
- **vs. RoboArena**: RoboArena also conducts real-world evaluation but lacks systematic generalization and Real2Sim support; ManipArena's green-screen design eliminates uncontrolled variables
- **vs. VLABench**: VLABench imposes high reasoning demands but operates in simulation; ManipArena brings high reasoning demands into the real world

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of green-screen controlled environment and hierarchical OOD evaluation is novel, though the innovation ceiling of benchmark work is inherently limited
- Experimental Thoroughness: ⭐⭐⭐⭐ Framework design is thorough and comprehensive with broad task coverage, but detailed baseline results from existing models are lacking
- Writing Quality: ⭐⭐⭐⭐⭐ Paper structure is clear, design principles are articulated with rigor, and each design decision is well motivated
- Value: ⭐⭐⭐⭐ Fills a critical gap in standardized real-world evaluation and provides an important contribution to the VLA community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](gecosrt_geometryaware_continual_adaptation_for_rob.md)

</div>

<!-- RELATED:END -->
