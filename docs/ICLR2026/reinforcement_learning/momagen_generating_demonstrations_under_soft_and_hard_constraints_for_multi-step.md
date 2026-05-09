---
title: >-
  [Paper Note] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation
description: >-
  [ICLR 2026][Reinforcement Learning][mobile manipulation] MoMaGen formulates demonstration data generation for bimanual mobile manipulation as a constrained optimization problem. By combining hard constraints (reachability, collision-free motion, visibility) with soft constraints (object visibility during navigation, retraction to compact poses), the framework automatically generates large-scale, diverse datasets from a single human teleoperation demonstration. The resulting visuomotor policy can be deployed on a physical robot with only 40 real demonstrations for fine-tuning.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - mobile manipulation
  - bimanual coordination
  - constrained optimization
  - automatic data generation
  - imitation learning
date: 2026-05-08
content_hash: e5f358f7aa5498e8
---

# MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation

**Conference**: ICLR 2026
**arXiv**: [2510.18316](https://arxiv.org/abs/2510.18316)
**Code**: [Project Page](https://momagen.github.io)
**Area**: Reinforcement Learning
**Keywords**: mobile manipulation, bimanual coordination, constrained optimization, automatic data generation, imitation learning

## TL;DR

MoMaGen formulates demonstration data generation for bimanual mobile manipulation as a constrained optimization problem. By combining hard constraints (reachability, collision-free motion, visibility) with soft constraints (object visibility during navigation, retraction to compact poses), the framework automatically generates large-scale, diverse datasets from a single human teleoperation demonstration. The resulting visuomotor policy can be deployed on a physical robot with only 40 real demonstrations for fine-tuning.

## Background & Motivation

- **Background**: Learning from large-scale human teleoperation data (imitation learning) has proven effective for training robotic manipulation skills. The X-Gen family of methods (MimicGen, SkillMimicGen, DexMimicGen, etc.) uses a small number of human demonstrations as seeds to automatically generate 25×–350× data variants in simulation, substantially reducing data collection costs. However, these methods primarily target tabletop manipulation with fixed-base robots.
- **Limitations of Prior Work**: Bimanual mobile manipulation introduces two core challenges: (1) **Reachability** — a mobile base requires replanning its pose for each new scene configuration; directly replaying the navigation segment from a source demonstration often places the base in a location from which the arm cannot reach the target after object randomization; (2) **Visibility** — the onboard moving camera means that naïve data augmentation can push task-relevant objects out of the camera field of view, causing visuomotor policies to fail.
- **Key Challenge**: Human teleoperation of bimanual mobile manipulation is extremely difficult (simultaneous control of the base and two high-DoF arms), making data collection costly; yet existing automatic data generation methods cannot handle base motion or camera visibility and are limited to simple tabletop tasks.
- **Goal**: Design a general automatic data generation framework for bimanual mobile manipulation that produces high-quality, diverse demonstrations under aggressive scene randomization (object positions, distractors, obstacles).
- **Key Insight**: Unify data generation as an optimization problem with hard and soft constraints. This abstraction accommodates the novel requirements of mobile manipulation and also subsumes prior X-Gen methods within the same framework — differing only in which constraints are selected.
- **Core Idea**: Introduce four constraint types — reachability (hard), object visibility during manipulation (hard), object visibility during navigation (soft), and retraction to a compact pose (soft) — and discover base poses and whole-body motion trajectories satisfying all constraints via a sample-and-verify loop.

## Method

### Overall Architecture

Given a human source demonstration and a randomized new scene configuration, MoMaGen decomposes the demonstration into subtasks (navigation segments + manipulation segments) and generates new trajectories subtask by subtask. For each subtask: (1) transfer the relative end-effector–to–object pose from the source demonstration to the new object position; (2) sample a base pose satisfying reachability and visibility constraints; (3) plan a base/torso navigation trajectory with soft visibility costs; (4) plan arm motion to a pre-grasp pose; (5) replay the contact-rich manipulation segment in task space; (6) retract to a compact pose. On failure, the system backtracks and resamples.

### Key Designs

1. **Constrained Optimization Formulation**:

    - *Function*: Unify automatic data generation as a constrained optimization problem.
    - *Mechanism*: The optimization variable is the full trajectory $\{a_t\}_{t \in [T]}$, subject to constraints including system dynamics $s_{t+1} = f(s_t, a_t)$, kinematic feasibility $\mathcal{G}_{\mathrm{kin}}(s_t, a_t) \leq 0$, collision freedom $\mathcal{G}_{\mathrm{coll}}(s_t, a_t) \geq 0$, visibility $\mathcal{G}_{\mathrm{vis}}(s_t, a_t, o_{i(t)}) \leq 0$, contact-segment relative pose preservation $\mathbf{T}_W^{E_k} = \mathbf{T}_W^{o_i} (\mathbf{T}_W^{o_{i,\text{src}}})^{-1} \mathbf{T}_W^{E_k}$, and task success. The objective $\mathcal{L}(\cdot)$ encodes user-specified soft constraints such as trajectory length and smoothness.
    - *Design Motivation*: Prior X-Gen methods implicitly solve constrained optimization problems but each selects a different, incomplete subset of constraints. The unified framework clarifies methodological differences and provides a principled basis for incorporating new constraints.

2. **Reachability Constraint & Base Pose Sampling**:

    - *Function*: Sample base poses in new scenes such that the arm can reach all target positions.
    - *Mechanism*: Base poses $\mathbf{T}^{\mathrm{base}}$ are randomly sampled from an annular region around the target object; inverse kinematics (IK) verifies that all required end-effector trajectories $\{\mathbf{T}_W^{E_k}\}$ lie within the arm workspace. Collision detection further eliminates poses that conflict with furniture or obstacles. cuRobo (a GPU-accelerated motion generation library) is used for efficient IK solving and collision-free path planning.
    - *Design Motivation*: MimicGen and DexMimicGen replay the source demonstration's base trajectory, which becomes inappropriate after object randomization. Under D1/D2 randomization (objects anywhere on furniture, with added obstacles), base placement must be replanned from scratch.

3. **Dual Visibility Constraints**:

    - *Function*: Ensure that task-relevant objects remain within the camera field of view.
    - *Mechanism*: **Hard constraint layer** — before the manipulation phase begins, the system verifies that the sampled base pose and head camera orientation afford an unoccluded view of the target object; otherwise it resamples. **Soft constraint layer** — during the navigation phase, a cost term is added to the motion planner to prefer orientations where the head camera faces the target, without enforcing it strictly. Formally, sampled $\mathbf{T}^{\mathrm{cam}}$ must satisfy $\mathcal{G}_{\mathrm{vis}}(s_t, a_t, o_{i(t)}) \leq 0$.
    - *Design Motivation*: Visuomotor policies rely on RGB images for decision-making; if task-relevant objects are frequently invisible in training data, the policy cannot learn reliable visual servoing behavior. Ablations show that removing visibility constraints causes a dramatic performance drop (Tidy Table: 0.40 → 0.05).

4. **Retraction as Soft Constraint**:

    - *Function*: Return the arms and torso to a compact configuration after each manipulation phase.
    - *Mechanism*: Upon completion of a manipulation subtask, the robot retracts its arms and torso to predefined "compact" joint angles, reducing the robot's footprint.
    - *Design Motivation*: Reducing the probability of collisions with the environment during subsequent navigation, particularly in obstacle-rich D2 scenes.

### Loss & Training

The data generation phase employs a constraint-based sample-and-verify loop (not gradient-based optimization) and does not involve a conventional loss function. The policy training phase uses standard behavior cloning (BC): $\arg\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}} [-\log \pi_\theta(a|s)]$.

Two policy learning methods are evaluated:
- **WB-VIMA**: Trained from scratch; inputs include proprioception and three RGB image streams (head and two wrists, fused into an egocentric point cloud); outputs target joint angles.
- **$\pi_0$**: Fine-tuned from a pretrained checkpoint using LoRA (rank=32); inputs include RGB images and proprioception; outputs target joint angles.

## Key Experimental Results

### Main Results

Four household tasks: Pick Cup (navigate + grasp cup), Tidy Table (move cup to sink across long distance), Put Dishes Away (bimanual independent dish stacking), Clean Frying Pan (bimanual coordinated pan scrubbing). Three randomization levels: D0 (±15 cm / ±15°), D1 (arbitrary positions on furniture), D2 (D1 + distractors and floor obstacles).

**Data Generation Success Rate Comparison**:

| Method | Pick Cup | Tidy Table | Put Dishes Away | Clean Frying Pan |
|--------|----------|------------|-----------------|------------------|
| MoMaGen (D0) | 0.86 | 0.80 | 0.38 | 0.51 |
| SkillMimicGen (D0) | 1.00 | 0.69 | 0.38 | 0.40 |
| DexMimicGen (D0) | 1.00 | 0.72 | 0.38 | 0.35 |
| MoMaGen (D1) | 0.60 | 0.64 | 0.34 | 0.20 |
| MoMaGen (D2) | 0.47 | 0.22 | 0.07 | 0.16 |

Note: Baseline methods achieve zero success rate under D1/D2 (objects fall outside reachable range after base pose replay) and are therefore omitted.

**Task-Relevant Object Visibility Comparison**:

| Method | Pick Cup | Tidy Table | Put Dishes Away | Clean Frying Pan |
|--------|----------|------------|-----------------|------------------|
| MoMaGen (D0) | 1.00 | 0.86 | 0.79 | 0.69 |
| SkillMimicGen (D0) | 1.00 | 0.40 | 0.71 | 0.65 |
| DexMimicGen (D0) | 1.00 | 0.39 | 0.71 | 0.67 |
| MoMaGen w/o vis. (D0) | 0.90 | 0.46 | 0.40 | 0.35 |
| MoMaGen (D1) | 0.93 | 0.89 | 0.78 | 0.80 |
| MoMaGen (D2) | 0.94 | 0.79 | 0.75 | 0.81 |

### Ablation Study

**Effect of Visibility Constraints on Policy Performance (WB-VIMA, 1000 demos, D0)**:

| Method | Pick Cup Success Rate | Tidy Table Success Rate |
|--------|-----------------------|-------------------------|
| MoMaGen (full) | 0.75 | 0.40 |
| w/o soft visibility | ~0.55 | ~0.05 |
| w/o hard visibility | ~0.50 | ~0.05 |
| w/o all visibility | ~0.45 | ~0.05 |

**$\pi_0$ Data Scaling Effect (Pick Cup D1)**:

| # Demonstrations | 500 | 1000 | 2000 |
|-----------------|-----|------|------|
| Success rate trend | Lower | Medium | Higher |

Under D1 randomization, increasing data volume yields significant performance gains, reflecting broader coverage of the state–action space.

**Sim-to-Real Results (Pick Cup D0, fine-tuned with 40 real demos)**:

| Method | With Sim Pretraining | Without Sim Pretraining |
|--------|----------------------|-------------------------|
| WB-VIMA | 10% | 0% |
| $\pi_0$ | 60% | 0% |

### Key Findings

- MoMaGen achieves an average data generation success rate of 63% under D0 and is the only method capable of handling D1/D2 randomization — baselines achieve zero success rate under D1/D2.
- Visibility constraints substantially affect policy quality: on Tidy Table, removing all visibility constraints reduces policy success rate from 0.40 to 0.05, an 87.5% decline.
- Data diversity is critical: MoMaGen's D1 data covers the entire tabletop rather than a narrow region; PCA projections reveal that arm and torso joint angle distributions are far broader than those of baselines.
- $\pi_0$ benefits significantly from simulation pretraining even given strong pretrained weights (10k+ hours of robot data): success rate improves from 0% to 60%.

## Highlights & Insights

- **The unified framework perspective is highly insightful**: Casting all X-Gen methods as instances of constrained optimization with different constraint selections clearly highlights methodological differences (MimicGen uses only the task success constraint; SkillMimicGen adds kinematics and collision constraints, etc.) and provides a principled basis for future extensions — new methods need only define appropriate hard/soft constraints.
- **The dual-layer visibility design reflects a deep understanding of visuomotor policy training**: Distinguishing between the manipulation phase (object must be visible → hard constraint) and the navigation phase (object preferably visible → soft constraint) strikes an engineering balance that ensures data quality without over-constraining the system and depressing generation rates. The 8× performance gap attributable to visibility underscores that this is a necessity, not a nicety.
- **End-to-end pipeline from a single demonstration to real-world deployment**: 1 human demonstration → 1,000 simulation variants → policy training → 40 real demonstration fine-tuning → 60% real-world success rate with $\pi_0$, demonstrating the practical value of the X-Gen paradigm for complex bimanual mobile manipulation.

## Limitations & Future Work

- **Dependency on complete scene knowledge**: The framework currently assumes full scene information during data generation (precise object poses and geometry), which is naturally available in simulation but requires additional perception systems (e.g., SAM2 for estimating relative object poses) in the real world.
- **Sequential navigation–manipulation only**: The current framework assumes alternating navigation and manipulation phases and does not support whole-body manipulation (e.g., simultaneously moving the base and arms while pushing open a door).
- **High GPU resource requirements**: The system relies on GPU-accelerated motion generators such as cuRobo, making it computationally intensive; simulation execution dominates total runtime (base motion planning: ~18 s vs. simulation execution: ~100 s).
- **Base pose sampling efficiency**: The current approach uniformly samples within an annular region; when feasible base poses are sparse, search becomes slow. Smarter sampling strategies (e.g., biasing toward regions with more free space) could improve efficiency.
- **Generation success rate degrades significantly under D2**: Floor obstacles constrain the navigation space; Put Dishes Away achieves only 7% success rate under D2, indicating substantial room for improvement in complex scenes.

## Related Work & Insights

- **vs. MimicGen**: MimicGen is the seminal X-Gen work but supports only single-arm, fixed-base tasks and replays base trajectories directly. MoMaGen overcomes this fundamental limitation through base pose sampling and reachability constraints.
- **vs. SkillMimicGen**: SkillMimicGen adds kinematic and collision constraints and handles obstacle scenes, but remains limited to fixed-base, single-arm settings. MoMaGen extends to mobile base, bimanual, and active-camera whole-body control.
- **vs. DexMimicGen**: DexMimicGen supports bimanual dexterous manipulation but lacks a mobile base and ignores visibility. MoMaGen builds on this by adding mobile base support, visibility constraints, and obstacle handling.
- **vs. DemoGen / PhysicsGen**: DemoGen and PhysicsGen introduce collision-free and system dynamics constraints, respectively, but neither supports a mobile base or active perception. MoMaGen is the only method that simultaneously satisfies all six constraint types.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified constrained optimization framework is a novel conceptual contribution; the reachability and dual-layer visibility constraint designs are original. However, the underlying techniques (IK, motion planning, cuRobo) are existing tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four tasks × three randomization levels × multiple baselines and ablations, plus data diversity analysis, policy training evaluation, and real-robot deployment, constituting a complete experimental chain.
- **Writing Quality**: ⭐⭐⭐⭐ — The unified framework formalization is clear, experimental design is logically rigorous, and figures are information-rich; the constrained optimization formulation is readable.
- **Value**: ⭐⭐⭐⭐ — Automatic data generation for bimanual mobile manipulation addresses a pressing community need; the framework is general and extensible to new tasks and robot platforms, though reliance on complete scene information limits direct real-world applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](../../NeurIPS2025/reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_scaling_tool_use.md)
- [\[AAAI 2026\] Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization](../../AAAI2026/reinforcement_learning/provably_efficient_multi-objective_bandit_algorithms_under_preference-centric_cu.md)

<!-- RELATED:END -->
