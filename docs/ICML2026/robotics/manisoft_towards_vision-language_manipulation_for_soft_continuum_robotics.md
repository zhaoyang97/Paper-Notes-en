---
title: >-
  [Paper Note] ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics
description: >-
  [ICML 2026][Robotics][Soft robots] To address the gap where vision-language manipulation research almost exclusively covers rigid arms while ignoring soft continuum robots…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Soft robots"
  - "vision-language manipulation"
  - "benchmark"
  - "hybrid simulation"
  - "hierarchical expert trajectories"
date: 2026-05-08
content_hash: de23901527253826
---

# ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics

**Conference**: ICML 2026  
**arXiv**: [2605.18617](https://arxiv.org/abs/2605.18617)  
**Code**: https://buaa-colalab.github.io/ManiSoft (Project Page)  
**Area**: Robotics / Soft Continuum Manipulators / VLA Benchmark  
**Keywords**: Soft robots, vision-language manipulation, benchmark, hybrid simulation, hierarchical expert trajectories

## TL;DR
To address the gap where vision-language manipulation research almost exclusively covers rigid arms while ignoring soft continuum robots, this paper constructs the ManiSoft benchmark. Utilizing a hybrid simulator that couples "Cosserat rod soft dynamics + MuJoCo rigid body contact + elastic force constraints," it defines four task categories reflecting the difficulties of soft arm control. Through an automated generation of 6,300 scenes and expert trajectories using a "high-level rule planner + low-level RL torque actuator," the study reveals that models like DP, RDT, and OpenVLA-OFT are moderately successful in clean scenes (~30%) but suffer a catastrophic drop in randomized environments (up to 29.4 points). The root causes of failure lie in the inability to estimate proprioceptive states from vision and the failure to exploit soft body deformability for obstacle avoidance.

## Background & Motivation

**Background**: Vision-language manipulation has become a core component of embodied AI. Benchmarks such as RLBench, ManiSkill, CALVIN, LIBERO, RoboVerse, and RoboTwin have matured in training and evaluating "image-to-instruction-to-execution" pipelines. However, the robotic arms in these benchmarks are **universally rigid arms**, characterized by readable joint angles, low-dimensional kinematics, and direct perception-to-control links. VLA models such as OpenVLA, $\pi$ series, RDT-1B, CogACT, and DexVLA have rapidly evolved under these assumptions.

**Limitations of Prior Work**: Rigid arms have structural shortcomings in cluttered or narrow spaces—rigid joint constraints mean the gripper cannot reach targets blocked by obstacles without "going around" them. Soft continuum arms (Cosserat rods, pneumatic/tendon-driven, low elastic modulus materials) can bend and deform as a whole to bypass obstacles. However, they introduce three new challenges: (i) **No reliable proprioception**—soft arms lack rigid joint encoders and must infer poses from external vision; (ii) **Low-level execution involves torque/tension/pressure** instead of joint target poses, making inverse kinematics extremely complex; (iii) **Distributed actuators** lead to high-dimensional and highly coupled action spaces. These issues make existing VLA models almost non-functional when directly transferred to soft arms.

**Key Challenge**: The mature assumptions of rigid-arm VLA (precise proprioception + low-dimensional joint space + analytical inverse kinematics) conflict with the physical reality of soft arms (visual proprioception + high-dimensional torque space + strongly coupled flexible dynamics) at almost every level. A benchmark is needed to "honestly expose these differences" to guide research.

**Goal**: (i) Provide a soft arm simulator capable of accurately simulating elastic deformation while handling contact friction; (ii) Design tasks that distinguish between four difficulty levels: basic trajectory control, fine-grained posing, contact-intensive stacking, and complex obstacle avoidance; (iii) Provide a scalable data generation pipeline with 6.3k expert trajectories; (iv) Benchmark mainstream VLA models to pinpoint failure modes.

**Key Insight**: The authors found that existing soft simulators (Elastica, SOFA) excel at elastic dynamics but have weak contact modeling, while rigid simulators (MuJoCo, SAPIEN, Habitat) excel at contact friction but cannot handle deformation. Therefore, **the two types of simulators are coupled using a "virtual spring"**: soft body deformation is simulated by Elastica, while the end-effector contact is handled by MuJoCo. These two components pull each other through Hooke's Law elastic constraints. Expert trajectories are also handled hierarchically—a high-level rule planner provides 6-DoF waypoints, and a low-level RL actuator translates these waypoints into torques.

**Core Idea**: Establish soft arm VLA research as a scalable benchmark through "soft-rigid hybrid simulation + hierarchical waypoint-torque experts," exposing the failure modes of existing VLAs through a two-tier evaluation of clean and randomized environments.

## Method

### Overall Architecture
ManiSoft consists of three components: (1) **Hybrid Simulator**: Models the soft arm as a three-part coupled system: "soft body (Cosserat rod, simulated by Elastica) + end-effector (simulated by MuJoCo) + elastic force constraints (virtual spring)"; environment visuals are rendered by Blender. (2) **Four Task Categories**: Collecting (COLL, placing targets in containers), Alignment (ALN, precise 6-DoF positioning), Stacking (STK, stacking tableware by size), and Arrangement (ARR, spatial constraints and obstacle avoidance). (3) **Automated Data Pipeline**: Procedurally samples 263 3D objects and candidate grasp poses to construct clean/randomized scenes, using GPT templates for diverse instructions. Expert trajectories use a two-stage generation: "high-level rule planner (SE(3) waypoints) + low-level RL torque actuator (waypoint tracking)." Finally, 6,300 scene-trajectory pairs are released across 109 manipulable objects in 17 categories and 154 obstacles in 35 categories, with an average trajectory length of 1,272 steps.

### Key Designs

1.  **Hybrid Soft-Rigid Simulator (Cosserat Rod + MuJoCo + Elastic Constraints)**:
    - **Function**: Simultaneously captures "accurate elastic deformation" and "stable contact friction" in one simulation stack, filling the capability gap of single-type simulators in soft arm manipulation.
    - **Mechanism**: Decouples the soft arm into two independent but coupled subsystems. The **soft body** is discretized into $N$ segments of Cosserat rods using Elastica. External driving torque $\boldsymbol{\tau}_e\in\mathbb{R}^{N\times 3}$ generates four types of strain (axial, shear, bending, torsion) along the rod, producing internal forces $\mathbf{f}_i$ and internal torques $\boldsymbol{\tau}_i$ that determine deformation. The **end-effector** and its contact friction are handled by MuJoCo's mature contact solver. **The two are coupled by a zero-length virtual spring**: when the relative displacement $\Delta\mathbf{x}\in\mathbb{R}^3$ and relative rotation $\Delta\boldsymbol{\theta}\in\mathbb{R}^3$ between the soft tip and end-effector are non-zero, restoring force and torque are generated via Hooke's Law: $\mathbf{F}=-k_F\Delta\mathbf{x}$ and $\mathbf{M}=-k_M\Delta\boldsymbol{\theta}$ ($k_F, k_M$ are adjustable stiffness), pulling both sides toward synchronized motion. Visual observations are rendered as RGB images using a fixed camera in Blender.
    - **Design Motivation**: Pure soft simulators cannot stably handle tabletop contact, and pure rigid simulators cannot deform continuously. The elastic constraint acts as a "soft connector" ensuring physical force closure while decoupling the numerical integration of the two simulators, making long-horizon, contact-intensive tasks like STK feasible.

2.  **Four Tasks + Two-Tier Randomized Evaluation Protocol**:
    - **Function**: Covers a difficulty gradient from "basic trajectory control" to "complex obstacle avoidance" under a unified interface, exposing generalization collapse under visual and physical changes.
    - **Mechanism**: For each timestep $t$, given instruction $\mathbf{L}$ and visual observation $\mathbf{V}_t$, the policy outputs $\mathbf{A}_t=(\boldsymbol{\tau}_e, S)$, where $\boldsymbol{\tau}_e$ is the external torque and $S\in\{0,1\}$ is the gripper state. **Key Decision**: Unlike rigid arm benchmarks that feed joint angles as proprioception, ManiSoft **deliberately provides only visual observations** without internal soft body states, forcing policies to learn how to determine deformation from images. Tasks are split into two levels: **clean** (fixed layout/appearance) and **randomized** (distractor obstacles, randomized textures/lighting, and diverse attribute-based language descriptions like "yellow bottle"). Metrics include success rate and completion steps (#Steps).
    - **Design Motivation**: (a) Omitting proprioception reflects the reality that most soft arms lack encoders; (b) Randomized settings distinguish between "scene overfitting" and robust visual-language generalization; (c) The four tasks provide a diagnostic gradient—low performance in ALN/STK despite high COLL success indicates a failure in precision control.

3.  **Hierarchical Waypoint-Torque Expert Trajectory Generation**:
    - **Function**: Decomposes high-quality long trajectory generation into two scalable steps, avoiding training collapse associated with directly learning long-horizon torque control via RL.
    - **Mechanism**: **High-level**: A rule-based planner generates 6-DoF waypoints $\hat P\in\mathrm{SE}(3)$ for sub-goals like "approach," "grasp," and "lift." **Low-level**: An RL actuator takes (target pose $\hat P$, proprioceptive history, current pose $P$) as input and outputs torque $\boldsymbol{\tau}_e$. Pose error is measured via SE(3) logarithm $[\mathbf{d}_p,\mathbf{d}_r]=\log(P^{-1}\hat P)$, with distance $d=\|\mathbf{d}_p\|_2+\alpha\|\mathbf{d}_r\|_2$. The reward consists of two parts: a distance reward $R_d=-d+k_1\mathbbm{1}_{d<d_1}+k_2\mathbbm{1}_{d<d_2}$ and a stability reward $R_s=-\mathrm{sgn}(\partial d/\partial t)\cdot\beta$ active when $d\le D$ (otherwise 0). $R_s$ encourages the error to decrease when close to the target, penalizing oscillation. Optimal parameters are $\beta=1, D=0.3$.
    - **Design Motivation**: Direct RL on torque sequences suffers from high-dimensional coupling and sparse rewards. Hierarchical decomposition assigns logical structure to rules and dynamic tracking to RL. The stability reward $R_s$ significantly reduces end-effector fluctuations, making trajectories smoother and more usable for downstream imitation learning.

### Loss & Training
- Low-level RL actuator: Total reward $R=R_d+R_s$ with distance and stability terms; optimal $\beta=1, D=0.3$ determined via ablation.
- Data Scale: 6,300 scene-trajectory pairs (2,100 clean + 4,200 randomized), avg. 40 instructions per scene, 4:1 train/test split.
- Evaluation: DP and RDT trained from scratch; OpenVLA-OFT fine-tuned via LoRA.

## Key Experimental Results

### Main Results
Success rates (ACC%) and completion steps (#Steps) for the four tasks:

| Model | COLL ACC | ALN ACC | STK ACC | ARR ACC | Avg. ACC | Avg. #Steps |
|------|----------|---------|---------|---------|----------|-------------|
| **Clean** | | | | | | |
| DP (~400M) | 63.0 | 18.3 | 15.0 | 30.0 | 31.6 | 520 |
| RDT (~1B) | 13.8 | 11.7 | 10.0 | 1.3 | 9.2 | 496 |
| OpenVLA-OFT (~400M) | 45.4 | 25.0 | 20.0 | 31.3 | **30.4** | 527 |
| **Randomized** | | | | | | |
| DP | 3.8 | 1.7 | 2.5 | 0.6 | 2.2 | 613 |
| RDT | 1.2 | 4.2 | 0.0 | 1.3 | 1.6 | 368 |
| OpenVLA-OFT | 32.7 | 26.7 | 35.0 | 13.7 | **27.0** | 554 |

In the clean setting, DP and OpenVLA-OFT are comparable (31.6% vs 30.4%), while RDT lags significantly (9.2%) due to overfitting on the 6.3k dataset. COLL is consistently the easiest, while STK is the hardest. **The diagnostic signal appears in the randomized setting**: DP drops 29.4 points to 2.2%, whereas OpenVLA-OFT only drops 3.4 points, maintaining 27.0%, showcasing the visual generalization advantage of pretrained VLM backbones.

### Ablation Study
ARR task breakdown by object category (Randomized):

| Model | Rubik's Cube ACC | Bottle ACC | Pen Cup ACC | Shoe ACC | ARR Avg. |
|------|------------------|------------|-------------|----------|----------|
| DP | 0.0 | 0.0 | — | 2.5 | 0.6 |
| RDT | 0.0 | 2.5 | 2.5 | 0.0 | 1.3 |
| OpenVLA-OFT | 15.0 | 7.5 | 25.0 | 7.5 | 13.7 |

Stability reward $R_s$ hyperparameter ablation for the RL actuator (control stability = variance of pose error, lower is better):

| $D \backslash \beta$ | 0 | 0.5 | 1 | 1.5 |
|-----|-----|------|------|------|
| 0.05 | 0.176 | 0.157 | 0.074 | 0.121 |
| 0.10 | 0.176 | 0.149 | 0.153 | 0.071 |
| 0.20 | 0.176 | 0.070 | 0.135 | 0.064 |
| 0.30 | 0.176 | 0.145 | **0.053** | 0.091 |

### Key Findings
- **Object geometry dictates difficulty**: Success rates are highest for regular geometric cubes and lowest for irregular non-convex shoes.
- **OpenVLA-OFT's "stop-moving" behavior**: The model often stagnates after a successful grasp, leading to lower COLL ACC (45.4%) than DP (63.0%), attributed to self-inhibitory loops induced by subtle visual changes in the gripper.
- **Failure Mode 1: Proprioceptive Ambiguity**: When targets are near the base requiring extreme bending, internal torques dominate. Policies fail to estimate pose accurately, leading to lateral drift.
- **Failure Mode 2: Inability to exploit soft body properties**: Policies attempt to move through obstacles instead of deforming around them, indicating that existing VLAs lack an understanding of soft body affordances.

## Highlights & Insights
- **Hybrid simulation with virtual springs** is a clever engineering choice that leverages optimized components (Elastica + MuJoCo) without the instability of unified simulators.
- **Deliberate omission of proprioception** forces research toward "visual pose estimation," which is the most critical and weak area of current VLAs.
- **Hierarchical structure with $R_s$** provides a template for controlling high-degree-of-freedom flexible systems (e.g., cable-driven arms).
- **Diagnostic gradient design** ensures that benchmark results are naturally interpretable, identifying exactly where a model fails.

## Limitations & Future Work
- **Low absolute success rates**: The benchmark is highly challenging, but this may deter users if no methods show initial promise.
- **Computational Cost**: The combination of Cosserat rods, MuJoCo, and Blender is computationally expensive, potentially limiting large-scale online RL research.
- **Instruction Diversity**: While attribute-based, instructions are still template-driven and may underestimate real-world linguistic complexity.
- **Sim-to-Real Gap**: The benchmark is currently simulation-only. Sim-to-real for soft materials remains a massive challenge.

## Related Work & Insights
- **vs. LIBERO / CALVIN / RLBench**: These rigid arm benchmarks assume precise proprioception, serving as "rigid vs. soft" comparisons to ManiSoft.
- **vs. ManiSkill / RoboTwin**: While these offer more diverse scenes, they are rigid-only. ManiSoft repurposes RoboTwin assets but redesigns tasks and the simulation stack for soft bodies.
- **vs. OpenVLA-OFT / RDT**: These VLAs were developed for rigid bodies; this study proves they suffer from "stop-moving" behaviors and proprioceptive distortion when applied to soft arms, suggesting a need for soft-body data or pose-estimation modules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)

</div>

<!-- RELATED:END -->
