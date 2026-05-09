---
title: >-
  [Paper Note] Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment
description: >-
  [AAAI2026][Robotics][dexterous manipulation] This paper proposes the PKDA framework, which automatically converts human hand manipulation videos into high-quality manipulation trajectories for multi-fingered dexterous hands via progressive kinematic-dynamic alignment, achieving an average transfer success rate of 73%.
tags:
  - AAAI2026
  - Robotics
  - dexterous manipulation
  - motion retargeting
  - reinforcement-learning
  - hand-object interaction
  - sim-to-real transfer
date: 2026-05-08
content_hash: b41f92661682c584
---

# Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment

**Conference**: AAAI2026
**arXiv**: [2511.10987](https://arxiv.org/abs/2511.10987)
**Code**: To be confirmed
**Area**: Robotics
**Keywords**: dexterous manipulation, motion retargeting, reinforcement-learning, hand-object interaction, sim-to-real transfer

## TL;DR

This paper proposes the PKDA framework, which automatically converts human hand manipulation videos into high-quality manipulation trajectories for multi-fingered dexterous hands via progressive kinematic-dynamic alignment, achieving an average transfer success rate of 73%.

## Background & Motivation

Manipulation data for multi-fingered dexterous robotic hands is extremely scarce, severely limiting data-driven dexterous manipulation policy learning. Existing data collection approaches face the following challenges:

- **Real hardware collection**: High cost, complex workflow, difficult to scale
- **Pure kinematic mapping** (e.g., Anyteleop): Only performs finger position mapping without contact dynamics optimization, unable to resist inertial disturbances, resulting in very low grasp success rates (only 12.5%)
- **Pure reinforcement learning methods** (e.g., D-Grasp): Low exploration efficiency, task-dependent reward design, limited generalization
- **Online teleoperation**: Requires expensive equipment and real-time human visual feedback, difficult to deploy at scale

The core motivation is: given only RGB videos of human hand manipulation, can manipulation skills be **automatically** transferred to dexterous hands with different morphologies, while simultaneously guaranteeing kinematic fidelity and stable physical interaction?

## Core Problem

1. **Structural differences**: Significant differences exist between human hands and robotic hands in joint degrees of freedom, finger lengths, and kinematic topology — how to perform accurate motion retargeting?
2. **Contact dynamics**: Simple pose mapping cannot transfer force closure and dynamic contact strategies, resulting in unstable grasps under perturbation
3. **Task diversity**: Reward design for different manipulation tasks (grasping, pouring, stamping, etc.) is difficult to unify, limiting the generalizability of the optimization framework

## Method

PKDA models dexterous manipulation transfer as a four-stage pipeline corresponding to four modules:

### 1. Interaction Perceptor

Extracts key interaction information from human hand manipulation videos:

- **Hand trajectory** $H = \{h_1, \dots, h_T\}$: Fingertip spatial positions (15D) + palm orientation (3D)
- **Object trajectory** $O = \{o_1, \dots, o_T\}$: Object centroid 3D position + 3D orientation
- **Contact points** $C = \{c_1, \dots, c_N\}$: 3D coordinates of $N \in \{2,3,4,5\}$ fingertip contact points

For scenes with known object models, HFL-Net is used to estimate hand-object pose; for unknown model scenes, Hold is used to jointly reconstruct 3D geometry, with convex decomposition optimization applied to handle mesh defects caused by occlusion.

### 2. Trajectory Proposer

Maps human hand trajectories to joint angle sequences for the dexterous hand, formulated as a nonlinear optimization problem:

$$\min_{\mathbf{q}_t} \left( w_f E_f + w_o E_o + w_s E_s \right)$$

- $E_f$: Fingertip position constraint — aligns absolute fingertip positions in **world coordinates** (rather than the conventional fingertip-to-wrist vectors), decoupling fingertip localization from wrist alignment and reducing sensitivity to morphological differences
- $E_o$: Palm orientation constraint — measures palm normal vector alignment via geodesic distance
- $E_s$: Temporal smoothness constraint — suppresses abrupt joint angle changes between adjacent frames

The optimized joint angle sequence is converted to control signals $A_{primary}$ via inverse dynamics.

### 3. ContactAdapt Optimizer

This is the core module of the paper, employing reinforcement learning to optimize grasp dynamics through three key designs:

**(a) RL-Configurator (Unified Task Configurator)**:

- **Thumb-guided pre-grasp initialization**: Selects as the RL initial state the configuration where hand and object are not yet in contact and the thumb is closest to its corresponding grasp point. Experiments show that thumb-guided initialization achieves 7.5% higher TSR than index-finger-guided and 10% higher than middle-finger-guided
- **Unified goal formulation**: Abstracts the initial stage of all manipulation tasks as "picking" — the goal is to bring the object to a pose that first deviates 0.1m from its initial position

**(b) Action Space Rescaling**:

- Compresses the wrist joint motion range from the global workspace to a local neighborhood $\mathcal{N}(\hat{q}_{pre}, \rho)$ around the pre-grasp pose
- Finger joints retain their full range of motion
- Ablation experiments show that removing this mechanism causes TSR to drop sharply from 77.5% to 37.5%, making it the most critical design component

**(c) Hierarchical Unified Reward**:

- **Approach reward** $r_{approach}$: Guides fingertips toward target contact points
- **Grasp reward** $r_{grasp}$: Activated when all fingertips enter the contact tolerance range ($\varepsilon = 0.06$m), comprising a contact reward and a pose imitation reward
- **Lift reward** $r_{lift}$: Activated when the thumb and at least one auxiliary finger are in contact with the object, with a piecewise design — linear reward for basic lifting below 0.02m, transitioning to pose adjustment above 0.02m

### 4. Wrist Trajectory Planner

Guided by dynamic object pose changes, the wrist trajectory is computed under the assumption of no relative sliding:

$$T_t = o_t \cdot (T_{grasp}^{-1} \cdot o_{grasp})^{-1}$$

A PD controller drives wrist motion to maintain semantic consistency of the manipulation (e.g., the complete action intent of "lift–tilt–place").

## Key Experimental Results

Experiments are conducted across three scenarios using three dexterous hands: Adroit Hand, Allegro Hand, and Leap Hand:

| Method | SR Grasp↑ | SR Follow↑ | TSR↑ |
|--------|-----------|------------|------|
| Anyteleop | 12.5% | 7.5% | 7.5% |
| PGDM | 72.5% | 72.5% | 72.5% |
| D-Grasp | 62.5% | 60% | 57.5% |
| **PKDA (40 sequences)** | **80%** | **80%** | **77.5%** |
| **PKDA (600 sequences)** | **84.2%** | **77.6%** | **73.3%** |

**Cross-hand generalization**: TSR of Adroit 77.5% / Allegro 72.5% / Leap 67.5%, with consistent position and rotation errors (0.054–0.058m, 31°–33°).

**Perception robustness**: Success rate remains no lower than 70% under pose estimation errors and object reconstruction defects.

**Learning efficiency**: PKDA converges faster in training steps compared to PGDM and D-Grasp (Fig. 4).

**Core ablation findings**:

- Absolute fingertip position retargeting vs. fingertip-to-wrist vectors: the former achieves 7.5% higher TSR
- Action space rescaling: removing it drops TSR from 77.5% to 37.5% (most critical component)
- Thumb-guided vs. index-/middle-finger-guided initialization: 7.5% / 10% higher respectively

**Real-world validation**: Successfully completes three tasks — shaking, pouring, and stamping — on a UR10 arm + Leap Hand, with simulation trajectories executed open-loop.

## Highlights & Insights

1. **Kinematic-dynamic synergy**: Kinematic mapping serves as high-quality initialization and exploration direction constraints for RL, while RL in turn optimizes contact dynamics — the two are mutually reinforcing
2. **Zero task-specific tuning**: The entire transfer process requires no task-specific parameter adjustment; for different dexterous hand configurations, only finger correspondence needs to be specified
3. **Elegant action space rescaling**: Compressing the wrist space while releasing the finger space effectively suppresses overshooting behavior, with ablation experiments confirming its greatest contribution
4. **Complete end-to-end pipeline**: Covers the full chain from raw video to simulation control signals to real robot deployment, encompassing perception, planning, and optimization
5. **New evaluation metric TSR**: A DTW-based semantic-level action intent similarity measure that focuses on manipulation intent rather than frame-by-frame trajectory reproduction

## Limitations & Future Work

- Primarily handles stable contact patterns; **dynamic multi-contact transitions** (e.g., in-hand flipping, rolling manipulation) are not yet addressed
- The wrist trajectory planner assumes no relative sliding after grasping, limiting applicability to tasks requiring in-hand manipulation
- Tactile feedback is not considered; incorporating tactile sensing may further improve contact optimization quality
- Real-world validation uses only open-loop control; closed-loop feedback could improve robustness
- When generalizing across hands, success rates for large dexterous hands decrease on small objects, and finger-to-object scale adaptation still requires improvement

## Related Work & Insights

| Method Category | Representative Work | Strengths | Weaknesses |
|----------------|--------------------|-----------|-----------| 
| Pure kinematic mapping | Anyteleop, DexMV | Simple implementation, fast | No dynamics optimization, unable to resist disturbances |
| Pure RL | D-Grasp, PGDM | Can explore complex interactions | Low exploration efficiency, task-dependent reward design |
| Kinematics + RL | PKDA (Ours) | Efficient, generalizable, no task-specific tuning | Limited to stable contact patterns |
| Online teleoperation | DexCap, AnyTeleop-RT | Real-time human feedback | High cost, not scalable |

Compared to PGDM: PGDM treats object trajectories as hard constraints for precise reproduction, trading efficiency for accuracy; PKDA prioritizes transferring manipulation intent, applying RL only for the grasping phase and PD control elsewhere, yielding significantly higher efficiency.

**Broader implications**:

- The paradigm of **kinematic-guided RL exploration** has general value: in high-dimensional action spaces, using simple mapping to determine the feasible region before fine-tuning with RL can substantially improve sample efficiency
- The **action space rescaling** idea can be generalized to other robot learning tasks: partitioning and selectively compressing/releasing action spaces according to the motion characteristics of different joints
- Complementary to the foundation model direction in dexterous manipulation: PKDA provides an efficient data generation pipeline that can serve as a data source for large-scale policy pretraining
- The thumb-first contact strategy design reflects biomechanical principles of human grasping and can inspire control architecture design for bioinspired dexterous hands

## Rating

- Novelty: ⭐⭐⭐⭐ — The progressive kinematic-dynamic alignment framework and action space rescaling are novel contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three dexterous hands, three scenarios, multiple baselines, comprehensive ablation, and real robot validation
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with modular presentation that aids comprehension, though some formula notation is dense
- Value: ⭐⭐⭐⭐ — Provides a practical dexterous manipulation data generation solution with direct impact on data-driven robot manipulation research

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](../../NeurIPS2025/robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[AAAI 2026\] To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance](to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)
- [\[AAAI 2026\] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)

<!-- RELATED:END -->
