---
title: >-
  [Paper Note] NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][closed-loop simulation] This paper proposes NeuroNCAP, a photorealistic closed-loop safety testing framework for autonomous driving based on NeRF rendering. Inspired by the Euro NCAP collision avoidance protocols, three types of safety-critical scenarios (stationary, frontal, and side collisions) are designed. It reveals that current state-of-the-art (SOTA) end-to-end planners (UniAD, VAD) fail catastrophically in closed-loop safety scenarios—w…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "closed-loop simulation"
  - "NeRF"
  - "safety testing"
  - "end-to-end planning"
  - "Euro NCAP"
date: 2026-05-08
content_hash: 033a8c8f69af35ae
---

# NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving

**Conference**: ECCV 2024  
**arXiv**: [2404.07762](https://arxiv.org/abs/2404.07762)  
**Code**: [https://github.com/atonderski/neuro-ncap](https://github.com/atonderski/neuro-ncap)  
**Area**: Autonomous Driving  
**Keywords**: closed-loop simulation, NeRF, safety testing, end-to-end planning, Euro NCAP

## TL;DR

This paper proposes NeuroNCAP, a photorealistic closed-loop safety testing framework for autonomous driving based on NeRF rendering. Inspired by the Euro NCAP collision avoidance protocols, three types of safety-critical scenarios (stationary, frontal, and side collisions) are designed. It reveals that current state-of-the-art (SOTA) end-to-end planners (UniAD, VAD) fail catastrophically in closed-loop safety scenarios—with collision rates as high as 88-92%—despite their perception modules functioning accurately.

## Background & Motivation

**The Rise and Traps of End-to-End Planning**: End-to-end autonomous driving models such as UniAD and VAD excel on the nuScenes planning benchmark. However, this benchmark relies on open-loop evaluation—the planner's predictions are never executed, and they are only compared with recorded human driving trajectories using displacement error. Studies have already shown a weak correlation between open-loop evaluation scores and actual driving quality.

**Three Core Problems of Open-Loop Evaluation**:
   - **Planner decisions do not affect the environment**: If the planner predicts an erroneous trajectory, the scene does not change, making it impossible to observe cascading effects.
   - **Only normal driving scenarios are evaluated**: nuScenes does not contain collision accident scenarios, failing to test model performance in safety-critical situations.
   - **Displacement error metric is misleading**: Zero error equates to human-level driving, but a low error does not necessarily imply better driving, as multiple different trajectories could be equally valid.

**Challenges of Closed-Loop Testing**: Traditional closed-loop simulators (such as CARLA and nuPlan) do not generate sensor data, making them incapable of testing sensor-input end-to-end planners. Crafted graphics-based simulators lack photorealism and scenario diversity.

**Core Idea**: NeRF is utilized to learn 3D scene representations from real-world driving data. Safety-critical scenarios are created by editing the 3D bounding boxes of dynamic objects. The planner is then allowed to "actually drive" in a closed loop: rendering images $\rightarrow$ predicting trajectories $\rightarrow$ executing control $\rightarrow$ propagating vehicle states $\rightarrow$ rendering new images, repeating the cycle. Standardified safety scenarios are designed by drawing inspiration from the Euro NCAP collision-avoidance testing protocols.

**Core Problem**: How do current SOTA end-to-end planners perform in safety-critical scenarios when their predicted trajectories are actually executed in closed loop?

## Method

### Overall Architecture

The closed-loop simulator consists of four iterative steps:
1. **Neural Renderer (NeRF)**: Given the ego-vehicle state and camera calibration, it renders photorealistic multi-view camera images.
2. **AD Model**: The end-to-end planner (e.g., UniAD or VAD) takes the rendered images and ego-vehicle state as input to predict trajectory waypoints for the next 3 seconds.
3. **Controller**: A Linear Quadratic Regulator (LQR) converts the trajectory waypoints into steering angle $\delta$ and acceleration $a$ control signals.
4. **Vehicle Model**: A kinematic bicycle model propagates the ego-vehicle state.

### Key Designs

#### 1. Neural Renderer — Photorealistic Scene Generation and Editing

**Function**: Learn 3D scene representations from real driving data, supporting rendering from novel viewpoints and dynamic object editing (adding, removing, or modifying trajectories).

**Mechanism**: NeuRAD, a NeRF method specifically designed for autonomous driving, is adopted as the renderer. It is trained for 100k steps using the large configuration (NeuRAD-L). It supports editing dynamic objects by modifying 3D bounding boxes, such as placing a vehicle that was originally driving in an adjacent lane to be stationary in the ego-lane. Additionally, data augmentation by flipping along the symmetry axis is employed to support rendering objects from arbitrary viewpoints.

**Design Motivation**: Learning from real-world data, NeRF naturally addresses the two major challenges faced by handcrafted graphics simulators: photorealism and scenario diversity. The capability for dynamic object editing allows the synthesis of safety-critical scenarios from normal driving data.

#### 2. Safety-Critical Scenario Design — Inspired by Euro NCAP

**Function**: Design three classes of collision scenarios to simulate typical safety events in the Euro NCAP collision avoidance testing protocols.

**Three Scenario Types**:

- **Stationary**: A stationary vehicle is placed in the ego-lane. The ego-vehicle can avoid collision through hard braking or steering. This is the simplest scenario.
- **Frontal**: The target vehicle cuts into the ego-lane from the opposite lane, creating a head-on collision path. Braking alone cannot completely prevent collision (it only reduces speed); steering is required to avoid it.
- **Side**: The target vehicle crosses the ego-lane perpendicularly. Collision can be avoided by braking to wait, or accelerating to pass along with minor steering adjustments.

**Scenario Construction**: Based on 14 validation sequences from nuScenes, suitable sequences and target vehicles are manually selected. The initial states of the ego and target vehicles are set such that a collision would occur in approximately 4 seconds if current speeds and steering are maintained. All non-stationary objects are removed, and one is randomly chosen as the target vehicle. Random perturbations are applied to the target vehicle's position, heading, and speed. Each scenario is run 100 times to obtain average results.

**Design Motivation**: Euro NCAP is an industry-standard vehicle safety assessment protocol. Although these scenario types are rare in normal driving, they are highly prone to collisions, representing necessary conditions for safety assessment. Random perturbations enhance testing robustness.

#### 3. NeuroNCAP Rating — Collision-Focused Evaluation Metric

**Function**: Design a 5-star rating system based on collision outcomes to replace displacement error metrics.

**Core Formula**:

$$\text{NNS} = \begin{cases} 5.0 & \text{无碰撞} \\ 4.0 \cdot \max(0, 1 - v_i / v_r) & \text{有碰撞} \end{cases}$$

where $v_i$ is the relative impact speed at the moment of collision, and $v_r$ is the reference impact speed if no action is taken. Completely avoiding the collision yields 5 stars; colliding but successfully decelerating is rated from 0 to 4 stars according to the deceleration ratio.

**Design Motivation**: Traditional displacement error metrics fail to reflect driving safety. Zero collisions is the baseline requirement for safe driving, but even in the event of a collision, deceleration can significantly reduce severity—which is exactly how Euro NCAP structures its scores.

#### 4. Vehicle Dynamics Model — Kinematic Bicycle Model

**Core Formula**:

$$\frac{dS}{dt} = \begin{pmatrix} v\cos\theta \\ v\sin\theta \\ \frac{v\tan\delta}{L} \\ a \end{pmatrix}$$

The state $S = (x, y, \theta, v)^T$ contains longitudinal/lateral position, heading, and velocity. $L$ is the wheelbase, while $\delta$ and $a$ are control signals. An LQR controller is used to convert waypoint sequences into control signals.

### Loss & Training

- The NeRF renderer uses the NeuRAD-L configuration, trained for 100k steps with default hyperparameters, combined with pose optimization to compensate for missing vertical displacement information in the nuScenes dataset.
- End-to-end planners (UniAD/VAD) directly use the pre-trained weights released by the authors without any modifications.
- Collision avoidance post-processing: UniAD originally contains a trajectory optimization step based on predicted occupancy grids; a similar post-processing is implemented for VAD (rasterizing predicted future objects into occupancy grids).

## Key Experimental Results

### Main Results

**NeuroNCAP Safety Assessment Results**

| Model | Post-processing | NNS Avg↑ | NNS Stationary | NNS Frontal | NNS Side | Collision Rate Avg↓ | Collision Rate Stationary | Collision Rate Frontal | Collision Rate Side |
|------|--------|----------|----------|----------|----------|------------|-----------|-----------|-----------|
| Baseline-U | - | 2.65 | 4.72 | 1.80 | 1.43 | 69.9% | 9.6% | 100% | 100% |
| Baseline-V | - | 2.67 | 4.82 | 1.85 | 1.32 | 68.7% | 6.0% | 100% | 100% |
| UniAD | ✗ | 0.73 | 0.84 | 0.10 | 1.26 | 88.6% | 87.8% | 98.4% | 79.6% |
| VAD† | ✗ | 0.66 | 0.47 | 0.04 | 1.45 | 92.5% | 96.2% | 99.6% | 81.6% |
| UniAD† | ✓ | 1.84 | 3.54 | 0.66 | 1.33 | 68.7% | 34.8% | 92.4% | 78.8% |
| VAD | ✓ | 2.75 | 3.77 | 1.44 | 3.05 | 50.7% | 28.7% | 73.6% | 49.8% |

**Key Findings**:
- End-to-end planners **suffer from collision rates as high as 88-92% without post-processing**, causing rampant collisions even in simple stationary scenarios.
- The naive baseline (a simple constant velocity + brake logic based on forward obstacles from perception outputs) achieves near-perfect scores in the stationary scenario (collision rate 6-9.6%), proving that perception itself is functioning properly.
- Post-processing significantly reduces collision rates in stationary scenarios (UniAD: 87.8% $\rightarrow$ 34.8%), but offers limited assistance in frontal and side scenarios.

### Ablation Study

**Open-loop Real-to-Sim Gap**

| Model | Data Type | ADE@1s↓ | ADE@2s↓ | ADE@3s↓ | CR@1s↓ | CR@2s↓ | CR@3s↓ | NDS↑ |
|------|----------|---------|---------|---------|--------|--------|--------|------|
| UniAD | Real Images | 0.44 | 0.75 | 1.16 | 0.00 | 0.12 | 0.21 | 0.490 |
| UniAD | Rendered Images | 0.47 | 0.80 | 1.24 | 0.00 | 0.12 | 0.24 | 0.489 |
| VAD | Real Images | 0.43 | 0.71 | 1.01 | 0.00 | 0.08 | 0.11 | 0.449 |
| VAD | Rendered Images | 0.44 | 0.76 | 1.16 | 0.00 | 0.00 | 0.08 | 0.413 |

The Real-to-Sim gap is extremely small—displacement errors, collision rates, and detection performance remain almost unchanged on rendered images, verifying that the NeRF rendering quality is sufficient.

**Target Vehicle Perception Recall in Closed-Loop Scenarios**

| Scenario | Model | Recall@0s (5-15m) | @0s (15-25m) | @1s (5-15m) | @2s (5-15m) | @3s (5-15m) |
|------|------|-------------------|--------------|-------------|-------------|-------------|
| Stationary | UniAD | 0.97 | 0.98 | 0.94 | 0.94 | 0.94 |
| Stationary | VAD | 1.00 | 0.96 | 0.97 | 0.93 | 0.91 |
| Frontal | UniAD | 0.83 | 0.97 | 0.82 | 0.80 | 0.77 |
| Side | UniAD | 0.92 | 0.96 | 0.92 | 0.90 | 0.72 |

In the most critical 5-25m range, the model's perception recall is >80%, **proving that the high collision rate is not caused by perception failure, but rather represents a fundamental flaw in the planning module**.

### Key Findings

1. **Planning-Perception Contradiction of End-to-End Planners**: The models successfully detect and predict the position and motion of the target vehicle with high accuracy, yet output trajectories that collide directly into it. This exposes a severe disconnect between perception capability and planning decisions.
2. **Double-Edged Sword of Post-processing**: Trajectory post-processing (occupancy-grid-based optimization) manages to avoid collisions in some scenarios but triggers catastrophic outcomes in others—such as waypoints being "repelled" by obstacles to the opposite side, causing the vehicle to accelerate and steer right into the target.
3. **Simple Logic vs. Deep Learning**: The naive baseline based on perception outputs (braking if an object is ahead) achieves near-perfect performance in stationary scenarios, highlighting the insufficient safety learning in end-to-end planners under safety-critical scenarios.
4. **Misleading Nature of Open-Loop Evaluation**: Models that demonstrate tiny displacement errors in open-loop evaluation exhibit collision rates exceeding 88% in closed-loop settings.

## Highlights & Insights

1. **The Proper Use of NeRF in Safety Testing**: Instead of using NeRF to train better models, it is leveraged to create testing scenarios that cannot be safely captured in the real world—marking one of the most valuable applications of NeRF in autonomous driving.
2. **Computer-Vision-izing Euro NCAP**: Introducing industry safety assessment standards into academic model evaluation bridges the gap between academic benchmarks and industrial safety requirements.
3. **A Wake-up Call for End-to-End Planning**: The core contribution of this work lies not in methodological innovation, but in its profound experimental findings—that current SOTA end-to-end planners are fundamentally unsafe in safety-critical scenarios. This serves as an important warning for the future direction of the field.
4. **Value of the Open-Source Framework**: Fully open-sourcing the simulator and scenario suites lowers the barrier of entry for safety evaluations.

## Limitations & Future Work

1. NeRF cannot render weather conditions not present in the training data (such as rainy days), and large drift trajectories or close-range objects may generate artifacts.
2. Simplifications in the kinematic vehicle model do not account for physical factors such as delay, friction, and suspension.
3. Inability to handle deformable objects (such as pedestrians) limits testing on scenarios involving vulnerable road users.
4. Target vehicles move along predefined trajectories and do not adapt reactively to the ego-vehicle's behaviors.
5. All models utilize the same LQR controller, whereas controllers and planners are tightly coupled in real-world deployments.
6. The study is based on only 14 sequences from nuScenes, which limits scenario diversity.

## Related Work & Insights

- **UniAD [Hu et al., CVPR 2023]** / **VAD [Jiang et al., ICCV 2023]**: The two SOTA end-to-end planners evaluated in this work.
- **NeuRAD [Tonderski et al., CVPR 2024]**: The neural renderer used by NeuroNCAP, specifically designed for autonomous driving.
- **nuPlan [Caesar et al., 2021]**: An object-level closed-loop simulator that does not generate sensor data.
- **Euro NCAP**: The industry standard for automotive safety assessment, serving as the inspiration for NeuroNCAP's scenario design.
- **Dauner et al., CoRL 2023**: An important study pointing out the weak correlation between open-loop evaluation and actual driving quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The integration of the NeRF closed-loop testing framework is novel, and the Euro NCAP-inspired scenario design holds practical significance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive quantitative evaluations (3 scenario types $\times$ multiple configurations $\times$ 100 random seeds), detailed verification of the Real-to-Sim gap, and deep qualitative analysis revealing failure modes.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logic, striking and convincing experimental findings, with excellent chart/figure layouts.
- **Value**: ⭐⭐⭐⭐⭐ — Possesses far-reaching cautionary implications for end-to-end autonomous driving research; the open-source framework can serve as a standardized evaluation tool in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[ICLR 2026\] BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](../../ICLR2026/autonomous_driving/bridgedrive_diffusion_bridge_policy_for_closed-loop_trajectory_planning_in_auton.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](../../CVPR2025/autonomous_driving/drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)

</div>

<!-- RELATED:END -->
