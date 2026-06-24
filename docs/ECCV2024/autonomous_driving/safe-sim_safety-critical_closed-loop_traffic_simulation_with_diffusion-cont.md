---
title: >-
  [Paper Note] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries
description: >-
  [ECCV 2024][Autonomous Driving][Safety-critical simulation] Safe-Sim proposes a closed-loop safety-critical simulation framework based on diffusion models. By introducing an adversarial term and a Partial Diffusion mechanism into the diffusion denoising process, it achieves fine-grained control over adversarial vehicle behavior types (collision angles, relative velocities, and collision categories), validating its effective assessment capabilities over multiple planners on nu…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Safety-critical simulation"
  - "diffusion models"
  - "adversarial generation"
  - "closed-loop simulation"
  - "controllability"
date: 2026-05-08
content_hash: d86de2d308068dc9
---

# Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries

**Conference**: ECCV 2024  
**arXiv**: [2401.00391](https://arxiv.org/abs/2401.00391)  
**Code**: [https://safe-sim.github.io/](https://safe-sim.github.io/) (with project page)  
**Area**: Autonomous Driving  
**Keywords**: Safety-critical simulation, diffusion models, adversarial generation, closed-loop simulation, controllability

## TL;DR

Safe-Sim proposes a closed-loop safety-critical simulation framework based on diffusion models. By introducing an adversarial term and a Partial Diffusion mechanism into the diffusion denoising process, it achieves fine-grained control over adversarial vehicle behavior types (collision angles, relative velocities, and collision categories), validating its effective assessment capabilities over multiple planners on nuScenes and nuPlan.

## Background & Motivation

**Background**: The key safety capability of autonomous vehicles (AVs) lies in handling near-collision events. However, such events are extremely rare on real roads, and actively testing these scenarios on public roads is neither safe nor legal. Therefore, simulation is an indispensable tool for evaluating AV safety, and modeling the behaviors of other road users is the core of traffic simulation.

**Limitations of Prior Work**:
1. **Manual scenario design is not scalable**: Traditional methods rely on human-designed failure-prone scenarios, which have limited coverage.
2. **Open-loop is unrealistic**: Most existing automated generation methods focus on static scenario generation rather than dynamic closed-loop execution. Other agents do not react to the planner's behavior, making it impossible to truly test interactive capabilities.
3. **Lack of controllability**: Existing methods usually generate a single adversarial outcome for each scenario, failing to explore different conditions and response modes. Researchers cannot control the collision type (head-on, side, rear-end) and severity.

**Key Challenge**: The framework must simultaneously satisfy three requirements: safety-criticality (generating collision scenarios), closed-loop reactivity (all agents responding to the planner), and controllability (adjustable adversarial behavior modes). No prior work has achieved all three simultaneously.

**Key Insight**: Leveraging the controllable generation capabilities of diffusion models—injecting adversarial objectives via guidance, controlling collision types via partial diffusion, and ensuring realism by training on real-world datasets.

**Core Idea**: The safety-critical simulation is decomposed into a two-level control: the guidance objective layer (adversarial collision + regularization constraints) to control the adversarial agent's aggressiveness towards the ego vehicle, and the partial diffusion layer to control the specific collision type through trajectory proposal initialization. Combining both enables unprecedented controllable adversarial simulation.

## Method

### Overall Architecture

In each simulation step, Safe-Sim: (1) controls the ego vehicle using the planner $\pi$ under test to produce a future trajectory; (2) generates trajectories for all non-ego reactive agents using a diffusion-based model $g_\theta$, where one or more agents are designated as adversarial agents; (3) injects adversarial and control targets into the adversarial agents through the guided diffusion process, while non-adversarial agents are guided only by regularization to maintain realistic behavior; (4) updates observations and replan after executing the first few steps, forming a closed loop. The model is trained on real-world driving datasets (nuScenes/nuPlan).

### Key Designs

1. **Guided Adversarial Diffusion**:
    - **Function**: Perturbs the predicted trajectory using guidance gradients during each denoising step to drive the adversarial agent towards colliding with the ego vehicle.
    - **Mechanism**: Employs reconstruction guidance (using clean data guidance) to inject guidance gradients into the estimated clean trajectory $\hat{\tau}_0$ at each denoising step:
     $$\tilde{\tau}_0 = \hat{\tau}_0 - \alpha \Sigma_k \nabla_{\tau_k} J(\hat{\tau}_0)$$
     The total guidance objective consists of adversarial terms and regularization terms:
     $$J(\tau) = \rho \underbrace{(J_{\text{coll}} + J_v + J_{\text{ttc}})}_{J_{\text{adv}}} + \underbrace{J_{\text{route}} + J_{\text{Gauss}}}_{J_{\text{reg}}}$$
     where $\rho$ is a scalar weight. For adversarial agents, $\rho > 0$, and for non-adversarial agents, $\rho = 0$.
    - **Design Motivation**: Trained on real-world data, the diffusion model naturally reflects normal driving distributions. The guidance mechanism injects adversarial behaviors without completely deviating from realism. The weight $\rho$ easily distinguishes between adversarial and non-adversarial agents without requiring separate model training.

2. **Collision Guidance Objective $J_{\text{coll}}$**:
    - **Function**: Encourages the adversarial agent to collide with the ego vehicle.
    - **Mechanism**: Minimizes the sum of distances between the adversarial agent and the ego vehicle over the trajectory planning horizon:
     $$J_{\text{coll}} = -\sum_{t=1}^T d(t)$$
    - **Design Motivation**: This is the most direct collision promotion objective. Adversarial agents are dynamically selected based on lane proximity or distance.

3. **Safety-Criticality Control Objectives $J_v$ and $J_{\text{ttc}}$**:
    - **Function**: Controls the relative velocity and urgency of the collision, adjusting the risk level of the scenario.
    - **Mechanism**: The TTC cost uses a Gaussian kernel of time-to-collision and crash distance under a constant velocity assumption:
     $$J_{\text{ttc}} = \sum_{t=1}^T -\exp\left(-\frac{\tilde{t}_{\text{col}(t)}^2}{2\lambda_t} - \frac{\tilde{d}_{\text{col}(t)}^2}{2\lambda_d}\right)$$
     The TTC cost tends to produce scenarios with high relative velocities and awkward collision angles that are difficult for the ego vehicle to evade.
    - **Design Motivation**: Different weights of TTC cost yield collision scenarios of varying severity, enabling continuously adjustable safety-criticality.

4. **Route Guidance $J_{\text{route}}$ and Gaussian Collision Guidance $J_{\text{Gauss}}$**:
    - **Function**: Regularization terms—route guidance prevents agents from going off-road, and Gaussian collision guidance prevents collisions among non-adversarial agents.
    - **Mechanism**: Route guidance penalizes deviations exceeding a predefined path margin $d_m$:
     $$J_{\text{route}} = \sum_{t=1}^T \max(0, |d_n(\tau_t, r) - d_m|)$$
     Gaussian collision guidance considers both tangential and normal distances, proving more effective than simple disc approximations:
     $$J_{\text{Gauss}} = \sum_{t=1}^T \sum_{i,j}^N \exp\left(-\frac{1}{2\sigma^2}(\lambda \cdot d_t^{ij}(t)^2 + d_n^{ij}(t)^2)\right)$$
    - **Design Motivation**: (1) Route guidance constrains agents to predefined lanes more accurately than prior off-road loss formulations; (2) Gaussian collision distance penalizes collisions more realistically by accounting for non-circular vehicle shapes, significantly reducing background collision rates.

5. **Partial Diffusion**:
    - **Function**: Initializes the diffusion process with trajectory proposals to control the collision type (head-on, side, rear-end, etc.).
    - **Mechanism**: A three-step process: (1) generate rule-based initial trajectory proposals $\tau_0$ for different collision types (by finding the intersection of ego and adversarial agent centerlines, selecting acceleration and offsets); (2) choose a partial diffusion ratio $\gamma$ to determine the starting step $k_p = \gamma \cdot K$, and add corresponding noise to the proposal: $\hat{\tau}_{k_p} = \sqrt{\bar{\alpha}_{k_p}}\tau_0 + \sqrt{1 - \bar{\alpha}_{k_p}}\epsilon$; (3) execute guided denoising from $k_p$ to step 0 to obtain realistic trajectories.
    - **Design Motivation**: Pure guidance-based optimization struggles to control specific collision styles since it only optimizes in one direction (minimizing distance). Partial diffusion utilizes proposals to "hint" at the collision type, which the diffusion model then "realizes" for realism. $\gamma$ adjusts the trade-off between user control and the model's data distribution.

### Loss & Training

- The diffusion model is trained using standard DDPM with $K=100$ steps.
- Trajectories are represented as action sequences $\tau_a = [a_0, ..., a_{T-1}]$, with state sequences derived from the initial state and unicycle kinematics.
- Scenario encoding uses an agent-centric rasterized map + ResNet encoder.
- Trajectory processing employs a UNet architecture with 1D temporal convolutional blocks.
- Planner and reactive agents update plans at a frequency of 2Hz.

## Key Experimental Results

### Main Results

**Safety-Critical Simulation Comparison (Rule-Based Planner, nuScenes):**

| Method | Collision Rate ↑ | Other Agent Off-road ↓ | Adv. Agent Off-road ↓ | Rel. Collision Velocity | Realism ↓ | Time (s) ↓ |
|------|---------|-------------|-------------|-------------|---------|---------|
| STRIVE | 36.4% | 2.2% | 11.4% | 5.52 | 0.85 | 427.2 |
| DiffScene | 18.2% | 11.4% | 9.0% | 16.4 | 0.52 | 105.4 |
| **Safe-Sim** | **43.2%** | **1.8%** | 11.4% | **-0.12** | **0.38** | **104.5** |

Safe-Sim achieves the highest collision rate and the best realism, while running at nearly 4x the speed of STRIVE.

**Results on nuPlan:**

| Method | Collision Rate ↑ | Realism ↓ |
|------|---------|--------|
| DiffScene | 56.7% | 0.42 |
| **Safe-Sim** | **80%** | **0.27** |

**Evaluation of Different Planners:**

| Planner | Ego-Adv Collision Rate | Ego-Other Collision Rate | Adv. Agent Off-road Rate | Realism |
|---------|-------------|---------------|-------------|--------|
| IDM | 49.3% | 58.2% | 3.0% | 0.78 |
| BC | 38.8% | 37.3% | 9.0% | 0.79 |
| PDM-Closed | 26.9% | 50.7% | 1.5% | 0.86 |
| BITS | 16.4% | 19.4% | 6.0% | 0.79 |

### Ablation Study

**Impact of TTC Weight on Collision Control:**

| TTC Weight | TTC Cost | TTC (s) | Collision Velocity (m/s) | Collision Angle (deg) | Collision Rate ↑ |
|---------|--------|--------|-------------|-------------|---------|
| 0.0 | 0.18 | 2.45 | -7.43 | – | 48.2% |
| 1.0 | 0.21 | 2.30 | 0.43 | – | 53.6% |
| 2.0 | 0.26 | 3.78 | -17.0 | – | 60.7% |

Increasing the TTC weight not only improves the collision rate but also alters the collision angle, making evasion harder for the ego vehicle.

**Controllability Ablation (Collision Diversity):**

| Configuration | Collision Angle Var ↑ | Collision Velocity Var ↑ | Collision Point Var ↑ |
|------|-------------|-------------|-----------|
| $J_{\text{adv}}$ only | 3.34 | 4.81 | 2.47 |
| $J_{\text{adv}} + J_{\text{reg}}$ | 2.22 | 2.99 | 1.62 |
| $J_{\text{adv}} + J_{\text{reg}}$ + Partial Diff. | **3.10** | **1.96** | **5.44** |

Partial Diffusion significantly increases the diversity of collision points (variance increases from 1.62 to 5.44), validating that the trajectory proposals successfully guide the geometric style of collisions.

### Key Findings

- The introduction of $J_{\text{reg}}$ reduces the collision rate (from 53.5% to 23.9%) but dramatically improves realism and reduces off-road rates.
- The partial diffusion ratio $\gamma$ controls the trade-off between proposal fidelity and realism; at $\gamma=0$, the collision rate is highest but trajectories are less natural.
- After being trained on nuScenes, Safe-Sim can transfer to nuPlan without fine-tuning.
- The Gaussian collision distance (considering tangential/normal aspects) significantly reduces collisions between non-adversarial agents compared to disc approximations.

## Highlights & Insights

1. **The first simulation framework that simultaneously achieves safety-criticality, closed-loop reactivity, and controllability**. The comparison in Table 1 is clear at a glance, filling an important gap in the literature.
2. **Elegant design of Partial Diffusion**—combines rule-based collision type proposals with a data-driven diffusion model. Proposals provide the "intent," and diffusion guarantees "realism."
3. **Comprehensive guidance objective design**: Adversarial terms (collision + TTC + velocity) + regularization terms (route + Gaussian collision), each having clear physical meanings.
4. **Multi-planner evaluation** demonstrates the generality of the framework—it can be used to compare and evaluate the safety performance of different planners.

## Limitations & Future Work

- Adversarial agents sometimes collide unrealistically with non-adversarial agents before reaching the ego vehicle.
- In some collision scenarios, the ego is not the "at-fault" party; generating more ego-at-fault scenarios would be more valuable for evaluation.
- Trajectory proposals are currently based on simple rules (centerline intersections); proposal generation for more complex scenarios (e.g., roundabouts) needs improvement.
- Closed-loop policy training using this framework has not yet been explored (currently used only for evaluation).
- Automated parameter search for adversarial behavior (instead of manual weight tuning) is an important direction for future work.

## Related Work & Insights

- Compared with STRIVE (adversarial optimization in VAE latent space), Safe-Sim directly guides diffusion in the trajectory space, which is more flexible and offers stronger controllability.
- Compared with CTG/CTG++ (controllable generation using diffusion but not safety-critical), Safe-Sim adds adversarial objectives and partial diffusion.
- Compared with DiffScene (safety-critical diffusion but open-loop), Safe-Sim achieves closed-loop interaction.
- **Insight**: The guidance mechanism of diffusion models is naturally suited for safety-critical simulation—different types of adversarial behaviors can be "programmed" via different combinations of guidance objectives.

## Rating

- Novelty: ⭐⭐⭐⭐ Partial diffusion and collision type control are novel contributions, though the guidance objective design represents a combination of established concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with dual datasets, multiple planners, controllability validation, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, intuitive architecture diagrams, and a highly convincing Table 1.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to AV safety evaluation with excellent engineering utility in the framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving](neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](../../CVPR2025/autonomous_driving/drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[CVPR 2025\] Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models](../../CVPR2025/autonomous_driving/closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)
- [\[ECCV 2024\] RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion](roofdiffusion_constructing_roofs_from_severely_corrupted_point_data_via_diffusio.md)

</div>

<!-- RELATED:END -->
