---
title: >-
  [Paper Note] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping
description: >-
  [ICLR 2026][3D Vision][real-to-sim-to-real] D-REX is proposed as a Gaussian-based differentiable real-to-sim-to-real engine that performs end-to-end object mass identification from visual observations and robot control s…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "real-to-sim-to-real"
  - "differentiable physics simulation"
  - "mass identification"
  - "dexterous grasping"
  - "Gaussian representation"
date: 2026-05-08
content_hash: f6bc6a1c44b4b6db
---

# D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping

**Conference**: ICLR 2026  
**arXiv**: [2603.01151](https://arxiv.org/abs/2603.01151)  
**Code**: [drex.github.io](https://drex.github.io)  
**Area**: 3D Vision / Robotic Manipulation  
**Keywords**: real-to-sim-to-real, differentiable physics simulation, mass identification, dexterous grasping, Gaussian representation

## TL;DR
D-REX is proposed as a Gaussian-based differentiable real-to-sim-to-real engine that performs end-to-end object mass identification from visual observations and robot control signals, and leverages the identified mass for force-aware dexterous grasping policy learning, effectively bridging the sim-to-real gap.

## Background & Motivation

**Background**: Simulation serves as a core platform for robot policy learning, yet the sim-to-real gap remains a fundamental challenge. Existing approaches include domain randomization, system identification, domain adaptation, and digital twin construction, each with its own limitations.

**Limitations of Prior Work**:
   - Constructing accurate digital twins requires integrating multiple pipelines such as geometric reconstruction and parameter identification, resulting in high complexity.
   - Estimating physical properties (e.g., mass) from visual observations is extremely difficult; initial estimates from SAM2/VLM typically deviate significantly from ground truth.
   - Existing grasping policies rely solely on position control and neglect force control — the same grasp pose can produce drastically different outcomes for objects of varying mass.
   - Non-differentiable simulators preclude backpropagation for parameter optimization.

**Key Challenge**: High-fidelity physical simulation requires accurate physical parameters, yet such parameters are difficult to obtain from visual observations; grasping policies require force control, but the appropriate force magnitude depends on unknown object mass.

**Goal**: (1) Identify object mass from robot–object interaction videos; (2) Use the identified mass for force-aware dexterous grasping policy learning.

**Key Insight**: Leverage a differentiable physics engine to backpropagate through simulated trajectories and optimize mass parameters so that simulated trajectories match real trajectories.

**Core Idea**: Construct a digital twin using differentiable simulation and Gaussian representation; identify mass via trajectory matching; subsequently learn a force–position hybrid grasping policy conditioned on the identified mass.

## Method

### Overall Architecture
A four-stage pipeline: (1) Real-to-Sim — reconstruct the visual and geometric models of the scene and objects from RGB video; (2) Mass Identification — identify object mass from robot–object interaction using a differentiable physics engine; (3) Human Demonstration Transfer — convert human demonstration videos into executable robot trajectories; (4) Policy Learning — train a force-aware grasping policy conditioned on the identified mass.

### Key Designs

1. **Visual and Geometric Reconstruction**

    - Function: Construct collision meshes and high-fidelity rendering representations from RGB video.
    - Mechanism: Videos captured by a smartphone are processed to train two sets of Gaussian primitives — 2D Gaussians (with normal estimation) provide accurate collision geometry, while 3D Gaussians ensure high-fidelity rendering. The outputs are a collision mesh $\mathcal{K}$ and Gaussian particles $\mathcal{P}$.
    - Design Motivation: Collision detection demands precise geometry, while visual supervision requires photorealistic rendering; since these two requirements differ, they are modeled separately.

2. **Differentiable Physics Engine and Mass Identification**

    - Function: Identify object mass from robot–object pushing interactions.
    - Mechanism: Identical pushing actions are executed in the real world and in simulation, collecting object trajectories $\{\mathbf{s}_t^{real}\}$ and $\{\mathbf{s}_t^{sim}(m)\}$. The following trajectory loss is minimized:
      $\min_{m>0} \mathcal{L}_{traj}(m) = \sum_{t=1}^T \|\mathbf{s}_t^{sim}(m) - \mathbf{s}_t^{real}\|_2^2$
    - Semi-implicit Euler integration is used for state updates $G([\mathbf{s}_t, \mathbf{u}_t], m, \theta)$, with a compliant penalty-based contact model; the entire computation graph is differentiable.
    - Gradients are backpropagated to the mass parameter via automatic differentiation: $\frac{\partial \mathcal{L}}{\partial m} = \sum_t \frac{\partial \mathcal{L}}{\partial \mathbf{s}_t^{sim}} \cdot \frac{\partial \mathbf{s}_t^{sim}}{\partial \mathbf{M}_t} \cdot \frac{\partial \mathbf{M}_t}{\partial m}$
    - Real object pose is obtained via FoundationPose.
    - Design Motivation: Direct optimization from interaction data eliminates the need for manual external force specification (cf. GradSim), and consistent robot control signals are used to model external forces.

3. **Human-to-Robot Demonstration Transfer**

    - Function: Convert human hand manipulation videos into robot-executable trajectories.
    - Mechanism: HaMeR and MCC-HO reconstruct hand joints and object 6-DoF pose from video frames → Dex-Retargeting maps these to robot hand degrees of freedom → output joint-angle actions $\mathbf{A}_t \in \mathbb{R}^{J_r}$.

4. **Force-Aware Policy Learning**

    - Function: Train a dexterous grasping policy that jointly controls position and force.
    - Mechanism: A multi-head network $\pi_\phi$ takes object collision mesh vertices (after positional encoding) as input and predicts three outputs: joint positions $\hat{\mathbf{A}} \in \mathbb{R}^{16}$, contact constraints $\hat{\mathbf{r}} \in \mathbb{R}^2$, and grasp force constraints $\hat{\mathbf{f}} = \frac{m \cdot g}{n_{active}}$.
    - Force constraints are computed based on the identified mass $m$, distributing gravitational load evenly across active contact points.
    - Two-stage training: position control is trained first, followed by retraining with force control constraints.

### Loss & Training
Mass identification uses trajectory MSE loss with Adam optimization for approximately 200 epochs (5–20 minutes). Policy training employs supervised learning on demonstration data with contact constraint losses.

## Key Experimental Results

### Mass Identification

| Object | VLM-Inferred Mass | Identified Mass | Ground Truth | Error% |
|--------|-------------------|-----------------|--------------|--------|
| Letter U | 500g | 110g | 125g | 12.0% |
| Letter A | 500g | 145g | 134g | 9.0% |
| Lego | 300g | 53g | 59g | 8.6% |
| Cookie | 500g | 200g | 210g | 4.8% |
| Ketchup | 1000g | 667g | 726g | 8.1% |

Experiment with identical geometry but varying density: identification errors across three density levels are all within 13g.

### Grasping Experiments

| Method | Overall Performance |
|--------|---------------------|
| DexGraspNet 2.0 | Low grasping success rate with high variance |
| Human2Sim2Robot | Performance degrades significantly as object mass increases |
| **D-REX** | Consistently high success rate and low variance across 8 objects |

Cross-evaluation shows that optimal policy performance is achieved only when the training mass matches the evaluation mass (75–95% when matched vs. 15–40% when mismatched), confirming the necessity of force control.

### Ablation Study
- Force-conditioned policy vs. position-only policy: force conditioning consistently outperforms across all objects.
- Identified mass vs. ground-truth mass: policies using identified mass perform comparably to those using ground-truth mass, and substantially outperform those using random mass.
- The pushing task design (virtual fulcrum + reduced friction) is critical for accurate mass identification.

## Highlights & Insights
- D-REX is the first to integrate differentiable physics simulation with Gaussian representation for mass identification within a real-to-sim-to-real framework.
- Force–position hybrid policy learning is an important complement to pure position-based policies, and the experimental results are convincing.
- The large discrepancy between VLM-inferred and ground-truth mass (e.g., 500g vs. 125g) underscores the necessity of physics-based identification.
- The end-to-end pipeline from video to digital twin to policy deployment is complete and practically usable.

## Limitations & Future Work
- Mass identification relies on simple pushing interactions and may not generalize to all object types.
- The reconstruction pipeline requires 30–35 minutes per object and mass identification requires 5–20 minutes, limiting real-time deployment.
- Only mass is identified; other physical parameters such as friction coefficients are not addressed.
- Demonstration transfer depends on the quality of hand estimation from HaMeR/MCC-HO, which may be unreliable under heavy occlusion.
- Evaluation is limited to tabletop grasping; more complex manipulation tasks (e.g., pouring, tool use) remain unvalidated.
- Stiffness and damping parameters of the compliant contact model still require manual specification.

## Related Work & Insights
- vs. GradSim: Both use differentiable simulation for system identification, but D-REX directly leverages robot control signals rather than manually specifying external forces, making it more practical.
- vs. DexGraspNet 2.0: Trained on large-scale simulation data but without force control, rendering it insensitive to mass variation.
- vs. Human2Sim2Robot: Learns from human videos but relies on position control only, causing objects with high mass to be dropped.
- vs. Gaussian-based digital twin methods: Most focus solely on visual reconstruction; D-REX further performs physical parameter identification.

The combination of differentiable simulation and visual representation can be extended to identify additional physical parameters (e.g., moment of inertia, stiffness). The force-aware policy learning framework is generalizable to broader manipulation tasks. Building on the experimental finding that "mass matters," future work may explore the influence of additional physical attributes on policy performance.

## Rating
- Novelty: ⭐⭐⭐⭐ (complete pipeline integrating differentiable simulation, Gaussian representation, and force-aware policy learning)
- Experimental Thoroughness: ⭐⭐⭐⭐ (real-world validation with multi-dimensional ablation)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (meaningful advancement for the sim-to-real community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](../../ICCV2025/3d_vision/embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](../../ICCV2025/3d_vision/radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[CVPR 2026\] AnthroTAP: Learning Point Tracking with Real-World Motion](../../CVPR2026/3d_vision/anthrotap_learning_point_tracking_with_real-world_motion.md)
- [\[ICLR 2026\] FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)

</div>

<!-- RELATED:END -->
