---
title: >-
  [Paper Note] Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments
description: >-
  [CVPR 2025][Robotics][Deformable Linear Objects] This paper proposes a quasi-static trajectory optimization framework based on the Globally Variational Strain (GVS) parameterized Cosserat rod model for dual-arm coordinated manipulation of hybrid deformable-rigid linear objects (hDLO) in constrained environments. By leveraging analytical gradients, the solver achieves a 33x speedup over finite differences, and a ~3cm deformation error is validated on a real dual-arm platform.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Deformable Linear Objects"
  - "Cosserat Rod Model"
  - "Trajectory Optimization"
  - "Dual-Arm Coordination"
  - "Constrained Environments"
date: 2026-05-08
content_hash: 237d0afd365e2bc0
---

# Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments

**Conference**: CVPR 2025  
**arXiv**: [2603.12940](https://arxiv.org/abs/2603.12940)  
**Code**: None  
**Area**: Robotic Manipulation / Deformable Objects  
**Keywords**: Deformable Linear Objects, Cosserat Rod Model, Trajectory Optimization, Dual-Arm Coordination, Constrained Environments

## TL;DR

This paper proposes a quasi-static trajectory optimization framework based on the Globally Variational Strain (GVS) parameterized Cosserat rod model for dual-arm coordinated manipulation of hybrid deformable-rigid linear objects (hDLO) in constrained environments. By leveraging analytical gradients, the solver achieves a 33x speedup over finite differences, and a ~3cm deformation error is validated on a real dual-arm platform.

## Background & Motivation

**Background**: The manipulation of deformable linear objects (DLOs, such as cables and ropes) is an active research direction in robotics. Numerous works focus on pure DLO manipulation planning, using sampling-based methods (RRT/PRM) or data-driven methods (RL/IL).

**Limitations of Prior Work**: (1) In real-world applications, objects are often hybrids of deformable and rigid components (e.g., cables with connectors, jointed flexible mechanisms), making pure DLO models inapplicable; (2) Sampling-based methods do not guarantee trajectory optimality, and random sampling may violate physical constraints; (3) Optimization-based methods are computationally expensive due to the high-dimensional configuration space and non-linear mechanics, and are highly dependent on initial guesses.

**Key Challenge**: The configuration space of an hDLO is theoretically infinite-dimensional (each point on the DLO has continuous pose variations), requiring a modeling framework that can both reduce dimensions and uniformly represent both deformable and rigid bodies.

**Goal**: To provide an optimization-based planning scheme for dual-arm coordinated manipulation of hDLOs in constrained environments, achieving highly efficient solving by leveraging the structural advantages and analytical gradients of the GVS model.

**Key Insight**: The GVS model naturally unifies the continuous deformation of DLOs and rigid joints using strain parameters, providing a finite-dimensional configuration space and analytical Jacobians, which makes gradient-based optimization feasible.

**Core Idea**: Reduce the infinite-dimensional configuration space of hDLOs to a finite dimension using the GVS strain model, solve inverse kinematics-statics (IKS) to provide a warm start, and then perform trajectory optimization satisfying environmental constraints.

## Method

### Overall Architecture

Input: hDLO model (topology and physical parameters of DLO + rigid links) + dual-arm manipulators + constrained environment (circular passage) + target end-effector poses. The solution is divided into two stages: (1) Inverse Kinematics-Statics (IKS): finding static equilibrium configurations that satisfy targets; (2) Trajectory Optimization: using the IKS solution as a warm start to solve for a time sequence that satisfies environmental constraints.

### Key Designs

1. **GVS Strain-Parameterized Modeling**:

    - **Function**: Reduces the infinite-dimensional configuration space of hDLOs to a finite dimension, uniformly representing both deformation and rigid-body motion.
    - **Mechanism**: The Cosserat rod model describes the pose of a DLO as a continuous mapping $g(X) \in SE(3)$ along the arc length. GVS parameterizes the strain field using limited basis functions as $\xi = \Phi(X)q + \xi^*$, where $q$ represents the generalized coordinates. Rigid joints are special cases (where $\Phi$ does not depend on $X$). Closed-chain constraints are formulated using $e_c(q) = \log(g_A^{-1}g_B) = 0$ in $\mathfrak{se}(3)$, and the complete static equilibrium condition is $Kq - F - Bu - A^T\lambda = 0$.
    - **Design Motivation**: Compared to FEM (maximal coordinates, high-dimensional), Discrete Elastic Rods (do not share joint abstractions), and data-driven methods (unexplainable), GVS provides a minimal coordinate representation and can directly reuse rigid-body kinematics.

2. **Analytical Gradient-Accelerated Optimization**:

    - **Function**: Solves IKS and trajectory optimization problems efficiently.
    - **Mechanism**: Utilizes recently proposed analytical derivatives of GVS statics (partial derivatives with respect to $q, u, \lambda$), avoiding finite differences. Environmental constraints (circular passage) are evaluated at any arc length by interpolating DLO positions in SE(3), with analytical derivatives propagated via the chain rule.
    - **Design Motivation**: Analytical gradients are 33 times faster than finite differences for the IKS problem, and trajectory optimization is completely infeasible using finite differences (requiring both warm starts and analytical gradients to converge).

3. **IKS Warm-Started Trajectory Optimization**:

    - **Function**: Generates feasible manipulation trajectories in constrained environments.
    - **Mechanism**: First solves IKS while ignoring environmental constraints (finding the target configuration) to obtain a linearly interpolated trajectory from start to target as an initial guess. Then, using this as a warm start, full constrained trajectory optimization (including circular passage constraints) is performed by discretizing the trajectory into $N$ steps and solving them simultaneously.
    - **Design Motivation**: Directly solving constrained trajectory optimization is non-convex and prone to divergence; IKS warm start significantly reduces the search space.

### Loss & Training

The optimization objective is to minimize actuator motion ($\sum \|u_{k+1} - u_k\|^2$). Constraints include: static equilibrium at each step, closed-chain constraints, environmental constraints (preventing the DLO from penetrating passage walls), and joint limits. The IPOPT solver is utilized.

## Key Experimental Results

### Main Results

| Metric | Ours (Optimization) | BiRRT | Description |
|------|----------|-------|------|
| 3-link hDLO, Goal Reached | ✓ | ✓ | Both methods succeeded |
| Average Deformation Error (Experiment) | ~3cm (5% link length) | N/A | Only the optimization method was evaluated in real-world experiments |
| IKS Solving Speed (Analytical) | 0.8s | N/A | Analytical gradient |
| IKS Solving Speed (Finite Difference) | 26.4s | N/A | 33x slower |
| Trajectory Optimization Feasibility | Feasible with analytical gradient | N/A | Finite difference fails to converge |

### Ablation Study

| hDLO Configuration | IKS Analytical (s) | IKS Finite Difference (s) | Speedup |
|-----------|-----------|---------------|-------|
| 2-link (1DLO+1rigid) | 0.3 | 8.2 | 27x |
| 3-link (2DLO+1rigid) | 0.8 | 26.4 | 33x |
| 5-link (3DLO+2rigid) | 2.1 | 65+ | 31x |

### Key Findings

- Analytical gradients are key to making trajectory optimization feasible—finite difference baselines completely fail to converge during the trajectory optimization stage.
- IKS warm-starting is crucial for trajectory optimization; without a warm start, optimization often gets trapped in local optima or fails to satisfy constraints.
- A deformation error of ~3cm on average (5% of link length) in real-world experiments validates the sim-to-real transferability.
- The trajectories found by the optimization method are smoother and shorter than those of BiRRT, though the computation time is longer.

## Highlights & Insights

- **Unification of the GVS Model**: A single framework handles both continuous deformation and discrete joints. This "minimal coordinate" representation paves the way for optimization-based methods and can be generalized to more robotic architectures.
- **Necessity of Analytical Gradients**: Not only does it provide a 33x speedup, but more importantly, it turns trajectory optimization from "infeasible" to "feasible." In high-dimensional non-linear systems, the value of analytical derivatives far exceeds simple acceleration.
- **Completeness of Practical Validation**: A complete validation pipeline from simulation to real-world dual-arm robotic experiments, with a convincing actual accuracy of 3cm error.

## Limitations & Future Work

- Only quasi-static motion is considered, which is inapplicable to scenarios requiring dynamic manipulation.
- The circular passage is a simplified constraint model; real-world environments (e.g., organs) have more complex shapes.
- Trajectory optimization may find local optima rather than global optima.
- DLO material parameters need to be calibrated beforehand, limiting generalization capabilities to unknown materials.
- Future work can integrate force control to realize online closed-loop adjustment.

## Related Work & Insights

- **vs. Sampling-based Methods (RRT/PRM)**: They do not guarantee optimality, and random sampling may violate physical constraints; this optimization-based method finds smooth, optimal trajectories.
- **vs. Data-driven Methods (RL/IL)**: They have low sample efficiency and do not guarantee physical feasibility; this work is based on precise physical models.
- **vs. FEM Methods**: Maximal coordinates lead to high-dimensional systems; the minimal coordinate representation of GVS is more suitable for optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ First to apply GVS analytical gradients to constrained trajectory optimization of hDLOs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + real experiments + comparison with BiRRT.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations and complete experimental design.
- Value: ⭐⭐⭐⭐ Practical application value for hDLO manipulation in industrial assembly and minimally invasive surgery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)
- [\[NeurIPS 2025\] Automaton Constrained Q-Learning](../../NeurIPS2025/robotics/automaton_constrained_q-learning.md)
- [\[ICML 2025\] Action-Constrained Imitation Learning](../../ICML2025/robotics/action-constrained_imitation_learning.md)
- [\[ICML 2026\] DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics](../../ICML2026/robotics/dlo-lab_benchmarking_deformable_linear_object_manipulations_with_differentiable_.md)
- [\[ECCV 2024\] GraspXL: Generating Grasping Motions for Diverse Objects at Scale](../../ECCV2024/robotics/graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)

</div>

<!-- RELATED:END -->
