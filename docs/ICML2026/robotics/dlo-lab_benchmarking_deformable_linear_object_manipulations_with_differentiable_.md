---
title: >-
  [Paper Note] DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics
description: >-
  [ICML 2026][Robotics][Deformable Linear Objects] DLO-Lab develops a differentiable simulator on the Genesis platform using Taichi, featuring a Discrete Elastic Rods (DER) kernel that supports bidirectional coupling…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Deformable Linear Objects"
  - "Differentiable Simulation"
  - "Robot Benchmark"
  - "Discrete Elastic Rods"
  - "Grasp Proposal"
date: 2026-05-08
content_hash: 49bcfdd2029daa42
---

# DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics

**Conference**: ICML 2026  
**arXiv**: [2606.04206](https://arxiv.org/abs/2606.04206)  
**Code**: Project Page https://dlo-lab-26.github.io/  
**Area**: robotics  
**Keywords**: Deformable Linear Objects, Differentiable Simulation, Robot Benchmark, Discrete Elastic Rods, Grasp Proposal  

## TL;DR
DLO-Lab develops a differentiable simulator on the Genesis platform using Taichi, featuring a Discrete Elastic Rods (DER) kernel that supports bidirectional coupling, bending plasticity, and closed-loop topology. It includes 10 benchmark tasks for ropes, cables, and rubber bands, alongside a specialized agent using VLMs for "grasp proposal + task decomposition." The framework facilitates a unified comparison of policy learning algorithms including PPO, SAC, SHAC, SAPO, CMA-ES, and GD, with sim-to-real validation via system identification.

## Background & Motivation
**Background**: Manipulation of Deformable Linear Objects (DLOs, such as ropes, cables, and rubber bands) is a long-standing robotic challenge. Prior work has either hard-coded single tasks (unknotting, wiring, shaping) or relied on real-world data, which lacks scalability and universality.

**Limitations of Prior Work**: Existing DLO simulators have various gaps: neural-network-based ones (Bi-LSTM, GNN, DEFORM) are differentiable but lack physical fidelity; PBD-based ones (SoftGym, XPBD) are fast but have coarse elastic potential modeling; DER-based models (Elastica, C-IPC, IMC) are high-fidelity but non-differentiable, preventing gradient-based policy optimization; MPM/Spring-Mass differentiable solutions (DaXBench, PhysTwin) struggle with bidirectional coupling with rigid/soft bodies or closed-loop topologies. No single platform simultaneously provides "elastic potential + bending plasticity + closed-loop topology + bidirectional coupling + differentiability," all of which are essential for realistic DLO manipulation.

**Key Challenge**: The engineering contradiction between physical fidelity (DER/FEM) and differentiability/coupling (automatic differentiation + MPM/SDF bidirectional contact). The former favors implicit timestepping and hard-constraint solvers, while the latter requires explicit timestepping and differentiable contact models.

**Goal**: (1) Build a DLO differentiable simulator possessing all five core characteristics; (2) design a benchmark task set reflecting unique DLO challenges (topological constraints, grasp sensitivity, long horizons); (3) provide a "DLO-specialized agent" using VLM physical common sense for automatic grasp selection and sub-task decomposition to enable standard RL/optimization for long-horizon tasks; (4) conduct a horizontal evaluation of MFRL, FO-MBRL, trajectory optimization, and evolutionary algorithms to establish baselines.

**Key Insight**: Utilizing the Genesis physical engine as a base with Taichi for automatic differentiation. DLOs are represented via DER (centerline vertices + adapted frames), soft-coupled with rigid bodies via SDF, and bidirectionally coupled with MPM soft bodies via Eulerian grids. Explicit gradient checkpointing enables differentiability over "arbitrarily long horizons." VLMs provide physical priors for "where to grasp and how many steps to take," which are difficult for end-to-end policies.

**Core Idea**: By combining a "differentiable DER kernel + bidirectional coupling + gradient checkpointing + VLM agent," DLO manipulation—a highly structured problem—is systematically benchmarked for the first time.

## Method
DLO-Lab is divided into three layers: the underlying physics simulator (Section 3), the mid-level benchmark task suite (Section 4.1-4.2), and the high-level DLO agent (Section 4.3).

### Overall Architecture
**Input**: Initial DLO state for each task (centerline vertices + frames), target conditions (e.g., S-shape, looping a ring, bypassing a post), robot arm configuration, and end-effector.

**Mechanism**: A self-developed DLO solver runs DER dynamics; bidirectional coupling is performed with Genesis's rigid body solver (SDF) and MPM solver (fluids/elastoplasticity); gradients are calculated throughout via Taichi autodiff, with gradient checkpointing used across timesteps.

**Policy Interface**: Standard MDP interface where state $\mathbf{S}=(\mathbf{x},\dot{\mathbf{x}},\mathbf{r},\mathbf{M},\dot{\mathbf{M}})$ includes all DLO vertex poses, rest configurations, and robot joint states. Observations consist of $(\mathbf{x},\dot{\mathbf{x}})\in\mathbb{R}^{N_v\times 6}$ and end-effector poses/joint configurations. Actions are Cartesian target poses for the end-effector (resolved via IK).

**Output**: Differentiable rewards and trajectory gradients $\partial r/\partial a_{0:T}$ for GD/SHAC/SAPO; the platform also supports sampling-based RL (PPO/SAC) and black-box optimization (CMA-ES).

### Key Designs

1. **Differentiable DLO Solver based on DER + Bidirectional Coupling**:
    - **Function**: Merges "high-fidelity DER dynamics" and "differentiability + bidirectional interaction with rigid/MPM soft bodies" into a single solver as the platform's physical foundation.
    - **Core Idea**: DLOs are represented by centerline vertices $\mathbf{x}=\{\mathbf{x}_i\in\mathbb{R}^3\}$ and adapted orthonormal frames $\mathbf{d}=\{(\mathbf{d}_1,\mathbf{d}_2,\mathbf{d}_3)^j\}$ for each edge, where $\mathbf{d}_3^j\equiv \mathbf{e}^j/|\mathbf{e}^j|$. Potential energy consists of stretching $U_s$, bending $U_b$, and twisting $U_t$. Gradients are derived and advanced via symplectic Euler. Bending plasticity is implemented by adjusting rest curvature using a yield threshold $\sigma_y$ and creep rate $r_c$. Closed-loop topology is handled by connecting the head and tail of the centerline. For rigid body coupling, DLO sample points query the rigid SDF; penetration depth $d(\mathbf{p})=r(\mathbf{p})-\mathrm{SDF}(\mathbf{p})$ is processed through a soft exponential factor $f_i=\min(\exp(d/\epsilon_s),1)$ for impulse-based friction response, with reaction forces applied to the rigid body. For MPM coupling, collisions between grid nodes and DLO vertices/edges are detected within the Eulerian grid loop, and repulsive impulses are applied based on relative velocity, normals, and mass ratios.
    - **Design Motivation**: DER is the most physically accurate discretization for slender rod geometries. Previous DER implementations (C-IPC, IMC) relied on implicit solvers and were non-differentiable. This work makes it explicit and runs on Taichi, marking the first time DER is both high-fidelity and differentiable while being compatible with heterogeneous materials.

2. **Gradient Checkpointing for Long-Horizon Backpropagation**:
    - **Function**: Enables "thousands of simulation steps + full differentiability" within limited GPU memory, determining whether first-order policy optimizers like GD/SHAC can utilize full trajectory gradients.
    - **Core Idea**: Following FluidLab, the trajectory is segmented during the forward pass. States are cached to CPU memory at the end of each segment, discarding intermediate GPU computation graphs. During the backward pass, checkpoints are traversed in reverse order; each segment re-runs a forward pass to reconstruct its local graph for backpropagation. This reduces memory consumption from $\mathcal{O}(T)$ to $\mathcal{O}(\sqrt{T})$ (optimal at segment length $\sim\sqrt{T}$), decoupling memory usage from the number of simulation steps.
    - **Design Motivation**: DLO tasks such as unknotting or wiring often reach thousands of steps. Standard autodiff causes immediate memory overflow. Checkpointing trades 1x extra forward computation for $\sqrt{T}$ memory savings, which is acceptable for policy optimization and makes first-order MBRL feasible for long-horizon DLO tasks.

3. **VLM-driven DLO Agent (Grasp Proposal + Task Decomposition)**:
    - **Function**: Adds structural priors for "where to grasp" and "how many steps to take"—which are difficult for end-to-end policies to learn—reducing long-horizon, topologically constrained tasks into multiple short-horizon sub-tasks.
    - **Core Idea**: Grasp proposal uses three prompting modes: (a) Candidate mode: VLM chooses from uniformly sampled candidate points rendered on the DLO; (b) Coefficient mode: VLM outputs a $[0,1]$ scalar representing the position along the DLO; (c) Marker mode: VLM clicks pixel coordinates based on visual markers. Candidate mode proved most reliable. For task decomposition, the VLM generates an initial plan including reward functions and horizons for each sub-task. Each sub-task is then solved via differentiable trajectory optimization. After execution, the VLM evaluates the state to decide whether to continue or re-plan, forming a closed loop.
    - **Design Motivation**: DLO manipulation possesses two fatal challenges for RL: grasping the wrong point can make a task kinematically infeasible (e.g., pulling the wrong strand in a knot), and long-horizon rewards are too sparse for end-to-end PPO/SAC exploration. Offloading symbolic and physical reasoning to VLMs allows the system to combine the semantic layer of world models with the numerical layer of optimizers.

### Loss & Training
- The simulator is differentiable, and all reward functions are designed to be smooth (contact smoothing, SDF distance smoothing). It supports both sampling-based RL and first-order optimization.
- The platform evaluates PPO, SAC (MFRL), SHAC, SAPO (FO-MBRL using analytical gradients from Taichi for actor optimization), GD (trajectory gradient descent on actions), and CMA-ES (gradient-free evolution strategies).
- Sim-to-real transfer utilizes the differentiable simulator for system identification: simulated rope projections are compared against real binary masks from video, and gradients are backpropagated to optimize material parameters (stretching/bending stiffness).

## Key Experimental Results

### Main Results
8 fixed-horizon tasks (Coiling, Gathering, Lifting, Separation, Slingshot, Unknotting, Wiring-post, Wrapping) + 2 long-horizon tasks (Letter Art, Wiring-ring). Results for fixed-horizon tasks report the max episodic return ± std over 3 seeds; long-horizon tasks are evaluated with the DLO agent.

| Task | PPO | SAC | SHAC | SAPO | GD | CMA-ES |
|------|-----|-----|------|------|-----|--------|
| Coiling | 9.40 | 8.28 | 11.55 | **11.57** | 11.59 | **11.73** |
| Gathering | 39.76 | 40.76 | 40.48 | 40.29 | 39.84 | **47.84** |
| Lifting | 247.38 | 250.29 | 214.24 | 204.54 | **255.55** | **335.59** |
| Separation | 114.31 | **134.71** | 96.29 | 105.27 | 115.52 | 84.86 |
| Slingshot | 6.90 | 7.23 | 6.90 | 6.90 | 6.90 | **11.07** |
| Unknotting | 3.29 | 2.95 | 45.88 | **46.30** | 3.44 | **57.21** |
| Wiring-post | 62.17 | 62.07 | 36.42 | 36.13 | 36.40 | **64.31** |
| Wrapping | 131.08 | 161.85 | 129.90 | 144.36 | 139.98 | **162.68** |

CMA-ES achieved the best results in 6/8 tasks. FO-MBRL (SHAC/SAPO) significantly outperformed PPO/SAC on the topological Unknotting task (46 vs 3), proving the necessity of differentiable gradients in contact-intensive tasks. GD performed well on smooth-reward tasks but failed in local optima for Unknotting and Wiring-post.

### Ablation Study
| Configuration | Key Finding | Description |
|------|----------|------|
| MFRL (PPO/SAC) vs Traj. Opt. | Traj. Opt. is significantly more sample-efficient | RL involves extra overhead for exploration and network fitting, which is penalized by sparse rewards and high-dimensional vertex states. |
| FO-MBRL vs MFRL (Unknotting) | 46 vs 3 | Analytical gradients allow SHAC/SAPO to find optimization directions at contact switches where PPO/SAC get stuck. |
| CMA-ES vs GD (Lifting/Slingshot) | CMA-ES leads by a wide margin | When DLO has no contact, gradients are zero; GD fails without warm-starting, while CMA-ES skips local optima via parallel sampling. |
| Grasp proposal: Candidate / Coefficient / Marker | Candidate mode is most stable | Discrete candidates best match VLM vision-language reasoning; numerical precision suffers in other modes. |
| Task decomposition (Letter Art, Wiring-ring) | Closed-loop re-planning boosts success | Single-phase optimization cannot handle serial dependencies like "bend into D, then re-grasp to bend into L." |
| System ID + Real-world Transfer | Differentiable sim optimized rope parameters | Open-loop success with zero-shot deployment; closed-loop Wiring-ring achieved 7/12 (≈58%) trials. |

### Key Findings
- **Differentiability acts as a "contact penetrator"**: On purely topological tasks like Unknotting, analytical gradients improve SHAC/SAPO performance by 15x over MFRL. However, when rewards depend on contacts that haven't occurred (Lifting/Slingshot), gradients vanish, and CMA-ES takes the lead.
- **Closed-loop policies are harder to learn than open-loop trajectories**: Under the same sample budget, PPO/SAC lose to CMA-ES. This is not a failure of RL algorithms, but a reflection that learning a robust policy while navigating exploration is inherently harder. Future DLO research must distinguish between open-loop and closed-loop baselines.
- **Robustness of CMA-ES**: Derived from (1) independence from local gradients, avoiding zero-gradient dead zones, and (2) large-scale parallel sampling that jumps out of local optima in non-smooth reward landscapes. Differentiable simulation does not replace sampling; they should be selected based on gradient availability.
- **VLM Synergy**: The combination of Candidate mode and task decomposition makes multi-stage topological tasks (Letter Art) solvable, whereas end-to-end RL is nearly impossible within reasonable budgets.
- **System Identification is key for Sim-to-Real**: Using the differentiable simulator to calibrate stiffness parameters allowed a 58% success rate in closed-loop Wiring-ring and zero-shot deployment for open-loop tasks, a level of validation often missing in previous DLO simulators.

## Highlights & Insights
- **The "DER + Autodiff + Coupling + Checkpoint" stack**: While individual components have appeared elsewhere, DLO-Lab is the first to integrate all five, providing a comprehensive "reference implementation" for DLO benchmarks.
- **Differentiable sim for System ID is highly effective**: Rather than relying on gradients to solve the entire policy, using gradients to calibrate physical parameters for sim-to-real deployment appears to be a more robust path than "end-to-end differentiable policy" optimization.
- **VLM as a Structural Prior**: Instead of direct action output, using VLMs for semantic/topological labels (where and when to re-grasp) is a pragmatic approach that avoids the numerical precision pitfalls of current VLMs.
- **Method Selection Guide**: Smooth reward + low contact $\rightarrow$ GD; Dense contact + gradient availability $\rightarrow$ SHAC/SAPO; Sparse contact / non-smooth / topological $\rightarrow$ CMA-ES; Generalist closed-loop policy $\rightarrow$ Requires 10x more samples.

## Limitations & Future Work
- DER remains a discretization of rods; extremely fine cables or soft ribbons may require higher resolution, hitting performance bottlenecks in the Taichi kernel.
- Bidirectional coupling currently covers rigid bodies (SDF) and MPM (soft/fluids) but excludes textiles (cloth) or granular materials (sand).
- VLM agent reliability depends on external APIs and prompt engineering; the agent's self-error correction analysis is currently insufficient.
- The 58% real-world success rate for Wiring-ring indicates a gap; robustness tests against simultaneous shifts in perception and physical parameters are needed.
- Tasks are currently limited to tabletop single/dual-arm parallel grippers; dexterous hands and mobile platforms are not yet included.

## Related Work & Insights
- **vs DaXBench (Chen et al., 2023)**: DaXBench uses MPM for everything; DLOs are viewed as particles. DLO-Lab uses DER to respect the 1D geometry, leading to lower parameter dimensions and higher fidelity.
- **vs PhysTwin (Jiang et al., 2025)**: PhysTwin uses spring-mass systems and lacks bending plasticity/topological constraints. DLO-Lab's DER core captures plasticity in copper wires and rubber bands.
- **vs C-IPC / IMC**: These are physically accurate but non-differentiable due to implicit solvers. DLO-Lab offers a compromise via explicit symplectic Euler for first-order policy optimization.
- **vs SoftGym / XPBD**: PBD is fast but lacks precision and closed-loop support. DLO-Lab uses PBD only for friction sub-modules, leaving the core dynamics to DER.
- **vs DEFORM (Chen et al., 2024)**: DEFORM approximates DER with NNs; DLO-Lab uses analytical DER + autodiff, avoiding generalization risks from training data.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining the five-component stack has significant engineering and research value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 algorithms × 10 tasks, VLM prompt ablation, and real-world deployment covers all necessary bases.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; Table 1 effectively highlights differences with existing simulators.
- Value: ⭐⭐⭐⭐⭐ Directly provides a unified infrastructure, baseline evaluations, and sim-to-real recipes for the DLO field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](robomme_benchmarking_and_understanding_memory_for_robotic_generalist_policies.md)
- [\[ICML 2026\] Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](plan_in_sandbox_navigate_in_open_worlds_learning_physics-grounded_abstracted_exp.md)
- [\[ACL 2026\] ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation](../../ACL2026/robotics/elasticflow_one-step_physics-consistent_policy_with_elastic_time_horizons_for_la.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](../../AAAI2026/robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)

</div>

<!-- RELATED:END -->
