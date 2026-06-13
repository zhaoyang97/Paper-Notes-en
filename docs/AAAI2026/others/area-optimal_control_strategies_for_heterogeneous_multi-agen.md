---
title: >-
  [Paper Note] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit
description: >-
  [AAAI 2026][pursuit-evasion game] This paper studies pursuit-evasion games with heterogeneous speeds involving multiple pursuers and a single evader. The evader's safe reachable set is defined as the intersection of Apol…
tags:
  - "AAAI 2026"
  - "pursuit-evasion game"
  - "heterogeneous multi-agent"
  - "Apollonius circle"
  - "area-optimal"
  - "closed-form control law"
  - "safe reachable set"
  - "zero-sum game"
date: 2026-05-08
content_hash: c3391ddc31380ba3
---

# Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit

**Conference**: AAAI 2026
**arXiv**: [2511.15036v2](https://arxiv.org/abs/2511.15036v2)  
**Code**: None  
**Area**: Other
**Keywords**: pursuit-evasion game, heterogeneous multi-agent, Apollonius circle, area-optimal, closed-form control law, safe reachable set, zero-sum game

## TL;DR

This paper studies pursuit-evasion games with heterogeneous speeds involving multiple pursuers and a single evader. The evader's safe reachable set is defined as the intersection of Apollonius circles for all pursuer–evader pairs. The capture strategy is modeled as a zero-sum game in which pursuers minimize and the evader maximizes the area of this intersection. Closed-form instantaneous optimal heading control laws are derived, and simulations verify that pursuers can systematically shrink the safe region to guarantee capture.

## Background & Motivation

### State of the Field

**Background**: Multi-agent pursuit-evasion is a classical problem in multi-agent systems and game theory, with broad applications in military interception, search and rescue, robotic encirclement, and UAV coordination. The main limitations of existing methods are:

1. **Homogeneity assumption**: Most methods assume identical pursuer speeds and cannot handle heterogeneous scenarios where different platforms (ground vehicles, UAVs, speedboats) operate at vastly different speeds.
2. **High computational cost**: Methods based on numerical optimization or Hamilton-Jacobi reachability analysis incur high computational complexity, making them unsuitable for real-time embedded control.
3. **Lack of analytical solutions**: Existing cooperative capture strategies largely rely on heuristic or learning-based approaches, without rigorous mathematical characterization or optimality guarantees.

The Apollonius circle—the geometric locus of points reachable simultaneously by two agents moving at different speeds—provides a natural geometric tool for characterizing heterogeneous-speed pursuit-evasion. This paper exploits the analytical properties of Apollonius circles to reformulate the pursuit-evasion game as an area optimization problem.

### Starting Point

**Goal**: **How can multiple pursuers with heterogeneous speeds cooperatively minimize the evader's safe reachable region? Can closed-form optimal control laws be derived to support real-time decision-making?**

## Method

### Overall Architecture

The objective of the pursuit-evasion game is reformulated from the conventional "minimize capture time" to "minimize/maximize the area of the safe reachable set." The safe reachable set is defined as the region the evader can reach before any pursuer—geometrically, the intersection of Apollonius circles for all pursuer–evader pairs. This area serves as the payoff function of the zero-sum game: pursuers cooperatively minimize the area, while the evader maximizes it.

### Key Designs

1. **Safe Reachable Set Definition (Intersection of Apollonius Circles)**:

    - For each pursuer–evader pair, the radius of the Apollonius circle is determined by the speed ratio $\alpha_i = v_e / v_{p_i}$, where $v_e$ is the evader speed and $v_{p_i}$ is the $i$-th pursuer speed.
    - The center lies on the line connecting the two agents at the point that internally or externally divides it in the speed ratio.
    - The interior of the circle represents points the evader can reach before that pursuer.
    - In the multi-pursuer setting, the safe reachable set is the intersection of all Apollonius circles—a convex region.
    - Capture occurs when this area shrinks to zero.

2. **Analytical Computation of Area Gradients**:

    - The gradient of the intersection area with respect to each agent's position admits a closed-form expression.
    - Using the analytical formulas for Apollonius circle parameters (center and radius), the gradient of the area with respect to each pursuer's and the evader's position is computed via the chain rule.
    - This constitutes the core technical contribution of the paper: transforming the optimization of a geometric intersection area into an analytically solvable gradient descent/ascent problem.

3. **Closed-Form Instantaneous Optimal Control Law**:

    - Optimal heading for each pursuer: the **negative** direction of the area gradient with respect to its position (gradient descent, reducing area).
    - Optimal heading for the evader: the **positive** direction of the area gradient with respect to its position (gradient ascent, increasing area).
    - The control law requires only the computation of Apollonius circle parameters and area gradients, involving neither numerical optimization nor forward simulation.
    - Computational complexity is minimal ($O(n)$, where $n$ is the number of pursuers), making it suitable for real-time embedded systems.

4. **Natural Handling of Heterogeneous Speeds**:

    - Different pursuer speeds are directly encoded in the radius and center of each corresponding Apollonius circle.
    - A faster pursuer corresponds to a larger Apollonius circle, imposing stronger spatial constraints on the evader.
    - The framework natively supports arbitrary speed configurations without special-case treatment.

### Game-Theoretic Formulation

The problem is cast as a continuous-time zero-sum differential game:
- Pursuer team strategy space: heading angle $\theta_{p_i} \in [0, 2\pi)$ for each pursuer.
- Evader strategy space: heading angle $\theta_e \in [0, 2\pi)$.
- Payoff function: safe reachable set area $A(t)$.
- Pursuer objective: $\min A(t)$; evader objective: $\max A(t)$.
- Greedy strategy: at each instant, each agent selects the instantaneously optimal heading.

### Loss & Training

No learning or training components are involved. The control law is a purely analytically derived instantaneous optimal strategy, computing the optimal heading angle directly from the current state (positions, speeds).

## Key Experimental Results

### Main Results

| Scenario | # Pursuers | Speed Configuration | Result | Notes |
|----------|-----------|---------------------|--------|-------|
| 2v1 heterogeneous | 2 | Different speeds | Safe region area monotonically decreases | Capture guaranteed |
| 3v1 heterogeneous | 3 | Different speeds | Faster convergence | Clear cooperative effect |
| vs. numerical method | — | — | Trajectories consistent with numerical solution | Validates analytical solution |
| Evader counterstrategy | — | — | Evader maximizes area | Zero-sum game equilibrium verified |

### Key Findings

- **Heterogeneous vs. homogeneous**: Under heterogeneous speeds, the optimal strategy differs **significantly** from that under the homogeneity assumption—faster pursuers take on greater encirclement responsibilities, while slower pursuers cooperate to narrow escape routes.
- **Effect of pursuer count**: Increasing the number of pursuers accelerates safe region shrinkage, but with diminishing marginal returns.
- **Evolution of safe region shape**: The Apollonius circle intersection transitions from a near-circular shape to an elongated region, eventually degenerating to zero area (the moment of capture).
- **Trajectory validation**: Trajectories generated by the analytical control law closely match those of numerical optimization methods with far higher computational cost.

## Highlights & Insights

- **Elegance of closed-form solutions**: The entirely closed-form optimal control law requires no numerical optimization and no iterative solving—the theoretical and practical value of a closed-form solution far exceeds that of approximate numerical solutions.
- **Geometric clarity**: Area minimization as an optimization objective for spatial control is highly intuitive, and the visualization of the Apollonius circle intersection makes strategy effectiveness immediately apparent.
- **Real-time feasibility**: $O(n)$ computational complexity enables direct deployment of the control law on resource-constrained embedded platforms (e.g., micro-UAVs) without requiring powerful computational backends.
- **Key insight behind the objective reformulation**: Replacing "time minimization" with "area minimization" is the central insight of the paper—the area objective enables analytical gradient computation, whereas the time objective typically requires solving an HJB equation.

## Limitations & Future Work

- **Simulation-only validation**: Physical robot experiments are absent; practical factors such as communication delay, localization error, kinematic constraints (turning radius), and obstacles are not considered.
- **2D planar assumption**: Extension to 3D space (e.g., UAV pursuit-evasion) requires handling the intersection of Apollonius spheres, which involves more complex geometric computation and may not admit closed-form solutions.
- **Single-evader assumption**: In multi-evader scenarios, the definition of the safe reachable set and the game structure become more complex, requiring pursuers to perform target assignment.
- **Instantaneous optimality ≠ global optimality**: The greedy strategy may not constitute a Nash equilibrium under certain extreme initial configurations—the evader may exploit the greedy nature of pursuers to construct counterexamples.
- **Constant-speed assumption**: Acceleration constraints, energy limitations, and other more realistic motion models are not considered.
- **No communication or sensing constraints**: All pursuers are assumed to have global observability with no communication delay.

## Related Work & Insights

- **vs. Voronoi-based pursuit-evasion**: Voronoi methods assume equal speed and cannot handle heterogeneous scenarios; Apollonius circles naturally encode speed differences.
- **vs. RL-based pursuit-evasion methods**: RL methods can adapt to complex environments but lack optimality guarantees and interpretability; this paper provides a provably optimal (in the instantaneous sense) control law.
- **vs. Hamilton-Jacobi reachability analysis**: HJ methods compute safe reachable sets most accurately but require solving PDEs whose computational cost grows exponentially with dimensionality (curse of dimensionality); the area optimization objective directly bypasses PDE solving.
- **vs. classical pursuit-evasion differential games**: Classical methods largely focus on "capture time minimization" or "escape probability"; the "area optimization" objective introduced here represents a novel perspective.

## Related Work & Insights

- The Apollonius circle is a general geometric tool for heterogeneous-speed problems, with potential applications in sensor coverage, communication range planning, and search task assignment.
- The methodological insight of substituting "area minimization" for "time minimization" is broadly instructive—in multi-agent coordination, choosing an **analytically tractable surrogate objective** may be more efficient than directly optimizing the true objective.
- This paper exemplifies rigorous theoretical work: the problem is clearly defined, the derivations are complete, and the conclusions are concise and insightful.

## Rating

- Novelty: ⭐⭐⭐⭐ The area-optimal problem formulation is novel; the closed-form heterogeneous control law is an important theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Validation is limited to simulation without physical experiments; scenarios are relatively simple.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous; geometric visualizations are intuitive.
- Value: ⭐⭐⭐ Offers theoretical value to the multi-agent control community, but the application scope is narrow and experimental validation is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[CVPR 2026\] AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](../../CVPR2026/others/assistmimic_physics_grounded_humanoid_assistance.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
