---
title: >-
  [Paper Note] Tree-Guided Diffusion Planner
description: >-
  [NeurIPS 2025][Image Generation][Diffusion planning] This paper proposes the Tree-guided Diffusion Planner (TDP), which formalizes test-time diffusion planning as a tree search problem. Through bi-level sampling—particle-guided generation of diverse parent trajectories for exploration, combined with fast conditional denoising to generate child trajectories for exploitation—TDP achieves a strong exploration–exploitation balance and substantially outperforms existing methods under non-convex objectives and non-differentiable constraints.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion planning
  - tree search
  - test-time guidance
  - zero-shot planning
  - trajectory generation
date: 2026-05-08
content_hash: 0606e2ec7504bbac
---

# Tree-Guided Diffusion Planner

**Conference**: NeurIPS 2025
**arXiv**: [2508.21800](https://arxiv.org/abs/2508.21800)
**Code**: [Project Page](https://tree-diffusion-planner.github.io)
**Area**: Diffusion Models / Planning & Generation
**Keywords**: Diffusion planning, tree search, test-time guidance, zero-shot planning, trajectory generation

## TL;DR

This paper proposes the Tree-guided Diffusion Planner (TDP), which formalizes test-time diffusion planning as a tree search problem. Through bi-level sampling—particle-guided generation of diverse parent trajectories for exploration, combined with fast conditional denoising to generate child trajectories for exploitation—TDP achieves a strong exploration–exploitation balance and substantially outperforms existing methods under non-convex objectives and non-differentiable constraints.

## Background & Motivation

Diffusion models have emerged as a powerful framework for offline planning, capable of generating coherent trajectories from offline demonstration data. However, existing test-time guided planning methods face several fundamental limitations:

**Limitations of gradient guidance**: Standard gradient guidance (e.g., classifier guidance) assumes the guidance function is convex and differentiable, yet real-world planning tasks frequently involve non-convex objectives (e.g., multi-goal navigation) and non-differentiable constraints (e.g., path planning through mandatory waypoints).

**In-distribution preference**: Pre-trained diffusion models tend to generate trajectories consistent with the training data distribution, making it difficult to discover solutions requiring compositional novelty. Gradient guidance is prone to local optima, as it prioritizes locally optimal trajectories within the learned distribution.

**Exploration–exploitation dilemma**: The guidance strength $\alpha$ is highly task-dependent, and exhaustive tuning is prohibitively costly in test-time guidance scenarios. Existing methods fail to adequately address the balance between maintaining trajectory feasibility and maximizing guidance scores.

**Limitations of supervised approaches**: Sequential methods such as MCTD require training task-specific value estimators; Hierarchical Diffuser relies on subgoal distributions annotated at training time; D-MPC requires few-shot fine-tuning—all of which constrain zero-shot generalization.

## Method

### Overall Architecture

TDP adopts a bi-level trajectory sampling framework. The first level generates diverse parent trajectories via Particle Guidance (PG) to promote exploration; the second level branches from parent trajectories using fast conditional denoising to generate child trajectories for exploitation. All trajectories form a tree structure, and the path with the highest-scoring leaf node is selected as the final plan.

### Key Designs

1. **State Decomposition**: Given a test-time guidance function, TDP automatically partitions the state vector into observation states and control states by examining the gradient of the guidance function with respect to each state feature—features with non-zero gradients are classified as observation states, while those with zero gradients are classified as control states. For example, in the KUKA robot arm task, block positions are observation states (affecting the objective function), while robot joint angles are control states (not directly affecting the objective but governing dynamics). This gradient-based automatic classification requires no task-specific prior knowledge.

2. **Parent Branching**: Fixed-potential Particle Guidance (PG) is applied to the control states, using the gradient $\nabla\Phi$ of a radial basis function (RBF) to impose repulsive forces among trajectories, thereby promoting diversity in exploration. Task gradient guidance $\nabla\mathcal{J}$ is applied to the observation states. The overall denoising update is:

$$\boldsymbol{\mu}^{i-1}_{\text{control}} \leftarrow \boldsymbol{\mu}^{i}_{\text{control}} + \alpha_p \Sigma^i \nabla\Phi(\boldsymbol{\mu}^{i}_{\text{control}})$$
$$\boldsymbol{\mu}^{i-1}_{\text{obs}} \leftarrow \boldsymbol{\mu}^{i}_{\text{obs}} + \alpha_g \Sigma^i \nabla\mathcal{J}(\boldsymbol{\mu}^{i}_{\text{obs}})$$

PG and gradient guidance together constitute a unified conditional distribution, with the combined guidance term $g_{\text{TDP}} = g_{\text{gg}} + g_{\text{pg}}$.

3. **Sub-Tree Expansion**: For each parent trajectory, a branching point $b \sim \text{Unif}(0, T_{\text{pred}})$ is randomly selected. The portion of the parent trajectory after the branching point is noised using fast denoising with $N_f \ll N$ steps, and child trajectories are generated conditioned on the parent trajectory prefix $\boldsymbol{C} = \{s_k\}_{k=0}^b$:

$$\boldsymbol{\tau}_{\text{child}}^{N_f} \sim q_{N_f}(\boldsymbol{\tau}_{\text{parent}}, \boldsymbol{C})$$

Child trajectories serve two purposes: (1) correcting dynamical infeasibilities introduced by particle guidance in the parent trajectories; and (2) performing efficient local search in the vicinity of parent trajectories to find superior solutions.

### Theoretical Guarantees

Proposition 1 establishes the necessity of bi-level sampling: guided sampling initialized from a standard Gaussian may converge to local optima that lie off the data manifold (off-subspace), whereas guided sampling initialized near unconditional samples converges to the global optimum. This provides a theoretical justification for TDP's strategy of first generating on-subspace parent trajectories before performing guided sub-tree expansion, which is superior to direct gradient guidance.

## Key Experimental Results

### Main Results — Maze2D Gold-picking

| Method | Medium | Large | Single-task Avg. | Multi-Medium | Multi-Large | Multi-task Avg. |
|--------|--------|-------|-----------------|-------------|-------------|----------------|
| Diffuser | 10.1 | 4.3 | 6.8 | 7.7 | 9.9 | 8.8 |
| Diffuserγ (TAT) | 12.3 | 9.3 | 10.8 | 8.6 | 23.1 | 15.9 |
| MCSS | 17.2 | 25.0 | 21.1 | 32.3 | 57.5 | 44.9 |
| MCSS+SS | 17.4 | 21.2 | 19.3 | 29.2 | 58.0 | 43.6 |
| TDP (w/o child) | 19.0 | 30.4 | 24.7 | 35.3 | 59.1 | 47.2 |
| TDP (w/o PG) | 39.1 | 41.1 | 40.1 | 75.9 | 64.9 | 70.4 |
| **TDP** | **39.8** | **47.6** | **43.7** | **74.7** | **70.0** | **72.4** |

### Ablation Study — Robot Arm Manipulation

| Method | PnWP | PnP (stack) | PnP (place) | PnP Avg. |
|--------|------|------------|------------|---------|
| Diffuser | 31.13 | 51.50 | 21.31 | 36.41 |
| AdaptDiffuser | 39.72 | 60.54 | 36.17 | 48.36 |
| MCSS | 35.69 | 59.91 | 31.37 | 45.64 |
| MCSS+SS | 36.24 | 56.80 | 35.50 | 46.15 |
| TDP (w/o child) | 35.53 | 60.00 | 32.19 | 46.10 |
| TDP (w/o PG) | 66.63 | 59.42 | 36.94 | 48.18 |
| **TDP** | **66.81** | **61.17** | **36.94** | **49.06** |

### Key Findings

- **Most significant gains on non-convex tasks**: In the PnWP task, TDP (66.81) nearly doubles MCSS (35.69), demonstrating the substantial advantage of bi-level sampling under non-convex guidance functions.
- **Particle guidance is critical for multi-modal exploration**: The performance gap between TDP and TDP (w/o PG) is modest on PnWP but pronounced in Maze2D multi-task settings, indicating that PG is especially important in scenarios requiring broad exploration.
- **Sub-tree expansion enables fine-grained optimization**: TDP (w/o child) scores only 35.53 on PnWP, far below the full TDP's 66.81, confirming that local search is essential for discovering globally optimal solutions.
- TDP finds **11% more goals than MCSS in AntMaze** while reducing the number of steps per goal, demonstrating superior exploration efficiency.
- TDP surpasses AdaptDiffuser, which requires task-specific training, demonstrating strong zero-shot test-time planning capability.

## Highlights & Insights

- **Reformulating planning as tree search** is a novel and natural perspective—parent nodes provide broad exploration while child nodes offer local refinement, perfectly matching the exploration–exploitation trade-off.
- The **automated state decomposition** design requires no task-specific prior knowledge, lending the method strong generality.
- The **theoretical analysis** (Proposition 1) clearly explains why on-subspace initialization is superior to standard Gaussian initialization, providing a principled foundation for the method.
- The fully zero-shot, training-free nature of TDP makes it applicable to a wide variety of new test-time objectives.

## Limitations & Future Work

- Bi-level trajectory generation introduces additional computational overhead, including pairwise distance computation among particles and extra forward passes for child trajectory generation.
- The hyperparameters for particle guidance ($\alpha_p$, $\alpha_g$) still require specification; while more robust than a single guidance strength, tuning remains necessary.
- Experiments are conducted in relatively simple settings (2D mazes, robot arms); performance in higher-dimensional spaces with more complex dynamics remains to be validated.
- In closed-loop planning, re-running the search at each step may limit real-time applicability.
- An in-depth comparison with learning-based test-time adaptation methods is absent.

## Related Work & Insights

- TDP combines the sampling diversity of diffusion models with the structured exploration of tree search, offering a new paradigm for applying diffusion models to complex decision-making problems.
- The use of Particle Guidance (PG) to promote sample diversity originates from the image generation literature; this work innovatively adapts it for diversity exploration in trajectory space.
- The state decomposition idea is generalizable to other generative planning scenarios that require distinguishing between controllable and uncontrollable states.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The bi-level sampling framework and automated state decomposition are original; the integration of tree search with diffusion planning is theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across three distinct task domains with thorough ablations, though validation in more complex real-world scenarios is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated and the connection between theory and experiments is natural, though some notation is dense.
- **Value**: ⭐⭐⭐⭐ Addresses a core pain point in test-time guidance for diffusion planning; the zero-shot capability holds practical promise.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)
- [\[NeurIPS 2025\] Safe and Stable Control via Lyapunov-Guided Diffusion Models](safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Guided Diffusion Sampling on Function Spaces with Applications to PDEs](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](../../CVPR2026/image_generation/image_generation_as_a_visual_planner_for_robotic_manipulation.md)

<!-- RELATED:END -->
