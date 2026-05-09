---
title: >-
  [Paper Note] A Differential and Pointwise Control Approach to Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Differential RL] This paper reformulates the RL problem via the differential dual form of continuous-time control, embeds physical priors through Hamiltonian structure, and proposes the dfPO algorithm for pointwise policy optimization. On scientific computing tasks (surface modeling, grid-based control, molecular dynamics), dfPO surpasses 12 RL baselines with fewer samples.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Differential RL
  - Hamiltonian structure
  - pointwise convergence
  - scientific computing RL
  - Pontryagin maximum principle
date: 2026-05-08
content_hash: 16763cb100b7f7f6
---

# A Differential and Pointwise Control Approach to Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2404.15617](https://arxiv.org/abs/2404.15617)
**Code**: [https://github.com/mpnguyen2/dfPO](https://github.com/mpnguyen2/dfPO)
**Area**: Reinforcement Learning / Continuous-Time Control / Scientific Computing
**Keywords**: Differential RL, Hamiltonian structure, pointwise convergence, scientific computing RL, Pontryagin maximum principle

## TL;DR
This paper reformulates the RL problem via the differential dual form of continuous-time control, embeds physical priors through Hamiltonian structure, and proposes the dfPO algorithm for pointwise policy optimization. On scientific computing tasks (surface modeling, grid-based control, molecular dynamics), dfPO surpasses 12 RL baselines with fewer samples.

## Background & Motivation
RL in scientific computing faces three key bottlenecks: (1) poor sample efficiency — scientific simulations are costly and cannot afford extensive trial-and-error; (2) lack of physical consistency — standard RL encodes neither physical laws nor structural priors, leading to trajectories that violate physical constraints; (3) weak theoretical guarantees — fine-grained convergence guarantees in continuous state-action spaces are largely absent. Model-based RL can improve sample efficiency but requires explicit reward models or their gradients (e.g., SVG, PILCO, iLQR), or assumes the ability to replan from intermediate states (e.g., shooting methods) — capabilities generally unavailable in black-box scientific simulators. This motivates a fundamentally different approach: constructing a physics-aligned RL framework grounded in continuous-time optimal control.

## Core Problem
How to design an RL algorithm that is both physically consistent and theoretically grounded for scientific computing scenarios characterized by low data, physical constraints, and black-box simulators? The core challenges are: (1) rewards are only observable at trajectory points, precluding direct access to the global reward function or its gradients; (2) the agent must generate complete trajectories from the initial time step without mid-episode resets or modifications; (3) pointwise policy quality must be guaranteed in continuous spaces.

## Method

### Overall Architecture
The authors convert the discrete-time cumulative reward maximization of a standard MDP into a continuous-time integral formulation, then introduce dual variables (adjoint variables $p$) via the Pontryagin Maximum Principle (PMP) to construct the Hamiltonian $H(s,p,a)$. Eliminating explicit action dependence through the stationarity condition yields a **differential dual system**: the state-adjoint pair $x=(s,p)$ evolves in phase space along the symplectic gradient flow $\dot{x}=S\nabla h(x)$ (where $S$ is the symplectic matrix). Discretization produces the dynamics operator $G(x)=x+\Delta t \cdot S\nabla g(x)$, where $g$ is a learnable score function approximating the Hamiltonian. The learning objective thus shifts from "maximizing cumulative reward" to "learning the optimal trajectory operator $G$."

### Key Designs
1. **Differential Dual Reformulation**: Rather than optimizing directly within the MDP framework, the paper first transforms the problem to continuous-time control, constructs the dual via PMP, and then discretizes the dual system. This yields two benefits: (a) the Hamiltonian structure naturally encodes physical priors (the symplectic form preserves phase-space structure); (b) the policy is defined on the extended space $(s,p)$, where $p$ encodes reward-action information ($p=a^*$ under regularized rewards), bypassing explicit action-space search.

2. **Score Function Learning**: Rather than directly learning a value function or policy network, the method learns a score function $g(x)\approx h(x)$ (approximating the Hamiltonian), obtaining the policy $G_\theta=\text{Id}+\Delta t \cdot S\nabla g_\theta$ via automatic differentiation, with training performed using smooth $L_1$ loss. This design naturally preserves trajectory consistency during policy updates.

3. **Stage-wise Temporal Expansion (dfPO Algorithm)**: Analogous to Dijkstra's algorithm, training proceeds stage by stage from step 1 to step $H-1$. At each stage $k$: (a) $N_k$ trajectories are sampled using the current policy $G_{\theta_{k-1}}$ and environment scores are queried; (b) new samples are added to a replay buffer, retaining only those on which the current policy performs well; (c) $g_{\theta_k}$ is trained to approximate both the environment score $g$ and the previous policy's score $g_{\theta_{k-1}}$ (to prevent abrupt policy changes); (d) the policy $G_{\theta_k}$ is updated via automatic differentiation.

### Loss & Training
- **Smooth $L_1$ loss** is used to train the score function $g_\theta$.
- Replay buffer filtering retains only samples on which the current policy already performs well, ensuring correct policy update directions.
- For scientific computing tasks, rewards take the regularized form $r(s,a)=\frac{1}{2}\|a\|^2 - \mathcal{F}(s)$, such that the adjoint variable $p$ equals the optimal action $a^*$.
- Hyperparameters are minimal: learning rate 0.001, batch size 32, with no complex tuning required.

## Key Experimental Results

| Task | dfPO | CrossQ | TQC | DDPG | TRPO | SAC | PPO |
|------|------|--------|-----|------|------|-----|-----|
| Surface Modeling (↓) | **6.32** | 6.42 | 6.67 | 15.92 | 6.48 | 7.41 | 20.61 |
| Grid-based (↓) | **6.06** | 7.23 | 7.12 | 6.58 | 7.10 | 7.00 | 7.11 |
| Molecular Dyn. (↓) | **53.34** | 923.90 | 76.87 | 68.20 | 1842.30 | 1361.31 | 1842.31 |

- Evaluated against 12 baselines (6 standard + 6 reward-shaped), dfPO achieves the best performance across all three tasks.
- The advantage is most pronounced on the molecular dynamics task: dfPO scores 53.34 vs. the second-best DDPG at 68.20.
- Performance on classic control tasks (Pendulum/MountainCar/CartPole) is also competitive.
- Statistical significance (t-test over 10 random seeds) confirms that dfPO's improvements are statistically significant.
- Training time is approximately 1 hour on an A100, comparable to SAC and lower than TQC/CrossQ at 2 hours.

### Ablation Study
- **Strong hyperparameter robustness**: dfPO uses default hyperparameters (lr=0.001, batch=32) and remains stable, while baselines exhibit large performance variation across different hyperparameter settings.
- **Reward shaping helps but is insufficient**: reward-shaped variants consistently outperform their standard counterparts, yet still fall short of dfPO.
- **Minimal model size**: dfPO models occupy only 0.17–0.66 MB, similar to PPO/TRPO, whereas DDPG requires 4–5 MB.
- On the molecular dynamics task, TRPO, PPO, S-TRPO, and S-PPO fail entirely (cost ≈ 1842), demonstrating that these methods cannot learn under extremely limited data (5,000 steps).

## Highlights & Insights
- **Perspective innovation**: Revisiting RL through the dual theory of continuous-time control naturally induces Hamiltonian structure, encoding physical priors as inductive biases rather than explicit constraints — even when the problem does not explicitly involve physics, the symplectic structure provides beneficial regularization.
- **Pointwise convergence guarantee**: Standard RL theory provides only global regret bounds, whereas dfPO proves a policy error bound of $\mathcal{O}(\epsilon)$ at **each individual time step** — a stronger guarantee that prevents severe policy degradation at specific steps (e.g., reward hacking).
- **Algorithmic simplicity**: Compared to TRPO's complex constrained optimization, dfPO requires only training a score function followed by automatic differentiation, making implementation straightforward.
- **Closed loop between theory and practice**: Theorem 3.2 provides an explicit formula for the required sample count, and experiments validate the advantage under low-data conditions.
- **Score function as Hamiltonian**: $g(x)$ simultaneously serves as a critic (evaluating trajectory quality) and a policy generator (producing actions via its gradient), unifying the two networks of the actor-critic paradigm.

## Limitations & Future Work
- **Strong theoretical assumptions**: The framework requires bounded Lipschitz constants for the dynamics operator $G$ and the policy network, and continuity of the initial distribution $\rho_0$, which excludes discontinuous dynamical systems.
- **Limited task diversity**: Validation is restricted to energy minimization tasks in scientific computing; more general RL settings (e.g., game playing, robotic manipulation with non-energy objectives) have not been tested.
- **Suboptimal regret bound**: The $\mathcal{O}(K^{5/6})$ bound is derived under restricted hypothesis spaces; the general bound $\mathcal{O}(K^{(2d+3)/(2d+4)})$ degrades with dimensionality.
- **Doubled extended-space dimensionality**: Replacing $(s,a)$ with $(s,p)$ raises the dimension to $d_S+d_A$, which may increase learning difficulty in high-dimensional problems.
- **Deterministic environment assumption**: The current framework assumes deterministic dynamics ($s_{k+1}=G(s_k)$); stochastic environments would require an SDE-based dual formulation.
- **Regularized reward dependency**: The elegant correspondence $p=a^*$ relies on the quadratic regularization form $r=\frac{1}{2}\|a\|^2-\mathcal{F}(s)$.

## Related Work & Insights
- **vs. TRPO/PPO**: dfPO derives from the continuous-time control dual and naturally embeds symplectic structure priors; TRPO/PPO are purely discrete-time methods relying on global value estimation. On low-data scientific computing tasks, PPO nearly fails entirely. dfPO implicitly performs trust-region-style updates (the score function simultaneously approximates the previous policy's score), but with simpler implementation.
- **vs. continuous-time RL (Wang et al. 2020, Jia & Zhou 2023)**: These works redefine the Q-function in continuous time via Hamiltonian-based formulations but require pointwise access to rewards and their gradients. dfPO requires only trajectory-level score evaluations, making it more suitable for black-box environments. The authors conjecture that $g$ is conceptually equivalent to the continuous-time $q$-function of Jia & Zhou.
- **vs. model-based RL (PILCO, SVG, iLQR)**: These methods require explicit reward models or replanning capabilities, which are unavailable in black-box scientific simulators. dfPO operates like a model-free method — requiring only score observations — while implicitly leveraging physical information through its differential structure.

## Rating
- Novelty: ⭐⭐⭐⭐ — The reformulation of RL via continuous-time control duality has theoretical depth, though the core ideas (PMP + symplectic structure) are classical in control theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 scientific computing tasks + 3 classic control tasks + 12 baselines + 10-seed statistical testing + ablations, though broader RL benchmark validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and Appendix D conveys physical intuition well, but the main text is notation-heavy and presupposes a strong mathematical background.
- Value: ⭐⭐⭐⭐ — Provides a theoretically grounded new paradigm for RL in scientific computing, though generality requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[NeurIPS 2025\] Actor-Free Continuous Control via Structurally Maximizable Q-Functions](actorfree_continuous_control_via_structurally_maximizable_qf.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)

</div>

<!-- RELATED:END -->
