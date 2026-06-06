---
title: >-
  [Paper Note] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients
description: >-
  [AAAI2026][Reinforcement Learning][optimization-based control] This paper proposes DiffOP, a framework that treats optimization-based control policies (e.g., MPC) as differentiable modules…
tags:
  - "AAAI2026"
  - "Reinforcement Learning"
  - "optimization-based control"
  - "implicit differentiation"
  - "policy gradient"
  - "model predictive control"
  - "bilevel optimization"
date: 2026-05-08
content_hash: 55d0af649272c84b
---

# DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients

**Conference**: AAAI2026
**arXiv**: [2411.07484](https://arxiv.org/abs/2411.07484)  
**Code**: [alwaysbyx/DiffOP](https://github.com/alwaysbyx/DiffOP)  
**Area**: Reinforcement Learning
**Keywords**: optimization-based control, implicit differentiation, policy gradient, model predictive control, bilevel optimization

## TL;DR

This paper proposes DiffOP, a framework that treats optimization-based control policies (e.g., MPC) as differentiable modules, derives analytic policy gradients via implicit differentiation to enable end-to-end reinforcement learning training, and provides the first non-asymptotic convergence guarantee for this setting.

## Background & Motivation

Real-world control systems (power grids, robotics, traffic networks, etc.) impose strict requirements on policy **interpretability, safety, and robustness**. Optimization-based control policies (e.g., MPC) generate actions by solving constrained optimization problems, offering inherent interpretability and constraint satisfaction.

Existing approaches suffer from two major issues:

1. **Objective Mismatch**: Conventional methods decouple the learning of dynamics models and cost functions from the control objective — a model may perform well in terms of prediction accuracy while failing to guide optimal control decisions.
2. **Limitations of Supervised Learning**: Recent differentiable optimization works (PDP, IDOC, etc.) operate primarily in a supervised imitation learning setting, relying on expert demonstrations and unable to improve policies through online interaction.
3. **Existing RL+MPC methods** (RLMPC-TD, RLMPC-DPG) rely on value function approximation and Q-learning updates, often converging to suboptimal solutions.

## Core Problem

How can optimization-based control policies, implicitly defined by optimization problems, be trained end-to-end in a **reinforcement learning setting**? Specifically, the following challenges must be addressed:

- The policy is implicitly defined by the **solution** of an optimization problem — how can gradients with respect to policy parameters be computed efficiently?
- Can the true control cost be directly optimized without relying on value function approximation?
- Does the learning process admit theoretical convergence guarantees?

## Method

### Overall Architecture

DiffOP defines the control policy as the solution to a parameterized optimization problem:

$$u_{0:H-1}^{\star}(x_{\text{init}};\theta) = \arg\min_u \sum_{i=0}^{H-1} c(x_i, u_i; \theta_c) + c_H(x_H; \theta_H)$$

where the dynamics $f(\cdot;\theta_f)$, stage cost $c(\cdot;\theta_c)$, and terminal cost $c_H(\cdot;\theta_H)$ are all learnable. The policy parameters $\theta = (\theta_c, \theta_H, \theta_f)$ jointly characterize the cost and dynamics models.

### Bilevel Optimization Formulation

Policy learning is formulated as a bilevel optimization problem:

- **Upper level**: Minimize the expected cumulative cost in the true system, $C(\theta) = \mathbb{E}[\sum_t c(x_t, u_t; \phi_c)]$
- **Lower level**: Solve a parameterized optimization problem at each decision step to obtain control actions

To support exploration, actions are sampled from a truncated Gaussian distribution: $u_i \sim \mathcal{N}(u_i^\star, \sigma^2 I)$, with truncation range controlled by hyperparameter $\beta$.

### Implicit Policy Gradient Derivation

The core technical contribution lies in applying the **implicit function theorem** to compute gradients of the optimal solution with respect to parameters. The procedure is as follows:

1. Write out the KKT conditions for the optimal trajectory $\zeta^\star$
2. Apply implicit differentiation to the KKT conditions to obtain an analytic expression for $\nabla_\theta u_i^\star$ (Proposition 1)
3. Combine with the REINFORCE gradient estimator to derive the full policy gradient (Proposition 2):

$$\nabla_\theta C(\theta) = \mathbb{E}\left[L(\tau)\sum_{t=0}^{T} \frac{1}{\sigma^2}[\nabla_\theta u_t^\star]^{\mathsf{T}}(u_t - u_t^\star)\right]$$

In practice, this is approximated via Monte Carlo sampling of $N$ trajectories.

### Algorithm (Algorithm 1)

Each iteration consists of three steps:

1. Sample $N$ trajectories; at each step, solve for the policy using an optimization solver (CasADi) and add noise for exploration
2. Compute implicit gradients $\nabla_\theta u_t^\star$ for each trajectory
3. Estimate policy gradients via Monte Carlo and update $\theta$ via gradient descent

### Convergence Guarantee

Under the bounded sensitivity assumption (Assumption 1) and bounded cost assumption (Assumption 2), the following is proven:

$$\min_{k=0,...,K-1} \|\nabla_\theta C(\theta^{(k)})\|^2 \leq \frac{16L_C(C(\theta^{(0)}) - C(\bar\theta))}{K} + 3\epsilon$$

That is, DiffOP converges to an $\epsilon$-stationary point within $\mathcal{O}(1/\epsilon)$ iterations, matching the convergence rate of standard policy gradient methods.

For the unconstrained strongly convex case, it is further shown that the bounded sensitivity condition follows directly from Lipschitz smoothness and strong convexity (Proposition 3).

### Execution Modes

DiffOP supports two deployment modes:

- **DiffOP (Step)**: Only the first action in the optimized sequence is executed at each step (analogous to standard MPC receding-horizon execution)
- **DiffOP (Traj)**: The full control sequence is generated and executed in open loop (stronger temporal consistency)

## Key Experimental Results

### Nonlinear Control Tasks (Cartpole / Robot Arm / Quadrotor)

| Method | Training Mode | Result |
|--------|---------------|--------|
| DiffOP (Step) | Online RL | Fastest convergence and lowest final cost across all tasks |
| DiffOP (Traj) | Online RL | Outperforms PDP (Offline) on Robot Arm and Quadrotor |
| RLMPC-TD | Online RL | Frequently converges to suboptimal solutions |
| RLMPC-DPG | Online RL | Unstable; cost fails to decrease consistently on some tasks |
| PDP (Offline) | Offline supervised | Limited by expert data quality |

### Voltage Control (IEEE 13-bus, 500 scenarios)

| Method | Transient Cost | Steady-State Cost |
|--------|----------------|-------------------|
| **DiffOP (Step)** | **-6.81** | **-0.11** |
| **DiffOP (Traj)** | **-6.80** | **-0.11** |
| TASRL | -6.76 | -0.11 |
| RLMPC-DPG | -6.11 | -0.10 |
| Stable-DDPG | -5.61 | -0.09 |
| PDP (Offline) | -5.86 | -0.09 |
| RLMPC-TD | -4.62 | -0.07 |

DiffOP achieves the best transient cost among all methods, with steady-state cost on par with TASRL. Post-training voltage trajectories remain within the safe operating range.

## Highlights & Insights

1. **First non-asymptotic convergence guarantee**: A convergence rate of $\mathcal{O}(1/\epsilon)$ is established, filling a theoretical gap in RL training of optimization-based policies
2. **General framework**: Requires neither LQR approximation nor value function approximation; applicable to general nonlinear constrained optimization problems
3. **Flexible deployment**: Unifies step-wise (receding-horizon) and trajectory-level (open-loop) execution modes
4. **Joint learning of cost and dynamics**: Eliminates objective mismatch by directly optimizing via environment feedback in an end-to-end manner
5. **Constraint handling**: Natural support for hard constraints is demonstrated in the voltage control experiments

## Limitations & Future Work

1. **Strong convexity assumption**: Theoretical convergence guarantees rely on strong convexity of the cost function with respect to control variables; sensitivity may be unbounded in non-convex settings
2. **Non-smoothness at constraint boundaries**: Changes in the active set of inequality constraints may cause gradient discontinuities, which are not covered by the theoretical analysis
3. **Sample efficiency**: Each policy update requires sampling $N$ complete trajectories, leading to substantial computational and sample costs
4. **Solver dependency**: An optimization solver (CasADi) must be invoked at every step, incurring higher inference overhead than pure neural network policies
5. **Simple exploration mechanism**: Only independent Gaussian noise is used; more structured exploration strategies are not considered

## Related Work & Insights

| Dimension | DiffOP | PDP | RLMPC-TD/DPG | Stable-DDPG/TASRL |
|-----------|--------|-----|--------------|-------------------|
| Training | Online RL | Offline supervised | Online RL | Online RL |
| Gradient computation | Implicit differentiation | PMP differentiation | Q-learning | Backpropagation |
| Value function | Not required | N/A | Required | Required |
| Constraint support | Native | Supported | Supported | By design |
| Convergence guarantee | Yes (non-asymptotic) | No | No | Partial (stability) |
| Policy form | Solution to opt. problem | Solution to opt. problem | Short-horizon MPC | Neural network |

**Broader implications and connections:**

- The **implicit differentiation + RL paradigm** can be extended to other decision-making systems embedding optimization layers (e.g., combinatorial optimization, scheduling)
- The bilevel optimization perspective provides a unified theoretical framework for understanding MPC parameter learning
- The voltage control experiments demonstrate that optimization-based policies are more deployable than black-box RL in safety-critical systems
- Future work may incorporate actor-critic methods to reduce variance, or leverage warm-starting to accelerate the solver

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of implicit differentiation and policy gradients offers technical novelty, though it builds upon prior work in IDOC/PMP
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two experimental groups (nonlinear control and voltage control) provide good coverage, but high-dimensional tasks and more real-world scenarios are lacking
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and the structure is clear
- **Value**: ⭐⭐⭐⭐ — Establishes a theoretical foundation for RL training of optimization-based policies, with practical relevance for safe control

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[ICML 2026\] Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](../../ICML2026/reinforcement_learning/chebyshev_policies_and_the_mountain_car_problem_reinforcement_learning_for_low-d.md)
- [\[ICML 2026\] Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation](../../ICML2026/reinforcement_learning/randomized_advantage_transformation_rat_computing_natural_policy_gradients_via_d.md)
- [\[AAAI 2026\] Test-driven Reinforcement Learning in Continuous Control](test-driven_reinforcement_learning_in_continuous_control.md)

</div>

<!-- RELATED:END -->
