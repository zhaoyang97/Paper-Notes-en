---
title: >-
  [Paper Note] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents
description: >-
  [NeurIPS 2025][Reinforcement Learning][Constrained Reinforcement Learning] This paper proposes a reward-based risk-aware constrained RL framework that applies Optimized Certainty Equivalent (OCE) risk measures to both ob…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Constrained Reinforcement Learning"
  - "Risk Aversion"
  - "Optimized Certainty Equivalent (OCE)"
  - "CVaR"
  - "Partial Lagrangian Relaxation"
date: 2026-05-08
content_hash: fc373ac625f1b455
---

# Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents

**Conference**: NeurIPS 2025
**arXiv**: [2510.20199](https://arxiv.org/abs/2510.20199)  
**Code**: [Available](https://github.com/baturaysaglam/risk-averse-constrained-RL)  
**Area**: Reinforcement Learning / Risk Aversion
**Keywords**: Constrained Reinforcement Learning, Risk Aversion, Optimized Certainty Equivalent (OCE), CVaR, Partial Lagrangian Relaxation

## TL;DR

This paper proposes a reward-based risk-aware constrained RL framework that applies Optimized Certainty Equivalent (OCE) risk measures to both objectives and constraints, establishes parametric strong duality, and delivers a modular algorithm that wraps standard RL solvers (e.g., PPO) as a black box.

## Background & Motivation

Constrained RL is a common framework for handling conflicting objectives (e.g., reaching a goal while avoiding walls in maze navigation). Standard constrained RL expresses objectives and constraints via expected cumulative rewards, which is insufficiently rigorous for high-stakes applications:

**Limitations of Expectation**: Expected values fail to capture catastrophic tail events in return distributions. For instance, maximizing average returns in portfolio management may overlook the risk of large losses.

Existing risk-averse RL paradigms each have shortcomings:

**Return-based**: Applies a risk measure in place of expectation to discounted cumulative returns, $\rho[-\sum \gamma^\tau r]$. This captures risk only over the overall return distribution and is insensitive to risk at **individual time steps**.

**Recursive Risk**: Recursively evaluates risk measures at each decision stage, generalizing the tower property of expectations. Computationally demanding and difficult to scale.

**This paper proposes a third paradigm — reward-based**: Applies the risk measure to the **occupancy measure**, i.e.:

$$R^* = \sup_{\nu^\pi} \frac{1}{1-\gamma} \cdot -\rho_{\nu^\pi}(-r(s,a))$$

**Key Advantage**: This paradigm provides **stepwise robustness** — capturing risk simultaneously across reward values and time steps. As CVaR's $\beta \to 0$, the objective degrades to the essential infimum of the worst reward over all state-action pairs at all time steps, a strictly stronger safety guarantee than the return-based approach.

**Core Challenges**:
- How to incorporate reward-based risk measures into constrained RL?
- How to establish duality guarantees (making Lagrangian relaxation exact)?
- How to design practical algorithms compatible with existing RL methods?

## Method

### Overall Architecture

For general OCE risk measures (including CVaR, mean semi-variance, etc.), the constrained problem is written as:

$$\sup_{\pi,t_0} \mathbb{E}[\sum \gamma^\tau r_0'(s_\tau, a_\tau, t_0)] \quad \text{s.t.} \quad \sup_{t_i} \mathbb{E}[\sum \gamma^\tau r_i'(s_\tau, a_\tau, t_i)] \geq c_i$$

where the modified reward $r_i'(s,a,t) = t - \frac{1}{\beta}(t - r_i(s,a))_+$ (for CVaR), and the auxiliary variable $t$ controls the degree of risk aversion.

### Key Designs

#### 1. Parametric Strong Duality and Partial Lagrangian Relaxation

**Function**: Proves that the constrained problem can be solved exactly via partial Lagrangian relaxation.

**Mechanism**:
- With $t \in \mathcal{T}$ fixed, the problem reduces to standard constrained RL → strong duality holds under Slater's condition (Proposition 3.3).
- A constraint qualification is introduced (Assumption 3.4): there exists a convex compact set $\mathcal{I} \subset \mathcal{T}$ such that Slater's condition holds for all $t \in \mathcal{I}$.
- Under this condition, the resulting partial dual problem is exactly equivalent to the original constrained problem.

$$D_\theta^* = \sup_{t \in \mathcal{T}} \inf_{\lambda \in \Lambda} \underbrace{\sup_\theta \mathcal{L}(\pi_\theta, t, \lambda)}_{\text{black-box RL}}$$

**Design Motivation**: To make the dual relaxation exact rather than approximate — a first in the risk-averse constrained RL literature.

#### 2. Modular Algorithm Design (Algorithm 1)

**Function**: Decomposes the problem into an inner RL subproblem and an outer $(t, \lambda)$ update.

For fixed $(t, \lambda)$, the inner problem is standard RL (with a modified reward function) solvable by any RL algorithm (e.g., PPO). The outer loop updates via SGDA:

$$\lambda^{(j+1)} \leftarrow \Pi_\Lambda(\lambda^{(j)} - \eta_\lambda \hat{\nabla}_\lambda \hat{\mathcal{L}})$$
$$t^{(j+1)} \leftarrow \Pi_\mathcal{T}(t^{(j)} + \eta_t \hat{\nabla}_t \hat{\mathcal{L}})$$

**Design Motivation**: The modular structure allows users to flexibly choose any combination of risk-neutral/risk-averse objectives and/or constraints, and to plug in any existing RL algorithm as a black-box subproblem solver.

#### 3. Approximate Optimality Guarantee (Theorem 3.5)

Under $\epsilon$-universal policy parameterization:

$$P^*(t^*) \geq \sup_t \inf_\lambda \sup_\theta \mathcal{L}(\pi_\theta, t, \lambda) \geq P^*(t^*) - \mathcal{O}\left(\frac{\epsilon}{1-\gamma}\right)$$

The gap between the parametric partial dual and the true primal problem depends only on the policy parameterization error.

### Convergence Analysis

**Theorem 3.12**: Under assumptions of Lipschitz smoothness, unbiased gradient oracles, and an approximate policy solver, the iteration complexity to recover an $\epsilon$-stationary point is:

$$\mathcal{O}\left(\frac{\ell^3(C^2+\sigma^2+\delta^2)(\text{diam}(\Lambda))^2 \hat{\Delta}_\Phi}{\epsilon^6}\right)$$

Key feature: only a single trajectory ($n=1$) is required, enabling online operation. When the inexact solver bias satisfies $\delta = \mathcal{O}(\epsilon^2)$, exact $\epsilon$-stationary points are recovered.

## Key Experimental Results

### Main Results: Safe Navigation (Safety-Gymnasium)

Point agent at Level 1 difficulty, 5–10M training steps:

| Environment | PPO Cumulative Cost ↓ | MARS Cumulative Cost ↓ | PPO Reward ↑ | MARS Reward ↑ |
|-------------|----------------------|----------------------|-------------|--------------|
| Button | 150.76 | **0.0** | 24.29 | 2.58 |
| Circle | 206.74 | **0.0** | 60.18 | 39.19 |
| Goal | 45.09 | **0.0** | 21.89 | 13.56 |
| Push | 38.48 | **0.0** | 0.93 | **2.42** |

MARS achieves **zero constraint violations** across all environments, being the only PPO-based method to do so. In the Push environment, the constraint even helps the agent achieve higher reward.

### Ablation Study: Safe Velocity Constraint (MuJoCo-v4)

| Agent | Velocity Threshold $c$ | $\beta$-upper Quantile | Converged $t$ | Match |
|-------|----------------------|----------------------|---------------|-------|
| HalfCheetah | 1.450 | 1.419 | 1.417 | ✓ |
| Hopper | 0.373 | 0.370 | 0.370 | ✓ |
| Swimmer | 0.228 | 0.248 | 0.207 | ≈ |
| Walker2d | 1.171 | 1.133 | 1.122 | ✓ |

### Key Findings

1. **$t$ Aligns with CVaR**: The converged $t$ value precisely matches the $\beta$-upper quantile of the trained velocity distribution, validating correct operation of the CVaR constraint.
2. **Stable $\lambda$ Convergence**: The dual variable $\lambda$ stabilizes and oscillates around a consistent value after sufficient training.
3. **Interpretable Policies**: Constrained agents move more cautiously (more stable velocity), whereas unconstrained PPO agents exhibit high velocity variance.
4. **Stabilizing Evaluation Reward**: Variance in evaluation reward decreases over training, demonstrating the effect of risk management.
5. **Only Zero-Violation Method**: In navigation tasks, MARS is the only PPO-based method in the literature to achieve strictly zero constraint violations.

## Highlights & Insights

1. **Reward-based vs. Return-based**: The paper clearly argues how reward-based risk measures provide robustness simultaneously in value and time dimensions — a stronger safety guarantee than return-based approaches.
2. **Exact Duality**: Unlike prior risk-constrained RL work (e.g., Chow et al.'s CVaR methods) that only yield approximate relaxations, this paper establishes exact equivalence under the constraint qualification — a significant theoretical contribution.
3. **High Practicality**: The black-box wrapper design enables direct use of PPO/SAC/TD3 or any other RL algorithm, lowering the implementation barrier.
4. **Flexibility**: Risk-neutral objectives can be freely combined with risk-averse constraints (as in the experiments), closely matching practical requirements.

## Limitations & Future Work

1. **Full Strong Duality Unproven**: Whether Assumption 3.4 holds unconditionally remains an open problem.
2. **High Computational Cost**: Each $(t, \lambda)$ update requires approximately solving a policy optimization subproblem, making the method slower than risk-neutral approaches.
3. **Step-Size Sensitivity**: Tuning $\eta_\lambda$ and $\eta_t$ is critical for convergence and requires considerable patience.
4. **Artificial Action Noise**: Experiments simulate risk by injecting 5% Gaussian noise into actions; real-world uncertainty may be more complex.
5. **CVaR Only Validated**: Although the theory covers general OCE measures, experiments validate only CVaR.

## Related Work & Insights

- Extends Bonetti et al.'s unconstrained reward-based risk-averse RL to the constrained setting.
- Compared to Chow et al.'s return-based CVaR constrained RL: the partial Lagrangian relaxation proposed here is exact under the constraint qualification.
- The convergence analysis framework builds on Lin et al.'s minimax optimization theory.
- Potential extensions to multi-agent settings or hierarchical RL are worth exploring.

## Rating
- Novelty: ⭐⭐⭐⭐ (Reward-based constrained RL with exact duality is a significant theoretical contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on navigation and velocity scenarios with detailed convergence curves)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical exposition; Table 1's formulation comparison is immediately informative)
- Value: ⭐⭐⭐⭐ (Strong in both theory and practice; modular design is highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)

</div>

<!-- RELATED:END -->
