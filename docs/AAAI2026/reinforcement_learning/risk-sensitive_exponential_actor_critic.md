---
title: >-
  [Paper Note] Risk-Sensitive Exponential Actor Critic
description: >-
  [AAAI2026][Reinforcement Learning][risk-sensitive RL] To address the high variance and numerical instability of policy gradients under the entropic risk measure…
tags:
  - "AAAI2026"
  - "Reinforcement Learning"
  - "risk-sensitive RL"
  - "entropic risk measure"
  - "policy gradient"
  - "actor-critic"
  - "numerical stability"
date: 2026-05-08
content_hash: 96f93756ffb64a8f
---

# Risk-Sensitive Exponential Actor Critic

**Conference**: AAAI2026
**arXiv**: [2602.07202](https://arxiv.org/abs/2602.07202)  
**Area**: Reinforcement Learning
**Keywords**: risk-sensitive RL, entropic risk measure, policy gradient, actor-critic, numerical stability

## TL;DR
To address the high variance and numerical instability of policy gradients under the entropic risk measure, this paper derives a complete set of on/off-policy risk-sensitive policy gradient theorems and proposes the rsEAC algorithm, which achieves stable risk-sensitive continuous control via log-domain critic parameterization and gradient normalization-clipping mechanisms.

## Background & Motivation
Standard RL optimizes expected return, but risk-aware decision-making is required in domains such as autonomous driving, robotics, and finance. The entropic risk measure is a widely used risk metric:

$$J^\beta(\pi_\theta) = \frac{1}{\beta} \log \mathbb{E}_{p_\pi(\tau)} \left[ e^{\beta \sum_t r_t} \right]$$

- $\beta > 0$: risk-seeking
- $\beta < 0$: risk-averse
- $\beta \to 0$: reduces to the standard risk-neutral objective

**Core difficulties of existing methods**:
- Gradient estimation via the likelihood ratio trick requires complete trajectories and is scaled by the exponentiated return, resulting in high variance and numerical instability
- The exponential value function $Z^\beta(s,a) = e^{\beta Q(s,a)}$ is highly susceptible to overflow/underflow under function approximation
- Existing model-free methods (e.g., R-AC) are limited to simple tasks and tabular settings

## Method

### Risk-Sensitive Policy Gradient Theorems
**Theorem 1 (Stochastic Policy)**:

$$\nabla_\theta J^\beta = \frac{1}{\beta} \int_S \rho_\pi^*(s) \int_A \nabla_\theta \pi_\theta(a|s) \cdot e^{\beta(Q^\beta(s,a) - V^\beta(s))} \, da \, ds$$

where $\rho_\pi^*$ is the state distribution under the exponential twisted dynamics. The key distinction is that the Q-value is replaced by the exponentiated advantage, which introduces numerical risk.

**Theorem 2 (Deterministic Policy)**:

$$\nabla_\theta J^\beta = \int_S \rho_\mu^*(s) \nabla_\theta \mu_\theta(s) \nabla_a Q^\beta_{\mu_\theta}(s,a) \big|_{a=\mu_\theta(s)} \, ds$$

This formulation avoids exponential terms and action integration, making it more suitable for practical use. Deterministic policy improvement under off-policy approximation is also established (Theorem 3).

### Log-Domain Critic Parameterization
Directly learning $Z_\psi(s,a) = e^{\beta Q(s,a)}$ causes numerical overflow/underflow. The key improvement is:

- Parameterize as $Z_\psi(s,a) = e^{Q_\psi(s,a)}$, where $Q_\psi$ is a neural network
- $\frac{1}{\beta} Q_\psi(s,a)$ approximates the soft-value function
- Gradients are computed in the log domain, yielding inherently greater stability

### Gradient Stabilization Mechanisms
Gradients of the exponential TD loss take the form $e^x(e^x - e^y)$, requiring stabilization:

1. **Helper function**: $f(x,y) = (1-e^{y-x})$ or $(e^{x-y}-1)$, bounded in $[-1,1]$, ensuring stability
2. **Batch normalization**: the exponential prefactor is normalized by subtracting the in-batch mean $z$
3. **Gradient clipping**: the range of exponential arguments is restricted to prevent gradient explosion/vanishing

### rsEAC Algorithm
Built on the TD3 framework:
- **Twin critics**: two $Q_\psi$ networks; minimum is taken when $\beta > 0$ and maximum when $\beta < 0$ (to control overestimation direction)
- **Actor**: deterministic policy $\mu_\theta$, updated via the off-policy deterministic gradient
- **Exploration**: Gaussian noise is added

## Key Experimental Results

### GridWorld (Tabular)
Validates the risk-modulation effect of $\beta$:
- $\beta = -1$: the risk-averse policy detours around cliff regions
- $\beta = 1$: the risk-seeking policy takes the shortest path along the cliff edge
- Large $|\beta|$ causes exponential value function overflow, confirming the necessity of stabilization

### Inverted Pendulum
- rsEAC learns high-return policies under both $\beta=1$ and $\beta=-1$
- R-AC yields poor policy quality and lacks risk sensitivity due to numerical instability

### MuJoCo Risk Variants (Swimmer / HalfCheetah / Ant)
Stochastic noise risk regions are introduced ($\mathcal{N}(0,10^2)$ or $\mathcal{N}(0,7^2)$):

| Method | Risk Behavior | Mean Return Performance |
|--------|--------------|------------------------|
| rsEAC | Low risk-region visitation | Comparable to MVPI |
| R-AC | Numerically unstable | Worst across all tasks |
| MVPI | Low risk-region visitation | Weaker on Ant |
| MG (PPO) | Moderate risk aversion | Moderate |

- rsEAC outperforms R-AC on all tasks, validating the critical role of the stabilization mechanism
- rsEAC outperforms MVPI on the high-dimensional Ant task

### Stability Comparison (CartPole)
- Direct learning of $Z_\psi$: value function estimates overflow/underflow; only 1 out of 4 $\beta$ settings learns the optimal policy
- $Q_\psi$ (proposed) + gradient normalization-clipping: optimal policies are learned under all $\beta$ settings

## Highlights & Insights
- **Complete theoretical framework**: the first work to derive all four policy gradient theorems for the entropic risk measure across on/off-policy × stochastic/deterministic settings
- **Practical stabilization scheme**: log-domain parameterization + batch normalization + clipping addresses the long-standing challenge of learning exponential value functions
- **Tunable risk**: a single parameter $\beta$ controls the degree of risk-seeking or risk-aversion
- **First model-free risk-sensitive actor-critic capable of handling complex continuous control tasks**

## Limitations & Future Work
- The inherent instability of the exponential function cannot be fully eliminated; failures may still occur under extreme $\beta$ values
- The risk parameter $\beta$ requires task-specific tuning, with no adaptive mechanism provided
- Validation is limited to MuJoCo continuous control tasks; safety-critical scenarios remain untested

## Rating
- Novelty: ⭐⭐⭐⭐ — Rigorous derivation of policy gradient theorems with clear engineering contributions in stabilization
- Experimental Thoroughness: ⭐⭐⭐ — Reasonable coverage of tabular and continuous control settings, though task diversity is somewhat limited
- Writing Quality: ⭐⭐⭐⭐ — Theory is presented clearly, and numerical issues are intuitively visualized
- Value: ⭐⭐⭐⭐ — Paves the way for practical deployment of risk-sensitive RL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[ICLR 2026\] Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator](../../ICLR2026/reinforcement_learning/emergence_of_spatial_representation_in_an_actor-critic_agent_with_hippocampus-in.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](../../NeurIPS2025/reinforcement_learning/global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
