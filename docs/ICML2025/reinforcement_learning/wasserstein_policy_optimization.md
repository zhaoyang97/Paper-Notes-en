---
title: >-
  [Paper Note] Wasserstein Policy Optimization
description: >-
  [ICML2025][Reinforcement Learning][Wasserstein Gradient Flow] Wasserstein Policy Optimization (WPO) is proposed, which projects the Wasserstein gradient flow from optimal transport theory onto the parameter space. This yields a closed-form update rule that both enjoys the benefits of deterministic policy gradients (DPG) utilizing action-value gradients and supports arbitrary distributions like classic stochastic policy gradients (SPG), without requiring reparameterization tri…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Wasserstein Gradient Flow"
  - "Policy Gradient"
  - "Optimal Transport"
  - "Actor-Critic"
  - "Continuous Action Spaces"
date: 2026-05-08
content_hash: ae03bd05b2a64d7b
---

# Wasserstein Policy Optimization

**Conference**: ICML2025  
**arXiv**: [2505.00663](https://arxiv.org/abs/2505.00663)  
**Code**: [google-deepmind/acme](https://github.com/google-deepmind/acme)  
**Area**: Policy Optimization / Reinforcement Learning / Continuous Control  
**Keywords**: Wasserstein Gradient Flow, Policy Gradient, Optimal Transport, Actor-Critic, Continuous Action Spaces

## TL;DR

Wasserstein Policy Optimization (WPO) is proposed, which projects the Wasserstein gradient flow from optimal transport theory onto the parameter space. This yields a closed-form update rule that both enjoys the benefits of deterministic policy gradients (DPG) utilizing action-value gradients and supports arbitrary distributions like classic stochastic policy gradients (SPG), without requiring reparameterization tricks.

## Background & Motivation

Policy optimization in continuous action spaces is a core challenge in deep reinforcement learning. Existing methods primarily fall into two categories:

**Classic Stochastic Policy Gradients (SPG)**: e.g., REINFORCE, PPO, MPO. They update parameters via $\nabla_\theta \log \pi_\theta(\mathbf{a}|\mathbf{s})$. While applicable to arbitrary stochastic policies, they only utilize scalar $Q$-values, leading to high variance and low sample efficiency in high-dimensional action spaces.

**Deterministic Policy Gradients (DPG)**: e.g., DDPG, TD3. They utilize $\nabla_\mathbf{a} Q(\mathbf{s}, \mathbf{a})$ to provide directional information, which is more efficient in high dimensions, but they are limited to deterministic policies and suffer from hard exploration. Extended schemes (e.g., SVG(0), SAC) rely on reparameterization tricks, which restricts the classes of policy distributions.

**Core Problem**: Can a policy update method be designed to simultaneously utilize the gradients of the action-value function (like DPG) and learn arbitrary stochastic policies (like SPG), without relying on reparameterization tricks?

## Method

### Core Idea: Wasserstein Gradient Flow

The starting point of WPO is the gradient flow theory on the space of probability distributions. Given any policy functional $\mathcal{J}[\pi]$, its steepest descent under the 2-Wasserstein metric satisfies the continuity equation:

$$\frac{\partial \pi}{\partial t} = -\nabla_\mathbf{a} \cdot \left(\pi \left(-\nabla_\mathbf{a} \frac{\delta \mathcal{J}}{\delta \pi}\right)\right)$$

For the expected return in an MDP, the functional derivative is $\frac{\delta \mathcal{J}}{\delta \pi}(\mathbf{s}, \mathbf{a}) = \frac{1}{1-\gamma} Q^\pi(\mathbf{s}, \mathbf{a}) d^\pi(\mathbf{s})$, rendering the velocity field $\nabla_\mathbf{a} Q^\pi(\mathbf{s}, \mathbf{a})$—which is precisely the gradient of the action-value function.

### Parametric Projection: From PDE to Closed-Form Updates

Projecting the non-parametric gradient flow onto the neural network parameter space $\theta$ by minimizing $D_\text{KL}[\pi_\theta \| \pi_\theta + \frac{\partial \pi}{\partial t} dt - \nabla_\theta \pi_\theta \Delta\theta]$ and utilizing a quadratic approximation of the Fisher Information Matrix yields:

$$\Delta\theta = \mathcal{F}_{\theta\theta}^{-1} \underbrace{\mathbb{E}_{\mathbf{a}\sim\pi}\left[\nabla_\theta \nabla_\mathbf{a} \log \pi_\theta(\mathbf{a}|\mathbf{s}) \cdot \nabla_\mathbf{a} Q^\pi(\mathbf{s}, \mathbf{a})\right]}_{\mathcal{F}_{t\theta}}$$

Here, $\mathcal{F}_{t\theta}$ is simplified into the above expression via integration by parts—the **core WPO update formula** (Equation 6).

### Practical Approximation and Loss

1. **Natural Gradient Approximation**: For Gaussian policies $\pi = \mathcal{N}(\mu_\theta, \Sigma_\theta)$, diagonal Fisher matrix structures are utilized to multiply $\nabla_{\mu_i} \log \pi$ by $\sigma_i^2$ and $\nabla_{\sigma_i} \log \pi$ by $\frac{1}{2}\sigma_i^2$ for variance normalization, preventing gradient explosion when the policy collapses.
2. **KL Regularization**: To prevent the policy from shifting too rapidly, an MPO-style KL penalty is adopted:

$$\max_\pi \mathbb{E}\left[\sum_t \gamma^t \left(\mathbb{E}_{\mathbf{a}\sim\pi}[r_t] - \alpha D_\text{KL}[\bar{\pi}(\cdot|\mathbf{s}_t) \| \pi(\cdot|\mathbf{s}_t)]\right)\right]$$

3. **Critic Update**: Standard $n$-step TD target with target networks.

### Equivalence Analysis with Existing Methods

Under a univariate Gaussian policy:
- WPO mean update $\Delta_\mu \theta = \mathbb{E}_pi[\nabla_a Q(a) \nabla_\theta \mu]$ is structurally consistent with the DPG update (but taking the gradient at the sampled point rather than at the mean).
- WPO variance update $\Delta_\sigma \theta = \mathbb{E}_pi[\frac{a-\mu}{\sigma} \nabla_a Q(a) \nabla_\theta \sigma]$ is equivalent to the SVG(0) reparameterized update.
- Furthermore, it can be proven that the expected WPO update = classic policy gradient $\mathbb{E}_pi[Q(a) \nabla_\theta \log \pi(a)]$.

**However**: (1) sampling variance differs—when $Q$ is locally linear w.r.t. actions, the WPO mean update has zero variance; (2) for non-Gaussian policies (such as Mixture of Gaussians), WPO is qualitatively different from SPG, converging faster and more stably.

## Key Experimental Results

### DeepMind Control Suite (28 tasks)

| Method | Overall Performance | High-dimensional Task (Humanoid CMU) | Stability |
|------|---------|------------------------|--------|
| **WPO** | Comparable to SOTA on almost all tasks | Learns faster in early stage of Walk | ✅ Most robust |
| MPO | Strong baseline | Stably learns | ✅ Robust |
| SAC | Poor convergence on some tasks | Completely fails to start | ⚠️ Sensitive to entropy weight |
| DDPG | Poor convergence on some tasks | Completely fails to start | ⚠️ Struggles with sparse rewards |

### High-Dimensional Scaling Experiments (Combined Humanoid Stand)

| Action Dimension | 1×21 | 3×65 | 5×105 |
|----------|-------|-------|-------|
| WPO Startup Speed | Comparable to MPO | Significantly faster than MPO | **Substantially leading** |
| SAC | Slowest convergence | Slowest convergence | Slowest convergence |

**Key Findings**: As the action dimension grows, the startup advantage of WPO increases accordingly, suggesting its advantage is more pronounced in hundred-dimensional action spaces.

### Nuclear Fusion Magnetic Control (Tokamak, 19-dimensional actions, 93-dimensional observations)

- The reward of WPO is slightly higher than that of MPO.
- The policy variance of WPO progressively approaches zero during training (matching expectations in a fully observable environment), whereas MPO maintains a near-constant variance.

## Highlights & Insights

1. **Theoretical Elegance**: Grounded in optimal transport theory, a clean closed-form update is derived through Wasserstein gradient flow $\rightarrow$ continuity equation $\rightarrow$ Fisher projection, unifying both the DPG and SPG paradigms.
2. **Strong Versatility**: It does not rely on reparameterization tricks and is applicable to any continuous policy distribution (exponential family, Gaussian mixtures, etc.), overcoming the distributional limits of SAC/SVG(0).
3. **Low-Variance Advantage**: When the action gradient of $Q$ points in consistent directions across sample points (e.g., locally linear), the variance of WPO updates is far lower than classic policy gradients.
4. **High-Dimensional Scalability**: Empirical results demonstrate that the performance advantage increases with action dimensions, substantially leading baselines in 105-dimensional joint control.
5. **Practical Feasibility**: It has been open-sourced and integrated into the Acme framework; engineering modifications are straightforward (variance normalization + KL regularization).

## Limitations & Future Work

1. **Degraded Performance under Partial Observability**: Poor performance in the Dog domain (state > observation) suggests supplementary adaptation might be required in POMDP settings.
2. **Evaluation Limited to Gaussian Policies**: Although arbitrary distributions are theoretically supported, experiments solely utilized diagonal Gaussian policies. The empirical performance with non-Gaussian policies (such as Gaussian mixtures, flow-based models) remains to be verified.
3. **Coarse Natural Gradient Approximation**: Only diagonal Fisher scaling at the policy distribution level is utilized, neglecting the Fisher matrix structure over the network parameter dimensions.
4. **Limited Hyperparameter Tuning**: The authors acknowledge that WPO underwent much less tuning compared to the years of accumulation for DDPG/SAC, indicating potential room for further performance gains.
5. **Inapplicable to Discrete Actions**: The theory relies on gradient flows in continuous spaces, preventing direct extension to discrete action spaces.

## Related Work & Insights

- **MPO** (Abdolmaleki et al., 2018): WPO inherits its KL regularization scheme from MPO; their update formulations are equivalent under Gaussian policies but exhibit different variance profiles.
- **SAC** (Haarnoja et al., 2018): The reparameterized version is equivalent to WPO under Gaussian distributions but fails to apply to non-reparameterizable distributions.
- **Other Applications of Wasserstein RL**: Abdullah et al. (2019) used it for robust MDPs, and Moskovitz et al. (2020) adopted it as a preconditioner—both differ from WPO (which works at the velocity-field level of gradient flows).

## Rating

- Novelty: ⭐⭐⭐⭐ — A novel bridge between optimal transport theory and policy optimization, featuring an elegant mathematical derivation for closed-form updates.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers Control Suite, high-dimensional scaling, and real nuclear fusion tasks, but lacks experiments on non-Gaussian policies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and rigorous, with excellent intuitive pedagogical figures (Fig 1-3).
- Value: ⭐⭐⭐⭐ — Provides a solid new algorithmic choice for high-dimensional continuous control, with particularly striking results on high-dimensional scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/latent_wasserstein_adversarial_imitation_learning.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](../../NeurIPS2025/reinforcement_learning/sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[ICLR 2026\] Single-stream Policy Optimization](../../ICLR2026/reinforcement_learning/single-stream_policy_optimization.md)
- [\[ACL 2025\] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](../../ACL2025/reinforcement_learning/bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)

</div>

<!-- RELATED:END -->
