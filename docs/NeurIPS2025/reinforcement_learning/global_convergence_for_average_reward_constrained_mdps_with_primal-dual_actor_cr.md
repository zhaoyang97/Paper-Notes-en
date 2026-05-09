---
title: >-
  [Paper Note] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic
description: >-
  [NeurIPS 2025][Reinforcement Learning][Constrained MDP] This paper proposes the Primal-Dual Natural Actor-Critic (PDNAC) algorithm, which achieves, for the first time, a global convergence rate of $\tilde{\mathcal{O}}(1/\sqrt{T})$ and a constraint violation rate of $\tilde{\mathcal{O}}(1/\sqrt{T})$ for average reward constrained MDPs under general parameterized policies, matching the theoretical lower bound.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Constrained MDP
  - Average Reward
  - Primal-Dual
  - Natural Policy Gradient
  - Global Convergence
date: 2026-05-08
content_hash: 90c8a9662c8ef127
---

# Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic

**Conference**: NeurIPS 2025
**arXiv**: [2505.15138](https://arxiv.org/abs/2505.15138)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Constrained MDP, Average Reward, Primal-Dual, Natural Policy Gradient, Global Convergence

## TL;DR

This paper proposes the Primal-Dual Natural Actor-Critic (PDNAC) algorithm, which achieves, for the first time, a global convergence rate of $\tilde{\mathcal{O}}(1/\sqrt{T})$ and a constraint violation rate of $\tilde{\mathcal{O}}(1/\sqrt{T})$ for average reward constrained MDPs under general parameterized policies, matching the theoretical lower bound.

## Background & Motivation

The infinite-horizon average reward setting is crucial for modeling realistic long-term objectives, such as delivery time constraints in transportation networks and resource budget limitations in communication networks. Constrained Markov Decision Processes (CMDPs) address these constraints by introducing cost functions, requiring the maximization of average reward while ensuring that average costs do not exceed prescribed thresholds.

**Core Gap**:

| Setting | Best Known Rate | Lower Bound |
|---------|----------------|-------------|
| Tabular CMDP | $\tilde{\mathcal{O}}(1/\sqrt{T})$ ✓ | $\Omega(1/\sqrt{T})$ |
| Linear MDP CMDP | $\tilde{\mathcal{O}}(1/\sqrt{T})$ ✓ | $\Omega(1/\sqrt{T})$ |
| **General Parameterized CMDP** | $\tilde{\mathcal{O}}(1/T^{1/5})$ ✗ | $\Omega(1/\sqrt{T})$ |

General parameterization—indexing policies via finite-dimensional parameters with $d \ll |\mathcal{S}||\mathcal{A}|$—is a key approach for handling large or infinite state spaces. However, the prior best rate of $\tilde{\mathcal{O}}(1/T^{1/5})$ falls far short of the lower bound.

**Key Challenge**: CMDPs under general parameterization lack strong convexity. When primal-dual methods are applied directly, convergence of the dual problem does not automatically translate to convergence of the primal problem. The choice of dual learning rate $\beta$ faces an inherent tension: a small $\beta$ leads to slow constraint violation convergence, while a large $\beta$ inflates the variance of primal updates.

## Method

### Overall Architecture

PDNAC addresses the CMDP by solving the following saddle-point optimization:
$$\max_{\theta \in \Theta} \min_{\lambda \geq 0} \mathcal{L}(\theta, \lambda) = J_r(\theta) + \lambda J_c(\theta)$$

The algorithm adopts a nested-loop structure:
- **Outer loop** ($K$ rounds): updates the primal parameter $\theta_k$ and dual parameter $\lambda_k$
- **Inner loop** ($H$ steps): runs a natural policy gradient (NPG) subroutine and estimates optimal critic parameters

**Key Parameter Design**: To achieve $\mathcal{O}(1/\sqrt{T})$ convergence, $H$ is set to be approximately constant while $K = \tilde{\Theta}(T)$, making the algorithm effectively near-single-timescale. This contrasts sharply with the typical unconstrained MDP setting of $K = H = \Theta(\sqrt{T})$.

### Key Designs

#### Critic Estimation (MLMC-based)

The critic subroutine estimates two quantities: the average reward/cost $J_g(\theta_k)$ and the value function $V_g^{\pi_{\theta_k}}$.

**Average Reward Estimation**: $J_g(\theta_k)$ is expressed as the solution to:
$$\min_{\eta \in \mathbb{R}} R_g(\theta_k, \eta) = \frac{1}{2}\sum_{s,a} \nu_g^{\pi_{\theta_k}}(s,a)\{\eta - g(s,a)\}^2$$

**Value Function Estimation**: A linear critic $\hat{V}_g(\zeta, s) = \langle \phi_g(s), \zeta \rangle$ approximates $V_g^{\pi_{\theta_k}}$.

**MLMC Estimator**: The key innovation is the use of Multi-Level Monte Carlo (MLMC) to estimate gradients. For each inner-loop step $(k,h)$:
1. Sample $Q_h^k \sim \text{Geom}(1/2)$
2. Collect samples along trajectories of length $l_{kh} = 2^{Q_h^k}$
3. Construct the MLMC gradient estimate: $\mathbf{v}_g = \mathbf{v}_{g,kh}^0 + 2^{Q_h^k}(\mathbf{v}_{g,kh}^{Q_h^k} - \mathbf{v}_{g,kh}^{Q_h^k - 1})$

**Advantages of MLMC**:
- Achieves the same bias as averaging $T_{\max}$ samples, but requires only $\tilde{\mathcal{O}}(\log T_{\max})$ samples
- Geometric distribution sampling eliminates the need for mixing time knowledge, removing the mixing time assumption of prior work
- Avoids storing trajectories of length $H$, reducing memory complexity by a factor of $H$

#### NPG Estimator

Given the critic estimate $\xi_g^k = [\eta_g^k, (\zeta_g^k)^\top]^\top$, the natural policy gradient is estimated over $H$ inner-loop steps:
$$\omega_{g,h+1}^k = \omega_{g,h}^k - \gamma_\omega \hat{\nabla}_\omega f_g(\theta_k, \omega_{g,h}^k, \xi_g^k)$$

where the gradient estimate uses the TD error as an advantage estimate:
$$\hat{A}_g = g(s,a) - \eta_g^k + \zeta_g^k(\phi_g(s') - \phi_g(s))$$

The final combined NPG direction is $\omega_k = \omega_r^k + \lambda_k \omega_c^k$.

### Loss & Training

**Primal-Dual Updates**:
$$\theta_{k+1} = \theta_k + \alpha \omega_k, \quad \lambda_{k+1} = \mathcal{P}_{[0, 2/\delta]}[\lambda_k - \beta \eta_c^k]$$

where $\alpha = T^{-1/2}$ (primal learning rate), $\beta = T^{-1/2}$ (dual learning rate), and $\delta$ is the Slater condition parameter.

**Key Convergence Decomposition** (Lemma 4.6): The global convergence rate decomposes as:
$$\frac{1}{K}\sum_k (\mathcal{L}(\pi^*, \lambda_k) - \mathcal{L}(\theta_k, \lambda_k)) \leq \sqrt{\epsilon_{\text{bias}}} + \text{NPG bias term} + \text{variance term} + \frac{1}{\alpha K}\text{KL divergence}$$

Precise control of the NPG bias and variance is achieved via the critic/actor convergence bounds established in Theorems 4.7 and 4.8.

## Key Experimental Results

### Main Results (Theoretical Comparison)

| Algorithm | Global Convergence | Constraint Violation | Unknown Mixing Time | Model-free | Setting |
|-----------|-------------------|---------------------|--------------------|-----------|----|
| Chen et al. (Alg.1) | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | ✗ | ✗ | Tabular |
| UC-CURL | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | 0 | ✓ | ✗ | Tabular |
| Ghosh (Alg.3) | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | ✗ | — | Linear MDP |
| Bai et al. (Prev. SOTA) | $\tilde{\mathcal{O}}(1/T^{1/5})$ | $\tilde{\mathcal{O}}(1/T^{1/5})$ | ✗ | ✓ | General Param. |
| **PDNAC (Ours)** | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | $\tilde{\mathcal{O}}(1/\sqrt{T})$ | ✗ | ✓ | **General Param.** |
| **PDNAC ($\tau$ unknown)** | $\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$ | $\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$ | ✓ | ✓ | **General Param.** |
| Lower Bound | $\Omega(1/\sqrt{T})$ | — | — | — | — |

### Ablation Study (Relationship Between Two Theorems)

**Theorem 4.9** (Known Mixing Time):
- $H = \tilde{\Theta}(\tau_{\text{mix}}^2)$, $K = T/H$
- Convergence rate: $\mathcal{O}(\sqrt{\epsilon_{\text{bias}}} + \sqrt{\epsilon_{\text{app}}} + 1/\sqrt{T})$

**Theorem 4.10** (Unknown Mixing Time):
- $H = T^\epsilon$, $K = T^{1-\epsilon}$
- Condition: $T \geq \tilde{\Theta}(\tau_{\text{mix}}^{2/\epsilon})$
- Convergence rate: $\mathcal{O}(\sqrt{\epsilon_{\text{bias}}} + \sqrt{\epsilon_{\text{app}}} + 1/T^{0.5-\epsilon})$
- Smaller $\epsilon$ yields near-optimal rates but requires a longer time horizon

**NPG and Critic Error Analysis** (Theorems 4.7 & 4.8):

| Error Term | Upper Bound Components |
|------------|----------------------|
| Critic bias $\|\mathbb{E}[\xi_g^k] - \xi_g^*\|^2$ | $1/T^2 + \tau_{\text{mix}}^2/T_{\max}$ |
| Critic variance $\mathbb{E}[\|\xi_g^k - \xi_g^*\|^2]$ | $1/T^2 + \tau_{\text{mix}}/H + \tau_{\text{mix}}/T_{\max}$ |
| NPG bias | Critic bias $+ \epsilon_{\text{app}} + \tau_{\text{mix}}^2/T^2$ |
| NPG variance | Critic variance $+ \epsilon_{\text{app}}$ |

### Key Findings

1. The theoretical lower bound $\Omega(1/\sqrt{T})$ is matched for the first time in general parameterized CMDPs.
2. A substantial improvement over the prior SOTA rate of $\tilde{\mathcal{O}}(1/T^{1/5})$ is demonstrated.
3. Setting $H$ to be approximately constant with $K \approx T$ is the critical design choice, rendering the algorithm near-single-timescale.
4. The MLMC estimator simultaneously addresses both sample efficiency and the unknown mixing time challenge.
5. $\epsilon_{\text{bias}}$ and $\epsilon_{\text{app}}$ determine irreducible approximation residuals.

## Highlights & Insights

- **Closing the Constrained-vs-Unconstrained Gap**: Prior work showed CMDP rates far inferior to unconstrained MDPs ($1/T^{1/5}$ vs. $1/T^{1/4}$); this paper demonstrates that the two can be matched.
- **Near-Single-Timescale Reduction**: By setting $H$ to be approximately constant, the nested-loop algorithm is effectively reduced to near-single-timescale, with MLMC maintaining low bias.
- **Multiple Benefits of MLMC**: (a) low sample complexity; (b) elimination of the mixing time assumption; (c) reduced memory footprint.
- **Careful Dual Learning Rate Balancing**: The symmetric choice $\alpha = \beta = T^{-1/2}$, combined with parameter tuning, resolves the inherent tension between constraint satisfaction and optimization.

## Limitations & Future Work

1. When the mixing time is unknown, the condition $T \geq \tilde{\Theta}(\tau_{\text{mix}}^{2/\epsilon})$ may require an extremely long time horizon for slowly mixing problems.
2. $\epsilon_{\text{bias}}$ and $\epsilon_{\text{app}}$ are irreducible approximation residuals, though they are negligible for sufficiently expressive neural network parameterizations.
3. The ergodicity assumption excludes non-ergodic MDPs.
4. The critic employs linear function approximation and is not extended to neural networks.
5. No empirical validation is provided; this is a purely theoretical contribution.
6. Extending the results to multiple constraints or time-varying constraints remains an open direction.

## Related Work & Insights

- **Bai et al. (2024)**: Prev. SOTA for general parameterized CMDPs, achieving $\tilde{\mathcal{O}}(1/T^{1/5})$ convergence.
- **Ganesh et al. (2024)**: Accelerated actor-critic methods for unconstrained average reward MDPs.
- **Suttle et al. (2023) & Patel et al. (2024)**: Use of MLMC to eliminate the mixing time assumption in unconstrained average reward settings.
- **Wei et al. (2022)** (Triple-QA): Zero constraint violation in tabular settings but with $\tilde{\mathcal{O}}(1/T^{1/6})$ convergence.
- **Insight**: The combination of MLMC techniques with primal-dual frameworks may generalize to more complex constrained RL scenarios.

## Rating

- **Novelty**: ★★★★★ — First to match the theoretical lower bound for general parameterized CMDPs; a breakthrough result.
- **Experimental Thoroughness**: ★★☆☆☆ — Purely theoretical; no empirical validation.
- **Value**: ★★★☆☆ — Outstanding theoretical contribution, though practical deployment requires further engineering.
- **Writing Quality**: ★★★★☆ — Theoretically rigorous with consistent notation, though the content density is very high.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[NeurIPS 2025\] Automaton Constrained Q-Learning](automaton_constrained_q-learning.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)

<!-- RELATED:END -->
