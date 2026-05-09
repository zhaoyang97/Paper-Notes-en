---
title: >-
  [Paper Note] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics
description: >-
  [NeurIPS 2025][Reinforcement Learning][Mean Field Games] This paper proposes the DEDA-FP algorithm, which for the first time simultaneously learns Nash equilibrium policies and population distributions in non-stationary mean field games (MFGs) with continuous state/action spaces. By combining deep RL for best response computation, supervised learning for mean policy representation, and conditional Normalizing Flow for modeling time-varying population distributions, DEDA-FP achieves over 10× sampling efficiency compared to existing methods.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Mean Field Games
  - Deep Reinforcement Learning
  - Fictitious Play
  - Normalizing Flow
  - Nash Equilibrium
date: 2026-05-08
content_hash: 015559bc4cee358f
---

# Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2510.22158](https://arxiv.org/abs/2510.22158)
**Code**: Unavailable
**Area**: Reinforcement Learning
**Keywords**: Mean Field Games, Deep Reinforcement Learning, Fictitious Play, Normalizing Flow, Nash Equilibrium

## TL;DR

This paper proposes the DEDA-FP algorithm, which for the first time simultaneously learns Nash equilibrium policies and population distributions in non-stationary mean field games (MFGs) with continuous state/action spaces. By combining deep RL for best response computation, supervised learning for mean policy representation, and conditional Normalizing Flow for modeling time-varying population distributions, DEDA-FP achieves over 10× sampling efficiency compared to existing methods.

## Background & Motivation

### State of the Field

Mean field games (MFGs) provide a powerful framework for modeling large-scale multi-agent systems: by taking the limit as the number of agents tends to infinity, the multi-agent problem reduces to an interaction between a representative agent and a population distribution. MFGs have broad applications in economics, finance, transportation, and communications, where state-action spaces are naturally continuous and population distributions typically evolve over time (non-stationary).

### Limitations of Prior Work

Existing MFG solution methods suffer from three major limitations:

**Discreteness constraints**: Most RL-based methods (Guo et al. 2019; Elie et al. 2020; Laurière et al. 2022a) are restricted to finite state/action spaces.

**Stationarity assumptions**: Most approaches (Perrin et al. 2021; Angiuli et al. 2023) assume that the population distribution does not evolve over time.

**Density inaccessibility**: Existing methods cannot directly evaluate the probability density $\mu(x)$ of the population distribution—they can only draw samples—making it impossible to precisely solve MFGs with local density dependence (e.g., congestion models).

No existing RL algorithm can learn a complete solution (Nash equilibrium policy + population distribution) for non-stationary MFGs in continuous spaces.

### Starting Point

The core approach constructs a three-component method: (1) deep RL (SAC/PPO) for best response computation; (2) supervised learning to approximate the mean policy and obtain a Nash equilibrium policy; (3) a time-conditioned Normalizing Flow to model the non-stationary population distribution, enabling both sampling and exact density evaluation. The entire framework is embedded within an iterative Fictitious Play (FP) scheme.

## Method

### Overall Architecture

DEDA-FP is built on the iterative framework of the Fictitious Play (FP) algorithm. Each iteration consists of three steps:
1. Compute the best response to the current mean policy using deep RL.
2. Update the neural network representation of the mean policy via supervised learning.
3. Train a conditional Normalizing Flow to learn the time-varying population distribution induced by the mean policy.

### Key Designs

1. **Best Response Computation (DRL)**: SAC or PPO is employed to solve $\pi_k^* = \arg\max_\pi J_{\mu_0}^N(\pi, \bar{G}_{k-1})$, where $\bar{G}_{k-1}$ is the population distribution model learned in the previous iteration. During each rollout, $N-1$ virtual agents follow the current mean policy $\bar{\pi}_{k-1}$ to simulate population behavior. This is a standard DRL procedure, except that the population distribution in the environment is drawn from the learned generative model rather than empirical samples.

2. **Nash Equilibrium Policy Learning (Supervised Learning)**: A replay buffer $\mathcal{M}_{SL}$ stores $(t, s, a)$ tuples from all historical best responses. A policy network $\bar{\pi}^{\bar{\theta}}$ is trained to minimize the negative log-likelihood:

$$\mathcal{L}_{\text{NLL}}(\bar{\theta}) = \mathbb{E}_{(t,s,a) \sim \mathcal{M}_{SL}}\left[-\log \bar{\pi}^{\bar{\theta}}(a|t,s)\right] = -\frac{1}{M}\sum_{i=1}^{M}\log \mathcal{N}(a_i; \mu_{\bar{\theta}}(s_i, t_i), \sigma_{\bar{\theta}}(s_i, t_i))$$

The policy network outputs the mean and standard deviation of a conditional Gaussian distribution, conditioned on both time $t$ and state $s$. This circumvents the difficulty of directly averaging neural network policies.

3. **Conditional Normalizing Flow (CNF) for Population Distribution Modeling**: An autoregressive variant of Neural Spline Flow is adopted, conditioned on time $t$. Given a sample $\mathbf{x}$ and time $t$, the density is computed via the change-of-variables formula:

$$p(\mathbf{x}|t) = p_0(f^{-1}(\mathbf{x}, t))\left|\det\left(\frac{\partial f^{-1}(\mathbf{x}, t)}{\partial \mathbf{x}}\right)\right|$$

The training objective is maximum likelihood estimation (equivalently, minimizing NLL). The key advantage of CNF is that it supports **both sampling and density evaluation**, which is essential for MFGs with local density dependence.

### Convergence Analysis

Theorem 1 establishes a convergence bound on exploitability, which depends on three sources of error:
- **Best response error** $\epsilon_{br}^k$: suboptimality of the DRL policy
- **Mean policy error** $\epsilon_{sl}^k$: fitting error of the supervised learning component
- **Distribution error** $\epsilon_{cnf}^k$: density estimation error of the Normalizing Flow

$$e_k^{\text{true}} < C_0 e_0^{\text{cnf}} + \frac{1}{k}\sum_{i=1}^{k-1}\left[(i+1)\epsilon_{br}^{i+1} + C_1(\epsilon_{sl}^{i+1} + \epsilon_{cnf}^{i+1}) + \frac{C_2}{i}\right]$$

## Key Experimental Results

### Beach Bar Problem (Local Density Dependence)

| Algorithm | Exploitability Trend | Distribution Quality | Direct Access to $\mu(x)$ |
|-----------|---------------------|----------------------|--------------------------|
| Algo 1 (Vanilla FP) | Converges | Coarse | Requires Gaussian convolution approximation |
| Algo 2 (+Policy Learning) | Converges | Moderate | Requires Gaussian convolution approximation |
| DEDA-FP | Converges, no performance degradation | Smooth and concentrated | **Exact density, no approximation** |

### Linear-Quadratic Model

| Algorithm | Exploitability | Policy Linearity | Distribution Concentration |
|-----------|---------------|-----------------|---------------------------|
| Algo 1 | ~0 (fast convergence) | Policy not learned | Basic |
| Algo 2 | ~0 (fast convergence) | Linear, matches theory | Moderate |
| DEDA-FP | ~0 (fast convergence) | Linear, matches theory | **Optimal** |

### 4-Rooms Exploration (2D, Entropy Maximization)

| Metric | Algo 1 | Algo 2 | DEDA-FP |
|--------|--------|--------|---------|
| Exploitability | Converges | Converges | Converges, comparable |
| Time to sample 5,000 trajectories | Baseline | ~Baseline | **>10× faster** |
| Distribution representation quality | Poor (sparse) | Moderate | **Precise (8,000 samples)** |
| Direct use of density reward | No | No | **Yes** |

### Key Findings

- DEDA-FP achieves exploitability convergence comparable to baseline methods while providing **substantially higher-quality** population distribution representations.
- CNF-based sampling is **more than 10× faster** than conventional trajectory simulation.
- For problems with local density dependence (Beach Bar, 4-Rooms), DEDA-FP directly utilizes the density $\mu(x)$ **without approximation**.
- The learned policies exhibit the expected linear structure in the LQ model, validating the correctness of the approach.

## Highlights & Insights

- **Complete MFG solution**: The first DRL method to simultaneously learn Nash equilibrium policies and population distributions, filling the gap for non-stationary MFGs in continuous spaces.
- **Elegant choice of CNF**: Unlike GANs or diffusion models, which only support sampling, Normalizing Flows provide both sampling and density evaluation, perfectly matching the requirements of MFG problems.
- **Modular design**: The three components (DRL, supervised learning, CNF) serve distinct roles and can be improved independently.
- **Theoretical guarantees**: The error propagation analysis clearly characterizes the contribution of each of the three error sources to the final exploitability.

## Limitations & Future Work

- The theoretical analysis remains incomplete, particularly with respect to fine-grained analysis of deep network training.
- The framework addresses only standard MFGs and has not been extended to multi-population, graphon, or common-noise settings.
- Approximate exploitability evaluation relies on environment approximations and is therefore not fully exact.
- The expressive capacity and training stability of Normalizing Flows in high-dimensional state spaces remain to be validated.
- Practical deployment considerations in real-world applications are not addressed.

## Related Work & Insights

- Compared to Perrin et al. (2021): their work handles continuous spaces but is limited to stationary MFGs and does not learn Nash equilibrium policies.
- Compared to Laurière et al. (2022a): their method learns Nash equilibrium policies but is restricted to discrete spaces.
- Compared to Zaman et al. (2020): their approach handles non-stationary continuous spaces but is limited to LQ models.
- Heinrich & Silver (2016)'s Neural Fictitious Self-Play inspired the use of supervised learning to approximate the mean policy.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified treatment of the three core challenges in non-stationary MFGs with continuous spaces.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three experiments of increasing complexity plus an additional financial application; high-dimensional experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly stated; the method is presented in a progressive manner (Algo 1 → 2 → 3).
- Value: ⭐⭐⭐⭐ Opens the door to solving complex MFGs with DRL, though broader generalizability requires further validation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Non-convex Entropic Mean-Field Optimization via Best Response Flow](non-convex_entropic_mean-field_optimization_via_best_response_flow.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)

<!-- RELATED:END -->
