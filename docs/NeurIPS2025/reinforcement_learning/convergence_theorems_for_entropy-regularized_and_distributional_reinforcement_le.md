---
title: >-
  [Paper Note] Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][entropy regularization] This paper proposes the **temperature decoupling gambit**, proving that in entropy-regularized reinforcement learning, by decoupling the evaluation temperature from the behavioral temperature, both the policy and the return distribution converge—as the temperature tends to zero—to an interpretable, diversity-preserving optimal policy.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - entropy regularization
  - distributional reinforcement learning
  - convergence
  - temperature decoupling
  - optimal policy
date: 2026-05-08
content_hash: b8eecbe4b2ff70d0
---

# Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.08526](https://arxiv.org/abs/2510.08526)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: entropy regularization, distributional reinforcement learning, convergence, temperature decoupling, optimal policy
**arXiv**: [2510.08526](https://arxiv.org/abs/2510.08526)
**Code**: None
**Area**: Reinforcement Learning

## TL;DR

This paper proposes the **temperature decoupling gambit**, proving that in entropy-regularized reinforcement learning, by decoupling the evaluation temperature from the behavioral temperature, both the policy and the return distribution converge—as the temperature tends to zero—to an interpretable, diversity-preserving optimal policy.

## Background & Motivation

**State of the Field**: Standard RL admits multiple optimal policies. Policy optimization methods can converge to some optimal policy, but which one is learned cannot be controlled—different optimal policies may visit different states, execute different actions, and yield different return distributions.

**Limitations of Prior Work**: Entropy-regularized RL (ERL) induces uniqueness by penalizing policies with a KL divergence term, yielding a unique optimal policy $\pi^{\tau,\star}$ for each positive temperature $\tau$. However, when $\tau \to 0$ (to recover RL optimality), the convergence of the policy in non-tabular (continuous) MDPs remains unknown.

**Root Cause**: ERL yields a unique policy for each $\tau > 0$, yet that policy is suboptimal for RL; when $\tau \to 0$ to recover RL optimality, one is again confronted with the indeterminacy of optimal policies.

**Paper Goals**: (1) Provide a scheme that guarantees policy convergence in the $\tau \to 0$ limit; (2) characterize the limiting policy; (3) establish the first convergent algorithm for estimating the optimal return distribution in distributional RL.

**Starting Point**: Inspired by the chess concept of a gambit—accepting a short-term sacrifice of $\tau$-ERL optimality in exchange for long-term convergence guarantees.

**Core Idea**: Construct a Boltzmann–Gibbs policy using a behavioral temperature $\tau$ strictly larger than the evaluation temperature $\sigma$ (requiring $\sigma/\tau \to 0$), guaranteeing convergence to the "optimality-filtered reference policy."

## Method

### Overall Architecture

1. Prove that the optimal Q-function of $\tau$-ERL satisfies $q_\tau^\star \to q_{\text{ref}}^\star$ (the reference optimal value function), while the policy need not converge.
2. Introduce the temperature-decoupled policy $\pi^{\tau,\sigma} := \mathcal{G}_\tau q_\sigma^\star$ (the BG policy under temperature $\tau$ evaluated using the $\sigma$-optimal Q-function).
3. Prove $\pi^{\tau,\sigma} \to \pi^{\text{ref},\star}$ (the optimality-filtered reference policy).
4. Extend the framework to distributional RL and establish a convergent return distribution estimator.

### Key Design 1: Bellman Reference Optimal Operator

- **Function**: Defines a new Bellman operator whose fixed point captures the optimal value actually attainable by ERL as $\tau \to 0$.
- **Mechanism**:
$$(\mathcal{B}_{\text{ref}}^\star q)(x,a) := r(x,a) + \gamma \int \text{ess}\sup_{\pi_x^{\text{ref}}} q(x', \cdot)\, dP_{x,a}(x')$$
  The standard $\sup$ is replaced by the $\text{ess}\sup$, which takes the maximum only over the support of the reference policy.
- **Key Result (Theorem 3)**: $q_\tau^\star \to q_{\text{ref}}^\star$ monotonically as $\tau \to 0$.
- **Design Motivation**: When the set of optimal actions has measure zero (as in continuous action spaces), $q_{\text{ref}}^\star < q^\star$, and any parameterized policy (Gaussian, diffusion, etc.) will almost surely fail to sample an optimal action. Hence $q_{\text{ref}}^\star$ represents the actual attainable ceiling.

### Key Design 2: Temperature Decoupling Gambit (Definition 3.4)

- **Function**: Constructs a policy with convergence guarantees as $\tau \to 0$.
- **Mechanism**: Given temperature $\tau > 0$, choose $\sigma = \sigma(\tau)$ (e.g., $\sigma = \tau^2$) and set $\pi^{\tau,\sigma} := \mathcal{G}_\tau q_\sigma^\star$, requiring $\sigma/\tau \to 0$ as $\tau \to 0$.
- **Key Inequality**:
$$\lim_{\tau \to 0} \sup_x \|(\mathcal{G}_\tau q_\sigma^\star)_x - (\mathcal{G}_\tau q_{\text{ref}}^\star)_x\|_{\text{TV}} \lesssim -\lim_{\tau \to 0} \frac{\sigma}{\tau} \log p_{\text{ref}}$$
  The condition $\sigma/\tau \to 0$ ensures the right-hand side vanishes.
- **Design Motivation**: In standard ERL, the log-probability in the BG policy is amplified by $\tau^{-1}$, so a Q-estimation error of order $O(\tau)$ does not vanish in policy space. Decoupling uses $\tau$ (slower to zero) as a smoothing parameter and $\sigma$ (faster to zero) as a Q-estimation precision parameter.

### Key Design 3: Optimality-Filtered Reference Policy (Definition 3.5)

- **Function**: Characterizes the limiting policy to which the temperature-decoupled policy converges.
- **Mechanism**:
$$\pi_x^{\text{ref},\star} \propto \pi_x^{\text{ref}} \odot \chi_{\mathsf{N}^\star_{\text{ref}}(x)}, \quad \mathsf{N}^\star_{\text{ref}}(x) := \{a : q^\star(x,a) = \text{ess}\sup_{\pi_x^{\text{ref}}} q^\star(x,\cdot)\}$$
  That is, the reference policy restricted and renormalized to the set of optimal actions.
- **Key Property**: When $\pi^{\text{ref}}$ is a uniform policy, $\pi^{\text{ref},\star}$ distributes uniformly over all optimal actions—the most diverse optimal policy.
- **Distinction from Standard ERL Limit**: Even in tabular MDPs where $\pi^{\tau,\star}$ converges (Theorem 2.2), the two limits differ. The standard limit selects an optimal policy by minimizing the KL divergence of the long-run occupancy measure, favoring average diversity across states; the decoupled limit $\pi^{\text{ref},\star}$ maximizes per-state action diversity.

### Key Design 4: Distributional ERL (DERL)

- **Soft Distributional Bellman Evaluation Operator** (Definition 4.1):
$$(\mathcal{T}_\tau^\pi \bar{\zeta})_{x,a} := (\mathtt{b}_{r(x,a),\gamma} \circ \mathtt{proj}^{\mathbb{R}} - \gamma\tau \mathtt{kl}[\pi] \circ \mathtt{proj}^{\mathsf{X}})_\# (\bar{\zeta}_{\_,\_} \otimes \check{P}_{x,a}^\pi)$$
  This is a $\gamma$-contraction (Theorem 4.2) with a unique fixed point $\bar{\zeta}^{\pi,\tau}$.
- **Soft Distributional Optimal Operator** (Definition 4.2): Constructs a BG policy from the mean of the current distribution and evaluates it; the commutativity lemma gives $\mathcal{Q}\mathcal{T}_\tau^\star = \mathcal{B}_\tau^\star \mathcal{Q}$.
- **Convergence Rate (Theorem 4.2)**: $\bar{d}_p(\bar{\zeta}^n, \bar{\zeta}^{\tau,\star}) \leq C_{p,\tau,\gamma} n \gamma^{n/p}$—iterative convergence, in sharp contrast to the non-convergent iterations of the standard distributional $\mathcal{T}^\star$.

### Loss & Training

This is a theoretical contribution; no explicit loss function is defined. Algorithms are based on dynamic programming iterations.

## Key Experimental Results

### Numerical Demonstration 1: Three-State MDP (Figure 3.1)

- Three states $\{x_0, x_1, x_2\}$, two actions (blue, green), $\gamma = 0.9$, $\pi^{\text{ref}} = \mathcal{U}(\mathsf{A})$.
- **Standard ERL** $\hat{\pi}^{\tau,\star}$: converges at $x_0$ to $\delta_{a_1}$ (degenerates to a deterministic policy).
- **Temperature Decoupling** $\hat{\pi}^{\tau,\sigma}$: converges at $x_0$ to $\mathcal{U}(\{a_1, a_2\})$ (preserves diversity).
- Both behave identically at $x_1, x_2$; the difference arises only at $x_0$—confirming the theoretical prediction.

### Numerical Demonstration 2: Return Distribution Convergence (Figures 4.3, 4.4)

- Illustrative MDP: starting from $x_1$, the blue action yields a deterministic return $2\gamma/(1-\gamma)$; the green action yields a Bernoulli return $4\gamma/(1-\gamma) \cdot \text{Bernoulli}(1/2)$.
- **Temperature-decoupled estimator** $\hat{\eta}^{\tau,\sigma}$: converges as $\tau \to 0$ to the return distribution of $\pi^{\text{ref},\star}$ (bimodal).
- **Standard ERL estimator** $\hat{\eta}^{\tau,\star}$: also converges, but to a different return distribution.
- **Stability of the distributional optimal operator**: iterations of $\mathcal{T}_\tau^\star$ converge stably (Figure 4.1, bottom row), whereas iterations of the standard $\mathcal{T}^\star$ diverge (Figure 4.1, top row).
- **Effect of numerical precision**: both methods converge under 64-bit precision; under 32-bit precision, the temperature-decoupled approach is more stable.

### Key Findings

- The temperature-decoupled policy and the standard ERL policy converge to **distinct** optimal policies, even in the same tabular MDP.
- The former preserves per-state action diversity ("no discrimination among optimal actions"), while the latter optimizes the KL divergence of the long-run occupancy measure.
- The non-convergence of the distributional Bellman optimal operator $\mathcal{T}^\star$ is perfectly resolved by entropy regularization.

## Highlights & Insights

1. **Elegant theoretical contribution**: The temperature decoupling gambit is a refined construction—sacrificing short-term ERL optimality for long-term RL convergence, precisely as in a chess gambit.
2. **Interpretability of $\pi^{\text{ref},\star}$**: The limiting policy has a clean mathematical definition (the reference policy projected onto the optimal action set) and maximizes per-state diversity under a uniform reference.
3. **Unification of three areas**: The framework connects policy optimization (ERL), distributional RL (DRL), and information theory (KL regularization within a single convergence-theoretic treatment.
4. **First convergent optimal return distribution estimator**: Theorem 4.2 is a landmark result in DRL—prior to this work, iterations of the distributional Bellman optimal operator were known to be non-convergent.
5. **TV-distance bound for BG policies (Theorem 3.3)**: An independently valuable technical tool.

## Limitations & Future Work

1. **Qualitative convergence**: Policy convergence is established in the sense of TV/weak convergence; explicit finite-temperature approximation error bounds are absent, leaving open how small $\tau$ must be in practice.
2. **Verifiability of Assumption 3.2**: In continuous action spaces, ensuring that the reference policy places sufficient mass near optimal actions is non-trivial.
3. **Lack of large-scale experiments**: Numerical demonstrations are confined to toy MDPs; practical performance on high-dimensional benchmarks (e.g., MuJoCo) is unverified.
4. **Computational cost**: Temperature decoupling requires estimating $q_\sigma^\star$ for potentially very small $\sigma$, which may demand high-precision arithmetic in practice.
5. **Gap to practical algorithms**: How to embed temperature decoupling into practical ERL algorithms such as SAC remains an important open problem.

## Related Work & Insights

- **Relation to SAC (Haarnoja et al., 2018)**: SAC employs MaxEnt RL with a fixed temperature; the temperature-decoupled policy offers a new theoretical perspective on its adaptive temperature schemes.
- **Complement to distributional RL (Bellemare et al., 2017)**: The non-convergence of return distributions under the control setting is a known open problem in standard DRL; this paper provides the first solution via ERL.
- **Implications for safe RL**: Convergent return distribution estimation is critical for risk-sensitive applications—this work enables, for the first time, accurate estimation of the return distribution of a specific, well-defined optimal policy.
- **Implications for exploration**: $\pi^{\text{ref},\star}$ maintains a uniform distribution over all optimal actions, which is naturally conducive to exploratory behavior.
- **Ziegler et al. (2019) / KL-constrained RLHF**: The temperature decoupling idea may inform temperature annealing schedules for the KL penalty in RLHF.

## Rating

⭐⭐⭐⭐⭐ (5/5)

A work of exceptional theoretical depth that resolves long-standing open problems in both ERL and DRL. The temperature decoupling gambit is simultaneously simple and profound; the characterization of $\pi^{\text{ref},\star}$ provides a precise mathematical definition of the "most diverse optimal policy." The primary limitations are the absence of large-scale empirical validation and quantitative convergence rates.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](../../ICLR2026/reinforcement_learning/entropy-preserving_reinforcement_learning.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)

<!-- RELATED:END -->
