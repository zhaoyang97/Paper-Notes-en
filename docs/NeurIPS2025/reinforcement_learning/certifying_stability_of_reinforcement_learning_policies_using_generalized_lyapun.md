---
title: >-
  [Paper Note] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions
description: >-
  [NEURIPS2025][Reinforcement Learning][Lyapunov stability] This paper proposes a Generalized Lyapunov Function framework that combines RL value functions with neural network residual terms, replacing the classical strict per-step descent requirement with a multi-step weighted descent condition to certify the stability of RL policies.
tags:
  - NEURIPS2025
  - Reinforcement Learning
  - Lyapunov stability
  - reinforcement-learning
  - stability certification
  - value function
  - region of attraction
date: 2026-05-08
content_hash: 0ba7ab8b25e1b9cc
---

# Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions

**Conference**: NEURIPS2025
**arXiv**: [2505.10947](https://arxiv.org/abs/2505.10947)
**Code**: [GitHub](https://github.com/ExistentialRobotics/Generalized_Policy_Stability)
**Area**: Reinforcement Learning
**Keywords**: Lyapunov stability, reinforcement-learning, stability certification, value function, region of attraction

## TL;DR

This paper proposes a Generalized Lyapunov Function framework that combines RL value functions with neural network residual terms, replacing the classical strict per-step descent requirement with a multi-step weighted descent condition to certify the stability of RL policies.

## Background & Motivation

RL policies have demonstrated strong performance in nonlinear control tasks, yet they lack formal stability guarantees for the closed-loop system. Classical Lyapunov methods require the candidate function to strictly decrease at every step, a condition that learned policies often fail to satisfy. RL value functions naturally encode long-term cumulative cost and are natural Lyapunov candidates; however, due to the discount factor $\gamma \in (0,1)$, discounted value functions do not directly satisfy the Lyapunov descent condition.

Prior work (e.g., Romain 2017) attempted to construct Lyapunov functions by augmenting value functions with quadratic residual terms, but the resulting conditions become overly conservative when $\gamma$ is small. For instance, on the scalar system $x_{k+1}=2x_k+u_k$, the true stability threshold is $\gamma > 1/3$, whereas classical LMI-based methods can only certify $\gamma > 0.8090$—a substantial gap.

## Core Problem

1. How can RL value functions be leveraged to construct effective stability certificates?
2. How can the conservatism of classical Lyapunov conditions be reduced so that a broader class of reasonable RL policies can be certified?
3. How can stability certification be extended to nonlinear systems and joint policy–certificate synthesis settings?

## Method

### 1. Definition of Generalized Lyapunov Functions

The classical strict per-step descent condition is relaxed to allow the Lyapunov function to temporarily increase within a single step, provided it decreases on a weighted average over $M$ steps:

$$\frac{1}{M}\sum_{i=1}^{M}\sigma_i(\mathbf{x})V(\mathbf{x}_i) - V(\mathbf{x}) < 0$$

where $\sigma_i(\mathbf{x}) \geq 0$ are state-dependent non-negative weights satisfying $\frac{1}{M}\sum_{i=1}^M \sigma_i(\mathbf{x}) \geq 1$. The paper proves that under this relaxed condition, the origin remains asymptotically stable (Theorem 4.2).

### 2. Theoretical Analysis for Linear Systems (LQR)

For the discounted LQR problem, the optimal value function $J_\gamma^*(\mathbf{x})=\mathbf{x}^\top \mathbf{P}_\gamma \mathbf{x}$ is augmented as:

$$V(\mathbf{x}) = J_\gamma^*(\mathbf{x}) + \frac{1}{\varpi}\mathbf{x}^\top \mathbf{S}_0 \mathbf{x}$$

Stability can be certified by solving a set of multi-step LMI conditions (Theorem 4.4). The multi-step formulation provides additional degrees of freedom in the weights $\sigma_i$, significantly lowering the certifiable lower bound on $\gamma$. On the scalar example, the lower bound improves from 0.809 to 0.623 at $M=2$, approaching the true threshold $\gamma^*=1/3$ as $M$ increases.

### 3. Post-hoc Certification of RL Policies for Nonlinear Systems

For a pre-trained RL policy $\boldsymbol{\pi}_{\text{RL}}$, a generalized Lyapunov candidate function is constructed as:

$$V(\mathbf{x};\boldsymbol{\theta}_1) = |J_\gamma^{\boldsymbol{\pi}_{\text{RL}}}(\mathbf{x}) - J_\gamma^{\boldsymbol{\pi}_{\text{RL}}}(\mathbf{0})| + |\varphi(\mathbf{x};\boldsymbol{\theta}_1) - \varphi(\mathbf{0};\boldsymbol{\theta}_1)| + \beta\|\mathbf{x}\|^2$$

where $\varphi$ is a neural network residual term and $\beta\|\mathbf{x}\|^2$ ensures strict positive definiteness. A step-weight network $\sigma(\mathbf{x};\boldsymbol{\theta}_2)$ is introduced, with a softmax $\times M$ output layer to ensure the weights sum to $M$. Parameters $\boldsymbol{\theta}_1$ and $\boldsymbol{\theta}_2$ are jointly trained by minimizing ReLU losses on violations of the multi-step descent condition.

### 4. Joint Policy–Certificate Synthesis

The framework is extended to simultaneously learn a neural controller $\boldsymbol{\pi}(\mathbf{x};\boldsymbol{\phi})$ and a Lyapunov certificate $V(\mathbf{x};\boldsymbol{\theta}_1)$, with the objective of maximizing the volume of the certified region of attraction (ROA). Training employs a stability loss, a region loss, and L1 regularization, with PGD-based falsification for counterexample mining. After training, the $\alpha$-$\beta$-CROWN verifier is used for formal verification.

A key theoretical result (Theorem 6.2) shows that even when the generalized condition does not guarantee forward invariance of the sublevel set $\mathcal{S}$, $\mathcal{S}$ remains a valid inner approximation of the ROA and the origin is asymptotically stable.

## Key Experimental Results

**Post-hoc Certification (Fixed Policy):**

| Environment | RL Method | M | Test Points | Descent Condition Satisfaction |
|---|---|---|---|---|
| Inverted Pendulum | PPO, SAC, TD-MPC | 15 | 10,000 | 100% |
| Cartpole | SAC, TD-MPC | 20 | 1,000,000 | 100% |

**Certified ROA Volume under Joint Synthesis:**

| System | M=1 | M=2 | M=3 |
|---|---|---|---|
| Inverted Pendulum | 42.9±1.2 | 76.7±1.3 | 89.2±1.2 |
| Path Tracking | 21.8±0.6 | 23.6±0.5 | 23.9±0.5 |
| 2D Quadrotor | 103.5±1.8 | 109.1±2.0 | 113.7±2.0 |

Increasing $M$ consistently enlarges the certified region, though verification time grows accordingly (e.g., from 11.7s to 39.2s for the Inverted Pendulum).

**Step-Weight Distribution Analysis:** The learned weights concentrate toward the end of the horizon (30–38% in the 80–100% interval), indicating that the network learns a strategy of tolerating initial non-monotone transients while relying on monotone descent in later steps.

## Highlights & Insights

1. **Theoretical elegance**: The framework proceeds from exact analysis of LQR to derive intuition, then generalizes to nonlinear systems, yielding a clear logical chain.
2. **Practical applicability**: Post-hoc certification can be applied directly to pre-trained RL policies (PPO/SAC/TD-MPC) without retraining.
3. **Reduced conservatism**: The multi-step weighted descent condition significantly relaxes classical Lyapunov requirements; in the LQR example, the certifiable threshold improves from 0.809 to near the true value of 0.333.
4. **Larger ROA via joint synthesis**: On the Inverted Pendulum, the certified ROA volume at $M=3$ is more than twice that at $M=1$.
5. **Open-source implementation**: Complete code is provided for full reproducibility.

## Limitations & Future Work

1. **Choice of $M$**: The horizon length $M$ is fixed prior to training; no principled method exists for automatically determining the optimal $M$ for a given system.
2. **High-dimensional systems untested**: Experiments are limited to low-dimensional systems (up to 6-dimensional state spaces); high-dimensional settings such as humanoid locomotion or dexterous manipulation remain unexplored.
3. **Fixed weights in joint synthesis**: Due to limitations of formal verification tools, $\sigma_i$ cannot be parameterized by a neural network in the joint synthesis phase and must be selected as fixed values via grid search.
4. **Verification time scales with $M$**: At $M=3$, verification for the 2D Quadrotor exceeds 5,600 seconds, making scalability a bottleneck.
5. **Deterministic system assumption**: The theoretical analysis is restricted to deterministic systems; although experiments involve stochastic environments, formal guarantees are absent in that setting.

## Related Work & Insights

| Method | Characteristics | Advantage of This Work |
|---|---|---|
| Classical Lyapunov (Chang 2019, Wu 2023) | Requires strict per-step descent | Multi-step condition is easier to satisfy; certifies more policies |
| Yang 2024 (Lyapunov-stable) | Joint synthesis + $\alpha$,$\beta$-CROWN verification | This work replaces the condition with the generalized variant, achieving larger ROA |
| Berkenkamp 2017 | GP-based safe RL | Requires model priors; this work is model-free |
| $k$-Inductive methods | Descent required only at the last step | Weighted averaging here is more flexible with adaptive weights |

The idea of augmenting value functions as Lyapunov candidates is broadly applicable and can extend to other learning-based control settings requiring stability guarantees. The multi-step relaxation is analogous to the success of $k$-step returns in RL—longer horizons provide more accurate signals. Future work may explore **partial state stability certification** (certifying only task-relevant state components), which is particularly important for high-dimensional robotic systems. Integration with robust Lyapunov analysis and input-to-state stability is also a natural extension.

## Rating

- Novelty: ⭐⭐⭐⭐ — Generalized Lyapunov conditions have precedent, but their combination with RL value function augmentation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Both linear/nonlinear and post-hoc/joint synthesis settings are evaluated, though high-dimensional experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative arc from LQR intuition to nonlinear generalization is exceptionally clear.
- Value: ⭐⭐⭐⭐ — Bridges an important gap between RL and control-theoretic stability analysis.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies](certifying_concavity_and_monotonicity_in_games_via_sum-of-squares_hierarchies.md)
- [\[NeurIPS 2025\] Actor-Free Continuous Control via Structurally Maximizable Q-Functions](actorfree_continuous_control_via_structurally_maximizable_qf.md)
- [\[NeurIPS 2025\] Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization](learning_intractable_multimodal_policies_with_reparameterization_and_diversity_r.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)

<!-- RELATED:END -->
