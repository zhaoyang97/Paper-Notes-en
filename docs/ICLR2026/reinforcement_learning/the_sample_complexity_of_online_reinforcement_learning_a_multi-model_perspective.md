---
title: >-
  [Paper Note] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective
description: >-
  [ICLR 2026][Reinforcement Learning][Sample Complexity] This paper proposes an online reinforcement learning algorithm for nonlinear dynamical systems with continuous state-action spaces. By combining multi-model posterior sampling with certainty-equivalence control, the algorithm enables online learning of unknown systems and provides non-asymptotic policy regret guarantees that scale from finite model sets to parametric model families.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Sample Complexity
  - Online Reinforcement Learning
  - Multi-Model Adaptive Control
  - Policy Regret
  - Nonlinear Dynamical Systems
date: 2026-05-08
content_hash: 1fc25f2221264ed7
---

# The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective

**Conference**: ICLR 2026
**arXiv**: [2501.15910](https://arxiv.org/abs/2501.15910)
**Code**: None
**Area**: Reinforcement Learning / Online Control
**Keywords**: Sample Complexity, Online Reinforcement Learning, Multi-Model Adaptive Control, Policy Regret, Nonlinear Dynamical Systems

## TL;DR

This paper proposes an online reinforcement learning algorithm for nonlinear dynamical systems with continuous state-action spaces. By combining multi-model posterior sampling with certainty-equivalence control, the algorithm enables online learning of unknown systems and provides non-asymptotic policy regret guarantees that scale from finite model sets to parametric model families.

## Background & Motivation

Online reinforcement learning faces a fundamental dilemma: the decision-maker must balance **exploration** (acquiring information about system dynamics) and **exploitation** (optimizing performance). Classical work has focused primarily on linear dynamical systems (e.g., LQR), analyzing sample complexity via two-step strategies (system identification followed by control design). However, many real-world systems exhibit nonlinear dynamics (e.g., robotics, transportation), and online control analysis in continuous state-action spaces is substantially more challenging than in the discrete setting.

Existing approaches have the following limitations:

**Machine learning community** (e.g., Dean et al., 2018; Simchowitz & Foster, 2020) focuses on linear dynamical systems, relying on two-step learning strategies (alternating least-squares estimation and optimal control design) that do not generalize to nonlinear systems.

**Adaptive control community** (e.g., Anderson et al., 2000; Hespanha et al., 2001) addresses asymptotic stability and boundedness, but lacks non-asymptotic performance characterization.

**Recent online switching control work** (e.g., Li et al., 2023; Kim & Lavaei, 2024) handles nonlinear dynamics, but regret bounds scale exponentially in the number of unstabilizing controllers.

The paper's motivation is to establish a unified analytical framework covering multiple levels of model complexity — from finite model sets to function classes of infinite cardinality to parametric model families — while delivering concise and practical online learning algorithms.

## Method

### Overall Architecture

The paper considers a standard stochastic nonlinear dynamical system:

$$x_{k+1} = f(x_k, u_k) + n_k$$

where $f$ is the unknown dynamics and $n_k \sim \mathcal{N}(0, \sigma^2 I)$ is process noise. The decision-maker aims to minimize cumulative loss $\sum_{k=1}^N l(x_k, u_k)$.

The core idea of the algorithm is **posterior sampling + certainty-equivalence control + persistent excitation**:
- Maintain one-step prediction errors for each candidate model based on past trajectories
- Sample from candidate models via a softmax distribution
- Apply the optimal feedback policy corresponding to the selected model
- Inject Gaussian excitation noise into the control input to ensure model identification

### Key Designs

#### 1. **Setting S1: Finite Model Set**

Given $m$ candidate nonlinear models $\{f_1, \dots, f_m\}$, the algorithm maintains the normalized one-step prediction error for each model:

$$s_k^i = \sum_{j=1}^{k-1} \frac{|x_{j+1} - f_i(x_j, u_j)|^2}{1 + |(x_j, u_j)|^2 / b^2}$$

and samples model index $i_k$ with probability $p_k^i \propto \exp(-\eta s_k^i)$. The normalization factor $1 + |(x_j, u_j)|^2 / b^2$ ensures that $s_k^i$ remains bounded even when states and inputs grow large.

**Design Motivation**: From a Bayesian perspective, $\exp(-s_k^i)$ is precisely the posterior probability of model $f_i$ given past trajectories (due to Gaussian noise). The parameter $\eta$ implements softmax temperature control.

**Core Result (Theorem 2.1)**: Policy regret is $\mathcal{O}(\ln N + \ln m)$, with logarithmic dependence on both the time horizon and the number of models.

#### 2. **Setting S2: Function Class of Infinite Cardinality**

When the candidate model set $F$ is a bounded subset of a normed vector space (e.g., a bounded Lipschitz function space), the algorithm constructs an $\epsilon$-packing $F_\epsilon$ via greedy covering, discretizing the infinite set into a finite collection, and then applies the S1 analysis framework.

**Core Result (Theorem 2.2)**: Regret is $\mathcal{O}(N\epsilon^2 + \ln(m(\epsilon))/\epsilon^2)$, where $m(\epsilon)$ is the packing number. For bounded $L$-Lipschitz functions, the regret growth rate is approximately $T^{(d_x+d_u)/(d_x+d_u+2)} = o(T)$, establishing no-regret learning.

#### 3. **Setting S3: Parametric Models**

When $F = \{f_\theta \mid \theta \in \Omega \subset \mathbb{R}^p\}$ (e.g., neural network parameterization), the algorithm directly samples parameters $\theta_k$ from a continuous posterior distribution.

**Design Highlight**: For linear feature maps $f_\theta(x,u) = \phi(x,u)^\top \theta$, the posterior is Gaussian, with mean and covariance computable via recursive least squares at a per-step cost of only $\mathcal{O}(p^2)$.

**Core Result (Theorem 2.3)**: Regret is $\mathcal{O}(\sqrt{Np})$, consistent with known results for linear systems.

### Loss & Training

The algorithm requires no explicit training and operates entirely online. Key technical components include:

1. **Excitation signal design**: $\sigma_{u_k}^2 \propto 2/k + \ln(m)/k^2$, ensuring sufficient early-stage exploration that gradually diminishes over time
2. **Model switching interval**: Model index is updated every $M$ steps to ensure persistent excitation conditions are met
3. **Core analysis**: The cost function $V$ is used as a Lyapunov function; regret bounds are established by conditioning on $\Pr(i_k = i^*)$ and $\Pr(i_k \neq i^*)$, combining the model convergence rate $\Pr(i_k \neq i^*) \leq M^2/(k-M)^2$ with the martingale structure of additive noise

## Key Experimental Results

### Main Results

Validation is performed on a 20-dimensional state, 5-dimensional input linear time-invariant system composed of four 5-dimensional leaky integrators in series, with a 5-step delay from input to state.

| Setting | Algorithm | Steps to Near-Optimal Performance | Parameter Space Dimension |
|---------|-----------|----------------------------------|--------------------------|
| S1 (Finite models) | Algorithm 1 (100 candidates) | ~10 steps | 100 models |
| S3 (Parametric) | Algorithm 3 | ~60 steps | $d_x^2 + d_x d_u = 500$ |

### Ablation Study

| Configuration | Key Finding | Explanation |
|--------------|-------------|-------------|
| S1 vs S3 | S1 converges 6× faster | Finite model set is more efficient but requires prior knowledge |
| $\eta$ parameter | Theoretical selection suffices | $\eta \leq \min\{1/(4M\sigma^2), 1/(2ML^2b^2)\}$ |
| $\sigma_{u_k}$ decay | Ensures boundedness and convergence | Two-phase schedule: first guarantees identification, then performance |

### Key Findings

1. **Model-based vs. model-free**: In model-based methods, a single trajectory provides information about the accuracy of all candidate models, yielding $\mathcal{O}(\ln m)$ regret dependence on model count; model-free methods can only extract information about the current policy, resulting in at least $\mathcal{O}(m)$ regret.
2. **Separation principle**: The algorithm naturally realizes a separation principle for nonlinear systems — system identification and optimal control are performed separately.
3. **State boundedness**: Under the condition $l(x,u) \geq \bar{L}_l |x|^2/2$, $\mathbb{E}[|x_k|^2]$ is uniformly bounded and only finitely many steps of persistent excitation are required.
4. **Finite-time convergence**: The model index sequence $\{i_k\}$ converges almost surely in finite time to the true model.

## Highlights & Insights

1. **Unified framework**: All three settings (finite / infinite cardinality / parametric) are handled within a single analytical framework, with the latter two naturally reducing to extensions of the first, yielding a coherent and unified analysis.
2. **Practical applicability**: Particularly for S1 and the special case of S3 with linear feature maps, the algorithm requires only Gaussian sampling and recursive least squares, making implementation straightforward.
3. **Recovery of classical results**: The regret bound $\mathcal{O}(\sqrt{T} \cdot (d_x^2 + d_x d_u))$ for linear dynamical systems matches existing literature, validating the correctness of the framework.
4. **Deep theoretical contribution**: The paper reveals the fundamental distinction between the logarithmic dependence $\ln(m)$ in model-based RL and the polynomial dependence $m$ in model-free RL.

## Limitations & Future Work

1. **Persistent excitation assumption**: Assumption 3.2 requires satisfaction under any initial state and excitation variance, which may be difficult to verify for certain degenerate systems.
2. **Computational feasibility**: The argmin and greedyCover operations in S2 are generally computationally intractable; S2 is primarily of theoretical interest.
3. **Full state observability only**: Output feedback (partial observability) is not addressed.
4. **Cost function smoothness**: Assumption 3.1 requires precise quadratic upper and lower bounds on the cost function, which may be restrictive for non-standard costs.
5. **No $\mathcal{H}_\infty$ setting**: Only expected ($\mathcal{H}_2$) performance is considered; robust (worst-case) performance is not discussed.

## Related Work & Insights

- **Relationship to Thompson Sampling**: The algorithm is essentially a nonlinear generalization of Thompson Sampling — sampling from the model posterior and applying a certainty-equivalence policy.
- **Connection to online learning theory**: The $\mathcal{O}(\ln m)$ regret for finite model sets aligns with classical results in online learning and multi-armed bandits.
- **Practical application potential**: Due to its simplicity and natural incorporation of prior knowledge, the algorithm is well-suited for engineering domains such as intelligent transportation systems and automated supply chains.

## Rating

- Novelty: ⭐⭐⭐⭐ (Unified framework is innovative, though core techniques are relatively standard)
- Experimental Thoroughness: ⭐⭐⭐ (Validation limited to linear systems; genuine nonlinear experiments are absent)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure; proof intuition is presented excellently)
- Value: ⭐⭐⭐⭐ (Solid theoretical contribution that lays a foundation for nonlinear online control)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[ICLR 2026\] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)

<!-- RELATED:END -->
