---
title: >-
  [Paper Note] KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies
description: >-
  [ICML2025][Reinforcement Learning][Exploration strategy coordination] This paper proposes KEA, which actively coordinates different exploration strategies by introducing a dynamic switching mechanism between a standard agent and a novelty-augmented agent. This resolves the issues of redundant sampling and inefficient exploration caused by policy interaction when combining SAC with novelty-based exploration.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Exploration strategy coordination"
  - "sparse rewards"
  - "Soft Actor-Critic"
  - "novelty-based exploration"
  - "RND"
  - "NovelD"
date: 2026-05-08
content_hash: a42f59b99dad5bf2
---

# KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies

**Conference**: ICML2025  
**arXiv**: [2503.18234](https://arxiv.org/abs/2503.18234)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning  
**Keywords**: Exploration strategy coordination, sparse rewards, Soft Actor-Critic, novelty-based exploration, RND, NovelD

## TL;DR

This paper proposes KEA, which actively coordinates different exploration strategies by introducing a dynamic switching mechanism between a standard agent and a novelty-augmented agent. This resolves the issues of redundant sampling and inefficient exploration caused by policy interaction when combining SAC with novelty-based exploration.

## Background & Motivation

**Core Problem:** When SAC is combined with novelty-based exploration methods (such as RND) in sparse-reward environments, complex interactions between the two exploration strategies lead to low exploration efficiency.

Specifically, this interaction exhibits a cyclic pattern:

**Phase T1**: High intrinsic rewards drive the agent to repeatedly visit a specific region, concentrating the action distribution in a few directions.

**Phase T2**: After the intrinsic reward in that region decays, policy entropy increases, introducing more randomness and raising the probability of exploring new directions.

**Phase T3**: Upon discovering an unvisited region, the agent is again attracted by high intrinsic rewards and refocuses on the newly discovered region.

This "visit $\rightarrow$ decay $\rightarrow$ randomness $\rightarrow$ discovery $\rightarrow$ re-visit" cycle relies on natural policy transitions, resulting in:

- A large amount of redundant experience collection (repeatedly visiting known high-novelty regions)
- Delays in discovering new states
- A decline in overall learning efficiency

**Limitations of Prior Work:** Reward shaping may lead to behaviors that deviate from the true goal. Although curiosity/novelty-based methods (ICM, RND, NovelD) encourage exploration, they lack an explicit coordination mechanism for the interaction between different exploration strategies.

## Method

### Overall Architecture

KEA integrates two agents and coordinates them via a switching mechanism:

- **Novelty-augmented agent $\mathcal{A}^{\text{N}}$**: Uses SAC + intrinsic rewards (RND/NovelD) to find relatively novel states within explored regions.
- **Standard agent $\mathcal{A}^{\text{S}}$**: Pure SAC (extrinsic rewards only), maintaining high randomness to explore uncharted regions.
- **Switching mechanism $\psi$**: Dynamically selects which agent's policy to use based on state novelty.

### Novelty Exploration Policy

The Soft Bellman update target of SAC is modified to consider both extrinsic and intrinsic rewards:

$$y_Q = (\beta^{\text{ext}} r^{\text{ext}} + \beta^{\text{int}} r^{\text{int}}) + \gamma \left( \min_{\theta'_{1,2}} Q_{\theta'_i}(s', a') - \alpha \log \pi^{\text{SAC}}(\cdot|s') \right)$$

where the intrinsic reward is computed by RND:

$$r_t^{\text{int}} = \| \hat{f}(s_t; \theta) - f(s_t) \|^2$$

$f$ is a randomly initialized target network, and $\hat{f}$ is a predictor network trained via gradient descent.

### Design of the Standard Agent

The key design of $\mathcal{A}^{\text{S}}$ is **delayed learning** — before receiving any extrinsic reward, the loss weight is set to zero, freezing gradient updates. This keeps the policy's action variance high (close to the initial random policy), thereby providing complementary random exploration when switching to this agent. Normal training is resumed once extrinsic rewards are obtained.

The two agents share a unified replay buffer, leveraging off-policy characteristics to collect diverse data.

### Switching Mechanism

The switching is decided based on the intrinsic reward of the current state:

$$\pi(s_t) = \psi(r_t^{\text{int}}, \pi^{\text{N}}(s_t), \pi^{\text{S}}(s_t))$$

$$\psi = \begin{cases} \pi^{\text{S}}(s_t), & \text{if } r_t^{\text{int}} > \sigma \\ \pi^{\text{N}}(s_t), & \text{otherwise} \end{cases}$$

- **Intrinsic reward > threshold $\sigma$**: The state is highly novel (near the boundary); switch to $\mathcal{A}^{\text{S}}$ for random exploration to increase the probability of entering uncharted regions.
- **Intrinsic reward $\le$ threshold $\sigma$**: The state is within known regions; switch to $\mathcal{A}^{\text{N}}$ to use novelty guidance toward newer regions.

## Key Experimental Results

### 2D Navigation Task (Sparse Rewards, 300K Steps)

| Method | Average Return |
|------|---------|
| SAC | 0.0 ± 0.0 |
| RND-SAC | 0.235 ± 0.184 |
| **KEA-RND-SAC** | **0.403 ± 0.042** |
| NovelD-SAC | 0.607 ± 0.042 |
| **KEA-NovelD-SAC** | 0.604 ± 0.051 (24% faster convergence) |

KEA-RND-SAC achieves a **70%+** Gain compared to RND-SAC, with significantly lower variance.

### DeepSea Hard Exploration Benchmark (100K Episodes)

| Method | N=10 | N=20 | N=24 | N=30 |
|------|------|------|------|------|
| SAC | 0.98 | 0.00 | 0.00 | 0.00 |
| RND-SAC | 0.99 | 0.89 | 0.67 | 0.35 |
| SOFE-DQN | 0.97 | 0.70 | 0.65 | 0.42 |
| **KEA-RND-SAC** | **0.99** | **0.92** | **0.81** | **0.54** |

At the high difficulty of N=30, KEA reaches 0.54, outperforming all baselines.

### DeepMind Control Suite Sparse Rewards (500K Steps)

| Method | Walker Run | Cheetah Run | Reacher Hard |
|------|-----------|-------------|--------------|
| SAC | 0.0 | 0.0 | 715.17 |
| RND-SAC | 287.65 | 512.02 | 790.32 |
| **KEA-RND-SAC** | **629.74** | **773.76** | **874.61** |
| NovelD-SAC | 553.26 | 647.29 | 860.40 |
| **KEA-NovelD-SAC** | **706.47** | **734.67** | 837.12 |

KEA-RND-SAC achieves a **119%** Gain on Walker Run and a **51%** Gain on Cheetah Run.

### Sensitivity Analysis of Switching Threshold

| Threshold $\sigma$ | Return | $\mathcal{A}^{\text{S}}$ Usage Rate |
|---------|------|------|
| 0.50 | 0.358 | 24% |
| 1.00 | **0.407** | 14% |
| 1.50 | 0.334 | 8% |

All threshold configurations outperform the baseline RND-SAC (0.235), showing that the method is insensitive to this hyperparameter.

### Generalization to Off-Policy Methods

- **SAC/SQL** (stochastic policy coupled with novelty exploration) $\rightarrow$ KEA provides significant Gains.
- **DQN-P** (proportional Q-value sampling) $\rightarrow$ KEA yields improvements.
- **DQN** ($\epsilon$-greedy independent of Q-value) $\rightarrow$ KEA has no obvious effect (exploration strategies do not interact).

## Highlights & Insights

1. **In-depth problem analysis**: The cyclic interaction issue between "novelty exploration $\leftrightarrow$ random exploration" is clearly characterized and intuitively demonstrated with 2D visualizations.
2. **Simple and effective design**: Only requires an additional standard agent and a simple threshold switching mechanism, without relying on complex meta-learning or hierarchical architectures.
3. **Plug-and-play**: Can be integrated with any novelty-based method (such as RND, NovelD) and is compatible with various off-policy algorithms.
4. **Efficient shared buffer**: Both agents share the same replay buffer, fully utilizing off-policy characteristics without wasting sample data.
5. **Thorough generalization analysis**: The paper explicitly points out the applicability condition for KEA — the base exploration policy must be coupled with novelty-based exploration.

## Limitations & Future Work

1. **Only applicable to off-policy methods**: The standard agent requires experience sharing, which cannot be directly scaled to on-policy methods (such as PPO).
2. **Threshold $\sigma$ is a fixed hyperparameter**: Although insensitive to the threshold, adaptive thresholds (e.g., based on quantiles of the intrinsic reward distribution) might further improve performance.
3. **Only validated with RND/NovelD intrinsic rewards**: Whether it is applicable to prediction-error-based methods like ICM remains unverified.
4. **"Delayed learning" design of the standard agent**: When extrinsic rewards are extremely sparse, $\mathcal{A}^{\text{S}}$ may remain purely random for a long time, potentially affecting exploitation efficiency in later stages.
5. **Limited experimental scale**: Not validated in high-dimensional observation spaces (e.g., pixel inputs) or more complex environments (e.g., complex MuJoCo robotic manipulation tasks).
6. **Computational overhead**: Maintaining two independent agents increases parameter size and training time, which is not fully discussed in terms of efficiency trade-offs.

## Related Work & Insights

- **RND** (Burda et al., 2018): Intrinsic rewards based on random network distillation, a core component of KEA.
- **NovelD** (Zhang et al., 2021): Combines novelty difference with count-based rewards, another exploration method compatible with KEA.
- **NGU** (Badia et al., 2020): An exploration strategy combining episodic memory with life-long novelty.
- **DeRL** (Schäfer et al., 2021): Decoupled representation learning for exploration, used as an experimental baseline.
- **SOFE** (Castanyer et al., 2024): Another exploration enhancement method, used as a baseline on DeepSea.

## Rating

- Novelty: ⭐⭐⭐ — The problem definition is clear, but the core method (dual agents with threshold switching) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple environments and baselines are included, with generalization and sensitivity analyses, but high-dimensional scenarios are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Good problem visualization, clear writing, and complete structure.
- Value: ⭐⭐⭐ — Resolves practical pain points of SAC + novelty-based exploration, but its scope is limited to off-policy methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptively Coordinating with Novel Partners via Learned Latent Strategies](../../NeurIPS2025/reinforcement_learning/adaptively_coordinating_with_novel_partners_via_learned_latent_strategies.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[ICML 2025\] Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration](controlling_underestimation_bias_in_constrained_reinforcement_learning_for_safe_.md)
- [\[NeurIPS 2025\] Exploration via Feature Perturbation in Contextual Bandits](../../NeurIPS2025/reinforcement_learning/exploration_via_feature_perturbation_in_contextual_bandits.md)
- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)

</div>

<!-- RELATED:END -->
