---
title: >-
  [Paper Note] Human-Inspired Multi-Level Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][rating-based RL] This paper proposes RbRL-KL, which augments rating-based RL (RbRL) with a KL divergence-driven policy loss term. By leveraging failure experiences across different rating levels with varying weights to repel the current policy, RbRL-KL outperforms standard RbRL across 6 DeepMind Control environments.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - rating-based RL
  - KL divergence
  - human feedback
  - multi-level learning
  - reward-free RL
date: 2026-05-08
content_hash: 5c29fd38727f0ab9
---

# Human-Inspired Multi-Level Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2501.07502](https://arxiv.org/abs/2501.07502)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: rating-based RL, KL divergence, human feedback, multi-level learning, reward-free RL

## TL;DR
This paper proposes RbRL-KL, which augments rating-based RL (RbRL) with a KL divergence-driven policy loss term. By leveraging failure experiences across different rating levels with varying weights to repel the current policy, RbRL-KL outperforms standard RbRL across 6 DeepMind Control environments.

## Background & Motivation

**Background**: In reward-free settings, RLHF infers rewards from human feedback. PbRL employs preference comparisons, while RbRL uses rating annotations for reward learning.

**Limitations of Prior Work**: RbRL utilizes ratings solely for reward learning, discarding the policy-directional information embedded in different rating levels.

**Key Challenge**: Failure experiences of varying performance levels are treated uniformly, whereas humans naturally distinguish between them—completely missing a ball versus going out of bounds represent errors of different severity.

**Goal**: Directly exploit multi-level rating information during policy learning, enabling the policy to distance itself from failure experiences of different performance levels by varying degrees.

**Key Insight**: KL divergence is used to measure the distributional similarity between the current policy and experiences at different rating levels, with penalties applied via decreasing weights.

**Core Idea**: A hierarchical policy loss based on KL divergence allows RL agents to extract directional information from multi-level failure experiences in a manner analogous to human learning.

## Method

### Overall Architecture

RbRL-KL extends standard RbRL with a third channel: low-level information extraction (RbRL reward learning), high-level information extraction (KL divergence policy direction), and joint training.

### Key Designs

1. **Tiered Rating Buffer Storage**:

    - $n$ rating classes are stored in separate buffers $R_0, \ldots, R_{n-1}$
    - The highest rating level is excluded from the KL loss; all remaining levels are treated as "failures" of varying performance

2. **Multivariate Gaussian Representation**:

    - Trajectory sets at each rating level and the current policy trajectories are parameterized as multivariate Gaussians $\mathcal{N}(\mu, \Sigma)$

3. **Hierarchical KL Divergence Policy Loss**:

    - Core formula: $\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log(\pi_\theta) \hat{R}(\sigma_\theta)] - \nabla_\theta \sum_{i=0}^{n-2} \omega_i D_{KL}(D_i \| D_{\pi_\theta})$
    - KL divergence computed analytically via multivariate Gaussians: $D_{KL}(P\|Q) = \frac{1}{2}(\text{Tr}(\Sigma_Q^{-1}\Sigma_P) + (\mu_P-\mu_Q)^T \Sigma_Q^{-1}(\mu_Q-\mu_P) + \ln\frac{\det\Sigma_Q}{\det\Sigma_P})$
    - Weights satisfy $\omega_0 > \omega_1 > \cdots > \omega_{n-2}$, imposing larger penalties for lower rating levels
    - The first term corresponds to standard policy gradient; the second repels the policy from suboptimal behaviors of varying degrees

### Loss & Training
- An initial phase of $M$ episodes collects ratings to train the reward predictor; subsequent updates jointly optimize both loss terms
- The KL loss module is added in a plug-and-play fashion without modifying the original RbRL framework

## Key Experimental Results

### Main Results (6 DeepMind Control Environments)

| Environment | RbRL(n=4) | RbRL-KL(n=4) | Gain% | RbRL(n=6) | RbRL-KL(n=6) | Gain% |
|------|-----------|-------------|-------|-----------|-------------|-------|
| Cartpole | 402.55 | **417.54** | +3.7 | 306.92 | **381.79** | +24.4 |
| Ball-in-cup | 789.30 | **861.47** | +9.1 | 828.62 | **873.92** | +5.5 |
| Finger-spin | 511.55 | **579.27** | +13.2 | 559.73 | **646.37** | +15.5 |
| HalfCheetah | 238.99 | **337.04** | +41.0 | 235.46 | **303.88** | +29.1 |
| Walker | 606.14 | **742.05** | +22.4 | 797.90 | **825.18** | +3.4 |
| Quadruped | 308.48 | **477.29** | +54.7 | 199.83 | **306.78** | +53.5 |

### Gain Percentage Across Different Numbers of Rating Classes

| Environment | n=3 | n=4 | n=5 | n=6 |
|------|-----|-----|-----|-----|
| Cartpole | +15.5% | +3.7% | +22.5% | +24.4% |
| HalfCheetah | +60.0% | +41.0% | +45.2% | +29.1% |
| Quadruped | -7.5% | +54.7% | +226.0% | +53.5% |

### Key Findings
- Significant gains are observed in high-complexity environments (HalfCheetah, Walker, Quadruped)
- Occasional negative gains at low rating class counts (n=3): coarse failure groupings lead to overly uniform KL penalties
- Uniform hyperparameters ($\omega_i$ decaying as $2^{-i}$) generalize across all environments

## Highlights & Insights
- **Human Learning Analogy**: The hierarchical KL penalty formalizes the intuition of "learning different lessons from different mistakes"
- **Modular Design**: Plug-and-play compatibility with PPO/DDPG/SAC
- **Multivariate Gaussian Approximation** is concise and effective, avoiding complex distributional estimation
- The approach is transferable to graded penalties in preference-based RL

## Limitations & Future Work
- The weights $\omega_i$ are set manually, lacking an adaptive mechanism
- The multivariate Gaussian assumption may be inaccurate for high-dimensional multimodal distributions
- The optimal choice of rating class count $n$ is environment-dependent
- Experimental environments are relatively simple

## Related Work & Insights
- **vs. RbRL (White et al. 2024)**: Original RbRL uses ratings only for reward learning; this work adds a policy-direction learning channel, making the two complementary
- **vs. PbRL (Christiano et al. 2017)**: PbRL relies on pairwise preferences and cannot assess the absolute quality of individual samples; RbRL-KL exploits the multi-level information in absolute ratings
- **vs. Wu et al. (2024) negative experience RL**: Their approach applies a uniform penalty for all failure experiences; this work distinguishes between failure levels for finer-grained shaping
- **vs. DQfD/DDPGfD**: These methods incorporate expert demonstrations into replay; this work uses multi-level non-expert failure experiences for policy shaping
- **vs. NAC (Gao et al. 2018)**: NAC initializes policies with noisy demonstrations and fine-tunes; this work continuously exploits multi-level information throughout training

### Hyperparameter Settings

| Parameter | Value | Description |
|------|------|------|
| Clip $\epsilon$ | 0.4 | PPO clip parameter |
| Learning rate $\alpha$ | 5e-5 | Unified across all environments |
| Batch size | 128 | Unified across all environments |
| Hidden layers | 3 | Unified across all environments |
| $\omega_0$ | 1.0 | Weight for lowest rating level |
| $\omega_1$ | 0.5 | Exponential decay |
| $\omega_2$ | 0.25 | Exponential decay |

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical KL loss is intuitively novel, though technically a combination of existing components
- Experimental Thoroughness: ⭐⭐⭐ 6 environments with 10 seeds is acceptable, but ablation studies and more complex benchmarks are lacking
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete formulations, and well-motivated presentation
- Value: ⭐⭐⭐⭐ Concisely and effectively exploits multi-level feedback, offering meaningful reference for RLHF research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization](learning_human-like_rl_agents_through_trajectory_optimization_with_action_quanti.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go](mixing_expert_knowledge_bring_human_thoughts_back_to_the_game_of_go.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)

</div>

<!-- RELATED:END -->
