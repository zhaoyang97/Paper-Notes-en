---
title: >-
  [Paper Note] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning
description: >-
  [AAAI2026][Reinforcement Learning][off-policy reinforcement learning] This paper proposes Behaviour Policy Optimization (BPO), which optimizes a dedicated behaviour policy for off-policy data collection such that the variance of return estimates is provably lower than on-policy collection, thereby improving the sample efficiency and stability of REINFORCE and PPO.
tags:
  - AAAI2026
  - Reinforcement Learning
  - off-policy reinforcement learning
  - variance reduction
  - importance sampling
  - behaviour policy
  - policy gradient
date: 2026-05-08
content_hash: ae557662d4f70930
---

# Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning

**Conference**: AAAI2026
**arXiv**: [2511.10843](https://arxiv.org/abs/2511.10843)
**Code**: [sacktock/BPO](https://github.com/sacktock/BPO)
**Area**: Reinforcement Learning
**Keywords**: off-policy reinforcement learning, variance reduction, importance sampling, behaviour policy, policy gradient

## TL;DR

This paper proposes Behaviour Policy Optimization (BPO), which optimizes a dedicated behaviour policy for off-policy data collection such that the variance of return estimates is provably lower than on-policy collection, thereby improving the sample efficiency and stability of REINFORCE and PPO.

## Background & Motivation

- Policy gradient methods (e.g., REINFORCE, PPO) rely on return estimates for policy updates; high-variance estimates lead to gradient oscillation, unstable learning, and poor sample efficiency.
- Traditional off-policy methods (e.g., IMPALA, ACER) primarily address data mismatch between multiple parallel workers and asynchronous policy updates by truncating importance weights to control variance.
- Recent advances in off-policy evaluation (ODI, Liu et al. 2024) reveal a counter-intuitive finding: **data collected by a carefully designed behaviour policy can yield return estimates with provably lower variance than on-policy collection**.
- However, ODI only considers undiscounted finite-horizon policy evaluation; directly applying it to online RL policy improvement faces challenges including non-stationary target policies and the unsuitability of full Monte Carlo returns.

## Core Problem

How to extend the idea of "designing an optimal behaviour policy to reduce return estimate variance" from off-policy evaluation to the online RL setting, so that both policy evaluation and policy improvement benefit from lower-variance estimates?

## Method

### Core Idea

Rather than collecting data on-policy under the current target policy $\pi$, BPO maintains a dedicated, optimized behaviour policy $\mu$ such that the importance-weighted return estimates derived from its off-policy data exhibit lower variance.

### Variance-Optimal Behaviour Policy

Based on the theoretical results of ODI, the one-step optimal behaviour policy $\hat{\mu}$ takes the form:

$$\hat{\mu}(a|s) \propto \pi(a|s) \sqrt{\hat{q}_\pi(s,a)}$$

where $\hat{q}_\pi(s,a)$ incorporates terms for $q_\pi^2$, the variance of the next-state value function, and subsequent PDIS variance. Intuitively, this policy assigns higher probability to actions with larger $|q_\pi|$, thereby reducing the variance of importance weights.

### Truncated Importance-Sampled TD(λ) Returns

The paper proposes a novel return estimator $G_t^{\text{TIS},\lambda}$ combining TD(λ) with truncated importance weights:

$$G_t^{\text{TIS},\lambda} = v_\pi(S_t) + \sum_{k=t}^{\infty} (\gamma\lambda)^{k-t} \left(\prod_{i=t}^{k-1} c_i\right) \delta_k$$

where $c_t = \min(\bar{c}, \pi(A_t|S_t)/\mu(A_t|S_t))$ is the truncated trace coefficient and $\delta_k$ is the TD error. This estimator can be computed efficiently via recursion (analogous to eligibility traces) and provides the following theoretical guarantees:

- **Unbiasedness (Theorem 1)**: When $\bar{c}, \bar{\rho} = \infty$, for any $\mu \in \Lambda$, $\mathbb{E}_\mu[G_t^{\text{TIS},\lambda}|S_t=s] = v_\pi(s)$.
- **Variance Reduction (Theorem 2)**: When $\lambda=1, \bar{c}, \bar{\rho}=\infty$, using $\hat{\mu}$ for data collection yields variance strictly no higher than on-policy collection under $\pi$.

### Learning $\hat{q}_\pi$

$\hat{q}_\pi$ can be expressed as another Q-function defined with a modified reward $\hat{r}_\pi(s,a) = 2r(s,a)q_\pi(s,a) - r^2(s,a)$ and discount factor $\gamma^2$. Thus, only one additional Q-network needs to be trained.

### Algorithm Implementation

BPO augments a base policy gradient algorithm with three auxiliary modules:

1. **Two Q-networks**: Separately estimate $q_\pi$ and $\hat{q}_\pi$ via Fitted Q-Evaluation (FQE), using symlog target transformation to stabilize training.
2. **Behaviour policy optimization**: For discrete action spaces, a cross-entropy loss is used to match the target distribution; for continuous action spaces, a log-probability distance loss is used.
3. **Off-policy data collection**: The optimized $\mu$ is used for rollouts, and truncated IS TD(λ) returns are used to compute advantage estimates.

### Integration with PPO

- The ratio in PPO's clipped surrogate objective is replaced by $r_t(\theta) = \pi_\theta(A_t|S_t) / \mu(A_t|S_t)$.
- The advantage estimate is replaced by $\hat{A}_t = \hat{G}_t^{\text{TIS},\lambda} - V_\omega(S_t)$.
- The value function uses truncated IS TD(λ) returns as regression targets.
- The policy and value function update rules remain identical to standard PPO; only the data source and return estimation procedure change.

## Key Experimental Results

### REINFORCE + ShortCorridor

- BPO converges faster and is more stable in later training stages, though the advantage is modest in this simple environment.
- Best truncation parameters: $\bar{c}=1.0, \bar{\rho}=1.5$.

### PPO + MuJoCo

Results on Ant-v5, HalfCheetah-v5, Hopper-v5, and Walker2d-v5 (10 independent runs):

| Environment | PPO | BPO (best config) | Gain |
|---|---|---|---|
| Ant-v5 (gSDE) | 1106±111 | **1690±125** | +53% |
| HalfCheetah-v5 (gSDE) | 3425±468 | 3742±408 | +9% |
| Hopper-v5 | 3527±670 | **4749±419** | +35% |
| Walker2d-v5 | 2126±492 | **2770±296** | +30% |
| Walker2d-v5 (default) | 2091±408 | **3044±332** | +46% |

- Nearly all BPO configurations outperform baseline PPO, with most improvements being statistically significant.
- BPO converges faster in early training and is more stable in later stages.
- Trajectory-level truncation (traj $\bar{c}=1.0$) performs best on Walker2d.

## Highlights & Insights

1. **Solid theoretical foundation**: Beyond providing intuition for variance reduction, the paper offers rigorous proofs of unbiasedness and variance reduction, elegantly extending off-policy evaluation theory to online RL.
2. **Counter-intuitive insight**: The paper demonstrates that on-policy collection is not variance-optimal, and that a carefully designed off-policy collection scheme can be strictly better.
3. **Practical estimator design**: The truncated IS TD(λ) return balances theoretical unbiasedness guarantees with practical variance control.
4. **Plug-and-play**: BPO functions as an add-on module that can be stacked on top of existing policy gradient algorithms without modifying their core update logic.
5. **Continuous action support**: An alternative loss function for continuous action spaces is designed with theoretical justification (Theorem 4).

## Limitations & Future Work

1. **Increased computational cost**: Training two additional Q-networks and a behaviour policy network increases both computational burden and hyperparameter tuning complexity.
2. **Hyperparameter sensitivity**: The choice of truncation parameters $\bar{c}, \bar{\rho}$ substantially affects performance, and the optimal configuration varies across environments.
3. **Limited experimental scope**: PPO is only evaluated on MuJoCo continuous control tasks; discrete high-dimensional tasks such as Atari or more complex environments are not considered.
4. **Q-network estimation error propagation**: Overestimation or approximation errors in FQE degrade behaviour policy quality; while symlog targets mitigate this, the issue is not fully resolved.
5. **Not extended to the actor-critic family**: Experiments are limited to REINFORCE and PPO; the effectiveness for A2C, SAC, TD3, and related methods remains unverified.

## Related Work & Insights

| Method | Core Idea | Difference from BPO |
|---|---|---|
| IMPALA (V-trace) | Truncated IS weights to correct asynchronous multi-worker data | BPO focuses on optimal behaviour policy design for a single worker; orthogonal direction |
| ACER | Truncated IS + experience replay | Does not actively optimize the behaviour policy; only passively truncates weights |
| GAE | Bias-variance tradeoff for multi-step returns | BPO reduces variance at the sampling distribution level; complementary to GAE |
| ODI (Liu et al.) | Designs optimal behaviour policy for off-policy evaluation | BPO extends this to online RL, introducing a TD(λ) estimator and handling non-stationarity |
| Retrace(λ) | TD(λ) with IS weights truncated to 1 | BPO's truncation is generally less aggressive, and the behaviour policy naturally yields well-behaved IS weights |
| ROS | Re-weights behaviour policy to cover under-represented states | Assumes complete trajectories and known policies; does not use IS correction |

BPO's core insight—**that the data collection policy can differ from and outperform the target policy**—has potential value for RLHF/RLAIF and other LLM alignment settings, where more efficient exploration policies could be designed. This echoes the spirit of curriculum learning: rather than uniformly sampling training data, the data distribution is adaptively selected to be most beneficial for learning. The BPO framework is in principle extensible to model-based RL, where a learned environment model could be used to design a globally optimal behaviour policy.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of variance-optimal sampling theory from off-policy evaluation to online RL; conceptually novel.
- Experimental Thoroughness: ⭐⭐⭐ — MuJoCo results are promising but environment coverage is incomplete; discrete high-dimensional tasks are absent.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and complete, progressing systematically from intuition to formal proof.
- Value: ⭐⭐⭐⭐ — Provides a general variance reduction framework with practical improvement value for policy gradient methods.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[ICLR 2026\] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](../../ICLR2026/reinforcement_learning/policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](../../NeurIPS2025/reinforcement_learning/boundary-to-region_supervision_for_offline_safe_reinforcement_learning.md)

<!-- RELATED:END -->
