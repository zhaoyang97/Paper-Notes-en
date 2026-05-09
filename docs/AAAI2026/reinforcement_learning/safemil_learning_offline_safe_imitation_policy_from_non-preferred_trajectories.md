---
title: >-
  [Paper Note] SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories
description: >-
  [AAAI 2026][Reinforcement Learning][Offline safe imitation learning] This paper proposes SafeMIL, which formulates cost function learning as a Multiple Instance Learning (MIL) problem to learn a safe imitation policy from a limited set of non-preferred trajectories and a large collection of unlabeled trajectories—without requiring step-level reward or cost annotations—achieving constraint satisfaction performance 3.7× better than the strongest baseline.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Offline safe imitation learning
  - multiple instance learning
  - constrained MDP
  - behavioral cloning
  - cost function learning
date: 2026-05-08
content_hash: 3b8b1e5f5e9187cb
---

# SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories

**Conference**: AAAI 2026
**arXiv**: [2511.08136](https://arxiv.org/abs/2511.08136)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Offline safe imitation learning, multiple instance learning, constrained MDP, behavioral cloning, cost function learning

## TL;DR

This paper proposes SafeMIL, which formulates cost function learning as a Multiple Instance Learning (MIL) problem to learn a safe imitation policy from a limited set of non-preferred trajectories and a large collection of unlabeled trajectories—without requiring step-level reward or cost annotations—achieving constraint satisfaction performance 3.7× better than the strongest baseline.

## Background & Motivation

### State of the Field

Deploying reinforcement learning in real-world settings faces two major challenges:

**High risk of online interaction**: In robotics, autonomous driving, and similar domains, online trial-and-error is prohibitively costly.

**Difficulty of reward function design**: Crafting appropriate reward functions for complex tasks often leads to unintended behaviors.

Imitation learning (IL) sidesteps reward design by learning from expert demonstrations, but conventional IL implicitly assumes all demonstrations are safe. When unsafe trajectories are mixed into the data, direct imitation may reproduce dangerous behaviors.

### Design Motivation

A practically relevant scenario exists in which:
- A **small number of non-preferred trajectories** are available (e.g., traffic accident recordings, reported toxic chat logs).
- A **large collection of unlabeled trajectories** exists (mixing safe and unsafe behaviors).
- Per-step reward and cost information is **unavailable**.

**Labeling a trajectory as "non-preferred" is far easier than annotating per-step costs.** For example:
- Autonomous driving: knowing that a particular driving segment "ran a red light" is easier than annotating precise safety costs for every frame.
- Chatbots: user reports of toxic content require only trajectory-level labels.

### Limitations of Prior Work

| Method | Online Interaction | Per-step Reward/Cost | Data Source |
|---|---|---|---|
| Standard RL | ✓ | ✓ | Online interaction |
| Offline Safe RL | ✗ | ✓ (cost annotation required) | Offline dataset |
| Standard IL | ✗ | ✗ | Expert demonstrations |
| T-REX/PEBBLE | ✓ (online RL required) | ✗ (reward from rankings) | Ranked trajectories |
| SafeDICE | ✗ | ✗ | Non-preferred + unlabeled |
| **SafeMIL (Ours)** | ✗ | ✗ | Non-preferred + unlabeled |

SafeMIL is the **first work to introduce MIL into the offline safe IL setting**.

## Method

### Overall Architecture

SafeMIL operates in two stages:
1. **Cost function learning**: A state-action-level cost function $\hat{c}_\theta(s, a)$ is learned from trajectory-level labels via the MIL framework.
2. **Safe policy learning**: The learned cost function is used to filter or reweight preferred trajectories from the unlabeled data, from which a safe policy is learned via behavioral cloning (BC).

### Key Designs

#### 1. Formulating Cost Function Learning as a MIL Problem

**MIL background**:
- Data is organized into *bags*: $\mathcal{B} = \{x_1, x_2, \ldots, x_K\}$
- Only bag-level labels $Y$ are available; instance-level labels are absent.
- A bag is positive ($Y=1$) if and only if it contains at least one positive instance.
- A bag is negative ($Y=0$) if and only if all instances are negative.

**Trajectory-to-MIL mapping**:
- **Negative bags**: $K$ trajectories sampled with replacement from the non-preferred dataset $\mathcal{D}^N$ (all trajectories are guaranteed non-preferred).
- **Unlabeled bags**: $K$ trajectories sampled with replacement from the unlabeled dataset $\mathcal{D}^U$.

**Key Lemma (Lemma 1)**: The probability that an unlabeled bag contains at least one preferred trajectory is:
$$P(\mathcal{B} \cap \mathcal{T}_p \neq \emptyset) = 1 - (1-\alpha)^K$$
where $\alpha$ is the proportion of preferred trajectories in the unlabeled data. As $K$ grows, this probability approaches 1, so unlabeled bags can be treated as **positive bags**.

#### 2. Bag Scoring Function Based on Symmetric Functions

Building on the fundamental theorem of symmetric functions, a permutation-invariant bag scoring function is designed as:

$$Score(\mathcal{B}) = g\left(\sum_{\tau \in \mathcal{B}} f(\tau)\right)$$

The following functional forms are adopted:
- $f(\tau) = \frac{1}{K} \sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t)$ (average discounted cost of a trajectory)
- $g$ = identity function

The resulting score is:
$$Score(\mathcal{B}) = \frac{1}{K} \sum_{\tau \in \mathcal{B}} \sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t)$$

**Intuition**: As $K \to \infty$, the score converges to the expected cumulative cost of the bag's trajectories. Non-preferred bags are expected to have higher scores than unlabeled bags, since the latter contain preferred trajectories.

**Theorem 1**: $P(Score(\mathcal{B}_n) > Score(\mathcal{B}_u)) = 1 - (1-\alpha)^K$, i.e., the probability that a negative bag scores higher than an unlabeled bag equals the probability that the unlabeled bag contains a preferred trajectory.

#### 3. Training the Cost Function with the Bradley-Terry Loss

Exploiting the relationship that negative bags should score higher than unlabeled bags, the cost function is trained using the Bradley-Terry model:

$$\mathcal{L}_\theta = -\mathbb{E}_{\mathcal{B}_n \sim \rho^N, \mathcal{B}_u \sim \rho^U} \left[ \log \frac{\exp(Score(\mathcal{B}_n))}{\exp(Score(\mathcal{B}_n)) + \exp(Score(\mathcal{B}_u))} \right]$$

This loss encourages $\hat{c}_\theta$ to assign higher cost values to non-preferred behaviors.

#### 4. Policy Learning from the Cost Function

Given $\hat{c}_\theta$, two policy learning strategies are available:

**a) Hard thresholding**: Select trajectories whose cumulative cost falls below a threshold $\hat{b}$ and apply BC:
$$\mathcal{T}_{\hat{c}_\theta} := \{\tau \in \mathcal{D}^U \mid \sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t) \leq \hat{b}\}$$

**b) Soft-weighted BC (default)**: Assign per-trajectory weights:
$$w(\tau) = \exp\left(-\sum_{t=0}^{T-1} \gamma^t \hat{c}_\theta(s_t, a_t) / \beta\right)$$

Weighted BC loss:
$$\min_\pi \sum_{\tau \in \mathcal{D}^U} \left[ w(\tau) \sum_{t=0}^{T-1} \mathcal{L}_\pi(s_t, a_t) \right]$$

A smaller $\beta$ imposes stronger penalties on high-cost trajectories.

#### 5. Extension to Partial Trajectories

Training on full trajectories is computationally expensive; SafeMIL therefore supports **partial trajectories** of length $H$:
- Partial trajectories sampled from non-preferred data may exhibit preferred behavior, meaning some instances in negative bags may be mislabeled.
- However, when the bag size $K$ is sufficiently large, the average cost of negative bags still exceeds that of unlabeled bags, and the scoring relationship holds.

### Loss & Training

- The cost function $\hat{c}_\theta$ and the policy network $\pi$ are trained alternately.
- Each iteration samples a pair of negative and unlabeled bags to update the cost function.
- The policy is simultaneously updated via weighted BC.
- Training steps: 1,000,000.
- Number of non-preferred trajectories: 50.
- Number of unlabeled trajectories: 200.

## Key Experimental Results

### Experimental Setup

**Environments**:
- MuJoCo velocity-constrained tasks: Walker-Velocity, Swimmer-Velocity, Ant-Velocity
- Navigation tasks: Point-Circle2, Point-Goal1, Point-Button1

**Data**: The DSRL (Datasets for offline Safe RL) benchmark is used, with all reward and cost information removed.

**Evaluation metrics**:
- Normalized Return (0 = random policy, 1 = constrained RL policy)
- Normalized Cost (0 = cost level of constrained RL policy)
- Normalized CVaR@20% Cost (mean cost of the worst 20% of runs)

### Main Results

**Velocity-constrained tasks (primary results in Fig. 1):**

| Method | Walker-Vel Cost | Swimmer-Vel Cost | Ant-Vel Cost | Safety |
|---|---|---|---|---|
| BC-Unlabeled | High (>0) | High (>0) | High (>0) | Learns non-preferred behaviors |
| SafeDICE | Moderate | Moderate | Moderate | Partial constraint satisfaction |
| DWBC-NU | Moderate | Moderate | Moderate | Unstable |
| T-REX-WBC | Moderate | Moderate | Moderate | Partial improvement |
| **SafeMIL** | **≈0** | **≈0** | **≈0** | **Best safety** |

**Navigation tasks**: SafeMIL achieves the best performance on Point-Goal1 and remains competitive with baselines on Point-Circle2 and Point-Button1.

**Across all environments**: The median safety performance of SafeMIL is **3.7× better** than the strongest baseline.

### Ablation Study

**Sensitivity to bag size $K$ (Swimmer-Velocity):**

| Bag size $K$ | Normalized Cost | Normalized Return | Note |
|---|---|---|---|
| 1 | High | Normal | No MIL effect |
| 8 | Reduced | Normal | Beginning to take effect |
| 16 | Further reduced | Normal | Noticeable improvement |
| 64 | Near 0 | Normal | Approaching stable |
| 128 | ≈0 | Normal | Best safety |

This aligns with theoretical expectations: larger $K$ increases the probability that unlabeled bags contain preferred trajectories, yielding more accurate cost function learning.

**Sensitivity to partial trajectory length $H$ ($K=128$, Swimmer-Velocity):**

| Trajectory length $H$ | Normalized Cost | Note |
|---|---|---|
| 1 | ≈0 | Stable |
| 5 | ≈0 | Stable |
| 10 | ≈0 | Stable |

When $K$ is sufficiently large, safety performance is insensitive to trajectory length—supporting the use of partial trajectories to reduce computational cost.

**Weighting scheme comparison**:
- Trajectory-level weighting (Eq. 12) vs. state-action-level weighting (Eq. 14)
- Both approaches yield similar performance on Swimmer-Velocity and Point-Goal1.

### Key Findings

1. **The MIL framework effectively bridges trajectory-level and state-level label propagation**: Accurate state-action-level cost functions are recovered from trajectory-level "non-preferred" labels alone.
2. **Bag size $K$ is the critical hyperparameter**: MIL signals are insufficient for small $K$; performance stabilizes for $K \geq 64$.
3. **Partial-trajectory training is computationally efficient without sacrificing performance**: Full trajectory length is unnecessary in practice.
4. **SafeMIL demonstrates a clear advantage on velocity-constrained tasks**: It nearly recovers the safety level of the constrained RL policy.
5. **Competitive performance on navigation tasks**: SafeMIL achieves the best results on Point-Goal1 and matches baselines on the remaining navigation tasks.

## Highlights & Insights

1. **Elegant application of MIL formulation**: The weakly supervised problem of identifying which state-action pairs are dangerous within a trajectory is naturally cast as the MIL question of which instances within a bag are positive.
2. **Only 50 non-preferred trajectories required**: The minimal annotation demand makes the method highly practical.
3. **Theoretical guarantees**: Lemma 1 and Theorem 1 provide probabilistic guarantees on the validity of the scoring function.
4. **Simple scoring function design**: No complex attention mechanisms or deeply nested architectures are needed; a straightforward summation score suffices.
5. **Broad applicability**: The method makes no environment-specific assumptions and is extensible to any safety-critical sequential decision-making scenario.

## Limitations & Future Work

1. **Unknown preferred trajectory ratio $\alpha$**: In practice, this quantity may require estimation or tuning.
2. **Bag size selection**: Larger $K$ is theoretically preferable but increases computational cost, necessitating a trade-off.
3. **Homogeneity assumption for non-preferred behavior costs**: Theorem 1 assumes non-preferred trajectories have similar costs, which may not hold in practice.
4. **Limited baseline comparison**: No comparison is made against online safe RL methods or a broader set of preference-based approaches.
5. **Continuous action spaces**: Validation is primarily conducted on MuJoCo and navigation tasks; more complex high-dimensional settings (e.g., autonomous driving) remain to be explored.
6. **Cost threshold $\hat{b}$ selection**: The hard-thresholding variant requires prior knowledge to set an appropriate threshold.

## Related Work & Insights

- **SafeDICE (Jang et al., 2023)**: Directly estimates the stationary distribution of the preferred policy; the most direct comparison baseline.
- **T-REX (Brown et al., 2019)**: Learns reward from ranked trajectories but requires online RL for optimization.
- **DWBC (Xu et al., 2022)**: Trains a discriminator via PU learning to serve as BC weights.
- **COptiDICE (Lee et al., 2022)**: Offline RL with constraints; serves as an upper-bound reference (uses complete reward and cost information).
- Key insight: **Weak supervision signals (trajectory-level labels) combined with an appropriate learning framework (MIL) can effectively substitute for costly step-level annotations.**

## Rating

- Novelty: ⭐⭐⭐⭐ (innovative MIL formulation for safe IL)
- Experimental Thoroughness: ⭐⭐⭐⭐ (6 environments + multi-dimensional sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear problem formulation, good integration of theory and experiments)
- Value: ⭐⭐⭐⭐ (highly practical; safe policies are learnable from only 50 non-preferred trajectories)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/boundary_to_region_supervision_for_offline_safe_rl.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)

<!-- RELATED:END -->
