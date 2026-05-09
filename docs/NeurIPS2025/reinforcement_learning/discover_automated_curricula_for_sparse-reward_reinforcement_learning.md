---
title: >-
  [Paper Note] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][sparse reward] This paper proposes DISCOVER, a goal selection strategy for sparse-reward long-horizon RL that simultaneously balances achievability, novelty, and relevance to construct curricula directed toward a target task. The authors theoretically prove that the number of steps to reach the goal scales linearly with goal distance rather than with the volume of the search space, and demonstrate significant improvements over prior state-of-the-art exploration strategies on high-dimensional navigation and manipulation tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - sparse reward
  - goal selection
  - exploration-exploitation
  - UCB
  - curriculum learning
  - goal-conditioned RL
date: 2026-05-08
content_hash: e9e15ab908554351
---

# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.19850](https://arxiv.org/abs/2505.19850)
**Code**: [Available](https://github.com/LeanderDiazBone/discover)
**Area**: Reinforcement Learning
**Keywords**: sparse reward, goal selection, exploration-exploitation, UCB, curriculum learning, goal-conditioned RL

## TL;DR
This paper proposes DISCOVER, a goal selection strategy for sparse-reward long-horizon RL that simultaneously balances achievability, novelty, and relevance to construct curricula directed toward a target task. The authors theoretically prove that the number of steps to reach the goal scales linearly with goal distance rather than with the volume of the search space, and demonstrate significant improvements over prior state-of-the-art exploration strategies on high-dimensional navigation and manipulation tasks.

## Background & Motivation

**Sparse rewards are a core challenge in RL**: the agent receives reward only upon task completion, necessitating deep exploration—which is practically infeasible via random exploration in long-horizon, high-dimensional spaces.

**The exploration dilemma in goal-conditioned RL**:
- HER relabeling provides learning signal from off-task goals
- However, goal selection faces its own exploration–exploitation trade-off: goals that are too difficult yield no learning signal; goals that are too easy provide no new information; irrelevant goals waste samples

**Biases in existing methods**:
- **HER**: always selects the final goal—too difficult to learn from
- **MEGA**: selects the rarest achieved goal—explores all directions uniformly, leading to exponential growth in high-dimensional spaces
- **DISCERN (uniform)**: randomly selects from achieved goals—similarly undirected
- All these methods lack a **relevance** dimension—they do not consider whether intermediate goals facilitate progress toward the final goal

## Method

### Core Idea: Goal Utility = Achievability + Novelty + Relevance

DISCOVER selects a goal $g_t$ from the set of achieved goals $\mathcal{G}_{\text{ach}}$ at each episode:

$$g_t = \arg\max_{g \in \mathcal{G}_{\text{ach}}} \underbrace{V(s_0, g) + \alpha_t \sigma(s_0, g)}_{\text{Achievability + Novelty}} + \underbrace{\beta_t [V(g, g^\star) + \alpha_t \sigma(g, g^\star)]}_{\text{Relevance}}$$

**Interpretation of the three principles**:
- High $V(s_0, g)$ → the policy can reach $g$ (**achievability**)
- High $\sigma(s_0, g)$ → uncertainty about reaching $g$ (**novelty**, encouraging exploration of unknown regions)
- High $V(g, g^\star)$ → $g$ is close to the final goal $g^\star$ (**relevance**, providing directionality)
- High $\sigma(g, g^\star)$ → uncertainty about the value of $g \to g^\star$ (**exploration of relevance**, preventing premature convergence to incorrect directions)

### Uncertainty Estimation
A **critic ensemble** (multiple Q-networks) is used to estimate the mean and variance $\sigma^2$ of $V$, bootstrapped entirely from the agent's own experience without any prior information.

### Adaptive Parameter Adjustment
$\alpha_t$ is automatically tuned to maintain a goal achievement rate of $p^\star \approx 50\%$:

$$\alpha_{t+1} = \Pi_{[0,1]}(\alpha_t + \eta(p_t - p^\star))$$

where $p_t$ is the achievement rate over the most recent $k_{\text{adapt}}$ episodes. If too low, $\alpha$ is decreased (easier goals selected); if too high, $\alpha$ is increased (more novel goals selected).

### Theoretical Guarantees (Simplified Setting)

Under linear feature space and Gaussian noise assumptions, the number of episodes required to reach the goal is bounded by:

$$N \leq \tilde{O}\left(\frac{Dd^2}{\kappa^3}\right)$$

- $D = -V^\star(s_0, g^\star)$: optimal distance from start to goal
- $d$: feature space dimensionality
- $\kappa$: expansion rate of the reachable set

**Key significance**: the bound depends only on distance $D$ (one-dimensional), not on the volume of the goal space $\mathcal{G}$. Prior methods such as MEGA have bounds that depend on $|\mathcal{G}|$, which grows exponentially in high-dimensional spaces. DISCOVER avoids the curse of dimensionality through directionality.

### Algorithm Overview

At each episode:
1. **SelectGoal**: select $g_t$ from $\mathcal{G}_{\text{ach}}$ according to the DISCOVER objective
2. **Rollout**: execute $\pi(g_t)$ until $g_t$ is achieved
3. Upon achievement, continue with random exploration until episode end
4. Update the replay buffer
5. Update parameters using off-policy RL (TD3) with HER relabeling

## Key Experimental Results

### High-Dimensional Pointmaze (Steps to 10% Success Rate, M steps)

| Dimensionality | HER | MEGA | Ach.+Nov. | DISCOVER |
|---|---|---|---|---|
| 2 | ∞ | 4.8M | 5.2M | **2.9M** |
| 3 | ∞ | ∞ | ∞ | **3.1M** |
| 4 | ∞ | ∞ | ∞ | **7.4M** |
| 5 | ∞ | ∞ | ∞ | **5.4M** |
| 6 | ∞ | ∞ | ∞ | **18.7M** |

∞ indicates failure to reach the threshold within 50M steps. Only DISCOVER succeeds in three or more dimensions—undirected exploration explodes exponentially in high-dimensional spaces, while DISCOVER tunnels toward the goal directionally.

### Antmaze + Arm Environments (Success Rate, Peak During Late Training)

| Environment / Method | HER | MEGA | DISCERN | Ach.+Nov. | **DISCOVER** |
|---|---|---|---|---|---|
| Antmaze (simple) | ~40% | ~80% | ~70% | ~75% | **~90%** |
| Antmaze (hard) | 0% | ~20% | ~15% | ~25% | **~60%** |
| Arm (simple) | ~50% | ~60% | ~55% | ~55% | **~85%** |
| Arm (hard) | 0% | 0% | 0% | 0% | **~30%** |

The advantage is most pronounced in hard configurations—other methods achieve either very low performance (Antmaze hard) or fail to converge entirely (Arm hard).

### Ablation Study Key Findings
- **Removing Relevance** (= Ach.+Nov.): fails on all high-dimensional Pointmaze tasks; Antmaze hard drops from 60% to 25%
- **Removing Novelty** (Ach.+Rel. only): prone to premature convergence to incorrect directions
- **Removing Achievability** (Nov.+Rel. only): selects unreachable goals, yielding no effective learning signal
- All three components are indispensable

### Leveraging Prior Knowledge
- Manually specified distance priors (e.g., L2 distance) or pretrained critics marginally accelerate exploration
- However, the fully bootstrapped version of DISCOVER is already sufficiently strong; gains from priors are limited

### Standard RL Methods Fail Completely
Methods such as TD3 + curiosity, TD3 + RND, and TD3 + count-based exploration fail to reach the goal even in simple environments—deep exploration requires subgoal decomposition.

## Highlights & Insights

- **Directionality is the key**: the central insight of DISCOVER is that achievability and novelty alone are insufficient; the agent must also know *which direction to explore*. This directionality is derived entirely from bootstrapped critic estimates, requiring no oracle
- **Resolving the curse of dimensionality in exploration**: the complexity of undirected exploration scales as $(R/\epsilon)^m$ (exponential in dimension), whereas DISCOVER reduces this to $O(D)$ (one-dimensional distance)—verified both theoretically and empirically
- **A new application of UCB**: the multi-armed bandit UCB principle is generalized to goal selection—each reachable goal serves as an "arm," and the reward reflects its contribution toward reaching the final task
- **Adaptive difficulty control**: maintaining approximately 50% goal achievement rate automatically balances exploration and exploitation in a simple and effective manner

## Limitations & Future Work
- Relies on a critic ensemble for uncertainty estimation, incurring additional computational cost for large critic networks
- Theoretical guarantees assume linear features, which diverges from the non-stationarity of bootstrapping in deep networks
- Goals can only be **selected** from previously achieved goals; **generating** new goals is not supported—limiting fine-grained control in continuous spaces
- The current work focuses on single-goal tasks; extension to multi-goal distributions requires further design
- Only TD3 is used as the backbone; compatibility with model-based methods (e.g., Dreamer) has not been validated

## Related Work & Insights
- **vs. MEGA (Pitis et al.)**: MEGA selects the rarest achieved goal (maximizing entropy of the goal distribution) but is entirely undirected. Adding the relevance term in DISCOVER turns failure into success on high-dimensional tasks
- **vs. HER**: HER selects only the final goal, equivalent to pure exploitation. DISCOVER outperforms HER by orders of magnitude on difficult tasks
- **Connection to Test-Time RL (TTRL)**: DISCOVER can be viewed as successful test-time RL, solving initially infeasible goals through millions of steps of autonomous exploration
- **Implications for LLMs**: the paper notes that the principle of directed exploration may apply to LLM self-improvement in complex goal spaces such as programming and mathematics—a direction worth monitoring

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified objective of Achievability + Novelty + Relevance is an elegant contribution; the theoretical connection to UCB is deep
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 environments × 2 difficulty levels, dimensionality sweep (2–6), 5 baselines, detailed ablations and visualizations, 10 seeds
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative connecting theory and experiments is coherent; Figure 2's intuition illustration and Figure 4's goal selection visualization are exceptionally clear
- Value: ⭐⭐⭐⭐⭐ Addresses the core exploration challenge in sparse-reward RL with theoretical guarantees, strong empirical evidence, and a concise, general method

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Thermodynamics of Reinforcement Learning Curricula](../../ICLR2026/reinforcement_learning/thermodynamics_of_reinforcement_learning_curricula.md)
- [\[NeurIPS 2025\] Learning Interestingness in Automated Mathematical Theory Formation](learning_interestingness_in_automated_mathematical_theory_formation.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)

<!-- RELATED:END -->
