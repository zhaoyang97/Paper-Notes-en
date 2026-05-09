---
title: >-
  [Paper Note] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Stackelberg Game] This paper proposes SCORER, a framework that models representation learning and value function learning in Deep Q-Learning as a Stackelberg game. Through two-timescale updates—where the Q-network acts as the slow-updating leader and the encoder as the fast-updating follower—SCORER achieves stable co-adaptation without modifying the network architecture.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Stackelberg Game
  - Representation Learning
  - Deep Q-Learning
  - Two-Timescale
  - Variance Minimization
date: 2026-05-08
content_hash: c6df223b48514a2f
---

# Stackelberg Coupling of Online Representation Learning and Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2508.07452](https://arxiv.org/abs/2508.07452)
**Code**: [https://github.com/fernando-ml/SCORER](https://github.com/fernando-ml/SCORER)
**Area**: Reinforcement Learning / Representation Learning
**Keywords**: Stackelberg Game, Representation Learning, Deep Q-Learning, Two-Timescale, Variance Minimization

## TL;DR

This paper proposes SCORER, a framework that models representation learning and value function learning in Deep Q-Learning as a Stackelberg game. Through two-timescale updates—where the Q-network acts as the slow-updating leader and the encoder as the fast-updating follower—SCORER achieves stable co-adaptation without modifying the network architecture.

## Background & Motivation

- **Deadly Triad**: In Deep Q-Learning, the combination of function approximation, bootstrapping, and off-policy learning induces instability, potentially leading to representation collapse and catastrophic learning failures.
- **Limitations of Monolithic Networks**: Conventional approaches jointly learn representations and value functions within a single network, forcing representations to continuously adapt to non-stationary value targets while value estimates depend on a shifting representation, forming a vicious cycle.
- **Gradient Conflicts from Auxiliary Losses**: Introducing auxiliary losses (e.g., self-supervised objectives) to stabilize representations may conflict with the primary value learning objective.
- **Mechanism**: Rather than simply adding auxiliary losses, SCORER fundamentally restructures the optimization problem by modeling it as a hierarchical Stackelberg game.

## Method

### Overall Architecture

SCORER decomposes the agent into two strategic players:
- **Leader (control network $Q_\theta$)**: Responsible for value estimation; updated slowly to provide stable targets.
- **Follower (perception network $f_\phi$)**: Responsible for representation learning; updated rapidly to compute the best response to the leader's strategy.

### 1. Leader Objective: Minimizing MSBE

$$\min_\theta \mathcal{L}_{\text{leader}}(Q_\theta, f_{\phi^*(\theta)}) \triangleq \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[(Y - Q_\theta(f_{\phi^*(\theta)}(s), a))^2\right]$$

### 2. Follower Objective: Minimizing Bellman Error Variance

$$\phi^*(\theta) \in \arg\min_\phi \mathcal{L}_{\text{follower}}(f_\phi, Q_\theta) \triangleq \text{Var}_{j \in B}[\delta_j(\phi, \theta)]$$

where $\delta_j(\phi, \theta) = Y_j - Q_\theta(f_\phi(s_j), a_j)$ is the Bellman error. Variance minimization is preferred over MSBE because it encourages representations that produce more consistent Bellman errors, making them more robust to noisy TD targets and directly countering the root cause of the deadly triad.

### 3. Two-Timescale Approximation of Bilevel Optimization

The problem is formally cast as bilevel optimization:

$$\min_\theta \mathcal{L}_{\text{leader}}(Q_\theta, f_{\phi^*(\theta)}) \quad \text{s.t.} \quad \phi^*(\theta) \in \arg\min_\phi \mathcal{L}_{\text{follower}}(f_\phi, Q_\theta)$$

The Stackelberg equilibrium is approximated via two-timescale gradient descent:
- Follower uses a larger learning rate $\alpha_{\phi,k}$ (fast timescale)
- Leader uses a smaller learning rate $\alpha_{\theta,k}$ (slow timescale)
- Satisfying $\lim_{k \to \infty} \alpha_{\theta,k} / \alpha_{\phi,k} = 0$

Update rules (with stop-gradient to block gradient flow):

$$\phi_{k+1} \leftarrow \phi_k - \alpha_{\phi,k} \nabla_\phi \mathcal{L}_{\text{follower}}(\phi_k; B_{\text{follower}}, Y, \bar{\theta_k})$$

$$\theta_{k+1} \leftarrow \theta_k - \alpha_{\theta,k} \nabla_\theta \mathcal{L}_{\text{leader}}(\theta_k; B_{\text{leader}}, Y, \bar{\phi_{k+1}})$$

### 4. Implementation Simplicity

SCORER is straightforward to implement—it requires only separate decaying learning rates for the two components, with no modifications to the network architecture or additional modules.

## Key Experimental Results

### Main Results: MinAtar Environments (Final IQM Return, 30 seeds)

| Algorithm | Method | Asterix | Breakout | Freeway | SpaceInvaders | Speed |
|-----------|--------|---------|----------|---------|---------------|-------|
| DQN | Baseline | 54.95 | 19.16 | 62.70 | 127.78 | 1.00x |
| DQN | **SCORER** | **54.78** | **65.69** | **63.03** | **148.71** | 0.99x |
| DDQN | Baseline | 50.77 | 36.47 | 62.22 | 116.72 | 1.00x |
| DDQN | **SCORER** | **52.59** | **64.44** | **62.68** | **146.67** | 1.00x |
| DuelingDQN | Baseline | 39.22 | 27.81 | 61.89 | 121.21 | 1.00x |
| DuelingDQN | **SCORER** | **52.28** | **60.04** | **62.27** | **139.08** | 1.01x |

### Key Findings

- On Breakout, DQN+SCORER achieves **more than a 3× improvement** in final score (19.16 → 65.69).
- SCORER enables replay-buffer-based methods to compete with advanced approaches such as PQN.
- Computational overhead is negligible (0.99–1.01×).

### Ablation Study

| Follower Objective | Performance |
|-------------------|-------------|
| Bellman Error Variance | **Best** |
| MSBE | Second |
| No Follower | Baseline |

- Variance minimization consistently outperforms direct MSBE minimization.
- Using a separate batch for the follower yields better performance.

## Highlights & Insights

1. **Game-Theoretic Novelty**: This is the first work to model the representation–control interaction in value-based RL as a Stackelberg game.
2. **Minimal Implementation**: Only learning rate schedules need to be modified; no architectural changes are required.
3. **Broad Applicability**: Effective across DQN, DDQN, Dueling DQN, R2D2, and PQN.
4. **Theoretical Grounding**: Two-timescale convergence theory guarantees convergence to first-order stationary points.

## Limitations & Future Work

- Requires tuning decay parameters for two separate learning rates.
- Theoretical analysis relies on first-order approximations, omitting the effect of implicit gradient terms.
- Validation is primarily conducted in discrete action spaces; effectiveness in continuous action settings remains unexplored.

## Related Work & Insights

- **Representation Learning in RL**: Auxiliary tasks (SPR, CURL), contrastive learning, self-supervised methods.
- **Two-Timescale Optimization**: TTSA theory (Borkar 1997; Hong et al. 2023).
- **Value Decomposition Methods**: Architectures such as Dueling DQN separate value components structurally but lack game-theoretic coupling.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces Stackelberg game formulation to representation–value function co-learning.
- **Technical Depth**: ⭐⭐⭐⭐ — Formal bilevel optimization framework with convergence analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple algorithms and environments with comprehensive ablations.
- **Value**: ⭐⭐⭐⭐⭐ — Simple to implement, plug-and-play, with no additional computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)

</div>

<!-- RELATED:END -->
