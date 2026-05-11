---
title: >-
  [Paper Note] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling
description: >-
  [NeurIPS 2025][Reinforcement Learning][Adversarial Robustness] This paper proposes CART (Conservative Adversarially Robust Decision Transformer)…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Adversarial Robustness"
  - "Decision Transformer"
  - "Stochastic Games"
  - "NashQ"
  - "Expectile Regression"
date: 2026-05-08
content_hash: f5824b945f29c51e
---

# Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2510.11877](https://arxiv.org/abs/2510.11877)
**Code**: Unavailable
**Area**: Reinforcement Learning
**Keywords**: Adversarial Robustness, Decision Transformer, Stochastic Games, NashQ, Expectile Regression

## TL;DR

This paper proposes CART (Conservative Adversarially Robust Decision Transformer), the first method to enhance the adversarial robustness of Decision Transformers in stochastic games. By modeling stage games and estimating NashQ values, CART addresses the over-optimism of ARDT under stochastic state transitions, achieving more accurate minimax value estimation and superior worst-case returns.

## Background & Motivation

The Decision Transformer (DT) reformulates reinforcement learning as a conditional sequence generation problem, generating actions conditioned on target returns. In adversarial settings, ARDT learns worst-case-aware policies by conditioning on minimax returns rather than return-to-go. However, ARDT exhibits a fundamental flaw:

**ARDT assumes deterministic state transitions.** In stochastic games, transitions are probabilistic. ARDT's minimax return computation:
$$Q_{\text{ARDT}}(s_t, a_t) = \min_{\bar{a}_t} \max_{a_{t+1}} \min_{\bar{a}_{t+1}} \cdots \hat{R}(\tau_{t:H})$$
operates directly on trajectories, ignoring the probability of reaching high-return subgames. This causes ARDT to be misled by rare but high-return trajectories, producing over-optimistic value estimates.

**Illustrative example**: In a stochastic game where action $a_0$ leads to a low-return state with 90% probability and a high-return state with 10% probability, ARDT may be misled by the rare 10% high-return trajectories, overestimating the value of $a_0$ while neglecting more robust alternatives. Experiments show that ARDT achieves a return of only 5.7 in this scenario, whereas CART achieves 8.0.

## Method

### Overall Architecture

The core idea of CART is to explicitly account for the stochasticity of state transitions in NashQ value computation. Each timestep's protagonist–adversary interaction is modeled as a stage game whose payoff function is defined as the expected value over subsequent states, thereby incorporating transition probabilities.

### Key Designs

1. **Stage Game Modeling and Payoff Function**: At each stage, the protagonist selects action $a$, and the adversary selects $\bar{a}$ after observing $a$. The payoff function accounts for transition stochasticity via an auxiliary state value function $V$:
$$\bar{Q}(s, a, \bar{a}) = \mathbb{E}_{s' \sim T(\cdot|s,a)}[r + V(s')]$$
where $V(s') = \max_{a'} Q(s', a')$. This means the payoff function takes an expectation over all possible successor states rather than estimating from a single trajectory.

2. **NashQ Conditioning Value**: As the conditioning variable $z$ for DT training, CART uses:
$$Q_{\text{CART}}(s, a) = \min_{\bar{a}} \bar{Q}(s, a, \bar{a})$$
This is the solution to the stage game, jointly accounting for adversarial robustness and transition stochasticity. The key distinction from ARDT is that ARDT performs min-max operations directly on trajectory returns, whereas CART integrates transition probabilities via an explicit $V$ function.

3. **Expectile Regression to Approximate Min/Max**: Enumerating all actions to compute $\min_{\bar{a}}$ or $\max_a$ directly from data is inefficient. CART employs expectile regression (ER) to jointly perform Q-learning and extremal approximation:

    - Learn the payoff function $\bar{Q}$: minimize the TD error $\mathcal{L}(\bar{Q}) = \mathbb{E}[\bar{Q}(s,a,\bar{a}) - V(s') - r]$
    - Estimate the NashQ value: $\mathcal{L}(Q) = \mathbb{E}[L_{\text{ER}}^{\alpha \to 0}(Q(s,a) - \bar{Q}(s,a,\bar{a}))]$ ($\alpha \to 0$ approximates $\min$)
    - Estimate the optimal state value: $\mathcal{L}(V) = \mathbb{E}[L_{\text{ER}}^{\alpha \to 1}(V(s') - Q(s',a'))]$ ($\alpha \to 1$ approximates $\max$)

### Loss & Training

Training proceeds in two phases:
1. **NashQ Value Estimation**: Alternately optimize the three losses for $\bar{Q}$, $Q$, and $V$ until convergence. Terminal state values are initialized via MSE.
2. **DT Training**: Using the converged $Q_{\text{CART}}$ as the conditioning value $z$, the model is trained with the standard DT loss:
$$\mathcal{L}_{\text{DT}}(\theta) = -\mathbb{E}[\log \pi_\theta(a_t | \tau_{0:t-1}, s_t, z)]$$
At inference, a high target return $z$ is set to guide the generation of robust policies.

## Key Experimental Results

### Main Results

Worst-case returns in synthetic stochastic games are compared (data collected by a uniformly random policy; tested against an optimal adversary):

| Method | Two-Stage Game Return | Avg. Return over 5 Variants | Over-Optimism Risk |
|--------|----------------------|-----------------------------|--------------------|
| DT | 6.0 | Low | High (no adversarial awareness) |
| ARDT | 5.7 | Moderate | High (ignores transition probabilities) |
| **CART** | **8.0** | **Highest** | **Low** |

### Ablation Study

| Game Variant | DT | ARDT | CART | Notes |
|-------------|-----|------|------|-------|
| Original Two-Stage | 6.0 | 5.7 | **8.0** | ARDT misled by 10% high-return trajectories |
| Rare Return = 100 | Low | Worse | **Stable** | Extreme rare high returns amplify ARDT's over-optimism |
| Transition Prob. 20/80 | Mid | Mid | **Optimal** | Varying transition ratios validates robustness |
| Three-Stage Game | Low | Low | **Optimal** | Longer decision horizons validate scalability |
| Mixed Variant | Low | Low | **Optimal** | Combines multiple sources of stochasticity |

### Key Findings

- CART consistently achieves the highest worst-case return with the lowest variance across all stochastic game variants.
- ARDT's robustness degrades as the target return increases, since higher target returns steer the model toward rare high-return trajectories.
- Introducing an explicit $V$ function is critical for addressing adversarial robustness under stochastic transitions.
- Expectile regression effectively approximates min/max operations, avoiding exhaustive search.

## Highlights & Insights

- The paper precisely identifies the theoretical flaw of ARDT in stochastic games: ignoring transition probabilities leads to over-optimistic value estimates.
- Nash Q-Learning concepts are elegantly integrated into the Decision Transformer framework via trajectory relabeling.
- Expectile regression as a differentiable approximation of min/max is a general technique that enables Q-learning and extremal optimization to be performed jointly and efficiently.

## Limitations & Future Work

- Experiments are conducted only on synthetic, short-horizon stochastic games; validation in more complex multi-agent competitive environments such as Poker is absent.
- The offline setting constrains data coverage — if the behavioral policy fails to sufficiently explore critical states, NashQ estimates may be inaccurate.
- The choice of the $\alpha$ parameter in expectile regression affects the quality of min/max approximation, but the paper does not provide a detailed discussion of tuning strategies.
- Only two-player zero-sum games are considered; extending to multi-player or general-sum games requires additional theoretical support.

## Related Work & Insights

- **Decision Transformer (DT)**: Formulates RL as conditional sequence generation; this work builds upon the DT framework.
- **ARDT**: Introduces adversarial robustness into DT but is restricted to deterministic transitions; CART is a direct extension of ARDT.
- **Nash Q-Learning**: A classical method for solving stochastic games; CART adopts its stage game formulation and NashQ value concept.
- **IQL (Implicit Q-Learning)**: Uses expectile regression for offline Q-learning; CART's value function learning is directly inspired by IQL.
- **Insight**: In adversarial RL, explicitly modeling environmental stochasticity is more important than relying solely on worst-case returns.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First work to address adversarial robustness of DT in stochastic games
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to synthetic game experiments; large-scale or real-world validation is lacking
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, motivation is well-grounded, and mathematical derivations in the method section are rigorous
- **Value**: ⭐⭐⭐⭐ Provides a theoretical foundation and practical approach for applying DT in multi-agent stochastic environments

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](../../ICLR2026/reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)

</div>

<!-- RELATED:END -->
