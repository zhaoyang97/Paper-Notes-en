---
title: >-
  [Paper Note] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction
description: >-
  [ICLR 2026][AI Safety][Distributionally Robust] This paper presents the first study of online learning in Distributionally Robust Markov Games (DRMGs), proposing the MORNAVI algorithm. Without relying on a simulator or offline data, MORNAVI efficiently learns optimal robust policies through online interaction, and provides the first provable regret bounds under both TV-divergence and KL-divergence uncertainty sets.
tags:
  - ICLR 2026
  - AI Safety
  - Distributionally Robust
  - Multi-Agent RL
  - Markov Games
  - online learning
  - Regret Bounds
date: 2026-05-08
content_hash: 5daf814cd5717219
---

# Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction

**Conference**: ICLR 2026
**arXiv**: [2508.02948](https://arxiv.org/abs/2508.02948)
**Code**: None
**Area**: AI Safety / Multi-Agent Reinforcement Learning
**Keywords**: Distributionally Robust, Multi-Agent RL, Markov Games, online learning, Regret Bounds

## TL;DR

This paper presents the first study of online learning in Distributionally Robust Markov Games (DRMGs), proposing the MORNAVI algorithm. Without relying on a simulator or offline data, MORNAVI efficiently learns optimal robust policies through online interaction, and provides the first provable regret bounds under both TV-divergence and KL-divergence uncertainty sets.

## Background & Motivation

Multi-agent systems face a fundamental challenge in real-world deployment: **model mismatch between training and deployment environments**. Such mismatch may stem from environmental noise, transition uncertainty, or even adversarial attacks. Multi-agent policies that perform well during training may fail catastrophically when confronted with these uncertainties.

Distributionally Robust Markov Games (DRMGs) address this by optimizing **worst-case performance**—seeking policies that achieve a robust Nash equilibrium for all agents over a defined set of environmental uncertainties.

However, existing approaches to DRMGs suffer from critical limitations:

- **Simulator dependence**: Many methods assume repeated access to an environment model (i.e., a generative simulator), which is often unavailable in real systems.
- **Offline dataset dependence**: Other methods require large-scale offline experience collected in advance, which may not exist in new environments.
- **Lack of online learning guarantees**: No existing method provides theoretical guarantees for the setting where agents interact directly with the environment and learn robust policies in real time.

In summary, online learning for DRMGs—where agents learn robust policies purely through direct environment interaction without any prior data—remains a completely unexplored problem.

## Method

### Overall Architecture

MORNAVI (Multiplayer Optimistic Robust Nash Value Iteration) is an algorithmic framework for online DRMGs. Its core idea is to layer **optimistic exploration** and **robust evaluation** on top of standard value iteration, enabling agents to efficiently balance exploration (information gathering) and exploitation (executing robust policies).

### Key Designs

1. **Formalization of Online DRMGs**:

    - **Function**: Formalizes the multi-agent robust learning problem as an online interaction framework.
    - **Mechanism**: Agents interact directly with the real environment each episode, observing state transitions and collecting rewards. The objective is to minimize cumulative regret—the performance gap between the online policy and the optimal robust policy.
    - **Design Motivation**: Online learning is the setting closest to real deployment—agents have neither the hindsight of a simulator nor the prior knowledge embedded in offline data, and must learn by doing.

2. **Optimistic Robust Nash Value Iteration**:

    - **Function**: Introduces the optimism principle into classical Nash Value Iteration to handle uncertainty.
    - **Mechanism**:
        - **Transition model estimation**: Agents maintain empirical estimates of environmental transition probabilities and construct confidence sets to quantify estimation uncertainty.
        - **Optimistic exploration**: The most optimistic model within the confidence set is selected for policy optimization, encouraging exploration of insufficiently visited state-action pairs.
        - **Robust evaluation**: For each candidate model, the worst-case value function is computed over the uncertainty set (TV or KL divergence ball), ensuring policy robustness.
    - **Design Motivation**: Optimism is a classical principle for balancing exploration and exploitation in online learning. Combining it with distributionally robust optimization simultaneously guarantees efficient exploration (low regret) and robustness (worst-case protection).

3. **Unified Treatment of TV and KL Divergence Uncertainty Sets**:

    - **Function**: Provides a unified algorithmic framework and theoretical guarantees for two widely used uncertainty measures.
    - **Mechanism**:
        - **Total Variation (TV) divergence**: Measures the maximum probability difference between two distributions; suitable for modeling bounded model shifts.
        - **Kullback-Leibler (KL) divergence**: Measures distributional distance in the information-theoretic sense; suitable for modeling multiplicative noise.
        - The robust Bellman operators under the two settings have different computational structures; MORNAVI designs efficient inner-loop optimizations for each.
    - **Design Motivation**: Different application scenarios call for different uncertainty measures. TV divergence is easier to optimize but may be overly conservative; KL divergence is more flexible but requires finer analysis. Supporting both increases the generality of the framework.

### Loss & Training

- **Regret minimization objective**: Minimize $\text{Regret}(K) = \sum_{k=1}^{K} [V^* - V^{\pi_k}]$, where $V^*$ is the optimal robust Nash equilibrium value and $V^{\pi_k}$ is the value of the policy at episode $k$.
- **Confidence set construction**: Confidence intervals for transition probabilities are constructed based on empirical visit counts and Hoeffding-type inequalities.
- **Inner robust optimization**: Under the TV uncertainty set, the inner optimization admits a closed-form solution; under the KL uncertainty set, a dualization approach is employed for efficient computation.

## Key Experimental Results

### Theoretical Results

The core contribution of this paper is theoretical rather than empirical.

| Uncertainty Set | Regret Bound | Notes |
|----------------|--------------|-------|
| TV divergence | $\tilde{O}(\text{poly}(S,A,H) \cdot \sqrt{K})$ | First regret bound for online multi-agent robust learning |
| KL divergence | $\tilde{O}(\text{poly}(S,A,H) \cdot \sqrt{K})$ | First result under KL uncertainty sets |

Here $S$ is the state space size, $A$ is the action space size, $H$ is the episode length, and $K$ is the total number of episodes. The $\sqrt{K}$ regret growth rate implies that average regret tends to zero, i.e., the algorithm eventually converges to the optimal robust policy.

### Core Theoretical Contributions

| Result | Significance |
|--------|-------------|
| First provable regret bounds for online DRMGs | Opens a new research direction |
| Efficient algorithm under TV divergence | Robust Bellman operator admits a closed-form solution |
| Efficient algorithm under KL divergence | More general uncertainty sets handled via dualization |
| Robust equilibrium in multiplayer games | Not restricted to two-player zero-sum; applies to general-sum games |

### Key Findings

1. **Online robust learning is feasible**: Without a simulator or offline data, optimal robust policies can be learned through online interaction alone with sublinear regret.
2. **The optimism principle is effective for robust optimization**: Although seemingly contradictory (optimism vs. robustness/pessimism), optimism operates along the exploration dimension while robustness operates along the model uncertainty dimension—the two are orthogonal and compatible.
3. **Distinct properties of TV and KL uncertainty sets**: The TV divergence setting has better problem structure (closed-form solutions), while the KL divergence setting requires more delicate analysis but offers more flexible modeling.

## Highlights & Insights

- **Pioneer problem formulation**: The first work to formally define and solve the online learning problem for DRMGs, filling a theoretical gap.
- **Theoretical rigor**: Complete regret bound proofs are provided, including upper bound analysis and key lemmas.
- **Practical relevance**: Real-world multi-agent deployment (drone formations, autonomous vehicle fleets, robotic cooperation, etc.) is inherently an online learning scenario; this work provides the theoretical foundation.
- **Unified framework**: Simultaneously addressing both TV and KL divergence—the two dominant uncertainty measures—enhances generality.

## Limitations & Future Work

1. **Tabular setting**: MORNAVI is built on tabular MDPs (finite state-action spaces); extension to continuous or high-dimensional spaces via function approximation is an important future direction.
2. **Computational complexity**: Each step requires solving an inner robust optimization over the uncertainty set; computational efficiency for large-scale problems needs improvement.
3. **Tightness of regret bounds**: The current $\tilde{O}(\sqrt{K})$ regret bound may not be minimax optimal in polynomial factors; establishing matching lower bounds remains an open problem.
4. **Finite-player assumption**: The theoretical analysis assumes a finite number of symmetric or asymmetric players; very large-scale multi-agent scenarios (e.g., mean-field games) require additional consideration.
5. **Uncertainty set selection**: The radii of the TV and KL divergence balls (i.e., the degree of uncertainty) are pre-specified hyperparameters; principled adaptive selection lacks theoretical guidance.
6. **Absence of empirical validation**: As a theory-focused work, experimental evaluation is lacking; even numerical verification in tabular environments would strengthen the paper's persuasiveness.

## Related Work & Insights

- **Relation to single-agent robust RL**: Online learning for robust MDPs has been preliminarily studied (e.g., robust UCRL); MORNAVI extends this to the multi-agent game setting, where complexity increases substantially due to the need to handle both policy coupling across agents and robustness simultaneously.
- **Relation to non-robust online MARL**: Online learning for standard Markov Games (e.g., Nash-VI) has a rich theoretical literature; this paper augments that foundation with a distributionally robust layer, resulting in a bilevel optimization structure.
- **Relation to distributionally robust optimization (DRO)**: DRO has been extensively studied in supervised learning; combining it with multi-agent online learning represents a natural yet novel intersection.
- **Implications for safety-critical applications**: In safety-critical multi-agent systems (e.g., interactions among autonomous vehicles), robustness is a hard requirement. The theoretical framework developed here provides a foundation for designing deployable, safe MARL algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Risk-Sensitive Agent Compositions](risk-sensitive_agent_compositions.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](../../CVPR2026/ai_safety/tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[ICLR 2026\] Action-Free Offline-to-Online RL via Discretised State Policies](action-free_offline-to-online_rl_via_discretised_state_policies.md)

<!-- RELATED:END -->
