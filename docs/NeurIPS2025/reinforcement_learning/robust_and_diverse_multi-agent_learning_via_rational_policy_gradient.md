---
title: >-
  [Paper Note] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient
description: >-
  [NeurIPS 2025][Reinforcement Learning][Multi-Agent Reinforcement Learning] This paper proposes the Rationality-Preserving Optimization (RPO) framework and the Rational Policy Gradient (RPG) algorithm. By introducing manipulator agents and opponent shaping techniques, RPG eliminates suicidal behavior induced by adversarial optimization in both cooperative and general-sum games, while simultaneously achieving policy robustness and diversity.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Multi-Agent Reinforcement Learning
  - Adversarial Optimization
  - Suicidal Behavior
  - Policy Gradient
  - Cooperative Games
date: 2026-05-08
content_hash: dfa883b73ee3070a
---

# Robust and Diverse Multi-Agent Learning via Rational Policy Gradient

**Conference**: NeurIPS 2025
**arXiv**: [2511.09535](https://arxiv.org/abs/2511.09535)
**Code**: [GitHub](https://github.com/niklaslauffer/rational-policy-gradient)
**Area**: Reinforcement Learning
**Keywords**: Multi-Agent Reinforcement Learning, Adversarial Optimization, Suicidal Behavior, Policy Gradient, Cooperative Games

## TL;DR

This paper proposes the Rationality-Preserving Optimization (RPO) framework and the Rational Policy Gradient (RPG) algorithm. By introducing manipulator agents and opponent shaping techniques, RPG eliminates suicidal behavior induced by adversarial optimization in both cooperative and general-sum games, while simultaneously achieving policy robustness and diversity.

## Background & Motivation

Training robust behavior in multi-agent reinforcement learning (MARL) is a long-standing challenge: agents must be able to adapt to the wide range of strategies that other agents may adopt.

**Success of adversarial optimization in zero-sum games**: In zero-sum settings, self-play naturally encourages agents to continuously identify and exploit weaknesses in opponents' strategies. In cooperative or general-sum games, however, self-play actively avoids exposing teammates' weaknesses (as doing so would harm shared rewards), resulting in brittle policies.

**The suicidal behavior problem**: Directly applying adversarial optimization in cooperative settings—i.e., incentivizing agents to minimize others' rewards—allows an adversary to trivially achieve its objective by refusing cooperation or actively sabotaging the team. For instance, in a simple cooperative matrix game, an adversary can simply select action $E$ (yielding a reward of $-1$ regardless of the victim's action), which is entirely unreasonable yet satisfies the adversarial objective.

**Failure of existing methods**: Methods such as CoMeDi attempt to prevent suicidal behavior by mixing observation distributions from self-play and cross-play, but still fail in certain game structures—suicidal behavior persists even in games with a single observation, indicating that the root cause lies beyond observation distributions.

The core insight is that **suicidal behavior is fundamentally irrational**—agents make choices that contradict their own interests. The solution should therefore directly constrain agents to remain rational: a policy must be a best response to at least one plausible partner strategy.

## Method

### Overall Architecture

The core architecture of RPG introduces a *manipulator* agent: for each original *base* agent, a corresponding manipulator is created. The base agent is trained exclusively in the manipulator's environment to maximize its own reward (ensuring rationality), while the manipulator employs opponent shaping to indirectly guide the base agent's learning toward the adversarial objective. After training, the manipulator is discarded and only the trained base agent is retained.

### Key Designs

1. **Rationality-Preserving Optimization (RPO) Formalization**:
   For each agent $i$, given an adversarial objective $O_i(\pi_1, \dots, \pi_m)$, RPO requires:
   $$\max_{\pi_i} O_i(\pi_1, \dots, \pi_m) \quad \text{s.t.} \quad \exists \pi'_{-i} \in \Pi_{-i} \text{ s.t. } \pi_i \in \text{BR}(\pi'_{-i})$$
   That is, the adversarial objective is optimized subject to the constraint that the policy must be a best response to at least one joint strategy of the other agents. This guarantees that the agent's behavior is "reasonable"—there exists some teammate strategy under which the agent's behavior is optimal.

   Key property: In zero-sum games, the RPO constraint is automatically satisfied (since minimizing the opponent's reward equals maximizing one's own), making RPO a strict generalization of adversarial training to zero-sum settings.

2. **Manipulator and Opponent Shaping Mechanism**:
   The objective is decomposed into two parts:
   - **Base agent** objective: $\max_{\pi_i} U(\pi_i, \pi^M_{-i})$ (best response to the manipulator, ensuring rationality)
   - **Manipulator** objective: $\max_{\pi^M_{-i}} O_i(\pi_1, \dots, \pi_m)$ (optimizing the adversarial objective by influencing the base agent's learning)

   The manipulator's gradient update involves **higher-order gradients**—gradients taken through the base agent's parameter update step:
   $$\theta^M_{-i} \leftarrow \theta^M_{-i} + \nabla_{\theta^M_{-i}} O_i(\theta'_1, \dots, \theta'_m)$$
   where $\theta'_i$ denotes the base agent's parameters after the update.

3. **Partner-Play Regularization**:
   To prevent distribution shift when the base agent (trained solely in the manipulator's environment) is evaluated against other base agents, a small number of cross-play trajectories between base agents are incorporated into training, weighted by a small coefficient $\epsilon$ as an auxiliary loss.

4. **Loaded DiCE Loss**:
   Standard RL surrogate losses support only first-order gradients, whereas the manipulator requires higher-order gradients. The Loaded DiCE surrogate loss (based on the magic box operator) is employed to support unbiased higher-order gradient computation via automatic differentiation.

5. **Five RPG Algorithm Variants**:
   - **AP-RPG**: Finds rational adversarial examples among pre-trained policies
   - **AT-RPG**: Robustifies agents during adversarial training
   - **PAIRED-RPG**: Robustification via regret minimization
   - **PAIRED-A-RPG**: Adversarial attack that maximizes the victim's regret
   - **AD-RPG**: Learns genuinely diverse policy populations and generates automatic curricula

### Loss & Training

Manipulator loss (Loaded DiCE):
$$\mathcal{L}^{O_i} = \sum_{e \in E} w_e \sum_t \gamma^t \square(\{a^{t' \leq t}_{j \in \{B,M\}}\}) r^t_e$$

Base agent loss (standard RL surrogate loss + partner-play regularization):
$$\mathcal{L} = \sum_{e \in E} w_e \sum_t \gamma^t \log(\pi(a_t|s_t)) r^t_e$$

At each iteration, the algorithm first performs $N$ lookahead steps to update the base agent (best response to the manipulator), then samples trajectories among the updated base agents, and finally updates the manipulator parameters.

## Key Experimental Results

### Main Results: Diversity (AD-RPG vs. Baselines, Overcooked Cramped Room)

| Algorithm | Self-Play Reward | Cross-Play Reward | Suicidal Behavior? |
|-----------|:---:|:---:|:---:|
| CoMeDi | 220 | 2 | Yes (blocks teammate by standing in front of the dish dispenser) |
| AD (vanilla adversarial diversity) | 240 | 1.25 | Yes |
| **AD-RPG** | **240** | **240** | **No** |

AD-RPG achieves the same high self-play reward as the baselines while maintaining high cross-play reward, demonstrating that the cramped room layout admits virtually no genuine diversity—low cross-play scores can only be achieved through suicidal behavior.

### Robustness Evaluation (Victim Performance under Adversarial Attack, STORM Environment)

| Victim Training Algorithm | Training Reward | AP Attack | PAIRED-A-RPG Attack | AP-RPG Attack |
|--------------------------|:---:|:---:|:---:|:---:|
| PAIRED | 0.13 | 0.0 | 0.50 | 0.42 |
| PAIRED-RPG | 0.93 | 0.0 | 0.84 | 0.85 |
| AT (fails due to suicidal behavior) | 0.0 | 0.0 | 0.0 | 0.0 |
| AT-RPG | 0.65 | 0.0 | 0.72 | 0.88 |
| AD-RPG | 0.98 | 0.0 | 0.25 | 0.96 |
| Self-play | 0.98 | 0.0 | 0.16 | 0.96 |

Policies trained with RPG variants maintain high rewards under rational adversarial attacks, whereas non-RPG methods fail entirely due to suicidal behavior during training.

### Ablation Study / Cross-Environment Generalization

| Environment | SP (Low Entropy) | SP (High Entropy) | AD | AD-RPG |
|-------------|:---:|:---:|:---:|:---:|
| Forced Coordination | Low cross-play | Low cross-play | Suicidal | **High cross-play** |
| Counter Circuit | Low | Low | Suicidal | **Substantially higher** |
| Hanabi (3) | Medium | Moderately high | Suicidal | **Highest** |
| Hanabi (4) | Medium | Moderately high | Suicidal | **Highest** |

AD-RPG eliminates suicidal behavior and maintains high robustness across all environments.

### Key Findings

- AD-RPG fully eliminates suicidal behavior in existing adversarial diversity algorithms—an open problem that had remained unresolved in the field.
- Adversarial examples found by RPG are "rational": e.g., in Overcooked, the method discovers that the victim assumes agents navigate clockwise while the adversary moves counterclockwise—a reasonable but incompatible assumption.
- Policies trained with PAIRED-RPG and AT-RPG achieve the best performance under adversarial attacks.
- Partner-play regularization is critical for preventing distribution shift.

## Highlights & Insights

- **Elegant formalization of rationality constraints**: Defining rationality as "there exists a teammate strategy under which my policy is a best response" is both concise and powerful, directly addressing the root cause of suicidal behavior.
- **Clever design of the manipulator architecture**: The base agent only needs to focus on rationality (maximizing its own reward), while the full complexity of the adversarial objective is delegated to the manipulator.
- **Strict generalization of zero-sum training**: In zero-sum games, RPO automatically reduces to standard adversarial training, establishing the generality of the approach.
- **One framework unifying five algorithms**: AP-RPG, AT-RPG, PAIRED-RPG, PAIRED-A-RPG, and AD-RPG are all instantiations of RPG.

## Limitations & Future Work

- Higher-order gradient computation introduces additional overhead (AD-RPG is approximately 6× slower than AD), requiring large batch sizes to stabilize estimation.
- No formal convergence guarantees are provided—it remains unclear under what conditions RPG is guaranteed to find an RPO solution.
- Gradient estimation variance for both the manipulator and base agent may be large in high-dimensional settings.
- Validation is currently limited to relatively simple environments (matrix games, Overcooked, STORM, simplified Hanabi); testing in more complex domains remains future work.

## Related Work & Insights

- **Opponent shaping (Foerster et al.)**: RPG is the first work to apply opponent shaping to adversarial training.
- **CoMeDi (Sarkar et al.)**: Prevents suicidal behavior by mixing observation distributions, but is shown to be insufficient both theoretically and empirically.
- **PAIRED (Dennis et al.)**: RPG extends this framework from environment design to co-player design.
- Insight: Combining gradient-free opponent shaping methods (e.g., M-FOS) with RPG may reduce computational overhead.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The rationality-preserving framework fundamentally resolves the core challenge of suicidal behavior, with significant theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across four environments, five algorithm variants, and three dimensions (robustness, diversity, adversarial examples).
- **Writing Quality**: ⭐⭐⭐⭐⭐ The matrix game example is intuitive, the theoretical exposition is concise, and the visual design is elegant.
- **Value**: ⭐⭐⭐⭐⭐ Resolves a core open problem in multi-agent adversarial optimization that has persisted for years, with far-reaching implications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)

<!-- RELATED:END -->
