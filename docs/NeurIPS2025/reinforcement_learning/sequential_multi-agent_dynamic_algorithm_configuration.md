---
title: >-
  [Paper Note] Sequential Multi-Agent Dynamic Algorithm Configuration
description: >-
  [NeurIPS 2025][Reinforcement Learning][Dynamic Algorithm Configuration] This paper proposes Seq-MADAC, a framework that models multi-hyperparameter dynamic configuration as a contextual sequential multi-agent MDP. By exploiting inherent inter-parameter dependencies via a Sequential Advantage Decomposition Network (SADN), it outperforms existing MARL methods on multi-objective optimization algorithm configuration.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Dynamic Algorithm Configuration
  - Multi-Agent Reinforcement Learning
  - Sequential Dependency
  - Advantage Decomposition
  - Hyperparameter Optimization
date: 2026-05-08
content_hash: e8034efbb876d71d
---

# Sequential Multi-Agent Dynamic Algorithm Configuration

**Conference**: NeurIPS 2025
**arXiv**: [2510.23535](https://arxiv.org/abs/2510.23535)
**Code**: [https://github.com/lamda-bbo/seq-madac](https://github.com/lamda-bbo/seq-madac)
**Area**: Reinforcement Learning
**Keywords**: Dynamic Algorithm Configuration, Multi-Agent Reinforcement Learning, Sequential Dependency, Advantage Decomposition, Hyperparameter Optimization

## TL;DR
This paper proposes Seq-MADAC, a framework that models multi-hyperparameter dynamic configuration as a contextual sequential multi-agent MDP. By exploiting inherent inter-parameter dependencies via a Sequential Advantage Decomposition Network (SADN), it outperforms existing MARL methods on multi-objective optimization algorithm configuration.

## Background & Motivation

**Background**: Dynamic Algorithm Configuration (DAC) applies RL to dynamically adjust hyperparameters during algorithm execution, demonstrating superiority over static configuration. Multi-agent DAC (MADAC) further handles multiple heterogeneous hyperparameters.

**Limitations of Prior Work**: Many algorithms exhibit inherent sequential dependencies among hyperparameters (e.g., selecting an operator type before setting its parameters), yet MADAC assumes parameter independence. CANDID DAC attempts to address this via SAQL but suffers from credit assignment and convergence issues.

**Key Challenge**: Inter-parameter dependencies introduce large numbers of illegal combinations in the joint action space; ignoring these dependencies leads to inefficient exploration.

**Core Idea**: The problem is modeled as a sequential MMDP in which each subsequent agent observes the actions of preceding agents, enabling differentiated credit assignment through advantage decomposition.

## Method

### Overall Architecture
Seq-MADAC models parameter configuration as a sequential decision-making process. At each time step, agents make decisions in a predefined order, with the $i$-th agent observing the actions of the preceding $i-1$ agents.

### Key Designs

1. **Sequential Advantage Decomposition Network (SADN)**:

    - The global advantage function is decomposed into a sum of individual agent advantages: $A(s,\boldsymbol{a}) = \sum_{i=1}^n A_i(s, \boldsymbol{a}_{1:i-1}, a_i)$
    - Each agent maintains an independent network modeling $A_i$, updated implicitly via backpropagation through the global advantage.
    - Satisfies the IGM principle: $\arg\max_{\boldsymbol{a}} Q(s,\boldsymbol{a})$ is equivalent to sequentially selecting the optimal action along each dimension.

2. **Independent Update Mechanism**:

    - The global advantage is updated via one-step TD error: $A(s,\boldsymbol{a}) \leftarrow A(s,\boldsymbol{a}) + \alpha[r + \gamma V(s') - V(s) - A(s,\boldsymbol{a})]$
    - Each agent network is updated independently, avoiding the chain error accumulation of ACE and the interference issues of SAQL.

3. **Contextual Sequential MMDP Formulation**: The distribution of problem instances is incorporated into the MDP definition, enabling generalization across problem instances.

## Key Experimental Results

### Main Results — MOEA/D Multi-Objective Optimization (IGD↑, higher is better)

| Problem | Dimension | SADN | ACE | SAQL | VDN | MAPPO |
|---------|-----------|------|-----|------|-----|-------|
| DTLZ2 | 6 | **4.593e-2** | 3.809e-2 | 3.851e-2 | 3.950e-2 | 3.906e-2 |
| WFG4 | 6 | **5.729e-2** | 4.537e-2 | 4.626e-2 | 4.817e-2 | 4.639e-2 |
| WFG6 | 9 | **6.641e-2** | 4.884e-2 | 5.111e-2 | 5.558e-2 | 6.194e-2 |

### Ablation Study — Seq-Sigmoid-Robust

| Setting | SADN | ACE | SAQL |
|---------|------|-----|------|
| 5-dim-1-random | Most stable | Performance degradation | Low performance |
| 10-dim-2-random | Most stable | Severe degradation | Low performance |

### Key Findings
- SADN achieves a clean sweep (0/9/0) in significance tests on the training set.
- ACE degrades in multi-instance environments due to chain update interruptions.
- Experiments comparing correct versus reversed ordering confirm the importance of modeling parameter dependencies.

## Highlights & Insights
- The sequential extension of the IGM principle is theoretically grounded, decomposing joint action optimization into sequential greedy selection.
- Advantage decomposition is more flexible than Q-value decomposition, overcoming the additive/monotonicity assumptions that limit VDN/QMIX.

## Limitations & Future Work
- The parameter ordering must be specified in advance; automatically learning the ordering would be more valuable.
- Experiments are concentrated on MOEA/D; validation on a broader range of algorithms is insufficient.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First work to model sequential parameter dependencies in DAC.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both synthetic and real-world tasks with comprehensive comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and well-presented.
- **Value**: ⭐⭐⭐⭐ Practically useful with a solid theoretical foundation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
