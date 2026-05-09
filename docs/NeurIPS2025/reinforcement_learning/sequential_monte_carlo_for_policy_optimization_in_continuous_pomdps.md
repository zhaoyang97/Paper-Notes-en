---
title: >-
  [Paper Note] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs
description: >-
  [NeurIPS 2025][Reinforcement Learning][POMDP] This paper proposes a nested Sequential Monte Carlo (SMC) algorithm grounded in non-Markovian Feynman-Kac models for policy optimization in continuous POMDPs, naturally capturing the value of information gathering without hand-crafted heuristics.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - POMDP
  - sequential Monte Carlo
  - policy gradient
  - partial observability
  - Feynman-Kac
date: 2026-05-08
content_hash: 3184af12226b35bc
---

# Sequential Monte Carlo for Policy Optimization in Continuous POMDPs

**Conference**: NeurIPS 2025
**arXiv**: [2505.16732](https://arxiv.org/abs/2505.16732)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: POMDP, sequential Monte Carlo, policy gradient, partial observability, Feynman-Kac

## TL;DR
This paper proposes a nested Sequential Monte Carlo (SMC) algorithm grounded in non-Markovian Feynman-Kac models for policy optimization in continuous POMDPs, naturally capturing the value of information gathering without hand-crafted heuristics.

## Background & Motivation
**Background**: Optimal decision-making in partially observable Markov decision processes (POMDPs) requires agents to balance uncertainty reduction (exploration) with immediate goal pursuit (exploitation).

**Limitations of Prior Work**: Existing policy optimization methods for continuous POMDPs either rely on suboptimal approximations (e.g., belief-point methods) or employ hand-crafted reward shaping to encourage exploration.

**Key Challenge**: The belief space of a POMDP is infinite-dimensional, making direct optimization computationally intractable; yet simplified approximations fail to preserve the value of information gathering.

**Key Insight**: The paper reformulates policy learning as probabilistic inference, naturally encoding information value through a Feynman-Kac model.

## Method

### Overall Architecture
POMDP policy optimization is mapped to probabilistic inference within a non-Markovian Feynman-Kac model: the optimal trajectory distribution is defined by the POMDP reward structure, and policy gradients under this distribution are estimated via nested SMC.

### Key Designs
1. **Feynman-Kac Model Construction**

    - Function: Encodes the POMDP value function as a Feynman-Kac path integral.
    - Mechanism: $\mathcal{Z}_\theta = \mathbb{E}_{\pi_\theta}\left[\prod_{t=0}^T G_t(s_{0:t}, o_{0:t})\right]$
    - Design Motivation: The FK model naturally encodes the value of information gathering through expected future observations.

2. **Nested SMC Algorithm**

    - Outer SMC: Samples trajectories in history space.
    - Inner SMC: Performs belief updates conditioned on a given history.
    - Mechanism: Outer particles represent distinct behavioral trajectories; inner particles track belief states.
    - Design Motivation: The nested structure decouples policy evaluation from belief maintenance.

3. **History-Dependent Policy Gradient**

    - Function: Computes gradients with respect to policy parameters.
    - Mechanism: Estimates the expectation of $\nabla_\theta \log \pi_\theta$ over SMC-sampled trajectories.
    - Supports non-Markovian policies that condition on the full observation history.

### Loss & Training
- Annealing strategy with progressively increasing particle counts.
- RNN/Transformer-parameterized policy networks to handle observation history sequences.
- A natural gradient variant is provided to improve convergence.

## Key Experimental Results

### Main Results: Continuous POMDP Benchmarks (Cumulative Reward ↑)

| Environment | QMDP | POMCP | Belief-PPO | RNN-PPO | **FK-SMC** |
|-------------|------|-------|------------|---------|-----------|
| Tiger | -12.3 | -5.7 | -8.2 | -6.1 | **-3.4** |
| LightDark | 45.2 | 78.3 | 62.1 | 71.5 | **85.7** |
| Navigation | 32.1 | 58.6 | 48.3 | 55.2 | **67.3** |
| Active Sensing | 18.7 | 42.3 | 31.5 | 38.9 | **52.1** |

### Ablation Study: Effect of SMC Particle Count

| Particle Count (outer/inner) | LightDark Reward | Compute Time (s/iter) |
|------------------------------|-----------------|----------------------|
| 16/16 | 72.3 | 0.8 |
| 32/32 | 79.5 | 2.1 |
| 64/64 | 83.1 | 5.6 |
| **128/128** | **85.7** | **14.2** |
| 256/256 | 86.1 | 38.5 |

### Policy Parameterization Ablation

| Policy Parameterization | LightDark Reward | Navigation Reward |
|------------------------|-----------------|------------------|
| Linear Policy | 61.2 | 38.7 |
| RNN Policy | 79.8 | 58.3 |
| GRU Policy | 82.4 | 62.1 |
| **Transformer Policy** | **85.7** | **67.3** |

### Key Findings
- FK-SMC substantially outperforms all baselines in environments requiring active information gathering.
- The largest improvement is observed in the Tiger environment, which demands the most delicate exploration–exploitation balance.
- 128 particles achieves the best trade-off between performance and computational cost.
- The advantage of non-Markovian policies (Transformer) becomes more pronounced in long-horizon tasks.

## Highlights & Insights
- **Theoretical Elegance**: The probabilistic inference framework naturally encodes information value via the FK model, offering a principled solution to the RL problem.
- **No Manual Heuristics**: Eliminates the need for hand-designed exploration bonuses such as curiosity rewards or information-gain rewards.
- The nested SMC algorithm admits a provably convergent theoretical guarantee.
- This work is the first to introduce FK path integrals into POMDP policy optimization.

## Limitations & Future Work
- Computational cost of nested SMC scales quadratically with particle count.
- Extension to high-dimensional observation spaces (e.g., images) requires integration with deep generative belief models.
- Comparison with Transformer-based memory approaches remains insufficient.
- The resampling step in continuous action spaces requires special treatment.
- Particle degeneracy in SMC becomes problematic for long horizons (> 100 steps).

## Related Work & Insights
- POMCP (Silver & Veness 2010): Monte Carlo tree search for POMDPs.
- SMC² (Chopin et al. 2013): Nested SMC framework.
- Control as Inference (Levine 2018): Probabilistic treatment of RL.
- Dreamer series (Hafner et al. 2020): World model-based approaches.
- Insight: The FK model framework is potentially extensible to risk-sensitive RL and multi-agent Dec-POMDPs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The FK + nested SMC paradigm for policy optimization is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-environment evaluation with particle count and policy ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and well-organized.
- Value: ⭐⭐⭐⭐ Advances both the theory and practice of POMDP policy optimization.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)
- [\[ICLR 2026\] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](../../ICLR2026/reinforcement_learning/policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)

<!-- RELATED:END -->
