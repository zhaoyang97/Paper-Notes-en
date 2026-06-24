---
title: >-
  [Paper Note] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination
description: >-
  [ICML 2025][Multi-Agent][Zero-shot Coordination] > Proposes the Cross-Environment Cooperation (CEC) paradigm, which trains agents via self-play across a large number of procedurally generated, diverse environments (rather than increasing partner diversity). This enables agents to learn general cooperative norms, achieving zero-shot coordination with unseen partners in unseen environments.
tags:
  - "ICML 2025"
  - "Multi-Agent"
  - "Zero-shot Coordination"
  - "Cross-Environment Cooperation"
  - "Procedural Generation"
  - "Multi-agent RL"
  - "Human-AI Collaboration"
date: 2026-05-08
content_hash: e54fde6be0585a35
---

# Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination

**Conference**: ICML 2025  
**arXiv**: [2504.12714](https://arxiv.org/abs/2504.12714)  
**Code**: [https://kjha02.github.io/publication/cross-env-coop](https://kjha02.github.io/publication/cross-env-coop)  
**Area**: Reinforcement Learning / Multi-Agent Collaboration / Zero-Shot Coordination  
**Keywords**: Zero-shot Coordination, Cross-Environment Cooperation, Procedural Generation, Multi-agent RL, Human-AI Collaboration  

## TL;DR

> Proposes the Cross-Environment Cooperation (CEC) paradigm, which trains agents via self-play across a large number of procedurally generated, diverse environments (rather than increasing partner diversity). This enables agents to learn general cooperative norms, achieving zero-shot coordination with unseen partners in unseen environments.

## Background & Motivation

### Problem Definition

Zero-Shot Coordination (ZSC) is a critical capability for building human-compatible AI—agents must cooperate on the fly with novel partners on novel tasks without prior coordination. Humans naturally excel at this ad-hoc cooperation: a chef who learns to cook a dish at home with parents can easily complete the same (or even more complex) cooking tasks in a different kitchen with a spouse.

### Limitations of Prior Work

**Self-Play (SP)**: In cooperative games, once an equilibrium strategy is found, neither player has an incentive to explore other equilibria. This leads to fragile policies that fail to adapt to new partners with different strategies.

**Population-Based Training (PBT)**: Such as Fictitious Co-Play (FCP), which increases "partner diversity" by maintaining a diverse pool of partners. Although it adapts to different partners in a single environment, it **completely fails to generalize to new environments, even with minor changes**. Each environmental change requires retraining the entire population, which is computationally expensive and unscalable.

**E3T (Efficient End-to-End Training)**: Reaches SOTA in single-task ZSC by adding noise to partner policies and training an auxiliary network to predict others' behaviors, but is similarly restricted to the specific environment seen during training.

### Key Insight

The authors propose a key hypothesis: **Environment Diversity > Partner Diversity**. Instead of training diverse partner policies on a single task, agents are trained via self-play with the same partner (a copy of themselves) across a large number of diverse environments. Diverse environments force agents to learn high-level task structures ("cook onions and serve meals") rather than low-level action sequences ("move left three steps and interact"), naturally yielding generalization to both new partners and new environments.

## Method

### Overall Architecture

The core process of CEC (Cross-Environment Cooperation) is as follows:

1. **Procedural Environment Generation**: Construct a generator capable of producing a massive number of solvable coordination tasks.
2. **Cross-Environment Self-Play Training**: Perform self-play using IPPO on the sampled diverse tasks.
3. **(Optional) Single-Task Fine-Tuning**: Fine-tune with a low learning rate on specific target environments (CEC-FT).

### Formal Definition

A two-player cooperative Markov game is defined as $\langle S, A, \mathcal{T}, R, H \rangle$, where a task $m \sim \mathcal{M}$ defines the initial state distribution $p(s_0|m)$, while sharing the transition dynamics $\mathcal{T}$ and reward function $R$.

**PBT Objective Function** (single task, multiple partners):

$$J(\pi_C) = \mathbb{E}_{\pi_i \sim P}[S(\pi_i, \pi_C, m)]$$

**CEC Objective Function** (multiple environments, self-play):

$$J(\pi_C) = \mathbb{E}_{m_i \sim \mathcal{M}}[S(\pi_C, \pi_C, m_i)]$$

where the cooperation score is defined as:

$$S(\pi_p, \pi_C, m) = \mathbb{E}_{\substack{s_0 \sim m, s \sim \mathcal{T} \\ a^p \sim \pi_p, a^C \sim \pi_C}} \left[\sum_{t=0}^{H} R(s_t, a_t^p, a_t^C)\right]$$

Key comparison: PBT computes the expectation over the partner distribution $P$, while CEC computes the expectation over the task distribution $\mathcal{M}$. CEC only needs to train a single policy and does not require maintaining a partner population.

### Procedural Environment Generation

#### Dual Destination (Toy Environment)

- Two agents in a grid need to reach different green goal cells.
- Reaching the opposite goal cell yields +3 reward, with a -1 penalty per step.
- CEC Version: Randomizes agent starting positions and goal positions.

#### Overcooked (Main Evaluation Environment)

Extended based on the Overcooked implementation in the JaxMARL project, using the wall structures of five classic layouts (Asymmetric Advantages, Coordination Ring, Counter Circuit, Cramped Room, Forced Coordination):

1. Randomly sample one of the five layouts as the base structure.
2. Remove all items and agents, keeping only the walls.
3. Randomly place essential items (plate stack, onion stack, pot, serving counter) on reachable walls.
4. Randomly place extra items on the remaining walls.
5. Randomly sample initial positions of agents (split layouts ensure players are on opposite sides of the partition).
6. Rotate by 90° with 50% probability, embedding into a 9x9x26 observation space.
7. Check if it duplicates any evaluation layout, and regenerate if it does.

This generator can produce $1.16 \times 10^{17}$ different solvable kitchen configurations. The entire pipeline is built on JAX, achieving a single-GPU training speed of **10 million steps per minute**.

### Neural Network Architecture

| Component | Layer | Details |
|------|------|------|
| Observation Encoder | Conv1 | 2×2 kernel, 64 filters, ReLU |
| | Conv2 | 2×2 kernel, 32 filters, ReLU |
| | FC1 | Fully Connected, 512 units, ReLU |
| | FC2 | Fully Connected, 512 units, ReLU |
| Recurrent Core | LSTM | Feature dimension 256, state reset at episode boundaries |
| Actor Head | FC1-FC4 | 256→192→128→64, ReLU |
| | Output | 6 action logits (Overcooked) |
| Critic Head | FC1-FC4 | 512→256→192→128, ReLU |
| | Output | 1 scalar value prediction |

**Necessity of Recurrent Network**: Experiments show that without LSTM, CEC fails to obtain even positive rewards within 300M steps of training, as LSTM provides basic meta-learning capabilities, enabling agents to infer partner intent within an episode.

### PPO Training Hyperparameters

- Learning rate: $3 \times 10^{-4}$ (annealed)
- Total training steps: $3 \times 10^9$
- GAE parameters: $\gamma=0.99, \lambda=0.95$
- PPO clipping: $\epsilon=0.2$
- Entropy coefficient: 0.005
- Gradient clipping: 0.5

### CEC-Finetune

After training the general CEC policy, five copies are created for the five evaluation layouts respectively. Self-play training is continued for $10^8$ steps on each layout with a reduced learning rate to obtain the CEC-FT model.

## Key Experimental Results

### Toy Environment Experiment (Dual Destination)

| Method | Fixed Task XP | Procedural Task XP |
|------|------------|----------------|
| IPPO (SP) | ~0.2 | ~0.05 |
| FCP | ~0.6 | ~0.15 |
| **CEC** | **~0.93** | **~0.97** |

- CEC normalized reward is 0.931 (standard error 0.013), which is only about 2.5% lower than the ideal oracle.
- Statistical significance: CEC vs FCP and IPPO both achieve $p < 0.001$ (t-test).

### Overcooked AI-AI Evaluation

#### 5 Classic Layouts (XP Performance)

| Method | Average XP Reward |
|------|------------|
| IPPO | ~50 |
| FCP | ~80 |
| E3T | ~90 |
| **CEC** | **~130** |
| **CEC-FT** | **~155** |

- CEC-FT significantly outperforms FCP and IPPO on classic layouts ($p < 0.01$).
- CEC (which has not seen these layouts) still outperforms all single-task baselines.

#### 100 Procedural Layouts (XP Performance)

| Method | Average XP Reward |
|------|------------|
| IPPO | ~0 |
| FCP | 0 |
| E3T | 0 |
| **CEC** | **~70** |
| CEC-FT | ~42 |

- FCP and E3T obtain **zero reward** on new layouts, completely failing to generalize.
- CEC significantly outperforms all baselines ($p < 0.0001$).
- CEC-FT exhibits degraded generalization, reflecting the trade-off between generality and specialization.

### Cross-Algorithm Collaboration Analysis

Through Empirical Game-Theoretic Analysis (EGTA), the cross-algorithm cooperation scores are used as the payoff matrix of a meta-game to compute the replicator dynamics gradient. The results show that on both the 5 classic layouts and 100 procedurally generated layouts, the gradients point towards CEC and CEC-FT, indicating they are likely equilibrium strategies.

### Human Experiment Results

80 participants cooperated with various AI models on two layouts: Counter Circuit and Coordination Ring.

#### Cooperation Scores (Quantitative)

| Method | Human Cooperation Score |
|------|-------------|
| IPPO | ~2.0 |
| FCP | ~4.0 |
| **CEC** | **~7.5** |
| **CEC-FT** | **~8.0** |
| E3T | ~9.5 |

- CEC is significantly better than FCP ($p < 0.001$).
- CEC is close to E3T in performance despite never having seen the evaluation layouts.

#### Subjective Evaluation (Qualitative, 7 Metrics)

CEC and CEC-FT scored the highest user ratings across all of the following subjective dimensions:
- Adaptability, consistency, enjoyability, coordination, low frustration, cooperative ability, and overall preference.
- CEC-FT significantly outperforms E3T ($p < 0.01, t=3.1233$).
- Cronbach's alpha = 0.874, validating the internal consistency of subjective metrics.

#### Collision Analysis

CEC has the lowest average collision frequency with humans, indicating that CEC learns general cooperative norms such as "giving way". Although this may decrease short-term rewards, it significantly improves user experience.

### Ablation Study

| Ablation item | Key Findings |
|--------|---------|
| Partial Observability | Under a 3×3 window, CEC (0.74) > PBT (0.61) > SP (0.03), showing consistent conclusions. |
| Multi-Task Variant | Under 4 effective strategies, CEC (0.404) > PBT (0.251) > SP (0.083). |
| CEC+E3T | Combining partner diversity, generalization to new layouts is better than CEC-FT, but classic layout performance drops. |
| Removing LSTM | Fails to converge, unable to obtain positive rewards within 300M steps. |
| CEC-FT | Enhances specific layout performance but loses generalization capability. |

### Behavioral Pattern Analysis

By visualizing the tile visit frequencies on the Counter Circuit layout:
- **IPPO**: Highly concentrated visit distribution with fixed routes (clockwise or counter-clockwise), indicating policy fragility.
- **CEC**: More uniform visit distribution, with high-frequency regions clustered around task-relevant items (pots, onion stacks, plate stacks, serving counter), indicating that CEC learns rich representations of task structure.

## Highlights & Insights

1. **Subversive Finding**: The conventional view that self-play is "insufficient" in cooperative games is challenged—the key is not to increase partner diversity, but to increase environment diversity. CEC trained via self-play outperforms PBT in zero-shot coordination.

2. **Emergence of General Cooperative Norms**: Cross-environment training not only enhances environmental generalization but also surprisingly improves partner generalization. Agents learn general cooperative norms such as "giving way to others" and "focusing on task-relevant items".

3. **Computational Efficiency Advantage**: CEC only needs to train a single policy, whereas PBT requires training an entire population for each task. Under the same computational budget, CEC invests computation in environmental diversity rather than partner diversity, yielding better results.

4. **Decoupling of Quantitative and Qualitative Evaluations**: E3T scores higher in reward, but CEC is superior in subjective human evaluation. This implies that **reward maximization does not equal good cooperation**—overly greedy policies may force humans to adapt to the AI rather than the AI adapting to humans.

5. **Engineering Contribution from JAX Acceleration**: Provides a JAX-based procedural Overcooked environment generator, reaching 10M steps/minute on a single GPU, providing scalable infrastructure for large-scale multi-agent collaboration research.

## Limitations & Future Work

1. **Incomplete Convergence of Training**: After 3B steps of training, the learning curve of CEC has not saturated, leaving its performance upper bound unexplored due to computational resource limits.
2. **Limited Environmental Complexity**: Although Overcooked is a standard benchmark, it remains relatively simple compared to real-world collaborative scenarios (such as household robotics).
3. **Human Experiment Bias**: Participants were limited to fluent English speakers, which may introduce cultural bias influencing collaborative behaviors and evaluations.
4. **Limited Effect of CEC+PBT Combination**: Combining CEC with E3T degrades training efficiency, potentially requiring larger networks and longer training times.
5. **Generalization vs. Specialization Trade-off**: While CEC-FT improves performance on specific layouts, it sacrifices generalization to new layouts. How to balance the two remains an open question.

## Related Work & Insights

- **Self-play Methods**: AlphaStar (Vinyals et al., 2019) and AlphaGo (Silver et al., 2017) achieved massive success in zero-sum games, but fail in cooperative games due to the multi-equilibrium problem.
- **PBT Methods**: FCP (Strouse et al., 2022), MEP (Zhao et al., 2022), and E3T (Yan et al., 2023) improve ZSC via partner diversity, but are restricted to a single environment.
- **Procedural Environment Generation**: Procgen (Cobbe et al., 2020) and MAESTRO (Samvelyan et al., 2023) demonstrate the value of environment diversity in single-agent or zero-sum settings.
- **Environment Diversity and Cooperation**: McKee et al. (2022) found that environmental diversity improves cooperation generalization, but did not test with humans or compare the relative effects of environment vs. partner diversity.
- **UED Methods**: Ruhdorfer et al. (2024) used unsupervised environment design in Overcooked, but did not guarantee the solvability of generated tasks, resulting in poor generalization.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall Rating | ⭐⭐⭐⭐ |

> The CEC paradigm proposed in this paper is simple yet effective; replacing partner diversity with environmental diversity to achieve zero-shot coordination is an elegant and inspiring idea. The experiments cover a complete evaluation pipeline from toy environments to large-scale Overcooked, and from AI-AI to Human-AI coordination, with particularly solid human experiment design. The main limitation is that the environmental complexity remains bounded, and the fact that 3B training steps did not fully converge means that the true potential of CEC has not yet been fully realized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection](../../CVPR2026/multi_agent/agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/multi_agent/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](../../ACL2025/multi_agent/multi-agent_collaboration_via_cross-team_orchestration.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](../../AAAI2026/multi_agent/learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)

</div>

<!-- RELATED:END -->
