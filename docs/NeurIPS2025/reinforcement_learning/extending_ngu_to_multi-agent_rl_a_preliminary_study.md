---
title: >-
  [Paper Note] Extending NGU to Multi-Agent RL: A Preliminary Study
description: >-
  [NeurIPS 2025][Reinforcement Learning][Never Give Up] This paper extends the single-agent NGU (Never Give Up) algorithm to multi-agent settings and conducts a systematic ablation across three design dimensions: shared re…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Never Give Up"
  - "Multi-Agent Reinforcement Learning"
  - "Intrinsic Motivation"
  - "Exploration"
  - "Sparse Rewards"
date: 2026-05-08
content_hash: 88cf02626b9e84a1
---

# Extending NGU to Multi-Agent RL: A Preliminary Study

**Conference**: NeurIPS 2025  
**arXiv**: [2512.01321](https://arxiv.org/abs/2512.01321)  
**Code**: [GitHub](https://github.com/juan-hernandez/extending-ngu-marl)  
**Area**: Reinforcement Learning  
**Keywords**: Never Give Up, Multi-Agent Reinforcement Learning, Intrinsic Motivation, Exploration, Sparse Rewards

## TL;DR
This paper extends the single-agent NGU (Never Give Up) algorithm to multi-agent settings and conducts a systematic ablation across three design dimensions: shared replay buffer, shared novelty signal, and heterogeneous β parameters. The results show that NGU combined with a shared experience replay buffer significantly outperforms a multi-agent DQN baseline on the PettingZoo simple_tag pursuit task.

## Background & Motivation

**Background**: Reinforcement learning has achieved remarkable success on benchmarks such as Atari, yet methods like DQN perform poorly in sparse-reward environments (e.g., Montezuma's Revenge) due to insufficient exploration. The NGU algorithm addresses this by combining episodic novelty with intrinsic motivation, achieving state-of-the-art performance on single-agent sparse-reward tasks.

**Limitations of Prior Work**: The sparse-reward problem is even more severe in multi-agent reinforcement learning (MARL), where agents must additionally contend with credit assignment, environmental non-stationarity, and coordinated exploration. Existing MARL exploration methods (e.g., EMC, MACE) typically introduce additional architectural complexity and computational overhead.

**Key Challenge**: Although sophisticated exploration mechanisms are effective in MARL, they generalize poorly, while the simpler and more powerful NGU framework has not been systematically adapted to multi-agent settings.

**Goal**: Can the core exploration mechanisms of NGU be directly extended to MARL? How do three design choices—shared experience, shared novelty, and heterogeneous exploration parameters—affect performance?

**Key Insight**: The paper retains NGU's core components (embedding network, inverse dynamics model, episodic memory) while removing RND and UVFA to reduce computational cost, and directly evaluates the resulting approach in a MARL environment.

**Core Idea**: Through ablation of three key design dimensions, the paper demonstrates that NGU combined with a shared replay buffer constitutes the most effective multi-agent exploration configuration.

## Method

### Overall Architecture
Each agent $i \in \{1, \dots, N\}$ maintains an independent Q-network, embedding network, episodic memory, and intrinsic reward computation module. Agents receive local observations as input and produce discrete actions as output. The embedding network is trained jointly with an inverse dynamics model to learn state representations, and novelty rewards are computed via k-nearest-neighbor distances over the episodic memory.

### Key Designs

1. **Intrinsic Reward Computation**:

    - Function: Computes an episodic memory-based novelty reward for each agent.
    - Mechanism: An embedding network $\phi: \mathcal{S} \to \mathbb{R}^d$ is trained via an inverse dynamics loss to predict the action $a_t$ between consecutive embeddings. At each step, the intrinsic reward for agent $i$ is computed as $r_{t,i}^{\text{intrinsic}} = f(\phi(s_{t+1}^i), \mathcal{M}_i)$, where $f$ is a k-nearest-neighbor distance function and $\mathcal{M}_i$ is the within-episode embedding memory buffer.
    - The total reward is $r_{t,i} = r_t^{\text{extrinsic}} + \beta_i \cdot r_{t,i}^{\text{intrinsic}}$.
    - Design Motivation: This preserves NGU's core novelty-driven exploration mechanism, encouraging agents to visit unseen states without requiring additional modules such as RND.

2. **Shared Replay Buffer**:

    - Function: Aggregates the experience of all agents into a single centralized replay buffer.
    - Mechanism: Pooling experience improves sample efficiency and mitigates non-stationarity, allowing each agent to learn from the trajectories of others.
    - Design Motivation: Individual agents accumulate limited experience in MARL; sharing experience effectively expands the usable sample pool and enables more stable Q-function estimation.

3. **Shared Novelty**:

    - Function: Marks a state embedding as "no longer novel" once it has been visited by $k$ distinct agents.
    - Mechanism: Cosine similarity is used to detect similarity among projected embeddings; when $k$ agents have visited the vicinity of a state, that state ceases to generate intrinsic rewards for all agents.
    - Design Motivation: Prevents multiple agents from repeatedly exploring the same regions, though excessively large $k$ values render the signal too sparse.

4. **Heterogeneous β Parameters**:

    - Function: Assigns different intrinsic/extrinsic reward trade-off parameters to different agents, e.g., $\{0.1, 0.2, 0.4\}$.
    - Mechanism: Some agents prioritize exploration (large β) while others prioritize exploitation (small β), inducing role specialization.
    - Design Motivation: Inspired by mixed-strategy thinking, the expectation is that behavioral diversity will improve overall team performance.

### Loss & Training
All configurations use identical hyperparameters: learning rate 0.001, buffer size $10^6$, batch size 128, $\gamma=0.99$, training frequency every 16 steps, 4 gradient steps per update, target network update interval 2000 steps, and $\epsilon$ linearly decayed from 1.0 to 0.1. Each configuration is evaluated over 200K timesteps with 15 independent seeds.

## Key Experimental Results

### Main Results

Experiments are conducted in the PettingZoo simple_tag_v3 environment, where multiple pursuers (red) cooperate to capture an evader (blue) under sparse shared rewards. The evader follows a default heuristic policy.

| Configuration | Replay Buffer | Mean Return Rank | Learning Stability |
|---------------|--------------|------------------|--------------------|
| Multi-DQN | Independent | Lowest | Unstable, high variance |
| Multi-NGU | Independent | Above average | Noticeably more stable |
| Multi-NGU + Shared Novelty (k=1) | Independent | Comparable to NGU | Stable |
| Multi-NGU + Heterogeneous β | Independent | Slightly below NGU | Moderate |
| Multi-DQN | Shared | Moderate | Improved |
| **Multi-NGU** | **Shared** | **Highest** | **Most stable** |
| Multi-NGU + Shared Novelty (k=1) | Shared | Near best | Stable |
| Multi-NGU + Heterogeneous β | Shared | Below standard NGU | Moderate |

### Ablation Study

| Ablation Dimension | Best Configuration | Key Findings |
|--------------------|-------------------|--------------|
| Replay buffer | Shared > Independent | All variants improve with sharing; NGU's advantage becomes more pronounced |
| Novelty sharing threshold k | k=1 optimal | k=2, k=3 degrade performance; novelty signal loses informativeness when averaged over more neighbors |
| Heterogeneous β values | {0.1,0.2,0.4} > {0.2,0.4,0.6} | Smaller β combinations are more stable, but none surpass uniform β=0.1 |
| β grid search | β=0.1 optimal | Among {0, 0.1, 0.5, 1.0}, β=0.1 achieves the highest and most stable return |

### Key Findings
- **Shared replay buffer is the largest source of gain**: The combination of NGU's intrinsic exploration with experience sharing yields the strongest performance.
- **Shared novelty at k=1 matches standard NGU**, but k>1 degrades performance—higher k values render the novelty signal excessively sparse.
- **Heterogeneous β fails to surpass uniform small β**: Role specialization does not offer benefits over consistently mild exploration in this environment.
- Shared novelty converges faster in early training but is eventually surpassed by standard NGU, suggesting a trade-off between rapid redundancy elimination early on and sustained novelty over the long run.

## Highlights & Insights
- **Transferability of NGU's core mechanism**: NGU remains effective after removing RND and UVFA, indicating that episodic novelty is the primary driver of its exploration capability. This simplification strategy generalizes to the multi-agent adaptation of other complex algorithms.
- **Conceptual equivalence of experience sharing and novelty sharing** is noteworthy: both fundamentally reduce the intrinsic reward contribution of already-observed experiences, yet their learning dynamics differ—the former indirectly enriches training data, while the latter directly modifies the reward signal.
- **Systematic ablation design**: The three dimensions are orthogonal and comprehensive; reporting with 15 independent runs and 95% confidence intervals provides a reliable reference for future work.

## Limitations & Future Work
- **Validation on a single environment (simple_tag)** precludes confident generalization to competitive, partially observable, or other settings.
- **Only DQN is used as the base algorithm**; stronger MARL algorithms such as VDN, QMIX, and MAPPO are not explored.
- **RND and UVFA are omitted**, yet these components may be critical in more complex or large-scale environments.
- **The environment scale is small** (a 2D bounded arena); large-scale benchmarks such as StarCraft are not evaluated.
- The shared novelty mechanism relies on a hard threshold, which could be replaced by attention-weighted or gradually decaying alternatives.

## Related Work & Insights
- **vs. EMC**: EMC uses Q-value prediction error as a curiosity signal combined with episodic memory to reinforce informative trajectories, resulting in greater architectural complexity; this paper achieves comparable performance with the simpler NGU mechanism.
- **vs. MACE**: MACE enables decentralized agents to share local novelty signals to approximate global novelty; however, this paper finds that directly sharing a replay buffer is more effective than sharing novelty signals.
- The findings suggest that in MARL, **experience sharing is more important than signal sharing**, offering a core insight for the design of new distributed exploration methods.

## Rating
- Novelty: ⭐⭐⭐ The method is a direct extension rather than a fundamentally new design, though the systematic ablation has research value.
- Experimental Thoroughness: ⭐⭐⭐ The ablation is comprehensive (15 seeds × 8 configurations), but only one environment is used and comparisons with MARL SOTA are absent.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with rigorous experimental setup and well-defined conclusions; its self-characterization as a preliminary study is appropriately honest.
- Value: ⭐⭐⭐ The work establishes an empirical foundation for the NGU→MARL direction, and the shared replay buffer finding has practical guidance value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)

</div>

<!-- RELATED:END -->
