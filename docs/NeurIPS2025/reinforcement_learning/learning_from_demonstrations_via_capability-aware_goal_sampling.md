---
title: >-
  [Paper Note] Learning from Demonstrations via Capability-Aware Goal Sampling
description: >-
  [NeurIPS 2025][Reinforcement Learning][Imitation Learning] This paper proposes Cago, a method that dynamically tracks an agent's attainment capability along expert demonstration trajectories and adaptively samples intermediate goals near the capability frontier, constructing an implicit curriculum to guide learning in long-horizon, sparse-reward tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Imitation Learning
  - Curriculum Learning
  - Goal-Conditioned Reinforcement Learning
  - Capability-Awareness
  - World Model
date: 2026-05-08
content_hash: 0a59a6b7293b6282
---

# Learning from Demonstrations via Capability-Aware Goal Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2601.08731](https://arxiv.org/abs/2601.08731)
**Code**: [GitHub](https://github.com/RU-Automated-Reasoning-Group/Cago)
**Area**: Reinforcement Learning
**Keywords**: Imitation Learning, Curriculum Learning, Goal-Conditioned Reinforcement Learning, Capability-Awareness, World Model

## TL;DR

This paper proposes Cago, a method that dynamically tracks an agent's attainment capability along expert demonstration trajectories and adaptively samples intermediate goals near the capability frontier, constructing an implicit curriculum to guide learning in long-horizon, sparse-reward tasks.

## Background & Motivation

**Background**: Imitation learning trains agents from expert demonstrations via methods such as behavioral cloning (BC), GAIL, and inverse reinforcement learning, yet significant challenges remain for long-horizon complex tasks.

**Limitations of Prior Work**:
- BC suffers from compounding errors.
- Distribution-matching methods (e.g., GAIL) perform "flat matching" in early training, failing to distinguish between mastered and unmastered segments.
- Reverse curriculum methods require resetting the agent to arbitrary states along demonstrations, which is unrealistic in real-world settings where quantities such as joint velocities are difficult to reproduce precisely.

**Key Challenge**: Existing methods do not account for the dynamic evolution of agent capability—they lack awareness of which portions of a task have been mastered and which remain challenging.

**Goal**: Construct an adaptive learning curriculum aligned with the agent's capability without requiring resets to arbitrary intermediate states.

**Key Insight**: Demonstrations are treated as structured roadmaps rather than direct imitation targets; the agent's current capability ceiling is continuously monitored to select intermediate goals.

**Core Idea**: Observation visit frequency is used to track the capability frontier, and goals that lie just beyond the agent's current capability are sampled to guide Go-Explore-style exploration.

## Method

### Overall Architecture

The framework operates as a three-step closed loop: (1) **Observation Visit Tracking**—recording the frequency with which the agent visits each position along demonstration trajectories; (2) **Capability-Aware Goal Sampling**—sampling intermediate goals near the capability frontier; (3) **Go-Explore-style Training**—a goal-conditioned policy first navigates to the sampled goal, a BC Explorer then continues exploration from that point, and the collected data trains both a World Model and the policy.

### Key Designs

1. **Observation Visit Tracking**:

    - *Function*: Maintains a dictionary $\text{Dict}_{visit}$ recording the agent's visit frequency at each step of every demonstration trajectory.
    - *Design Motivation*: Visit frequency directly reflects the agent's ability to reach corresponding states.
    - *Mechanism*: Updated at each environment step as $\text{Dict}_{visit}[\tau^{(i)}][j] += 1$ when $\text{sim}(s_t, s_j^{(i)}) \leq \epsilon$; supports L2 distance (state space) and MSE (visual environments).
    - *Novelty*: Only resets to the initial demonstration state, eliminating the need to reset to arbitrary intermediate states.

2. **Capability-Aware Goal Sampling**:

    - *Function*: Samples goals of appropriate difficulty near the capability frontier.
    - *Design Motivation*: Goals that are too easy provide no learning signal; goals that are too difficult cause divergence.
    - *Mechanism*:
        - Capability ceiling: $j^* = \max\{j | \text{Dict}_{visit}[\tau^{(i)}][j] \geq \lambda_{visit}\}$
        - Sampling window: $\mathcal{G}_{cap}(\pi^G, \tau^{(i)}) = \{s_k \in \tau^{(i)} | |k - j^*| \leq \delta \cdot L_i\}$
    - $\lambda_{visit}$: visit frequency threshold (e.g., 100); $\delta$: window size (e.g., 10% of trajectory length).
    - *Novelty*: Unlike JSRL's uniform curriculum, this approach genuinely reflects the agent's current capability.

3. **Go-Explore-style Data Collection**:

    - *Function*: Each episode is divided into a Go phase and an Explore phase.
    - *Design Motivation*: The two-phase structure ensures collected data is both close to the demonstration distribution and exploratory.
    - *Mechanism*: In the Go phase, the goal-conditioned policy $\pi^G(\cdot|s, g)$ attempts to reach the sampled goal; in the Explore phase, the BC Explorer $\pi^E$ (a behavioral cloning policy) continues exploration from the reached state.
    - *Novelty*: The BC Explorer provides higher-quality exploration than random exploration.

4. **World Model and Policy Training**:

    - *Function*: Trains the goal-conditioned policy using imagined trajectories from the World Model.
    - *Design Motivation*: Data collected via Go-Explore is close to the demonstration distribution, yielding a more accurate World Model in these regions.
    - *Mechanism*: Built on the Dreamer framework; uses a temporal distance function $D_t(s,g)$ as reward $r^G(s,g) = -D_t(s,g)$.
    - *Novelty*: Theorem 1 proves that the BC Explorer effectively reduces the upper bound on model prediction error.

5. **Goal Predictor**:

    - *Function*: Infers the final goal from the current observation at test time.
    - *Design Motivation*: Demonstration trajectories are unavailable during testing.
    - *Mechanism*: $\mathcal{P}_\phi: s \mapsto \hat{g}$, trained by minimizing $\|\mathcal{P}_\phi(s_t^{(i)}) - s_L^{(i)}\|_2^2$; the final policy is $\pi(s) = \pi^G(s, \mathcal{P}(s))$.

### Loss & Training

- World Model: supervised loss under the Dreamer framework.
- Policy: Actor-Critic with temporal distance reward.
- Goal Predictor: MSE regression loss.
- BC Explorer: behavioral cloning loss.
- Each task uses only 10–20 demonstrations.

## Key Experimental Results

### Main Results

**MetaWorld Very Hard Tasks (Success Rate %, averaged over 8 seeds)**:

| Method | Disassemble | PickPlaceWall | ShelfPlace | StickPull | StickPush |
|--------|-------------|---------------|------------|-----------|-----------|
| Dreamer | ~10% | ~5% | ~10% | ~5% | ~15% |
| JSRL | ~25% | ~20% | ~30% | ~20% | ~30% |
| MoDem | ~40% | ~35% | ~40% | ~30% | ~45% |
| Cal-QL | ~15% | ~10% | ~15% | ~10% | ~20% |
| **Cago** | **~70%** | **~60%** | **~65%** | **~55%** | **~70%** |

**Adroit Dexterous Manipulation Tasks (Success Rate after 1M steps)**:

| Method | Door | Hammer | Pen |
|--------|------|--------|-----|
| MoDem | ~60% | ~70% | ~55% |
| **Cago** | **~80%** | **~85%** | **~75%** |

**ManiSkill Hard Tasks**: Cago is the only method capable of achieving high success rates under limited demonstrations.

### Ablation Study

**Component Importance (Disassemble / StickPush / Pen, 5 seeds)**:

| Variant | Description | Performance |
|---------|-------------|-------------|
| Cago (full) | Capability-aware sampling + BC Explorer | Best |
| Cago-FinalGoal | BC Explorer only, always targets final goal | Significant drop |
| Cago-StepBased | Goal sampled proportionally to training steps | Drop |
| Cago-NoExplorer | Capability-aware sampling only, no BC Explorer | Notable drop |
| Cago-RandomExplorer | Random exploration replaces BC Explorer | Drop |

### Key Findings

- Capability-aware goal sampling is the core contribution; removing it causes significant performance degradation.
- The normalized goal position progresses naturally from 0 to 1 over training, confirming the effectiveness of the adaptive curriculum.
- The BC Explorer is critical for data quality; random exploration performs considerably worse.
- The method functions effectively with as few as 10 demonstrations.
- The visual-input variant (Cago-Visual) achieves comparable performance, demonstrating strong generalization.

## Highlights & Insights

- **Core insight of "capability-awareness"**: Unlike existing methods that assume fixed curricula or global distribution matching, Cago genuinely tracks the agent's dynamic learning state.
- **Reset only to initial states**: This is far more practical than reverse curriculum methods, as it avoids the need to reproduce latent variables such as joint velocities.
- **Dual validation via theory and experiment**: Theorem 1 provides theoretical error-bound guarantees, and experiments span three major benchmarks.
- **Goal-conditioned extension of the Go-Explore paradigm**: The classic exploration strategy is elegantly combined with demonstration-guided learning.

## Limitations & Future Work

- The method relies on resets to the initial demonstration state, which is far less restrictive than arbitrary-state resets but remains a constraint.
- The similarity metric $\text{sim}(\cdot,\cdot)$ and threshold $\epsilon$ may require task-specific tuning.
- The generalization of the Goal Predictor to out-of-distribution scenarios warrants further investigation.
- Future work could explore integrating LLMs or VLMs as goal predictors for more abstract tasks.

## Related Work & Insights

- **Comparison with JSRL**: JSRL uses a predefined curriculum rather than capability-aware adaptation.
- **Comparison with MoDem**: MoDem achieves rapid convergence via demonstration oversampling but is limited in final performance.
- The Go-Explore paradigm is reinterpreted in Cago through goal conditioning and demonstration guidance.
- The Dreamer world model provides the infrastructure for imagination-based training under capability-aware sampling.

## Rating

- Novelty: ⭐⭐⭐⭐ Capability-aware goal sampling is intuitive and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three major benchmarks, 11 tasks, comprehensive ablations, and visual extension.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, fluent method description, and complete theoretical analysis.
- Value: ⭐⭐⭐⭐ Substantial improvements on long-horizon sparse-reward tasks with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Distribution Learning Meets Graph Structure Sampling](distribution_learning_meets_graph_structure_sampling.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](../../ICLR2026/reinforcement_learning/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)

</div>

<!-- RELATED:END -->
