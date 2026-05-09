---
title: >-
  [Paper Note] First-Order Representation Languages for Goal-Conditioned RL
description: >-
  [AAAI 2026][Reinforcement Learning][Goal-conditioned reinforcement learning] This paper investigates the application of first-order relational languages to goal-conditioned RL and generalized planning. It proposes representing goals as subsets or lifted versions of sets of atoms, and combines this with HER to automatically construct easy-to-hard goal curricula, enabling the learning of generalizable policies on large-scale sparse-reward planning problems.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Goal-conditioned reinforcement learning
  - first-order representation languages
  - Hindsight Experience Replay
  - generalized planning
  - curriculum learning
date: 2026-05-08
content_hash: 8f58b645b63b8a14
---

# First-Order Representation Languages for Goal-Conditioned RL

**Conference**: AAAI 2026
**arXiv**: [2512.19355](https://arxiv.org/abs/2512.19355)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Goal-conditioned reinforcement learning, first-order representation languages, Hindsight Experience Replay, generalized planning, curriculum learning

## TL;DR

This paper investigates the application of first-order relational languages to goal-conditioned RL and generalized planning. It proposes representing goals as subsets or lifted versions of sets of atoms, and combines this with HER to automatically construct easy-to-hard goal curricula, enabling the learning of generalizable policies on large-scale sparse-reward planning problems.

## Background & Motivation

**State of the Field**: First-order relational languages serve two primary purposes in MDP-based planning and reinforcement learning: (1) compactly describing MDPs, and (2) representing and learning policies with generalization capabilities that are not tied to specific instances or state spaces. Goal-conditioned RL is an important direction in this context, where the agent must learn a policy capable of reaching any given goal state.

**Limitations of Prior Work**: When training instances are large and goals cannot be reached through random exploration, goal-conditioned RL faces severe challenges. Hindsight Experience Replay (HER) alleviates this by relabeling failed trajectories as if the agent had intended to reach the states it actually visited. However, standard HER, when states and goals are represented propositionally, offers limited generalization—it struggles to transfer experience from small instances to large ones.

**Root Cause**: Generalizable goal-conditioned policies must handle varying state and goal spaces, and standard vector representations fail to capture relational structure. Although first-order representations are more expressive, how to effectively leverage them to accelerate goal-conditioned RL learning remains an open problem.

**Paper Goals**: To investigate whether goal-conditioned RL can achieve further performance gains when states and goals are represented as sets of atoms.

**Starting Point**: The authors propose three goal representation schemes: (1) the goal as a complete state; (2) the goal as a subset of the original goal atoms; and (3) the goal as a lifted version of the subgoal. The latter two representations automatically construct curricula from simple to complex goals, thereby accelerating learning.

**Core Idea**: By exploiting the structural properties of first-order atomic representations, complex goals are automatically decomposed into simpler subgoals, forming a curriculum learning mechanism that enables effective learning of generalizable policies on large-scale sparse-reward problems.

## Method

### Overall Architecture

The MDP states and goals are represented as sets of first-order atoms. The agent is trained via HER-style learning to obtain a goal-conditioned policy. The key innovation lies in how goals are represented and relabeled—progressing from full-state goals to subgoals to lifted subgoals—incrementally increasing generalization capability and the effectiveness of the induced curriculum.

### Key Designs

1. **Goal as Subsets of Atoms**:

    - **Function**: Decomposes complex goals into simpler subgoals to reduce learning difficulty.
    - **Mechanism**: The original goal is a conjunction of atoms, e.g., $\{on(A,B), on(B,C), clear(A)\}$. This is decomposed into subsets such as $\{on(A,B)\}$, which serve as simpler intermediate goals. During HER relabeling, in addition to using the fully reached state as a new goal, any atomic subset satisfied by that state can also be used as a new goal, substantially increasing the volume of effective training data.
    - **Design Motivation**: In combinatorial domains such as Blocksworld, the complete goal may involve numerous constraints, making the distance from a random state to the full goal very large. Subgoal decomposition naturally induces a curriculum from simple to complex.

2. **Lifted Subgoals**:

    - **Function**: Further enhances generalization, enabling policies to transfer across object instances.
    - **Mechanism**: Concrete objects in subgoals are replaced by variables; e.g., $\{on(A,B)\}$ is lifted to $\{on(X,Y)\}$. This transforms "place block A on block B" into a universal goal "place any block on another block." Lifted subgoals can be matched by any state satisfying the pattern, further increasing the effective training data for HER relabeling.
    - **Design Motivation**: The central challenge in generalized planning is transferring from small to large instances. Lifted goals eliminate object-specificity, allowing policies to learn relational patterns rather than operations between particular objects.

3. **Automatic Curriculum Generation**:

    - **Function**: Begins training with simple goals and progressively increases complexity.
    - **Mechanism**: In early training, relabeling predominantly produces simple single-atom or two-atom subgoals, allowing the policy to first master simple tasks. As training progresses, subgoal complexity is gradually increased until the full original goal can be achieved. This curriculum arises automatically through the combination of HER relabeling and the nested structure of subgoal decomposition.
    - **Design Motivation**: In large-scale planning problems with sparse rewards, directly learning the full goal is nearly intractable. Curriculum learning reduces exploration difficulty through a gradual progression.

### Loss & Training

The framework adopts a standard goal-conditioned RL training paradigm (e.g., goal-conditioned DQN or PPO) with sparse binary rewards (+1 for reaching the goal, 0 otherwise). The HER relabeling strategy is extended to multi-level relabeling, simultaneously using full states, subsets, and lifted subsets as virtual goals.

## Key Experimental Results

### Main Results

Evaluated on large instances across multiple planning domains (e.g., Blocksworld, Logistics).

| Method | Success Rate (Large Instances) | Generalization | Data Efficiency | Notes |
|--------|-------------------------------|----------------|-----------------|-------|
| Standard RL | Negligible | None | Poor | Random exploration cannot reach goals |
| HER (full goal) | Moderate | Limited | Moderate | Standard HER baseline |
| HER + Subset Goals | High | Good | Good | Subgoal decomposition yields significant gains |
| HER + Lifted Subgoals | Highest | Strongest | Best | Cross-object generalization + best curriculum effect |

### Ablation Study

| Configuration | Success Rate | Notes |
|---------------|-------------|-------|
| Full goal (baseline) | Low | Goal too complex to learn directly |
| Subset goals | Moderate–High | Automatic curriculum is effective |
| Lifted subgoals | Highest | Dual gains from generalization and curriculum |
| Random curriculum | Low | Structured curriculum far outperforms random |

### Key Findings

- The subset goal and lifted subgoal variants successfully learn generalizable policies on large-scale planning instances where standard methods fail entirely, demonstrating the practical value of first-order representations in goal-conditioned RL.
- Automatic curriculum generation is critical to success—initializing training with simple subgoals avoids the sparse-reward exploration bottleneck.
- Lifted goals provide cross-object generalization: the same relational pattern can be applied to any concrete object instantiation.
- The paper also honestly discusses the limitations and future directions of the proposed approach.

## Highlights & Insights

- The idea of **using atomic subsets as an automatic curriculum** is elegant and natural—no manual curriculum design is required; the compositional structure of goals itself defines a hierarchy from simple to complex.
- The **generalization gains from first-order lifting** validate the value of combining symbolic AI with deep RL—explicit modeling of relational structure yields generalization capabilities inaccessible to propositional representations.
- HER obtains a "free" data augmentation effect under first-order representations: a single trajectory can be relabeled into a much larger number of effective training samples.

## Limitations & Future Work

- The method assumes that states and goals can be precisely represented as sets of atoms, and does not directly apply to continuous or pixel-level observations.
- Lifted goals require a typed object system and may not be applicable in domains where object types are unclear.
- Learned policies may still exhibit generalization limitations on very large instances.
- Integration with graph neural networks could allow GNNs to directly process first-order relational structures, enhancing representational capacity.

## Related Work & Insights

- **vs. Standard HER**: Standard HER relabels in propositional space; this work relabels in first-order atomic space, yielding a structured curriculum effect.
- **vs. STRIPS Planners**: Classical planners solve instances exactly but cannot learn or generalize. The proposed method combines the structural advantages of planning with the learning capability of RL.
- **vs. Curriculum Learning Methods**: Most curriculum learning approaches require manually designed curricula. This work automatically generates curricula through goal decomposition, making it more generally applicable.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of first-order representations, HER, and automatic curriculum is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Three variants validated across multiple domains with honest discussion of limitations
- Writing Quality: ⭐⭐⭐⭐ Logic is clear with a well-structured progressive presentation of the three methods
- Value: ⭐⭐⭐⭐ Provides an effective paradigm for integrating symbolic AI with deep RL

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICLR 2026\] Spectral Bellman Method: Unifying Representation and Exploration in RL](../../ICLR2026/reinforcement_learning/spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](../../NeurIPS2025/reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[ICLR 2026\] Dual Goal Representations](../../ICLR2026/reinforcement_learning/dual_goal_representations.md)
- [\[AAAI 2026\] ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India](regal_a_first_look_at_ppo-based_legal_ai_for_judgment_prediction_and_summarizati.md)

<!-- RELATED:END -->
