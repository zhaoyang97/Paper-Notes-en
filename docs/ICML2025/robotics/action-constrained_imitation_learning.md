---
title: >-
  [Paper Note] Action-Constrained Imitation Learning
description: >-
  [ICML2025][Robotics][Imitation Learning] Formulates a new problem of "Action-Constrained Imitation Learning (ACIL)" where a constrained agent learns from an unconstrained expert; proposes DTWIL, which generates alternative constrained trajectories via MPC and DTW distance to eliminate occupancy measure mismatch, outperforming baselines significantly on various robotic tasks.
tags:
  - "ICML2025"
  - "Robotics"
  - "Imitation Learning"
  - "Action Constraints"
  - "Dynamic Time Warping"
  - "Trajectory Alignment"
  - "MPC"
date: 2026-05-08
content_hash: 2c129623d1815245
---

# Action-Constrained Imitation Learning

**Conference**: ICML2025  
**arXiv**: [2508.14379](https://arxiv.org/abs/2508.14379)  
**Code**: [GitHub - ACRL-Baselines](https://github.com/NYCU-RL-Bandits-Lab/ACRL-Baselines)  
**Area**: Reinforcement Learning  
**Keywords**: Imitation Learning, Action Constraints, Dynamic Time Warping, Trajectory Alignment, MPC

## TL;DR
Formulates a new problem of "Action-Constrained Imitation Learning (ACIL)" where a constrained agent learns from an unconstrained expert; proposes DTWIL, which generates alternative constrained trajectories via MPC and DTW distance to eliminate occupancy measure mismatch, outperforming baselines significantly on various robotic tasks.

## Background & Motivation

### New Problem Definition
Traditional imitation learning assumes that the expert and the imitator share the same action space. However, capability gaps are common in practice: for instance, learning to control a low-power robotic arm (constrained torque) by imitating demonstrations from a high-power robotic arm (unconstrained).

### Fundamental Challenge of Occupancy Measure Mismatch
When expert actions exceed the feasible set of the imitator, action projection leads to severe trajectory deviations, known as "occupancy measure distortion." For instance, in maze navigation, a constrained agent may fail to turn in time, leading to collisions.

### Limitations of Prior Work
- ACRL methods use projection layers to enforce constraints, but in IL, projection leads to trajectory deviation.
- LfO methods ignore the capability gap and attempt to replicate infeasible trajectories.

## Method

### DTWIL Framework (Two Stages)

**Stage 1: Generate Constrained Alternative Demonstrations**
- For each expert trajectory, MPC is used to generate an alternative trajectory that respects constraints and maintains similar state sequences.
- Key point: The alternative trajectory may be longer than the expert trajectory (requiring more steps to complete the same action).

**Stage 2: Learn from Alternative Demonstrations using Arbitrary IL Methods**
- Either BC or Inverse RL can be applied, as the framework decouples constraint satisfaction from the imitation learning algorithm.

### MPC Trajectory Alignment
Reformulates trajectory alignment as a planning problem:
- At each step, MPC is used to solve a finite-horizon subproblem.
- Candidate rollouts are generated using a learned dynamics model.
- The rollout with the minimal DTW distance is selected.
- Only the first action is executed, enabling step-by-step adaptation.

### DTW as the Alignment Criterion
DTW naturally handles sequences of different lengths by "warping" time to align state sequences. It is recursively solved as:
$$d_{DTW}(\sigma_{0:i}, \sigma'_{0:j}) = ||\sigma_i - \sigma'_j||_2 + \min\{d_{DTW}(\sigma_{0:i-1}, \sigma'_{0:j}), ...\}$$

### Progress Parameter Tracking
Introduces a progress parameter `t_pg` to track the current alignment position on the expert trajectory, allowing the agent to use more steps to accomplish the same state transitions.

## Key Experimental Results

### MuJoCo Locomotion Tasks

| Task | BC | GAIL | BCO | DTWIL-BC | DTWIL-GAIL |
|------|-----|------|-----|----------|-----------|
| HalfCheetah | 32.1 | 45.3 | 38.7 | **78.5** | **82.3** |
| Walker2d | 28.4 | 41.2 | 35.6 | **72.8** | **76.1** |
| Hopper | 35.2 | 48.7 | 42.3 | **81.4** | **85.2** |

### Maze2d Navigation Tasks

| Constraint Severity | BC Projection | LfO | DTWIL |
|---------|--------|-----|-------|
| Mild (80% action range) | 85.2% | 72.3% | **95.1%** |
| Moderate (50%) | 52.1% | 45.6% | **82.4%** |
| Severe (20%) | 12.3% | 18.7% | **61.5%** |

### Key Findings
1. The tighter the constraints, the larger the relative advantage of DTWIL.
2. DTWIL is robust to downstream IL methods (both BC and GAIL benefit).
3. Alternative trajectories are typically 20-50% longer than expert trajectories but feature highly similar state sequences.
4. The quality of the MPC dynamics model determines the performance upper bound.

## Highlights & Insights

1. ACIL provides a clear and practical definition of a new problem.
2. Modeling trajectory alignment as a planning problem is an elegant conceptual transition.
3. DTW naturally addresses the requirement of "using more steps to complete the same action."
4. The two-stage decoupled design makes the framework compatible with any IL method.
5. It consistently outperforms projection methods across all constraint severities, especially under severe constraints.

## Limitations & Future Work

1. Learning a dynamics model is required, making data efficiency and model accuracy the bottlenecks.
2. Generating candidate rollouts in MPC can be inefficient in high-dimensional action spaces.
3. The DTW distance only considers state similarity without explicitly constraining action smoothness.
4. Adaptive determination of the optimal alternative trajectory length $K^*$ still requires better strategies.
5. While thoroughly validated in continuous control scenarios, the discrete action space remains to be tested.

## Related Work & Insights

- Fundamental difference from ACRL: ACRL operates with reward functions and can learn via trial-and-error, whereas ACIL only has demonstrations and suffers from capability gaps.
- Difference from LfO: LfO does not consider trajectory infeasibility.
- Insights: DTW alignment can be extended to cross-morphology imitation learning (cross-morphology IL).

## Rating
- Novelty: 5.0/5 — New problem + new framework
- Experimental Thoroughness: 4.5/5 — Multiple tasks and constraint severities
- Writing Quality: 4.5/5 — Clear problem definition
- Value: 4.5/5 — Direct significance for safe robot deployment

## Supplementary Technical Details

### Application of CEM Optimizer in MPC
When MPC generates candidate rollouts, the Cross-Entropy Method (CEM) is adopted to iteratively optimize the action sequence distribution. In each round, $N$ candidates are sampled, and the top-$K$ candidates with the smallest DTW distance are selected to update the sampling distribution, incorporating rejection sampling to ensure all candidate actions satisfy the constraints.

### Adaptive Update of the Progress Parameter
After MPC selects the optimal trajectory at each step, the position aligned to the expert trajectory is back-tracked via the DTW alignment matrix to automatically update the progress parameter, allowing the agent to use more steps in difficult segments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Automaton Constrained Q-Learning](../../NeurIPS2025/robotics/automaton_constrained_q-learning.md)
- [\[ICML 2025\] Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism](robot-gated_interactive_imitation_learning_with_adaptive_intervention_mechanism.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](../../NeurIPS2025/robotics/safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [\[NeurIPS 2025\] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](../../NeurIPS2025/robotics/beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)
- [\[CVPR 2026\] NIL: No-data Imitation Learning](../../CVPR2026/robotics/nil_no-data_imitation_learning.md)

</div>

<!-- RELATED:END -->
