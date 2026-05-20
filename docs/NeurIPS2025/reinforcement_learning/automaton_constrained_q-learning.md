---
title: >-
  [Paper Note] Automaton Constrained Q-Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][LTL] This paper proposes ACQL (Automaton Constrained Q-Learning), which translates Linear Temporal Logic (LTL) task specifications into automata and combines goal-conditioned learni…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "LTL"
  - "Automaton"
  - "Safety Constraints"
  - "Goal-Conditioned RL"
  - "CMDP"
date: 2026-05-08
content_hash: 2d0834d03a19597b
---

# Automaton Constrained Q-Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.05061](https://arxiv.org/abs/2510.05061)  
**Code**: None  
**Area**: Reinforcement Learning / Safe RL
**Keywords**: LTL, Automaton, Safety Constraints, Goal-Conditioned RL, CMDP

## TL;DR

This paper proposes ACQL (Automaton Constrained Q-Learning), which translates Linear Temporal Logic (LTL) task specifications into automata and combines goal-conditioned learning with minimal safety constraints. ACQL is the first scalable method to simultaneously support sequential temporal goals and non-stationary safety constraints in continuous control environments.

## Background & Motivation

### Starting Point

**Key Insight**: Real-world robotic tasks require achieving **sequential sub-goals** while adhering to **dynamic safety constraints** (e.g., a warehouse robot must restock shelves in order while avoiding obstacles and remaining within battery range).

### Root Cause

**Key Challenge**: Goal-conditioned RL (GCRL) can reach individual goals but cannot reason over goal sequences or enforce safety; safe RL can only handle **static** safety constraints.

### State of the Field

**Background**: LTL can express complex temporal tasks, but existing LTL+RL methods rely on sparse binary rewards and perform poorly in complex continuous environments.

### Limitations of Prior Work

**Limitations of Prior Work**: There is no scalable method that simultaneously supports temporal goals and safety constraints.

## Method

### Overall Architecture

ACQL constructs an augmented product CMDP through the following steps:

1. Translate STL (Signal Temporal Logic) specifications into a Deterministic Büchi Automaton (DBA).
2. Extract a **safety constraint map** $S: \mathcal{Q} \to \Phi$ and a **liveness condition / sub-goal map** $G: \mathcal{Q} \to \mathcal{G}^+$ from the automaton.
3. Compose the MDP with the automaton into an augmented CMDP with state space $\mathcal{S}^A = \mathcal{S} \times \mathcal{G}^+ \times \mathcal{Q}$.
4. Learn a reward Q-function $Q^r$ and a safety Q-function $Q^c$; the policy is obtained by constrained maximization: $\pi^*(s^A) = \arg\max_{a: Q^c(s^A,a) > \mathcal{L}} Q^r(s^A, a)$.

### Key Designs

1. **Goal-Conditioned Learning + HER to Address Sparse Rewards**:
    - Function: Incorporate the sub-goal list $g^+$ into the product CMDP state and apply HER to learn from failed trajectories.
    - Mechanism: Sub-goals corresponding to automaton states are explicitly encoded into observations; HER retrospectively assigns rewards for sub-goals that were reached.
    - Design Motivation: Sparse rewards are the core bottleneck of prior LTL+RL methods; GCRL techniques substantially mitigate this issue.

2. **Minimal Safety Constraint as a Replacement for Cumulative Cost**:
    - Function: Replace the standard CMDP cumulative cost constraint with a minimal safety constraint $\mathbb{E}[\min_t c^A(s_t, a_t)] > \mathcal{L}$.
    - Mechanism: Safety evaluation is recast as a classification problem (safe / unsafe) rather than regression over future cumulative costs; the safety Q-function is defined as $Q^c(s,a) = \mathbb{E}[\min_t c^A(s_t, a_t)]$ with a progressively scheduled discount factor $\gamma_c \to 1$.
    - Design Motivation: Predicting cumulative costs requires modeling long-range dependencies, leading to high variance and slow convergence; the minimal safety formulation reduces the problem to binary classification, and $\mathcal{L}=0$ applies universally across all tasks.

### Loss & Training

- Reward Q-function: Standard TD target $y^r_t = r^A_t + \gamma Q^r_{\bar\psi}(s^A_{t+1}, \pi_j(s^A_{t+1}))$

- Safety Q-function: Hamilton–Jacobi reachability-based target $y^c_t = \gamma_c \min\{c^A(s_t, a_t), Q^c_{\bar\theta}(s^A_{t+1}, \pi_j(s^A_{t+1}))\} + (1-\gamma_c) c^A(s_t, a_t)$

- $\gamma_c$ is progressively scheduled from a small value to $1.0$ (three-timescale stochastic approximation, with convergence guarantees).

## Key Experimental Results

### Main Results

| Task | Agent | CRM-RS | LOF | **ACQL** |
|------|-------|--------|-----|---------|
| Dual-goal sequential navigation | Ant | 0% success | ~20% | **~80%** |
| Branching navigation | Ant | 0% success | ~10% | **~70%** |
| Single goal + safety constraint | Quadcopter | ~30% | ~40% | **~90%** |
| Dual goal + vanishing safety | PointMass | ~50% | ~20% | **~95%** |

### Ablation Study

- Removing HER: Performance degrades substantially, confirming the critical role of sub-goals + HER in overcoming sparse rewards.
- Replacing minimal safety with cumulative cost: Safety constraint violation rates increase, and $\mathcal{L}$ requires manual tuning.
- Fixing $\gamma_c=1$ (no scheduling): Safety Q-function learning becomes unstable.

### Key Findings

- ACQL significantly outperforms CRM-RS and LOF across all tested tasks.
- Non-stationary safety constraints (e.g., "avoid region $o_1$ before reaching $g_1$, but may pass through it afterward") pose a fatal challenge for baseline methods.
- ACQL is successfully deployed on a 6-DOF robotic arm, completing goal-reaching tasks in a cabinet environment with safety constraints.

## Highlights & Insights

- **First unified framework**: Elevates Safe RL and GCRL to the task class expressible by LTL.
- The minimal safety constraint is more concise and general than cumulative cost ($\mathcal{L}=0$ applies to all tasks) and yields more stable learning.
- A single goal-conditioned policy handles all sub-goals, unlike LOF which requires training a separate policy for each edge.
- Convergence is theoretically guaranteed via three-timescale stochastic approximation.

## Limitations & Future Work

- Evaluation is limited to recurrence-class LTL formulas; more general LTL fragments are not covered.
- Obstacles in the experimental environments are introduced solely through task specifications (no physical obstacles), which may limit realism.
- The method is restricted to the online RL setting; zero-shot generalization to new LTL tasks is not addressed.
- Scalability to high-dimensional continuous observation spaces (e.g., visual inputs) remains unverified.

## Related Work & Insights

- Reward Machines (CRM-RS) handle safety implicitly via episode termination, which lacks robustness.
- LOF requires training an independent policy for each sub-goal, limiting scalability.
- The minimal safety idea, drawn from Hamilton–Jacobi reachability analysis, elegantly bridges formal methods and deep RL.

## Rating

- Theoretical Innovation: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[AAAI 2026\] Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking](../../AAAI2026/reinforcement_learning/constrained_and_robust_policy_synthesis_with_satisfiability-modulo-probabilistic.md)
- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
