---
title: >-
  [Paper Note] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning
description: >-
  [ICLR 2026][Robotics][Abstract world models] This paper proposes ExoPredicator, a framework that jointly learns symbolic state abstractions and causal processes (encompassing both endogenous actions and exogenous mechani…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Abstract world models"
  - "exogenous causal processes"
  - "temporal planning"
  - "predicate invention"
  - "variational inference"
  - "LLM guidance"
date: 2026-05-08
content_hash: 59df838c192fc51d
---

# ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning

**Conference**: ICLR 2026
**arXiv**: [2509.26255](https://arxiv.org/abs/2509.26255)  
**Code**: N/A  
**Area**: Robot Planning / World Models
**Keywords**: Abstract world models, exogenous causal processes, temporal planning, predicate invention, variational inference, LLM guidance

## TL;DR
This paper proposes ExoPredicator, a framework that jointly learns symbolic state abstractions and causal processes (encompassing both endogenous actions and exogenous mechanisms). Via variational Bayesian inference combined with LLM-based proposals, ExoPredicator learns causal world models with stochastic delays from a small number of trajectories, achieving rapid generalization in planning across five tabletop robot environments.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: 1. In long-horizon embodied planning, the world changes not only due to agent actions but also through **exogenous processes** (e.g., water heating, domino cascades) that unfold concurrently with agent actions.
2. Existing abstract world models (e.g., STRIPS) assume instantaneous actions and cannot model delayed effects or autonomous exogenous processes.
3. Classical PDDL does not directly support exogenous processes (requiring the PDDL+ extension), and suffers from combinatorial explosion when discretizing time.
4. VLM/VLA models lack compositional world models and generalize poorly to novel tasks.
5. **Core Problem**: How can one learn world models that simultaneously abstract the state space and the temporal progression of causal processes?

## Method

### State Abstraction
- Predicates are implemented as Python functions with VLM queries, mapping raw states to sets of Boolean features (grounded atoms).
- Candidate predicates are proposed by an LLM, and a greedy search selects the optimal subset.

### Causal Process Representation
Each causal process $L = \langle \text{Par}, C, O, E, W, p^{\text{delay}} \rangle$:
- **Endogenous processes**: Actions directly controlled by the agent (e.g., Pick/Place/SwitchOn).
- **Exogenous processes**: Processes that unfold autonomously once their preconditions are satisfied (e.g., kettle filling, domino cascades).
- Each process has a stochastic delay distribution $p^{\text{delay}}$; effects are realized only after the delay.
- NoOp action: waiting for exogenous processes to complete.

### Learning Algorithm (Three Nested Levels)
1. **Parameter learning**: Variational inference optimizes delay distributions and weight parameters, introducing a variational distribution to approximate the arrival times of causal effects.
2. **Structure learning**: LLM proposes candidate process structures → Bayesian model selection scoring.
3. **Predicate invention**: LLM proposals + greedy local search, optimizing data likelihood combined with a model prior.

### Planning
- "Big-step" transition function $\mathcal{T}_{\text{big}}$: skips time intervals during which the abstract state remains unchanged.
- A* search with a fast-forward heuristic.

## Key Experimental Results

### Five PyBullet Environments

| Environment | Description | Exogenous Processes |
|-------------|-------------|---------------------|
| Coffee | Coffee machine produces coffee → poured into cup | Coffee dispensing, pouring |
| Grow | Watering plants (color matching) | Plant growth |
| Boil | Filling water + boiling (avoiding overflow) | Filling, heating, overflow |
| Domino | Pushing domino cascade | Inter-domino collisions |
| Fan | Using fan to blow ball to target | Wind moving ball |

### Main Results

| Method | Coffee | Grow | Boil | Domino | Fan |
|--------|--------|------|------|--------|-----|
| Manual | ~100% | ~70% | ~85% | ~80% | ~100% |
| **Ours** | ~100% | ~85% | ~80% | ~90% | ~100% |
| ViLa-fs | ~80% | ~30% | ~30% | ~10% | ~20% |
| MAPLE | ~20% | ~10% | ~5% | ~5% | ~5% |
| VisPred | ~60% | ~20% | ~20% | ~15% | ~30% |

- ExoPredicator consistently outperforms VLM planning, HRL, and STRIPS learning baselines across all domains.
- Convergence is achieved with only 1–2 demonstrations and at most 3 rounds of online learning.
- On Grow and Domino, the method even surpasses the hand-designed Manual baseline, attributed to better delay parameters learned via variational inference.

### Ablation Study
- No LLM: The search space reaches $2^{50}$, rendering the approach infeasible.
- No Bayes: Full reliance on the LLM prior alone proves unreliable.
- Manual-d (hand-crafted abstraction without delay learning): Near-zero performance, demonstrating the critical role of delay parameter learning.

## Highlights & Insights
- This is the first framework to jointly learn symbolic state abstractions and exogenous causal processes (including stochastic delays).
- Variational inference elegantly addresses the combinatorial explosion in temporal reasoning over causal effect sequences.
- The LLM proposal + Bayesian scoring search strategy balances efficiency and reliability.
- Generalizable world models are learned from very few demonstrations (1–2), and generalize to more objects and more complex goals.

## Limitations & Future Work
- Validation is limited to PyBullet tabletop environments and has not been extended to large-scale or high-noise settings.
- Learning of exogenous process preconditions is incomplete (e.g., disjunctive overflow conditions in Boil).
- The framework relies on predefined closed-loop skills (Pick/Place, etc.) and does not learn the skills themselves.
- Predicate proposal quality depends on the LLM, and the approach may fail for rare physical phenomena.

## Related Work & Insights
- **PDDL/PDDL+**: Requires manual authoring of temporal planning domain descriptions; this work learns them automatically.
- **HRL (MAPLE)**: Does not explicitly model exogenous dynamics, making exploration difficult.
- **VLM planning (ViLA)**: Lacks a world model, resulting in poor compositional generalization.
- **VisualPredicator**: Learns STRIPS-style operators but does not model delays or exogenous processes.
- **Causal RL**: Constructs causal graphs at the feature level, without addressing symbolic abstraction or temporal processes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[ICLR 2026\] AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception](anytouch_2_general_optical_tactile_representation_learning_for_dynamic_tactile_p.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICML 2026\] Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](../../ICML2026/robotics/plan_in_sandbox_navigate_in_open_worlds_learning_physics-grounded_abstracted_exp.md)

</div>

<!-- RELATED:END -->
