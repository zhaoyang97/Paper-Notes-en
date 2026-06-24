---
title: >-
  [Paper Note] Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking
description: >-
  [AAAI 2026][Reinforcement Learning][Markov Decision Processes] This paper proposes the first framework capable of efficiently computing robust policies under arbitrary structural constraints. By tightly integrating a SAT solver with probabilistic model checking algorithms, the framework enables constrained and robust policy synthesis for finite Markov Decision Processes (MDPs), with feasibility and competitiveness validated across hundreds of benchmarks.
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Markov Decision Processes"
  - "Robust Policy"
  - "Constraint Satisfaction"
  - "Probabilistic Model Checking"
  - "SAT Solving"
date: 2026-05-08
content_hash: 0063c7ee85999385
---

# Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking

**Conference**: AAAI 2026
**arXiv**: [2511.08078](https://arxiv.org/abs/2511.08078)  
**Code**: None  
**Area**: Reinforcement Learning / Formal Verification
**Keywords**: Markov Decision Processes, Robust Policy, Constraint Satisfaction, Probabilistic Model Checking, SAT Solving

## TL;DR

This paper proposes the first framework capable of efficiently computing robust policies under arbitrary structural constraints. By tightly integrating a SAT solver with probabilistic model checking algorithms, the framework enables constrained and robust policy synthesis for finite Markov Decision Processes (MDPs), with feasibility and competitiveness validated across hundreds of benchmarks.

## Background & Motivation

**Background**: Computing optimal reward policies for a given finite MDP is fundamental to applications in planning, controller synthesis, and verification. Standard methods such as value iteration and policy iteration can find optimal policies in polynomial time. However, practical applications impose additional requirements: policies must remain performant under perturbations of MDP parameters (robustness), and their representation or implementation cost must satisfy extra structural constraints.

**Limitations of Prior Work**: (1) Robust policy synthesis must account for uncertainty sets over MDP parameters, extending a single optimization problem into a min-max problem over the entire uncertainty set, significantly increasing computational complexity. (2) Structural constraints—such as memory size limits, determinism requirements, state aggregation, or cost budgets—transform the problem from continuous optimization to combinatorial optimization. (3) Existing methods typically handle only specific types of constraints or robustness requirements, lacking a unified framework.

**Key Challenge**: A fundamental tension exists between flexibility and efficiency—frameworks expressive enough to handle arbitrary constraints are generally computationally inefficient, while efficient methods are typically limited to specific problem fragments.

**Goal**: To design a policy synthesis framework that simultaneously achieves flexibility (supporting arbitrary structural constraints expressible in first-order theories) and efficiency (through tight integration of SAT solving and probabilistic model checking).

**Key Insight**: Drawing inspiration from SMT (Satisfiability Modulo Theories), the authors decompose constrained policy synthesis into two components: a SAT solver handles the combinatorial constraint portion, while a probabilistic model checking algorithm handles MDP analysis, with the two components interacting tightly through conflict-driven learning.

**Core Idea**: Construct a Satisfiability-Modulo-Probabilistic-Model-Checking (SM-PMC) framework that fuses the combinatorial search capability of SAT solvers with the MDP analysis capability of probabilistic model checkers, enabling flexible and efficient constrained/robust policy synthesis.

## Method

### Overall Architecture

The SM-PMC framework takes as input one or more MDPs, an optimization objective (e.g., maximizing expected reward), and structural constraints expressed in first-order theory. The framework uses a SAT solver to generate candidate policies satisfying the structural constraints, a probabilistic model checker to evaluate candidate policy performance on the MDP, and then leverages the evaluation results to generate conflict clauses fed back to the SAT solver, iterating until an optimal feasible policy is found or infeasibility is proven.

### Key Designs

1. **First-Order Theory Encoding**:

    - Function: Expresses diverse structural constraints in a unified formal language.
    - Mechanism: The state-action space of the MDP is encoded as propositional variables, and structural constraints on the policy (e.g., determinism, finite memory, cost budgets, state aggregation) are encoded as first-order logic formulas. This allows users to flexibly specify arbitrary constraint types without requiring dedicated algorithms for each.
    - Design Motivation: Existing methods such as MILP and convex optimization can only handle specific constraint forms. First-order theory offers strong expressiveness, enabling unified treatment of diverse constraint types.

2. **Tight SAT-PMC Integration**:

    - Function: Enables efficient information transfer between combinatorial search and probabilistic analysis.
    - Mechanism: The SAT solver searches the space of constraint-satisfying policies and produces candidate policies; the PMC (Probabilistic Model Checker) analyzes the probabilistic properties of the MDP under a given policy (e.g., expected reward, reachability probability). When the PMC determines that a candidate policy fails to meet performance requirements, it generates an explanatory conflict lemma that provides pruning information to the SAT solver, preventing search over similar infeasible regions.
    - Design Motivation: Compared to naive enumerate-and-verify approaches, tight integration achieves exponential search space reduction through conflict-driven learning, and is the primary source of efficiency gains.

3. **Robust Policy Synthesis Extension**:

    - Function: Finds policies with performance guarantees under MDP parameter uncertainty.
    - Mechanism: Robustness requirements are encoded as constraints over a set of MDPs (the uncertainty set)—the policy must satisfy performance requirements on all MDPs in the set. The framework integrates PMC queries over multiple MDPs into a single SAT search process, accelerating solving by sharing conflict information.
    - Design Motivation: The naive approach to robust policy synthesis solves each possible MDP separately and intersects the results, which is computationally prohibitive. SM-PMC's integrated approach allows analyses of different MDPs to provide mutual pruning information.

### Loss & Training

No neural network training is involved. The objective function is the maximization of expected reward or reachability probability on the MDP, with constraints expressed in first-order theory. Optimization is solved iteratively via the SAT solver's branch-and-bound framework.

## Key Experimental Results

### Main Results

Evaluation is conducted on hundreds of benchmarks covering two major problem classes: constrained policy synthesis and robust policy synthesis.

| Task Type | Metric | SM-PMC | Baseline | Notes |
|-----------|--------|--------|----------|-------|
| Constrained Policy Synthesis | Solve Time | Competitive | MILP, etc. | More flexible on general constraints |
| Constrained Policy Synthesis | Solve Rate | High | Specialized methods | Handles arbitrary constraint types |
| Robust Policy Synthesis | Solve Time | Competitive | SOTA methods | Comparable to optimal methods for specific fragments |
| Robust Policy Synthesis | Policy Quality | Optimal/Near-optimal | Heuristic methods | Provides exact solutions rather than approximations |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full SM-PMC | Fastest | Tight SAT+PMC integration |
| Without Conflict Learning | Orders of magnitude slower | Lacks PMC feedback pruning |
| Enumerate + Verify | Non-scalable | Naive method fails on large state spaces |
| MILP Only | Partial constraints | Handles only linear constraints; limited flexibility |

### Key Findings

- Conflict-driven learning is critical to efficiency—conflict feedback from PMC to SAT reduces the search space by orders of magnitude.
- The framework is highly flexible with respect to constraint type, handling diverse constraints that previously required dedicated algorithms.
- On robust policy synthesis, the framework is competitive with SOTA methods specifically designed for this task, while additionally supporting structural constraints.
- The scalability of the framework is primarily limited by the state space size of the underlying MDP; it performs well on medium-scale MDPs.

## Highlights & Insights

- **Flexibility of the unified framework** is the paper's primary contribution. Unifying multiple policy synthesis variants (constrained, robust, constrained+robust) within a single framework eliminates the engineering burden of designing specialized algorithms for each variant.
- **The fusion of SAT and PMC** represents a cross-domain methodological innovation—importing techniques from the formal methods (SAT/SMT) community into the planning and reinforcement learning domain, bridging disciplinary boundaries.
- The framework has abundant practical applications: robotic systems and autonomous driving planning that impose strict safety and robustness requirements on controllers can directly benefit.

## Limitations & Future Work

- The framework targets finite-state MDPs and does not directly apply to continuous state spaces or large-scale MDPs.
- Scalability is limited by the inherent complexity of SAT solvers and PMC, potentially rendering the approach impractical for very large state spaces.
- No comparison is made with deep reinforcement learning approaches to robustness, as the two classes of methods target different problem scales.
- Future work could explore integration with neural network policy parameterization, using SM-PMC for verification rather than synthesis.

## Related Work & Insights

- **vs. MILP methods**: MILP methods are efficient but restricted to linear constraints; SM-PMC supports arbitrary first-order constraints but may be slower than MILP on linear constraint subproblems.
- **vs. Robust MDP methods**: Traditional Robust MDP methods assume rectangular uncertainty sets; SM-PMC can handle more general uncertainty descriptions.
- **vs. Deep RL robust methods**: Deep RL methods scale to large continuous problems but yield only approximate solutions; SM-PMC targets medium-scale discrete problems but provides exact solutions.

## Rating

- Novelty: ⭐⭐⭐⭐ The integrated SAT-PMC framework is an innovative methodological contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ Hundreds of benchmarks covering multiple problem variants
- Writing Quality: ⭐⭐⭐⭐ Rigorous formal presentation with clearly defined problem statements
- Value: ⭐⭐⭐⭐ Provides a practical tool for policy synthesis in safety-critical systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SafeMPO: Constrained Reinforcement Learning via Probabilistic Incremental Improvement](../../ICLR2026/reinforcement_learning/safempo_constrained_reinforcement_learning_with_probabilistic_incremental_improv.md)
- [\[ICLR 2026\] Boolean Satisfiability via Imitation Learning](../../ICLR2026/reinforcement_learning/boolean_satisfiability_via_imitation_learning.md)
- [\[ICLR 2026\] Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration](../../ICLR2026/reinforcement_learning/off-policy_safe_reinforcement_learning_with_cost-constrained_optimistic_explorat.md)
- [\[AAAI 2026\] G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation](g-ubs_towards_robust_understanding_of_implicit_feedback_via_group-aware_user_beh.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](../../ICLR2026/reinforcement_learning/model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)

</div>

<!-- RELATED:END -->
