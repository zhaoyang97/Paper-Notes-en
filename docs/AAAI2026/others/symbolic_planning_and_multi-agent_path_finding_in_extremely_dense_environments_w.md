---
title: >-
  [Paper Note] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents
description: >-
  [AAAI 2026][Block Rearrangement] This paper introduces the Block Rearrangement Problem (BRaP) as a formal problem definition and proposes five solving algorithms based on configuration space search…
tags:
  - "AAAI 2026"
  - "Block Rearrangement"
  - "MAPF"
  - "Symbolic Planning"
  - "PDDL"
  - "LaCAM"
  - "Dense Environments"
  - "Warehouse Robotics"
date: 2026-05-08
content_hash: 6fb68475126f5b6e
---

# Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents

**Conference**: AAAI 2026
**arXiv**: [2509.01022](https://arxiv.org/abs/2509.01022)  
**Code**: None  
**Area**: Other
**Keywords**: Block Rearrangement, MAPF, Symbolic Planning, PDDL, LaCAM, Dense Environments, Warehouse Robotics

## TL;DR

This paper introduces the Block Rearrangement Problem (BRaP) as a formal problem definition and proposes five solving algorithms based on configuration space search, PDDL symbolic planning, and MAPF. Among them, BR-LaCAM achieves a 92% success rate with millisecond-level solving speed on grids up to 80×80 under extreme density conditions.

## Background & Motivation

1. **Practical demand**: Amazon Robotics warehouses employ dense storage grids to maximize space utilization, but when a required shelf (block) is deeply buried, complex rearrangement plans are needed to move it out.
2. **Insufficient existing problem definitions**: Classical problems such as sliding puzzles, Rush Hour, and Sokoban share structural similarities with warehouse rearrangement, but none fully captures the constraints specific to BRaP (unassigned blocks, following conflicts, multi-target sets, etc.).
3. **Combinatorial explosion under high density**: Grids are nearly fully occupied (very few empty cells), causing configuration space to grow exponentially with the number of blocks, making traditional methods difficult to scale.
4. **Following conflicts overlooked**: Most MAPF solvers neglect following conflicts (a agent occupying the position vacated by another in the previous step), yet warehouse scenarios must account for them due to the prohibition of platooning.
5. **Coordination of unassigned agents**: A large number of goal-free blocks must be actively moved aside to make way for assigned blocks—a coordination challenge insufficiently studied in classical MAPF.
6. **Lack of baseline methods**: As a newly defined problem, BRaP requires a set of baseline algorithms to establish performance references and guide future research.

## Method

### Problem Formulation

BRaP is defined on a graph $G=(V,E)$: given a set of assigned blocks $\mathcal{I}$ (with goals) and a set of unassigned blocks $\mathcal{J}$ (without goals), each assigned block $i$ has an initial position $s_i$ and a target vertex set $V_i$. The action set is $A=\{\text{move}, \text{wait}, \text{complete}\}$, and vertex conflicts, edge conflicts, and following conflicts must all be avoided. The objective is to find a minimum-cost conflict-free path plan such that all assigned blocks reach their respective targets.

### Algorithm 1: Configuration Space Search

BRaP is modeled as a graph search over the configuration space. Each state encodes the current time, positions of assigned blocks, the set of empty vertices, and the set of completed blocks. A* search is employed with an admissible heuristic: assuming each assigned block moves along its least-obstructed path and each blocking block requires only one move to clear, the joint lower bound is computed as the average cost across all assigned blocks. Branch factor is reduced by restricting each step to a single action.

### Algorithm 2: PDDL-based Configuration Space Search

The configuration space search is encoded in PDDL. Predicates `(emp ?u)`, `(asb ?u)`, `(blk ?u)`, `(cmp ?u)`, and `(goal ?u)` represent vertex states, and three actions `move_blk`, `move_asb`, and `complete` are defined. The fast-downward planner is used to generate sequential single-action plans.

### Algorithm 3: Priority-based Configuration Space Search

This variant permits each assigned block to execute one action per step, enabling parallel execution. Priorities are assigned based on heuristic distance, and plans are generated sequentially for each assigned block. The plan of a higher-priority block is passed as a constraint to lower-priority blocks, ensuring conflict-free parallel execution.

### Algorithm 4: Heuristic Approach (Least-Obstruction Path)

A purely heuristic method designed for large-scale grids. For each assigned block, the least-obstructed path $U_i$ is computed, and obstructing blocks along the path are cleared to the nearest empty vertex before the assigned block advances. A vertex blocking time function $\psi$ coordinates parallel movement across multiple assigned blocks.

### Algorithm 5: BR-LaCAM (Core Contribution)

Built on the LaCAM framework, with BR-PIBT designed as the configuration generator:

- **Priority management**: A block's priority resets to a random value in $(0,1)$ upon reaching its goal; it is incremented by 1 when displaced, ensuring that repeatedly interrupted blocks eventually acquire the highest priority.
- **Recursive planning**: Blocks are processed in descending priority order; each block attempts to move toward its optimal neighbor. If the target cell is occupied, the occupant is recursively requested to yield, forming a displacement chain until an empty vertex is reached.
- **Temporary goal assignment**: Assigned blocks select temporary goals from the nearest unoccupied targets; unassigned blocks prefer to move toward empty vertices to maintain system fluidity.
- **Integration with LaCAM**: BR-PIBT generates single-step successor configurations; if a previously explored state is produced, the algorithm backtracks and expands low-level constraints, progressively covering all possible successors. BR-LaCAM maintains completeness and supports anytime improvement.

## Key Experimental Results

### Experimental Setup

Evaluation is conducted on 13,860 test instances with grid sizes ranging from 4×10 to 80×80. The number of assigned blocks accounts for at most 12.5% of all vertices, and empty vertices account for at most 25%. Three goal types are tested: Goal B (boundary goals), Goal R1 (random goals, count = number of assigned blocks), and Goal R2 (random goals, count = 2× number of assigned blocks). The time limit is 10 seconds per instance.

### Table 1: Action Cost Matrix

| Block Type | move | wait | complete |
|:---:|:---:|:---:|:---:|
| Assigned | 2 | 1 | 2 |
| Unassigned | 2 | 0 | N/A |

### Table 2: Success Rates and Cost Ratios by Grid Size

| Grid Size | BR-LaCAM | Heuristic | Priority | Config | PDDL |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4×10 | **100%** | 93% | 91% | 87% | 89% |
| 6×10 | **100%** | 97% | 87% | 76% | 76% |
| 10×10 | **100%** | 96% | 60% | 42% | 34% |
| 20×20 | **100%** | 95% | 15% | 7% | 4% |
| 40×40 | **99%** | 91% | 8% | 3% | 0% |
| 80×80 | **92%** | 80% | 2% | 0% | 0% |

The overall mean cost ratio of BR-LaCAM is 1.02 (std. 0.21), while Heuristic achieves 1.18 (std. 0.23); the remaining algorithms almost completely fail on large grids.

### Table 3: Success Rates by Goal Type

| Goal Type | BR-LaCAM | Heuristic | Priority | Config | PDDL |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Goal B (Boundary) | **100%** | 97% | 56% | 49% | 45% |
| Goal R1 (Strict Random) | **98%** | 82% | 33% | 22% | 22% |
| Goal R2 (Relaxed Random) | **99%** | 100% | 56% | 47% | 44% |

Goal R1 is the most challenging setting; BR-LaCAM exhibits higher cost variance (std. 0.34) in this scenario, reflecting cost fluctuations caused by the myopic behavior of BR-PIBT when goals are sparsely distributed.

## Key Findings

- BR-LaCAM achieves the best performance across all scales and goal types, with an overall success rate of 99%, and is the only algorithm capable of reliably handling 80×80 grids.
- The Heuristic method is the only competitive alternative to BR-LaCAM (93% success rate), but incurs approximately 16% higher cost ratio.
- Configuration Space Search and PDDL methods almost completely fail on grids larger than 20×20, confirming the severity of exponential explosion in configuration space.
- Increasing the number of empty vertices substantially reduces search depth, improves success rates, and lowers solution cost.
- Both BR-LaCAM and Heuristic find initial solutions within 1 millisecond in over 50% of instances, and solve over 90% of instances within 1 second.

## Highlights & Insights

- **First formal definition of BRaP**: The warehouse shelf rearrangement problem is rigorously formulated as a graph search problem, establishing clear connections to and distinctions from classical problems such as sliding puzzles, MAPF, and MAPFUA.
- **Elegant BR-PIBT configuration generator design**: Through priority incrementing, recursive displacement chains, and temporary goal assignment, the fundamental limitation of original PIBT in handling following conflicts is resolved.
- **Completeness guarantee**: BR-LaCAM maintains completeness over the finite configuration space; failures are attributable solely to time limits rather than algorithmic deficiencies.
- **Industrial-scale validation**: Systematic evaluation on 13,860 instances covering grids from 4×10 to 80×80 demonstrates practical deployability.
- **Anytime property**: An initial solution is produced rapidly and continuously improved thereafter, balancing real-time responsiveness with solution quality.

## Limitations & Future Work

- Each step is restricted to a single action (Config/PDDL) or one action per block (Priority), leaving parallel execution underutilized.
- The myopic behavior of BR-PIBT leads to high cost variance when goals are sparsely distributed (std. up to 0.34 in the Goal R1 setting).
- Dynamic scenarios are not considered: all block positions and goals are assumed known prior to planning, with no support for online task arrivals.
- Only one termination condition is tested for goal types (transition to obstacle); the other two variants (transition to empty vertex / transition to unassigned) are not evaluated experimentally.
- The heuristic function is relatively simple (independent per-block estimates averaged together) and may be overly loose when assigned blocks are highly coupled.
- Obstacles are modeled as a fixed rectangular block region in the lower-right corner; more complex layout configurations are not tested.

## Related Work & Insights

- **Sliding puzzles and Rush Hour**: Classical combinatorial search problems; BRaP extends them by introducing multiple assigned blocks and unassigned blocks (Gozon & Yu 2024, Cian et al. 2022).
- **PDDL symbolic planning**: Automatic planning via PDDL state and action descriptions, previously applied to logistics, robotic path planning, and related domains (Helmert 2009, Davesa Sureda et al. 2024).
- **Classical MAPF**: Stern et al. 2019 defines the standard MAPF problem; LaCAM (Okumura 2023) enables fast solving in large-scale, high-density scenarios.
- **MAPFUA**: Felner & Stern 2026 define a MAPF variant with unassigned agents; BRaP is a concrete instantiation of this setting in extremely dense environments.
- **TAPF**: Ma & Koenig 2016 study joint goal assignment and path planning, but do not include unassigned agents.
- **Learning-based methods**: Reinforcement learning (Damani et al. 2021) and graph neural networks (Li et al. 2020) have been applied to MAPF, but do not specifically address the dense constraints of BRaP.

## Rating

- Novelty: ⭐⭐⭐⭐ First formal definition of BRaP with clear connections to MAPF/MAPFUA
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13,860 instances with systematic coverage of diverse parameter combinations
- Writing Quality: ⭐⭐⭐⭐ Rigorous problem definition and clear algorithmic descriptions
- Value: ⭐⭐⭐⭐ Direct applicability to warehouse logistics automation; BR-LaCAM is practically deployable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)
- [\[AAAI 2026\] Finding Diverse Solutions Parameterized by Cliquewidth](finding_diverse_solutions_parameterized_by_cliquewidth.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning](bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)

</div>

<!-- RELATED:END -->
