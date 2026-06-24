---
title: >-
  [Paper Note] Hybrid Quantum-Classical Multi-Agent Pathfinding
description: >-
  [ICML 2025][Autonomous Driving][MAPF] Proposed the first optimal hybrid quantum-classical MAPF algorithms, QP and QCP, converting the path selection problem of MAPF into QUBO subproblems solvable on quantum hardware. By utilizing a conflict graph and column generation framework, theoretical optimality is achieved, and feasibility is validated on real quantum hardware.
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "MAPF"
  - "QUBO"
  - "Quantum Computing"
  - "Column Generation"
  - "Branch-and-Cut-and-Price"
date: 2026-05-08
content_hash: 206dc22cf0458de8
---

# Hybrid Quantum-Classical Multi-Agent Pathfinding

**Conference**: ICML 2025  
**arXiv**: [2501.14568](https://arxiv.org/abs/2501.14568)  
**Area**: Autonomous Driving  
**Keywords**: MAPF, QUBO, Quantum Computing, Column Generation, Branch-and-Cut-and-Price  

## TL;DR

Proposed the first optimal hybrid quantum-classical MAPF algorithms, QP and QCP, converting the path selection problem of MAPF into QUBO subproblems solvable on quantum hardware. By utilizing a conflict graph and column generation framework, theoretical optimality is achieved, and feasibility is validated on real quantum hardware.

## Background & Motivation

### Background

**Background**: Background**: MAPF is an NP-hard problem, which is crucial in large-scale applications such as UAV traffic. Classical optimal solvers like CBS and BCP are limited on large-scale instances.

### Limitations of Prior Work

**Limitations of Prior Work**: Limitations of Prior Work**: Quantum computing holds promise for breaking the bottlenecks of NP-hard problems, but existing QUBO-MAPF methods either use edge-based representations that lead to problem sizes far exceeding current quantum hardware capabilities, or lack theoretical optimality guarantees.

### Key Challenge

**Key Challenge**: Key Challenge**: The limited qubit count and high error rates of NISQ-era quantum hardware vs. the MAPF problem requiring an exponential number of variables.

### Approach

**Approach**: Goal**: Design quantum algorithms for MAPF that can run on current quantum hardware and possess theoretical optimality guarantees.

### Additional Note

**Additional Note**: Key Insight**: Convert the high-level ILP problem in the BCP framework into a compact QUBO problem, reducing the number of constraints through conflict graphs.

### Additional Note

**Additional Note**: Core Idea**: Achieve optimal MAPF solvable on quantum hardware through iterative path set expansion + conflict graph QUBO solving + theoretical optimality determination criteria.

## Method

### Overall Architecture

A two-level iterative optimization is adopted: the outer loop (Separation) identifies path conflicts and adds constraints; the inner loop (Pricing) generates new paths and solves the Restricted Master Problem (RMP) via QUBO. When the pricing criterion is met, the solution is globally optimal.

### Key Designs

**Design 1: Path-based QUBO Formulation**
- **Function**: Encode each potential path as a binary variable, embedding constraints into the QUBO matrix via a conflict graph.
- **Mechanism**: Connect edges between two conflicting paths in the conflict graph, converting inequality constraints into quadratic penalty terms. The conflict graph can be decomposed into independent subproblems to be solved in parallel.
- **Design Motivation**: Avoid directly handling an exponential number of constraints, as conflict graphs provide a compact representation.

**Design 2: Optimality Determination Criterion**
- **Function**: Prove when to stop adding new paths, guaranteeing that the current solution is optimal.
- **Mechanism**: A reduced cost criterion based on Lagrangian duality, relaxing the non-negativity assumption in classical column generation.
- **Design Motivation**: Prior quantum MAPF methods lacked stopping criteria and could only rely on heuristics.

**Design 3: Hardware-Aware QUBO Decomposition**
- **Function**: Decompose the QUBO problem into independent subproblems that can be efficiently solved on quantum annealers.
- **Mechanism**: Utilize connected components of the conflict graph to identify independent subproblems, each of which can be solved in parallel on different quantum processors.
- **Design Motivation**: Adapt to the limited qubit count and topological constraints of current quantum hardware.

### Loss & Training

Objective function: $\min \sum_{a \in A} \sum_{p \in \mathcal{P}_a} c_p z_p$, minimizing the sum of all path lengths, constraining each agent to choose exactly one path without conflict.

## Key Experimental Results

### Main Results

| Method | 5 agents | 10 agents | Optimality |
|------|----------|-----------|--------|
| Prior QUBO | Timeout | Timeout | No |
| QP (Ours) | ✓ | ✓ | Yes |
| QCP (Ours) | ✓ | ✓ | Yes |
| CBS | ✓ | Partial | Yes |
| BCP | ✓ | ✓ | Yes |

### Ablation Study

| Component | Impact |
|------|------|
| Conflict Graph Decomposition | Reduces QUBO size by 60-80% on average |
| Cut Step (QCP vs QP) | QCP converges faster but introduces more constraints |
| Physical Quantum Hardware vs. Simulation | D-Wave annealer can handle small instances |

### Key Findings

1. QP/QCP dominates prior QUBO methods on benchmark data and achieves solution quality comparable to classical BCP.
2. Conflict graph decomposition is key to making the problem compatible with current quantum hardware—reducing the required qubit count to an acceptable range.
3. Small-scale instances were successfully run on the D-Wave quantum annealer, validating the practicality of the framework.

## Highlights & Insights

1. The first quantum MAPF algorithm with theoretical optimality guarantees, filling an important gap.
2. The framework design is modular, compatible with more powerful future quantum hardware.
3. The combination of conflict graph and column generation cleverly dissolves large problems into quantum-solvable small problems.

## Limitations & Future Work

1. Current quantum hardware can only handle small-scale instances; practical large-scale applications still await hardware progress.
2. Online replanning scenarios in dynamic environments are not considered.
3. Branching strategies are not yet deeply integrated with quantum computing.

## Related Work & Insights

- BCP (Lam et al.) is currently the strongest classical optimal MAPF solver, and this paper quantizes it.
- The QUBO-MAPF by Martín & Martin lacks optimality guarantees, which this paper resolves via the pricing criterion.
- Insight: The hybrid quantum-classical paradigm may be the correct path for the NISQ era—using quantum to solve core subproblems, while classical algorithms manage the framework and decision-making.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ★★★★★ |
| Practicality | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)
- [\[CVPR 2026\] Unsupervised Multi-agent and Single-agent Perception from Cooperative Views](../../CVPR2026/autonomous_driving/unsupervised_multi-agent_and_single-agent_perception_from_cooperative_views.md)
- [\[ICCV 2025\] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](../../ICCV2025/autonomous_driving/srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)
- [\[CVPR 2025\] Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels](../../CVPR2025/autonomous_driving/learning_to_detect_objects_from_multi-agent_lidar_scans_without_manual_labels.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
