---
title: >-
  [Paper Note] Learning to Solve Orienteering Problem with Time Windows and Variable Profits
description: >-
  [ICLR 2026][Optimization][Orienteering Problem] This paper proposes DeCoST, a learning-based two-stage framework that decouples the coupled discrete routing decisions and continuous service time allocation in OPTWVP. The…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Orienteering Problem"
  - "Time Windows"
  - "Variable Profits"
  - "Discrete-Continuous Decoupling"
  - "Linear Programming"
date: 2026-05-08
content_hash: cb6dccb371a83f1e
---

# Learning to Solve Orienteering Problem with Time Windows and Variable Profits

**Conference**: ICLR 2026
**arXiv**: [2603.06260](https://arxiv.org/abs/2603.06260)  
**Code**: [GitHub](https://github.com/SwonGao/DeCoST)  
**Area**: Combinatorial Optimization / Vehicle Routing
**Keywords**: Orienteering Problem, Time Windows, Variable Profits, Discrete-Continuous Decoupling, Linear Programming

## TL;DR

This paper proposes DeCoST, a learning-based two-stage framework that decouples the coupled discrete routing decisions and continuous service time allocation in OPTWVP. The first stage employs a parallel decoder to jointly generate routes and initial service times, while the second stage applies LP to optimally allocate service times (globally optimal). Cross-stage coordination is achieved via pTAR feedback. DeCoST achieves optimality gaps of only 0.83%–3.31% on OPTWVP instances with 50–500 nodes, with inference up to 45× faster than metaheuristics.

## Background & Motivation

**Background**: The Orienteering Problem (OP) is an important variant of VRP that requires selecting a subset of nodes to visit within a fixed time budget to maximize collected rewards. In practical settings—factory scheduling, logistics, and robot planning—rewards typically grow with service duration (variable profits), and nodes are accessible only within specific time windows. This motivates the OPTWVP, which requires jointly optimizing discrete routing and continuous service time allocation.

**Limitations of Prior Work**:

1. **Discrete-Continuous Coupling**: Route changes reshape the feasible service time intervals for all nodes, which in turn affects rewards and routing decisions—bidirectional dependencies cause exponential growth in the joint search space.
2. **Limitations of NCO Methods**: Existing neural combinatorial optimization (NCO) methods handle only discrete route selection and lack the capability to allocate continuous variables (i.e., service times).
3. **Myopia of Decomposition Methods**: Naive decomposition causes first-stage routing decisions to be made without foresight of optimal service times, and subsequent local adjustments are insufficient to correct structural bias.
4. **Scalability of Metaheuristics**: Manually designed heuristic rules combined with exhaustive local search require re-solving the continuous subproblem upon each route change, incurring high computational cost.

**Key Challenge**: Discrete routes and continuous service times are inherently interdependent, yet joint optimization is computationally intractable, while naive decoupling leads to myopic decisions.

**Goal**: Intelligent decoupling with cross-stage coordination. The first stage is service-time-aware (parallel decoding of routes and times); the second stage optimizes exactly (globally optimal via LP); pTAR feedback enables the first stage to anticipate second-stage consequences.

## Method

### Overall Architecture

DeCoST formulates OPTWVP as a constrained Markov decision process (CMDP), where the action space comprises both discrete (node selection) and continuous (normalized service time ratio) components:

$$a_i = (v^{(i)}, d_{v^{(i)}}/d_{v^{(i)}\max})$$

The overall objective is to maximize the expected total reward:

$$\max_{\pi_\theta} J_r(\pi_\theta) = \mathbb{E}_{(\tau,\mathbf{d})\sim\pi_\theta}[R(\tau, \mathbf{d}|\mathcal{G})]$$

where $R(\tau, \mathbf{d}|\mathcal{G}) = \sum_{i=1}^{l-1} p_{v^{(i)}} d_{v^{(i)}}$, subject to time budget, service time upper bound, and time window constraints.

### Key Designs 1: Parallel Decoding in the First Stage

The first stage adopts a parallel decoder architecture:

- **Route Decoder**: Attention-based mechanism for selecting the next node to visit.
- **Service Time Decoder (STD)**: Simultaneously predicts the initial service time allocation for each selected node.
- **Spatial Encoding**: Encodes edge features (inter-node distances) as attention biases to enhance graph structure comprehension.
- **Feasibility Masking**: Dynamically excludes candidate nodes that violate constraints—(i) nodes from which the agent cannot return to the depot within the time limit; (ii) nodes whose arrival time exceeds their time window.

The parallel structure ensures that routing decisions are made with awareness of service time impacts, avoiding the myopia of pure route-oriented decoding.

### Key Designs 2: LP-Based Global Optimality in the Second Stage

Given a fixed route $\tau$ from the first stage, service time allocation reduces to a linear program:

$$\max_{\mathbf{s},\mathbf{d}} \mathbf{p}^T \mathbf{d}, \quad \text{s.t.} \quad \mathbf{s}^- \leq \mathbf{s} \leq \mathbf{s}^+, \quad \mathbf{0} \leq \mathbf{d} \leq \mathbf{d}_{\max}, \quad (I-U)\mathbf{s} + \mathbf{d} + \mathbf{t} \leq \mathbf{0}$$

An STO algorithm is proposed to solve this via a greedy strategy: nodes are sorted in descending order of unit profit, and the maximum feasible service time is allocated to each node sequentially while updating the earliest start times of subsequent nodes. **The global optimality of STO is rigorously proven** (Theorem 4.1), and the algorithm supports parallel computation.

### Key Designs 3: pTAR Cross-Stage Supervisory Feedback

The profit-weighted Time Allocation Ratio (pTAR) measures the profit efficiency of a solution:

$$pTAR(\mathbf{d}) = \sum_{i \in \tau} \frac{p_i d_i}{t_i}$$

A repulsive supervision loss is introduced to prevent the STD from prematurely converging to the LP-conditional optimum (which varies with different routes):

$$\mathcal{L}_{pTAR} = -(pTAR(\hat{\mathbf{d}}) - pTAR(\mathbf{d}^*))^2$$

The total loss is a weighted combination of the REINFORCE loss and the pTAR loss: $\mathcal{L}_{total} = \beta_1 \mathcal{L} + \beta_2 \mathcal{L}_{pTAR}$.

## Key Experimental Results

### Main Results: OPTWVP Performance at Different Scales

| Method | Category | n=50, TW100 Score (Gap) | n=100, TW100 Score (Gap) | n=50, TW500 Score (Gap) |
|--------|----------|------------------------|--------------------------|------------------------|
| B&C (Gurobi) | Exact | 15.2 (0.00%) | 26.1 (0.00%) | 51.9 (0.00%) |
| Greedy-PRS | Heuristic | 12.9 (15.7%) | 21.9 (16.2%) | 46.1 (11.4%) |
| ILS | Metaheuristic | 14.5 (4.34%) | 24.9 (4.2%) | 47.8 (7.82%) |
| POMO | NCO | 11.3 (25.3%) | 11.5 (55.7%) | 31.3 (39.6%) |
| GFACS | NCO | 12.3 (18.6%) | 25.2 (3.38%) | 47.4 (8.57%) |
| **DeCoST** | **NCO** | **15.1 (1.06%)** | **25.6 (1.97%)** | **51.4 (0.83%)** |

DeCoST achieves the smallest optimality gaps across all settings (0.83%–3.31%), substantially outperforming other NCO and metaheuristic methods. POMO performs worst (gap up to 55.7%) due to its inability to handle service times.

### Large-Scale Experiments (n=500, TW=100)

| Method | Score | Gap | Runtime (ms) |
|--------|-------|-----|-------------|
| B&C (Gurobi) | 82.3 | 0.00% | 68,400 |
| ILS | 78.2 | 4.98% | 8,803 |
| POMO | 58.6 | 28.8% | 747 |
| **DeCoST** | **79.6** | **3.31%** | **1,329** |

At the 500-node scale, DeCoST maintains a 3.31% optimality gap while being 6.6× faster than ILS and 51× faster than Gurobi.

### Ablation Study (n=50, TW=100)

| Configuration | SE | STO | SL | Score | Gap |
|--------------|----|----|-----|-------|-----|
| Baseline (POMO) | ✗ | ✗ | ✗ | 11.3 | 25.3% |
| + Spatial Encoding | ✓ | ✗ | ✗ | 11.6 | 23.0% |
| + STO | ✓ | ✓ | ✗ | 14.9 | 2.28% |
| **DeCoST (Full)** | **✓** | **✓** | **✓** | **15.1** | **1.06%** |

STO is the most critical component, reducing the gap sharply from 23.0% to 2.28%; the pTAR supervision further reduces it to 1.06%.

## Highlights & Insights

- **"Decouple but Coordinate" Design Philosophy**: Rather than naively separating the two optimization problems, DeCoST achieves information exchange between stages via pTAR feedback—more efficient than joint optimization and better than independent optimization.
- **Leveraging LP Global Optimality**: Once the route is fixed, the continuous subproblem admits an exact LP solution—a "fortunate" structural property of OPTWVP that the paper identifies and rigorously proves.
- **Broader Significance for NCO**: Most NCO methods address purely discrete problems; DeCoST is among the first to systematically tackle mixed discrete-continuous optimization in the NCO paradigm, filling an important gap.
- **STO as a General Plugin**: The second stage of DeCoST can be combined with different first-stage constructive solvers, consistently improving solution quality across various baselines.

## Limitations & Future Work

- The global optimality of LP relies on the assumption of a linear profit function; nonlinear profits would require nonlinear programming.
- The optimality gap for large instances (n=500) remains 3.31%, leaving room for improvement.
- The trade-off parameters $\beta_1, \beta_2$ for pTAR require manual tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ Two-stage decoupling + LP optimality + pTAR cross-stage feedback
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple scales, multiple time windows, comparisons with exact/heuristic/NCO methods, and ablations
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and rigorous mathematics
- Value: ⭐⭐⭐⭐ Directly applicable to combinatorial optimization with continuous variables

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](../../ICML2026/optimization/on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](../../ICML2026/optimization/transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICLR 2026\] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)

</div>

<!-- RELATED:END -->
