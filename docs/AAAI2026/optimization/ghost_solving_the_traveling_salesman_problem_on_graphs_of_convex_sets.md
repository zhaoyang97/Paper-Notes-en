---
title: >-
  [Paper Note] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets
description: >-
  [AAAI 2026][Optimization][Traveling Salesman Problem] This paper proposes GHOST, a hierarchical optimal search algorithm for solving the Traveling Salesman Problem on Graphs of Convex Sets (GCS). By combining combinatorial path search with convex trajectory optimization, and employing a novel abstract path expansion algorithm to compute admissible lower bounds that guide best-first search, GHOST achieves optimality guarantees while outperforming the monolithic mixed-integer convex programming (MICP) baseline by orders of magnitude in runtime.
tags:
  - AAAI 2026
  - Optimization
  - Traveling Salesman Problem
  - Graphs of Convex Sets
  - Trajectory Optimization
  - Hierarchical Search
  - Robot Motion Planning
date: 2026-05-08
content_hash: 8a5b73da8e3df837
---

# GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets

**Conference**: AAAI 2026
**arXiv**: [2511.06471](https://arxiv.org/abs/2511.06471)
**Code**: [https://github.com/reso1/ghost](https://github.com/reso1/ghost)
**Area**: Optimization
**Keywords**: Traveling Salesman Problem, Graphs of Convex Sets, Trajectory Optimization, Hierarchical Search, Robot Motion Planning

## TL;DR

This paper proposes GHOST, a hierarchical optimal search algorithm for solving the Traveling Salesman Problem on Graphs of Convex Sets (GCS). By combining combinatorial path search with convex trajectory optimization, and employing a novel abstract path expansion algorithm to compute admissible lower bounds that guide best-first search, GHOST achieves optimality guarantees while outperforming the monolithic mixed-integer convex programming (MICP) baseline by orders of magnitude in runtime.

## Background & Motivation

Trajectory optimization is widely used in robot motion planning; however, even a single obstacle can render the configuration space non-convex and computationally intractable. The **Graphs of Convex Sets (GCS)** framework decomposes the configuration space into convex regions connected via a graph structure, reformulating non-convex trajectory planning as convex optimization subproblems.

While shortest-path problems on GCS have been thoroughly studied, many practical robot tasks—such as area coverage, inspection, and structured exploration—require **sequentially visiting multiple regions at minimum cost**, which naturally gives rise to the Traveling Salesman Problem on GCS (GCS-TSP).

The key distinction between GCS-TSP and classical TSP is that **edge weights are not fixed but depend on the specific trajectory traversing each convex set**. This tightly couples combinatorial optimization (visit ordering) with continuous kinematic feasibility, rendering classical TSP methods inapplicable for three reasons:

1. No static metric closure exists—each transition cost depends on the concrete trajectory endpoints within the convex sets.
2. Monolithic MICP formulations scale poorly and cannot handle incomplete GCS instances with complex constraints.
3. Existing methods either lack optimality guarantees or cannot handle high-order continuity constraints.

## Method

### Overall Architecture

GHOST employs a **two-level hierarchical search** architecture:

- **High level**: Systematically explores TSP paths over the complete graph $\hat{G}$ induced by the GCS, ordered by non-decreasing lower-bound cost.
- **Low level**: For each abstract path, searches for feasible GCS paths (expansions) and computes optimal trajectories via convex optimization.

The core component is the **abstract path expansion algorithm**, which computes admissible lower bounds for a given abstract path (an arbitrary sequence of consecutive sets), providing strong pruning capability to reduce unnecessary convex optimization calls.

### Key Designs

#### 1. **Lower Bound Graph (LBG) Construction**: Tight Lower Bound Estimation via Triplets

**Problem**: In a standard graph, edge weights provide path cost estimates, but the traversal cost of edge $(u,v)$ in GCS depends on trajectory endpoints $\mathbf{x}_u \in \mathcal{X}_u$ and $\mathbf{x}_v \in \mathcal{X}_v$, which are unconstrained until the full path is determined. Edge-based lower bounds are therefore loose.

**Solution**: The LBG is defined over **triplets** $(u,v,w)$ (predecessor–current–successor) rather than edges alone. This captures geometric information about the "volume" of convex set $\mathcal{X}_v$, yielding tighter lower bounds.

The LBG $\mathcal{H} = (\mathcal{P}, \mathcal{F})$ is a directed hypergraph where:
- **Passages** $(u,v,w) \in \mathcal{P}$: formed by each pair of adjacent edges $(u,v),(v,w) \in E$.
- **Hyperedges** $(p,p')$: connecting each $p = (\cdot,u,v)$ to $p' = (u,v,\cdot)$.
- Each triplet $p$ is annotated with a lower-bound cost $lb_p$, computed via a convex program with relaxed constraints.

**Key Theorem (Theorem 1 – Lower Bound Path Cost)**: For any abstract path $\hat{\pi}$ and the minimum lower-bound-cost path $\pi^*$ expanded from it:

$$\mathcal{L}(\pi^*) \leq \mathcal{L}(\pi) \leq c(\tau)$$

holds for any path $\pi$ expanded from $\hat{\pi}$ and any trajectory $\tau$ conditioned on $\pi$.

**Design Motivation**: Triplets provide tighter lower bounds than edges by incorporating geometric information about the convex sets. Tighter lower bounds → stronger pruning → fewer convex optimization calls.

#### 2. **Path Expansion Algorithm (Algorithm 1)**: Multi-Label A\*-Style Search

**Function**: Given an abstract path $\hat{\pi} = (v_1, v_2, \ldots, v_k)$ (where consecutive vertices may not be adjacent in the GCS), enumerate all GCS paths expanded from $\hat{\pi}$ in non-decreasing order of lower-bound cost.

**Mechanism**:

Search states consist of a triplet $p$ and a label $\ell$ indicating progress toward the next endpoint $v_{\ell+1}$. Each search node $n$ stores:
- Current path $n.\pi$
- Current cost $n.g$
- Estimated final cost $n.f = n.g + h_{\hat{\pi}}(p, n.\ell)$

**Admissible Heuristic (Lemma 2)**:

$$h_{\hat{\pi}}(p, n.\ell) = \mathcal{L}(\pi^*_{w, v_{\ell+1}}) + \sum_{i=\ell+1}^{k-1}\mathcal{L}(\pi^*_{v_i, v_{i+1}})$$

This aggregates the minimum possible cost of remaining sub-paths and never overestimates the true cost.

Optionally, nodes with $n_c.f > \bar{c}$ are pruned to support efficient hierarchical search within GHOST.

#### 3. **Restricted TSP (RTSP) Reformulation**: ILP over Abstract Triplets

Classical TSP ILPs use edge variables, but edge costs in GCS do not provide meaningful estimates. This paper introduces **triplet variables** $y_{\hat{p}}$ to minimize lower-bound path cost:

$$\min_{y_{\hat{p}}} \sum_{\hat{p} \in \hat{\mathcal{P}}} b(\hat{p}) y_{\hat{p}}$$

subject to:
- Each vertex is visited exactly once (Eq. 6).
- Flow conservation (Eq. 7).
- Edge inclusion/exclusion constraints (Eqs. 8–9).
- DFJ-style subtour elimination (Eq. 10).

The lower-bound cost of a triplet is: $b(\hat{p}) = \frac{1}{2}\mathcal{L}(\pi^*_{u,v}) + b_{\text{mid}}(\hat{p}) + \frac{1}{2}\mathcal{L}(\pi^*_{v,w})$

**Theorem 3 (Lower Bound Path Cost)**: For abstract path $\hat{\pi}$, $\mathcal{L}(\hat{\pi}) \leq \mathcal{L}(\pi) \leq c(\tau)$.

### GHOST Framework (Algorithm 2)

**High-level search** (generalization of the Lawler–Murty procedure):
- The OPEN list is ordered by lower-bound path cost $n.\underline{c}$.
- At each step, the minimum-cost node is popped, and EvalNode is called to expand and evaluate it.
- Termination occurs when $n.\underline{c} \geq c(n^*.\tau)$ (the current best trajectory cost).

**Low-level evaluation (EvalNode)**:
- Invokes the path expansion algorithm to generate paths.
- Computes optimal trajectories for each path via convex optimization.
- Stops when the lower bound of the expanded path exceeds the current best.

### Bounded Suboptimal Variant: ε-GHOST

Given parameter $\epsilon \in [0,1)$:
- The termination condition is relaxed to $c(n^*.\tau) - n.\underline{c} \leq \epsilon \cdot c(n^*.\tau)$.
- The pruning threshold is adjusted to $(1-\epsilon) \cdot c(n^*.\tau)$.

**Theorem 4**: ε-GHOST returns a solution whose cost is at most $\frac{1}{1-\epsilon}$ times optimal.

**Corollary 5**: At $\epsilon=0$, GHOST returns an optimal solution.

## Key Experimental Results

### Main Results

Three GCS-TSP problem types are evaluated:

**Point-GCS** (complete GCS, 2D point sets, $N=5$–$25$ vertices):

| Method | Solution Quality | Runtime | Optimality Guarantee |
|--------|-----------------|---------|----------------------|
| **GHOST** | **Optimal** | **Fastest** | **Yes** |
| ECG | Sub-optimal | Slower | No |
| Greedy | Sub-optimal | Moderate | No |
| MICP | Optimal | Fails for $N>13$ | Yes |

**Linear-GCS** (incomplete GCS, $M=10$–$50$ edges):

| Method | Maximum Tractable Scale | Notes |
|--------|------------------------|-------|
| GHOST | $M=50+$ | All instances solved |
| MICP | $M\leq19$ | Fails at larger scales |

**Bézier-GCS** (incomplete GCS, degree-4 Bézier curves, $M=10$–$50$):

| Method | Result |
|--------|--------|
| GHOST | Successfully solved |
| MICP | Completely intractable |

### Ablation Study

| Configuration | Solution Quality | Runtime | Optimality |
|---------------|-----------------|---------|------------|
| GHOST (full) | Optimal | Baseline | Yes |
| ECG (no LBG) | Degraded | Slower | No |
| Greedy (no systematic search) | Worst | Unstable | No |
| 0.5-GHOST | Near-optimal | Faster | Bounded suboptimal |

Key comparisons:
- **LBG vs. ECG**: LBG provides provable lower bounds, enabling GHOST to monitor the optimality gap and prune early.
- **Systematic search vs. greedy**: Systematic search yields better solutions and more reliable performance.
- **0.5-GHOST** typically achieves the same solution quality as GHOST with higher efficiency.

### Key Findings

1. **GHOST outperforms MICP by orders of magnitude**: MICP fails to produce feasible solutions for Point-GCS with $N>13$, Linear-GCS with $M>19$, or any Bézier-GCS instance.
2. **LBG integration is critical**: It provides provable lower bounds and strong pruning capability.
3. **GHOST solves previously intractable problems**: Complex instances involving high-order continuity constraints and incomplete GCS.
4. **ε-GHOST offers a flexible optimality–efficiency trade-off**: Practically valuable in time-critical scenarios.
5. **Trajectory caching is effective**: The same path may be expanded from different abstract paths; caching avoids redundant convex optimization calls.

## Highlights & Insights

1. **Problem modeling contribution**: GCS-TSP captures the essential tight coupling between combinatorial and continuous optimization in robot planning, representing a meaningful generalization of classical TSP to practical robotics applications.
2. **Elegant hierarchical search architecture**: The high level explores visit orderings while the low level verifies feasibility; the two-level lower bounds mutually support each other to enable efficient pruning.
3. **Innovation of triplet-based lower bounds**: Tighter than edge-based lower bounds by exploiting the geometry of convex sets.
4. **Complete theoretical guarantees**: Optimality (Corollary 5) and bounded suboptimality (Theorem 4) together provide rigorous solution quality assurances.
5. **Application versatility**: Coverage planning, inspection planning, 7-DoF manipulator task-and-motion planning—demonstrating the generality of the framework.

## Limitations & Future Work

1. **Scalability**: Although far superior to MICP, GHOST's performance on very large-scale GCS instances (hundreds of convex sets) remains to be validated.
2. **Single-threaded implementation**: The paper notes limited benefit from cross-path parallel evaluation, though parallelization may be more valuable at larger scales.
3. **Convex decomposition quality**: GHOST's performance depends on the quality of the input GCS; convex decomposition methods themselves are outside the scope of this work.
4. **LBG construction cost**: Although precomputable, the number of triplets is $\mathcal{O}(|E|^2/|V|)$ for dense GCS instances, which may be substantial.
5. **Restricted to convex cost functions**: The method requires cost functions that are positive, bounded away from zero, and convex, which may not hold in all application scenarios.

## Related Work & Insights

- The work builds on the GCS framework and GCS shortest-path solvers, extending the problem from point-to-point planning to multi-target visitation.
- A generalization of the Lawler–Murty procedure provides a systematic framework for enumerating suboptimal TSP solutions.
- The application of multi-label A\* search in path expansion represents a novel combination of existing graph search techniques.
- The ε-GHOST bounded suboptimal variant offers a theoretically guaranteed approximation scheme for time-critical robot applications.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The GCS-TSP problem formulation is original; the hierarchical search framework and triplet-based lower bound design are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three problem types, multiple baselines, and ablation studies provide sufficient coverage, though large-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation, algorithm description, and theoretical proofs are all presented with exceptional clarity.
- Value: ⭐⭐⭐⭐ — Addresses an important combinatorial optimization problem in robot motion planning with strong practical relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[ICLR 2026\] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](../../ICLR2026/optimization/cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)
- [\[ICLR 2026\] Learning to Solve Orienteering Problem with Time Windows and Variable Profits](../../ICLR2026/optimization/learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
