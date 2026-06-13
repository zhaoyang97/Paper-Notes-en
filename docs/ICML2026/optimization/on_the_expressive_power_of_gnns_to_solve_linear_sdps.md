---
title: >-
  [Paper Note] On the Expressive Power of GNNs to Solve Linear SDPs
description: >-
  [ICML 2026][Optimization][Semidefinite Programming] This paper characterizes the minimum GNN expressivity required to learn solutions for linear SDPs from the perspective of the Weisfeiler–Leman hierarchy for the first t…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Semidefinite Programming"
  - "GNN Expressivity"
  - "Weisfeiler-Leman"
  - "PDHG"
  - "warm-start"
date: 2026-05-08
content_hash: afb1233f6a77e399
---

# On the Expressive Power of GNNs to Solve Linear SDPs

**Conference**: ICML 2026  
**arXiv**: [2604.27786](https://arxiv.org/abs/2604.27786)  
**Code**: Not disclosed  
**Area**: Optimization / Graph Neural Networks / Learning to Optimize  
**Keywords**: Semidefinite Programming, GNN Expressivity, Weisfeiler-Leman, PDHG, warm-start

## TL;DR
This paper characterizes the minimum GNN expressivity required to learn solutions for linear SDPs from the perspective of the Weisfeiler–Leman hierarchy for the first time. It demonstrates that standard variable-constraint bipartite message passing (VC-WL) and higher-order VC-2-WL are insufficient, whereas the VC-2-FWL architecture, equivalent to 2-FWL, is sufficient to simulate the update steps of a PDHG solver. Using high-quality predictions as warm-starts on synthetic datasets and SDPLIB achieves speedups of up to approximately 80%.

## Background & Motivation

**Background**: Learning to optimize (L2O) for Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) using GNNs is relatively mature. The mainstream approach constructs a bipartite graph of constraints and variables and employs message passing equivalent to 1-WL (e.g., GNNs aligned with IPM/PDHG by Gasse et al.), which can approximate optimal solutions or significantly accelerate IPM on small-to-medium scale problems.

**Limitations of Prior Work**: Directly applying the same logic to **Semidefinite Programming (SDP)** fails. The core variable in SDP is not a vector of components but an entire symmetric positive semidefinite matrix $X \in S^n_+$, where the entries $X_{ij}$ are equivariant under simultaneous row and column permutations. Traditional V-C bipartite graphs cannot represent this second-order symmetry, preventing GNNs from fitting the optimal solution regardless of training duration.

**Key Challenge**: Prior L2O literature has only investigated the expressivity thresholds for LP/QP/SOCP (where 1-WL often suffices), but the "constraint-variable-entry" structure of SDP constitutes a third-order tensor structure. The question of **which level of WL is sufficient to recover the optimal solution** remains unanswered, which has stalled the development of "neural SDP solvers."

**Goal**: 1) Formalize the minimum expressivity required for the mapping from SDP instances to the optimal matrix solution $X^*$; 2) Prove the impossibility of standard GNN-like architectures; 3) Propose a sufficient architecture and validate it experimentally; 4) Use predictions as warm-starts for classical solvers.

**Key Insight**: The SDP is formulated as a "hypergraph" composed of constraint matrices $A_k$, the objective matrix $C$, and the variable matrix $X$. By applying the hash updates of 1-WL/2-WL/2-FWL to this tensor graph, VC-WL, VC-2-WL, and VC-2-FWL are constructed respectively. Sufficiency is then proven by simulating the first-order solver PDHG—if a GNN can simulate one iteration of the solver, it can necessarily approximate the converged solution.

**Core Idea**: Standard message passing is replaced by "pairwise node joint aggregation" equivalent to 2-FWL, ensuring that the stable coloring of the GNN is finer than the trajectory partition under PDHG iterations, thereby establishing a sufficient condition for expressivity.

## Method

### Overall Architecture
The input is an SDP instance $(C, \{A_k\}_{k=1}^m, \{b_k\}_{k=1}^m)$. The authors encode it as a **graph with two types of nodes**: variable nodes corresponding to the entries $(i,j) \in [n]\times[n]$ of the matrix $X$, and constraint nodes corresponding to each $A_k$. "Second-order interactions" between variable nodes are weighted by $A_{k,ij}$. After $T$ rounds of permutation-equivariant message passing, variable embeddings are decoded into a predicted matrix $\hat X$ and supervised against the unique minimum Frobenius norm optimal solution $X^*$ (proven unique in Proposition 1.1). Finally, $\hat X$ is utilized as a warm-start initial point for the PDHG solver to accelerate convergence.

### Key Designs

1.  **Impossibility Results (VC-WL / VC-2-WL are insufficient)**:
    - **Function**: Defines "which GNNs definitely cannot learn SDPs."
    - **Mechanism**: The authors construct a family of SDP instances that are structurally isomorphic but possess different optimal solutions. They prove that VC-WL (1-WL equivalent) yields identical node representations after stable coloring; thus, any GNN based on it must produce identical outputs, failing to distinguish different optimal solutions. This conclusion remains valid when extended to VC-2-WL (a variant of standard 2-WL).
    - **Design Motivation**: This transfers the classic "WL identical colors $\Rightarrow$ GNN identical outputs" argument from GNN expressivity research to the SDP setting to provide a **clear lower bound**, warning against wasting computational resources on 1-WL-like architectures.

2.  **Sufficient Architecture VC-2-FWL (Simulating PDHG)**:
    - **Function**: Provides a realizable architecture capable of learning the optimal solutions of linear SDPs.
    - **Mechanism**: Colors are assigned to node pairs $(u,v)\in V^2$, with update rules following folklore 2-FWL: $c^t_{uv} := \text{hash}(c^{t-1}_{uv}, \{\!\{(c^{t-1}_{wv}, c^{t-1}_{uw}) \mid w\in V\}\!\})$. The key proof explicitly maps one iteration of PDHG (including matrix-matrix products and eigendecomposition structural information in PSD projection) to several hash steps of VC-2-FWL. Provided the embedding dimension is sufficient, updates of both dual and primal variables can be precisely simulated. This implies that the stable coloring of VC-2-FWL is finer than the state partition of the PDHG convergence trajectory.
    - **Design Motivation**: Proving "convergence to the optimal solution" directly is difficult. Instead, proving "the ability to simulate a solver known to converge" is a standard L2O technique (the SDP version of the work on IPM by Qian et al.). This establishes VC-2-FWL as the **weakest known GNN architecture capable of solving SDPs**.

3.  **Warm-start Integration**:
    - **Function**: Transitions learned predictions $\hat X$ into practical acceleration.
    - **Mechanism**: After training, $\hat X$ is used as the initial point $X^{(0)}$ for PDHG, allowing the classical solver to begin closer to the optimum. The remaining optimization is handled by PDHG to ensure precision. This addresses the deployment issue where "GNN outputs may not be strictly feasible": feasibility and optimality are guaranteed by PDHG, while the GNN provides a high-quality starting point.
    - **Design Motivation**: Pure neural solvers are unsuitable for high-precision scientific computing; warm-starting is the most robust paradigm for combining ML with traditional optimization, translating theoretical expressivity ($\hat X \approx X^*$) into "reduced wall-clock time" engineering benefits.

### Loss & Training
The supervised objective is the Frobenius error between the predicted matrix and the minimum Frobenius norm optimal solution $\|\hat X - X^*\|_F$, with the objective gap $|\langle C, \hat X\rangle - \langle C, X^*\rangle|$ as an auxiliary metric. Training data consist of synthetic SDP instances (covering randomly generated symmetric matrix families) and various SDP types from SDPLIB (e.g., max-cut relaxation, $\theta$-function).

## Key Experimental Results

### Main Results
On synthetic SDPs and multiple SDPLIB benchmarks, the VC-2-FWL architecture **consistently outperforms** theoretically weaker baselines such as VC-WL and VC-2-WL in both prediction error and objective gap.

| Setting | Metric | VC-WL | VC-2-WL | VC-2-FWL |
| :--- | :--- | :--- | :--- | :--- |
| Synthetic SDP | $\|\hat X - X^*\|_F$ | Highest | Medium | Lowest |
| SDPLIB | Objective Gap | Largest | Medium | Smallest |
| Warm-start PDHG | Convergence Time Reduction | Minimal | Moderate | **Up to 80%** |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
| :--- | :--- | :--- |
| VC-WL (1-WL equivalent) | Highest fitting error; unable to distinguish certain instances | Validates Impossibility Theorem |
| VC-2-WL (2-WL equivalent) | Better than VC-WL but still deviates from optimum | Standard 2-WL is insufficient |
| VC-2-FWL (2-FWL equivalent) | Lowest error and smallest objective gap | Satisfies sufficiency |
| Cold-start PDHG only | Baseline convergence time | Control group |
| VC-2-FWL warm-start | Max convergence time reduction ~80% | Expressivity $\rightarrow$ Practical Acceleration |

### Key Findings
- The expressivity hierarchy strictly aligns with "actual fitting quality": The theoretical relationship VC-WL $\sqsubset$ VC-2-WL $\sqsubset$ VC-2-FWL translates to a strict decrease in prediction error on both synthetic and real benchmarks.
- There is a clear leverage effect where "reducing GNN error slightly" results in "significantly fewer solver iterations"—as long as the prediction falls within the basin of attraction of the optimal solution, the remaining iterations for PDHG are drastically reduced.
- Error mode analysis shows that VC-WL primarily fails on "highly symmetric constraint matrix families," which is the classic blind spot of 1-WL discriminative power, consistent with theoretical predictions.

## Highlights & Insights
- **First characterization of SDP-GNN expressivity**: This work extends the established "WL hierarchy as GNN expressivity" framework from LP/QP to SDP, filling a major gap in L2O theory. This analysis template can be directly transferred to broader conic programming beyond SOCP.
- **Elegant "solver simulation equals expressivity" paradigm**: Rather than proving direct convergence of $\hat X$, the authors prove the GNN can simulate one step of PDHG—converting the "optimal solution recovery" problem into a "discrete dynamical system simulation" problem, which is more tractable.
- **Theory-guided architecture**: Future efforts to train neural SDP solvers can bypass the "trial-and-error" phase of 1-WL/2-WL architectures and start directly from 2-FWL-equivalent designs.
- **Finite threshold upper bound**: Proving VC-2-FWL is "sufficient" does not imply it is "necessary," but impossibility results suggest 1-WL/2-WL are inadequate. The remaining gap (e.g., 3-WL?) presents an opportunity for future exploration.

## Limitations & Future Work
- Experiments were mainly conducted on synthetic data and medium-sized SDPLIB. **In actual large-scale SDPs (e.g., $n>10^4$ in combinatorial optimization), the memory cost of VC-2-FWL ($O(n^2)$ node pairs) explodes rapidly**, and a scalable implementation has yet to be provided.
- Sufficiency proofs only address "linear SDP + unique minimum norm optimal solution," without covering non-unique solution sets, degenerate solutions, or semi-infinite SDPs.
- Warm-start benefits depend on PDHG as the downstream solver; compatibility with more precise solvers like IPM or ADMM remains to be validated.
- Future directions: Sparsification or subgraph sampling to scale VC-2-FWL, extending analysis to second-order cones or mixed-integer SDPs, and integration with modern low-rank SDP algorithms.

## Related Work & Insights
- **vs Qian et al. (PDHG-GNN for LP)**: They proved that 1-WL-like GNNs suffice to simulate PDHG for LP; this paper proves SDP requires an upgrade to 2-FWL, revealing the correspondence between "cone dimension" and "WL level."
- **vs Yau et al. (GNN for low-rank SDP relaxation of Max-CSP)**: While they analyzed GNNs as approximation algorithms for CSP, this paper targets general linear SDPs directly to study whether GNNs can recover the optimal matrix solution.
- **vs Chen et al. (GNN expressivity for QP / SOCP)**: This work continues their research line of "WL hierarchy $\leftrightarrow$ convex optimization solution recovery," placing SDP precisely on the expressivity map.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to link SDP solution recovery to WL levels.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic + SDPLIB, but lacks ultra-large-scale and multi-solver comparisons.
- Writing Quality: ⭐⭐⭐⭐ Theoretical sections are clear; warm-start engineering details could be more substantial.
- Value: ⭐⭐⭐⭐ Benchmark significance for the "neural convex optimization" theory community; industrial application still requires scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICLR 2026\] Learning to Solve Orienteering Problem with Time Windows and Variable Profits](../../ICLR2026/optimization/learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)
- [\[ICML 2026\] Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming](provably_data-driven_lagrangian_relaxation_for_mixed_integer_linear_programming.md)
- [\[ICML 2026\] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)

</div>

<!-- RELATED:END -->
