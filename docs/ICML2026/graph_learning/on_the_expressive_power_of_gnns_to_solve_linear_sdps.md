---
title: >-
  [Paper Note] On the Expressive Power of GNNs to Solve Linear SDPs
description: >-
  [ICML 2026][Graph Learning][Semidefinite Programming] This work, from the perspective of the Weisfeiler–Leman hierarchy, for the first time characterizes the minimal GNN expressiveness required to learn solutions to line…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Semidefinite Programming"
  - "GNN Expressiveness"
  - "Weisfeiler-Leman"
  - "PDHG"
  - "warm-start"
date: 2026-05-08
content_hash: 0f37d7a0fd36e5ef
---

# On the Expressive Power of GNNs to Solve Linear SDPs

**Conference**: ICML 2026  
**arXiv**: [2604.27786](https://arxiv.org/abs/2604.27786)  
**Code**: Not released  
**Area**: Optimization / Graph Neural Networks / Learning to Optimize  
**Keywords**: Semidefinite Programming, GNN Expressiveness, Weisfeiler-Leman, PDHG, warm-start

## TL;DR
This work, from the perspective of the Weisfeiler–Leman hierarchy, for the first time characterizes the minimal GNN expressiveness required to learn solutions to linear SDPs. It proves that standard variable-constraint bipartite message passing (VC-WL) and higher-order VC-2-WL are insufficient, while the VC-2-FWL architecture, equivalent to 2-FWL, is sufficient to simulate the update steps of the PDHG solver. On synthetic and SDPLIB datasets, using high-quality predictions as warm-starts yields up to 80% acceleration.

## Background & Motivation

**Background**: Using GNNs to learn to solve linear programming (LP) and mixed-integer programming (MILP) (learning-to-optimize, L2O) is already mature. The mainstream approach constructs a constraint-variable bipartite graph and applies message passing equivalent to 1-WL (e.g., Gasse et al.'s IPM/PDHG-aligned GNN), which can approach optimal solutions or significantly accelerate IPM on small to medium-scale problems.

**Limitations of Prior Work**: Directly applying the same approach to **semidefinite programming (SDP)** fails. The core variable in SDP is not a vector component but an entire symmetric positive semidefinite matrix $X \in S^n_+$, whose entries $X_{ij}$ are equivariant under simultaneous row and column permutations. Traditional V-C bipartite graphs cannot express this second-order symmetry, causing GNNs to fail to approximate the optimal solution regardless of training duration.

**Key Challenge**: Existing L2O literature has only studied the expressiveness threshold for LP/QP/SOCP (mostly 1-WL suffices), but for SDP, the "constraint-variable-entry" forms a third-order tensor structure. **Which WL level is sufficient to recover the optimal solution** remains unanswered, directly hindering the development of "neural SDP solvers."

**Goal**: 1) Formalize the minimal expressiveness required to map an SDP instance to its optimal matrix solution $X^*$; 2) Prove the impossibility for standard GNN architectures; 3) Provide a sufficient architecture and experimental validation; 4) Use predictions as warm-starts for classical solvers.

**Key Insight**: Represent the SDP as a "hypergraph" composed of constraint matrices $A_k$, objective matrix $C$, and variable matrix $X$, and apply the hash updates of 1-WL/2-WL/2-FWL directly to this tensor graph, constructing VC-WL, VC-2-WL, and VC-2-FWL respectively. By simulating the first-order PDHG solver, sufficiency is established: as long as the GNN can simulate one iteration of the solver, it can approximate the converged solution.

**Core Idea**: Replace standard message passing with "pairwise node joint aggregation" equivalent to 2-FWL, so that the GNN's stable coloring is finer than the trajectory partition under PDHG iterations, thus providing a sufficient condition for expressiveness.

## Method

### Overall Architecture
The input is an SDP instance $(C, \{A_k\}_{k=1}^m, \{b_k\}_{k=1}^m)$. The authors encode it as a **graph with two types of nodes**: variable nodes correspond to entries $(i,j) \in [n]\times[n]$ of matrix $X$, and constraint nodes correspond to each $A_k$; "second-order interactions" between variable nodes are weighted by $A_{k,ij}$. After $T$ rounds of permutation-equivariant message passing, the model decodes the variable embeddings into a predicted matrix $\hat X$, supervised against the unique minimum Frobenius norm optimal solution $X^*$ (uniqueness proved in Proposition 1.1). Finally, $\hat X$ is used as the warm-start initial point for the PDHG solver to accelerate convergence.

### Key Designs

1. **Impossibility Results (VC-WL / VC-2-WL Insufficiency)**:

    - Function: Defines "which GNNs cannot learn SDP."
    - Mechanism: The authors construct a family of structurally isomorphic SDP instances with different optimal solutions, proving that VC-WL (1-WL equivalent) gives identical node representations after stable coloring, so any GNN based on it must output the same result and cannot distinguish different optima. Extending the analysis to VC-2-WL (a variant of standard 2-WL) yields the same conclusion.
    - Design Motivation: Transfers the classic "WL same color $\Rightarrow$ GNN same output" argument from GNN expressiveness research to the SDP setting, providing a **clear lower bound** and warning future work not to waste compute on 1-WL architectures.

2. **Sufficient Architecture VC-2-FWL (Simulating PDHG)**:

    - Function: Provides a realizable architecture capable of learning the optimal solution to linear SDPs.
    - Mechanism: Colors are assigned to node pairs $(u,v)\in V^2$, with updates using folklore 2-FWL: $c^t_{uv} := \text{hash}(c^{t-1}_{uv}, \{\!\{(c^{t-1}_{wv}, c^{t-1}_{uw}) \mid w\in V\}\!\})$. The key proof explicitly maps one PDHG iteration (including matrix-matrix products and spectral decomposition in PSD projection) into several hash steps of VC-2-FWL—given sufficient embedding dimension, both dual and primal updates can be precisely simulated. This means VC-2-FWL's stable coloring is finer than the state partition of PDHG's convergence trajectory, thus capable of fitting the optimal solution.
    - Design Motivation: Directly proving "can approximate the optimal solution" is difficult; instead, proving "can simulate a known convergent solver" is a standard L2O technique (the SDP version of Qian et al.'s IPM work). This makes VC-2-FWL the **weakest known GNN architecture capable of solving SDPs**.

3. **Warm-start Integration**:

    - Function: Turns the learned prediction $\hat X$ into practical acceleration.
    - Mechanism: After training, $\hat X$ is used as the initial point $X^{(0)}$ for PDHG, allowing the classical solver to start closer to the optimum, with the remaining optimization handled by PDHG for precision. This avoids the deployment challenge that "GNN outputs may not be strictly feasible": feasibility and optimality are guaranteed by PDHG, and the GNN only needs to provide a good starting point.
    - Design Motivation: Pure neural solvers are unusable in high-precision scientific computing; warm-start is the most robust paradigm for combining ML with traditional optimization, translating the expressiveness $\hat X \approx X^*$ into tangible "wall-clock time reduction."

### Loss & Training
The supervised objective is the Frobenius error between the predicted and minimum Frobenius norm optimal matrix, $\|\hat X - X^*\|_F$, with the objective gap $|\langle C, \hat X\rangle - \langle C, X^*\rangle|$ as an auxiliary metric. Training data includes synthetic SDP instances (covering randomly generated symmetric matrices) and various SDPLIB SDPs (e.g., max-cut relaxations, $\theta$-function, etc.).

## Key Experimental Results

### Main Results
On synthetic SDPs and multiple SDPLIB benchmarks, the VC-2-FWL architecture consistently **outperforms** theoretically weaker baselines such as VC-WL and VC-2-WL in both prediction error and objective gap.

| Setting | Metric | VC-WL | VC-2-WL | VC-2-FWL |
|------|------|------|------|------|
| Synthetic SDP | $\|\hat X - X^*\|_F$ | Highest | Medium | Lowest |
| SDPLIB | Objective gap | Largest | Medium | Smallest |
| Warm-start PDHG | Convergence time reduction | Almost none | Moderate | **Up to 80%** |

### Ablation Study

| Configuration | Key Phenomenon | Description |
|------|---------|------|
| VC-WL (1-WL equivalent) | Highest fitting error, some instances completely indistinguishable | Confirms impossibility theorem |
| VC-2-WL (2-WL equivalent) | Better than VC-WL, but still deviates from optimum | Standard 2-WL insufficient |
| VC-2-FWL (2-FWL equivalent) | Lowest error, smallest objective gap | Satisfies sufficiency |
| Cold-start PDHG only | Baseline convergence time | Control |
| VC-2-FWL warm-start | Up to 80% reduction in convergence time | Expressiveness → practical acceleration |

### Key Findings
- The expressiveness hierarchy aligns strictly with "actual fitting quality": the theoretical relation VC-WL ⊏ VC-2-WL ⊏ VC-2-FWL translates into strictly decreasing prediction error on both synthetic and real benchmarks.
- There is a clear leverage effect in turning "slightly lower GNN error" into "significantly fewer solver iterations"—as long as the prediction falls within the basin of attraction of the optimal solution, the remaining PDHG iterations are greatly reduced.
- Error pattern analysis shows VC-WL mainly fails on "highly symmetric constraint matrix families," which is the classic blind spot of 1-WL's distinguishing power, consistent with theoretical predictions.

## Highlights & Insights
- **First expressiveness characterization for SDP-GNNs**: Extends the mature "WL hierarchy = GNN expressiveness" framework from LP/QP to SDP, filling a major gap in L2O theory; this analysis template can be directly transferred to broader conic programs beyond SOCP.
- **Elegant paradigm of 'simulating solvers = gaining expressiveness'**: Rather than directly proving $\hat X$ convergence, it proves the GNN can simulate one PDHG step—thus reframing "approximating the optimal solution" as "simulating a discrete dynamical system," which is more tractable.
- **A typical case of theory guiding architecture**: Future attempts at neural SDP solvers can skip the "pitfall period" of failed 1-WL/2-WL architectures and start directly from 2-FWL equivalents.
- **Finite threshold upper bound**: Proving VC-2-FWL is "sufficient" does not mean "necessary," but the impossibility results imply 1-WL/2-WL are insufficient, leaving a gap (3-WL? 3-FWL?) for future exploration.

## Limitations & Future Work
- Experiments mainly focus on synthetic data and medium-scale SDPLIB; **for truly large-scale SDPs (e.g., combinatorial optimization relaxations with $n>10^4$), the memory cost of VC-2-FWL (number of node pairs $O(n^2)$) quickly explodes**, and no scalable implementation is provided.
- The sufficiency proof only covers "linear SDP + unique minimum norm optimal solution," and does not address cases with non-unique solutions, degenerate solutions, or semi-infinite SDPs.
- Warm-start benefits depend on PDHG as the downstream solver; compatibility with more precise solvers such as IPM or ADMM remains to be verified.
- Future directions: sparsification or subgraph sampling to make VC-2-FWL scalable, extending the analysis to second-order cone/mixed-integer SDPs, and combining with modern low-rank SDP algorithms.

## Related Work & Insights
- **vs Qian et al. (PDHG-GNN for LP)**: They proved 1-WL GNNs suffice to simulate LP's PDHG; this work shows SDP requires 2-FWL, revealing a correspondence between "cone dimension" and "WL hierarchy."
- **vs Yau et al. (GNN for low-rank SDP relaxation of Max-CSP)**: Their work analyzes GNNs as approximate algorithms for CSPs, while this work targets general linear SDPs and investigates whether GNNs can recover the optimal matrix solution.
- **vs Chen et al. (GNN for QP / SOCP expressiveness)**: This work continues their "WL hierarchy ↔ convex optimization solution recovery" research, precisely locating SDP in the expressiveness map.
- **vs Yau et al. (GNNs on low-rank SDP relaxations)**: They treat GNNs as CSP approximators; this work directly studies whether GNNs can recover the original SDP optimal matrix solution $X^*$, focusing on different objects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to connect SDP solution recovery to the WL hierarchy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both synthetic and SDPLIB, but lacks large-scale and multi-solver comparisons.
- Writing Quality: ⭐⭐⭐⭐ Theory is clear; warm-start engineering details could be further enriched.
- Value: ⭐⭐⭐⭐ Sets a benchmark for the "neural convex optimization" theory community; industrial deployment still requires further extension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](../../AAAI2026/graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)

</div>

<!-- RELATED:END -->
