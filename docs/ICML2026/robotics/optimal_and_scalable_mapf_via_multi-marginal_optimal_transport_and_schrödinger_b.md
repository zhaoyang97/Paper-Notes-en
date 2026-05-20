---
title: >-
  [Paper Note] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges
description: >-
  [ICML 2026][Robotics][MAPF] This paper proves that anonymous multi-robot path planning (MAPF) can be formulated as a **Markovian Multi-Marginal Optimal Transport (MMOT)** problem…
tags:
  - "ICML 2026"
  - "Robotics"
  - "MAPF"
  - "Multi-Marginal Optimal Transport"
  - "Schrödinger bridge"
  - "Total Unimodularity"
  - "Sinkhorn"
date: 2026-05-08
content_hash: 7290cf4496c9d918
---

# Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges

**Conference**: ICML 2026  
**arXiv**: [2605.10917](https://arxiv.org/abs/2605.10917)  
**Code**: Not released  
**Area**: Robotics / Multi-Agent Path Planning / Optimal Transport  
**Keywords**: MAPF, Multi-Marginal Optimal Transport, Schrödinger bridge, Total Unimodularity, Sinkhorn

## TL;DR
This paper proves that anonymous multi-robot path planning (MAPF) can be formulated as a **Markovian Multi-Marginal Optimal Transport (MMOT)** problem, compressing the original $K^{T+1}$-dimensional transport tensor into a polynomial-size LP (P1), with total unimodularity guaranteeing integer optimality. It then generalizes to the Schrödinger bridge, yielding a Sinkhorn-style entropic relaxation (P2) that produces a "shadow transport." Finally, pruning and solving an LP (P3) on the shadow recovers integer solutions, achieving 3.6×–7.1× speedup and <10% cost gap at $K^{1.15}$ complexity.

## Background & Motivation

**Background**: MAPF (having multiple robots reach goals on a shared graph without collisions) is classically solved by Conflict-Based Search (CBS), SAT encoding, and time-expanded flow networks. Optimal algorithms are feasible at moderate scale, but large-scale anonymous MAPF (any robot to any goal) remains challenging.

**Limitations of Prior Work**: Existing IP/LP formulations (time-expanded network flow) can yield optimal solutions, but **no one has systematically characterized the source of LP integrality**—it is empirically known that integer solutions exist in some cases, but a unified framework for "what structural conditions guarantee total unimodularity (TU)" is lacking. For large-scale problems (thousands of nodes, tens of thousands of variables), only approximations are practical.

**Key Challenge**: Achieving "optimality + integrality" typically requires IP (NP-hard), while "scalability" usually means distributed heuristics (no guarantees). MAPF lacks a unified framework that bridges these extremes with theoretical guarantees and scalability.

**Goal**: 1) Provide a unified optimal transport perspective for MAPF; 2) Prove that the LP is polynomial and integral under this view; 3) Use probabilistic relaxation (Schrödinger bridge) to obtain a scalable Sinkhorn algorithm; 4) Convert the probabilistic relaxation back to executable integer trajectories.

**Key Insight**: Treat all possible joint trajectories of $N$ robots over $T$ steps as a $(T+1)$-order tensor $\mathbf{P}\in\mathbb{R}_{\ge 0}^{K\times\cdots\times K}$, where each entry is the probability mass of a path. MAPF seeks the minimum-cost transport plan matching initial and terminal marginals. This is naturally MMOT, but since robot motion is Markovian, the tensor admits a standard factorization $\mathbf{P}_{i_0,\ldots,i_T} \propto \prod_t [\Pi_t]_{i_{t-1}i_t}$, reducing variables from $O(K^{T+1})$ to $O(K^2T)$.

**Core Idea**: MAPF = Markovian MMOT; under the anonymous setting, the LP is totally unimodular under natural assumptions, so polynomial-time integer optimal solutions exist. Introducing the Schrödinger bridge yields a scalable probabilistic solver, and pruning plus LP recovers integrality.

## Method

### Overall Architecture
The approach follows three steps: (1) **P1**: Formulate MAPF as an LP over Markovian transition matrices $\{\Pi_t\}_{t=1}^T$, prove total unimodularity for integer optimality; (2) **P2**: Use the Gibbs kernel $\bar g_{ij,t} \propto \exp(-c_{ij,t}/\varepsilon)$ as a reference distribution to write the Schrödinger bridge, yielding an entropic regularization of P1, and solve for the "shadow" fractional transport $\tilde\Pi_t$ via multi-marginal Sinkhorn; (3) **P3**: Prune the graph using high-mass edges from the shadow, then resolve the LP on the reduced graph to recover integer solutions $\hat\Pi_t$. This pipeline bridges "optimality + integrality" and "scalability," reducing overall complexity from classical IPM's $O(K^{1.68})$ to $O(K^{1.15})$.

### Key Designs

1. **P1: MMOT-LP Formulation for MAPF and Total Unimodularity Guarantee**:

    - **Function**: Provides a polynomial-time LP formulation for anonymous MAPF, guaranteeing $\{0,1\}$ integer solutions.
    - **Mechanism**: Decision variables are transition matrices $\{\Pi_t\}$ between adjacent time steps, with objective $\sum_t \langle \Pi_t, C_t\rangle$. Three key constraints: **gluing** (mass conservation $\Pi_t^\top\mathbf{1} = \Pi_{t+1}\mathbf{1}$) ensures Markovianity; **terminal** fixes initial and final distributions $\Pi_1\mathbf{1}=\mu, \Pi_T^\top\mathbf{1}=\nu$; **vertex-capacity** $0\le\Pi_t^\top\mathbf{1}\le\mathbf{1}$ prevents >1 robot per vertex. Assumption 3.1 gives natural structure (self-loops allowed, non-overlapping edges can be parallel, move cost > wait cost > 0, target wait cost = 0). Lemma 3.3 proves that under these assumptions, the LP constraint matrix is totally unimodular (TU), so all basic feasible solutions (extreme points) are integer, with polynomial $O(KT)$ variables. Theorem 3.4 translates integer solutions back to "collision-free, non-overlapping trajectories, each robot reaching an independent goal."
    - **Design Motivation**: Previous MAPF time-expanded IPs lacked a clear explanation for the existence of integer solutions; this work provides TU as a **first-principles** justification, and unifies min-cost / min-move / min-makespan objectives via cost matrix adjustments (e.g., Assumption 3.5 uses exponentially growing $c_{ij,t} = B^t \tilde c_{ij}$ to implicitly enforce min-makespan, Lemma 3.7 gives $T^* \le N + K - 1$ makespan upper bound and $O(\log K)$ binary search for minimal horizon).

2. **P2: Schrödinger Bridge and Entropic Relaxation**:

    - **Function**: Uses a probabilistic framework to obtain a highly parallelizable Sinkhorn solver, producing a "shadow map" of important edges.
    - **Mechanism**: Generalizes P1 to finding a joint distribution $\mathbf{P}$ minimizing $\mathrm{KL}(\mathbf{P}\,\|\,\mathbf{G})$ over constraint set $\mathcal{C}$, where $\mathbf{G}$ is a reference Markov tensor. Lemma 4.1 shows $\mathrm{KL}(\mathbf{P}\|\mathbf{G})$ decomposes into a sum over time layers $\mathrm{KL}(\frac{1}{N}\Pi_t\|\mathbf{G}_t)$ plus boundary terms. With the Gibbs kernel $g_{ij,t}=\exp(-c_{ij,t}/\varepsilon)$ as reference, Lemma 4.2 expresses the objective as P1 with entropic regularization: $\min \sum_t \langle\Pi_t,C_t\rangle + \varepsilon\sum_{i,j}\pi_{ij,t}(\log\pi_{ij,t}-1)$. This is P2, solvable efficiently by multi-marginal Sinkhorn block coordinate descent (Appendix G). Note P2 relaxes the vertex-capacity constraint, so solutions may be fractional—this is precisely the "shadow": it indicates which edges are favored by the optimal transport. As $\varepsilon\to 0$, the shadow contracts onto the min-cost geometric corridor.
    - **Design Motivation**: Classical multi-marginal Sinkhorn is mature for general MMOT, but direct application to MAPF lacked a bridge to "why this is an entropic relaxation of P1." This work rigorously connects the Schrödinger bridge to P1, making P2 not just an engineering accelerator but a probabilistically interpretable "prior-aware" solver (structural preferences can be injected via different references $\mathbf{G}$).

3. **P3: Shadow Pruning + LP Recovery of Integrality**:

    - **Function**: Converts the fast but fractional solution of P2 back to executable $\{0,1\}$ integer paths.
    - **Mechanism**: Adds a KL penalty to $\Pi_t$ towards the shadow $\tilde\Pi_t$, linearizing to the objective $\sum_t \sum_{i,j}\pi_{ij,t}(c_{ij,t} - \lambda\log(\tilde\pi_{ij,t}+\delta))$, and prunes all edges with mass $\le\eta$ (i.e., $\Pi_t \subseteq [\tilde\Pi_t]_\eta$). This is equivalent to resolving P1 on the sparse subgraph highlighted by the shadow: still TU, still integer, but the number of variables drops from $|\mathcal{E}|T$ to $\zeta|\mathcal{E}|T$, with $\zeta\in[0.2, 0.4]$ in experiments. Three hyperparameters $\varepsilon, \lambda, \eta$ jointly control the "optimality—scalability" trade-off: $\lambda=\eta=0$ degenerates to P1; increasing $\varepsilon$ blurs the shadow, increases pruning, and raises cost.
    - **Design Motivation**: Treats P2 as a "feature selector" for P3—this is an elegant use of "fractional mass distribution as structural prior" rarely seen in OT literature, preserving P1's optimality certificate while leveraging P2's scalability.

### Loss & Training
This is a non-learning method, with no loss/training; hyperparameters are chosen based on 260 runs at $K=10000$: $\varepsilon=0.2, \lambda=0$ are robust defaults, yielding 4.3% cost gap and 5× speedup. Sinkhorn iterations are few in practice (dozens suffice for pruning).

## Key Experimental Results

### Main Results
On $K = W\times H$ grids (side 50–150, 5% robot density, $T=30$, Gurobi solver), 162 independent runs:

| Method | Solve Time Scaling with $K$ | Speedup | Cost Gap | Integrality |
|--------|-----------------------------|---------|----------|-------------|
| P1 (Original LP) | $O(K^{1.68})$ | 1× | 0% (optimal) | 100% |
| P2 + P3 pipeline | $O(K^{1.15})$ | **3.6× – 7.1×** | **< 10%** | 100% (all solutions verified integer) |

### Ablation Study

| Setting | Key Phenomenon | Notes |
|---------|---------------|-------|
| Edge retention drops from 100% to ~20–40% | Cost gap < 10%, feasibility maintained | Shadow pruning is efficient |
| $\varepsilon = 0.2, \lambda = 0$ (default) | 4.3% cost gap, 5× speedup | Robust balance |
| Increasing $\varepsilon$ | Shadow diffuses, more pruning, cost gap increases | $\varepsilon$ is dominant factor |
| Varying $\lambda$ | Minor effect | Linearized KL weight is secondary |
| Compared to CBM (Ma & Koenig 2016) | P2+P3 more stable at large scale | See Appendix H.5 |

### Key Findings
- Shadow pruning yields greater benefits as problem size increases: as $K\uparrow$, fewer edges suffice for feasibility, with 60–80% of edges pruned.
- TU property is preserved after pruning, which is key to P3's stable integer solutions.
- The trade-off between optimality (P1) and scalability (P2 → P3) is continuously tunable via three hyperparameters, providing engineers with a smooth control slider.

## Highlights & Insights
- Bridging MAPF to MMOT/Schrödinger bridge offers a unified perspective: not only clarifying the source of LP integrality (TU), but also naturally introducing probabilistic Sinkhorn acceleration. This "classical combinatorial optimization + modern OT tools" bridge is transferable to related problems (vehicle routing, multi-commodity flow).
- The idea of "shadow as feature selector" is generalizable: any integer LP with entropic relaxation can use Sinkhorn to identify "important variables," then return to LP for refinement.
- Using $B^t$ exponential cost growth to implicitly enforce min-makespan is smarter than explicit max-min formulations (which break TU), but care is needed for numerical overflow; the paper also provides an $O(\log K)$ binary search alternative.
- **The three hyperparameters $\varepsilon, \lambda, \eta$ provide a continuous "optimality vs scalability" tuning lever**, allowing engineers to smoothly switch between the two extremes as needed, rather than making a binary choice.

## Limitations & Future Work
- Mainly targets **anonymous** MAPF; non-anonymous (fixed robot-goal assignments) requires a more general MMOT formulation, which the authors note is extensible but not implemented.
- Assumes the graph satisfies Assumption 3.1's "no diagonal collision" and related conditions; for real multi-robot systems with kinematic constraints (e.g., turning radius), continuous motion must first be discretized.
- The Schrödinger bridge reference $\mathbf{G}$ must be the Gibbs kernel to yield entropic regularization; other priors (e.g., risk aversion, travel preferences) can be incorporated in form but require solver re-derivation.
- Complexity results are based on grid graph experiments; general sparse graphs and real-time scenarios with dynamic obstacles are not directly validated.

## Related Work & Insights
- **vs CBS / SAT-based MAPF**: This work provides a first-principles explanation from the "polytope integrality" perspective, addressing the lack of guarantees in classical heuristics; CBS may still be faster at moderate scale, but this approach scales better.
- **vs Time-expanded network flow (Yu & LaValle, Ma)**: These also provide LP/IP formulations but do not explicitly prove integrality; this work formalizes it via TU and adds the Schrödinger probabilistic perspective.
- **vs Yau et al. (GNN for low-rank SDP relaxation of CSP)**: Both use "convex relaxation + recovery" for combinatorial problems, but this work applies it to OT/MAPF and requires no learning.
- **vs Sinkhorn-based MMOT (Lin, Haasler, Carlier)**: This is the first application of multi-marginal Sinkhorn to MAPF, a setting requiring both scalability and 0/1 solutions.
- **vs OR community's µMAPF / LNS variants**: These are heuristic high-performance solvers but lack strict optimality guarantees; P1 here can serve as an upper bound and ground truth for such heuristics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified MAPF↔MMOT/Schrödinger bridge perspective is truly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale scaling, parameter sensitivity, and CBM comparison are covered, but lacks dynamic/continuous scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clean, and the three-stage P1/P2/P3 structure is clear and accessible.
- Value: ⭐⭐⭐⭐⭐ Direct engineering significance for large-scale MAPF applications such as warehouse robotics and multi-UAV coordination.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](../../CVPR2026/optimization/otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](../../ICLR2026/optimization/rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](../../ICML2025/optimization/nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)

</div>

<!-- RELATED:END -->
