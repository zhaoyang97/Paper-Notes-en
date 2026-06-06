---
title: >-
  [Paper Note] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges
description: >-
  [ICML 2026][Robotics][MAPF] This paper proves that anonymous multi-agent path finding (MAPF) is a class of **Markovian multi-marginal optimal transport (MMOT)**…
tags:
  - "ICML 2026"
  - "Robotics"
  - "MAPF"
  - "Multi-Marginal Optimal Transport"
  - "Schrödinger bridge"
  - "Total Unimodularity"
  - "Sinkhorn"
date: 2026-05-08
content_hash: ad83957849669c50
---

# Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges

**Conference**: ICML 2026  
**arXiv**: [2605.10917](https://arxiv.org/abs/2605.10917)  
**Code**: Not released  
**Area**: Robotics / Multi-Agent Path Finding / Optimal Transport  
**Keywords**: MAPF, Multi-Marginal Optimal Transport, Schrödinger bridge, Total Unimodularity, Sinkhorn

## TL;DR
This paper proves that anonymous multi-agent path finding (MAPF) is a class of **Markovian multi-marginal optimal transport (MMOT)**, compressing the original $K^{T+1}$-dimensional transport tensor into a polynomial-scale LP (P1) and guaranteeing integer optimality through total unimodularity. It then generalizes this to a Schrödinger bridge to obtain a Sinkhorn-style entropic relaxation (P2) yielding a "shadow transport." Finally, it performs pruning on the shadow and solves an LP (P3) to recover integer solutions, achieving 3.6×–7.1× speedup with <10% cost gap under $O(K^{1.15})$ complexity.

## Background & Motivation

**Background**: Classic MAPF solvers (resolving collision-free paths for multiple robots on a shared graph) primarily rely on Conflict-Based Search (CBS), SAT encoding, or time-expanded flow networks. While optimal algorithms are feasible for medium scales, large-scale anonymous MAPF (where any robot can reach any target) remains a challenge.

**Limitations of Prior Work**: Although existing IP/LP formulations (e.g., time-expanded network flow) can provide optimal solutions, **the source of integer optimality in their LP relaxations has not been systematically characterized.** Practitioners know empirically that integer solutions exist in certain cases, but a unified framework identifying "structural conditions sufficient for total unimodularity (TU)" is missing. Furthermore, these methods do not scale well to large problems (thousands of nodes, tens of thousands of variables).

**Key Challenge**: Achieving both "optimality and integrality" usually implies ILP (NP-hard), while "scalability" often leads to distributed heuristics (without guarantees). MAPF lacks a unified framework that bridges these two ends with theoretical guarantees and large-scale efficiency.

**Goal**: 1) Establish a unified optimal transport perspective for MAPF; 2) Prove that the resulting LP is polynomial-time solvable and yields integer solutions; 3) Derive a scalable Sinkhorn-style algorithm via probabilistic relaxation (Schrödinger bridge); 4) Convert the benefits of probabilistic relaxation back into executable integer trajectories.

**Key Insight**: View all possible joint trajectories of $N$ robots over $T$ steps as a $(T+1)$-order tensor $\mathbf{P}\in\mathbb{R}_{\ge 0}^{K\times\cdots\times K}$, where each entry is the probability mass of a specific path. MAPF becomes finding the minimum-cost transport plan satisfying start/end marginals. This is naturally an MMOT. Since robot motions are Markovian, the tensor admits a standard factorized form $\mathbf{P}_{i_0,\ldots,i_T} \propto \prod_t [\Pi_t]_{i_{t-1}i_t}$, reducing potential variables from $O(K^{T+1})$ to $O(K^2T)$.

**Core Idea**: MAPF = Markovian MMOT; its LP formulation under the anonymous setting is totally unimodular under natural assumptions, ensuring polynomial-time integer optimality. Probabilistic relaxation via Schrödinger bridges provides a scalable solver, and the integer solution is recovered through pruning and LP refinement.

## Method

### Overall Architecture
The pipeline follows a three-step process: (1) **P1**: Formulate MAPF as an LP using Markovian tensor parameterization for transport plans $\{\Pi_t\}_{t=1}^T$ between adjacent time steps, and prove TU for integer optimality; (2) **P2**: Formulate the Schrödinger bridge using a Gibbs kernel $\bar g_{ij,t} \propto \exp(-c_{ij,t}/\varepsilon)$ as a reference distribution, obtaining an entropic regularization of P1 solved by multi-marginal Sinkhorn for "shadow" fractional transport $\tilde\Pi_t$; (3) **P3**: Perform graph pruning using high-quality edges from the shadow (retaining edges with high mass) and re-solve the LP on the reduced graph to recover integer solutions $\hat\Pi_t$. This pipeline bridges optimality and scalability, reducing complexity from $O(K^{1.68})$ for classic IPMs to $O(K^{1.15})$.

### Key Designs

1. **P1: MAPF MMOT-LP and TU Guarantee**:

    - **Function**: Provide a polynomial-time solvable LP formulation for anonymous MAPF that guarantees $\{0,1\}$ integer solutions.
    - **Mechanism**: Decision variables are the transition matrices $\{\Pi_t\}$ between adjacent time steps. The objective is $\sum_t \langle \Pi_t, C_t\rangle$, subject to three key constraints: **gluing** (mass conservation $\Pi_t^\top\mathbf{1} = \Pi_{t+1}\mathbf{1}$) to ensure Markovian consistency, **terminal** constraints fixing start/end distributions $\Pi_1\mathbf{1}=\mu, \Pi_T^\top\mathbf{1}=\nu$, and **vertex-capacity** $0\le\Pi_t^\top\mathbf{1}\le\mathbf{1}$ to prevent collisions. Assumption 3.1 identifies natural structures (self-loops allowed, non-overlapping edges can be parallel, move cost > wait cost > 0, target wait cost = 0). Lemma 3.3 proves that the constraint matrix is totally unimodular (TU) under these assumptions, meaning all basic feasible solutions (vertices) are integers. Theorem 3.4 translates these integer solutions back to collision-free paths.
    - **Design Motivation**: Previous time-expanded IP for MAPF lacked a clear answer for "why integer solutions exist." This paper provides a **first-principles** explanation via TU and unifies various objectives like min-cost or min-makespan by adjusting $C_t$.

2. **P2: Schrödinger Bridge and Entropic Relaxation**:

    - **Function**: Utilize a probabilistic framework to obtain a massively parallelizable Sinkhorn solver, generating a "shadow map" of important edges.
    - **Mechanism**: Generalizes P1 to find a joint distribution $\mathbf{P}$ that minimizes $\mathrm{KL}(\mathbf{P}\,\|\,\mathbf{G})$ over constraint set $\mathcal{C}$, where $\mathbf{G}$ is a reference Markovian tensor. When the reference is a Gibbs kernel $g_{ij,t}=\exp(-c_{ij,t}/\varepsilon)$, Lemma 4.2 reduces the objective to the entropic regularization of P1: $\min \sum_t \langle\Pi_t,C_t\rangle + \varepsilon\sum_{i,j}\pi_{ij,t}(\log\pi_{ij,t}-1)$. This P2 formulation is solved via multi-marginal Sinkhorn block coordinate descent. By relaxing vertex-capacity, the "shadow" solution $\tilde\Pi_t$ reveals where the optimal transport mass tends to flow.
    - **Design Motivation**: While Sinkhorn is mature for general MMOT, its application to MAPF requires a bridge showing it is an entropic relaxation of P1. This makes P2 a probabilistic "prior-aware" solver rather than just an engineering accelerator.

3. **P3: Shadow Pruning + LP Recovery**:

    - **Function**: Convert the fast but fractional solutions from P2 back into executable $\{0,1\}$ integer paths.
    - **Mechanism**: A KL penalty is added to pull $\Pi_t$ toward the shadow $\tilde\Pi_t$, linearized to the objective $\sum_t \sum_{i,j}\pi_{ij,t}(c_{ij,t} - \lambda\log(\tilde\pi_{ij,t}+\delta))$. All edges with mass $\le\eta$ are pruned. This is equivalent to re-solving P1 on a "sparse subgraph" highlighted by the shadow. This sub-problem remains TU and integer, but the number of variables is reduced from $|\mathcal{E}|T$ to $\zeta|\mathcal{E}|T$, where experiments show $\zeta\in[0.2, 0.4]$. Three hyperparameters $\varepsilon, \lambda, \eta$ regulate the "optimality-scalability" trade-off.
    - **Design Motivation**: Using P2 as a "feature selector" for P3 is an elegant use of fractional mass as a structural prior, retaining P1's optimality certificates while inheriting P2's scalability.

## Key Experimental Results

### Main Results
On $K = W\times H$ grid graphs (side lengths 50–150, 5% robot density, $T=30$, using the Gurobi solver):

| Method | Time Scaling vs $K$ | Speedup | Cost Gap | Integrality |
|------|--------------------|---------|---------|--------|
| P1 (Original LP) | $O(K^{1.68})$ | 1× | 0% (Opt) | 100% |
| P2 + P3 pipeline | $O(K^{1.15})$ | **3.6× – 7.1×** | **< 10%** | 100% |

### Ablation Study

| Setting | Observation | Explanation |
|------|---------|------|
| Edge retention 100% $\to$ ~20-40% | Cost gap < 10%, feasibility maintained | Shadow pruning is highly efficient |
| $\varepsilon = 0.2, \lambda = 0$ (Default) | 4.3% cost gap, 5× speedup | Robust balance |
| Increasing $\varepsilon$ | Shadow diffuses, more pruning, higher cost gap | $\varepsilon$ is the dominant factor |
| Compared with CBM (Ma & Koenig 2016) | P2+P3 more stable at large scales | See Appendix H.5 |

### Key Findings
- Shadow pruning yields higher returns as problem size increases; for large $K$, fewer edges are needed to maintain feasibility (60-80% of edges can be pruned).
- TU property is preserved after pruning, which is central to P3's ability to consistently provide integer solutions.
- The trade-off between optimality (P1) and scalability (P2 $\to$ P3) is continuously adjustable via the three hyperparameters.

## Highlights & Insights
- Mapping MAPF to MMOT/Schrödinger bridges provides a beautiful unified perspective: it clarifies the source of LP integrality (TU) and naturally introduces scalable Sinkhorn acceleration. This "classic combinatorial optimization + modern OT tools" bridge is highly transferable to problems like vehicle routing or multi-commodity flow.
- The concept of "shadow as feature selector" is powerful: for any integer LP with an entropic relaxation, one can use Sinkhorn to identify important variables before refinement.
- Using an exponentially growing cost $B^t$ to implicitly approximate min-makespan is more clever than explicit max-min formulations (which break TU), though it requires care regarding numerical precision.

## Limitations & Future Work
- Primarily targets **anonymous** MAPF; non-anonymous settings (fixed agent-to-target assignments) require a more general MMOT formulation.
- Assumes the graph satisfies structure like Assumption 3.1; real-world robots with kinematic constraints (e.g., turning radii) require careful discretization.
- The Schrödinger bridge reference $\mathbf{G}$ must be a Gibbs kernel to reduce to entropic regularization; other priors (e.g., risk aversion) are theoretically possible but require re-deriving the solver.
- Complexity conclusions are based on grid graph experiments; performance on general sparse graphs or dynamic obstacles needs further validation.

## Related Work & Insights
- **vs CBS / SAT-based MAPF**: This paper provides a first-principles explanation via "polytope integrality," filling the gap left by heuristic methods. While CBS might be faster for medium scales, this method scales better.
- **vs Time-expanded network flow**: While others have used LP/IP, this work explicitly formalizes the TU guarantee and adds the probabilistic Schrödinger perspective.
- **vs Sinkhorn-based MMOT**: This is the first practical application of multi-marginal Sinkhorn to a domain requiring strict $0/1$ solutions like MAPF.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The MAPF↔MMOT/Schrödinger bridge perspective is genuinely original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong large-scale scaling and sensitivity analysis, though missing dynamic/continuous scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clean; the P1/P2/P3 structure is logical.
- Value: ⭐⭐⭐⭐⭐ High engineering significance for large-scale warehouse robotics and UAV coordination.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](../../NeurIPS2025/robotics/grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)
- [\[ICCV 2025\] Certifiably Optimal Anisotropic Rotation Averaging](../../ICCV2025/robotics/certifiably_optimal_anisotropic_rotation_averaging.md)
- [\[AAAI 2026\] To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance](../../AAAI2026/robotics/to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)
- [\[ICML 2026\] WestWorld: Scalable Trajectory World Models with Knowledge Encoding](westworld_a_knowledge-encoded_scalable_trajectory_world_model_for_diverse_roboti.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)

</div>

<!-- RELATED:END -->
