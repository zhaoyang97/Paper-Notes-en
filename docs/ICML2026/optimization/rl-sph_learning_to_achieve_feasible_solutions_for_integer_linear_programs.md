---
title: >-
  [Paper Note] RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs
description: >-
  [ICML 2026][Optimization][Integer Linear Programming] This paper proposes RL-SPH—an end-to-end reinforcement learning heuristic that does not rely on external ILP solvers and can independently produce 100% feasible solut…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Integer Linear Programming"
  - "Start Primal Heuristic"
  - "Reinforcement Learning"
  - "Graph Transformer"
  - "Feasibility Reward"
date: 2026-05-08
content_hash: e9735cad97526bd0
---

# RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs

**Conference**: ICML 2026  
**arXiv**: [2411.19517](https://arxiv.org/abs/2411.19517)  
**Code**: Repository not directly provided in the paper  
**Area**: Reinforcement Learning / Combinatorial Optimization  
**Keywords**: Integer Linear Programming, Start Primal Heuristic, Reinforcement Learning, Graph Transformer, Feasibility Reward

## TL;DR
This paper proposes RL-SPH—an end-to-end reinforcement learning heuristic that does not rely on external ILP solvers and can independently produce 100% feasible solutions. By leveraging "feasibility reward + two-phase policy + feasibility-aware neighborhood search," the Graph Transformer Agent reduces the average primal gap by 28.6 times on ILPs with non-binary integer variables.

## Background & Motivation
**Background**: When solving NP-hard integer linear programs (ILPs), primal heuristics are used to quickly find feasible solutions. Recent end-to-end learning-based primal heuristics (E2EPH) employ GNNs to learn commonalities across multiple ILP instances, directly predicting variable assignments, which are then repaired into feasible solutions by solvers like Gurobi/SCIP.

**Limitations of Prior Work**: Existing E2EPH methods almost entirely rely on external ILP solvers for feasibility, as ML predictions often violate constraints if not precise. A few attempts to independently generate feasible solutions (DDIM, DiffILO) still struggle with stable convergence and have feasibility rates far below 100%. Moreover, most methods target only 0/1 binary variables; when faced with non-binary integer variables (NBI) with larger value ranges, the probability of constraint violation increases significantly.

**Key Challenge**: There is a fundamental mismatch between the "global one-shot output" of end-to-end ML prediction and the "element-wise hard satisfaction" required by ILP constraints—end-to-end supervised learning uses MSE/CE for training, lacking explicit feedback channels for feasibility signals.

**Goal**: To train a start primal heuristic that does not rely on solvers and can autonomously produce feasible solutions (including non-binary integers), and to validate feasibility rate and solution quality on both binary/non-binary and various CO benchmarks.

**Key Insight**: The authors frame the search for feasible solutions as a sequential decision process—each variable's "increment by 1 / stay / decrement by 1" is an action, constraint violation is a negative reward, and achieving feasibility itself is a large reward, thus embedding the non-differentiable feasibility constraint directly into the RL reward.

**Core Idea**: By using "two-phase reward + feasibility-aware neighborhood selection + ILP-GT Graph Transformer," the agent first quickly enters the feasible region, then iteratively optimizes within it, all without any external solver.

## Method

### Overall Architecture
The input is any ILP instance $M=(\mathbf{c},\mathbf{A},\mathbf{b},l,u)$. RL-SPH first encodes it as a "variable-constraint" bipartite graph and obtains an initial solution $\mathbf{x}_0$ via LP relaxation or random assignment. At each step $t$, feasibility-aware selection picks $\tilde n = 2\lceil\log_2 n\rceil$ mutable variables (others are frozen), which are fed into the ILP-GT actor to predict the action set $\mathcal{A}_t$ (each variable chooses from $\{+1,0,-1\}$), updating the solution $\mathbf{x}_{t+1}$. The left-hand side $\mathbf{lhs}_{t+1}=\mathbf{Ax}_{t+1}$, feasibility vector $\mathbf{f}_{t+1}=\mathbf{b}-\mathbf{lhs}_{t+1}$, and objective value $obj_{t+1}$ are recalculated. The critic's value estimate is used for actor training; whenever a better feasible solution than the incumbent is found, $\mathbf{x}_b, obj_b$ are updated. The process continues until the external time budget is exhausted or all baselines complete their search.

### Key Designs

1. **Two-Phase Feasibility Reward**:

    - **Function**: Decouples the two distinct subgoals of "achieving feasibility first" and "optimizing within the feasible region," each driven by different rewards.
    - **Mechanism**: The main reward in phase 1 is $\mathcal{R}_{t,\text{F}}=\mathcal{R}_{t,\text{bound}}+\frac{1}{\sqrt{\tilde n}}\mathcal{R}_{t,\text{const}}$, where $\mathcal{R}_{t,\text{bound}}=-\sum_i \mathbb{I}(x_{t+1,i}\notin[l_i,u_i])$ penalizes out-of-bounds variables, and $\mathcal{R}_{t,\text{const}}=\sum_j \min(f_{t+1,j},0)-\min(f_{t,j},0)$ rewards the "improvement" for each violated constraint. Only when all variables are within bounds and constraints are improved is the $\Delta obj$ term added. Upon entering phase 2, the reward switches to $\mathcal{R}_{t,\text{p2}}$, giving positive $\Delta obj$ reward for feasible solutions, negative $\mathcal{R}_{t,\text{F}}$ for infeasible ones, and using a toward-optimal bias $\alpha=2$ to encourage exploration toward regions better than the incumbent. Additionally, staying in place is penalized by $-100$ to prevent agent stagnation.
    - **Design Motivation**: The authors prove Proposition 1—so long as $\mathcal{R}_{t,\text{const}}>0$ and $\mathcal{R}_{t,\text{bound}}=0$ persist, the agent will eventually enter the feasible region. This transforms feasibility, a hard constraint, into a "reward-feasibility alignment" theoretical guarantee for the agent, avoiding the implicit feasibility blind spot of end-to-end supervised methods.

2. **ILP-Specific Graph Transformer (ILP-GT)**:

    - **Function**: Encodes the selected $\tilde n$ variables, their objective coefficients and constraint rows $(\mathbf{c}^\top|\mathbf{A})$, and reward context tokens (phase/obj/feasibility vectors) into a Transformer encoder; the actor outputs actions, the critic outputs values.
    - **Mechanism**: Equilibration scaling normalizes $(\mathbf{c}^\top|\mathbf{A})$ to $[-1,1]$ for stable training; continuous variable values are encoded using Periodic Embedding $\operatorname{PE}(z)=\oplus(\sin(\tilde z),\cos(\tilde z))$ (where $\tilde z=[2\pi w_1 z,\dots,2\pi w_k z]$), combined with a binary `bnd_lim` bit indicating boundary contact. The reward context includes phase indicator, PE-encoded $obj$, and $\mathbf{f}_t$ normalized by $\sqrt{|\mathbf{b}|+|\mathbf{b}-\mathbf{lhs}_t|}$. Finally, phase-separated actor/critic heads and a shared backbone form an input sequence of length $\tilde n+3$.
    - **Design Motivation**: Traditional GCNs aggregate only first-order neighborhoods and are insufficient for modeling long-range variable dependencies in ILPs. The Transformer's fully connected attention naturally captures cross-constraint dependencies, and PE addresses the challenge of embedding unbounded integer values.

3. **Feasibility-Aware Neighborhood Search**:

    - **Function**: At each step, only $\tilde n=p+q$ variables most likely to improve feasibility are unfrozen, avoiding blind search in $n$-dimensional space.
    - **Mechanism**: In phase 1, $p$ seed variables are randomly sampled weighted by their frequency in violated constraints, then $q$ neighbors co-occurring with seeds in the same violated constraints are selected for joint action. In phase 2, seeds with ample slack (less likely to cause constraint oscillation) are prioritized. $p=q=\lceil\log_2 n\rceil$ limits Transformer input size. In phase 1, state rollback occurs only for out-of-bounds variables; in phase 2, rollback happens unless the new solution strictly improves the incumbent, ensuring monotonic search progress.
    - **Design Motivation**: Single-variable local search is too slow, and naive LNS requires a feasible initial solution. This method integrates RENS's "subproblem repair" and LNS's "large neighborhood" concepts into an RL framework independent of solvers, enabling the agent to act on a $\Theta(\log n)$-sized variable subset simultaneously.

### Loss & Training
An Actor-Critic framework is used: the actor encourages actions with $\mathcal{R}_{t,\text{total}}>V_\theta$ and suppresses low-reward actions; the critic uses regression loss to approximate true returns. Each instance remains in phase 1 for a preset number of steps to ensure sufficient training before switching instances. Training requires only 1,000 instances, with an average training time of 30 minutes per instance, about 14.7× faster than E2EPH baselines.

## Key Experimental Results

### Main Results
On five NP-hard ILP benchmarks (MVC, IS, SC, CA, NBI), RL-SPH is compared with four types of SPH (FP / RENS / DHF / RHF) and three E2EPH methods (PAS / DDIM / DiffILO), all under a unified 1000-second wall-clock budget.

| Metric | RL-SPH | Existing SPH/E2EPH Baseline | Gain |
|--------|--------|-----------------------------|------|
| Feasibility Rate (FR) | 100% (5/5 benchmarks) | Most <100%, some timeout failures | Only method with full feasibility |
| Avg. Primal Gap | 1× | 28.6× | 28.6× better |
| Avg. Primal Integral | 1× | 2.6× | 2.6× better |
| Training Time | ~30 min | E2EPH avg. 7+ h | ~14.7× speedup |

### Ablation Study

| Configuration | Key Phenomenon | Notes |
|---------------|---------------|-------|
| Full RL-SPH | 100% FR + lowest gap | Baseline |
| Remove phase-separated head | Gap rises significantly | Two-phase reward requires two-phase network heads |
| $\alpha=1$ (no toward-optimal bias) | Phase 2 exploration stagnates | Exploration bias is essential |
| Naive GNN replaces ILP-GT | Performance drops on large instances | Long-range dependencies require Transformer |
| Single-variable actions (no feasibility-aware selection) | Convergence speed severely degrades | Large neighborhood simultaneous updates are key for acceleration |

### Key Findings
- The two-phase reward is the fundamental guarantee for 100% feasibility: Proposition 1 strictly links "reward > 0" to "eventual entry into the feasible region," and empirically, phase 1 typically reaches feasibility within dozens of steps.
- RL-SPH's advantage is most pronounced on non-binary integer benchmarks (NBI)—existing E2EPH methods see constraint violation rates soar as variable domains widen, while RL's $\{+1,0,-1\}$ incremental actions naturally suit integer search.
- RL-SPH can serve as a warm-start for LNS and local branching, significantly reducing subsequent solving time; it also maintains feasibility advantages on real MIPLIB instances.

## Highlights & Insights
- Translates feasibility constraints into reward signals, with a theoretical Proposition proving reward-feasibility alignment—rare among ML for CO works to provide formal feasibility guarantees.
- The two-phase design elegantly resolves the often-confused dual objectives of "feasibility first" and "then optimality," preventing RL from stalling under sparse rewards.
- The dual-track encoding of Periodic Embedding + `bnd_lim` ("continuous value + discrete boundary flag") can be directly transferred to any scenario requiring unbounded integers as Transformer input (e.g., scheduling, layout).
- The adaptive neighborhood size $\tilde n=2\lceil\log_2 n\rceil$ enables scalability to ILPs with tens of thousands of variables.

## Limitations & Future Work
- The authors acknowledge RL-SPH lacks a built-in termination condition and relies on baseline completion time or external budget; automatically detecting "local optimality reached" remains open.
- The integer action magnitude is fixed at $\pm 1$ (see Appendix I.7); for ILPs with very large value ranges (e.g., transport problems with $10^4$ scale), hierarchical actions or adaptive step sizes may be needed.
- The initial solution depends on LP relaxation or random assignment; future work could combine supervised warm-starts to further reduce phase 1 steps.
- There is no explicit modeling or proof of objective optimality—only "high-quality feasible solutions" are obtained, leaving a gap with the exact guarantees of branch-and-bound.

## Related Work & Insights
- **vs PAS / Predict-and-Search (Han et al. 2023)**: PAS predicts fixed + trust region but still relies on SCIP for repair; RL-SPH completely bypasses solvers and supports non-binary integers.
- **vs DiffILO (Geng et al. 2025)**: DiffILO samples solutions directly via diffusion models but has unstable feasibility rates; RL-SPH achieves 100% FR through reward alignment.
- **vs Classic RENS / LNS**: RL-SPH draws on RENS's "fix part of variables to solve subproblems" and LNS's "large neighborhood search," but replaces solvers and heuristic rules with RL.
- **Insight**: In other learning tasks where outputs must satisfy hard constraints (e.g., circuit routing, vehicle routing), using constraint improvement as reward + two-phase strategy may also be effective.

## Rating
- Novelty: ⭐⭐⭐⭐ First E2EPH with theoretical feasibility alignment guarantee, naturally supports non-binary integers
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks + 7 baselines + MIPLIB + solver integration + hyperparameter analysis, comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Method diagrams and Algorithm 1 are clear, reward formulas require careful reading
- Value: ⭐⭐⭐⭐ Provides an engineering-ready route for "ML independently solves ILP," directly reusable for LNS warm-start

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICML 2026\] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICLR 2026\] Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)](../../ICLR2026/optimization/adaptive_rollout_allocation_for_online_reinforcement_learning_with_verifiable_re.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)

</div>

<!-- RELATED:END -->
