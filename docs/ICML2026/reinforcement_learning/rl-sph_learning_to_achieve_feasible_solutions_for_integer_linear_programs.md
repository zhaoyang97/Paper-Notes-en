---
title: >-
  [Paper Note] RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs
description: >-
  [ICML 2026][Reinforcement Learning][Integer Linear Programming] This paper proposes RL-SPH—an end-to-end reinforcement learning heuristic that does not rely on external ILP solvers and independently generates 100% feasib…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Integer Linear Programming"
  - "Start Primal Heuristic"
  - "Graph Transformer"
  - "Feasibility Reward"
date: 2026-05-08
content_hash: 0b64ee5cc7b0ca19
---

# RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs

**Conference**: ICML 2026  
**arXiv**: [2411.19517](https://arxiv.org/abs/2411.19517)  
**Code**: Repository address not directly provided in the paper  
**Area**: Reinforcement Learning / Combinatorial Optimization  
**Keywords**: Integer Linear Programming, Start Primal Heuristic, Reinforcement Learning, Graph Transformer, Feasibility Reward

## TL;DR
This paper proposes RL-SPH—an end-to-end reinforcement learning heuristic that does not rely on external ILP solvers and independently generates 100% feasible solutions. Using a "feasibility reward + dual-phase strategy + feasibility-aware neighborhood search," it enables a Graph Transformer Agent to reduce the average primal gap by 28.6x on ILP instances containing non-binary integer variables.

## Background & Motivation
**Background**: When solving NP-hard Integer Linear Programs (ILP), primal heuristics are used to quickly find feasible solutions. Recent end-to-end learning-based primal heuristics (E2EPH) use GNNs to learn commonalities across multiple ILP instances, directly predicting variable values which are then repaired into feasible solutions by solvers like Gurobi/SCIP.

**Limitations of Prior Work**: The feasibility of existing E2EPH almost entirely depends on external ILP solvers because imprecise ML predictions violate constraints. A few works attempting independent feasibility (DDIM, DiffILO) still struggle with stable convergence, with feasibility rates far below 100%. Furthermore, most methods target 0/1 binary variables; the probability of constraint violation increases exponentially when encountering Non-Binary Integer (NBI) variables with larger ranges.

**Key Challenge**: There is a fundamental misalignment between the "global one-shot output" of end-to-end ML predictions and the "element-wise hard satisfaction" of ILP constraints. End-to-end supervised learning trained with MSE/CE lacks an explicit feedback channel for feasibility signals.

**Goal**: To train a start primal heuristic that is solver-independent and capable of self-consistently producing feasible solutions (including NBI variables), and to validate feasibility rates and solution quality across binary/non-binary CO benchmarks.

**Key Insight**: The authors treat the search for a feasible solution as a sequential decision process—where "increment by 1 / no change / decrement by 1" for each variable serves as an action. Constraint violations act as negative rewards, while achieving feasibility serves as a large positive reward, thereby embedding non-differentiable feasibility constraints directly into the RL reward.

**Core Idea**: Utilize a "dual-phase reward + feasibility-aware neighborhood selection + ILP-GT (Graph Transformer)" to allow the Agent to first enter the feasible region rapidly and then optimize within it, without requiring any external solvers throughout the process.

## Method

### Overall Architecture
The input is an arbitrary ILP instance $M=(\mathbf{c},\mathbf{A},\mathbf{b},l,u)$. RL-SPH first encodes it into a "variable-constraint" bipartite graph and obtains an initial solution $\mathbf{x}_0$ via LP relaxation or random assignment. At each step $t$, feasibility-aware selection identifies $\tilde n = 2\lceil\log_2 n\rceil$ changeable variables (freezing others), which are fed into the ILP-GT actor to predict action sets $\mathcal{A}_t$ (choosing from $\{+1,0,-1\}$ for each variable). The solution is updated to $\mathbf{x}_{t+1}$, and the left-hand side $\mathbf{lhs}_{t+1}=\mathbf{Ax}_{t+1}$, feasibility vector $\mathbf{f}_{t+1}=\mathbf{b}-\mathbf{lhs}_{t+1}$, and objective $obj_{t+1}$ are recalculated. Critic estimates are used for actor training; $\mathbf{x}_b, obj_b$ are updated whenever a better feasible solution than the incumbent is found. The process continues until the external time budget is exhausted or all baselines finish searching.

### Key Designs

1.  **Two-Phase Reward**:
    *   **Function**: Decouples the two distinct sub-objectives of "achieving feasibility first" and "optimizing within the feasible region," driven by different rewards.
    *   **Mechanism**: In phase 1, the primary reward is $\mathcal{R}_{t,\text{F}}=\mathcal{R}_{t,\text{bound}}+\frac{1}{\sqrt{\tilde n}}\mathcal{R}_{t,\text{const}}$, where $\mathcal{R}_{t,\text{bound}}=-\sum_i \mathbb{I}(x_{t+1,i}\notin[l_i,u_i])$ penalizes boundary violations, and $\mathcal{R}_{t,\text{const}}=\sum_j \min(f_{t+1,j},0)-\min(f_{t,j},0)$ rewards "improvement" for each violated constraint. The $\Delta obj$ term is only added if variables are within bounds and constraints improve. After entering phase 2, it switches to $\mathcal{R}_{t,\text{p2}}$, giving positive $\Delta obj$ for feasible improvements and negative $\mathcal{R}_{t,\text{F}}$ for infeasible solutions, with a toward-optimal bias $\alpha=2$ to encourage exploration toward regions better than the incumbent. Additionally, a $-100$ penalty is applied to idling to prevent stagnation.
    *   **Design Motivation**: The authors prove Proposition 1—as long as $\mathcal{R}_{t,\text{const}}>0$ and $\mathcal{R}_{t,\text{bound}}=0$ hold continuously, the agent will eventually enter the feasible region. This converts the hard constraint of feasibility into a theoretical guarantee of "reward-feasibility alignment" for the agent, avoiding the implicit feasibility blind spots of end-to-end supervised methods.

2.  **Graph Transformer for ILP (ILP-GT)**:
    *   **Function**: Encodes the $\tilde n$ selected variables at each step, their objective coefficients and constraint rows $(\mathbf{c}^\top|\mathbf{A})$, and "reward context tokens" like phase/obj/feasibility vectors into a Transformer encoder.
    *   **Mechanism**: Equilibration scaling normalizes $(\mathbf{c}^\top|\mathbf{A})$ to $[-1,1]$ to stabilize training. Continuous variable values are encoded via Periodic Embedding $\operatorname{PE}(z)=\oplus(\sin(\tilde z),\cos(\tilde z))$ where $\tilde z=[2\pi w_1 z,\dots,2\pi w_k z]$, paired with a `bnd_lim` binary bit indicator. The reward context includes phase identifiers, PE-transformed $obj$, and $\mathbf{f}_t$ normalized by $\sqrt{|\mathbf{b}|+|\mathbf{b}-\mathbf{lhs}_t|}$. Finally, a phase-separated actor/critic head with a shared backbone forms an input sequence of length $\tilde n+3$.
    *   **Design Motivation**: Standard GCNs only aggregate first-order neighborhoods, which is insufficient for modeling long-range variable correlations in ILPs. The all-to-all attention of Transformers naturally captures cross-constraint dependencies, while PE resolves the embedding issues of unbounded integer values.

3.  **Feasibility-Aware Search Strategy**:
    *   **Function**: Only allows $\tilde n=p+q$ variables with the most potential to improve feasibility to change at each step, avoiding blind searches in $n$-dimensional space.
    *   **Mechanism**: In phase 1, $p$ seed variables are weighted-randomly sampled based on "frequency of appearance in violated constraints," followed by $q$ neighbors that co-occur most frequently in those violations. In phase 2, priority is given to seed variables with ample slack that are unlikely to cause constraint oscillation. Scaling $p=q=\lceil\log_2 n\rceil$ limits the Transformer input size. Additionally, phase 1 only rolls back state if a variable exceeds bounds, while phase 2 rolls back whenever a new solution does not strictly improve the incumbent, ensuring monotonic search progress.
    *   **Design Motivation**: Single-variable local search is too slow, and naive LNS requires a feasible initial solution. This merges the "sub-problem repair" concept of RENS with the "large neighborhood" concept of LNS into a solver-independent RL framework, allowing the agent to manipulate a $\Theta(\log n)$ subset of variables simultaneously.

### Loss & Training
Uses Actor-Critic: the actor encourages actions where $\mathcal{R}_{t,\text{total}}>V_\theta$ and suppresses low-reward actions; the critic uses regression loss to approximate the true return. Each instance stays in phase 1 for a preset number of steps to ensure sufficient training before switching. Training requires only 1,000 instances, with an average training time of 30 minutes per instance, roughly 14.7x faster than E2EPH baselines.

## Key Experimental Results

### Main Results
RL-SPH is compared against four types of SPH (FP / RENS / DHF / RHF) and three E2EPH (PAS / DDIM / DiffILO) on 5 NP-hard ILP benchmarks (MVC, IS, SC, CA, NBI), with a unified 1000-second wall-clock budget.

| Metric | RL-SPH | Existing SPH/E2EPH baselines | Gain |
|----------|--------|------------------------|------|
| Feasibility Rate (FR) | 100% (5/5 benchmarks) | Most <100%, some timeout failures | Only one fully feasible |
| Avg. Primal Gap | 1× | 28.6× | 28.6× Better |
| Avg. Primal Integral | 1× | 2.6× | 2.6× Better |
| Training Time | ~30 min | E2EPH avg. 7+ h | ~14.7× Speedup |

### Ablation Study

| Configuration | Key Observation | Explanation |
|------|---------|------|
| Full RL-SPH | 100% FR + Lowest gap | Baseline |
| W/o phase-separated head | Gap significantly increases | Dual-phase rewards require matching network heads |
| $\alpha=1$ (No toward-optimal bias) | Phase 2 exploration stalls | Exploration bias is indispensable |
| Naive GNN instead of ILP-GT | Performance drops on large instances | Long-range dependencies require Transformers |
| Single-variable actions | Convergence speed severely degrades | Multi-variable updates in large neighborhoods are key to speed |

### Key Findings
- Dual-phase rewards are the fundamental guarantee for 100% feasibility: Proposition 1 strictly links "reward > 0" with "eventual entry into the feasible region." Empirically, phase 1 usually enters the feasible region within dozens of steps.
- RL-SPH shows the most significant advantage on non-binary integer (NBI) benchmarks—existing E2EPH suffer from surging constraint violation rates as variable domains widen, while RL's $\{+1,0,-1\}$ incremental actions are naturally suited for integer search.
- RL-SPH can serve as a warm-start for LNS or local branching, significantly reducing subsequent solving time; it maintains feasibility advantages on real MIPLIB instances.

## Highlights & Insights
- "Translating" feasibility constraints into reward signals and providing theoretical proof for reward-feasibility alignment is a rare formal feasibility guarantee in ML for CO literature.
- The dual-phase design cleverly resolves the conflict between "achieving feasibility" and "achieving optimality," preventing RL from getting stuck under sparse rewards.
- The "continuous value + discrete bound flag" dual-track encoding of Periodic Embedding + `bnd_lim` can be migrated to any scenario requiring unbounded integers to be fed into Transformers (e.g., scheduling, layout).
- The adaptive neighborhood size $\tilde n=2\lceil\log_2 n\rceil$ allows the method to scale to ILPs with tens of thousands of variables.

## Limitations & Future Work
- The authors acknowledge that RL-SPH lacks a built-in termination condition, relying on baseline completion time or external budgets; automatically determining "local optimality" remains an open problem.
- Integer action magnitude is fixed at $\pm 1$ (discussed in Appendix I.7); for ILPs with huge value ranges (e.g., transport volumes of $10^4$), hierarchical actions or adaptive step sizes might be necessary.
- Initial solution depends on LP relaxation or randomness; future work could combine supervised warm-starts to further reduce phase 1 steps.
- It does not explicitly model optimality proofs for the objective function, only "obtaining high-quality feasible solutions," leaving a gap between this and the exact guarantees of branch-and-bound.

## Related Work & Insights
- **vs PAS / Predict-and-Search (Han et al. 2023)**: PAS still requires SCIP for repair after fixed + trust region prediction; RL-SPH skips the solver entirely and applies to non-binary integers.
- **vs DiffILO (Geng et al. 2025)**: DiffILO uses diffusion models to sample solutions directly, but feasibility rates are unstable; RL-SPH achieves 100% FR through reward alignment.
- **vs Classic RENS / LNS**: RL-SPH draws from RENS's idea of "fixing a subset of variables to solve sub-problems" and LNS's "large neighborhood search," but replaces solvers and heuristic rules with RL.
- **Insight**: In other learning tasks where outputs must satisfy hard constraints (e.g., circuit routing, VRP), treating constraint improvement as a reward combined with dual-phase strategies may be equally applicable.

## Rating
- Novelty: ⭐⭐⭐⭐ First E2EPH with theoretical feasibility alignment guarantees, naturally supporting non-binary integers.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 benchmarks + 7 baselines + MIPLIB + solver integration + hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ Method diagrams and Algorithm 1 are clear; reward formula cases require careful reading.
- Value: ⭐⭐⭐⭐ Provides an engineering path for "ML-independent ILP solving," usable as a warm-start for LNS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](../../AAAI2026/reinforcement_learning/deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](../../ACL2026/reinforcement_learning/a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[NeurIPS 2025\] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](../../NeurIPS2025/reinforcement_learning/a_unifying_view_of_linear_function_approximation_in_offpolic.md)

</div>

<!-- RELATED:END -->
