---
title: >-
  [Paper Note] Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion
description: >-
  [ICML 2026][Autonomous Driving][max-min fairness] This paper unifies "max-min multi-objective fairness" and "hard constraint satisfaction" into a single MORL framework. By reformulating the problem as a convex program us…
tags:
  - "ICML 2026"
  - "Autonomous Driving"
  - "max-min fairness"
  - "constrained MORL"
  - "occupancy measure"
  - "dual convex optimization"
  - "projected gradient descent"
date: 2026-05-08
content_hash: cc644065c3a68eb9
---

# Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion

**Conference**: ICML 2026  
**arXiv**: [2605.31388](https://arxiv.org/abs/2605.31388)  
**Code**: https://github.com/Giseung-Park/Constrained-Maxmin-MORL  
**Area**: Reinforcement Learning / Multi-Objective RL / Constrained Optimization  
**Keywords**: max-min fairness, constrained MORL, occupancy measure, dual convex optimization, projected gradient descent

## TL;DR
This paper unifies "max-min multi-objective fairness" and "hard constraint satisfaction" into a single MORL framework. By reformulating the problem as a convex program using occupancy measures and deriving a dual convex optimization problem over weights $(u,w)$, the authors implement a projected gradient descent algorithm that simultaneously achieves fairness and constraint feasibility with theoretical guarantees of geometric convergence rates.

## Background & Motivation

**Background**: The mainstream approach in Multi-Objective Reinforcement Learning (MORL) is to use a scalarization function $f(J_1(\pi),\ldots,J_K(\pi))$ to combine multiple returns into a single scalar for single-objective RL. When $f$ is linear, it becomes a weighted sum $\sum_k w_k J_k(\pi)$, which is simple but fails to reflect "fairness." For instance, in traffic signal control, a scheduler may prioritize "minimizing the maximum waiting time" (max-min fairness, $f=\min$) rather than minimizing the total average wait time across all directions.

**Limitations of Prior Work**: Existing max-min MORL methods assume an unconstrained environment. However, real-world systems often involve hard constraints: a scheduler must maximize throughput fairness within a power budget, or a traffic light must minimize the longest wait time within a greenhouse gas emission limit. Integrating constraints into max-min MORL is not a trivial task—current max-min MORL algorithms either optimize imprecise lower bounds (Fan et al. 2023; Peng et al. 2025) or rely on biased gradients through Gaussian smoothing (Park et al. 2024). Conversely, established constrained RL methods (CPO, RCPO, Lagrangian) focus on $K=1$ scalar rewards and are ill-equipped to handle the non-differentiability introduced by $f=\min$.

**Key Challenge**: The max-min objective $\min_k J_k(\pi)$ is non-convex and non-differentiable in the policy space, causing standard Lagrangian dual analysis to fail. To incorporate constraints, one must identify a mathematical structure capable of simultaneously handling "non-differentiable max-min" and "inequality constraints."

**Goal**: To establish a unified MORL framework capable of optimizing $\min_k J_k(\pi)$ while satisfying $J_{K+l}(\pi)\ge C^{(l)}$, accompanied by a provably convergent algorithm.

**Key Insight**: The authors observe that while max-min is non-convex in the policy space, it is **convex in the occupancy measure $\rho(s,a)$ space**, a classic result from Puterman 1994. By rewriting policy optimization as a convex program regarding $\rho$, constraints naturally become linear inequalities and the max-min objective becomes a min-of-linear form, transforming the entire problem into a standard convex program with a Lagrangian dual.

**Core Idea**: Convexification of the primal problem via occupancy measures → derivation of its entropy-regularized dual → formulation of a convex optimization problem involving only "constraint multipliers $u$ + objective weights $w$" → use of projected gradient descent to learn both sets of weights, achieving max-min fairness and constraint satisfaction in a joint iteration.

## Method

### Overall Architecture

The input is a MOMDP $\langle\mathcal{S},\mathcal{A},T,\mu_0,r,\gamma\rangle$, where the first $K$ dimensions of the reward $r:\mathcal{S}\times\mathcal{A}\to\mathbb{R}^{K+L}$ are objectives for "max-min fairness," and the subsequent $L$ dimensions are constraints requiring $J\ge C$. The algorithm outputs a softmax policy $\pi(\cdot|s)=\mathrm{softmax}\{Q(s,\cdot)/\beta\}$ that maximizes the minimum objective return while satisfying all constraints.

The workflow follows a **bilevel iteration**:

- **Inner Loop**: With fixed weights $(u,w)$, the Q-function is trained to $Q^*_{u,w}$ via soft Bellman iterations (corresponding to an entropy-regularized "scalarized RL subproblem" with reward $\sum_l u_l c^{(l)}+\sum_k w_k r^{(k)}$).
- **Outer Loop**: Based on the $\pi^*_{u,w}$ obtained from the inner loop, $\nabla_{(u,w)}\mathcal{L}(u,w)$ is estimated. A projected gradient descent step is performed on $(u,w)$ to project them onto the constraint multiplier space $\mathbb{R}_+^L$ and the objective weight simplex $\Delta^K$.

Mathematically, the primal problem is transformed into a convex program through occupancy measures. The authors further prove that the dual can be written as:

$$\min_{u\in\mathbb{R}_+^L,\,w\in\Delta^K} \mathcal{L}(u,w) = \sum_s \mu_0(s)\,v^*_{u,w}(s) - \sum_{l=1}^L u_l C^{(l)}$$

where $v^*_{u,w}$ is the fixed point of the entropy-regularized Bellman operator $\mathcal{T}_{u,w}$.

### Key Designs

1. **Occupancy Measure Convexification + Dual Convexification (Double Convex Structure)**:
    - **Function**: Converts non-convex and non-differentiable max-min MORL in the policy space into a finite-dimensional convex optimization over $(u,w)$.
    - **Mechanism**: The policy $\pi$ is replaced by the occupancy measure $\rho(s,a)$, making the primal max-min problem a convex program—the objective is $\max_\rho \min_k \sum_{s,a} r^{(k)}(s,a)\rho(s,a)$ with Bellman flow and linear return constraints. Solving its dual yields a convex loss $\mathcal{L}(u,w)$ over multipliers $u\in\mathbb{R}_+^L$ and weights $w\in\Delta^K$. The non-differentiability of $f=\min$ disappears in the dual, as max-min is equivalent to maximizing a linear function over the simplex $\Delta^K$.
    - **Design Motivation**: To avoid dealing with subgradients of $\min$ directly on policy parameters. While Lagrangian duality fails in the original max-min form due to non-differentiability, the double convexification via occupancy measures allows max-min and inequality constraints to be unified into a single convex loss.

2. **Unified Gradient Formula (Theorem 3.3)**:
    - **Function**: Uses the same entropy-regularized policy $\pi^*_{u,w}$ to simultaneously calculate $\nabla_u \mathcal{L}$ and $\nabla_w \mathcal{L}$.
    - **Mechanism**: The authors prove $\nabla_u v^*_{u,w}(s) = v_c^{\pi^*_{u,w}}(s)$ and $\nabla_w v^*_{u,w}(s) = v_r^{\pi^*_{u,w}}(s)$, where $v_c$ and $v_r$ are value functions evaluating constrained rewards $\{c^{(l)}\}$ and objective rewards $\{r^{(k)}\}$, respectively.
    - **Design Motivation**: This compresses "constraint updates" and "max-min weight updates" into a single value function evaluation. The gradient for $w$ automatically amplifies "bottleneck" objectives (dimensions with small returns), while the gradient for $u$ pushes multipliers of violated constraints higher to enforce feasibility.

3. **Entropy Regularization + Projected Gradient Descent (with Smoothness Guarantee)**:
    - **Function**: Ensures solvability and numerical stability while providing a geometric convergence rate.
    - **Mechanism**: Adding an entropy term $\beta \sum_s \mathcal{H}_\rho(s)\rho(s)$ introduces only an $O(\beta\log|\mathcal{A}|/(1-\gamma))$ approximation error. Entropy regularization keeps $\pi^*_{u,w}(a|s)>0$ strictly positive, making the Hessian $H[\mathcal{L}]$ positive definite under Slater's condition. This renders the dual $\alpha$-smooth and strongly convex.
    - **Design Motivation**: Entropy regularization acts as both a "theoretical crutch" (guaranteeing positivity and smoothness) and an "algorithmic crutch" (enabling soft Bellman Q-updates). PGD is more stable than standard Lagrangian methods as $w$ remains on the simplex without specialized normalization tricks.

### Loss & Training Strategy

The outer objective is the convex loss $\mathcal{L}(u,w) = \sum_s \mu_0(s) v^*_{u,w}(s) - \sum_l u_l C^{(l)}$. The inner loop uses entropy-regularized soft Bellman updates: $Q(s,a) \leftarrow [u;w]^\top [c;r] + \gamma \sum_{s'} T(s'|s,a)\beta\log\sum_{a'}\exp(Q(s',a')/\beta)$. In continuous state spaces, a gradient network $g_\theta(s)\in\mathbb{R}^{L+K}$ parameterizes the estimate of $\nabla v^*_{u,w}(s)$, paired with SAC-style Q-networks. The hyperparameter $\beta$ was tuned within $\{0.1, 0.03, 0.01, 0.003, 0.001\}$, with $\beta=0.03$ yielding the lowest error.

## Key Experimental Results

### Main Results

Evaluations were conducted on tabular settings and three real-world environments. For the tabular setup, an LP was used to find the ground truth.

| Setting | Algorithm | Metric | Value | Comparison |
|------|------|------|------|------|
| Tabular MOMDP | **Constrained max-min (ours)** | Optimality Error ↓ | **0.004** | unconstr. max-min: 0.325 / constr. max-avg: 0.657 / unconstr. max-avg: 1.008 |
| Building (HVAC, $C_{th}=180$) | **Ours** | Energy / Min Comfort ↑ | **178.7 / 639.8** | MA-SAC-L: 171.4 / 620.9 (feat/poor fairness); Max-min GS: 202.1 / 653.6 (vioat); ARAM: 276.9 / 664.3 (severe violat) |
| MoAnt-v5 ($C_{th}=50$) | **Ours** | Control Cost / Min Return ↑ | **28.3 / 92.2** | MA-SAC-L: 47.8 / 83.0; MA-SAC: 275.3 / 98.8 (violat); ARAM: 620.7 / 101.3 (severe violat) |
| Traffic 16-lane ($C_{th}=70{,}000$) | **Ours** | CO₂ / Min Lane Return ↑ | **69,147 / −25,229** | MA-CPGO: 67,887 / −27,830; Max-min GS: 73,162 / −21,527 (violat); ARAM: 88,748 / −19,700 (severe violat) |

Notably, only the proposed method and the "max-average + Lagrangian" baseline strictly satisfied constraints. Among feasible methods, ours achieved higher "worst-case returns," validating the effectiveness of max-min fairness.

### Ablation Study

| Configuration | Energy ($C_{th}=180$) | Worst-case Return | Description |
|------|-------------------|------------|------|
| Full model | 178.7 | 639.8 | Complete algorithm (joint $(u,w)$ updates) |
| w/o $w$ update | 178.1 | 626.7 | No max-min weight learning → worst return drops by 13 |
| w/o $u$ update | 200.7 | 653.5 | No multiplier learning → constraint violation (>180) |
| w/o $(u,w)$ update | 222.0 | 646.9 | No weight learning → violates and unfair |

### Key Findings

- **Complementary $u$ and $w$**: Ablations show that removing $w$ updates hurts fairness, while removing $u$ updates leads to constraint violations. Simultaneous updates are essential.
- **Sensitivity of $\beta$**: $\beta=0.1$ introduces high approximation error, while $\beta<0.01$ decreases the convergence coefficient $\lambda/\alpha$. The sweet spot is $\beta\in[0.01, 0.03]$.
- **Failure of existing max-min methods**: Max-min GS and ARAM fail to satisfy constraints in all real-world scenarios, suggesting that constraints cannot be simply tacked onto max-min MORL.
- **MA-SAC-L Limitation**: This method optimizes for max-average while satisfying constraints, failing to address fairness between objectives, resulting in lower "worst-case" performance.

## Highlights & Insights

- **Occupancy measure convexification is an underutilized tool**: Non-convex objectives in the policy space often become convex in the occupancy measure space. This avoids the necessity of subgradient methods for $\min$.
- **Elegant Unified Gradient (Theorem 3.3)**: It is surprising that gradients for both multipliers $u$ and weights $w$ are simply value functions of different reward channels under the same policy.
- **The Triple Role of Entropy Regularization**: (1) Smoothing the hard-min; (2) ensuring the dual Hessian is positive definite; (3) enabling a closed-form soft Bellman update.
- **Transferable Design**: The joint PGD framework for objective weights and constraint multipliers can be applied to other fair-constrained tasks like fair federated learning or fair-RLHF.

## Limitations & Future Work

- **Tabular Theory vs. Neural Practice**: Geometric convergence via Theorem 3.6 relies on assumptions that may not strictly hold during neural network approximation.
- **Entropy Parameter $\beta$**: $\beta$ is a hyperparameter coupled with the convergence rate. A potential future direction includes developing an "adaptive $\beta$" mechanism.
- **Scaling to Multiple Constraints ($L \ge 2$)**: Real-world experiments focused on $L=1$. Framework support exists, but performance under complex multiple constraints is unexplored.
- **High-dimensional Objectives**: Traffic $(K=16)$ was the scale limit in the study; stability for very large $K$ (e.g., $K=100$) requires further validation.

## Related Work & Insights

- **vs. Park et al. 2024 (Gaussian Smoothing)**: Their method requires multiple network copies and results in biased gradients. This work provides **exact** gradients with a single network and supports constraints.
- **vs. Byeon et al. 2025 (ARAM)**: ARAM uses game-theoretic ideas but lacks constraint support. Experiments show ARAM severely violates constraints in practical settings.
- **vs. CPO / Lagrangian RL**: These assume scalar rewards ($K=1$). This work unifies objective weights and multipliers into a single dual framework, offering a cleaner structural solution for MORL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](../../CVPR2026/autonomous_driving/remot_reinforcement_learning_with_motion_contrast_triplets.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](../../CVPR2026/autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
