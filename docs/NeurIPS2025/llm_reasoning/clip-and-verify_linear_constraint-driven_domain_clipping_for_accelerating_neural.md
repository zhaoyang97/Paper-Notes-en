---
title: >-
  [Paper Note] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerated Neural Network Verification
description: >-
  [NeurIPS 2025][LLM Reasoning][Neural network verification] This paper proposes the Clip-and-Verify verification pipeline, which leverages linear constraints generated "for free" during linear bound propagation. Two GPU-e…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Neural network verification"
  - "branch-and-bound"
  - "linear constraints"
  - "domain clipping"
  - "α"
  - "β-CROWN"
date: 2026-05-08
content_hash: 2871cb1e06c701e7
---

# Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerated Neural Network Verification

**Conference**: NeurIPS 2025
**arXiv**: [2512.11087](https://arxiv.org/abs/2512.11087)
**Code**: [https://github.com/Verified-Intelligence/Clip_and_Verify](https://github.com/Verified-Intelligence/Clip_and_Verify)
**Area**: LLM Reasoning
**Keywords**: Neural network verification, branch-and-bound, linear constraints, domain clipping, α,β-CROWN

## TL;DR

This paper proposes the Clip-and-Verify verification pipeline, which leverages linear constraints generated "for free" during linear bound propagation. Two GPU-efficient algorithms—complete clipping (coordinate ascent dual solving) and relaxed clipping (closed-form input domain shrinkage)—are used to tighten intermediate-layer bounds across the entire network. The approach reduces the number of BaB subproblems by up to 96% on multiple benchmarks, and serves as a core component of the VNN-COMP 2025 winning verifier.

## Background & Motivation

**Background**: Neural network verification aims to formally prove that a network satisfies safety properties (e.g., robustness) over a specified input domain. The most effective current approach combines Branch-and-Bound (BaB) with linear bound propagation (e.g., CROWN/α,β-CROWN)—which recursively relaxes nonlinear activation functions to compute upper and lower bounds for each neuron. A property is verified when the final-layer lower bound exceeds zero; otherwise, the search continues by splitting subproblems.

**Limitations of Prior Work**: The number of BaB subproblems grows exponentially, and loose intermediate-layer bounds are the root cause of the inability to verify subdomains early. Existing approaches face the following dilemma: (1) β-CROWN theoretically supports optimizing intermediate-layer bounds, but in practice the overhead of layer-wise optimization is prohibitive, as the number of hidden-layer neurons exceeds that of the output layer by orders of magnitude; (2) LP solvers (e.g., Gurobi) yield tight bounds but are too slow and cannot be parallelized on GPUs; (3) BaB introduces linear constraints at each split (activation splits/input splits) that encode information for input domain reduction, yet existing frameworks do not exploit them sufficiently.

**Key Challenge**: Tighter intermediate-layer bounds lead to fewer BaB subproblems and faster verification, but the computational cost of directly optimizing intermediate-layer bounds is disproportionate to the benefit. An efficient mechanism is needed to tighten intermediate-layer bounds opportunistically at minimal overhead.

**Goal**: How can intermediate-layer bounds across the entire network be opportunistically tightened during BaB at minimal computational cost, without relying on expensive external solvers?

**Key Insight**: The bounding planes produced at each layer during linear bound propagation are themselves linear constraints on the input; together with constraints introduced by BaB splits, they can be used to shrink the input domain or directly improve intermediate-layer bounds. The key insight is to reformulate the constrained bound optimization problem as a one-dimensional dual problem, enabling GPU-parallel exact solving via piecewise-linear structure and breakpoint sorting.

**Core Idea**: Linear constraints obtained "for free" during BaB are converted into input domain reduction and intermediate-layer bound tightening. Efficient GPU-parallel clipping is achieved through closed-form or coordinate ascent solutions to the dual problem.

## Method

### Overall Architecture

Clip-and-Verify is a verification pipeline embedded within the standard BaB loop. At each BaB node: (1) linear constraints are collected (from final-layer bounding planes or activation splits); (2) relaxed clipping shrinks the input domain $\mathcal{X} \rightarrow \mathcal{X}'$, after which intermediate-layer bounds are re-concretized over the smaller input domain; (3) complete clipping directly optimizes bounds for critical unstable neurons. The two clipping strategies are complementary—relaxed clipping is inexpensive and globally effective, while complete clipping is precise but locally applied.

### Key Designs

1. **Complete Clipping — Coordinate Ascent Dual Solving**:

    - **Function**: Directly and precisely optimizes the bound of each neuron under linear constraints.
    - **Mechanism**: Given the minimization of a linear function $\mathbf{a}^\top \mathbf{x} + c$ over a box domain $\mathcal{X}$ subject to a linear constraint $\mathbf{g}^\top \mathbf{x} + h \leq 0$, a Lagrangian dual problem is constructed. The key theorem (Theorem 3.1) states that the dual objective $D(\beta) = (\mathbf{a} + \beta\mathbf{g})^\top \hat{\mathbf{x}} - \sum_j |(\mathbf{a} + \beta\mathbf{g})_j| \epsilon_j + c + \beta h$ is a concave piecewise-linear function of $\beta \in \mathbb{R}_+$, and the optimal solution can be found exactly by enumerating breakpoints $\beta_j = -a_j/g_j$. For multiple constraints, coordinate ascent is applied: each $\beta$ is optimized in turn with others fixed, and each step reduces to a single-constraint problem. This is equivalent to the greedy algorithm for the continuous knapsack problem, with complexity $\mathcal{O}(n \log n)$.
    - **Design Motivation**: The high-dimensional LP is converted into a one-dimensional concave maximization problem, and its piecewise-linear structure avoids iterative optimization while naturally supporting GPU parallelism. Compared to Gurobi's LP solver, this approach is 880× faster (0.0028s vs. 2.47s) with comparable bound accuracy (error 0.00085 vs. 0.0007).

2. **Relaxed Clipping — Closed-Form Input Domain Shrinkage**:

    - **Function**: Shrinks the input domain at minimal cost to indirectly tighten intermediate-layer bounds across the entire network.
    - **Mechanism**: For a single linear constraint $\mathbf{a}^\top \mathbf{x} + c \leq 0$ and a box domain $\mathcal{X}$, Theorem 3.2 provides a closed-form update for each coordinate: $x_i^{(clip)} = (-\sum_{j \neq i} \{a_j \hat{x}_j - |a_j|\epsilon_j\} - c) / a_i$, updating the upper or lower bound depending on the sign of $a_i$. The clipped box is the tightest axis-aligned hyperrectangle containing the feasible set. For multiple constraints, each constraint is applied independently in parallel, and the tightest bounds across all results are retained. The computation requires only $O(n)$ arithmetic operations. After obtaining the tighter input domain, all intermediate-layer bounds are updated via re-concretization (rather than full re-propagation).
    - **Design Motivation**: While complete clipping is exact, it requires per-neuron solving, which is costly in large networks. Relaxed clipping updates only the shared input domain once, allowing all intermediate-layer bounds to benefit automatically. For example, in a 3-layer network with 4096+2048+100 neurons, full re-propagation takes 10s per subproblem, whereas re-concretization takes only 0.3s.

3. **Integration with BaB — Constraint Sources and Pipeline Design**:

    - **Function**: Seamlessly acquires constraints and applies clipping in both input BaB and activation BaB modes.
    - **Mechanism**: (a) *Input BaB*: The final-layer bounding plane $\underline{\mathbf{a}}^{(L)\top}\mathbf{x} + \underline{c}^{(L)} \leq 0$ naturally separates verified and unverified regions. After each split, child subproblems inherit parent constraints, enabling progressively more precise clipping. The BaB workflow is modified so that relaxed clipping occurs after splitting rather than before. (b) *Activation BaB*: Activation splits $z_j^{(i)} \geq s$ or $z_j^{(i)} \leq s$ are relaxed via bound propagation into linear constraints in the input space and can be cached and reused. The top-$k$ unstable neurons with the highest BaBSR scores are selected for complete clipping. (c) *Infeasibility detection*: If clipping produces $\underline{x} > \overline{x}$ in any dimension, the current constraint combination is infeasible, and the subproblem is immediately marked as verified.
    - **Design Motivation**: Constraints arise naturally during BaB at no additional cost. In input BaB, child subproblems inherit parent constraints so that clipping effects accumulate progressively; in activation BaB, large numbers of split constraints are collected and exploited.

## Key Experimental Results

### Main Results

Input BaB benchmark comparison:

| Method | acasxu subproblems | Time (s) | #verified | nn4sys subproblems | #verified |
|--------|-------------------|----------|-----------|-------------------|-----------|
| α,β-CROWN | 7,154,387 (baseline) | 280.51 | 138 | 4,440,252 (baseline) | 194 |
| +Relaxed clipping | 3,124,100 ↓56.3% | 151.37 | 139 | 2,691,750 ↓39.4% | 194 |
| +Relaxed+Reordering | 2,557,715 ↓64.2% | 150.25 | 139 | 2,300,894 ↓48.2% | 194 |
| +Complete clipping | 1,533,068 ↓**78.6%** | 168.57 | **139** | 2,141,288 ↓51.8% | 194 |

Activation BaB benchmark comparison (number verified):

| Dataset | β-CROWN | BICCOS | C&V+β-CROWN | C&V+BICCOS | Upper bound |
|---------|---------|--------|------------|------------|-------------|
| oval22 | 20 | 25 | 22 | **27** | 29 |
| cifar10-resnet | 60 | 63 | 63 | **64** | 72 |
| cifar100-2024 | 119 | 121 | 126 | **131** | 168 |
| tinyimagenet-2024 | 135 | 138 | 140 | **144** | 157 |

### Ablation Study

Control system verification (necessity of complete clipping):

| Task | α,β-CROWN time | C&V (relaxed) time | C&V (complete) time | Subproblem reduction |
|------|---------------|-------------------|--------------------|--------------------|
| cartpole | 1602s | 484s | **142s** | ↓95.5% |
| Quadrotor-2D | timeout | 209,504s | **78,818s** | ↓57.7% |
| Quad-2D-Large | timeout | timeout | **104,614s** | Only method to complete |

### Key Findings

- Complete clipping achieves the most significant subproblem reduction (up to 96%), but due to per-neuron solving overhead, it does not always yield the shortest total runtime—relaxed clipping is sometimes faster overall.
- Complete clipping is the only method capable of verifying the most challenging control system properties.
- Compared to Gurobi LP: the coordinate ascent GPU solver is 880× faster with comparable bound accuracy.
- Relaxed and complete clipping are complementary: relaxed clipping is inexpensive and globally effective and should always be enabled; complete clipping provides precise tightening for selected critical neurons.
- Clip-and-Verify is orthogonal to and composable with BICCOS; C&V+BICCOS achieves the best results on nearly all benchmarks.

## Highlights & Insights

- **Mathematical elegance of the dual reformulation**: The $n$-dimensional LP is converted into a 1D piecewise-linear concave maximization problem. Its breakpoint structure enables exact solving that is naturally GPU-parallel, nearly three orders of magnitude faster than general-purpose LP solvers. The essential technique is the dual of the continuous knapsack problem.
- **Engineering insight of the "free lunch"**: Linear bound propagation already computes bounding planes at each layer. Clip-and-Verify cleverly reuses this existing information as linear constraints for clipping, introducing virtually no additional computation.
- **Infeasibility detection as a byproduct**: The clipping process naturally identifies infeasible subdomains (contradictory activation assignments), which can be immediately marked as verified, further reducing the BaB search space.
- **Practical impact**: As a core component of α,β-CROWN, the VNN-COMP 2025 winning verifier, the method has already been integrated into a deployed system.

## Limitations & Future Work

- Complete clipping remains too expensive to apply to all unstable neurons in large networks, necessitating heuristic selection of the top-$k$ neurons. The BaBSR heuristic is effective but is not the optimal selection strategy.
- Coordinate ascent does not guarantee global optimality for multi-constraint problems; improvement at each step is assured only by the concavity of the dual objective. Convergence properties and the choice of iteration count require further analysis.
- The effectiveness of relaxed clipping is limited by the quality of the box-domain approximation—when the feasible set is far from box-shaped, its impact is limited.
- The approach has only been validated on feedforward networks and Vision Transformers; applicability to more complex architectures (e.g., recurrent networks, GNNs) remains unexplored.
- The memory overhead of caching and reusing constraints may be non-negligible in deep networks.

## Related Work & Insights

- **vs. GCP-CROWN (MIP cuts)**: GCP-CROWN strengthens relaxations using MIP cutting planes, but requires an external MILP solver and incurs high overhead. Clip-and-Verify uses pure GPU coordinate ascent, which is faster and more scalable, achieving comparable or higher verification counts on most benchmarks.
- **vs. BICCOS**: BICCOS strengthens relaxations through inter-neuron constraints (quadratic cross-constraints). Clip-and-Verify is orthogonal to BICCOS and can be combined with it—C&V+BICCOS achieves the best verification accuracy on nearly all benchmarks.
- **vs. β-CROWN intermediate-layer optimization**: β-CROWN theoretically supports tightening intermediate-layer bounds by optimizing β parameters, but in practice this is abandoned due to the excessive number of neurons. Clip-and-Verify provides a computationally feasible alternative for intermediate-layer bound tightening.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual reformulation and the insight of reusing existing constraints are elegant, though the core mathematical tool (the dual of the continuous knapsack) is not a new invention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple years of VNN-COMP benchmarks, control system verification, comparisons with diverse tools, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous; the roles and complementarity of the two algorithms are clearly articulated.
- **Value**: ⭐⭐⭐⭐⭐ As a core component of the VNN-COMP 2025 winning system, its practical impact has already been demonstrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion](two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[AAAI 2026\] Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs](../../AAAI2026/llm_reasoning/graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)
- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](../../ACL2026/llm_reasoning/towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)

</div>

<!-- RELATED:END -->
