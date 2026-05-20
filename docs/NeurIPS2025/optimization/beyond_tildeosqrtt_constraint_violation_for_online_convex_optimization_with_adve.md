---
title: >-
  [Paper Note] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints
description: >-
  [NeurIPS 2025][Optimization][Online convex optimization] This paper studies Constrained Online Convex Optimization (COCO) with adversarial constraints and introduces a tunable parameter $\beta$ to achieve a precise trade…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Online convex optimization"
  - "adversarial constraints"
  - "constraint violation"
  - "regret-violation tradeoff"
  - "safety constraints"
date: 2026-05-08
content_hash: bc728fcf4df67c8c
---

# Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints

**Conference**: NeurIPS 2025
**arXiv**: [2505.06709](https://arxiv.org/abs/2505.06709)  
**Code**: None  
**Area**: Online Optimization / Constrained Optimization
**Keywords**: Online convex optimization, adversarial constraints, constraint violation, regret-violation tradeoff, safety constraints

## TL;DR

This paper studies Constrained Online Convex Optimization (COCO) with adversarial constraints and introduces a tunable parameter $\beta$ to achieve a precise tradeoff between $\tilde{O}(T^\beta)$ regret and $\tilde{O}(T^{1-\beta})$ cumulative constraint violation (CCV), surpassing the previously known optimal bound of $\tilde{O}(\sqrt{T})$ constraint violation.

## Background & Motivation

**Problem Setting**: In Constrained Online Convex Optimization (COCO) with adversarial constraints, at each round:
1. The learner selects an action from a convex decision set.
2. An adversary reveals a convex cost function and a convex constraint function.
3. The learner's goal is to minimize cumulative regret and cumulative constraint violation (CCV).

**Prior Results**: The best-known algorithms achieve $O(\sqrt{T})$ regret and $\tilde{O}(\sqrt{T})$ CCV.

**Core Motivation**: In safety-critical applications such as autonomous driving and medical decision-making, satisfying safety constraints is non-negotiable. Trading regret for reduced constraint violation is therefore of significant practical importance. The existing $\tilde{O}(\sqrt{T})$ CCV bound implies that constraints may still be substantially violated even after many rounds, which is unacceptable in safety-sensitive scenarios.

**Contributions**: The paper proposes a tunable parameter $\beta \in [0, 1]$ that enables a smooth tradeoff between regret and CCV:
- $\beta = 1/2$: recovers the known optimal $O(\sqrt{T})$ regret and $\tilde{O}(\sqrt{T})$ CCV.
- $\beta \to 1$: CCV approaches a constant, but regret grows nearly linearly.
- In practice, choosing $\beta > 1/2$ can substantially reduce constraint violation while keeping regret manageable.

## Method

### Overall Architecture

The paper adopts a hierarchical algorithmic design. It first addresses a special **Constrained Expert** problem, then reduces the general COCO problem to this special case via a covering argument.

### Key Designs

**1. Constrained Expert Problem**:
- The decision set is the probability simplex $\Delta_N$ (distributions over $N$ experts).
- Both cost and constraint functions are linear.
- This is an important special case of COCO and serves as the foundation for the general solution.

**Algorithm Core**: The approach leverages a newly proposed **adaptive small-loss regret bound** to design a computationally efficient algorithm. The key insight is that when constraint function values are small (i.e., near feasibility), the algorithm can more aggressively optimize cost; when constraint violation is large, the algorithm shifts focus toward reducing violation.

**Regret Bound**: $O(\sqrt{T \ln N} + T^\beta)$ regret and $\tilde{O}(T^{1-\beta} \ln N)$ CCV.

**2. Reduction from Constrained Expert to General COCO**:
- An $\epsilon$-net cover is constructed over the convex decision set, yielding $N$ "experts" (discretized actions).
- The constrained expert algorithm is then applied over these experts.
- The choice of coverage precision $\epsilon$ governs the tradeoff between discretization error and the number of experts.
- **Final result**: $\tilde{O}(\sqrt{dT} + T^\beta)$ regret and $\tilde{O}(dT^{1-\beta})$ CCV, where $d$ is the dimension of the decision set.

**3. First-Order Methods for the Smooth Case**:
- An additional $M$-smoothness assumption is imposed (Lipschitz-continuous gradients for cost and constraint functions).
- A computationally efficient first-order algorithm is designed.
- Achieves $O(\sqrt{MT} + T^\beta)$ regret and $\tilde{O}(MT^{1-\beta})$ CCV.

### Loss & Training

- **Adaptive Learning Rate**: The algorithm's learning rate is dynamically adjusted based on the cumulative degree of constraint violation.
- **Small-Loss Regret Bound**: The regret bound scales with the learner's actual cumulative loss rather than the worst-case $T$, automatically yielding tighter bounds when constraint violation is small.
- **Selection Strategy for $\beta$**: In practice, $\beta$ can be tuned according to the stringency of safety requirements — a higher safety demand corresponds to a larger $\beta$.

## Key Experimental Results

### Main Results: Theoretical Bound Comparison for Regret–CCV Tradeoff

| Method | Regret Upper Bound | CCV Upper Bound | Parameter | Computational Efficiency |
|---|---|---|---|---|
| Prior SOTA | $O(\sqrt{T})$ | $\tilde{O}(\sqrt{T})$ | None | Polynomial time |
| **Ours (general convex)** | $\tilde{O}(\sqrt{dT} + T^\beta)$ | $\tilde{O}(dT^{1-\beta})$ | $\beta \in [0,1]$ | Via covering argument |
| **Ours (expert setting)** | $O(\sqrt{T\ln N} + T^\beta)$ | $\tilde{O}(T^{1-\beta}\ln N)$ | $\beta \in [0,1]$ | Computationally efficient |
| **Ours ($M$-smooth)** | $O(\sqrt{MT} + T^\beta)$ | $\tilde{O}(MT^{1-\beta})$ | $\beta \in [0,1]$ | First-order method |

### Ablation Study: Performance Under Different Values of $\beta$

| $\beta$ | Regret Order | CCV Order | Regret–CCV Product | Applicable Scenario |
|---|---|---|---|---|
| 0 | $\tilde{O}(\sqrt{dT})$ | $\tilde{O}(dT)$ | $\tilde{O}(d^{3/2}T^{3/2})$ | Pure regret minimization |
| 1/4 | $\tilde{O}(\sqrt{dT} + T^{1/4})$ | $\tilde{O}(dT^{3/4})$ | — | Moderate safety requirements |
| 1/2 | $\tilde{O}(\sqrt{dT} + \sqrt{T})$ | $\tilde{O}(d\sqrt{T})$ | $\tilde{O}(d^{3/2}T)$ | Classical setting |
| 3/4 | $\tilde{O}(\sqrt{dT} + T^{3/4})$ | $\tilde{O}(dT^{1/4})$ | — | High safety requirements |
| 1 | $\tilde{O}(\sqrt{dT} + T)$ | $\tilde{O}(d)$ | $\tilde{O}(d^{3/2}T)$ | Strict constraint satisfaction |

### Key Findings

1. **Pareto-Optimal Tradeoff**: Regret and CCV cannot be simultaneously improved; $\beta$ provides a mechanism for smooth movement along the Pareto frontier.
2. **Dimension Dependence**: The bounds for the general convex case depend on the decision set dimension $d$, rather than a larger measure of function class complexity.
3. **Critical Role of Adaptive Small-Loss Bounds**: This technical innovation enables precise control of constraint violation without sacrificing regret.
4. **Computational Efficiency**: Both the constrained expert algorithm and the first-order smooth method run in polynomial time.

## Highlights & Insights

1. **Theoretical Foundation for Safety-Critical Applications**: This work provides the first formal characterization of the tunable tradeoff between regret and constraint violation in online learning.
2. **Elegant Hierarchical Reduction**: Solving the simpler expert problem first and then extending to general convex settings via covering arguments offers a methodologically instructive design pattern.
3. **Novel Technique: Adaptive Small-Loss Bounds**: This technique goes beyond standard online learning tools and may find broad applicability in other online optimization problems.
4. **Practical Guidance for $\beta$**: The parameter $\beta$ offers engineers a clear and actionable safety–performance knob when deploying online learning systems.

## Limitations & Future Work

1. **Absence of Lower Bounds**: It remains unclear whether the proposed $\tilde{O}(\sqrt{dT} + T^\beta)$ regret and $\tilde{O}(dT^{1-\beta})$ CCV bounds are Pareto-optimal.
2. **Realism of the Adversarial Assumption**: Fully adversarial constraints may be overly conservative; practical constraints may exhibit certain statistical regularity.
3. **Convexity Assumption**: Extensions to non-convex cost or constraint functions have not been explored.
4. **Efficiency of the Covering Argument**: The $\epsilon$-net reduction may lead to computational explosion in high-dimensional decision sets.
5. **Insufficient Numerical Experiments**: The paper provides no empirical simulations to verify the tightness of the theoretical bounds.

## Related Work & Insights

- **Online Convex Optimization**: The classical framework of Hazan (2016) provides the theoretical foundation for this work.
- **Constrained Online Learning**: Mahdavi et al. (2012) and Yu et al. (2020) established foundational results for COCO.
- **Safe Online Learning**: The $\beta$-tuning mechanism offers a new perspective for the theoretical analysis of safe reinforcement learning.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 4 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 2 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)

</div>

<!-- RELATED:END -->
