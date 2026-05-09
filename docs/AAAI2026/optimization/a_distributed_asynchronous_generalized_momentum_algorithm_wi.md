---
title: >-
  [Paper Note] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds
description: >-
  [AAAI 2026][Optimization][Distributed Optimization] This paper proposes a totally asynchronous Generalized Momentum (GM) distributed optimization algorithm that guarantees linear convergence without assuming any upper bound on communication or computation delays. On a Fashion-MNIST classification task, the proposed method requires 71% fewer iterations than gradient descent, 41% fewer than Heavy Ball, and 19% fewer than Nesterov accelerated gradient.
tags:
  - AAAI 2026
  - Optimization
  - Distributed Optimization
  - Total Asynchrony
  - Generalized Momentum
  - No Delay Bound
  - Linear Convergence
date: 2026-05-08
content_hash: 44d5c3a5769d41f8
---

# A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds

**Conference**: AAAI 2026
**arXiv**: [2508.08218](https://arxiv.org/abs/2508.08218)
**Code**: [GitHub](https://github.com/aaai26tagm-sim/TAGM)
**Area**: Optimization / Distributed Computing
**Keywords**: Distributed Optimization, Total Asynchrony, Generalized Momentum, No Delay Bound, Linear Convergence

## TL;DR
This paper proposes a totally asynchronous Generalized Momentum (GM) distributed optimization algorithm that guarantees linear convergence without assuming any upper bound on communication or computation delays. On a Fashion-MNIST classification task, the proposed method requires 71% fewer iterations than gradient descent, 41% fewer than Heavy Ball, and 19% fewer than Nesterov accelerated gradient.

## Background & Motivation
Large-scale machine learning problems typically require parallel/distributed optimization to accelerate training. Following the classical taxonomy of Bertsekas & Tsitsiklis (1989), parallel algorithms fall into three categories:

1. **Synchronous algorithms**: All processors must compute and communicate at every round. These suffer from the straggler penalty — the slowest processor bottlenecks the entire system.
2. **Partially asynchronous algorithms**: Allow bounded delays, but require a known delay upper bound, which is often unavailable or nonexistent in practice.
3. **Totally asynchronous algorithms**: Allow arbitrarily long delays and are the most flexible, but existing methods (primarily based on gradient descent) converge slowly.

The central limitation of existing totally asynchronous algorithms is: **either they require a delay upper bound to guarantee convergence (unrealistic), or they forgo this requirement at the cost of slow convergence (restricted to GD)**. Momentum methods (Heavy Ball, Nesterov accelerated gradient) substantially outperform GD in centralized settings, yet lack theoretical guarantees in totally asynchronous distributed environments.

## Core Problem
How to design a distributed optimization algorithm that simultaneously satisfies the following three properties:
1. **Total asynchrony**: No assumption or knowledge of any delay upper bound is required (computation delays, transmission delays, and reception delays among processors can all be arbitrarily long).
2. **Momentum acceleration**: Leverages momentum techniques to achieve faster convergence than GD.
3. **Theoretical guarantees**: Admits rigorous convergence proofs and convergence rate analysis.

The fundamental challenge lies in the inherent tension between momentum, which introduces dependence on historical iterates, and total asynchrony, which allows each processor's "historical information" to be arbitrarily stale.

## Method

### Overall Architecture
Consider $n$ processors communicating over a graph $G=(V,E)$, collaboratively minimizing $f(x) = \sum_{i=1}^{n} f_i(x_{V_i})$, where $x$ is partitioned across processors and each $f_i$ depends only on the variables of processor $i$ and its neighbors.

The approach proceeds in three steps:
1. Starting from the centralized GM update rule parameterized by $\gamma$ (step size), $\lambda$ (momentum for gradient evaluation point), and $\beta$ (momentum for iterate update).
2. Showing that a single GM step does not yield an $\infty$-norm contraction mapping, but that **two consecutive steps** do — thereby constructing a "two-step synchronous GM" (dGM) algorithm.
3. Applying the framework of Bertsekas & Tsitsiklis to derive totally asynchronous convergence guarantees from the contraction property of the synchronous version.

### Key Designs

1. **Generalized Momentum Update Law**:
   The centralized form is:
   $$x(l+1) = \Pi_X\left[x(l) - \gamma \nabla f\big(x(l) + \lambda(x(l)-x(l-1))\big) + \beta(x(l)-x(l-1))\right]$$
   where $\gamma$ is the step size, $\lambda$ controls the momentum shift in the gradient evaluation point (analogous to Nesterov's look-ahead), and $\beta$ controls the iterate momentum term (analogous to Heavy Ball). Setting $\beta=\lambda$ recovers NAG, $\lambda=0$ recovers HB, and $\beta=\lambda=0$ recovers GD. This unified framework enables search over a richer parameter space for optimal configurations.

2. **Two-Step Contraction Mapping**:
   This is the paper's central technical contribution. A single GM update does not satisfy the $\infty$-norm contraction property due to the momentum term. The authors show that **applying two consecutive GM updates** constitutes an $\infty$-norm contraction with coefficient $\alpha = \max\{\alpha_1, \alpha_2\} \in (0,1)$, where:
   - $\alpha_1 = |1+\beta - \gamma\mu(1+\lambda)|^2 + (\beta - \gamma\lambda\mu)|2+\beta - \gamma\mu(1+\lambda)|$
   - $\alpha_2 = 1 - \gamma\mu + 2(\beta - \lambda\gamma\mu)$

   This enables the definition of the double-step synchronous GM (dGM) algorithm, where each time step consists of two GM updates. Since each dGM step is a contraction mapping, the totally asynchronous convergence framework of Bertsekas & Tsitsiklis applies directly.

3. **Totally Asynchronous Protocol (Algorithm 1)**:
   Each processor $i$ maintains a local copy $z^i = (x^i, y^i)$ of all relevant variables. Under the totally asynchronous protocol:
   - Processor $i$ executes the dGM update at times $k \in K_i$ (its computation time set), and remains idle otherwise.
   - Communication uses stale information: the copy of processor $j$'s variables held by processor $i$ may correspond to a much earlier version $\tau_j^i(k)$.
   - The only assumption (Assumption 4) is that no processor permanently halts computation or communication — but the interval between successive operations can be arbitrarily long.

   The paper introduces the notion of an **operation cycle**: $\text{ops}(k)$ counts the number of complete operation cycles completed by time $k$, where one cycle requires every processor to complete at least one computation, send, and receive. Convergence rate is expressed as a linear function of $\text{ops}(k)$.

### Loss & Training
The objective function is required to satisfy three assumptions:
1. **Decomposability of the constraint set $X$**: $X = X_1 \times \cdots \times X_n$, a bounded convex compact set enabling parallelizable projections.
2. **Twice continuous differentiability**: Ensures the Hessian exists and is continuous.
3. **$\mu$-diagonal dominance** (most critical): $H_{ii}(x) \geq \mu + \sum_{j \neq i} |H_{ij}(x)|$. Bertsekas & Tsitsiklis have shown that the absence of diagonal dominance leads to divergence in totally asynchronous algorithms. Intuitively, this requires each variable's self-influence on the gradient to dominate the sum of all cross-variable coupling effects.

Algorithm parameters must satisfy the constraint set $C_1 \cup C_2$: $C_1$ corresponds to $\beta \leq \lambda$ (including NAG), and $C_2$ to $\lambda \leq \beta$ (including HB).

## Key Experimental Results
Experiments use 16 processors (13th Gen Intel Core i7-13700, 2.10 GHz) to train an $\ell_2$-regularized multiclass logistic regression classifier on Fashion-MNIST (70K samples, 10 classes) with regularization parameter $\theta=0.01$. All algorithms are run until 85% test accuracy is reached.

| Probability $p$ | GD Iters | HB Iters | NAG Iters | GM Iters | vs GD | vs HB | vs NAG |
|-----------------|----------|----------|-----------|----------|-------|-------|--------|
| 1.0 | 666 | 309 | 218 | **168** | 74.8% | 45.6% | 22.9% |
| 0.5 | 1351 | 651 | 475 | **377** | 72.1% | 42.1% | 20.6% |
| 0.1 | 8486 | 4154 | 3018 | **2424** | 71.4% | 41.6% | 19.7% |
| 0.05 | 24309 | 11539 | 8425 | **6722** | 72.3% | 41.7% | 20.2% |

Parameter settings: GD ($\gamma=0.1$), HB ($\gamma=0.1, \beta=0.075$), NAG ($\gamma=0.1, \beta=\lambda=0.35$), GM ($\gamma=0.1, \beta=0.5, \lambda=0.05$).

### Ablation Study
- **Optimal GM parameter combination**: GM uses $\beta=0.5$ (large iterate momentum) but $\lambda=0.05$ (small look-ahead), in sharp contrast to NAG's $\beta=\lambda=0.35$, demonstrating that decoupling and independently tuning these two parameters yields significant gains in asynchronous settings.
- **Stable advantage across delay regimes**: As $p$ decreases from 1.0 (synchronous) to 0.05 (extreme delays), the iteration reduction of GM relative to all baselines remains stable (vs GD: 71–75%, vs HB: 41–46%, vs NAG: 19–23%), indicating that GM's acceleration is robust to the magnitude of delays.
- **Operation complexity analysis (Theorem 4)**: For a given accuracy $\epsilon$, each processor requires $\rho = \frac{\log(D/\epsilon)}{\log(1/\alpha)}$ computations and $\rho|V_i|$ communications, where $\alpha$ is the contraction constant. A smaller $\alpha$ implies fewer required operations.

## Highlights & Insights
- **Unified framework**: GD, HB, and NAG are unified as special cases of GM. This is the first work to achieve momentum acceleration under total asynchrony, filling a theoretical gap.
- **Elegance of the two-step contraction**: If a single step is not a contraction mapping, prove that two steps are. This insight is concise yet non-trivial, reducing an apparently intractable problem to a tractable one.
- **Abstraction via operation cycles**: Using operation cycles rather than wall-clock time as the convergence measure elegantly sidesteps the incomparability of time across processors in asynchronous environments.
- **Expanded parameter space**: By independently tuning $\beta$ and $\lambda$ (rather than coupling them as in NAG), the algorithm identifies better operating points within the parameter feasibility set under asynchronous constraints.

## Limitations & Future Work
- **Strong assumptions (strong convexity + diagonal dominance)**: The $\mu$-diagonal dominance requirement on the Hessian precludes strong inter-variable coupling, ruling out many practical ML problems (deep learning loss functions almost never satisfy this condition). The authors note, however, that this is a necessary condition for totally asynchronous convergence.
- **Experiments limited to convex problems**: Validation is restricted to $\ell_2$-regularized logistic regression; non-convex optimization (e.g., neural network training) is not explored. Extending the framework to non-convex settings is an important future direction.
- **Manual hyperparameter tuning**: $\gamma, \lambda, \beta$ are selected by manual search within the $C_1 \cup C_2$ constraint set; no adaptive parameter selection mechanism is provided.
- **Single dataset**: Only Fashion-MNIST is used, lacking validation on larger-scale or more diverse problems.
- **Verification of diagonal dominance**: How to efficiently verify or enforce Hessian diagonal dominance in practice is not adequately discussed.

## Related Work & Insights
1. **vs Totally asynchronous GD (Bertsekas & Tsitsiklis 1989; Wu et al. 2023)**: GM strictly generalizes GD (the $\beta=\lambda=0$ case), achieving at least 71% fewer iterations through momentum. Theoretically, GD is a special case within the $C_2$ feasibility set.
2. **vs Totally asynchronous HB (Hustig-Schultz et al. 2023)**: HB corresponds to GM with $\lambda=0$, incorporating only iterate momentum without look-ahead at the gradient evaluation point. GM achieves an additional 41% speedup by introducing nonzero $\lambda$.
3. **vs Totally asynchronous NAG (Pond et al. 2024)**: NAG is the special case $\beta=\lambda$, coupling the two momentum parameters. By decoupling them, GM identifies a better parameter configuration in asynchronous settings, yielding a further 19% reduction in iterations.
4. **vs Partially asynchronous methods (Assran et al. 2020; Zhou et al. 2018, etc.)**: These methods require a known delay upper bound, limiting their applicability. GM converges without any such assumption.

The paper's core technical paradigm — "a single step is not a contraction, but $k$ steps are" — is broadly inspiring and may apply to other algorithm designs requiring asynchronous convergence guarantees. The operation cycle abstraction is also transferable to settings such as federated learning. As noted in the Conclusion, a key future direction is **online adaptive parameter tuning**, dynamically adjusting $\gamma, \lambda, \beta$ based on runtime information to further accelerate convergence.

## Rating
- Novelty: ⭐⭐⭐⭐ — The unified framework and two-step contraction technique are genuinely novel, though the core mathematical machinery follows the Bertsekas & Tsitsiklis framework.
- Experimental Thoroughness: ⭐⭐⭐ — Only one dataset and one model; parameter sweeps are comprehensive but the experimental scope is narrow.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-organized, with tight logical flow from problem to method to theory to experiments; theorems are stated rigorously.
- Value: ⭐⭐⭐⭐ — Fills a theoretical gap in totally asynchronous momentum optimization, though practical applicability is limited by the strong convexity and diagonal dominance assumptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[CVPR 2026\] Dynamic Momentum Recalibration in Online Gradient Learning](../../CVPR2026/optimization/dynamic_momentum_recalibration_in_online_gradient_learning.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](../../ICLR2026/optimization/mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[NeurIPS 2025\] perturbation bounds for low-rank inverse approximations under noise](../../NeurIPS2025/optimization/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)

</div>

<!-- RELATED:END -->
