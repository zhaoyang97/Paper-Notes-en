---
title: >-
  [Paper Note] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions
description: >-
  [AAAI2026][Optimization][Global optimization] This paper proposes ECPv2, which introduces three innovations—adaptive lower bound, Worst-$m$ memory…
tags:
  - "AAAI2026"
  - "Optimization"
  - "Global optimization"
  - "Lipschitz functions"
  - "black-box optimization"
  - "random projection"
  - "no-regret"
date: 2026-05-08
content_hash: e8df895a802be242
---

# ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions

**Conference**: AAAI2026
**arXiv**: [2511.16575](https://arxiv.org/abs/2511.16575)  
**Authors**: Fares Fourati (KAUST), Mohamed-Slim Alouini (KAUST), Vaneet Aggarwal (Purdue)
**Code**: [GitHub](https://github.com/fouratifares/ECP)  
**Area**: Optimization
**Keywords**: Global optimization, Lipschitz functions, black-box optimization, random projection, no-regret

## TL;DR

This paper proposes ECPv2, which introduces three innovations—adaptive lower bound, Worst-$m$ memory, and fixed random projection—to reduce the per-run complexity of Lipschitz global optimization from $\Omega(n^2 d)$ to $\Omega(n(m+d)\log n)$, while maintaining an $O(n^{-1/d})$ regret convergence rate that matches the minimax lower bound.

## Background & Motivation

### Core Challenge of Global Optimization
Global optimization is a classical problem in the optimization community: the objective function may be non-convex and non-smooth, and is accessible only through black-box evaluations. In settings such as robotic control, hyperparameter tuning for machine learning, and black-box LLM query optimization, each function evaluation is costly, necessitating approximation of the global optimum within a limited budget.

### Strengths and Limitations of ECP
The predecessor work ECP (Every Call is Precious) proposed a conservative acceptance criterion: a candidate point is evaluated only if it could potentially be the global maximum under the Lipschitz assumption, ensuring every function call is informative. ECP achieves a no-regret guarantee in theory and outperforms Bayesian optimization, CMA-ES, and other methods on various benchmarks. However, ECP suffers from two bottlenecks:

1. **Computational bottleneck**: The acceptance condition requires computing distances to all historical points, yielding a total complexity of $\Omega(n^2 d)$.
2. **Excessive conservatism**: When the parameter $\varepsilon_t$ is small, the acceptance region may be empty, causing a large number of futile rejections.

## Method

### Innovation 1: Adaptive Lower Bound $\varepsilon_t^{\oslash}$

ECPv2 introduces a theoretically derived adaptive lower bound to prevent empty acceptance regions:

$$\varepsilon_t^{\oslash} = \frac{\max_i f(x_i) - \min_j f(x_j)}{\text{diam}(\mathcal{X})}$$

**Lemma 1** proves that if any point $x$ satisfies the ECP acceptance condition, then $\varepsilon_t$ must be no smaller than this lower bound. In practice, $\varepsilon_t$ is updated at each step as $\varepsilon_t = \max(\tau_{n,d} \cdot \varepsilon_{t-1}, \varepsilon_t^{\oslash})$, with an additional maintenance cost of only $O(1)$ via incremental tracking of the maximum and minimum values.

**Lemma 2** further proves that the acceptance region obtained with the lower bound is a superset of the original ECP acceptance region, i.e., $\mathcal{A}_{\text{ECP}}(\varepsilon_t, t) \subseteq \mathcal{A}_t(\varepsilon_t, t)$.

### Innovation 2: Worst-$m$ Memory Mechanism

Instead of checking the acceptance condition against all $t$ historical points, ECPv2 compares only against the $m$ points with the lowest function values. The index set of the worst $m$ points is defined as:

$$\mathcal{I}_t^m = \arg\min_{S \subseteq \{1,\ldots,t\}, |S|=m} \sum_{i \in S} f(x_i)$$

The modified acceptance condition becomes:

$$\min_{i \in \mathcal{I}_t^m} \left(f(x_i) + \max\{\varepsilon_t, \varepsilon_t^{\oslash}\} \cdot \|x - x_i\|_2\right) \geq \max_{j \in [t]} f(x_j)$$

Intuitively, points with the largest gap impose the strongest constraints on the acceptance condition, while excluding high-value points relaxes the exploration space. Setting $m = n$ recovers the original ECP.

### Innovation 3: Fixed Random Projection for Acceleration

Leveraging the Johnson–Lindenstrauss lemma, distance computations are projected from $\mathbb{R}^d$ to $\mathbb{R}^{d'}$, where $d' = 8\log(\beta n) / (\delta^2 - \delta^3)$. The projection matrix $\mathbf{P}$ is drawn i.i.d. from $\mathcal{N}(0,1)$, guaranteeing that all pairwise distances are distorted by at most $\delta$ with probability at least $1 - 1/\beta^2$:

$$(1-\delta)\|x_i - x_j\|_2^2 \leq \|\mathbf{P}x_i - \mathbf{P}x_j\|_2^2 \leq (1+\delta)\|x_i - x_j\|_2^2$$

After projection, the parameter in the acceptance condition is adjusted to $\tilde{\varepsilon}_t = \max\{\varepsilon_t, \varepsilon_t^{\oslash}\} / \sqrt{1-\delta}$ to compensate for the lower-bound distortion of distances.

### Theoretical Guarantees

**Corollary 1**: The combination of the three innovations guarantees $\mathcal{A}_{\text{ECP}}(\varepsilon_t, t) \subseteq \mathcal{A}_{\text{ECPv2}}(\varepsilon_t, t, m, \mathbf{P})$ with high probability.

**Theorem 2 (Finite-time regret bound)**: For any $f \in \text{Lip}(k)$, with probability at least $1 - 1/\beta^2 - \xi$,

$$\mathcal{R}_{\text{ECPv2},f}(n) \leq k \cdot \text{diam}(\mathcal{X}) \cdot \log_{\tau_{n,d}}\!\left(\frac{k}{\varepsilon_1}\right)^{1/d} \cdot \left(\frac{\ln(1/\xi)}{n}\right)^{1/d}$$

This bound matches the minimax lower bound $\Omega(k \cdot n^{-1/d})$, preserving the optimal convergence rate.

## Key Experimental Results

### Complexity Comparison

| Method | Memory | Runtime | Regret Upper Bound |
|--------|--------|---------|-------------------|
| AdaLIPO | $O(nd)$ | $\Omega(n^2 d)$ | $O(n^{-1/d})$ |
| ECP | $O(nd)$ | $\Omega(n^2 d)$ | $O(n^{-1/d})$ |
| **ECPv2** | $O((m+d)\log n)$ | $\Omega(n(m+d)\log n)$ | $O(n^{-1/d})$ |

### High-Dimensional Benchmark Experiments
- Rosenbrock function with $d \in \{3, 100, 200, 300, 500\}$: ECPv2 ($\beta=5, \delta=2/3, m=8$) matches or surpasses ECP after $n=200$ evaluations with significantly reduced wall-clock time.
- Rosenbrock ($d=500$) and Powell ($d=1000$): ECPv2 converges approximately twice as fast as ECP with higher optimization scores.
- Worst-$m$ ablation: Acceptance test cost is reduced by 4–5× for $m \in \{8, 16, 32, 64, 128\}$; small values of $m$ sometimes outperform the full method.
- Adaptive lower bound in isolation: Runtime is approximately 2× faster under low evaluation budgets.

### Hyperparameter Selection
- $\beta = 5$: Guarantees a 96% success rate for distance preservation.
- $\delta = 2/3$: Analytically derived value that minimizes the optimal projection dimensionality.

## Highlights & Insights

- **Substantial efficiency gains**: Runtime reduced from $O(n^2 d)$ to $O(n(m+d)\log n)$ and memory from $O(nd)$ to $O((m+d)\log n)$, with significant implications for high-dimensional, large-budget settings.
- **Complete theoretical guarantees**: The regret bound matches the minimax lower bound, the acceptance region strictly contains that of ECP, and each of the three innovations has independent theoretical support.
- **Modular design**: The three innovations can be used independently or in combination, offering flexibility across different settings.
- **Principled hyperparameters**: $\beta=5$ and $\delta=2/3$ are analytically derived rather than heuristically tuned.

## Limitations & Future Work

- **Restricted to Lipschitz continuity**: No theoretical guarantees are provided for non-Lipschitz functions.
- **Curse of dimensionality in $n^{-1/d}$ regret**: This is an inherent limitation of Lipschitz optimization, not an algorithmic issue.
- **Selection of $m$ in Worst-$m$**: While experiments suggest that small $m$ suffices, no theoretical guidance exists for adaptive selection of $m$.
- **Probabilistic failure from random projection**: The method fails with probability $1/\beta^2$; although this can be controlled by increasing $\beta$, doing so increases the projection dimensionality.

## Related Work & Insights

- **AdaLIPO / AdaLIPO+**: Relies on uniform random exploration to estimate the Lipschitz constant, leading to low sample efficiency; ECPv2 avoids wasted evaluations through its conservative acceptance criterion.
- **DIRECT**: Deterministic space partitioning incurs high computational cost in high dimensions and tends to be overly conservative; ECPv2's random sampling with projection acceleration is better suited for high-dimensional settings.
- **Bayesian Optimization (BO)**: Depends on Gaussian process model assumptions and is sensitive to hyperparameters; ECPv2 requires only the Lipschitz assumption and is thus more general.
- **CMA-ES / Dual Annealing**: Lack no-regret guarantees and require extensive evaluations and hyperparameter tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ — Each of the three innovations is theoretically grounded, though the core idea constitutes an improvement over ECP rather than an entirely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multi-dimensional benchmarks with detailed ablations, but lacks validation on real-world application scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rigorous theoretical derivations, and a complete definition–lemma–theorem framework.
- Value: ⭐⭐⭐⭐ — Significantly advances the scalability of Lipschitz optimization and addresses the practical deployment bottlenecks of ECP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](../../ICML2026/optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](../../ICML2026/optimization/learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[CVPR 2026\] BlazeFL: Fast and Deterministic Federated Learning Simulation](../../CVPR2026/optimization/blazefl_fast_and_deterministic_federated_learning_simulation.md)

</div>

<!-- RELATED:END -->
