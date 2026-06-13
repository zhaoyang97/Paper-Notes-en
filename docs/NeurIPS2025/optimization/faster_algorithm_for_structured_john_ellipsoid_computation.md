---
title: >-
  [Paper Note] Faster Algorithms for Structured John Ellipsoid Computation
description: >-
  [NeurIPS 2025][Optimization][John ellipsoid] For computing the John ellipsoid of a symmetric convex polytope $P = \{x \in \mathbb{R}^d : -\mathbf{1}_n \leq Ax \leq \mathbf{1}_n\}$…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "John ellipsoid"
  - "convex optimization"
  - "input sparsity"
  - "treewidth"
  - "non-negative matrix factorization"
  - "Lewis weights"
  - "leverage scores"
date: 2026-05-08
content_hash: c8e0f8b53b919250
---

# Faster Algorithms for Structured John Ellipsoid Computation

**Conference**: NeurIPS 2025
**arXiv**: [2211.14407](https://arxiv.org/abs/2211.14407)  
**Code**: None  
**Area**: Convex Optimization / Algorithm Design
**Keywords**: John ellipsoid, convex optimization, input sparsity, treewidth, non-negative matrix factorization, Lewis weights, leverage scores

## TL;DR
For computing the John ellipsoid of a symmetric convex polytope $P = \{x \in \mathbb{R}^d : -\mathbf{1}_n \leq Ax \leq \mathbf{1}_n\}$, this paper proposes two fast algorithms: a near-input-sparsity algorithm based on sketching with per-iteration cost $\widetilde{O}(\text{nnz}(A) + d^\omega)$, and a treewidth-based algorithm with per-iteration cost $O(n\tau^2)$, both significantly improving upon the prior state-of-the-art cost of $O(nd^2)$.

## Background & Motivation

**John Ellipsoid**: Fritz John's theorem guarantees the existence and uniqueness of a maximum-volume inscribed ellipsoid (the John ellipsoid) for any convex body. This is a fundamental problem in convex optimization.

**Applications**: High-dimensional sampling, linear programming, online learning, differential privacy, uncertainty quantification, and D-optimal experimental design.

**Prior State-of-the-Art**: Cohen, Cousins, Lee, and Yang (COLT 2019) proposed a fixed-point iteration based on $\ell_\infty$ Lewis weights, with per-iteration cost $O(nd^2)$ and $O(\varepsilon^{-1}\log(n/d))$ iterations.

**Computational Bottleneck**: Each iteration requires computing quadratic forms $a^\top B^{-1} a$ where $B$ is a weighted version of $A^\top A$. The standard approach via Cholesky decomposition costs $O(nd^2)$.

**Goal**: Exploit matrix structure (sparsity or small treewidth) to break the $O(nd^2)$ barrier.

## Method

### Problem Formulation

Given $A \in \mathbb{R}^{n \times d}$ of rank $d$, finding the maximum-volume inscribed ellipsoid is equivalent to solving:

$$\min_{w \geq 0} \sum_{i=1}^n w_i - \log\det\left(\sum_{i=1}^n w_i a_i a_i^\top\right) - d$$

Optimality conditions: $\sum_i w_i = d$, and $a_j^\top Q^{-1} a_j = 1$ whenever $w_j \neq 0$, where $Q = \sum_i w_i a_i a_i^\top$.

### Fixed-Point Iteration (Base Framework)

Each iteration updates the weights as:

$$w_{k+1,i} = w_{k,i} \cdot a_i^\top (A^\top \text{diag}(w_k) A)^{-1} a_i$$

This is equivalent to computing the leverage scores of $B_k = \sqrt{\text{diag}(w_k)} A$: $w_{k+1,i} = \|(B_k^\top B_k)^{-1/2} \sqrt{w_{k,i}} a_i\|_2^2$.

### Algorithm 1: Near-Input-Sparsity Algorithm (Theorem 1.1)

**Core Idea**: Two-level acceleration via leverage score sampling and random matrix sketching.

1. **Leverage Score Sampling**: Construct a sampling matrix $D_k$ using approximate leverage scores $\tilde{\sigma}$ such that:
   $$(1-\varepsilon_0) B_k^\top B_k \preceq B_k^\top D_k B_k \preceq (1+\varepsilon_0) B_k^\top B_k$$
   Only $s = \Theta(\varepsilon_0^{-2} d \log(d/\delta))$ sampled rows are needed.

2. **Random Sketching**: Apply a Gaussian matrix $S_k \in \mathbb{R}^{s \times d}$ for dimensionality reduction:
   $$\hat{w}_{k+1,i} = \frac{1}{s}\|\widetilde{Q}_k \sqrt{w_{k,i}} a_i\|_2^2, \quad \widetilde{Q}_k = S_k (B_k^\top D_k B_k)^{-1/2}$$

3. **Approximate Leverage Score Computation**: Computed in $\widetilde{O}(\varepsilon_\sigma^{-2}(\text{nnz}(A) + d^\omega))$ time using the method of Drineas et al.

**Per-Iteration Cost**: $\widetilde{O}(\varepsilon^{-1}\text{nnz}(A) + \varepsilon^{-2} d^\omega)$

**Total Complexity**: $\widetilde{O}((\varepsilon^{-1}\text{nnz}(A) + \varepsilon^{-2} d^\omega) \cdot \varepsilon^{-1}\log(n/d))$

### Algorithm 2: Small-Treewidth Algorithm (Theorem 1.2)

**Core Idea**: Exploit the treewidth $\tau$ of the incidence graph of $A$ to accelerate Cholesky decomposition.

1. **Treewidth**: Construct the incidence graph $G_A$ of $A$, where vertices correspond to columns and edges connect pairs of columns that are both nonzero in some row. Treewidth $\tau$ measures how "tree-like" the graph is.

2. **Fast Cholesky Decomposition**: When the treewidth is $\tau$, the factorization $A^\top W A = LL^\top$ can be computed in $O(n\tau^2)$ time, with each row of $L$ having at most $\tau$ nonzero entries.

3. **Sparse $L$ Solves**: By exploiting the sparsity of $L$, computing $b_{k,i}^\top (L_k L_k^\top)^{-1} b_{k,i}$ requires only $O(\tau^2)$ operations.

**Per-Iteration Cost**: $O(n\tau^2)$

**Practical Relevance**: Treewidth is typically $O(d^{1/2} \sim d^{3/4})$ on the Netlib LP benchmark; the largest MATPOWER power systems instance has $n=20467$, $d=12659$, yet treewidth only $\tau=35$.

### Error Analysis: Telescope Lemma

The final approximation quality is decomposed into an initialization term and an accumulated per-step error:

$$\phi_i(u) \leq \frac{1}{T}\log\frac{n}{d} + \varepsilon/250 + \varepsilon_0$$

Choosing $T = O(\varepsilon^{-1}\log(n/d))$ ensures both terms are $O(\varepsilon)$.

### Output Ellipsoid Guarantee

For a $(1+\varepsilon)$-approximate John ellipsoid $Q$:

$$\frac{1}{\sqrt{1+\varepsilon}} \cdot Q \subseteq P \subseteq \sqrt{d} \cdot Q$$

Volume guarantee: $\text{vol}(\frac{1}{\sqrt{1+\varepsilon}} Q) \geq \exp(-d\varepsilon/2) \cdot \text{vol}(Q^*)$.

## Key Experimental Results

### Complexity Comparison

| Algorithm | Iterations | Per-Iteration Cost |
|-----------|------------|--------------------|
| CCLY19 | $\varepsilon^{-1}\log(n/d)$ | $nd^2$ |
| Alg. 1 (sketching) | $\varepsilon^{-1}\log(n/d)$ | $\varepsilon^{-1}\text{nnz}(A) + \varepsilon^{-2}d^\omega$ |
| Alg. 2 (treewidth) | $\varepsilon^{-1}\log(n/d)$ | $n\tau^2$ |

### Speedup Analysis
- When $A$ is sparse ($\text{nnz}(A) \ll nd$): Algorithm 1 offers substantial improvement.
- When $A$ is dense but $n > d^\omega$: Algorithm 1 still outperforms CCLY19.
- When $\tau \ll d$ (e.g., $\tau = O(d^{1/2})$): Algorithm 2 reduces cost from $nd^2$ to $nd$, a $d$-fold speedup.
- MATPOWER case: with $\tau = 35$, Algorithm 2 reduces cost from $O(2 \times 10^{11})$ to $O(2.5 \times 10^7)$, approximately a $10^4\times$ speedup.

## Highlights & Insights
- **Two Complementary Acceleration Paths**: Sparsity is exploited via sketching and sampling; structural properties are exploited via treewidth, covering distinct practical scenarios.
- **Near-Input-Sparsity Time**: Since reading the input already requires $O(\text{nnz}(A))$, Algorithm 1 is nearly optimal.
- **Telescope Lemma Innovation**: Unlike CCLY19, which analyzes only sketching error, this paper simultaneously handles error accumulation from both sketching and sampling.
- **Solid Theoretical Contributions**: High-probability guarantees with precise approximation factor analysis.

## Limitations & Future Work
- The work is purely theoretical with no numerical experiments to validate practical efficiency.
- The $d^\omega$ term in approximate leverage score computation may remain a bottleneck for large $d$.
- The treewidth algorithm requires computing a tree decomposition first (NP-hard to solve exactly; only $O(\tau \log^3 n)$ approximations are available).
- Only centrally symmetric convex polytopes are considered; asymmetric cases are not covered.
- Algorithm 1 relies on random matrices and carries a failure probability $\delta$.
- Parallelization potential is not discussed.

## Related Work & Insights
- **vs. CCLY19**: Both use the same fixed-point iteration framework; this paper reduces per-iteration cost from $O(nd^2)$ to $\widetilde{O}(\text{nnz}(A) + d^\omega)$ and $O(n\tau^2)$, respectively.
- **vs. Interior-Point Methods**: Traditional $O(nd^3)$ cost is prohibitive; both proposed algorithms are faster in their respective applicable regimes.
- **vs. First-Order Methods**: First-order methods require many iterations but have cheap per-step cost; the proposed methods require fewer iterations ($O(\log n)$) with high per-step efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ Both acceleration paths represent genuinely new technical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical; no numerical experiments.
- Writing Quality: ⭐⭐⭐⭐ Proof structure is clear; technical overviews aid comprehension.
- Value: ⭐⭐⭐⭐ Substantial improvement to a foundational convex optimization algorithm with broad application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](../../ICLR2026/optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)
- [\[NeurIPS 2025\] FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)

</div>

<!-- RELATED:END -->
