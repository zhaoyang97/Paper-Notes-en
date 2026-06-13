---
title: >-
  [Paper Note] Center-Outward q-Dominance: A Sample-Computable Proxy for Strong Stochastic Dominance in Multi-Objective Optimisation
description: >-
  [AAAI 2026][Stochastic multi-objective optimisation] Building on the center-outward distribution function from optimal transport theory, this paper proposes the q-dominance relation as a computable approximation of stron…
tags:
  - "AAAI 2026"
  - "Stochastic multi-objective optimisation"
  - "stochastic dominance"
  - "optimal transport"
  - "hyperparameter tuning"
  - "center-outward quantiles"
date: 2026-05-08
content_hash: 3d13da56d91fe894
---

# Center-Outward q-Dominance: A Sample-Computable Proxy for Strong Stochastic Dominance in Multi-Objective Optimisation

**Conference**: AAAI 2026
**arXiv**: [2511.12545](https://arxiv.org/abs/2511.12545)  
**Code**: None  
**Area**: Other
**Keywords**: Stochastic multi-objective optimisation, stochastic dominance, optimal transport, hyperparameter tuning, center-outward quantiles

## TL;DR

Building on the center-outward distribution function from optimal transport theory, this paper proposes the q-dominance relation as a computable approximation of strong first-order stochastic dominance (strong FSD). It proves that q-dominance over the full quantile range implies strong FSD, derives explicit sample-size thresholds for Type I error control, and validates practical utility in hyperparameter tuning ranking and noisy multi-objective optimisation.

## Background & Motivation

### Problem Definition

**Stochastic Multi-Objective Optimisation Problem (SMOOP)**: When objective functions involve randomness (e.g., repeated runs of an algorithm in hyperparameter optimisation), each candidate solution corresponds to the distribution of a multivariate random vector. Comparing solutions requires **ranking distributions** rather than deterministic points.

### Limitations of Prior Work

**Scalarisation**: Converts multi-objective problems into single-objective ones, discarding distributional structure and depending on predefined scalarisation functions.

**Moment surrogates** (mean/variance/CVaR): Replace stochastic objectives with summary statistics, ignoring dependence structure.

**Weak FSD**: Implemented via CDF comparisons, but overly permissive in dimension $d>1$ — it may assert $\mathbf{X} \succeq_w \mathbf{Y}$ even when $\mathbf{X}$ is inferior to $\mathbf{Y}$ under certain monotone utility functions.

**Rioux et al. (2025)** OT approach: Computes one optimal coupling per distribution pair, lacks a globally consistent reference frame for multi-distribution comparisons, and incurs computational cost that grows quadratically in the number of distribution pairs.

### Gap Between Weak and Strong FSD

The paper presents a clear counterexample: in two dimensions, $\mathbf{X}=\{(0,1),(1,0)\}$ weakly dominates $\mathbf{Y}=\{(0,0),(1,1)\}$ in the CDF sense, but does not strongly dominate it — under utility $u(x_1,x_2)=\exp(x_1+x_2)$, $\mathbb{E}[u(\mathbf{X})] < \mathbb{E}[u(\mathbf{Y})]$.

### Core Problem

A **computable, globally consistent** approximation of strong FSD is needed, with the following properties: no hyperparameters, sample-size guarantees, and a natural hierarchy of relaxations.

## Method

### Overall Architecture

**Core Idea**: Each distribution is mapped to the uniform distribution $U_d$ on the unit ball via its center-outward distribution function, establishing a **unified reference frame**. Sample points sharing the same center-outward rank and sign are naturally paired, enabling quantile-by-quantile comparison.

### Key Designs

#### 1. **Center-Outward Distribution/Quantile Functions**

Based on McCann's theorem and the theory of Hallin (2017, 2021):

- **Center-outward quantile function** $\mathbf{Q}^\pm$: The optimal quadratic transport map from $U_d$ to target distribution $P$, satisfying $\mathbf{Q}^\pm \# U_d = P$.
- **Center-outward distribution function** $\mathbf{F}^\pm$: Its inverse, mapping $P$ to $U_d$, satisfying $\mathbf{F}^\pm \# P = U_d$.

Key properties: $\|\mathbf{F}^\pm(\mathbf{Z})\|$ follows a uniform distribution on $[0,1]$ (rank); $\mathbf{F}^\pm(\mathbf{Z})/\|\mathbf{F}^\pm(\mathbf{Z})\|$ is uniformly distributed on the unit sphere (sign); and the two are independent.

**Empirical estimation**: Constructed via an augmented grid on the unit ball ($n_R$ concentric hyperspheres $\times$ $n_S$ unit vectors), with optimal assignment solved using the Hungarian algorithm ($O(n^3)$).

#### 2. **Definition of q-Dominance**

For $q \in [0,1)$, $P_1 \succeq_q P_2$ if and only if for all $\mathbf{y} \in \mathbb{C}_{P_2}(q)$ (the $q$-quantile region of $P_2$):

$$\mathbf{x} := \mathbf{Q}_1^\pm(\mathbf{F}_2^\pm(\mathbf{y})) \geq \mathbf{y}$$

**Intuition**: $\mathbf{y}$ is mapped to the unit ball via $P_2$'s distribution function, then mapped back to data space via $P_1$'s quantile function; the resulting point $\mathbf{x}$ is componentwise no less than $\mathbf{y}$.

**Relationship to strong FSD** (Theorem 5): If $P_1 \succeq_q P_2$ holds for all $q \in [0,1)$, then $P_1$ strongly first-order stochastically dominates $P_2$.

#### 3. **Finite-Sample Testing and Sample-Size Thresholds**

Theorem 6 establishes: under bi-Lipschitz conditions, for any confidence level $0<\delta<1$, there exists an explicit threshold $n^*(\delta)$ such that when the sample size $n \geq n^*(\delta)$, empirical q-dominance correctly reflects theoretical q-dominance with probability at least $1-\delta$.

The expression for $n^*(\delta)$, while complex (involving dimension $d$, Lipschitz constants, the minimum gap $\Delta$ of quantile functions, etc.), is **fully explicit** — no Monte Carlo or bootstrap estimation is required.

### Loss & Training

This paper does not involve neural network training. The core algorithmic pipeline is:

1. Compute the empirical center-outward quantile map for each distribution (computed once).
2. Perform pairwise q-dominance tests on the shared augmented grid.
3. Construct non-dominated fronts based on q-dominance using a sorting algorithm (Algorithm 1).

## Key Experimental Results

### Main Results: Multi-Objective Hyperparameter Optimisation Ranking

Seven HPO methods (Random Search, ParEGO, SMS-EGO, EHVI, MEGO, MIES, etc.) are compared on the YAHPO-MO benchmark:

| Method | Avg. Rank (q-dominance) | Avg. Rank (HVI) | Key Difference |
|--------|------------------------|-----------------|----------------|
| MIES | **Best (~2.0)** | Moderate | Best under q-dominance |
| ParEGO | Moderate (~3.1) | **Best** | Best under HVI |
| Random Search | Worst | Worst | Consistent across both |

| Analysis Dimension | q-dominance Ranking | HVI Ranking |
|--------------------|--------------------|--------------------|
| Method discriminability | More compact rankings | Larger spread |
| Methods not significantly better than RS | Only EHVI (1 method) | EHVI + SMS-EGO + MIES (3 methods) |
| Information retention | Retains full distributional information | Uses expected values only |

### Ablation Study: Noise-Augmented ZDT Benchmark

| ZDT Problem | NSGA-II + q-dom | NSGA-II mean | NSGA-II single | Notes |
|-------------|----------------|-------------|----------------|-------|
| ZDT1 | **Fastest convergence** | Slow | Slowest | Significant advantage of q-dominance |
| ZDT2 | **Fastest convergence** | Slow | Slowest | Same as above |
| ZDT3 | **Fastest convergence** | Slow | Slowest | Same as above |
| ZDT4 | All three similar | Similar | Similar | ZDT4 sensitive to noise |
| ZDT6 | **Fastest convergence** | Slow | Slowest | Significant advantage of q-dominance |

Experimental setup: $\sigma=0.1$, population size 20, 200 generations, 64 samples per solution ($n_R=n_S=8$).

### Key Findings

1. **MIES vs. ParEGO case study**: On lcbench 167152, MIES yields better outcomes with high probability but carries slightly higher risk; HVI (expected value) favours ParEGO due to its lower variance. q-dominance correctly identifies MIES's distributional advantage.
2. Under q-dominance ranking, meaningful rankings can still be produced when HVI fails to distinguish methods (i.e., when expected values are close).
3. Replacing mean-based selection with q-dominance selection in NSGA-II significantly improves convergence on 5 out of 6 ZDT problems.

## Highlights & Insights

1. **Globally consistent reference frame**: Each distribution requires only one mapping to the uniform distribution; all comparisons share the same reference — avoiding the inconsistency across different couplings in the Rioux et al. approach.
2. **Natural hierarchy of relaxations**: Different values of $q$ correspond to dominance relations of varying strictness, and these relaxations are obtained at **zero additional cost** after the mapping is computed once.
3. **No hyperparameters**: Unlike the method of Rioux et al., which requires tuning regularisation parameter $\lambda$ and function $h$.
4. **Explicit sample-size thresholds**: Provides actionable finite-sample guarantees.

## Limitations & Future Work

1. **$O(n^3)$ complexity of the Hungarian algorithm** limits scalability to large-sample settings.
2. The expression for $n^*(\delta)$ depends on Lipschitz constants and $\Delta$, which are difficult to estimate accurately in practice.
3. For dimension $d \geq 5$, the available range of $\theta$ narrows, potentially limiting practical utility.
4. Experimental scope is limited (comparison with only a few baselines); the authors explicitly note that large-scale benchmarking is left for future work.
5. The theoretical relationship with Rioux et al. (2025) has not been fully clarified.

## Related Work & Insights

- The center-outward quantile theory of **Hallin (2021)** provides the core mathematical tool for this work.
- The entropic OT approach of **Rioux et al. (2025)** is used for approximate stochastic dominance testing but lacks global consistency.
- The selection mechanism in **NSGA-II** can be directly replaced by q-dominance, offering broad applicability.
- The concept of q-dominance may generalise to other settings requiring distribution comparison, such as robust optimisation and Bayesian optimisation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The combination of center-outward quantiles and stochastic dominance represents a genuinely novel theoretical contribution.)
- Experimental Thoroughness: ⭐⭐⭐ (Limited experimental scale, but sufficient to validate the theory.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematically rigorous, clearly structured, with smooth theorem–proof–application flow.)
- Value: ⭐⭐⭐⭐ (Provides a theoretically more grounded foundation for stochastic multi-objective optimisation.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)
- [\[ICML 2026\] Adaptive Multi-Round Allocation with Stochastic Arrivals](../../ICML2026/others/adaptive_multi-round_allocation_with_stochastic_arrivals.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](online_linear_regression_with_paid_stochastic_features.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)

</div>

<!-- RELATED:END -->
