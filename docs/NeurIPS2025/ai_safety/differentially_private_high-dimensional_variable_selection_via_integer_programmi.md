---
title: >-
  [Paper Note] Differentially Private High-dimensional Variable Selection via Integer Programming
description: >-
  [NeurIPS 2025][AI Safety][Differential Privacy] This paper proposes two pure differentially private sparse variable selection methods (top-R and mistakes) that leverage modern mixed integer programming (MIP) techniques to efficiently explore non-convex objective landscapes, achieving state-of-the-art support recovery rates in high-dimensional settings (p up to 10,000) while providing theoretical recovery guarantees.
tags:
  - NeurIPS 2025
  - AI Safety
  - Differential Privacy
  - Sparse Variable Selection
  - Mixed Integer Programming
  - Exponential Mechanism
  - Best Subset Selection
date: 2026-05-08
content_hash: 914aa018bf77b7e7
---

# Differentially Private High-dimensional Variable Selection via Integer Programming

**Conference**: NeurIPS 2025
**arXiv**: [2510.22062](https://arxiv.org/abs/2510.22062)
**Code**: [github.com/petrosprastakos/DP-variable-selection](https://github.com/petrosprastakos/DP-variable-selection)
**Area**: AI Safety / Differential Privacy
**Keywords**: Differential Privacy, Sparse Variable Selection, Mixed Integer Programming, Exponential Mechanism, Best Subset Selection

## TL;DR
This paper proposes two pure differentially private sparse variable selection methods (top-R and mistakes) that leverage modern mixed integer programming (MIP) techniques to efficiently explore non-convex objective landscapes, achieving state-of-the-art support recovery rates in high-dimensional settings (p up to 10,000) while providing theoretical recovery guarantees.

## Background & Motivation

**Background**: High-dimensional sparse variable selection (Best Subset Selection, BSS) is a classical problem in statistics, aiming to minimize a loss function subject to the constraint that the $\ell_0$-norm of the coefficients does not exceed $s$. Recent advances in MIP techniques have enabled non-private BSS to be solved efficiently at the scale of millions of variables.

**Limitations of Prior Work**: Existing DP variable selection methods either rely on Lasso (with support recovery probability that does not tend to one), require enumeration over all $\binom{p}{s}$ feasible support sets (exponential complexity, not scalable), or only provide approximate DP guarantees.

**Key Challenge**: The exponential mechanism for variable selection requires traversing an exponentially large output space, making both computation and sampling infeasible.

**Goal**: Design scalable pure DP variable selection algorithms that leverage MIP techniques to avoid exhaustive enumeration of all support sets.

**Key Insight**: The key observation is that candidate sets far from the optimal support incur large objective values and thus carry negligible probability mass under the exponential mechanism. Consequently, it suffices to compute the exact objective values for only the top $R$ optimal support sets, using a lower bound for all others.

**Core Idea**: Use a MIP solver to efficiently identify the top $R$ optimal support sets, substitute the objective value of the $R$-th support set as a lower bound for all remaining sets, and construct a truncated exponential mechanism to achieve pure DP.

## Method

### Overall Architecture
The input consists of a data matrix $X$, observations $y$, and privacy parameter $\epsilon$. Both methods are based on modified versions of the exponential mechanism and output a feature subset $S$ of size $s$.

### Key Designs

1. **Top-R Method (Algorithm 1)**:

    - **Function**: Samples a support set from a modified exponential mechanism.
    - **Mechanism**: A MIP solver first identifies the top $R$ support sets with the smallest objective values. True objective values are used for $k \leq R$; for $k > R$, the objective value of the $R$-th support set serves as a lower bound.
    - **Sampling distribution**: $P_0(k) \propto \exp\!\left(-\varepsilon \cdot R(S_k, D)/(2\Delta)\right)$ for $k \leq R$; $P_0(R+1) \propto \bigl(\binom{p}{s} - R\bigr)\exp\!\left(-\varepsilon \cdot R(S_R, D)/(2\Delta)\right)$.
    - **Privacy guarantee**: Pure $\varepsilon$-DP as $T \to \infty$ (Theorem 1), requiring only standard bounded-data assumptions.

2. **Mistakes Method**:

    - **Function**: Samples by grouping support sets according to their "error count" relative to the optimal support.
    - **Mechanism**: Support sets are partitioned into $s+1$ groups based on the number of features $t$ that differ from the optimal support; only the objective value of the optimal support is computed within each group.
    - **Privacy guarantee**: Pure $\varepsilon$-DP under an objective-gap condition.
    - **Design Motivation**: Only $s+1$ MIP solves are required (vs. $R$ for top-R), and the theoretical recovery condition is weaker.

3. **MIP Solver — Outer Approximation Algorithm (Algorithm 2)**:

    - **Function**: Efficiently computes the top-$R$ support sets.
    - **Mechanism**: Constructs cutting-plane approximations for the ridge-penalized objective (using Danskin's theorem for gradient computation) and solves the MIP iteratively via outer approximation.
    - Supports arbitrary convex loss functions (least squares, hinge loss, Huber loss, etc.) within the same algorithmic framework.

### Loss & Training
- Global sensitivity: $\Delta = 2b_y^2 + 2b_x^2 r^2 s$ (derived from bounded-data assumptions).
- Experimental settings: $R = 2 + (p-s) \cdot s$, $b_x = b_y = 0.5$, $r = 1.1$.

## Key Experimental Results

### Main Results: Support Recovery Rate ($p=10000$, $s=5$, $\mathrm{SNR}=5$, $\rho=0.1$, $\varepsilon=1$)

| Method | $n=200$ | $n=400$ | $n=600$ | $n=800$ | DP Type |
|--------|---------|---------|---------|---------|---------|
| Top-R (Ours) | ~15% | ~55% | ~85% | ~95% | Pure $\varepsilon$-DP |
| Mistakes (Ours) | ~25% | ~70% | ~92% | ~98% | Pure $\varepsilon$-DP |
| MCMC (Roy & Tewari) | ~5% | ~20% | ~45% | ~60% | Approximate DP |
| Samp-Agg (Lasso) | <5% | ~10% | ~15% | ~20% | Pure $\varepsilon$-DP |

### Theoretical Recovery Condition Comparison

| Method | Sufficient Condition on $\tau$ | Additional Assumptions |
|--------|-------------------------------|----------------------|
| Non-private BSS | $\tau \gg \log(p)/n$ | None |
| Exponential Mechanism (enumeration) | $\tau \gg \max\!\left(\log(p)/n,\; s\log(p)/(n\varepsilon)\right)$ | Requires enumeration of $\binom{p}{s}$ sets |
| Top-R (Ours) | $\tau \gg \max\!\left(\log(p)/n,\; s^2\log(p)/(n\varepsilon)\right)$ | Bounded data only |
| Mistakes (Ours) | $\tau \gg \max\!\left(s\log(p)/n,\; s\log(p)/(n\varepsilon)\right)$ | Requires objective-gap condition |

### Key Findings
- The mistakes method numerically outperforms top-R across all parameter settings, consistent with theory.
- Both proposed methods substantially outperform the MCMC and Samp-Agg baselines, achieving near-perfect recovery at sufficiently large $n$.
- In the low-privacy regime, the $\log(p)/n$ term dominates for top-R, matching the non-private optimal rate.
- Extension to hinge loss for sparse classification also proves effective.

## Highlights & Insights
- Incorporating MIP solvers into DP sampling elegantly transforms the exponential enumeration problem into a finite number of MIP calls. This idea is transferable to any DP combinatorial optimization problem.
- The mistakes method compresses the exponential support space into $s+1$ partitions, each requiring only a single MIP solve.
- The outer approximation algorithm exploits Danskin's theorem so that subproblem gradients can be computed analytically, avoiding numerical differentiation.
- The choice of $R$ reflects a three-way trade-off among privacy, accuracy, and computational cost: larger $R$ yields higher accuracy but increases both computational cost and privacy loss.
- Pure DP guarantees are stronger than approximate DP, commanding greater trust in high-sensitivity domains such as healthcare and finance.

## Limitations & Future Work
- The privacy guarantee of the mistakes method relies on an additional objective-gap assumption.
- Theoretical recovery guarantees are limited to BSS (least squares), even though experiments demonstrate effectiveness for hinge loss as well.
- The recovery condition for top-R includes an extra factor of $s$ compared to the full exponential mechanism.
- MIP solvers may encounter scalability bottlenecks at very large scales ($p > 10^5$).

## Related Work & Insights
- **vs. Roy & Tewari (2024, MCMC)**: Their approach approximates the exponential mechanism via MCMC, providing only approximate DP; this paper provides pure DP with significantly better empirical performance.
- **vs. Lei et al. (2024)**: Their method directly enumerates all support sets under the exponential mechanism, which is not scalable; this paper avoids enumeration via MIP.
- **vs. Kifer & Smith (2011, propose-test-release)**: Their method has a non-zero failure probability that does not vanish with $n$; the proposed methods recover the support with high probability for sufficiently large $n$.
- The truncated exponential mechanism framework can be generalized to DP subset selection, DP feature engineering, and other combinatorial optimization settings.
- The methods have direct practical relevance for feature selection over sensitive data in healthcare, finance, and recommender systems.
- All experiments were conducted on a 20-core, 64 GB RAM cluster using the Gurobi solver; wall-clock runtimes are comparable to the MCMC baseline.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of MIP and DP is a genuinely new direction; both truncated exponential mechanism designs are elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-parameter ablations, $p$ up to 10,000, and extension to hinge loss.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical results are stated clearly; algorithm pseudocode is complete.
- **Value**: ⭐⭐⭐⭐ Provides the first practical pure DP solution with theoretical guarantees for DP variable selection.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[NeurIPS 2025\] Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)

<!-- RELATED:END -->
