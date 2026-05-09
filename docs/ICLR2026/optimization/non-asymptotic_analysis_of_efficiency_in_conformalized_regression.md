---
title: >-
  [Paper Note] Non-Asymptotic Analysis of Efficiency in Conformalized Regression
description: >-
  [ICLR 2026][Optimization][conformal prediction] This work establishes the first non-asymptotic efficiency bounds for Conformalized Quantile Regression (CQR) and Conformalized Median Regression (CMR) trained with SGD, explicitly characterizing the joint dependence of prediction set length bias on training sample size $n$, calibration sample size $m$, and miscoverage level $\alpha$.
tags:
  - ICLR 2026
  - Optimization
  - conformal prediction
  - quantile regression
  - non-asymptotic analysis
  - prediction set efficiency
  - uncertainty quantification
date: 2026-05-08
content_hash: a8d60f295c19b8fc
---

# Non-Asymptotic Analysis of Efficiency in Conformalized Regression

**Conference**: ICLR 2026
**arXiv**: [2510.07093](https://arxiv.org/abs/2510.07093)
**Code**: None
**Area**: Optimization / Statistical Learning Theory
**Keywords**: conformal prediction, quantile regression, non-asymptotic analysis, prediction set efficiency, uncertainty quantification

## TL;DR

This work establishes the first non-asymptotic efficiency bounds for Conformalized Quantile Regression (CQR) and Conformalized Median Regression (CMR) trained with SGD, explicitly characterizing the joint dependence of prediction set length bias on training sample size $n$, calibration sample size $m$, and miscoverage level $\alpha$.

## Background & Motivation

Conformal prediction is a distribution-free framework that provides prediction sets with coverage guarantees for black-box models. In regression, the **efficiency** of conformal prediction is typically measured by the expected length of prediction intervals—subject to the coverage condition $\mathbb{P}[Y \in \mathcal{C}(X)] \geq 1-\alpha$, smaller prediction sets are preferred.

Existing efficiency analyses suffer from two main limitations:

**Asymptotic analysis**: proves that prediction sets converge to oracle sets as sample size tends to infinity, but provides no finite-sample guarantees.

**Existing non-asymptotic bounds**: typically treat $\alpha$ as a constant and consider only the effect of calibration set size $m$, ignoring the influence of training set size $n$ and the value of $\alpha$.

In safety-critical applications (medical, financial, autonomous driving), $\alpha$ is often chosen to be very small to ensure high coverage, yet the efficiency behavior in this regime remains unclear. This paper fills the theoretical gap by providing the first non-asymptotic characterization of efficiency with respect to all three quantities $(n, m, \alpha)$.

## Method

### Overall Architecture

The paper analyzes two variants of split conformal prediction:
- **Conformalized Quantile Regression (CQR)**: estimates conditional upper and lower quantiles to construct adaptive asymmetric prediction intervals.
- **Conformalized Median Regression (CMR)**: estimates the conditional median, uses absolute residuals as the nonconformity score, and constructs symmetric prediction intervals.

Both methods employ linear models trained via SGD. The analysis characterizes the expected absolute deviation between the prediction set length and the oracle interval length.

### Key Designs

1. **Efficiency Bound for CQR-SGD (Theorem 3.2)**:

    - Core result: $\mathbb{E}[||{\mathcal{C}(X)}| - |{\mathcal{C}^*(X)}||] \leq \mathcal{O}(n^{-1/2} + (\alpha^2 n)^{-1} + m^{-1/2} + \exp(-\alpha^2 m))$
    - Interpretation of the four terms:
        - $n^{-1/2}$: standard training error from quantile regression.
        - $(\alpha^2 n)^{-1}$: amplification of training error as $\alpha$ decreases.
        - $m^{-1/2}$: finite-sample effect of the calibration set.
        - $\exp(-\alpha^2 m)$: exponential calibration term that deteriorates as $\alpha$ decreases.
    - Assumptions: well-specified linear model (Assumption 3.1), bounded covariance (Assumption 3.2), conditional density regularity (Assumption 3.3).

2. **Efficiency Bound for CMR-SGD (Theorem 4.1)**:

    - Under an additional quantile symmetry assumption (Assumption 4.2), CMR achieves bounds of the same order as CQR.
    - CMR produces constant-length prediction sets (independent of the input), making the analysis more concise.

3. **Phase Transition Analysis (Section 3.2.1)**:

    - When $n = \Theta(m)$, the bound simplifies to $\mathcal{O}(n^{-1/2} + (\alpha^2 n)^{-1})$.
    - Three regimes of $\alpha$:
        - $\alpha = \Omega(n^{-1/4})$: bound is $\mathcal{O}(n^{-1/2})$, independent of $\alpha$.
        - $n^{-1/2} \ll \alpha \ll n^{-1/4}$: bound transitions to $\mathcal{O}((\alpha^2 n)^{-1})$.
        - $\alpha = \Theta(n^{-1/2})$: extreme regime where $\alpha$ dominates the bound.
    - Provides practical guidance on data allocation: when $\alpha$ is sufficiently large, training and calibration sets should be of the same order; when $\alpha$ is small, more training data is needed.

### Proof Strategy

The proof proceeds in three main steps (corresponding to three propositions):
1. **Population quantile bound (Prop B.5)**: controls the effect of parameter estimation error on the quantile of the score distribution.
2. **Finite-sample score quantile gap (Prop B.7)**: controls the discrepancy between the $(1-\alpha)_m$ and $(1-\alpha)$ quantiles.
3. **Empirical score quantile concentration (Prop B.11)**: applies the DKW inequality to bound the deviation between empirical and population quantiles.

Key technical ingredients include exploiting the strong convexity of the pinball loss to obtain an $\mathcal{O}(1/n)$ centering error bound for SGD (Theorem 3.1), and using conditional density regularity to ensure a lower-bounded density near the score quantile.

## Key Experimental Results

### Synthetic Data Experiments

Experiments use piecewise affine conditional densities to validate the theoretical bounds along three dimensions:

| Dimension | Theoretical Prediction | Empirical Observation |
|-----------|----------------------|-----------------------|
| Training size $n$ | log Δ vs log $n$ slope ranges from −1 to −0.5 | slope transitions from −1 to −0.5 as $\alpha$ increases |
| Calibration size $m$ | log-log slope approaches −0.5 | measured slope close to −0.5 |
| Miscoverage level $\alpha$ | Δ ~ $\alpha^{-2}$ | fitted exponent $b_1 = -2.24$, close to theoretical value −2 |

### Real Data Experiments

Using MEPS Panel 19/20 datasets:
- Increasing calibration size $m$ consistently reduces length bias.
- Under fixed sample size, larger $\alpha$ yields smaller bias and lower variance.
- Both CQR and CMR exhibit trends consistent with theoretical predictions.

### Key Findings

- In the regime dominated by $\mathcal{O}((\alpha^2 n)^{-1})$, the log-log regression slope of measured Δ vs $1/(n\alpha^2)$ is approximately 0.92, close to the theoretical value of 1.
- The phase transition in the efficiency bound is clearly validated by experiments.

## Highlights & Insights

1. **First triple $(n, m, \alpha)$ efficiency bound**: prior work either considered only $m$ or treated $\alpha$ as a constant; this paper is the first to reveal the critical role of $\alpha$.
2. **Assumptions directly on the data distribution**: unlike prior work that imposes assumptions on the score distribution, the assumptions here are placed directly on the data distribution, making them more natural and verifiable.
3. **Optimizer-agnostic analysis framework**: although SGD is used as the running example, the framework generalizes to other optimizers by substituting the corresponding training error bound (e.g., SAGA/SVRG can yield exponential convergence rates).
4. **Practical data allocation guidance**: depending on the magnitude of $\alpha$, the results guide how to allocate data between training and calibration sets to minimize excess prediction set length.

## Limitations & Future Work

1. **Linear model assumption**: theoretical results are restricted to linear quantile regression; extending to nonlinear models (e.g., neural networks) requires new technical tools.
2. **Conditional density regularity**: requires the conditional density to be bounded above and below ($f_{\min} \leq f_{Y|X} \leq f_{\max}$), excluding heavy-tailed distributions and cases where the density vanishes.
3. **Symmetry assumption for CMR**: Assumption 4.2 requires quantiles to be symmetric around the median, limiting the applicability of the CMR analysis.
4. **Absence of lower bounds**: only upper bounds are established; whether matching lower bounds exist remains an important open problem.

## Related Work & Insights

- **Romano et al. (2019)**: originators of CQR; this paper provides a theoretical characterization of its efficiency.
- **Lei et al. (2018)**: incorporates training error into efficiency analysis, but treats $\alpha$ as a constant.
- **Bars & Humbert (2025)**: non-asymptotic analysis of volume-minimizing conformal methods under a finite function class assumption; consistent with the present results when $\alpha$ is fixed.
- **Rakhlin et al. (2012)**: provides optimal SGD convergence rates for strongly convex objectives; serves as the theoretical foundation for Theorem 3.1.

## Rating

- Novelty: ⭐⭐⭐⭐ — first non-asymptotic bound jointly characterizing $(n, m, \alpha)$.
- Experimental Thoroughness: ⭐⭐⭐⭐ — synthetic and real data experiments validate theoretical predictions from multiple angles.
- Writing Quality: ⭐⭐⭐⭐⭐ — clear theoretical exposition, well-structured proof framework, intuitive experiments.
- Value: ⭐⭐⭐⭐ — fills an important theoretical gap in conformal prediction efficiency and provides practical data allocation guidance.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[ICLR 2026\] RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](../../NeurIPS2025/optimization/kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)

<!-- RELATED:END -->
