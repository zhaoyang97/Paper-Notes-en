---
title: >-
  [Paper Note] Integer Programming for Generalized Causal Bootstrap Designs
description: >-
  [ICML2025][Optimization][Causal bootstrap] Proposes an integer programming (IP) based method to numerically solve for the worst-case copula, extending the design uncertainty quantification of causal bootstrap from "complete randomization + difference-in-means estimator" to any known probability assignment and linear/quadratic treatment estimators, and proves asymptotic validity.
tags:
  - "ICML2025"
  - "Optimization"
  - "Causal bootstrap"
  - "Integer programming"
  - "Design uncertainty"
  - "Variance upper bound"
  - "Worst-case copula"
  - "Finite population inference"
date: 2026-05-08
content_hash: 6091d6c57e639708
---

# Integer Programming for Generalized Causal Bootstrap Designs

**Conference**: ICML2025  
**arXiv**: [2410.21464](https://arxiv.org/abs/2410.21464)  
**Code**: Not publicly available (open-source solvers mentioned in the paper)  
**Area**: Optimization / Causal Inference / Experimental Design  
**Keywords**: Causal bootstrap, Integer programming, Design uncertainty, Variance upper bound, Worst-case copula, Finite population inference

## TL;DR

Proposes an integer programming (IP) based method to numerically solve for the worst-case copula, extending the design uncertainty quantification of causal bootstrap from "complete randomization + difference-in-means estimator" to any known probability assignment and linear/quadratic treatment estimators, and proves asymptotic validity.

## Background & Motivation

In experimental causal inference, sources of uncertainty are divided into two categories:

**Sampling uncertainty**: The randomness arising when samples are drawn from a superpopulation. Classical bootstrap and Wald-type confidence intervals primarily estimate this source.

**Design uncertainty**: The randomness brought by the random assignment of treatment/control under the finite population model. Each individual can only observe one potential outcome (treatment or control), so the statistics fluctuate under different randomizations.

In scenarios such as spatial experiments (e.g., country-level A/B testing), where the sample size is small (tens to hundreds), treatment effect heterogeneity is large, and the samples cover almost the entire population, sampling uncertainty methods are often **too conservative**. In Neyman's variance decomposition, the variance term $S_\tau^2$ (variance of individual-level treatment effects) depends on the unobserved joint distribution and cannot be directly estimated.

**Limitations of Prior Work**:

- The causal bootstrap methods of Aronow et al. (2014) and Imbens & Menzel (2021) only apply to **complete randomization assignment + difference-in-means estimator**, utilizing the analytical solution of the isotone copula (sorted pairing).
- For non-standard combinations such as matched-pair designs, Horvitz-Thompson estimators, and doubly-robust estimators, the isotone copula is no longer the known worst-case distribution, and a general method is lacking.

**Design Motivation**: To **numerically solve** for the worst-case copula via integer programming, bypassing the reliance on analytical solutions, thereby generalizing to more estimators and assignment mechanisms.

## Method

### Overall Architecture

The goal is to find the joint potential outcome distribution $F_{0,1}$ that maximizes the estimator's variance:

$$\max_{Y_i(0), Y_i(1) \in \mathcal{Y}^2} \mathbf{Var}_Z[\hat{\tau}] \quad \text{s.t.} \quad F_{0,1} \in \mathcal{C}$$

where $\mathcal{C}$ is the set of constraints on the joint distribution to ensure consistency with the observed marginal distributions.

### Key Designs

**Decision Variables**: Introduce indicator variables $X_{ik}^{(a)} := \mathbb{I}(Y_i(a) = y_k)$, where $a \in \{0,1\}$ is the treatment status, $i$ represents individuals, and $y_k$ represents discrete outcome values.

**Objective Function**: The variance can be expressed as a quadratic form $\mathbf{Var}_Z[\hat{\tau}] = \mathbf{X}^T \mathbf{Q} \mathbf{X}$, where $\mathbf{Q} = \mathbf{Y}^T \mathbf{\Sigma}_{ZZ} \mathbf{Y}$, and $\mathbf{\Sigma}_{ZZ}$ is the covariance matrix of treatment assignment. Since $X$ is a binary variable, the quadratic term can be linearized, transforming the problem into a standard integer linear program.

**Constraints** (taking complete randomization as an example):

- **(a) Observation Lock**: $X_{ik}^{(Z_i)} = 1$ if and only if $Y_i^{obs} = y_k$, showing that observed potential outcomes are fixed.
- **(b) Support Constraint** (optional): $X_{ik}^{(a)} = 0$ if $y_k \notin \text{supp}(F_a)$, restricting the potential outcomes to take only observed values.
- **(c) Unique Matching**: $\sum_k X_{ik}^{(a)} = 1$, ensuring each individual takes exactly one value under each treatment status.
- **(d) Marginal Matching**: $\sum_i X_{ik}^{(a)} (Z_i/N_1 - (1-Z_i)/N_0) = 0$, ensuring consistency between observed and missing marginal distributions. Since constraint (d) might be infeasible, a relaxation parameter $\epsilon$ is introduced.

**Lemma 2.1**: When $\epsilon \geq 1/\min(N_0, N_1)$, the integer program is guaranteed to be feasible.

### Generalization of Estimators

**Linear Treatment Estimators**: Handled as $\hat{\tau} = \sum_i Z_i a_i + b_i$, which includes the Horvitz-Thompson estimator. The variance is $\sum_{i,j} a_i a_j \mathbf{Cov}[Z_i, Z_j]$, requiring only the replacement of $\mathbf{Q}$ with $\mathbf{Q}' = \mathbf{Y}^T \mathbf{U}^T \mathbf{\Sigma}_{ZZ} \mathbf{U} \mathbf{Y}$.

**Quadratic Treatment Estimators**: Handled as $\hat{\tau} = \sum_i b_i + \sum_j Z_i Z_j a_{ij}$, which includes linear regression fitting of treatment variables and covariates. The variance involves fourth-order moments $\mathbf{Cov}[Z_i Z_j, Z_k Z_l]$.

**Doubly-Robust Estimator**: Utilizes out-of-sample fitted covariate prediction functions, transforming the problem into applying the Horvitz-Thompson estimator to the residuals $Y_i' = Y_i - Z_i \hat{f}_1(W_i) - (1-Z_i)\hat{f}_0(W_i)$.

### Generalization of Assignment Mechanisms

**General Known Probability Assignment**: Generalizes constraint (d) to marginal matching based on propensity score weighting:

$$(-1)^b \sum_i X_{ik}^{(a)} \left(\frac{Z_i}{P_i} - \frac{1-Z_i}{1-P_i}\right) \leq \epsilon N$$

**Conditionally Unconfounded Assignment**: Utilizes conditional marginal distribution matching of covariates $W_i$, adding finer-grained constraints to tighten the confidence intervals.

### Asymptotic Validity

**Theorem 4.1**: For unconfounded assignment, $\mathbb{P}(V^* \geq \mathbf{Var}_Z[\hat{\tau}]) \geq 1 - \beta$, where $\beta = 8\exp(-\epsilon^2 N \tilde{P}/4) + O(\text{Cov}(Z_i,Z_j))$, which approaches 0 as $N$ increases.

**Theorem 4.3**: For individual assignment with bounded confounding, it is also asymptotically valid when the relaxation $\epsilon \geq \delta$ (confounding degree).

## Key Experimental Results

Uses IMF public GDP data from 2017–2019 (top 50 countries) to simulate country-level spatial experiments.

### Main Results (Table 1 - A/A Test, 95% CI Width)

| Covariates | Method | Completely Randomized Coverage | Matched-Pair Coverage | Completely Randomized CI | Matched-Pair CI |
|--------|------|--------------|--------------|-----------|-----------|
| None | Sampling Bootstrap | 90% | 100% | 3967 | 4110 |
| None | Conservative Var. | 98% | 100% | 3991 | 1152 |
| None | Isotone Copula | 87% | **0%** | 3348 | 44 |
| None | **Opt. Causal Boot.** | 87% | **100%** | 3348 | **2233** |
| 2018 GDP | Sampling Bootstrap | 91% | 100% | 141 | 142 |
| 2018 GDP | **Opt. Causal Boot.** | 90% | **100%** | **133** | **102** |

### Key Findings

1. **Under Complete Randomization**: The proposed method (Ours) is completely consistent with the isotone copula, verifying the correctness of the IP solver.
2. **Under Matched-Pair Design**: The coverage rate of the isotone copula drops to **0%** (variance estimate is zero), while the proposed method achieves **100%** coverage with a reasonable CI width.
3. **CI Width**: Compared to sampling bootstrap, the proposed method reduces the CI width by **$\geq 10\%$** in all experiments, and up to **50%** in the difference-in-means estimator + multiplicative effect scenario.
4. After including the historical GDP covariate, the CI width drops significantly from ~3000+ to ~130, demonstrating the massive benefit of covariate adjustment.

### Scalability

- Matched-pair design: an average of 671 seconds for 50 units (IP solver).
- Completely randomized design: LP relaxation takes only 22 seconds.
- Computational complexity increases with the number of units, making it suitable for datasets of up to several hundred units.

## Highlights & Insights

1. **Methodological Breakthrough**: Replaces analytical solutions with integer programming to solve the worst-case copula, unlocking the applicability of causal bootstrap to non-standard designs (e.g., matched-pair) and non-standard estimators (e.g., doubly-robust, linear regression).
2. **Failure Case of Isotone Copula**: The 0% coverage rate of the isotone copula under matched-pair design serves as an excellent counterexample—when pairing is good, counterfactual imputation leads to variance degenerating to zero.
3. **Theoretical Completeness**: Covers asymptotic validity proofs for three types of assignment mechanisms: unconfounded, conditionally unconfounded, and bounded confounding.
4. **Practical Value**: Spatial experiments (such as country/city-level A/B testing) are common scenarios in industry. This work directly addresses the key pain point of uncertainty quantification in these settings.

## Limitations & Future Work

1. **Computational Scalability**: The solver time of integer programming grows rapidly with the number of units. Currently, it is only suitable for small samples of up to several hundred units, requiring approximation algorithms for large-scale experiments.
2. **Insufficient Coverage**: Under complete randomization with heavy-tailed data, the coverage of all bootstrap methods falls below the nominal level (87-91% vs 95%), with the slow convergence of the marginal distribution being the fundamental cause.
3. **Uncovered General Confounded Assignment**: Only the asymptotic validity for individual assignment with bounded confounding is proved. Asymptotic validity for general confounding mechanisms is left for future work.
4. **Discretization Assumption**: Continuous outcomes must be discretized to the support of the observed values. When the support of missing potential outcomes exceeds the observed range, the upper bound is no longer guaranteed to be valid.
5. **Covariate Dimension**: Conditional marginal matching converges more slowly under high-dimensional/continuous covariates, and the practical improvement may be limited.

## Related Work & Insights

- **Neyman (1923)**: Foundational work for finite population causal inference, the starting point of variance decomposition.
- **Aronow et al. (2014)**: Proved that the isotone copula is the variance-maximizing copula under complete randomization + DIM.
- **Imbens & Menzel (2021)**: Proposed the causal bootstrap resampling method, which is the direct predecessor of this work.
- **Harshaw et al. (2021)**: Obtained quadratic variance bounds via convex optimization, but limited to quadratic forms.
- **Ji et al. (2023)**: Estimated bounds of joint distribution functionals using optimal transport, a dual method but does not explicitly solve the coupling.
- **Insight**: Transforming unidentifiable problems in causal inference into mathematical programming problems is a general paradigm, which can be extended to other counterfactual estimation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — The approach of solving for the worst-case copula using integer programming is highly novel, unlocking previously intractable design $\times$ estimator combinations.
- Experimental Thoroughness: ⭐⭐⭐ — The simulation design with IMF GDP data is reasonable, but it only uses one dataset, lacking validation on a wider variety of real-world scenarios.
- Writing Quality: ⭐⭐⭐⭐ — Clear development from simple to general cases, with mathematically complete theoretical proofs.
- Value: ⭐⭐⭐⭐ — Possesses direct practical value for small-sample causal inference scenarios like spatial experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FMIP: Joint Continuous-Integer Flow for Mixed-Integer Linear Programming](../../ICLR2026/optimization/fmip_joint_continuous-integer_flow_for_mixed-integer_linear_programming.md)
- [\[ICML 2026\] Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming](../../ICML2026/optimization/provably_data-driven_lagrangian_relaxation_for_mixed_integer_linear_programming.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICML 2025\] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](sdp-crown_efficient_bound_propagation_for_neural_network_verification_with_tight.md)

</div>

<!-- RELATED:END -->
