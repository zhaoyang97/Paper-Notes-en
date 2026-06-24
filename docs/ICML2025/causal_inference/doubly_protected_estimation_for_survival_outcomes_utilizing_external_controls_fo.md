---
title: >-
  [Paper Note] Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials
description: >-
  [ICML 2025][Causal Inference][External controls] Proposing a doubly protected estimation framework for survival outcomes that corrects covariate shift via density ratio weighting and detects outcome drift via DR-Learner to selectively borrow comparable external controls, achieving robustness to external data heterogeneity while guaranteeing consistency and efficiency gains.
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "External controls"
  - "survival analysis"
  - "doubly robust estimation"
  - "restricted mean survival time"
  - "selective borrowing"
date: 2026-05-08
content_hash: 130d23eb3e23546b
---

# Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials

**Conference**: ICML 2025  
**arXiv**: [2410.18409](https://arxiv.org/abs/2410.18409)  
**Code**: To be released (the paper mentions open-sourcing after acceptance)  
**Area**: Biostatistics / Clinical Trials  
**Keywords**: External controls, survival analysis, doubly robust estimation, restricted mean survival time, selective borrowing

## TL;DR

Proposing a doubly protected estimation framework for survival outcomes that corrects covariate shift via density ratio weighting and detects outcome drift via DR-Learner to selectively borrow comparable external controls, achieving robustness to external data heterogeneity while guaranteeing consistency and efficiency gains.

## Background & Motivation

### Background

**Background**: In clinical trials (especially for rare diseases), the sample size of control groups is often limited, leading to insufficient statistical power.

### Key Challenge

**Key Challenge**: External controls (historical studies/real-world data) can supplement control group information, but direct usage introduces bias.

### Limitations of Prior Work

**Limitations of Prior Work**: FDA guidelines explicitly point out that external controls face issues such as **selection bias, temporal inconsistency, unobserved confounding, and outcome differences**.

### Key Insight

**Key Insight**: Most existing methods are based on Cox model assumptions, which lack flexibility; or they only handle covariate shift while ignoring outcome drift.

### Additional Explanation

**Additional Explanation**: Semiparametric efficient estimation (based on EIF) has gained popularity in causal inference recently, but has not yet been fully generalized to survival analysis scenarios with external controls.

**Design Motivation**: A flexible framework that simultaneously handles covariate shift and outcome drift, accommodates machine learning, and provides valid statistical inference (confidence intervals) is highly needed.

## Method

### Overall Architecture

The method consists of three steps:

1. **Efficient Integrative Estimation** (assuming demographic homogeneity): Derive the semiparametrically efficient influence function (EIF) of the RMST difference on the combined dataset, and construct a doubly robust estimator $\hat{\theta}_\tau^{\text{acw}}$.
2. **Selective Borrowing** (handling outcome drift): Construct pseudo-outcomes via DR-Learner to detect the bias of each external control, and filter out comparable subsets using penalized selection.
3. **Adaptive Integrative Output**: Perform final estimation $\hat{\theta}_\tau^{\text{adapt}}$ using only the selected comparable external controls.

### Key Designs

**Estimand Target**: Restricted Mean Survival Time (RMST) difference
$$\theta_\tau = \int_0^\tau \{S_1(t|R=1) - S_0(t|R=1)\} dt$$

**Doubly Robust EIF**:
- The EIF for the treatment group $S_1(t|R=1)$ uses only trial data (established theory exists).
- The EIF for the control group $S_0(t|R=1)$ integrates trial + external data, optimally combining the two data sources using the density ratio $q_R(X)$ and variance-weighting $r(t,X)$.
- The overall EIF is the time integral of the difference between the two.

**Bias Detection**:
- Define the bias parameter $b_{i,0}$ as the difference in RMST for external control $i$ between the trial population and the external population.
- Construct pseudo-outcomes $\xi_i$ using DR-Learner, leveraging cross-fitting to reduce bias.
- Apply SCAD/MCP/adaptive lasso penalties to $\xi_i$ to select external controls with zero bias.

**Density Ratio Correction**: Reweight external data to the trial population distribution based on the density ratio of $\pi_R(X) = P(R=1|X)$.

### Loss & Training

- Conditional survival curves $S_a(t|X,R=r)$: Flexible models like Cox models or random survival forests can be used.
- Censoring survival function $S^C(t|X,R)$: Similarly modeled flexibly.
- Propensity scores $\pi_R(X)$, $\pi_A(X)$: Modeled using SuperLearner (logistic regression + random forest).
- Adopt cross-fitting to avoid overfitting bias.
- Penalty parameter $\lambda_N$ is selected via a BIC-type criterion.

## Key Experimental Results

### Main Results

Five simulation settings simulate five categories of bias sources of concern to the FDA:

| Setting | Bias Type | Description |
|------|----------|------|
| Setting 1 | Covariate shift only | External and trial share the same survival parameters |
| Setting 2 | Unobserved confounding | Introduce unobserved factor U, larger external shift |
| Setting 3 | Temporal inconsistency | 50% of external controls have temporal shift δ∈{0,5} |
| Setting 4 | Different covariate effects | External covariate coefficient 0.5 vs trial 0.2 |
| Setting 5 | Different baseline hazards | Time-varying hazard functions differ (Weibull vs exponential) |

- External sample size $N_e=500$ is fixed, and the trial control group size $N_0$ varies from 50 to 400.
- Under Setting 1, all integrative methods outperform the trial-only estimator (lower Root-MSE).
- Under Settings 2-5, the proposed $\hat{\theta}_\tau^{\text{adapt}}$ achieves minimal bias, outperforming naive full borrowing and TransCox transfer learning.

### Ablation Study

- The method remains robust under different censoring rates (40%/60%).
- Choice of penalty function (adaptive lasso vs SCAD vs MCP): Consistent results.
- DR-Learner vs plug-in bias detection: DR-Learner is significantly superior in small samples.

### Key Findings

- **Galcanezumab migraine trial** real-world data application: After borrowing external data from the FDA Adverse Event Reporting System (FAERS), the estimation precision of the RMST difference improves by approximately 25%, and the confidence interval narrows significantly.
- Selective borrowing correctly identifies approximately 70-90% of comparable external controls.

## Highlights & Insights

1. **Double Protection**: Doubly robust to covariate shift + selective borrowing for outcome drift, offering progressive two-layer protection mechanisms.
2. **Solid Theory**: Proven rate double robustness, asymptotic normality, and local efficiency, with variance strictly bounded by the trial-only estimator.
3. **Extremely Flexible**: Bypasses the Cox model assumption, allowing arbitrary machine learning models to estimate nuisance functions as long as the product convergence rate is $o(N^{-1/2})$.
4. **Generalizable**: The framework can be directly extended to any estimand target based on the survival function $S_a(t)$ (e.g., median survival time).

## Limitations & Future Work

- Currently assumes a "comparable subset" exists in external controls, but in practice, bias could be continuous (partial borrowing would be more flexible).
- The power of DR-Learner's bias detection remains limited in small samples, which might mistakenly include some biased external controls.
- Only considers a single external data source; multi-source external data integration requires further study.
- Extreme values of the density ratio (positivity violation) may cause variance inflation, requiring truncation.
- Computational cost: Requires fitting multiple conditional survival models + cross-fitting, which needs engineering optimization for practical deployment.

## Related Work & Insights

- **Gao et al. (2024a)**: Doubly robust integration of external controls under non-survival outcomes; this paper generalizes it to survival analysis.
- **Li et al. (2023b)**: A transfer learning framework based on penalized Cox models, which only addresses proportional hazards.
- **Chen et al. (2022)**: Propensity score weighted KM estimation, ignoring time-varying drift.
- **Kennedy (2020), Kallus & Oprescu (2023)**: Theoretical foundations of the DR-Learner.
- **Tsiatis (2006)**: Semiparametric theory and EIF derivation framework under monotonic coarsening.

**Insight**: Extending the doubly robust + transfer learning framework in causal inference to survival analysis provides a theoretically guaranteed statistical tool for external control integration as encouraged by the FDA.

## Rating

- Novelty: ⭐⭐⭐⭐ — First semiparametric efficient framework in survival analysis handling both covariate shift and outcome drift.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five FDA-concerned scenarios + real data, but lacks high-dimensional covariate experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations, though notation is dense and requires patient reading.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses FDA guidance requirements, highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation](transformer-based_spatial-temporal_counterfactual_outcomes_estimation.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](../../ICLR2026/causal_inference/direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)
- [\[ICLR 2026\] Privacy-Protected Causal Survival Analysis Under Distribution Shift](../../ICLR2026/causal_inference/privacy-protected_causal_survival_analysis_under_distribution_shift.md)
- [\[ICLR 2026\] Conformalized Survival Counterfactuals Prediction for General Right-Censored Data](../../ICLR2026/causal_inference/conformalized_survival_counterfactuals_prediction_for_general_right-censored_dat.md)
- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](../../ICLR2026/causal_inference/an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)

</div>

<!-- RELATED:END -->
