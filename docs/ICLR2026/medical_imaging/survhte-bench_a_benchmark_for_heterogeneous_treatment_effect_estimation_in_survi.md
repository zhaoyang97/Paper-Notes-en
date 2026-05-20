---
title: >-
  [Paper Note] SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis
description: >-
  [ICLR2026][Medical Imaging][heterogeneous treatment effects] This paper introduces SurvHTE-Bench, the first comprehensive benchmark for heterogeneous treatment effect (HTE) estimation on right-censored survival data…
tags:
  - "ICLR2026"
  - "Medical Imaging"
  - "heterogeneous treatment effects"
  - "survival analysis"
  - "right-censored data"
  - "causal inference"
  - "benchmark"
  - "CATE"
  - "meta-learners"
  - "precision medicine"
date: 2026-05-08
content_hash: f95ec76a011e18c5
---

# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

**Conference**: ICLR2026
**arXiv**: [2603.05483](https://arxiv.org/abs/2603.05483)  
**Code**: [GitHub](https://github.com/Shahriarnz14/SurvHTE-Bench)  
**Area**: Medical Imaging
**Keywords**: heterogeneous treatment effects, survival analysis, right-censored data, causal inference, benchmark, CATE, meta-learners, precision medicine

## TL;DR

This paper introduces SurvHTE-Bench, the first comprehensive benchmark for heterogeneous treatment effect (HTE) estimation on right-censored survival data, encompassing 40 synthetic datasets, 10 semi-synthetic datasets, and 2 real-world datasets. It systematically evaluates 53 estimation methods under varying causal assumption violations and censoring levels, finding that no single method dominates, and that survival meta-learners—particularly S-Learner-Survival and Matching-Survival—are most robust under high censoring and assumption violations.

## Background & Motivation

### Problem Definition

HTE estimation aims to quantify the differential treatment effect across individuals, which is central to precision medicine and personalized policy-making. In survival analysis settings, observation times are subject to right censoring (i.e., some individuals do not experience the target event before the study ends), imposing a triple challenge on HTE estimation:

**Counterfactual unobservability**: Each individual can only be observed under one treatment condition.

**Confounding**: In observational studies, treatment assignment is influenced by covariates.

**Censoring mechanism**: Censoring may be related to event times (informative censoring), violating standard assumptions.

### Limitations of Existing Evaluation Practices

Despite the proliferation of survival HTE estimation methods (causal survival forests, survival meta-learners, outcome imputation approaches, etc.), evaluation practices remain highly fragmented:

- Individual studies rely on custom simulation data with heterogeneous assumption settings and censoring levels.
- No unified benchmark with known ground truth exists.
- Fair comparison across methods is infeasible.
- Robustness of estimators under **simultaneous multiple assumption violations** is unknown.

Existing causal inference benchmarks (e.g., CausalBench) target fully observed outcomes, while survival ATE benchmarks do not cover individual-level heterogeneous effects. **A standardized benchmark for survival HTE estimation has been absent**—this constitutes the core motivation of this paper.

### Causal Identification Assumptions

Estimating the conditional average treatment effect (CATE) relies on five key assumptions:

- **(A1) Consistency**: The observed outcome equals the potential outcome, $T_i = T_i(W_i)$.
- **(A2) Ignorability**: Potential outcomes are independent of treatment assignment given covariates.
- **(A3) Positivity**: Treatment probability is bounded away from 0 and 1 for all covariate values.
- **(A4) Ignorable censoring**: Censoring time is independent of event time given covariates and treatment.
- **(A5) Censoring positivity**: The censoring probability is not equal to 1.

In practice, these assumptions are frequently violated—unobserved prognostic factors undermine ignorability, treatment guidelines undermine positivity, and prognosis-related dropout induces informative censoring. The core objective of SurvHTE-Bench is to measure estimator behavior under these violations.

## Method

### Benchmark Design: A Three-Tier Data Structure

#### 1. Synthetic Data (40 Datasets)

Systematically varied along two orthogonal axes:

**8 causal configurations** (treatment assignment × assumption violation combinations):

| Configuration | Randomized | Ignorability | Positivity | Ignorable Censoring |
|------|:---:|:---:|:---:|:---:|
| RCT-50 | ✓ | ✓ | ✓ | ✓ |
| RCT-5 | ✓ | ✓ | ✓ | ✓ |
| OBS-CPS | ✗ | ✓ | ✓ | ✓ |
| OBS-UConf | ✗ | ✗ | ✓ | ✓ |
| OBS-NoPos | ✗ | ✓ | ✗ | ✓ |
| OBS-CPS-InfC | ✗ | ✓ | ✓ | ✗ |
| OBS-UConf-InfC | ✗ | ✗ | ✓ | ✗ |
| OBS-NoPos-InfC | ✗ | ✓ | ✗ | ✗ |

**5 survival scenarios** (event time distribution × censoring rate):

| Scenario | Survival Time Distribution | Censoring Rate |
|------|:---:|:---:|
| A | Cox | Low (<30%) |
| B | AFT | Low (<30%) |
| C | Poisson | Medium (30–70%) |
| D | AFT | High (>70%) |
| E | Poisson | High (>70%) |

The 8 × 5 = 40 synthetic datasets each contain 50,000 samples with 5-dimensional uniformly distributed covariates and binary treatment. Both potential outcomes $T_i(0)$ and $T_i(1)$ are observed, ensuring CATE ground truth availability.

#### 2. Semi-Synthetic Data (10 Datasets)

Real covariates are combined with simulated treatments and outcomes:

- **ACTG semi-synthetic** (1 dataset): 23-dimensional covariates from an HIV clinical trial with moderate censoring (51%).
- **MIMIC semi-synthetic** (9 datasets): 36-dimensional covariates from the MIMIC-IV ICU database, spanning 53%–88% censoring, with covariate-dependent treatment assignment and nonlinear interaction mechanisms.

#### 3. Real-World Data (2 Datasets)

- **Twins dataset**: Twin birth data with known ground truth (11,400 twin pairs, censoring rate 84.8%).
- **ACTG 175 dataset**: HIV antiretroviral treatment clinical trial (2,139 patients, baseline censoring 13.7%, artificially increased to >90%).

### Method Taxonomy: Three Families, 53 Variants

**Family 1: Outcome Imputation Methods (42 variants)**

Censored event times are first handled via an imputation algorithm, followed by a standard CATE estimator:
- Imputation algorithms: Pseudo-obs, Margin, IPCW-T
- CATE estimators: S-/T-/X-/DR-Learner (each paired with Lasso/Random Forest/XGBoost) + Double-ML + Causal Forest

**Family 2: Direct Survival Causal Methods (2 methods)**

Causal inference is extended directly to time-to-event outcomes:
- SurvITE: Neural network-based balanced representation learning
- Causal Survival Forests: Extension of generalized random forests to survival data

**Family 3: Survival Meta-Learners (9 variants)**

Base learners within meta-learner frameworks are replaced with survival models:
- Meta-learner types: S-/T-/Matching-Learner
- Survival base models: Random Survival Forests, DeepSurv, DeepHit

### Evaluation Metrics

- **CATE RMSE**: Root mean squared error of individual-level treatment effect estimates.
- **ATE Bias**: Systematic bias in population-level average treatment effect estimates.
- **Auxiliary metrics**: Imputation MAE, regression/survival model fit (C-index, AUC).

## Key Experimental Results

### Overall Ranking on Synthetic Data (Borda Count)

Methods are ranked by CATE RMSE on each dataset; results are averaged across 40 datasets × 10 random splits:

| Rank | Method | Mean Rank | Family |
|:---:|------|:---:|:---:|
| 1 | S-Learner-Survival (DeepSurv) | 5.17 | Survival meta-learner |
| 2 | Matching-Survival (DeepSurv) | 5.42 | Survival meta-learner |
| 3 | Double-ML + Margin | 6.65 | Outcome imputation |
| — | Causal Survival Forests | 5.10* | Direct survival causal |

*Note: 5.10 is the family-level rank (best variant per family selected from 11 families).

Family-level rankings (best variant per dataset per family): S-Learner-Survival (3.30) > Matching-Survival (3.48) > Double-ML (3.98) > Causal Survival Forests (5.10).

### Effect of Assumption Violations on Performance

| Scenario | Dominant Method Trend | Key Finding |
|------|------|------|
| RCT-50 (ideal conditions) | Outcome imputation methods prevail | Double-ML (3.60) and Causal Forest (5.60) on par with survival meta-learners |
| RCT-5 (severely imbalanced treatment) | Double-ML leads | T-Learner-Survival drops to last (9.00) due to sparse treated samples |
| OBS-UConf (ignorability violated) | Survival meta-learners stable | Survival meta-learners and CSF show consistent ATE bias; imputation methods exhibit increased bias |
| OBS-NoPos (positivity violated) | Double-ML/X-Learner dominant | CSF rank drops sharply, sensitive to deterministic treatment assignment regions |
| Multiple violations | Survival meta-learners regain advantage | Most robust under simultaneous positivity + other assumption violations |
| InfC (informative censoring) | Survival methods consistently lead | All methods degrade; CATE RMSE variance increases markedly |

### Effect of Censoring Rate

| Censoring Level | Best Family | Representative Method |
|------|------|------|
| Low (Scenarios A, B) | Outcome imputation | Double-ML ranks first |
| Medium (Scenario C) | Competitive equilibrium | Families perform comparably |
| High (Scenarios D, E) | Survival meta-learners | S-Learner-Survival (1.6), Matching-Survival (2.4) dominate |

Under Scenario D (high censoring + AFT distribution), ATE bias diverges substantially for nearly all estimators, demonstrating that RMST-based treatment effect estimation under high censoring remains an extremely challenging task.

### Semi-Synthetic Data

CATE RMSE comparison across the MIMIC-ii–v series (censoring 53%–88%), reported as mean ± standard deviation over 10 repetitions:

| Method | ACTG (51%) | MIMIC-v (53%) | MIMIC-ii (88%) |
|------|:---:|:---:|:---:|
| Double-ML | **10.651±0.24** | **7.891±0.05** | 7.954±0.05 |
| S-Learner-Survival | 11.713±0.24 | **7.897±0.04** | **7.921±0.04** |
| Matching-Survival | 12.523±0.29 | 7.912±0.04 | 7.949±0.04 |
| SurvITE | 12.714±0.56 | 7.906±0.07 | **7.931±0.05** |
| CSF | 11.674±0.17 | 7.893±0.04 | 7.963±0.06 |

Key findings: (1) Double-ML is optimal on moderate-dimensional ACTG data; (2) survival methods (SurvITE and S-Learner-Survival) are most stable on high-censoring MIMIC data; (3) RMSE gaps between methods are compressed in real covariate spaces.

### Real-World Data

- **Twins dataset**: S-Learner and DR-Learner (with imputation) and S-Learner-Survival perform best (RMSE ≈ 7.2 days); Double-ML performs worst (inconsistent with synthetic rankings, suggesting dataset-specific patterns).
- **ACTG 175 dataset**: Under artificially high censoring, CSF produces the most stable estimates; survival meta-learners (T-/Matching-Learner) exhibit notable instability.

## Highlights & Insights

- **First survival HTE benchmark**: Fills the gap in HTE evaluation for right-censored survival data and establishes a reproducible, extensible standardized evaluation platform.
- **Systematic method taxonomy**: For the first time, 53 methods are unified under a three-family framework, including several previously unpublished natural extension variants.
- **Comprehensive assumption violation analysis**: Tests not only individual assumption violations but also simultaneous multiple violations, revealing the true robustness boundaries of each method.
- **Practical selection guide**: Provides practitioners with a clear method selection roadmap—Double-ML for low censoring, S-Learner-Survival for high censoring, and survival meta-learners under multiple violations.

## Limitations & Future Work

- Assumption violations are modeled as binary (present/absent), without modeling **gradual violation severity** (e.g., Rosenbaum Γ sensitivity analysis).
- Only **static binary treatments** and fixed baseline covariates are considered; time-varying treatments, instrumental variables, and dynamic covariates are not covered.
- The primary estimand focuses on RMST; although survival probability results are included in the appendix, **clinically common metrics such as median survival time and time-varying hazard ratios** are not addressed.
- The covariate structure of synthetic data (5-dimensional uniform distribution) may **inadequately represent real high-dimensional medical data**.
- Some real-world datasets (MIMIC-IV) require credentialed access, limiting reproducibility.

## Related Work & Insights

- **Non-censored HTE benchmarks**: Shimoni et al. (2018), Crabbé et al. (2022), CausalBench (2024), targeting fully observed outcomes.
- **Survival ATE benchmarks**: Voinot et al. (2025), targeting population-average effects without individual-level heterogeneity.
- **Causal survival forests**: Cui et al. (2023), extending generalized random forests to survival data with limited evaluation scope.
- **SurvITE**: Curth et al. (2021), a neural network approach based on balanced representations.
- **Survival meta-learners**: Bo et al. (2024), Noroozizadeh et al. (2025), adapting meta-learners to survival models.
- **Outcome imputation**: Qi et al. (2023), proposing IPCW-T and related censored time substitution strategies.
- **Doubly debiased machine learning**: Chernozhukov et al. (2018), the Double-ML framework.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ⭐⭐⭐ |
| Validity | ⭐⭐⭐⭐ |
| Significance | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Overall**: ⭐⭐⭐⭐ — As the first comprehensive survival HTE benchmark, the experimental design is rigorous (40 synthetic + 10 semi-synthetic + 2 real datasets, 53 methods), filling an important gap. The core findings—that no single method dominates and that censoring rate and assumption violations determine method selection—carry significant practical value. However, there remains room for improvement in progressive modeling of assumption violations and estimand diversity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ConSurv: Multimodal Continual Learning for Survival Analysis](../../AAAI2026/medical_imaging/consurv_multimodal_continual_learning_for_survival_analysis.md)
- [\[CVPR 2026\] MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation](../../CVPR2026/medical_imaging/medgen-bench_contextually_entangled_benchmark_for_open-ended_multimodal_medical_.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](incentives_in_federated_learning_with_heterogeneous_agents.md)

</div>

<!-- RELATED:END -->
