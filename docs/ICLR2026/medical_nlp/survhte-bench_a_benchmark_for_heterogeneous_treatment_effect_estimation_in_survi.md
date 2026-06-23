---
title: >-
  [Paper Note] SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis
description: >-
  [ICLR 2026][Medical NLP][heterogeneous treatment effects] Proposes SurvHTE-Bench, the first comprehensive benchmark for Heterogeneous Treatment Effect (HTE) estimation for right-censored survival data. It includes 40 synthetic datasets, 10 semi-synthetic datasets, and 2 real datasets to systematically evaluate 53 estimation methods under different causal assumption violations
tags:
  - ICLR 2026
  - Medical NLP
  - heterogeneous treatment effects
  - survival analysis
  - right-censored data
  - causal inference
  - benchmark
  - CATE
date: 2026-05-08
content_hash: 70e85f1f672bf56b
---
# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

**Conference**: ICLR2026  
**arXiv**: [2603.05483](https://arxiv.org/abs/2603.05483)  
**Code**: [GitHub](https://github.com/Shahriarnz14/SurvHTE-Bench)  
**Area**: Medical Imaging  
**Keywords**: heterogeneous treatment effects, survival analysis, right-censored data, causal inference, benchmark, CATE, meta-learners, precision medicine

## TL;DR

Proposes SurvHTE-Bench, the first comprehensive benchmark for Heterogeneous Treatment Effect (HTE) estimation for right-censored survival data. It includes 40 synthetic datasets, 10 semi-synthetic datasets, and 2 real datasets to systematically evaluate 53 estimation methods under different causal assumption violations and censoring levels. The study finds that no single method dominates, but survival meta-learners (especially S-Learner-Survival and Matching-Survival) are the most robust in high censoring and assumption violation scenarios.

## Background & Motivation

### Definition

Heterogeneous Treatment Effect (HTE) estimation aims to quantify the differential efficacy of the same treatment for different individuals, which is central to precision medicine and personalized policy formulation. In survival analysis scenarios, observation times are affected by right-censoring (i.e., some individuals do not experience the target event before the study ends), making HTE estimation face a triple challenge:

**Counterfactual Unobservability**: Only one outcome (treated or untreated) can be observed for each individual.

**Confounding Factors**: In observational studies, treatment assignment is influenced by covariates.

**Censoring Mechanism**: Censoring may be related to event time (informative censoring), violating standard assumptions.

### Limitations of Prior Work

Despite the numerous survival HTE estimation methods proposed recently (Causal Survival Forests, survival meta-learners, outcome imputation methods, etc.), evaluation practices remain highly fragmented:

- Each study uses custom simulated data with different assumption settings and censoring levels.
- Lack of a unified benchmark with known ground truth.
- Fair comparisons between different methods are impossible.
- The robustness of estimators under **simultaneous multiple assumption violations** is unknown.

Existing causal inference benchmarks (e.g., CausalBench) target fully observed outcomes, while survival ATE benchmarks do not cover individual-level heterogeneous effects. **The survival HTE estimation field has lacked a standardized benchmark until now**—this is the core motivation of this paper.

### Causal Identification Assumptions

Estimating the Conditional Average Treatment Effect (CATE) relies on five key assumptions:

- **(A1) Consistency**: The observed outcome equals the potential outcome $T_i = T_i(W_i)$.
- **(A2) Ignorability**: Potential outcomes are independent of treatment assignment given covariates.
- **(A3) Positivity**: The probability of treatment is bounded away from 0 and 1 for all covariate levels.
- **(A4) Ignorable Censoring**: Censoring time is independent of event time given covariates and treatment.
- **(A5) Censoring Positivity**: The probability of censoring is not 1.

In practice, these assumptions are often violated—unobserved prognostic factors break ignorability, treatment guidelines break positivity, and prognosis-related dropouts lead to informative censoring. The core goal of SurvHTE-Bench is to measure estimator behavior under these violations.

## Method

### Overall Architecture

SurvHTE-Bench aims to answer a seemingly simple but systematically unverified question: which HTE estimators remain reliable when survival data includes right-censoring and multiple causal assumption violations? The benchmark is structured across three layers: "Data → Method → Metrics". First, a set of **datasets with known CATE ground truth** is created (ranging from pure synthetic to semi-synthetic to real), covering combinations of ideal RCTs to multiple assumption violations, and low to extreme (>90%) censoring. Next, 53 existing estimation methods are unified into three main families for head-to-head competition. Finally, unified CATE RMSE and ATE bias are used for scoring, with Borda Count rankings across datasets. This maps the performance of any method to a coordinate system of "which censoring level and which assumption violations," rather than disparate claims.

### Key Designs

**1. Three-tier Data Gradient: From synthetic to real, shifting "calculability" of ground truth and "realism" of data.**

The hardest part of the benchmark is balancing the need for CATE ground truth with the realism of clinical data. This is addressed via three tiers:

The first tier, **Synthetic Data**, is the only place where precise CATE ground truth is available because both potential outcomes $T_i(0)$ and $T_i(1)$ are generated by known mechanisms. Two axes are decoupled: one manages **treatment assignment and causal assumptions**, degrading from RCT-50 / RCT-5 to violations of ignorability (UConf), positivity (NoPos), and informative censoring (InfC), totaling 8 configurations. The other axis manages **event time distribution and censoring rate**, using Cox / AFT / Poisson distributions with low (<30%) / medium (30–70%) / high (>70%) censoring, totaling 5 scenarios.

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

| Scenario | Survival Distribution | Censoring Rate |
|------|:---:|:---:|
| A | Cox | Low (<30%) |
| B | AFT | Low (<30%) |
| C | Poisson | Medium (30-70%) |
| D | AFT | High (>70%) |
| E | Poisson | High (>70%) |

There are $8 \times 5 = 40$ synthetic datasets, each with 50,000 samples, 5-dimensional uniform covariates, and binary treatment. Decoupling the axes allows for clean attribution of method failure to either assumption violations or censoring levels.

The second tier, **Semi-synthetic Data**, replaces "clean" synthetic covariates with real clinical covariates while simulating treatment assignment and survival outcomes. This preserves ground truth while reintroducing complex treatment dependencies and non-linear interactions found in real data. It includes 10 datasets: 1 ACTG semi-synthetic (23 covariates, 51% censoring) and 9 MIMIC semi-synthetics (36 covariates, 53%–88% censoring).

The third tier, **Real Data**, moves closer to practical application. Twins (twin birth data, 11,400 pairs, 84.8% censoring) provides approximate counterfactuals via identical twins. ACTG 175 (HIV trial, 2,139 patients, 13.7% baseline censoring) is artificially censored to >90% to stress-test extreme cases. These tiers verify if synthetic rankings transfer to real scenarios.

**2. Unified Classification of 53 Methods: Organizing sporadic estimators into three families and filling in unpublished extensions.**

Family I: **Outcome Imputation Methods** (42 variants): Use Pseudo-obs / Margin / IPCW-T to fill in censored times as usable outcomes, then apply standard CATE estimators (S-/T-/X-/DR-Learner with Lasso / Random Forest / XGBoost, plus Double-ML and Causal Forest). Family II: **Direct Survival Causal Methods** (2 methods): Perform inference directly on time-to-event outcomes without imputation, including SurvITE based on balanced representation learning and Causal Survival Forests. Family III: **Survival Meta-learners** (9 variants): Replace base learners in meta-learners with survival models (S-/T-/Matching-learner with Random Survival Forests / DeepSurv / DeepHit), including several previously unpublished natural extensions.

**3. Unified Evaluation Metrics: Primary individual-level CATE RMSE, secondary population-level ATE Bias, and Borda Count aggregation.**

The primary metric is **CATE RMSE**, measuring individual-level accuracy. **ATE Bias** measures systematic population-level shifts. Diagnostic metrics like imputation MAE and model fit (C-index, AUC) help locate error sources. Methods are ranked by CATE RMSE on each dataset, and **Borda Count** (sum of ranks across datasets) is used for final aggregation to prevent any single dataset from dominating the conclusions.

## Key Experimental Results

### Synthetic Data Overall Ranking (Borda Count)

Ranked by CATE RMSE across 40 datasets × 10 random splits:

| Rank | Method | Avg Rank | Method Family |
|:---:|------|:---:|:---:|
| 1 | S-Learner-Survival (DeepSurv) | 5.17 | Survival Meta-learner |
| 2 | Matching-Survival (DeepSurv) | 5.42 | Survival Meta-learner |
| 3 | Double-ML + Margin | 6.65 | Outcome Imputation |
| — | Causal Survival Forests | 5.10* | Direct Survival Causal |

*Note: 5.10 is the rank at the method family level (selecting the best variant among 11 families).

Method family rankings (best variant per dataset): S-Learner-Survival (3.30) > Matching-Survival (3.48) > Double-ML (3.98) > Causal Survival Forests (5.10).

### Impact of Hypothesis Violations

| Scenario | Best Method Trend | Key Findings |
|------|------|------|
| RCT-50 (Ideal) | Outcome Imputation dominates | Double-ML (3.60) and Causal Forest (5.60) match survival meta-learners |
| RCT-5 (Unbalanced) | Double-ML leads | T-Learner-Survival drops to last (9.00) due to sparse treatment group samples |
| OBS-UConf (Ignorability) | Survival Meta-learner stable | ATE bias for survival meta-learners and CSF is consistent; imputation bias grows |
| OBS-NoPos (Positivity) | Double-ML/X-Learner strong | CSF rank drops significantly; sensitive to deterministic treatment regions |
| Multiple Violations | Survival Meta-learner regains lead | Survival meta-learners are most robust under joint positivity + other violations |
| InfC (Informative) | Survival methods lead | All methods degrade; CATE RMSE variance increases significantly |

### Impact of Censoring Rate

| Censoring Level | Best Family | Representative Method |
|------|------|------|
| Low (Scenes A, B) | Outcome Imputation | Double-ML ranks first |
| Medium (Scene C) | Balanced competition | Families are close |
| High (Scenes D, E) | Survival Meta-learner | S-Learner-Survival (1.6), Matching-Survival (2.4) dominate others |

In Scenario D (High Censoring + AFT), almost all estimators' ATE bias diverges sharply, indicating that RMST-based effect estimation remains highly challenging under high censoring.

### Semi-synthetic Data

CATE RMSE comparison for MIMIC-ii–v series (53%–88% censoring) (mean ± SD of 10 repeats):

| Method | ACTG (51%) | MIMIC-v (53%) | MIMIC-ii (88%) |
|------|:---:|:---:|:---:|
| Double-ML | **10.651±0.24** | **7.891±0.05** | 7.954±0.05 |
| S-Learner-Survival | 11.713±0.24 | **7.897±0.04** | **7.921±0.04** |
| Matching-Survival | 12.523±0.29 | 7.912±0.04 | 7.949±0.04 |
| SurvITE | 12.714±0.56 | 7.906±0.07 | **7.931±0.05** |
| CSF | 11.674±0.17 | 7.893±0.04 | 7.963±0.06 |

Key Findings: (1) Double-ML is optimal in medium-dimensional ACTG data; (2) Survival methods (SurvITE and S-Learner-Survival) are most stable in high-censoring MIMIC data; (3) RMSE gaps between methods narrow in real covariate spaces.

### Main Results

- **Twins Dataset**: S-Learner and DR-Learner (with imputation) and S-Learner-Survival performed best (RMSE ≈ 7.2 days). Double-ML performed worst, suggesting data-specific patterns inconsistent with synthetic results.
- **ACTG 175 Dataset**: Under artificial high censoring, CSF is the most stable. Survival meta-learners (T-/Matching-learner) show significant instability.

## Highlights & Insights

- **First Survival HTE Benchmark**: Fills the gap in HTE evaluation for right-censored survival data, establishing a reproducible and standardized platform.
- **Systematic Method Classification**: Unifies 53 methods into three families, including several previously unpublished natural extensions.
- **Comprehensive Assumption Violation Analysis**: Tests both single and multiple simultaneous violations, revealing the real boundaries of method robustness.
- **Practical Selection Guide**: Provides a roadmap—use Double-ML for low censoring, S-Learner-Survival for high censoring, and survival meta-learners for multiple violations.

## Limitations & Future Work

- Assumption violations are binary (present/absent) without modeling **progressive violation severity** (e.g., Rosenbaum Γ sensitivity analysis).
- Only considers **static binary treatments** and fixed baseline covariates; time-varying treatments, instrumental variables, and dynamic covariates are not covered.
- Target estimands focus on RMST; though survival probabilities are in the appendix, **common clinical metrics like median survival time or time-varying hazard ratios** are missing.
- Synthetic covariate structures (5D uniform) may **not fully represent high-dimensional medical data**.
- Some real datasets (MIMIC-IV) require credentialed access, impacting reproducibility.

## Related Work & Insights

- **Non-censored HTE Benchmarks**: Shimoni et al. (2018), Crabbé et al. (2022), CausalBench (2024) target fully observed outcomes.
- **Survival ATE Benchmarks**: Voinot et al. (2025) focus on population average effects rather than individual heterogeneity.
- **Causal Survival Forests**: Cui et al. (2023) extended generalized random forests to survival data but had limited evaluation scope.
- **SurvITE**: Curth et al. (2021) neural network method based on balanced representations.
- **Survival meta-learners**: Bo et al. (2024), Noroozizadeh et al. (2025) adapted meta-learners for survival models.
- **Outcome Imputation**: Qi et al. (2023) proposed IPCW-T for censoring time replacement strategies.
- **Double Machine Learning**: Chernozhukov et al. (2018) Double-ML framework.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Significance | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Total Rating**: ⭐⭐⭐⭐ — As the first comprehensive survival HTE benchmark, the experimental design is rigorous (40 synthetic + 10 semi-synthetic + 2 real datasets, 53 methods), filling a major gap. Core findings (no single best method, choice depends on censoring and violations) are of high practical value. Future improvements in progressive violation modeling and estimand diversity are possible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)
- [\[ICLR 2026\] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)
- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](../../ACL2026/medical_nlp/heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)
- [\[NeurIPS 2025\] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](../../NeurIPS2025/medical_nlp/healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_nlp/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->
