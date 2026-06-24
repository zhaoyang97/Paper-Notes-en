---
title: >-
  [Paper Note] One Size Fits None: Rethinking Fairness in Medical AI
description: >-
  [ACL 2025][Medical LLM][Fairness] This paper conducts a subpopulation performance analysis across three multimodal medical prediction tasks (ICU mortality, graft failure, and emergency triage), exposing performance disparities among groups that are otherwise masked by aggregated metrics. It advocates for tightly coupling fairness with transparency to promote responsible medical AI deployment through routine subpopulation reporting.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Fairness"
  - "Subpopulation Analysis"
  - "Medical ML"
  - "Multimodal"
  - "Clinical Decision Making"
date: 2026-05-08
content_hash: fae910dcf1cef53f
---

# One Size Fits None: Rethinking Fairness in Medical AI

**Conference**: ACL 2025  
**arXiv**: [2506.14400](https://arxiv.org/abs/2506.14400)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: Fairness, Subpopulation Analysis, Medical ML, Multimodal, Clinical Decision Making

## TL;DR
This paper conducts a subpopulation performance analysis across three multimodal medical prediction tasks (ICU mortality, graft failure, and emergency triage), exposing performance disparities among groups that are otherwise masked by aggregated metrics. It advocates for tightly coupling fairness with transparency to promote responsible medical AI deployment through routine subpopulation reporting.

## Background & Motivation

**Background**: Medical ML models are increasingly utilized to assist clinical decision-making, spanning scenarios such as mortality prediction, organ transplant prognosis, and emergency triage. Typically, these models report a single aggregated metric (such as AUC-ROC) over the entire test set, demonstrating relatively strong performance.

**Limitations of Prior Work**: Real-world medical datasets often suffer from high noise, high rates of missingness, and severe class imbalance. When certain patient subpopulations (e.g., specific ethnicities, genders, or age groups) are over- or under-represented in the training data, model performance on these groups can decline significantly, a disparity that remains hidden under aggregated metrics.

**Key Challenge**: A fundamental conflict exists between pursuing "one-size-fits-all" high overall performance and ensuring equitable performance across diverse subpopulations. Existing approaches largely overlook systematic subpopulation-level evaluations, resulting in unnoticed performance degradation on marginalized groups.

**Goal**: (a) How to systematically reveal subpopulation performance disparities across different medical tasks? (b) How to responsibly utilize biased models under real-world constraints where completely eliminating disparities is infeasible?

**Key Insight**: The authors utilize three real-world clinical datasets covering different countries (US, Germany) and diverse data modalities (text, structured data, and time-series data) to expose hidden performance disparities by slicing the evaluation test set into subpopulations.

**Core Idea**: By making subpopulation performance transparent, fairness transitions from being "invisible" to "manageable", rather than simply demanding the elimination of all disparities.

## Method

### Overall Architecture

Rather than proposing a new model architecture, this paper presents a **subpopulation fairness evaluation paradigm**. The overall workflow is as follows:
- Train predictive models on three multimodal medical datasets individually.
- Evaluate overall performance on a fixed reference test set.
- Slice the test set into subpopulations based on patient clinical features (age, gender, ethnicity, donor type, etc.).
- Compare the performance of each subpopulation with the reference set performance.
- Analyze the underlying causes of the disparities combining both medical and technical perspectives.

### Three Clinical Prediction Tasks

1. **ICU Mortality Prediction (Mortality)**
    - Data Source: MIMIC-III (US ICU data)
    - Input: Demographics + time-series vital signs + admission note text
    - Task: Predict in-hospital mortality 48 hours post-admission (binary classification)
    - Model: Multimodal architecture based on interpolation embedding and time-aware attention, using interleaved self-attention and cross-attention layers to fuse modalities
    - Metrics: AUC-ROC and AUPRC

2. **Graft Failure Prediction (Graft Failure)**
    - Data Source: German Transplant Center
    - Input: Structured data (demographics, comorbidities) + time-series lab results/vital signs + clinical text
    - Task: Predict graft failure within 360 days post-visit (binary classification)
    - Model: Gradient Boosting Regressor capable of handling static, time-series, and text data
    - Metrics: AUC-ROC and AUPRC

3. **Emergency Triage Prediction (Triage)**
    - Data Source: Semi-structured ambulance records from a German emergency department
    - Input: Structured features (vital signs, pain scores, Glasgow Coma Scale) + short text notes
    - Task: Classify patient acuity according to the Manchester Triage System (multiclass classification)
    - Model: Transformer for text processing + feed-forward network to integrate structured features + expert rules to boost recall for high-acuity categories
    - Metrics: Precision / Recall / F1

### Subpopulation Analysis Method

The core method is **test set slicing and comparison**: on a single pre-trained model, the test set is filtered based on patient attributes (e.g., female patients only, Black patients only), and then the performance of each subpopulation is compared with that of the full reference test set. This approach is straightforward, requires no alterations to the model training, and is applicable to auditing deployed systems for fairness.

### Statistical Significance Verification

A **one-sided non-parametric bootstrap hypothesis test** is conducted to verify if the differences among subpopulations are statistically significant, ruling out random fluctuations due to small sample sizes. On the Mortality task, the performance gap between White and Black patients was confirmed to be statistically significant.

## Key Experimental Results

### Main Results: Subpopulation Analysis for ICU Mortality Prediction (Table 1)

| Subpopulation | AUC-ROC | AUPRC |
|---|---|---|
| Reference Set (All) | 0.89 | 0.61 |
| Advanced Age (>75) | 0.86 | 0.59 |
| Male | 0.90 | 0.65 |
| Female | 0.88 | 0.57 |
| White | 0.89 | 0.62 |
| Black | 0.86 | **0.45** |
| Asian| 0.91 | 0.56 |
| Hispanic | 0.97 | 0.77 |

Key Findings: AUPRC is highly sensitive to class imbalance. The AUPRC for Black patients is only 0.45 (substantially below the overall 0.61), and drops further to 0.36 for Black female patients. While AUC-ROC shows minimal divergence, the gap in AUPRC reveals severe recall bias. Significance testing confirms that the White vs. Black performance gap is statistically significant, whereas other comparisons like gender or Hispanic vs. White are not.

### Subpopulation Analysis for Graft Failure Prediction (Table 2)

| Subpopulation | AUC-ROC | AUPRC |
|---|---|---|
| Reference Set | 0.94 | 0.55 |
| Younger Age | 0.96 | 0.72 |
| Advanced Age | 0.93 | 0.51 |
| Male | 0.95 | 0.61 |
| Female | 0.94 | **0.49** |
| Living Donor | 0.98 | 0.70 |
| Deceased Donor | 0.93 | 0.53 |

Key Findings: The AUPRC for female patients (0.49), elderly patients (0.51), and patients who received organs from deceased donors (0.53) are all significantly lower than the reference set baseline (0.55). Medical Explanation: Sarcopenia (low muscle mass) in elderly patients renders creatinine-based eGFR equations unreliable; male overrepresentation combined with the non-use of sex-adjusted eGFR calculation leads to systemic gender biases.

### Subpopulation Analysis for Triage Prediction (Table 3, Selected)

| Subpopulation / Category | Precision | Recall | F1 |
|---|---|---|---|
| Reference Set → Red | 0.21 | 0.86 | 0.34 |
| Children (<18) → Red | 0.30 | 0.78 | 0.44 |
| Advanced Age (>85) → Red | 0.16 | 0.88 | 0.27 |
| Missing Age Info → Red | 0.36 | 0.67 | 0.47 |
| Reference Set → Orange | 0.20 | 0.53 | 0.29 |
| Advanced Age (>85) → Orange | 0.13 | 0.44 | 0.20 |

Key Findings: For elderly patients, precision in high-urgency categories (Red/Orange) drops precipitously. When age information is missing, recall drops across all categories. Notably, approximately 30% of triage labels in the real-world dataset suffer from labeling errors, showing that label noise severely constrains the reliability of such evaluations.

## Highlights & Insights

- **Pragmatic Fairness Perspective**: Shifts from the idealistic target of "eliminating all disparities" to advocating for transparency to make biases "manageable," fostering responsible application under real-world constraints.
- **Multitask & Multimodal Coverage**: Spans three distinct clinical domains (ICU, transplantation, and emergency triage) utilizing heterogeneous model architectures (attention networks, gradient boosting, Transformer hybrids), enhancing the generalizability of the findings.
- **Binary Medical-Technical Analysis**: Beyond explaining bias via data distribution shifts, the study incorporates medical domain knowledge (such as the impact of creatinine on eGFR calculation and ~30% triage label noise) to formulate deeper medical diagnoses of model behavior.
- **Policy Alignment**: Connects subpopulation reporting with high-risk AI documentation requirements under the EU AI Act, advocating for a model documentation standard akin to "drug package inserts".

## Limitations & Future Work

- Simple subpopulation division (slicing on only a single demographic feature) without systematically exploring **intersectionality analysis** (e.g., the combined effects of race + sex + age).
- Statistical significance was not quantified across all datasets, only a bootstrap significance test was conducted for the Mortality task.
- Two of the datasets originate from Germany, and one represents a single hospital in the USA, leading to **limited geographical generalizability**.
- The issue of label noise (especially ~30% error rate in the triage task) fundamentally bounds the evaluation validity, yet no mitigation strategy is proposed.
- The study remains primarily a **diagnostic analysis** (identifying issues), without implementing specific **bias mitigation methodologies** (e.g., fairness-constrained training, data augmentation, subpopulation-specific modeling).

## Related Work & Insights

- Compared to the fairness survey by Mehrabi et al. (2021), this study emphasizes **empirical analysis in the medical domain** over abstract generalized frameworks.
- Unlike the FairMedFM benchmark by Jin et al. (2024), this work expands to **multimodality** (text + structured + time-series) instead of focusing exclusively on medical imaging foundation models.
- In contrast to the fairness-aware optimization objectives proposed by Sivarajkumar et al. (2023), this study refrains from altering model training, focusing instead on the **evaluation and reporting** stages.
- Adapting the "information leaflet" concept from Samhammer et al. (2023), the paper proposes that medical ML models should compile extensive documentation of subpopulation performance.

## Insights & Connections

- The core concept that "transparency is the prerequisite of fairness" is highly generic and valuable for all high-risk AI applications.
- The methodology of slicing the evaluation set is simple but effective, and can be directly transferred to audit other multimodal prediction systems.
- The interference of label noise on fairness evaluation is worthy of in-depth study—how to reasonably evaluate subpopulation fairness when labels are unreliable remains an open question.

## Rating
- Novelty: ⭐⭐⭐ (The methodology itself is not novel, but the perspective and systematic analysis are valuable)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive coverage across three tasks, including medical analysis and statistical testing)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, sound and comprehensive logic)
- Value: ⭐⭐⭐⭐ (Highly pragmatic for promoting real-world fairness in medical AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](../../ACL2026/medical_nlp/beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2025\] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](clinical_coding_eight_recommendations.md)
- [\[ACL 2026\] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](../../ACL2026/medical_nlp/inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2025\] A Retrieval-Based Approach to Medical Procedure Matching in Romanian](a_retrieval-based_approach_to_medical_procedure_matching_in_romanian.md)

</div>

<!-- RELATED:END -->
