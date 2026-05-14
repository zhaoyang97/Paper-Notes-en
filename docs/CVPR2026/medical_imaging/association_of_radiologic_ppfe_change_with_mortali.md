---
title: >-
  [Paper Note] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR 2026][Medical Imaging][PPFE] Across the NLST (n=7,980) and SUMMIT (n=8,561) large-scale lung cancer screening cohorts, deep learning-based automatic segmentation is used to quantify longitudinal PPFE changes (dPPFE…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "PPFE"
  - "lung cancer screening"
  - "low-dose CT"
  - "longitudinal quantitative analysis"
  - "imaging biomarker"
date: 2026-05-08
content_hash: 4ffe55635ffbfd8a
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**Conference**: CVPR 2026
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)
**Code**: None
**Area**: Medical Image Analysis
**Keywords**: PPFE, lung cancer screening, low-dose CT, longitudinal quantitative analysis, imaging biomarker

## TL;DR

Across two independent large-scale lung cancer screening cohorts, deep learning-based automatic segmentation is employed to quantify longitudinal PPFE changes, providing the first validation of the independent prognostic value of PPFE progression in a screening population.

## Background & Motivation

**Pleuroparenchymal fibroelastosis (PPFE)** is a rare fibrotic lung abnormality predominantly affecting the upper lobes, and has been associated with increased mortality in patients with established interstitial lung disease (e.g., IPF, HP). However, prior studies focused on patient populations with confirmed fibrotic diagnoses, who inherently carry a poor prognosis.

**The clinical significance of longitudinal PPFE progression in lung cancer screening populations—typically asymptomatic or mildly symptomatic—remains unclear.** This represents an important knowledge gap: the NHS is expected to enroll one million individuals in annual CT screening by 2028. If existing low-dose CT scans from screening programs could simultaneously identify individuals at high PPFE-related risk, the additional clinical value of screening would be substantially enhanced.

**The root cause of the challenge lies in** the infeasibility of manual or semi-automatic PPFE assessment at population scale, and the absence of any prior systematic validation of PPFE progression's prognostic significance in screening cohorts using fully automated quantitative methods. The paper's starting point is to apply deep learning-based fully automated segmentation combined with annualized change computation, with dual validation across two independent cohorts.

## Method

### Overall Architecture

A retrospective longitudinal study design: a deep learning model based on the nnU-Net architecture automatically segments upper-lung PPFE lesions on baseline and follow-up CT scans, computes the annualized volumetric change (ΔPPFE), dichotomizes subjects into progressive/non-progressive groups using a distribution-derived threshold, and employs Cox proportional hazards models to assess independent association with all-cause mortality.

### Key Designs

1. **Automated PPFE Segmentation Model**:

    - Function: Automatically segments PPFE lesion volume in the upper lung on low-dose CT.
    - Mechanism: Based on the nnU-Net architecture, trained with 5-fold cross-validation on 100 manually annotated cases from SUMMIT, achieving Dice = 0.91. Segmentation is restricted to the region above the carina, with the apical 5 mm excluded to avoid confusion with benign pleural thickening.
    - Design Motivation: Manual segmentation is infeasible in cohorts of tens of thousands; restricting the segmentation region reduces false positives.

2. **ΔPPFE Progression Threshold Definition**:

    - Function: Binarizes the continuous annualized PPFE change into progressive/non-progressive categories.
    - Mechanism: One-half of the standard deviation of baseline PPFE volume in NLST (0.41 cm³/year) is used as the threshold, consistent with established biomarker methodology for moderate effect sizes.
    - Design Motivation: Derived in NLST and directly applied to SUMMIT to assess cross-cohort generalizability; 5.4% of NLST subjects and 1.5% of SUMMIT subjects are classified as progressive.

3. **Multivariable Cox Regression Analysis**:

    - Function: Evaluates the independent association between ΔPPFE and all-cause mortality.
    - Mechanism: Adjusted for age, sex, smoking history, height, baseline PPFE volume, and interaction terms; SUMMIT analyses additionally adjust for FVC% predicted and visual ILA score.
    - Design Motivation: Thorough confounder adjustment isolates the independent prognostic contribution of PPFE progression.

### Loss & Training

The segmentation model uses a composite Dice + cross-entropy loss with SGD optimization over 1,000 epochs. The statistical analysis framework includes Cox PH models (survival analysis), negative binomial GLM (hospitalization counts), and ordinal logistic regression (mMRC dyspnea score).

## Key Experimental Results

### Main Results

| Cohort | Adjustment | HR | 95% CI | p-value |
|--------|-----------|-----|--------|---------|
| NLST | Multivariable | 1.25 | 1.01–1.56 | 0.042 |
| SUMMIT | Multivariable | 3.14 | 1.66–5.97 | <0.001 |
| SUMMIT | +FVC+ILA | 2.55 | 1.34–4.85 | 0.004 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Baseline PPFE alone | Inconsistent | Weak and non-significant in NLST; significant in SUMMIT but weaker than ΔPPFE |
| After adjusting for ILA and FVC | HR remains significant | ΔPPFE represents a biological process independent of classical UIP-pattern fibrosis |
| Baseline-CT-only subgroup | IRR = 1.24 | Baseline PPFE associated with hospitalizations, but not with mMRC or medication use |

### Key Findings

- Kaplan–Meier curves in both cohorts demonstrate significantly lower survival in the progressive PPFE group (log-rank p < 0.001).
- ΔPPFE shows no association with cardiovascular events (MACE5), reflecting respiratory-specific rather than systemic risk.
- Progressive PPFE is strongly associated with higher respiratory hospitalization rates (IRR = 2.79) and greater use of corticosteroids/antibiotics (IRR = 1.55).

## Highlights & Insights

- Dual validation across two independent large-scale screening cohorts yields consistent results; automated quantification makes large-cohort epidemiological research of rare conditions feasible. The paradigm of "leveraging existing screening CTs to identify additional high-risk individuals" has direct public health value.

## Limitations & Future Work

- Retrospective observational design precludes causal inference.
- Low-dose CT resolution is inferior to diagnostic HRCT, potentially underestimating subtle PPFE.
- Anatomical overlap may exist between PPFE and ILA.
- The progression threshold requires external validation in non-screening populations.

## Related Work & Insights

- **vs. Gudmundsson et al. (2023)**: Validated the association between PPFE progression and mortality in IPF/HP patients; the present study is the first to extend this finding to an asymptomatic screening population.
- **vs. Jacob et al. (2018)**: Employed manual/semi-automatic PPFE assessment; the present study achieves fully automated deep learning-based analysis at a scale of tens of thousands.

## Rating

- Novelty: ⭐⭐⭐ No methodological innovation; contribution lies in the clinical finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two independent large cohorts, multiple endpoints, complete subgroup analyses.
- Writing Quality: ⭐⭐⭐⭐ Clinical paper style with clear structure.
- Value: ⭐⭐⭐⭐ Direct clinical guidance for lung cancer screening practice.

---
title: >-
  [Paper Note] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR 2026][Medical Imaging][PPFE] In the NLST (n=7980) and SUMMIT (n=8561) large-scale lung cancer screening cohorts, deep learning-based automatic segmentation is used to quantify longitudinal PPFE changes (dPPFE) on low-dose CT, validating their independent association with all-cause mortality (HR=1.25/3.14) and respiratory morbidity.
tags:
  - CVPR 2026
  - Medical Imaging
  - PPFE
  - Lung Cancer Screening
  - Low-dose CT
  - Longitudinal Quantitative Analysis
  - Imaging Biomarker
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**Conference**: CVPR 2026
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)
**Code**: None
**Area**: Medical Image Analysis
**Keywords**: PPFE, lung cancer screening, low-dose CT, longitudinal quantitative analysis, imaging biomarker

## TL;DR
Across the NLST (n=7,980) and SUMMIT (n=8,561) large-scale lung cancer screening cohorts, deep learning-based automatic segmentation is used to quantify longitudinal PPFE changes (dPPFE) on low-dose CT, validating their independent association with all-cause mortality (HR=1.25/3.14) and respiratory morbidity.

## Background & Motivation

Pleuroparenchymal fibroelastosis (PPFE) is a rare fibrotic lung abnormality predominantly affecting the upper lobes, and has been shown to be associated with increased mortality in established interstitial lung diseases such as IPF and HP. Prior research has focused largely on PPFE in populations with confirmed fibrotic diagnoses; however, the clinical significance of longitudinal PPFE progression in lung cancer screening populations—typically asymptomatic or mildly symptomatic—remains unclear. With the NHS expected to enroll one million individuals in annual CT screening by 2028, understanding the implications of PPFE in screening populations is increasingly important.

## Core Problem
In asymptomatic lung cancer screening populations, can automatically quantified longitudinal PPFE changes on low-dose CT independently predict mortality and adverse respiratory outcomes?

## Method

### Overall Architecture
A retrospective longitudinal study: a deep learning model based on nnU-Net segments PPFE lesions on baseline and follow-up CT scans, computes the annualized volumetric change (dPPFE), and evaluates its independent association with mortality via Cox proportional hazards models. Two independent cohorts (NLST and SUMMIT) provide mutual validation.

### Key Designs
1. **Automated PPFE Segmentation Model**: Based on the nnU-Net architecture, trained with 5-fold cross-validation on 100 manually annotated cases from SUMMIT, achieving Dice = 0.91. Segmentation is restricted to the region above the carina, with the apical 5 mm excluded to avoid confusion with benign pleural thickening.
2. **dPPFE Progression Threshold**: One-half of the standard deviation of baseline PPFE volume in NLST (0.41 cm$^3$/year) is used as the threshold, consistent with established biomarker methodology. This identifies 5.4% of NLST subjects and 1.5% of SUMMIT subjects as progressive. The threshold is derived in NLST and applied directly to SUMMIT.
3. **Multivariable Cox Regression**: Adjusted for age, sex, smoking history (pack-years), height, baseline PPFE volume, and interaction terms; SUMMIT analyses additionally adjust for FVC% predicted and visual ILA score. Negative binomial GLM (hospitalization counts) and ordinal logistic regression (mMRC score) are used for secondary endpoints.

### Loss & Training
- Segmentation model: Composite Dice + cross-entropy loss, SGD optimizer, 1,000 epochs.
- Statistical analysis: Cox PH model (survival), negative binomial GLM (hospitalization rate), ordinal logistic regression (dyspnea score).

## Key Experimental Results

| Cohort | dPPFE HR | 95% CI | p-value |
|--------|---------|--------|---------|
| NLST (multivariable) | 1.25 | 1.01–1.56 | 0.042 |
| SUMMIT (multivariable) | 3.14 | 1.66–5.97 | <0.001 |
| SUMMIT (+FVC+ILA) | 2.55 | 1.34–4.85 | 0.004 |

| Outcome (SUMMIT) | IRR/OR | p-value |
|------------------|--------|---------|
| Respiratory hospitalization | IRR = 2.79 | <0.001 |
| Corticosteroid/antibiotic use | IRR = 1.55 | 0.011 |
| mMRC dyspnea score | OR = 1.40 | 0.055 |

- Kaplan–Meier curves in both cohorts show significantly lower survival in the progressive PPFE group (log-rank p < 0.001).
- dPPFE shows no association with cardiovascular events (MACE5), indicating respiratory-specific rather than systemic risk.
- Baseline-CT-only subgroup: baseline PPFE is associated with hospitalizations (IRR = 1.24) but not with mMRC or medication use.

### Ablation Study Highlights
- Baseline PPFE alone has weak and inconsistent prognostic value; dPPFE provides incremental information.
- dPPFE remains independent after adjusting for ILA and FVC, representing a biological process distinct from classical UIP-pattern fibrosis.
- Exclusion of subjects without follow-up may introduce survival bias; secondary analyses confirm that baseline PPFE also carries partial risk information.

## Highlights & Insights
- Dual validation across two independent large-scale real-world screening cohorts yields consistent and robust results.
- Automated quantification combined with longitudinal analysis makes large-cohort epidemiological research of rare conditions feasible.
- Direct association with clinically actionable outcomes (hospitalization rate, medication use, dyspnea score) is demonstrated with a rigorous methodological framework.

## Limitations & Future Work
- Retrospective observational design precludes causal inference.
- Low-dose CT resolution is inferior to diagnostic HRCT, potentially underestimating subtle PPFE.
- Anatomical overlap may exist between PPFE and ILA, though statistical adjustment preserves independence.
- The progression threshold requires external validation in non-screening populations.

## Related Work & Insights
- **vs. Gudmundsson et al. (2023)**: Validated the association between PPFE progression and mortality in IPF/HP patients; the present study is the first to extend this to an asymptomatic screening population.
- **vs. Jacob et al. (2018)**: Employed manual/semi-automatic PPFE prognostic assessment; the present study achieves fully automated deep learning-based analysis at large-cohort scale.
- The paradigm of automated quantitative imaging biomarkers combined with longitudinal analysis is generalizable to other incidental screening findings (e.g., coronary artery calcium, osteoporosis).

## Rating
- Novelty: ⭐⭐⭐ No methodological innovation (nnU-Net + Cox); contribution lies in the clinical finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two independent large cohorts, multiple endpoints, complete subgroup analyses.
- Writing Quality: ⭐⭐⭐⭐ Clinical paper style with clear structure.
- Value: ⭐⭐⭐⭐ Direct guidance for lung cancer screening clinical practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[CVPR 2026\] Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)
- [\[CVPR 2026\] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)
- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/medical_imaging/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)

</div>

<!-- RELATED:END -->
