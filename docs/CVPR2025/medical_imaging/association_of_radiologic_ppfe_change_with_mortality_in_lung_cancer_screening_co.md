---
title: >-
  [Paper Note] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR2025][Medical Imaging][PPFE] This study validates that the progression of PPFE (pleuroparenchymal fibroelastosis) automatically quantified by deep learning is independently associated with all-cause mortality across two large-scale lung cancer screening cohorts (NLST: 7,980 cases; SUMMIT: 8,561 cases). It proposes that longitudinal changes in PPFE can serve as an imaging biomarker to identify individuals at high risk for respiratory morbidity in screening populations.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "PPFE"
  - "Lung Cancer Screening"
  - "CT Imaging Biomarker"
  - "Mortality"
  - "Longitudinal Analysis"
date: 2026-05-08
content_hash: f59c64cbda4df846
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**Conference**: CVPR2025  
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)  
**Code**: Not publicly available  
**Area**: Medical Imaging  
**Keywords**: PPFE, Lung Cancer Screening, CT Imaging Biomarker, Mortality, Longitudinal Analysis

## TL;DR

This study validates that the progression of PPFE (pleuroparenchymal fibroelastosis) automatically quantified by deep learning is independently associated with all-cause mortality across two large-scale lung cancer screening cohorts (NLST: 7,980 cases; SUMMIT: 8,561 cases). It proposes that longitudinal changes in PPFE can serve as an imaging biomarker to identify individuals at high risk for respiratory morbidity in screening populations.

## Background & Motivation

**Clinical Significance of PPFE**: PPFE is an upper-lobe-predominant fibrotic lung disorder associated with increased mortality in patients with interstitial lung disease, but its prognostic significance in lung cancer screening populations remains unclear.

**Large-Scale Implementation of Lung Cancer Screening**: The UK NHS expects 1 million individuals to undergo annual CT screening by 2028, making it clinically vital to understand the impact of PPFE on screening populations.

**Lack of Longitudinal PPFE Quantification**: Previous studies have focused primarily on cross-sectional baseline analyses, and the relationship between longitudinal PPFE progression and mortality has only been reported in patients with IPF/HP.

**Advantages of Automated Quantification**: Visual assessment lacks sensitivity for PPFE, whereas automated deep learning segmentation can provide continuous quantitative metrics.

**Research Value of Smoker-Enriched Populations**: Both NLST and SUMMIT enrolled heavy or former smokers, providing large enriched cohorts to study this relatively rare condition.

**PPFE May Progress Independently of ILD**: The progression pathway of upper-lobe PPFE may not rely on typical basal UIP-pattern fibrosis.

## Method

### Overall Architecture

Retrospective longitudinal cohort study: automated segmentation and quantification of PPFE volume on baseline and follow-up LDCT scans using a deep learning algorithm. Annualized change ($\Delta\text{PPFE}$) is calculated, progressive PPFE is defined based on a distribution threshold, and its association with mortality is evaluated using Cox proportional hazards models.

### Key Designs

- **Automated PPFE Quantification**: A deep learning segmentation model automatically quantifies upper-lung PPFE volume ($\text{cm}^3$), calculated on both baseline and follow-up scans.
- **Definition of Progression**: $$\Delta\text{PPFE} = (\text{PPFE}_{\text{follow-up}} - \text{PPFE}_{\text{baseline}}) / \text{time interval}$$; the threshold for progression is defined as half of the standard deviation of baseline PPFE in NLST ($\ge 0.41 \text{ cm}^3/\text{year}$).
- **Survival Analysis**: Univariable and multivariable Cox PH models adjusted for age, sex, smoking pack-years, height, baseline PPFE, and interaction terms.
- **Additional Adjustments in SUMMIT**: FVC% predicted and visual ILA score.
- **Clinical Outcome Analysis**: Negative binomial GLM (for number of respiratory hospitalizations, antibiotic/corticosteroid use) and ordinal logistic regression (for mMRC score).

### Loss & Training

This is a clinical epidemiological study and does not involve training loss functions; details of the PPFE segmentation model training are provided in the supplementary material.

## Key Experimental Results

### Multivariable Cox Model Results

| Variable | NLST HR (95% CI) | SUMMIT HR (95% CI) |
|------|-------------------|---------------------|
| Progressive PPFE | **1.25 (1.01-1.56), p=0.042** | **3.14 (1.66-5.97), p<0.001** |
| Age | 1.10 (1.09-1.11) | 1.08 (1.07-1.10) |
| Male | 1.30 (1.11-1.51) | 1.57 (1.21-2.05) |
| Baseline PPFE | 1.02 (n.s.) | 1.06 (n.s.) |

### SUMMIT Clinical Outcome Associations

| Outcome | IRR/OR (95% CI) | p-value |
|------|-----------------|------|
| Respiratory Hospitalization | IRR=2.79 (1.69-4.60) | <0.001 |
| Antibiotic/Corticosteroid Use | IRR=1.55 (1.10-2.19) | 0.011 |
| mMRC Score | OR=1.40 (0.99-1.97) | 0.055 |

### Baseline Cohort Characteristics

| Characteristic | NLST | SUMMIT |
|------|------|--------|
| Total Participants | 7,980 | 8,561 |
| Proportion of Progressive PPFE | 5.4% (431) | 1.5% (124) |
| Follow-up Duration | 10 years | 2 years |

### Key Findings

- Progressive PPFE independently predicted mortality in both cohorts, remaining significant even after adjusting for ILA and FVC (HR=2.55, p=0.004).
- No significant association was found between major adverse cardiovascular events (MACE5) and PPFE, suggesting that PPFE primarily impacts the respiratory system.
- Baseline PPFE alone only partially captured respiratory risk, whereas longitudinal change provides a stronger prognostic signal.

## Highlights & Insights

1. **External Validation Across Dual Cohorts**: Cross-validated across two independent large-scale screening cohorts (US NLST and UK SUMMIT), demonstrating high consistency.
2. **Clinical Translatability**: Progressive PPFE can be automated and "piggybacked" on existing lung cancer screening LDCT scans without requiring extra examinations.
3. **Multidimensional Outcome Evaluation**: The study assesses not just mortality, but also various clinical endpoints such as hospitalizations, medication use, and symptoms.
4. **Automated Quantification is More Sensitive than Visual Scoring**: Quantitative PPFE can detect prognostic changes missed by visual assessment.

## Limitations & Future Work

1. The training details and performance metrics of the PPFE segmentation model are only in the supplementary materials, leaving the core validation somewhat opaque.
2. The follow-up period for SUMMIT is only 2 years; the long-term prognostic association might be underestimated.
3. The progression threshold ($0.41 \text{ cm}^3/\text{year}$) is derived from the distribution characteristics of NLST, and its external validity when directly applied to SUMMIT remains to be validated.
4. The prevalence of progressive PPFE differs significantly between the two cohorts (5.4% vs 1.5%), which may reflect differences in study populations or scan parameters.
5. The study is observational in nature, which makes it impossible to establish causal relationships.

## Related Work & Insights

- **Prognostic Value of PPFE in ILD**: Gudmundsson et al. (2021, 2023) reported that PPFE quantification is associated with mortality in IPF and HP.
- **Automated PPFE Quantification**: Quantitative upper-lobe PPFE is more sensitive than visual assessment (Gudmundsson et al. 2021).
- **Imaging Biomarkers in Lung Cancer Screening**: LDCT is useful not only for cancer screening but also for identifying comorbidities such as ILA and emphysema.
- **Application of Deep Learning in Chest CT**: Automated segmentation of lung lesions has been widely used in ILD quantification and prognostic research.

## Rating

- Novelty: ⭐⭐⭐ (Clear hypothesis but no major methodological innovation; the core contribution lies in large-scale clinical validation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Dual cohorts over 16,000+ cases, comprehensive multivariable adjustments, and various clinical outcomes)
- Writing Quality: ⭐⭐⭐⭐ (Clinical-style paper with rigorous statistical method descriptions)
- Value: ⭐⭐⭐⭐ (Direct clinical implications for the added value of lung cancer screening, potentially influencing screening workflow design)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[ICML 2026\] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](../../ICML2026/medical_imaging/auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)
- [\[CVPR 2025\] Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)
- [\[CVPR 2026\] Temporal Inversion for Learning Interval Change in Chest X-Rays](../../CVPR2026/medical_imaging/temporal_inversion_for_learning_interval_change_in_chest_x-rays.md)

</div>

<!-- RELATED:END -->
