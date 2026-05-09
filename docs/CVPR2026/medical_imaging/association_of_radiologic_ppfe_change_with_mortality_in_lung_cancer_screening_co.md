---
title: >-
  [Paper Note] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts
description: >-
  [CVPR 2026][Medical Imaging][Pleuroparenchymal fibroelastosis] Across two large-scale lung cancer screening cohorts (NLST n=7,980; SUMMIT n=8,561), this study employs deep learning to automatically segment PPFE volumes and defines "progressive PPFE" based on annualized volume change. Cox proportional hazards models demonstrate that PPFE progression is an independent predictor of all-cause mortality (NLST HR=1.25; SUMMIT HR=3.14), and is significantly associated with respiratory hospitalization rates, antibiotic/corticosteroid usage, and other clinical endpoints.
tags:
  - CVPR 2026
  - Medical Imaging
  - Pleuroparenchymal fibroelastosis
  - lung cancer screening
  - deep learning segmentation
  - longitudinal imaging analysis
  - survival analysis
date: 2026-05-08
content_hash: 39abd29937b43758
---

# Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts

**Conference**: CVPR 2026
**arXiv**: [2603.09531](https://arxiv.org/abs/2603.09531)
**Code**: To be confirmed
**Area**: Medical Imaging
**Keywords**: Pleuroparenchymal fibroelastosis, lung cancer screening, deep learning segmentation, longitudinal imaging analysis, survival analysis

## TL;DR

Across two large-scale lung cancer screening cohorts (NLST n=7,980; SUMMIT n=8,561), this study employs deep learning to automatically segment PPFE volumes and defines "progressive PPFE" based on annualized volume change. Cox proportional hazards models demonstrate that PPFE progression is an independent predictor of all-cause mortality (NLST HR=1.25; SUMMIT HR=3.14), and is significantly associated with respiratory hospitalization rates, antibiotic/corticosteroid usage, and other clinical endpoints.

## Background & Motivation

Pleuroparenchymal fibroelastosis (PPFE) is an upper-lobe-predominant fibrotic abnormality characterized by subpleural elastosis and parenchymal fibrosis. In patients with interstitial lung disease (ILD), the presence of PPFE has been associated with worse prognosis and higher mortality. However, existing research is subject to the following limitations:

**Population bias**: Prior studies have focused on ILD clinical cohorts, with little attention paid to general lung cancer screening (LCS) populations—the prevalence and clinical significance of PPFE in LCS cohorts remain systematically unexplored.

**Limitations of cross-sectional analyses**: Most studies evaluate only the presence or absence of PPFE (binary classification) without accounting for PPFE **progression** over time—yet the dynamic trajectory of fibrotic disease may carry greater predictive value than its static state.

**Bottleneck of manual assessment**: Radiologic evaluation of PPFE relies on expert readers, is highly subjective and labor-intensive, and is difficult to apply routinely in large-scale screening cohorts.

**Lack of quantitative methods**: No reproducible automated tools exist for tracking longitudinal changes in PPFE volume.

The central motivation of this study is: **In the broader LCS population, can longitudinal PPFE progression independently predict mortality?** If so, quantitative PPFE assessment would constitute a low-cost, additive imaging biomarker within the LCS workflow.

## Method

### Overall Architecture

The overall pipeline consists of three stages:

1. **Automated deep learning segmentation**: PPFE regions are automatically segmented on low-dose CT, and volumes are computed.
2. **Longitudinal change quantification**: Annualized volume change ΔPPFE is derived from baseline and follow-up scans; a threshold-based criterion defines "progressive PPFE."
3. **Survival analysis and clinical association**: Cox models test the independent association between PPFE progression and mortality; negative binomial regression and ordinal logistic regression examine associations with healthcare resource utilization.

### Key Designs

1. **Automated PPFE Segmentation Model**

    - A 3D U-Net–based segmentation network trained on expert-annotated CT data.
    - Input: low-dose chest CT scans; output: voxel-wise PPFE probability maps.
    - Post-processing: thresholding + connected-component analysis → PPFE volume (mL).
    - The model achieves sufficient segmentation accuracy on an independent validation set to ensure reliability of downstream statistical analyses.

2. **Computation of Annualized Change ΔPPFE**

    - Definition: $\Delta\text{PPFE} = \frac{V_{\text{follow-up}} - V_{\text{baseline}}}{T}$, where $T$ is the interscan interval in years.
    - Progressive PPFE is defined by dichotomization: ΔPPFE exceeding a **half-standard-deviation threshold** (0.5 SD) is classified as progression.
    - Rationale for the half-SD threshold over alternatives such as the median: in longitudinal imaging analysis, half a standard deviation is a widely accepted proxy for the minimal clinically important difference (MCID).

3. **Cox Proportional Hazards Model**

    - Outcome: all-cause mortality.
    - Primary predictor: progressive PPFE (binary).
    - Covariates: age, sex, BMI, smoking status (pack-years), baseline lung function (FEV1/FVC), baseline PPFE volume, emphysema proportion, and others.
    - Models are fitted separately in the NLST and SUMMIT cohorts to assess reproducibility.

4. **Clinical Endpoint Association Analyses (SUMMIT only)**

    - Respiratory-related hospitalization rate: negative binomial regression → incidence rate ratio (IRR).
    - Antibiotic/corticosteroid use frequency: negative binomial regression → IRR.
    - Trend in mMRC dyspnea score change: ordinal logistic regression → OR.

### Loss & Training

- The segmentation model is trained with a combined Dice Loss and Cross-Entropy Loss to address the small volume and class imbalance of PPFE regions.
- Data augmentation: random flipping, rotation, elastic deformation, and intensity jittering.
- The training set consists of expert-annotated subsets; the validation set is used for early stopping.

## Key Experimental Results

### Main Results

**Cox Proportional Hazards Models — Progressive PPFE and All-Cause Mortality**:

| Cohort | N | Median Follow-up | Progression Rate | HR (95% CI) | p-value |
|--------|---|------------------|------------------|-------------|---------|
| NLST | 7,980 | ~6.5 years | — | **1.25** (1.01–1.55) | 0.042 |
| SUMMIT | 8,561 | ~3 years | — | **3.14** (1.73–5.71) | **<0.001** |

- PPFE progression is an **independent predictor of mortality** in both independent cohorts.
- The higher HR in SUMMIT may reflect the shorter follow-up period and concentration of events.

**Clinical Endpoint Associations in the SUMMIT Cohort**:

| Clinical Endpoint | Metric | Estimate (95% CI) | p-value |
|-------------------|--------|-------------------|---------|
| Respiratory hospitalization rate | IRR | **2.79** (1.52–5.13) | <0.01 |
| Antibiotic/corticosteroid use | IRR | **1.55** (1.12–2.13) | <0.01 |
| mMRC deterioration trend | OR | **1.40** (0.98–2.01) | 0.065 |

### Ablation Study

| Analysis Variant | HR (NLST) | HR (SUMMIT) | Note |
|-----------------|-----------|-------------|------|
| Baseline PPFE volume only (no progression) | ~1.1 (ns) | ~1.3 | Weak predictive power of static measure |
| ΔPPFE as continuous variable | Significant | Significant | Continuous form also effective |
| Progressive PPFE (binary, 0.5 SD) | **1.25** | **3.14** | Best clinical interpretability |
| Unadjusted for covariates | Higher HR | Higher HR | HR attenuates but remains significant after adjustment |

### Key Findings

- **Progression outperforms presence**: Baseline PPFE volume alone does not reach statistical significance in NLST, whereas progressive PPFE does—demonstrating that dynamic change carries greater predictive value than static state.
- **Cross-cohort reproducibility**: Consistent directional findings are obtained in NLST (US multicenter RCT) and SUMMIT (UK single-center prospective study), two cohorts with entirely different designs.
- **Clinical actionability**: PPFE progression is associated with respiratory hospitalizations and medication use, conferring concrete value for clinical decision-making beyond a purely statistical finding.
- **Feasibility of automation**: Deep learning segmentation enables quantitative PPFE assessment as a zero-additional-cost add-on to routine LCS workflows.

## Highlights & Insights

- **Paradigm shift from static to dynamic**: The core contribution lies not in the segmentation model per se, but in proposing a complete analytical framework of "longitudinal change quantification + progression definition + survival analysis"—a paradigm transferable to other imaging biomarkers (e.g., emphysema, coronary artery calcification).
- **Dual-cohort validation design**: NLST and SUMMIT differ in country, screening protocol, CT acquisition parameters, and follow-up duration; the consistency of results substantially strengthens the credibility of the conclusions.
- **Practical orientation**: The methodology is tailored to clinical workflows—leveraging low-dose CT already acquired during LCS, requiring no additional examinations or annotations, and achieving a high degree of automation.
- **Statistical justification of the half-SD threshold**: Adopting a predefined MCID criterion avoids the overfitting risk of data-driven optimal threshold search.

## Limitations & Future Work

- **Limited causal inference**: As an observational cohort study, a causal relationship between PPFE progression and mortality cannot be established, and residual confounding may remain.
- **Insufficient segmentation model details**: The paper provides limited description of the segmentation architecture, training data size, and generalizability validation—robustness across different CT protocols and scanners remains unclear.
- **Threshold sensitivity**: Although the half-SD threshold is theoretically motivated, alternative thresholds (e.g., interquartile range, ROC-optimal cutpoint) are not sufficiently explored for their impact on conclusions.
- **Ambiguity in PPFE pathological definition**: Discrepancies exist between radiologic and pathologically confirmed PPFE; automated segmentation may include fibrotic regions that do not represent true PPFE.
- **All-cause mortality only**: Respiratory-specific mortality is not distinguished from other causes of death, limiting specificity.

## Related Work & Insights

- Compared with PPFE prognostic studies in ILD populations (e.g., Reddy et al., Kato et al.), this study is the first to extend the study population to LCS participants.
- The methodological approach to longitudinal imaging biomarker analysis can draw on established practices in analogous domains, such as annual FEV1 decline rate and coronary artery calcification progression.
- The paradigm of deep learning segmentation combined with longitudinal tracking is extendable to other incidental LCS findings (e.g., bone density, cardiovascular calcification, hepatic steatosis).

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic investigation of the prognostic value of PPFE progression in an LCS population; the combination of longitudinal analysis and automation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Independent dual-cohort validation, multiple clinical endpoints, and well-adjusted multivariable models.
- Writing Quality: ⭐⭐⭐⭐ Rigorous clinical paper style with standardized statistical reporting.
- Value: ⭐⭐⭐⭐ Offers a low-cost, additive risk stratification tool for LCS with strong clinical translatability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](sparse_task_vector_mixup_wsi_prognosis.md)
- [\[CVPR 2026\] Novel Architecture of RPA in Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_de.md)
- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gender-adversarial_attention_multi.md)
- [\[CVPR 2026\] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
