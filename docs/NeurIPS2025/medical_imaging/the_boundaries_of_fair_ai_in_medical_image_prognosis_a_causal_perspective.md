---
title: >-
  [Paper Note] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective
description: >-
  [NeurIPS 2025][Medical Imaging][Fairness] FairTTE is the first comprehensive framework to systematically investigate fairness in time-to-event (TTE) prediction for medical imaging. It leverages causal analysis to quantif…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Fairness"
  - "Time-to-Event Prediction"
  - "Causal Analysis"
  - "Distribution Shift"
date: 2026-05-08
content_hash: 26a967ac234b178f
---

# The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective

**Conference**: NeurIPS 2025
**arXiv**: [2510.08840](https://arxiv.org/abs/2510.08840)  
**Code**: [https://github.com/pth1993/FairTTE](https://github.com/pth1993/FairTTE)  
**Area**: Medical Imaging / AI Fairness
**Keywords**: Fairness, Time-to-Event Prediction, Causal Analysis, Medical Imaging, Distribution Shift

## TL;DR
FairTTE is the first comprehensive framework to systematically investigate fairness in time-to-event (TTE) prediction for medical imaging. It leverages causal analysis to quantify five sources of bias, and through training over 20,000 models, reveals the limitations of existing fairness methods — particularly the fundamental challenge of maintaining fairness under distribution shift.

## Background & Motivation

Machine learning models are increasingly applied to medical image analysis, yet they may exhibit bias against specific demographic groups. Existing fairness research has primarily focused on **diagnostic tasks** (classification, segmentation), while largely neglecting **prognostic tasks** — predicting disease outcomes or time-to-progression (i.e., time-to-event prediction / survival analysis).

Fairness in TTE prediction faces unique challenges:

**Data scarcity**: Lack of publicly available datasets that simultaneously contain medical images, time-to-event outcomes, and sensitive attributes.

**Unclear bias mechanisms**: Limited understanding of how bias in medical images specifically affects fairness in TTE prediction.

**Absence of unified metrics**: No widely accepted fairness measures for TTE prediction.

**Censoring issues**: Pervasive right-censoring in TTE data renders standard evaluation metrics inapplicable.

The core starting point is to characterize the data-generating process of TTE data using structural causal models (SCMs), systematically decompose sources of bias, understand why existing methods frequently fail, and reveal the fundamental boundaries of fair TTE prediction from a causal perspective.

## Method

### Overall Architecture

The FairTTE framework comprises three levels: (1) a theoretical bias analysis framework grounded in causal inference; (2) a collection of large-scale multi-modal medical imaging datasets; and (3) an evaluation pipeline integrating state-of-the-art TTE prediction models and fairness algorithms.

### Key Designs

1. **Causal Structure**:

    - An unobserved latent health state $Z$ is introduced, decomposing features $X$ into: $X_Z$ (target features correlated with $Z$) and $X_A$ (features correlated with sensitive attribute $A$).
    - Unbiased setting: $A$ affects only $X_A$ and no other variables; learning the invariant feature $X_Z$ suffices for fairness.
    - Biased setting: $A$ may influence $T$, $C$, $X_Z$, etc., through multiple causal paths, leading to $P(t|x_z, a) \neq P(t|x_z, a')$.
    - The causal graph explicitly reveals why learning invariant representations is not always sufficient.

2. **Quantitative Decomposition of Five Bias Sources**:

    - Based on a Bayesian decomposition, the label function is factorized as: $P(y,\delta|x,a) = \underbrace{PMI(x_z,y)}_{\text{image–time mutual info}} \cdot \underbrace{PMI(x_z,\delta)}_{\text{image–censoring mutual info}} \cdot \underbrace{P(y|\delta,a)}_{\text{TTE distribution}} \cdot \underbrace{P(\delta|a)}_{\text{censoring rate}}$
    - Five bias sources: (i) distributional differences in image features; (ii) differences in mutual information between $X_Z$ and $Y$; (iii) differences in mutual information between $X_Z$ and $\Delta$; (iv) differences in TTE distributions; (v) differences in censoring rates.
    - Each bias source can be quantified independently (Wasserstein distance, normalized mutual information, etc.).

3. **Theoretical Fairness Analysis**:

    - **Theorem 1**: The fairness error upper bound is $\mathcal{F}_{Er}(h) \leq \max_{a,a'}(\eta(\mathcal{H},f_a,f_{a'}) + \mathcal{D}(\mathcal{H},D_a,D_{a'}))$
    - Here $\eta$ is the minimum joint prediction error (difficult to reduce when label functions $f_a \neq f_{a'}$), and $\mathcal{D}$ is the subgroup distribution distance (reducible via fair representation learning).
    - **Proposition 2**: Under covariate shift, if the representation $Z$ is sufficient for the target, learning a fair representation enables fair TTE prediction.
    - Core insight: When causal paths cause the label functions of different groups to differ fundamentally, achieving fairness becomes inherently difficult.

### Loss & Training
- TTE prediction models: DeepHit (discrete-time competing risks), Nnet-survival, PMF.
- Fairness algorithms: 5 state-of-the-art methods covering three strategy categories — pre-processing (SR), in-training (DI, FRL, DRO), and post-processing (CSA).
- Model selection: the model with optimal fairness on the validation set is selected, allowing at most a 5% drop in predictive performance.
- Backbone networks: 2D EfficientNet (AREDS/MIMIC-CXR), 3D ResNet-18 (ADNI).
- 10 random seeds are used to ensure stability.

## Key Experimental Results

### Main Results

**Inter-group Performance Gap ($Ct^d$) Across Datasets and Sensitive Attributes**

| Dataset–Sensitive Attribute | DeepHit Best Group | DeepHit Worst Group | Gap | Best Fairness Algorithm Effect |
|------|------|------|----------|------|
| ADNI–Age | ~80% | ~66% | 14.19 | DI: −66.22% (gap reduction) |
| AREDS–Race | ~88% | ~77% | 11.09 | DRO: −37.21% (gap reduction) |
| MIMIC-CXR–Age | ~79% | ~73% | 5.93 | DRO: −95.74% (gap reduction) |
| AREDS–Sex | ~82% | ~81% | 1.32 | SR: −85.27% (gap reduction) |

### Ablation Study

| Experimental Setting | Key Finding | Notes |
|------|---------|------|
| Pre-trained vs. trained from scratch | Pre-training improves accuracy (especially ADNI) but does not improve fairness | p > 0.05 in 18/24 settings |
| Distribution shift (Y noise) | IBS deteriorates significantly; ranking metrics less affected | Directly impacts error-based measures |
| Distribution shift (Δ flip) | $Ct^d$ and $AUCt^d$ severely degrade | Disrupts comparable quantities |
| Distribution shift (X noise) | All metrics degrade | Feature quality diminishes |
| Bias quantification correlation | High bias degree ↔ large performance gap | AREDS–Race most representative |

### Key Findings
- **Bias is pervasive**: Significant inter-group performance disparities are observed across all dataset–sensitive attribute combinations.
- **Age and race exhibit greater bias than sex**: Consistent across all datasets.
- **Existing fairness methods have limited effectiveness**: No method consistently outperforms the baseline DeepHit across all settings; some even exacerbate unfairness in certain cases.
- **Fairness–accuracy trade-off**: Improvements in fairness are frequently accompanied by reductions in predictive accuracy.
- **Pre-training does not automatically improve fairness**: It raises accuracy but leaves fairness metrics largely unchanged.
- **Fairness is harder to maintain under distribution shift**: Different types of shift affect fairness in distinct ways.
- Quantified bias sources are highly correlated with the degree of model unfairness, validating the practical utility of the causal decomposition framework.

## Highlights & Insights
- This is the first study to systematically introduce causal analysis into fairness research for TTE prediction in medical imaging.
- The decomposition into five bias sources provides a fine-grained diagnostic tool for understanding *why* unfairness arises.
- Theorem 1 clearly explains why fairness is fundamentally unachievable when the label function itself depends on the sensitive attribute (e.g., age genuinely affects survival time).
- The large-scale experiments (20,000+ models) ensure the reliability of the conclusions.
- The intrinsic connection between fairness and distribution shift is a profound insight: achieving fairness is equivalent to enabling the model to generalize to a test distribution that retains only fair causal paths.

## Limitations & Future Work
- Only single-risk settings and non-informative right-censoring are considered; competing risks and informative censoring are not addressed.
- Group fairness (inter-group performance gap) is adopted; individual fairness is not considered.
- In certain clinical scenarios, sensitive attributes are genuine risk factors (e.g., age and mortality), making strict group fairness potentially inappropriate.
- The causal directions in the causal graph require domain knowledge from clinical experts to establish.
- How to distinguish "fair causal paths" from "unfair causal paths" is not explored.

## Related Work & Insights
- MedFair (Zong et al., 2023) and FairSeg (Tian et al., 2024) focus on fairness in diagnostic tasks.
- This paper extends fairness research to the prognostic domain, filling an important gap.
- The causal inference perspective provides a deeper understanding of fairness than purely statistical approaches.
- Fairness under distribution shift is an important but insufficiently studied direction.
- Inspired directions: developing fairness algorithms that simultaneously address multiple bias sources and remain robust under distribution shift.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of fairness in TTE prediction; the causal decomposition framework is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × multiple sensitive attributes × 3 models × 5 algorithms × 10 seeds — remarkable scale.
- Writing Quality: ⭐⭐⭐⭐ Clear framework with tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for fairness research in medical imaging; insights are profound and actionable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](../../AAAI2026/medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[NeurIPS 2025\] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)

</div>

<!-- RELATED:END -->
