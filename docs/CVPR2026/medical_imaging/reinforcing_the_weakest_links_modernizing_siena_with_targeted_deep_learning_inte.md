---
title: >-
  [Paper Note] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration
description: >-
  [CVPR 2026][Medical Imaging][brain atrophy estimation] By replacing the classical skull stripping (BET2) and tissue segmentation (FAST) modules in the SIENA brain atrophy pipeline with deep learning alternatives (SynthStrip, SynthSeg), this work significantly improves the clinical sensitivity and robustness of PBVC estimation while preserving the interpretability of the overall pipeline.
tags:
  - CVPR 2026
  - Medical Imaging
  - brain atrophy estimation
  - SIENA
  - skull stripping
  - deep learning module replacement
  - longitudinal MRI
date: 2026-05-08
content_hash: dfb4723f3d451f47
---

# Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration

**Conference**: CVPR 2026
**arXiv**: [2603.12951](https://arxiv.org/abs/2603.12951)
**Code**: [GitHub](https://github.com/Raciti/Enhanced-SIENA)
**Area**: Medical Imaging
**Keywords**: brain atrophy estimation, SIENA, skull stripping, deep learning module replacement, longitudinal MRI

## TL;DR

By replacing the classical skull stripping (BET2) and tissue segmentation (FAST) modules in the SIENA brain atrophy pipeline with deep learning alternatives (SynthStrip, SynthSeg), this work significantly improves the clinical sensitivity and robustness of PBVC estimation while preserving the interpretability of the overall pipeline.

## Background & Motivation

Percentage Brain Volume Change (PBVC) is a key MRI biomarker for assessing the progression of neurodegenerative diseases such as Alzheimer's disease (AD) and Parkinson's disease (PD). SIENA is the most widely used longitudinal whole-brain atrophy estimation pipeline, with a workflow comprising: BET2 skull stripping → FLIRT symmetric affine registration → FAST tissue segmentation → boundary shift estimation → PBVC output.

However, SIENA's reliance on two classical image processing modules introduces notable weaknesses:
- **BET2 skull stripping**: sensitive to signal inhomogeneity, motion artifacts, and severe atrophy; small parameter perturbations can lead to substantial differences in estimated atrophy rates.
- **FAST tissue segmentation**: performance degrades when upstream skull stripping is imperfect, and it cannot operate robustly independent of preceding modules.
- These errors propagate downstream through registration and boundary detection steps, biasing the final PBVC estimates.

Recent end-to-end deep learning methods (DeepBVC, EAM, BrainLossNet) can directly predict PBVC but sacrifice the interpretability of SIENA. The authors therefore propose a **modular replacement** strategy—replacing only the most fragile processing steps while retaining the remainder of the pipeline.

## Method

### Overall Architecture

The core SIENA workflow (symmetric registration, boundary identification, displacement estimation) is preserved unchanged; only two front-end modules are replaced, yielding four pipeline variants:

| Pipeline | Skull Stripping | Tissue Segmentation |
|----------|----------------|-------------------|
| SIENA Vanilla | BET2 | FAST |
| SIENA-SS | **SynthStrip** | FAST |
| SIENA-SEG | BET2 | **SynthSeg** |
| SIENA-SS-SEG | **SynthStrip** | **SynthSeg** |

### Key Designs

1. **SynthStrip Integration (skull stripping replacement)**: SynthStrip is trained with domain randomization, providing strong generalization across acquisition protocols and artifacts. Since SynthStrip outputs only a brain mask without a skull mask, the authors design an additional skull mask derivation procedure: Gaussian smoothing of the brain mask ($\sigma=1.0$) → ray casting outward from boundary voxels along surface normals (maximum distance 30 mm) → detection of the inner skull boundary using BET2 intensity-gradient heuristics → generation of a skull mask for use in registration.

2. **SynthSeg Integration (tissue segmentation replacement)**: SynthSeg outputs anatomical structure labels (>30 classes), which are merged into the three-class labels required by SIENA: CSF (ventricles, lateral ventricles, etc.), GM (cortex, thalamus, hippocampus, amygdala, cerebellar cortex, etc.), and WM (cerebral/cerebellar white matter, brainstem). Domain randomization training likewise confers robustness to heterogeneous datasets.

3. **Pipeline integrity**: All variants share SIENA's FLIRT symmetric registration, boundary identification, and PBVC estimation steps. Careful mask conversion ensures full compatibility between the new modules and the original pipeline.

### Loss & Training

No new training procedures are introduced. Both SynthStrip and SynthSeg are used as pretrained models (FreeSurfer v7.4.1) and serve as plug-and-play replacements within the SIENA pipeline. Evaluation employs three complementary criteria: (1) correlation with disease progression; (2) scan-order consistency; (3) computational efficiency.

## Key Experimental Results

### Main Results

Pearson correlation coefficients between PBVC and clinical measures on the ADNI cohort (AD, N=1006):

| Measure | SIENA Vanilla | SIENA-SS | SIENA-SEG | SIENA-SS-SEG |
|---------|-------------|----------|-----------|-------------|
| ΔMMSE | -0.226 | **-0.497** | -0.252 | -0.384 |
| ΔCDR-SB | -0.258 | **-0.608** | -0.290 | -0.453 |
| ΔADAS-13 | -0.254 | **-0.524** | -0.271 | -0.405 |
| ΔFAQ | -0.260 | **-0.540** | -0.257 | -0.394 |
| ΔBPF | -0.118 | **-0.249** | -0.098 | -0.167 |

Scan-order consistency (MFRR, ADNI):

| Pipeline | MFRR (%) | Relative Improvement |
|----------|---------|-------------------|
| SIENA Vanilla | 0.379 | — |
| SIENA-SS | 0.067 | 82.4% |
| SIENA-SEG | 0.307 | 18.9% |
| SIENA-SS-SEG | **0.046** | **87.8%** |

On the PPMI cohort (PD), SIENA-SS-SEG achieves a **99.1%** reduction in MFRR.

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Skull stripping only (SS) | ΔCDR-SB: -0.608 (Z=-11.98, p<0.001) | Largest gain in clinical sensitivity |
| Segmentation only (SEG) | Most metrics show no significant improvement | Limited effect when replaced in isolation |
| Full replacement (SS-SEG) | Lowest MFRR, best scan-order consistency | Optimal robustness |
| GPU acceleration | SEG/SS-SEG: ~1002–1166 s vs. CPU 1855 s | Up to 46% speedup |

### Key Findings

- Skull stripping is the most fragile component of the SIENA pipeline—replacing it yields far greater clinical gains than replacing the segmentation module alone.
- The full replacement variant achieves the best scan-order consistency but slightly lower clinical correlation than skull-stripping-only replacement.
- GPU acceleration is primarily attributable to SynthSeg; SynthStrip contributes comparatively less speedup.

## Highlights & Insights

- The **modular modernization** philosophy has broad applicability: rather than overhauling a mature pipeline, only the weakest links are reinforced, preserving clinical trustworthiness and interpretability.
- The scan-order consistency analysis (Forward–Reverse Residual) serves as an elegant metric for evaluating pipeline robustness.
- The ray-casting-based skull mask derivation is a concise and practical solution to the compatibility issue arising from SynthStrip's lack of a skull mask output.
- Validation on two large-scale longitudinal cohorts (ADNI: N=1006; PPMI: N=310) with rigorous statistical testing (Steiger's Z, Bonferroni correction) strengthens the credibility of the findings.

## Limitations & Future Work

- Improvements on the PPMI cohort do not reach statistical significance, possibly due to milder atrophy in PD and smaller sample size.
- The pipeline still relies on SIENA's FLIRT linear registration; incorporating deep learning registration (e.g., SynthMorph) could be explored.
- Skull mask derivation remains partially dependent on BET2 heuristics; an end-to-end skull segmentation model would be the ideal solution.
- Only whole-brain atrophy is evaluated; extension to regional atrophy (e.g., hippocampal volume change) is not addressed.

## Related Work & Insights

- The domain randomization training strategy of **SynthStrip/SynthSeg** (FreeSurfer) is the foundation that makes modular replacement feasible.
- This work is complementary to end-to-end methods such as DeepBVC and EAM: modular replacement preserves interpretability, while end-to-end methods prioritize estimation accuracy.
- The approach is analogous to modular upgrades in other medical imaging pipelines (e.g., deep learning replacements for registration and bias field correction).

## Rating

- Novelty: ⭐⭐⭐ — Methodologically a combination of established tools, but the systematic study of modular modernization has independent value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two large-scale cohorts, multi-dimensional evaluation metrics, and rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated, and thorough experimental analysis.
- Value: ⭐⭐⭐⭐ — Directly applicable to clinical neuroimaging practice; open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2026\] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)

</div>

<!-- RELATED:END -->
