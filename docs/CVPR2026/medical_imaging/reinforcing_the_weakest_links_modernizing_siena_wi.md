---
title: >-
  [Paper Note] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration
description: >-
  [CVPR 2026][Medical Imaging][SIENA] This work selectively replaces the classical skull stripping (BET2) and tissue segmentation (FAST) modules in the SIENA longitudinal brain atrophy pipeline with deep learning alternatives (SynthStrip/SynthSeg). Evaluated on two large-scale longitudinal cohorts—ADNI (N=1006) and PPMI (N=310)—the proposed modifications substantially improve the correlation between PBVC and clinical disease progression (correlation coefficients increase by over 100%), while reducing scan-order error by up to 99.1%.
tags:
  - CVPR 2026
  - Medical Imaging
  - SIENA
  - brain atrophy
  - longitudinal MRI
  - SynthStrip
  - SynthSeg
  - modular modernization
date: 2026-05-08
content_hash: 5b0e230755a97825
---

# Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration

**Conference**: CVPR 2026
**arXiv**: [2603.12951](https://arxiv.org/abs/2603.12951)
**Code**: [GitHub](https://github.com/Raciti/Enhanced-SIENA.git)
**Area**: Medical Imaging / Brain Atrophy Assessment
**Keywords**: SIENA, brain atrophy, longitudinal MRI, SynthStrip, SynthSeg, modular modernization

## TL;DR

This work selectively replaces the classical skull stripping (BET2) and tissue segmentation (FAST) modules in the SIENA longitudinal brain atrophy pipeline with deep learning alternatives (SynthStrip/SynthSeg). Evaluated on two large-scale longitudinal cohorts—ADNI (N=1006) and PPMI (N=310)—the proposed modifications substantially improve the correlation between PBVC and clinical disease progression (correlation coefficients increase by over 100%), while reducing scan-order error by up to 99.1%.

## Background & Motivation

**Background**: SIENA is the most widely used tool for assessing longitudinal brain atrophy (PBVC, Percentage Brain Volume Change). It estimates atrophy rates by analyzing boundary displacements in registered brain images, has been validated across numerous clinical trials, and offers strong interpretability—each intermediate step can be individually inspected.

**Limitations of Prior Work**: SIENA relies on classical FSL algorithms: BET2 for skull stripping (based on intensity heuristics and deformable surface models) and FAST for tissue segmentation (based on similar principles). These algorithms are parameter-sensitive—minor adjustments to BET2's fractional intensity threshold can cause substantial variation in estimated atrophy rates—and are prone to failure under severe neurodegeneration, signal inhomogeneity, or motion artifacts. Skull stripping errors propagate downstream into registration and segmentation steps.

**Key Challenge**: End-to-end deep learning methods (e.g., DeepBVC, EAM) can directly predict PBVC but sacrifice the interpretability and clinical trustworthiness of SIENA; fully retaining SIENA leaves it vulnerable to the fragility of classical image processing steps.

**Goal**: To improve robustness and clinical sensitivity by selectively replacing the weakest image processing steps, while preserving SIENA's validated and interpretable core framework.

**Key Insight**: Rather than replacing the entire pipeline, this work follows a "reinforce the weakest links" philosophy—identifying skull stripping and tissue segmentation as the two bottlenecks and replacing them with domain-randomization-trained deep learning solutions (SynthStrip/SynthSeg).

**Core Idea**: Replace BET2 with SynthStrip and FAST with SynthSeg to achieve maximum robustness gains in SIENA with minimal architectural changes.

## Method

### Overall Architecture

The core SIENA pipeline is preserved (symmetric skull-constrained registration → boundary detection → displacement estimation → bidirectional averaging), with only the two preprocessing modules replaced. This yields four pipeline variants: SIENA Vanilla (BET2+FAST), SIENA-SS (SynthStrip+FAST), SIENA-SEG (BET2+SynthSeg), and SIENA-SS-SEG (SynthStrip+SynthSeg).

### Key Designs

1. **SynthStrip Integration and Skull Mask Derivation**:
    - Function: Replace BET2 with SynthStrip for brain extraction, and derive the skull mask required by SIENA from SynthStrip outputs.
    - Mechanism: SynthStrip outputs only a brain mask, not a skull mask. To maintain compatibility, a derivation pipeline is designed: Gaussian smoothing of the brain mask ($\sigma=1.0$) → surface normal estimation from gradients → ray casting along normals (up to 30 mm) → inner skull boundary detection using BET2's intensity gradient heuristic → aggregation of detected points to construct the skull mask.
    - Design Motivation: SIENA's skull-constrained registration requires a skull mask as a stable anatomical reference to prevent longitudinal atrophy from being erroneously normalized away.

2. **SynthSeg Integration and Label Mapping**:
    - Function: Replace FAST with SynthSeg for tissue segmentation, mapping fine-grained anatomical labels to the three tissue classes required by SIENA.
    - Mechanism: SynthSeg outputs detailed anatomical structure labels (cortex, thalamus, hippocampus, etc.) rather than three tissue classes. The mapping rules are: ventricles (lateral, third, fourth, etc.) → CSF; cortex + subcortical gray matter (thalamus, caudate, putamen, hippocampus, etc.) → GM; white matter + brainstem → WM.
    - Design Motivation: SIENA's boundary detection requires only CSF/GM/WM segmentation, and SynthSeg's domain-randomization training confers stronger generalization across acquisition protocols.

### Loss & Training

No training is required. SynthStrip and SynthSeg are used with pretrained weights—obtained via domain-randomization training for strong generalization—and serve as drop-in replacements. Experiments are conducted under FSL v6.0.7.17 and FreeSurfer v7.4.1, with SynthSeg's robust mode enabled.

## Key Experimental Results

### Main Results

Pearson correlation coefficients between PBVC and clinical deterioration in the ADNI cohort (AD, N=1006):

| Clinical Metric | SIENA Vanilla (r) | SIENA-SS (r) | Gain | Significance |
|---------|-------------------|--------------|---------|-----------|
| MMSE | -0.226 | -0.497 | +119.9% | p<0.001 |
| CDR-SB | -0.258 | -0.608 | +135.7% | p<0.001 |
| ADAS-13 | -0.254 | -0.524 | +106.3% | p<0.001 |
| FAQ | -0.260 | -0.540 | +107.7% | p<0.001 |
| BPF | -0.118 | -0.249 | +111.0% | p<0.001 |

Scan-order consistency (MFRR↓, lower is better):

| Pipeline | ADNI MFRR | Improvement | PPMI MFRR | Improvement |
|------|-----------|------|-----------|------|
| Vanilla | 0.379% | - | 0.246% | - |
| SIENA-SS | 0.067% | -82.4% | 0.002% | -99.0% |
| SIENA-SS-SEG | 0.046% | **-87.8%** | 0.002% | **-99.1%** |

### Ablation Study

| Configuration | Clinical Correlation | Scan Symmetry | Runtime | Notes |
|----------|----------|-----------|---------|------|
| Skull stripping only (SS) | Largest gain (all metrics >100%) | 82–99% improvement | Comparable to Vanilla | Skull stripping is the weakest link |
| Segmentation only (SEG) | Limited and inconsistent gains | Moderate improvement | GPU: −46% (1002s vs. 1855s) | FAST directly models tissue classes; substitution yields limited gains |
| Both replaced (SS-SEG) | Slightly below SS alone | **Best symmetry** | GPU speedup achieved | Complementarity between modules manifests in symmetry |

### Key Findings

- **Skull stripping is unambiguously the weakest link**: replacing this step alone yields over 100% improvement across all clinical correlation metrics, with Steiger Z-tests all reaching p<0.001.
- Scan-order error is reduced from 0.379% to 0.046% (ADNI) and from 0.246% to 0.002% (PPMI), nearly eliminating directional bias.
- Effect sizes on the PPMI cohort are smaller and do not reach statistical significance—PD-related brain atrophy is slower than in AD, and the sample size (N=310) is relatively small.
- GPU acceleration reduces runtime by 46%, while CPU runtime remains comparable to the original SIENA.

## Highlights & Insights

- The "modular modernization" strategy has broad translational value—reinforcing the weakest links rather than replacing the entire pipeline preserves clinical trust and interpretability.
- A 99.1% reduction in scan-order error is a remarkable robustness improvement, revealing BET2 as the primary source of directional bias.
- The three-dimensional evaluation framework (clinical correlation + scan symmetry + computational efficiency) is comprehensive and well-motivated.
- The strong generalization of SynthStrip/SynthSeg via domain-randomization training demonstrates the substantial potential of synthetic data training in medical imaging.

## Limitations & Future Work

- No in vivo ground truth for brain atrophy exists; pipeline quality can only be assessed via proxy measures (correlation with clinical scales).
- The mapping rules from SynthSeg anatomical labels to three tissue classes may not be optimal; alternative mapping strategies are not systematically compared.
- Effects on PPMI are non-significant; validation on larger PD cohorts is needed.
- No cross-framework comparison with other brain atrophy methods (BSI, BrainLossNet, etc.) is conducted.
- Only whole-brain atrophy is evaluated; improvements in regional atrophy estimation are not explored.

## Related Work & Insights

- **vs. DeepBVC/EAM**: End-to-end DL methods predict PBVC directly but lack transparency and rely on noisy SIENA-generated targets for training; this work retains SIENA's interpretable framework.
- **vs. BrainLossNet**: Estimates PBVC from deformation fields but requires SIENA values for calibration, thus remaining indirectly dependent on SIENA's accuracy.
- **vs. BSI**: A classical method requiring manual brain extraction; SIENA is fully automated but fragile; this work uses DL to reinforce the automated steps.
- **Insight**: A "minimally invasive upgrade" strategy for established clinical tools may achieve broader clinical adoption than wholesale replacement.

## Rating

- Novelty: ⭐⭐⭐ — The method amounts to direct module substitution with limited technical novelty, though the systematic methodology for identifying the weakest link has methodological value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two large cohorts (1006+310 subjects), six clinical metrics, scan symmetry, and runtime analysis; evaluation is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, methodology is rigorous, and statistical analysis is well-grounded (Fisher z-transform, Steiger test, Bonferroni correction).
- Value: ⭐⭐⭐⭐ — Offers direct practical value to the clinical neuroimaging community; existing SIENA users can benefit immediately.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)
- [\[CVPR 2026\] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_le.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)

<!-- RELATED:END -->
