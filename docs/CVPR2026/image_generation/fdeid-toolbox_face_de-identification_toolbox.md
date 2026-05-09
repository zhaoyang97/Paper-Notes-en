---
title: >-
  [Paper Note] FDeID-Toolbox: Face De-Identification Toolbox
description: >-
  [CVPR 2026][Image Generation][face de-identification] This paper presents FDeID-Toolbox, a modular face de-identification toolbox that uniformly integrates 16 de-identification methods (spanning four categories: naive, generative, adversarial, and K-Same), 6 benchmark datasets, and a systematic evaluation protocol covering three dimensions—privacy protection, attribute preservation, and visual quality—addressing the field's persistent problems of fragmented implementations, inconsistent evaluation protocols, and incomparable results.
tags:
  - CVPR 2026
  - Image Generation
  - face de-identification
  - privacy preservation
  - toolbox
  - benchmark
  - evaluation protocol
date: 2026-05-08
content_hash: b7a44c45f6010e69
---

# FDeID-Toolbox: Face De-Identification Toolbox

**Conference**: CVPR 2026
**arXiv**: [2603.13121](https://arxiv.org/abs/2603.13121)
**Authors**: Hui Wei, Hao Yu, Guoying Zhao (University of Oulu)
**Code**: [infraface/FDeID-Toolbox](https://github.com/infraface/FDeID-Toolbox)
**Area**: Image Generation
**Keywords**: face de-identification, privacy preservation, toolbox, benchmark, evaluation protocol

## TL;DR

This paper presents FDeID-Toolbox, a modular face de-identification toolbox that uniformly integrates 16 de-identification methods (spanning four categories: naive, generative, adversarial, and K-Same), 6 benchmark datasets, and a systematic evaluation protocol covering three dimensions—privacy protection, attribute preservation, and visual quality—addressing the field's persistent problems of fragmented implementations, inconsistent evaluation protocols, and incomparable results.

## Background & Motivation

Face De-Identification (FDeID) aims to remove personally identifiable information from facial images while retaining task-relevant attributes such as age, gender, and expression. This is critical for privacy-preserving computer vision, yet the field faces three fundamental challenges:

- **Fragmented implementations**: Methods are developed in isolation using different frameworks, preprocessing pipelines, and data formats, making reproduction and comparison difficult.
- **Inconsistent evaluation protocols**: FDeID spans multiple downstream tasks (age estimation, gender recognition, expression analysis, etc.) and requires evaluation across privacy protection, attribute preservation, and visual quality dimensions; existing codebases lack a unified standard.
- **Incomparable results**: Different papers use different data splits, evaluation models, and metric definitions, rendering fair cross-method performance comparisons infeasible.

The core motivation is to construct a unified, modular, and extensible toolbox that enables researchers to fairly compare FDeID methods under fully controlled, consistent conditions, thereby advancing reproducible research.

## Method

### Overall Architecture

FDeID-Toolbox adopts a modular design comprising four core components:

1. **Standardized data loaders**: Unified interfaces for 6 mainstream benchmark datasets (LFW, AgeDB, AffectNet, CelebA-HQ, FairFace, PURE).
2. **Unified method implementations**: 16 de-identification methods ranging from classical approaches to state-of-the-art generative models.
3. **Flexible inference pipeline**: YAML configuration file-driven, with CLI parameters capable of overriding any configuration value.
4. **Systematic evaluation protocol**: Covering privacy, attribute preservation, and quality evaluation dimensions.

### BaseDeIdentifier Abstract Base Class

All methods inherit from a unified `BaseDeIdentifier` abstract base class providing a standardized interface:

- `process_frame(frame, face_bbox)`: Applies de-identification to the facial region of a single frame.
- `process_batch(frames, face_bboxes)`: Native batch processing support.
- `get_name()` / `get_config()`: Method metadata.
- Factory function `get_deidentifier(config)` for automatic instantiation via configuration dictionary.

### 16 De-Identification Methods (Four Categories)

| Category | Method | Identifier | Mechanism |
|----------|--------|-----------|-----------|
| Naive | Gaussian Blur | blur | Gaussian blurring of the face region |
| Naive | Pixelation | pixelate | Pixelation (mosaic) of the face region |
| Naive | Black Mask | mask | Black mask overlaid on the face |
| Generative | CIAGAN | ciagan | Conditional identity anonymization GAN |
| Generative | AMT-GAN | amtgan | Adversarial makeup transfer GAN |
| Generative | Adv-Makeup | advmakeup | Adversarial makeup generation |
| Generative | WeakenDiff | weakendiff | Diffusion model-based identity weakening |
| Generative | DeID-rPPG | deid_rppg | De-identification preserving rPPG signals |
| Generative | G2Face | g2face | Generative face replacement |
| Adversarial | PGD | pgd | Projected Gradient Descent adversarial perturbation |
| Adversarial | MI-FGSM | mifgsm | Momentum Iterative FGSM |
| Adversarial | TI-DIM | tidim | Translation-Invariant Diverse Input Method |
| Adversarial | TI-PIM | tipim | Translation-Invariant Patch Input Method |
| Adversarial | Chameleon | chameleon | Natural adversarial perturbation |
| K-Same | k-Same-Average | average | k-nearest neighbor face averaging |
| K-Same | k-Same-Select | select | k-nearest neighbor selection and replacement |
| K-Same | k-Same-Furthest | furthest | k-furthest neighbor replacement |

### Three-Dimensional Evaluation Framework

**Privacy Protection Dimension**:
- Verification Accuracy: Face verification accuracy (lower is better, indicating successful identity concealment)
- TAR@FAR: True acceptance rate at a given false acceptance rate
- PSR (Privacy Success Rate): Rate of successful privacy protection
- Evaluation models: ArcFace, CosFace, AdaFace

**Attribute Preservation Dimension**:
- Age: MAE (Mean Absolute Error)
- Gender: Classification accuracy
- Expression: Classification accuracy
- Facial landmarks: NME (Normalized Mean Error)
- Ethnicity: Classification accuracy
- rPPG: Heart rate MAE and RMSE

**Visual Quality Dimension**:
- Reference-based metrics: PSNR, SSIM, LPIPS
- Reference-free distribution metric: FID
- Reference-free quality metric: NIQE

## Key Experimental Results

### Table 1: Privacy Protection Evaluation (LFW Dataset)

| Method | Category | ArcFace Acc↓ | CosFace Acc↓ | AdaFace Acc↓ | PSR↑ |
|--------|----------|-------------|-------------|-------------|------|
| Original | - | 99.8 | 99.7 | 99.8 | 0.0 |
| Blur | Naive | 56.2 | 58.1 | 55.8 | 87.4 |
| Pixelate | Naive | 62.4 | 63.7 | 61.9 | 79.3 |
| Mask | Naive | 50.1 | 50.3 | 50.0 | 99.6 |
| CIAGAN | Generative | 53.8 | 55.2 | 54.1 | 91.2 |
| AMT-GAN | Generative | 58.7 | 60.3 | 57.9 | 82.6 |
| WeakenDiff | Generative | 51.4 | 52.8 | 51.1 | 96.8 |
| G2Face | Generative | 52.1 | 53.5 | 51.8 | 95.3 |
| PGD | Adversarial | 67.3 | 69.1 | 66.8 | 64.5 |
| Chameleon | Adversarial | 61.8 | 63.4 | 60.9 | 76.2 |
| k-Same-Avg | K-Same | 55.4 | 57.2 | 54.9 | 89.1 |
| k-Same-Furthest | K-Same | 53.1 | 54.8 | 52.7 | 93.5 |

Generative methods (WeakenDiff, G2Face) approach Black Mask in privacy protection, whereas the latter entirely destroys visual information. Adversarial methods offer comparatively weaker privacy protection.

### Table 2: Visual Quality vs. Attribute Preservation Trade-off (AgeDB / CelebA-HQ)

| Method | FID↓ | SSIM↑ | LPIPS↓ | Age MAE↓ | Gender Acc↑ | Landmark NME↓ |
|--------|------|-------|--------|----------|-------------|---------------|
| Blur | 142.5 | 0.71 | 0.38 | 8.2 | 78.4 | 12.3 |
| Pixelate | 156.8 | 0.65 | 0.42 | 9.1 | 75.6 | 14.7 |
| Mask | 198.3 | 0.52 | 0.56 | 15.3 | 62.1 | N/A |
| CIAGAN | 78.4 | 0.82 | 0.21 | 4.5 | 89.3 | 5.8 |
| AMT-GAN | 85.2 | 0.79 | 0.24 | 5.1 | 87.8 | 6.2 |
| WeakenDiff | 62.1 | 0.86 | 0.17 | 3.8 | 91.5 | 4.6 |
| G2Face | 58.7 | 0.88 | 0.15 | 3.4 | 92.1 | 4.2 |
| PGD | 45.2 | 0.93 | 0.08 | 2.1 | 95.8 | 2.8 |
| Chameleon | 52.3 | 0.91 | 0.11 | 2.6 | 94.2 | 3.1 |
| k-Same-Avg | 95.6 | 0.76 | 0.28 | 5.8 | 84.7 | 7.4 |
| k-Same-Furthest | 108.3 | 0.73 | 0.31 | 6.5 | 82.3 | 8.1 |

Key findings:
- **Privacy–attribute preservation trade-off**: Adversarial methods (PGD, Chameleon) best preserve attributes but provide the weakest privacy protection; Black Mask offers the strongest privacy protection but completely eliminates attribute information.
- **Generative methods achieve the best balance**: WeakenDiff and G2Face attain the optimal trade-off between privacy protection (PSR > 95%) and attribute preservation (Age MAE < 4, Gender Acc > 91%).
- **Naive and K-Same methods**: Exhibit inferior visual quality (FID > 95) with moderate attribute preservation.
- **Value of unified evaluation**: Only under consistent conditions do the true strengths and weaknesses of each method category become apparent; advantages claimed by certain methods in their original papers no longer hold under unified evaluation.

## Highlights & Insights

- **Unified abstract interface**: The `BaseDeIdentifier` base class is cleanly designed (`process_frame` + `process_batch`); integrating a new method requires only ~30 lines of code, making the toolbox highly extensible.
- **YAML configuration-driven**: All experiments are fully specified through a single YAML file, with CLI overrides for any parameter, ensuring complete reproducibility.
- **First unified three-dimensional evaluation**: Privacy protection, attribute preservation, and visual quality have never previously been systematically compared within a single framework using consistent evaluation models and data splits.
- **Broad coverage**: 16 methods spanning four categories, 6 datasets covering diverse downstream tasks, and 8 evaluation dimensions (privacy + 5 attributes + rPPG + quality) constitute the most comprehensive FDeID benchmark to date.
- **Lightweight dependencies**: Pure PyTorch implementation with no complex C++ extensions or conflicting frameworks, lowering the barrier to adoption.
- **Factory pattern design**: Methods can be switched via a configuration dictionary, facilitating large-scale automated experimentation.

## Limitations & Future Work

- **Technical report nature**: As a toolbox paper, methodological innovation is limited; the core contribution lies in engineering integration and standardization.
- **Incomplete method uploads**: The GitHub repository indicates that K-Same and certain generative methods are still being uploaded, and completeness remains to be verified.
- **Limited dataset scale**: Datasets such as LFW are relatively small with limited facial diversity; generalization to large-scale real-world scenarios requires further validation.
- **No video-level evaluation**: Although the interface supports frame-by-frame processing, temporal consistency evaluation for video de-identification is absent.
- **No formal privacy guarantees**: Evaluation relies solely on empirical metrics (face verification) without theoretical privacy guarantees such as differential privacy.
- **Limited diffusion model methods**: Only WeakenDiff is diffusion model-based; numerous recently proposed diffusion-based de-identification methods have not been incorporated.

## Related Work & Insights

- **Traditional de-identification**: Naive methods such as Gaussian Blur, Pixelation, and Black Mask are simple and effective but offer poor attribute preservation, serving as long-standing baselines.
- **K-Same family**: k-Same-Pixel, k-Same-Select, and k-Same-Furthest aggregate faces in feature space based on k-anonymity principles, providing theoretical privacy guarantees but limited visual quality.
- **Generative methods**: CIAGAN (conditional identity anonymization GAN), AMT-GAN (adversarial makeup transfer), DeepPrivacy (GAN inpainting), and FALCO (attention-guided conditional generation) offer high visual quality but involve complex training.
- **Adversarial perturbation methods**: PGD, MI-FGSM, TI-DIM, and others add imperceptible pixel-level perturbations to deceive recognition models; they induce minimal visual change but their privacy protection is model-dependent, limiting generalizability.
- **Diffusion model methods**: WeakenDiff, RiDDLE, and others leverage the generative capacity of diffusion models to achieve high-quality de-identification and represent a current research focus.
- **FDeID-Toolbox positioning**: Rather than proposing new methods, this work unifies the implementations and evaluations of existing methods, filling the gap left by the absence of a standardized benchmark in the field.

## Rating

- **Novelty**: ⭐⭐⭐ — A toolbox/benchmark paper with limited methodological innovation, though the unified evaluation framework design constitutes a clear contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 16 methods × 6 datasets × three-dimensional evaluation with strong diversity across method categories.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, detailed description of modular design, and abundant code examples.
- **Value**: ⭐⭐⭐⭐ — Provides the FDeID field with a much-needed standardized evaluation platform; has strong potential to become the standard benchmark tool for this research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification](pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)
- [\[CVPR 2026\] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)

</div>

<!-- RELATED:END -->
