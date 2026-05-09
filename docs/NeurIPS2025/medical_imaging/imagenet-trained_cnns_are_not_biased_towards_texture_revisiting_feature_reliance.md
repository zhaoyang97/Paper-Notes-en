---
title: >-
  [Paper Note] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
description: >-
  [NeurIPS 2025][Medical Imaging][texture bias] By proposing a systematic feature suppression framework—rather than cue-conflict experiments—this work re-evaluates the feature reliance of CNNs, finding that CNNs are not inherently texture-biased but instead rely primarily on local shape features; moreover, feature reliance patterns differ substantially across domains (CV/MI/RS).
tags:
  - NeurIPS 2025
  - Medical Imaging
  - texture bias
  - feature reliance
  - CNN
  - feature suppression
  - domain-specific analysis
date: 2026-05-08
content_hash: da5f6a292875ddea
---

# ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression

**Conference**: NeurIPS 2025
**arXiv**: [2509.20234](https://arxiv.org/abs/2509.20234)
**Code**: [GitHub](https://github.com/tomburgert/feature-reliance)
**Area**: Medical Imaging / Computer Vision
**Keywords**: texture bias, feature reliance, CNN, feature suppression, domain-specific analysis

## TL;DR

By proposing a systematic feature suppression framework—rather than cue-conflict experiments—this work re-evaluates the feature reliance of CNNs, finding that CNNs are not inherently texture-biased but instead rely primarily on local shape features; moreover, feature reliance patterns differ substantially across domains (CV/MI/RS).

## Background & Motivation

**Background**: The highly influential cue-conflict experiments of Geirhos et al. (2019) proposed that ImageNet-trained CNNs are inherently biased toward texture (the texture bias hypothesis), whereas humans rely more on shape. This conclusion has profoundly shaped subsequent research on model interpretability and robustness.

**Limitations of Prior Work**: The cue-conflict paradigm contains three critical methodological issues: (i) **features are not isolated**—color and local shape information are retained within texture cues; (ii) **texture signal overload**—texture covers not only the target region but also the background, creating spatially unbalanced signals; (iii) **human interface biases toward shape**—humans select categories via silhouette icons, potentially guiding shape-based responses.

**Key Challenge**: The widely accepted conclusion that "CNNs are texture-biased" may be an artifact of experimental design rather than an intrinsic property of CNNs.

**Goal**: To propose an unbiased, domain-agnostic framework for evaluating feature reliance.

**Key Insight**: Shifting from "conflict-based preference" to "suppression-based reliance"—quantifying dependence by systematically suppressing a single feature type and measuring the resulting performance degradation.

**Core Idea**: Feature reliance should be measured by the performance drop after removing the target feature, rather than by preference under conflicting stimuli.

## Method

### Overall Architecture

A domain-agnostic feature reliance evaluation framework is proposed. For each of the three feature types—shape, texture, and color—two complementary suppression transformations are applied. Model accuracy under suppression conditions, expressed as a ratio relative to baseline accuracy (relative accuracy), is used to quantify reliance on each feature type.

### Key Designs

1. **Feature Definitions and Suppression Transformations**:

    - **Function**: Two complementary suppression methods are designed for each of the three feature types.
    - **Design Motivation**: Using two transformations with distinct mechanisms prevents artifacts introduced by any single transformation from confounding the conclusions.
    - **How**: 
        - Shape suppression: Patch Shuffle (randomly shuffling patch positions) + Patch Rotation (rotating patches while preserving locality)
        - Texture suppression: Bilateral Filter (edge-preserving smoothing) + Gaussian Blur
        - Color suppression: Grayscale conversion + Channel Shuffle
    - **Novelty**: Unlike cue-conflict methods, no adversarial stimuli are generated; transformations are applied directly to original images.

2. **Quantitative Validation Metrics**:

    - **Function**: Four quantitative metrics verify that each transformation effectively suppresses the target feature without affecting others.
    - **Design Motivation**: Empirical evidence is required to demonstrate that suppression transformations are both effective and feature-selective.
    - **How**:
        - Texture metrics: Local Variance (LV) and High-Frequency Energy (HFE), expected to decrease.
        - Shape metrics: Edge-SSIM (ESSIM) and Gradient Correlation (GC), expected to be preserved.
    - **Key Result**: Bilateral Filter best preserves shape (ESSIM: 0.74, GC: 0.85) while effectively suppressing texture (LV: 0.54, HFE: 0.49).

3. **Human–Model Comparison Experiment Design**:

    - **Function**: Feature reliance is compared between humans and models under identical suppression conditions.
    - **Design Motivation**: This eliminates the shape-guiding bias in the human interface used by Geirhos et al.
    - **How**: 20 human participants selected categories via text labels (not silhouette icons); each trial consisted of a 300 ms fixation, 200 ms image presentation, and 200 ms pink-noise mask; each participant viewed only one suppression variant of each image.
    - **Novelty**: Alphabetically ordered text-label grids were used to avoid any shape-guided response.

### Cross-Domain Analysis

The same framework is applied across three domains (CV/MI/RS) using a fixed ResNet50 architecture. Suppression intensity is varied continuously to generate suppression curves, which are normalized for cross-dataset and cross-domain comparison.

## Key Experimental Results

### Main Results: Human vs. CNN Feature Reliance (Relative Accuracy)

| Model | Global Shape | Local Shape | Texture | Color | Original Accuracy |
|-------|-------------|-------------|---------|-------|-------------------|
| **Humans** | 0.965 | **0.763** | 0.979 | 0.999 | 0.969 |
| ResNet50-standard | 0.832 | **0.276** | 0.795 | 0.924 | 0.954 |
| ResNet50-sota | 0.943 | **0.618** | 0.867 | 0.948 | 0.931 |
| ConvNeXtV2 | 0.949 | **0.647** | 0.925 | 0.969 | 0.940 |
| ViT | 0.930 | **0.636** | 0.921 | 0.977 | 0.929 |
| CLIP ViT | 0.959 | **0.758** | 0.949 | 0.984 | 0.936 |

**Key Findings**: All CNNs exhibit the largest performance drop under **local shape suppression**, not texture suppression. This directly challenges the texture bias hypothesis.

### Ablation Study: Cross-Domain Feature Reliance Differences

| Domain | Most Relied-Upon Feature | Observations |
|--------|--------------------------|--------------|
| **Computer Vision** | Local shape | Shape suppression causes the largest performance drop; texture/color suppression has smaller impact |
| **Medical Imaging** | Color | Grayscale conversion leads to significant accuracy drops, reflecting the critical role of color in diagnosis |
| **Remote Sensing** | Texture | Texture suppression has the greatest impact; RS categories are largely defined by textural patterns (e.g., fields, residential areas) |

### Key Findings

- **Core Conclusion**: CNNs are not inherently texture-biased. ResNet50-standard retains 80% performance under texture suppression (close to the 83% retained under global shape suppression) but retains only 28% under local shape suppression.
- **Effect of Modern Training**: ResNet50-sota improves the retention rate under local shape suppression from 28% to 62%; ConvNeXtV2 further improves this to 65%.
- **Vision–Language Models Closest to Humans**: CLIP ViT most closely matches human performance across all suppression conditions (local shape retention: 0.758 vs. human 0.763).
- **ViTs Are Not More Shape-Biased Than CNNs**: The feature reliance profile of ViT is highly similar to that of ResNet50-sota.

## Highlights & Insights

- **Paradigm Shift**: Transitioning from "feature preference" (preference in conflict) to "feature reliance" (reliance through suppression) yields a conceptually cleaner framework.
- **Counter-Intuitive Finding**: The widely cited texture bias is in fact a misattribution of local shape dependence—the texture cues used in cue-conflict experiments contain embedded local shape information.
- **High Value of Domain Differentiation**: The feature reliance profiles of CV, MI, and RS are entirely distinct, indicating that a model's perceptual strategy is shaped by data and domain characteristics rather than fixed architectural biases.
- **Rigorous Experimental Design**: Using two complementary transformations per feature type with quantitative validation substantially enhances the credibility of the conclusions.

## Limitations & Future Work

- Feature suppression cannot achieve perfect isolation: some low-frequency texture information may persist after texture suppression, and shape suppression cannot eliminate all shape cues.
- Suppression transformations may introduce artifacts that independently affect model behavior (e.g., block-like structure from Patch Shuffle).
- Results for pretrained models may be influenced by the similarity between data augmentation strategies and suppression transformations (e.g., Cutout vs. Patch Shuffle).
- The human experiment was conducted under controlled conditions (brief exposure, limited categories) and may not fully reflect real-world visual perception.

## Related Work & Insights

- **Geirhos et al. (2019)**: The original work proposing the texture bias hypothesis, whose methodology is directly challenged in this paper.
- **Hermann et al. (2020)**: Demonstrated that texture bias stems primarily from training objectives and augmentation strategies rather than network architecture.
- **Insights**: When evaluating model behavior, implicit biases in experimental design can lead to fundamentally flawed conclusions. This lesson has broad relevance—standard evaluation protocols warrant more careful scrutiny of their underlying assumptions.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework is conceptually clear, though individual suppression transformations are not entirely novel; the contribution lies in their systematic combination and in challenging prevailing views.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Human–model comparison, multiple architectures, multiple domains, multiple datasets, and quantitative validation are all comprehensively covered.
- Writing Quality: ⭐⭐⭐⭐⭐ The argumentation is logically rigorous, and the critique of prior work is well-substantiated.
- Value: ⭐⭐⭐⭐⭐ Challenges one of the most influential hypotheses in the field; the cross-domain analysis is of substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)
- [\[NeurIPS 2025\] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)
- [\[NeurIPS 2025\] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM](unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](../../ICCV2025/medical_imaging/alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)

</div>

<!-- RELATED:END -->
