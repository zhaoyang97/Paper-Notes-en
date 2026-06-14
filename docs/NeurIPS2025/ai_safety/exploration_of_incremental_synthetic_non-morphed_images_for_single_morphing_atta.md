---
title: >-
  [Paper Note] Exploration of Incremental Synthetic Non-Morphed Images for Single Morphing Attack Detection
description: >-
  [NeurIPS 2025 (LXAI Workshop)][AI Safety][S-MAD] This paper systematically investigates the effect of incrementally introducing synthetic non-morphed face images into Single Morphing Attack Detection (S-MAD) training. Re…
tags:
  - "NeurIPS 2025 (LXAI Workshop)"
  - "AI Safety"
  - "S-MAD"
  - "synthetic face data"
  - "incremental training"
  - "EfficientNet"
  - "MobileNet"
  - "cross-dataset generalization"
date: 2026-05-08
content_hash: d0be312de86b51bb
---

# Exploration of Incremental Synthetic Non-Morphed Images for Single Morphing Attack Detection

**Conference**: NeurIPS 2025 (LXAI Workshop)
**arXiv**: [2510.09836](https://arxiv.org/abs/2510.09836)  
**Code**: No public code  
**Area**: AI Security
**Keywords**: S-MAD, synthetic face data, incremental training, EfficientNet, MobileNet, cross-dataset generalization

## TL;DR
This paper systematically investigates the effect of incrementally introducing synthetic non-morphed face images into Single Morphing Attack Detection (S-MAD) training. Results show that a moderate proportion of synthetic data (~75% increment) can improve cross-dataset generalization (EER reduced from 6.17% to 6.10%), while excessive use or training exclusively on synthetic data leads to severe performance degradation (EER rising to ~38%).

## Background & Motivation
Face morphing attacks represent a critical security threat to biometric systems in the AI era: by blending facial images of two or more subjects, a morphed image can simultaneously match the biometric features of multiple individuals. Morphing Attack Detection (MAD) falls into two categories:
- **D-MAD** (Differential MAD): compares a live capture against a reference document image
- **S-MAD** (Single-image MAD): determines whether a single image is morphed without any reference

S-MAD faces a fundamental data dilemma: large-scale real bona fide face datasets are difficult to obtain due to privacy regulations, yet deep learning methods require substantial data. Synthetic data is a potential solution, but the optimal introduction strategy and proportion remain unclear.

## Core Problem
How can S-MAD generalization be enhanced by controllably incorporating synthetic non-morphed images into training? What is the optimal proportion of synthetic data? Is training exclusively on synthetic data viable?

## Method

### Dataset Composition

Three datasets are used:

| Dataset | Subjects | Bona fide | Morphs | Morphing Tools | Notes |
|--------|--------|-----------|-------|---------|------|
| FERET | 529 | 529×3 | 529×4×3 | FaceFusion, FaceMorpher, OpenCV, UBO | PS300/PS600/Resized |
| FRGCv2 | 533 | 984×3 | 964×4×3 | Same as above | PS300/PS600/Resized |
| SMDD | - | 15,000 | 25,000 | StyleGAN2-ADA | Fully synthetic |

Three image processing conditions are applied: print/scan at 300 dpi (PS300), print/scan at 600 dpi (PS600), and digital resizing. Four morphing tools span the quality range from low (FaceMorpher) to high (FaceFusion).

### Training Framework
- **Preprocessing**: MTCNN face alignment (scale factor 0.9, output 369×369), data augmentation (random flip/rotation/color jitter), ImageNet normalization
- **Models**: EfficientNet-B2 (2.9M parameters) and MobileNetV3-large (7.7M parameters), initialized with ImageNet pretrained weights
- **Optimizer**: Adam ($\beta_1=0.99$, $\beta_2=0.999$), learning rate $1\times10^{-5}$ (grid search optimal), batch size 64, 100 epochs
- **Loss function**: categorical cross-entropy

$$L_i = H(y_k, \hat{y_k}) = -\frac{1}{n}\sum_x (y_k) \log(\hat{y_k})$$

### Incremental Synthetic Data Strategy
Non-morphed samples $S_{(SMDD,j)}$ are randomly drawn from the SMDD subset and added to the training set at proportion $j$:

$$S_{(SMDD,j)} \subset D_{SMDD}, \quad |S_{(SMDD,j)}| = m < n$$

Three experimental configurations:
1. **No synthetic data**: FERET train / FRGCv2 test (and vice versa)
2. **Incremental addition**: SMDD non-morphed images added at 10%, 20%, 30%, 50%, 75%, and 100% proportions
3. **Synthetic-only training**: trained solely on SMDD, evaluated on real datasets

### Evaluation Metrics
- **MACER** (Morphing Attack Classification Error Rate): proportion of morphed samples misclassified as bona fide
- **BPCER** (Bona fide Presentation Classification Error Rate): proportion of genuine samples misclassified as morphed
- **D-EER** (Detection Equal Error Rate): operating point where MACER = BPCER
- **BPCER@5/10/20**: BPCER at MACER = 5%/10%/20%

## Key Experimental Results

### Train on FERET → Test on FRGCv2

| Model | Synthetic Ratio | Total Bona fide | D-EER(%) ↓ | BPCER5(%) ↓ | BPCER10(%) ↓ |
|------|------------|-------------|------------|-------------|-------------|
| MobileNetV3 | 0% | 1,587 | 6.17 | 1.42 | 3.59 |
| EfficientNet-B2 | 0% | 1,587 | 6.47 | 1.97 | 4.57 |
| EfficientNet-B2 | 50% | 2,387 | 6.09 | 1.83 | 3.70 |
| **MobileNetV3** | **75%** | **2,787** | **6.10** | **1.05** | **3.05** |
| **EfficientNet-B2** | **75%** | **2,787** | **6.09** | **1.39** | **4.20** |
| MobileNetV3 | 100% | 3,174 | 8.17 | 2.58 | 6.23 |
| EfficientNet-B2 | Synthetic only | 25,000 | 37.96 | 61.24 | 76.51 |
| MobileNetV3 | Synthetic only | 25,000 | 38.95 | 62.32 | 73.73 |

### Train on FRGCv2 → Test on FERET

| Model | Synthetic Ratio | D-EER(%) ↓ | BPCER5(%) ↓ |
|------|------------|------------|-------------|
| **EfficientNet-B2** | **10%** | **8.68** | **3.47** |
| **MobileNetV3** | **10%** | **10.20** | **3.66** |
| EfficientNet-B2 | 0% | 9.61 | 2.64 |
| EfficientNet-B2 | 75% | 14.05 | 8.95 |
| EfficientNet-B2 | Synthetic only | 37.57 | 57.54 |

### Key Findings

1. **Optimal synthetic ratio is dataset-dependent**: the optimal ratio is 75% for FERET→FRGCv2 and 10% for FRGCv2→FERET, indicating that the best proportion depends on the characteristics of the original training set.
2. **Excessive synthetic data is detrimental**: performance begins to decline at 100% (8.17% vs. 6.10% at 75%).
3. **Synthetic-only training fails severely**: EER spikes to 37–39%, reflecting a synthetic-to-real domain gap that causes generalization collapse.
4. **Lightweight architectures remain competitive**: both EfficientNet-B2 and MobileNetV3 achieve competitive performance under constrained parameter budgets, making them suitable for practical deployment (border inspection, mobile verification).

### Average EER Summary

| Configuration | EfficientNet-B2 $\overline{EER}$ | MobileNetV3 $\overline{EER}$ |
|------|--------------------------------|------------------------------|
| FERET→FRGCv2 (all experiments) | 6.86% | 6.93% |
| FRGCv2→FERET (all experiments) | 10.46% | 12.05% |

## Highlights & Insights
- **Systematic incremental study**: a complete proportion sweep from 0% to 100% and synthetic-only training provides clear practical guidance on synthetic data incorporation.
- **Rigorous cross-dataset evaluation**: training and testing on entirely different datasets (FERET vs. FRGCv2) yield a genuine assessment of generalization.
- **Deployment-oriented design**: lightweight architectures (2.9M–7.7M parameters) are selected with resource-constrained scenarios such as mobile and border-control applications in mind.
- **Broad morphing tool coverage**: four morphing tools of varying quality (FaceFusion/FaceMorpher/OpenCV/UBO) combined with GAN-generated samples cover the diversity of real-world attacks.

## Limitations & Future Work
- **Workshop length constraints**: certain experimental details and analytical depth are limited; in particular, quantitative analysis of synthetic data domain shift is absent.
- **Single synthetic data source**: only SMDD (StyleGAN2-ADA) is used; alternative generation methods such as diffusion models are not explored.
- **Absence of modern backbones**: Vision Transformers and more recent efficient architectures are not evaluated.
- **No privacy-utility quantification**: while privacy constraints motivate the work, the actual privacy benefit of varying synthetic proportions is not quantified.
- **Unpredictability of the optimal ratio**: the large variation in optimal proportion across datasets (10% vs. 75%) is not accompanied by theoretical guidance for prediction.
- **Limited data augmentation strategies**: print-scan simulation, compression artifact augmentation, and other operationally realistic techniques are not explored.
- **Impact of weak morphing tools**: FaceMorpher produces visually apparent artifacts that may reduce overall detection difficulty.

## Rating
- Novelty: ⭐⭐⭐⭐ The incremental synthetic data study is systematic, but the methodological contribution is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ The proportion sweep is comprehensive, but deeper analysis and ablation studies are lacking.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, though workshop length constraints limit depth.
- Value: ⭐⭐⭐⭐ Provides practical guidance on synthetic data usage for security applications.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[AAAI 2026\] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference](../../AAAI2026/ai_safety/diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)
- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)
- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)

</div>

<!-- RELATED:END -->
