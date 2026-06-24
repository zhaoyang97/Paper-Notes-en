---
title: >-
  [Paper Note] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI
description: >-
  [CVPR2025][Medical Imaging][Ovarian Cancer] Detects ovarian cancer and its subtypes on histopathology images using 15 CNN variants (LeNet, ResNet, VGG, Inception), selects InceptionV3 (ReLU) as the optimal model (average 94.58% accuracy), and interprets model predictions using three XAI methods: LIME, SHAP, and Integrated Gradients.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Ovarian Cancer"
  - "CNN"
  - "Explainable AI"
  - "LIME"
  - "SHAP"
  - "Integrated Gradients"
date: 2026-05-08
content_hash: 76aa6ef341fc3d4b
---

# Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**Conference**: CVPR2025  
**arXiv**: [2603.11818](https://arxiv.org/abs/2603.11818)  
**Code**: Undisclosed  
**Area**: Medical Imaging  
**Keywords**: Ovarian Cancer, CNN, Explainable AI, LIME, SHAP, Integrated Gradients

## TL;DR

Detects ovarian cancer and its subtypes on histopathology images using 15 CNN variants (LeNet, ResNet, VGG, Inception), selects InceptionV3 (ReLU) as the optimal model (average 94.58% accuracy), and interprets model predictions using three XAI methods: LIME, SHAP, and Integrated Gradients.

## Background & Motivation

**High lethality of ovarian cancer**: It is the 7th most common cancer among women globally. Due to the lack of early screening methods, it is usually detected at late stages and possesses a high metastasis rate.

**Limitations of existing detection methods**: Transvaginal ultrasound and CA-125 blood tests lack sufficient accuracy, and definitive diagnosis relies on invasive biopsies.

**Potential of deep learning in cancer detection**: It has been successfully applied in fields such as breast and cervical cancer, but research in the field of ovarian cancer is relatively limited.

**Black-box problem**: Explainability of medical AI decisions is crucial for clinical acceptance.

**Dataset availability**: The OvarianCancer&Subtypes histopathological dataset provided by Mendeley contains five categories.

**Limitations of prior work**: Kasture et al. achieved only 84.64% accuracy using VGG16, and relied on a larger augmented dataset (24,742 images).

## Method

### Overall Architecture

Data augmentation → Tensor conversion & normalization → Training & comparison of 15 CNN variants → Selection of InceptionV3-A → XAI explainability analysis.

### Key Designs

- **Data Augmentation**: The Albumentations library was used for rotation (up to 180°), horizontal/vertical flips, and brightness/contrast/saturation/hue adjustments, expanding the dataset from 498 images to 2,490 images (498 per category for 5 categories).
- **Model Selection**: Tested 3 LeNet variants, 4 ResNet variants (34/50/101), 4 VGG variants (16-A/B/C, 19), and 4 Inception variants (V1-A/B, V3-A/B).
- **InceptionV3-A Architecture**: Adds Batch Normalization to InceptionV1, replaces 7×7 convolutions with 3×3 convolutions, modifies the Inception module filter configuration, uses ReLU activation, and a Softmax output layer.
- **VGG uses transfer learning** while Inception is trained from scratch—although VGG achieves the highest accuracy, frozen layers from transfer learning make XAI interpretation extremely difficult; therefore, InceptionV3 trained from scratch is selected to facilitate XAI analysis.
- **XAI Methods**: LIME (Local Interpretable Model-agnostic Explanations, restricted to showing 10 key features), Integrated Gradients (gradient integration attribution), and SHAP (a local variant of Shapley values).
- **Hyperparameter Tuning**: The ResNet series adopts a random search strategy, randomly sampling 10 sets of parameters in the range of learning rate [0.0001, 0.1] and dropout [0.0, 0.9].

### Loss & Training

Categorical cross-entropy loss, with Softmax output for 5 classes: $\text{softmax}(z)_i = e^{z_i} / \sum_{j=1}^{N} e^{z_j}$

## Key Experimental Results

### Model Performance Comparison (Top 5, Augmented Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| VGG19 | **97.19%** | **97.31%** | **97.19%** | **97.20%** |
| VGG16-A | 96.99% | 96.98% | 96.99% | 96.97% |
| VGG16-B | 96.18% | 96.27% | 96.18% | 96.20% |
| VGG16-C | 96.18% | 96.32% | 96.18% | 96.18% |
| **InceptionV3-A** | **94.58%** | **94.75%** | **94.58%** | **94.62%** |

### Comparison with Prior Work

| Model | Original Dataset | Augmented Dataset |
|------|-----------|-----------|
| VGG16-O (Kasture et al.) | 50% | 84.64% (20 epoch, 24742 images) |
| VGG16-A (Ours) | 77.78% | 96.99% (80 epoch, 2490 images) |
| InceptionV3-A (Ours) | 20.20% | 94.58% (80 epoch, 2490 images) |

### Key Findings

- VGG variants achieve the highest performance but are discarded because transfer learning hindered XAI interpretation.
- InceptionV3 trained from scratch achieves only 20.20% on the small original dataset, but its performance improves significantly after augmentation.
- The three XAI methods show consistent key feature regions on the "Serous" class, validating the rationality of the model's decisions.
- The differences between LIME and SHAP/IG arise because LIME restricts the number of displayed key features (only 10), rather than conflicting model explanations.

## Highlights & Insights

1. **Systematic Model Comparison**: A comprehensive horizontal comparison of 15 CNN variants, covering classical to modern architectures.
2. **Cross-Validation with Multiple XAI Methods**: Comparative analysis of three complementary methods (LIME, SHAP, and Integrated Gradients) enhances credibility.
3. **Data Efficiency**: Surpasses the accuracy of prior work using only 2,490 augmented images, compared to the 24,742 images used previously.
4. **Explainability-Aware Model Selection**: Rather than blindly choosing the model with the highest accuracy, XAI compatibility was taken into consideration.

## Limitations & Future Work

1. **Extremely Small Dataset**: The original dataset has only 498 images (~100 per class), and even when augmented to 2,490 images, it remains far below clinical requirements, questioning the generalizability of the conclusions.
2. **Limited Methodological Innovation**: Purely model selection and hyperparameter tuning, with no new architectures or methods proposed.
3. **Lack of External Validation**: Evaluated only via an 80-20 split on a single Mendeley dataset, without cross-dataset or cross-institution generalization testing.
4. **No Non-Invasive Data Used**: The claimed goal is non-invasive detection, but histopathologic images (requiring surgery/biopsy to obtain) are used, contradicting the motivation.
5. **Abnormally Poor ResNet Performance** (ResNet-50 achieves only 34.14%); the reason is not thoroughly analyzed, and the random search with only 10 iterations and 3 epochs of training for hyperparameter selection may be highly insufficient.
6. **Fairness of VGG via Transfer Learning vs. Inception Trained From Scratch**: The comparison baseline is inconsistent, as transfer learning yields natural advantages.
7. **Lack of Comparison with Modern Methods**: No comparisons are made with newer methods such as ViTs, Swin Transformers, or CLIP.
8. **Coarse Hyperparameter Search Strategy**: ResNet hyperparameter selection relies on only 10 random samples and 3 training epochs, which is insufficient to find reasonable configurations.
9. **Shallow XAI Analysis**: Shows only visualizations of individual samples, lacking quantitative faithfulness evaluations.

## Related Work & Insights

- **Ovarian Cancer AI Detection**: Zhou et al. reviewed the applications of AI in ovarian cancer diagnosis; FaRe-ConvNN by Hema et al. achieved 97% accuracy.
- **CNN Classification Architectures**: LeNet-5, ResNet, VGGNet, and GoogLeNet/Inception each have distinct characteristics.
- **XAI Methods**: LIME (Ribeiro 2016), SHAP (Lundberg 2017), and Integrated Gradients (Sundararajan 2017).
- **Prior Work on the Same Dataset**: Kasture et al. achieved 84.64% on the augmented data using VGG16.
- **OCT Research**: Schwartz et al. used OCT recordings + LSTM for ovarian cancer detection (AUC 0.81), providing an alternative pathway for non-invasive detection.

## Rating

- Novelty: ⭐⭐⭐⭐ (Standard CNN comparison experiments + XAI application, lacking methodological innovation)
- Experimental Thoroughness: ⭐⭐⭐ (Dataset is too small, lacks external validation, abnormal ResNet performance is not analyzed)
- Writing Quality: ⭐⭐⭐⭐ (Complete structure but redundant narratives)
- Value: ⭐⭐⭐ (As an applied work, it basically achieves its goals, but remains far from actual clinical deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2025\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2025\] Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)
- [\[ICML 2025\] Efficient Noise Calculation in Deep Learning-based MRI Reconstructions](../../ICML2025/medical_imaging/efficient_noise_calculation_in_deep_learning-based_mri_reconstructions.md)

</div>

<!-- RELATED:END -->
