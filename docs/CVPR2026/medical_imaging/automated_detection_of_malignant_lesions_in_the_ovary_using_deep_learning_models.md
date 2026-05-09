---
title: >-
  [Paper Note] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI
description: >-
  [CVPR 2026][Medical Imaging][Ovarian cancer detection] This paper systematically compares 15 variants across four major CNN families — LeNet, ResNet, VGG, and Inception — for ovarian cancer histopathology image classification. InceptionV3-ReLU is selected as the final model (average metrics ~94%), and three XAI methods — LIME, SHAP, and Integrated Gradients — are applied to provide interpretability for the classification results.
tags:
  - CVPR 2026
  - Medical Imaging
  - Ovarian cancer detection
  - CNN classification
  - Explainable AI
  - Histopathology
  - InceptionV3
date: 2026-05-08
content_hash: e73f7760f5d1569b
---

# Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**Conference**: CVPR 2026
**arXiv**: [2603.11818](https://arxiv.org/abs/2603.11818)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Ovarian cancer detection, CNN classification, Explainable AI, Histopathology, InceptionV3

## TL;DR

This paper systematically compares 15 variants across four major CNN families — LeNet, ResNet, VGG, and Inception — for ovarian cancer histopathology image classification. InceptionV3-ReLU is selected as the final model (average metrics ~94%), and three XAI methods — LIME, SHAP, and Integrated Gradients — are applied to provide interpretability for the classification results.

## Background & Motivation

Ovarian cancer is the 7th most common cancer among women worldwide and one of the deadliest gynecological malignancies. Unlike breast cancer (mammography/CBE) and cervical cancer (Pap smear), no reliable early screening method currently exists for ovarian cancer. Existing detection modalities include:

**Transvaginal ultrasound**: Limited sensitivity and high false-positive rate

**CA-125 blood test**: Poor specificity; elevated levels can result from non-cancerous conditions

**Tissue biopsy**: The gold standard for diagnosis, but invasive and time-consuming

**Root Cause**: **A non-invasive, rapid, and accurate detection method is needed to reduce delayed diagnosis rates for ovarian cancer.** While deep learning has demonstrated strong performance in medical imaging-assisted diagnosis in recent years, its application in the ovarian cancer domain remains relatively limited.

**Starting Point**: On the publicly available Mendeley ovarian cancer histopathology dataset, this paper systematically compares multiple CNN variants, identifies the optimal model, and applies explainable AI (XAI) to reveal the model's decision rationale, thereby enhancing clinical trustworthiness.

## Method

### Overall Architecture

The overall pipeline consists of: dataset acquisition → data augmentation → tensor conversion and normalization → training and evaluation of 15 CNN variants → optimal model selection → XAI interpretability analysis (LIME + SHAP + Integrated Gradients).

The dataset is the OvarianCancer&SubtypesDatasetHistopathology from Mendeley, comprising 5 classes: Clear Cell, Endometrioid, Mucinous, Non-Cancerous, and Serous, with a total of 498 original images.

### Key Designs

1. **Data Augmentation Strategy**:

    - **Function**: Expands the dataset from 498 to 2,490 images while maintaining inter-class balance.
    - **Mechanism**: Composite augmentation using the Albumentations library — rotation (up to 180°), horizontal/vertical flipping, and random variation of brightness, contrast, saturation, and hue; 4 augmented images are generated per original image.
    - **Design Motivation**: The original dataset is extremely small (~100 images per class), insufficient for training deep CNNs. The randomized probability parameters in Albumentations provide greater diversity than fixed transformations.

2. **Tensor Conversion and Normalization**:

    - **Function**: Converts augmented images to TensorFlow tensors, normalizing pixel values from $[0, 255]$ to $[0, 1]$.
    - **Mechanism**: Uses `image_dataset_from_directory()` with label mode set to `int` (non-one-hot) and an 80/20 random train/test split.
    - **Design Motivation**: float32 with $[0,1]$ normalization accelerates convergence in convolutional operations; integer labels facilitate future class extensions.

3. **Systematic Evaluation of 15 CNN Variants**:

    - **Function**: Covers a total of 15 models across LeNet (3 variants), ResNet (4 variants), VGG (4 variants), and Inception (4 variants).
    - LeNet variants: baseline (lr=0.001) → +Dropout → +Step Decay.
    - ResNet variants: ResNet-34 (32×32), ResNet-34 (224×224), ResNet-50, ResNet-101; hyperparameters optimized via random search over learning rate and dropout rate.
    - VGG variants: VGG16-A/B/C (using ReLU/tanh/+lr+dropout respectively) and VGG19; all employ transfer learning (frozen convolutional layers, only fully connected layers trained).
    - Inception variants: V1-A/B (ReLU/tanh) and V3-A/B (+BatchNorm/ReLU and tanh).
    - **Design Motivation**: Systematic comparison to identify the optimal architecture for this specific dataset.

4. **XAI Interpretability Analysis**:

    - **Function**: Applies three XAI methods to the selected InceptionV3-A model.
    - **LIME**: Generates locally interpretable superpixel-level feature importance maps, limited to displaying the 10 most important features.
    - **Integrated Gradients**: Integrates gradients from a baseline input to the actual input, producing pixel-level attribution maps.
    - **SHAP**: Provides local explanations based on Shapley values, visualizing positive/negative contributions of each pixel to each class prediction.
    - **Design Motivation**: VGG's transfer learning structure is difficult to analyze with XAI, whereas InceptionV3 trained from scratch is more amenable to gradient-based XAI methods.

### Loss & Training

- All models use **Softmax** activation in the output layer: $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}$
- Loss function: Categorical cross-entropy
- ResNet hyperparameter search: 10 random samples drawn from lr $\in [0.0001, 0.1]$ and dropout $\in [0.0, 0.9]$, each run for 3 epochs to select the optimal configuration
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC curve, and AUC are uniformly reported

## Key Experimental Results

### Main Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LeNet-A | 61.85% | 62.20% | 61.85% | 61.96% |
| ResNet-34 (224) | 57.03% | 59.39% | 57.03% | 57.70% |
| ResNet-50 | 34.14% | 47.75% | 34.14% | 33.47% |
| VGG16-A (transfer) | **96.99%** | 96.98% | 96.99% | 96.97% |
| VGG19 (transfer) | **97.19%** | **97.31%** | **97.19%** | **97.20%** |
| InceptionV3-A | 94.58% | 94.75% | 94.58% | 94.62% |
| InceptionV1-A | 78.92% | 81.58% | 78.92% | 79.33% |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| VGG16-O (Kasture et al.) | 50% (original) / 84.64% (augmented 20k) | Baseline comparison paper |
| VGG16-A (Ours) | 77.78% (original) / 96.99% (augmented) | Substantial gains from tensor conversion + normalization |
| InceptionV3-A (original data) | 20.20% | Training from scratch on small data performs poorly |
| InceptionV3-A (augmented data) | 94.58% | Significant improvement after augmentation |
| ReLU vs. tanh (InceptionV3) | 94.58% vs. 82.13% | ReLU substantially outperforms tanh |
| ReLU vs. tanh (InceptionV1) | 78.92% vs. 85.74% | tanh performs better in V1 |

### Key Findings

- VGG models achieve the best performance under transfer learning (~97%), but the encapsulated nature of transfer learning makes in-depth XAI analysis difficult.
- InceptionV3-A, trained from scratch, achieves 94.58% on the augmented dataset, balancing accuracy with interpretability.
- Data augmentation is the decisive factor for CNN training on small datasets — InceptionV3 accuracy jumps from 20.20% to 94.58%.
- All three XAI methods identify overlapping critical feature regions on the same samples, validating the consistency of the classification rationale.

## Highlights & Insights

- **Systematic Comparison**: The exhaustive evaluation of 15 model variants is relatively comprehensive within the ovarian cancer histopathology domain.
- **Triangulated XAI Validation**: The concurrent application of LIME, SHAP, and Integrated Gradients with cross-method comparison strengthens the credibility of interpretability conclusions.
- **Principled Model Selection Trade-off**: Rather than purely optimizing for accuracy (VGG19 = 97.19%), the paper selects InceptionV3 by also considering XAI compatibility, reflecting a practically-oriented perspective.

## Limitations & Future Work

- **Extremely Small Dataset**: Only 2,490 augmented images (498 originals); a substantial gap from clinical scale raises concerns about generalizability.
- **No External Validation**: All experiments are conducted on a single dataset without evaluation on an independent test set or multi-center data.
- **Outdated Architectures**: Modern architectures such as Vision Transformers, EfficientNet, and ConvNeXt are not explored.
- **No Clinical Comparison**: Model performance is not compared against diagnostic accuracy of professional pathologists.
- **Qualitative XAI Analysis**: The three XAI methods are compared only visually, without quantitative consistency metrics (e.g., IoU, correlation coefficients).
- **Single Data Source**: The image quality and staining protocols of the Mendeley dataset may not represent the diversity encountered in real clinical settings.

## Related Work & Insights

- Kasture et al. achieved 84.64% on the same dataset using VGG16; this paper's VGG16-A reaches 96.99%, with the improvement primarily attributable to preprocessing (tensor conversion + normalization).
- Wang et al.'s MRI-based ovarian tumor differentiation (87% accuracy) uses a different modality but proposes the concept of AI-assisted diagnosis for primary-care physicians.
- Hsu et al.'s ensemble CNN on ultrasound (ResNet-18/50 + Xception) proposes an 80–100% confidence threshold as a standard for clinical deployment.

## Rating
- **Novelty**: ⭐⭐ No methodological innovation; a standard combination of CNN classification and XAI.
- **Experimental Thoroughness**: ⭐⭐⭐ The systematic comparison of 15 variants is comprehensive, but the dataset is too small and no external validation is performed.
- **Writing Quality**: ⭐⭐⭐ Structure is clear, but some details are redundant and mathematical descriptions lean toward introductory textbook style.
- **Value**: ⭐⭐⭐ Provides a useful reference for ovarian cancer CAD, but remains a considerable distance from clinical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)
- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_hi.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)
- [\[CVPR 2026\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiationinduced_cont.md)

<!-- RELATED:END -->
