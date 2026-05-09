---
title: >-
  [Paper Note] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI
description: >-
  [CVPR 2026][Medical Imaging][Ovarian Cancer Detection] This work systematically compares 15 CNN variants (LeNet/ResNet/VGG/Inception) on five-class classification of ovarian cancer histopathology images. InceptionV3-A (ReLU) is selected as the final model, achieving 94% across comprehensive metrics, with comparative explainability analysis conducted using three XAI methods: LIME, SHAP, and Integrated Gradients.
tags:
  - CVPR 2026
  - Medical Imaging
  - Ovarian Cancer Detection
  - CNN Comparison
  - Explainable AI
  - Histopathology
  - InceptionV3
date: 2026-05-08
content_hash: bba9e8c1467a642d
---

# Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**Conference**: CVPR 2026
**arXiv**: [2603.11818](https://arxiv.org/abs/2603.11818)
**Code**: None
**Area**: Medical Image Classification / Explainable AI
**Keywords**: Ovarian Cancer Detection, CNN Comparison, Explainable AI, Histopathology, InceptionV3

## TL;DR

This work systematically compares 15 CNN variants (LeNet/ResNet/VGG/Inception) on five-class classification of ovarian cancer histopathology images. InceptionV3-A (ReLU) is selected as the final model, achieving 94% across comprehensive metrics, with comparative explainability analysis conducted using three XAI methods: LIME, SHAP, and Integrated Gradients.

## Background & Motivation

**Background**: Ovarian cancer is the 7th most common cancer among women worldwide and carries an extremely high mortality rate. A core challenge is the lack of effective early screening methods — unlike breast cancer (mammography) or cervical cancer (Pap smear), ovarian cancer can only be confirmed through invasive biopsy. Deep learning has shown progress in detecting various cancers, yet DL-based approaches for ovarian cancer remain limited.

**Limitations of Prior Work**: (1) Existing non-invasive detection methods (transvaginal ultrasound, CA-125 blood test, pelvic examination) lack sufficient accuracy to serve as reliable screening tools; (2) Definitive diagnosis relies on biopsy, which is invasive and time-consuming; (3) Existing DL approaches mostly employ single models, lacking systematic multi-architecture comparison and XAI-based explainability support, leading to low clinical trust and adoption.

**Key Challenge**: An automated detection system must simultaneously achieve high accuracy and interpretability; however, available histopathology datasets are extremely small (only 498 images), and high-performing models (VGG with transfer learning) are difficult to explain effectively via XAI due to frozen pretrained feature layers.

**Goal**: To build a high-accuracy classification model on a small-scale ovarian cancer histopathology dataset and provide transparent support for clinical decision-making through XAI.

**Key Insight**: Cast a wide net by comparing 15 CNN variants, selecting models based on both accuracy and XAI feasibility rather than solely pursuing peak performance.

**Core Idea**: Model selection should consider interpretability alongside accuracy — InceptionV3 (trained from scratch) is preferred over the higher-accuracy VGG (transfer learning) because the former is more amenable to effective XAI explanation.

## Method

### Overall Architecture

Acquire 5-class histopathology images (Clear Cell, Endometrioid, Mucinous, Non-Cancerous, Serous) totaling 498 images from the Mendeley dataset → Apply Albumentations-based data augmentation to expand to 2,490 images → Systematically train 15 CNN variants → Select the best model (InceptionV3-A) through comprehensive evaluation → Apply three XAI methods (LIME/SHAP/IG) for comparative explainability analysis.

### Key Designs

1. **Data Augmentation Pipeline**
   - Albumentations library is used for rotations (up to 180°), horizontal/vertical flips, and random brightness/contrast/saturation/hue transformations.
   - Each original image generates 4 augmented copies, expanding the dataset from 498 to 2,490 images (~498 per class, maintaining class balance).
   - After conversion to tensors, RGB values are normalized from 0–255 to 0–1, significantly improving training stability.
   - Random 80:20 train-test split (1,992 training / 498 testing).

2. **Systematic Comparison of 15 CNN Variants**
   - **LeNet series** (3 variants): Base (lr=0.001) / +Dropout / +Step Decay, trained for 100 epochs.
   - **ResNet series** (4 variants): ResNet-34 at two resolutions (32×32 and 224×224), ResNet-50, and ResNet-101; optimal lr and dropout rate determined via random search (10 iterations × 3 epochs).
   - **VGG series** (4 variants): VGG16-A/B/C and VGG19, all using ImageNet transfer learning with frozen feature layers and only fully connected layers trained.
   - **Inception series** (4 variants): InceptionV1-A (ReLU) / V1-B (Tanh) / V3-A (ReLU + BatchNorm) / V3-B (Tanh + BatchNorm), all trained from scratch for 80 epochs.

3. **Comparative XAI Analysis Using Three Methods**
   - **LIME**: Locally interpretable; generates superpixel-level explanation maps (limited to 10 most important features), revealing local rationale behind predictions.
   - **Integrated Gradients**: Gradient attribution method that integrates gradients along the path from a baseline to the input, producing pixel-level importance maps.
   - **SHAP**: Shapley value-based attribution method quantifying the marginal contribution of each pixel to the prediction.
   - Highlighted regions produced by all three methods exhibit significant overlap, validating the consistency and reliability of black-box explanations.

### Loss & Training

Softmax output layer with cross-entropy loss. LeNet variants are trained for 100 epochs; ResNet variants use random search to determine optimal hyperparameters (lr range: 0.0001–0.1, dropout range: 0.0–0.9); VGG variants use ImageNet pretrained weights with frozen convolutional layers and only fully connected layers trained; Inception variants are trained from scratch for 80 epochs. The VGG series is excluded from final selection — despite achieving the highest accuracy, the frozen feature layers from transfer learning impede effective XAI application.

## Key Experimental Results

### Main Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| VGG19 (Transfer Learning) | 97.19% | 97.31% | 97.19% | 97.20% |
| VGG16-A (Transfer Learning) | 96.99% | 96.98% | 96.99% | 96.97% |
| **InceptionV3-A (Selected)** | **94.58%** | **94.75%** | **94.58%** | **94.62%** |
| InceptionV1-B | 85.74% | 86.26% | 85.74% | 85.42% |
| LeNet-A | 61.85% | 62.20% | 61.85% | 61.96% |
| ResNet-34 (224) | 57.03% | 59.39% | 57.03% | 57.70% |
| ResNet-50 | 34.14% | 47.75% | 34.14% | 33.47% |

Although the VGG series achieves the highest scores, it is excluded because the black-box feature layers from transfer learning render XAI-based explanation ineffective.

### Ablation Study

| Comparison Dimension | Conclusion |
|----------------------|------------|
| VGG16-A vs. VGG16-O (Kasture et al.) | Same dataset: 96.99% (ours) vs. 84.64% (theirs) on augmented data; 77.78% vs. 50% (+27.78 pp) on original data, attributed to tensor conversion and normalization. |
| InceptionV3-A (ReLU) vs. InceptionV3-B (Tanh) | 94.58% vs. 82.13%; ReLU significantly outperforms Tanh by 12.45 pp. |
| ResNet-34 32×32 vs. 224×224 | 43.78% vs. 57.03%; input resolution has a substantial impact on ResNet performance. |
| Three XAI Methods | LIME/SHAP/IG highlighted regions show consistent overlap, validating explanation reliability. |

### Key Findings

- Model complexity does not guarantee performance: ResNet-50/101 perform extremely poorly (34–43%), likely due to insufficient hyperparameter search and training epochs.
- Data preprocessing (tensor conversion + normalization) yields substantial gains on small datasets, exceeding the baseline work by 27 pp.
- XAI feasibility is an important dimension of model selection — a 2.4 pp accuracy trade-off in exchange for interpretability is considered worthwhile.

## Highlights & Insights

- The systematic comparison of 15 models provides valuable reference for CNN selection in medical imaging.
- Incorporating XAI feasibility into model selection — rather than solely maximizing accuracy — reflects deployment-oriented and clinical-trust-driven considerations.
- The comparative XAI analysis demonstrates the complementarity and consistency of LIME, SHAP, and IG.
- The data augmentation and normalization strategy on an extremely small dataset offers useful practical insights.

## Limitations & Future Work

- The dataset is extremely small (498 original → 2,490 augmented images); generalizability is questionable and no cross-validation is performed.
- Only classical CNN architectures are explored; ViT, medical pretrained models (e.g., BiomedCLIP), or more modern architectures are not investigated.
- Multi-center/multi-institution data validation and real-world clinical testing are absent.
- The ResNet series performs extremely poorly (34–57%); the hyperparameter search strategy (only 10 iterations × 3 epochs) is likely severely inadequate.
- Per-class ROC-AUC details (e.g., confusion between Clear Cell and Serous subtypes) are not reported.
- Augmentation strategies are limited to simple geometric and color transforms; more advanced techniques (e.g., MixUp, CutMix, Mosaic) are not explored.

## Related Work & Insights

- **vs. Kasture et al. (VGG16-O)**: On the same dataset, this work's VGG16-A achieves 96.99% vs. 84.64% on augmented data, demonstrating the value of data preprocessing.
- **vs. Hsu et al.**: Their work employs ensemble learning with ResNet-18/50/Xception for ultrasound-based ovarian cancer detection, achieving higher accuracy but relying on larger datasets.
- **vs. Wang et al.**: Their approach uses DL for MRI-based benign/malignant ovarian differentiation, reaching 87% accuracy, but on a different imaging modality.
- The augmentation strategy for medical small-data scenarios and the concept of "interpretability-oriented model selection" offer practical reference for similar work.

## Rating

- **Novelty**: ⭐⭐ All methodological components follow standard practices; no new architecture or technique is proposed.
- **Experimental Thoroughness**: ⭐⭐ Dataset is too small; generalization validation, cross-validation, and statistical significance testing are absent.
- **Writing Quality**: ⭐⭐⭐ Structure is clear, but some descriptions are redundant and equation numbering is not tightly integrated with the main text.
- **Value**: ⭐⭐ Provides some reference as an introductory medical AI study, but insufficient innovation and experimental depth to support a top-venue publication.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_hi.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_le.md)

<!-- RELATED:END -->
