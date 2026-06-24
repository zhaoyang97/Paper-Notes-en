---
title: >-
  [Paper Note] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification
description: >-
  [CVPR2025][Medical Imaging][Brain Tumor Classification] Presents a brain tumor classification framework combining Non-Negative Matrix Factorization (NNMF) feature extraction, statistical feature selection, lightweight CNN classification, and diffusion-based feature space denoising. While maintaining ~85% clean accuracy, it improves robust accuracy under AutoAttack from 0.47% to 59.5%.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Brain Tumor Classification"
  - "NNMF"
  - "Diffusion Denoising"
  - "Adversarial Robustness"
  - "AutoAttack"
date: 2026-05-08
content_hash: fcf958e7523d931e
---

# Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification

**Conference**: CVPR2025  
**arXiv**: [2603.13182](https://arxiv.org/abs/2603.13182)  
**Code**: To be confirmed  
**Area**: Medical Images  
**Keywords**: Brain Tumor Classification, NNMF, Diffusion Denoising, Adversarial Robustness, AutoAttack

## TL;DR

Presents a brain tumor classification framework combining Non-Negative Matrix Factorization (NNMF) feature extraction, statistical feature selection, lightweight CNN classification, and diffusion-based feature space denoising. While maintaining ~85% clean accuracy, it improves robust accuracy under AutoAttack from 0.47% to 59.5%.

## Background & Motivation

- Brain tumor MRI classification is a critical task in computer-aided diagnosis; early and accurate detection helps improve treatment planning and patient survival rates.
- Although deep learning models can achieve high classification accuracy, they are highly vulnerable to adversarial perturbations. Minor, imperceptible modifications can collapse classification accuracy, posing a severe hazard in safety-critical scenarios like medical diagnosis.
- Although deep learning models are highly vulnerable to adversarial perturbations—minor, imperceptible modifications can collapse classification accuracy, posing a severe hazard in safety-critical scenarios like medical diagnosis.
- NNMF can decompose non-negative data into interpretable part-based representations, making it more suitable than PCA for naturally non-negative medical image pixel values.
- The denoising principles of diffusion models can be applied to purify adversarial perturbations in the feature space.
- **Core Motivation**: To combine interpretable, low-dimensional NNMF features with a diffusion denoising defense to construct a lightweight framework that balances both classification accuracy and adversarial robustness.

## Method

### Overall Architecture (Four-Stage Pipeline)
1. Data Preprocessing → 2. NNMF Feature Extraction + Statistical Selection → 3. Lightweight CNN Classification → 4. Diffusion-Based Feature Space Denoising Defense

### Stage 1: Data Preprocessing
- Data Source: Kaggle brain MRI dataset, approximately 2,200 images (Normal: 730 train / 219 val / 97 test; Tumor: 770 train / 210 val / 118 test), split into 70% / 20% / 10%.
- COCO annotation format is reorganized into a folder structure to suit the classification task.
- **Note**: Lacking patient ID information, the risk of slice-level leakage cannot be fully ruled out; the paper acknowledges that the reported performance may be overly optimistic.
- Training/validation/testing only uses the original single split from Kaggle, without conducting multiple random-split validations.

### Stage 2: NNMF Feature Extraction
- Images converted to grayscale → resized to 128x128 → normalized to [0,1] → vectorized into a non-negative data matrix $V$.
- NNMF decomposition using KL divergence + multiplicative update rules: $V \approx WH$, with rank $R=15$.
- The training set learns the basis matrix $W$, while validation/test sets are projected onto the fixed $W$ via non-negative least squares.
- L2 normalization of all feature vectors.
- **Statistical Feature Selection**: Calculates AUC, Cohen's d, and Welch's t-test p-value for each NNMF component, selecting the Top-M most discriminative features.

### Stage 3: Lightweight CNN Classification
- Trains a lightweight CNN on the selected subset of NNMF features (instead of high-dimensional image inputs).
- Validation set is monitored to prevent overfitting.
- Validation accuracy ~83%, test accuracy ~85.1%.

### Stage 4: Diffusion-Based Feature Space Denoising

This stage consists of three sub-steps:

1. **Forward Diffusion Data Generation**: Progressively adds Gaussian noise to clean NNMF features using a linear noise schedule over fixed steps, generating paired training data $(x_t, x_0)$.
2. **Denoising Network Training**: Encodes the timestep $t$ using sinusoidal embeddings, concatenates it with the noisy features, and trains a regression network using MSE loss to predict the clean features $x_0$.
3. **Test-time Defense**: Test samples are first forward-perturbed with noise up to step $t$, then restored using the denoising network, and finally fed into the classifier.
4. **EOT (Expectation over Transformation)**: Due to the randomness in the defense, predictions are averaged over K=8 random samples.

### Robustness Evaluation
- Uses AutoAttack ($L_\infty$, $\epsilon=0.10$) containing both APGD-CE and Square attacks.
- Models trained in MATLAB are exported to ONNX → Python/PyTorch execution of AutoAttack.

## Key Experimental Results

| Model | Clean Accuracy | Robust Accuracy (AutoAttack) | MCC | ROC-AUC | Brier Score |
|------|-----------|----------------------|-----|---------|-------------|
| Baseline (Undefended) | 86.05% | 0.47% | -0.99 | 0.0075 | 0.4702 |
| Defended (Diffusion Denoised) | 85.12% | **59.53%** | 0.17 | 0.7485 | 0.2150 |

**Key Findings**:
- Baseline completely collapses under AutoAttack (accuracy $\to$ 0.47%, MCC $\to$ -0.99, indicating completely reversed predictions).
- The diffusion defense improves robust accuracy to 59.5% at the cost of only ~1% in clean accuracy.
- The confusion matrix on the validation set shows that misclassifications between the two classes are relatively balanced.
- GPU acceleration reduces the total run time from 201.5s to 116.6s (1.73$\times$ speedup), with the AutoAttack defense stage benefiting the most.

## Highlights & Insights

- **Feature-Space Diffusion Defense**: Unlike common image-space denoising, this work performs diffusion-denoising in the low-dimensional NNMF feature space, minimizing computational overhead.
- **Interpretability**: NNMF basis components can be directly visualized as spatial patterns, and feature selection is statistically grounded (a triple criterion of AUC + Cohen's d + p-value).
- **Modular Design**: Individual stages (feature extraction / classification / defense) are decoupled, facilitating replacement and extension.
- **Practical Toolchain**: MATLAB training + ONNX export + Python AutoAttack, demonstrating a cross-platform pipeline.
- **Significant Robustness Improvement**: From almost complete collapse (0.47%) to nearly 60% robust accuracy, with only a ~1% penalty in clean accuracy.

## Limitations & Future Work

- **Dataset too small and lacking patient-level splits**: ~2,200 images without patient IDs; slice-level leakage may lead to overestimated performance.
- **Binary classification is overly simplified**: Only normal vs. tumor, with no multi-class tumor subtyping (e.g., glioma grading) involved.
- **Low clean accuracy** (85.1%), which is far below existing SOTA brain tumor classification methods, partly due to the excessive loss of discriminative information during NNMF compression.
- **Fixed diffusion step t=41**: Does not explore adaptive diffusion schedules or multi-step purification strategies.
- **Unclear meaning of $\epsilon=0.10$ in the feature space**: The actual magnitude of image perturbation corresponding to this perturbation budget is not discussed.
- **Poor writing quality**: Numerous grammatical errors, improper word choices, and confusing expressions affect readability.
- **Lack of discussion on the choice of NNMF rank R=15**: Explores no ablation studies regarding the choice of rank.

## Related Work & Insights

- **CNN-based Brain Tumor Detection**: The five-layer CNN of Hossain et al. achieves 97.87% on BRATS, but relies on pixel-level segmentation + end-to-end image input.
- **NMF-CNN Fusion**: Chan et al. use NMF to enhance CNN features in acoustic event detection (F1 improved from 23.7% to 30.39%).
- **Semi-NMF Networks**: Huang et al. utilize Semi-NMF to construct convolutional filters, bypassing backpropagation.
- **Joint Classification-Denoising Models**: Thiry & Guth utilize denoising objectives to enhance adversarial robustness on CIFAR-10/ImageNet.
- **AutoAttack**: A unified attack benchmark proposed by Croce & Hein, revealing that many "robust" defenses can actually be broken.
- **Adversarial Training**: Adversarial training methods like PGD are mainstream solutions for enhancing robustness, whereas this work opts for a more lightweight feature-space diffusion denoising route.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of NNMF + diffusion denoising possesses some novelty, though the individual components are not new)
- Experimental Thoroughness: ⭐⭐⭐ (Small dataset, lacking patient-level split, no ablation studies, and binary-only classification)
- Writing Quality: ⭐⭐⭐ (Numerous grammatical errors and confusing expressions, poor readability)
- Value: ⭐⭐⭐⭐ (The feature-space diffusion defense concept is inspiring, but experiments are insufficient for thorough validation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2025\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[ICLR 2026\] Sequential Information Bottleneck Fusion: Towards Robust and Generalizable Multi-Modal Brain Tumor Segmentation](../../ICLR2026/medical_imaging/sequential_information_bottleneck_fusion_towards_robust_and_generalizable_multi-.md)
- [\[CVPR 2025\] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)

</div>

<!-- RELATED:END -->
