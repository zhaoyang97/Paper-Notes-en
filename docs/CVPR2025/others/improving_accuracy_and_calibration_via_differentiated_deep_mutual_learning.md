---
title: >-
  [Paper Note] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning
description: >-
  [CVPR 2025][Deep Mutual Learning] Proposes Diff-DML (Differentiated Deep Mutual Learning), which simultaneously improves accuracy and uncertainty calibration quality while maintaining the prediction diversity of the ensemble models through two core designs: Differentiated Training Strategy (DTS) and Diversity-Preserving Learning Objective (DPLO).
tags:
  - "CVPR 2025"
  - "Deep Mutual Learning"
  - "Uncertainty Calibration"
  - "Ensemble Methods"
  - "Prediction Diversity"
  - "Overconfidence"
date: 2026-05-08
content_hash: ac5c2ab645685211
---

# Improving Accuracy and Calibration via Differentiated Deep Mutual Learning

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Model Calibration and Ensemble Learning  
**Keywords**: Deep Mutual Learning, Uncertainty Calibration, Ensemble Methods, Prediction Diversity, Overconfidence

## TL;DR

Proposes Diff-DML (Differentiated Deep Mutual Learning), which simultaneously improves accuracy and uncertainty calibration quality while maintaining the prediction diversity of the ensemble models through two core designs: Differentiated Training Strategy (DTS) and Diversity-Preserving Learning Objective (DPLO).

## Background & Motivation

**Background**: Deep neural networks have achieved excellent prediction accuracy across various tasks. However, in safety-critical applications (such as autonomous driving and medical diagnosis), high accuracy alone is insufficient; reliable uncertainty estimation is also required.

**Limitations of Prior Work**:
- Modern DNNs trained with cross-entropy loss are prone to **overconfidence**, especially on ambiguous samples.
- Many calibration techniques (such as temperature scaling and label smoothing) improve calibration at the expense of **sacrificing accuracy** or **increasing computational overhead**.
- Traditional Deep Mutual Learning (DML) improves performance through mutual learning among multiple models, but the models gradually converge, leading to a **loss of prediction diversity**, which is detrimental to calibration.

**Key Challenge**: The calibration benefit of ensemble methods stems from the prediction diversity of member models, but the mutual learning process causes models to homogenize, leading to a loss of diversity, which constitutes a fundamental contradiction.

**Goal**: To preserve the prediction diversity of member models within the mutual learning framework, thereby simultaneously improving both accuracy and calibration.

**Key Insight**: Start from the issue of diversity loss in mutual learning and maintain differences among models through differentiated training strategies and learning objectives.

**Core Idea**: Use differentiated data augmentation and a differentiated KL divergence learning objective to ensure that individual models in mutual learning maintain sufficient prediction diversity, thereby achieving the calibration gain of the ensemble.

## Method

### Overall Architecture

Diff-DML is based on the Deep Mutual Learning (DML) framework, training multiple networks to learn from each other. However, it introduces two key innovations to maintain prediction diversity: Differentiated Training Strategy (DTS) and Diversity-Preserving Learning Objective (DPLO).

### Key Designs

1. **Differentiated Training Strategy (DTS)**:
    - **Function**: Ensures that models receive differentiated training signals at the source by applying different data augmentation strategies to different models.
    - **Mechanism**: Each member model uses a different combination of data augmentations (e.g., varying cropping strategies, color jittering, etc.), ensuring that even during mutual learning, the models maintain diversity by observing data from multiple perspectives.
    - **Design Motivation**: In traditional DML, all models receive the same input, leading to rapid convergence after mutual learning. Differentiated input is the most direct approach to maintaining diversity.

2. **Diversity-Preserving Learning Objective (DPLO)**:
    - **Function**: Modifies the KL divergence objective of mutual learning to penalize overly similar prediction distributions while encouraging models to learn from each other.
    - **Mechanism**: Introduces a diversity term based on the standard KL divergence mutual learning loss, penalizing models when their predictions are too similar, thereby maintaining diversity at the optimization objective level.
    - **Design Motivation**: Relying solely on different data augmentations may be insufficient to maintain long-term diversity; explicit diversity constraints must be provided at the loss function level.

3. **Theoretical Analysis Support**:
    - **Function**: Theoretically proves that the diversified learning framework of Diff-DML can leverage ensemble benefits while preventing the loss of prediction diversity observed in traditional DML.
    - **Mechanism**: Demonstrates the critical role of prediction diversity in calibration quality by analyzing the variance decomposition of the ensemble model.
    - **Design Motivation**: To provide a theoretical guarantee for the effectiveness of the method.

### Loss & Training

The overall loss consists of three parts:
- **Classification Loss**: Standard cross-entropy loss, ensuring the classification accuracy of each model.
- **Mutual Learning Loss**: Modified KL divergence, encouraging models to learn soft label knowledge from each other.
- **Diversity Regularization**: Penalizes overly similar model predictions to maintain ensemble diversity.

During the training process, multiple models are trained synchronously. Each model employs a differentiated data augmentation strategy and learns from the others via the DPLO objective.

## Key Experimental Results

### Main Results

Results using the ResNet34 model on the CIFAR-100 dataset:

| Metric | Diff-DML vs MDCA (SOTA) | Gain |
|------|------------------------|---------|
| Accuracy | Absolute Gain | +1.3% / +3.1% |
| ECE | Relative Reduction | -49.6% / -43.8% |
| Classwise-ECE | Relative Reduction | -7.7% / -13.0% |

An extensive evaluation conducted across multiple benchmark datasets validates the effectiveness of the proposed method.

### Ablation Study

- While using DTS and DPLO individually can bring improvements, their combined use yields the best performance.
- The efficacy of the differentiated data augmentation increases as the degree of variance between augmentation strategies grows.
- The weight of the diversity regularization requires careful tuning.

### Key Findings

- In traditional DML, models converge at an accelerated pace in the later stages of training, leading to a sharp decline in diversity.
- Diff-DML maintains stable prediction diversity throughout the entire training process.
- There is a strong positive correlation between prediction diversity and calibration quality.
- The method performs consistently across different architectures (such as ResNet, WideResNet, etc.).

## Highlights & Insights

1. **Deep Problem Insight**: Accurately identifies the overlooked issue of diversity loss in traditional mutual learning, providing thorough theoretical justification and experimental validation.
2. **Simple and Effective Solution**: Both DTS and DPLO designs are simple, require no additional complex modules, and have low implementation overhead.
3. **Unification of Theory and Experiment**: Provides theoretical analysis proving the importance of diversity for calibration and validates theoretical predictions via experiments.
4. **Dual Indicator Improvement**: Simultaneously improves accuracy and calibration quality without introducing additional inference overhead.

## Limitations & Future Work

1. **Ensemble Inference Overhead**: Running multiple models is still required during inference, with computational overhead scaling linearly with the number of member models.
2. **Data Augmentation Strategy Selection**: The design of differentiated augmentation strategies currently lacks automated methods.
3. **Large-scale Validation**: Primarily validated on medium-scale datasets such as CIFAR-100; performance on large-scale datasets remains to be confirmed.
4. **Combination with Post-processing Calibration**: The combined effect with post-processing methods such as temperature scaling can be explored.

## Related Work & Insights

- **Deep Mutual Learning (DML)**: The baseline framework of this work.
- **MDCA**: The prior SOTA calibration method.
- **Diversity Theory in Ensemble Methods**: The dual decomposition theorem indicates that ensemble performance depends on the diversity of the member models.
- **Inspiration for Future Research**: The idea of maintaining diversity in mutual learning can be extended to scenarios such as knowledge distillation and federated learning.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)
- [\[CVPR 2025\] Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach](wear_classification_of_abrasive_flap_wheels_using_a_hierarchical_deep_learning_a.md)
- [\[NeurIPS 2025\] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering](../../NeurIPS2025/others/mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)

</div>

<!-- RELATED:END -->
