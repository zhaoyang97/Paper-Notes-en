---
title: >-
  [Paper Note] Rethinking Unsupervised Outlier Detection via Multiple Thresholding
description: >-
  [ECCV 2024][Self-Supervised Learning][Unsupervised Outlier Detection] This work proposes the Multi-T (Multiple Thresholding) module, which generates two thresholds to isolate inliers and outliers within a target dataset, respectively. By utilizing the identified inliers to train a clean normal manifold and using the outliers for feature denoising, Multi-T significantly enhances the performance of existing outlier scoring methods.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Unsupervised Outlier Detection"
  - "Multiple Thresholding"
  - "Outlier Scoring"
  - "Feature Normalization"
  - "Self-Supervised Signals"
date: 2026-05-08
content_hash: 2d7f365dc070b05f
---

# Rethinking Unsupervised Outlier Detection via Multiple Thresholding

**Conference**: ECCV 2024  
**arXiv**: [2407.05382](https://arxiv.org/abs/2407.05382)  
**Code**: Yes ([https://github.com/zhliu-uod/Multi-T](https://github.com/zhliu-uod/Multi-T))  
**Area**: Self-Supervised Learning  
**Keywords**: Unsupervised Outlier Detection, Multiple Thresholding, Outlier Scoring, Feature Normalization, Self-Supervised Signals

## TL;DR

This work proposes the Multi-T (Multiple Thresholding) module, which generates two thresholds to isolate inliers and outliers within a target dataset, respectively. By utilizing the identified inliers to train a clean normal manifold and using the outliers for feature denoising, Multi-T significantly enhances the performance of existing outlier scoring methods.

## Background & Motivation

### Core Problem

Unsupervised outlier detection (UOD) aims to assign an outlier score to each sample in an unlabeled dataset. Current mainstream methods (such as DeepSVDD, OCSVM, LVAD, etc.) focus on learning a discriminative scoring function $F(\cdot)$, but overlook a critical step: **how to convert continuous scores into binary labels**.

### Two Major Limitations of the Existing Paradigm

**Contaminated Training Set**: Most methods assume that inliers constitute the majority and directly learn the normal manifold on the entire target dataset. However, when the outlier ratio increases, the dataset mean shifts towards the outliers (the mean-shift problem), causing the learned "normal manifold" to be contaminated.

**Scoring Methods Limited to Single-Pass Inference**: Existing detectors are typically non-iterative. Due to the lack of predicted labels, they cannot exploit self-supervised signals from the dataset to iteratively improve themselves. Performance would be significantly enhanced if one could first "label" the data and then refine the detection.

### Why a Single Threshold is Infeasible?

Traditional single-threshold methods attempt to find a threshold $\phi$ to precisely separate the score distributions of inliers and outliers. However, due to the imperfection of the initial scoring function and the influence of the outlier ratio, an **overlapping region $A$** inevitably exists in the score distribution:

$$D = F_{\text{init}}(\mathbf{X}) = I \cup A \cup O$$

Within this overlapping region, any choice of a single threshold will lead to misclassifications.

### Core Idea

Rather than attempting precise separation, the proposed method **discards the overlapping region** and generates two thresholds:
- $\phi_{\text{in}}$: Samples with scores below this value are definitely inliers $\rightarrow$ used to train a clean normal manifold.
- $\phi_{\text{out}}$: Samples with scores above this value are definitely outliers $\rightarrow$ used for feature denoising via Shell Normalization.

Although some samples in the "gray area" remain unclassified, both the obtained inlier and outlier sets exhibit high purity, making them highly reliable for enhancing subsequent detection methods.

## Method

### Overall Architecture

Multi-T is a training-free, plug-and-play module. Its workflow consists of three stages:
1. **Preparation**: Feature extraction + initial outlier scoring function.
2. **Multi-T Module**: Generates two thresholds to separate inliers and outliers.
3. **Integration**: Feeds the identified inliers and outliers into existing detection methods to obtain an enhanced scoring function.

### Key Designs

1. **Initial Outlier Scoring Function (LVAD-S)**:

    - **Function**: Computes the initial outlier score for each sample.
    - **Mechanism**: Computes the Euclidean distance of Ergodic-set normalized features to the dataset mean:
    $F_{\text{init}}(\mathbf{X}) = \{\text{Dist}(\text{E-norm}(\mathbf{x}_i), \text{E-norm}(\mathbf{m}_X))\}_{i=1}^n$
      where the reference vector $\mathbf{v}_E$ for E-norm is the global scalar mean across all dimensions of all samples.
    - **Design Motivation**: E-norm is invariant to the outlier ratio (as it uses a global scalar mean instead of a dimension-wise mean), providing a robust foundation for initial scoring.

2. **Stage 1: Identifying Uncontaminated Inliers (Iterative 3-Sigma Filtering)**:

    - **Function**: Extracts a highly pure set of inliers from the initial score distribution.
    - **Mechanism**:
        - Sort the score sequence in ascending order and fit the inlier distribution using linear regression (since sorted scores of inliers tend to increase approximately linearly).
        - Identify the boundary of the linear region: $I = \{\hat{F}(\mathbf{x}_i) \mid i < \max_i\{g(a_i) > \hat{F}(\mathbf{x}_i)\}\}$
        - Iteratively apply the 3-sigma rule to filter out outliers: $\phi_{\text{out}}^b = \text{mean}(I^b) + 3 \cdot \text{std}(I^b)$
        - Remove samples that exceed the threshold and repeat until convergence.
        - The final inlier threshold: $\phi_{\text{in}} = \max(I^{b^*})$
    - **Design Motivation**: Shell Theory indicates that the distance scores of inliers in a high-dimensional space follow a Gaussian-like distribution, where the 3-sigma rule can effectively filter out anomalies. The iterative process progressively mitigates the biasing effects of outliers on the mean and standard deviation.

3. **Stage 2: Adaptive Outlier Threshold Selection**:

    - **Function**: Adaptively selects an appropriate outlier threshold based on the outlier ratio $\gamma$.
    - **Mechanism**: Estimates $\gamma$ by comparing the ranking similarity obtained from two normalization methods:
        - Shell Normalization (S-norm): Uses the predicted outlier mean as the reference vector, which is sensitive to $\gamma$.
        - Ergodic-set Normalization (E-norm): Uses the global scalar mean, which is invariant to $\gamma$.
        - Compute the Pearson correlation coefficient $\rho$ of their rank indices:
       - $\rho > 0.3$ (high $\gamma$) $\rightarrow$ use the converged threshold $\phi_{\text{out}}^*$ (more aggressive)
       - $0.1 \leq \rho \leq 0.3$ (medium $\gamma$) $\rightarrow$ use the first-round threshold $\phi_{\text{out}}^1$
       - $\rho < 0.1$ (low $\gamma$) $\rightarrow$ use the global 3-sigma threshold $\phi_{\text{out}}^0$ (most conservative)
    - **Design Motivation**: When $\gamma$ is high, S-norm performs well, resulting in high ranking consistency between the two normalizations, which allows for aggressive outlier selection. Conversely, when $\gamma$ is low, S-norm is unreliable, and a conservative approach is adopted. This mechanism ensures robust performance under various outlier ratios.

### Loss & Training

The Multi-T module itself **does not require training** and is a purely statistical approach. It can be integrated in two ways:

**Direct Application** (distance to the inlier mean):
$$F_{\text{Multi-T}}(\mathbf{X}) = \{\text{Dist}(\text{S-norm}(\mathbf{x}_i, \mathbf{v}'_S), \text{S-norm}(\mathbf{m}_{X'_{\text{in}}}, \mathbf{v}'_S))\}_{i=1}^n$$

**Integration with Existing Methods**:
$$F_{M+\text{Multi-T}}(\mathbf{X}) = M.\text{fit}(\{\text{S-norm}(\mathbf{x}_i) \mid \mathbf{x}_i \in X'_{\text{in}}\}).\text{predict}(\{\text{S-norm}(\mathbf{x}_i) \mid \mathbf{x}_i \in X\})$$

In other words, the model $M$ is trained on clean inliers and evaluated on all features normalized via S-norm, with both steps benefiting from the Multi-T module.

## Key Experimental Results

### Main Results: AUC Comparison against SOTA Outlier Detection Methods

| Method | STL-10 (ResNet) | STL-10 (CLIP) | CIFAR-10 (CLIP) | MIT-Places (CLIP) | MNIST |
|------|:---:|:---:|:---:|:---:|:---:|
| IF | 0.836 | 0.943 | 0.891 | 0.868 | 0.776 |
| ECOD | 0.907 | 0.981 | 0.935 | 0.943 | 0.734 |
| LVAD | 0.954 | 0.968 | 0.917 | 0.919 | 0.867 |
| DeepSVDD | 0.622 | 0.597 | 0.509 | 0.549 | 0.513 |
| **Multi-T** | **0.968** | **0.989** | **0.957** | **0.974** | **0.897** |
| DeepSVDD + Multi-T | 0.925 | 0.921 | 0.819 | 0.832 | 0.732 |
| DeepSVDD Gain | +48.7% | +54.3% | +60.9% | +51.6% | +37.9% |
| OCSVM + Multi-T | 0.957 | 0.965 | 0.916 | 0.924 | 0.863 |

Multi-T achieves SOTA performance across almost all combinations of datasets and feature extractors, significantly boosting the performance of weaker baselines.

### Ablation Study: Enhancing Various Methods with Multi-T (STL-10)

| Method | Without Multi-T | With Multi-T | Gain |
|------|:---:|:---:|:---:|
| IF (ResNet) | 0.836 | 0.899 | +7.5% |
| ECOD (ResNet) | 0.907 | 0.919 | +1.3% |
| ABOD (ResNet) | 0.665 | 0.883 | +32.8% |
| PCA (ResNet) | 0.865 | 0.945 | +9.2% |
| GMM (ResNet) | 0.859 | 0.952 | +10.8% |
| IF (CLIP) | 0.943 | 0.983 | +4.2% |
| PCA (CLIP) | 0.984 | 0.994 | +1.0% |

Multi-T consistently improves all methods (both statistical and deep learning-based), with weaker methods benefiting the most.

### Quality Evaluation of Threshold Learning (STL-10)

| Outlier Scoring Function | Thresholding Method | $F_{0.1}$ | $F_{10}$ | Average |
|------|------|:---:|:---:|:---:|
| LVAD-S | Highest $F_{0.1}$ Baseline | 0.911 | 0.454 | 0.682 |
| LVAD-S | Highest $F_{10}$ Baseline | 0.382 | 0.967 | 0.674 |
| LVAD-S | **Multi-T (Ours)** | **0.840** | **0.869** | **0.855** |
| +GT Norm | **Multi-T (Ours)** | **0.917** | **0.980** | **0.949** |

Traditional methods suffer from a severe imbalance between $F_{0.1}$ and $F_{10}$, whereas Multi-T achieves both balanced and superior results for both metrics.

### Key Findings

- **The Comeback of DeepSVDD**: The baseline AUC of DeepSVDD is only 0.622 (STL-10) but surges to 0.925 (+48.7%) with Multi-T. This suggests that the bottleneck of DeepSVDD lies in the quality of the training data rather than the model itself.
- **Efficiency Advantage**: Multi-T takes only 1.2 seconds to process 10,000 ResNet-50 samples, which is orders of magnitude faster than deep learning baselines.
- **Impact of Feature Extractors**: CLIP features consistently outperform ResNet-50. Multi-T is effective for both, reaching an AUC of up to 0.994 with PCA + CLIP + Multi-T.
- **Robustness Across Outlier Ratios**: Evaluating across the range of $\gamma \in [0.05, 0.4]$, Multi-T demonstrates stable and robust performance.

## Highlights & Insights

1. **Paradigm Shift**: Shifting from "learning a better scoring function" to "exploiting self-supervised signals through thresholding" to retroactively feedback predicted labels to the baseline method. A simple distance-based method combined with Multi-T can outperform complex SOTA detectors.
2. **Training-Free Design**: Multi-T is a purely statistical implementation that requires no training or hyperparameter tuning, processing large-scale data in just 1.2 seconds.
3. **Estimating Outlier Ratio via Rank Correlation**: Leverages the structural consistency and complementarity of two normalization methods to implicitly estimate $\gamma$, eliminating the need for prior knowledge.
4. **Practicality of Multiple Thresholds**: Abandoning the illusion of precise separation to instead focus on ensuring high purity at both ends of the inlier and outlier distributions proves to be a pragmatic and highly effective design.

## Limitations & Future Work

- Samples falling into the "gray area" between the two thresholds are left unutilized, representing a partial waste of data.
- The cutoff thresholds for $\rho$ (0.1, 0.3) for threshold selection are set empirically and may lack robustness under extreme distributions.
- The quality of the initial scoring function still dictates the performance upper bound of Multi-T; if the initial step fails to provide discriminative scores, the sorting and linear fitting procedures may become ineffective.
- The evaluation is primarily conducted on image classification datasets, leaving validation in real-world scenarios such as industrial defect detection or time-series anomaly detection for future work.

## Related Work & Insights

- **Relationship with Shell Renormalization**: Multi-T utilizes Shell Normalization for feature denoising but improves upon the outlier selection mechanism, using a comparison with Ergodic-set Normalization to estimate $\gamma$.
- **Difference from Semi-Supervised Anomaly Detection**: Multi-T does not require a predefined train/test split; all operations are executed globally on the same unlabeled target dataset.
- **Directions for Inspiration**: The multiple thresholding approach can be extended to other scenarios requiring clean subset extraction from noisy labels, such as learning with noisy labels and data cleaning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The shift in perspective from "scoring" to "thresholding" is highly novel. The design of using multiple thresholds coupled with self-supervised signals is inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 datasets, multiple feature extractors, a wide range of outlier ratios, and compares against 20+ methods.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-structured, although some notations are slightly complex.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, offers substantial improvements, and provides high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](../../ICML2026/self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[ECCV 2024\] FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning](flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)
- [\[ECCV 2024\] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [\[ICLR 2026\] Rethinking Unsupervised Cross-Modal Flow Estimation: Learning from Decoupled Optimization and Consistency Constraint](../../ICLR2026/self_supervised/rethinking_unsupervised_cross-modal_flow_estimation_learning_from_decoupled_opti.md)
- [\[ICML 2026\] Towards One-for-All Anomaly Detection for Tabular Data](../../ICML2026/self_supervised/towards_one-for-all_anomaly_detection_for_tabular_data.md)

</div>

<!-- RELATED:END -->
