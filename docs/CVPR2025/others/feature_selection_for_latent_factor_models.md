---
title: >-
  [Paper Note] Feature Selection for Latent Factor Models
description: >-
  [CVPR 2025][Feature Selection] A class-specific feature selection method based on signal-to-noise ratio (SNR) is proposed for low-rank generative models (PPCA/LFA/ELF). Accommodating a new class requires only $O(1)$ computation without retraining models of historical classes, thereby circumventing catastrophic forgetting. Furthermore, a novel non-parametric latent factor model, ELF, is proposed, and its effectiveness is validated on microarray cancer classification and high-d…
tags:
  - "CVPR 2025"
  - "Feature Selection"
  - "Latent Factor Models"
  - "Signal-to-Noise Ratio"
  - "Class-Incremental Learning"
  - "Low-Rank Generative Models"
date: 2026-05-08
content_hash: 6742656d02b9f1f5
---

# Feature Selection for Latent Factor Models

**Conference**: CVPR 2025  
**arXiv**: [2412.10128](https://arxiv.org/abs/2412.10128)  
**Code**: None  
**Area**: Others  
**Keywords**: Feature Selection, Latent Factor Models, Signal-to-Noise Ratio, Class-Incremental Learning, Low-Rank Generative Models

## TL;DR
A class-specific feature selection method based on signal-to-noise ratio (SNR) is proposed for low-rank generative models (PPCA/LFA/ELF). Accommodating a new class requires only $O(1)$ computation without retraining models of historical classes, thereby circumventing catastrophic forgetting. Furthermore, a novel non-parametric latent factor model, ELF, is proposed, and its effectiveness is validated on microarray cancer classification and high-dimensional feature selection.

## Background & Motivation

### Background

**Background**: Feature selection for high-dimensional data is a core problem in machine learning. Existing methods are categorized into global methods based on margin maximization (e.g., SVM feature selection) and statistical methods (e.g., PCA dimensionality reduction).

**Limitations of Prior Work**: Margin maximization methods require joint optimization across all classes, necessitating a complete retraining (with $O(C)$ complexity) when a new class is added, which does not support incremental learning. Supervised PCA also requires full retraining on new data.

**Key Challenge**: In real-world scenarios, the number of classes continuously grows (e.g., new disease types or newly discovered species), yet existing feature selection methods fail to add new classes incrementally, requiring retraining of all models every time.

**Goal**: Design class-specific feature selection methods where each class is modeled independently, enabling feature selection for a newly added class with only $O(1)$ computation and no need to retrain existing classes.

**Key Insight**: Model the data distribution of each class separately using low-rank generative models (PCA, Factor Analysis, or ELF), and measure the discriminativeness of each feature within that class using the signal-to-noise ratio (signal variance / noise variance). Features with high SNR are selected.

**Core Idea**: A low-rank generative model is trained independently for each class to select discriminative features using SNR. Adding a new class only requires training a new model rather than retraining all models.

## Method

### Overall Architecture
For each class $c$: (1) fit a low-rank generative model (PPCA/LFA/ELF) on its training data to decompose the data into low-rank signal + noise; (2) calculate the SNR for each feature dimension as $\text{SNR} = \text{signal variance} / \text{noise variance}$; (3) select feature dimensions with high SNR as the discriminative features for this class. During classification, decisions are made by combining the feature selection results of all classes.

### Key Designs

1. **ELF (Estimation of Latent Factors)**:

    - **Function**: A novel non-parametric latent factor model, which is more flexible than LFA.
    - **Mechanism**: Minimizes the weighted Frobenius norm $\min_{W,\Sigma} \|X - WZ\|_\Sigma$, where $\Sigma$ is the heteroscedastic noise covariance matrix (diagonal), and $W$ is the low-rank factor loading matrix. Unlike standard LFA, ELF does not assume a Gaussian distribution and employs weighted regression using feature weights (the inverse of noise variance). A semi-orthogonality constraint $W^T \Sigma^{-1} W = I$ is enforced to guarantee a unique solution.
    - **Design Motivation**: The assumption of isotropic noise in PPCA is too restrictive (identical noise variance), while the EM algorithm of LFA is sensitive to initialization. ELF achieves a better balance in heteroscedastic noise modeling and stability.

2. **SNR Feature Selection**:

    - **Function**: Quantify the discriminativeness of each feature within each class.
    - **Mechanism**: For the $j$-th feature of class $c$, $\text{SNR}_j = \sigma_j^2(\text{signal}) / \sigma_j^2(\text{noise})$. The signal variance arises from the contribution of the factor loading matrix $W$ to this feature, and the noise variance comes from the diagonal elements of $\Sigma$. High SNR indicates that the feature carries meaningful class-discriminative information, while low SNR indicates that it is dominated by noise.
    - **Design Motivation**: SNR directly quantifies the "information content" of features, which is more straightforward than mutual information or variance explained ratio.

3. **Class-Incremental Feature Selection**:

    - **Function**: No need to retrain existing class models when adding a new class.
    - **Mechanism**: Each class is modeled and its features selected independently without mutual coupling. When a new class $c_{new}$ is added, it only requires fitting a new low-rank model and calculating the SNR, which has a complexity of $O(1)$. This contrasts with margin maximization methods that require joint optimization over all classes with $O(C)$ complexity.
    - **Design Motivation**: Continuously increasing classes are common in real-world scenarios, making $O(1)$ incremental capability crucial for practical deployment.

### Loss & Training
PPCA utilizes closed-form ML estimation; LFA uses the EM algorithm; ELF uses alternating optimization (optimizing $W$ with $\Sigma$ fixed, and then optimizing $\Sigma$ with $W$ fixed).

## Key Experimental Results

### Main Results
- Microarray cancer data classification: The feature subset selected by SNR (50-200 dimensions) outperforms full dimensions and other feature selection methods.
- High-dimensional synthetic data: ELF achieves a higher feature recovery rate than PPCA and LFA under heteroscedastic noise.
- Scalability: The time required to add the 101st class is the same as adding the 2nd class ($O(1)$).

### Ablation Study

| Configuration | Description |
|------|------|
| PPCA (Isotropic Noise) | Performs poorly on heteroscedastic data |
| LFA (EM) | Sensitive to initialization |
| ELF (Weighted Frobenius) | Most stable and accurate |
| Margin Maximization vs SNR | SNR has an overwhelming advantage in incremental scenarios |

### Key Findings
- Biomarkers identified by SNR feature selection highly coincide with known cancer markers, validating the biological significance of the method.
- ELF significantly outperforms PPCA under heteroscedastic noise, proving the necessity of modeling non-isotropic noise.
- $O(1)$ incremental capability makes the method uniquely suited for continual learning and online feature selection scenarios.

## Highlights & Insights
- **$O(1)$ Incremental Feature Selection**: Each class is processed independently, and adding a new class does not affect existing classes—this is especially important in the era of continual learning.
- **Combining Theory and Practice**: Accompanied by rigorous theoretical analyses (Theorems 1-2 + Proposition 1) while being validated on real biological data.
- **Contributions of the ELF Model Itself**: Serving as a non-parametric alternative to LFA, it is more stable in noisy heteroscedastic scenarios.

## Limitations & Future Work
- The low-rank assumption might not be flexible enough for non-linear feature relationships.
- The SNR metric assumes feature independence, neglecting correlations among features.
- Only validated on classification tasks; other tasks such as regression are not addressed.
- Lacks comparison with deep learning-based feature selection methods.

## Related Work & Insights
- **vs CFSS**: CFSS jointly optimizes the margins of all classes, requiring $O(C)$ retraining for new classes, whereas ours achieves $O(1)$ incremental capability.
- **vs Supervised PCA**: SPCA uses a single model for all classes and does not support incremental learning, whereas ours builds independent models for each class.
- **vs Network Pruning (FSA/TISP)**: Network pruning focuses on redundant parameters in deep networks; ours focuses on selection in the raw feature space.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the ELF model and SNR incremental feature selection is novel; the $O(1)$ incremental capability is an important feature.
- Experimental Thoroughness: ⭐⭐⭐ The theoretical analysis is thorough, but the experimental datasets are relatively small, and comparison with large-scale or deep learning methods is lacking.
- Writing Quality: ⭐⭐⭐⭐ Rigorously derived mathematical proofs.
- Value: ⭐⭐⭐ Provides theoretical contributions to the intersection of feature selection and incremental learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](../../NeurIPS2025/others/distributionally_robust_feature_selection.md)
- [\[ICML 2025\] Latent Variable Estimation in Bayesian Black-Litterman Models](../../ICML2025/others/latent_variable_estimation_in_bayesian_black-litterman_models.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](../../ACL2025/others/latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[CVPR 2025\] EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching](edm_equirectangular_projection-oriented_dense_kernelized_feature_matching.md)

</div>

<!-- RELATED:END -->
