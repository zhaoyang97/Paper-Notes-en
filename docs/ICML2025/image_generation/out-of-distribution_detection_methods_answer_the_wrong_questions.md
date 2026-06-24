---
title: >-
  [Paper Note] Out-of-Distribution Detection Methods Answer the Wrong Questions
description: >-
  [ICML2025][Image Generation][OOD detection] This paper systematically demonstrates that current mainstream OOD detection methods (feature-based and logit-based) fundamentally answer the wrong questions—they detect "whether features are anomalous" or "whether the model is uncertain" rather than "whether the input comes from a different distribution." It also proves that various common improvement strategies cannot resolve this fundamental misalignment.
tags:
  - "ICML2025"
  - "Image Generation"
  - "OOD detection"
  - "distribution shift"
  - "feature space"
  - "uncertainty"
  - "model misspecification"
date: 2026-05-08
content_hash: 4c32fdedb9e0ef61
---

# Out-of-Distribution Detection Methods Answer the Wrong Questions

**Conference**: ICML2025  
**arXiv**: [2507.01831](https://arxiv.org/abs/2507.01831)  
**Code**: None (analysis/position paper)  
**Area**: Medical Images  
**Keywords**: OOD detection, distribution shift, feature space, uncertainty, model misspecification

## TL;DR

This paper systematically demonstrates that current mainstream OOD detection methods (feature-based and logit-based) fundamentally answer the wrong questions—they detect "whether features are anomalous" or "whether the model is uncertain" rather than "whether the input comes from a different distribution." It also proves that various common improvement strategies cannot resolve this fundamental misalignment.

## Background & Motivation

Out-of-Distribution (OOD) detection aims to identify whether test samples originate from a distribution different from the training data, which is crucial for the safe deployment of models. Current mainstream methods rely on features or logits of supervised models trained on in-distribution (ID) data for OOD detection and have achieved good results on standard benchmarks.

However, the authors raise a fundamental question: **the premise of using a classifier trained only on ID data to identify OOD samples is itself flawed**. The Key Challenge lies in:

- A classifier that only distinguishes between cats and dogs may output a highly confident but incorrect classification when facing an image of an airplane, simply because the airplane contains certain features used to distinguish cats from dogs.
- This is not a "difficulty" in OOD detection, but rather that the methods inherently answer the wrong question.

## Method

### Problem Formalization

Given a supervised model $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$, the predictive distribution of the model is:

$$p_\theta(y=k|x) = \text{softmax}(f_\theta(x)_k)$$

Decomposing the model into a feature extractor $e_\theta: \mathcal{X} \rightarrow \mathcal{F}$ and a classification layer $c_\theta: \mathcal{F} \rightarrow \mathbb{R}^K$:

$$p_\theta(y=c|x) = \text{softmax}(c_\theta \circ e_\theta(x))_c$$

OOD detection methods define a scoring function $s(x^*, f_\theta, \mathcal{D}_{tr})$, which is compared against a threshold to determine whether a sample is OOD.

### Two Failure Modes of Feature Methods

**Failure Mode 1: Indistinguishable features**. Features learned by supervised models solely serve ID class separation, and features of OOD samples can highly overlap with ID features. Taking Mahalanobis distance as an example:

$$s(x) = -\min_c (e_\theta(x) - \mu_c) \Sigma^{-1} (e_\theta(x) - \mu_c)^\top$$

Experiments show that even when training an Oracle binary classifier with simultaneous access to ID and OOD features, perfect separation remains impossible in near-OOD scenarios (e.g., ImageNet vs. ImageNet-OOD), presenting an irreducible lower bound of error.

**Failure Mode 2: Irrelevant features**. Even if the model learns discriminative features, selecting the correct dimensions from a large pool of irrelevant features is difficult. Experiments show that applying PCA to ViT features to keep only the 32-256 most relevant principal components improves the average Mahalanobis AUROC by more than 10 percentage points. However, the optimal feature subset heavily depends on the specific OOD dataset and is non-transferable.

### Fundamental Misalignment of Logit Methods

Logit methods equate **label uncertainty** (the model's uncertainty regarding ID labels) with **OOD uncertainty** (whether a sample is OOD), which are fundamentally different quantities.

- **ID samples often exhibit high uncertainty**: In ImageNet, a large number of multi-label images (e.g., containing concepts from multiple classes simultaneously) naturally generate high label uncertainty, yet these samples are explicitly ID.
- **OOD samples often exhibit low uncertainty**: The model's uncertainty on the "texture" category in Textures is indistinguishable from ID samples. Across 14 models using methods like MSP, the average FPR@95 exceeds 60%.

### Failure Analysis of Various Improvement Strategies

**Hybrid methods** (features + logits): Introducing a simple baseline Hybrid-Add (normalized Maha + MSP) reveals that improvements are highly model- and dataset-dependent, and fail to resolve the fundamental issue of indistinguishable ID/OOD features.

**Outlier Exposure**: Adding an OOD term to the training loss:

$$\mathcal{L} = \mathbb{E}_{(x,y)\sim\mathcal{D}_{in}} \ell_{CE}(f(x), y) + \alpha \mathbb{E}_{x'\sim\mathcal{D}_{out}} \ell_{CE}(f(x'), y_u)$$

While it can improve semantic shift detection, the model accuracy under covariate shift drops by over 10%, sacrificing generalization capabilities.

**Epistemic uncertainty** (Bayesian methods/ensembles): As ID data increases, posterior collapse leads to reduced epistemic uncertainty, which paradoxically degrades OOD detection performance—entirely opposite to the expected behavior.

**Introducing an "unknown" class**: This is only effective when training OOD samples are highly similar to test OOD samples, which is rarely satisfied in practice.

**Scaling model/data size**: Even when ViT-G/14 DINOv2 is pre-trained on internet-scale data, there remains a >5% irreducible error in near-OOD tasks.

**Generative models**: $p(x)$ answers a different question from $p(\text{OOD}|x)$. According to Bayes' rule $p(\text{OOD}|x) \propto p(x|\text{OOD})/p(x)$, knowing $p(x)$ does not tell us this ratio. Experiments show that better generative models can instead lead to worse OOD detection.

## Key Experimental Results

| Analysis Dimension | Key Findings | Models Involved |
|---------|---------|---------|
| Indistinguishable features | Oracle binary classifier still exhibits significant errors on near-OOD | ResNet-18/50, ViT-S/B/16 |
| Irrelevant features | PCA feature selection on average improves Maha AUROC > 10pp | Same as above |
| Logit FPR@95 | MSP/MaxLogit/Energy/Entropy on average > 60% | 14 models (ResNet/ViT/ConvNextV2) |
| Side-effect of Outlier Exposure | Accuracy drops by > 10% under covariate shift | ResNet-18 on CIFAR-10 |
| Model Scale | ViT-G/14 still has > 5% irreducible error | 12 models of different scales |
| Generative Models | Better GMM/NF models → worse OOD detection | GMM, RealNVP, DiT |

## Highlights & Insights

1. **Profound Argument**: Instead of merely pointing out the limitations of a specific method, it fundamentally questions the soundness of the entire paradigm—a crucial reflection for the field.
2. **Systematic and Comprehensive Analysis**: Covers 7 major categories—feature-based, logit-based, hybrid methods, Outlier Exposure, Bayesian methods, model scaling, and generative models—demonstrating their limitations one by one.
3. **Error Decomposition Framework**: Decomposes the error of feature methods into two independent sources, "indistinguishable features" and "irrelevant features," providing a quantitative analysis tool.
4. **Counter-intuitive Finding**: Better generative models can lead to worse OOD detection (the fundamental difference between $p(x)$ and $p(\text{OOD}|x)$), challenging common assumptions.
5. **Wide Experimental Coverage**: Involves 54 models, 9 architectures, 6 pre-training strategies, and multiple datasets (ImageNet, CIFAR, CelebA, etc.), enhancing the persuasiveness.

## Limitations & Future Work

1. **Lack of Constructive Solutions**: The paper focuses primarily on critical analysis; although it provides directional suggestions in Section 6, no concrete alternative method is proposed.
2. **Primary Focus on the Vision Domain**: Although it briefly covers NLP (Multi-NLI/SNLI), the vast majority of experiments are based on image classification.
3. **Insuffient Discussion on Practical Deployment**: In practice, OOD detection often acts as one component of a safety mechanism rather than the sole means; the paper does not fully discuss its value at the system level.
4. **Under-explored Self-Supervised/Contrastive Learning**: Large-scale self-supervised pre-training may learn more general feature representations, which the paper analyzes only to a limited extent.
5. **Distinction between Near-OOD and Far-OOD**: The paper's arguments are strongest in near-OOD scenarios, but existing methods might still possess practical value in far-OOD scenarios.

## Related Work & Insights

- **Hendrycks & Gimpel (2016)**: MSP baseline, which remains a highly competitive method to this day.
- **Lee et al. (2018)**: Mahalanobis distance method, a representative of feature-based methods.
- **Kirichenko et al. (2020)**: Inductive bias issues in normalizing flows.
- **Yang et al. (2024b)**: Distinguishing semantic shift and covariate shift in benchmarks.
- **Insight**: Future OOD detection should directly estimate $p(\text{OOD}|x)$ rather than relying on proxy signals, potentially requiring OOD detection objectives to be incorporated during the training phase.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Fundamental questioning of the entire OOD detection paradigm, unique perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (54 models / 9 architectures / 6 pre-training settings / multiple datasets)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, step-by-step argumentation)
- Value: ⭐⭐⭐⭐ (Profound analysis but lacks concrete constructive solutions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection](../../ICCV2025/image_generation/synthesizing_near-boundary_ood_samples_for_out-of-distribution_detection.md)
- [\[ICML 2025\] Unsupervised Learning for Class Distribution Mismatch (UCDM)](unsupervised_learning_for_class_distribution_mismatch.md)
- [\[ICML 2025\] Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization](expressive_score-based_priors_for_distribution_matching_with_geometry-preserving.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](../../CVPR2025/image_generation/any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)

</div>

<!-- RELATED:END -->
