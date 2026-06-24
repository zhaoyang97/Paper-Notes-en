---
title: >-
  [Paper Note] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][Test-time prompt tuning] The A-TPT framework is proposed to promote angular diversity by maximizing the minimum pairwise angular distance of normalized text features on the unit hypersphere. This addresses the miscalibration issue caused by overconfident VLM predictions in Test-time Prompt Tuning (TPT), outperforming existing TPT calibration methods on both natural distribution shifts and medical datasets.
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Test-time prompt tuning"
  - "CLIP"
  - "calibration"
  - "angular diversity"
  - "hyperspherical uniform distribution"
date: 2026-05-08
content_hash: 7635a4d6df231e92
---

# A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models

**Conference**: ICLR 2026  
**arXiv**: [2510.26441](https://arxiv.org/abs/2510.26441)  
**Code**: Coming soon  
**Area**: Multimodal VLM / Calibration  
**Keywords**: Test-time prompt tuning, CLIP, calibration, angular diversity, hyperspherical uniform distribution  

## TL;DR
The A-TPT framework is proposed to promote angular diversity by maximizing the minimum pairwise angular distance of normalized text features on the unit hypersphere. This addresses the miscalibration issue caused by overconfident VLM predictions in Test-time Prompt Tuning (TPT), outperforming existing TPT calibration methods on both natural distribution shifts and medical datasets.

## Background & Motivation

**Background**: TPT adapts VLMs (e.g., CLIP) to new tasks by optimizing learnable prompt vectors with unlabeled samples at inference time. While it improves accuracy, it often leads to increased calibration error (overconfidence).

**Limitations of Prior Work**: C-TPT improves calibration by maximizing Average Text Feature Dispersion (ATFD), but features can still cluster. O-TPT enforces angular separation using orthogonality constraints, but when the number of classes $N$ exceeds the embedding dimension $|D|$ (e.g., 1000 classes in ImageNet-1k vs. 512-dim CLIP), strict orthogonality is mathematically impossible, causing features to cluster instead.

**Key Challenge**: Neither dispersion ($L_2$ distance) nor orthogonality constraints guarantee a uniform angular distribution of features on the hypersphere—the former can allow all features to bias toward one direction while being far from the centroid, and the latter fails at high class counts.

**Goal**: Propose a TPT calibration method that effectively promotes angular diversity of text features in both $N > |D|$ and $N < |D|$ scenarios.

**Key Insight**: Link the calibration problem to the **Tammes problem** (placing points optimally on a hypersphere)—maximizing the minimum pairwise angular distance ensures a uniform distribution of features.

**Core Idea**: By maximizing the minimum pairwise angular distance between normalized text features (rather than average dispersion or orthogonality), a uniform distribution is achieved on the hypersphere, significantly improving VLM calibration during inference.

## Method

### Overall Architecture
A-TPT aims to resolve the side effect where TPT boosts accuracy at the cost of calibration (overconfidence) during inference. It follows the TPT setup—performing unlabeled optimization of learnable prompt vectors for each test sample—but adds an angular diversity regularizer to the original entropy minimization target. The workflow involves: concatenating learnable prompts to each class name to generate text features → performing multi-view data augmentation on the image and updating prompts via entropy minimization to adapt to the sample → simultaneously constraining the **minimum pairwise angular distance** of normalized text features for all classes to be as large as possible, forcing them to spread uniformly on the unit hypersphere. Finally, classification is performed using these more dispersed text features to reduce calibration error without sacrificing accuracy.

### Key Designs

**1. Angular Diversity Regularization: Replacing Dispersion/Orthogonality with Maximizing Minimum Angular Distance**

Prior strategies failed to guarantee true uniform distribution on the hypersphere: C-TPT's ATFD maximizes $L_2$ distances to the centroid, yet features can cluster together in one direction. O-TPT forces pairwise orthogonality, which becomes mathematically impossible when the number of classes exceeds the embedding dimension (e.g., ImageNet-1k's 1000 classes vs. CLIP's 512 dimensions), leading to feature collapse. A-TPT maps this to the **Tammes problem** (arranging $N$ points on a hypersphere to maximize their separation): it calculates angular distances $\theta_{ij} = \arccos(\text{sim}(t_i, t_j))$ for all class pairs $(i,j)$ and includes $\max \min_{i \neq j} \theta_{ij}$ in the objective. This "pushes apart the closest pair," acting as a numerical approximation to the Tammes problem. The benefit is dimensionality independence: when $N > |D|$, maximizing the minimum angular distance remains well-defined, finding the optimal arrangement of $N$ points in a finite-dimensional space; when $N < |D|$, it fully utilizes the hypersphere space unlike wasteful orthogonality constraints.

**2. Integration with TPT: As a Regularization Term for Entropy Minimization**

The angular diversity loss does not replace the original TPT objective but is added as a regularizer for joint optimization: $\mathcal{L} = \mathcal{L}_{entropy} + \lambda \mathcal{L}_{angular}$, where $\lambda$ balances adaptation capability and calibration constraints. Thus, $\mathcal{L}_{entropy}$ ensures the prompt adapts to the current test sample to maintain TPT's accuracy gains, while $\mathcal{L}_{angular}$ pushes text features toward a uniform distribution to improve calibration within the same optimization step. Both terms undergo gradient descent during inference for each test sample, which is why A-TPT reduces calibration error without dropping accuracy.

### Loss & Training
Gradient descent is performed on learnable prompt vectors for each test sample (unlabeled). Multiple views are generated via data augmentation to minimize prediction entropy, combined with the angular diversity regularizer $\mathcal{L}_{angular}$. The temperature parameter $\tau=0.01$ is fixed.

## Key Experimental Results

### Main Results

**ECE↓ (Expected Calibration Error) on Caltech101 with CLIP ViT-B/16:**

| Method | Caltech101 ECE↓ |
|------|-----------------|
| Zero-shot CLIP | 5.66 |
| TPT | 6.18 |
| **Ours (A-TPT)** | **2.23** |

A-TPT achieves the lowest ECE across multiple natural distribution shift datasets (Caltech101, OxfordPets, DTD, EuroSAT, ImageNet) and generalizes to medical datasets. Full comparisons with C-TPT / O-TPT are available in the original paper.

### Ablation Study

Comparison of calibration constraints relative to TPT (accuracies are mostly maintained; differences are primarily in ECE):

| Configuration | ECE Trend | Description |
|------|---------|------|
| TPT (No calibration) | Highest | Accurate but overconfident |
| + ATFD (C-TPT) | Slight decrease | Only maximizes $L_2$ dispersion from centroid; limited improvement |
| + Orthogonality (O-TPT) | Slight decrease | Fails when $N > |D|$ due to impossibility of orthogonality |
| + Angular Diversity (A-TPT) | **Lowest** | Maximizes minimum pairwise angular distance; globally optimal |

### Key Findings
- Empirical analysis shows a negative correlation between Angular Distance (AD) and ECE—larger AD leads to better calibration, validating the theoretical rationale of angular diversity.
- When $N > |D|$ (e.g., ImageNet-1k 1000 classes vs. 512 dimensions), O-TPT's orthogonality constraint fails, while A-TPT remains effective.
- t-SNE visualizations clearly show that angularly diverse features are not only dispersed but also well-aligned with class labels.
- A-TPT significantly reduces ECE without sacrificing accuracy and generalizes effectively to medical datasets.

## Highlights & Insights
- **ML Application of the Tammes Problem**: Linking hyperspherical optimal point placement to VLM calibration is a clever cross-domain approach. Maximizing minimum pairwise angular distance characterizes "uniform distribution" more fundamentally than $L_2$ dispersion or orthogonality.
- **Decoupling Calibration and Accuracy**: Within groups of similar accuracy, calibration performance variance is primarily driven by angular diversity—providing a new perspective on understanding VLM calibration.
- **Consistency between Theory and Practice**: Uniform distribution on the hypersphere preserves maximum information (Wang & Isola, 2020), which aligns with calibration improvements.

## Limitations & Future Work
- Maximizing the minimum angular distance is non-convex and may converge to local optima.
- Computational overhead per test sample—TPT already has significant inference-time costs, and additional regularization increases this.
- Experiments focused on classification; downstream tasks like detection or segmentation remain unverified.
- The temperature parameter $\tau=0.01$ is fixed; adaptive temperature might offer further improvements.

## Related Work & Insights
- **vs. C-TPT**: C-TPT uses ATFD to maximize $L_2$ distance from the centroid; A-TPT directly optimizes minimum pairwise angular distance to guarantee uniformity more effectively.
- **vs. O-TPT**: O-TPT uses orthogonality constraints which are mathematically impossible when $N > |D|$; A-TPT's Tammes problem framework naturally handles such cases.
- **vs. Wang & Isola 2020**: They proved the importance of uniformity in contrastive learning; A-TPT applies this insight to TPT calibration.

## Rating
- Novelty: ⭐⭐⭐⭐ Connecting the Tammes problem to VLM calibration is novel, though the core idea (maximizing min distance) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multi-dataset (including medical), multi-backbone, and detailed visual/theoretical support.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and persuasive visualizations.
- Value: ⭐⭐⭐⭐ Provides a practical improvement for test-time calibration of VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning](../../CVPR2026/multimodal_vlm/soc_semantic_orthogonal_calibration_for_test-time_prompt_tuning.md)
- [\[CVPR 2026\] Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining](../../CVPR2026/multimodal_vlm/improving_calibration_in_test-time_prompt_tuning_for_vision-language_models_via_.md)
- [\[ICLR 2026\] Flatness-Guided Test-Time Adaptation for Vision-Language Models](flatness_guided_test-time_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[ICLR 2026\] Bilateral Information-aware Test-time Adaptation for Vision-Language Models](bilateral_information-aware_test-time_adaptation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
