---
title: >-
  [Paper Note] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier
description: >-
  [AAAI 2026 (Oral)][Feature attribution] This paper proposes DFAX, the first distribution-based feature attribution method, which quantifies feature importance by comparing the conditional probability density of a target instance under the target class versus non-target classes. It provides the first formal definition of feature attribution, and demonstrates significant improvements over SHAP/LIME and other baselines across 10 datasets while being orders of magnitude faster.
tags:
  - AAAI 2026 (Oral)
  - Feature attribution
  - explainable AI
  - kernel density estimation
  - model-agnostic
  - distributional methods
date: 2026-05-08
content_hash: 63cda6626a4e8ab8
---

# Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2511.09332](https://arxiv.org/abs/2511.09332)  
**Code**: None  
**Area**: Explainable AI / Feature Attribution  
**Keywords**: Feature attribution, explainable AI, kernel density estimation, model-agnostic, distributional methods

## TL;DR

This paper proposes DFAX, the first distribution-based feature attribution method, which quantifies feature importance by comparing the conditional probability density of a target instance under the target class versus non-target classes. It provides the first formal definition of feature attribution, and demonstrates significant improvements over SHAP/LIME and other baselines across 10 datasets while being orders of magnitude faster.

## Background & Motivation

Feature attribution is a core technique in explainable AI, aiming to assign contribution scores to each input feature of a black-box model to help users understand model decisions. Existing methods fall into two main families:

**Local approximation methods** (e.g., LIME, DLIME, MAPLE): fit simple surrogate models (e.g., linear regression) around the target instance to extract feature importance. However, these methods rely only on a local subset of the dataset and cannot fully exploit global information.

**Perturbation-based methods** (e.g., SHAP, PFI): measure feature importance by perturbing or masking features and observing changes in model output. Many implementations (e.g., SHAP's Shapley sampling values) create synthetic instances by mixing feature values from the target instance and a background dataset, producing out-of-distribution (OOD) data.

**Core Problem**: Despite years of research, feature attribution has lacked a **formal problem definition**. Without a clear standard, comparisons and evaluations among methods lack theoretical grounding.

**Paper Goals**:
- Provide a formal definition of feature attribution (Definition 1), explicitly requiring that explanations be grounded in the underlying data distribution $\mathcal{P}$ without using OOD instances
- Analyze the compliance of existing methods under this definition, revealing that many popular methods (e.g., LIME, common SHAP implementations) fail to satisfy it
- Propose a method that performs feature attribution directly from a distributional perspective, overcoming the limitations of prior work

## Method

### Overall Architecture

DFAX (Distributional Feature Attribution eXplanations) adopts a fundamentally new distributional perspective: it computes feature attribution scores directly from the data distribution, rather than fitting local surrogate models or applying perturbations. The core idea is to compare, for each feature, the probability density of the target instance under the target class versus the non-target classes.

### Key Designs

#### 1. **Formal Problem Definition (Definition 1)**

Given a classifier $f$, a target instance $\mathbf{x}^*$, and a dataset $\mathbf{X}$ (all i.i.d. samples from distribution $\mathcal{P}$), the task of feature attribution is to compute a score $I(\mathbf{x}^*, s | \mathbf{X})$ for each feature $s \in \mathcal{A}$, satisfying:
- The score quantifies the influence of feature value $\mathbf{x}_s^*$ on the classifier's prediction $y^*$
- The explanation model $I(\cdot|\mathbf{X})$ must be constructed directly from the **unmodified** dataset $\mathbf{X}$
- Any modification of $\mathbf{X}$ that alters the underlying distribution invalidates the explanation

The key implication of this definition is the prohibition of synthetic/OOD data for generating explanations. LIME, which generates synthetic neighborhoods via random perturbations, and SHAP's sampling-based implementations, which create synthetic instances by mixing feature values, both violate this criterion.

#### 2. **The DFAX Method (Definition 2)**

For a target instance $\mathbf{x}^*$ (predicted class $y^*$) and feature $s$, the DFAX attribution score is defined as:

$$I(\mathbf{x}^*, s | \mathbf{X}) = K^s(\mathbf{x}^* | \mathbf{X}_{\{y^*\}}) - K^s(\mathbf{x}^* | \mathbf{X} \setminus \mathbf{X}_{\{y^*\}})$$

where:
- $K^s$ denotes kernel density estimation (KDE) on the one-dimensional subspace defined by feature $s$
- $\mathbf{X}_{\{y^*\}}$ is the subset of the dataset predicted as the target class by the classifier
- The first term measures the probability density of the target instance within the **target-class data**
- The second term measures the density within **all other class data**
- A larger difference indicates that the feature value is more discriminative for the target class

**Design Motivation**: If a feature value exhibits high probability density in the target-class distribution and low density in all other class distributions, it serves as a key discriminating feature for the target class and should receive a high attribution score.

#### 3. **KDE Implementation and Acceleration**

DFAX supports two KDE implementations:
- **DFAX_G**: uses Gaussian kernel density estimation (GKDE), with bandwidth $\gamma = \frac{1}{2\sigma^2}$ as the hyperparameter
- **DFAX_S**: uses SiNNE (simplified iNNE), with subsampling size $\psi$ and ensemble count $t$ as hyperparameters

**Key acceleration technique**: If the kernel function can be approximated by a finite-dimensional feature map (e.g., via the Nyström method), the kernel mean embedding $\hat{\Phi}(\mathbf{X})$ of the dataset can be precomputed, after which density estimation for each target instance requires only $\mathcal{O}(1)$ time.

### Loss & Training

DFAX involves no training and operates as a lazy learning method, analogous to KNN. Evaluation uses the standard deletion score (AUC of classification probability as important features are progressively removed; lower is better) and insertion score (AUC as important features are progressively added; higher is better).

**Distinctive advantages of DFAX**:
- **Fully decoupled from the classifier**: once explanations are constructed, the classifier need not be queried again, making it suitable for scenarios where the classifier is unavailable or querying is expensive
- **Global information utilization**: leverages the distribution of the entire dataset $\mathbf{X}$ rather than a local subset
- **Intrinsic data explainability**: replacing predictions with ground-truth labels enables explanation of the inherent class structure of the data

## Key Experimental Results

### Main Results

Evaluations are conducted on 10 real-world datasets (covering tabular, text, and image data, ranging from 520 to 70,000 samples) using multiple classifiers (RF, LR, SVM, MLP, ResNet, etc.).

| Method | Deletion ↓ Mean | Deletion Rank | Insertion ↑ Mean | Insertion Rank |
|--------|----------------|---------------|-----------------|----------------|
| **DFAX_G** | **0.3244** | **1.5** | **0.7708** | **1.2** |
| **DFAX_S** | 0.3344 | 1.6 | 0.7470 | 2.0 |
| DLIME | 0.4595 | 4.1 | 0.6612 | 3.8 |
| LINEX | 0.5287 | 4.8 | 0.6326 | 4.1 |
| SLISE | 0.5457 | 5.5 | 0.6068 | 5.1 |
| SHAP | 0.5246 | 5.4 | 0.5463 | 7.3 |
| MAPLE | 0.5671 | 6.1 | 0.5800 | 6.1 |
| Random | 0.5709 | 7.0 | 0.5838 | 6.4 |

DFAX ranks first or second on 9 out of 10 datasets, substantially outperforming all baselines. The insertion scores of SHAP and MAPLE fall below the random baseline.

### Ablation Study

| Configuration | Key Performance | Notes |
|---------------|----------------|-------|
| DFAX_G (GKDE) | Deletion 0.3244, Insertion 0.7708 | Overall best |
| DFAX_S (SiNNE) | Deletion 0.3344, Insertion 0.7470 | Slightly below GKDE |
| HER2st gene attribution | Accuracy 95.64% vs. DLIME 79.51% | Maintains high predictive accuracy using only half the genes |
| RottenTomatoes sentiment words | Successfully identifies key words such as *compelling* / *bad* | DLIME selects irrelevant words such as *real* / *humor* |

### Key Findings

1. **Computational efficiency**: DFAX is the fastest method across all datasets, typically 1–2 orders of magnitude faster than competing methods. On MNIST, DFAX takes approximately 0.01 seconds, while SHAP requires approximately 100 seconds and SLISE approximately 1,000 seconds.
2. **Definition compliance analysis**: Methods that satisfy Definition 1 (DLIME, SLISE, MAPLE) consistently outperform non-compliant methods (LIME, SHAP sampling implementations), validating the practical significance of the formal definition.
3. **Spatial transcriptomics application**: When identifying key cancer genes on the HER2st dataset, DFAX maintains 95.64% predictive accuracy using only half the genes, compared to 79.51% for DLIME, demonstrating strong potential for real-world scientific discovery applications.

## Highlights & Insights

- **First formal definition of the feature attribution problem**, providing theoretical foundations and evaluation standards for the field while exposing fundamental flaws in popular methods such as LIME and SHAP
- **New distributional paradigm**: departs from the two dominant traditions of surrogate model fitting and perturbation-based observation, instead grounding feature importance directly in conditional probability density
- **Elegant simplicity and efficiency**: the method is defined by a single subtraction operation, enables $\mathcal{O}(1)$ inference after precomputation, and delivers strong performance without hyperparameter tuning
- **Complete decoupling from the classifier**: once predictions are obtained, no further classifier queries are required, making the method suitable for privacy-sensitive or query-expensive settings

## Limitations & Future Work

- Currently limited to single-feature attribution; extension to **feature group attribution** is identified by the authors as ongoing work
- KDE may suffer from reduced efficiency in high-dimensional spaces (the curse of dimensionality), though the paper mitigates this by computing KDE per dimension
- **Axiomatic properties** (e.g., efficiency, symmetry as satisfied by Shapley values) are not discussed and are listed as future work
- A labeled dataset $\mathbf{X}$ (or classifier predictions) is required to partition class subsets

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First formal definition combined with a novel distributional perspective; a pioneering contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 datasets, 6 baselines, comprehensive quantitative, qualitative, and efficiency evaluation
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure with a complete logical chain from definition to analysis, method, and experiments
- **Value**: ⭐⭐⭐⭐ — Significant theoretical and practical contributions to the XAI field; accepted as Oral

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[ICCV 2025\] VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](../../ICCV2025/interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)

<!-- RELATED:END -->
