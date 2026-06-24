---
title: >-
  [Paper Note] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models
description: >-
  [ICML 2025][Multimodal VLM][Vision-Language Models] By analyzing the root cause of VLM calibration failure during prompt tuning (text feature shift), this paper proposes a Dynamic Outlier Regularization (DOR) method. It utilizes high semantic similarity nouns from WordNet as text outliers to constrain feature drift during fine-tuning, significantly reducing calibration error.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Prompt Tuning"
  - "Calibration"
  - "CLIP"
  - "Outlier Regularization"
date: 2026-05-08
content_hash: c3adda76d5681feb
---

# Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models

**Conference**: ICML 2025  
**arXiv**: [2410.02681](https://arxiv.org/abs/2410.02681)  
**Code**: [ml-stat-Sustech/Outlier-Calibration](https://github.com/ml-stat-Sustech/Outlier-Calibration)  
**Area**: Multimodal VLMs  
**Keywords**: Vision-Language Models, Prompt Tuning, Calibration, CLIP, Outlier Regularization

## TL;DR

By analyzing the root cause of VLM calibration failure during prompt tuning (text feature shift), this paper proposes a Dynamic Outlier Regularization (DOR) method. It utilizes high semantic similarity nouns from WordNet as text outliers to constrain feature drift during fine-tuning, significantly reducing calibration error.

## Background & Motivation

**Background**: Large-scale vision-language models like CLIP can achieve excellent classification performance on downstream tasks through prompt tuning. Methods such as CoOp, MaPLe, and KgCoOp have demonstrated good accuracy in base-to-new generalization settings.

**Limitations of Prior Work**: Despite good accuracy, these methods suffer from severe issues in **calibration**. Specifically, there is a significant discrepancy between the model's confidence and its actual accuracy, which is unacceptable in safety-critical applications (such as medical diagnosis and autonomous driving).

**Key Challenge**: Standard prompt tuning methods (e.g., CoOp) exhibit overconfidence on new classes, whereas regularized methods (e.g., KgCoOp) mitigate overconfidence on new classes but lead to underconfidence on base classes. Neither strategy can simultaneously calibrate both base and new classes.

**Goal**: To identify the root cause of calibration failure in prompt tuning and propose a unified method that simultaneously reduces calibration error for both base and new classes.

**Key Insight**: The authors delve into the changes in the feature space and discover that **text feature shift** is the root cause of calibration failure. The fine-tuned text encoder produces a feature distribution different from that of zero-shot CLIP, leading to a shift in the classification decision boundary.

**Core Idea**: Utilize nouns in WordNet that are semantically close to target classes but do not belong to any known classes as "outlier anchors". Features of these outliers are constrained through regularization to remain consistent before and after fine-tuning, thereby suppressing text feature drift.

## Method

### Overall Architecture

The DOR method adds an outlier regularization term to standard prompt tuning (e.g., CoOp). During training, a batch of nouns semantically similar to target classes is dynamically sampled from WordNet as "text outliers" at each epoch. The loss function then constrains the feature representations of these outliers in the fine-tuned text encoder to remain as close as possible to their features in the original zero-shot CLIP. The total loss is $L_{total} = L_{ce} + \lambda \cdot L_{dor}$.

### Key Designs

1. **Text Outlier Selection Strategy**:

    - **Function**: Select words that are semantically close but non-overlapping with the target task classes from a large-scale noun database to serve as regularization anchors.
    - **Mechanism**: Utilizing approximately 80,000 nouns in WordNet, the cosine similarity between each noun and the target classes in the zero-shot CLIP text feature space is calculated. The Top-K nouns with the highest similarity (excluding words that completely overlap with target classes) are selected as near-OOD outliers.
    - **Design Motivation**: Semantically close outliers reside near the classification decision boundaries, and their feature drifts have the most significant impact on calibration. Compared to random or far-OOD sampling, near-OOD outliers more effectively constrain feature stability near the decision boundary region.

2. **Dynamic Outlier Regularization Loss (DOR Loss)**:

    - **Function**: Constrain the magnitude of text feature drift during fine-tuning.
    - **Mechanism**: $L_{dor} = 1 - \frac{1}{B}\sum_{b=1}^{B} \text{sim}(\psi(t'_{o_b}), \psi(t_{o_b}))$, where $\psi(t'_{o_b})$ and $\psi(t_{o_b})$ represent the text features of the outlier words $o_b$ from the fine-tuned and zero-shot CLIP models, respectively, and $\text{sim}$ denotes the cosine similarity.
    - **Design Motivation**: Directly constraining target class features limits the adaptation capability of the model. Indirect constraints through outliers allow target features to adapt appropriately for higher accuracy while preventing excessive structural drifts in the overall feature space.

3. **Dynamic Sampling Mechanism**:

    - **Function**: Re-sample the outlier set at each training epoch to prevent overfitting to fixed outliers.
    - **Mechanism**: At the start of each epoch, a new batch of outliers is weighted-sampled from the candidate outlier pool based on semantic similarity, ensuring the diversity of regularization signals.
    - **Design Motivation**: Fixed outliers might lead to feature consistency only on those specific words while ignoring drifts in other regions. Dynamic sampling provides broader coverage.

### Loss & Training

The total loss function is a weighted sum of the standard cross-entropy loss and the DOR regularization term:

$$L_{total} = L_{ce}(x, y) + \lambda \cdot L_{dor}$$

where $\lambda$ controls the regularization strength. The DOR loss is based on cosine similarity, measuring the discrepancy between text features of outliers before and after fine-tuning. Regarding training hyperparameters, $\lambda$ is tuned on the validation set, the number of outliers $B$ is set to a moderate value (e.g., 50-100), and outlier sampling is updated once per epoch. DOR is a plug-and-play module that can be directly integrated into any prompt tuning method, such as CoOp, MaPLe, KgCoOp, or TCP.

## Key Experimental Results

### Main Results

Base-to-new generalization calibration results on 11 datasets (ECE %, lower is better):

| Method | Base ECE | New ECE | HM ECE |
|:---|:---:|:---:|:---:|
| Zero-shot CLIP | 4.30 | 5.79 | 5.05 |
| CoOp | 3.07 | 14.58 | 8.82 |
| CoOp + DOR | **2.67** | **6.49** | **4.58** |
| MaPLe | 3.10 | 8.98 | 6.04 |
| MaPLe + DOR | **2.73** | **6.33** | **4.53** |
| KgCoOp | 4.86 | 6.56 | 5.71 |
| KgCoOp + DOR | **3.31** | **5.47** | **4.39** |
| TCP | 2.92 | 7.82 | 5.37 |
| TCP + DOR | **2.57** | **5.95** | **4.26** |

DOR achieves significant ECE reduction across all four methods, with an average reduction of 8.09%.

Domain generalization experiments (ImageNet $\rightarrow$ ImageNet variants):

| Method | Source ECE | Target ECE |
|:---|:---:|:---:|
| CoOp | 2.43 | 7.18 |
| CoOp + DOR | **2.28** | **4.89** |

### Ablation Study

Comparison of outlier selection strategies (CoOp method, 11-dataset average, ECE %):

| Outlier Type | Base ECE | New ECE | HM ECE |
|:---|:---:|:---:|:---:|
| No Regularization (Original CoOp) | 3.07 | 14.58 | 8.83 |
| Near-OOD (Ours) | **2.68** | 7.09 | **4.89** |
| Far-OOD | 2.95 | 7.72 | 5.34 |
| Random | 2.80 | 7.33 | 5.07 |
| Oracle (Known New Classes) | 3.13 | **4.34** | 3.74 |

Visual prompt tuning validation (VPT method, DTD dataset):

| Method | New ECE |
|:---|:---:|
| VPT | 13.04 |
| VPT + DOR | **8.40** |

### Key Findings

- As a plug-and-play module, DOR significantly reduces calibration errors across four prompt tuning methods (CoOp, MaPLe, KgCoOp, and TCP), with an average reduction of 8.09%.
- The performance of Near-OOD outliers is significantly better than Far-OOD and Random strategies, validating the importance of semantic proximity for calibration correction.
- DOR reduces calibration errors with almost no loss in classification accuracy: base class accuracy is 83.20% vs 82.97%, and new class accuracy remains largely comparable.
- The finding that text feature shift is the root cause of calibration failure is highly generalizable; DOR is even effective for Visual Prompt Tuning (VPT).
- DOR is equally effective in domain generalization scenarios: ECE decreases from 7.18% to 4.89% on ImageNet variants.

## Highlights & Insights

- **Deep Root-Cause Analysis**: Instead of simply applying standard calibration methods (e.g., temperature scaling), this work identifies text feature shift as the root cause from a feature-space perspective, providing valuable theoretical guidance for future research.
- **Elegant Design**: As a lightweight plug-and-play module, DOR does not change the training pipeline of existing prompt tuning methods, only introducing a single regularization term, making it extremely practical.
- **WordNet as Outlier Source**: It cleverly leverages the rich semantic structure provided by linguistic knowledge bases, requiring no extra annotated data.
- **Comprehensive Experimental Validation**: Covering 4 prompt tuning methods, 11 datasets, domain generalization, and visual prompt tuning, demonstrating high credibility.

## Limitations & Future Work

- Outlier sampling relies on WordNet, which might need to be replaced with domain-specific vocabularies for non-English scenarios or professional domains (e.g., medical terms).
- The Top-K parameter for text outlier selection and $\lambda$ require adjustment on a validation set, which increases hyperparameter tuning costs.
- The current analysis focuses primarily on feature drift on the text side; the impact of vision-side features on calibration remains to be explored.
- It is validated only on classification tasks; calibration issues in dense prediction tasks like detection and segmentation are not yet addressed.

## Related Work & Insights

- **vs CoOp**: Learnable prompts in CoOp exhibit reasonable calibration on base classes (3.07%) but suffer from heavy overconfidence on new classes (14.58%). DOR reduces the new-class ECE to 6.49%.
- **vs KgCoOp**: KgCoOp mitigates overconfidence on new classes by constraining the learnable prompt not to deviate far from hand-crafted prompts, but leads to underconfidence on base classes (4.86%). DOR addresses feature drift directly, simultaneously improving both ends.
- **vs Temperature Scaling**: Post-processing calibration methods require an additional validation set and cannot improve the representation learning itself. DOR directly optimizes feature space stability during training.
- **vs ProDA**: ProDA improves generalization through distribution alignment but does not focus specifically on calibration, whereas DOR explicitly targets calibration.

## Rating

- Novelty: ⭐⭐⭐⭐ The first to systematically analyze calibration issues in prompt tuning and locate the cause as text feature shift. The outlier regularization design is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage with 11 datasets, 4 methods, domain generalization, visual prompt tuning, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis, natural motivation derivation, and well-organized experiments.
- Value: ⭐⭐⭐⭐ Calibration is crucial for VLM deployment, and the plug-and-play nature of DOR gives it high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2025\] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](../../CVPR2025/multimodal_vlm/dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [\[ICLR 2026\] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](../../ICLR2026/multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](../../CVPR2026/multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](../../ICCV2025/multimodal_vlm/fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
