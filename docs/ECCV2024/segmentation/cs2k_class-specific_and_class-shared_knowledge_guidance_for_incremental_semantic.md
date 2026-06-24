---
title: >-
  [Paper Note] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation
description: >-
  [ECCV2024][Segmentation][Incremental Semantic Segmentation] Proposed the Cs2K framework, which synergistically mitigates catastrophic forgetting and underfitting of new categories in incremental semantic segmentation from two aspects: class-specific knowledge (prototype-guided pseudo-labeling + prototype-guided class adaptation) and class-shared knowledge (weight-guided selective consolidation).
tags:
  - "ECCV2024"
  - "Segmentation"
  - "Incremental Semantic Segmentation"
  - "Class-specific Knowledge"
  - "Class-shared Knowledge"
  - "Prototype"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 1cfe9eb78a6bd3c7
---

# Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation

**Conference**: ECCV2024  
**arXiv**: [2407.09047](https://arxiv.org/abs/2407.09047)  
**Code**: To be confirmed  
**Area**: Image Segmentation  
**Keywords**: Incremental Semantic Segmentation, Class-specific Knowledge, Class-shared Knowledge, Prototype, catastrophic forgetting

## TL;DR
Proposed the Cs2K framework, which synergistically mitigates catastrophic forgetting and underfitting of new categories in incremental semantic segmentation from two aspects: class-specific knowledge (prototype-guided pseudo-labeling + prototype-guided class adaptation) and class-shared knowledge (weight-guided selective consolidation).

## Background & Motivation
Incremental Semantic Segmentation (ISS) requires models to retain their segmentation capability on old classes while learning new ones. Prior methods suffer from two types of bias:

1. **Lack of class-specific knowledge guidance**: Relying solely on old model weights (class-shared knowledge) fails to targetedly correct decision boundaries for old categories, leading the model to bias toward new categories.
2. **Indiscriminate constraints on class-shared knowledge**: Equalling fusing or constraining all weights of the old model biases the model toward old categories, leading to insufficient learning of new categories.

The Key Challenge lies in: the class distribution in datasets across different training steps varies dramatically (each step only contains annotations of current foreground classes), and old class pixels are labeled as background, causing category over-representation and drastic changes in decision boundaries.

## Core Problem
How to simultaneously leverage both **class-specific knowledge** (old class prototypes) and **class-shared knowledge** (old model weights) to balance performance between old and new categories, thereby overcoming catastrophic forgetting without storing past samples?

## Method

### Overall Architecture
Cs2K consists of three core modules; the first two are designed from the perspective of class-specific knowledge, while the third leverages class-shared knowledge:

### 1. Prototype-guided Pseudo Labeling (PPL)
**Function**: Utilizes old class prototypes to correct old class pixels falsely classified into the background, generating high-quality pseudo-labels.

- At the end of step $t{-}1$, compute the prototype $\eta_c$ for each old class (the average of all pixel features of this class).
- For background pixels at the current step $t$, compute the similarity weight $\kappa_{i,c}^t$ between their features and each old class prototype (based on the softmax of feature distance).
- Multiply the similarity weight by the prediction probability of the old model to obtain the corrected probability, which is used to correct pseudo-labels:
    - If the ground truth (GT) is a foreground class → use GT directly.
    - If the GT is background and the corrected probability points to an old class → set the pseudo-label to that old class.
    - Otherwise → pseudo-label remains as background.
- Train the model using the generated pseudo-labels via cross-entropy loss $\mathcal{L}_{pl}$.

**Key Designs**: Prototypes are unaffected by outliers and treat classes with different frequencies of occurrence equally, making them more reliable than direct predictions of the old model.

### 2. Prototype-guided Class Adaptation (PCA)
**Function**: Enhances training by utilizing old class prototypes to maintain discriminability between old and new classes.

Two augmentation strategies are included:

- **Self-prototype Augmentation**: $\Gamma_c = \eta_c + \mu \cdot s^t$, where $\mu \sim \mathcal{N}(0,1)$, and $s^t$ is a dynamic scaling factor weighted by the number of classes, helping the model explore the feature space.
- **Inter-prototype Augmentation**: $\Pi_c = \lambda \cdot \eta_c + (1{-}\lambda) \cdot \eta_{c'}$, performing Mixup interpolation on different old class prototypes to enhance discriminability between classes.

The augmented prototypes are fed into the classifier and jointly trained with cross-entropy loss $\mathcal{L}_{pa}$ to maintain the classifier's decision-making ability for old classes without requiring old samples.

### 3. Weight-guided Selective Consolidation (WSC)
**Function**: Selectively consolidates new and old models at the weight level to balance old and new knowledge.

- Compute the importance of each parameter of the old model to old classes, $F_i^{t-1}$, using Fisher Information.
- Rank the parameters by importance, and select the Top-$\beta$ proportion of important weights for weighted fusion: $\Theta_i^t = \omega \cdot \Theta_i^{t-1} + (1{-}\omega) \cdot \Theta_i^t$.
- Directly use the new model's parameters for the remaining weights.
- Both $\beta$ and $\omega$ are dynamic factors adaptively adjusted according to the ratio of the number of old classes to new classes:
    - $\beta$ is designed via a sigmoid function; more old classes lead to retaining more important weights.
    - $\omega$ is designed via a power function to control the constraint strength of the old weights.

### Total Loss
$$\mathcal{L} = \mathcal{L}_{pl} + \mathcal{L}_{pa}$$

After training, perform WSC to consolidate weights. The entire method is plug-and-play and can be combined with baseline methods such as MiB, PLOP, etc.

## Key Experimental Results

### Pascal VOC 2012

| Method | 15-1 (all) | 10-1 (all) | 5-3 (all) |
|------|-----------|-----------|----------|
| MiB | 32.2 | 12.6 | 46.7 |
| PLOP | 54.6 | 30.5 | 28.7 |
| MiB+EWF | 65.5 | 37.3 | 51.8 |
| PLOP+EWF | 67.0 | 51.9 | 47.7 |
| **MiB+Cs2K** | **68.0** | **39.3** | **56.2** |
| **PLOP+Cs2K** | **70.4** | **61.5** | **54.8** |

- Under the 10-1 scenario (11 steps, the most challenging), PLOP+Cs2K achieves **9.6%** higher mIoU than PLOP+EWF.
- The improvement is particularly prominent on new classes: the gain is 13.7% on new classes in the 15-1 scenario and 16.9% on new classes in the 10-1 scenario.

### ADE20K
- 100-10 scenario: MiB+Cs2K reaches 34.1 mIoU and PLOP+Cs2K reaches 35.4 mIoU, both surpassing their corresponding EWF variants.
- 100-5 scenario: MiB+Cs2K reaches 34.2 mIoU, which is 2.1% higher than MiB+EWF.

### Ablation Study (15-1 Scenario)

| Removed Module | mIoU (all) | Drop |
|---------|-----------|---------|
| W/o PPL | 65.3 | -5.1 |
| W/o PCA | 68.7 | -1.7 |
| W/o WSC | 48.6 | -21.8 |
| **Full Cs2K** | **70.4** | - |

WSC contributes the most (-21.8), demonstrating that selective consolidation at the weight level is the core guarantee of performance.

## Highlights & Insights
1. **Dual Knowledge Collaboration Framework**: A systematic combination of class-specific knowledge and class-shared knowledge, representing an early exploration in this direction.
2. **Prototype-guided Pseudo-label Correction**: Weighting and correcting pseudo-labels from the old model using prototype distance, which is more robust than relying solely on old model predictions or entropy threshold filtering.
3. **Selective Weight Consolidation**: Fuses important parameters based on Fisher information instead of equally constraining all parameters, preventing insufficient learning of new classes.
4. **Plug-and-play Design**: Can be directly applied on top of existing methods like MiB and PLOP.
5. **Dynamic Hyperparameters**: $\beta$, $\omega$, and $s^t$ are all adaptively adjusted according to incremental steps without manual tuning.

## Limitations & Future Work
1. **Performance Gap with Joint Training**: Performance still falls short of the joint training upper bound in long-sequence tasks, which is explicitly mentioned by the authors in the conclusion.
2. **Prototype Quality Dependent on the Prior Step**: Prototypes are computed and frozen at the end of step $t{-}1$. If the model quality in the prior step is poor, the prototype will also be biased.
3. **Fisher Information Computational Overhead**: Requires extra forward propagation to calculate Fisher Information for all parameters, increasing training costs.
4. **Domain Shift Unconsidered**: If domain gaps exist between data from different steps (beyond just class differences), the current prototype-based approach may fail.
5. **Insufficient Validation in Large-scale/Many-step Scenarios**: Tested on at most 150 classes of ADE20K, lack of validation on larger scale datasets.

## Related Work & Insights
- **vs. EWF**: EWF equally fuses all weights of the old and new models without distinguishing parameter importance; Cs2K selectively fuses important parameters using Fisher information and additionally introduces prototype-level knowledge guidance.
- **vs. PLOP**: PLOP only uses multi-scale feature distillation to constrain representation consistency; Cs2K adds prototype-guided pseudo-label correction and selective weight consolidation on top of this.
- **vs. RCIL / GSC**: These methods exhibit unstable performance across different scenarios; Cs2K shows stable improvements in all scenarios.
- **vs. Rehearsal-based methods (ALIFE, etc.)**: Does not require storing old samples, preserving data privacy while using only lightweight prototypes instead.

## Inspiration & Connections
- **Prototype augmentation strategies** are worth referencing in other incremental learning scenarios, such as incremental object detection and incremental instance segmentation.
- **The selective weight consolidation concept** can be generalized to the model merging field, selectively merging parameters based on task importance.
- The "distance weight $\times$ probability" paradigm in pseudo-label correction is applicable to any scenario requiring label repair using prototypes when annotations are scarce.

## Rating
- Novelty: ⭐⭐⭐⭐ — For the first time systematically combining two types of knowledge, with well-designed modules.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation on multiple scenarios of VOC and ADE20K, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clearly categorized with well-defined motivation.
- Value: ⭐⭐⭐⭐ — The plug-and-play framework offers practical reference value to the ISS community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)
- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](../../ICCV2025/segmentation/know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[ECCV 2024\] CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation](cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)
- [\[ECCV 2024\] Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation](learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)
- [\[CVPR 2026\] Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/leveraging_class_distributions_in_clip_for_weakly_supervised_semantic_segmentati.md)

</div>

<!-- RELATED:END -->
