---
title: >-
  [Paper Note] Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation
description: >-
  [ECCV 2024][Medical Imaging][Semi-supervised Segmentation] Proposes AD-MT (Alternate Diverse Mean Teacher), which addresses the confirmation bias problem in semi-supervised medical image segmentation through random periodic alternate updating of two teacher models and an entropy-based conflict-combating strategy, comprehensively outperforming SOTA methods on ACDC, LA, and Pancreas datasets.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Semi-supervised Segmentation"
  - "Mean Teacher"
  - "Confirmation Bias"
  - "Pseudo Label"
  - "Conflict-Combating"
date: 2026-05-08
content_hash: f74a4ea4b78f93aa
---

# Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2311.17325](https://arxiv.org/abs/2311.17325)  
**Code**: [https://github.com/zhenzhao/AD-MT](https://github.com/zhenzhao/AD-MT)  
**Area**: Medical Image Segmentation / Semi-supervised Learning  
**Keywords**: Semi-supervised Segmentation, Mean Teacher, Confirmation Bias, Pseudo Label, Conflict-Combating

## TL;DR
Proposes AD-MT (Alternate Diverse Mean Teacher), which addresses the confirmation bias problem in semi-supervised medical image segmentation through random periodic alternate updating of two teacher models and an entropy-based conflict-combating strategy, comprehensively outperforming SOTA methods on ACDC, LA, and Pancreas datasets.

## Background & Motivation
**Background**: Mainstream methods in semi-supervised medical image segmentation (SSMIS) are based on consistency regularization, generating pseudo-labels for unlabeled data via a teacher-student framework. The core challenge is confirmation bias, where a single model inevitably generates noisy pseudo-labels and self-reinforces.

**Limitations of Prior Work**:
   - **Single-Teacher (Mean Teacher)**: Employs only a single perspective, lacking a correction mechanism for pseudo-label noise.
   - **Multi-Student Co-training (e.g., MC-Net+)**: Introduces additional training parameters, and differences arising solely from different initializations or learning rates are insufficient.
   - **Multi-Teacher Ensemble (e.g., PS-MT)**: The teacher updating strategies are not carefully designed, leading to insufficient diversity; furthermore, simple averaging discards conflicting information between teachers.

**Key Challenge**: Seeking diverse teacher supervision to alleviate confirmation bias, but making teachers sufficiently different without introducing extra training costs is difficult; additionally, conflicting predictions between teachers are typically discarded, thereby wasting information.

**Goal**: (a) How to make the two teacher models sufficiently different? (b) How to leverage (rather than discard) conflicting predictions between teachers?

**Key Insight**: Intervening at the teacher updating dimension—ensuring diversity through a triple redundancy of complementary data batches, different augmentation strategies, and random switching periods; during conflicts, comparing the teacher ensemble entropy with student entropy and choosing the more confident one.

**Core Idea**: Alternately update two teachers (maximizing diversity via complementary data, different augmentations, and random periods) and utilize an entropy-based conflict-combating module to learn from both consistent and conflicting predictions.

## Method

### Overall Architecture
One trainable student model + two non-trainable teacher models (EMA updated). Only one teacher is updated per iteration, with the two teachers updating alternately. Both teachers simultaneously generate pseudo-labels for unlabeled data, which are fused via the Conflict-Combating Module to supervise the student's predictions on strongly augmented data. The total loss is formulated as $\mathcal{L} = \mathcal{L}_x + \lambda_t \mathcal{L}_u$.

### Key Designs

1. **Random Periodic Alternate (RPA) Updating Module**:

    - **Function**: To ensure the two teacher models are as distinct as possible.
    - *Triple Diversity Strategy*:
        - *Complementary Data Batches*: Only one teacher is updated at a time, making the unlabeled data seen by the two teachers during a training period fully complementary.
        - *Different Augmentation Strategies*: T1 uses color-jittering while T2 uses copy-paste augmentation—distinctly different in strength and nature.
        - *Random Switching Periods*: Instead of fixed alternation intervals, a new period is randomly generated from $[0, \mathcal{T}_{max}]$ at each switch.
    - **Design Motivation**: Introducing divergence across three dimensions simultaneously—data, augmentation, and update pacing—to ensure teachers generate truly distinct "perspectives" in the feature space.

2. **Conflict-Combating Module (CCM)**:

    - **Function**: To handle pixels where the two teachers' predictions do not match—not by discarding, but by leveraging them.
    - **Mechanism**: Pixel-wise processing:
        - *When teachers agree*: Use an entropy-based weighted ensemble $\psi_i = \frac{w_1 q_i^{t_1} + w_2 q_i^{t_2}}{w_1 + w_2}$, where $w_k = e^{-H_{t_k}}$ (low entropy = high weight).
        - *When teachers conflict*: Compare the entropy of the ensemble prediction $H_{\psi_i}$ with that of the student prediction $H_{q_i^s}$, selecting the one with lower entropy (higher confidence) as supervision.
    - **Design Motivation**: In the late stages of training, the student might become more accurate than the teachers in certain regions (as the student processes all data); selecting the student during conflicts avoids penalizing the student with incorrect teacher predictions.

### Loss & Training
- Supervised Loss: Average of Dice + CE.
- Unsupervised Loss: Also Dice + CE, but using the pseudo-labels fused via CCM.
- Confidence Threshold $\tau$: 0.95 for 2D datasets and 0.75 for 3D datasets (a high threshold in 3D would filter out too much information).
- EMA parameter of 0.99, maximum period $\mathcal{T}_{max} = 0.5$ epoch.

## Key Experimental Results

### Main Results

| Dataset | Labeled Ratio | Metric (Dice%) | AD-MT | BCP (prev SOTA) | Gain |
|--------|---------|-------------|-------|-----------------|------|
| LA (3D) | 5% (4 cases) | Dice | **89.63** | 88.02 | +1.61 |
| LA (3D) | 10% (8 cases) | Dice | **90.55** | 89.62 | +0.93 |
| ACDC (2D) | 5% (3 cases) | Dice | **88.75** | 87.59 | +1.16 |
| ACDC (2D) | 10% (7 cases) | Dice | **89.46** | 88.84 | +0.62 |
| Pancreas (3D) | 10% (6 cases) | Dice | **80.21** | 73.83 | **+6.38** |
| Pancreas (3D) | 20% (12 cases) | Dice | 82.61 | **82.91** | -0.30 |

### Ablation Study

| Configuration | ACDC Dice | ACDC 95HD | Description |
|------|-----------|-----------|------|
| T1 only | 86.83 | 2.65 | Single-teacher baseline |
| T2 only | 86.22 | 2.43 | Copy-paste augmentation is slightly worse |
| T1+T2+RPA (no CCM) | 87.88 | 2.03 | Alternate updating yields 1%+ improvement |
| **T1+T2+RPA+CCM (Full)** | **88.75** | **1.48** | CCM further yields 0.87% improvement |

### Key Findings
- **Most significant improvement under Pancreas 10% (+6.38%)**: The advantage of AD-MT is most pronounced in low-data scenarios, indicating that diverse supervision from the two teachers carries the highest value when labeled data is extremely scarce.
- **RPA module contributes the most (+1.05%)**: Transitioning from a single teacher to RPA twin teachers represents the largest source of improvement.
- **CCM is more valuable in later training stages**: Conflicts between teachers increase in later stages of training, where the student model has already acquired substantial capability and can provide valuable alternative supervision.
- **Threshold $\tau$ sensitivity differs between 2D and 3D datasets**: A high threshold (0.95) works well for 2D, while a lower threshold (0.75) is better for 3D.
- **Outperforms BCP without pre-training**: BCP requires an additional pre-training phase, whereas AD-MT's end-to-end training is more elegant and concise.

## Highlights & Insights
- **Addressing confirmation bias from the perspective of teacher updating strategies**: Rather than relying on architectural differences or extra loss functions, the method creates differences between teachers across three dimensions: data partition, augmentation strategy, and update frequency. The approach is simple yet effective.
- **Leveraging rather than discarding conflicts**: Standard approaches typically average or discard conflicting teacher predictions. In contrast, CCM compares entropy to select the most confident prediction, an insight that offers high referential value for multi-model ensemble methods.
- **Random periods are key**: Fixed alternating periods might lead to a fixed data distribution observed by each of the two teachers; randomization breaks this rigid pattern.

## Limitations & Future Work
- **Evaluated only on U-Net/V-Net backbones**: Validation has not been performed on stronger backbones (e.g., nnU-Net, Swin UNETR).
- **Slightly lower performance than BCP on the Pancreas 20% setting**: The comparative advantage diminishes under higher labeled ratios, indicating that diverse supervision experiences diminishing returns as labeled data increases.
- **The augmentation strategies for the two teachers are manually designed**: The choice of color-jittering vs. copy-paste was not automatically searched, and other augmentation combinations might perform better.
- **Risks associated with CCM's student substitution strategy**: In the early stages of training, the student is quite weak, and choosing the student's prediction during conflicts may introduce more noise—suggesting the need for a warm-up mechanism.

## Related Work & Insights
- **vs. PS-MT**: PS-MT also utilizes multiple teachers but updates them at different epochs. This paper implements alternation within the same epoch, combined with complementary data and distinct augmentations, enabling more thorough differentiation.
- **vs. MC-Net+**: MC-Net+ uses co-training with two students, introducing additional parameters. AD-MT only trains a single student (with teachers updated via EMA), incurring no additional training costs.
- **vs. BCP**: BCP requires a pre-training schedule to establish a reliable initialization, while AD-MT runs end-to-end without pre-training—yet achieves a 6.38% gain on the Pancreas 10% setting.

## Rating
- Novelty: ⭐⭐⭐⭐ RPA and CCM designs are innovative, and the triple-differentiation concept of the alternate updating strategy is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 3 datasets, 2D/3D setups, detailed ablation studies, threshold sensitivity, and class-level analysis.
- Writing Quality: ⭐⭐⭐⭐ Diagrams are clear, and the contrastive framework against existing methods is well-depicted.
- Value: ⭐⭐⭐⭐ Simple yet highly effective, presenting direct value to semi-supervised medical image segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/sam_dpo_semi_supervised.md)
- [\[NeurIPS 2025\] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](../../NeurIPS2025/medical_imaging/vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](../../AAAI2026/medical_imaging/dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)
- [\[CVPR 2026\] Divide, Conquer, and Aggregate: Asymmetric Experts for Class-Imbalanced Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/divide_conquer_and_aggregate_asymmetric_experts_for_class-imbalanced_semi-superv.md)

</div>

<!-- RELATED:END -->
