---
title: >-
  [Paper Note] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction
description: >-
  [ICCV 2025][Segmentation][Semi-supervised segmentation] This paper proposes ConformalSAM, a framework that leverages Conformal Prediction to calibrate the output uncertainty of the foundation segmentation model SEEM on target domains. Unreliable pixel labels are filtered out before serving as supervision signals for unlabeled data. Combined with a late-stage self-reliance training strategy, the framework achieves 81.21 mIoU on PASCAL VOC under the 1/16 labeled setting.
tags:
  - ICCV 2025
  - Segmentation
  - Semi-supervised segmentation
  - conformal prediction
  - SAM/SEEM
  - uncertainty calibration
  - pseudo labels
date: 2026-05-08
content_hash: 6090fc2774cfe5db
---

# ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction

**Conference**: ICCV 2025
**arXiv**: [2507.15803](https://arxiv.org/abs/2507.15803)
**Code**: N/A
**Area**: Image Segmentation
**Keywords**: Semi-supervised segmentation, conformal prediction, SAM/SEEM, uncertainty calibration, pseudo labels

## TL;DR

This paper proposes ConformalSAM, a framework that leverages Conformal Prediction to calibrate the output uncertainty of the foundation segmentation model SEEM on target domains. Unreliable pixel labels are filtered out before serving as supervision signals for unlabeled data. Combined with a late-stage self-reliance training strategy, the framework achieves 81.21 mIoU on PASCAL VOC under the 1/16 labeled setting.

## Background & Motivation

The core challenge of semi-supervised semantic segmentation (SSSS) is how to effectively exploit large amounts of unlabeled data. A natural approach is to use foundation segmentation models such as SAM/SEEM to directly generate pseudo labels for unlabeled data — however, experiments show this **actually degrades performance**:
- PASCAL VOC 1/16 split: 50.65 mIoU with labels only → drops to 42.00 when SEEM pseudo labels are added
- Reason: A domain gap exists between SEEM's pretraining data and the target domain, resulting in inconsistent prediction quality on the target domain

Core problem: **How can one reliably exploit the powerful capabilities of foundation models while filtering out their unreliable predictions?**

This paper adopts Conformal Prediction (CP) as the uncertainty calibration tool because: (1) CP is a black-box method that requires only a small amount of labeled data for calibration; (2) it provides theoretically guaranteed coverage; (3) it does not require modifying the foundation model.

## Method

### Overall Architecture

ConformalSAM employs a two-stage training procedure:
- **Stage I**: Joint training with CP-calibrated SEEM pseudo labels and ground-truth labels
- **Stage II**: SEEM pseudo labels are discarded and training switches to Self-Reliance

### Key Designs

1. **CP-calibrated Foundation Model Inference (Stage I)**:

    - **Calibration procedure**: The labeled set $D_l$ is used as the calibration set
        - SEEM generates a probability map $P_i \in \mathbb{R}^{K \times H \times W}$ for each labeled image
        - Nonconformity scores are computed as: $\hat{P}_i^j(a,b) = 1 - P_i^j(a,b)$ (only for pixels of the ground-truth class)
        - Nonconformity scores are aggregated across all images to compute the $(1-\alpha)$ quantile threshold $\hat{q}_\alpha$
    - **Calibrated inference**: For unlabeled image $x_i$, the prediction set at pixel $(a,b)$ is $\mathcal{C}_i(a,b) = \{j: \hat{P}_i^j(a,b) \leq \hat{q}_\alpha(a,b)\}$
    - **Class-conditional filtering**: Since background pixels dominate, when both background and non-background classes appear in the prediction set, non-background classes are prioritized:
    $M_i(a,b) = \begin{cases} \arg\min_j \mathcal{C}_i[j], & |\mathcal{C}_i| > 0 \land 0 \notin \mathcal{C}_i \\ \arg\min_{j \neq 0} \mathcal{C}_i[j], & |\mathcal{C}_i| > 0 \land 0 \in \mathcal{C}_i \\ \text{NaN}, & |\mathcal{C}_i| = 0 \end{cases}$
    - When the prediction set is empty, the pixel label is set to NaN (ignored), effectively filtering low-confidence predictions
    - Miscoverage rate $\alpha = 0.05$

2. **Self-Reliance Training Strategy (Stage II)**:

    - SEEM-generated masks are discarded; the model's own pseudo labels are used instead
    - Dynamic weight decay strategy: $\mathcal{L} = (1 - \lambda(t)) \times \mathcal{L}_s + \lambda(t) \times \mathcal{L}_u$
    - $\lambda(t)$ decays exponentially, so that the model relies increasingly on ground-truth supervision in later epochs
    - PASCAL VOC: Stage I for 60 epochs, Stage II for 20 epochs
    - ADE20K: Stage I for 30 epochs, Stage II for 10 epochs

3. **Flexible Plug-in Design**:

    - The Stage II self-training framework can be replaced by other methods such as AllSpark
    - ConformalSAM(AllSpark): Stage I uses CP-calibrated pseudo labels, Stage II switches to AllSpark
    - This demonstrates the generality and composability of the framework

### Loss & Training

- Labeled data: standard cross-entropy loss
- Unlabeled data (Stage I): NaN pixels are ignored; CE is computed only on high-confidence pixels selected by CP
- Stage II applies exponentially decayed weights to balance supervised and unsupervised losses
- SegFormer-B5 is used as the segmentation backbone

## Key Experimental Results

### Main Results

| Method | VOC 1/16(92) | VOC 1/8(183) | VOC 1/4(366) | VOC 1/2(732) | VOC Full |
|------|------------|------------|------------|------------|---------|
| UniMatch | 75.2 | 77.2 | 78.8 | 79.9 | - |
| AllSpark | 76.07 | 78.41 | 79.77 | 80.75 | 82.12 |
| **ConformalSAM(AllSpark)** | 80.69 | 81.29 | 81.33 | 82.69 | 83.44 |
| **ConformalSAM** | **81.21** | **82.22** | **81.84** | **83.52** | **83.85** |

| Method | ADE20K 1/128(158) | 1/64(316) | 1/32(632) | 1/16(1263) | 1/8(2526) |
|------|-----------------|---------|---------|----------|---------|
| AllSpark | 16.17 | 23.03 | 26.42 | 28.40 | 32.10 |
| **ConformalSAM** | **26.21** | **30.02** | **33.33** | **34.64** | **36.25** |

### Ablation Study

| Configuration | SEEM | CP | SR | VOC 1/16 | VOC 1/2 |
|------|------|----|----|---------|---------|
| Semi-Baseline | ✗ | ✗ | ✗ | 52.89 | 74.22 |
| +SEEM (direct) | ✓ | ✗ | ✗ | 42.00 | 44.99 |
| +SEEM+CP | ✓ | ✓ | ✗ | 78.09 | 79.10 |
| +SEEM+CP+SR | ✓ | ✓ | ✓ | **81.21** | **83.52** |

| CP Variant | α=0.1 | α=0.05 | α=0.01 |
|--------|-------|--------|--------|
| Pixel-wise | 74.31 | **78.09** | 68.01 |
| Image-wise | 75.99 | 75.54 | 44.59 |
| K-Means | 69.36 | 69.13 | 44.16 |

### Key Findings

- **Critical role of CP**: Directly applying SEEM reduces mIoU by 8.65; adding CP yields a gain of 25.2 mIoU (1/16 setting)
- Class-conditional filtering is essential: compared to vanilla CP, it brings an average gain of 34.11 mIoU
- Pixel-wise CP outperforms image-wise, K-Means, GenAnn, and other CP variants
- $\alpha=0.05$ is consistently the optimal miscoverage rate
- The SR strategy contributes an additional average gain of 3.76 mIoU
- On ADE20K under the 1/128 setting, the improvement reaches 10.04 mIoU (AllSpark: 16.17 → 26.21)
- When integrated as a plug-in into AllSpark, an average gain of 2.07 mIoU is achieved

## Highlights & Insights

- **First application of CP to calibrate pseudo labels from foundation segmentation models in SSSS**, with a concise idea backed by strong empirical validation
- Class-conditional filtering addresses the critical failure mode of SEEM in segmentation tasks — foreground being overwhelmed by background pixels
- The two-stage strategy follows a clear design logic: exploit foundation model knowledge early, then avoid overfitting to SEEM noise in later training
- As a plug-in framework, it can be freely combined with existing SSSS methods such as AllSpark

## Limitations & Future Work

- Effectiveness depends on the overlap between foundation model knowledge and the target task — gains are smaller on datasets with novel categories such as ADE20K and Cityscapes
- CP calibration requires labeled data; calibration accuracy may be insufficient in extremely low-label scenarios (e.g., tens of images)
- Only SEEM is evaluated as the foundation model; stronger models such as SAM2 and GLAMM remain unexplored
- The switching point for SR training is determined empirically (60 epochs) and may require adjustment for different datasets
- In-depth comparison with prompt-engineering-based SAM approaches is absent

## Related Work & Insights

- **UniMatch/AllSpark**: Current SSSS state-of-the-art methods; ConformalSAM is complementary to them
- **SemiSAM/CPC-SAM**: Leverage SAM via improved prompting, whereas this paper directly uses SEEM outputs with CP calibration
- **Conformal Prediction**: Imported from classification/detection into segmentation as an uncertainty calibration tool
- CP holds broad promise for calibrating outputs of other foundation models, including LLMs

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of applying CP to calibrate foundation segmentation models is novel, though the two-stage training itself is relatively straightforward
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets (VOC/VOC-aug/ADE20K), plug-in validation, and comprehensive CP variant ablations
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and ablation design is well-conceived, though the method section is equation-heavy
- Value: ⭐⭐⭐⭐ Demonstrates a general paradigm for safely leveraging foundation models to assist downstream training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](../../AAAI2026/segmentation/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[ICCV 2025\] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss](prompt_guidance_and_human_proximal_perception_for_hot_prediction_with_regional_j.md)
- [\[ICCV 2025\] Auto-Vocabulary Semantic Segmentation](auto-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
