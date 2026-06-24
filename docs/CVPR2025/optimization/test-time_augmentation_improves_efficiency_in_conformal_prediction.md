---
title: >-
  [Paper Note] Test-Time Augmentation Improves Efficiency in Conformal Prediction
description: >-
  [CVPR 2025][Optimization][Conformal Prediction] It is discovered that test-time data augmentation (TTA) can systematically improve the efficiency of conformal prediction. By learning augmentation weights on a calibration set to optimize the augmentation aggregation strategy, the prediction set size is reduced by 10-17% on ImageNet with ResNet-50 while strictly preserving the coverage guarantee.
tags:
  - "CVPR 2025"
  - "Optimization"
  - "Conformal Prediction"
  - "Test-Time Augmentation"
  - "Prediction Set Efficiency"
  - "Learnable Aggregation"
  - "Coverage Guarantee"
date: 2026-05-08
content_hash: 9e17cf49a5efab04
---

# Test-Time Augmentation Improves Efficiency in Conformal Prediction

**Conference**: CVPR 2025  
**arXiv**: [2505.22764](https://arxiv.org/abs/2505.22764)  
**Code**: To be released  
**Area**: Other / Uncertainty Estimation  
**Keywords**: Conformal Prediction, Test-Time Augmentation, Prediction Set Efficiency, Learnable Aggregation, Coverage Guarantee

## TL;DR

It is discovered that test-time data augmentation (TTA) can systematically improve the efficiency of conformal prediction. By learning augmentation weights on a calibration set to optimize the augmentation aggregation strategy, the prediction set size is reduced by 10-17% on ImageNet with ResNet-50 while strictly preserving the coverage guarantee.

## Background & Motivation

### Background

**Background**: Conformal Prediction attaches statistical guarantees to classifiers—outputting a prediction set containing the true label, which guarantees coverage at a user-specified confidence level $1-\alpha$. However, larger prediction sets are less useful (at worst, containing all classes), making "efficiency" (prediction set size) a key metric.

### Limitations of Prior Work

**Limitations of Prior Work**: The efficiency of conformal prediction depends entirely on the output quality of the underlying classifier—if the classifier assigns a high probability to the true class, the prediction set is small; otherwise, it is large. Existing methods improve efficiency by refining conformal score functions (e.g., RAPS with regularization) but do not consider improving the model's probabilistic outputs.

### Key Challenge

**Key Challenge**: A single forward pass corresponds to only one perspective of the data, leading to unstable probability estimation from the classifier. Some samples may receive lower-than-actual probability simply due to sub-optimal perspectives, causing them to be included in large prediction sets.

### Key Insight

**Key Insight**: TTA provides more stable probability estimates of the classifier using multiple augmentations followed by aggregation, without altering model parameters. The key issue lies in how to leverage TTA while preserving exchangeability (the theoretical premise of conformal prediction).

### Core Idea

**Core Idea**: Learning augmentation aggregation weights on an independent calibration set + calculating conformal thresholds using a separate set of calibration data = TTA improves efficiency while ensuring coverage.

## Method

### Key Designs

1. **Learnable Augmentation Aggregation Weights**:

    - **Function**: Learn the optimal contribution of each augmentation to prediction quality.
    - **Mechanism**: Optimize weight $\theta$ on calibration set $D^{(TTA)}$ using cross-entropy: $\hat\pi_x(y) = \sigma(\theta^T A(f, \mathcal{A}, x))$, where $A$ is the classifier output matrix under various augmentations. Certain augmentations (e.g., blurring/sharpening reduction) consistently receive zero weights, indicating they do not benefit classification.
    - **Design Motivation**: TTA-Avg (equal-weight averaging) yields limited improvement (35.60 vs 37.75), whereas learning weights achieves substantial improvement (31.25).

2. **Data Partitioning for Exchangeability**:

    - **Function**: Theoretically guarantee that TTA does not violate the coverage guarantee of conformal prediction.
    - **Mechanism**: The calibration data is split into two parts: $D^{(TTA)}$ is used to learn aggregation weights, and $D^{(cal)}$ is used to compute conformal thresholds. Since the threshold is computed independently after weight learning, exchangeability is preserved.
    - **Design Motivation**: Simultaneously learning weights and computing thresholds on the same dataset violates the exchangeability assumption, causing the coverage guarantee to fail.

### Loss & Training

TTA weight learning: Cross-entropy loss is minimized on $D^{(TTA)}$ without modifying model parameters. Twelve augmentation strategies (rotation, translation, color jittering, etc.) are used, followed by a weighted aggregation after multiple forward passes.

## Key Experimental Results

### Main Results

ImageNet ResNet-50 prediction set size ↓ ($\alpha$=0.01):

| Method | Prediction Set Size | Change |
|------|-----------|------|
| RAPS Baseline | 37.75 | — |
| RAPS + TTA-Avg | 35.60 | -5.7% |
| **RAPS + TTA-Learned** | **31.25** | **-17.2%** |

### Ablation Study

| Configuration | Prediction Set Size |
|------|-----------|
| Simple Augmentation (2 types) | 32.70 |
| Extended Augmentation (12 types) | **31.25** |
| TTA-Avg | 35.60 |
| **TTA-Learned** | **31.25** |

### Key Findings
- **Learning weights is significantly superior to equal-weight averaging**: 31.25 vs 35.60. The contributions of augmentations are uneven—augmentations like blurring or excessive sharpening reduction prove useless.
- **Remains effective under distribution shift**: Improves performance on ImageNet-C as well, demonstrating high robustness.
- **TTA improves the rank of the true class**: For misclassified images, the rank of the true class improves from ~200 to ~100, which lowers the conformal score.

## Highlights & Insights
- **First systematic integration of TTA and Conformal Prediction**—Transforming TTA from an empirical trick for "accuracy improvement" to a rigorous method that "reduces prediction set size while theoretically maintaining coverage guarantees."
- **Explainability of augmentation weights**—Which augmentations are beneficial or redundant can be directly observed, providing theoretical guidance for selecting data augmentation strategies.

## Limitations & Future Work
- Computational overhead of multiple forward passes (12 augmentations = 12x inference).
- Validation is limited to image classification.
- Requires a sufficiently large calibration set.
- Exchangeability assumptions restrict certain data partitioning strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of TTA and conformal prediction is novel and theoretically rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on ImageNet with multiple scales and distribution shifts.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and elegant theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a practical tool for the uncertainty estimation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](conformal_prediction_for_zero-shot_models.md)
- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/optimization/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)

</div>

<!-- RELATED:END -->
