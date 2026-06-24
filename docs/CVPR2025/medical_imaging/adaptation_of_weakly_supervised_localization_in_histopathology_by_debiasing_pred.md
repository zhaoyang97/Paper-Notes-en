---
title: >-
  [Paper Note] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions
description: >-
  [CVPR2025][Medical Imaging][Source-Free Domain Adaptation] Proposes the SFDA-DeP method. Inspired by machine unlearning, it identifies and corrects the prediction bias (over-predicting certain classes) of the source model in the target domain. This addresses the challenge of amplified prediction bias in weakly supervised localization models during cross-organ/cross-center domain adaptation in histopathology.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Source-Free Domain Adaptation"
  - "Weakly Supervised Localization"
  - "Histopathology"
  - "machine unlearning"
  - "Prediction Bias"
date: 2026-05-08
content_hash: b80f11267cd6bb45
---

# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**Conference**: CVPR2025  
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)  
**Code**: [anonymous.4open.science/r/SFDA-DeP-1797](https://anonymous.4open.science/r/SFDA-DeP-1797/)  
**Area**: Medical Image  
**Keywords**: Source-Free Domain Adaptation, Weakly Supervised Localization, Histopathology, machine unlearning, Prediction Bias

## TL;DR

Proposes the SFDA-DeP method. Inspired by machine unlearning, it identifies and corrects the prediction bias (over-predicting certain classes) of the source model in the target domain. This addresses the challenge of amplified prediction bias in weakly supervised localization models during cross-organ/cross-center domain adaptation in histopathology.

## Background & Motivation

**Clinical significance of WSOL**: Weakly Supervised Object Localization (WSOL) enables concurrent classification and ROI localization using only image-level labels, heavily reducing the annotation burden in pathology.

**Domain shift as a core challenge**: Differences in staining protocols, scanner-specific characteristics, and tissue preparation workflows across institutions lead to severe performance degradation when models are deployed cross-center.

**Amplification effect of prediction bias**: Under severe domain shift, the source model over-predicts certain classes, resulting in highly skewed pseudo-label distributions. Traditional SFDA methods (such as SFDA-DE) based on self-training tend to run into a loop that reinforces this bias instead.

**Inaccessibility of source data**: Source-Free Domain Adaptation (SFDA) better complies with clinical privacy regulations, but the absence of source data makes bias correction significantly more challenging.

**Specificity of the localization task**: Classification bias propagates down to spatial Class Activation Maps (CAMs), inducing inconsistent localization results.

**Severe cross-organ shift**: When migrating from GlaS (colon) to CAMELYON16/17 (breast), predictions almost entirely shift toward the cancer class.

## Method

### Overall Architecture

SFDA-DeP models Source-Free Domain Adaptation as an iterative process of bias identification and correction, consisting of three core components: forget/retain set splitting, forget loss, and localization supervision.

### Key Designs

- **Bias detection**: Calculates the prediction frequency of each class on the target domain to identify the dominant class $\mathcal{B}$ that is being over-predicted.
- **Forget/Retain set splitting**: From the samples predicted as the dominant class, the top-$\rho$ samples with the highest normalized entropy are selected as the forget set $\mathbb{B}_f$ (representing uncertain samples near the decision boundary), and the rest constitute the retain set.
- **Retain loss**: Standard cross-entropy to preserve pseudo-label predictions for reliable samples: $\mathcal{L}_{\text{retain}} = \mathbb{E}_{x_i \in \mathbb{B}_r}[-\log(p_i(\hat{y}))]$
- **Forget loss**: Reverse cross-entropy, forcing the model to "forget" dominant class predictions for uncertain samples: $\mathcal{L}_{\text{forget}} = \mathbb{E}_{x_i \in \mathbb{B}_f}[-\log(1 - p_i(\hat{y}))]$
- **Localization supervision**: A lightweight pixel-level classification head $h$ that performs pixel-level binary classification using foreground/background pseudo-labels extracted from CAMs: $\mathcal{L}_{\text{loc}} = -(1-Y_p)\log(h(z_p)_0) - Y_p\log(h(z_p)_1)$
- **Periodic update**: Reconstructs the forget/retain sets every $m$ epochs to prevent overfitting to pseudo-labels.

### Loss & Training

$$\mathcal{L} = \lambda_{\text{retain}}\mathcal{L}_{\text{retain}} + \lambda_{\text{forget}}\mathcal{L}_{\text{forget}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}}$$

## Key Experimental Results

### Datasets

- GlaS (colon gland segmentation), CAMELYON16 (breast lymph node), CAMELYON17 (5 centers: C17-0 to C17-4)

### PixelCAM: GlaS → Cross-Domain Average Performance

| Method | PxAP | CL (Classification Accuracy) |
|------|------|---------------|
| Source only | 36.9 | 49.3 |
| SFDA-DE | 28.0 | 54.6 |
| ERL | 25.4 | 59.9 |
| RGV | 34.7 | 52.1 |
| **SFDA-DeP (Ours)** | **44.1** | **67.1** |

### SAT: GlaS → Cross-Domain Average Performance

| Method | PxAP | CL |
|------|------|-----|
| Source only | 21.3 | 52.1 |
| SFDA-DE | 21.6 | 68.7 |
| **SFDA-DeP (Ours)** | **30.3** | **69.2** |

### DeepMIL: GlaS → Cross-Domain Average Performance

| Method | PxAP | CL |
|------|------|-----|
| Source only | 20.9 | 49.8 |
| SFDA-DE | 20.5 | 53.9 |
| CDCL | 27.3 | 55.5 |
| **SFDA-DeP (Ours)** | **40.7** | **73.4** |

### Key Findings

- SFDA-DeP consistently outperforms state-of-the-art SFDA baselines across all WSOL backbones (PixelCAM, SAT, DeepMIL).
- In comparison to SFDA-DE, it achieves a gain of +16.1 PxAP / +12.5 CL on PixelCAM, and +20.2 PxAP / +19.5 CL on DeepMIL.
- Traditional SFDA methods (such as SFDA-DE) tend to amplify bias under severe domain shifts. Their classification performance is sometimes inferior to source-only models (e.g., PxAP drops from 37.2 to 14.5 on PixelCAM for C17-0).
- Dynamic resampling of the forget/retain sets is a critical component, as static splitting leads to a significant performance drop.
- The pixel-level localization loss contributes significantly to the improvement in PxAP, while bringing complementary gains to classification accuracy.
- Simultaneous and substantial improvements are achieved in both localization and classification tasks.

## Highlights & Insights

1. **Valuable problem formulation**: For the first time, this work systematically reveals the mechanism of SFDA failure in WSOL scenarios caused by amplified prediction bias.
2. **Clever adaptation of machine unlearning**: The domain adaptation problem is modeled analogously to "forgetting old decision boundaries and establishing new ones."
3. **No source data required**: Completely source-free, adhering to clinical data privacy regulations.
4. **High versatility**: Effective across both CNN (ResNet-50) and Transformer (DeiT-Tiny) backbones.

## Limitations & Future Work

1. The evaluation is limited to binary classification (cancer vs. normal) and has not been extended to multi-class fine-grained classification (e.g., cancer subtypes).
2. The forget ratio $\rho$ and loss weights must be tuned on a validation set, and the hyperparameter sensitivity analysis is insufficient.
3. Considerable performance gaps exist across CAMELYON17 centers (e.g., classification accuracy on C17-1 drops to 41.3%), indicating room for improvement in cross-center robustness.
4. Pixel-level localization supervision relies on the quality of CAMs. The efficacy is bounded if the source model's CAM contains severe intrinsic bias.
5. Comparison with prompt-based foundation model adaptation methods (e.g., SAM) is lacking.
6. Samples in the forget set are simply pushed away from the dominant class, which might inadvertently push them towards incorrect minority classes rather than the true labels.

## Related Work & Insights

- **WSOL methods**: DeepMIL, SAT, PixelCAM, NEGEV, etc., which obtain spatial localization from image-level labels via CAM mechanisms.
- **SFDA methods**: SFDA-DE, CDCL, ERL, RGV, etc., which are based on pseudo-labeling/clustering self-training, but yield limited effectiveness under biased predictions.
- **Machine unlearning**: Traditionally employed for privacy deletion, this work innovatively adopts it to correct prediction bias instead of deleting classes.

## Rating

- Novelty: ⭐⭐⭐⭐ (The synergistic combination of machine unlearning, SFDA, and WSOL presents a highly novel entry point.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluated across 3 WSOL backbones, multiple target domains, and multiple SFDA baselines.)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem analysis; Fig. 1 intuitively demonstrates the bias amplification phenomenon.)
- Value: ⭐⭐⭐⭐ (Addresses a practical bottleneck in cross-center deployment in pathology.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](../../NeurIPS2025/medical_imaging/match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)
- [\[CVPR 2025\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)

</div>

<!-- RELATED:END -->
