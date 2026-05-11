---
title: >-
  [Paper Note] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability
description: >-
  [ICCV 2025][Object Detection][Automated Model Evaluation] This paper proposes PCR (Prediction Consistency and Reliability), an automated evaluation method that estimates object detection model performance without human a…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Automated Model Evaluation"
  - "NMS"
  - "Prediction Consistency"
  - "Reliability"
date: 2026-05-08
content_hash: 83c90e226b7b9692
---

# Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability

**Conference**: ICCV 2025
**arXiv**: [2508.12082](https://arxiv.org/abs/2508.12082)
**Code**: [GitHub](https://github.com/YonseiML/autoeval-det)
**Area**: Object Detection / Model Evaluation
**Keywords**: Automated Model Evaluation, object detection, NMS, Prediction Consistency, Reliability

## TL;DR
This paper proposes PCR (Prediction Consistency and Reliability), an automated evaluation method that estimates object detection model performance without human annotations. PCR analyzes the spatial consistency and confidence reliability of bounding boxes before and after NMS to estimate mAP, and constructs a corruption-based meta-dataset for more realistic and scalable evaluation.

## Background & Motivation
Machine learning models require performance evaluation prior to deployment, particularly when the target domain exhibits distribution shift relative to the training domain. Annotating test data is typically costly, making Automated Model Evaluation (AutoEval)—estimating model performance on unlabeled test data—highly valuable.

AutoEval has been extensively studied in image classification (by measuring feature distribution distances to estimate covariate shift), but direct extension to object detection poses fundamental challenges:

**Classification vs. Detection**: Detection performance is affected by complex factors such as scale variation, occlusion, background clutter, and object interactions, which cannot be captured by covariate shift alone.

**Limitations of Prior Work**: The only existing AutoEval method designed for detection, BoS (Box Stability), relies on Monte Carlo dropout, which is stochastic, requires additional forward passes, and does not exploit confidence information.

**Meta-dataset Issues**: Prior work constructs meta-datasets using strong data augmentations, yielding unrealistic images and a narrow mAP distribution.

This paper adopts a bottom-up strategy, observing that conventional detectors generate a large number of candidate boxes before NMS—boxes typically discarded yet containing valuable localization and classification information.

## Method

### Overall Architecture
PCR consists of two complementary scores: a Consistency Score (measuring localization failure patterns in low-confidence predictions) and a Reliability Score (measuring detection trustworthiness in high-confidence predictions). The two scores are combined via linear regression to estimate mAP.

### Key Designs

1. **Consistency Score**:

    - Function: Measures the spatial consistency between each final post-NMS prediction and the merged bounding box of its corresponding pre-NMS candidates, focusing on low-confidence predictions.
    - Mechanism:
        - For each final prediction box, all associated pre-NMS candidate boxes are merged into a single enclosing box $B_{\text{merge}}^{(i)}$.
        - Consistency is measured by the average of IoU and Center Closeness (CC): $S^{C(i)} = \frac{\text{IoU}(B_{\text{final}}^{(i)}, B_{\text{merge}}^{(i)}) + \text{CC}(B_{\text{final}}^{(i)}, B_{\text{merge}}^{(i)})}{2}$
        - CC is defined as: $\text{CC} = 1 - \frac{\sqrt{(x_f - x_m)^2 + (y_f - y_m)^2}}{\sqrt{w_f^2 + h_f^2}/2}$, providing a scale-invariant center distance measure.
        - Image-level consistency is obtained via sigmoid-weighted averaging, with a negative-scale sigmoid $\sigma_C$ emphasizing low-confidence predictions.
    - Design Motivation: A low-confidence final prediction that is highly consistent with the merged box suggests the detector repeatedly localizes the same region without a true object present—a signal of detection failure. This score is strongly negatively correlated with mAP.

2. **Reliability Score**:

    - Function: Measures what fraction of pre-NMS candidates associated with high-confidence predictions also exhibit high confidence.
    - Mechanism: $S^R = \frac{\sum_{i=1}^{N} \sum_{j=1}^{K_i} \mathbb{I}[h(B_{\text{final}}^{(i)}) > c] \cdot \sigma_R(h(B^{(ij)}))}{\sum_{B^{(ij)} \in \mathcal{P}} \sigma_R(h(B^{(ij)}))}$, where $\sigma_R$ assigns weights close to 1 for high-confidence candidates and a floor value of $\alpha$ for low-confidence ones.
    - Design Motivation: If candidates surrounding a high-confidence prediction also carry high confidence, the model demonstrates strong consensus in both classification and localization—a signal of successful detection. This score is strongly positively correlated with mAP.

3. **Corruption-based Meta-dataset**:

    - Function: Constructs a more realistic evaluation meta-dataset using corruption transforms from ImageNet-C.
    - Mechanism: Employs 10 corruption types × 5 severity levels = 50 datasets. Corruptions that alter bounding box coordinates (e.g., zoom blur) are excluded.
    - Design Motivation: Augmentation-based meta-datasets produce unrealistic images and narrow mAP distributions (concentrated around 25–35%). The corruption-based meta-dataset is more realistic and covers a mAP range from near 0% to 40%.

### AutoEval Pipeline
After collecting the consistency and reliability scores, a least-squares linear regression is applied: $\widehat{\text{mAP}} = w_0 + w_1 \cdot \bar{S}^C + w_2 \cdot \bar{S}^R$. Leave-one-out cross-validation is used for training and evaluation.

### Hyperparameter Settings
Confidence threshold $c=0.5$, consistency sigmoid scale $k_C=-60$, reliability sigmoid scale $k_R=10$, floor value $\alpha=0.2$.

## Key Experimental Results

### Main Results (Vehicle Detection, Average RMSE over Four Detectors × Two Meta-datasets)

| Method | Avg. RMSE↓ | Avg. Rank↓ | Notes |
|--------|-----------|-----------|-------|
| PS | 7.74 | 3.13 | Prediction Score |
| ES | 8.62 | 4.75 | Entropy Score |
| AC | 9.27 | 5.13 | Average Confidence |
| ATC | 9.17 | 4.38 | Average Thresholded Confidence |
| BoS | 6.94 | 2.50 | Box Stability (MC dropout) |
| **PCR** | **4.61** | **1.13** | Ours |

### Ablation Study

| Configuration | Avg. RMSE↓ | Notes |
|--------------|-----------|-------|
| $S^C$ only | 6.75 | Already outperforms baselines |
| $S^R$ only | 6.64 | Also outperforms baselines |
| PCR ($S^C + S^R$) | 6.57 | Best combined performance |
| Consistency w/ IoU only | 6.75 | No CC metric |
| Consistency w/ IoU + CC | 6.64 | CC complements IoU |
| No confidence weighting ($S_{all}^C$) | 8.24 | No distinction by confidence level |
| With confidence weighting ($S^C$) | 6.64 | Focuses on low-confidence predictions |

### Key Findings
- PCR achieves the best performance on **pedestrian detection** as well (Avg. RMSE 3.57 vs. 6.22 for BoS), ranking **first across all settings**.
- PCR and BoS are complementary; their combination reduces RMSE from 5.69 to 5.15, as they capture different dimensions of consistency.
- PCR can simultaneously estimate mAP50 and mAP75 (RMSE of 10.18 and 7.94, respectively, both best in class).
- The corruption-based meta-dataset covers a broader mAP range than the augmentation-based one, and is especially effective in the low-performance regime.
- As corruption severity increases, PCR's RMSE grows more steadily than BoS, indicating superior robustness.

## Highlights & Insights
- The paper identifies a meaningful link between confidence and localization quality: low-confidence predictions exhibit low IoU, while high-confidence predictions exhibit high IoU. This observation is translated into a label-free evaluation signal.
- The method cleverly exploits "discarded" pre-NMS candidates—a natural byproduct of a single forward pass requiring no additional computation.
- PCR is deterministic (vs. BoS's stochasticity), efficient (no additional forward passes required), and leverages confidence information.
- The CC metric addresses the failure mode of IoU when merged boxes are elongated, reflecting careful design consideration.

## Limitations & Future Work
- The method is specific to NMS-based detectors; applicability to end-to-end detectors (e.g., DETR-family) remains unexplored.
- The linear regression assumes a linear relationship between consistency/reliability and mAP, which may not hold in complex scenarios.
- Only 250 images are sampled per dataset, representing a relatively small sample size.
- Hyperparameters (particularly the confidence threshold $c$ and sigmoid scale parameters) may require tuning for different settings.
- Validation is limited to vehicle and pedestrian detection; generalization to multi-class detection remains to be confirmed.

## Related Work & Insights
- **vs. BoS**: BoS compares predictions before and after MC dropout perturbations, introducing stochasticity and requiring additional forward passes. PCR leverages deterministic pre- and post-NMS information, offering greater efficiency and stability.
- **vs. Classification AutoEval Methods (PS/ES/AC/ATC)**: These methods rely solely on confidence statistics, ignoring the spatial relationships intrinsic to detection. PCR jointly considers both localization and classification dimensions.
- **Insight**: Although NMS discards redundant boxes, the "internal consensus" information carried by those boxes can reflect the model's actual performance.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of mining evaluation signals from pre-NMS candidates is novel and intuitively well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four detectors, two meta-datasets, two detection scenarios, comprehensive ablations and combination analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logically rigorous; the derivation chain from observation to method is complete, with clear and intuitive figures.
- Value: ⭐⭐⭐⭐ — Fills a gap in AutoEval for object detection; the method is practical and open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[NeurIPS 2025\] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](../../NeurIPS2025/object_detection/automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)
- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [\[ICCV 2025\] Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding](intervening_in_black_box_concept_bottleneck_model_for_enhancing_human_neural_net.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)

</div>

<!-- RELATED:END -->
