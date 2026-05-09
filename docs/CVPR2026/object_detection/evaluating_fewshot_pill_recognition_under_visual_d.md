---
title: >-
  [Paper Note] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift
description: >-
  [CVPR 2026][Object Detection][Few-shot learning] This paper systematically evaluates few-shot pill recognition under cross-dataset domain shift from a deployment-oriented perspective. It reveals a decoupling phenomenon in which semantic classification saturates at 1-shot while localization and recall degrade sharply under occlusion and overlap, and demonstrates that the visual realism of training data is the dominant factor governing few-shot generalization.
tags:
  - CVPR 2026
  - Object Detection
  - Few-shot learning
  - pill recognition
  - domain shift
  - deployment robustness
date: 2026-05-08
content_hash: 9571745d8018deeb
---

# Evaluating Few-Shot Pill Recognition Under Visual Domain Shift

**Conference**: CVPR 2026
**arXiv**: [2603.10833](https://arxiv.org/abs/2603.10833)
**Code**: N/A
**Area**: Object Detection
**Keywords**: Few-shot learning, pill recognition, domain shift, object detection, deployment robustness

## TL;DR

This paper systematically evaluates few-shot pill recognition under cross-dataset domain shift from a deployment-oriented perspective. It reveals a decoupling phenomenon in which semantic classification saturates at 1-shot while localization and recall degrade sharply under occlusion and overlap, and demonstrates that the visual realism of training data is the dominant factor governing few-shot generalization.

## Background & Motivation

Adverse drug events (ADEs) are a significant source of preventable harm, motivating the development of automated pill recognition systems. Real-world deployment, however, faces substantial challenges:

1. **Visual complexity**: Pills in practice are often stored in pill organizers, introducing cluttered scenes, overlapping occlusions, and specular reflections.
2. **Annotation scarcity**: Constructing large-scale annotated datasets in medical settings is costly.
3. **Evaluation distortion**: Most existing few-shot pill recognition studies are conducted under controlled conditions where training and test distributions are close, thereby overestimating real-world deployment robustness.
4. **Absence of cross-dataset evaluation**: Systematic cross-dataset evaluation under domain shift is exceedingly rare in few-shot object detection.

The central objective of this work is not to propose a new architecture, but to systematically examine the generalization behavior and failure modes of few-shot pill recognition from a **deployment diagnostic** perspective.

## Method

### Overall Architecture

A two-stage few-shot object detection framework based on Faster R-CNN (FsDet) is adopted:

1. **Base training stage**: A detector is trained on base categories to learn general visual representations, region proposal mechanisms, and feature embeddings.
2. **Few-shot fine-tuning stage**: The detector is fine-tuned using a small number of annotated samples from novel categories.

### Key Designs

1. **Cross-domain evaluation protocol**:
    - Base training is performed on either the CURE or MEDISEG dataset.
    - Few-shot fine-tuning and evaluation are conducted on entirely separate, unseen deployment datasets.
    - Strict data leakage prevention is enforced across all three stages.
    - A 5-way K-shot setting is used ($K \in \{1, 5, 10\}$).

2. **Contrastive design of two base training datasets**:
    - **CURE**: 8,973 images, 196 classes, single pill per image, captured under controlled conditions without occlusion — visually simple.
    - **MEDISEG**: 8,262 images, 32 classes, multiple pill instances per image, pill organizer scenes with occlusion and clutter — visually realistic.
    - These two datasets are deliberately selected for their stark contrast to investigate the effect of base-domain realism on few-shot adaptation.

3. **Classification-centric, error-driven evaluation metrics**:
    - Average precision (AP) is not used as the primary metric, as heterogeneous annotation granularity renders AP incomparable across datasets.
    - Core metrics: foreground classification accuracy (FG-Acc), false negative rate (FN rate), classification loss, RPN loss, and total loss.
    - These metrics isolate semantic recognition from localization artifacts and remain fairly comparable under annotation heterogeneity.

### Loss & Training

- **Few-shot fine-tuning**: SGD with momentum 0.9, weight decay $10^{-4}$, fixed learning rate $10^{-3}$.
- All shot settings are trained for a fixed 2,000 iterations with no early stopping.
- The backbone (ResNet + FPN) is frozen; the RPN is partially trainable; ROI heads are fully fine-tuned.
- The classification layer is re-initialized for novel categories.
- Base training data is not revisited, and no additional data augmentation is applied.

## Key Experimental Results

### Main Results: Cross-Domain Few-Shot Adaptation

| Metric | CURE 1-shot | CURE 5-shot | CURE 10-shot | MEDISEG 1-shot | MEDISEG 5-shot | MEDISEG 10-shot |
|--------|------------|------------|-------------|---------------|---------------|----------------|
| FG Acc | 0.989±.004 | 0.980±.004 | 0.977±.004 | 0.994±.005 | 0.991±.002 | 0.983±.003 |
| FN rate | 0.011±.004 | 0.020±.004 | 0.023±.004 | 0.006±.005 | 0.009±.002 | 0.017±.003 |
| loss_cls | 0.005±.001 | 0.014±.001 | 0.019±.002 | 0.005±.001 | 0.011±.001 | 0.015±.002 |
| total_loss | 0.015±.003 | 0.039±.003 | 0.055±.005 | 0.014±.002 | 0.032±.003 | 0.044±.003 |

Key observation: foreground classification accuracy reaches ≥0.989 at 1-shot; models trained on MEDISEG exhibit a false negative rate 45% lower than those trained on CURE.

### Ablation Study: Overlap and Occlusion Stress Test

| Metric | CURE 1-shot | CURE 5-shot | CURE 10-shot | MEDISEG 1-shot | MEDISEG 5-shot | MEDISEG 10-shot |
|--------|------------|------------|-------------|---------------|---------------|----------------|
| FG Acc | 0.131 | 0.372 | 0.558 | 0.406 | 0.625 | 0.740 |
| FN rate | 0.816 | 0.465 | 0.342 | 0.513 | 0.246 | 0.210 |
| loss_cls | 0.351 | 0.421 | 0.320 | 0.383 | 0.279 | 0.191 |
| loss_rpn_cls | 0.863 | 0.224 | 0.133 | 0.312 | 0.182 | 0.059 |
| total_loss | 1.326 | 0.844 | 0.674 | 0.963 | 0.680 | 0.445 |

### Key Findings

1. **Semantic recognition saturates rapidly**: Foreground classification accuracy reaches 0.989+ at 1-shot; marginal returns from additional annotations diminish quickly.
2. **Decoupling of localization and classification**: Semantic classification is strong under standard evaluation, yet localization and recall degrade sharply under the occlusion stress test (CURE 1-shot FG Acc drops from 0.989 to 0.131).
3. **Training data realism is the dominant factor**: Models trained on MEDISEG (multi-pill, realistic scenes) outperform those trained on CURE (single-pill, controlled scenes) by **210%** in FG Acc on the 1-shot overlap test.
4. **Diminishing returns from increased supervision**: The gain from 1-shot to 5-shot is substantial (MEDISEG FG Acc: 0.406→0.625), while the gain from 5-shot to 10-shot is considerably smaller (+18%).
5. **Loss increase is not a degradation signal**: Total loss grows with the number of shots but does not indicate recognition degradation — rather, it reflects a more complex optimization landscape.
6. **MEDISEG advantage is greatest at low shot counts**: The relative advantage is 210% at 1-shot and narrows to 33% at 10-shot, indicating that realistic training data is especially critical under extreme data scarcity.

## Highlights & Insights

- **Novel deployment diagnostic perspective**: Few-shot learning is reframed as a **diagnostic tool** for deployment readiness rather than a pure adaptation strategy — varying supervision levels exposes stability–robustness trade-offs.
- **Discovery of classification–localization decoupling as a systematic failure mode**: High semantic recognition accuracy can mask severe localization failures, an insight entirely hidden under conventional AP evaluation.
- **Data realism > data scale**: CURE contains 196 classes versus MEDISEG's 32, yet the latter generalizes better in cross-domain few-shot settings due to higher visual complexity.
- **Thoughtful evaluation design**: The decision to abandon AP in favor of classification-error signals due to annotation heterogeneity is well-justified.

## Limitations & Future Work

1. **No new method is proposed**: This is a purely analytical study with no methodological contribution.
2. **Limited number of novel classes**: Only a 5-way setting is employed, constrained by annotation availability.
3. **Whole-image bounding box annotations in CURE limit localization metrics**: AP is not comparable across datasets, necessitating reliance on classification-centric evaluation.
4. **Stronger detectors are not explored**: Only Faster R-CNN is used; the behavior of DETR-based or single-stage detectors remains unknown.
5. **Data augmentation effects are not studied**: Synthetic occlusion augmentation in few-shot occlusion scenarios may offer a low-cost path to improved robustness.
6. **No cross-dataset baseline comparison**: The evaluation protocol deviates from standard few-shot benchmarks, precluding direct comparison with other methods.

## Related Work & Insights

- **FsDet (Faster R-CNN)** is a classical two-stage few-shot detection framework; this paper conducts cross-domain analysis upon it.
- **CURE** and **MEDISEG** represent the two extremes of controlled versus realistic settings — motivating reflection on how training data design affects deployment robustness.
- The cross-dataset evaluation discussions in the **domain generalization literature** align with the philosophy of this work.
- **Implication**: In safety-critical applications, one should not merely pursue benchmark performance but should systematically investigate failure modes and data–performance interactions.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)
- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](ewdetr_evolving_world_object_detection.md)
- [\[CVPR 2026\] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)

</div>

<!-- RELATED:END -->
