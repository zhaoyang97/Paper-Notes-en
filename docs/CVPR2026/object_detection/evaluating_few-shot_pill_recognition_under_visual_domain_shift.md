---
title: >-
  [Paper Note] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift
description: >-
  [CVPR 2026][Object Detection][few-shot object detection] This paper systematically evaluates the generalization of pill recognition under cross-domain few-shot conditions from a deployment perspective. It reveals a decoupling phenomenon in which semantic classification saturates at 1-shot while localization and recall degrade sharply under occlusion and overlap, and demonstrates that the visual realism of training data is far more critical than data volume or shot count.
tags:
  - CVPR 2026
  - Object Detection
  - few-shot object detection
  - pill recognition
  - domain shift
  - deployment readiness
  - cross-dataset evaluation
date: 2026-05-08
content_hash: 78e75cf997fb9077
---

# Evaluating Few-Shot Pill Recognition Under Visual Domain Shift

**Conference**: CVPR 2026
**arXiv**: [2603.10833](https://arxiv.org/abs/2603.10833)
**Code**: None (based on FsDet/Detectron2 open-source framework)
**Area**: Object Detection / Medical Imaging
**Keywords**: few-shot object detection, pill recognition, domain shift, deployment readiness, cross-dataset evaluation

## TL;DR
This paper systematically evaluates the generalization of pill recognition under cross-domain few-shot conditions from a deployment perspective. It reveals a decoupling phenomenon in which semantic classification saturates at 1-shot while localization and recall degrade sharply under occlusion and overlap, and demonstrates that the visual realism of training data is far more critical than data volume or shot count.

## Background & Motivation

**Background**: Adverse drug events (ADEs) are a significant source of preventable medical harm, and automated pill recognition systems have attracted considerable attention. Existing systems are typically trained and evaluated under controlled conditions (single pill, clean background, uniform lighting) and achieve impressive performance.

**Limitations of Prior Work**: Real-world deployment environments differ substantially from controlled settings—pills are stored in dosette boxes, with multiple pills overlapping, occluding one another, exhibiting specular reflections, and appearing against cluttered backgrounds. Existing few-shot pill recognition studies are evaluated almost exclusively on in-distribution data (training and testing drawn from similar visual conditions), and the high accuracy reported likely severely overestimates true robustness.

**Key Challenge**: Can few-shot learning remain effective under cross-domain scenarios? Existing evaluation protocols sidestep the most critical deployment challenges—namely, the systematic domain shift between training data (controlled single-pill images) and deployment environments (cluttered multi-pill scenes). Standard mAP metrics are also inadequate for fair comparison across heterogeneous annotation conditions.

**Goal**
- What is the true generalization capability of few-shot adaptation under cross-dataset domain shift?
- Which factor has a greater impact on few-shot performance: the visual realism of base training data, or data volume?
- Are semantic classification and localization performance consistent under few-shot conditions with occlusion?
- Can few-shot fine-tuning serve as a diagnostic tool for deployment readiness?

**Key Insight**: Rather than pursuing architectural innovations, this work designs a rigorous cross-domain evaluation protocol (CURE controlled single-pill vs. MEDISEG real-world multi-pill → novel deployment environment), and replaces traditional mAP with classification-centric metrics to enable fair evaluation.

**Core Idea**: Reframe few-shot fine-tuning as a deployment readiness diagnostic tool, using cross-domain and overlap stress tests to expose the systematic failure mode of classification–localization decoupling.

## Method

### Overall Architecture
A two-stage few-shot object detection framework implemented on top of FsDet (Frustratingly Simple Few-Shot Object Detection) / Faster R-CNN.

**Pipeline**: Base training (training on base classes from CURE or MEDISEG) → Few-shot fine-tuning (fine-tuning with 1/5/10-shot support sets from the novel deployment dataset) → Query set evaluation (516 multi-pill cluttered scene images) + Overlap-only stress test (133 severely overlapping scene images).

### Key Designs

1. **Dataset Design and Contrast**

   - Function: Two datasets with substantially different visual complexity are deliberately selected as sources for base training.
   - Mechanism: CURE (8,973 images / 196 classes / single-pill controlled environment / full-image bbox annotations) vs. MEDISEG (8,262 images / 32 classes / multi-pill real-world scenes / instance-level bbox annotations). The two datasets have non-overlapping class sets, and neither overlaps with the novel classes.
   - Design Motivation: By controlling the visual realism of the base domain, the variable of "training data realism" is isolated to assess its effect on few-shot generalization. CURE offers larger data volume and more categories but visually simple scenes; MEDISEG offers fewer images and categories but higher visual complexity—constituting a natural "quantity vs. quality" experiment.

2. **Few-Shot Adaptation Protocol**

   - Function: Perform 5-way K-shot adaptation on the novel deployment dataset.
   - Mechanism: $K \in \{1, 5, 10\}$; support sets are sampled from the deployment dataset, while the query set (516 images) and overlap-only set (133 images) are strictly separated. Fine-tuning is fixed at 2,000 iterations with SGD + momentum 0.9, lr $= 1\times10^{-3}$; the backbone is frozen and only ROI heads and partial RPN layers are fine-tuned.
   - Design Motivation: A fixed training budget eliminates confounds from training duration; freezing the backbone preserves base knowledge; strict data separation ensures that observed differences are attributable to base-domain characteristics rather than data leakage.

3. **Classification-Centric Evaluation Framework**

   - Function: Replace traditional mAP with foreground classification accuracy (FG-Acc), false negative rate (FN rate), RPN classification loss, and total loss as primary metrics.
   - Mechanism: $\text{FG-Acc} = \frac{\text{number of correctly classified foreground proposals}}{\text{total foreground proposals}}$, $\text{FN} = \frac{\text{number of missed GT objects}}{\text{total GT objects}}$.
   - Design Motivation: CURE (full-image bboxes) and MEDISEG (instance bboxes) differ in annotation granularity, leading to inconsistent IoU matching; AP is therefore not comparable across annotation strategies. Classification and error metrics isolate semantic recognition from localization failures, exposing failure modes that mAP obscures.

4. **Overlap-Only Stress Test**

   - Function: A subset of 133 images with severe pill overlap is filtered from the deployment dataset to serve as an independent test set.
   - Mechanism: Each image is manually verified to contain significant occlusion or boundary ambiguity; instance-level bbox and segmentation mask annotations are provided. The overlap-only set shares the label space with the standard evaluation set but presents a structurally more challenging scene configuration.
   - Design Motivation: Standard evaluation may conflate performance on easy and hard scenes; the overlap-only set isolates the most challenging visual conditions to directly expose model fragility under occlusion.

### Loss & Training
- Base training: Standard Faster R-CNN training with fixed hyperparameters across all experiments.
- Few-shot fine-tuning: SGD, momentum = 0.9, weight decay $= 1\times10^{-4}$, lr $= 1\times10^{-3}$, 2,000 iterations.
- Backbone (ResNet + FPN) is frozen; RPN is partially trainable (with restricted lr); ROI heads are fully fine-tuned.
- Classification heads are re-initialized for novel classes.
- No additional data augmentation (only standard Detectron2 transforms).

## Key Experimental Results

### Main Results: Few-Shot Adaptation on the Standard Evaluation Set

| Configuration | FG Classification Accuracy | False Negative Rate | Classification Loss | Total Loss |
|---|---|---|---|---|
| CURE 1-shot | 0.989 ± 0.001 | 0.011 | 0.008 | 0.015 |
| CURE 5-shot | 0.981 ± 0.002 | 0.009 | 0.023 | 0.036 |
| CURE 10-shot | 0.977 ± 0.003 | 0.009 | 0.034 | 0.055 |
| MEDISEG 1-shot | 0.994 ± 0.005 | 0.006 | 0.011 | 0.021 |
| MEDISEG 5-shot | 0.990 ± 0.002 | 0.005 | 0.010 | 0.019 |
| MEDISEG 10-shot | 0.983 ± 0.002 | 0.005 | 0.019 | 0.030 |

**Key Findings**: Semantic classification saturates at 1-shot (CURE: 0.989, MEDISEG: 0.994), with minor degradation as shot count increases. The false negative rate under MEDISEG base training is 45% lower than under CURE (0.006 vs. 0.011).

### Ablation Study: Overlap-Only Stress Test

| Configuration | FG Classification Accuracy | False Negative Rate | Classification Loss | RPN Loss | Total Loss |
|---|---|---|---|---|---|
| CURE 1-shot | 0.131 | 0.816 | 0.351 | 0.863 | 1.326 |
| CURE 5-shot | 0.372 | 0.465 | 0.421 | 0.224 | 0.844 |
| CURE 10-shot | 0.558 | 0.342 | 0.320 | 0.133 | 0.674 |
| MEDISEG 1-shot | 0.406 | 0.513 | 0.383 | 0.312 | 0.963 |
| MEDISEG 5-shot | 0.625 | 0.246 | 0.279 | 0.182 | 0.680 |
| MEDISEG 10-shot | 0.740 | 0.210 | 0.191 | 0.059 | 0.445 |

### Key Findings

- **Classification–Localization Decoupling**: FG-Acc approaches 1.0 in the standard evaluation, but collapses to 0.131 (−87%) for CURE 1-shot in the overlap setting, while MEDISEG also drops to 0.406. Semantic recognition remains reliable when localization succeeds, but overlap causes dramatic deterioration in localization and recall.
- **Training Data Realism > Data Volume**: Under the most challenging 1-shot overlap condition, MEDISEG (fewer categories and images, but visually realistic) achieves 3.1× the FG-Acc of CURE (more categories and images, but visually simple): 0.406 vs. 0.131. This advantage is consistent across all shot settings.
- **Diminishing Returns**: The improvement from 1→5-shot is substantial (MEDISEG overlap FG-Acc: 0.406→0.625, +54%), while the gain from 5→10-shot noticeably diminishes (+18%), supporting the practical recommendation that moderate supervision is sufficient.
- **Decreasing Variance**: The standard deviation of MEDISEG 1-shot FG-Acc is ±0.005, dropping to ±0.002 at 5-shot (−60%), indicating that additional supervision primarily improves stability rather than peak accuracy.

## Highlights & Insights

- **Few-Shot Fine-Tuning as a Diagnostic Tool**: This is the most insightful contribution of the paper. Rather than treating few-shot learning solely as a data-efficient adaptation strategy, the paper leverages varying shot levels to expose trade-offs between model stability, robustness, and domain sensitivity—providing direct guidance for deployment decisions.
- **Clear Revelation of Classification–Localization Decoupling**: By combining classification-centric metrics (rather than mAP alone) with the overlap stress test, the paper quantitatively disentangles the distinct failure modes of semantic recognition and spatial localization. This finding transfers broadly to object detection evaluation in dense and occluded scenes.
- **Evaluation Protocol Design**: The pragmatic decision to abandon AP in favor of classification metrics when facing annotation heterogeneity is a methodological contribution worth adopting in cross-dataset evaluation settings.

## Limitations & Future Work

- **Acknowledged Limitations**: Full-image bboxes in CURE restrict the use of localization metrics; the non-standard few-shot benchmark precludes direct comparison with other methods; the number of novel classes is limited by annotation cost.
- **Unexplored Architectural Dimensions**: Only FsDet/Faster R-CNN is evaluated; stronger few-shot detectors (e.g., DeFRCN, FSCE) and alternative backbones are not explored. It remains unclear whether the observed classification–localization decoupling is architecture-agnostic.
- **Absence of Remedies**: The paper identifies the problem but proposes no improvements. Potential directions include: (1) occlusion-aware region proposal enhancement; (2) overlap-aware data augmentation during the few-shot fine-tuning phase; (3) mixed base+novel training strategies.
- **Localization Improvement Directions**: Instance segmentation masks (already annotated in the paper) could be incorporated into training rather than used solely for evaluation, to investigate whether this improves localization in overlapping scenes.

## Related Work & Insights

- **vs. Conventional Few-Shot Detection Evaluation**: Conventional methods (TFA, FsDet, FSCE, etc.) are evaluated on held-out splits of PASCAL VOC/COCO, where training and testing are drawn from the same distribution. The cross-dataset evaluation introduced here reveals genuine failure modes masked by in-distribution evaluation.
- **vs. EPillID / Original CURE Work**: These works demonstrate promising results under controlled conditions, but the present paper shows that such results are unreliable in deployment environments, particularly in overlapping scenes.
- **Insights**: The "few-shot as diagnosis" paradigm can be transferred to safety-critical domains such as autonomous driving and industrial inspection—using varying shot levels and out-of-domain data to probe model weaknesses has greater deployment value than pursuing state-of-the-art benchmark performance.

## Rating
- Novelty: ⭐⭐⭐ Not an architectural contribution, but the framing of "few-shot as a diagnostic tool" is novel and the evaluation protocol design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two base-domain comparisons, dual evaluation sets (standard + overlap), multiple shot settings, and both quantitative and qualitative analysis; the experimental design is rigorous.
- Writing Quality: ⭐⭐⭐⭐ The exposition is clear, the chain from experimental motivation to conclusions is complete, and the argument for classification–localization decoupling builds progressively.
- Value: ⭐⭐⭐⭐ Offers direct guidance for medical AI deployment; the findings that "data realism > data volume" and that classification–localization decoupling exists carry broad transferable reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)
- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)
- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](ewdetr_evolving_world_object_detection.md)

</div>

<!-- RELATED:END -->
