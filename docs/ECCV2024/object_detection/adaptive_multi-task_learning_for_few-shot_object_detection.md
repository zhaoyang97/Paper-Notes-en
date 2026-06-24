---
title: >-
  [Paper Note] Adaptive Multi-task Learning for Few-Shot Object Detection
description: >-
  [ECCV 2024][Object Detection][Few-Shot Object Detection] This paper proposes an adaptive multi-task learning method (MTL-FSOD) that dynamically adjusts the gradient scales of classification and localization tasks using a precision-driven gradient balancer to alleviate their conflict. It also introduces CLIP-based knowledge distillation and a classification refinement scheme to enhance individual task performance, achieving consistent improvements across multiple few-shot obje…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Few-Shot Object Detection"
  - "Multi-Task Learning"
  - "Gradient Balancing"
  - "Knowledge Distillation"
  - "CLIP"
date: 2026-05-08
content_hash: 4753bee62c4955df
---

# Adaptive Multi-task Learning for Few-Shot Object Detection

**Conference**: ECCV 2024  
**Code**: [https://github.com/RY-Paper/MTL-FSOD](https://github.com/RY-Paper/MTL-FSOD)  
**Area**: Object Detection  
**Keywords**: Few-Shot Object Detection, Multi-Task Learning, Gradient Balancing, Knowledge Distillation, CLIP

## TL;DR

This paper proposes an adaptive multi-task learning method (MTL-FSOD) that dynamically adjusts the gradient scales of classification and localization tasks using a precision-driven gradient balancer to alleviate their conflict. It also introduces CLIP-based knowledge distillation and a classification refinement scheme to enhance individual task performance, achieving consistent improvements across multiple few-shot object detection benchmarks.

## Background & Motivation

**Background**: Few-Shot Object Detection (FSOD) aims to detect novel category objects using extremely few annotated samples. Prevailing approaches are usually based on two-stage detection frameworks (such as Faster R-CNN) and adapt to novel classes via meta-learning or fine-tuning strategies. However, most methods utilize shared feature maps for the two sub-tasks of classification and localization.

**Limitations of Prior Work**: There is an inherent contradiction in the feature requirements of the classification and localization tasks: localization requires features sensitive to scale and position to accurately regress bounding box coordinates, whereas classification demands features robust to scale and position variations to achieve generalization in category discrimination. While this conflict has been extensively studied in standard object detection, it is more severe in few-shot scenarios—because samples are scarce, every dimension of the feature representation is more precious, leading to a fiercer "competition" for features between the two tasks. Although a few existing works have attempted to address this issue, they do not offer comprehensive solutions.

**Key Challenge**: There is a fundamental conflict between position/scale-invariant features needed for classification and position/scale-sensitive features needed for localization, which is amplified under the few-shot setting where shared features struggle to simultaneously satisfy the preferences of both tasks.

**Goal**: (1) How to effectively balance the gradient conflicts between classification and localization tasks to prevent the optimization of one task from undermining the other? (2) In scenarios of extreme sample scarcity, how to leverage external knowledge (large-scale pre-trained models) to enhance the performance of individual sub-tasks?

**Key Insight**: The authors propose dynamically adjusting the learning pace of the two tasks from the gradient level—increasing gradient weights for the poorly performing task while decreasing them for the other—thereby achieving precision-driven adaptive balancing. Meanwhile, they leverage the vision-language alignment capability of CLIP to boost classification accuracy in few-shot scenarios.

**Core Idea**: Resolve the classification-localization conflict from the optimization perspective using a precision-driven gradient balancer, and enhance few-shot classification capabilities from the knowledge perspective using CLIP knowledge distillation.

## Method

### Overall Architecture

MTL-FSOD is built upon a two-stage detection framework. The input image first passes through a shared feature extractor (backbone) to obtain feature maps, and then Region Proposal Network (RPN) generates candidate regions. Based on the RoI features, the model splits into classification and localization branches to perform predictions. The key innovations are reflected at two levels: (1) During training, a Precision-driven Gradient Balancer (PGB) dynamically adjusts the respective backpropagation gradient weights based on the current relative performance of the two tasks; (2) On the classification branch, a CLIP-based knowledge distillation module transfers large-scale pre-trained vision-language knowledge to the few-shot classifier, paired with a classification refinement scheme to further improve classification accuracy.

### Key Designs

1. **Precision-driven Gradient Balancer (PGB)**:

    - Function: Dynamically adjusts the gradient scales of classification loss and localization loss during training, alleviating optimization conflicts between the two tasks.
    - Mechanism: In each training iteration, precision metrics (such as classification accuracy and mean IoU) are computed for both classification and localization tasks on the current batch. Then, based on the relative precision discrepancy between the two tasks, gradient scaling factors are dynamically calculated: the task with poorer performance receives a larger gradient weight, while the task with better performance is appropriately suppressed. Specifically, letting $p_c$ and $p_l$ denote the precision metrics for classification and localization respectively, the gradient scaling factors are computed based on $\frac{p_c}{p_c + p_l}$ and $\frac{p_l}{p_c + p_l}$, ensuring that the learning progress of the two tasks remains roughly synchronized.
    - Design Motivation: Fixed loss weights cannot adapt to the dynamic changes in task difficulty during training. In few-shot scenarios, the difficulty of the classification task increases sharply with the introduction of novel classes. If fixed weights are still applied, localization gradients might "drown out" classification gradients, leading to severe degradation in classification performance.

2. **CLIP-based Knowledge Distillation Module**:

    - Function: Leverages the rich vision-language alignment knowledge pre-trained in CLIP to enhance the discriminative ability of the few-shot classifier.
    - Mechanism: The image encoder of CLIP acts as the teacher model, while the classification branch of the detector serves as the student model. For each RoI region, features are extracted using both the teacher and student, and the student's representation space is progressively aligned with the teacher's via feature alignment loss (such as L2 distance or cosine similarity). Consequently, the student can indirectly acquire semantic understanding capabilities learned by CLIP on large-scale datasets, thereby achieving accurate category discrimination even with rare annotations.
    - Design Motivation: CLIP is pre-trained on hundreds of millions of image-text pairs, possessing powerful zero-shot classification capabilities and rich semantic knowledge. Through knowledge distillation, such knowledge can be efficiently injected into the lightweight detection classification head, compensating for the lack of data in few-shot scenarios.

3. **Classification Refinement Scheme**:

    - Function: Refines classification results via post-processing to further improve the classification accuracy of novel classes.
    - Mechanism: Uses CLIP's text encoder to generate text embeddings for each class (e.g., "a photo of a [class name]"), and then computes the similarity between RoI features and text embeddings of all classes to obtain a second set of classification scores. The final classification result is a weighted fusion of the detector's original classification score and the CLIP-guided classification score, with the weight automatically determined on the validation set.
    - Design Motivation: Few-shot detectors often have unreliable classification confidence on novel classes (due to extremely scarce training samples), whereas the text-image alignment from CLIP provides a complementary signal independent of large amounts of annotations. Intergrating their fusion can significantly reduce classification error rates.

### Loss & Training

The overall loss is a weighted sum of classification, localization, and knowledge distillation losses: $L = \alpha_c L_{cls} + \alpha_l L_{loc} + \beta L_{KD}$, where $\alpha_c$ and $\alpha_l$ are dynamically determined by PGB, and $\beta$ is the weight for knowledge distillation. Training is divided into two phases: the base training phase, which uses abundant base class data to train the complete model; and the novel fine-tuning phase, which freezes most parameters and only fine-tunes the classification head and the gradient balancer.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | Ours (MTL-FSOD) | Prev. SOTA | Gain |
|--------|------|------|------|----------|------|
| PASCAL VOC | Novel Split 1, 1-shot | nAP50 | Consistent Improvement | FSOD baselines | Significant Outperformance |
| PASCAL VOC | Novel Split 1, 5-shot | nAP50 | Consistent Improvement | FSOD baselines | Significant Outperformance |
| MS COCO | 10-shot | nAP | Consistent Improvement | FSOD baselines | Steady Improvement |
| MS COCO | 30-shot | nAP | Consistent Improvement | FSOD baselines | Steady Improvement |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Baseline (Shared Features) | Baseline nAP | Standard two-stage detector |
| + PGB | nAP Improvement | Gradient balancing effectively alleviates task conflicts |
| + CLIP Distillation | Further nAP Improvement | External knowledge enhances classification capability |
| + Classification Refinement | Multi-fold nAP Improvement | CLIP text embeddings provide complementary classification signals |
| Full Model | Highest nAP | Best performance achieved through joint collaboration of all modules |

### Key Findings

- PGB consistently improves performance across different shot settings, indicating that the adaptiveness of gradient balancing is robust to varying data volumes.
- CLIP knowledge distillation brings particularly significant gains to novel classes while having a minor effect on base classes, suggesting that knowledge transfer primarily benefits data-scarce scenarios.
- The classification refinement scheme is most effective in the 1-shot setting, and its efficacy decreases as the shot count increases, which aligns with expectations.
- The method yields consistent improvements on multiple strong baselines (such as TFA, FSCE, and DeFRCN), demonstrating the generalizability of its methodology.

## Highlights & Insights

- Addressing multi-task conflicts at the gradient level is a clean and effective paradigm, and precision-driven dynamic balancing is more elegant than fixed weights or manual parameter tuning.
- The integration of CLIP goes beyond simple feature concatenation, leveraging knowledge transfer at both the distillation and refinement levels, which reflects a deep consideration of utilizing large-scale model knowledge.
- Functioning as a plug-and-play enhancement, the method can be applied to different few-shot detection baselines, exhibiting great versatility.
- The design mentality of PGB can also be generalized to other multi-task learning scenarios.

## Limitations & Future Work

- Although the code is open-source, it is marked as work-in-progress, making the practical reproducibility unclear.
- Whether the choices of precision metrics for the gradient balancer (accuracy for classification, IoU for localization) are optimal remains to be explored.
- Utilizing CLIP as a teacher model incurs additional computational and memory overheads, which may not be practical in resource-constrained environments.
- Verified only on two-stage detectors, future adaptation is needed for single-stage detectors (such as the YOLO series) and DETR-like methods.
- For more extreme zero-shot scenarios (0-shot), the applicability of the current framework requires further exploration.

## Related Work & Insights

- One of the core challenges in the FSOD domain is the classification-localization conflict, and this work provides a systematic solution.
- Leveraging large-scale pre-trained models (e.g., CLIP) to enhance few-shot learning has become a prevailing trend, and the distillation + refinement strategy proposed in this paper is highly instructive.
- The gradient balancing approach is related to multi-task learning methods like GradNorm and MGDA, yet it is more concise and targeted.

## Rating
- Novelty: ⭐⭐⭐⭐ While the innovation of individual modules is moderate, the overall formulation is rational.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated under multiple settings on both VOC and COCO, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and systematic methodology description.
- Value: ⭐⭐⭐⭐ Provides a practical multi-task optimization solution for few-shot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/object_detection/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [\[ECCV 2024\] DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)
- [\[ECCV 2024\] Plain-Det: A Plain Multi-Dataset Object Detector](plain-det_a_plain_multi-dataset_object_detector.md)

</div>

<!-- RELATED:END -->
