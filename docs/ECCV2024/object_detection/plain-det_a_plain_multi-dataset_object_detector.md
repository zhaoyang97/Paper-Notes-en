---
title: >-
  [Paper Note] Plain-Det: A Plain Multi-Dataset Object Detector
description: >-
  [ECCV 2024][Object Detection][Multi-dataset training] Plain-Det proposes a simple and flexible multi-dataset object detection framework. By incorporating semantic space calibration, class-aware query compositor, and hardness-indicated dynamic sampling strategies, it achieves 51.9 mAP on COCO (matching the SOTA at that time) and can be flexibly scaled to new datasets while maintaining robust performance.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Multi-dataset training"
  - "Semantic space calibration"
  - "Sparse query"
  - "Dynamic sampling"
date: 2026-05-08
content_hash: f714c75205315808
---

# Plain-Det: A Plain Multi-Dataset Object Detector

**Conference**: ECCV 2024  
**arXiv**: [2407.10083](https://arxiv.org/abs/2407.10083)  
**Code**: [https://github.com/ChengShiest/Plain-Det](https://github.com/ChengShiest/Plain-Det)  
**Area**: Object Detection / Segmentation  
**Keywords**: Multi-dataset training, Object detection, Semantic space calibration, Sparse query, Dynamic sampling

## TL;DR

Plain-Det proposes a simple and flexible multi-dataset object detection framework. By incorporating semantic space calibration, class-aware query compositor, and hardness-indicated dynamic sampling strategies, it achieves 51.9 mAP on COCO (matching the SOTA at that time) and can be flexibly scaled to new datasets while maintaining robust performance.

## Background & Motivation

As a fundamental task in computer vision, object detection requires large-scale annotated data. However, dense annotation is costly. A practical strategy is to integrate and jointly train on multiple existing detection datasets.

**Limitations of Prior Work**:
- Inconsistent taxonomies across different datasets (e.g., "dolphin" is a class in O365 but background in COCO).
- Significant distribution discrepancies (number of categories ranges from 80 to 1203, and image quantities differ by up to 17 times).
- Existing methods (such as UniDet, ScaleDet) require manually constructing a unified label space, which introduces more noise and requires reconstructing the taxonomy as the number of datasets increases.
- Multi-dataset training often leads to performance degradation rather than improvement.

**Key Challenge**: How to share and mutually leverage knowledge across multiple datasets while maintaining flexibility?

**Key Insight**: Instead of pursuing a unified label space, the proposed method retains independent classification heads for each dataset, implementing implicit knowledge transfer through a shared semantic space established by CLIP text embeddings. Meanwhile, it discovers and utilizes three critical insights in multi-dataset training: frequency bias in the semantic space, superiority of sparse queries, and the "emergent property."

## Method

### Overall Architecture

Plain-Det is based on an arbitrary query-based detector (e.g., Deformable-DETR, Sparse R-CNN), introducing three core components:
1. Dataset-specific classification heads + a shared semantic space with a frozen classifier.
2. Class-Aware Query Compositor.
3. Hardness-indicated Sampling strategy.

Overall architecture: shared backbone + encoder $\rightarrow$ shared decoder $\rightarrow$ shared regression head $\rightarrow$ dataset-independent classification heads.

### Key Designs

1. **Semantic Space Calibration**:

    - **Function**: Correct the frequency bias in CLIP text embeddings.
    - **Core Idea**: The similarity between high-frequency words (e.g., "person") and other words in CLIP training data tends to be biased toward higher values. The authors found that the embedding of the empty string NULL can serve as a frequency-driven base vector.
    - **Formula**: $\hat{W}^m = \text{Norm}(W^m - \text{Enc}_{\text{text}}(\texttt{NULL}))$
    - **Design Motivation**: When directly using CLIP text embeddings as classifier weights, the similarity between high-frequency nouns is excessively high, which compromises classification accuracy. Subtracting the NULL embedding yields a more reasonable similarity matrix.

2. **Class-Aware Query Compositor**:

    - **Function**: Generate object queries that account for both dataset priors and image prior in multi-dataset scenarios.
    - **Core Idea**: Use dataset classifier embeddings to generate a weak dataset prior query, which is then combined with global image features.
    - **Formula**: $\mathcal{Q}^b = \text{MLP}(\hat{W}^m)$ (dataset prior), $\mathcal{W} = \text{MLP}(\text{Max-Pool}(Enc(I)))$ (image prior), $\mathcal{Q}^c = \mathcal{W} \mathcal{Q}^b$ (final query)
    - **Design Motivation**: Preliminary experiments show that strong dataset priors (top-K pixel features) lead to significant performance drops in multi-dataset training, while purely learned queries lack dataset adaptability. Weak prior queries strike a balance between the two.

3. **Hardness-indicated Sampling**:

    - **Function**: Dynamically adjust the sampling ratio according to each dataset's loss during the training process.
    - **Core Discovery - Emergent Property**: Even if a detector trained on multiple datasets exhibits low accuracy on a given dataset in certain iterations, it can quickly recover with a few dataset-specific iterations. Multi-dataset training naturally imparts a more generalized detection capability.
    - **Formula**: $w_m = \frac{L_m}{\min(\{L_i\})} \cdot [\frac{\max(\{S_i\})}{S_m}]^{1/2}$, where $L_m$ is the loss and $S_m$ is the dataset size.
    - **Design Motivation**: Static sampling causes severe training fluctuations in large-scale datasets (such as O365), whereas dynamic sampling adaptively balances training.

### Loss & Training

- Each dataset maintains its original loss function and sampling strategy (e.g., LVIS uses RFS to handle long-tail distribution).
- Classification loss: Cross-Entropy; Regression loss: GIoU + L1.
- Strict multi-dataset setting: Under the premise that the total number of iterations remains unchanged when adding datasets, the performance gains solely originate from mutual assistance among the datasets.

## Key Experimental Results

### Main Results

| Dataset Combination | COCO mAP | LVIS mAP | O365 mAP | Average mAP |
|-----------|----------|----------|----------|---------|
| Single-dataset Training | 45.6 | 33.6 | 32.2 | 43.2 |
| Joint Training W/O Plain-Det | 44.2 | 29.1 | 27.4 | 40.0 |
| Plain-Det (L+C+O+D) | **51.9** | **40.9** | **33.3** | **47.4** |

Comparison with SOTA multi-dataset detectors (L+C+O+D setting):

| Method | COCO AP | LVIS AP | Average mAP | Gain Relative to Single-dataset |
|------|---------|---------|---------|-----------------|
| UniDet | 45.5 | - | 35.0 | +1.3 |
| ScaleDet | 47.1 | 36.8 | 41.9 | +2.4 |
| **Plain-Det** | **51.9** | **40.9** | **46.4** | **+4.8** |

### Ablation Study

| Configuration | COCO AP | LVIS AP | Average mAP | Description |
|------|---------|---------|---------|------|
| Baseline (w/o partition head) | 39.3 | 23.8 | 31.6 | Most basic setting |
| + Partition head | 38.1 | 24.0 | 31.1 | Slightly decreased but ensures flexibility |
| + Partition head + Sparse queries | 44.2 | 28.7 | 36.5 | Sparse queries bring significant improvement |
| + Above + Semantic calibration | 45.3 | 30.2 | 37.8 | Label space calibration is effective |
| + Above + Dynamic sampling | 47.1 | 32.4 | 35.0 | Dynamic sampling improves mAP by 4.1% |

### Key Findings

1. **More datasets lead to better performance**: Under strict settings (no increase in overall iterations), COCO AP increases from 37.2 (LVIS only) to 51.9 (4-dataset joint), surpassing the 45.6 of single-dataset training.
2. **Emergent Property**: Detectors jointly trained on multiple datasets exhibit more general detection capabilities, which can be quickly activated with a few dataset-specific iterations.
3. **Zero-shot Transfer**: Plain-Det achieves 46.1 mAP on the ODinW benchmark, outperforming GLIP (44.0) while using less data (3.6M vs 5.5M).
4. **Cross-detector Architecture Compatibility**: A 3.1% improvement is also achieved on Sparse R-CNN.

## Highlights & Insights

- **NULL embedding calibration is an extremely simple but elegant insight**: Simply subtracting the CLIP embedding of empty strings significantly corrects the classifier's frequency bias at almost zero cost.
- **The discovery of emergent properties is valuable**: It reveals that multi-dataset training naturally endows the model with more general capabilities, an insight that could inspire other multi-task/multi-domain learning studies.
- **A genuinely "plain" design**: It avoids manual construction of a unified taxonomy and extra alignment modules, directly leveraging CLIP semantic space + dataset-specific heads.
- **High training efficiency**: Achieves higher performance on COCO with fewer epochs (36 epochs vs. 192 epochs for ScaleDet).

## Limitations & Future Work

- The semantic space relies on the CLIP model, which may inherit biases from CLIP's training data.
- The theoretical foundation of the NULL calibration method is not fully established, lacking in-depth analysis of why the NULL embedding happens to capture frequency bias.
- The maximum scale of the current experiments is around 4M images + 2249 classes; the performance under larger scales remains unknown.
- Inference requires specifying the dataset to select the classification head, which is less flexible than a unified taxonomy.

## Related Work & Insights

- **UniDet** learns a unified label space, but lacks flexibility.
- **ScaleDet** also uses CLIP embeddings but did not notice the frequency bias issue.
- **Detic** expands vocabularies using ImageNet 21K classification labels.
- The semantic calibration method proposed in this work can be extended to other scenarios that use CLIP text embeddings as classifiers (e.g., open-vocabulary detection).

## Rating

- Novelty: ⭐⭐⭐⭐ (All three insights are novel, with NULL calibration being particularly ingenious)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Tested on 17 datasets, multi-architecture verification, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, rich figures and tables)
- Value: ⭐⭐⭐⭐ (High practicality, generalizability of the method)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](../../CVPR2025/object_detection/object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [\[ECCV 2024\] Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer](portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)

</div>

<!-- RELATED:END -->
