---
title: >-
  [Paper Note] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning
description: >-
  [CVPR 2025][Interpretability][Self-Supervised Learning] Approaching the analysis from the perspective of childhood visual development, this work systematically evaluates the capabilities of 22 self-supervised learning (SSL) models on mid-level vision tasks (depth estimation, surface normals, object segmentation, geometric correspondence, etc.). The study reveals that while a substantial performance gap remains between SSL models and supervised models on high-level semantic ta…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Self-Supervised Learning"
  - "Mid-Level Vision"
  - "Depth Estimation"
  - "3D Perception"
  - "Representation Evaluation"
date: 2026-05-08
content_hash: 87e4774c4a7cd99c
---

# Probing the Mid-Level Vision Capabilities of Self-Supervised Learning

**Conference**: CVPR 2025  
**arXiv**: [2411.17474](https://arxiv.org/abs/2411.17474)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Self-Supervised Learning, Mid-Level Vision, Depth Estimation, 3D Perception, Representation Evaluation

## TL;DR

Approaching the analysis from the perspective of childhood visual development, this work systematically evaluates the capabilities of 22 self-supervised learning (SSL) models on mid-level vision tasks (depth estimation, surface normals, object segmentation, geometric correspondence, etc.). The study reveals that while a substantial performance gap remains between SSL models and supervised models on high-level semantic tasks, this gap is significantly smaller for mid-level vision capabilities like 3D spatial perception.

## Background & Motivation

**Background**: Self-supervised learning has achieved approximately 70% of the performance of supervised learning on high-level semantic tasks such as ImageNet classification. However, mid-level vision capabilities—including 3D spatial perception (depth, surface normals), object segmentation, and geometric correspondence—have been largely neglected in SSL evaluations. These capabilities are crucial in human visual development: infants develop mature 3D spatial perception within their first year, long before semantic understanding.

**Limitations of Prior Work**: (1) Evaluation of SSL models is almost exclusively focused on high-level tasks such as classification and detection, leaving mid-level vision capabilities understudied; (2) the relative strengths and weaknesses of different SSL approaches (contrastive learning vs. masked modeling vs. clustering, etc.) on mid-level tasks remain unknown; (3) there is a lack of a systematic benchmark to measure the 3D spatial perception quality of SSL representations.

**Key Challenge**: The SSL field primarily pursues semantic-level representation quality (measured by classification accuracy), neglecting spatial and geometric information in visual representations, which is crucial for applications such as robotic manipulation, navigation, and AR.

**Goal**: To comprehensively evaluate SSL models using multiple mid-level vision tasks that span the "front line of 3D understanding", revealing which SSL methods are most effective at learning 3D spatial representations.

**Key Insight**: Drawing insights from developmental psychology: infants develop spatial perception with minimal supervision from the visual experience of head-mounted camera perspectives. If an SSL model is trained on 200 hours of infant head-mounted camera video (simulating childhood vision), can it acquire similar mid-level vision capabilities?

**Core Idea**: To systematically evaluate 6 mid-level vision tasks (object segmentation, depth estimation, surface normals, object geometric correspondence, scene geometric correspondence, and mid-level image similarity) across 22 SSL models. The findings reveal that the mid-level vision capabilities of SSL models are much closer to those of supervised models than their high-level semantic capabilities, with different SSL methods performing distinctively across various tasks.

## Method

### Overall Architecture

Purely evaluative study. This work selects 22 mainstream SSL models (covering Jigsaw, RotNet, NPID, SimCLR, MoCo v2-v3, BYOL, SimSiam, SwAV, DINO, iBOT, MAE, MaskFeat, etc.) pre-trained on ImageNet-1K using ResNet-50 and ViT-B/16 backbones. Six mid-level vision tasks are evaluated by freezing the feature extractor and training only a linear probe head or a lightweight decoder.

### Key Designs

1. **Comprehensive Mid-Level Vision Evaluation System**:

    - Function: Evaluating the spatial/geometric perception capabilities of SSL representations across multiple dimensions.
    - Mechanism: The six tasks cover the core capabilities of mid-level vision: (a) **generic object segmentation** (VOC07/VOC12, foreground-background binary segmentation, mIoU/F1/Acc); (b) **depth estimation** (NYU indoor depth and NAVI object depth, $\delta_i$ threshold accuracy and RMSE); (c) **surface normal estimation** (angular error and threshold accuracy); (d) **object geometric correspondence** (recall under 3D metric errors); (e) **scene geometric correspondence** (recall under 2D projection errors); (f) **mid-level image similarity** (determining which image is more similar in terms of mid-level features).
    - Design Motivation: Mid-level vision bridges low-level (edge detection) and high-level (classification) processing, and is key to constructing a unified 3D world representation. These six tasks progressively evaluate different aspects of spatial perception, from 2D grouping (segmentation) and 3D geometry (depth/normal/correspondence) to similarity judgment.

2. **Comparison of Multi-Category SSL Methods**:

    - Function: Identifying which category of SSL paradigm is most beneficial for mid-level visual representation learning.
    - Mechanism: Twenty-two SSL methods are divided into five main categories: (a) pretext tasks (Jigsaw, RotNet) – predicting rotation angles/jigsaw permutations; (b) instance discrimination (NPID, PIRL) – treating each image as an individual class; (c) contrastive learning (SimCLR, MoCo v2/v3, BYOL, SimSiam, Barlow Twins) – pulling closer representations of different augmentations of the same image; (d) clustering methods (SwAV, DeepCluster-v2, SeLa-v2, ClusterFit) – clustering in feature space to assign pseudo-labels; (e) masked modeling (MAE, MaskFeat, iBOT) – reconstructing masked image patches. ImageNet-1K pre-training is uniformly used to control data variables.
    - Design Motivation: The learning objectives of different SSL paradigms vary dramatically—contrastive learning encourages global invariance, masked modeling encourages local reconstruction, and clustering encourages semantic aggregation. How these distinct inductive biases affect mid-level vision capabilities needs to be systematically uncovered.

3. **Simulation Experiments of Childhood Visual Experience**:

    - Function: Exploring whether SSL models trained on "infant-like" visual experiences can acquire mid-level vision capabilities.
    - Mechanism: Using 200 hours of head-mounted camera video from a single child (aged 6-25 months) in the SAYCam dataset, embedding models and generative models are trained. The performance of these models on mid-level vision tasks is evaluated as a control comparison against models trained on ImageNet.
    - Design Motivation: If practical 3D perception representations can be learned purely from childhood visual experience, it would reveal that mid-level vision capabilities might not require large-scale, highly diverse data, but instead stem from more fundamental signals such as temporal consistency.

### Loss & Training

This is a purely evaluative study; each SSL model uses the pre-trained checkpoint from its original paper. Downstream tasks are evaluated using linear probing (frozen features + linear head) or a DPT decoder.

## Key Experimental Results

### Main Results (Mid-level vision performance of SSL models on ViT-B/16 backbone, with selected representative methods)

| Method | VOC12 mIoU | NYU Depth $\delta_1$↑ | Surface Normal $\delta_1$↑ | Geometric Correspondence Recall↑ |
|------|-----------|------------------|-------------------|----------------|
| MAE | 69.63 | Medium | Medium | Lower |
| MoCo v3 | 74.11 | Medium | Medium | Medium |
| DINO | **79.94** | Higher | Higher | **Highest** |
| iBOT | **84.72** | **Highest** | **Highest** | High |

### Ablation Study (Comparison of the gap between high-level and mid-level performance)

| Task Level | SSL vs. Supervised Performance Ratio |
|---------|-------------------|
| High-Level Semantics (Classification) | ~70% |
| Mid-Level Vision (3D Spatial Perception) | **~85-90%** |

### Key Findings

- **SSL's mid-level vision capabilities are far better than expected**: While SSL reaches only ~70% of supervised performance on high-level semantic tasks, the gap narrows significantly (~85-90%) on 3D spatial perception tasks, suggesting that SSL naturally tends to learn spatial structures.
- **iBOT and DINO consistently lead in mid-level tasks**: iBOT, which combines self-distillation and masked modeling, achieves the best performance in both object segmentation (mIoU 84.72%) and depth/normal estimation. DINO performs best in geometric correspondence.
- **Masked modeling methods (MAE/MaskFeat) perform relatively poorly**: Although they perform well after classification fine-tuning, their frozen features show insufficient mid-level vision capability, indicating that the learned mask reconstruction signal favors low-level texture rather than mid-level spatial structure.
- **Pretext task methods (Jigsaw/RotNet) are surprisingly competitive**: Jigsaw directly learns spatial relations by predicting patch arrangements, performing on par with contrastive learning methods in certain geometric tasks.
- DenseCL (pixel-level contrastive learning) shows an advantage in tasks requiring spatial precision, validating the benefit of dense-level self-supervised objectives for mid-level vision.

## Highlights & Insights

- The interdisciplinary framework of **examining SSL from the perspective of developmental psychology** is highly inspiring: human infants develop 3D spatial perception without semantic labels, which closely aligns with the label-free learning paradigm of SSL. The strong performance of SSL on mid-level vision tasks further supports this analogy.
- The finding that **different SSL objectives lead to distinct mid-level capabilities** provides a practical guide for SSL method selection: applications requiring 3D perception should opt for self-distillation methods (DINO/iBOT), while applications requiring dense prediction should choose pixel-level methods (DenseCL).
- The evaluation framework itself is a significant contribution to the SSL community, providing a multi-dimensional representation quality assessment tool that goes beyond classification accuracy.

## Limitations & Future Work

- Evaluation is limited to linear probing or lightweight decoders, leaving the differences under full fine-tuning unexplored.
- The infant video experiment has a small scale (200 hours of a single child), and the generalizability of the findings remains to be validated.
- Recent large-scale SSL methods (such as DINOv2, I-JEPA, etc.) are not included in the evaluation.
- Future work could explore SSL objective functions specifically designed for mid-level vision capabilities and the effectiveness of multi-task SSL pre-training.

## Related Work & Insights

- **vs. DINO**: DINO is widely considered one of the best SSL feature extractors. This study further confirms its advantage in mid-level vision and reveals that its self-distillation mechanism is key.
- **vs. MAE**: While MAE shows excellent classification performance after fine-tuning, its frozen features exhibit weaker mid-level vision capabilities, indicating that the masked reconstruction objective learns texture patterns rather than spatial structures.
- **vs. DUSt3R/MASt3R**: Recent cross-view geometric pre-training methods exhibit excellent performance on 3D tasks. The findings in this paper suggest that combining self-distillation (DINO) with cross-view geometric objectives might be an optimal strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically evaluate the mid-level vision capabilities of SSL, with an innovative interdisciplinary perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 22 models, 6 tasks, and multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ The introduction of developmental psychology enriches the narrative.
- Value: ⭐⭐⭐⭐ Provides an important evaluation framework and key insights to the SSL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](../../ICML2026/interpretability/interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](../../ICML2026/interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)

</div>

<!-- RELATED:END -->
