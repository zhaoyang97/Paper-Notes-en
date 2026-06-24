---
title: >-
  [Paper Note] Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach
description: >-
  [CVPR 2025][Grinding wheel wear classification] This paper proposes a hierarchical visual classification framework based on EfficientNetV2, which decomposes the wear state of abrasive flap wheels into three levels (usage state $\rightarrow$ wear type $\rightarrow$ severity), achieving classification accuracy of 93.8% to 99.3% across various subtasks.
tags:
  - "CVPR 2025"
  - "Grinding wheel wear classification"
  - "hierarchical classification"
  - "transfer learning"
  - "EfficientNetV2"
  - "Grad-CAM"
date: 2026-05-08
content_hash: 274701d052a2e499
---

# Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach

**Conference**: CVPR 2025  
**arXiv**: [2603.12852](https://arxiv.org/abs/2603.12852)  
**Code**: None  
**Area**: Other  
**Keywords**: Grinding wheel wear classification, hierarchical classification, transfer learning, EfficientNetV2, Grad-CAM

## TL;DR
This paper proposes a hierarchical visual classification framework based on EfficientNetV2, which decomposes the wear state of abrasive flap wheels into three levels (usage state $\rightarrow$ wear type $\rightarrow$ severity), achieving classification accuracy of 93.8% to 99.3% across various subtasks.

## Background & Motivation

**Background**: Abrasive flap wheels are widely used for finishing complex free-form surfaces in aerospace and mold manufacturing, adapting to different surface shapes due to their flexible nature. Tool condition monitoring in automated grinding is key to achieving adaptive process control.

**Limitations of Prior Work**: Due to their flexible structure, abrasive flap wheels exhibit complex and highly variable wear patterns (e.g., concave/convex contour changes, flap tearing), and traditional rule-based image processing methods are error-prone when dealing with such high-variance wear patterns. Existing CNN-based wear monitoring methods mainly target rigid tools like milling cutters and turning tools, lacking specialized frameworks for the wear characteristics of flexible tools.

**Key Challenge**: The wear of abrasive flap wheels is multi-dimensional (geometry, flap integrity, severity), whereas a single multi-class classifier requires sufficient training data for each wear combination, leading to data fragmentation issues.

**Goal**: To design a multi-level hierarchical classification framework that decomposes wear detection into multiple independent sub-decisions, where each sub-network is responsible for only one specific classification task.

**Key Insight**: Logical dependency relationships exist between different wear features. Hierarchical decomposition not only reduces the classification difficulty of each sub-network but also enables the detection of misclassifications through logical consistency checks.

**Core Idea**: Decompose the wear analysis of abrasive flap wheels into a three-level hierarchical decision tree (usage state $\rightarrow$ profile type + flap tearing $\rightarrow$ severity), using independent EfficientNetV2 classifiers for each level, and utilizing logical constraints between hierarchies to enhance robustness.

## Method

### Overall Architecture
The input consists of radial and axial perspective images of the abrasive flap wheel. A three-level hierarchical classification is sequentially performed to judge: Level 1 determines the usage state (new/used), Level 2 determines the wear profile type (rectangular/concave/convex) and flap tearing (present/absent), and Level 3 evaluates the severity of the concave or convex profile (partial/complete deformation).

### Key Designs

1. **Hierarchical Classification Architecture**:

    - **Function**: Decomposes complex multi-class wear classification into multiple binary or ternary sub-classification tasks.
    - **Mechanism**: Unlike a monolithic classifier, hierarchical classification allows each sub-network to focus on a specific wear dimension. Level 1 uses radial images to determine if the tool is new or used, Level 2 simultaneously performs profile classification and flap tear detection, and Level 3 refines the severity of deformity. Each sub-network can be trained independently, avoiding the data scarcity issues caused by category combination explosion.
    - **Design Motivation**: The optimal camera angles vary for different wear features—flap tearing is easiest to observe from the axial perspective, whereas contour changes require a radial view. The hierarchical structure allows different sub-networks to utilize different perspective inputs.

2. **Logical Consistency Check**:

    - **Function**: Detects contradictory classification results between levels.
    - **Mechanism**: Defines 11 consistent outcome paths and 3 contradictory paths. For example, a logical contradiction occurs if Level 1 is classified as "new" but Level 2 detects flap tearing or a non-rectangular profile. Contradictory results can trigger re-detection or manual review.
    - **Design Motivation**: Leverages physical prior knowledge of wear to provide additional validation for classification results, improving reliability in industrial environments.

3. **Transfer Learning based on EfficientNetV2**:

    - **Function**: Achieves high-accuracy classification on limited datasets.
    - **Mechanism**: Uses ImageNet pre-trained EfficientNetV2 as the feature extractor. EfficientNetV2-L is used for flap tear detection, and EfficientNetV2-S is used for the remaining sub-networks. The AdamW optimizer is employed with both the learning rate and weight decay set to $1 \times 10^{-4}$.
    - **Design Motivation**: Labeled data in industrial scenarios is limited (approximately 13,240 images). Transfer learning can fully leverage pre-trained features.

### Loss & Training
Each sub-network is trained independently using standard cross-entropy loss. During training, data augmentation techniques such as random rotation, brightness, and contrast adjustments are applied to the images. Axial images are converted to grayscale for flap tear detection.

## Key Experimental Results

### Main Results
Evaluated on a total of 13,240 images from 105 abrasive flap wheels with different wear conditions:

| Subtask | Accuracy | F1-score | AUC |
|--------|--------|----------|-----|
| Usage State Classification (Level 1) | 98.6% | 0.983 | 0.999 |
| Flap Profile Classification (Level 2) | 95.4% | 0.954 | 0.99 |
| Flap Tear Detection (Level 2) | 93.8% | 0.935 | 0.98 |
| Concave Severity (Level 3) | 99.3% | 0.993 | 1.00 |
| Convex Severity (Level 3) | 95.0% | 0.948 | ≥0.98 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Hierarchical Classification (Ours) | 93.8-99.3% across subtasks | Each subtask is optimized independently |
| Logical Consistency Check | Detects contradictory paths | Enhances industrial reliability |
| EfficientNetV2-L (Tearing) | 93.8% | Larger model enhances accuracy in difficult-to-distinguish tasks |
| EfficientNetV2-S (Others) | >95% | Lightweight models are sufficient |

### Key Findings
- Flap tear detection (93.8%) is the most difficult subtask. The average confidence of misclassification (0.76) is significantly lower than the overall average (0.92), which can be further filtered using a confidence threshold.
- The unidirectional misclassification pattern of "complete $\rightarrow$ partial" in convex profile classification indicates that the visual boundary between partial and complete convex profiles is gradual.
- Grad-CAM visualization validates that the model has learned physically relevant features (flap edge shapes, contour change regions).
- Rotational symmetry causes some concave/convex misclassifications; thus, vertical flip data augmentation is recommended.

## Highlights & Insights
- **Hierarchical Decomposition Approach**: Decomposing complex multi-attribute classification tasks into multiple simple sub-decisions avoids the data demand issue caused by category combination explosion and can be transferred to other industrial inspection scenarios.
- **Logical Consistency Check**: Establishing constraint relationships between levels using domain priors to detect misclassifications is a lightweight yet effective way to improve robustness.
- **Grad-CAM Validating Physical Features**: Not only used to interpret the model, but also to verify whether the model has learned physically relevant wear features.

## Limitations & Future Work
- The dataset scale is relatively small (approx. 13K images), and the generalization capability on few-shot categories remains to be validated.
- Lack of direct comparison experiments with monolithic multi-class classifiers.
- Flap tear detection has the lowest accuracy, and attention mechanisms or object detection methods have not been attempted.
- The error propagation problem between levels has not been analyzed in depth.

## Related Work & Insights
- **vs. Direct Multi-Class Classification**: Hierarchical classification decomposition reduces the difficulty of sub-tasks and data requirements but may introduce error propagation.
- **vs. Indirect Methods (Force/Vibration Signals)**: Direct visual detection decouples monitoring from workpiece geometric complexity.
- This paper demonstrates a practical case of combining industrial inspection problems with deep learning.

## Rating
- Novelty: ⭐⭐⭐⭐ No breakthrough in methodology, mainly an engineering combination of hierarchical classification and transfer learning
- Experimental Thoroughness: ⭐⭐⭐⭐ Lack of comparison with end-to-end multi-class classifiers
- Writing Quality: ⭐⭐⭐⭐⭐ Clear description of industrial context, detailed Grad-CAM analysis
- Value: ⭐⭐⭐⭐ Practical value for wear detection of abrasive flap wheels, limited generality

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[ICML 2025\] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](../../ICML2025/others/function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)
- [\[CVPR 2026\] HierUQ: Hierarchical Uncertainty Quantification with Adaptive Granularity Reconciliation for Degraded Image Classification](../../CVPR2026/others/hieruq_hierarchical_uncertainty_quantification_with_adaptive_granularity_reconci.md)
- [\[CVPR 2026\] Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach](../../CVPR2026/others/revisiting_f-measure_optimization_in_multi-label_classification_a_sampling-based.md)

</div>

<!-- RELATED:END -->
