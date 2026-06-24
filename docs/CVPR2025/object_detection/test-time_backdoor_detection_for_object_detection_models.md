---
title: >-
  [Paper Note] Test-Time Backdoor Detection for Object Detection Models
description: >-
  [CVPR 2025][Object Detection][Backdoor Attack Detection] TRACE (TRAnsformation Consistency Evaluation) proposes the first test-time backdoor sample detection method for object detection models. Based on two key observations—that poisoned samples yield more consistent detection results across different backgrounds, and clean samples are more consistent under different focus information—it detects poisoned samples by calculating the variance of object confidence after applying…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Backdoor Attack Detection"
  - "Test-Time Defense"
  - "Transformation Consistency"
  - "Black-Box Detection"
date: 2026-05-08
content_hash: c58c58fd3dd06f73
---

# Test-Time Backdoor Detection for Object Detection Models

**Conference**: CVPR 2025  
**arXiv**: [2503.15293](https://arxiv.org/abs/2503.15293)  
**Code**: None  
**Area**: Object Detection / AI Security  
**Keywords**: Backdoor Attack Detection, Object Detection, Test-Time Defense, Transformation Consistency, Black-Box Detection

## TL;DR
TRACE (TRAnsformation Consistency Evaluation) proposes the first test-time backdoor sample detection method for object detection models. Based on two key observations—that poisoned samples yield more consistent detection results across different backgrounds, and clean samples are more consistent under different focus information—it detects poisoned samples by calculating the variance of object confidence after applying transformations to the foreground and background, achieving black-box general-purpose detection and improving AUROC by 30% compared to the SOTA.

## Background & Motivation

**Background**: Deep learning object detection models are widely deployed in safety-critical scenarios such as autonomous driving and security surveillance. Since model training may be outsourced or utilize third-party datasets, there is an inherent risk of backdoor attacks. Backdoor attacks embed a predefined trigger in a small portion of training samples, causing the model to exhibit anomalous behavior when encountering the trigger during inference.

**Limitations of Prior Work**: While extensive work exists for backdoor detection in image classification, backdoor detection in object detection remains practically unexplored. The impact of backdoor attacks on object detection is significantly more complex than on classification: (1) **"Ghost" object attacks**—the trigger causes the model to detect non-existent objects; (2) **"Vanish" object attacks**—the trigger causes the model to miss actually existing objects. These compounding effects render existing classification backdoor detection methods completely ineffective.

**Key Challenge**: Backdoor detection methods for classification usually focus on flip patterns of output classes, whereas object detection outputs a variable number of bounding boxes. The structural differences in output spaces mean that existing detection metrics cannot be directly migrated.

**Goal**: Design a test-time, black-box backdoor detection method that can determine whether an input sample contains a backdoor trigger without accessing model parameters, and is effective against multiple types of object detection backdoor attacks.

**Key Insight**: Starting from adversarial transformations, the authors identify two distinct behavioral differences between poisoned and clean samples on object detection outputs: (1) The effect of a backdoor trigger is independent of background variations—even when the background is replaced, the trigger still functions, hence poisoned samples exhibit highly consistent detection results across different backgrounds; (2) Clean samples depend on real visual content; when focusing on different regions, detection results naturally change, exhibiting higher consistency under targeted focus.

**Core Idea**: Apply foreground and background transformations to each test sample, and then calculate the consistency (variance of confidence scores) of the detection results after transformation. If a sample exhibits high consistency (low variance) typical of poisoned samples, it is flagged as suspicious.

## Method

### Overall Architecture
The input to TRACE is a test image to be inspected and a black-box object detection model. The output is a decision of whether the image is poisoned (along with a corresponding anomaly score). The process consists of three steps: (1) Apply foreground and background transformations to the input image respectively to generate multiple transformed versions; (2) Feed all transformed versions into the object detection model to collect detection results (bounding boxes, categories, and confidence scores) for each version; (3) Compute the consistency score of detection results across transformations to determine whether the sample is poisoned.

### Key Designs

1. **Background Transformation**:

    - **Function**: Test whether the trigger effect is independent of the background content.
    - **Mechanism**: Perform foreground/background separation on the input image (using off-the-shelf segmentation models or simple saliency detection), then replace the background with different random images or solid colors to generate $N$ background-transformed versions. After passing all versions to the detector, compute the variance $\sigma^2_{bg}$ of the confidence scores of each bounding box across the $N$ versions. **Key Observation**: The triggers of poisoned samples are typically situated in the foreground, and background replacement does not affect the trigger, leading to highly consistent detection results (small $\sigma^2_{bg}$). Clean samples rely on overall scene context, so background changes alter detection confidence (large $\sigma^2_{bg}$).
    - **Design Motivation**: A backdoor trigger is a localized pattern independent of scene semantics, and its activation does not require global context. This characteristic can be exposed via background replacement.

2. **Foreground Transformation**:

    - **Function**: Test whether detection results depend on real visual content.
    - **Mechanism**: Perform local foreground occlusion or cropping on the input image to generate $M$ versions. Each version preserves different foreground regions and is fed into the detector to calculate the variance of confidence scores $\sigma^2_{fg}$. **Key Observation**: Object detection for clean samples relies on the visual features of real targets, so occluding different regions leads to large changes in detection results (large $\sigma^2_{fg}$). In contrast, anomalous detection results of poisoned samples are primarily driven by the trigger. As long as the trigger region is not occluded, the results remain stable (small $\sigma^2_{fg}$).
    - **Design Motivation**: This complements background transformation. Background transformation detects the property "triggers are independent of background", while foreground transformation detects "detection results do not rely on genuine targets". The combination of both provides a more comprehensive assessment.

3. **Consistency Scoring & Decision**:

    - **Function**: Quantify transformation consistency into an anomaly score.
    - **Mechanism**: For each test sample, synthesize the consistency of detection results under foreground and background transformations. The concrete formula is: $S = \lambda \cdot (1/\sigma^2_{bg}) + (1-\lambda) \cdot (1/\sigma^2_{fg})$, where $\lambda$ is a balancing coefficient. A higher score indicates a higher probability that the sample is poisoned. The final binary classification decision is made by setting a threshold or using an anomaly detection algorithm.
    - **Design Motivation**: Using the reciprocal of the variance as a metric for consistency is intuitive and computationally simple. Combining information from both foreground and background dimensions reduces false positive rates.

### Loss & Training
TRACE is a test-time method and does not require training. The entire process is a black-box operation during inference, relying only on query access to the inputs and outputs of the object detection model.

## Key Experimental Results

### Main Results

Evaluation is conducted on the PASCAL VOC and MS COCO datasets against multiple object detection backdoor attacks (ghost object attacks, vanish object attacks, misclassification attacks) using detectors like Faster R-CNN and YOLO.

| Method | VOC Ghost (AUROC) | VOC Vanish (AUROC) | COCO Ghost (AUROC) | COCO Vanish (AUROC) |
|------|-------------------|--------------------|--------------------|---------------------|
| STRIP (Adapted from classification) | 52.3 | 55.1 | 50.8 | 53.2 |
| SentiNet | 58.7 | 54.2 | 56.1 | 52.9 |
| Spectral Signature | 61.4 | 59.8 | 58.3 | 56.7 |
| **TRACE** | **93.1** | **89.7** | **91.2** | **87.5** |

### Ablation Study

| Configuration | AUROC (VOC avg) | Description |
|------|----------------|------|
| TRACE (bg + fg) | 91.4 | Complete method, combining both transformations |
| bg transformation only | 86.2 | Only background transformation, dropped by 5.2% |
| fg transformation only | 83.7 | Only foreground transformation, dropped by 7.7% |
| N=3 transformed versions | 85.6 | Decreased number of transformations leads to lower detection capability |
| N=10 transformed versions | 91.4 | Sufficient number of transformations |
| N=20 transformed versions | 91.8 | Diminishing marginal returns |

### Key Findings
- TRACE improves AUROC by approximately 30 percentage points over the best existing detection methods, demonstrating that methods adapted directly from classification tasks are indeed unsuitable for object detection.
- The contributions of background transformation and foreground transformation are complementary, and using them in combination yields the best results; background transformation is more effective against "ghost object" attacks, whereas foreground transformation is more critical for "vanish object" attacks.
- The number of transformations $N$ saturates around 10; increasing the number of transformations further yields limited gains.
- Under adaptive attacks, TRACE still maintains robust performance. To evade TRACE, attackers must make the trigger sensitive to both foreground and background transformations, which directly contradicts the goal of stable trigger activation.

## Highlights & Insights
- **The two observations (background consistency + foreground independence) accurately capture the essential characteristics of backdoor triggers**: Triggers operate as semantic-free shortcuts, which inherently means they are decoupled from context. This insight itself is a significant conceptual contribution.
- **The black-box and training-free detection paradigm is highly practical**: In real-world scenarios, models are often deployed via APIs, where users only have black-box query access to inputs and outputs. TRACE fully satisfies this constraint.
- **The method can theoretically be extended to backdoor detection in other object-level tasks**, such as instance segmentation.

## Limitations & Future Work
- Foreground/background separation depends on the quality of existing segmentation methods; inaccurate segmentation may degrade detection performance.
- For triggers embedded in the background rather than the foreground (e.g., global perturbation triggers), background transformation may fail.
- Computational overhead scales linearly with the number of transformations; $N=10$ implies more than 10 inferences per sample.
- Clean-label attacks (attacks where triggers do not alter labels during training) are not considered, and their behavior patterns might differ.

## Related Work & Insights
- **vs STRIP**: STRIP detects classification backdoors by superposing multiple images, relying on variations in output entropy. However, this method is inapplicable to the varying count of bounding boxes in object detection.
- **vs SentiNet**: SentiNet locates suspicious regions to detect triggers, but it requires white-box access to intermediate feature maps. TRACE is completely black-box.
- **vs Neural Cleanse**: Neural Cleanse requires a large number of clean samples and model parameters to reverse-engineer triggers. TRACE is a sample-by-sample, test-time method requiring no extra data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first test-time backdoor detection method for object detection, with two profound and novel core observations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple attack types, datasets, and detectors, including adaptive attacks.
- Writing Quality: ⭐⭐⭐⭐ Clear logical progression from observation to method to validation.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in object detection backdoor detection, offering a simple and practical method with direct implications for deployment security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)
- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)
- [\[CVPR 2025\] Interpreting Object-level Foundation Models via Visual Precision Search](interpreting_object-level_foundation_models_via_visual_precision_search.md)

</div>

<!-- RELATED:END -->
