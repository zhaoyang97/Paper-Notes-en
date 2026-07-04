---
title: >-
  [Paper Note] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection
description: >-
  [CVPR 2025][Object Detection][Long-Tailed Object Detection] SimLTD proposes a simple and intuitive three-stage framework—pre-training on head classes, transferring to tail classes, and fine-tuning on hybrid-sampled data—optionally integrated with semi-supervised learning on unlabeled images, which comprehensively outperforms existing methods that rely on ImageNet labels on the LVIS v1 benchmark.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Long-Tailed Object Detection"
  - "Semi-Supervised Learning"
  - "Pseudo-Labeling"
  - "Model Transfer"
  - "LVIS"
date: 2026-05-08
content_hash: 4863d93b0f0d197d
---

# SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2412.20047](https://arxiv.org/abs/2412.20047)  
**Code**: None  
**Area**: Object Detection  
**Keywords**: Long-Tailed Object Detection, Semi-Supervised Learning, Pseudo-Labeling, Model Transfer, LVIS

## TL;DR
SimLTD proposes a simple and intuitive three-stage framework—pre-training on head classes, transferring to tail classes, and fine-tuning on hybrid-sampled data—optionally integrated with semi-supervised learning on unlabeled images, which comprehensively outperforms existing methods that rely on ImageNet labels on the LVIS v1 benchmark.

## Background & Motivation

**Background**: Object detection has made tremendous progress on balanced datasets like PASCAL VOC and MS-COCO, but its performance drops significantly on large-vocabulary long-tailed datasets like LVIS. Existing long-tailed detection methods mainly follow two paradigms: multi-stage training (e.g., LST, which requires 7 stages + knowledge distillation) and leveraging external ImageNet labels (e.g., Detic, RichSem, requiring ~14M labeled images) to supplement tail-class samples.

**Limitations of Prior Work**: (1) Multi-stage methods like LST are overly complex, and multiple knowledge transfer stages easily lead to catastrophic forgetting; (2) Methods relying on ImageNet labels require large-scale labeled auxiliary databases, which is impractical in custom domains like industrial scenarios and aerial remote sensing; (3) For genuinely rare categories (e.g., scarce fish species), collecting more annotated data from the internet is simply impossible.

**Key Challenge**: The fundamental challenge of long-tailed detection lies in the scarcity of tail-class samples (down to just 1), whereas existing solutions are either overly complex or strictly dependent on large-scale supervised auxiliary data.

**Goal**: To design a simple and general framework that does not rely on additional image-level annotations, while optionally utilizing easily accessible unlabeled images to boost long-tailed detection performance.

**Key Insight**: Through empirical analysis, the authors find that a strong representation model pre-trained on COCO shows significantly improved performance when transferred to the LVIS tail classes—the stronger the pre-training, the better the transfer. This inspires a direct "head pre-training $\rightarrow$ tail transfer" paradigm without the extra complexity of meta-learning or knowledge distillation.

**Core Idea**: Decomposing long-tailed detection into three simple steps—head pre-training for strong representations, tail transfer learning for adapting to rare classes, and hybrid fine-tuning to balance overall performance—with an option to incorporate pseudo-labeling semi-supervised learning for further enhancement.

## Method

### Overall Architecture
Given a long-tailed dataset $\mathcal{D}_{\text{ltd}}$ (e.g., LVIS), it is divided into a head set $\mathcal{D}_{\text{head}}$ (866 classes) and a tail set $\mathcal{D}_{\text{tail}}$ (337 classes) using a threshold $M=10$. The three-step workflow is as follows: Step 1 trains a detector on the head set (optionally including unlabeled data for semi-supervised learning) to obtain strong representations; Step 2 freezes the backbone and only re-initializes and fine-tunes the detection head on the tail set for transfer learning; Step 3 samples $k$ instances from each category across all classes to construct $\mathcal{D}_k$, fine-tuning the detection head to balance head and tail performance. It is compatible with various detectors such as Faster R-CNN, Deformable DETR, and DINO.

### Key Designs

1. **Head Pre-training + Semi-Supervised Enhancement**:

    - **Function**: Learn powerful visual representations on data-rich head classes.
    - **Mechanism**: In addition to standard data augmentation, Simple Copy-Paste and Repeat Factor Sampling are used to address class imbalance within the head set. Optionally, unlabeled images are introduced for pseudo-label learning via a student-teacher framework: $\mathcal{L} = \mathcal{L}_{\text{sup}} + \alpha \mathcal{L}_{\text{pseudo}}$. Three semi-supervised methods are explored: SoftER Teacher, MixTeacher, and MixPL.
    - **Design Motivation**: The head classes have relatively sufficient data, making them an ideal scenario for pre-training representation learning; semi-supervised learning further exploits cheap unlabeled data.

2. **Tail Transfer Learning (Frozen Backbone + Re-initialized Detection Head)**:

    - **Function**: Efficiently transfer representations learned from head classes to tail classes.
    - **Mechanism**: The detection head (classifier and regressor) learned from the head classes is removed and re-initialized with random weights, and then the detection head is trained exclusively on the tail set while keeping the backbone frozen. Semi-supervised pseudo-labeling is also optionally incorporated. To facilitate pseudo-label generation for rare classes, random copy-pasting of tail-class instances onto unlabeled images is innovatively applied.
    - **Design Motivation**: Empirical analysis shows that stronger pre-training leads to better transfer, and freezing the backbone avoids overfitting due to scarce tail data.

3. **Hybrid Sampling Fine-Tuning (Exemplar Replay)**:

    - **Function**: Balance head and tail classes, preventing the forgetting of head-class knowledge after transfer.
    - **Mechanism**: A relatively balanced small dataset $\mathcal{D}_k$ is constructed by sampling $k$ instances ($k \leq M$) per category from the entire long-tailed dataset, which is used for the final fine-tuning of the detection head. If a tail class has fewer than $k$ instances, all of them are used.
    - **Design Motivation**: Training exclusively on the tail classes in Step 2 causes forgetting of head classes. The hybrid sampling fine-tuning in Step 3 achieves a balance between "remembering the head and learning the tail".

### Loss & Training
The same detection losses (classification + regression) are used in all three stages, with an additional pseudo-label loss during the semi-supervised stage. Data augmentations include resizing, flipping, photometric distortion, and SCP+RFS, while semi-supervised learning additionally uses translation, shearing, rotation, and Cutout.

## Key Experimental Results

### Main Results

| Method | External Data | Backbone | mAP | AP_r↑ | AP_c | AP_f |
|------|---------|----------|-----|-------|------|------|
| Seesaw Loss | None | R101-FPN | 27.8 | 18.7 | 27.0 | 32.8 |
| RichSem | ImageNet-21K | R50 | 35.1 | 26.0 | 32.6 | 41.8 |
| **SimLTD (Supervised)** | **None** | **R50** | **35.0** | **32.0** | 34.0 | 37.5 |
| RichSem | ImageNet-21K | Swin-B | 46.4 | 38.5 | 45.1 | 51.3 |
| **SimLTD (Supervised)** | **None** | **Swin-B** | **47.2** | **42.7** | 46.7 | 49.9 |
| Detic | ImageNet-21K | R50 | 32.5 | 26.2 | 31.3 | 36.6 |
| **SimLTD (Semi-Supervised)** | **COCO-unlabeled** | **R50** | **39.4** | **32.6** | 38.5 | 43.6 |

### Ablation Study

| Configuration | mAP | AP_r↑ | Description |
|------|-----|-------|------|
| Head Pre-training $\rightarrow$ Direct Full-set Fine-tuning | 31.2 | 21.5 | No transfer learning |
| Head Pre-training $\rightarrow$ Tail Transfer $\rightarrow$ No Fine-tuning | 15.3 | 25.1 | Severe forgetting of head classes |
| **Full SimLTD** | **35.0** | **32.0** | Complete three-stage pipeline |
| SimLTD + SoftER Teacher | 30.3 | 23.3 | Faster R-CNN |
| SimLTD + MixTeacher | 31.8 | 23.4 | Faster R-CNN |
| **SimLTD + MixPL** | **39.4** | **32.6** | DINO, R50 |

### Key Findings
- Supervised SimLTD doesn't use any external labeled data, yet its tail-class AP_r (32.0) already outperforms RichSem (26.0) which relies on ImageNet-21K labels, while achieving a comparable mAP on R50 (35.0 vs 35.1).
- On Swin-B, SimLTD's AP_r (42.7) significantly outperforms RichSem (38.5), demonstrating the scalability of a stronger backbone paired with the three-stage strategy.
- Using only COCO-unlabeled2017 (~120k unlabeled images), the semi-supervised SimLTD improves the mAP of DINO/R50 from 35.0 to 39.4.
- The augmentation strategy of copy-pasting tail-class instances onto unlabeled images is particularly crucial for semi-supervised learning.

## Highlights & Insights
- **Overcoming Complexity with Simplicity**: With just three simple steps and without the need for meta-learning or knowledge distillation, it significantly outperforms complex methods that require 7-stage training or 14M extra labels. The simplicity of the method is highly appealing.
- **Replacing Labeled Auxiliary Data with Unlabeled Data**: It is demonstrated that leveraging unlabeled images via pseudo-labeling semi-supervised learning can replace or even outperform schemes depending on large labeled databases like ImageNet-21K, greatly boosting practical feasibility.
- **Empirical Discovery of Transfer Learning**: A stronger pre-trained model leads to a better tail transfer effect; this simple yet important finding provides a clear pathway for improving long-tailed learning.

## Limitations & Future Work
- The three stages require separate training, and the total computational cost remains substantial (~460 GPU hours for Swin-L).
- Freezing the backbone during tail transfer may limit the backbone's ability to adapt to tail-class features.
- The choice of $k$ (the number of samples per class in hybrid sampling) may require tuning for different datasets.
- Future work could explore unifying the three stages into an end-to-end curriculum learning scheme.

## Related Work & Insights
- **vs LST**: LST requires 7 stages + knowledge distillation, whereas SimLTD only needs 3 steps without distillation, being simpler and more effective.
- **vs Detic/RichSem**: These methods strictly depend on ImageNet-21K labeled data, whereas SimLTD achieves comparable or even superior performance using unlabeled data + pseudo-labels, offering broader applicability.
- The generality of the framework (supporting both CNN and Transformer detectors) allows it to serve as a strong baseline for long-tailed detection.

## Rating
- Novelty: ⭐⭐⭐ The method itself is a simple combination of existing techniques, but the combination is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Various backbones, multiple detectors, supervised/semi-supervised comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and well-defined motivation.
- Value: ⭐⭐⭐⭐ Provides a simple and efficient baseline for long-tailed detection, offering valuable reference for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)
- [\[ECCV 2024\] Rectify the Regression Bias in Long-Tailed Object Detection](../../ECCV2024/object_detection/rectify_the_regression_bias_in_long-tailed_object_detection.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/object_detection/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)

</div>

<!-- RELATED:END -->
